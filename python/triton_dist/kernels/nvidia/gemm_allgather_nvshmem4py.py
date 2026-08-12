################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
GEMM + AllGather using nvshmem4py (decoupled from triton_dist dependency)

This module implements GEMM + AllGather overlap optimization using nvshmem4py API.
Key features:
1. Uses nvshmem.core API instead of triton_dist.utils
2. Supports multicast via nvshmem.core.get_multicast_tensor()
3. Supports multimem_st instruction for efficient AllGather
"""

import torch
import torch.distributed as dist
import dataclasses
from typing import List, Optional, Tuple
import triton
import triton.language as tl

try:
    import deep_gemm
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False
    deep_gemm = None

# nvshmem4py imports
try:
    import nvshmem.core as nvshmem
    NVSHMEM_AVAILABLE = True
except ImportError:
    NVSHMEM_AVAILABLE = False
    nvshmem = None

try:
    from cuda.core import Device
except ImportError:
    try:
        from cuda.core.experimental import Device
    except ImportError:
        raise RuntimeError(
            "Failed to import Device from cuda.core or cuda.core.experimental. "
            "Please ensure cuda-core is installed: pip install cuda-core"
        )


# ============================================================================
# NVSHMEM Context Manager (decoupled from triton_dist)
# ============================================================================

class NVSHMEMContext:
    """
    NVSHMEM resource manager - decoupled from triton_dist dependency

    Features:
    1. Initialize NVSHMEM via PyTorch ProcessGroup (UID mode)
    2. Automatic resource tracking and cleanup
    3. Support fp8 dtype (automatically converted to int8 storage)

    Correspondence with triton-dist:
    - init_nvshmem_by_torch_process_group → NVSHMEMContext.from_process_group()
    - nvshmem_create_tensor → ctx.create_tensor()
    - nvshmem_create_tensors → ctx.get_peer_tensors()
    - nvshmem_free_tensor_sync → ctx.free_tensor()
    - finalize_distributed → ctx.finalize()
    """

    _instance: Optional['NVSHMEMContext'] = None

    def __init__(self, rank: int, world_size: int, local_rank: int = None):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank if local_rank is not None else rank
        self._allocated_tensors: List[torch.Tensor] = []
        self._finalized = False

    @classmethod
    def from_process_group(cls, pg: dist.ProcessGroup) -> 'NVSHMEMContext':
        """Initialize NVSHMEM from PyTorch ProcessGroup (UID mode)"""
        if not NVSHMEM_AVAILABLE:
            raise RuntimeError("nvshmem4py not available. Please install nvshmem.")

        if cls._instance is not None and not cls._instance._finalized:
            return cls._instance

        torch.cuda.synchronize()

        rank = pg.rank()
        world_size = pg.size()
        local_rank = torch.cuda.current_device()

        # Set device
        dev = Device(local_rank)
        dev.set_current()

        # UID broadcast
        broadcast_objects = [nvshmem.get_unique_id(empty=(rank != 0))]
        dist.broadcast_object_list(
            broadcast_objects,
            src=dist.get_global_rank(pg, 0),
            group=pg
        )
        dist.barrier(group=pg)

        # Initialize NVSHMEM (UID mode)
        nvshmem.init(
            device=dev,
            uid=broadcast_objects[0],
            rank=rank,
            nranks=world_size,
            initializer_method="uid"
        )

        cls._instance = cls(rank, world_size, local_rank)
        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional['NVSHMEMContext']:
        """Get current instance"""
        return cls._instance

    def create_tensor(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Create NVSHMEM symmetric memory tensor"""
        if self._finalized:
            raise RuntimeError("NVSHMEMContext has been finalized")

        torch.cuda.synchronize()

        # NVSHMEM does not support fp8, use int8 for storage
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            tensor = nvshmem.tensor(shape, dtype=torch.int8)
            tensor = tensor.view(dtype)
        else:
            tensor = nvshmem.tensor(shape, dtype=dtype)

        self._allocated_tensors.append(tensor)
        torch.cuda.synchronize()
        return tensor

    def get_peer_tensor(self, tensor: torch.Tensor, peer_rank: int) -> torch.Tensor:
        """Get peer tensor for specified rank"""
        if peer_rank == self.rank:
            return tensor
        return nvshmem.get_peer_tensor(tensor, peer_rank)

    def get_peer_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Get peer tensors for all ranks"""
        return [self.get_peer_tensor(tensor, r) for r in range(self.world_size)]

    def get_multicast_tensor(self, tensor: torch.Tensor, team=None) -> torch.Tensor:
        """Get multicast tensor (requires NVSwitch V3+)"""
        if team is None:
            team = nvshmem.Teams.TEAM_WORLD
        return nvshmem.get_multicast_tensor(nvshmem.Teams.TEAM_WORLD, tensor)

    def free_tensor(self, tensor: torch.Tensor) -> None:
        """Free NVSHMEM tensor"""
        torch.cuda.synchronize()
        # Use data_ptr to check if in allocated list, avoid shape comparison issues
        tensor_ptr = tensor.data_ptr()
        for i, t in enumerate(self._allocated_tensors):
            if t.data_ptr() == tensor_ptr:
                self._allocated_tensors.pop(i)
                break
        nvshmem.free_tensor(tensor)
        torch.cuda.synchronize()

    def barrier_all(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Global barrier synchronization"""
        if stream is None:
            stream = torch.cuda.current_stream()

        # Wrap stream
        class StreamWrapper:
            def __init__(self, pt_stream):
                self.pt_stream = pt_stream
                self.handle = pt_stream.cuda_stream

            def __cuda_stream__(self):
                return (0, self.handle)

        nvshmem.barrier(nvshmem.Teams.TEAM_WORLD, stream=StreamWrapper(stream))

    def finalize(self) -> None:
        """Cleanup NVSHMEM resources - free tracked tensors and finalize NVSHMEM"""
        if self._finalized:
            return

        for tensor in list(self._allocated_tensors):
            try:
                nvshmem.free_tensor(tensor)
            except Exception:
                pass

        self._allocated_tensors.clear()
        torch.cuda.synchronize()
        nvshmem.finalize()

        self._finalized = True
        NVSHMEMContext._instance = None


# ============================================================================
# GEMM + AllGather Context (nvshmem4py version)
# ============================================================================

@dataclasses.dataclass
class GemmAGContextNVSHMEM:
    """
    GEMM + AllGather Context - using nvshmem4py

    Decoupled from triton_dist dependency, uses nvshmem.core API
    """
    rank: int
    num_ranks: int

    symm_gemm_out_buf: torch.Tensor
    symm_residual_out_buf: torch.Tensor
    symm_ag_out_buf: torch.Tensor

    gemm_barrier_buf: torch.Tensor
    multi_st_barrier_buf: torch.Tensor
    grid_barrier_buf: torch.Tensor
    tile_barrier_buf: torch.Tensor

    NUM_COMM_SMS: int
    ag_stream: torch.cuda.Stream
    TILE_MAP_LEVEL: int = 0

    # nvshmem context
    nvshmem_ctx: NVSHMEMContext = None

    # Multicast buffer (optional)
    mc_ag_out_buf: Optional[torch.Tensor] = None

    # Peer tensor pointers for kernel
    peer_ag_out_ptrs: Optional[torch.Tensor] = None
    peer_multi_st_barrier_ptrs: Optional[torch.Tensor] = None

    def finalize(self):
        """Free NVSHMEM resources"""
        if self.nvshmem_ctx:
            self.nvshmem_ctx.free_tensor(self.symm_gemm_out_buf)
            self.nvshmem_ctx.free_tensor(self.symm_residual_out_buf)
            self.nvshmem_ctx.free_tensor(self.symm_ag_out_buf)
            self.nvshmem_ctx.free_tensor(self.gemm_barrier_buf)
            self.nvshmem_ctx.free_tensor(self.multi_st_barrier_buf)

    def get_gemm_out_buf(self, input, weight):
        M, N = input.shape[0], weight.shape[0]
        assert self.symm_gemm_out_buf.numel() >= M * N
        offset = M * N * self.rank
        return self.symm_gemm_out_buf.reshape(-1)[offset : offset + M * N].reshape(M, N)

    def get_direct_gemm_out_buf(self, M, N):
        assert self.symm_gemm_out_buf.numel() >= M * N
        offset = M * N * self.rank
        return self.symm_gemm_out_buf.reshape(-1)[offset : offset + M * N].reshape(M, N)

    def get_ag_out_buf(self, input, weight):
        M, N = input.shape[0], weight.shape[0]
        assert self.symm_ag_out_buf.numel() >= M * N
        offset = M * N * self.rank
        return self.symm_ag_out_buf.reshape(-1)[offset : offset + M * N].reshape(M, N)

    def reset_all_barrier_buf(self):
        self.gemm_barrier_buf.zero_()
        self.tile_barrier_buf.zero_()
        self.grid_barrier_buf.zero_()
        self.multi_st_barrier_buf.zero_()


def create_gemm_ag_context_nvshmem(
    ag_stream: torch.cuda.Stream,
    rank: int,
    world_size: int,
    local_world_size: int,
    max_M: int,
    N: int,
    dtype: torch.dtype,
    MIN_BLOCK_SIZE_M: int = 16,
    MIN_BLOCK_SIZE_N: int = 16,
    NUM_COMM_SMS: int = 16,
    TILE_MAP_LEVEL: int = 0,
    enable_multicast: bool = False,
):
    """
    Create GEMM + AllGather Context - nvshmem4py version

    Args:
        ag_stream: CUDA stream for AllGather
        rank: Current rank
        world_size: Total number of ranks
        local_world_size: Number of ranks per node
        max_M: Maximum M dimension
        N: N dimension
        dtype: Data type
        MIN_BLOCK_SIZE_M: Minimum block size in M dimension
        MIN_BLOCK_SIZE_N: Minimum block size in N dimension
        NUM_COMM_SMS: Number of SMs for communication
        TILE_MAP_LEVEL: Tile mapping level
        enable_multicast: Whether to enable multicast (requires NVSwitch V3+)

    Returns:
        GemmAGContextNVSHMEM instance
    """
    assert local_world_size == world_size, "Only intra-node supported"

    nvshmem_ctx = NVSHMEMContext.get_instance()
    if nvshmem_ctx is None:
        raise RuntimeError("NVSHMEMContext not initialized. Call NVSHMEMContext.from_process_group() first.")

    # Allocate symmetric memory
    gemm_out_buf = nvshmem_ctx.create_tensor((world_size, max_M, N), dtype)
    residual_out_buf = nvshmem_ctx.create_tensor((world_size, max_M, N), dtype)
    symm_ag_out_buf = nvshmem_ctx.create_tensor((max_M * world_size, N), dtype)
    gemm_barrier_buf = nvshmem_ctx.create_tensor(
        (world_size, triton.cdiv(max_M, MIN_BLOCK_SIZE_M), triton.cdiv(N, MIN_BLOCK_SIZE_N)),
        torch.int32
    )
    multi_st_barrier_buf = nvshmem_ctx.create_tensor((world_size * NUM_COMM_SMS,), torch.int32)
    grid_barrier_buf = torch.zeros((1,), dtype=torch.int32, device=torch.cuda.current_device())
    tile_barrier_buf = torch.zeros(
        (world_size, triton.cdiv(max_M, MIN_BLOCK_SIZE_M), triton.cdiv(N, MIN_BLOCK_SIZE_N)),
        dtype=torch.int32,
        device=torch.cuda.current_device()
    )

    # Initialize
    gemm_out_buf.zero_()
    residual_out_buf.zero_()
    gemm_barrier_buf.zero_()
    multi_st_barrier_buf.zero_()

    nvshmem_ctx.barrier_all()

    # Get multicast buffer (optional)
    mc_ag_out_buf = None
    if enable_multicast:
        try:
            mc_ag_out_buf = nvshmem_ctx.get_multicast_tensor(symm_ag_out_buf)
        except Exception as e:
            print(f"Warning: Multicast not supported: {e}")

    # Get peer tensor pointer array (for dynamic access in kernel)
    peer_ag_out_ptrs = torch.tensor(
        [nvshmem_ctx.get_peer_tensor(symm_ag_out_buf, r).data_ptr() for r in range(world_size)],
        dtype=torch.int64,
        device=torch.cuda.current_device()
    )
    peer_multi_st_barrier_ptrs = torch.tensor(
        [nvshmem_ctx.get_peer_tensor(multi_st_barrier_buf, r).data_ptr() for r in range(world_size)],
        dtype=torch.int64,
        device=torch.cuda.current_device()
    )

    ctx = GemmAGContextNVSHMEM(
        rank=rank,
        num_ranks=world_size,
        symm_gemm_out_buf=gemm_out_buf,
        symm_residual_out_buf=residual_out_buf,
        symm_ag_out_buf=symm_ag_out_buf,
        gemm_barrier_buf=gemm_barrier_buf,
        multi_st_barrier_buf=multi_st_barrier_buf,
        grid_barrier_buf=grid_barrier_buf,
        tile_barrier_buf=tile_barrier_buf,
        NUM_COMM_SMS=NUM_COMM_SMS,
        ag_stream=ag_stream,
        TILE_MAP_LEVEL=TILE_MAP_LEVEL,
        nvshmem_ctx=nvshmem_ctx,
        mc_ag_out_buf=mc_ag_out_buf,
        peer_ag_out_ptrs=peer_ag_out_ptrs,
        peer_multi_st_barrier_ptrs=peer_multi_st_barrier_ptrs,
    )

    return ctx


# ============================================================================
# Triton Kernels (using standard triton.jit, decoupled from triton_dist.jit)
# ============================================================================

@triton.jit
def get_swizzled_block_idx(block_idx, num_pid_m, num_pid_n,
                           kNumMulticast=2, kIsMulticastOnA=True, kNum1DBlocksPerGroup=8, sm90_odd_fix=True):
    primary_num_blocks   = num_pid_n if kIsMulticastOnA else num_pid_m
    secondary_num_blocks = num_pid_m if kIsMulticastOnA else num_pid_n

    num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup

    group_idx = block_idx // num_blocks_per_group
    first_block_idx = group_idx * kNum1DBlocksPerGroup
    in_group_idx = block_idx % num_blocks_per_group

    num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx)

    # SM90 odd fix: when multicast>1 and group primary count is odd, split into (even part) + (tail 1)
    if sm90_odd_fix and kNumMulticast > 1 and (num_blocks_in_group % 2 == 1):
        # same logic as scheduler.cuh
        if in_group_idx < ((num_blocks_in_group ^ 1) * secondary_num_blocks):
            num_blocks_in_group = (num_blocks_in_group ^ 1)
        else:
            in_group_idx = in_group_idx - ((num_blocks_in_group ^ 1) * secondary_num_blocks)
            first_block_idx += (num_blocks_in_group ^ 1)
            num_blocks_in_group = 1

    # convert to (m,n)
    if kIsMulticastOnA:
        m_block_idx = in_group_idx // num_blocks_in_group
        n_block_idx = first_block_idx + (in_group_idx % num_blocks_in_group)
    else:
        m_block_idx = first_block_idx + (in_group_idx % num_blocks_in_group)
        n_block_idx = in_group_idx // num_blocks_in_group

    return m_block_idx, n_block_idx


@triton.jit
def load_v4_b32(ptr):
    """Vectorized load 4x int32, aligned with language_extra.py load_v4_b32 (no scope/semantic, i.e. ld.global.v4.b32).
    In @triton.jit, tl.constexpr string concatenation is unreliable, so PTX is fixed.
    For scope/semantic, please use the version in language_extra.py directly.
    """
    val = tl.inline_asm_elementwise(
        asm="ld.global.v4.b32 {$0,$1,$2,$3}, [$4];",
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1
    )
    return val


@triton.jit
def st_v4_b32(ptr, val0, val1, val2, val3):
    """Vectorized store 4x int32, aligned with language_extra.py st_v4_b32 (no scope/semantic, i.e. st.global.v4.b32).
    In @triton.jit, tl.constexpr string concatenation is unreliable, so PTX is fixed.
    """
    tl.inline_asm_elementwise(
        asm="""
        st.global.v4.b32 [$1], {$2,$3,$4,$5};
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r,r,r,r"),  # no use output
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def multimem_st_v4(ptr, val0, val1, val2, val3, suffix: tl.constexpr):
    """Multimem store 4x int32 (NVSwitch multicast), aligned with language_extra.py _multimem_st_v4_impl.

    suffix must match the element type of ptr (compile-time constant):
      - float32  → "f32"
      - bfloat16 → "bf16x2"
      - float16  → "f16x2"
    PTX ISA: multimem.st's v4 mode only supports f32/f16x2/bf16x2, not b32.
    """
    tl.static_assert(
        suffix == "f32" or suffix == "bf16x2" or suffix == "f16x2",
        "multimem.st.v4 only supports f32, bf16x2, f16x2"
    )
    tl.inline_asm_elementwise(
        asm=f"""
        multimem.st.global.v4.{suffix} [$1], {{$2, $3, $4, $5}};
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r,r,r,r"),  # no use output
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def ld_sys(ptr):
    """Load with sys scope (acquire semantic), aligned with language_extra.py ld(scope="sys", semantic="acquire")"""
    return tl.inline_asm_elementwise(
        asm="ld.global.acquire.sys.b32 $0, [$1];",
        constraints=("=r,l"),
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def ld_gpu(ptr):
    """Load with gpu scope (relaxed semantic), aligned with language_extra.py ld(scope="gpu", semantic="relaxed")"""
    return tl.inline_asm_elementwise(
        asm="ld.global.relaxed.gpu.b32 $0, [$1];",
        constraints=("=r,l"),
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def st_sys(ptr, val):
    """Store with sys scope (release semantic), aligned with language_extra.py st(scope="sys", semantic="release")"""
    tl.inline_asm_elementwise(
        asm="""
        st.global.release.sys.b32 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r"),
        args=[ptr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def st_gpu(ptr, val):
    """Store with gpu scope (relaxed semantic), aligned with language_extra.py st(scope="gpu", semantic="relaxed")"""
    tl.inline_asm_elementwise(
        asm="""
        st.global.relaxed.gpu.b32 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r"),
        args=[ptr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def __syncthreads():
    """Block synchronization - equivalent to CUDA __syncthreads(), using PTX bar.sync"""
    tl.inline_asm_elementwise(
        asm="bar.sync 0;",
        constraints=("=r"),
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def tid(axis: tl.constexpr = 0):
    """Get thread index (threadIdx.x/y/z)"""
    if axis == 0:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1
        )
    elif axis == 1:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.y;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1
        )
    else:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.z;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1
        )


@triton.jit
def num_warps():
    """Get number of warps"""
    # Triton does not provide a direct way to get num_warps
    # Here we assume it's a compile-time constant, should be passed in kernel arguments
    return 32  # Default value, should be determined at compile time


@triton.jit
def consumer_all_gather_kernel_nvshmem(
    symm_input_ptr,
    symm_ag_out_ptr,
    ag_out_ptr,
    gemm_barrier_ptr,
    multi_st_barrier_ptr,
    peer_ag_out_ptrs,  # GPU-side pointer array
    peer_multi_st_barrier_ptrs,  # GPU-side barrier pointer array
    mc_ag_out_ptr,  # Multicast pointer (optional)
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
    USE_MULTIMEM_ST: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """
    Consumer AllGather Kernel - nvshmem4py version

    Key improvements:
    1. Uses standard @triton.jit instead of @triton_dist.jit
    2. Passes rank/world_size as parameters instead of dl.rank()/dl.num_ranks()
    3. Uses pointer arrays instead of dl.symm_at() for dynamic access
    4. Supports multimem_st (multicast)
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    thread_idx = tid(0)
    block_dim = 32 * 32  # num_warps() * 32

    # Vectorized size
    VEC_SIZE: tl.constexpr = 128 // tl.constexpr(symm_input_ptr.dtype.element_ty.primitive_bitwidth)
    ELEM_BYTES: tl.constexpr = tl.constexpr(symm_input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    tl.static_assert(BLOCK_SIZE_N % VEC_SIZE == 0)

    # Determine PTX suffix for multimem.st based on element type (compile-time constant)
    # PTX ISA: multimem.st.v4 only supports f32/f16x2/bf16x2
    if tl.constexpr(symm_input_ptr.dtype.element_ty == tl.float32):
        MULTIMEM_SUFFIX: tl.constexpr = "f32"
    elif tl.constexpr(symm_input_ptr.dtype.element_ty == tl.bfloat16):
        MULTIMEM_SUFFIX: tl.constexpr = "bf16x2"
    elif tl.constexpr(symm_input_ptr.dtype.element_ty == tl.float16):
        MULTIMEM_SUFFIX: tl.constexpr = "f16x2"
    else:
        tl.static_assert(False, "multimem_st_v4 only supports float32/bfloat16/float16")

    if not USE_MULTIMEM_ST:
        # Non-multicast mode: use peer tensor pointer arrays
        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            pid_m, pid_n = get_swizzled_block_idx(tile_id, num_pid_m, num_pid_n)

            if thread_idx == 0:
                gemm_barrier_idx = pid_m * num_pid_n + pid_n
                while ld_sys(gemm_barrier_ptr + gemm_barrier_idx) != 1:
                    pass

            __syncthreads()

            tile_m = tl.minimum(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            tile_n = tl.minimum(N - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)
            VEC_PER_ROW = tile_n // VEC_SIZE
            cur_tile_nelem = tile_m * tile_n

            for idx in range(thread_idx, cur_tile_nelem // VEC_SIZE, block_dim):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                # offset: element offset (in original dtype units)
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                # load_v4_b32/st_v4_b32 takes byte addresses, convert element offset to byte offset
                byte_offset = offset * ELEM_BYTES
                # Byte base address of symm_input_ptr
                src_byte_addr = symm_input_ptr.to(tl.int64) + byte_offset

                val0, val1, val2, val3 = load_v4_b32(src_byte_addr.to(tl.pointer_type(tl.int32)))

                # Broadcast to all peers: write to peer's symm_ag_out[rank*M:(rank+1)*M, :]
                dst_byte_base = rank * M * N * ELEM_BYTES + byte_offset
                for peer_rank in range(world_size):
                    peer_ptr = tl.load(peer_ag_out_ptrs + peer_rank)  # int64 byte address
                    st_v4_b32((peer_ptr + dst_byte_base).to(tl.pointer_type(tl.int32)), val0, val1, val2, val3)

    else:
        # Multicast mode: use multimem_st to write to all peers at once
        for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
            pid_m, pid_n = get_swizzled_block_idx(tile_id, num_pid_m, num_pid_n)

            if thread_idx == 0:
                gemm_barrier_idx = pid_m * num_pid_n + pid_n
                while ld_sys(gemm_barrier_ptr + gemm_barrier_idx) != 1:
                    pass

            __syncthreads()

            tile_m = tl.minimum(M - pid_m * BLOCK_SIZE_M, BLOCK_SIZE_M)
            tile_n = tl.minimum(N - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)
            VEC_PER_ROW = tile_n // VEC_SIZE
            cur_tile_nelem = tile_m * tile_n

            for idx in range(thread_idx, cur_tile_nelem // VEC_SIZE, block_dim):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                # offset is element offset within symm_input (M, N)
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                byte_offset = offset * ELEM_BYTES
                src_byte_addr = symm_input_ptr.to(tl.int64) + byte_offset

                val0, val1, val2, val3 = load_v4_b32(src_byte_addr.to(tl.pointer_type(tl.int32)))
                # mc_ag_out is the multicast pointer of symm_ag_out (world_size*M, N)
                # Each rank writes to its own row region, mc_ag_out_ptr is byte address
                mc_byte_offset = (rank * M * N + offset) * ELEM_BYTES
                multimem_st_v4(mc_ag_out_ptr + mc_byte_offset, val0, val1, val2, val3, MULTIMEM_SUFFIX)

        __syncthreads()

        # Barrier: wait for all peers' multimem_st to complete
        if thread_idx < world_size:
            peer_ptr = tl.load(peer_multi_st_barrier_ptrs + thread_idx).to(tl.pointer_type(tl.int32))
            st_sys(peer_ptr + rank * NUM_COMM_SMS + pid, 1)

        if thread_idx < world_size:
            multi_st_barrier_idx = thread_idx * NUM_COMM_SMS + pid
            while ld_sys(multi_st_barrier_ptr + multi_st_barrier_idx) != 1:
                pass
            st_sys(multi_st_barrier_ptr + multi_st_barrier_idx, 0)

    # Reset barriers
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        pid_m, pid_n = get_swizzled_block_idx(tile_id, num_pid_m, num_pid_n)
        gemm_barrier_idx = pid_m * num_pid_n + pid_n
        if thread_idx == 0:
            st_sys(gemm_barrier_ptr + gemm_barrier_idx, 0)


# ============================================================================
# Host Functions
# ============================================================================

def consumer_all_gather_nvshmem(
    symm_input,
    symm_ag_out,
    ag_out,
    gemm_barrier,
    multi_st_barrier,
    peer_ag_out_ptrs,
    peer_multi_st_barrier_ptrs,
    mc_ag_out,
    BLOCK_SIZE_M=16,
    BLOCK_SIZE_N=64,
    NUM_COMM_SMS=16,
    USE_MULTIMEM_ST=False,
    rank=0,
    world_size=1,
):
    """Launch consumer all-gather kernel"""
    M, N = symm_input.shape

    grid = (NUM_COMM_SMS,)
    consumer_all_gather_kernel_nvshmem[grid](
        symm_input,
        symm_ag_out,
        ag_out,
        gemm_barrier,
        multi_st_barrier,
        peer_ag_out_ptrs,
        peer_multi_st_barrier_ptrs,
        mc_ag_out.data_ptr() if mc_ag_out is not None else 0,
        M,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_COMM_SMS=NUM_COMM_SMS,
        USE_MULTIMEM_ST=USE_MULTIMEM_ST,
        rank=rank,
        world_size=world_size,
        num_warps=32
    )


# ============================================================================
# GEMM Kernels (nvshmem4py version)
# ============================================================================

@triton.jit
def kernel_persistent_gemm_notify_nvshmem(
    a_ptr, b_ptr, c_ptr,
    gemm_barrier_ptr, tile_barrier_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_GEMM_SMS: tl.constexpr,
    rank: tl.constexpr,
    TILE_MAP_LEVEL: tl.constexpr = 0,
):
    """
    Persistent GEMM Kernel (with notification mechanism) - nvshmem4py version

    Notifies communication kernel via barrier after computation is done
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS):
        # Swizzle tile mapping
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (tile_id % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # K-dimension loop
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator += tl.dot(a, b.T)

        # Store result
        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

        # Notify communication kernel via barrier
        # thread 0 of each block notifies after each tile is done (matches triton_dist original)
        thread_idx = tid(0)
        if thread_idx == 0:
            if TILE_MAP_LEVEL == 0:  # tile_wise_map_to_comm
                gemm_barrier_idx = pid_m * num_pid_n + pid_n
                st_gpu(gemm_barrier_ptr + gemm_barrier_idx, 1)


def persistent_gemm_notify_nvshmem(a, b, out, gemm_barrier, tile_barrier, gemm_config, rank=0):
    """
    Host function to launch persistent GEMM kernel - nvshmem4py version
    """
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Check constraints
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, _ = b.shape

    grid = lambda META: (
        min(META["NUM_GEMM_SMS"],
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )

    kernel_persistent_gemm_notify_nvshmem[grid](
        a, b, out, gemm_barrier, tile_barrier,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        **gemm_config.all_kwargs(),
        rank=rank,
    )

    return out


# ============================================================================
# High-Level Operations (nvshmem4py version)
# ============================================================================

def allgather_op_nvshmem(
    ctx: GemmAGContextNVSHMEM,
    c: torch.Tensor,
    BLOCK_SIZE_M: int = 16,
    BLOCK_SIZE_N: int = 64,
    NUM_COMM_SMS: int = 16,
    USE_MULTIMEM_ST: bool = False,
    copy_to_local: bool = True,
):
    """
    Execute AllGather operation - nvshmem4py version

    Args:
        ctx: GemmAGContextNVSHMEM context
        c: Input tensor [M, N]
        BLOCK_SIZE_M: Block size in M dimension
        BLOCK_SIZE_N: Block size in N dimension
        NUM_COMM_SMS: Number of SMs for communication
        USE_MULTIMEM_ST: Whether to use multicast
        copy_to_local: Whether to copy to local buffer

    Returns:
        ag_out: AllGather result [M * world_size, N]
    """
    M, N = c.shape

    # Get current rank's buffer
    offset = M * N * ctx.rank
    symm_c = ctx.symm_gemm_out_buf.reshape(-1)[offset : offset + M * N].reshape(M, N)
    symm_c.copy_(c)

    symm_ag_out = ctx.symm_ag_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    peer_ag_out_ptrs = ctx.peer_ag_out_ptrs
    peer_multi_st_barrier_ptrs = ctx.peer_multi_st_barrier_ptrs
    mc_ag_out = ctx.mc_ag_out_buf

    ag_out = torch.empty((M * ctx.num_ranks, N), dtype=c.dtype, device=c.device)

    # Set barrier to 1 (indicating data is ready)
    gemm_barrier.fill_(1)

    # Execute AllGather
    consumer_all_gather_nvshmem(
        symm_c,
        symm_ag_out,
        ag_out,
        gemm_barrier,
        multi_st_barrier,
        peer_ag_out_ptrs,
        peer_multi_st_barrier_ptrs,
        mc_ag_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_COMM_SMS=NUM_COMM_SMS,
        USE_MULTIMEM_ST=USE_MULTIMEM_ST,
        rank=ctx.rank,
        world_size=ctx.num_ranks,
    )

    # If using multimem_st and need to copy to local
    if USE_MULTIMEM_ST and copy_to_local:
        ag_out.copy_(symm_ag_out.reshape(-1)[:M * ctx.num_ranks * N].reshape(M * ctx.num_ranks, N))

    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ag_out.reshape(-1)[:M * ctx.num_ranks * N].reshape(M * ctx.num_ranks, N)

    return ag_out



def gemm_op_nvshmem(
    ctx: GemmAGContextNVSHMEM,
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_config: triton.Config,
    As: Optional[torch.Tensor] = None,
    Bs: Optional[torch.Tensor] = None,
):
    """
    Execute pure GEMM operation (without AllGather) - nvshmem4py version

    Args:
        ctx: GemmAGContextNVSHMEM context
        a: Input tensor [M, K]
        b: Weight tensor [N, K]
        gemm_config: Triton GEMM configuration
        As: A's scale (for int8)
        Bs: B's scale (for int8)

    Returns:
        out: GEMM result [M, N]
    """
    M, N = a.shape[0], b.shape[0]

    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(a, b)
    gemm_barrier = ctx.gemm_barrier_buf
    tile_barrier = ctx.tile_barrier_buf

    with_scale = (As is not None and Bs is not None)

    if with_scale:
        assert a.dtype == torch.int8
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    else:
        out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    persistent_gemm_notify_nvshmem(a, b, symm_c, gemm_barrier, tile_barrier, gemm_config, rank=ctx.rank)
    out.copy_(symm_c.reshape(-1)[:M * N].reshape(M, N))

    return out


def gemm_allgather_op_nvshmem(
    ctx: GemmAGContextNVSHMEM,
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_config: triton.Config,
    copy_to_local: bool = True,
    USE_MULTIMEM_ST: bool = False,
    As: Optional[torch.Tensor] = None,
    Bs: Optional[torch.Tensor] = None,
):
    """
    Execute GEMM + AllGather fused operation - nvshmem4py version

    Args:
        ctx: GemmAGContextNVSHMEM context
        a: Input tensor [M, K]
        b: Weight tensor [N, K]
        gemm_config: Triton GEMM configuration
        copy_to_local: Whether to copy to local buffer
        USE_MULTIMEM_ST: Whether to use multicast
        As: A's scale (for int8)
        Bs: B's scale (for int8)

    Returns:
        ag_out: AllGather result [M * world_size, N]
    """
    M, N = a.shape[0], b.shape[0]
    NUM_COMM_SMS = ctx.NUM_COMM_SMS
    BLOCK_SIZE_M = gemm_config.all_kwargs()["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.all_kwargs()["BLOCK_SIZE_N"]

    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(a, b)
    symm_ag_out = ctx.symm_ag_out_buf
    gemm_barrier = ctx.gemm_barrier_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    peer_ag_out_ptrs = ctx.peer_ag_out_ptrs
    peer_multi_st_barrier_ptrs = ctx.peer_multi_st_barrier_ptrs
    mc_ag_out = ctx.mc_ag_out_buf
    tile_barrier = ctx.tile_barrier_buf

    with_scale = (As is not None and Bs is not None)

    if with_scale:
        assert a.dtype == torch.int8
        ag_out = torch.empty((M * ctx.num_ranks, N), dtype=torch.bfloat16, device=a.device)
    else:
        ag_out = torch.empty((M * ctx.num_ranks, N), dtype=a.dtype, device=a.device)

    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream
    ag_stream.wait_stream(current_stream)

    if not USE_MULTIMEM_ST:  # Multimem kernel will reset barrier inside the ar kernel
        ctx.nvshmem_ctx.barrier_all(current_stream)
        ctx.reset_all_barrier_buf()
        ctx.nvshmem_ctx.barrier_all(current_stream)

    # Launch GEMM kernel
    persistent_gemm_notify_nvshmem(a, b, symm_c, gemm_barrier, tile_barrier, gemm_config, rank=ctx.rank)

    # Launch AllGather kernel in separate stream
    with torch.cuda.stream(ag_stream):
        consumer_all_gather_nvshmem(
            symm_c,
            symm_ag_out,
            ag_out,
            gemm_barrier,
            multi_st_barrier,
            peer_ag_out_ptrs,
            peer_multi_st_barrier_ptrs,
            mc_ag_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            NUM_COMM_SMS=NUM_COMM_SMS,
            USE_MULTIMEM_ST=USE_MULTIMEM_ST,
            rank=ctx.rank,
            world_size=ctx.num_ranks,
        )

    current_stream.wait_stream(ag_stream)

    # Copy to local buffer if needed
    if USE_MULTIMEM_ST and copy_to_local:
        ag_out.copy_(symm_ag_out.reshape(-1)[:M * ctx.num_ranks * N].reshape(M * ctx.num_ranks, N))

    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ag_out.reshape(-1)[:M * ctx.num_ranks * N].reshape(M * ctx.num_ranks, N)

    return ag_out


def deepgemm_allgather_op_nvshmem(
    ctx: GemmAGContextNVSHMEM,
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    gemm_config: triton.Config,
    copy_to_local: bool = True,
    USE_MULTIMEM_ST: bool = False,
):
    """
    Execute DeepGEMM (FP8) + AllGather fused operation - nvshmem4py version

    Uses deep_gemm.fp8_gemm_nt instead of Triton GEMM kernel, with enable_overlap=True
    and signal parameter to implement tile-level overlap with AllGather consumer kernel.

    Args:
        ctx: GemmAGContextNVSHMEM context
        a: (fp8_tensor [M, K], scale_tensor) input tensor and its scale
        b: (fp8_tensor [N, K], scale_tensor) weight tensor and its scale
        gemm_config: Triton GEMM configuration (for extracting BLOCK_SIZE_M/N)
        copy_to_local: Whether to copy result to local buffer (effective when USE_MULTIMEM_ST=True)
        USE_MULTIMEM_ST: Whether to use multicast store for AllGather

    Returns:
        ag_out: AllGather result [M * world_size, N], dtype=bfloat16
    """
    if not DEEP_GEMM_AVAILABLE:
        raise RuntimeError("deep_gemm is not installed. Please install deep_gemm to use deepgemm_allgather_op_nvshmem.")

    M, N = a[0].shape[0], b[0].shape[0]
    NUM_COMM_SMS = ctx.NUM_COMM_SMS
    BLOCK_SIZE_M = gemm_config.all_kwargs()["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.all_kwargs()["BLOCK_SIZE_N"]

    assert a[0].shape[1] == b[0].shape[1], "Incompatible dimensions"
    assert a[0].dtype == b[0].dtype, "Incompatible dtypes"

    # symm_c output is bfloat16 (deep_gemm fp8_gemm_nt outputs bf16)
    symm_c = ctx.get_gemm_out_buf(a[0], b[0])
    symm_ag_out = ctx.symm_ag_out_buf
    multi_st_barrier = ctx.multi_st_barrier_buf
    peer_ag_out_ptrs = ctx.peer_ag_out_ptrs
    peer_multi_st_barrier_ptrs = ctx.peer_multi_st_barrier_ptrs
    mc_ag_out = ctx.mc_ag_out_buf

    # deep_gemm outputs bfloat16
    ag_out = torch.empty((M * ctx.num_ranks, N), dtype=symm_c.dtype, device=symm_c.device)
    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream
    ag_stream.wait_stream(current_stream)

    if not USE_MULTIMEM_ST:  # Multimem kernel will reset barrier inside the ar kernel
        ctx.nvshmem_ctx.barrier_all(current_stream)
        ctx.reset_all_barrier_buf()
        ctx.nvshmem_ctx.barrier_all(current_stream)

    # Launch DeepGEMM, enable_overlap=True makes it write signal after each tile completes
    deep_gemm.fp8_gemm_nt(
        a, b, symm_c,
        c=None,
        disable_ue8m0_cast=True,
        recipe=None,
        enable_overlap=True,
        signal=ctx.gemm_barrier_buf,
    )

    # Launch AllGather consumer kernel on separate stream
    with torch.cuda.stream(ag_stream):
        consumer_all_gather_nvshmem(
            symm_c,
            symm_ag_out,
            ag_out,
            ctx.gemm_barrier_buf,
            multi_st_barrier,
            peer_ag_out_ptrs,
            peer_multi_st_barrier_ptrs,
            mc_ag_out,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            NUM_COMM_SMS=NUM_COMM_SMS,
            USE_MULTIMEM_ST=USE_MULTIMEM_ST,
            rank=ctx.rank,
            world_size=ctx.num_ranks,
        )

    current_stream.wait_stream(ag_stream)

    # Copy to local buffer if needed
    if USE_MULTIMEM_ST and copy_to_local:
        ag_out.copy_(symm_ag_out.reshape(-1)[:M * ctx.num_ranks * N].reshape(M * ctx.num_ranks, N))

    if USE_MULTIMEM_ST and not copy_to_local:
        return symm_ag_out.reshape(-1)[:M * ctx.num_ranks * N].reshape(M * ctx.num_ranks, N)

    return ag_out
