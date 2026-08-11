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
GEMM + AllGather using PyTorch symmetric_memory (decoupled from triton_dist dependency)

This module implements GEMM + AllGather overlap optimization using PyTorch's
native torch.distributed._symmetric_memory API (CUDA backend).
Key features:
1. Uses torch.distributed._symmetric_memory API instead of triton_dist.utils
2. Supports multicast via handle.multicast_ptr (requires NVSwitch V3+)
3. Supports multimem_st instruction for efficient AllGather

Note on backend choice (PyTorch 2.9 / 2.10):
- CUDA backend  : get_multicast_ptr() implemented (returns mc_addr_ from cuMulticast*)
- NVSHMEM backend: get_multicast_ptr() is // TODO -> nullptr (stub)
- NCCL backend  : get_multicast_ptr() is // TODO -> nullptr (stub)
Therefore we explicitly set_backend("CUDA") so multicast actually works.

Differences from the nvshmem4py version (semantically equivalent):
- symm_mem.empty() + symm_mem.rendezvous() instead of nvshmem.tensor()
- handle.get_buffer(peer, shape, dtype) instead of nvshmem.get_peer_tensor(t, r)
- handle.buffer_ptrs[r] (int64 device ptr) for in-kernel peer access
- handle.multicast_ptr (int64 mc VA) instead of nvshmem.get_multicast_tensor(t)
- handle.barrier() (stream-bound, signal_pad based) instead of nvshmem.barrier()
- Explicit free is a no-op (Python GC reclaims symm_mem buffers)
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
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


# ============================================================================
# SymmetricMemory Context Manager (PyTorch native API)
# ============================================================================

class SymmMemContext:
    """
    Symmetric Memory resource manager using PyTorch native API

    Features:
    1. Initialize symmetric memory via PyTorch ProcessGroup
       (set_backend("CUDA"))
    2. Allocate via symm_mem.empty() + symm_mem.rendezvous()
    3. Peer access via handle.buffer_ptrs
    4. Multicast support via handle.multicast_ptr
    5. No explicit free - tensors are garbage collected by Python

    Correspondence with NVSHMEMContext (nvshmem4py version):
    - from_process_group     → SymmMemContext.from_process_group()
    - create_tensor          → ctx.create_tensor()   (symm_mem.empty + rendezvous)
    - get_peer_tensor(s)     → ctx.get_peer_tensor(s) (handle.buffer_ptrs)
    - get_multicast_tensor   → ctx.get_multicast_tensor() (handle.multicast_ptr)
    - free_tensor            → ctx.free_tensor()  (no-op, GC managed)
    - barrier_all            → ctx.barrier_all()  (handle.barrier or dist.barrier)
    - finalize               → ctx.finalize()     (no-op)
    """

    _instance: Optional['SymmMemContext'] = None

    def __init__(self, rank: int, world_size: int, local_rank: int, group_name: str):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.group_name = group_name
        # Keep tensor references alive (symm_mem allocations are tracked via Python GC)
        self._allocated_tensors: List[torch.Tensor] = []
        # Cache rendezvous handles keyed by tensor data_ptr to avoid repeated rendezvous calls
        self._handle_cache: dict = {}
        self._finalized = False

    @classmethod
    def from_process_group(cls, pg: dist.ProcessGroup) -> 'SymmMemContext':
        """Initialize symmetric memory from PyTorch ProcessGroup.

        Note: torch.distributed._symmetric_memory.empty() does NOT accept a
        group_name argument; the underlying NVSHMEM backend uses the WORLD
        group by default. We use the WORLD group_name for rendezvous.
        This matches the pattern in pytorch/test/distributed/
        test_symmetric_memory.py (group_name = dist.group.WORLD.group_name).
        """
        if cls._instance is not None and not cls._instance._finalized:
            return cls._instance

        torch.cuda.synchronize()

        rank = pg.rank()
        world_size = pg.size()
        local_rank = torch.cuda.current_device()

        # Select the symmetric memory backend. On PyTorch 2.9/2.10 only the
        # CUDA backend actually implements get_multicast_ptr(); the NVSHMEM
        # backend is a stub (returns nullptr). Use CUDA backend to enable
        # multicast on NVSwitch V3+ hardware.
        try:
            symm_mem.set_backend("CUDA")
        except RuntimeError:
            pass

        # symm_mem.empty() implicitly uses WORLD; use WORLD group for rendezvous
        world_group_name = dist.group.WORLD.group_name
        cls._instance = cls(rank, world_size, local_rank, world_group_name)
        return cls._instance

    @classmethod
    def get_instance(cls) -> Optional['SymmMemContext']:
        """Get current instance"""
        return cls._instance

    def _get_handle(self, tensor: torch.Tensor):
        """Get (and cache) rendezvous handle for a symmetric tensor."""
        key = tensor.data_ptr()
        hdl = self._handle_cache.get(key)
        if hdl is None:
            hdl = symm_mem.rendezvous(tensor, group=self.group_name)
            self._handle_cache[key] = hdl
        return hdl

    def create_tensor(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """Create symmetric memory tensor via symm_mem.empty() + rendezvous().

        Eagerly performs rendezvous so that the returned tensor is immediately
        usable for peer access via get_peer_tensor / get_peer_buffer_ptrs /
        get_multicast_tensor. PyTorch docs require empty() and rendezvous() to
        be called in the same order across all ranks - we enforce that by
        rendezvousing right after allocation.
        """
        if self._finalized:
            raise RuntimeError("SymmMemContext has been finalized")

        dev = torch.device(f"cuda:{self.local_rank}")

        # NVSHMEM does not support fp8, use int8 for storage
        if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            tensor = symm_mem.empty(*shape, dtype=torch.int8, device=dev)
            tensor = tensor.view(dtype)
        else:
            tensor = symm_mem.empty(*shape, dtype=dtype, device=dev)

        # Eager rendezvous - ensures same call order on all ranks
        self._get_handle(tensor)
        self._allocated_tensors.append(tensor)
        return tensor

    def get_peer_tensor(self, tensor: torch.Tensor, peer_rank: int) -> torch.Tensor:
        """Get peer tensor for specified rank (host-side view).

        Uses handle.get_buffer(peer_rank, shape, dtype) which returns a tensor
        view backed by the peer's symmetric buffer. The returned tensor can be
        read/written from host just like the local symmetric tensor.
        """
        hdl = self._get_handle(tensor)
        # NVSHMEM does not support fp8, the underlying allocation uses int8.
        # Return as the requested dtype via view.
        if tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            buf = hdl.get_buffer(peer_rank, tensor.shape, torch.int8)
            return buf.view(tensor.dtype)
        return hdl.get_buffer(peer_rank, tensor.shape, tensor.dtype)

    def get_peer_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Get peer tensors for all ranks (returns list of length world_size)."""
        return [self.get_peer_tensor(tensor, r) for r in range(self.world_size)]

    def get_peer_buffer_ptrs(self, tensor: torch.Tensor) -> List[int]:
        """Get peer buffer pointers for all ranks (PyTorch symm_mem specific).

        This is the recommended way to access peer buffers - the int64 pointers
        are passed into kernels for device-side peer access.
        """
        hdl = self._get_handle(tensor)
        return list(hdl.buffer_ptrs)

    def get_multicast_tensor(self, tensor: torch.Tensor) -> int:
        """Get multicast pointer for a symmetric tensor (requires NVSwitch V3+).

        Returns the multicast pointer as int64 (0 if multicast not supported).
        """
        hdl = self._get_handle(tensor)
        return hdl.multicast_ptr

    def free_tensor(self, tensor: torch.Tensor) -> None:
        """Drop internal reference to a symmetric tensor.

        PyTorch symm_mem relies on Python GC for actual deallocation, so this
        only removes the context's own reference and cached handle. Callers
        must drop their own references to trigger reclamation.
        Provided for API compatibility with NVSHMEMContext.
        """
        tensor_ptr = tensor.data_ptr()
        for i, t in enumerate(self._allocated_tensors):
            if t.data_ptr() == tensor_ptr:
                self._allocated_tensors.pop(i)
                break
        self._handle_cache.pop(tensor_ptr, None)

    def barrier_all(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Global barrier synchronization (stream-bound, CUDA Graph compatible).

        Uses handle.barrier() which enqueues a device-side barrier on the
        current stream (based on signal_pad), matching the semantics of
        nvshmem.barrier(stream).

        Requires at least one symmetric tensor to have been rendezvoused; if
        none exists yet, falls back to host-side dist.barrier().
        """
        # Pick any rendezvoused handle to issue the barrier on
        if self._handle_cache:
            hdl = next(iter(self._handle_cache.values()))
            if stream is not None:
                with torch.cuda.stream(stream):
                    hdl.barrier()
            else:
                hdl.barrier()
        else:
            # No symmetric tensor yet - fall back to host barrier
            if stream is not None:
                stream.synchronize()
            torch.cuda.synchronize()
            dist.barrier()

    def finalize(self) -> None:
        """Cleanup - drop tensor references; symm_mem uses Python GC."""
        if self._finalized:
            return

        self._allocated_tensors.clear()
        self._handle_cache.clear()
        torch.cuda.synchronize()

        self._finalized = True
        SymmMemContext._instance = None


# ============================================================================
# GEMM + AllGather Context (PyTorch symmetric_memory version)
# ============================================================================

@dataclasses.dataclass
class GemmAGContextPyTorch:
    """
    GEMM + AllGather Context - using PyTorch symmetric_memory API

    Decoupled from triton_dist dependency, uses
    torch.distributed._symmetric_memory API (PyTorch native CUDA backend).
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

    # Symmetric memory context
    symm_mem_ctx: SymmMemContext = None

    # Multicast buffer (optional, int64 ptr from hdl.multicast_ptr; 0 if unsupported)
    mc_ag_out_buf: Optional[int] = None

    # Peer tensor pointers for kernel (device-side int64 arrays)
    peer_ag_out_ptrs: Optional[torch.Tensor] = None
    peer_multi_st_barrier_ptrs: Optional[torch.Tensor] = None

    def finalize(self):
        """Free symmetric memory resources (no-op for PyTorch symm_mem - GC managed)."""
        if self.symm_mem_ctx:
            self.symm_mem_ctx.free_tensor(self.symm_gemm_out_buf)
            self.symm_mem_ctx.free_tensor(self.symm_residual_out_buf)
            self.symm_mem_ctx.free_tensor(self.symm_ag_out_buf)
            self.symm_mem_ctx.free_tensor(self.gemm_barrier_buf)
            self.symm_mem_ctx.free_tensor(self.multi_st_barrier_buf)

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


def create_gemm_ag_context_pytorch(
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
    Create GEMM + AllGather Context - PyTorch symmetric_memory version

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
        GemmAGContextPyTorch instance
    """
    assert local_world_size == world_size, "Only intra-node supported"

    symm_mem_ctx = SymmMemContext.get_instance()
    if symm_mem_ctx is None:
        raise RuntimeError("SymmMemContext not initialized. Call SymmMemContext.from_process_group() first.")

    # Allocate symmetric memory using symm_mem.empty() + rendezvous()
    gemm_out_buf = symm_mem_ctx.create_tensor((world_size, max_M, N), dtype)
    residual_out_buf = symm_mem_ctx.create_tensor((world_size, max_M, N), dtype)
    symm_ag_out_buf = symm_mem_ctx.create_tensor((max_M * world_size, N), dtype)
    gemm_barrier_buf = symm_mem_ctx.create_tensor(
        (world_size, triton.cdiv(max_M, MIN_BLOCK_SIZE_M), triton.cdiv(N, MIN_BLOCK_SIZE_N)),
        torch.int32
    )
    multi_st_barrier_buf = symm_mem_ctx.create_tensor((world_size * NUM_COMM_SMS,), torch.int32)

    # Non-symmetric buffers (local only)
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

    symm_mem_ctx.barrier_all()

    # Get multicast buffer (optional)
    mc_ag_out_buf = None
    if enable_multicast:
        try:
            mc_ag_out_buf = symm_mem_ctx.get_multicast_tensor(symm_ag_out_buf)
            if mc_ag_out_buf == 0:
                mc_ag_out_buf = None
        except Exception as e:
            print(f"Warning: Multicast not supported: {e}")
            mc_ag_out_buf = None

    # Get peer tensor pointer arrays (for dynamic access in kernel)
    peer_ag_out_ptrs = torch.tensor(
        symm_mem_ctx.get_peer_buffer_ptrs(symm_ag_out_buf),
        dtype=torch.int64,
        device=torch.cuda.current_device()
    )
    peer_multi_st_barrier_ptrs = torch.tensor(
        symm_mem_ctx.get_peer_buffer_ptrs(multi_st_barrier_buf),
        dtype=torch.int64,
        device=torch.cuda.current_device()
    )

    ctx = GemmAGContextPyTorch(
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
        symm_mem_ctx=symm_mem_ctx,
        mc_ag_out_buf=mc_ag_out_buf,
        peer_ag_out_ptrs=peer_ag_out_ptrs,
        peer_multi_st_barrier_ptrs=peer_multi_st_barrier_ptrs,
    )

    return ctx


# ============================================================================
# Triton Kernels (identical to nvshmem4py version - kernel code is backend-agnostic)
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
    """Vectorized load 4x int32"""
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
    """Vectorized store 4x int32"""
    tl.inline_asm_elementwise(
        asm="""
        st.global.v4.b32 [$1], {$2,$3,$4,$5};
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r,r,r,r"),
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def multimem_st_v4(ptr, val0, val1, val2, val3, suffix: tl.constexpr):
    """Multimem store 4x int32 (NVSwitch multicast)"""
    tl.static_assert(
        suffix == "f32" or suffix == "bf16x2" or suffix == "f16x2",
        "multimem.st.v4 only supports f32, bf16x2, f16x2"
    )
    tl.inline_asm_elementwise(
        asm=f"""
        multimem.st.global.v4.{suffix} [$1], {{$2, $3, $4, $5}};
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r,r,r,r"),
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def ld_sys(ptr):
    """Load with sys scope (acquire semantic)"""
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
    """Load with gpu scope (relaxed semantic)"""
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
    """Store with sys scope (release semantic)"""
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
    """Store with gpu scope (relaxed semantic)"""
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
    """Block synchronization - equivalent to CUDA __syncthreads()"""
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
def consumer_all_gather_kernel_pytorch(
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
    Consumer AllGather Kernel - PyTorch symmetric_memory version

    Identical to nvshmem4py version - kernel code is backend-agnostic.
    The only difference is how peer pointers are obtained (buffer_ptrs vs get_peer_tensor).
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
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                byte_offset = offset * ELEM_BYTES
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
                offset = (row_id + pid_m * BLOCK_SIZE_M) * N + col_id * VEC_SIZE + pid_n * BLOCK_SIZE_N
                byte_offset = offset * ELEM_BYTES
                src_byte_addr = symm_input_ptr.to(tl.int64) + byte_offset

                val0, val1, val2, val3 = load_v4_b32(src_byte_addr.to(tl.pointer_type(tl.int32)))
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

def consumer_all_gather_pytorch(
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
    consumer_all_gather_kernel_pytorch[grid](
        symm_input,
        symm_ag_out,
        ag_out,
        gemm_barrier,
        multi_st_barrier,
        peer_ag_out_ptrs,
        peer_multi_st_barrier_ptrs,
        mc_ag_out if mc_ag_out is not None else 0,
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
# GEMM Kernels (identical to nvshmem4py version)
# ============================================================================

@triton.jit
def kernel_persistent_gemm_notify_pytorch(
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
    Persistent GEMM Kernel (with notification mechanism) - PyTorch version

    Identical computation logic as nvshmem4py version.
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
        thread_idx = tid(0)
        if thread_idx == 0:
            if TILE_MAP_LEVEL == 0:  # tile_wise_map_to_comm
                gemm_barrier_idx = pid_m * num_pid_n + pid_n
                st_gpu(gemm_barrier_ptr + gemm_barrier_idx, 1)


def persistent_gemm_notify_pytorch(a, b, out, gemm_barrier, tile_barrier, gemm_config, rank=0):
    """
    Host function to launch persistent GEMM kernel - PyTorch version
    """
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, _ = b.shape

    grid = lambda META: (
        min(META["NUM_GEMM_SMS"],
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),
    )

    kernel_persistent_gemm_notify_pytorch[grid](
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
# High-Level Operations (PyTorch symmetric_memory version)
# ============================================================================

def allgather_op_pytorch(
    ctx: GemmAGContextPyTorch,
    c: torch.Tensor,
    BLOCK_SIZE_M: int = 16,
    BLOCK_SIZE_N: int = 64,
    NUM_COMM_SMS: int = 16,
    USE_MULTIMEM_ST: bool = False,
    copy_to_local: bool = True,
):
    """
    Execute AllGather operation - PyTorch symmetric_memory version
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
    consumer_all_gather_pytorch(
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


def gemm_op_pytorch(
    ctx: GemmAGContextPyTorch,
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_config: triton.Config,
    As: Optional[torch.Tensor] = None,
    Bs: Optional[torch.Tensor] = None,
):
    """
    Execute pure GEMM operation (without AllGather) - PyTorch version
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

    persistent_gemm_notify_pytorch(a, b, symm_c, gemm_barrier, tile_barrier, gemm_config, rank=ctx.rank)
    out.copy_(symm_c.reshape(-1)[:M * N].reshape(M, N))

    return out


def gemm_allgather_op_pytorch(
    ctx: GemmAGContextPyTorch,
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_config: triton.Config,
    copy_to_local: bool = True,
    USE_MULTIMEM_ST: bool = False,
    As: Optional[torch.Tensor] = None,
    Bs: Optional[torch.Tensor] = None,
):
    """
    Execute GEMM + AllGather fused operation - PyTorch version
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
        ctx.symm_mem_ctx.barrier_all(current_stream)
        ctx.reset_all_barrier_buf()
        ctx.symm_mem_ctx.barrier_all(current_stream)

    # Launch GEMM kernel
    persistent_gemm_notify_pytorch(a, b, symm_c, gemm_barrier, tile_barrier, gemm_config, rank=ctx.rank)

    # Launch AllGather kernel in separate stream
    with torch.cuda.stream(ag_stream):
        consumer_all_gather_pytorch(
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


def deepgemm_allgather_op_pytorch(
    ctx: GemmAGContextPyTorch,
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    gemm_config: triton.Config,
    copy_to_local: bool = True,
    USE_MULTIMEM_ST: bool = False,
):
    """
    Execute DeepGEMM (FP8) + AllGather fused operation - PyTorch version

    Uses deep_gemm.fp8_gemm_nt with enable_overlap=True
    """
    if not DEEP_GEMM_AVAILABLE:
        raise RuntimeError("deep_gemm is not installed. Please install deep_gemm to use deepgemm_allgather_op_pytorch.")

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
        ctx.symm_mem_ctx.barrier_all(current_stream)
        ctx.reset_all_barrier_buf()
        ctx.symm_mem_ctx.barrier_all(current_stream)

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
        consumer_all_gather_pytorch(
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