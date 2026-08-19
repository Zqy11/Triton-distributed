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
AllGather -> GEMM Overlap using PyTorch symmetric memory (symm_mem)

This module implements the AllGather + GEMM overlap pattern where:
1. cp_engine_full_mesh_pull_ag: host-side AllGather via Copy Engine
   (full-mesh pull) enqueued on ag_stream; uses PtoP data copies and
   _SymmetricMemory.stream_write_value32 for signal writes so AG runs
   on the CE concurrently with the GEMM without consuming SMs, and
   signals per-rank-shard when each peer's data is ready.
2. consumer_bf16_a_block_fp8_matmul: GEMM kernel that polls ag_signal[src_rank]
   with system-scope acquire (ld_sys) before computing each tile, reads bf16 A
   from symmetric memory, on-the-fly quantizes to fp8, and performs block-wise
   fp8 matmul.

Also includes:
- Per-token-group FP8 quantization kernels (from SGLang)
- Block-wise FP8 matmul kernels (w8a8 and bf16-a fused quantize variants)
"""

import logging
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory
import dataclasses
from typing import List, Optional, Tuple
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ============================================================================
# Triton JIT Helpers (PTX inline assembly)
# ============================================================================

@triton.jit
def __syncthreads():
    """PTX bar.sync (block-level __syncthreads)."""
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
    """PTX threadIdx.x/y/z."""
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
def ld_sys(ptr):
    """ld.global.acquire.sys.b32 — system-scope acquire load."""
    return tl.inline_asm_elementwise(
        asm="ld.global.acquire.sys.b32 $0, [$1];",
        constraints=("=r,l"),
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def st_sys(ptr, val):
    """st.global.release.sys.b32 — system-scope release store."""
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


# ============================================================================
# AllGather -> GEMM Context (PyTorch symm_mem version)
# ============================================================================

@dataclasses.dataclass
class AllGatherGemmContextSymmMem:
    """Context for AG->GEMM overlap. Holds symm_mem buffers and handles."""

    rank: int
    num_ranks: int
    NUM_COMM_SMS: int
    NUM_GEMM_SMS: int
    ag_stream: torch.cuda.Stream

    symm_input_buf: torch.Tensor           # [world_size, M, K]
    symm_ag_a_buf: torch.Tensor            # [M * world_size, K]
    ag_signal_buf: torch.Tensor            # [world_size] uint32

    # symm_mem rendezvous handles
    input_hdl: object = None
    ag_hdl: object = None
    signal_hdl: object = None

    mc_ag_a_buf: Optional[torch.Tensor] = None  # deprecated (NVLS)
    peer_signal_ptrs: Optional[torch.Tensor] = None
    peer_symm_input_bufs: Optional[List[torch.Tensor]] = None
    group: Optional[object] = None

    def finalize(self):
        """Release symm_mem resources with a barrier for lock-step teardown."""
        self.symm_input_buf = None
        self.symm_ag_a_buf = None
        self.ag_signal_buf = None
        self.input_hdl = None
        self.ag_hdl = None
        self.signal_hdl = None
        self.mc_ag_a_buf = None
        self.peer_signal_ptrs = None
        self.peer_symm_input_bufs = None
        if dist.is_initialized():
            dist.barrier(group=self.group)

    def get_input_buf(self, M, K):
        """Return this rank's [M, K] shard view in symm_input_buf."""
        return self.symm_input_buf[self.rank, :M, :]


def create_allgather_gemm_context_symm_mem(
    ag_stream: torch.cuda.Stream,
    rank: int,
    world_size: int,
    max_M: int,
    K: int,
    NUM_COMM_SMS: int = 0,
    enable_multicast: bool = False,
    group: Optional[object] = None,
):
    """Create AG->GEMM context with symm_mem buffers.

    group: process group for symm_mem rendezvous (defaults to WORLD;
        set to TP group when pipeline parallelism is used).
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before creating the context.")

    device = torch.cuda.current_device()
    rendezvous_group = group if group is not None else dist.group.WORLD

    symm_input_buf = symm_mem.empty(
        (world_size, max_M, K), dtype=torch.bfloat16, device=device,
    )
    symm_ag_a_buf = symm_mem.empty(
        (max_M * world_size, K), dtype=torch.bfloat16, device=device,
    )
    ag_signal_buf = symm_mem.empty(
        (world_size,), dtype=torch.uint32, device=device,
    )

    symm_input_buf.zero_()
    symm_ag_a_buf.zero_()
    ag_signal_buf.zero_()

    input_hdl = symm_mem.rendezvous(symm_input_buf, group=rendezvous_group)
    ag_hdl = symm_mem.rendezvous(symm_ag_a_buf, group=rendezvous_group)
    signal_hdl = symm_mem.rendezvous(ag_signal_buf, group=rendezvous_group)

    input_hdl.barrier()

    mc_ag_a_buf = None
    if enable_multicast:
        logger.warning(
            "enable_multicast is deprecated in the symm_mem migration; "
            "mc_ag_a_buf will be None. Reactivate via ag_hdl.multicast_ptr if needed."
        )

    peer_signal_ptrs = torch.tensor(
        [signal_hdl.get_buffer(r, (world_size,), torch.uint32).data_ptr()
         for r in range(world_size)],
        dtype=torch.int64,
        device=device,
    )

    peer_symm_input_bufs = [
        input_hdl.get_buffer(r, (world_size, max_M, K), torch.bfloat16)
        for r in range(world_size)
    ]

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = num_sms - NUM_COMM_SMS

    ctx = AllGatherGemmContextSymmMem(
        rank=rank,
        num_ranks=world_size,
        NUM_COMM_SMS=NUM_COMM_SMS,
        NUM_GEMM_SMS=num_gemm_sms,
        ag_stream=ag_stream,
        symm_input_buf=symm_input_buf,
        symm_ag_a_buf=symm_ag_a_buf,
        ag_signal_buf=ag_signal_buf,
        input_hdl=input_hdl,
        ag_hdl=ag_hdl,
        signal_hdl=signal_hdl,
        mc_ag_a_buf=mc_ag_a_buf,
        peer_signal_ptrs=peer_signal_ptrs,
        peer_symm_input_bufs=peer_symm_input_bufs,
        group=group,
    )

    return ctx


# ============================================================================
# CP-Engine Full-Mesh-Pull AllGather (host-side, using _SymmetricMemory)
# ============================================================================

def cp_engine_full_mesh_pull_ag(
    rank: int,
    world_size: int,
    M_local: int,
    K: int,
    symm_input: torch.Tensor,
    symm_ag_a: torch.Tensor,
    peer_symm_input_bufs: List[torch.Tensor],
    ag_signal: torch.Tensor,
):
    """AllGather via Copy Engine full-mesh pull (PtoP data + PtoP signal).

    Signal writes via _SymmetricMemory.stream_write_value32 (cuStreamWriteValue32).
    Caller must wrap in `with torch.cuda.stream(ag_stream):`.
    """
    # Self-shard: local copy + signal via _SymmetricMemory.stream_write_value32
    local_dst = symm_ag_a[rank * M_local : (rank + 1) * M_local, :]
    local_dst.copy_(symm_input)
    _SymmetricMemory.stream_write_value32(ag_signal, rank, 1)

    # Remote shards in rotated order
    for offset in range(1, world_size):
        src_rank = (rank + offset) % world_size
        remote_src = peer_symm_input_bufs[src_rank][src_rank, :M_local, :]
        local_dst = symm_ag_a[src_rank * M_local : (src_rank + 1) * M_local, :]
        local_dst.copy_(remote_src)
        # cuStreamWriteValue32 issues a system level fence before the write
        _SymmetricMemory.stream_write_value32(ag_signal, src_rank, 1)


# ============================================================================
# Per-Token-Group FP8 Quantization Kernels (from SGLang)
# ============================================================================

@triton.jit
def _per_token_group_quant_fp8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,
    eps,
    fp8_max,
    BLOCK: tl.constexpr,
):
    """Per-token-group FP8 quantization kernel (from SGLang).

    Each program handles one group of size BLOCK (= group_size).
    Computes absmax, scale, and quantizes to FP8 in a single pass.
    """
    g_id = tl.program_id(0)
    y_ptr += g_id * y_stride
    y_q_ptr += g_id * y_stride
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / fp8_max
    y_s_inv = 1.0 / y_s
    y_q = tl.clamp(y * y_s_inv, -fp8_max, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int = 128,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group FP8 quantization using Triton kernel.

    Equivalent to DeepGEMM's per_token_cast_to_fp8 but ~20x faster
    (single Triton kernel vs multi-step PyTorch ops).

    Args:
        x: Input tensor [M, K], must be contiguous.
        group_size: Quantization group size (typically 128).
        eps: Minimum scale to avoid division by zero.
        dtype: FP8 output dtype.

    Returns:
        Tuple of (quantized fp8 tensor [M, K], scale [M, K // group_size]).
    """
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    assert x.shape[-1] % group_size == 0, (
        f"Last dim {x.shape[-1]} must be divisible by group_size {group_size}"
    )
    assert x.is_contiguous(), "Input must be contiguous"

    fp8_max = float(torch.finfo(dtype).max)

    M, K = x.shape
    num_groups = M * (K // group_size)

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty((M, K // group_size), device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(group_size)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_group_quant_fp8[(num_groups,)](
        x, x_q, x_s,
        group_size,
        group_size,
        eps,
        fp8_max=fp8_max,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    return x_q, x_s


# ============================================================================
# Block-wise FP8 Matmul Kernels (from SGLang)
# ============================================================================

@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
tensor `C`.
"""

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Block-wise FP8 matmul using Triton kernel.

    Args:
        A: The input tensor (fp8), shape [..., K], row-major.
        B: The weight tensor (fp8), shape [N, K], column-major (transposed).
        As: The per-token-group quantization scale for `A`, shape [..., K // block_k].
        Bs: The per-block quantization scale for `B`, shape [N // block_n, K // block_k].
        block_size: The block sizes [block_m, block_n, block_k], e.g. [64, 128, 128].
        output_dtype: The dtype of the output tensor.

    Returns:
        torch.Tensor: The result of matmul, shape [..., N].
    """
    assert len(block_size) == 3
    block_m, block_n, block_k = block_size[0], block_size[1], block_size[2]

    assert A.shape[-1] == B.shape[-1]
    assert A.is_contiguous()

    M = A.numel() // A.shape[-1]
    N, K = B.shape

    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    # Default config (block-wise quant: BLOCK_SIZE_K must be divisible by block_size[2])
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }

    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=needs_masking,
    )

    return C


# ============================================================================
# Fused BF16-A + Block-wise FP8 GEMM Kernels (quantize A on-the-fly)
# ============================================================================

@triton.jit
def _bf16_a_block_fp8_matmul(
    # Pointers to inputs and output
    A,          # bf16 activation, shape [M, K], row-major
    B,          # fp8 weight, shape [N, K], column-major (transposed)
    C,          # output, shape [M, N]
    Bs,         # per-block scale for B, shape [N // group_n, K // group_k]
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Block-wise FP8 matmul with on-the-fly A quantization.

    Loads bf16 A tiles, quantizes to fp8 per-group on the fly,
    then performs dot product with fp8 B tiles and block-wise scaling.
    This fuses the quantize + GEMM into a single kernel, saving
    one kernel launch and the global memory round-trip for A_fp8 / A_scale.
    """
    fp8_max = 448.0  # float(torch.finfo(torch.float8_e4m3fn).max)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers (bf16, row-major)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B pointers (fp8, column-major)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # B scale pointers
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load bf16 A tile
        if needs_masking:
            a_bf16 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a_bf16 = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        # On-the-fly quantize A: per-row-group absmax -> scale -> cast to fp8
        a_absmax = tl.max(tl.abs(a_bf16), axis=1)  # [BLOCK_SIZE_M]
        a_amax_safe = tl.maximum(a_absmax, 1e-12)  # avoid div-by-zero
        a_s = a_amax_safe * (1.0 / fp8_max)  # [BLOCK_SIZE_M], per-row scale (scalar mul)
        a_rcp_s = fp8_max / a_amax_safe  # [BLOCK_SIZE_M], 1/scale (scalar div, only BLOCK_M elems)
        a_fp8 = (a_bf16 * a_rcp_s[:, None]).to(B.dtype.element_ty)  # FMUL + saturate cast (no clamp)

        # Load B scale
        b_s = tl.load(Bs_ptrs)

        # Accumulate: dot(fp8_A, fp8_B) * a_scale * b_scale
        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a_fp8, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def bf16_a_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-wise FP8 matmul with on-the-fly A (bf16) quantization using Triton kernel.

    Fuses per-token-group bf16->fp8 quantization of A into the GEMM kernel,
    eliminating the separate quantize kernel launch and the global memory
    round-trip for A_fp8 and A_scale.

    Args:
        A: The activation tensor (bf16), shape [..., K], row-major.
        B: The weight tensor (fp8), shape [N, K], column-major (transposed).
        Bs: The per-block quantization scale for `B`, shape [N // block_n, K // block_k].
        block_size: The block sizes [block_m, block_n, block_k], e.g. [64, 128, 128].
            block_k must match the quantization group size for A (typically 128).
        output_dtype: The dtype of the output tensor.

    Returns:
        torch.Tensor: The result of matmul, shape [..., N].
    """
    assert len(block_size) == 3
    block_m, block_n, block_k = block_size[0], block_size[1], block_size[2]

    assert A.shape[-1] == B.shape[-1]
    assert A.is_contiguous()
    assert A.dtype == torch.bfloat16, f"Expected bf16 A, got {A.dtype}"

    M = A.numel() // A.shape[-1]
    N, K = B.shape

    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    # Default config
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }

    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _bf16_a_block_fp8_matmul[grid](
        A,
        B,
        C,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=needs_masking,
    )

    return C


# ============================================================================
# Consumer GEMM Kernel (non-persistent, polls AG signal, then computes bf16-A block-FP8 matmul)
# ============================================================================

@triton.jit
def consumer_bf16_a_block_fp8_matmul(
    # Pointers
    A_ptr,           # bf16 [M, K] gathered A (symm_mem)
    B_ptr,           # fp8 [N, K] weight (col-major)
    C_ptr,           # output [M, N]
    Bs_ptr,          # B scale [N//group_n, K//group_k]
    ag_signal_ptr,   # [world_size] uint32
    # Dimensions
    M, N, K,
    M_local,
    M_local_tiles,
    # Block quantization
    group_n, group_k,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_Bs_k, stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Non-persistent consumer GEMM: polls AG signal per rank-shard, then
    computes bf16-A on-the-fly-quantized block-FP8 matmul.

    Rank-aware tile rotation aligns CTA scheduling with AG signal arrival
    order (self-shard first, then peers in rotated order).
    """
    fp8_max = 448.0
    pid = tl.program_id(axis=0)

    num_pid_m = M_local_tiles * world_size
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Rank-aware tile rotation
    logical_src_rank = min(pid_m // M_local_tiles, world_size - 1)
    tile_in_rank = pid_m - logical_src_rank * M_local_tiles
    src_rank = (rank + logical_src_rank) % world_size
    pid_m = src_rank * M_local_tiles + tile_in_rank

    tile_row_start = src_rank * M_local + tile_in_rank * BLOCK_SIZE_M

    # Poll AG signal for src_rank's shard
    if tid(0) == 0:
        while ld_sys(ag_signal_ptr + src_rank) != 1:
            pass
    __syncthreads()

    offs_am = (tile_row_start + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs_ptr + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a_bf16 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a_bf16 = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        # On-the-fly A quantization: per-row absmax
        a_absmax = tl.max(tl.abs(a_bf16), axis=1)
        a_s = tl.maximum(a_absmax, 1e-12) / fp8_max
        a_fp8 = tl.clamp(a_bf16 / a_s[:, None], -fp8_max, fp8_max)
        a_fp8 = a_fp8.to(B_ptr.dtype.element_ty)

        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a_fp8, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        Bs_ptrs += scale_step_k * stride_Bs_k

    c = accumulator.to(tl.bfloat16)

    offs_cm = tile_row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    rank_row_end = (src_rank + 1) * M_local
    c_mask = (offs_cm[:, None] < rank_row_end) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# AllGather -> GEMM Operation (AG first, GEMM follows with tile-level signaling)
# ============================================================================

def allgather_gemm_op_symm_mem(
    ctx: AllGatherGemmContextSymmMem,
    a_bf16: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: list,
    output_dtype: torch.dtype = torch.bfloat16,
    GROUP_SIZE_M: int = 32,
):
    """AG->GEMM overlap: CE-driven AllGather on ag_stream + consumer GEMM on
    current_stream with per-rank-shard signal polling for compute-comm overlap.

    Args:
        ctx: AllGatherGemmContextSymmMem context
        a_bf16: bf16 activation tensor [M, K] (local shard for this rank)
        b_fp8: fp8 weight tensor [N, K], column-major (transposed)
        b_scale: per-block quantization scale for B [N // group_n, K // group_k]
        block_size: [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
        output_dtype: output dtype
        GROUP_SIZE_M: swizzle group size for GEMM

    Returns:
        c: GEMM result [M * world_size, N]
    """
    assert len(block_size) == 3
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size

    M_local, K = a_bf16.shape
    N = b_fp8.shape[0]
    M = M_local * ctx.num_ranks

    assert a_bf16.dtype == torch.bfloat16, f"Expected bf16 A, got {a_bf16.dtype}"
    assert a_bf16.shape[1] == b_fp8.shape[1], "K dimension mismatch"
    assert a_bf16.is_contiguous()

    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)

    # Copy A shard to symmetric memory
    symm_input = ctx.get_input_buf(M_local, K)
    symm_input.copy_(a_bf16)

    symm_ag_a = ctx.symm_ag_a_buf
    ag_signal = ctx.ag_signal_buf

    assert ag_signal.numel() >= ctx.num_ranks

    c = torch.empty((M, N), dtype=output_dtype, device=a_bf16.device)

    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream

    ctx.input_hdl.barrier()
    ag_signal.fill_(0)
    ctx.input_hdl.barrier()

    ag_stream.wait_stream(current_stream)

    # Step 1: AG on ag_stream (CE, no SM consumption)
    with torch.cuda.stream(ag_stream):
        cp_engine_full_mesh_pull_ag(
            rank=ctx.rank,
            world_size=ctx.num_ranks,
            M_local=M_local,
            K=K,
            symm_input=symm_input,
            symm_ag_a=symm_ag_a,
            peer_symm_input_bufs=ctx.peer_symm_input_bufs,
            ag_signal=ag_signal,
        )

    # Step 2: consumer GEMM on current_stream
    needs_masking = bool(K % BLOCK_SIZE_K != 0)
    M_local_tiles = triton.cdiv(M_local, BLOCK_SIZE_M)
    num_pid_m_grid = M_local_tiles * ctx.num_ranks
    num_tiles = num_pid_m_grid * num_pid_n

    consumer_bf16_a_block_fp8_matmul[(num_tiles,)](
        symm_ag_a,
        b_fp8,
        c,
        b_scale,
        ag_signal,
        M, N, K,
        M_local,
        M_local_tiles,
        BLOCK_SIZE_N,  # group_n
        BLOCK_SIZE_K,  # group_k
        symm_ag_a.stride(0),
        symm_ag_a.stride(1),
        b_fp8.stride(1),
        b_fp8.stride(0),
        c.stride(0),
        c.stride(1),
        b_scale.stride(1),
        b_scale.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        needs_masking=needs_masking,
        rank=ctx.rank,
        world_size=ctx.num_ranks,
        num_warps=4,
        num_stages=3,
    )

    current_stream.wait_stream(ag_stream)

    return c


# ============================================================================
# Triton BF16 Matmul Kernel (standard C = A @ B, no quantization)
# ============================================================================

@triton.jit
def _bf16_matmul(
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Standard Triton BF16 matmul kernel: C = A @ B.

    A: [M, K] row-major bf16
    B: [K, N] column-major bf16 (transposed storage)
    C: [M, N] row-major bf16
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def bf16_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    block_size: list = None,
    output_dtype: torch.dtype = torch.bfloat16,
    GROUP_SIZE_M: int = 32,
) -> torch.Tensor:
    """Standard BF16 matmul using Triton kernel: C = A @ B.

    Args:
        A: [M, K] row-major bf16 activation.
        B: [K, N] column-major bf16 weight (transposed storage).
        block_size: [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K].
        output_dtype: output dtype.
        GROUP_SIZE_M: swizzle group size for L2 cache optimization.

    Returns:
        C: [M, N] output tensor.
    """
    if block_size is None:
        block_size = [64, 128, 128]
    assert len(block_size) == 3
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size

    M, K = A.shape
    _, N = B.shape

    assert A.is_contiguous()
    assert A.dtype == B.dtype

    C = torch.empty((M, N), dtype=output_dtype, device=A.device)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _bf16_matmul[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
        num_stages=3,
    )

    return C


# ============================================================================
# Consumer BF16 GEMM Kernel (non-persistent, polls AG signal, then computes bf16 matmul)
# ============================================================================

@triton.jit
def consumer_bf16_matmul(
    # Pointers
    A_ptr,           # bf16 [M, K] gathered A (symm_mem)
    B_ptr,           # bf16 [K, N] weight (col-major)
    C_ptr,           # output [M, N]
    ag_signal_ptr,   # [world_size] uint32
    # Dimensions
    M, N, K,
    M_local,
    M_local_tiles,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Non-persistent consumer BF16 GEMM: polls AG signal per rank-shard, then
    computes bf16 matmul C = A @ B.

    Rank-aware tile rotation aligns CTA scheduling with AG signal arrival
    order (self-shard first, then peers in rotated order).
    """
    pid = tl.program_id(axis=0)

    num_pid_m = M_local_tiles * world_size
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Rank-aware tile rotation
    logical_src_rank = min(pid_m // M_local_tiles, world_size - 1)
    tile_in_rank = pid_m - logical_src_rank * M_local_tiles
    src_rank = (rank + logical_src_rank) % world_size
    pid_m = src_rank * M_local_tiles + tile_in_rank

    tile_row_start = src_rank * M_local + tile_in_rank * BLOCK_SIZE_M

    # Poll AG signal for src_rank's shard
    if tid(0) == 0:
        while ld_sys(ag_signal_ptr + src_rank) != 1:
            pass
    __syncthreads()

    offs_am = (tile_row_start + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.bfloat16)

    offs_cm = tile_row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    rank_row_end = (src_rank + 1) * M_local
    c_mask = (offs_cm[:, None] < rank_row_end) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# AllGather -> BF16 GEMM Operation (AG first, BF16 GEMM follows with tile-level signaling)
# ============================================================================

def allgather_bf16_gemm_op_symm_mem(
    ctx: AllGatherGemmContextSymmMem,
    a_bf16: torch.Tensor,
    b_bf16: torch.Tensor,
    block_size: list,
    output_dtype: torch.dtype = torch.bfloat16,
    GROUP_SIZE_M: int = 32,
):
    """AG->BF16 GEMM overlap: CE-driven AllGather on ag_stream + consumer BF16
    GEMM on current_stream with per-rank-shard signal polling.

    Unlike allgather_gemm_op_symm_mem which uses FP8 block-wise matmul, this
    performs a pure BF16 matmul (C = A @ B) without quantization.

    Args:
        ctx: AllGatherGemmContextSymmMem context
        a_bf16: bf16 activation tensor [M, K] (local shard for this rank)
        b_bf16: bf16 weight tensor [K, N], column-major (transposed storage)
        block_size: [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
        output_dtype: output dtype
        GROUP_SIZE_M: swizzle group size for GEMM

    Returns:
        c: GEMM result [M * world_size, N]
    """
    assert len(block_size) == 3
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size

    M_local, K = a_bf16.shape
    _, N = b_bf16.shape
    M = M_local * ctx.num_ranks

    assert a_bf16.dtype == torch.bfloat16, f"Expected bf16 A, got {a_bf16.dtype}"
    assert b_bf16.dtype == torch.bfloat16, f"Expected bf16 B, got {b_bf16.dtype}"
    assert a_bf16.shape[1] == b_bf16.shape[0], "K dimension mismatch between A and B"
    assert a_bf16.is_contiguous()

    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)

    # Copy A shard to symmetric memory
    symm_input = ctx.get_input_buf(M_local, K)
    symm_input.copy_(a_bf16)

    symm_ag_a = ctx.symm_ag_a_buf
    ag_signal = ctx.ag_signal_buf

    assert ag_signal.numel() >= ctx.num_ranks

    c = torch.empty((M, N), dtype=output_dtype, device=a_bf16.device)

    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream

    ctx.input_hdl.barrier()
    ag_signal.fill_(0)
    ctx.input_hdl.barrier()

    ag_stream.wait_stream(current_stream)

    # Step 1: AG on ag_stream (CE, no SM consumption)
    with torch.cuda.stream(ag_stream):
        cp_engine_full_mesh_pull_ag(
            rank=ctx.rank,
            world_size=ctx.num_ranks,
            M_local=M_local,
            K=K,
            symm_input=symm_input,
            symm_ag_a=symm_ag_a,
            peer_symm_input_bufs=ctx.peer_symm_input_bufs,
            ag_signal=ag_signal,
        )

    # Step 2: consumer BF16 GEMM on current_stream
    needs_masking = bool(K % BLOCK_SIZE_K != 0)
    M_local_tiles = triton.cdiv(M_local, BLOCK_SIZE_M)
    num_pid_m_grid = M_local_tiles * ctx.num_ranks
    num_tiles = num_pid_m_grid * num_pid_n

    consumer_bf16_matmul[(num_tiles,)](
        symm_ag_a,
        b_bf16,
        c,
        ag_signal,
        M, N, K,
        M_local,
        M_local_tiles,
        symm_ag_a.stride(0),
        symm_ag_a.stride(1),
        b_bf16.stride(0),
        b_bf16.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        needs_masking=needs_masking,
        rank=ctx.rank,
        world_size=ctx.num_ranks,
        num_warps=4,
        num_stages=3,
    )

    current_stream.wait_stream(ag_stream)

    return c