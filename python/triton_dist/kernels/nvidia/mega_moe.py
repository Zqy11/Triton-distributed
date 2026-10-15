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
MegaMoE kernels — FP8 quantization, NVSHMEM dispatch, and group GEMM

This module provides:
- Per-token-group FP8 quantization (Triton kernel)
- Block-wise FP8 weight quantization
- Warp-level NVSHMEM dispatch (putmem + signal + postprocess) with per-expert recv buffer
- Block-level NVSHMEM dispatch (putmem_signal_block) with per-rank recv buffer
- Block-wise FP8 group GEMM with [EXPERTS_PER_RANK, WS*MAX_M, K] input layout
"""

from typing import Tuple
import functools

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
import torch.distributed._symmetric_memory as symm_mem

from triton.language import core


# ============================================================================
# Constants & helpers
# ============================================================================

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0

# Map torch dtypes to Triton scalar types for kernel constexpr
_TORCH_TO_TRITON_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


def cdiv(a, b):
    return (a + b - 1) // b


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.empty(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    torch.cumsum(splits, 0, out=out[1:])
    out[0] = 0
    return out


# ============================================================================
# Triton JIT Helpers (PTX inline assembly)
# ============================================================================

@triton.jit
def tid(axis: tl.constexpr = 0):
    """Get thread index (threadIdx.x/y/z) via inline PTX."""
    if axis == 0:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
    elif axis == 1:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.y;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.z;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )


# ============================================================================
# CTA Barrier + Device-scope Signal Helpers (inline PTX)
# ============================================================================

@triton.jit
def __syncthreads():
    """PTX bar.sync 0 — CTA-wide barrier (synchronize all warps)."""
    tl.inline_asm_elementwise(
        asm="bar.sync 0;",
        constraints="=r",
        args=(),
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _release_signal_local(addrs):
    """Device-scope release store: set local signal_buf[e] = 1."""
    tl.inline_asm_elementwise(
        asm="st.release.gpu.global.b32 [$1], 1;",
        constraints="=r, l", args=[addrs],
        dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def _wait_signal_local(addrs):
    """Spin on a local int32 flag until it becomes 1 (device-scope acquire load)."""
    tl.inline_asm_elementwise(
        asm="""
        {
            .reg .u32 %tmp; .reg .pred %p;
            wait: ld.acquire.gpu.global.b32 %tmp, [$1];
            setp.eq.u32 %p, %tmp, 1;
            @!%p bra wait;
        }
        """,
        constraints="=r, l", args=[addrs],
        dtype=tl.int32, is_pure=False, pack=1,
    )


@core.extern
def _getmem_warp_extern_wrapper(dest, source, size_bytes, pe, _semantic=None):
    """Extern for nvshmemx_getmem_warp (warp-level blocking get).

    Args must be (int64, int64, int64, int32) tensors — caller converts constexpr
    sizes to runtime int64 before calling.
    """
    return core.extern_elementwise(
        "", "",
        [dest, source, size_bytes, pe],
        {
            (
                core.dtype("int64"),  # dest ptr (local)
                core.dtype("int64"),  # source ptr (on remote PE)
                core.dtype("int64"),  # size in bytes
                core.dtype("int32"),  # pe number
            ): ("nvshmemx_getmem_warp", core.dtype("int32"))
        },
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# NVSHMEM Dispatch
# ============================================================================
#
# Warp-level dispatch using PyTorch's NVSHMEM backend
# (torch.distributed._symmetric_memory):
#   - DispatchContext pre-allocates all symmetric buffers via symm_mem (no per-call alloc)
#   - Warp-level getmem_warp pulls tokens from remote send buffers into local 3D output
#   - Single fused kernel: getmem_warp loop → compact 3D output (no Phase 2 compaction)

# NVSHMEM availability check
_NVSHMEM_AVAILABLE = symm_mem.is_nvshmem_available()


# -- Dispatch context (pre-allocates NVSHMEM symmetric buffers via symm_mem) --

class DispatchContext:
    """Pre-allocated NVSHMEM buffers for FP8 dispatch.

    Uses torch.distributed._symmetric_memory (symm_mem) for tensor creation,
    which works with PyTorch's built-in NVSHMEM backend. No triton_dist needed.
    """

    def __init__(
        self,
        max_m: int,
        hidden: int,
        num_experts: int,
        world_size: int,
        rank: int,
        num_scale_groups: int,
        group_name: str,
    ):
        dev = f"cuda:{rank}"
        epr = num_experts // world_size
        self.send_buf, self.barrier_hdl = self._create_tensor((max_m, hidden), FP8_DTYPE, dev, group_name)
        self.send_scales, _ = self._create_tensor((max_m, num_scale_groups), torch.float32, dev, group_name)

        # Symmetric buffer for cumsum exchange (size = num_experts + 1 for cumsum format).
        # Used by kernel to pull peer cumsum via getmem_warp — replaces host-side all_gather.
        self.split_send, _ = self._create_tensor((num_experts + 1,), torch.int32, dev, group_name)

        # Legacy buffers from push-mode warp dispatch — still allocated for backward
        # compatibility but not used by the new pull-mode _dispatch_warp_pull_kernel.
        self.split_recv, _ = self._create_tensor((num_experts * 2,), torch.int32, dev, group_name)
        self.signal, _ = self._create_tensor((world_size * 2,), torch.int64, dev, group_name)
        self.signal.fill_(0)

        # getmem_warp destination buffers: symmetric-heap (registered) memory is
        # required so the NIC's RMA writes have a valid lkey on the IB transport.
        ws_max_m = world_size * max_m
        self.recv_data, _ = self._create_tensor(
            (epr * ws_max_m, hidden), FP8_DTYPE, dev, group_name)
        self.recv_scales, _ = self._create_tensor(
            (epr * ws_max_m, num_scale_groups), torch.float32, dev, group_name)
        num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        max_ctas = max(num_sms, epr)
        self.all_splits_buf, _ = self._create_tensor(
            (max_ctas * world_size * (num_experts + 1),), torch.int32, dev, group_name)

        self.max_m = max_m
        self.hidden = hidden
        self.num_experts = num_experts
        self.experts_per_rank = epr
        self.num_scale_groups = num_scale_groups
        self.world_size = world_size
        self.rank = rank
        self.num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.call_count = 1
        self.MOD_VALUE = 1000000
        self.group_name = group_name

    @staticmethod
    def _create_tensor(shape, dtype, device, group_name):
        """Create a symmetric memory tensor via symm_mem.

        Returns (tensor, handle) where handle is a _SymmetricMemory object
        with a .barrier() method for lightweight cross-rank synchronization.
        """
        # NVSHMEM does not support fp8 directly, use int8 storage
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            tensor = symm_mem.empty(*shape, dtype=torch.int8, device=device)
            tensor = tensor.view(dtype)
        else:
            tensor = symm_mem.empty(*shape, dtype=dtype, device=device)
        hdl = symm_mem.rendezvous(tensor, group=group_name)
        return tensor, hdl

    def finalize(self):
        pass  # symm_mem tensors are garbage collected


def create_dispatch_context(
    max_m: int,
    hidden: int,
    num_experts: int,
    world_size: int,
    rank: int,
    num_scale_groups: int,
    group_name: str = None,
):
    if group_name is None:
        group_name = dist.group.WORLD.group_name
    symm_mem.set_backend("NVSHMEM")
    return DispatchContext(max_m, hidden, num_experts, world_size, rank, num_scale_groups, group_name)


# ============================================================================
# FP8 Quantization Helpers
# ============================================================================

def block_quantize_weight_fp8(
    w: torch.Tensor,
    group_n: int = 128,
    group_k: int = 128,
    dtype: torch.dtype = FP8_DTYPE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization for weights.

    Args:
        w: Weight tensor [E, N, K], must be contiguous.
        group_n: Block size for N dimension.
        group_k: Block size for K dimension.
        dtype: FP8 output dtype.

    Returns:
        Tuple of (quantized fp8 tensor [E, N, K], scale [E, N // group_n, K // group_k]).
    """
    assert w.dim() == 3, f"Expected 3D weight [E, N, K], got {w.dim()}D"
    E, N, K = w.shape
    assert N % group_n == 0, f"N({N}) must be divisible by group_n({group_n})"
    assert K % group_k == 0, f"K({K}) must be divisible by group_k({group_k})"

    fp8_max = float(torch.finfo(dtype).max)

    w_reshaped = w.view(E, N // group_n, group_n, K // group_k, group_k)
    absmax = w_reshaped.abs().float().amax(dim=(2, 4))  # [E, N // group_n, K // group_k]
    scales = torch.clamp(absmax / fp8_max, min=1e-10)

    scales_expanded = scales.view(E, N // group_n, 1, K // group_k, 1)
    w_fp8 = (w_reshaped / scales_expanded).clamp(-fp8_max, fp8_max).to(dtype)
    w_fp8 = w_fp8.view(E, N, K)

    return w_fp8, scales


# ============================================================================
# Triton Per-Token-Group FP8 Quantization
# ============================================================================

@triton.jit
def per_token_quant_kernel(
    X,                  # bf16 [M, K]
    OutFP8,             # fp8 [M, K]
    OutScales,          # float32 [M, num_groups]
    K,
    X_stride_m,
    Out_stride_m,
    Scale_stride_m,
    FP8_MAX: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Per-token-group FP8 quantization on contiguous bf16 input.

    Each block processes BLOCK_M rows. For each K-group of size GROUP_K,
    compute absmax -> scale -> saturating cast to fp8.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    for g in range(0, tl.cdiv(K, GROUP_K)):
        offs_k = g * GROUP_K + tl.arange(0, GROUP_K)

        x_ptrs = X + offs_m[:, None] * X_stride_m + offs_k[None, :]
        x_bf16 = tl.load(x_ptrs)
        x_fp32 = x_bf16.to(tl.float32)

        x_absmax = tl.max(tl.abs(x_fp32), axis=1)
        x_amax_safe = tl.maximum(x_absmax, 1e-4)
        x_scale = x_amax_safe / FP8_MAX            # sf = amax / 448.0
        x_inv_s = 1.0 / x_scale                    # 1.0 / sf
        x_fp8 = (x_fp32 * x_inv_s[:, None]).to(OutFP8.dtype.element_ty)

        out_ptrs = OutFP8 + offs_m[:, None] * Out_stride_m + offs_k[None, :]
        tl.store(out_ptrs, x_fp8)

        scale_ptrs = OutScales + offs_m * Scale_stride_m + g
        tl.store(scale_ptrs, x_scale)


@triton.jit
def per_token_quant_gather_kernel(
    X,                  # bf16 [M, K] — source input (contiguous)
    GatherIdx,          # int32 [EM] — output row → source row mapping
    OutFP8,             # fp8 [MAX_M, K] — send buffer (pre-allocated)
    OutScales,          # float32 [MAX_M, NSG] — scale send buffer
    K,
    X_stride_m,
    Out_stride_m,
    Scale_stride_m,
    EM,                 # number of gathered rows
    NUM_GROUPS,         # K // GROUP_K — total quantization groups along K
    FP8_MAX: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_G: tl.constexpr,   # quantization groups handled per CTA (K-parallelism)
):
    """Fused per-token-group FP8 quantization + gather into pre-allocated buffer.

    Each CTA processes BLOCK_M gathered rows × BLOCK_G quantization groups. The
    grid is 2-D — (cdiv(EM, BLOCK_M), cdiv(NUM_GROUPS, BLOCK_G)) — so the K
    dimension (always large) is used to supply extra parallelism when the row
    count is small, instead of looping every group inside a handful of CTAs and
    leaving most SMs idle.

    Per-token-group quantization is fully independent across groups, so splitting
    K across CTAs is exact: each (row, group) absmax/scale/store is unchanged.
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < EM

    gather_idx = tl.load(GatherIdx + offs_m, mask=row_mask, other=0)

    g_base = pid_g * BLOCK_G
    for gi in tl.static_range(0, BLOCK_G):
        g = g_base + gi
        if g < NUM_GROUPS:
            offs_k = g * GROUP_K + tl.arange(0, GROUP_K)

            x_ptrs = X + gather_idx[:, None] * X_stride_m + offs_k[None, :]
            x_bf16 = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
            x_fp32 = x_bf16.to(tl.float32)

            x_absmax = tl.max(tl.abs(x_fp32), axis=1)
            x_amax_safe = tl.maximum(x_absmax, 1e-4)
            x_scale = x_amax_safe / FP8_MAX
            x_inv_s = 1.0 / x_scale
            x_fp8 = (x_fp32 * x_inv_s[:, None]).to(OutFP8.dtype.element_ty)

            out_ptrs = OutFP8 + offs_m[:, None] * Out_stride_m + offs_k[None, :]
            tl.store(out_ptrs, x_fp8, mask=row_mask[:, None])

            scale_ptrs = OutScales + offs_m * Scale_stride_m + g
            tl.store(scale_ptrs, x_scale, mask=row_mask)


def per_token_quant_fp8(
    x: torch.Tensor,            # bf16 [M, K]
    group_k: int = 128,
    BLOCK_M: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-kernel Triton per-token-group FP8 quantization.

    Args:
        x: Input tensor [M, K] in bf16, must be contiguous.
        group_k: Quantization group size along K dimension.
        BLOCK_M: Rows per thread block.

    Returns:
        (x_fp8 [M, K] in FP8_DTYPE, x_scales [M, K // group_k] in float32)
    """
    M, K = x.shape
    assert K % group_k == 0, f"K({K}) must be divisible by group_k({group_k})"
    assert x.dtype == torch.bfloat16, f"Input must be bf16, got {x.dtype}"
    assert x.is_contiguous(), "Input must be contiguous"

    x_fp8 = torch.empty(M, K, dtype=FP8_DTYPE, device=x.device)
    x_scales = torch.empty(M, K // group_k, dtype=torch.float32, device=x.device)

    grid = (cdiv(M, BLOCK_M),)
    per_token_quant_kernel[grid](
        x, x_fp8, x_scales,
        K,
        x.stride(0),
        x_fp8.stride(0),
        x_scales.stride(0),
        FP8_MAX=448.0,
        GROUP_K=group_k,
        BLOCK_M=BLOCK_M,
    )
    return x_fp8, x_scales


def quant_gather(
    ctx,                        # DispatchContext or BlockDispatchContext
    x: torch.Tensor,            # bf16 [M, K] — source input, must be contiguous
    gather_idx: torch.Tensor,   # int32 [EM] — output row → source row mapping
    group_k: int = 128,
    BLOCK_M: int = 64,
):
    """Fused per-token-group FP8 quantization + gather, writing directly into pre-allocated buffers.

    Combines quant_gather and the send_buf copy into a single kernel:
    reads x[gather_idx], quantizes to FP8, and stores the result directly into
    ctx.send_buf[:EM] and ctx.send_scales[:EM], avoiding an extra copy.

    Args:
        ctx: DispatchContext or BlockDispatchContext providing send_buf, send_scales, num_sms.
        x: Input tensor [M, K] in bf16, must be contiguous.
        gather_idx: Gather indices [EM] mapping output row → input row.
        group_k: Quantization group size along K dimension.
        BLOCK_M: Rows per thread block.
    """
    out_fp8 = ctx.send_buf
    out_scales = ctx.send_scales
    M, K = x.shape
    EM = gather_idx.shape[0]
    assert K % group_k == 0, f"K({K}) must be divisible by group_k({group_k})"
    assert x.dtype == torch.bfloat16, f"Input must be bf16, got {x.dtype}"
    assert x.is_contiguous(), "Input must be contiguous"
    assert out_fp8.shape[1] >= K, f"out_fp8 K dim {out_fp8.shape[1]} < {K}"
    assert out_scales.shape[1] >= K // group_k

    num_groups = K // group_k
    row_blocks = cdiv(EM, BLOCK_M)

    # ---- K-dimension parallelism (2-D grid) ----
    num_sms = ctx.num_sms
    target_ctas = num_sms * 2
    if row_blocks >= target_ctas:
        BLOCK_G = num_groups
    else:
        grid_g = min(num_groups, max(1, cdiv(target_ctas, row_blocks)))
        BLOCK_G = cdiv(num_groups, grid_g)
    BLOCK_G = max(1, min(BLOCK_G, num_groups))

    grid = (row_blocks, cdiv(num_groups, BLOCK_G))
    per_token_quant_gather_kernel[grid](
        x, gather_idx, out_fp8, out_scales,
        K,
        x.stride(0),
        out_fp8.stride(0),
        out_scales.stride(0),
        EM,
        num_groups,
        FP8_MAX=448.0,
        GROUP_K=group_k,
        BLOCK_M=BLOCK_M,
        BLOCK_G=BLOCK_G,
    )
# ============================================================================
# NVSHMEM barrier kernel — lightweight cross-PE sync via signal-pad.
# ============================================================================

@requires_nvshmem
@triton.jit
def _nvshmem_barrier_all_kernel():
    """1-CTA kernel calling nvshmem_barrier_all. All PEs must launch."""
    nvshmem.barrier_all()


# ============================================================================
# Warp Dispatch: warp-level getmem_warp (pull mode) → 3D expert-contiguous output
# ============================================================================
#
# Uses warp-level nvshmemx_getmem_warp (blocking) for per-expert pull dispatch.
# Each CTA handles one or more local experts, pulling tokens from all source
# ranks via NVSHMEM getmem.  The recv_offset is accumulated on-the-fly, so the
# output is compacted into [EPR, WS*MAX_M, K] with no holes — no Phase 2 needed.
#
# Pipeline:
#   quant_gather (unchanged, separate kernel)
#   → copy cumsum to symmetric split_send + barrier (host-side, lightweight)
#   → _dispatch_warp_pull_kernel (in-kernel getmem_warp pulls cumsum + token data)
#   → fp8_group_gemm_3d (group GEMM with per-expert input, no m_indices needed)

@requires_nvshmem
@triton.jit
def _dispatch_warp_pull_kernel(
    # Symmetric send buffers (source for gets, pre-filled by quant_gather)
    data_src,           # symmetric fp8 [MAX_M, K]
    scale_src,          # symmetric fp32 [MAX_M, NSG]
    # Symmetric cumsum buffer (source for in-kernel cumsum pull)
    split_send,         # symmetric int32 [NUM_EXPERTS + 1] — local rank's cumsum
    # Local buffer for all ranks' cumsum (filled in-kernel by getmem_warp)
    all_splits,         # int32 [WS * (NUM_EXPERTS + 1)] — filled by kernel
    # Local recv buffers (write targets, 3D expert-contiguous)
    recv_data,          # fp8 [EPR * WS * MAX_M, K] (allocated by caller)
    recv_scales,        # fp32 [EPR * WS * MAX_M, NSG]
    # Local output
    expert_recv_count,  # int32 [EPR] — output (written inside kernel)
    # Scalar params
    rank,
    # Constants
    K: tl.constexpr,
    NSG: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    """Warp-per-rank pull dispatch: in-kernel cumsum pull + parallel token pull.

    Phase 1: Each warp r pulls rank r's full cumsum (CUMSUM_STRIDE * 4 bytes)
    from peer's split_send into local all_splits via getmem_warp.  This replaces
    host-side NCCL all_gather — the pull overlaps with NVLink warmup for Phase 2.

    Phase 2: For each expert, 8 warps simultaneously pull tokens from their
    assigned rank via getmem_warp.  Each warp independently computes recv_offset
    from the locally-filled all_splits.

    No signal exchange, no grid barrier, no Phase 2 compaction.
    """
    FP8_BYTES: tl.constexpr = 1
    F32_BYTES: tl.constexpr = 4
    WARP_SIZE: tl.constexpr = 32

    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    threadidx = tid(axis=0)
    warp_id = threadidx // WARP_SIZE

    WS_MAX_M: tl.constexpr = WORLD_SIZE * MAX_M
    CUMSUM_STRIDE: tl.constexpr = NUM_EXPERTS + 1

    # Convert constexpr sizes to runtime int64 tensors for extern calls.
    k_bytes = (pid * 0 + K * FP8_BYTES).to(tl.int64)
    nsg_bytes = (pid * 0 + NSG * F32_BYTES).to(tl.int64)
    cumsum_bytes = (pid * 0 + CUMSUM_STRIDE * F32_BYTES).to(tl.int64)

    # Per-CTA region in all_splits: each CTA writes to its own partition to
    # avoid concurrent getmem_warp writes to the same address (NVSHMEM UB).
    my_split_base = pid * WORLD_SIZE * CUMSUM_STRIDE

    # --- Phase 1: In-kernel cumsum pull (one-time, before expert loop) ---
    # Each warp pulls the full cumsum array from its assigned rank.
    # 8 warps × 1540 bytes = 12.3KB total, via NVLink (very fast).
    r = warp_id
    if r < WORLD_SIZE:
        _getmem_warp_extern_wrapper(
            (all_splits + my_split_base + r * CUMSUM_STRIDE).to(tl.int64),
            split_send.to(tl.int64),
            cumsum_bytes,
            r,
        )

    __syncthreads()

    # --- Phase 2: Warp-per-rank parallel token pull ---
    # Each CTA handles one or more local experts in round-robin
    for expert_idx in range(pid, EXPERTS_PER_RANK, num_ctas):
        g = rank * EXPERTS_PER_RANK + expert_idx  # global expert id

        # Each warp handles one source rank; all warps work in parallel.
        r = warp_id
        if r < WORLD_SIZE:
            # Read this rank's cumsum for expert g (from local all_splits)
            split_base = my_split_base + r * CUMSUM_STRIDE
            m_st = tl.load(all_splits + split_base + g)
            m_ed = tl.load(all_splits + split_base + g + 1)
            count = m_ed - m_st

            if count > 0:
                # Compute recv_offset: sum of counts from ranks 0..r-1
                recv_off = 0
                for prev_r in range(r):
                    prev_base = my_split_base + prev_r * CUMSUM_STRIDE
                    prev_st = tl.load(all_splits + prev_base + g)
                    prev_ed = tl.load(all_splits + prev_base + g + 1)
                    recv_off += (prev_ed - prev_st)

                dst_row_base = expert_idx * WS_MAX_M + recv_off

                # This warp handles ALL tokens for (expert, rank) sequentially.
                for token_off in range(count):
                    token_idx = m_st + token_off
                    dst_row = dst_row_base + token_off

                    # Get fp8 data: K bytes (warp-level blocking)
                    _getmem_warp_extern_wrapper(
                        (recv_data + dst_row * K).to(tl.int64),
                        (data_src + token_idx * K).to(tl.int64),
                        k_bytes,
                        r,
                    )
                    # Get scales: NSG * 4 bytes (warp-level blocking)
                    _getmem_warp_extern_wrapper(
                        (recv_scales + dst_row * NSG).to(tl.int64),
                        (scale_src + token_idx * NSG).to(tl.int64),
                        nsg_bytes,
                        r,
                    )

        # Compute total recv_count for this expert (warp 0 does it)
        if warp_id == 0:
            total = 0
            for rr in range(WORLD_SIZE):
                sb = my_split_base + rr * CUMSUM_STRIDE
                total += (tl.load(all_splits + sb + g + 1) - tl.load(all_splits + sb + g))
            tl.store(expert_recv_count + expert_idx, total)

        __syncthreads()


def dispatch_warp(
    ctx: DispatchContext,
    num_tokens: int,                    # actual number of gathered tokens in send_buf (= M * topk)
    send_split_cumsum: torch.Tensor,   # int32 [num_experts + 1]
    num_input_tokens: int,             # M — original token count (pre-gather), for recv allocation
    BM: int = 64,
):
    """Warp pull dispatch: NVSHMEM getmem_warp pull → 3D expert-contiguous output.
    Output layout: [EXPERTS_PER_RANK, M*WS, K] (expert-contiguous, compact).

    Each expert can receive at most M*WS tokens (M per rank * WS ranks), so the
    recv buffer is allocated with M*WS rows per expert — strictly tight, no topk slack.

    Assumes ``ctx.send_buf`` and ``ctx.send_scales`` have already been filled
    (e.g. by ``quant_gather``) with the FP8 data for ``num_tokens`` rows.

    Returns:
        recv_data: fp8 [EPR, M*WS, K]
        recv_scales: fp32 [EPR, M*WS, NSG]
        expert_recv_count: int32 [EPR]

    Note: recv_data/recv_scales are views of ctx symmetric buffers and are
    overwritten by the next dispatch_warp call.
    """
    assert num_tokens <= ctx.max_m

    EXPERTS_PER_RANK = ctx.experts_per_rank
    WS = ctx.world_size
    K = ctx.hidden
    NSG = ctx.num_scale_groups
    NUM_EXPERTS = ctx.num_experts
    device = ctx.send_buf.device

    # Copy cumsum to symmetric buffer, then NVSHMEM barrier (signal-pad, not NCCL).
    # Stream ordering: copy_() → barrier kernel → dispatch kernel.
    ctx.split_send[:NUM_EXPERTS + 1].copy_(send_split_cumsum)
    _nvshmem_barrier_all_kernel[(1,)](num_warps=1)

    # Launch pull dispatch kernel — grid = num SMs (enough for parallelism)
    grid = (max(ctx.num_sms, EXPERTS_PER_RANK),)

    # getmem_warp destinations must live in the symmetric heap (registered
    # memory) for the IB transport — plain torch.empty buffers have no lkey.
    # Per-CTA regions: each CTA writes to its own partition to avoid
    # concurrent getmem_warp writes to the same address (NVSHMEM UB).
    all_splits = ctx.all_splits_buf

    # Output views of the symmetric recv buffers (3D expert-contiguous, no holes)
    # Each expert can receive at most M*WS tokens — tightly allocated.
    RECV_MAX_M = num_input_tokens * WS
    recv_rows = EXPERTS_PER_RANK * RECV_MAX_M
    recv_data = ctx.recv_data.view(-1)[: recv_rows * K].view(EXPERTS_PER_RANK, RECV_MAX_M, K)
    recv_scales = ctx.recv_scales.view(-1)[: recv_rows * NSG].view(EXPERTS_PER_RANK, RECV_MAX_M, NSG)
    expert_recv_count = torch.empty(
        EXPERTS_PER_RANK, dtype=torch.int32, device=device)

    _dispatch_warp_pull_kernel[grid](
        ctx.send_buf,
        ctx.send_scales,
        ctx.split_send,
        all_splits,
        recv_data, recv_scales,
        expert_recv_count,
        ctx.rank,
        K=K, NSG=NSG, MAX_M=num_input_tokens,
        EXPERTS_PER_RANK=EXPERTS_PER_RANK,
        NUM_EXPERTS=NUM_EXPERTS,
        WORLD_SIZE=WS,
        NUM_WARPS=8,
        num_warps=8,
    )

    ctx.call_count = (ctx.call_count + 1) % ctx.MOD_VALUE

    return recv_data, recv_scales, expert_recv_count


# ============================================================================
# Group GEMM 3D — accepts [EXPERTS_PER_RANK, WS*MAX_M, K] input (dispatch output)
# ============================================================================
#
# Uses upper-bound grid (EPR * max_blocks_per_expert) — no DtoH sync needed.
# Blocks beyond an expert's valid token count early-return in the kernel.
# Expert ID and local row are derived from pid_m via integer division, so no
# m_indices table or _build_m_indices_kernel is needed.

@triton.jit
def moe_fp8_groupgemm_kernel_3d(
    A,              # fp8 [EPR * WS_MAX_M, K] (3D viewed as flat)
    B,              # fp8 [E, N, K]
    C,              # output [EPR * WS_MAX_M, N] (3D viewed as flat)
    A_scales,       # [EPR * WS_MAX_M, K // group_k]
    B_scales,       # [E, N // group_n, K // group_k]
    expert_recv_count,  # [EPR] int32 — valid tokens per expert
    self_signal,    # int32 [EPR] — local per-expert signal buf (acquire poll)
    N,
    K,
    E,
    group_n,
    group_k,
    A_stride_m,
    A_stride_k,
    A_s_stride_m,
    A_s_stride_k,
    B_stride_e,
    B_stride_n,
    B_stride_k,
    B_s_stride_e,
    B_s_stride_n,
    B_s_stride_k,
    C_stride_m,
    C_stride_n,
    WS_MAX_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MAX_BLOCKS_PER_EXPERT: tl.constexpr,
    USE_ACQUIRE_POLL: tl.constexpr,  # gate per-expert _wait_signal_local poll
    OUT_DTYPE: tl.constexpr,
    swap_ab: tl.constexpr = False,
):
    """Block-wise FP8 group GEMM with 3D [EPR, WS*MAX_M, K] input layout.

    Derives expert_id and local_row directly from pid_m via integer division:
      expert_id = pid_m // MAX_BLOCKS_PER_EXPERT
      local_row = (pid_m % MAX_BLOCKS_PER_EXPERT) * BLOCK_M
    Blocks beyond an expert's valid token count early-return.
    Contiguous access — no gather table or m_indices needed.

    Overlap: each CTA polls _wait_signal_local(self_signal[expert_id]) before
    reading any recv_buf tile so it doesn't race the dispatch pull kernel.
    Once the acquire-load sees 1, all prior dispatch stores (data, scales,
    and local_recv_count write) are visible via device-scope release-acquire.

    swap_ab (as in sglang's fused_moe_kernel): swaps the dot operands so
    BLOCK_N rides the MMA M-dim, which helps small BLOCK_M tiles on SM90.
    """
    pid = tl.program_id(0)
    num_block_n = tl.cdiv(N, BLOCK_N)

    # Total M-blocks across all experts (host-known upper bound, no DtoH sync)
    total_m_blocks = E * MAX_BLOCKS_PER_EXPERT

    # Group-M tiling for L2 cache locality
    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(total_m_blocks - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    # Derive expert_id and local_row from pid_m (no m_indices table)
    expert_id = pid_m // MAX_BLOCKS_PER_EXPERT
    local_row = (pid_m % MAX_BLOCKS_PER_EXPERT) * BLOCK_M

    # -- Acquire poll: wait until pull kernel has written expert_id's tiles --
    if USE_ACQUIRE_POLL:
        if tid(0) == 0:
            _wait_signal_local(self_signal + expert_id)
        __syncthreads()

    # Early return if beyond this expert's valid tokens
    valid_count = tl.load(expert_recv_count + expert_id)
    if local_row >= valid_count:
        return

    flat_row_start = expert_id * WS_MAX_M + local_row

    offs_m = tl.arange(0, BLOCK_M)
    token_mask = (local_row + offs_m) < valid_count

    offs_token = flat_row_start + offs_m
    # Clamp padding rows to 0 so pointer arithmetic stays in-bounds
    offs_token = tl.where(token_mask, offs_token, 0)

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_token[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    b_ptrs = B + expert_id.to(tl.int64) * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    As_ptrs = A_scales + offs_token * A_s_stride_m

    offs_bsn = pid_n * BLOCK_N // group_n
    Bs_ptrs = B_scales + expert_id.to(tl.int64) * B_s_stride_e + offs_bsn * B_s_stride_n

    n_tiles_k_per_group_k = group_k // BLOCK_K

    if swap_ab:
        accumulator = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs, mask=token_mask, other=0.0)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        if swap_ab:
            # Operands swapped; scales follow: b_s (scalar) broadcast over rows, a_s over cols
            accumulator += tl.dot(tl.trans(b, (1, 0)), tl.trans(a, (1, 0))) * (
                b_s * a_s[None, :]
            )
        else:
            accumulator += tl.dot(a, b) * (a_s[:, None] * b_s)

        a_ptrs += BLOCK_K * A_stride_k
        b_ptrs += BLOCK_K * B_stride_k
        As_ptrs += scale_step_k * A_s_stride_k
        Bs_ptrs += scale_step_k * B_s_stride_k

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    accumulator = accumulator.to(OUT_DTYPE)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_token[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# swap_ab benefits SM90 GPUs (H20, H100, H200, etc.) for BLOCK_M < 64.
@functools.lru_cache(maxsize=8)
def _should_enable_swap_ab(BLOCK_M: int, BLOCK_N: int) -> bool:
    if BLOCK_M >= 64 or BLOCK_N < 64:
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # no CUDA device visible
        return False
    return (major, minor) == (9, 0)


def _pick_block_m(avg_tokens_per_expert: float) -> int:
    """Pick the GEMM tile M from the expected token count per expert.

    Small per-expert workloads want small tiles (less padding waste);
    small tiles then get swap_ab to restore MMA granularity on SM90.
    """
    if avg_tokens_per_expert < 16:
        return 16
    if avg_tokens_per_expert < 64:
        return 32
    return 64


def fp8_group_gemm_3d(
    input_3d: torch.Tensor,          # fp8 [EPR, M*WS, K]
    weight: torch.Tensor,            # fp8 [E, N, K]
    a_scales_3d: torch.Tensor,       # fp32 [EPR, M*WS, K // group_k]
    b_scales: torch.Tensor,          # fp32 [E, N // group_n, K // group_k]
    expert_recv_count: torch.Tensor, # int32 [EPR]
    group_n: int = 128,
    group_k: int = 128,
    BLOCK_M: int = None,  # None = auto (requires num_tokens_per_rank + TOPK), else 64
    num_tokens_per_rank: int = None,  # routing info for auto BLOCK_M
    TOPK: int = None,  # routing info for auto BLOCK_M
    output_dtype: torch.dtype = torch.bfloat16,
    swap_ab: bool = None,  # None = auto-detect (SM90 + BLOCK_M < 64)
    self_signal: torch.Tensor = None,    # int32 [EPR] - local signal buf, set by dispatch pull
):
    """3D FP8 group GEMM — accepts [EPR, M*WS, K] input from dispatch.

    Uses upper-bound grid (EPR * max_blocks_per_expert) — no DtoH sync needed.
    Blocks beyond an expert's valid token count early-return in the kernel.
    WS_MAX_M (rows per expert) is derived from input_3d.shape — no max_m needed.

    Args:
        input_3d: FP8 activations [EPR, M*WS, K], expert-contiguous (3D).
        weight: FP8 weight [E, N, K].
        a_scales_3d: Per-token-group scales [EPR, M*WS, K // group_k].
        b_scales: Per-expert per-block B scales [E, N // group_n, K // group_k].
        expert_recv_count: Valid token count per expert [EPR].
        self_signal: (optional) int32 [EPR] local per-expert signal buffer.
        BLOCK_M: None = auto tile from routing info (num_tokens_per_rank *
            TOPK / EPR, see _pick_block_m); falls back to 64 when the routing
            info is not provided.
    """
    EPR, WS_MAX_M, K = input_3d.shape
    E, N, _K = weight.shape

    if BLOCK_M is None:
        if num_tokens_per_rank is None or TOPK is None:
            BLOCK_M = 64
        else:
            # Expected hits per expert under uniform routing
            BLOCK_M = _pick_block_m(num_tokens_per_rank * TOPK / EPR)

    use_acquire_poll = self_signal is not None
    if use_acquire_poll:
        signal_arg = self_signal
    else:
        # Stub: kernel unconditionally derefs self_signal but USE_ACQUIRE_POLL
        # is False, so the poll body never executes.
        signal_arg = torch.zeros(1, dtype=torch.int32, device=input_3d.device)

    # View 3D as flat 2D for kernel pointer arithmetic
    input_flat = input_3d.view(EPR * WS_MAX_M, K)
    scales_flat = a_scales_3d.view(EPR * WS_MAX_M, K // group_k)
    output = torch.empty(EPR * WS_MAX_M, N, dtype=output_dtype, device=input_3d.device)

    # GEMM grid sizing — WS_MAX_M is already the tight upper bound (M*WS).
    max_blocks_per_expert = cdiv(WS_MAX_M, BLOCK_M)
    total_m_blocks = EPR * max_blocks_per_expert

    num_block_n = cdiv(N, 128)
    grid = (total_m_blocks * num_block_n,)
    BLOCK_N = 128
    BLOCK_K = group_k
    GROUP_M = 8

    if swap_ab is None:
        swap_ab = _should_enable_swap_ab(BLOCK_M, BLOCK_N)

    moe_fp8_groupgemm_kernel_3d[grid](
        input_flat,
        weight,
        output,
        scales_flat,
        b_scales,
        expert_recv_count,
        signal_arg,
        N,
        K,
        E,
        group_n,
        group_k,
        input_flat.stride(0),
        input_flat.stride(1),
        scales_flat.stride(0),
        scales_flat.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        b_scales.stride(0),
        b_scales.stride(1),
        b_scales.stride(2),
        output.stride(0),
        output.stride(1),
        WS_MAX_M=WS_MAX_M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        MAX_BLOCKS_PER_EXPERT=max_blocks_per_expert,
        USE_ACQUIRE_POLL=use_acquire_poll,
        OUT_DTYPE=_TORCH_TO_TRITON_DTYPE[output_dtype],
        swap_ab=swap_ab,
        num_stages=3,
        num_warps=4,
    )

    return output.view(EPR, WS_MAX_M, N)


# ============================================================================
# Group GEMM 2D — accepts [WS*MAX_M, K] input (block dispatch output)
# ============================================================================
#
# Block dispatch recv layout is [WS*MAX_M, K] — per-rank, within each rank
# sorted by expert. This is a 2D layout, NOT the 3D [EXPERTS_PER_RANK, WS*MAX_M, K]
# expert-contiguous layout used by the warp dispatch.
#
# To compute GEMM on this layout, we build m_indices that map each GEMM
# block to its starting row in the 2D buffer, plus expert_ids that tell
# which weight to use. Each (rank, expert) pair may produce multiple blocks.


@triton.jit
def _build_m_indices_2d_kernel(
    recv_splits,          # int32 [WS * EXPERTS_PER_RANK] — per-(rank, expert) counts
    row_offsets,          # int32 [WS*MAX_M] — output: actual row in 2D buffer per gathered token
    m_indices,            # int32 [max_blocks] — output: offset into row_offsets per GEMM block
    expert_ids,           # int32 [max_blocks] — output: expert id per block
    counts_per_row,       # int32 [max_blocks] — output: valid token count per block
    expert_recv_count,    # int32 [EXPERTS_PER_RANK] — output: total tokens per expert (all ranks merged)
    block_counter,        # int32 [1] — zero-init
    row_counter,          # int32 [1] — zero-init
    BLOCK_M: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Build gather-based index tables for 2D per-rank layout.

    Merges all ranks' tokens for each expert into a single contiguous segment
    in row_offsets, so the GEMM kernel sees ~EXPERTS_PER_RANK blocks (not EXPERTS_PER_RANK*WS blocks).

    For expert e:
      total_rows = sum over r of recv_splits[r*EXPERTS_PER_RANK + e]
      row_offsets[row_start .. row_start+total_rows] = actual row indices
      num_blocks = cdiv(total_rows, BLOCK_M)
      m_indices[b] = row_start + b*BLOCK_M
      expert_ids[b] = e
      counts_per_row[b] = min(BLOCK_M, total_rows - b*BLOCK_M)

    Grid: (EXPERTS_PER_RANK,) — one CTA per local expert.
    """
    expert_id = tl.program_id(0)

    # Count total rows for this expert across all ranks
    total_rows = 0
    for r in range(WORLD_SIZE):
        count_r = tl.load(recv_splits + r * EXPERTS_PER_RANK + expert_id)
        total_rows += count_r
    num_blocks = tl.cdiv(total_rows, BLOCK_M)

    # Atomic allocate block range and row range
    block_off = tl.atomic_add(block_counter, num_blocks)
    row_off = tl.atomic_add(row_counter, total_rows)

    # Write row_offsets: gather all ranks' token rows into contiguous segment
    idx = row_off
    for r in range(WORLD_SIZE):
        count_r = tl.load(recv_splits + r * EXPERTS_PER_RANK + expert_id)
        if count_r > 0:
            prefix = 0
            for e_prev in range(expert_id):
                prefix += tl.load(recv_splits + r * EXPERTS_PER_RANK + e_prev)
            flat_start = r * MAX_M + prefix
            for j in range(count_r):
                tl.store(row_offsets + idx, flat_start + j)
                idx += 1

    # Write index tables
    tl.store(expert_recv_count + expert_id, total_rows)
    for b in range(num_blocks):
        tl.store(m_indices + block_off + b, row_off + b * BLOCK_M)
        tl.store(expert_ids + block_off + b, expert_id)
        remaining = total_rows - b * BLOCK_M
        tl.store(counts_per_row + block_off + b, tl.minimum(remaining, BLOCK_M))


@triton.jit
def fp8_groupgemm_kernel_2d(
    A,              # fp8 [WS*MAX_M, K] — per-rank layout from block dispatch
    B,              # fp8 [E, N, K]
    C,              # output [WS*MAX_M, N]
    A_scales,       # [WS*MAX_M, K // group_k]
    B_scales,       # [E, N // group_n, K // group_k]
    row_offsets,    # [total_rows] int32 — actual row index in A for each gathered token
    m_indices,      # [total_blocks] int32 — offset into row_offsets per GEMM block
    expert_ids,     # [total_blocks] int32 — expert id per block
    counts_per_row, # [total_blocks] int32 — valid token count per block
    total_blocks_ptr,  # int32 [1] — actual block count on device (avoids DtoH sync)
    N,
    K,
    E,
    group_n,
    group_k,
    A_stride_m,
    A_stride_k,
    A_s_stride_m,
    A_s_stride_k,
    B_stride_e,
    B_stride_n,
    B_stride_k,
    B_s_stride_e,
    B_s_stride_n,
    B_s_stride_k,
    C_stride_m,
    C_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    swap_ab: tl.constexpr = False,
):
    """2D FP8 group GEMM with gather — merges all ranks' tokens per expert.

    Each block loads row_offsets[m_indices[pid_m] : +BLOCK_M] to get the
    actual row indices in A (which may be non-contiguous across ranks),
    then computes A[gathered rows] @ B[expert_id]^T → C.

    swap_ab (as in sglang's fused_moe_kernel): swaps the dot operands so
    BLOCK_N rides the MMA M-dim, which helps small BLOCK_M tiles on SM90.
    """
    pid = tl.program_id(0)
    num_block_n = tl.cdiv(N, BLOCK_N)

    # Load actual block count from device memory — avoids DtoH sync
    total_blocks = tl.load(total_blocks_ptr)
    if pid >= total_blocks * num_block_n:
        return

    # Group-M tiling for L2 cache locality
    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(total_blocks - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    row_ptr_base = tl.load(m_indices + pid_m)
    expert_id = tl.load(expert_ids + pid_m)
    remaining = tl.load(counts_per_row + pid_m)

    offs_m = tl.arange(0, BLOCK_M)
    token_mask = offs_m < remaining

    # Gather: load actual row indices from row_offsets table
    offs_token = tl.load(row_offsets + row_ptr_base + offs_m, mask=token_mask, other=0)

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_token[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    b_ptrs = B + expert_id.to(tl.int64) * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    As_ptrs = A_scales + offs_token * A_s_stride_m

    offs_bsn = pid_n * BLOCK_N // group_n
    Bs_ptrs = B_scales + expert_id.to(tl.int64) * B_s_stride_e + offs_bsn * B_s_stride_n

    n_tiles_k_per_group_k = group_k // BLOCK_K

    if swap_ab:
        accumulator = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs, mask=token_mask, other=0.0)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        if swap_ab:
            # Operands swapped; scales follow: b_s (scalar) broadcast over rows, a_s over cols
            accumulator += tl.dot(tl.trans(b, (1, 0)), tl.trans(a, (1, 0))) * (
                b_s * a_s[None, :]
            )
        else:
            accumulator += tl.dot(a, b) * (a_s[:, None] * b_s)

        a_ptrs += BLOCK_K * A_stride_k
        b_ptrs += BLOCK_K * B_stride_k
        As_ptrs += scale_step_k * A_s_stride_k
        Bs_ptrs += scale_step_k * B_s_stride_k

    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))

    accumulator = accumulator.to(OUT_DTYPE)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_token[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def fp8_group_gemm_2d(
    input: torch.Tensor,            # fp8 [WS*max_m, K] — per-rank layout
    weight: torch.Tensor,           # fp8 [E, N, K] — local experts only [EXPERTS_PER_RANK, N, K]
    a_scales: torch.Tensor,         # fp32 [WS*max_m, K // group_k]
    b_scales: torch.Tensor,         # fp32 [EXPERTS_PER_RANK, N // group_n, K // group_k]
    recv_splits: torch.Tensor,      # int32 [WS * EXPERTS_PER_RANK] — per-(rank, expert) counts
    MAX_M: int,
    EXPERTS_PER_RANK: int,
    WS: int,
    group_n: int = 128,
    group_k: int = 128,
    BLOCK_M: int = None,  # None = auto (requires num_tokens_per_rank + TOPK), else 64
    num_tokens_per_rank: int = None,  # routing info for auto BLOCK_M
    TOPK: int = None,  # routing info for auto BLOCK_M
    output_dtype: torch.dtype = torch.bfloat16,
    swap_ab: bool = None,  # None = auto-detect (SM90 + BLOCK_M < 64)
):
    """2D FP8 group GEMM for block-level dispatch output.

    Args:
        input: FP8 activations [WS*MAX_M, K] from block dispatch.
        weight: FP8 weight [EXPERTS_PER_RANK, N, K] — local experts only.
        a_scales: Per-token-group scales [WS*MAX_M, K // group_k].
        b_scales: Per-expert per-block B scales [EXPERTS_PER_RANK, N // group_n, K // group_k].
        recv_splits: Per-(rank, expert) counts [WS * EXPERTS_PER_RANK].
        MAX_M: Per-rank recv buffer capacity (for buffer indexing).
        EXPERTS_PER_RANK: Experts per rank.
        WS: World size.
        BLOCK_M: None = auto tile from routing info (num_tokens_per_rank *
            TOPK / EXPERTS_PER_RANK, see _pick_block_m); falls back to 64
            when the routing info is not provided.
    """
    WS_MAX_M = WS * MAX_M
    K = input.shape[1]
    E = weight.shape[0]  # EXPERTS_PER_RANK local experts
    N = weight.shape[1]

    if BLOCK_M is None:
        if num_tokens_per_rank is None or TOPK is None:
            BLOCK_M = 64
        else:
            # Expected hits per expert under uniform routing
            BLOCK_M = _pick_block_m(num_tokens_per_rank * TOPK / EXPERTS_PER_RANK)

    # Build row_offsets (gather table) + m_indices + expert_ids + counts_per_row
    max_total_rows = WS_MAX_M
    max_total_blocks = EXPERTS_PER_RANK * cdiv(WS_MAX_M, BLOCK_M)
    row_offsets = torch.empty(max_total_rows, dtype=torch.int32, device=input.device)
    m_indices = torch.empty(max_total_blocks, dtype=torch.int32, device=input.device)
    expert_ids = torch.empty(max_total_blocks, dtype=torch.int32, device=input.device)
    counts_per_row = torch.empty(max_total_blocks, dtype=torch.int32, device=input.device)
    expert_recv_count = torch.empty(EXPERTS_PER_RANK, dtype=torch.int32, device=input.device)
    block_counter = torch.zeros(1, dtype=torch.int32, device=input.device)
    row_counter = torch.zeros(1, dtype=torch.int32, device=input.device)

    _build_m_indices_2d_kernel[(EXPERTS_PER_RANK,)](
        recv_splits,
        row_offsets,
        m_indices, expert_ids, counts_per_row,
        expert_recv_count,
        block_counter, row_counter,
        BLOCK_M=BLOCK_M, MAX_M=MAX_M, EXPERTS_PER_RANK=EXPERTS_PER_RANK, WORLD_SIZE=WS,
    )

    # No DtoH sync — use host-known upper bound for grid, kernel early-returns
    # for blocks beyond the actual count (loaded from block_counter on device).

    output = torch.empty(WS_MAX_M, N, dtype=output_dtype, device=input.device)

    num_block_n = cdiv(N, 128)
    grid = (max_total_blocks * num_block_n,)
    BLOCK_N = 128
    BLOCK_K = group_k
    GROUP_M = 8

    if swap_ab is None:
        swap_ab = _should_enable_swap_ab(BLOCK_M, BLOCK_N)

    fp8_groupgemm_kernel_2d[grid](
        input,
        weight,
        output,
        a_scales,
        b_scales,
        row_offsets,
        m_indices,
        expert_ids,
        counts_per_row,
        block_counter,
        N, K, E,
        group_n, group_k,
        input.stride(0), input.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        weight.stride(0), weight.stride(1), weight.stride(2),
        b_scales.stride(0), b_scales.stride(1), b_scales.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        OUT_DTYPE=_TORCH_TO_TRITON_DTYPE[output_dtype],
        swap_ab=swap_ab,
        num_stages=3,
        num_warps=4,
    )

    return output


# ============================================================================
# Block Dispatch (reference: tutorials/04-deepseek-infer-all2all.py)
# ============================================================================
#
# Simple block-level FP8 dispatch using torch NVSHMEM putmem_signal_block.
# Each CTA (one per target rank) sends all tokens destined for that rank's
# experts in one contiguous block — same pattern as the tutorial.
#
# Differences from the warp dispatch above:
#   - Block-level put (not per-token warp-level)
#   - Double-buffered recv layout: [WS * MAX_M * 2, K] (per-rank, not per-expert)
#   - No postprocess inside kernel — host-side slicing + concat like tutorial
#   - Simpler: no grid barrier, no warp-level extern wrapper
#
# Signal scheme: NVSHMEM_SIGNAL_ADD, 3 putmem_signal_block calls per dispatch
# (data + scales + splits), each adding 1 to the signal.


class BlockDispatchContext:
    """Pre-allocated NVSHMEM buffers for block-level FP8 dispatch.

    Buffer layout mirrors tutorials/04-deepseek-infer-all2all.py:
      recv_buf: [WS * MAX_M * 2, K] double-buffered, per-rank [rank][MAX_M] layout
      recv_scales: [WS * MAX_M * 2, NSG] double-buffered
      split_recv: [NUM_EXPERTS * 2] double-buffered, per-(rank, expert) counts
      signal: [WS * 2] int64, double-buffered

    Uses symm_mem for symmetric buffer creation (same as DispatchContext).
    """

    def __init__(
        self,
        max_m: int,
        hidden: int,
        num_experts: int,
        world_size: int,
        rank: int,
        num_scale_groups: int,
        group_name: str,
    ):
        dev = f"cuda:{rank}"
        epr = num_experts // world_size

        self.send_buf = self._create_tensor((max_m, hidden), FP8_DTYPE, dev, group_name)
        self.send_scales = self._create_tensor((max_m, num_scale_groups), torch.float32, dev, group_name)

        self.recv_buf = self._create_tensor(
            (world_size * max_m * 2, hidden), FP8_DTYPE, dev, group_name)
        self.recv_scales = self._create_tensor(
            (world_size * max_m * 2, num_scale_groups), torch.float32, dev, group_name)

        self.split_send = self._create_tensor((num_experts,), torch.int32, dev, group_name)
        self.split_recv = self._create_tensor((num_experts * 2,), torch.int32, dev, group_name)

        self.signal = self._create_tensor((world_size * 2,), torch.int64, dev, group_name)
        self.signal.fill_(0)

        self.max_m = max_m
        self.hidden = hidden
        self.num_experts = num_experts
        self.experts_per_rank = epr
        self.num_scale_groups = num_scale_groups
        self.world_size = world_size
        self.rank = rank
        self.num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.call_count = 1

    @staticmethod
    def _create_tensor(shape, dtype, device, group_name):
        """Create a symmetric memory tensor via symm_mem (same as DispatchContext)."""
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            tensor = symm_mem.empty(*shape, dtype=torch.int8, device=device)
            tensor = tensor.view(dtype)
        else:
            tensor = symm_mem.empty(*shape, dtype=dtype, device=device)
        symm_mem.rendezvous(tensor, group=group_name)
        return tensor

    def finalize(self):
        pass


def create_block_dispatch_context(
    max_m: int,
    hidden: int,
    num_experts: int,
    world_size: int,
    rank: int,
    num_scale_groups: int,
    group_name: str = None,
):
    """Create a BlockDispatchContext for block-level FP8 dispatch."""
    if group_name is None:
        group_name = dist.group.WORLD.group_name
    symm_mem.set_backend("NVSHMEM")
    return BlockDispatchContext(max_m, hidden, num_experts, world_size, rank, num_scale_groups, group_name)


@requires_nvshmem
@triton.jit
def _dispatch_block_kernel(
    data_src,        # symmetric fp8 [MAX_M, K] — local send buffer (pre-filled)
    data_dst,        # symmetric fp8 [WS * MAX_M * 2, K] — recv buffer (double-buffered)
    scale_src,       # symmetric fp32 [MAX_M, NSG] — local scale send buffer
    scale_dst,       # symmetric fp32 [WS * MAX_M * 2, NSG] — scale recv buffer
    splits_src,      # symmetric int32 [NUM_EXPERTS] — scratch for sending counts
    splits_dst,      # symmetric int32 [NUM_EXPERTS * 2] — per-(rank, expert) recv counts
    signal,          # symmetric int64 [WS * 2] — signal buffer (double-buffered)
    splits_cumsum,   # int32 [NUM_EXPERTS + 1] — cumsum provided by caller
    rank,
    call_count,
    K: tl.constexpr,
    NSG: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    EPR_POW2: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Block-level dispatch kernel — one CTA per target rank.

    Each CTA (pid = target rank) sends all tokens destined for pid's experts
    to PE pid using 3 putmem_signal_block calls:
      1. FP8 data  → data_dst  on PE pid + signal ADD 1
      2. FP32 scales → scale_dst on PE pid + signal ADD 1
      3. Int32 splits → splits_dst on PE pid + signal ADD 1
    Then waits for PE pid's data to arrive via signal_wait_until.

    Signal buffer layout (on each PE):
      signal[act_pos * WORLD_SIZE + src_rank]
    Updated by: CTA pid on PE src_rank adds 1 (×3 calls) to signal[src_rank] on PE pid.
    Waited on:   signal[pid] on PE rank — updated by PE pid's CTAs sending to us.
    """
    NVSHMEM_SIGNAL_ADD: tl.constexpr = 10
    NVSHMEM_CMP_EQ: tl.constexpr = 0
    FP8_BYTES: tl.constexpr = 1
    F32_BYTES: tl.constexpr = 4

    pid = tl.program_id(0)
    threadidx = tid(axis=0)

    act_pos = call_count % 2

    # Source segment in send_buf: tokens for pid's experts (global expert range)
    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK
    m_st = tl.load(splits_cumsum + exp_st)
    m_ed = tl.load(splits_cumsum + exp_ed)
    num_rows = m_ed - m_st

    # Destination on PE pid: our slot at [rank * MAX_M] in the active buffer half
    dst_off = rank * MAX_M
    data_dst_ptr = data_dst + (act_pos * WORLD_SIZE * MAX_M + dst_off) * K
    scale_dst_ptr = scale_dst + (act_pos * WORLD_SIZE * MAX_M + dst_off) * NSG
    split_dst_ptr = splits_dst + act_pos * NUM_EXPERTS + rank * EXPERTS_PER_RANK
    signal_ptr = signal + act_pos * WORLD_SIZE + rank

    # Compute & store per-expert splits for pid's experts
    split_src_ptr = splits_src + exp_st
    epr_offs = tl.arange(0, EPR_POW2)
    epr_mask = epr_offs < EXPERTS_PER_RANK
    off0 = exp_st + epr_offs
    off1 = off0 + 1
    cumsum_sts = tl.load(splits_cumsum + off0, mask=epr_mask)
    cumsum_eds = tl.load(splits_cumsum + off1, mask=epr_mask)
    tl.store(split_src_ptr + epr_offs, cumsum_eds - cumsum_sts, mask=epr_mask)

    # Signal scheme: 3 ADDs of 1 per dispatch per (src_rank → dst_rank) pair.
    # Each buffer position (act_pos) is used every 2 calls → expected = N * 3,
    # where N = (call_count + 1) // 2 is how many times this position was used.
    call_count_rt = pid * 0 + call_count
    expected_sig = ((call_count_rt.to(tl.int64) + 1) // 2) * 3
    one_t = call_count_rt * 0 + 1                    # runtime int32 tensor (= 1)
    split_bytes = call_count_rt * 0 + EXPERTS_PER_RANK * 4       # bytes for EXPERTS_PER_RANK int32 split entries

    # 1. Put FP8 data → PE pid + ADD signal
    nvshmem.putmem_signal_block(
        data_dst_ptr,
        data_src + m_st * K,
        num_rows * K * FP8_BYTES,
        signal_ptr,
        one_t,
        NVSHMEM_SIGNAL_ADD,
        pid,
    )

    # 2. Put FP32 scales → PE pid + ADD signal
    nvshmem.putmem_signal_block(
        scale_dst_ptr,
        scale_src + m_st * NSG,
        num_rows * NSG * F32_BYTES,
        signal_ptr,
        one_t,
        NVSHMEM_SIGNAL_ADD,
        pid,
    )

    # 3. Put int32 splits → PE pid + ADD signal
    nvshmem.putmem_signal_block(
        split_dst_ptr,
        split_src_ptr,
        split_bytes,
        signal_ptr,
        one_t,
        NVSHMEM_SIGNAL_ADD,
        pid,
    )

    # Wait for PE pid's 3 transfers to us (signal[pid] on PE rank updated by PE pid)
    if threadidx == 0:
        nvshmem.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            NVSHMEM_CMP_EQ,
            expected_sig,
        )


def dispatch_block(
    ctx: BlockDispatchContext,
    num_tokens: int,                    # actual number of gathered tokens in send_buf
    send_split_cumsum: torch.Tensor,   # int32 [num_experts + 1]
):
    """Block-level FP8 dispatch — one kernel, one CTA per target rank.

    Assumes ``ctx.send_buf`` and ``ctx.send_scales`` have already been filled
    (e.g. by ``quant_gather``) with the FP8 data for ``num_tokens`` rows.

    Returns active-half slices of recv buffers (views, not copies):
      recv_data:   fp8 [WS * MAX_M, K]
      recv_scales: fp32 [WS * MAX_M, NSG]
      recv_splits: int32 [NUM_EXPERTS]

    recv layout:
      recv_data[r * MAX_M : r * MAX_M + actual_count] = tokens from rank r
      recv_splits[r * EXPERTS_PER_RANK + e] = count from rank r for local expert e
    """
    assert num_tokens <= ctx.max_m

    K = ctx.hidden
    NSG = ctx.num_scale_groups
    MAX_M = ctx.max_m
    EXPERTS_PER_RANK = ctx.experts_per_rank
    WS = ctx.world_size
    NUM_EXPERTS = ctx.num_experts

    act_pos = ctx.call_count % 2

    grid = (WS,)
    _dispatch_block_kernel[grid](
        ctx.send_buf,
        ctx.recv_buf,
        ctx.send_scales,
        ctx.recv_scales,
        ctx.split_send,
        ctx.split_recv,
        ctx.signal,
        send_split_cumsum,
        ctx.rank,
        ctx.call_count,
        K=K,
        NSG=NSG,
        MAX_M=MAX_M,
        EXPERTS_PER_RANK=EXPERTS_PER_RANK,
        EPR_POW2=triton.next_power_of_2(EXPERTS_PER_RANK),
        NUM_EXPERTS=NUM_EXPERTS,
        WORLD_SIZE=WS,
    )

    ctx.call_count = ctx.call_count + 1

    data_st = act_pos * WS * MAX_M
    data_ed = data_st + WS * MAX_M
    split_st = act_pos * NUM_EXPERTS
    split_ed = split_st + NUM_EXPERTS

    return (
        ctx.recv_buf[data_st:data_ed],
        ctx.recv_scales[data_st:data_ed],
        ctx.split_recv[split_st:split_ed],
    )