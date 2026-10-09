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
MegaMoE kernels — FP8 quantization, NVLink load/store dispatch, and group GEMM

This module provides:
- Per-token-group FP8 quantization (Triton kernel)
- Block-wise FP8 weight quantization
- Block-level FP8 dispatch via intra-node NVLink load/store + host-side recv_hdl.barrier()
- 2D FP8 group GEMM with [WS*MAX_M, K] per-rank input layout
"""

from typing import Tuple
import functools

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import torch.distributed._symmetric_memory as symm_mem


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


# Use the default CUDA symmetric memory backend (not NVSHMEM), so that
# symm_mem handle .barrier() is a real GPU-side barrier (signal-pad based)
# rather than the NVSHMEM no-op stub.
# The CUDA backend uses CUDA IPC handles for cross-GPU symmetric memory and
# works on any multi-GPU node with NVLink — NVSHMEM is not required.


class BlockDispatchContext:
    """Pre-allocated symmetric memory buffers for block-level FP8 dispatch
    via direct NVLink load/store (intra-node, CUDA symm_mem backend).

    Buffer layout mirrors tutorials/04-deepseek-infer-all2all.py:
      recv_buf: [WS * MAX_M * 2, K] double-buffered, per-rank [rank][MAX_M] layout
      recv_scales: [WS * MAX_M * 2, NSG] double-buffered
      split_recv: [NUM_EXPERTS * 2] double-buffered, per-(rank, expert) counts

    Synchronization:
      cross-rank sync via recv_hdl.barrier() host-side collective, ensuring all
      ranks' NVLink stores are visible before GEMM reads them.
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

        # Precompute shapes for recv buffers
        recv_shape = (world_size * max_m * 2, hidden)
        recv_scales_shape = (world_size * max_m * 2, num_scale_groups)
        split_recv_shape = (num_experts * 2,)

        # Create symmetric buffers. Order must be identical on all ranks
        # because symm_mem.rendezvous() is a collective operation.
        #   1. send_buf    2. send_scales  3. recv_buf     4. recv_scales
        #   5. split_send  6. split_recv
        self.send_buf, _ = self._create_tensor((max_m, hidden), FP8_DTYPE, dev, group_name)
        self.send_scales, _ = self._create_tensor(
            (max_m, num_scale_groups), torch.float32, dev, group_name)

        self.recv_buf, self.recv_hdl = self._create_tensor(recv_shape, FP8_DTYPE, dev, group_name)
        self.recv_scales, scales_hdl = self._create_tensor(
            recv_scales_shape, torch.float32, dev, group_name)

        self.split_send, _ = self._create_tensor((num_experts,), torch.int32, dev, group_name)
        self.split_recv, split_hdl = self._create_tensor(
            split_recv_shape, torch.int32, dev, group_name)

        # Build peer pointer arrays for direct NVLink load/store.
        # Each element is the device address of peer rank r's buffer.
        self.peer_data_ptrs = self._build_peer_ptrs(
            self.recv_hdl, recv_shape, torch.int8, dev, world_size)
        self.peer_scale_ptrs = self._build_peer_ptrs(
            scales_hdl, recv_scales_shape, torch.float32, dev, world_size)
        self.peer_split_ptrs = self._build_peer_ptrs(
            split_hdl, split_recv_shape, torch.int32, dev, world_size)

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

    @staticmethod
    def _create_tensor(shape, dtype, device, group_name):
        """Create a symmetric memory tensor and return (tensor, symm_mem handle)."""
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            tensor = symm_mem.empty(*shape, dtype=torch.int8, device=device)
            tensor = tensor.view(dtype)
        else:
            tensor = symm_mem.empty(*shape, dtype=dtype, device=device)
        hdl = symm_mem.rendezvous(tensor, group=group_name)
        return tensor, hdl

    @staticmethod
    def _build_peer_ptrs(hdl, sizes, storage_dtype, device, world_size):
        """Build int64 tensor of peer buffer device addresses.

        ``hdl.get_buffer(peer_r, ...)`` returns a local tensor that references
        peer rank *peer_r*'s symmetric buffer via NVLink.  Its ``data_ptr()``
        is the device address the kernel stores to.
        """
        ptrs = torch.tensor(
            [hdl.get_buffer(r, sizes, storage_dtype).data_ptr()
             for r in range(world_size)],
            dtype=torch.int64, device=device,
        )
        return ptrs

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
    return BlockDispatchContext(max_m, hidden, num_experts, world_size, rank, num_scale_groups, group_name)


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
# Fused Triton Per-Token-Group FP8 Quantization + Gather
# ============================================================================

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

    # Load gather indices for this block's rows (shared across all its groups)
    gather_idx = tl.load(GatherIdx + offs_m, mask=row_mask, other=0)

    g_base = pid_g * BLOCK_G
    for gi in tl.static_range(0, BLOCK_G):
        g = g_base + gi
        if g < NUM_GROUPS:
            offs_k = g * GROUP_K + tl.arange(0, GROUP_K)

            # Gather: load source rows via gather_idx
            x_ptrs = X + gather_idx[:, None] * X_stride_m + offs_k[None, :]
            x_bf16 = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
            x_fp32 = x_bf16.to(tl.float32)

            x_absmax = tl.max(tl.abs(x_fp32), axis=1)
            x_amax_safe = tl.maximum(x_absmax, 1e-4)
            x_scale = x_amax_safe / FP8_MAX
            x_inv_s = 1.0 / x_scale
            x_fp8 = (x_fp32 * x_inv_s[:, None]).to(OutFP8.dtype.element_ty)

            # Store fp8 data + scales directly into output buffer
            out_ptrs = OutFP8 + offs_m[:, None] * Out_stride_m + offs_k[None, :]
            tl.store(out_ptrs, x_fp8, mask=row_mask[:, None])

            scale_ptrs = OutScales + offs_m * Scale_stride_m + g
            tl.store(scale_ptrs, x_scale, mask=row_mask)


def quant_gather(
    ctx: BlockDispatchContext,
    x: torch.Tensor,             # bf16 [M, K] — source input, must be contiguous
    gather_idx: torch.Tensor,    # int32 [EM] — output row → source row mapping
    group_k: int = 128,
    BLOCK_M: int = 64,
):
    """Fused per-token-group FP8 quantization + gather, writing directly into pre-allocated buffers.

    Combines quant_gather and the send_buf copy into a single kernel:
    reads x[gather_idx], quantizes to FP8, and stores the result directly into
    ctx.send_buf[:EM] and ctx.send_scales[:EM], avoiding an extra copy.

    Args:
        ctx: BlockDispatchContext providing send_buf, send_scales, num_sms.
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
    # Per-token-group quantization is independent across K groups, so we split
    # K across CTAs (grid_g) to increase parallelism when the row count is small,
    # and collapse to grid_g=1 (equivalent to 1-D) when rows already saturate SMs.
    num_sms = ctx.num_sms
    target_ctas = num_sms * 2  # ~2 resident CTAs/SM at 50% theoretical occupancy
    if row_blocks >= target_ctas:
        BLOCK_G = num_groups                     # 1-D, no K split
    else:
        grid_g = min(num_groups, max(1, cdiv(target_ctas, row_blocks)))
        BLOCK_G = cdiv(num_groups, grid_g)       # spread K to reach target
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
# Block-Level Dispatch via Intra-Node NVLink Load/Store
# ============================================================================
#
# Block-level FP8 dispatch using direct tl.store to peer symmetric memory
# over NVLink — replaces NVSHMEM putmem_signal_block with lower-latency
# SM-driven load/store.  Cross-rank sync via recv_hdl.barrier() (CUDA
# symm_mem backend — real GPU-side signal-pad barrier).
#
# Synchronization: all CTAs push data to peers, then host-side
# recv_hdl.barrier() ensures all ranks' stores are visible before GEMM.
#
# Differences from the fused warp-level dispatch above:
#   - Block-level tl.store to peer (not per-token warp-level putmem)
#   - Double-buffered recv layout: [WS * MAX_M * 2, K] (per-rank, not per-expert)
#   - No postprocess inside kernel — host-side slicing + concat like tutorial
#   - Simpler: no warp-level extern wrapper


@triton.jit
def _dispatch_block_push_kernel(
    # Local send buffers (read-only in this kernel)
    data_src,            # local fp8 [max_m, K] — send buffer (pre-filled by host)
    scale_src,           # local fp32 [max_m, NSG] — scale send buffer
    splits_cumsum,       # int32 [NUM_EXPERTS + 1] — cumsum provided by caller
    # Peer pointer arrays — device addresses of each peer rank's buffer
    peer_data_ptrs,      # int64 [WS] — peer recv data buffer pointers (full double-buffer)
    peer_scale_ptrs,     # int64 [WS] — peer recv scale buffer pointers
    peer_split_ptrs,     # int64 [WS] — peer recv split buffer pointers
    # Scalar params
    act_pos,             # 0 or 1 — double-buffer slot
    rank: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    # Constexpr sizes
    K: tl.constexpr,
    NSG: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    EPR_POW2: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    # Constexpr block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_NSG: tl.constexpr,
    NP: tl.constexpr,
):
    """Intra-node NVLink dispatch via direct load/store to peer symmetric memory.

    Replaces ``nvshmem.putmem_signal_block`` with direct ``tl.store`` to peer
    buffers over NVLink.  Cross-rank synchronization is handled by host-side
    ``recv_hdl.barrier()`` — the kernel itself contains no barriers, each CTA
    independently stores and exits.

    Each CTA: ``target_rank = pid // NP``, ``sub_id = pid % NP``.
    NP sub-blocks per target rank split the row range evenly, giving
    ``grid = (WS * NP,)`` to utilise more SMs beyond just one CTA per rank.

    Each sub-block stores its fraction of FP8 data + scales to peer.
    sub-block 0 also writes per-expert splits (rank-level, not row-level).
    """
    pid = tl.program_id(0)
    target_rank = pid // NP
    sub_id = pid % NP

    # Source segment in send_buf: tokens for target_rank's experts
    exp_st = target_rank * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK
    m_st_full = tl.load(splits_cumsum + exp_st)
    m_ed_full = tl.load(splits_cumsum + exp_ed)
    total_rows = m_ed_full - m_st_full

    # Even split of rows among NP sub-blocks
    sub_st = (total_rows * sub_id) // NP
    sub_ed = (total_rows * (sub_id + 1)) // NP
    num_rows = sub_ed - sub_st
    m_st = m_st_full + sub_st

    # Destination offset in peer's recv buffer: act_pos half + our slot
    buf_base = act_pos * WORLD_SIZE * MAX_M + rank * MAX_M
    split_base = act_pos * NUM_EXPERTS + rank * EXPERTS_PER_RANK

    # Load peer buffer pointers for target rank, cast to typed pointers
    peer_data_ptr = tl.load(peer_data_ptrs + target_rank).to(tl.pointer_type(tl.float8e4nv))
    peer_scale_ptr = tl.load(peer_scale_ptrs + target_rank).to(tl.pointer_type(tl.float32))
    peer_split_ptr = tl.load(peer_split_ptrs + target_rank).to(tl.pointer_type(tl.int32))

    # ---- 1. Store per-expert splits (only sub-block 0) ----
    if sub_id == 0:
        epr_offs = tl.arange(0, EPR_POW2)
        epr_mask = epr_offs < EXPERTS_PER_RANK
        off0 = exp_st + epr_offs
        off1 = off0 + 1
        cumsum_sts = tl.load(splits_cumsum + off0, mask=epr_mask)
        cumsum_eds = tl.load(splits_cumsum + off1, mask=epr_mask)
        split_vals = cumsum_eds - cumsum_sts
        tl.store(peer_split_ptr + split_base + epr_offs, split_vals, mask=epr_mask)

    # ---- 2. Copy FP8 data: local → peer via NVLink load/store ----
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    k_mask_base = offs_k < K

    dst_st = buf_base + sub_st
    src_data_base = data_src + m_st.to(tl.int64) * K
    dst_data_base = peer_data_ptr + dst_st.to(tl.int64) * K

    for i in range(tl.cdiv(num_rows, BLOCK_M)):
        row_mask = offs_m + i * BLOCK_M < num_rows
        for k_blk in range(tl.cdiv(K, BLOCK_K)):
            k_offs = k_blk * BLOCK_K + offs_k
            km = k_mask_base & (k_offs < K)
            mask = row_mask[:, None] & km[None, :]
            data = tl.load(
                src_data_base + (i * BLOCK_M + offs_m[:, None]) * K + k_offs[None, :],
                mask=mask, other=0.0,
            )
            tl.store(
                dst_data_base + (i * BLOCK_M + offs_m[:, None]) * K + k_offs[None, :],
                data, mask=mask,
            )

    # ---- 3. Copy scales: local → peer via NVLink load/store ----
    offs_s = tl.arange(0, BLOCK_NSG)
    s_mask = offs_s < NSG

    src_scale_base = scale_src + m_st.to(tl.int64) * NSG
    dst_scale_base = peer_scale_ptr + dst_st.to(tl.int64) * NSG

    for i in range(tl.cdiv(num_rows, BLOCK_M)):
        row_mask = offs_m + i * BLOCK_M < num_rows
        mask = row_mask[:, None] & s_mask[None, :]
        scale = tl.load(
            src_scale_base + (i * BLOCK_M + offs_m[:, None]) * NSG + offs_s[None, :],
            mask=mask, other=0.0,
        )
        tl.store(
            dst_scale_base + (i * BLOCK_M + offs_m[:, None]) * NSG + offs_s[None, :],
            scale, mask=mask,
        )


def dispatch_block(
    ctx: BlockDispatchContext,
    num_tokens: int,                    # actual number of gathered tokens in send_buf
    send_split_cumsum: torch.Tensor,   # int32 [num_experts + 1]
    num_sub_blocks: int = None,        # NP: sub-blocks per rank (None=auto)
):
    """Block-level FP8 dispatch via intra-node NVLink load/store.

    Assumes ``ctx.send_buf`` and ``ctx.send_scales`` have already been filled
    (e.g. by ``quant_gather``) with the FP8 data for ``num_tokens`` rows.

    One kernel, ``NP`` CTAs per target rank (``grid = WS * NP``).  Each CTA
    handles 1/NP of the row range for its target rank.  Split values (per-expert
    counts) are written only by sub-block 0 of each rank group.

    Synchronization: host-side ``recv_hdl.barrier()`` after kernel launch,
    ensuring all ranks' NVLink stores are visible before GEMM reads them.

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

    if num_sub_blocks is not None:
        NP = num_sub_blocks
    else:
        # Auto: aim for ~16-32 total CTAs, capped at 8 sub-blocks per rank
        NP = max(1, min(8, ctx.num_sms // (WS * 4)))

    act_pos = ctx.call_count % 2

    grid = (WS * NP,)
    _dispatch_block_push_kernel[grid](
        # Local send buffers
        ctx.send_buf,
        ctx.send_scales,
        send_split_cumsum,
        # Peer pointer arrays
        ctx.peer_data_ptrs,
        ctx.peer_scale_ptrs,
        ctx.peer_split_ptrs,
        # Scalars
        act_pos,
        ctx.rank,
        WS,
        # Constexpr sizes
        K=K,
        NSG=NSG,
        MAX_M=MAX_M,
        EXPERTS_PER_RANK=EXPERTS_PER_RANK,
        EPR_POW2=triton.next_power_of_2(EXPERTS_PER_RANK),
        NUM_EXPERTS=NUM_EXPERTS,
        # Block sizes
        BLOCK_M=16,
        BLOCK_K=min(triton.next_power_of_2(K), 1024),
        BLOCK_NSG=triton.next_power_of_2(NSG),
        NP=NP,
        num_warps=8,
    )

    ctx.recv_hdl.barrier()

    ctx.call_count = (ctx.call_count + 1) % ctx.MOD_VALUE

    data_st = act_pos * WS * MAX_M
    data_ed = data_st + WS * MAX_M
    split_st = act_pos * NUM_EXPERTS
    split_ed = split_st + NUM_EXPERTS

    return (
        ctx.recv_buf[data_st:data_ed],
        ctx.recv_scales[data_st:data_ed],
        ctx.split_recv[split_st:split_ed],
    )


# ============================================================================
# 2D Group GEMM — for block-level dispatch output [WS*MAX_M, K]
# ============================================================================
#
# Block dispatch recv layout is [WS*MAX_M, K] — per-rank, within each rank
# sorted by expert. This is a 2D layout, NOT the 3D [EXPERTS_PER_RANK, WS*MAX_M, K]
# expert-contiguous layout used by the fused dispatch.
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
    max_total_blocks = cdiv(WS_MAX_M, BLOCK_M) + EXPERTS_PER_RANK
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