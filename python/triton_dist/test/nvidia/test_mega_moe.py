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
Test MegaMoE kernels — fused dispatch + FP8 group GEMM correctness & performance

Usage:
    python test_mega_moe.py -M 32 -N 6144 --perf
    python test_mega_moe.py -M 32 -N 6144 --perf --profile
"""

import os
import random
import argparse
import time
import sys
import importlib.util

# Must be set before any CUDA context is created — required for symm_mem GPU-side barriers
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

import torch
import torch.distributed as dist

# Directly import profiler_utils module file, bypass triton_dist package dependency
_profiler_utils_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "profiler_utils.py"
)
_profiler_utils_spec = importlib.util.spec_from_file_location(
    "profiler_utils", _profiler_utils_path
)
_profiler_utils_module = importlib.util.module_from_spec(_profiler_utils_spec)
sys.modules["profiler_utils"] = _profiler_utils_module
_profiler_utils_spec.loader.exec_module(_profiler_utils_module)
group_profile = _profiler_utils_module.group_profile

# deep_gemm quantization utilities
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

# Directly import mega_moe module file, bypass triton_dist.__init__.py dependency issues
_mega_moe_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "kernels", "nvidia", "mega_moe.py"
)
_mega_moe_spec = importlib.util.spec_from_file_location("mega_moe", _mega_moe_path)
_mega_moe_module = importlib.util.module_from_spec(_mega_moe_spec)
_mega_moe_spec.loader.exec_module(_mega_moe_module)

# Export needed symbols from mega_moe module
FP8_DTYPE = _mega_moe_module.FP8_DTYPE
FP8_MAX = _mega_moe_module.FP8_MAX
cdiv = _mega_moe_module.cdiv
splits_to_cumsum = _mega_moe_module.splits_to_cumsum
quant_local = _mega_moe_module.quant_local
block_quantize_weight_fp8 = _mega_moe_module.block_quantize_weight_fp8
create_block_dispatch_context = _mega_moe_module.create_block_dispatch_context
fp8_group_gemm_2d = _mega_moe_module.fp8_group_gemm_2d
_pick_block_m = _mega_moe_module._pick_block_m
_should_enable_swap_ab = _mega_moe_module._should_enable_swap_ab


def initialize_distributed(local_rank: int, world_size: int):
    """Initialize distributed environment via TCP rendezvous (no torchrun needed).

    Uses MASTER_ADDR / MASTER_PORT env vars (defaults: 127.0.0.1:8361).
    """
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "8361")

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=local_rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    return local_rank, local_rank, world_size


# ──────────────────────────── helpers ────────────────────────────

OUTPUT_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def generate_round_robin_exp_indices(token_num: int, total_num_experts: int, topk: int,
                                seed: int = 42, world_size: int = 8, rank: int = 0):
    """Uniform expert indices: every WS consecutive entries span all ranks,
    so per-peer send counts are even and every expert gets equal load."""
    experts_per_rank = total_num_experts // world_size
    total_assignments = token_num * topk

    # Interleave: [r0_e0, r1_e0, ..., rW_e0, r0_e1, ...]
    local_experts = torch.arange(experts_per_rank)
    rank_offsets = torch.arange(world_size) * experts_per_rank
    interleaved = (local_experts[:, None] + rank_offsets[None, :]).flatten()

    # Rotate per rank so different ranks cover different experts
    interleaved = interleaved.roll(rank * experts_per_rank)

    num_tiles = (total_assignments + total_num_experts - 1) // total_num_experts
    flat = interleaved.repeat(num_tiles)[:total_assignments]

    return flat.view(token_num, topk).to(torch.int32)


def calc_scatter_index_stable(chosen_experts: torch.Tensor):
    return chosen_experts.flatten().argsort(stable=True).argsort().int().view(chosen_experts.shape)


def assert_allclose(actual: torch.Tensor,
                    expected: torch.Tensor,
                    atol: float = 1e-2,
                    rtol: float = 1e-2,
                    name: str = ""):
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: Shape mismatch: {actual.shape} vs {expected.shape}")
    diff = (actual - expected).abs()
    max_diff = diff.max().item()
    if max_diff > atol + rtol * expected.abs().max().item():
        raise AssertionError(
            f"{name}: Max diff {max_diff} exceeds tolerance (atol={atol}, rtol={rtol})")
    return True


def perf_func(func, warmup_iters=10, iters=100, *args, **kwargs):
    """Performance measurement"""
    torch.distributed.barrier()
    # Warmup
    for _ in range(warmup_iters):
        output = func(*args, **kwargs)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        output = func(*args, **kwargs)

    torch.cuda.synchronize()
    end = time.perf_counter()

    duration_ms = (end - start) / iters * 1000
    return output, duration_ms


def fp8_group_gemm_deepgemm(
    input: torch.Tensor,         # fp8 [experts_per_rank, m_max, K]
    weight: torch.Tensor,           # fp8 [E, N, K]
    a_scales: torch.Tensor,      # fp32 [experts_per_rank, m_max, K // group_k]
    b_scales: torch.Tensor,         # fp32 [E, N // group_n, K // group_k]
    expert_recv_count: torch.Tensor,  # int32 [experts_per_rank]
    expected_m: int,
    group_n: int,
    group_k: int,
    output_dtype: torch.dtype,
    BM: int,
):
    """DeepGEMM reference — uses [experts_per_rank, m_max, K] layout.

    Follows SGLang's pattern: expected_m = ceil(total_tokens / experts_per_rank),
    using the average per-expert count rather than the max.
    """
    experts_per_rank, m_max, K = input.shape
    E, N, _K = weight.shape

    d = torch.empty((experts_per_rank, m_max, N), dtype=output_dtype, device=input.device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
        (input, a_scales),
        (weight, b_scales),
        d,
        expert_recv_count,
        expected_m,
        disable_ue8m0_cast=True,
    )
    return d


# ─────────────────────── shared-activation tests ────────────────────────


def test_moe_dispatch_block(
    ctx,
    rank: int,
    world_size: int,
    num_tokens: int = 128,
    hidden: int = 7168,
    intermediate: int = 2048,
    num_experts: int = 384,
    topk: int = 6,
    BM: int = None,
    group_n: int = 128,
    group_k: int = 128,
):
    """Test shared-activation dispatch correctness (no cross-rank data move).

    Each rank quantizes its `num_tokens` tokens in original order into the
    local segment of the shared NVSHMEM buffer, then writes per-token topk_ids /
    topk_weights into the same segment. After host-side barrier, every rank
    accesses peer data via pointer indirection (buffer_ptrs) and runs the GEMM,
    which scans per-rank topk_ids to gather rows for each local expert and
    atomically accumulates `A[row] @ W[local_e]` into the output row(s).
    """
    if rank == 0:
        print(f"[shared-activation] M={num_tokens}, K={hidden}, "
              f"total_E={num_experts}, topk={topk}, group_k={group_k}, "
              f"world_size={world_size}")

    assert num_experts % world_size == 0
    experts_per_rank = num_experts // world_size
    max_m = ctx.max_m
    assert max_m >= num_tokens

    torch.manual_seed(rank)
    x_bf16 = torch.randn([num_tokens, hidden], dtype=torch.bfloat16, device="cuda")

    exp_indices = generate_round_robin_exp_indices(
        num_tokens, num_experts, topk, world_size=world_size, rank=rank).cuda()
    topology_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, dtype=torch.float32, device="cuda")

    # ── 1. Quant + copy topk metadata into VMM local view ─────────────
    quant_local(
        ctx, x_bf16,
        exp_indices.to(torch.int32), topology_weights,
        num_tokens=num_tokens,
        group_k=group_k, BLOCK_M=BM or 64)

    # ── 2. Cross-rank sync ─────────────────────────────────────────────
    torch.distributed.barrier()
    ctx.barrier()

    # ── 3. Get VMM 3D views ────────────────────────────────────
    vmm_fp8, vmm_scales, vmm_ids, vmm_weights, a_stride_ws, as_stride_ws = ctx.get_vmm_views()
    topk_stride_ws = ctx.vmm_ids.stride_ws

    # ── 4. Run GEMM with VMM 3D access ──────────────────────────
    weight_bf16 = torch.randn(
        [experts_per_rank, intermediate, hidden],
        dtype=torch.bfloat16, device="cuda")
    weight_fp8, weight_scales = block_quantize_weight_fp8(
        weight_bf16, group_n=group_n, group_k=group_k)

    num_scale_groups = hidden // group_k
    gemm_out = fp8_group_gemm_2d(
        vmm_fp8, weight_fp8, vmm_scales, weight_scales,
        vmm_ids,
        local_expert_start=rank * experts_per_rank,
        num_tokens_per_rank=num_tokens,
        TOPK=topk,
        MAX_M=max_m,
        A_STRIDE_WS=a_stride_ws,
        A_S_STRIDE_WS=as_stride_ws,
        TOPK_STRIDE_WS=topk_stride_ws,
        EXPERTS_PER_RANK=experts_per_rank,
        WS=world_size,
        hidden=hidden,
        num_scale_groups=num_scale_groups,
        group_n=group_n, group_k=group_k,
        BLOCK_M=BM, output_dtype=torch.bfloat16,
    )

    # ── 5. Reference: per-expert matmul -> scatter-add to expected ─────
    # SGLang pattern: m_max based on actual total assignments, aligned to 256 for TMA
    total_assignments = num_tokens * topk
    m_max = ((total_assignments + 255) // 256) * 256
    ref_input = torch.zeros(
        experts_per_rank, m_max, hidden,
        dtype=FP8_DTYPE, device="cuda")
    ref_scales = torch.zeros(
        experts_per_rank, m_max, hidden // group_k,
        dtype=torch.float32, device="cuda")
    expert_recv_count = torch.zeros(
        experts_per_rank, dtype=torch.int32, device="cuda")
    local_exp_start = rank * experts_per_rank

    # Build per-expert gathered input for DeepGEMM reference
    for e_local in range(experts_per_rank):
        target_e = local_exp_start + e_local
        dst_offset = 0
        for r in range(world_size):
            r_ids = ctx.get_peer_buffer(r, 'ids', num_tokens)
            r_fp8 = ctx.get_peer_buffer(r, 'fp8', num_tokens)
            r_scales = ctx.get_peer_buffer(r, 'scales', num_tokens)
            hit_mask = (r_ids == target_e).any(dim=1)
            hit_indices = hit_mask.nonzero(as_tuple=False).squeeze(1)
            count = hit_indices.shape[0]
            if count == 0:
                continue
            ref_input[e_local, dst_offset:dst_offset + count] = \
                r_fp8[hit_indices]
            ref_scales[e_local, dst_offset:dst_offset + count] = \
                r_scales[hit_indices]
            dst_offset += count
        expert_recv_count[e_local] = dst_offset

    total = expert_recv_count.sum().item()
    expected_m = (total - 1) // experts_per_rank + 1
    expected_m = ((expected_m + (BM or 64) - 1) // (BM or 64)) * (BM or 64)

    ref_out = fp8_group_gemm_deepgemm(
        ref_input, weight_fp8, ref_scales, weight_scales,
        expert_recv_count, expected_m,
        group_n=group_n, group_k=group_k,
        output_dtype=torch.bfloat16, BM=BM or 64,
    )
    # ref_out: [EPR, m_max, N] — packed layout (tokens contiguous per expert)

    # ── 6. Compare: per-expert, mapped scattered→packed ────────────────
    # gemm_out: [EPR, WS*MAX_M, N] — SCATTERED: each token at its global row
    #   (r*MAX_M + token_idx).
    # ref_out:  [EPR, m_max, N] — PACKED: tokens contiguous per expert,
    #   ref_out[e, 0:count, :] in order rank0's hits, rank1's hits, ...
    # Must map: for each (expert, rank, hit_token): compare
    #   gemm_out[e, global_row, :] vs ref_out[e, packed_offset, :]
    max_diff = 0.0
    max_val = 0.0
    max_diff_info = None
    nonzero_diff = 0
    total_elements = 0

    for e_local in range(experts_per_rank):
        target_e = local_exp_start + e_local
        count = expert_recv_count[e_local].item()
        if count == 0:
            continue
        ref_offset = 0
        for r in range(world_size):
            r_ids = ctx.get_peer_buffer(r, 'ids', num_tokens)
            hit_mask = (r_ids == target_e).any(dim=1)
            hit_indices = hit_mask.nonzero(as_tuple=False).squeeze(1)
            count_r = hit_indices.shape[0]
            if count_r == 0:
                continue
            for j in range(count_r):
                global_row = r * max_m + hit_indices[j].item()
                g_row = gemm_out[e_local, global_row].to(torch.float32)
                ref_row = ref_out[e_local, ref_offset + j].to(torch.float32)
                diff_row = (g_row - ref_row).abs()
                d = diff_row.max().item()
                if d > max_diff:
                    max_diff = d
                    max_diff_info = (e_local, r, hit_indices[j].item(), global_row,
                                     ref_offset + j, diff_row.argmax().item())
                max_val = max(max_val, g_row.abs().max().item(),
                              ref_row.abs().max().item())
                nonzero_diff += (diff_row > 0.001).sum().item()
            ref_offset += count_r
        total_elements += count * ref_out.shape[2]

    gemm_atol, gemm_rtol = 0.5, 0.1
    threshold = gemm_atol + gemm_rtol * max_val

    if rank == 0:
        print(f"  GEMM output max_diff: {max_diff:.4f}")

    if max_diff < threshold:
        if rank == 0:
            print(f"  [PASS] shared-activation GEMM test passed!")
    else:
        if rank == 0:
            print(f"  [FAIL] shared-activation GEMM FAILED "
                  f"max_diff={max_diff:.4f}")
        raise AssertionError(
            f"gemm_diff={max_diff:.4f} threshold={threshold:.4f}")

    return ctx


def test_moe_dispatch_block_performance(
    ctx,
    rank: int,
    world_size: int,
    num_tokens: int = 128,
    hidden: int = 7168,
    intermediate: int = 2048,
    num_experts: int = 384,
    topk: int = 6,
    BM: int = None,
    group_n: int = 128,
    group_k: int = 128,
    warmup: int = 10,
    iters: int = 100,
):
    """Benchmark: stage-wise breakdown of shared-activation MoE pipeline.

    Stage 0 — quant_local    : FP8 quant + topk_ids/weights copy into VMM local view.
    Stage 1 — barrier       : cross-rank host-side symm_mem barrier.
    Stage 2 — fp8_groupgemm : read-side filter + 2D group GEMM (atomic-add).
    Stage 3 — DeepGEMM ref  : grouped-FP8 reference for throughput comparison.
    Full pipeline           : quant_local + barrier + GEMM, end-to-end.
    """
    if rank == 0:
        print(f"[shared-activation perf] rank={rank}/{world_size}, M={num_tokens}, "
              f"K={hidden}, N={intermediate}, total_E={num_experts}, topk={topk}, "
              f"BM={BM if BM is not None else 'auto'}, group_n={group_n}, group_k={group_k}")

    assert num_experts % world_size == 0
    experts_per_rank = num_experts // world_size
    max_m = ctx.max_m
    WS_MAX_M = world_size * max_m

    torch.manual_seed(rank)
    x_bf16 = torch.randn(
        [num_tokens, hidden], dtype=torch.bfloat16, device="cuda")
    exp_indices = generate_round_robin_exp_indices(
        num_tokens, num_experts, topk, world_size=world_size, rank=rank).cuda()
    topology_weights = torch.full(
        (num_tokens, topk), 1.0 / topk, dtype=torch.float32, device="cuda")
    weight_bf16 = torch.randn(
        [experts_per_rank, intermediate, hidden],
        dtype=torch.bfloat16, device="cuda")
    weight_fp8, weight_scales = block_quantize_weight_fp8(
        weight_bf16, group_n=group_n, group_k=group_k)

    vmm_fp8, vmm_scales, vmm_ids, vmm_weights, a_stride_ws, as_stride_ws = ctx.get_vmm_views()
    topk_stride_ws = ctx.vmm_ids.stride_ws

    exp_int32 = exp_indices.to(torch.int32)

    # One-time setup: quant + copy metadata into VMM, barrier, then
    # build DeepGEMM reference layout once outside the timing loops.
    quant_local(
        ctx, x_bf16, exp_int32, topology_weights,
        num_tokens=num_tokens,
        group_k=group_k, BLOCK_M=BM or 64)

    torch.distributed.barrier()
    ctx.barrier()

    num_scale_groups = hidden // group_k
    # SGLang pattern: m_max based on actual total assignments, aligned to 256 for TMA
    total_assignments = num_tokens * topk
    m_max = ((total_assignments + 255) // 256) * 256
    ref_input = torch.zeros(
        experts_per_rank, m_max, hidden,
        dtype=FP8_DTYPE, device="cuda")
    ref_scales = torch.zeros(
        experts_per_rank, m_max, hidden // group_k,
        dtype=torch.float32, device="cuda")
    expert_recv_count = torch.zeros(
        experts_per_rank, dtype=torch.int32, device="cuda")
    local_exp_start = rank * experts_per_rank
    for e_local in range(experts_per_rank):
        target_e = local_exp_start + e_local
        dst_offset = 0
        for r in range(world_size):
            r_ids = ctx.get_peer_buffer(r, 'ids', num_tokens)
            r_fp8 = ctx.get_peer_buffer(r, 'fp8', num_tokens)
            r_scales = ctx.get_peer_buffer(r, 'scales', num_tokens)
            hit_mask = (r_ids == target_e).any(dim=1)
            hit_indices = hit_mask.nonzero(as_tuple=False).squeeze(1)
            count = hit_indices.shape[0]
            if count == 0:
                continue
            ref_input[e_local, dst_offset:dst_offset + count] = \
                r_fp8[hit_indices]
            ref_scales[e_local, dst_offset:dst_offset + count] = \
                r_scales[hit_indices]
            dst_offset += count
        expert_recv_count[e_local] = dst_offset

    # ── Stage 0: quant_local (quant + topk copy into VMM) ─────────────
    def stage_quant():
        quant_local(
            ctx, x_bf16, exp_int32, topology_weights,
            num_tokens=num_tokens,
            group_k=group_k, BLOCK_M=BM or 64)

    _, dur_quant = perf_func(stage_quant, warmup, iters)

    # ── Stage 1: cross-rank barrier (host-side symm_mem collective) ────
    def stage_barrier():
        ctx.barrier()

    _, dur_barrier = perf_func(
        stage_barrier, warmup, iters)

    # ── Stage 2: 2D group GEMM with VMM contiguous access ───────────
    def stage_gemm_2d():
        return fp8_group_gemm_2d(
            vmm_fp8, weight_fp8, vmm_scales, weight_scales,
            vmm_ids,
            local_expert_start=rank * experts_per_rank,
            num_tokens_per_rank=num_tokens,
            TOPK=topk,
            MAX_M=max_m,
            A_STRIDE_WS=a_stride_ws,
            A_S_STRIDE_WS=as_stride_ws,
            TOPK_STRIDE_WS=topk_stride_ws,
            EXPERTS_PER_RANK=experts_per_rank,
            WS=world_size,
            hidden=hidden,
            num_scale_groups=num_scale_groups,
            group_n=group_n, group_k=group_k,
            BLOCK_M=BM, output_dtype=torch.bfloat16,
        )

    _, dur_gemm_2d = perf_func(stage_gemm_2d, warmup, iters)

    # ── Stage 3: DeepGEMM reference ────────────────────────────────────
    total = expert_recv_count.sum().item()
    expected_m = (total - 1) // experts_per_rank + 1
    expected_m = ((expected_m + (BM or 64) - 1) // (BM or 64)) * (BM or 64)
    def stage_gemm_deepgemm():
        return fp8_group_gemm_deepgemm(
            ref_input, weight_fp8, ref_scales, weight_scales,
            expert_recv_count, expected_m,
            group_n=group_n, group_k=group_k,
            output_dtype=torch.bfloat16, BM=BM or 64,
        )

    _, dur_deepgemm = perf_func(stage_gemm_deepgemm, warmup, iters)

    # ── Full pipeline: pack + barrier + GEMM, end-to-end ───────────────
    def full_pipeline():
        quant_local(
            ctx, x_bf16, exp_int32, topology_weights,
            num_tokens=num_tokens,
            group_k=group_k, BLOCK_M=BM or 64)
        ctx.barrier()
        return fp8_group_gemm_2d(
            vmm_fp8, weight_fp8, vmm_scales, weight_scales,
            vmm_ids,
            local_expert_start=rank * experts_per_rank,
            num_tokens_per_rank=num_tokens,
            TOPK=topk,
            MAX_M=max_m,
            A_STRIDE_WS=a_stride_ws,
            A_S_STRIDE_WS=as_stride_ws,
            TOPK_STRIDE_WS=topk_stride_ws,
            EXPERTS_PER_RANK=experts_per_rank,
            WS=world_size,
            hidden=hidden,
            num_scale_groups=num_scale_groups,
            group_n=group_n, group_k=group_k,
            BLOCK_M=BM, output_dtype=torch.bfloat16,
        )

    _, dur_full = perf_func(
        full_pipeline, warmup, iters)

    # ── Reporting ──────────────────────────────────────────────────────
    # Aggregate across all ranks for the GFLOPS accounting.
    total_tokens = num_tokens * topk
    data_bytes = total_tokens * hidden
    scale_bytes = total_tokens * (hidden // group_k) * 4
    total_bytes = data_bytes + scale_bytes
    flops = total_tokens * 2 * hidden * intermediate
    sum_stages = dur_quant + dur_barrier + dur_gemm_2d

    if rank == 0:
        gemm_bm = BM if BM is not None else \
            _pick_block_m(num_tokens * topk / experts_per_rank)
        gemm_swap = _should_enable_swap_ab(gemm_bm, 128)
        swap_tag = "(swapab)" if gemm_swap else "(no swapab)"
        print(f"  Stage timings (avg over {iters} iters):")
        print(f"    quant_local:          {dur_quant:.3f} ms")
        print(f"    barrier (cross-rank): {dur_barrier:.3f} ms")
        print(f"    fp8_group_gemm_2d:    {dur_gemm_2d:.3f} ms {swap_tag}")
        print(f"    DeepGEMM reference:   {dur_deepgemm:.3f} ms")
        print(f"    ───────────────────────────────────────")
        print(f"    Sum of stages:        {sum_stages:.3f} ms")
        print(f"    Full pipeline (e2e):  {dur_full:.3f} ms")
        print(f"    data volume:          {total_bytes / 1e6:.2f} MB")
        if dur_gemm_2d > 0:
            print(f"    GEMM 2D throughput:   {flops / dur_gemm_2d / 1e9:.2f} GFLOPS")
        if dur_deepgemm > 0:
            print(f"    DeepGEMM throughput:  {flops / dur_deepgemm / 1e9:.2f} GFLOPS")
            if dur_gemm_2d > 0:
                print(f"    GEMM speedup vs DeepGEMM: {dur_deepgemm / dur_gemm_2d:.2f}x")
        if dur_full > 0:
            print(f"    Pipeline throughput:  {flops / dur_full / 1e9:.2f} GFLOPS")


# ──────────────────────────── main ────────────────────────────

def worker(local_rank: int, num_gpus: int, args: argparse.Namespace):
    """Per-GPU worker — spawned by torch.multiprocessing.spawn."""
    # Restore fork start method so that multiprocessing.Pool in profiler_utils
    # (trace merge) inherits deep_gemm instead of re-importing it via spawn.
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    rank, local_rank, world_size = initialize_distributed(local_rank, num_gpus)

    # Per-rank capacity = number of tokens (each rank writes its `args.M`
    # tokens into its own segment of the shared VMM buffer in original order).
    num_scale_groups = args.K // args.group_k
    max_m = args.M

    # ── Shared-activation dispatch tests ──
    if rank == 0:
        print("=" * 60)
        print("Shared-Activation Dispatch Test")
        print("=" * 60)

    block_ctx = create_block_dispatch_context(
        max_m, args.K, args.experts, world_size, rank,
        num_scale_groups, topk=args.topk)

    test_moe_dispatch_block(
        block_ctx,
        rank=rank,
        world_size=world_size,
        num_tokens=args.M,
        hidden=args.K,
        intermediate=args.N,
        num_experts=args.experts,
        topk=args.topk,
        BM=args.BM,
        group_k=args.group_k,
    )

    if args.perf:
        if rank == 0:
            print()

        profile_name = f"mega_moe_load_{args.M}x{args.K}x{args.N}_e{args.experts}_topk{args.topk}"
        with group_profile(profile_name, args.profile):
            test_moe_dispatch_block_performance(
                block_ctx,
                rank=rank,
                world_size=world_size,
                num_tokens=args.M,
                hidden=args.K,
                intermediate=args.N,
                num_experts=args.experts,
                topk=args.topk,
                BM=args.BM,
                group_n=args.group_n,
                group_k=args.group_k,
                warmup=args.warmup,
                iters=args.iters,
            )

    if rank == 0:
        if args.profile:
            from pathlib import Path
            trace_path = Path("prof") / f"{profile_name}_merged.json"
            print(f"\nProfiling trace saved to: {trace_path}")
        print("\nAll tests passed!")

    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Test MegaMoE FP8 Group GEMM")
    parser.add_argument("-M", type=int, default=128, help="Number of tokens")
    parser.add_argument("-K", type=int, default=7168, help="Hidden dimension (K)")
    parser.add_argument("-N", type=int, default=2048, help="Intermediate dimension (N)")
    parser.add_argument("--experts", type=int, default=384, help="Total number of experts")
    parser.add_argument("--topk", type=int, default=6, help="Top-K expert selection")
    parser.add_argument("--BM", type=int, default=None,
                        help="Block size M for group GEMM (default: auto from per-expert tokens)")
    parser.add_argument("--group-n", type=int, default=128, help="FP8 quantization block size for N")
    parser.add_argument("--group-k", type=int, default=128, help="FP8 quantization block size for K")
    parser.add_argument("--dtype", default="bfloat16", help="Output data type", choices=list(OUTPUT_DTYPE_MAP.keys()))
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Bench iterations")
    parser.add_argument("--perf", action="store_true", help="Run performance benchmark")
    parser.add_argument("--profile", default=False, action="store_true",
                        help="dump torch.profiler.profile")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    print(f"Launching {num_gpus} GPU worker(s)...")
    torch.multiprocessing.spawn(worker, args=(num_gpus, args), nprocs=num_gpus)


if __name__ == "__main__":
    main()