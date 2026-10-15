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
Test MegaMoE kernels — warp/block dispatch + FP8 group GEMM correctness & performance

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
per_token_quant_fp8 = _mega_moe_module.per_token_quant_fp8
quant_gather = _mega_moe_module.quant_gather
block_quantize_weight_fp8 = _mega_moe_module.block_quantize_weight_fp8
DispatchContext = _mega_moe_module.DispatchContext
create_dispatch_context = _mega_moe_module.create_dispatch_context
dispatch_warp = _mega_moe_module.dispatch_warp
fp8_group_gemm_3d = _mega_moe_module.fp8_group_gemm_3d
_NVSHMEM_AVAILABLE = _mega_moe_module._NVSHMEM_AVAILABLE
create_block_dispatch_context = _mega_moe_module.create_block_dispatch_context
dispatch_block = _mega_moe_module.dispatch_block
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


# ─────────────────────── dispatch tests ────────────────────────

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


def test_moe_dispatch_warp(
    ctx: DispatchContext,
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
    """Test warp dispatch (single kernel) + group GEMM.

    Pipeline: quant_gather → dispatch_warp (warp-level getmem pull) → fp8_group_gemm_3d
    """
    if rank == 0:
        print(f"[warp dispatch+gemm] M={num_tokens}, K={hidden}, "
              f"N={intermediate}, total_E={num_experts}, topk={topk}, "
              f"BM={BM if BM is not None else 'auto'}, group_n={group_n}, group_k={group_k}, world_size={world_size}")

    if not _NVSHMEM_AVAILABLE:
        if rank == 0:
            print("  [SKIP] NVSHMEM not available")
        return

    assert num_experts % world_size == 0
    local_experts = num_experts // world_size
    num_scale_groups = hidden // group_k
    max_m = num_tokens * topk

    torch.manual_seed(rank)
    x_bf16 = torch.randn([num_tokens, hidden], dtype=torch.bfloat16, device="cuda")

    # Generate routing (deterministic via local torch.Generator)
    exp_indices = generate_round_robin_exp_indices(
        num_tokens, num_experts, topk, world_size=world_size, rank=rank).cuda()

    # quant_gather (unchanged)
    global_exp = exp_indices.view(-1).to(torch.int32)
    send_order = global_exp.argsort(stable=True)
    gather_idx = (send_order // topk).to(torch.int32)
    quant_gather(
        ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)

    send_splits = torch.bincount(global_exp, minlength=num_experts).to(torch.int32)
    send_split_cumsum = splits_to_cumsum(send_splits)

    # warp dispatch
    out_token, out_scales, expert_recv_count = dispatch_warp(
        ctx, max_m, send_split_cumsum, num_input_tokens=num_tokens, BM=BM or 64)

    if rank == 0:
        print(f"  rank={rank}: dispatched {max_m} -> "
              f"{expert_recv_count.sum().item()} tokens")

    # ── Verify dispatch out_token: reconstruct expected from all ranks' send_buf ──
    send_fp8 = ctx.send_buf[:max_m]
    send_scales = ctx.send_scales[:max_m]
    all_fp8 = [torch.empty_like(send_fp8) for _ in range(world_size)]
    all_scales = [torch.empty_like(send_scales) for _ in range(world_size)]
    all_cumsums = [torch.empty_like(send_split_cumsum) for _ in range(world_size)]
    dist.all_gather(all_fp8, send_fp8)
    dist.all_gather(all_scales, send_scales)
    dist.all_gather(all_cumsums, send_split_cumsum)

    dispatch_data_err = 0.0
    dispatch_scale_err = 0.0
    for e in range(local_experts):
        g = rank * local_experts + e
        n = expert_recv_count[e].item()
        if n == 0:
            continue

        exp_data_parts, exp_scale_parts = [], []
        for r in range(world_size):
            st = all_cumsums[r][g].item()
            ed = all_cumsums[r][g + 1].item()
            if ed > st:
                exp_data_parts.append(all_fp8[r][st:ed])
                exp_scale_parts.append(all_scales[r][st:ed])

        if not exp_data_parts:
            continue

        exp_data = torch.cat(exp_data_parts, dim=0)
        exp_scales = torch.cat(exp_scale_parts, dim=0)

        data_diff = (out_token[e, :n].to(torch.float32) -
                     exp_data.to(torch.float32)).abs().max().item()
        scale_diff = (out_scales[e, :n].to(torch.float32) -
                      exp_scales.to(torch.float32)).abs().max().item()
        dispatch_data_err = max(dispatch_data_err, data_diff)
        dispatch_scale_err = max(dispatch_scale_err, scale_diff)

    if dispatch_data_err > 0 or dispatch_scale_err > 1e-6:
        if rank == 0:
            print(f"  [FAIL] dispatch output mismatch: "
                  f"data_err={dispatch_data_err:.6f}, scale_err={dispatch_scale_err:.6f}")
        raise AssertionError(
            f"dispatch output mismatch: data_err={dispatch_data_err:.6f}, "
            f"scale_err={dispatch_scale_err:.6f}")

    # GEMM
    weight_bf16 = torch.randn([local_experts, intermediate, hidden],
                              dtype=torch.bfloat16, device="cuda")
    weight_fp8, weight_scales = block_quantize_weight_fp8(
        weight_bf16, group_n=group_n, group_k=group_k)

    gemm_out = fp8_group_gemm_3d(
        out_token, weight_fp8, out_scales, weight_scales,
        expert_recv_count,
        group_n=group_n, group_k=group_k, BLOCK_M=BM,
        num_tokens_per_rank=num_tokens, TOPK=topk,
        output_dtype=torch.bfloat16,
    )

    # DeepGEMM reference
    total = expert_recv_count.sum().item()
    expected_m = (total - 1) // local_experts + 1
    expected_m = ((expected_m + (BM or 64) - 1) // (BM or 64)) * (BM or 64)
    ref_out = fp8_group_gemm_deepgemm(
        out_token, weight_fp8, out_scales, weight_scales,
        expert_recv_count, expected_m,
        group_n=group_n, group_k=group_k,
        output_dtype=torch.bfloat16, BM=BM or 64,
    )

    # Compare valid rows only
    gemm_diff = 0.0
    gemm_max_val = 0.0
    for e in range(local_experts):
        n = expert_recv_count[e].item()
        if n > 0:
            diff = (gemm_out[e, :n].to(torch.float32) -
                    ref_out[e, :n].to(torch.float32)).abs().max().item()
            gemm_diff = max(gemm_diff, diff)
            gemm_max_val = max(gemm_max_val, gemm_out[e, :n].abs().max().item())

    if rank == 0:
        print(f"  GEMM output max_diff: {gemm_diff:.4f}")

    gemm_atol, gemm_rtol = 0.5, 0.1
    gemm_threshold = gemm_atol + gemm_rtol * gemm_max_val
    if gemm_diff < gemm_threshold:
        if rank == 0:
            print(f"  [PASS] warp dispatch + GEMM test passed!")
        return ctx
    else:
        if rank == 0:
            print(f"  [FAIL] warp dispatch + GEMM test FAILED")
        raise AssertionError(
            f"gemm_diff={gemm_diff:.4f} (threshold={gemm_threshold:.4f})")


def test_moe_dispatch_warp_performance(
    ctx: DispatchContext,
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
    """Benchmark pipeline: quant_gather → dispatch_warp → fp8_group_gemm_3d."""
    if rank == 0:
        print(f"[perf] rank={rank}/{world_size}, M={num_tokens}, K={hidden}, "
              f"N={intermediate}, total_E={num_experts}, topk={topk}, "
              f"BM={BM if BM is not None else 'auto'}, group_n={group_n}, group_k={group_k}")

    assert num_experts % world_size == 0
    experts_per_rank = num_experts // world_size
    max_m = num_tokens * topk

    torch.manual_seed(rank)
    x_bf16 = torch.randn([num_tokens, hidden], dtype=torch.bfloat16, device="cuda")

    exp_indices = generate_round_robin_exp_indices(
        num_tokens, num_experts, topk, world_size=world_size, rank=rank).cuda()

    global_exp = exp_indices.view(-1).to(torch.int32)
    send_order = global_exp.argsort(stable=True)
    gather_idx = (send_order // topk).to(torch.int32)

    send_splits = torch.bincount(global_exp, minlength=num_experts).to(torch.int32)
    send_split_cumsum = splits_to_cumsum(send_splits)

    weight_bf16 = torch.randn([experts_per_rank, intermediate, hidden],
                              dtype=torch.bfloat16, device="cuda")
    weight_fp8, weight_scales = block_quantize_weight_fp8(
        weight_bf16, group_n=group_n, group_k=group_k)

    quant_gather(
        ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)

    # Stage 1: quant_gather
    def stage_quant():
        return quant_gather(ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)
    _, dur_quant = perf_func(stage_quant, warmup, iters)

    # Stage 2: dispatch (warp pull)
    def stage_dispatch():
        return dispatch_warp(ctx, max_m, send_split_cumsum, num_input_tokens=num_tokens, BM=BM or 64)
    (out_token, out_scales, expert_recv_count), dur_dispatch = \
        perf_func(stage_dispatch, warmup, iters)

    # Stage 3: fp8_group_gemm_3d
    def stage_gemm():
        return fp8_group_gemm_3d(
            out_token, weight_fp8, out_scales, weight_scales,
            expert_recv_count,
            group_n=group_n, group_k=group_k, BLOCK_M=BM,
            num_tokens_per_rank=num_tokens, TOPK=topk,
            output_dtype=torch.bfloat16,
        )
    _, dur_gemm = perf_func(stage_gemm, warmup, iters)

    # Stage 3b: DeepGEMM reference for GEMM-only performance comparison
    # SGLang pattern: compact out_token into tight m_max for better TMA efficiency
    m_max = ((max_m + 255) // 256) * 256
    ref_input = torch.zeros(experts_per_rank, m_max, hidden, dtype=FP8_DTYPE, device="cuda")
    ref_scales = torch.zeros(experts_per_rank, m_max, hidden // group_k, dtype=torch.float32, device="cuda")
    for e in range(experts_per_rank):
        n = expert_recv_count[e].item()
        if n > 0:
            ref_input[e, :n] = out_token[e, :n]
            ref_scales[e, :n] = out_scales[e, :n]

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

    # Full pipeline end-to-end
    def full_pipeline():
        quant_gather(ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)
        ot, os_, erc = dispatch_warp(ctx, max_m, send_split_cumsum, num_input_tokens=num_tokens, BM=BM or 64)
        return fp8_group_gemm_3d(
            ot, weight_fp8, os_, weight_scales,
            erc,
            group_n=group_n, group_k=group_k, BLOCK_M=BM,
            num_tokens_per_rank=num_tokens, TOPK=topk,
            output_dtype=torch.bfloat16,
        )
    _, dur_full = perf_func(full_pipeline, warmup, iters)

    sum_stages = dur_quant + dur_dispatch + dur_gemm
    total_tokens = max_m  # = num_tokens * topk
    flops = total_tokens * 2 * hidden * intermediate
    gflops_gemm = flops / dur_gemm / 1e9
    gflops_deepgemm = flops / dur_deepgemm / 1e9
    gflops_full = flops / dur_full / 1e9

    if rank == 0:
        gemm_bm = BM if BM is not None else \
            _pick_block_m(num_tokens * topk / experts_per_rank)
        gemm_swap = _should_enable_swap_ab(gemm_bm, 128)
        swap_tag = "(swapab)" if gemm_swap else "(no swapab)"
        print(f"  Stage timings (avg over {iters} iters):")
        print(f"    quant_gather:        {dur_quant:.3f} ms")
        print(f"    dispatch (warp):     {dur_dispatch:.3f} ms")
        print(f"    fp8_group_gemm_3d:   {dur_gemm:.3f} ms {swap_tag}")
        print(f"    DeepGEMM reference:  {dur_deepgemm:.3f} ms")
        print(f"    ───────────────────────────────────────")
        print(f"    Sum of stages:          {sum_stages:.3f} ms")
        print(f"    Full pipeline (e2e):    {dur_full:.3f} ms")
        print(f"    GEMM throughput:        {gflops_gemm:.2f} GFLOPS")
        print(f"    DeepGEMM throughput:    {gflops_deepgemm:.2f} GFLOPS")
        print(f"    GEMM speedup vs DeepGEMM: {dur_deepgemm / dur_gemm:.2f}x")
        print(f"    Pipeline throughput:    {gflops_full:.2f} GFLOPS")


# ─────────────────────── block dispatch tests ────────────────────────


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
    """Test block dispatch + GEMM.

    Pipeline: quant_gather → dispatch_block → fp8_group_gemm_2d
    """
    if rank == 0:
        print(f"[block dispatch+gemm] M={num_tokens}, K={hidden}, "
              f"N={intermediate}, total_E={num_experts}, topk={topk}, "
              f"BM={BM if BM is not None else 'auto'}, group_n={group_n}, group_k={group_k}, world_size={world_size}")

    if not _NVSHMEM_AVAILABLE:
        if rank == 0:
            print("  [SKIP] NVSHMEM not available")
        return

    assert num_experts % world_size == 0
    experts_per_rank = num_experts // world_size
    num_scale_groups = hidden // group_k
    max_m = num_tokens * topk

    torch.manual_seed(rank)
    x_bf16 = torch.randn([num_tokens, hidden], dtype=torch.bfloat16, device="cuda")

    exp_indices = generate_round_robin_exp_indices(
        num_tokens, num_experts, topk, world_size=world_size, rank=rank).cuda()

    global_exp = exp_indices.view(-1).to(torch.int32)
    send_order = global_exp.argsort(stable=True)
    gather_idx = (send_order // topk).to(torch.int32)
    quant_gather(
        ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)

    send_splits = torch.bincount(global_exp, minlength=num_experts).to(torch.int32)
    send_split_cumsum = splits_to_cumsum(send_splits)

    torch.distributed.barrier()

    recv_buf, recv_scales_buf, recv_splits = dispatch_block(
        ctx, max_m, send_split_cumsum)

    # Diagnostic: show dispatch distribution (after barrier for correct cross-rank visibility)
    torch.distributed.barrier()
    if rank == 0:
        total_recv = recv_splits.sum().item()
        print(f"  rank={rank}: dispatched {max_m} -> {total_recv} tokens")

    # ── Verify dispatch out_token: reconstruct expected from all ranks' send_buf ──
    send_fp8 = ctx.send_buf[:max_m]
    send_scales = ctx.send_scales[:max_m]
    all_fp8 = [torch.empty_like(send_fp8) for _ in range(world_size)]
    all_scales = [torch.empty_like(send_scales) for _ in range(world_size)]
    all_cumsums = [torch.empty_like(send_split_cumsum) for _ in range(world_size)]
    dist.all_gather(all_fp8, send_fp8)
    dist.all_gather(all_scales, send_scales)
    dist.all_gather(all_cumsums, send_split_cumsum)

    dispatch_data_err = 0.0
    dispatch_scale_err = 0.0
    for src_rank in range(world_size):
        src_fp8 = all_fp8[src_rank]
        src_scales = all_scales[src_rank]
        src_cumsum = all_cumsums[src_rank]
        recv_start = src_rank * max_m
        src_offset = 0

        for e in range(experts_per_rank):
            g = rank * experts_per_rank + e
            expected_cnt = (src_cumsum[g + 1] - src_cumsum[g]).item()
            actual_cnt = recv_splits[src_rank * experts_per_rank + e].item()

            if expected_cnt != actual_cnt:
                raise AssertionError(
                    f"split mismatch: rank={src_rank} expert={g} "
                    f"expected={expected_cnt} actual={actual_cnt}")

            if expected_cnt > 0:
                expected_fp8 = src_fp8[src_cumsum[g]:src_cumsum[g + 1]]
                expected_scales = src_scales[src_cumsum[g]:src_cumsum[g + 1]]
                actual_fp8 = recv_buf[recv_start + src_offset:recv_start + src_offset + expected_cnt]
                actual_scales = recv_scales_buf[recv_start + src_offset:recv_start + src_offset + expected_cnt]

                d = (expected_fp8.float() - actual_fp8.float()).abs().max().item()
                s = (expected_scales - actual_scales).abs().max().item()
                dispatch_data_err = max(dispatch_data_err, d)
                dispatch_scale_err = max(dispatch_scale_err, s)
                src_offset += expected_cnt

    if dispatch_data_err > 0 or dispatch_scale_err > 1e-6:
        if rank == 0:
            print(f"  [FAIL] dispatch output mismatch: "
                  f"data_err={dispatch_data_err:.6f}, scale_err={dispatch_scale_err:.6f}")
        raise AssertionError(
            f"dispatch output mismatch: data_err={dispatch_data_err:.6f}, "
            f"scale_err={dispatch_scale_err:.6f}")

    # GEMM (2D — directly on per-rank [WS*MAX_M, K] layout, no rearrange needed)
    weight_bf16 = torch.randn([experts_per_rank, intermediate, hidden],
                              dtype=torch.bfloat16, device="cuda")
    weight_fp8, weight_scales = block_quantize_weight_fp8(
        weight_bf16, group_n=group_n, group_k=group_k)

    gemm_out = fp8_group_gemm_2d(
        recv_buf, weight_fp8, recv_scales_buf, weight_scales,
        recv_splits,
        MAX_M=max_m, EXPERTS_PER_RANK=experts_per_rank, WS=world_size,
        group_n=group_n, group_k=group_k, BLOCK_M=BM,
        num_tokens_per_rank=num_tokens, TOPK=topk,
        output_dtype=torch.bfloat16,
    )

    # DeepGEMM reference — needs per-expert 3D layout [experts_per_rank, m_max, K]
    # SGLang pattern: m_max based on actual total assignments, aligned to 256 for TMA
    m_max = ((max_m + 255) // 256) * 256
    ref_input = torch.zeros(experts_per_rank, m_max, hidden, dtype=FP8_DTYPE, device="cuda")
    ref_scales = torch.zeros(experts_per_rank, m_max, num_scale_groups, dtype=torch.float32, device="cuda")
    expert_recv_count = torch.zeros(experts_per_rank, dtype=torch.int32, device="cuda")

    for e in range(experts_per_rank):
        dst_offset = 0
        for r in range(world_size):
            count = recv_splits[r * experts_per_rank + e].item()
            if count > 0:
                prefix = 0
                for e_prev in range(e):
                    prefix += recv_splits[r * experts_per_rank + e_prev].item()
                src_start = r * max_m + prefix
                ref_input[e, dst_offset:dst_offset + count] = recv_buf[src_start:src_start + count]
                ref_scales[e, dst_offset:dst_offset + count] = recv_scales_buf[src_start:src_start + count]
                dst_offset += count
        expert_recv_count[e] = dst_offset

    total = expert_recv_count.sum().item()
    expected_m = (total - 1) // experts_per_rank + 1
    expected_m = ((expected_m + (BM or 64) - 1) // (BM or 64)) * (BM or 64)

    ref_out = fp8_group_gemm_deepgemm(
        ref_input, weight_fp8, ref_scales, weight_scales,
        expert_recv_count, expected_m,
        group_n=group_n, group_k=group_k,
        output_dtype=torch.bfloat16, BM=BM or 64,
    )

    # Compare: gather 2D GEMM output into 3D for comparison
    gemm_diff = 0.0
    gemm_max_val = 0.0
    for e in range(experts_per_rank):
        count = expert_recv_count[e].item()
        if count > 0:
            for r in range(world_size):
                cnt_r = recv_splits[r * experts_per_rank + e].item()
                if cnt_r > 0:
                    prefix = 0
                    for e_prev in range(e):
                        prefix += recv_splits[r * experts_per_rank + e_prev].item()
                    src_start = r * max_m + prefix
                    ref_offset = 0
                    for r_prev in range(r):
                        ref_offset += recv_splits[r_prev * experts_per_rank + e].item()
                    diff = (gemm_out[src_start:src_start + cnt_r].to(torch.float32) -
                            ref_out[e, ref_offset:ref_offset + cnt_r].to(torch.float32)).abs().max().item()
                    gemm_diff = max(gemm_diff, diff)
                    gemm_max_val = max(gemm_max_val, gemm_out[src_start:src_start + cnt_r].abs().max().item())

    if rank == 0:
        print(f"  GEMM output max_diff: {gemm_diff:.4f}")

    gemm_atol, gemm_rtol = 0.5, 0.1
    gemm_threshold = gemm_atol + gemm_rtol * gemm_max_val
    if gemm_diff < gemm_threshold:
        if rank == 0:
            print(f"  [PASS] block dispatch + GEMM test passed!")
        return ctx
    else:
        if rank == 0:
            print(f"  [FAIL] block dispatch + GEMM test FAILED")
        raise AssertionError(
            f"gemm_diff={gemm_diff:.4f} (threshold={gemm_threshold:.4f})")


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
    """Benchmark pipeline: quant_gather → dispatch_block → fp8_group_gemm_2d."""
    if rank == 0:
        print(f"[perf] rank={rank}/{world_size}, M={num_tokens}, K={hidden}, "
              f"N={intermediate}, total_E={num_experts}, topk={topk}, "
              f"BM={BM if BM is not None else 'auto'}, group_n={group_n}, group_k={group_k}")

    assert num_experts % world_size == 0
    experts_per_rank = num_experts // world_size
    max_m = num_tokens * topk

    torch.manual_seed(rank)
    x_bf16 = torch.randn([num_tokens, hidden], dtype=torch.bfloat16, device="cuda")

    exp_indices = generate_round_robin_exp_indices(
        num_tokens, num_experts, topk, world_size=world_size, rank=rank).cuda()

    global_exp = exp_indices.view(-1).to(torch.int32)
    send_order = global_exp.argsort(stable=True)
    gather_idx = (send_order // topk).to(torch.int32)
    quant_gather(
        ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)

    send_splits = torch.bincount(global_exp, minlength=num_experts).to(torch.int32)
    send_split_cumsum = splits_to_cumsum(send_splits)

    weight_bf16 = torch.randn([experts_per_rank, intermediate, hidden],
                              dtype=torch.bfloat16, device="cuda")
    weight_fp8, weight_scales = block_quantize_weight_fp8(
        weight_bf16, group_n=group_n, group_k=group_k)

    # Stage 1: quant_gather
    def stage_quant():
        return quant_gather(ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)
    _, dur_quant = perf_func(stage_quant, warmup, iters)

    # Stage 2: dispatch (block)
    def stage_dispatch():
        return dispatch_block(ctx, max_m, send_split_cumsum)
    (recv_buf, recv_scales_buf, recv_splits), dur_dispatch = \
        perf_func(stage_dispatch, warmup, iters)

    # Stage 3: fp8_group_gemm_2d (directly on per-rank layout, no rearrange)
    def stage_gemm_2d():
        return fp8_group_gemm_2d(
            recv_buf, weight_fp8, recv_scales_buf, weight_scales,
            recv_splits,
            MAX_M=max_m, EXPERTS_PER_RANK=experts_per_rank, WS=world_size,
            group_n=group_n, group_k=group_k, BLOCK_M=BM,
            num_tokens_per_rank=num_tokens, TOPK=topk,
            output_dtype=torch.bfloat16,
        )
    _, dur_gemm_2d = perf_func(stage_gemm_2d, warmup, iters)

    # Stage 3b: DeepGEMM reference (requires 3D rearrange)
    # SGLang pattern: m_max based on actual total assignments, aligned to 256 for TMA
    m_max = ((max_m + 255) // 256) * 256
    ref_input = torch.zeros(experts_per_rank, m_max, hidden, dtype=FP8_DTYPE, device="cuda")
    ref_scales = torch.zeros(experts_per_rank, m_max, hidden // group_k, dtype=torch.float32, device="cuda")
    expert_recv_count = torch.zeros(experts_per_rank, dtype=torch.int32, device="cuda")

    for e in range(experts_per_rank):
        dst_offset = 0
        for r in range(world_size):
            count = recv_splits[r * experts_per_rank + e].item()
            if count > 0:
                prefix = 0
                for e_prev in range(e):
                    prefix += recv_splits[r * experts_per_rank + e_prev].item()
                src_start = r * max_m + prefix
                ref_input[e, dst_offset:dst_offset + count] = recv_buf[src_start:src_start + count]
                ref_scales[e, dst_offset:dst_offset + count] = recv_scales_buf[src_start:src_start + count]
                dst_offset += count
        expert_recv_count[e] = dst_offset

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

    # Full pipeline end-to-end
    def full_pipeline():
        quant_gather(ctx, x_bf16, gather_idx, group_k=group_k, BLOCK_M=BM or 64)
        rb, rsb, rs = dispatch_block(ctx, max_m, send_split_cumsum)
        return fp8_group_gemm_2d(
            rb, weight_fp8, rsb, weight_scales,
            rs,
            MAX_M=max_m, EXPERTS_PER_RANK=experts_per_rank, WS=world_size,
            group_n=group_n, group_k=group_k, BLOCK_M=BM,
            num_tokens_per_rank=num_tokens, TOPK=topk,
            output_dtype=torch.bfloat16,
        )
    _, dur_full = perf_func(full_pipeline, warmup, iters)

    sum_stages = dur_quant + dur_dispatch + dur_gemm_2d
    total_tokens = max_m  # = num_tokens * topk
    flops = total_tokens * 2 * hidden * intermediate

    if rank == 0:
        gemm_bm = BM if BM is not None else \
            _pick_block_m(num_tokens * topk / experts_per_rank)
        gemm_swap = _should_enable_swap_ab(gemm_bm, 128)
        swap_tag = "(swapab)" if gemm_swap else "(no swapab)"
        print(f"  Stage timings (avg over {iters} iters):")
        print(f"    quant_gather:        {dur_quant:.3f} ms")
        print(f"    dispatch (block):    {dur_dispatch:.3f} ms")
        print(f"    fp8_group_gemm_2d:   {dur_gemm_2d:.3f} ms {swap_tag}")
        print(f"    DeepGEMM reference:  {dur_deepgemm:.3f} ms")
        print(f"    ───────────────────────────────────────")
        print(f"    Sum of stages:          {sum_stages:.3f} ms")
        print(f"    Full pipeline (e2e):    {dur_full:.3f} ms")
        if dur_gemm_2d > 0:
            print(f"    GEMM 2D throughput:     {flops / dur_gemm_2d / 1e9:.2f} GFLOPS")
        if dur_deepgemm > 0:
            print(f"    DeepGEMM throughput:    {flops / dur_deepgemm / 1e9:.2f} GFLOPS")
            print(f"    GEMM speedup vs DeepGEMM: {dur_deepgemm / dur_gemm_2d:.2f}x")
        if dur_full > 0:
            print(f"    Pipeline throughput:    {flops / dur_full / 1e9:.2f} GFLOPS")


# ──────────────────────────── main ────────────────────────────

def worker(local_rank: int, num_gpus: int, args: argparse.Namespace):
    """Per-GPU worker — spawned by torch.multiprocessing.spawn."""
    # Restore fork start method so that multiprocessing.Pool in profiler_utils
    # (trace merge) inherits deep_gemm instead of re-importing it via spawn.
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    rank, local_rank, world_size = initialize_distributed(local_rank, num_gpus)

    # ── Warp-level dispatch tests ──
    if rank == 0:
        print("=" * 60)
        print("Warp-Level Dispatch Test")
        print("=" * 60)

    # Create warp dispatch context
    num_scale_groups = args.K // args.group_k
    max_m = args.M * args.topk
    ctx = create_dispatch_context(
        max_m, args.K, args.experts, world_size, rank, num_scale_groups)

    # warp dispatch + GEMM correctness test
    test_moe_dispatch_warp(
        ctx,
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
    )

    # ── Block dispatch tests ──
    if rank == 0:
        print()
        print("=" * 60)
        print("Block Dispatch Test")
        print("=" * 60)

    block_ctx = create_block_dispatch_context(
        max_m, args.K, args.experts, world_size, rank, num_scale_groups)

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
        group_n=args.group_n,
        group_k=args.group_k,
    )

    # ── Performance benchmarks ──
    if args.perf:
        if rank == 0:
            print()

        profile_name = f"mega_moe_nvshmem_{args.M}x{args.K}x{args.N}_e{args.experts}_topk{args.topk}"
        with group_profile(profile_name, args.profile):
            if rank == 0:
                print("─ Warp performance ─")
            test_moe_dispatch_warp_performance(
                ctx,
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
                print("\n─ Block performance ─")
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