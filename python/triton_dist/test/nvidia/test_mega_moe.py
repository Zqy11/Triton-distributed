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
quant_gather = _mega_moe_module.quant_gather
block_quantize_weight_fp8 = _mega_moe_module.block_quantize_weight_fp8
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
    """Test block-level FP8 dispatch correctness (no GEMM).

    Verification: all_gather sorted FP8 tokens from all ranks, build expected
    recv layout, compare FP8 data + scales + split counts element-wise.
    """
    if rank == 0:
        print(f"[block dispatch+gemm] M={num_tokens}, K={hidden}, "
              f"N={intermediate}, total_E={num_experts}, topk={topk}, "
              f"BM={BM if BM is not None else 'auto'}, group_n={group_n}, group_k={group_k}, "
              f"world_size={world_size}")

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

    torch.distributed.barrier()

    recv_buf, recv_scales_buf, recv_splits = dispatch_block(
        ctx, max_m, send_split_cumsum)

    # All-gather sorted FP8 data and cumsums to build expected recv.
    # NB: dist.all_gather (NCCL) also provides the real cross-rank memory fence
    # that makes NVLink tl.store from other ranks visible locally.
    # recv_hdl.barrier() alone may not fence raw Triton tl.store operations.
    send_fp8 = ctx.send_buf[:max_m]
    send_scales = ctx.send_scales[:max_m]
    all_fp8 = [torch.empty_like(send_fp8) for _ in range(world_size)]
    all_scales = [torch.empty_like(send_scales) for _ in range(world_size)]
    all_cumsums = [torch.empty_like(send_split_cumsum) for _ in range(world_size)]
    dist.all_gather(all_fp8, send_fp8)
    dist.all_gather(all_scales, send_scales)
    dist.all_gather(all_cumsums, send_split_cumsum)

    # Diagnostic: show dispatch distribution (after all_gather for correct visibility)
    if rank == 0:
        total_sent = max_m  # already expanded by quant_gather (orig * topk)
        total_recv = recv_splits.sum().item()
        print(f"  rank={rank}: dispatched {total_sent} -> {total_recv} tokens")

    data_err = 0.0
    scale_err = 0.0
    split_err = 0
    total_checked = 0

    for src_rank in range(world_size):
        src_fp8 = all_fp8[src_rank]
        src_scales = all_scales[src_rank]
        src_cumsum = all_cumsums[src_rank]

        recv_start = src_rank * max_m
        src_offset = 0

        for e in range(experts_per_rank):
            g = rank * experts_per_rank + e  # global expert index
            expected_cnt = (src_cumsum[g + 1] - src_cumsum[g]).item()

            # Verify split count
            actual_cnt = recv_splits[src_rank * experts_per_rank + e].item()
            if expected_cnt != actual_cnt:
                split_err = max(split_err, abs(expected_cnt - actual_cnt))

            if expected_cnt > 0:
                expected_fp8 = src_fp8[src_cumsum[g]:src_cumsum[g + 1]]
                expected_scales = src_scales[src_cumsum[g]:src_cumsum[g + 1]]

                actual_fp8 = recv_buf[recv_start + src_offset:recv_start + src_offset + expected_cnt]
                actual_scales = recv_scales_buf[recv_start + src_offset:recv_start + src_offset + expected_cnt]

                d = (expected_fp8.float() - actual_fp8.float()).abs().max().item()
                data_err = max(data_err, d)
                s = (expected_scales - actual_scales).abs().max().item()
                scale_err = max(scale_err, s)
                total_checked += expected_cnt
                src_offset += expected_cnt

    if rank == 0:
        pass  # correctness verified silently

    if not (data_err == 0 and scale_err == 0 and split_err == 0):
        if rank == 0:
            print(f"  [FAIL] data_err={data_err}, scale_err={scale_err}, split_err={split_err}")
        raise AssertionError(
            f"data_err={data_err}, scale_err={scale_err}, split_err={split_err}")

    # ── 2D GEMM on block dispatch output ──

    weight_bf16 = torch.randn([experts_per_rank, intermediate, hidden],
                              dtype=torch.bfloat16, device="cuda")
    weight_fp8, weight_scales = block_quantize_weight_fp8(
        weight_bf16, group_n=group_n, group_k=group_k)

    # Run 2D GEMM directly on [WS*max_m, K] per-rank layout
    gemm_out = fp8_group_gemm_2d(
        recv_buf, weight_fp8, recv_scales_buf, weight_scales,
        recv_splits,
        MAX_M=max_m, EXPERTS_PER_RANK=experts_per_rank, WS=world_size,
        group_n=group_n, group_k=group_k, BLOCK_M=BM,
        num_tokens_per_rank=num_tokens, TOPK=topk,
        output_dtype=torch.bfloat16,
    )

    # Reference: DeepGEMM on rearranged 3D layout
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
        n = expert_recv_count[e].item()
        if n > 0:
            for r in range(world_size):
                count = recv_splits[r * experts_per_rank + e].item()
                if count > 0:
                    prefix = 0
                    for e_prev in range(e):
                        prefix += recv_splits[r * experts_per_rank + e_prev].item()
                    src_start = r * max_m + prefix
                    # Find where this expert's tokens from rank r are in the 3D ref
                    ref_offset = 0
                    for r_prev in range(r):
                        ref_offset += recv_splits[r_prev * experts_per_rank + e].item()
                    for e_accum in range(e):
                        pass  # already counted by expert_recv_count layout
                    # Actually just compare the 2D output with ref at same positions
                    diff = (gemm_out[src_start:src_start + count].to(torch.float32) -
                            ref_out[e, ref_offset:ref_offset + count].to(torch.float32)).abs().max().item()
                    gemm_diff = max(gemm_diff, diff)
                    gemm_max_val = max(gemm_max_val, gemm_out[src_start:src_start + count].abs().max().item())

    if rank == 0:
        print(f"  GEMM output max_diff: {gemm_diff:.4f}")

    gemm_atol, gemm_rtol = 0.5, 0.1
    gemm_threshold = gemm_atol + gemm_rtol * gemm_max_val
    if gemm_diff < gemm_threshold:
        if rank == 0:
            print(f"  [PASS] block dispatch + GEMM test passed!")
    else:
        if rank == 0:
            print(f"  [FAIL] block dispatch + GEMM test FAILED")
        raise AssertionError(f"gemm_diff={gemm_diff:.4f} (threshold={gemm_threshold:.4f})")

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
    """Benchmark block dispatch + 2D GEMM pipeline."""
    if rank == 0:
        print(f"[block perf] rank={rank}/{world_size}, M={num_tokens}, K={hidden}, "
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

    # Stage 0: fused quant+gather into send_buf
    def stage_quant_gather():
        return quant_gather(
            ctx, x_bf16, gather_idx,
            group_k=group_k, BLOCK_M=BM or 64)

    _, dur_quant = perf_func(stage_quant_gather, warmup, iters)

    # Stage 1: dispatch (push)
    def stage_dispatch():
        return dispatch_block(ctx, max_m, send_split_cumsum)

    (recv_buf, recv_scales_buf, recv_splits), dur_dispatch = \
        perf_func(stage_dispatch, warmup, iters)

    # Stage 2: 2D GEMM on [WS*max_m, K] per-rank layout
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

    # Stage 2b: DeepGEMM reference (requires 3D layout rearrange)
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
        rb, rsb, rs = dispatch_block(ctx, max_m, send_split_cumsum)
        return fp8_group_gemm_2d(
            rb, weight_fp8, rsb, weight_scales, rs,
            MAX_M=max_m, EXPERTS_PER_RANK=experts_per_rank, WS=world_size,
            group_n=group_n, group_k=group_k, BLOCK_M=BM,
            num_tokens_per_rank=num_tokens, TOPK=topk,
            output_dtype=torch.bfloat16,
        )
    _, dur_full = perf_func(full_pipeline, warmup, iters)

    total_tokens = max_m  # = num_tokens * topk
    data_bytes = total_tokens * hidden                      # FP8 data (1 byte/elem)
    scale_bytes = total_tokens * (hidden // group_k) * 4    # FP32 scales
    total_bytes = data_bytes + scale_bytes
    flops = total_tokens * 2 * hidden * intermediate
    sum_stages = dur_quant + dur_dispatch + dur_gemm_2d

    if rank == 0:
        gemm_bm = BM if BM is not None else \
            _pick_block_m(num_tokens * topk / experts_per_rank)
        gemm_swap = _should_enable_swap_ab(gemm_bm, 128)
        swap_tag = "(swapab)" if gemm_swap else "(no swapab)"
        print(f"  Stage timings (avg over {iters} iters):")
        print(f"    quant_gather:        {dur_quant:.3f} ms")
        print(f"    dispatch (push):     {dur_dispatch:.3f} ms")
        print(f"    fp8_group_gemm_2d:   {dur_gemm_2d:.3f} ms {swap_tag}")
        print(f"    DeepGEMM reference:  {dur_deepgemm:.3f} ms")
        print(f"    ───────────────────────────────────────")
        print(f"    Sum of stages:          {sum_stages:.3f} ms")
        print(f"    Full pipeline (e2e):    {dur_full:.3f} ms")
        print(f"    GEMM 2D throughput:     {flops / dur_gemm_2d / 1e9:.2f} GFLOPS")
        print(f"    DeepGEMM throughput:    {flops / dur_deepgemm / 1e9:.2f} GFLOPS")
        print(f"    GEMM speedup vs DeepGEMM: {dur_deepgemm / dur_gemm_2d:.2f}x")
        print(f"    Pipeline throughput:    {flops / dur_full / 1e9:.2f} GFLOPS")


# ──────────────────────────── main ────────────────────────────

def worker(local_rank: int, num_gpus: int, args: argparse.Namespace):
    """Per-GPU worker — spawned by torch.multiprocessing.spawn."""
    # Restore fork start method so that multiprocessing.Pool in profiler_utils
    # (trace merge) inherits deep_gemm instead of re-importing it via spawn.
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    rank, local_rank, world_size = initialize_distributed(local_rank, num_gpus)

    # Create block dispatch context
    num_scale_groups = args.K // args.group_k
    max_m = args.M * args.topk

    # ── Block-level dispatch tests ──
    if rank == 0:
        print("=" * 60)
        print("Block-Level Dispatch Test")
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
        group_k=args.group_k,
    )

    if args.perf:
        if rank == 0:
            print()

        profile_name = f"mega_moe_tl_store_{args.M}x{args.K}x{args.N}_e{args.experts}_topk{args.topk}"
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