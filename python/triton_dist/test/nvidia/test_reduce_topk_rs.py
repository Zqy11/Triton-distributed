import argparse
import os
from typing import Optional

import triton
import torch
from triton_dist.autotuner import contextual_autotune
from triton_dist.kernels.nvidia import (create_moe_rs_context)
from triton_dist.kernels.nvidia.moe_reduce_rs import run_moe_reduce_rs, get_auto_triton_config, moe_grouped_gemm_kernel, \
    MoEReduceRSContext, reduce_topk_reduce_scatter_a2a_intra_node, reduce_topk_reduce_scatter_fused_a2a_intra_node, reduce_topk_reduce_scatter_multimem_intra_node
from triton_dist.kernels.nvidia.moe_utils import calc_gather_scatter_index_triton
from triton_dist.utils import dist_print, initialize_distributed, sleep_async
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose


def create_rand_tensor(rank, shape, dtype=torch.float16, device="cuda"):
    return (-2 * torch.rand(shape, dtype=dtype, device=device) + 1) / 10 * (rank + 1)


def select_experts(pg: torch.distributed.ProcessGroup, num_ranks: int, topk: int, dtype: torch.dtype,
                   router_logits_shard: torch.Tensor):
    score = torch.softmax(router_logits_shard, dim=-1)
    local_topk_weight, local_topk_ids = torch.topk(score, topk)
    # do all-gather
    ntokens_per_rank = router_logits_shard.shape[0]
    ntokens = ntokens_per_rank * num_ranks
    full_topk_ids = torch.zeros((ntokens, topk), dtype=torch.int32, device="cuda")
    full_topk_weight = torch.zeros((ntokens, topk), dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(full_topk_weight, local_topk_weight, group=pg)
    torch.distributed.all_gather_into_tensor(full_topk_ids, local_topk_ids.to(torch.int32), group=pg)
    return full_topk_ids, full_topk_weight


THRESHOLD_MAP = {torch.float16: (1e-2, 1e-2), torch.bfloat16: (1e-2, 1e-2)}


def moe_reduce_rs_torch(x: torch.Tensor, w: torch.Tensor, chosen_experts: torch.Tensor, expert_weight: torch.Tensor,
                        pg: torch.distributed.ProcessGroup):
    M, _ = x.shape
    ntokens, topk = expert_weight.shape
    num_experts, _, hidden_dim = w.shape
    world_size = pg.size()
    ntokens_per_rank = ntokens // world_size
    assert ntokens * topk == M
    assert x.shape[1] == w.shape[1]
    grouped_gemm_out = torch.zeros((M, hidden_dim), dtype=x.dtype, device="cuda")
    chosen_experts = chosen_experts.view(-1)
    expert_weight = expert_weight.view(-1)
    for i in range(num_experts):
        mask = chosen_experts == i
        if mask.sum():
            grouped_gemm_out[mask] = (x[mask] @ w[i]) * expert_weight[mask, None]
    out_reduce_topk = torch.sum(grouped_gemm_out.reshape(ntokens, topk, hidden_dim), dim=1, keepdim=False)
    out_rs = torch.zeros((ntokens_per_rank, hidden_dim), dtype=x.dtype, device="cuda")
    torch.distributed.reduce_scatter_tensor(out_rs, out_reduce_topk, group=pg)
    return out_rs


def run_moe_reduce_rs_triton_non_overlap(x: torch.Tensor, weights: torch.Tensor, chosen_experts: torch.Tensor,
                                         expert_weight: torch.Tensor, n_chunks=2,
                                         rs_ctx: MoEReduceRSContext = None,
                                         config: Optional[triton.Config] = None):
    M, K_per_rank = x.shape
    N = weights.shape[-1]
    num_experts = weights.shape[0]
    topk = chosen_experts.shape[1]
    ntokens = M // topk
    ntokens_per_rank = ntokens // torch.distributed.get_world_size()
    config = config or get_auto_triton_config(M, N, K_per_rank, topk, num_experts, n_chunks, False, x.dtype)
    block_size_m = config.kwargs["BLOCK_SIZE_M"]
    _, _, gather_index, expert_index, M_pad_gpu = calc_gather_scatter_index_triton(chosen_experts, num_experts,
                                                                                   block_size_m)
    grouped_gemm_out = torch.empty(
        (M, N),
        dtype=x.dtype,
        device=torch.cuda.current_device(),
    )
    out = torch.empty((ntokens_per_rank, N), dtype=x.dtype, device=torch.cuda.current_device())
    M_pad_approx = (triton.cdiv(M, block_size_m) + num_experts) * block_size_m
    grid = lambda META: (triton.cdiv(M_pad_approx, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    moe_grouped_gemm_kernel[grid](x, weights, grouped_gemm_out, expert_weight, gather_index, expert_index, M_pad_gpu,
                                  N, x.shape[1], num_experts, x.stride(0), x.stride(1), weights.stride(0),
                                  weights.stride(1), weights.stride(2), grouped_gemm_out.stride(0),
                                  grouped_gemm_out.stride(1), topk, **config.all_kwargs())

    rs_ctx.gemm_done_flag[:n_chunks] = 1
    N_per_chunk = rs_ctx.N // n_chunks
    block_size_m = max(1, 16 * 1024 // N_per_chunk // x.itemsize)  # each thread with a uint4 load
    block_size_n = N_per_chunk
    reduce_topk_reduce_scatter_a2a_intra_node(grouped_gemm_out, rs_ctx, ntokens, n_chunks, out, block_size_m, block_size_n)
    # reduce_topk_reduce_scatter_fused_a2a_intra_node(grouped_gemm_out, rs_ctx, ntokens, n_chunks, out, block_size_m, block_size_n)
    # reduce_topk_reduce_scatter_multimem_intra_node(grouped_gemm_out, rs_ctx, ntokens, n_chunks, out, block_size_m, block_size_n)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)  # num_tokens
    parser.add_argument("N", type=int)  # hidden_size
    parser.add_argument("K", type=int)  # intermediate_size
    parser.add_argument("E", type=int)  # num_experts
    parser.add_argument("TOPK", type=int)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="perf iterations")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autotune", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.autotune:
        configs = [
            triton.Config({"BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK}, num_stages=s, num_warps=w)
            for BN in [128]
            for BK in [32, 64]
            for s in [3, 4]
            for w in [4, 8]
        ]
        from triton_dist.kernels.nvidia import moe_reduce_rs
        moe_reduce_rs.moe_gather_rs_grouped_gemm_kernel = triton.autotune(configs=configs, key=["M", "N", "K"])(
            moe_reduce_rs.moe_gather_rs_grouped_gemm_kernel)
        run_moe_reduce_rs = contextual_autotune(is_dist=True)(run_moe_reduce_rs)
        moe_reduce_rs.moe_grouped_gemm_kernel = triton.autotune(configs=configs,
                                                                key=["M", "N",
                                                                     "K"])(moe_reduce_rs.moe_grouped_gemm_kernel)
        run_moe_reduce_rs_triton_non_overlap = contextual_autotune(is_dist=True)(run_moe_reduce_rs_triton_non_overlap)

    tp_group = initialize_distributed(args.seed)
    RANK = tp_group.rank()
    WORLD_SIZE = tp_group.size()
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE"))

    ntokens = args.M  # this is actually the tokens
    ntokens_per_rank = ntokens // WORLD_SIZE
    hidden_size = args.N
    intermediate_size = args.K
    num_experts = args.E
    topk = args.TOPK

    max_token_num = ntokens * topk
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    iters = args.iters
    warmup_iters = args.warmup

    router_logits = create_rand_tensor(RANK, (ntokens_per_rank, num_experts), device="cuda", dtype=dtype)

    M = ntokens * topk
    K_per_rank = intermediate_size // WORLD_SIZE
    intermediate_states = create_rand_tensor(RANK, (M, K_per_rank), device="cuda", dtype=dtype)
    w = create_rand_tensor(RANK, (num_experts, K_per_rank, hidden_size), device="cuda", dtype=dtype)
    if args.debug:
        intermediate_states.fill_((RANK + 1) / 10)
        for n in range(num_experts):
            w.fill_((n + 1) // hidden_size)
        router_logits = torch.arange(0, num_experts, device="cuda", dtype=router_logits.dtype).repeat(
            (ntokens_per_rank, 1))

    choosed_expert, expert_weight = select_experts(tp_group, WORLD_SIZE, topk, dtype, router_logits)

    rs_ctx = create_moe_rs_context(RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, max_token_num, hidden_size, num_experts, topk, dtype)

    func_torch = lambda: moe_reduce_rs_torch(intermediate_states, w, choosed_expert, expert_weight, tp_group)
    func_triton_non_overlap = lambda: run_moe_reduce_rs_triton_non_overlap(intermediate_states, w, choosed_expert,
                                                                           expert_weight, n_chunks=7, rs_ctx=rs_ctx)

    # runs
    output_torch = func_torch()
    output_triton_non_overlap = func_triton_non_overlap()

    atol, rtol = THRESHOLD_MAP[dtype]
    assert_allclose(output_triton_non_overlap, output_torch, atol=atol, rtol=rtol)

    with group_profile(f"moe_rs_non_overlap_{os.environ['TORCHELASTIC_RUN_ID']}", do_prof=args.profile, group=tp_group):
        sleep_async(100)
        output, duration_ms_triton_non_overlap = perf_func(func_triton_non_overlap, iters=iters,
                                                           warmup_iters=warmup_iters)

    dist_print(f"triton non-overlap #{RANK} {duration_ms_triton_non_overlap:0.2f} ms/iter", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    rs_ctx.finalize()
    # finalize_distributed()
    torch.distributed.destroy_process_group()
