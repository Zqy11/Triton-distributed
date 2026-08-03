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
import argparse
import os

import deep_gemm
import torch
import triton
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

from triton_dist.deepgemm import get_best_config
from triton_dist.kernels.nvidia.gemm_allgather import create_gemm_ag_context, deepgemm_allgather_op
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import dist_print, initialize_distributed, nvshmem_barrier_all_on_stream, sleep_async


def torch_gemm_ag(A, weight, tp_group):
    local_out = torch.matmul(A, weight.T)
    full_out = torch.empty((A.shape[0] * tp_group.size(), weight.shape[0]),
                           dtype=local_out.dtype, device=local_out.device)
    torch.distributed.all_gather_into_tensor(full_out, local_out, group=tp_group)
    return full_out


def verify(ctx, A, weight, gemm_config, tp_group, USE_MULTIMEM_ST=False, atol=0.1, rtol=0.1):
    torch_out = torch_gemm_ag(A, weight, tp_group)

    ctx.gemm_barrier_buf.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    a = per_token_cast_to_fp8(A, use_ue8m0=False)
    b = per_block_cast_to_fp8(weight, use_ue8m0=False)
    dist_out = deepgemm_allgather_op(ctx, a, b, gemm_config,
                                     copy_to_local=True, USE_MULTIMEM_ST=USE_MULTIMEM_ST)

    assert_allclose(torch_out.to(torch.bfloat16), dist_out.to(torch.bfloat16),
                    atol=atol, rtol=rtol, verbose=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument("--num_comm_sms", default=16, type=int)
    parser.add_argument("--check", default=False, action="store_true")
    parser.add_argument("--verify-iters", default=5, type=int)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--use-multimem-st", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)
    args = parse_args()
    dist_print(args)

    assert torch.cuda.get_device_capability()[0] >= 9, "DeepGEMM fp8 overlap requires SM90+"

    tp_group = initialize_distributed(args.seed)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - args.num_comm_sms
    deep_gemm.set_num_sms(NUM_GEMM_SMS)

    # BLOCK_SIZE_M/N must match deepgemm's actual tile size, otherwise barrier index mismatch causes deadlock
    cfg = get_best_config(args.M, args.N, args.K, num_sms=NUM_GEMM_SMS)
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = cfg.block_m, cfg.block_n, cfg.block_k
    dist_print(f"tile: M={BLOCK_SIZE_M} N={BLOCK_SIZE_N} K={BLOCK_SIZE_K} gemm_sms={NUM_GEMM_SMS}")

    gemm_config = triton.Config({
        "BLOCK_SIZE_M": BLOCK_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE_N,
        "BLOCK_SIZE_K": BLOCK_SIZE_K, "GROUP_SIZE_M": 1, "NUM_GEMM_SMS": NUM_GEMM_SMS,
    }, num_stages=2, num_warps=8)

    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context(
        ag_stream, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE,
        args.M, args.N, torch.bfloat16,
        NUM_COMM_SMS=args.num_comm_sms,
        TILE_MAP_LEVEL=0,  # deepgemm only supports per-tile granularity
    )
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    def make_data():
        A = torch.randn((args.M, args.K), dtype=torch.bfloat16, device="cuda") * 0.01
        weight = torch.randn((args.N, args.K), dtype=torch.bfloat16, device="cuda") * 0.01
        return A, weight

    A, weight = make_data()

    if args.check:
        for i in range(args.verify_iters):
            A, weight = make_data()
            verify(ctx, A, weight, gemm_config, tp_group, USE_MULTIMEM_ST=args.use_multimem_st)
        dist_print(f"RANK[{RANK}]: pass.", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        ctx.finalize()
        torch.distributed.destroy_process_group()
        exit(0)

    a_fp8 = per_token_cast_to_fp8(A, use_ue8m0=False)
    b_fp8 = per_block_cast_to_fp8(weight, use_ue8m0=False)

    def _deepgemm_ag():
        ctx.gemm_barrier_buf.fill_(0)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        return deepgemm_allgather_op(ctx, a_fp8, b_fp8, gemm_config,
                                     copy_to_local=True, USE_MULTIMEM_ST=args.use_multimem_st)

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "local")
    with group_profile(f"deepgemm_ag_{args.M}x{args.N}x{args.K}_{run_id}", args.profile, group=tp_group):
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        deepgemm_out, deepgemm_perf = perf_func(_deepgemm_ag, iters=args.iters, warmup_iters=args.warmup)

        torch.cuda.synchronize()
        sleep_async(100)
        torch_out, torch_perf = perf_func(lambda: torch_gemm_ag(A, weight, tp_group),
                                          iters=args.iters, warmup_iters=args.warmup)

    dist_print(f"deepgemm+ag #{RANK}, total={deepgemm_perf:0.4f}", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch+ag #{RANK}, total={torch_perf:0.4f}", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"speedup #{RANK}: {torch_perf/deepgemm_perf:.2f}x", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    ctx.finalize()
    torch.distributed.destroy_process_group()
