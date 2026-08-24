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
Test AllGather + GEMM Overlap (AG first, GEMM follows) - nvshmem4py version

Usage:
    torchrun --nproc_per_node=4 test_ag_gemm_nvshmem4py.py

Features:
    1. AllGather + DeepGEMM performance benchmark (8-way comparison)
    2. Correctness verification for Triton block FP8 GEMM and fused quantize+GEMM
    3. NVSHMEM AllGather + GEMM overlap performance
"""

import os
import torch
import torch.distributed as dist
import argparse
import time
import triton
from typing import Optional

# Directly import module file, bypass triton_dist.__init__.py dependency issues
import importlib.util

# Import deepgemm get_best_config
_deepgemm_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "deepgemm.py"
)
_deepgemm_spec = importlib.util.spec_from_file_location("deepgemm", _deepgemm_path)
_deepgemm_module = importlib.util.module_from_spec(_deepgemm_spec)
_deepgemm_spec.loader.exec_module(_deepgemm_module)
get_best_config = _deepgemm_module.get_best_config

# Import from the new allgather_gemm module
_ag_gemm_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "kernels", "nvidia", "allgather_gemm_nvshmem4py.py"
)
_ag_gemm_spec = importlib.util.spec_from_file_location("allgather_gemm_nvshmem4py", _ag_gemm_path)
_ag_gemm_module = importlib.util.module_from_spec(_ag_gemm_spec)
_ag_gemm_spec.loader.exec_module(_ag_gemm_module)

# Export needed classes and functions from module
NVSHMEMContext = _ag_gemm_module.NVSHMEMContext
AllGatherGemmContextNVSHMEM = _ag_gemm_module.AllGatherGemmContextNVSHMEM
create_allgather_gemm_context_nvshmem = _ag_gemm_module.create_allgather_gemm_context_nvshmem
allgather_gemm_op_nvshmem = _ag_gemm_module.allgather_gemm_op_nvshmem
w8a8_block_fp8_matmul_triton = _ag_gemm_module.w8a8_block_fp8_matmul_triton
bf16_a_block_fp8_matmul_triton = _ag_gemm_module.bf16_a_block_fp8_matmul_triton
per_token_group_quant_fp8 = _ag_gemm_module.per_token_group_quant_fp8

# deep_gemm optional import
try:
    import deep_gemm
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False
    deep_gemm = None


def make_fp8_inputs(M: int, K: int, N: int, seed: int = 42):
    """
    Construct (fp8_tensor, scale) tuple inputs required by deep_gemm.fp8_gemm_nt.

    Use deep_gemm's built-in per_token_cast_to_fp8 and per_block_cast_to_fp8 for quantization.

    Returns:
        a_tuple, b_tuple, a_ref_bf16, b_ref_bf16
        where a_ref_bf16 / b_ref_bf16 are used for reference GEMM computation.
    """
    if not DEEP_GEMM_AVAILABLE:
        raise RuntimeError("deep_gemm not available, cannot create FP8 inputs")

    from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

    torch.manual_seed(seed)
    # Generate bf16 reference data
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Use deep_gemm's built-in quantization functions
    use_ue8m0 = False  # Use e4m3 scale format
    a_tuple = per_token_cast_to_fp8(a_bf16, use_ue8m0=use_ue8m0)
    b_tuple = per_block_cast_to_fp8(b_bf16, use_ue8m0=use_ue8m0)

    return a_tuple, b_tuple, a_bf16, b_bf16


def deepgemm_reference(a_bf16: torch.Tensor, b_bf16: torch.Tensor,
                        tp_group: dist.ProcessGroup) -> torch.Tensor:
    """
    DeepGEMM reference implementation: bf16 matmul + AllGather.
    Used to compare with deepgemm_allgather_op_nvshmem output.
    """
    local_out = torch.matmul(a_bf16, b_bf16.T).to(torch.bfloat16)
    world_size = dist.get_world_size(tp_group)
    M, N = local_out.shape
    ag_out = torch.empty((M * world_size, N), dtype=torch.bfloat16, device=local_out.device)
    dist.all_gather_into_tensor(ag_out, local_out, group=tp_group)
    return ag_out


def initialize_distributed():
    """Initialize distributed environment"""
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)

    # Initialize PyTorch distributed
    dist.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        device_id=torch.device(f"cuda:{LOCAL_RANK}"),
    )

    assert dist.is_initialized()

    # Create TP group
    tp_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    dist.barrier(tp_group)

    return RANK, LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, tp_group


def assert_allclose(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2, name: str = ""):
    """Verify correctness"""
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: Shape mismatch: {actual.shape} vs {expected.shape}")

    diff = (actual - expected).abs()
    max_diff = diff.max().item()

    if max_diff > atol + rtol * expected.abs().max().item():
        raise AssertionError(f"{name}: Max diff {max_diff} exceeds tolerance (atol={atol}, rtol={rtol})")

    return True


def perf_func(func, warmup_iters=10, iters=100, *args, **kwargs):
    """Performance measurement"""
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


def test_allgather_deepgemm_performance(rank, world_size, tp_group,
                                         M=1024, N=7168, K=2048,
                                         warmup=10, iters=100):
    """
    AllGather + DeepGEMM performance test (AG->GEMM pattern).

    Compare eight items:
      1. NCCL AllGather only
      2. Quantize only (bf16 -> fp8, per_token_group_quant_fp8)
      3. DeepGEMM only (on gathered+quantized A, [M*ws, K] @ [K, N])
      4. NCCL AllGather + Quantize + DeepGEMM (sequential, no overlap)
      5. PyTorch fused_all_gather_scaled_matmul
      5b.PyTorch fused_all_gather_matmul (bf16 input, no quant)
      6. Triton block-wise FP8 matmul (on gathered A, per-token-group + per-block quant)
      7. Fused bf16-A quantize + Triton block FP8 matmul (quantize inside kernel)
      8. NVSHMEM AllGather + GEMM overlap (AG first, compute follows with tile-level signaling)
    """
    print(f"[Rank {rank}] Testing AllGather+DeepGEMM performance (M={M}, N={N}, K={K})...")

    if not DEEP_GEMM_AVAILABLE:
        print(f"[Rank {rank}] deep_gemm not available, skipping performance test.")
        return

    a_tuple, b_tuple, a_bf16, b_bf16 = make_fp8_inputs(M, K, N, seed=rank + 42)

    nccl_ag_a_bf16_buf = torch.empty((M * world_size, K), dtype=torch.bfloat16, device="cuda")
    nccl_ag_gemm_buf = torch.empty((M * world_size, N), dtype=torch.bfloat16, device="cuda")
    # Pre-populate gathered A for #2 and #3
    dist.all_gather_into_tensor(nccl_ag_a_bf16_buf, a_bf16, group=tp_group)

    # 1. NCCL AllGather only
    def run_nccl_ag_only():
        dist.all_gather_into_tensor(nccl_ag_a_bf16_buf, a_bf16, group=tp_group)
        return nccl_ag_a_bf16_buf

    _, dur_nccl_ag = perf_func(run_nccl_ag_only, warmup, iters)

    # 2. Quantize only (bf16 [M*ws, K] -> fp8)
    def run_quantize_only():
        return per_token_group_quant_fp8(nccl_ag_a_bf16_buf, group_size=128)

    _, dur_quantize = perf_func(run_quantize_only, warmup, iters)

    # 3. DeepGEMM only on gathered+quantized input [M*ws, K] @ [K, N]
    a_gathered_tuple = per_token_group_quant_fp8(nccl_ag_a_bf16_buf, group_size=128)

    def run_deepgemm_gathered():
        deep_gemm.fp8_gemm_nt(
            a_gathered_tuple, b_tuple, nccl_ag_gemm_buf,
            c=None,
            disable_ue8m0_cast=True,
            recipe=None,
        )
        return nccl_ag_gemm_buf

    _, dur_deepgemm_gathered = perf_func(run_deepgemm_gathered, warmup, iters)

    # 4. NCCL AllGather + Quantize + DeepGEMM (sequential, no overlap)
    def run_nccl_ag_deepgemm():
        # AllGather bf16 A shard [M, K] -> [M*ws, K]
        dist.all_gather_into_tensor(nccl_ag_a_bf16_buf, a_bf16, group=tp_group)
        # Quantize gathered A to fp8
        a_gathered_tuple_seq = per_token_group_quant_fp8(nccl_ag_a_bf16_buf, group_size=128)
        # GEMM: [M*ws, K] @ [K, N] -> [M*ws, N]
        deep_gemm.fp8_gemm_nt(
            a_gathered_tuple_seq, b_tuple, nccl_ag_gemm_buf,
            c=None,
            disable_ue8m0_cast=True,
            recipe=None,
        )
        return nccl_ag_gemm_buf

    _, dur_nccl_ag_deepgemm = perf_func(run_nccl_ag_deepgemm, warmup, iters)

    # 5. PyTorch fused_all_gather_scaled_matmul
    group_name = tp_group.group_name
    A_shard_fp8 = a_tuple[0]
    B_fp8 = b_tuple[0].T  # [K, N], non-contiguous = column-major, required by cuBLASLt
    A_scale_symm = torch.tensor(1.0, device="cuda")
    B_scale_symm = torch.tensor(1.0, device="cuda")

    def run_symm_fused():
        ag_out, mm_outs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            A_shard_fp8,
            [B_fp8],
            A_scale_symm,
            [B_scale_symm],
            gather_dim=0,
            group_name=group_name,
            biases=[None],
            result_scales=[None],
            out_dtypes=[torch.bfloat16],
            use_fast_accum=[True],
        )
        return mm_outs[0]

    _, dur_symm = perf_func(run_symm_fused, warmup, iters)

    # 5b. PyTorch fused_all_gather_matmul (bf16 input, no quant)
    # b_bf16 shape is [N, K] (row-major). fused_all_gather_matmul expects B as [K, N].
    # b_bf16.T is [K, N] with stride (1, K) -> column-major, exactly what cuBLAS wants.
    B_bf16_KN = b_bf16.T  # [K, N], column-major (stride (1, K))
    assert B_bf16_KN.shape == (K, N) and B_bf16_KN.stride(0) == 1, \
        f"B_bf16_KN shape={B_bf16_KN.shape}, stride={B_bf16_KN.stride()}"

    def run_symm_fused_bf16():
        ag_out, mm_outs = torch.ops.symm_mem.fused_all_gather_matmul(
            a_bf16,
            [B_bf16_KN],
            gather_dim=0,
            group_name=group_name,
        )
        return mm_outs[0]

    _, dur_symm_bf16 = perf_func(run_symm_fused_bf16, warmup, iters)

    # 6. Triton block-wise FP8 matmul (w8a8_block_fp8_matmul) on gathered input
    def run_triton_block_fp8_gemm():
        return w8a8_block_fp8_matmul_triton(
            a_gathered_tuple[0], b_tuple[0],
            a_gathered_tuple[1], b_tuple[1],
            block_size=[64, 128, 128],
            output_dtype=torch.bfloat16,
        )

    _, dur_triton_gemm = perf_func(run_triton_block_fp8_gemm, warmup, iters)

    # 7. Fused bf16-A quantize + Triton block FP8 matmul on gathered input
    def run_fused_bf16_a_gemm():
        return bf16_a_block_fp8_matmul_triton(
            nccl_ag_a_bf16_buf, b_tuple[0],
            b_tuple[1],
            block_size=[64, 128, 128],
            output_dtype=torch.bfloat16,
        )

    _, dur_fused_gemm = perf_func(run_fused_bf16_a_gemm, warmup, iters)

    # Compute bf16 reference for correctness checks
    ref_gathered_out = torch.matmul(nccl_ag_a_bf16_buf, b_bf16.T).to(torch.bfloat16)

    # Verify Triton block FP8 matmul correctness (compare with bf16 reference)
    triton_out = run_triton_block_fp8_gemm()
    try:
        assert_allclose(triton_out, ref_gathered_out, atol=6e-2, rtol=6e-2,
                        name="Triton block FP8 matmul (gathered)")
        if rank == 0:
            print(f"[Rank {rank}] Triton block FP8 matmul correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] Triton block FP8 matmul correctness test FAILED: {e}")
        print(f"  output shape={triton_out.shape}, ref shape={ref_gathered_out.shape}")
        print(f"  output max={triton_out.abs().max():.4f}, ref max={ref_gathered_out.abs().max():.4f}")
        raise

    # Verify fused bf16-A quantize+GEMM correctness
    fused_out = run_fused_bf16_a_gemm()
    try:
        assert_allclose(fused_out, ref_gathered_out, atol=6e-2, rtol=6e-2,
                        name="Fused bf16-A block FP8 matmul (gathered)")
        if rank == 0:
            print(f"[Rank {rank}] Fused bf16-A block FP8 matmul correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] Fused bf16-A block FP8 matmul correctness test FAILED: {e}")
        print(f"  output shape={fused_out.shape}, ref shape={ref_gathered_out.shape}")
        print(f"  output max={fused_out.abs().max():.4f}, ref max={ref_gathered_out.abs().max():.4f}")
        raise

    # 8. NVSHMEM AllGather + GEMM overlap (AG first, compute follows)
    # Initialize NVSHMEM context for AG->GEMM
    NVSHMEMContext.from_process_group(tp_group)
    ag_gemm_stream = torch.cuda.Stream()
    ag_gemm_ctx = create_allgather_gemm_context_nvshmem(
        ag_stream=ag_gemm_stream,
        rank=rank,
        world_size=world_size,
        max_M=M,
        K=K,
        NUM_COMM_SMS=0,
        enable_multicast=True,
    )
    block_size = [64, 128, 128]  # [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]

    ag_gemm_out = None
    def run_nvshmem_ag_gemm():
        nonlocal ag_gemm_out
        ag_gemm_out = allgather_gemm_op_nvshmem(
            ag_gemm_ctx,
            a_bf16,      # [M, K] bf16 shard
            b_tuple[0],  # [N, K] fp8 weight
            b_tuple[1],  # scale
            block_size=block_size,
            output_dtype=torch.bfloat16,
            GROUP_SIZE_M=32,
        )
        return ag_gemm_out

    _, dur_nvshmem_ag_gemm = perf_func(run_nvshmem_ag_gemm, warmup, iters)

    # Verify NVSHMEM AG+GEMM correctness
    # (a) Verify AllGather result: symm_ag_a_buf should match NCCL AllGather exactly
    nvshmem_ag_result = ag_gemm_ctx.symm_ag_a_buf[:M * world_size, :]
    try:
        assert_allclose(nvshmem_ag_result, nccl_ag_a_bf16_buf, atol=0, rtol=0,
                        name="NVSHMEM AG result (vs NCCL AG)")
        if rank == 0:
            print(f"[Rank {rank}] NVSHMEM AG result correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] NVSHMEM AG result correctness test FAILED: {e}")
        print(f"  nvshmem shape={nvshmem_ag_result.shape}, nccl shape={nccl_ag_a_bf16_buf.shape}")
        print(f"  nvshmem max={nvshmem_ag_result.abs().max():.4f}, nccl max={nccl_ag_a_bf16_buf.abs().max():.4f}")
        raise

    # (b) Verify GEMM result
    try:
        assert_allclose(ag_gemm_out, ref_gathered_out, atol=6e-2, rtol=6e-2,
                        name="NVSHMEM AG+GEMM overlap")
        if rank == 0:
            print(f"[Rank {rank}] NVSHMEM AG+GEMM overlap correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] NVSHMEM AG+GEMM overlap correctness test FAILED: {e}")
        print(f"  output shape={ag_gemm_out.shape}, ref shape={ref_gathered_out.shape}")
        print(f"  output max={ag_gemm_out.abs().max():.4f}, ref max={ref_gathered_out.abs().max():.4f}")
        raise

    # Cleanup
    torch.cuda.synchronize()
    ag_gemm_ctx.finalize()

    # Calculate performance metrics
    # GEMM on gathered input [M*ws, K] @ [K, N], FLOPs = 2*M*ws*K*N
    ag_gemm_flops = 2 * M * world_size * K * N
    # AllGather bf16 A [M, K] (2 bytes/elem), logical volume = M*K*2*ws
    ag_in_bytes = M * K * 2 * world_size
    ag_bw = ag_in_bytes / dur_nccl_ag / 1e9 * 1000
    # Quantize throughput: bf16 [M*ws, K] -> fp8, read 2 bytes + write 1 byte per elem
    quantize_bytes = M * world_size * K * 3  # read bf16 (2B) + write fp8 (1B)
    quantize_bw = quantize_bytes / dur_quantize / 1e9 * 1000
    gemm_tflops_gathered = ag_gemm_flops / dur_deepgemm_gathered / 1e12 * 1000
    gemm_tflops_nccl = ag_gemm_flops / dur_nccl_ag_deepgemm / 1e12 * 1000
    dur_sum = dur_nccl_ag + dur_quantize + dur_deepgemm_gathered
    gemm_tflops_symm = ag_gemm_flops / dur_symm / 1e12 * 1000
    ag_bw_symm = ag_in_bytes / dur_symm / 1e9 * 1000
    gemm_tflops_symm_bf16 = ag_gemm_flops / dur_symm_bf16 / 1e12 * 1000
    ag_bw_symm_bf16 = ag_in_bytes / dur_symm_bf16 / 1e9 * 1000
    gemm_tflops_triton = ag_gemm_flops / dur_triton_gemm / 1e12 * 1000
    gemm_tflops_fused = ag_gemm_flops / dur_fused_gemm / 1e12 * 1000
    gemm_tflops_nvshmem_ag_gemm = ag_gemm_flops / dur_nvshmem_ag_gemm / 1e12 * 1000

    if rank == 0:
        print(f"[Rank {rank}] AllGather+DeepGEMM Performance results:")
        print(f"  1. NCCL AllGather:              {dur_nccl_ag:.3f} ms, {ag_bw:.2f} GB/s")
        print(f"  2. Quantize (bf16->fp8):        {dur_quantize:.3f} ms, {quantize_bw:.2f} GB/s")
        print(f"  3. DeepGEMM (gathered):         {dur_deepgemm_gathered:.3f} ms, {gemm_tflops_gathered:.2f} TFLOPS")
        print(f"  4. NCCL AG+Quant+DeepGEMM:      {dur_nccl_ag_deepgemm:.3f} ms, {gemm_tflops_nccl:.2f} TFLOPS (AG {dur_nccl_ag:.3f} + Quant {dur_quantize:.3f} + GEMM {dur_deepgemm_gathered:.3f} = {dur_sum:.3f} ms)")
        print(f"  5. Fused AG+GEMM:               {dur_symm:.3f} ms, {gemm_tflops_symm:.2f} TFLOPS, {ag_bw_symm:.2f} GB/s")
        print(f"  5b.Fused AG+GEMM (bf16):        {dur_symm_bf16:.3f} ms, {gemm_tflops_symm_bf16:.2f} TFLOPS, {ag_bw_symm_bf16:.2f} GB/s")
        print(f"  6. Triton block FP8 (gathered): {dur_triton_gemm:.3f} ms, {gemm_tflops_triton:.2f} TFLOPS")
        print(f"  7. Fused bf16-A quant+GEMM:     {dur_fused_gemm:.3f} ms, {gemm_tflops_fused:.2f} TFLOPS")
        print(f"  8. NVSHMEM AG+GEMM overlap:     {dur_nvshmem_ag_gemm:.3f} ms, {gemm_tflops_nvshmem_ag_gemm:.2f} TFLOPS")


def main():
    parser = argparse.ArgumentParser(description="Test AllGather + GEMM Overlap with nvshmem4py")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=7168)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    # Initialize distributed
    rank, local_rank, world_size, local_world_size, tp_group = initialize_distributed()

    print(f"[Rank {rank}] Starting tests: M={args.M}, N={args.N}, K={args.K}")

    test_allgather_deepgemm_performance(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K,
                                        warmup=args.warmup, iters=args.iters)

    print(f"[Rank {rank}] All tests passed!")

    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()