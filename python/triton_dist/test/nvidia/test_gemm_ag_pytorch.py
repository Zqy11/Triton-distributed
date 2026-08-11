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
Test GEMM + AllGather - PyTorch symmetric_memory version

Usage:
    torchrun --nproc_per_node=4 test_gemm_ag_pytorch.py

Features:
    1. Test PyTorch symmetric memory initialization
    2. Test AllGather correctness
    3. Test Multicast functionality
    4. Performance comparison (vs NCCL AllGather)

This version uses torch.distributed._symmetric_memory (PyTorch native NVSHMEM
backend) instead of nvshmem4py for symmetric memory management.
"""

import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import argparse
import time
import triton
from typing import Optional

# Directly import module file, bypass triton_dist.__init__.py dependency issues
import importlib.util

_deepgemm_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "deepgemm.py"
)
_deepgemm_spec = importlib.util.spec_from_file_location("deepgemm", _deepgemm_path)
_deepgemm_module = importlib.util.module_from_spec(_deepgemm_spec)
_deepgemm_spec.loader.exec_module(_deepgemm_module)
get_best_config = _deepgemm_module.get_best_config

_module_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "kernels", "nvidia", "gemm_allgather_pytorch.py"
)
_spec = importlib.util.spec_from_file_location("gemm_allgather_pytorch", _module_path)
_gemm_ag_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gemm_ag_module)

# Export needed classes and functions from module
SymmMemContext = _gemm_ag_module.SymmMemContext
GemmAGContextPyTorch = _gemm_ag_module.GemmAGContextPyTorch
create_gemm_ag_context_pytorch = _gemm_ag_module.create_gemm_ag_context_pytorch
allgather_op_pytorch = _gemm_ag_module.allgather_op_pytorch
gemm_op_pytorch = _gemm_ag_module.gemm_op_pytorch
gemm_allgather_op_pytorch = _gemm_ag_module.gemm_allgather_op_pytorch
deepgemm_allgather_op_pytorch = _gemm_ag_module.deepgemm_allgather_op_pytorch
persistent_gemm_notify_pytorch = _gemm_ag_module.persistent_gemm_notify_pytorch

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
    Used to compare with deepgemm_allgather_op_pytorch output.
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


def init_symm_mem(pg: dist.ProcessGroup):
    """Initialize PyTorch symmetric memory via SymmMemContext."""
    torch.cuda.synchronize()

    rank = pg.rank()
    world_size = pg.size()

    ctx = SymmMemContext.from_process_group(pg)

    print(f"[Rank {rank}] PyTorch symmetric memory (NVSHMEM backend) initialized successfully")
    return rank, world_size


def assert_allclose(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2, name: str = ""):
    """Verify correctness"""
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: Shape mismatch: {actual.shape} vs {expected.shape}")

    diff = (actual - expected).abs()
    max_diff = diff.max().item()

    if max_diff > atol + rtol * expected.abs().max().item():
        raise AssertionError(f"{name}: Max diff {max_diff} exceeds tolerance (atol={atol}, rtol={rtol})")

    return True


def torch_allgather_reference(input_tensor: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
    """PyTorch AllGather reference"""
    world_size = dist.get_world_size(tp_group)
    output = torch.empty((input_tensor.shape[0] * world_size, *input_tensor.shape[1:]),
                         dtype=input_tensor.dtype, device=input_tensor.device)
    dist.all_gather_into_tensor(output, input_tensor, group=tp_group)
    return output


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


def test_symm_mem_init(rank, world_size, tp_group):
    """Test PyTorch symmetric memory initialization"""
    print(f"[Rank {rank}] Testing PyTorch symmetric memory initialization...")

    # Initialize SymmMemContext
    ctx = SymmMemContext.from_process_group(tp_group)

    # Create test tensor
    tensor = ctx.create_tensor([1024, 1024], torch.bfloat16)
    print(f"[Rank {rank}] Created tensor: shape={tensor.shape}, dtype={tensor.dtype}")

    # Get peer buffer pointers (PyTorch symm_mem exposes raw ptrs via handle.buffer_ptrs)
    peer_ptrs = ctx.get_peer_buffer_ptrs(tensor)
    print(f"[Rank {rank}] Got {len(peer_ptrs)} peer buffer pointers")

    # Verify peer tensor addresses
    for i, ptr in enumerate(peer_ptrs):
        print(f"[Rank {rank}] Peer {i} buffer ptr: {ptr}")

    # Barrier
    ctx.barrier_all()

    torch.cuda.synchronize()
    ctx.free_tensor(tensor)
    print(f"[Rank {rank}] PyTorch symmetric memory initialization test passed!")


def test_allgather_correctness(rank, world_size, tp_group, M=1024, N=7168, dtype=torch.bfloat16):
    """Test AllGather correctness"""
    print(f"[Rank {rank}] Testing AllGather correctness (M={M}, N={N}, dtype={dtype})...")

    # Initialize SymmMemContext
    SymmMemContext.from_process_group(tp_group)

    # Create context
    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=dtype,
        NUM_COMM_SMS=16,
        enable_multicast=True,
    )

    # Generate test data
    torch.manual_seed(rank + 42)
    input_tensor = torch.randn(M, N, dtype=dtype, device="cuda")

    # Run AllGather
    output = allgather_op_pytorch(
        ctx,
        input_tensor,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        NUM_COMM_SMS=16,
        USE_MULTIMEM_ST=True,
        copy_to_local=False,
    )
    print(f"[Rank {rank}] AllGather output shape: {output.shape}")

    # Reference AllGather
    ref_output = torch_allgather_reference(input_tensor, tp_group)
    print(f"[Rank {rank}] Reference output shape: {ref_output.shape}")

    # Verify
    try:
        assert_allclose(output, ref_output, atol=6e-2, rtol=6e-2, name="AllGather")
        print(f"[Rank {rank}] AllGather correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] AllGather correctness test FAILED: {e}")
        print(f"[Rank {rank}] Output shape: {output.shape}, ref shape: {ref_output.shape}")
        print(f"[Rank {rank}] Output max: {output.abs().max()}, ref max: {ref_output.abs().max()}")
        raise

    torch.cuda.synchronize()
    ctx.finalize()


def test_allgather_multicast(rank, world_size, tp_group, M=1024, N=7168, dtype=torch.bfloat16):
    """Test Multicast AllGather"""
    print(f"[Rank {rank}] Testing Multicast AllGather (M={M}, N={N}, dtype={dtype})...")

    SymmMemContext.from_process_group(tp_group)

    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=dtype,
        NUM_COMM_SMS=16,
        enable_multicast=True,  # Enable multicast
    )

    if ctx.mc_ag_out_buf is None:
        print(f"[Rank {rank}] Multicast not supported, skipping...")
        torch.cuda.synchronize()
        ctx.finalize()
        return

    torch.manual_seed(rank + 42)
    input_tensor = torch.randn(M, N, dtype=dtype, device="cuda")

    # Run Multicast AllGather
    output = allgather_op_pytorch(
        ctx,
        input_tensor,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=128,
        NUM_COMM_SMS=16,
        USE_MULTIMEM_ST=True,
        copy_to_local=False,
    )

    # Reference AllGather
    ref_output = torch_allgather_reference(input_tensor, tp_group)

    # Verify
    try:
        assert_allclose(output, ref_output, atol=6e-2, rtol=6e-2, name="Multicast AllGather")
        print(f"[Rank {rank}] Multicast AllGather test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] Multicast AllGather test FAILED: {e}")
        raise

    torch.cuda.synchronize()
    ctx.finalize()


def test_performance(rank, world_size, tp_group, M=1024, N=7168, dtype=torch.bfloat16, warmup=10, iters=100):
    """Performance test"""
    print(f"[Rank {rank}] Testing performance (M={M}, N={N}, dtype={dtype})...")

    SymmMemContext.from_process_group(tp_group)

    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=dtype,
        NUM_COMM_SMS=16,
        enable_multicast=True,
    )

    torch.manual_seed(rank + 42)
    input_tensor = torch.randn(M, N, dtype=dtype, device="cuda")

    # PyTorch symm_mem AllGather performance
    def symm_mem_allgather():
        return allgather_op_pytorch(
            ctx,
            input_tensor,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=128,
            NUM_COMM_SMS=16,
            USE_MULTIMEM_ST=True,
            copy_to_local=False,
        )

    output_symm_mem, duration_symm_mem = perf_func(symm_mem_allgather, warmup, iters)

    # NCCL AllGather performance
    def nccl_allgather():
        return torch_allgather_reference(input_tensor, tp_group)

    output_nccl, duration_nccl = perf_func(nccl_allgather, warmup, iters)

    # Calculate bandwidth
    size_bytes = M * N * input_tensor.element_size() * world_size
    bw_symm_mem = size_bytes / duration_symm_mem * 1000 / 1e9  # GB/s
    bw_nccl = size_bytes / duration_nccl * 1000 / 1e9  # GB/s

    # Verify correctness
    assert_allclose(output_symm_mem, output_nccl, atol=6e-2, rtol=6e-2, name="Performance test")

    print(f"[Rank {rank}] Performance results:")
    print(f"  PyTorch symm_mem: {duration_symm_mem:.3f} ms, {bw_symm_mem:.2f} GB/s")
    print(f"  NCCL:             {duration_nccl:.3f} ms, {bw_nccl:.2f} GB/s")
    print(f"  Speedup: {bw_symm_mem / bw_nccl:.2f}x")

    torch.cuda.synchronize()
    ctx.finalize()


def test_multi_scale(rank, world_size, tp_group):
    """Multi-scale test"""
    print(f"[Rank {rank}] Running multi-scale tests...")

    test_configs = [
        (256, 1024, torch.bfloat16),
        (1024, 4096, torch.bfloat16),
        (1024, 7168, torch.bfloat16),
        (4096, 8192, torch.bfloat16),
        (1024, 1024, torch.float16),
        (1024, 7168, torch.float16),
    ]

    for M, N, dtype in test_configs:
        print(f"\n[Rank {rank}] Testing M={M}, N={N}, dtype={dtype}")
        try:
            test_allgather_correctness(rank, world_size, tp_group, M=M, N=N, dtype=dtype)
        except Exception as e:
            print(f"[Rank {rank}] Test FAILED for M={M}, N={N}: {e}")


def test_gemm_correctness(rank, world_size, tp_group, M=1024, N=7168, K=2048, dtype=torch.bfloat16):
    """Test pure GEMM correctness"""
    print(f"[Rank {rank}] Testing GEMM correctness (M={M}, N={N}, K={K}, dtype={dtype})...")

    # Initialize SymmMemContext
    SymmMemContext.from_process_group(tp_group)

    # Create context
    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=dtype,
        NUM_COMM_SMS=16,
        enable_multicast=True,
    )

    # Generate test data
    torch.manual_seed(rank + 42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(N, K, dtype=dtype, device="cuda")

    # GEMM configuration
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - ctx.NUM_COMM_SMS
    gemm_config = triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
         'GROUP_SIZE_M': 8, 'NUM_GEMM_SMS': NUM_GEMM_SMS}
    )

    # Run GEMM
    output = gemm_op_pytorch(ctx, a, b, gemm_config)

    # Reference GEMM
    ref_output = torch.matmul(a, b.T)

    # Verify
    try:
        assert_allclose(output, ref_output, atol=1e-2, rtol=1e-2, name="GEMM")
        print(f"[Rank {rank}] GEMM correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] GEMM correctness test FAILED: {e}")
        print(f"[Rank {rank}] Output shape: {output.shape}, ref shape: {ref_output.shape}")
        print(f"[Rank {rank}] Output max: {output.abs().max()}, ref max: {ref_output.abs().max()}")
        raise

    # Cleanup
    torch.cuda.synchronize()
    ctx.finalize()


def test_gemm_allgather_correctness(rank, world_size, tp_group, M=1024, N=7168, K=2048, dtype=torch.bfloat16):
    """Test GEMM + AllGather fusion correctness"""
    print(f"[Rank {rank}] Testing GEMM+AllGather correctness (M={M}, N={N}, K={K}, dtype={dtype})...")

    # Initialize SymmMemContext
    SymmMemContext.from_process_group(tp_group)

    # Create context
    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=dtype,
        NUM_COMM_SMS=8,
        enable_multicast=True,
    )

    # Generate test data
    torch.manual_seed(rank + 42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(N, K, dtype=dtype, device="cuda")

    # GEMM configuration
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - ctx.NUM_COMM_SMS
    gemm_config = triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
         'GROUP_SIZE_M': 8, 'NUM_GEMM_SMS': NUM_GEMM_SMS}
    )

    # Run GEMM + AllGather
    output = gemm_allgather_op_pytorch(ctx, a, b, gemm_config, copy_to_local=False, USE_MULTIMEM_ST=True)

    # Reference GEMM + AllGather
    local_gemm = torch.matmul(a, b.T)
    ref_output = torch_allgather_reference(local_gemm, tp_group)

    # Verify
    try:
        assert_allclose(output, ref_output, atol=1e-2, rtol=1e-2, name="GEMM+AllGather")
        print(f"[Rank {rank}] GEMM+AllGather correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] GEMM+AllGather correctness test FAILED: {e}")
        print(f"[Rank {rank}] Output shape: {output.shape}, ref shape: {ref_output.shape}")
        print(f"[Rank {rank}] Output max: {output.abs().max()}, ref max: {ref_output.abs().max()}")
        raise

    # Cleanup
    torch.cuda.synchronize()
    ctx.finalize()


def test_gemm_allgather_multicast(rank, world_size, tp_group, M=1024, N=7168, K=2048, dtype=torch.bfloat16):
    """Test GEMM + AllGather Multicast mode"""
    print(f"[Rank {rank}] Testing GEMM+AllGather Multicast (M={M}, N={N}, K={K}, dtype={dtype})...")

    # Initialize SymmMemContext
    SymmMemContext.from_process_group(tp_group)

    # Create context (enable multicast)
    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=dtype,
        NUM_COMM_SMS=8,
        enable_multicast=True,
    )

    if ctx.mc_ag_out_buf is None:
        print(f"[Rank {rank}] Multicast not supported, skipping...")
        torch.cuda.synchronize()
        ctx.finalize()
        return

    # Generate test data
    torch.manual_seed(rank + 42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(N, K, dtype=dtype, device="cuda")

    # GEMM configuration
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - ctx.NUM_COMM_SMS
    gemm_config = triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
         'GROUP_SIZE_M': 8, 'NUM_GEMM_SMS': NUM_GEMM_SMS}
    )

    # Run GEMM + AllGather (Multicast)
    output = gemm_allgather_op_pytorch(ctx, a, b, gemm_config, copy_to_local=False, USE_MULTIMEM_ST=True)

    # Reference GEMM + AllGather
    local_gemm = torch.matmul(a, b.T)
    ref_output = torch_allgather_reference(local_gemm, tp_group)

    # Verify
    try:
        assert_allclose(output, ref_output, atol=1e-2, rtol=1e-2, name="GEMM+AllGather Multicast")
        print(f"[Rank {rank}] GEMM+AllGather Multicast test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] GEMM+AllGather Multicast test FAILED: {e}")
        raise

    # Cleanup
    torch.cuda.synchronize()
    ctx.finalize()


def test_gemm_allgather_performance(rank, world_size, tp_group, M=1024, N=7168, K=2048,
                                    dtype=torch.bfloat16, warmup=10, iters=100):
    """GEMM + AllGather performance test"""
    print(f"[Rank {rank}] Testing GEMM+AllGather performance (M={M}, N={N}, K={K}, dtype={dtype})...")

    # Initialize SymmMemContext
    SymmMemContext.from_process_group(tp_group)

    # Create context
    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=dtype,
        NUM_COMM_SMS=8,
        enable_multicast=True,
    )

    # Generate test data
    torch.manual_seed(rank + 42)
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(N, K, dtype=dtype, device="cuda")

    # GEMM configuration
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - ctx.NUM_COMM_SMS
    gemm_config = triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64,
         'GROUP_SIZE_M': 8, 'NUM_GEMM_SMS': NUM_GEMM_SMS}
    )

    # PyTorch symm_mem GEMM+AllGather performance
    def symm_mem_gemm_ag():
        return gemm_allgather_op_pytorch(ctx, a, b, gemm_config, copy_to_local=False, USE_MULTIMEM_ST=True)

    output_symm_mem, duration_symm_mem = perf_func(symm_mem_gemm_ag, warmup, iters)

    # PyTorch reference (GEMM + NCCL AllGather)
    def torch_gemm_ag():
        local_gemm = torch.matmul(a, b.T)
        return torch_allgather_reference(local_gemm, tp_group)

    output_torch, duration_torch = perf_func(torch_gemm_ag, warmup, iters)

    # Verify correctness
    assert_allclose(output_symm_mem, output_torch, atol=1e-2, rtol=1e-2, name="GEMM+AllGather performance")

    # Calculate performance metrics
    # GEMM FLOPs: 2 * M * N * K
    gemm_flops = 2 * M * N * K
    gemm_tflops = gemm_flops / duration_symm_mem / 1e12 * 1000  # TFLOPS

    # AllGather bandwidth: M * N * dtype_size * world_size
    ag_bytes = M * N * a.element_size() * world_size
    ag_bw = ag_bytes / duration_symm_mem / 1e9 * 1000  # GB/s

    print(f"[Rank {rank}] GEMM+AllGather Performance results:")
    print(f"  PyTorch symm_mem: {duration_symm_mem:.3f} ms")
    print(f"  PyTorch NCCL:     {duration_torch:.3f} ms")
    print(f"  GEMM Performance: {gemm_tflops:.2f} TFLOPS")
    print(f"  AllGather Bandwidth: {ag_bw:.2f} GB/s")
    print(f"  Speedup: {duration_torch / duration_symm_mem:.2f}x")

    # Cleanup
    torch.cuda.synchronize()
    ctx.finalize()


def test_deepgemm_allgather_correctness(rank, world_size, tp_group,
                                        M=1024, N=7168, K=2048):
    """
    Test DeepGEMM (FP8) + AllGather fusion correctness.

    Verify that deepgemm_allgather_op_pytorch output matches
    bf16 matmul + NCCL AllGather reference result within bf16 tolerance.
    """
    print(f"[Rank {rank}] Testing DeepGEMM+AllGather correctness "
          f"(M={M}, N={N}, K={K}, dtype=fp8->bf16)...")

    if not DEEP_GEMM_AVAILABLE:
        print(f"[Rank {rank}] deep_gemm not available, skipping DeepGEMM test.")
        return

    SymmMemContext.from_process_group(tp_group)

    # Context uses bfloat16 (deepgemm outputs bf16)
    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=torch.bfloat16,
        NUM_COMM_SMS=8,
        enable_multicast=True,
    )

    # Construct FP8 inputs
    a_tuple, b_tuple, a_bf16, b_bf16 = make_fp8_inputs(M, K, N, seed=rank + 42)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - ctx.NUM_COMM_SMS
    blockconfig = get_best_config(M, N, K, NUM_GEMM_SMS)
    deep_gemm.set_num_sms(NUM_GEMM_SMS)

    BLOCK_SIZE_M = blockconfig.block_m
    BLOCK_SIZE_N = blockconfig.block_n
    BLOCK_SIZE_K = blockconfig.block_k
    kNumMulticast = blockconfig.multicast_config.num_multicast
    kIsMulticastOnA = blockconfig.multicast_config.is_multicast_on_a
    GROUP_SIZE_M = 1
    gemm_config = triton.Config(
        {
            'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K, "GROUP_SIZE_M":
            GROUP_SIZE_M, "NUM_GEMM_SMS": NUM_GEMM_SMS
        }, num_stages=2, num_warps=8)

    # Run DeepGEMM + AllGather
    output = deepgemm_allgather_op_pytorch(
        ctx, a_tuple, b_tuple, gemm_config,
        copy_to_local=False, USE_MULTIMEM_ST=True,
    )

    # Reference: bf16 matmul + NCCL AllGather
    ref_output = deepgemm_reference(a_bf16, b_bf16, tp_group)

    # Verify (fp8 quantization error, relax to bf16 overlap tolerance)
    try:
        assert_allclose(output, ref_output, atol=6e-2, rtol=6e-2,
                        name="DeepGEMM+AllGather")
        print(f"[Rank {rank}] DeepGEMM+AllGather correctness test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] DeepGEMM+AllGather correctness test FAILED: {e}")
        print(f"  output shape={output.shape}, ref shape={ref_output.shape}")
        print(f"  output max={output.abs().max():.4f}, ref max={ref_output.abs().max():.4f}")
        raise

    torch.cuda.synchronize()
    ctx.finalize()


def test_deepgemm_allgather_multicast(rank, world_size, tp_group,
                                      M=1024, N=7168, K=2048):
    """Test DeepGEMM (FP8) + AllGather Multicast mode"""
    print(f"[Rank {rank}] Testing DeepGEMM+AllGather Multicast "
          f"(M={M}, N={N}, K={K})...")

    if not DEEP_GEMM_AVAILABLE:
        print(f"[Rank {rank}] deep_gemm not available, skipping.")
        return

    SymmMemContext.from_process_group(tp_group)

    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=torch.bfloat16,
        NUM_COMM_SMS=8,
        enable_multicast=True,
    )

    if ctx.mc_ag_out_buf is None:
        print(f"[Rank {rank}] Multicast not supported, skipping.")
        torch.cuda.synchronize()
        ctx.finalize()
        return

    a_tuple, b_tuple, a_bf16, b_bf16 = make_fp8_inputs(M, K, N, seed=rank + 42)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - ctx.NUM_COMM_SMS
    blockconfig = get_best_config(M, N, K, NUM_GEMM_SMS)
    deep_gemm.set_num_sms(NUM_GEMM_SMS)

    BLOCK_SIZE_M = blockconfig.block_m
    BLOCK_SIZE_N = blockconfig.block_n
    BLOCK_SIZE_K = blockconfig.block_k
    kNumMulticast = blockconfig.multicast_config.num_multicast
    kIsMulticastOnA = blockconfig.multicast_config.is_multicast_on_a
    GROUP_SIZE_M = 1
    gemm_config = triton.Config(
        {
            'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K, "GROUP_SIZE_M":
            GROUP_SIZE_M, "NUM_GEMM_SMS": NUM_GEMM_SMS
        }, num_stages=2, num_warps=8)

    output = deepgemm_allgather_op_pytorch(
        ctx, a_tuple, b_tuple, gemm_config,
        copy_to_local=False, USE_MULTIMEM_ST=True,
    )

    ref_output = deepgemm_reference(a_bf16, b_bf16, tp_group)

    try:
        assert_allclose(output, ref_output, atol=6e-2, rtol=6e-2,
                        name="DeepGEMM+AllGather Multicast")
        print(f"[Rank {rank}] DeepGEMM+AllGather Multicast test passed!")
    except AssertionError as e:
        print(f"[Rank {rank}] DeepGEMM+AllGather Multicast test FAILED: {e}")
        raise

    torch.cuda.synchronize()
    ctx.finalize()


def test_deepgemm_allgather_performance(rank, world_size, tp_group,
                                        M=1024, N=7168, K=2048,
                                        warmup=10, iters=100):
    """
    DeepGEMM + AllGather performance test.

    Compare two implementations:
      1. DeepGEMM FP8 + PyTorch symm_mem AllGather (overlap)
      2. DeepGEMM FP8 + NCCL AllGather (no overlap, sequential)
    """
    print(f"[Rank {rank}] Testing DeepGEMM+AllGather performance (M={M}, N={N}, K={K})...")

    if not DEEP_GEMM_AVAILABLE:
        print(f"[Rank {rank}] deep_gemm not available, skipping performance test.")
        return

    SymmMemContext.from_process_group(tp_group)

    ag_stream = torch.cuda.Stream()
    ctx = create_gemm_ag_context_pytorch(
        ag_stream=ag_stream,
        rank=rank,
        world_size=world_size,
        local_world_size=world_size,
        max_M=M,
        N=N,
        dtype=torch.bfloat16,
        NUM_COMM_SMS=8,
        enable_multicast=True,
    )

    a_tuple, b_tuple, a_bf16, b_bf16 = make_fp8_inputs(M, K, N, seed=rank + 42)

    # 1. DeepGEMM + NCCL AllGather (no overlap, sequential)
    #    Test this FIRST with full SM count for fair baseline
    deepgemm_novlp_out = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    nccl_ag_out = torch.empty((M * world_size, N), dtype=torch.bfloat16, device="cuda")

    def run_deepgemm_ag_seq():
        # GEMM first
        deep_gemm.fp8_gemm_nt(
            a_tuple, b_tuple, deepgemm_novlp_out,
            c=None,
            disable_ue8m0_cast=True,
            recipe=None,
        )
        # Then NCCL AllGather (no overlap)
        dist.all_gather_into_tensor(nccl_ag_out, deepgemm_novlp_out, group=tp_group)
        return nccl_ag_out

    _, dur_deepgemm_seq = perf_func(run_deepgemm_ag_seq, warmup, iters)

    # 2. DeepGEMM + PyTorch symm_mem AllGather (overlap)
    #    Set reduced SM count for overlap (reserve SMs for communication)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_GEMM_SMS = NUM_SMS - ctx.NUM_COMM_SMS
    blockconfig = get_best_config(M, N, K, NUM_GEMM_SMS)
    deep_gemm.set_num_sms(NUM_GEMM_SMS)

    BLOCK_SIZE_M = blockconfig.block_m
    BLOCK_SIZE_N = blockconfig.block_n
    BLOCK_SIZE_K = blockconfig.block_k
    kNumMulticast = blockconfig.multicast_config.num_multicast
    kIsMulticastOnA = blockconfig.multicast_config.is_multicast_on_a
    GROUP_SIZE_M = 1
    gemm_config = triton.Config(
        {
            'BLOCK_SIZE_M': BLOCK_SIZE_M, 'BLOCK_SIZE_N': BLOCK_SIZE_N, "BLOCK_SIZE_K": BLOCK_SIZE_K, "GROUP_SIZE_M":
            GROUP_SIZE_M, "NUM_GEMM_SMS": NUM_GEMM_SMS
        }, num_stages=2, num_warps=8)

    def run_deepgemm_ag():
        return deepgemm_allgather_op_pytorch(
            ctx, a_tuple, b_tuple, gemm_config,
            copy_to_local=False, USE_MULTIMEM_ST=True,
        )

    out_deepgemm, dur_deepgemm = perf_func(run_deepgemm_ag, warmup, iters)

    # Calculate performance metrics
    gemm_flops = 2 * M * N * K
    ag_bytes = M * N * 2 * world_size  # bfloat16 = 2 bytes

    gemm_tflops_overlap = gemm_flops / dur_deepgemm / 1e12 * 1000
    gemm_tflops_novlp = gemm_flops / dur_deepgemm_seq / 1e12 * 1000
    ag_bw_overlap = ag_bytes / dur_deepgemm / 1e9 * 1000
    ag_bw_novlp = ag_bytes / dur_deepgemm_seq / 1e9 * 1000

    print(f"[Rank {rank}] DeepGEMM+AllGather Performance results:")
    print(f"  Overlap:    {dur_deepgemm:.3f} ms, {gemm_tflops_overlap:.2f} TFLOPS, {ag_bw_overlap:.2f} GB/s")
    print(f"  No-overlap: {dur_deepgemm_seq:.3f} ms, {gemm_tflops_novlp:.2f} TFLOPS, {ag_bw_novlp:.2f} GB/s")
    print(f"  Overlap speedup: {dur_deepgemm_seq / dur_deepgemm:.2f}x")

    torch.cuda.synchronize()
    ctx.finalize()


def test_barrier_correctness(rank, world_size, tp_group):
    """Test barrier mechanism correctness"""
    print(f"[Rank {rank}] Testing barrier mechanism...")

    # Initialize SymmMemContext
    ctx = SymmMemContext.from_process_group(tp_group)

    # Create test tensor
    tensor = ctx.create_tensor([1024], torch.int32)
    tensor.fill_(rank)

    # Check before barrier
    torch.cuda.synchronize()
    print(f"[Rank {rank}] Before barrier: tensor[0] = {tensor[0].item()}")

    # Barrier
    ctx.barrier_all()

    # Check after barrier
    torch.cuda.synchronize()
    print(f"[Rank {rank}] After barrier: tensor[0] = {tensor[0].item()}")

    # Cleanup
    torch.cuda.synchronize()
    ctx.free_tensor(tensor)
    print(f"[Rank {rank}] Barrier test passed!")


def test_peer_tensor_access(rank, world_size, tp_group):
    """Test peer tensor access via PyTorch symm_mem API"""
    print(f"[Rank {rank}] Testing peer tensor access (PyTorch symm_mem)...")

    # Initialize SymmMemContext
    ctx = SymmMemContext.from_process_group(tp_group)

    # Create test tensor
    tensor = ctx.create_tensor([1024], torch.float32)
    tensor.fill_(float(rank))

    torch.cuda.synchronize()

    # Get all peer buffer pointers (PyTorch symm_mem exposes raw int64 ptrs)
    peer_ptrs = ctx.get_peer_buffer_ptrs(tensor)

    # Verify addresses
    for i, ptr in enumerate(peer_ptrs):
        print(f"[Rank {rank}] Peer {i} buffer ptr: {ptr}")

    # Write different values to each rank's tensor
    tensor.fill_(float(rank * 100))

    ctx.barrier_all()

    # Verify own value
    val = tensor[0].item()
    expected = float(rank * 100)
    print(f"[Rank {rank}] Own tensor value: {val}, expected: {expected}")
    assert abs(val - expected) < 1e-5, f"Own value mismatch: {val} vs {expected}"

    # Cross-rank verification via NCCL all_gather
    # (PyTorch symm_mem does not support host-side peer reads like nvshmem4py)
    all_values = [torch.zeros(1, dtype=torch.float32, device="cuda") for _ in range(world_size)]
    val_tensor = torch.tensor([tensor[0].item()], dtype=torch.float32, device="cuda")
    dist.all_gather(all_values, val_tensor, group=tp_group)

    for i, v in enumerate(all_values):
        val = v.item()
        expected = float(i * 100)
        print(f"[Rank {rank}] Rank {i} value (via all_gather): {val}, expected: {expected}")
        assert abs(val - expected) < 1e-5, f"Rank {i} value mismatch: {val} vs {expected}"

    # Cleanup
    torch.cuda.synchronize()
    ctx.free_tensor(tensor)
    print(f"[Rank {rank}] Peer tensor access test passed!")


def test_different_block_sizes(rank, world_size, tp_group, M=1024, N=7168):
    """Test different block size configurations"""
    print(f"[Rank {rank}] Testing different block sizes...")

    # Initialize SymmMemContext
    SymmMemContext.from_process_group(tp_group)

    # Generate test data
    torch.manual_seed(rank + 42)
    dtype = torch.bfloat16
    input_tensor = torch.randn(M, N, dtype=dtype, device="cuda")

    # Test different block size configurations
    block_configs = [
        (16, 64, 16),
        (32, 64, 16),
        (64, 64, 16),
        (16, 128, 16),
        (32, 128, 16),
        (64, 128, 16),
    ]

    ref_output = torch_allgather_reference(input_tensor, tp_group)

    for BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_COMM_SMS in block_configs:
        print(f"[Rank {rank}] Testing BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, NUM_COMM_SMS={NUM_COMM_SMS}")

        try:
            # Create context
            ag_stream = torch.cuda.Stream()
            ctx = create_gemm_ag_context_pytorch(
                ag_stream=ag_stream,
                rank=rank,
                world_size=world_size,
                local_world_size=world_size,
                max_M=M,
                N=N,
                dtype=dtype,
                NUM_COMM_SMS=NUM_COMM_SMS,
                enable_multicast=True,
            )

            # Run AllGather
            output = allgather_op_pytorch(
                ctx,
                input_tensor,
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                NUM_COMM_SMS=NUM_COMM_SMS,
                USE_MULTIMEM_ST=True,
                copy_to_local=False,
            )

            # Verify
            assert_allclose(output, ref_output, atol=6e-2, rtol=6e-2,
                          name=f"AllGather(BLOCK_M={BLOCK_SIZE_M}, BLOCK_N={BLOCK_SIZE_N})")

            print(f"[Rank {rank}] ✓ Passed")

            # Cleanup
            torch.cuda.synchronize()
            ctx.finalize()

        except Exception as e:
            print(f"[Rank {rank}] ✗ Failed: {e}")
            raise

    print(f"[Rank {rank}] All block size tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Test GEMM AllGather with PyTorch symmetric_memory")
    parser.add_argument("--test", type=str, default="all",
                       choices=["init", "correctness", "multicast", "perf", "multi",
                               "gemm", "gemm_ag", "gemm_ag_mc", "gemm_ag_perf",
                                "deepgemm_ag", "deepgemm_ag_mc", "deepgemm_ag_perf",
                               "barrier", "peer", "blocks",
                               "all"])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=7168)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    # Initialize distributed
    rank, local_rank, world_size, local_world_size, tp_group = initialize_distributed()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"[Rank {rank}] Starting tests: test={args.test}, M={args.M}, N={args.N}, K={args.K}, dtype={dtype}")

    if args.test == "init" or args.test == "all":
        test_symm_mem_init(rank, world_size, tp_group)

    if args.test == "correctness" or args.test == "all":
        test_allgather_correctness(rank, world_size, tp_group, M=args.M, N=args.N, dtype=dtype)

    if args.test == "multicast" or args.test == "all":
        test_allgather_multicast(rank, world_size, tp_group, M=args.M, N=args.N, dtype=dtype)

    if args.test == "perf" or args.test == "all":
        test_performance(rank, world_size, tp_group, M=args.M, N=args.N, dtype=dtype,
                       warmup=args.warmup, iters=args.iters)

    if args.test == "multi" or args.test == "all":
        test_multi_scale(rank, world_size, tp_group)

    # GEMM related tests
    if args.test == "gemm" or args.test == "all":
        test_gemm_correctness(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K, dtype=dtype)

    if args.test == "gemm_ag" or args.test == "all":
        test_gemm_allgather_correctness(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K, dtype=dtype)

    if args.test == "gemm_ag_mc" or args.test == "all":
        test_gemm_allgather_multicast(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K, dtype=dtype)

    if args.test == "gemm_ag_perf" or args.test == "all":
        test_gemm_allgather_performance(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K,
                                       dtype=dtype, warmup=args.warmup, iters=args.iters)

    # DeepGEMM (FP8) related tests
    if args.test == "deepgemm_ag" or args.test == "all":
        test_deepgemm_allgather_correctness(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K)

    if args.test == "deepgemm_ag_mc" or args.test == "all":
        test_deepgemm_allgather_multicast(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K)

    if args.test == "deepgemm_ag_perf" or args.test == "all":
        test_deepgemm_allgather_performance(rank, world_size, tp_group, M=args.M, N=args.N, K=args.K,
                                            warmup=args.warmup, iters=args.iters)

    # Basic functionality tests
    if args.test == "barrier" or args.test == "all":
        test_barrier_correctness(rank, world_size, tp_group)

    if args.test == "peer" or args.test == "all":
        test_peer_tensor_access(rank, world_size, tp_group)

    if args.test == "blocks" or args.test == "all":
        test_different_block_sizes(rank, world_size, tp_group, M=args.M, N=args.N)

    print(f"[Rank {rank}] All tests passed!")

    # Cleanup
    ctx = SymmMemContext.get_instance()
    if ctx is not None:
        ctx.finalize()

    dist.destroy_process_group()
    os._exit(0)


if __name__ == "__main__":
    main()