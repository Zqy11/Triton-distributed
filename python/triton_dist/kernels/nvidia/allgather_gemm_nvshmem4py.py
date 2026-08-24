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
AllGather -> GEMM Overlap using nvshmem4py (AG first, GEMM follows with tile-level signaling)

This module implements the AllGather + GEMM overlap pattern where:
1. cp_engine_full_mesh_pull_ag: host-side AllGather via Copy Engine
   (full-mesh pull) enqueued on ag_stream; uses PtoP data + signal writes so
   AG runs on the CE concurrently with the GEMM without consuming SMs, and
   signals per-rank-shard when each peer's data is ready.
2. consumer_bf16_a_block_fp8_matmul: GEMM kernel that polls ag_signal[src_rank]
   with system-scope acquire (ld_sys) before computing each tile, reads bf16 A
   from symmetric memory, on-the-fly quantizes to fp8, and performs block-wise
   fp8 matmul.

Also includes:
- Per-token-group FP8 quantization kernels (from SGLang)
- Block-wise FP8 matmul kernels (w8a8 and bf16-a fused quantize variants)
"""

import torch
import dataclasses
from typing import List, Optional, Tuple
import triton
import triton.language as tl

# Import shared NVSHMEMContext and PTX helpers from gemm_allgather_nvshmem4py
import importlib.util
import os

import cuda.bindings.driver as cuda

_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gemm_allgather_nvshmem4py.py"
)
_spec = importlib.util.spec_from_file_location("gemm_allgather_nvshmem4py", _module_path)
_gemm_ag_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gemm_ag_module)

NVSHMEMContext = _gemm_ag_module.NVSHMEMContext
ld_sys = _gemm_ag_module.ld_sys
st_sys = _gemm_ag_module.st_sys
__syncthreads = _gemm_ag_module.__syncthreads
tid = _gemm_ag_module.tid
load_v4_b32 = _gemm_ag_module.load_v4_b32
multimem_st_v4 = _gemm_ag_module.multimem_st_v4


def stream_write_value32(tensor: torch.Tensor, offset: int, value: int):
    """Write a 32-bit value to tensor[offset] via cuStreamWriteValue32.
    Issues a system-scope fence before the write, pairing with ld.sys in consumer kernels."""
    (err,) = cuda.cuStreamWriteValue32(
        torch.cuda.current_stream().cuda_stream,
        tensor.data_ptr() + offset * tensor.element_size(),
        value,
        cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
    )
    assert err == cuda.CUresult.CUDA_SUCCESS, f"cuStreamWriteValue32 failed: {err}"


# ============================================================================
# AllGather -> GEMM Context (nvshmem4py version)
# ============================================================================

@dataclasses.dataclass
class AllGatherGemmContextNVSHMEM:
    """
    Context for AllGather -> GEMM overlap operation.
    Contains symmetric memory buffers needed for the AG->GEMM pipeline.

    - symm_input_buf: each rank's A shard [M, K] in symmetric memory
    - symm_ag_a_buf: gathered A [M*world_size, K] in symmetric memory (multicast)
    - ag_signal_buf: signal buffer for per-rank-shard notification [world_size] int32
    """
    rank: int
    num_ranks: int
    NUM_COMM_SMS: int
    NUM_GEMM_SMS: int
    ag_stream: torch.cuda.Stream

    symm_input_buf: torch.Tensor           # [world_size, M, K] per-rank A shards
    symm_ag_a_buf: torch.Tensor            # [M * world_size, K] gathered A
    ag_signal_buf: torch.Tensor            # [world_size] int32, per-rank-shard signal

    # nvshmem context
    nvshmem_ctx: NVSHMEMContext = None

    # Multicast buffer (optional)
    mc_ag_a_buf: Optional[torch.Tensor] = None

    # Peer tensor pointers for kernel
    peer_signal_ptrs: Optional[torch.Tensor] = None

    # Peer symm_input tensor list (for CP-engine full_mesh_pull AG)
    # peer_symm_input_bufs[i] is a torch.Tensor view of rank i's symm_input_buf
    peer_symm_input_bufs: Optional[List[torch.Tensor]] = None

    def finalize(self):
        """Free NVSHMEM resources"""
        if self.nvshmem_ctx:
            self.nvshmem_ctx.free_tensor(self.symm_input_buf)
            self.nvshmem_ctx.free_tensor(self.symm_ag_a_buf)
            self.nvshmem_ctx.free_tensor(self.ag_signal_buf)
            # Note: mc_ag_a_buf is a multicast tensor, cannot be freed with free_tensor

    def get_input_buf(self, M, K):
        """Get this rank's A shard from symmetric memory"""
        offset = M * K * self.rank
        return self.symm_input_buf.reshape(-1)[offset : offset + M * K].reshape(M, K)

    def reset_all_barrier_buf(self):
        self.ag_signal_buf.zero_()


def create_allgather_gemm_context_nvshmem(
    ag_stream: torch.cuda.Stream,
    rank: int,
    world_size: int,
    max_M: int,
    K: int,
    NUM_COMM_SMS: int = 0,
    enable_multicast: bool = False,
):
    """
    Create context for AllGather -> GEMM overlap operation.

    Args:
        ag_stream: CUDA stream for AllGather
        rank: Current rank
        world_size: Total number of ranks
        max_M: Maximum M dimension per rank (shard size)
        K: K dimension of A
        NUM_COMM_SMS: Number of SMs for communication
        enable_multicast: Whether to enable multicast

    Returns:
        AllGatherGemmContextNVSHMEM instance
    """
    nvshmem_ctx = NVSHMEMContext.get_instance()
    if nvshmem_ctx is None:
        raise RuntimeError("NVSHMEMContext not initialized. Call NVSHMEMContext.from_process_group() first.")

    # Symmetric memory for A shards: [world_size, max_M, K]
    symm_input_buf = nvshmem_ctx.create_tensor((world_size, max_M, K), torch.bfloat16)
    # Symmetric memory for gathered A: [max_M * world_size, K]
    symm_ag_a_buf = nvshmem_ctx.create_tensor((max_M * world_size, K), torch.bfloat16)
    # Signal buffer: [world_size] - one signal per rank-shard (per-rank granularity)
    # Each entry signals that the corresponding rank's [M_local, K] shard has been
    # pulled into symm_ag_a_buf[i*M_local:(i+1)*M_local].
    ag_signal_buf = nvshmem_ctx.create_tensor((world_size,), torch.int32)

    # Initialize
    symm_input_buf.zero_()
    symm_ag_a_buf.zero_()
    ag_signal_buf.zero_()

    nvshmem_ctx.barrier_all()

    # Get multicast buffer
    mc_ag_a_buf = None
    if enable_multicast:
        try:
            mc_ag_a_buf = nvshmem_ctx.get_multicast_tensor(symm_ag_a_buf)
        except Exception as e:
            print(f"Warning: Multicast not supported: {e}")

    # Get peer tensor pointer arrays
    peer_signal_ptrs = torch.tensor(
        [nvshmem_ctx.get_peer_tensor(ag_signal_buf, r).data_ptr() for r in range(world_size)],
        dtype=torch.int64,
        device=torch.cuda.current_device()
    )

    # Cache peer symm_input tensor views for CP-engine full_mesh_pull AG.
    # peer_symm_input_bufs[i] is a torch.Tensor view of rank i's symm_input_buf
    # with shape [world_size, max_M, K]. We pull rank i's own shard from
    # peer_symm_input_bufs[i][i, :, :] via Copy Engine (D2D), then signal locally.
    peer_symm_input_bufs = [
        nvshmem_ctx.get_peer_tensor(symm_input_buf, r) for r in range(world_size)
    ]

    # Calculate NUM_GEMM_SMS: total SMs minus communication SMs
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = num_sms - NUM_COMM_SMS

    ctx = AllGatherGemmContextNVSHMEM(
        rank=rank,
        num_ranks=world_size,
        NUM_COMM_SMS=NUM_COMM_SMS,
        NUM_GEMM_SMS=num_gemm_sms,
        ag_stream=ag_stream,
        symm_input_buf=symm_input_buf,
        symm_ag_a_buf=symm_ag_a_buf,
        ag_signal_buf=ag_signal_buf,
        nvshmem_ctx=nvshmem_ctx,
        mc_ag_a_buf=mc_ag_a_buf,
        peer_signal_ptrs=peer_signal_ptrs,
        peer_symm_input_bufs=peer_symm_input_bufs,
    )

    return ctx


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
# AllGather + Notify Kernel (multimem_st only)
# ============================================================================

@triton.jit
def all_gather_notify_kernel_nvshmem(
    symm_input_ptr,            # this rank's A shard in symmetric memory [M, K]
    symm_ag_out_ptr,           # gathered A output [M*world_size, K] in symmetric memory
    ag_signal_ptr,             # signal buffer [num_pid_m] int32 (one per M-row tile)
    peer_signal_ptrs,          # GPU-side pointer array for peer ag_signal buffers
    mc_ag_out_ptr,             # Multicast pointer for symm_ag_out
    M,                         # local shard M (each rank's rows)
    K,                         # K dimension of A
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    num_pid_m: tl.constexpr,   # total M-row tiles in gathered A = cdiv(M*world_size, BLOCK_SIZE_M)
):
    """
    AllGather + Notify kernel (multimem_st only).

    Copies each rank's symm_input [M, K] to symm_ag_out [M*world_size, K] via multimem_st.
    After each M-row-tile's full K range is written, signals ag_signal[global_pid_m] = 1
    on ALL ranks (via peer_signal_ptrs) so the consumer GEMM on every rank can proceed.

    Signal layout: ag_signal[global_pid_m] = 1
    """
    pid = tl.program_id(0)
    thread_idx = tid(0)
    block_dim = 32 * 32  # num_warps * 32

    # Number of M-row-tiles in this rank's local shard
    num_pid_m_local = tl.cdiv(M, BLOCK_SIZE_M)
    # Number of K-tiles (sub-tiles within each M-row)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)

    ELEM_BYTES: tl.constexpr = 2  # bfloat16
    VEC_SIZE: tl.constexpr = 128 // 16  # 8 bf16 elements per 128-bit vector
    MULTIMEM_SUFFIX: tl.constexpr = "bf16x2"

    for local_tile_id in range(pid, num_pid_m_local, NUM_COMM_SMS):
        global_pid_m = rank * num_pid_m_local + local_tile_id

        # Write all K-tiles for this M-row-tile via multimem_st
        for pid_k in range(num_pid_k):
            tile_m = tl.minimum(M - local_tile_id * BLOCK_SIZE_M, BLOCK_SIZE_M)
            tile_k = tl.minimum(K - pid_k * BLOCK_SIZE_K, BLOCK_SIZE_K)
            VEC_PER_ROW = tile_k // VEC_SIZE
            cur_tile_nelem = tile_m * tile_k

            for idx in range(thread_idx, cur_tile_nelem // VEC_SIZE, block_dim):
                row_id = idx // VEC_PER_ROW
                col_id = idx % VEC_PER_ROW
                # Element offset within this rank's shard
                offset = (row_id + local_tile_id * BLOCK_SIZE_M) * K + col_id * VEC_SIZE + pid_k * BLOCK_SIZE_K
                byte_offset = offset * ELEM_BYTES
                src_byte_addr = symm_input_ptr.to(tl.int64) + byte_offset
                val0, val1, val2, val3 = load_v4_b32(src_byte_addr.to(tl.pointer_type(tl.int32)))
                # Multicast to all ranks: write to symm_ag_out[rank*M*K + offset]
                mc_byte_offset = (rank * M * K + offset) * ELEM_BYTES
                multimem_st_v4(mc_ag_out_ptr + mc_byte_offset, val0, val1, val2, val3, MULTIMEM_SUFFIX)

        # After all K-tiles for this M-row are written, signal ALL ranks
        # Each rank's GEMM needs to know this M-row tile's data is ready
        if thread_idx == 0:
            for peer_rank in range(world_size):
                peer_signal_ptr = tl.load(peer_signal_ptrs + peer_rank).to(tl.pointer_type(tl.int32))
                st_sys(peer_signal_ptr + global_pid_m, 1)


# ============================================================================
# CP-Engine Full-Mesh-Pull AllGather (host-side)
# ============================================================================

def cp_engine_full_mesh_pull_ag(
    rank: int,
    world_size: int,
    M_local: int,
    K: int,
    symm_input: torch.Tensor,                       # this rank's [M_local, K] shard view into symm_input_buf
    symm_ag_a: torch.Tensor,                        # local [M_local*world_size, K] gathered buffer
    peer_symm_input_bufs: List[torch.Tensor],       # peer views: peer_symm_input_bufs[i] is rank i's [world_size, max_M, K] buffer
    ag_signal: torch.Tensor,                        # local ag_signal [world_size] int32
    ag_stream: torch.cuda.Stream,
):
    """
    AllGather via Copy Engine (full-mesh pull) — counterpart to
    `cp_engine_producer_all_gather_full_mesh_pull` in allgather.py.

    For each peer rank src_rank in rotated order (rank, rank+1, ..., rank+world_size-1):
      - If src_rank == rank: data already local; signal via cuStreamWriteValue32.
      - Else: PtoP data copy from peer_symm_input_bufs[src_rank][src_rank, :, :]
        into symm_ag_a[src_rank*M_local : (src_rank+1)*M_local, :], then signal
        via cuStreamWriteValue32.

    The data layout in symm_ag_a is COMPACT (no padding): each rank's shard
    occupies exactly M_local contiguous rows.  The consumer GEMM kernel is
    responsible for computing the correct row offset via
    src_rank * M_local + tile_in_rank * BLOCK_SIZE_M.

    Signal writes use cuStreamWriteValue32 (via stream_write_value32), which issues
    a system-level memory fence before the write. This ensures all prior data
    writes (PtoP copy_ and local copy_) are visible to other GPUs before the
    signal value becomes visible. The consumer GEMM kernel polls with ld.sys
    (system-scope acquire), which correctly pairs with this fence.

    NOTE: caller must wrap this call in `with torch.cuda.stream(ag_stream):`
    so that all enqueued copy_ ops and signal writes land on ag_stream.
    """
    # Self-shard: data is already local, copy and signal via cuStreamWriteValue32.
    local_dst = symm_ag_a[rank * M_local : (rank + 1) * M_local, :]
    local_dst.copy_(symm_input)
    stream_write_value32(ag_signal, rank, 1)

    # Pull remote shards in rotated order: (rank+1, rank+2, ..., rank-1) % world_size
    # The rotated order spreads the hot-spot: not all ranks pull from rank 0 first.
    for offset in range(1, world_size):
        src_rank = (rank + offset) % world_size
        # Remote source: rank `src_rank`'s own shard sits at peer's symm_input_buf[src_rank, :, :]
        remote_src = peer_symm_input_bufs[src_rank][src_rank, :M_local, :]
        local_dst = symm_ag_a[src_rank * M_local : (src_rank + 1) * M_local, :]
        local_dst.copy_(remote_src)  # PtoP CE data copy
        # Signal: cuStreamWriteValue32 with system-scope fence ensures data
        # visibility before the signal arrives on remote GPUs.
        stream_write_value32(ag_signal, src_rank, 1)


# ============================================================================
# Consumer GEMM Kernel (non-persistent, polls AG signal, then computes bf16-A block-FP8 matmul)
# ============================================================================

@triton.jit
def consumer_bf16_a_block_fp8_matmul(
    # Pointers
    A_ptr,           # symm_ag_out: bf16 [M, K] (gathered A in symmetric memory, compact layout)
    B_ptr,           # fp8 weight [N, K], column-major (transposed)
    C_ptr,           # output [M, N]
    Bs_ptr,          # per-block scale for B [N // group_n, K // group_k]
    ag_signal_ptr,   # signal from AG kernel [world_size] int32 (per-rank-shard)
    # Dimensions
    M,               # M_local * world_size (total rows)
    N,
    K,
    M_local,         # real rows per rank — for correct row offset computation
    M_local_tiles,   # = cdiv(M_local, BLOCK_SIZE_M); pid_m -> src_rank mapping
    # Block quantization
    group_n,
    group_k,
    # Strides
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
    needs_k_masking: tl.constexpr,   # True when K % BLOCK_SIZE_K != 0
    needs_m_masking: tl.constexpr,   # True when M_local % BLOCK_SIZE_M != 0
    rank: tl.constexpr,              # local rank, for rank-aware tile rotation
    world_size: tl.constexpr,
):
    """
    Non-persistent consumer GEMM kernel: polls AG signal, then computes
    bf16-A block-FP8 matmul.

    One CTA per output tile (grid = num_pid_m * num_pid_n). Mirrors
    `_bf16_a_block_fp8_matmul` exactly except for the AG signal poll
    inserted before the K-loop, so the scheduler can freely interleave
    GEMM CTAs with the concurrent AG kernel for compute-comm overlap.

    Signal granularity is per-rank-shard: ag_signal[src_rank] == 1 means
    the entire [M_local, K] shard belonging to rank `src_rank` has been
    pulled into symm_ag_a.

    Rank-aware tile rotation: tiles are remapped so the first
    M_local_tiles tiles in pid_m space target THIS rank's self-shard
    (signaled first), then peers in rotated order. CTAs launch roughly
    in pid order, so this matches the cp_engine_full_mesh_pull_ag
    signal-arrival order, minimizing spin-wait at GEMM start.

    Compact layout support: the AG buffer uses a compact layout where
    rank i's M_local rows are at A[i*M_local : (i+1)*M_local, :].
    When M_local is not a multiple of BLOCK_SIZE_M, the tile grid has
    M_local_tiles = cdiv(M_local, BLOCK_SIZE_M) tiles per rank, and
    the last tile in each rank's shard is a "tail tile" that may extend
    beyond the rank's real data.  Row offsets for both A and C are
    computed as src_rank * M_local + tile_in_rank * BLOCK_SIZE_M, and
    A loads / C stores are masked for rows >= M.
    """
    fp8_max = 448.0

    pid = tl.program_id(axis=0)
    # num_pid_m = M_local_tiles * world_size (may exceed cdiv(M, BLOCK_SIZE_M)
    # when M_local is not a multiple of BLOCK_SIZE_M)
    num_pid_m = M_local_tiles * world_size
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Rank-aware tile rotation: remap pid_m so that the first
    # M_local_tiles rows correspond to THIS rank's self-shard
    # (which is signaled first by cp_engine_full_mesh_pull_ag),
    # then peers in rotated order (rank+1, rank+2, ..., rank-1).
    # This aligns CTA scheduling order with AG signal arrival order.
    logical_src_rank = min(pid_m // M_local_tiles, world_size - 1)
    tile_in_rank = pid_m - logical_src_rank * M_local_tiles
    src_rank = (rank + logical_src_rank) % world_size

    # Poll AG signal at per-rank-shard granularity.
    if tid(0) == 0:
        while ld_sys(ag_signal_ptr + src_rank) != 1:
            pass
    __syncthreads()

    # Row offsets — use src_rank * M_local (compact layout), NOT pid_m * BLOCK_SIZE_M.
    # This correctly indexes into the compact AG buffer where each rank's shard
    # occupies exactly M_local contiguous rows.
    offs_row = src_rank * M_local + tile_in_rank * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Mask for M-dimension tail tiles.  Two conditions must hold:
    #   1. offs_row < M  (global out-of-bounds for the last rank's tail tile)
    #   2. offs_row < (src_rank + 1) * M_local  (within this rank's shard;
    #      prevents reading into the next rank's shard whose AG signal may
    #      not have arrived yet)
    # Only computed when M_local is not a multiple of BLOCK_SIZE_M.
    if needs_m_masking:
        shard_end = (src_rank + 1) * M_local
        row_mask = (offs_row < M) & (offs_row < shard_end)

    # A pointers (bf16, row-major, from symmetric memory)
    a_ptrs = A_ptr + (offs_row[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B pointers (fp8, column-major)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # B scale pointers
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs_ptr + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load bf16 A tile and fp8 B tile
        if needs_k_masking:
            k_mask = offs_k < K - k * BLOCK_SIZE_K
            if needs_m_masking:
                a_bf16 = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            else:
                a_bf16 = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        else:
            if needs_m_masking:
                a_bf16 = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
            else:
                a_bf16 = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        # On-the-fly quantize A: per-row-per-BLOCK_SIZE_K absmax
        a_absmax = tl.max(tl.abs(a_bf16), axis=1)  # [BLOCK_SIZE_M]
        a_s = tl.maximum(a_absmax, 1e-12) / fp8_max  # avoid div-by-zero
        a_fp8 = tl.clamp(a_bf16 / a_s[:, None], -fp8_max, fp8_max)
        a_fp8 = a_fp8.to(B_ptr.dtype.element_ty)

        # Load B scale
        b_s = tl.load(Bs_ptrs)

        # Accumulate: dot(fp8_A, fp8_B) * a_scale * b_scale
        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a_fp8, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        Bs_ptrs += scale_step_k * stride_Bs_k

    # Epilogue: cast and store
    c = accumulator.to(tl.bfloat16)

    # Output uses the same row offsets as A (compact layout).
    # For tail tiles, restrict the store mask to this rank's shard to avoid
    # clobbering rows that belong to the next rank's tile.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_row[:, None] + stride_cn * offs_cn[None, :]
    if needs_m_masking:
        c_mask = row_mask[:, None] & (offs_cn[None, :] < N)
    else:
        c_mask = (offs_row[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# AllGather -> GEMM Operation (AG first, GEMM follows with tile-level signaling)
# ============================================================================

def allgather_gemm_op_nvshmem(
    ctx: AllGatherGemmContextNVSHMEM,
    a_bf16: torch.Tensor,          # bf16 activation [M, K] (local shard)
    b_fp8: torch.Tensor,           # fp8 weight [N, K], column-major
    b_scale: torch.Tensor,         # per-block scale [N // group_n, K // group_k]
    block_size: list,              # [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
    output_dtype: torch.dtype = torch.bfloat16,
    GROUP_SIZE_M: int = 32,
):
    """
    AllGather -> GEMM overlap operation (AG first, compute follows).

    Overlaps AllGather communication with GEMM computation at per-rank-shard
    granularity:
    1. cp_engine_full_mesh_pull_ag: host-side AllGather via Copy Engine
       (full-mesh pull). For each src_rank in rotated order, issues a PtoP
       data copy of that rank's shard into symm_ag_a, followed by a PtoP
       signal write into ag_signal[src_rank]. All ops are enqueued on
       ag_stream so the CE runs concurrently with the GEMM on current_stream
       and does not consume any SMs.
    2. consumer_bf16_a_block_fp8_matmul: non-persistent GEMM kernel (one CTA per
       output tile) that polls ag_signal[src_rank] with system-scope acquire
       (ld_sys) before computing each tile, reads bf16 A from symmetric memory,
       on-the-fly quantizes to fp8, and performs block-wise fp8 matmul.

    The GEMM kernel is non-persistent: grid = num_pid_m * num_pid_n. CTAs are
    launched freely and the GPU block scheduler interleaves them with the
    concurrent CE-driven AG on ag_stream — tiles whose A shard is not yet ready
    stall on the signal poll, allowing the CE to make progress and enabling true
    computation-communication overlap.

    Args:
        ctx: AllGatherGemmContextNVSHMEM context
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
    M = M_local * ctx.num_ranks  # total rows after AllGather

    assert a_bf16.dtype == torch.bfloat16, f"Expected bf16 A, got {a_bf16.dtype}"
    assert a_bf16.shape[1] == b_fp8.shape[1], "K dimension mismatch"
    assert a_bf16.is_contiguous()

    # When M_local is not a multiple of BLOCK_SIZE_M, the last tile in each
    # rank's shard is a "tail tile" that partially overlaps real data.  The
    # consumer kernel handles this by:
    #   - Using src_rank * M_local (not pid_m * BLOCK_SIZE_M) for both A and C
    #     row addressing, so the compact AG layout is indexed correctly.
    #   - Masking A loads and C stores for rows beyond M.
    M_local_tiles = triton.cdiv(M_local, BLOCK_SIZE_M)

    # Tile counts
    num_pid_m = M_local_tiles * ctx.num_ranks
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)

    # Copy A shard to symmetric memory
    symm_input = ctx.get_input_buf(M_local, K)
    symm_input.copy_(a_bf16)

    symm_ag_a = ctx.symm_ag_a_buf
    ag_signal = ctx.ag_signal_buf
    peer_signal_ptrs = ctx.peer_signal_ptrs
    mc_ag_a = ctx.mc_ag_a_buf

    # Verify signal buffer is large enough (per-rank-shard granularity)
    assert ag_signal.numel() >= ctx.num_ranks, (
        f"Signal buffer too small: need {ctx.num_ranks}, have {ag_signal.numel()}"
    )

    # Output buffer
    c = torch.empty((M, N), dtype=output_dtype, device=a_bf16.device)

    # Barrier sync and reset
    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream

    ctx.nvshmem_ctx.barrier_all(current_stream)

    # Reset signal buffer to 0 (consumer GEMM polls for signal == 1)
    ag_signal.fill_(0)
    ctx.nvshmem_ctx.barrier_all(current_stream)

    # Ensure ag_stream waits for signal reset and barrier to complete
    # before launching AG copies
    ag_stream.wait_stream(current_stream)

    # Step 1: enqueue AG on ag_stream.
    # Data layout is COMPACT: rank i's M_local rows are at
    # symm_ag_a[i*M_local : (i+1)*M_local, :].
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
            ag_stream=ag_stream,
        )

    # Step 2: launch consumer GEMM on current_stream.
    # Grid = num_pid_m * num_pid_n (non-persistent, one CTA per output tile).
    # The kernel computes A/C row offsets as src_rank * M_local + tile_in_rank
    # * BLOCK_SIZE_M, matching the compact AG layout.
    needs_k_masking = bool(K % BLOCK_SIZE_K != 0)
    needs_m_masking = bool(M_local % BLOCK_SIZE_M != 0)
    num_tiles = num_pid_m * num_pid_n

    consumer_bf16_a_block_fp8_matmul[(num_tiles,)](
        symm_ag_a,
        b_fp8,
        c,
        b_scale,
        ag_signal,
        M,
        N,
        K,
        M_local,
        M_local_tiles,
        BLOCK_SIZE_N,  # group_n
        BLOCK_SIZE_K,  # group_k
        symm_ag_a.stride(0),   # stride_am
        symm_ag_a.stride(1),   # stride_ak
        b_fp8.stride(1),       # stride_bk
        b_fp8.stride(0),       # stride_bn
        c.stride(0),           # stride_cm
        c.stride(1),           # stride_cn
        b_scale.stride(1),     # stride_Bs_k
        b_scale.stride(0),     # stride_Bs_n
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        needs_k_masking=needs_k_masking,
        needs_m_masking=needs_m_masking,
        rank=ctx.rank,
        world_size=ctx.num_ranks,
        num_warps=4,
        num_stages=3,
    )

    # Wait for AG stream to complete before returning
    current_stream.wait_stream(ag_stream)

    return c
