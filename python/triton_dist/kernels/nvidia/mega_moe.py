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
MegaMoE kernels — FP8 quantization, shared-activation dispatch, and group GEMM

This module provides:
- Per-token-group FP8 quantization (Triton kernel)
- Block-wise FP8 weight quantization
- Shared-activation style dispatch: each rank writes its own segment of a
  symmetric NVSHMEM buffer in original token order; GEMM side reads peer
  data via kernel pointer indirection (buffer_ptrs + tl.cast to typed pointer)
  to filter local experts (no cross-rank data move, zero-copy P2P reads).
- 2D FP8 group GEMM with pointer-indirect gather from per-rank [MAX_M, K] buffers
"""

from typing import Tuple
import ctypes
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


# ============================================================================
# VMM continuous-VA remapping on top of symm_mem
# ============================================================================
#
# symm_mem allocates per-rank independent VAs (each rank's buffer_ptrs[r] is
# an isolated cuMemAddressReserve).  VMM remap creates a NEW contiguous VA span
# [0 .. WS * block_size) and re-maps every rank's physical allocation into it
# via cuMemRetainAllocationHandle + cuMemMap.  The GEMM kernel then sees a
# single [WS * MAX_M, ...] tensor with a constant base pointer — no pointer
# indirection, enabling compiler pipelining and hardware coalescing.

_cuda_driver = None


def _get_cuda_driver():
    global _cuda_driver
    if _cuda_driver is None:
        from cuda.bindings import driver
        _cuda_driver = driver
    return _cuda_driver


def _check_drv(result_tuple, label):
    drv = _get_cuda_driver()
    err = result_tuple[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    return result_tuple[1] if len(result_tuple) > 1 else None


class _DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


def _dtype_to_dlpack(dtype: torch.dtype):
    # DLPack type codes: kDLInt=0, kDLUInt=1, kDLFloat=2, kDLBfloat=4
    mapping = {
        torch.float8_e4m3fn: (2, 8),    # kDLFloat, 8 bits
        torch.float32: (2, 32),
        torch.float16: (2, 16),
        torch.bfloat16: (4, 16),
        torch.int32: (0, 32),
        torch.int8: (0, 8),
        torch.uint8: (1, 8),
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported VMM dtype: {dtype}")
    return mapping[dtype]


def _tensor_from_cuda_ptr(ptr, shape, dtype, device_id):
    """Wrap a raw device pointer into a PyTorch tensor via DLPack (zero-copy)."""
    dl_code, dl_bits = _dtype_to_dlpack(dtype)
    shape_array = (ctypes.c_int64 * len(shape))(*shape)
    dl_dtype = _DLDataType(dl_code, dl_bits, 1)
    dl_device = _DLDevice(2, device_id)  # kDLCUDA = 2

    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = dl_device
    managed.dl_tensor.ndim = len(shape)
    managed.dl_tensor.dtype = dl_dtype
    managed.dl_tensor.shape = ctypes.cast(
        ctypes.pointer(shape_array), ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = ctypes.c_void_p(0)

    # No-op deleter
    _deleter_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    _deleter_fn = _deleter_type(lambda _: None)
    managed.deleter = ctypes.cast(_deleter_fn, ctypes.c_void_p)

    refs = [managed, shape_array, dl_device, dl_dtype, _deleter_fn]
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.byref(managed), b"dltensor", None)
    return torch.from_dlpack(capsule), refs


class VMMContiguousBuffer:
    """Remaps symm_mem per-rank buffers into a single contiguous VA span.

    After symm_mem.rendezvous, each rank's buffer is at an independent VA.
    This class creates a NEW contiguous VA [0 .. WS * block_size), re-maps all
    peer allocations into it via cuMemRetainAllocationHandle + cuMemMap, and
    returns a PyTorch tensor view with shape [WS * padded_rows, ...].

    The original symm_mem handle is untouched — barrier/signal_pad still work.
    The GEMM kernel sees a single contiguous tensor; padding rows at the end
    of each rank segment are never read (token_mask handles that).
    """

    def __init__(
        self,
        hdl,                # symm_mem handle (from rendezvous)
        src_tensor,         # torch.Tensor — original symm_mem tensor (kept alive to
                           #   prevent CUDA from freeing the physical allocation)
        user_rows,          # int — logical rows the user writes (e.g. max_m)
        row_bytes,          # int — bytes per row (dtype_size * prod(shape_tail))
        world_size,         # int
        rank,               # int
        dtype,              # torch.dtype
        shape_tail,         # tuple — dims after the row dim, e.g. (hidden,) or (topk,)
    ):
        self._refs = []
        self._src_tensor = src_tensor  # keep physical allocation alive
        self._mapped_vas = []
        self._base_va = 0
        self._total_bytes = 0
        self._block_size = 0
        self._closed = False

        drv = _get_cuda_driver()
        device_id = torch.cuda.current_device()

        # hdl.buffer_size is the LOGICAL tensor size (bytes), but the physical
        # allocation (from cuMemCreate inside symm_mem) is aligned to GPU
        # allocation granularity.  cuMemAddressReserve and cuMemMap require
        # sizes that are multiples of this granularity.
        buffer_ptrs = hdl.buffer_ptrs  # list[int]

        # Query granularity from the first rank's allocation
        probe_handle = _check_drv(
            drv.cuMemRetainAllocationHandle(int(buffer_ptrs[0])),
            "cuMemRetainAllocationHandle(probe)",
        )
        alloc_prop = _check_drv(
            drv.cuMemGetAllocationPropertiesFromHandle(probe_handle),
            "cuMemGetAllocationPropertiesFromHandle",
        )
        granularity = int(_check_drv(
            drv.cuMemGetAllocationGranularity(
                alloc_prop,
                drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        ))
        _check_drv(drv.cuMemRelease(probe_handle), "cuMemRelease(probe)")

        raw_size = hdl.buffer_size
        block_size = ((raw_size + granularity - 1) // granularity) * granularity
        self._block_size = block_size
        total_bytes = block_size * world_size
        self._total_bytes = total_bytes

        base_va = int(_check_drv(
            drv.cuMemAddressReserve(total_bytes, granularity, 0, 0),
            "cuMemAddressReserve",
        ))
        self._base_va = base_va

        access_desc = drv.CUmemAccessDesc()
        access_desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = device_id
        access_desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        for r in range(world_size):
            peer_ptr = int(buffer_ptrs[r])
            _, handle = drv.cuMemRetainAllocationHandle(peer_ptr)

            address = base_va + r * block_size
            _check_drv(
                drv.cuMemMap(address, block_size, 0, handle, 0),
                f"cuMemMap(rank={r})",
            )
            self._mapped_vas.append(address)
            _check_drv(
                drv.cuMemSetAccess(address, block_size, [access_desc], 1),
                f"cuMemSetAccess(rank={r})",
            )
            _check_drv(drv.cuMemRelease(handle), "cuMemRelease")

        # Build a 3D tensor view: (WS, user_rows, *shape_tail)
        # with strides (block_size // dtype_size, row_elements, ..., 1).
        #
        # block_size is aligned to GPU granularity (e.g. 2MB) and is NOT a
        # multiple of row_bytes.  A flat 2D tensor would have wrong byte
        # offsets for rank > 0.  The 3D layout with explicit strides ensures
        # rank r's data starts at byte r * block_size in the VA.
        dtype_size = torch.empty((), dtype=dtype).element_size() if dtype not in (
            torch.float8_e4m3fn, torch.float8_e5m2) else 1
        row_elements = 1
        for s in shape_tail:
            row_elements *= s
        stride_ws = block_size // dtype_size  # elements between rank segments

        # Build stride tuple: (stride_ws, row_elements, *contiguous_strides(shape_tail))
        # The shape_tail dimensions use standard contiguous strides.
        tail_strides = []
        acc = 1
        for s in reversed(shape_tail):
            tail_strides.append(acc)
            acc *= s
        tail_strides.reverse()
        strides_tuple = tuple([stride_ws, row_elements] + tail_strides)
        view_shape = (world_size, user_rows, *shape_tail)

        # For fp8, create as int8 then view
        alloc_dtype = torch.int8 if dtype in (
            torch.float8_e4m3fn, torch.float8_e5m2) else dtype
        alloc_dtype_size = 1 if dtype in (
            torch.float8_e4m3fn, torch.float8_e5m2) else dtype_size

        # Create flat 1D buffer covering the entire VMM range
        flat_elements = total_bytes // alloc_dtype_size
        flat_tensor, refs = _tensor_from_cuda_ptr(
            base_va, (flat_elements,), alloc_dtype, device_id)
        self._refs = refs

        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            flat_tensor = flat_tensor.view(dtype)

        self.global_view = flat_tensor.as_strided(view_shape, strides_tuple)
        self.local_view = self.global_view[rank]
        self.stride_ws = stride_ws
        self.user_rows = user_rows
        self.block_size = block_size

    def close(self):
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize()
        del self.global_view
        del self.local_view
        del self._refs
        drv = _get_cuda_driver()
        for va in self._mapped_vas:
            _check_drv(drv.cuMemUnmap(va, self._block_size), "cuMemUnmap")
        _check_drv(drv.cuMemAddressFree(self._base_va, self._total_bytes), "cuMemAddressFree")
        del self._src_tensor


# Use the default CUDA symmetric memory backend (not NVSHMEM), so that
# symm_mem handle .barrier() is a real GPU-side barrier (signal-pad based)
# rather than the NVSHMEM no-op stub.
# The CUDA backend uses CUDA IPC handles for cross-GPU symmetric memory and
# works on any multi-GPU node with NVLink — NVSHMEM is not required.


# ============================================================================
# Shared-Activation Dispatch (owner-write, read-side filtering, VMM continuous VA)
# ============================================================================
#
# symm_mem.empty() allocates a PER-RANK independent symmetric heap.
# After rendezvous, VMMContiguousBuffer remaps all ranks' physical allocations
# into a single contiguous virtual address [0 .. WS * block_size) via
# cuMemRetainAllocationHandle + cuMemMap.  GEMM kernels see a single
# [WS * padded_rows, ...] tensor with a constant base pointer — no per-rank
# pointer indirection.
#
# Synchronization: symm_mem handle .barrier() (real GPU-side signal-pad
# barrier with CUDA backend).
#
# Per-rank layout (each tensor is [MAX_M, ...] on each rank, VMM-remapped
# into 3D (WS, MAX_M, ...) with explicit strides):
#   fp8:           fp8    [MAX_M, hidden]
#   scales:        fp32   [MAX_M, NSG]
#   topk_ids:      int32  [MAX_M, topk]
#   topk_weights:  fp32   [MAX_M, topk]
#
# All read/write goes through vmm_*.local_view (write side) and
# vmm_*.global_view (GEMM read side).  VMMContiguousBuffer internally
# retains the original symm_mem tensor to keep the physical allocation alive.


class BlockDispatchContext:
    """Pre-allocated symmetric memory buffers for shared-activation dispatch
    (intra-node, CUDA symm_mem backend + VMM contiguous VA remap).

    Each rank allocates per-rank-sized [MAX_M, ...] symmetric tensors and
    writes its own tokens in original token order.  After symm_mem rendezvous,
    VMMContiguousBuffer remaps all ranks' physical allocations into a single
    contiguous VA.  After barrier, the GEMM kernel reads the VMM tensor with a
    single base pointer (no per-rank pointer indirection).
    """

    def __init__(
        self,
        max_m: int,
        hidden: int,
        num_experts: int,
        world_size: int,
        rank: int,
        num_scale_groups: int,
        topk: int,
        group_name: str,
    ):
        dev = f"cuda:{rank}"
        epr = num_experts // world_size

        # Per-rank symmetric allocation.  Each rank gets its
        # own [MAX_M, ...] heap; peer data is NOT in a continuous VA range.
        # Order must be identical on all ranks (rendezvous is collective).
        fp8_tensor, self.fp8_hdl = self._create_tensor(
            (max_m, hidden), FP8_DTYPE, dev, group_name)
        scales_tensor, self.scales_hdl = self._create_tensor(
            (max_m, num_scale_groups), torch.float32, dev, group_name)
        ids_tensor, self.ids_hdl = self._create_tensor(
            (max_m, topk), torch.int32, dev, group_name)
        weights_tensor, self.weights_hdl = self._create_tensor(
            (max_m, topk), torch.float32, dev, group_name)

        # VMM remap: create contiguous VA spans for each buffer.
        # VMMContiguousBuffer retains src_tensor to keep the physical
        # allocation alive; GEMM kernels use a single base pointer.
        self.vmm_fp8 = VMMContiguousBuffer(
            self.fp8_hdl, fp8_tensor, max_m, max_m * hidden, world_size, rank,
            FP8_DTYPE, (hidden,))
        self.vmm_scales = VMMContiguousBuffer(
            self.scales_hdl, scales_tensor, max_m, max_m * num_scale_groups * 4, world_size, rank,
            torch.float32, (num_scale_groups,))
        self.vmm_ids = VMMContiguousBuffer(
            self.ids_hdl, ids_tensor, max_m, max_m * topk * 4, world_size, rank,
            torch.int32, (topk,))
        self.vmm_weights = VMMContiguousBuffer(
            self.weights_hdl, weights_tensor, max_m, max_m * topk * 4, world_size, rank,
            torch.float32, (topk,))

        self.max_m = max_m
        self.hidden = hidden
        self.num_experts = num_experts
        self.experts_per_rank = epr
        self.num_scale_groups = num_scale_groups
        self.topk = topk
        self.world_size = world_size
        self.rank = rank
        self.num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.local_expert_start = rank * epr

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

    def local_segment(self, num_tokens: int = None):
        """Return views into the calling rank's own segment of the 4 VMM tensors.

        All four views are slices of the VMM local_view (3D [WS, max_m, *tail]
        indexed at dim-0=rank).  Writes to these views go directly into the VMM
        contiguous VA, so the GEMM kernel sees them after barrier with no copy.

        Args:
            num_tokens: Number of tokens to slice (default: max_m).

        Returns:
            (fp8_local, scales_local, topk_ids_local, topk_weights_local)
        """
        n = self.max_m if num_tokens is None else num_tokens
        return (self.vmm_fp8.local_view[:n],
                self.vmm_scales.local_view[:n],
                self.vmm_ids.local_view[:n],
                self.vmm_weights.local_view[:n])

    def get_vmm_views(self):
        """Return VMM 3D tensors for the GEMM kernel.

        Returns: (fp8_global, scales_global, ids_global, weights_global,
                  fp8_stride_ws, scales_stride_ws)
        Each global tensor is 3D: (WS, max_m, *shape_tail) with explicit strides.
        The GEMM kernel decomposes logical row index → (rank, local_row) and
        uses stride_ws to compute the VMM address.
        """
        return (self.vmm_fp8.global_view,
                self.vmm_scales.global_view,
                self.vmm_ids.global_view,
                self.vmm_weights.global_view,
                self.vmm_fp8.stride_ws,
                self.vmm_scales.stride_ws)

    def get_peer_buffer(self, peer_rank: int, name: str, num_tokens: int = None):
        """Get a tensor view of a peer rank's shared buffer.

        Args:
            peer_rank: rank whose data to access.
            name: one of 'fp8', 'scales', 'ids', 'weights'.
            num_tokens: number of rows to view (default: max_m).
        """
        n = self.max_m if num_tokens is None else num_tokens
        _map = {
            'fp8':     (self.fp8_hdl,     FP8_DTYPE,       (n, self.hidden)),
            'scales':  (self.scales_hdl,  torch.float32,   (n, self.num_scale_groups)),
            'ids':     (self.ids_hdl,     torch.int32,     (n, self.topk)),
            'weights': (self.weights_hdl, torch.float32,   (n, self.topk)),
        }
        hdl, dtype, shape = _map[name]
        # For fp8, get_buffer may need int8 then view
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            buf = hdl.get_buffer(peer_rank, shape, torch.int8)
            return buf.view(dtype)
        return hdl.get_buffer(peer_rank, shape, dtype)

    def barrier(self):
        """Cross-rank sync via CUDA symm_mem signal-pad barrier."""
        self.fp8_hdl.barrier()
        self.scales_hdl.barrier()
        self.ids_hdl.barrier()
        self.weights_hdl.barrier()

    def finalize(self):
        pass


def create_block_dispatch_context(
    max_m: int,
    hidden: int,
    num_experts: int,
    world_size: int,
    rank: int,
    num_scale_groups: int,
    topk: int = 6,
    group_name: str = None,
):
    """Create a BlockDispatchContext for shared-activation style dispatch."""
    if group_name is None:
        group_name = dist.group.WORLD.group_name
    return BlockDispatchContext(
        max_m, hidden, num_experts, world_size, rank,
        num_scale_groups, topk, group_name)


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
# Fused Triton Per-Token-Group FP8 Quantization (writes into VMM local view)
# ============================================================================

@triton.jit
def per_token_quant_kernel(
    X,                  # bf16 [M, K] — source input (contiguous)
    SrcTopkIds,         # int32 [M, TOPK] — source topk ids
    SrcTopkWeights,     # float32 [M, TOPK] — source topk weights
    OutFP8,             # fp8 [MAX_M, K] — VMM local view (output)
    OutScales,          # float32 [MAX_M, NSG] — VMM local view (output)
    OutIds,             # int32 [MAX_M, TOPK] — VMM local view (output)
    OutWeights,         # float32 [MAX_M, TOPK] — VMM local view (output)
    K,
    X_stride_m,
    SrcIds_stride_m,
    Out_stride_m,
    Scale_stride_m,
    Ids_stride_m,
    NUM_TOKENS,         # number of tokens to process
    NUM_GROUPS,         # K // GROUP_K — total quantization groups along K
    TOPK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_G: tl.constexpr,   # quantization groups handled per CTA (K-parallelism)
    TOPK_POW2: tl.constexpr, # next_power_of_2(TOPK)
):
    """Fused per-token-group FP8 quantization + topk_ids/weights copy into VMM local view.

    Row index = absolute token index in [0, NUM_TOKENS).  Each CTA processes
    BLOCK_M rows × BLOCK_G quantization groups, and also copies the
    corresponding topk_ids/topk_weights rows.  No gather — tokens are written
    in original order.
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < NUM_TOKENS

    g_base = pid_g * BLOCK_G
    for gi in tl.static_range(0, BLOCK_G):
        g = g_base + gi
        if g < NUM_GROUPS:
            offs_k = g * GROUP_K + tl.arange(0, GROUP_K)

            x_ptrs = X + offs_m[:, None] * X_stride_m + offs_k[None, :]
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

    # --- Copy topk_ids and topk_weights (only on the first K-group CTA) ---
    if pid_g == 0:
        topk_offs = tl.arange(0, TOPK_POW2)
        topk_mask = topk_offs < TOPK

        # Read source topk_ids → write to VMM local view
        src_ids_ptrs = SrcTopkIds + offs_m[:, None] * SrcIds_stride_m + topk_offs[None, :]
        src_ids = tl.load(src_ids_ptrs, mask=row_mask[:, None] & topk_mask[None, :], other=0)
        dst_ids_ptrs = OutIds + offs_m[:, None] * Ids_stride_m + topk_offs[None, :]
        tl.store(dst_ids_ptrs, src_ids, mask=row_mask[:, None] & topk_mask[None, :])

        # Read source topk_weights → write to VMM local view
        src_w_ptrs = SrcTopkWeights + offs_m[:, None] * SrcIds_stride_m + topk_offs[None, :]
        src_w = tl.load(src_w_ptrs, mask=row_mask[:, None] & topk_mask[None, :], other=0.0)
        dst_w_ptrs = OutWeights + offs_m[:, None] * Ids_stride_m + topk_offs[None, :]
        tl.store(dst_w_ptrs, src_w, mask=row_mask[:, None] & topk_mask[None, :])


def quant_local(
    ctx: BlockDispatchContext,
    x: torch.Tensor,             # bf16 [M, K] — source input, must be contiguous
    topk_ids: torch.Tensor,      # int32 [M, TOPK] — source topk expert ids
    topk_weights: torch.Tensor,  # fp32 [M, TOPK] — source topk weights
    num_tokens: int = None,
    group_k: int = 128,
    BLOCK_M: int = 64,
):
    """Fused FP8 quantization + topk metadata copy into VMM local view.

    Reads x (bf16), quantizes to FP8 per-token-group, and stores fp8 + scales +
    topk_ids + topk_weights directly into the VMM local views, so the GEMM
    kernel sees the data through the VMM contiguous VA with no extra copy.

    Args:
        ctx: BlockDispatchContext providing vmm_*.local_view, num_sms.
        x: Input tensor [M, K] in bf16, must be contiguous.
        topk_ids: Source topk expert ids [M, TOPK] in int32.
        topk_weights: Source topk weights [M, TOPK] in float32.
        num_tokens: Number of tokens (default: x.shape[0]).
        group_k: Quantization group size along K dimension.
        BLOCK_M: Rows per thread block.
    """
    out_fp8 = ctx.vmm_fp8.local_view
    out_scales = ctx.vmm_scales.local_view
    out_ids = ctx.vmm_ids.local_view
    out_weights = ctx.vmm_weights.local_view

    M, K = x.shape
    n = M if num_tokens is None else num_tokens
    TOPK = topk_ids.shape[1]
    assert K % group_k == 0, f"K({K}) must be divisible by group_k({group_k})"
    assert x.dtype == torch.bfloat16, f"Input must be bf16, got {x.dtype}"
    assert x.is_contiguous(), "Input must be contiguous"
    assert out_fp8.shape[1] >= K, f"out_fp8 K dim {out_fp8.shape[1]} < {K}"
    assert out_scales.shape[1] >= K // group_k
    assert topk_ids.dtype == torch.int32, "topk_ids must be int32"
    assert topk_weights.dtype == torch.float32, "topk_weights must be float32"

    num_groups = K // group_k
    row_blocks = cdiv(n, BLOCK_M)

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
    per_token_quant_kernel[grid](
        x, topk_ids, topk_weights,
        out_fp8, out_scales, out_ids, out_weights,
        K,
        x.stride(0),
        topk_ids.stride(0),
        out_fp8.stride(0),
        out_scales.stride(0),
        out_ids.stride(0),
        n,
        num_groups,
        TOPK=TOPK,
        FP8_MAX=448.0,
        GROUP_K=group_k,
        BLOCK_M=BLOCK_M,
        BLOCK_G=BLOCK_G,
        TOPK_POW2=triton.next_power_of_2(TOPK),
    )


# ============================================================================
# 2D Group GEMM — VMM contiguous input, no pointer indirection
# ============================================================================
#
# A / A_scales / topk_ids are VMM-backed [WS*padded_rows, ...] tensors with a
# single contiguous virtual address.  The scan kernel reads topk_ids directly
# to build row_offsets / m_indices / expert_ids / counts_per_row tables.
# The GEMM kernel uses a single base pointer + direct row indexing.


@triton.jit
def _build_m_indices_from_topk_kernel(
    topk_ids,           # int32 3D (WS, max_m, TOPK) — VMM with stride_ws
    row_offsets,        # int32 [max_hit_rows] — logical row index (r*MAX_M + local_row)
    m_indices,          # int32 [max_blocks] — output: offset into row_offsets per GEMM block
    expert_ids,         # int32 [max_blocks] — output: local expert id per block
    counts_per_row,     # int32 [max_blocks] — output: valid token count per block
    expert_recv_count,  # int32 [EPR] — output: total tokens per local expert
    block_counter,      # int32 [1] — zero-init
    row_counter,        # int32 [1] — zero-init
    # Constexpr sizes
    LOCAL_EXPERT_START: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_POW2: tl.constexpr,
    NUM_TOKENS_PER_RANK: tl.constexpr,
    MAX_M: tl.constexpr,
    TOPK_STRIDE_WS: tl.constexpr,
    EPR: tl.constexpr,
    WS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
):
    """VMM scan kernel: topk_ids is a 3D (WS, max_m, TOPK) tensor.

    Uses TOPK_STRIDE_WS (elements between ranks) to index into the VMM tensor.
    Stores logical row indices (r * MAX_M + local_row) in row_offsets.
    The GEMM kernel decomposes these to compute VMM addresses.
    """
    expert_id = tl.program_id(0)
    target_e = LOCAL_EXPERT_START + expert_id

    topk_offs = tl.arange(0, TOPK_POW2)
    topk_mask = topk_offs < TOPK

    total_rows = 0
    for r in range(WS):
        base = r * TOPK_STRIDE_WS
        for blk in range(tl.cdiv(NUM_TOKENS_PER_RANK, SCAN_BLOCK)):
            offs = blk * SCAN_BLOCK + tl.arange(0, SCAN_BLOCK)
            mask = offs < NUM_TOKENS_PER_RANK
            ids = tl.load(
                topk_ids + base + offs[:, None] * TOPK + topk_offs[None, :],
                mask=mask[:, None] & topk_mask[None, :], other=-1,
            )
            hit_per_row = (ids == target_e)
            has_hit = tl.sum(hit_per_row.to(tl.int32), axis=1) > 0
            has_hit = has_hit & mask
            total_rows += tl.sum(has_hit.to(tl.int32))

    num_blocks = tl.cdiv(total_rows, BLOCK_M)
    row_off = tl.atomic_add(row_counter, total_rows)
    block_off = tl.atomic_add(block_counter, num_blocks)
    tl.store(expert_recv_count + expert_id, total_rows)

    for b in range(num_blocks):
        tl.store(m_indices + block_off + b, row_off + b * BLOCK_M)
        tl.store(expert_ids + block_off + b, expert_id)
        remaining = total_rows - b * BLOCK_M
        tl.store(counts_per_row + block_off + b, tl.minimum(remaining, BLOCK_M))

    local_idx = 0
    for r in range(WS):
        base = r * TOPK_STRIDE_WS
        c_base = r * MAX_M
        for blk in range(tl.cdiv(NUM_TOKENS_PER_RANK, SCAN_BLOCK)):
            offs = blk * SCAN_BLOCK + tl.arange(0, SCAN_BLOCK)
            mask = offs < NUM_TOKENS_PER_RANK
            ids = tl.load(
                topk_ids + base + offs[:, None] * TOPK + topk_offs[None, :],
                mask=mask[:, None] & topk_mask[None, :], other=-1,
            )
            hit_per_row = (ids == target_e)
            row_hit = (tl.sum(hit_per_row.to(tl.int32), axis=1) > 0) & mask
            hits_i = row_hit.to(tl.int32)
            positions = tl.cumsum(hits_i) - hits_i
            total_in_block = tl.sum(hits_i)
            c_hit_rows = c_base + offs  # logical row index for C and decomposition

            tl.store(
                row_offsets + row_off + local_idx + positions,
                c_hit_rows, mask=row_hit,
            )
            local_idx += total_in_block


@triton.jit
def moe_fp8_groupgemm_kernel_2d(
    A,              # fp8 3D (WS, max_m, K) — VMM with stride_ws
    B,              # fp8 [E, N, K]
    C,              # output [EPR, WS*MAX_M, N]
    A_scales,       # fp32 3D (WS, max_m, NSG) — VMM with stride_ws
    B_scales,       # [E, N // group_n, K // group_k]
    row_offsets,    # [total_rows] int32 — logical row index (r*MAX_M + local_row)
    m_indices,      # [total_blocks] int32 — offset into row_offsets per GEMM block
    expert_ids,     # [total_blocks] int32 — local expert id per block
    counts_per_row, # [total_blocks] int32 — valid token count per block
    total_blocks_ptr,  # int32 [1] — actual block count on device
    N,
    K,
    E,
    group_n,
    group_k,
    A_stride_ws,    # stride between ranks in VMM (in elements)
    A_stride_m,     # hidden
    A_stride_k,     # 1
    A_s_stride_ws,  # stride between ranks for scales in VMM
    A_s_stride_m,   # num_scale_groups
    A_s_stride_k,   # 1
    B_stride_e,
    B_stride_n,
    B_stride_k,
    B_s_stride_e,
    B_s_stride_n,
    B_s_stride_k,
    C_stride_e,
    C_stride_m,
    C_stride_n,
    MAX_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    swap_ab: tl.constexpr = False,
):
    """2D FP8 group GEMM with VMM 3D input.

    A is a 3D (WS, max_m, K) fp8 tensor with VMM strides.
    row_offsets stores logical indices (r*MAX_M + local_row).
    The kernel decomposes to (r, local_row) and uses stride_ws for VMM addressing.
    C is a standard 2D tensor indexed directly by the logical index.

    swap_ab (as in sglang's fused_moe_kernel): swaps the dot operands so
    BLOCK_N rides the MMA M-dim, which helps small BLOCK_M tiles on SM90.
    """
    pid = tl.program_id(0)
    num_block_n = tl.cdiv(N, BLOCK_N)

    total_blocks = tl.load(total_blocks_ptr)
    if pid >= total_blocks * num_block_n:
        return

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

    # Logical row indices (r*MAX_M + local_row) — used directly for C
    offs_token = tl.load(row_offsets + row_ptr_base + offs_m, mask=token_mask, other=0)
    # Decompose for VMM A addressing
    r_idx = offs_token // MAX_M
    local_row = offs_token % MAX_M

    offs_k = tl.arange(0, BLOCK_K)

    # VMM address: A + r * stride_ws + local_row * stride_m + k
    a_ptrs = A + r_idx[:, None] * A_stride_ws + local_row[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    b_ptrs = B + expert_id.to(tl.int64) * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    # VMM scales address
    As_ptrs = A_scales + r_idx * A_s_stride_ws + local_row * A_s_stride_m

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

    e_local = tl.load(expert_ids + pid_m)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # C is standard 2D: index directly with logical row
    c_ptrs = C + e_local.to(tl.int64) * C_stride_e + offs_token[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
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
    A: torch.Tensor,               # fp8 3D (WS, max_m, hidden) — VMM with stride_ws
    weights: torch.Tensor,          # fp8 [EPR, N, K]
    A_scales: torch.Tensor,         # fp32 3D (WS, max_m, NSG) — VMM with stride_ws
    b_scales: torch.Tensor,         # fp32 [EPR, N // group_n, K // group_k]
    topk_ids: torch.Tensor,         # int32 3D (WS, max_m, topk) — VMM with stride_ws
    local_expert_start: int,
    num_tokens_per_rank: int,
    TOPK: int,
    MAX_M: int,
    A_STRIDE_WS: int,
    A_S_STRIDE_WS: int,
    TOPK_STRIDE_WS: int,
    EXPERTS_PER_RANK: int,
    WS: int,
    hidden: int,
    num_scale_groups: int,
    group_n: int = 128,
    group_k: int = 128,
    BLOCK_M: int = None,  # None = auto from expected tokens per expert (16/32/64)
    output_dtype: torch.dtype = torch.bfloat16,
    swap_ab: bool = None,  # None = auto-detect (SM90 + BLOCK_M < 64)
):
    """VMM-based 2D FP8 group GEMM with 3D VMM tensors.

    A, A_scales, topk_ids are 3D (WS, max_m, *) tensors with VMM strides.
    The GEMM kernel decomposes logical row indices to compute VMM addresses.

    BLOCK_M=None picks the tile from the expected per-expert hit count under
    uniform routing (num_tokens_per_rank * TOPK / EXPERTS_PER_RANK): small
    workloads get small tiles, which auto-enable swap_ab on SM90 (see
    _pick_block_m).
    """
    WS_MAX_M = WS * MAX_M
    K = hidden
    E = weights.shape[0]
    N = weights.shape[1]

    if BLOCK_M is None:
        # Expected hits per expert under uniform routing
        BLOCK_M = _pick_block_m(num_tokens_per_rank * TOPK / EXPERTS_PER_RANK)

    A_stride_m = hidden
    A_stride_k = 1
    A_s_stride_m = num_scale_groups
    A_s_stride_k = 1

    max_hit_rows = EXPERTS_PER_RANK * WS * num_tokens_per_rank
    row_offsets = torch.empty(max_hit_rows, dtype=torch.int32, device=weights.device)
    max_total_blocks = EXPERTS_PER_RANK * cdiv(WS * num_tokens_per_rank, BLOCK_M)
    m_indices = torch.empty(max_total_blocks, dtype=torch.int32, device=weights.device)
    expert_ids = torch.empty(max_total_blocks, dtype=torch.int32, device=weights.device)
    counts_per_row = torch.empty(max_total_blocks, dtype=torch.int32, device=weights.device)
    expert_recv_count = torch.empty(EXPERTS_PER_RANK, dtype=torch.int32, device=weights.device)
    block_counter = torch.zeros(1, dtype=torch.int32, device=weights.device)
    row_counter = torch.zeros(1, dtype=torch.int32, device=weights.device)

    SCAN_BLOCK = 128

    _build_m_indices_from_topk_kernel[(EXPERTS_PER_RANK,)](
        topk_ids,
        row_offsets,
        m_indices, expert_ids, counts_per_row,
        expert_recv_count,
        block_counter, row_counter,
        LOCAL_EXPERT_START=local_expert_start,
        TOPK=TOPK,
        TOPK_POW2=triton.next_power_of_2(TOPK),
        NUM_TOKENS_PER_RANK=num_tokens_per_rank,
        MAX_M=MAX_M,
        TOPK_STRIDE_WS=TOPK_STRIDE_WS,
        EPR=EXPERTS_PER_RANK,
        WS=WS,
        BLOCK_M=BLOCK_M,
        SCAN_BLOCK=SCAN_BLOCK,
    )

    output = torch.empty(EXPERTS_PER_RANK, WS_MAX_M, N, dtype=output_dtype, device=weights.device)

    num_block_n = cdiv(N, 128)
    grid = (max_total_blocks * num_block_n,)
    BLOCK_N = 128
    BLOCK_K = group_k
    GROUP_M = 8

    if swap_ab is None:
        swap_ab = _should_enable_swap_ab(BLOCK_M, BLOCK_N)

    moe_fp8_groupgemm_kernel_2d[grid](
        A,
        weights,
        output,
        A_scales,
        b_scales,
        row_offsets,
        m_indices,
        expert_ids,
        counts_per_row,
        block_counter,
        N, K, E,
        group_n, group_k,
        A_STRIDE_WS, A_stride_m, A_stride_k,
        A_S_STRIDE_WS, A_s_stride_m, A_s_stride_k,
        weights.stride(0), weights.stride(1), weights.stride(2),
        b_scales.stride(0), b_scales.stride(1), b_scales.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        MAX_M=MAX_M,
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