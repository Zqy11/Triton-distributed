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
Test PyTorch symmetric memory (symm_mem) `multimem_all_gather_out` correctness

Usage:
    torchrun --nproc_per_node=8 test_symm_mem.py
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def test_multimem_all_gather():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group_name = dist.group.WORLD.group_name
    
    local_M = 512
    K = 1024
    
    local_shard = torch.randn(local_M, K, device="cuda", dtype=torch.bfloat16)
    local_shard.fill_(rank)
    
    global_M = local_M * world_size
    output = symm_mem.empty(global_M, K, dtype=torch.bfloat16, device="cuda")
    hdl = symm_mem.rendezvous(output, group=dist.group.WORLD)
    
    torch.ops.symm_mem.multimem_all_gather_out(local_shard, group_name, output)
    
    # Verify results
    for r in range(world_size):
        chunk = output[r * local_M : (r + 1) * local_M]
        assert chunk.eq(r).all(), f"Rank {rank}: chunk {r} verification failed"
    
    print(f"Rank {rank}: all-gather verification passed")


def main():
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    test_multimem_all_gather()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
