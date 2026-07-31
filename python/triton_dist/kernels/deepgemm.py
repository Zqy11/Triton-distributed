import triton


@triton.jit
def get_swizzled_block_idx(block_idx, num_pid_m, num_pid_n,
                           kNumMulticast=2, kIsMulticastOnA=True, kNum1DBlocksPerGroup=8, sm90_odd_fix=True):
    primary_num_blocks   = num_pid_n if kIsMulticastOnA else num_pid_m
    secondary_num_blocks = num_pid_m if kIsMulticastOnA else num_pid_n

    num_blocks_per_group = secondary_num_blocks * kNum1DBlocksPerGroup

    group_idx = block_idx // num_blocks_per_group
    first_block_idx = group_idx * kNum1DBlocksPerGroup
    in_group_idx = block_idx % num_blocks_per_group

    num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx)

    # SM90 odd fix: when multicast>1 and group primary count is odd, split into (even part) + (tail 1)
    if sm90_odd_fix and kNumMulticast > 1 and (num_blocks_in_group % 2 == 1):
        # same logic as scheduler.cuh
        if in_group_idx < ((num_blocks_in_group ^ 1) * secondary_num_blocks):
            num_blocks_in_group = (num_blocks_in_group ^ 1)
        else:
            in_group_idx = in_group_idx - ((num_blocks_in_group ^ 1) * secondary_num_blocks)
            first_block_idx += (num_blocks_in_group ^ 1)
            num_blocks_in_group = 1

    # convert to (m,n)
    if kIsMulticastOnA:
        m_block_idx = in_group_idx // num_blocks_in_group
        n_block_idx = first_block_idx + (in_group_idx % num_blocks_in_group)
    else:
        m_block_idx = first_block_idx + (in_group_idx % num_blocks_in_group)
        n_block_idx = in_group_idx // num_blocks_in_group

    return m_block_idx, n_block_idx
