# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Utility functions for pre-computing parameters and index tensors for Lean Attention.
"""

import torch
from bisect import bisect_right

def get_lean_attention_params(
    causal, batch_size, max_seqlen_q, max_seqlen_k, num_heads, BLOCK_M, BLOCK_N, num_SMs
):
    """
    Calculates core Lean Attention workload distribution parameters, mirroring the
    logic from the host code.

    Returns:
        A tuple containing:
        - tiles_per_head (int): Total number of lean tiles for a single attention head.
        - num_m_blocks (int): Number of blocks needed to cover the query sequence.
        - lean_griddimz (int): Number of workgroups (CTAs) to launch.
        - high_load_wgs (int): Number of workgroups that handle an extra tile.
        - max_tiles_per_wg (int): Maximum number of lean tiles any single workgroup will process.
    """
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # For causal attention, the workload is triangular.
        # The total number of lean tiles for a single head/batch item.
        for i in range(num_m_blocks):
            tiles_per_head += (i + 1) * (BLOCK_M // BLOCK_N)
        tiles_per_head *= batch_size
    else:
        # For non-causal, the workload is rectangular.
        num_n_blocks_total = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
        tiles_per_head = num_m_blocks * num_n_blocks_total

    total_tiles = tiles_per_head * num_heads
    lean_griddimz = num_SMs

    if lean_griddimz == 0:
        return 0, 0, 0, 0, 0

    # Distribute total tiles across all workgroups
    max_tiles_per_wg = (total_tiles + lean_griddimz - 1) // lean_griddimz
    high_load_wgs = total_tiles % lean_griddimz
    if high_load_wgs == 0 and total_tiles > 0:
        high_load_wgs = lean_griddimz

    return (
        tiles_per_head,
        num_m_blocks,
        lean_griddimz,
        high_load_wgs,
        max_tiles_per_wg,
    )

def find_group_python(x, num_m_blocks, MASKED_BLOCKS):
    """
    A Python implementation of the Triton find_group JIT function. It maps a global
    lean tile index `x` (relative to a head) to its corresponding query block and workload info.

    Returns:
        A tuple containing:
        - q_block_idx (int): The index of the Q block this tile belongs to.
        - task_size (int): The number of K/V blocks this Q block processes.
        - total_blocks_processed (int): The starting lean tile index for this Q block's workload.
    """
    total_blocks_processed = 0
    # The ping-pong iteration order for M-blocks ensures better cache usage
    for i in range(num_m_blocks):
        pair_idx = i // 2
        if (i % 2) == 0:
            # Even tasks are from the top (e.g., 0, 1, 2...)
            q_block_idx = pair_idx
        else:
            # Odd tasks are from the bottom (e.g., N-1, N-2, ...)
            q_block_idx = num_m_blocks - 1 - pair_idx

        task_size = (q_block_idx + 1) * MASKED_BLOCKS

        if total_blocks_processed + task_size > x:
            # Found the group this tile belongs to
            return q_block_idx, task_size, total_blocks_processed

        total_blocks_processed += task_size

    # This should not be reached if x is a valid tile index
    return -1, -1, -1


def precompute_lean_attention_indices(
    causal: bool,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_SMs: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
    """
    Calculates the max number of output tiles for any workgroup and also
    pre-computes a lookup tensor to replace the 'find_group' kernel call.

    Returns:
        A tuple containing:
        - max_output_tile_cnt (int): The compile-time constant for the kernel's persistent loop.
        - tile_indices (torch.Tensor or None): A [N, 3] tensor on the CUDA device containing
          pre-computed (q_block_idx, task_size, start_offset) for each lean tile.
          Returns None if not causal.
    """
    # Step 1: Get the same core parameters that the kernel will use
    (
        tiles_per_head,
        num_m_blocks,
        num_wgs,
        high_load_wgs,
        max_tiles_per_wg,
    ) = get_lean_attention_params(
        causal, batch_size, max_seqlen_q, max_seqlen_k, num_heads, BLOCK_M, BLOCK_N, num_SMs
    )

    if num_wgs == 0:
        return 0, None

    # Step 2: Pre-compute the indices for every lean tile if causal
    tile_indices = None
    m_block_boundaries = []
    if causal:
        MASKED_BLOCKS = BLOCK_M // BLOCK_N
        if BLOCK_M % BLOCK_N != 0:
            raise ValueError("For causal attention, BLOCK_M must be a multiple of BLOCK_N")

        tiles_per_head_per_batch = tiles_per_head // batch_size
        
        # Create the tensor to hold the pre-computed indices for a single head/batch item.
        # This will be broadcasted/indexed appropriately in the kernel.
        single_head_indices = torch.empty((tiles_per_head_per_batch, 3), dtype=torch.int32)

        # Populate the tensor by running the python find_group logic
        for i in range(tiles_per_head_per_batch):
            q_idx, task_size, start_offset = find_group_python(i, num_m_blocks, MASKED_BLOCKS)
            single_head_indices[i, 0] = q_idx
            single_head_indices[i, 1] = task_size
            single_head_indices[i, 2] = start_offset
        
        # Move the completed lookup tensor to the GPU
        tile_indices = single_head_indices.to("cuda")

        # Also compute the boundaries for the analytical max_tile calculation
        total_blocks = 0
        for i in range(num_m_blocks):
            pair_idx = i // 2
            q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
            task_size = (q_block_idx + 1) * MASKED_BLOCKS
            total_blocks += task_size
            m_block_boundaries.append(total_blocks)


    # Step 3: Calculate max_output_tile_cnt using the fast analytical method
    max_total_output_tiles = 0
    for wg_id in range(num_wgs):
        total_output_tiles_for_wg = 0

        # Determine the range of global lean tile indices for this WG
        if wg_id < high_load_wgs:
            start_iter = max_tiles_per_wg * wg_id
            end_iter = start_iter + max_tiles_per_wg
        else:
            start_iter = (max_tiles_per_wg - 1) * (wg_id - high_load_wgs) + high_load_wgs * max_tiles_per_wg
            end_iter = start_iter + (max_tiles_per_wg - 1)

        start_head = start_iter // tiles_per_head
        end_head = (end_iter - 1) // tiles_per_head

        for head_idx in range(start_head, end_head + 1):
            head_start_iter = head_idx * tiles_per_head

            wg_start_in_head = max(start_iter, head_start_iter)
            wg_end_in_head = min(end_iter, head_start_iter + tiles_per_head)

            if not causal:
                total_output_tiles_for_wg += 1
                continue

            # --- Causal Logic using Binary Search ---
            tiles_per_head_per_batch = tiles_per_head // batch_size
            
            start_batch_idx = (wg_start_in_head - head_start_iter) // tiles_per_head_per_batch
            end_batch_idx = (wg_end_in_head - 1 - head_start_iter) // tiles_per_head_per_batch

            for batch_idx in range(start_batch_idx, end_batch_idx + 1):
                batch_start_iter = head_start_iter + batch_idx * tiles_per_head_per_batch
                
                relative_start = max(wg_start_in_head, batch_start_iter) - batch_start_iter
                relative_end = min(wg_end_in_head, batch_start_iter + tiles_per_head_per_batch) - batch_start_iter
                
                start_m_idx = bisect_right(m_block_boundaries, relative_start)
                end_m_idx = bisect_right(m_block_boundaries, relative_end - 1)

                tiles_in_this_batch_item = (end_m_idx - start_m_idx) + 1
                total_output_tiles_for_wg += tiles_in_this_batch_item

        max_total_output_tiles = max(max_total_output_tiles, total_output_tiles_for_wg)
    return max_total_output_tiles, tile_indices
