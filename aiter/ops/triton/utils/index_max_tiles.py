# --- User-provided utility functions from the previous prompt ---
import torch
from bisect import bisect_right

def get_lean_attention_params(
    causal, batch_size, max_seqlen_q, max_seqlen_k, num_heads, BLOCK_M, BLOCK_N, num_SMs
):
    """
    Calculates core Lean Attention workload distribution parameters.
    """
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        for i in range(num_m_blocks):
            tiles_per_head += (i + 1) * (BLOCK_M // BLOCK_N)
        tiles_per_head *= batch_size
    else:
        num_n_blocks_total = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
        tiles_per_head = num_m_blocks * num_n_blocks_total

    total_tiles = tiles_per_head * num_heads
    lean_griddimz = num_SMs

    if lean_griddimz == 0:
        return 0, 0, 0, 0, 0

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
    A Python implementation of the Triton find_group JIT function.
    """
    total_blocks_processed = 0
    for i in range(num_m_blocks):
        pair_idx = i // 2
        q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
        task_size = (q_block_idx + 1) * MASKED_BLOCKS
        if total_blocks_processed + task_size > x:
            return q_block_idx, task_size, total_blocks_processed
        total_blocks_processed += task_size
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
    Calculates max output tiles and pre-computes a lookup tensor to replace 'find_group'.
    """
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
        return 0, None, 0

    tile_indices = None
    m_block_boundaries = []
    tiles_per_head_per_batch = 0
    if causal:
        MASKED_BLOCKS = BLOCK_M // BLOCK_N
        if BLOCK_M % BLOCK_N != 0:
            raise ValueError("For causal attention, BLOCK_M must be a multiple of BLOCK_N")

        tiles_per_head_per_batch = tiles_per_head // batch_size
        single_head_indices = torch.empty((tiles_per_head_per_batch, 3), dtype=torch.int32)

        for i in range(tiles_per_head_per_batch):
            q_idx, task_size, start_offset = find_group_python(i, num_m_blocks, MASKED_BLOCKS)
            single_head_indices[i, 0] = q_idx
            single_head_indices[i, 1] = task_size
            single_head_indices[i, 2] = start_offset
        
        tile_indices = single_head_indices.to("cuda")

        total_blocks = 0
        for i in range(num_m_blocks):
            pair_idx = i // 2
            q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
            task_size = (q_block_idx + 1) * MASKED_BLOCKS
            total_blocks += task_size
            m_block_boundaries.append(total_blocks)

    max_total_output_tiles = 0
    for wg_id in range(num_wgs):
        total_output_tiles_for_wg = 0
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
        
    return max_total_output_tiles, tile_indices, tiles_per_head_per_batch


def calculate_max_output_tiles_analytically(
    causal: bool,
    batch_size: int,
    max_seqlen_q: int,
    n_ctx: list[int],
    num_heads: int,
    num_SMs: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
    """
    Calculates the maximum number of output tiles any single workgroup will process
    using a fast, analytical method with binary search.

    This version is corrected to robustly handle both causal and non-causal attention
    with batching and ragged sequence lengths.

    Args:
        n_ctx (list[int]): A list of sequence lengths for each item in the batch.
                           This is required for accurate non-causal ragged calculation.
    """
    max_seqlen_k = sum(n_ctx)
    MASKED_BLOCKS = BLOCK_M // BLOCK_N
    if causal and BLOCK_M % BLOCK_N != 0:
        raise ValueError("For causal, BLOCK_M must be a multiple of BLOCK_N")

    (
        tiles_per_head,
        num_m_blocks,
        num_wgs,
        high_load_wgs,
        max_tiles_per_wg,
    ) = get_lean_attention_params(
        causal,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        BLOCK_M,
        BLOCK_N,
        num_SMs,
    )

    if num_wgs == 0:
        return 0

    boundaries = []
    if causal:
        # For causal, the boundaries are defined by the end of each M-block's workload
        # within a single batch item.
        total_blocks = 0
        for i in range(num_m_blocks):
            pair_idx = i // 2
            q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
            task_size = (q_block_idx + 1) * MASKED_BLOCKS
            total_blocks += task_size
            boundaries.append(total_blocks)
    else:
        # For non-causal, the boundaries are defined by the end of each batch item's workload.
        num_m_blocks_non_causal = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
        n_blocks_per_item = [(n + BLOCK_N - 1) // BLOCK_N for n in n_ctx]
        tiles_per_item = [num_m_blocks_non_causal * n_blocks for n_blocks in n_blocks_per_item]
        total_tiles = 0
        for tiles in tiles_per_item:
            total_tiles += tiles
            boundaries.append(total_tiles)

    max_total_output_tiles = 0
    # Loop through each workgroup to find the one that spans the most output tiles.
    for wg_id in range(num_wgs):
        total_output_tiles_for_wg = 0

        # Determine the range of global lean tile indices for this WG
        if wg_id < high_load_wgs:
            start_iter = max_tiles_per_wg * wg_id
            end_iter = start_iter + max_tiles_per_wg
        else:
            start_iter = (max_tiles_per_wg - 1) * (
                wg_id - high_load_wgs
            ) + high_load_wgs * max_tiles_per_wg
            end_iter = start_iter + (max_tiles_per_wg - 1)

        start_head = start_iter // tiles_per_head
        end_head = (end_iter - 1) // tiles_per_head

        # Loop through each head this workgroup touches
        for head_idx in range(start_head, end_head + 1):
            head_start_iter = head_idx * tiles_per_head
            wg_start_in_head = max(start_iter, head_start_iter)
            wg_end_in_head = min(end_iter, head_start_iter + tiles_per_head)

            if not causal:
                # For non-causal, use batch boundaries
                relative_start = wg_start_in_head - head_start_iter
                relative_end = wg_end_in_head - head_start_iter
                start_idx = bisect_right(boundaries, relative_start)
                end_idx = bisect_right(boundaries, relative_end - 1)
                total_output_tiles_for_wg += (end_idx - start_idx) + 1
            else:
                # For causal, we must loop through each batch item within the head
                tiles_per_head_per_batch = tiles_per_head // batch_size
                start_batch_idx = (wg_start_in_head - head_start_iter) // tiles_per_head_per_batch
                end_batch_idx = (wg_end_in_head - 1 - head_start_iter) // tiles_per_head_per_batch

                for batch_idx in range(start_batch_idx, end_batch_idx + 1):
                    batch_start_iter = head_start_iter + batch_idx * tiles_per_head_per_batch
                    # Get range relative to this specific batch item's workload
                    relative_start = max(wg_start_in_head, batch_start_iter) - batch_start_iter
                    relative_end = min(wg_end_in_head, batch_start_iter + tiles_per_head_per_batch) - batch_start_iter
                    
                    # Use M-block boundaries for causal
                    start_m_idx = bisect_right(boundaries, relative_start)
                    end_m_idx = bisect_right(boundaries, relative_end - 1)
                    total_output_tiles_for_wg += (end_m_idx - start_m_idx) + 1

        max_total_output_tiles = max(max_total_output_tiles, total_output_tiles_for_wg)

    return max_total_output_tiles
