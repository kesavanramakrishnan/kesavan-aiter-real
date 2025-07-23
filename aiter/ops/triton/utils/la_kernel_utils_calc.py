#!/usr/bin/env python3
import sys
import collections

# This is a direct Python port of the get_num_splits_and_buffer_sizes function
# from the user's script to ensure calculations are identical.
def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    head_size,
    BLOCK_M,
    BLOCK_N,
    num_SMs,
):
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        for i in range(0, num_m_blocks):
            tiles_per_head += (i + 1) * (BLOCK_M // BLOCK_N)
        tiles_per_head = tiles_per_head * batch_size
    else:
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k
    lean_griddimz = num_SMs
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
    )

# This is a direct Python port of the find_group function from the Triton kernel.
def find_group_py(x, BLOCK_RATIO, num_m_blocks):
    total_blocks_processed = 0
    for i in range(num_m_blocks):
        pair_idx = i // 2
        q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
        task_size = (q_block_idx + 1) * BLOCK_RATIO
        if total_blocks_processed + task_size > x:
            return q_block_idx, task_size, total_blocks_processed
        total_blocks_processed += task_size
    # Should not be reached if x is valid
    return num_m_blocks - 1, 0, total_blocks_processed

def simulate_cpu_work_distribution(
    batch, causal, h, n_ctx_q, n_ctx, d, total_programs, BLOCK_M, BLOCK_N
):
    """
    Simulates the work distribution of the la_persistent Triton kernel on the CPU.
    It counts the number of UNIQUE OUTPUT TILES each workgroup (pid) processes.
    """
    # --- Setup: Replicate host-side calculations ---
    sum_n_ctx = sum(n_ctx)
    BLOCK_RATIO = BLOCK_M // BLOCK_N

    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        _, # lean_griddimz is not needed for the loop simulation
    ) = get_num_splits_and_buffer_sizes(
        causal, batch, n_ctx_q, sum_n_ctx, h, h, d, BLOCK_M, BLOCK_N, total_programs
    )

    print("--- CPU Simulation Parameters ---")
    print(f"num_m_blocks: {num_m_blocks}, tiles_per_head: {tiles_per_head}")
    print(f"high_load_wgs: {high_load_wgs}, max_tiles_per_wg: {max_tiles_per_wg}\n")

    # This dictionary will store a set of unique output tiles for each pid.
    output_tiles_per_pid = collections.defaultdict(set)

    # --- Simulation: Loop over each workgroup (pid) ---
    for pid in range(total_programs):
        # Determine the start and end global tile IDs for this pid
        if pid < high_load_wgs:
            start_iter = max_tiles_per_wg * pid
            end_iter = start_iter + max_tiles_per_wg
        else:
            start_iter = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
            end_iter = start_iter + (max_tiles_per_wg - 1)

        # Iterate over every single lean tile assigned to this workgroup
        for iter_gid in range(start_iter, end_iter):
            # --- Replicate kernel's index calculation logic ---
            tile_head_idx = iter_gid // tiles_per_head
            
            if causal:
                tile_batch_idx = (iter_gid % tiles_per_head) // (tiles_per_head // batch)
                local_tile_x = iter_gid - (tile_head_idx * tiles_per_head) - (tile_batch_idx * (tiles_per_head // batch))
                
                # Find which M-block (q_block) this lean tile belongs to
                q_block_idx, _, _ = find_group_py(
                    local_tile_x, BLOCK_RATIO, num_m_blocks
                )
                
                # An output tile is uniquely identified by its head, batch, and M-block index
                output_tile_id = (tile_head_idx, tile_batch_idx, q_block_idx)
                output_tiles_per_pid[pid].add(output_tile_id)
            else: 
                # In the non-causal case, all lean tiles for a given head/batch
                # contribute to the same single output tile.
                tile_batch_idx = (iter_gid % tiles_per_head) // num_n_blocks
                output_tile_id = (tile_head_idx, tile_batch_idx, 0) # m_block is always 0
                output_tiles_per_pid[pid].add(output_tile_id)

    # Convert the dictionary of sets into a list of counts.
    final_counts = [len(output_tiles_per_pid[i]) for i in range(total_programs)]
    return final_counts


def main():
    # --- Use the same configuration as the user's test script ---
    batch = 1
    causal = True
    h = 4
    n_ctx_q = 2048
    n_ctx = [2048]
    d = 128
    total_programs = 16
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Run the simulation
    cpu_counts = simulate_cpu_work_distribution(
        batch, causal, h, n_ctx_q, n_ctx, d, total_programs, BLOCK_M, BLOCK_N
    )

    # Print the final result
    print("--- CPU Simulation Result ---")
    print("This array shows the number of UNIQUE OUTPUT TILES processed by each workgroup (pid).")
    print(cpu_counts)


if __name__ == "__main__":
    sys.exit(main())
