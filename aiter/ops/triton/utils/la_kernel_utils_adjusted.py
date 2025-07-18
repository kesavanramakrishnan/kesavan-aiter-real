# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import sys
import triton
import triton.language as tl


# Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs
def persistent_lean_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Mp: torch.Tensor,
    Lp: torch.Tensor,
    Op: torch.Tensor,  # (total_programs, n_ctx_q, d)
    locks: torch.Tensor,
    batch_num_block_n: torch.Tensor,
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    causal: bool,
    batch_size: int,
    sm_scale: torch.float16,
):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
    H = q.shape[1]

    BLOCK_RATIO = BLOCK_M // BLOCK_N

    qk_scale = sm_scale * 1.44269504

    (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        causal,
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        H,
        HEAD_DIM_Q,
        BLOCK_M,
        BLOCK_N,
        total_programs,
    )

    grid = (total_programs, 1, 1)

    o = torch.empty_like(q, dtype=v.dtype)
    
    # Counter for workgroup tile contributions
    wg_tile_count = torch.zeros((total_programs,), dtype=torch.int32, device=q.device)

    la_persistent[grid](
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        o,
        wg_tile_count, # Pass counter to kernel
        batch_num_block_n,
        locks,
        q.stride(0),  # N_CTX_Q
        q.stride(1),  # H
        q.stride(2),  # Head_Dim
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        Op.stride(0),  # total_programs
        Op.stride(1),  # n_ctx_q
        Op.stride(2),  # head_dim
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_RATIO=BLOCK_RATIO,
        batch_size=batch_size,
        causal=causal,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        # leanAttention params
        high_load_wgs=high_load_tbs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
    )
    # Synchronize to ensure kernel is complete before returning
    torch.cuda.synchronize()
    return o, wg_tile_count


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
    ##### Lean Atteion: Calculate Splits and Tile Sizes #####
    ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal (Ping-pong schedule)
        # The total number of tiles is the sum of workloads for each m_block
        for i in range(0, num_m_blocks):
            tiles_per_head += (i + 1) * (BLOCK_M // BLOCK_N)
        tiles_per_head = tiles_per_head * batch_size
    else:
        # Decode or Not Causal
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k  # Total tiles across all heads
    print(f"Total Lean Tiles: {total_tiles}")
    lean_griddimz = num_SMs  # CTA launch grid

    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

    # Find max number of splits
    num_splits = 0
    even_split = False
    if total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
    else:
        even_split = False
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )

    # high_load_tbs is the remainder of total_tile / num_cta
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    # Needed for causal. This is (per batch n_ctx) // BLOCK_N
    num_n_blocks = num_n_blocks // batch_size

    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
        num_splits,
        even_split,
    )

@triton.jit
def find_group(x, BLOCK_RATIO, num_m_blocks: tl.constexpr):
    """
    Finds the output tile (M-block) that the current N-block belongs to.
    This function implements a load-balancing scheduling strategy for causal attention
    by pairing the first and last M-blocks, second and second-to-last, and so on.
    """
    total_blocks_processed = 0
    final_q_block_idx = 0
    final_task_size = 0
    final_total_blocks = 0
    found = False
    # Iterate through the tasks in the desired ping-pong order
    for i in range(0, num_m_blocks):
        # Determine the actual Q block index for the current task in the ping-pong sequence
        pair_idx = i // 2
        if (i % 2) == 0:
            # Even tasks are from the top (e.g., 0, 1, 2...)
            q_block_idx = pair_idx
        else:
            # Odd tasks are from the bottom (e.g., N-1, N-2, ...)
            q_block_idx = num_m_blocks - 1 - pair_idx

        # Calculate the size of this task's workload (number of K/V blocks to process)
        task_size = (q_block_idx + 1) * BLOCK_RATIO

        # Check if the global tile `x` falls within this task's range
        if total_blocks_processed + task_size > x and found == False:
            # We found it. Return the Q index, the size of its workload, and its starting tile.
            final_q_block_idx, final_task_size, final_total_blocks = q_block_idx, task_size, total_blocks_processed
            found = True

        # Add this task's size to the running total and move to the next
        total_blocks_processed += task_size
    
    # Should not be reached if x is valid
    return final_q_block_idx, final_task_size, final_total_blocks


@triton.jit
def la_persistent(
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    wg_tile_count, # Counter for workgroup contributions
    batch_num_block_n,
    locks,
    stride_qm,  # n_ctx_q
    stride_qh,  # Head
    stride_qk,  # head_dim
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,  # n_ctx_q
    stride_oh,  # Head
    stride_on,  # head_dim
    stride_oph,  # total_programs
    stride_opm,  # n_ctx_q
    stride_opn,  # head_dim
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_RATIO: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
):
    current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (
            current_pid - high_load_wgs
        ) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

    # Loop context length
    while iter < cta_end_tile_gid:
        tile_head_idx = iter // tiles_per_head
        if causal:
            tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
            
            local_tile_x = iter - (tile_head_idx * tiles_per_head) - (tile_batch_idx * (tiles_per_head // batch_size))

            q_idx, per_head_tile_size, tile_group_start = find_group(
                local_tile_x,
                BLOCK_RATIO,
                num_m_blocks
            )
            q_idx += tile_batch_idx * num_m_blocks
            
            tile_iter = (
                tile_head_idx * tiles_per_head
                + (tile_batch_idx * (tiles_per_head // batch_size))
                + tile_group_start
            )
            tile_iter_end = tile_iter + per_head_tile_size
        else:
            tile_idx = (tile_head_idx * batch_size)
            tile_iter = tile_head_idx * tiles_per_head
            if batch_size == 1:
                req_size = tiles_per_head
            else:
                req_size = tl.load(batch_num_block_n)
            tile_iter_end = tile_iter + req_size
            for b in range(1, batch_size):
                next_req_size = tl.load(batch_num_block_n + b)
                local_head_iter = iter % tiles_per_head
                if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                    tile_iter = tile_iter + req_size
                    tile_idx = tile_idx + b
                    tile_iter_end = tile_iter + (next_req_size - req_size)
                req_size = next_req_size

        if iter == tile_iter:
            host_block = True
        else:
            host_block = False

        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

        if cta_end_tile_gid >= tile_iter_end:
            finishing_block = True
        else:
            finishing_block = False

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)

        if causal:
            b_seq_size = tile_batch_idx * num_n_blocks
        else:
            tile_batch_idx = tile_idx % batch_size
            b_seq_size = 0
            if tile_batch_idx > 0:
                b_seq_size = tl.load(batch_num_block_n + tile_batch_idx - 1)

        k_offs = ((b_seq_size + local_iter) * BLOCK_N + offs_n[None, :]) * stride_kn + tile_head_idx * stride_kh + offs_k[:, None] * stride_kk
        v_offs = ((b_seq_size + local_iter) * BLOCK_N + offs_n[:, None]) * stride_vn + tile_head_idx * stride_vh + offs_k[None, :] * stride_vk
        
        k_ptrs = K + k_offs
        v_ptrs = V + v_offs

        if not causal:
             q_idx = tile_batch_idx
             
        q_offs = (q_idx * BLOCK_M + offs_m[:, None]) * stride_qm + tile_head_idx * stride_qh + offs_k[None, :] * stride_qk
        q_ptrs = Q + q_offs

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q = tl.load(q_ptrs)

        for l_iter_offset in range(local_iter, local_iter_end):
            k = tl.load(k_ptrs)
            qk = tl.dot(q, k)
            qk = qk * qk_scale

            if causal and (BLOCK_RATIO > 1):
                if l_iter_offset == (tile_iter_end - tile_iter) - 2:
                    mask = offs_m[:, None] >= offs_n[None, :]
                    qk = tl.where(mask, qk, float("-inf"))
                if l_iter_offset == (tile_iter_end - tile_iter) - 1:
                    mask = (offs_m[:, None] >= BLOCK_N) & (offs_n[None, :] <= (offs_m[:, None] - BLOCK_N))
                    qk = tl.where(mask, qk, float("-inf"))
            if causal and (BLOCK_RATIO == 1):
                if (iter + (l_iter_offset - local_iter)) == (tile_iter_end - 1):
                    mask = offs_m[:, None] >= offs_n[None, :]
                    qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            
            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs)
            acc += tl.dot(p.to(v.dtype), v)
            
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            m_i = m_ij.to(m_i.dtype)

            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
        
        # Epilogue
        if not host_block:
            mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
            lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
            op_ptrs = Op + current_pid * stride_oph + offs_m[:, None] * stride_opm + offs_k[None, :] * stride_opn
            
            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.atomic_xchg(locks + current_pid, 1)

        if host_block:
            o_h_offs = (q_idx * BLOCK_M + offs_m[:, None]) * stride_om + tile_head_idx * stride_oh + offs_k[None, :] * stride_on
            o_ptrs = Out + o_h_offs

            if not finishing_block:
                last_cta = current_pid + 1
                temp_end_gid = cta_end_tile_gid
                split = 1
                while (split < num_splits) and (temp_end_gid < tile_iter_end):
                    if last_cta < high_load_wgs:
                        temp_end_gid += max_tiles_per_wg if (tile_iter_end - temp_end_gid) >= max_tiles_per_wg else (tile_iter_end - temp_end_gid)
                    else:
                        temp_end_gid += (max_tiles_per_wg - 1) if (tile_iter_end - temp_end_gid) >= (max_tiles_per_wg-1) else (tile_iter_end - temp_end_gid)
                    last_cta += 1
                    split += 1
                
                for cta in range((current_pid + 1), last_cta):
                    while tl.atomic_cas(locks + cta, 1, 1) != 1:
                        pass
                    
                    offs_mplp = cta * BLOCK_M + offs_m
                    mp_ptrs = Mp + offs_mplp
                    lp_ptrs = Lp + offs_mplp
                    op_h_offs = cta * stride_oph + offs_m[:, None] * stride_opm + offs_k[None, :] * stride_opn
                    op_ptrs = Op + op_h_offs

                    m_cta = tl.load(mp_ptrs)
                    l_cta = tl.load(lp_ptrs)
                    acc_cta = tl.load(op_ptrs)

                    m_new = tl.maximum(m_cta, m_i)
                    alpha = tl.math.exp2(m_cta - m_new)
                    alpha1 = tl.math.exp2(m_i - m_new)
                    l_new = alpha * l_cta + alpha1 * l_i
                    acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                    
                    m_i = m_new
                    l_i = l_new
                    
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty))

        # --- CORRECTED COUNTING LOGIC ---
        # Add the number of tiles processed in this iteration to the workgroup's total.
        tl.atomic_add(wg_tile_count + current_pid, local_iter_end)
        iter = iter + local_iter_end


def main():
    # --- CONFIGURATION CHANGED TO A LARGER SCENARIO ---
    batch = 1
    causal = True
    h = 4
    n_ctx_q = 2048
    n_ctx = [2048]
    d = 128
    total_programs = 16 # Increased number of workgroups

    init_dtype = torch.float16
    BLOCK_M = 128
    BLOCK_N = 64
    # --- END OF CONFIGURATION CHANGE ---

    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        pass

    list_num_block_n = [(int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, dtype=torch.int32, device='cuda')

    sm_scale = 0.5

    q = torch.empty((n_ctx_q * batch, h, d), device='cuda', dtype=init_dtype).normal_(mean=0.0, std=0.5)
    k = torch.empty((sum_n_ctx, h, d), device='cuda', dtype=init_dtype).normal_(mean=0.0, std=0.5)
    v = torch.empty((sum_n_ctx, h, d), device='cuda', dtype=init_dtype).normal_(mean=0.0, std=0.5)

    Mp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, BLOCK_M, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # Triton LeanAttention output
    _, wg_tile_count = persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        causal,
        batch,
        sm_scale,
    )
    
    # Print the results of the workgroup contribution counter from the host
    print("\nWork distribution for CORRECTED implementation:")
    print(wg_tile_count)


if __name__ == "__main__":
    sys.exit(main())

