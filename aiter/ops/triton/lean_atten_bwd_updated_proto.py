# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Lean Attention Backward Pass - DK/DV Kernel
===========================================
This file contains the Triton implementation for the dk/dv kernel of the
Lean Attention backward pass. It uses a persistent, Stream-K style approach
similar to the forward pass, with computations adapted to the transposed domain
for maximum efficiency.

Reference forward pass: lean_atten_val.py
Reference backward logic: mha_onekernel_bwd.py
"""

import torch
import triton
import triton.language as tl

# This function computes D = rowsum(dO * O)
# It's a necessary preprocessing step for the main backward pass.
@triton.jit
def _bwd_preprocess(
    o_ptr,
    do_ptr,
    delta_ptr,
    stride_o_b,
    stride_o_h,
    stride_o_m,
    stride_o_k,
    stride_delta_b,
    stride_delta_h,
    stride_delta_m,
    # FP8-related parameters would go here if needed
    seqlen_q,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Each program instance computes a block of the D vector.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # For simplicity, this reference does not handle varlen or FP8 yet.
    # It assumes a single batch and head for stride calculation passed from the host.
    # The full implementation would handle batch and head indexing.
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # Pointers to the current block of O and dO
    o_ptrs = o_ptr + pid_bh * stride_o_h + (offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k)
    do_ptrs = do_ptr + pid_bh * stride_o_h + (offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k)

    # Boundary checks
    mask_m = offs_m < seqlen_q
    mask_mk = mask_m[:, None] & (offs_k[None, :] < BLOCK_DMODEL)

    # Load O and dO
    o = tl.load(o_ptrs, mask=mask_mk, other=0.0)
    do = tl.load(do_ptrs, mask=mask_mk, other=0.0)

    # Compute D (delta)
    delta = tl.sum(o * do, axis=1)

    # Write D back to global memory
    delta_ptrs = delta_ptr + pid_bh * stride_delta_h + offs_m
    tl.store(delta_ptrs, delta, mask=mask_m)


def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
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

    # print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
    # print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
    # print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    # print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
        # Does not support ragged batch for causal.
        tiles_per_head = tiles_per_head * batch_size
    else:
        # Decode or Not Causal
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k  # Total tiles across all heads

    # StreamK Lean has as many threadblocks as SMs
    # This should be a function of tile size and number of scratchpad space
    # LeanAttention assign 3 CTAs per SM (bounded by LDS size)
    lean_griddimz = num_SMs  # CTA launch grid
    # if (total_tiles <= 2 * 2 * num_SMs):
    #    lean_griddimz = min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks)
    # else:
    #    lean_griddimz = min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks)

    # Max number lean tiles per task block (CTA)
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
def _bwd_dkdv_persistent_inner(
    DK, DV, Q, K, V, DO, L, D,
    dKp, dVp, locks, # Partial results and synchronization
    stride_dk_bh, stride_dk_n, stride_dk_d,
    stride_dv_bh, stride_dv_n, stride_dv_d,
    stride_q_bh, stride_q_m, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_do_bh, stride_do_m, stride_do_d,
    stride_l_bh, stride_l_m,
    stride_d_bh, stride_d_m,
    stride_dkp_g, stride_dkp_n, stride_dkp_d, # Strides for partials
    stride_dvp_g, stride_dvp_n, stride_dvp_d,
    sm_scale,
    seqlen_q, seqlen_k,
    # Lean Attention Scheduling Params
    iter, cta_end_tile_gid, current_pid,
    tiles_per_head, num_m_blocks, num_splits, high_load_wgs, max_tiles_per_wg,
    # Triton Kernel Params
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    This is the inner logic for the persistent dk/dv kernel.
    It computes a partial result for a dK/dV output block based on one Q block.
    The full dK/dV is the sum of these partial results over all Q blocks.
    """
    # Identify which head and which k-block this thread block works on
    tile_head_idx = iter // tiles_per_head # This is the combined (batch, k_head) index

    # In the backward pass for dk/dv, the "output" is a block of dk/dv,
    # which corresponds to a block of k/v. The reduction is over the q dimension.
    # So, a "LeanTile" processes one (k_block, q_block) pair.
    k_block_idx = (iter % tiles_per_head) // num_m_blocks
    q_block_idx = (iter % tiles_per_head) % num_m_blocks

    # The final output tile we are contributing to is indexed by head and k_block
    output_tile_idx = tile_head_idx * num_m_blocks + k_block_idx

    # Determine if this CTA is the "host" for this output tile.
    # The host is the first CTA to work on this tile (i.e., q_block_idx == 0)
    # and is responsible for the final reduction and write-back.
    host_block = (q_block_idx == 0)

    # --- Initial Setup ---
    offs_n = k_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # --- Load K and V for the current k-block ---
    # These will be kept in SRAM for the duration of this CTA's work on this tile.
    mask_n = offs_n < seqlen_k
    mask_nk = mask_n[:, None] & (offs_d[None, :] < HEAD_DIM)
    
    k_ptrs = K + tile_head_idx * stride_k_bh + (offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d)
    v_ptrs = V + tile_head_idx * stride_v_bh + (offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d)
    k = tl.load(k_ptrs, mask=mask_nk, other=0.0)
    v = tl.load(v_ptrs, mask=mask_nk, other=0.0)

    # --- GQA Logic: Accumulate over the Q-head group ---
    Q_GROUP_SIZE: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    
    # Absolute k_head and batch index
    k_head_abs = tile_head_idx % NUM_K_HEADS
    batch_idx = tile_head_idx // NUM_K_HEADS
    
    # Starting absolute q_head index for this group
    q_head_start_abs = k_head_abs * Q_GROUP_SIZE

    # Accumulators for this k_block, summed over the q_head group for this q_block
    dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    for q_head_offset in range(Q_GROUP_SIZE):
        q_head_abs = q_head_start_abs + q_head_offset
        q_head_bh_idx = batch_idx * NUM_Q_HEADS + q_head_abs

        # --- Load data for the current q-block and q-head ---
        mask_m = offs_m < seqlen_q
        mask_md = mask_m[:, None] & (offs_d[None, :] < HEAD_DIM)
        
        q_ptrs = Q + q_head_bh_idx * stride_q_bh + (offs_m[None, :] * stride_q_m + offs_d[:, None] * stride_q_d)
        q_t = tl.load(q_ptrs, mask=mask_md.T, other=0.0) # Load Q transposed: d x m

        do_ptrs = DO + q_head_bh_idx * stride_do_bh + (offs_m[:, None] * stride_do_m + offs_d[None, :] * stride_do_d)
        do = tl.load(do_ptrs, mask=mask_md, other=0.0) # Load dO: m x d

        l_t_ptrs = L + q_head_bh_idx * stride_l_bh + offs_m
        l_t = tl.load(l_t_ptrs, mask=mask_m, other=0.0) # Load L_transpose: m

        d_t_ptrs = D + q_head_bh_idx * stride_d_bh + offs_m
        d_t = tl.load(d_t_ptrs, mask=mask_m, other=0.0) # Load D_transpose: m

        # --- Core Computations in Transposed Domain ---
        s_t = tl.dot(k, q_t) * sm_scale
        p_t = tl.exp(s_t - l_t[None, :])
        p_t = tl.where(mask_n[:, None] & mask_m[None, :], p_t, 0.0)

        # 1. Accumulate dV = P_transpose @ dO
        dv_acc += tl.dot(p_t.to(do.dtype), do)

        # 2. Compute dP_transpose = V @ dO.T
        dp_t = tl.dot(v, tl.trans(do))

        # 3. Compute dS_transpose = P_transpose * (dP_transpose - D_transpose)
        ds_t = p_t * (dp_t - d_t[None, :])

        # 4. Accumulate dK = dS_transpose @ Q
        dk_acc += tl.dot(ds_t.to(q_t.dtype), tl.trans(q_t)) * sm_scale

    # --- Reduction and Write-back ---
    if not host_block:
        # This is a non-host block, so we write our partial results to temporary storage.
        dkp_ptrs = dKp + current_pid * stride_dkp_g + (offs_n[:, None] * stride_dkp_n + offs_d[None, :] * stride_dkp_d)
        dvp_ptrs = dVp + current_pid * stride_dvp_g + (offs_n[:, None] * stride_dvp_n + offs_d[None, :] * stride_dvp_d)
        
        tl.store(dkp_ptrs, dk_acc)
        tl.store(dvp_ptrs, dv_acc)
        
        # Signal that this CTA is done with its partial computation
        tl.atomic_xchg(locks + current_pid, 1)

    if host_block:
        # This is the host block. It will accumulate its own results and then
        # wait for and accumulate results from other contributing CTAs.
        
        last_cta = current_pid
        temp_end_gid = iter # Start from the current tile
        
        # Determine the range of other CTAs contributing to this output tile.
        for i in range(num_m_blocks - 1):
            temp_end_gid += 1
            if temp_end_gid >= cta_end_tile_gid:
                last_cta += 1
                if last_cta < high_load_wgs:
                    temp_end_gid = max_tiles_per_wg * last_cta
                else:
                    temp_end_gid = (max_tiles_per_wg - 1) * (last_cta - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        
        # Wait for non-host CTAs and reduce their partial results
        for cta_id in range(current_pid + 1, last_cta + 1):
             # This logic assumes a simple mapping, a more robust lock/signal is needed for complex schedules
            if cta_id < tl.num_programs(0):
                while tl.atomic_cas(locks + cta_id, 1, 1) != 1:
                    pass # Spin-wait for the lock
                
                # Load partial dK and dV
                dkp_ptrs = dKp + cta_id * stride_dkp_g + (offs_n[:, None] * stride_dkp_n + offs_d[None, :] * stride_dkp_d)
                dvp_ptrs = dVp + cta_id * stride_dvp_g + (offs_n[:, None] * stride_dvp_n + offs_d[None, :] * stride_dvp_d)

                dk_partial = tl.load(dkp_ptrs)
                dv_partial = tl.load(dvp_ptrs)

                # Accumulate the results
                dk_acc += dk_partial
                dv_acc += dv_partial
        
        # Host block writes the final result to global memory
        dk_ptrs = DK + tile_head_idx * stride_dk_bh + (offs_n[:, None] * stride_dk_n + offs_d[None, :] * stride_dk_d)
        dv_ptrs = DV + tile_head_idx * stride_dv_bh + (offs_n[:, None] * stride_dv_n + offs_d[None, :] * stride_dv_d)
        
        tl.store(dk_ptrs, dk_acc.to(DK.dtype.element_ty), mask=mask_nk)
        tl.store(dv_ptrs, dv_acc.to(DV.dtype.element_ty), mask=mask_nk)

@triton.jit
def la_bwd_dkdv_persistent(
    DK, DV, Q, K, V, DO, L, D,
    dKp, dVp, locks, # Partial results and synchronization
    stride_dk_bh, stride_dk_n, stride_dk_d,
    stride_dv_bh, stride_dv_n, stride_dv_d,
    stride_q_bh, stride_q_m, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_do_bh, stride_do_m, stride_do_d,
    stride_l_bh, stride_l_m,
    stride_d_bh, stride_d_m,
    stride_dkp_g, stride_dkp_n, stride_dkp_d, # Strides for partials
    stride_dvp_g, stride_dvp_n, stride_dvp_d,
    sm_scale,
    seqlen_q, seqlen_k,
    # Lean Attention Scheduling Params
    high_load_wgs, max_tiles_per_wg, tiles_per_head, num_m_blocks, num_splits, max_output_tile_cnt,
    # Triton Kernel Params
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Persistent kernel for the backward pass (dK, dV).
    This kernel launches one CTA per SM and each CTA processes a fixed number of LeanTiles.
    """
    current_pid = tl.program_id(0)

    # Calculate the range of global tile IDs this CTA is responsible for
    if current_pid < high_load_wgs:
        iter_start = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter_start + max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (current_pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter_start + (max_tiles_per_wg - 1)
    
    iter = iter_start
    # This loop is unrolled by the Triton compiler. Each iteration processes one LeanTile.
    for i in tl.static_range(max_output_tile_cnt + 1):
        if iter < cta_end_tile_gid:
            _bwd_dkdv_persistent_inner(
                DK, DV, Q, K, V, DO, L, D,
                dKp, dVp, locks,
                stride_dk_bh, stride_dk_n, stride_dk_d,
                stride_dv_bh, stride_dv_n, stride_dv_d,
                stride_q_bh, stride_q_m, stride_q_d,
                stride_k_bh, stride_k_n, stride_k_d,
                stride_v_bh, stride_v_n, stride_v_d,
                stride_do_bh, stride_do_m, stride_do_d,
                stride_l_bh, stride_l_m,
                stride_d_bh, stride_d_m,
                stride_dkp_g, stride_dkp_n, stride_dkp_d,
                stride_dvp_g, stride_dvp_n, stride_dvp_d,
                sm_scale, seqlen_q, seqlen_k,
                iter, cta_end_tile_gid, current_pid,
                tiles_per_head, num_m_blocks, num_splits, high_load_wgs, max_tiles_per_wg,
                NUM_Q_HEADS=NUM_Q_HEADS,
                NUM_K_HEADS=NUM_K_HEADS,
                HEAD_DIM=HEAD_DIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )
            iter += 1
