# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Lean Attention -- BACKWARD PASS
===============
This file adapts the Lean Attention forward pass to perform the backward pass.
It uses the same Stream-K style tiling strategy to efficiently compute 
gradients (dQ, dK, dV) using atomic additions for reduction.

The core idea is to linearize the backward pass workload (all q_block vs k_block
interactions) and distribute it evenly across all SMs.
"""

import torch
import triton
import triton.language as tl

# ==============================================================================
# Backward Pass Python Launcher
# ==============================================================================

def persistent_lean_attention_backward(
    # Core tensors for backward pass
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    softmax_lse: torch.Tensor,
    # Output gradient tensors
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    # Forward-pass style parameters
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    causal: bool,
    sm_scale: torch.float16,
    num_warps: int,
    waves_per_eu: int,
):
    """
    Main Python launcher for the Lean Attention backward pass.
    """
    # shape constraints
    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM in {16, 32, 64, 128, 256}

    batch_size, max_seqlen_q, num_heads, _ = q.shape
    _, max_seqlen_k, num_heads_k, _ = k.shape

    # 1. Pre-computation of Delta
    delta = torch.empty_like(softmax_lse)
    grid_pre = (triton.cdiv(max_seqlen_q, BLOCK_M), batch_size, num_heads)
    _bwd_preprocess_lean[grid_pre](
        out, do, delta,
        out.stride(0), out.stride(2), out.stride(1), out.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        max_seqlen_q,
        BLOCK_M=BLOCK_M,
        HEAD_DIM=HEAD_DIM,
    )

    # 2. Calculate Stream-K splits for the main backward workload.
    (
        total_tiles,
        tiles_per_cta,
        high_load_wgs,
        grid_size,
        num_m_blocks,
        num_n_blocks,
        tiles_per_head,
        tiles_per_batch
    ) = get_num_splits_and_buffer_sizes_bwd(
        causal, batch_size, max_seqlen_q, max_seqlen_k,
        num_heads, num_heads_k, BLOCK_M, BLOCK_N, total_programs
    )

    # 3. Launch the main backward kernel.
    grid = (grid_size, 1, 1)
    la_persistent_bwd[grid](
        # Inputs
        q, k, v, do, delta, softmax_lse,
        # Outputs
        dq, dk, dv,
        # Strides
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        do.stride(0), do.stride(2), do.stride(1), do.stride(3),
        dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3),
        dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3),
        dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3),
        delta.stride(0), delta.stride(1), delta.stride(2),
        # Dimensions
        max_seqlen_q, max_seqlen_k, num_heads, num_heads_k,
        # Block sizes & Compile-Time Constants
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
        NUM_M_BLOCKS=num_m_blocks, NUM_N_BLOCKS=num_n_blocks,
        TILES_PER_HEAD=tiles_per_head, TILES_PER_BATCH=tiles_per_batch,
        # Lean / Stream-K parameters
        total_tiles=total_tiles,
        tiles_per_cta=tiles_per_cta,
        high_load_wgs=high_load_wgs,
        sm_scale=sm_scale,
        causal=causal,
        num_warps=num_warps,
    )
    return

# ==============================================================================
# Workload Calculation Function
# ==============================================================================

def get_num_splits_and_buffer_sizes_bwd(
    causal, batch_size, max_seqlen_q, max_seqlen_k,
    num_heads, num_heads_k, BLOCK_M, BLOCK_N, num_SMs,
):
    grid_size = num_SMs
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
    
    # Assert that divisors will be non-zero
    assert num_m_blocks > 0 and num_n_blocks > 0 and num_heads > 0
    
    tiles_per_head = num_m_blocks * num_n_blocks
    tiles_per_batch = num_heads * tiles_per_head
    total_tiles = batch_size * tiles_per_batch

    tiles_per_cta = total_tiles // grid_size
    high_load_wgs = total_tiles % grid_size
    
    return total_tiles, tiles_per_cta, high_load_wgs, grid_size, num_m_blocks, num_n_blocks, tiles_per_head, tiles_per_batch

# ==============================================================================
# Triton Kernels for Backward Pass
# ==============================================================================

@triton.jit
def _bwd_preprocess_lean(
    o_ptr, do_ptr, delta_ptr,
    stride_o_b, stride_o_h, stride_o_m, stride_o_k,
    stride_delta_b, stride_delta_h, stride_delta_m,
    max_seqlen_q,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < max_seqlen_q

    o_offs = (bid * stride_o_b + hid * stride_o_h +
              offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k)
    do_offs = (bid * stride_o_b + hid * stride_o_h +
               offs_m[:, None] * stride_o_m + offs_k[None, :] * stride_o_k)

    o = tl.load(o_ptr + o_offs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptr + do_offs, mask=mask_m[:, None], other=0.0)

    delta = tl.sum(o * do, axis=1)

    delta_offs = (bid * stride_delta_b + hid * stride_delta_h + offs_m * stride_delta_m)
    tl.store(delta_ptr + delta_offs, delta, mask=mask_m)


@triton.jit
def la_persistent_bwd(
    # Pointers to Tensors
    q_ptr, k_ptr, v_ptr, do_ptr, delta_ptr, lse_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_deltab, stride_deltah, stride_deltam,
    # Dimensions
    max_seqlen_q, max_seqlen_k, num_heads, num_heads_k,
    # Block sizes & Compile-Time Constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr, NUM_N_BLOCKS: tl.constexpr,
    TILES_PER_HEAD: tl.constexpr, TILES_PER_BATCH: tl.constexpr,
    # Lean Attention parameters
    total_tiles: tl.constexpr,
    tiles_per_cta: tl.constexpr,
    high_load_wgs: tl.constexpr,
    sm_scale: tl.constexpr,
    causal: tl.constexpr,
):
    pid = tl.program_id(0)

    # 1. Determine the range of tiles for this specific CTA.
    num_tiles_per_cta = tl.where(pid < high_load_wgs, tiles_per_cta + 1, tiles_per_cta)
    start_tile = tl.where(pid < high_load_wgs, pid * (tiles_per_cta + 1),
                          high_load_wgs * (tiles_per_cta + 1) + (pid - high_load_wgs) * tiles_per_cta)
    end_tile = start_tile + num_tiles_per_cta

    # 2. Main loop over the assigned range of tiles.
    for current_tile in range(start_tile, end_tile):
        # a. Map the linear `current_tile` index back to the 4D grid.
        batch_idx = current_tile // TILES_PER_BATCH
        head_tile_idx = current_tile % TILES_PER_BATCH
        head_idx = head_tile_idx // TILES_PER_HEAD
        block_tile_idx = head_tile_idx % TILES_PER_HEAD
        m_block_idx = block_tile_idx // NUM_N_BLOCKS
        n_block_idx = block_tile_idx % NUM_N_BLOCKS
        
        # --- Core Gradient Computation for one tile ---
        offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = n_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        
        q_ptrs = q_ptr + (batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        k_ptrs = k_ptr + (batch_idx * stride_kb + head_idx * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd)
        v_ptrs = v_ptr + (batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        do_ptrs = do_ptr + (batch_idx * stride_dob + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
        lse_ptrs = lse_ptr + batch_idx * stride_deltab + head_idx * stride_deltah + offs_m
        delta_ptrs = delta_ptr + batch_idx * stride_deltab + head_idx * stride_deltah + offs_m
        
        mask_m = offs_m < max_seqlen_q
        mask_n = offs_n < max_seqlen_k
        
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        k_tile = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse_tile = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        delta_tile = tl.load(delta_ptrs, mask=mask_m, other=0.0)
        
        qk_tile = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        if causal: qk_tile += tl.where((offs_m[:, None] >= offs_n[None, :]), 0, float("-inf"))
        p_tile = tl.exp(qk_tile - lse_tile[:, None])
        
        partial_dv = tl.dot(tl.trans(p_tile.to(do_tile.dtype)), do_tile)
        dp_tile = tl.dot(do_tile, tl.trans(v_tile))
        ds_tile = p_tile * (dp_tile - delta_tile[:, None])
        partial_dq = tl.dot(ds_tile.to(k_tile.dtype), k_tile) * sm_scale
        partial_dk = tl.dot(tl.trans(ds_tile.to(q_tile.dtype)), q_tile) * sm_scale
        
        # --- Reduction using atomic adds ---
        dq_out_ptrs = dq_ptr + (batch_idx * stride_dqb + head_idx * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd)
        dk_out_ptrs = dk_ptr + (batch_idx * stride_dkb + head_idx * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd)
        dv_out_ptrs = dv_ptr + (batch_idx * stride_dvb + head_idx * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd)

        tl.atomic_add(dq_out_ptrs, partial_dq, mask=mask_m[:, None])
        tl.atomic_add(dk_out_ptrs, partial_dk, mask=mask_n[:, None])
        tl.atomic_add(dv_out_ptrs, partial_dv, mask=mask_n[:, None])
