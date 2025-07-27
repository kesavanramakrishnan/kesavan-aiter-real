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
    num_m_blocks = triton.cdiv(max_seqlen_q, BLOCK_M)
    num_n_blocks = triton.cdiv(max_seqlen_k, BLOCK_N)
    
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
def _bwd_dkdv_inner(
    dk,
    dv,  # output
    Q,
    k,
    v,
    DO,
    M,
    D,
    sm_scale,  # input tensor
    stride_qm,
    stride_qk,
    stride_dom,
    stride_dok,
    stride_dropoutm,
    stride_dropoutn,
    stride_deltam,
    BLOCK_M: tl.constexpr,  # 16
    BLOCK_N: tl.constexpr,  # 128
    HEAD_DIM: tl.constexpr,  #
    ACTUAL_HEAD_DIM: tl.constexpr,  #
    dropout_p,
    philox_seed,
    batch_philox_offset,
    dropout_offset,
    alibi_slope,
    seqlen_q,
    seqlen_k,  # max sequence length for q and k
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  # iteration numbers
    descale_q,
    descale_k,
    descale_v,
    descale_do,  # fp8 descale factors from user
    MASK: tl.constexpr,  # causal masking, only apply to tiles on mask diagonal
    ENABLE_DROPOUT: tl.constexpr,  # activate dropout
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,  # activate exp2
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr,
    DEBUG_TRITON_DETAIL: tl.constexpr,
):
    # if HEAD_DIM is padded
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)  # start_m + (0, 15)
    offs_n = start_n + tl.arange(0, BLOCK_N)  # start_m + (0, 127)
    offs_k = tl.arange(0, HEAD_DIM)
    # mask to make sure not OOB of seqlen_q
    mask_n = offs_n < seqlen_k
    # Q and DO are (seqlen_q, head_dim)
    # qT_ptrs = (1, BLOCK_M) + (HEAD_DIM, 1), transpose of q
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    # do_ptrs = (BLOCK_M, 1) + (1, HEAD_DIM), NOT transposed
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)

    for blk_idx in range(num_steps):
        if DEBUG_TRITON:
            print(f"iter {blk_idx}: curr_m = {curr_m}")  # noqa: E701
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        # update the mask because offs_m advanced
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < ACTUAL_HEAD_DIM
            mask_do &= offs_k[None, :] < ACTUAL_HEAD_DIM
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        # generate dropout mask
        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = (
                curr_philox_offset
                + offs_m[None, :] * stride_dropoutm
                + offs_n[:, None] * stride_dropoutn
            )
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = (
                    offs_m[None, :] * stride_dropoutm
                    + offs_n[:, None] * stride_dropoutn
                )
                dropout_mask = tl.load(curr_dropout_offset + dropout_offs, mask=mask_nm)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        if IS_FP8:
            qkT = tl.dot(k, qT) * descale_q * descale_k
        else:
            qkT = tl.dot(k, qT)
        qkT_scaled = qkT * sm_scale

        if USE_ALIBI:
            relative_pos_block = offs_n[:, None] + seqlen_q - seqlen_k - offs_m[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qkT_scaled += alibi_block

        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f"qT: {qT.shape}\n", qT)
                print(f"k: {k.shape}\n", k)
                print(f"qkT scaled: {qkT.shape}\n", qkT_scaled)
        # TODO: remove the scaling of m later when we removed re-scaling in fwd
        if USE_EXP2:
            pT = tl.math.exp2(qkT_scaled * RCP_LN2 - m[None, :] * RCP_LN2)
        else:
            pT = tl.math.exp(qkT_scaled - m[None, :])

        # Autoregressive masking.
        if MASK:
            # offset offs_m with delta_qk since the causal mask starts at
            # bottom right of the (seqlen_q, seqlen_k) matrix
            causal_mask = (offs_m[None, :] - delta_qk) >= offs_n[:, None]
            mask = causal_mask & mask_nm
            if DEBUG_TRITON_DETAIL:
                if start_n == 256:
                    print(f"causal_mask: {causal_mask.shape}\n", causal_mask)
                    print(
                        f"qkT after causal: {qkT.shape}\n",
                        tl.where(causal_mask, qkT * sm_scale, 0.0),
                    )
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        # Compute dV.
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = _compute_fp8_scaling_factors(
                    pT_dropout, FP8_MAX
                )
                dv += (
                    tl.dot((pT_dropout * scale_p_dropout).to(do.type.element_ty), do)
                    * descale_p_dropout
                    * descale_do
                )
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        else:
            if IS_FP8:
                scale_pT, descale_pT = _compute_fp8_scaling_factors(pT, FP8_MAX)
                dv += (
                    tl.dot((pT * scale_pT).to(do.type.element_ty), do)
                    * descale_pT
                    * descale_do
                )
            else:
                dv += tl.dot(pT.to(do.type.element_ty), do)

        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f"pT: {pT.shape}\n", pT)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)
        # Compute dP and dS.
        if IS_FP8:
            dpT = tl.dot(v, tl.trans(do)) * descale_v * descale_do
        else:
            dpT = tl.dot(v, tl.trans(do))
        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale
        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        if IS_FP8:
            scale_dsT, descale_dsT = _compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += (
                tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT))
                * descale_dsT
                * descale_q
            )
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qm
        do_ptrs += step_m * stride_dom
    return dk, dv


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
    if is_pod:
        current_pid = pod_pid
    else:
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
        # Calculate index of current head output tile
        # The tiles_per_head is the sum of # BLOCK_N in K/V sequence of all batches
        tile_head_idx = iter // tiles_per_head
        #TODO: add in o load
        # To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
        # [tile_iter, tile_iter_end) are in the form of global tile id
        if causal:
            tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
            # Does not support ragged batching. All requests in the batch have the same context length (per_head_tile_size)
            # tiles_per_head: total sum of # BLOCK_N in K/V sequence of all batches
            # per_head_tile_size: per head # BLOCK_N of each output tile
            per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
                iter
                - (tile_head_idx * tiles_per_head)
                - (tile_batch_idx * (tiles_per_head // batch_size)),
                MASKED_BLOCKS,
                num_m_blocks
            )
            tile_iter = (
                tile_head_idx * tiles_per_head
                + (tile_batch_idx * (tiles_per_head // batch_size))
                + total_blocks
            )
            tile_iter_end = tile_iter + (per_head_tile_size)
            tile_idx = (
                tile_head_idx * batch_size + tile_batch_idx
            ) * num_m_blocks + per_head_tile_idx
        else:
            tile_idx = (
                tile_head_idx * batch_size
            )  # Output tile idx, 1 output tile per head per batch
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
        # Local lean tile ID within a loop of an output tile
        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

        if iter == tile_iter:
            host_block = True
        else:
            host_block = False
        # finishing_block: the output tile is finished within this block
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
                b_seq_size = tl.load(
                    batch_num_block_n + tile_batch_idx - 1
                )  # Previous batch size

        k_offs = (
            (b_seq_size + local_iter) * BLOCK_N * stride_kn
            + tile_head_idx * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_k[:, None] * stride_kk
        )
        v_offs = (
            (b_seq_size + local_iter) * BLOCK_N * stride_vn
            + tile_head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )

        k_ptrs = K + k_offs
        k_ptrs = tl.multiple_of(k_ptrs, (16, 1))
        v_ptrs = V + v_offs
        v_ptrs = tl.multiple_of(v_ptrs, (1, 16))

        if causal:
            q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
        else:
            q_idx = tile_batch_idx
        q_offs = (
            q_idx * BLOCK_M * stride_qm
            + tile_head_idx * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        q_ptrs = Q + q_offs
        q_ptrs = tl.multiple_of(q_ptrs, (1, 16))

        if causal:
            q_start_m = q_idx * BLOCK_M

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q = tl.load(q_ptrs)

        for l_iter in range(local_iter, local_iter_end):
            # -- compute qk ----
            # k = tl.load(k_ptrs, cache_modifier=".cg")
            k = tl.load(k_ptrs)
            qk = tl.dot(q, k)
            qk = qk * qk_scale

            # if ((iter + (l_iter - local_iter)) == (tile_iter_end - 1)) and causal:
            #    mask = offs_m[:, None] >= offs_n[None, :]
            # Apply the causal mask
            #    qk = tl.where(mask, qk, float("-inf"))
            if causal and (MASKED_BLOCKS > 1):
                if l_iter == (tile_iter_end - tile_iter) - 2:
                    mask = offs_m[:, None] >= offs_n[None, :]
                    qk = tl.where(mask, qk, float("-inf"))
                if l_iter == (tile_iter_end - tile_iter) - 1:
                    mask = (offs_m[:, None] >= BLOCK_N) & (
                        offs_n[None, :] <= (offs_m[:, None] - BLOCK_N)
                    )
                    qk = tl.where(mask, qk, float("-inf"))

            if causal and (MASKED_BLOCKS == 1):
                # if (l_iter == (tile_iter_end - tile_iter) - 1):
                if (iter + (l_iter - local_iter)) == (tile_iter_end - 1):
                    mask = offs_m[:, None] >= offs_n[None, :]
                    qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)  # p.shape = [BLOCK_M, BLOCK_N]
            pT = tl.trans(p)  # pT.shape = [BLOCK_N, BLOCK_M]

            dV = tl.dot(pT.to(do.type.element_ty), do) # calculate dV

            v = tl.load(v_ptrs)

            dPT = tl.dot(v.to(do.dtype), tl.trans(do)) # calculate dP

            dST_block = pT * (dPT_block - delta[None, :])

            dk_block = tl.dot(dST_block.to(q.dtype), q) * sm_scale

            # 5. Atomically add the block gradients to global memory
            dk_store_ptrs = dk_ptr + (k_ptrs - k_ptr)
            dv_store_ptrs = dv_ptr + (v_ptrs - v_ptr)

            tl.atomic_add(dk_store_ptrs, dk_block.to(dk_ptr.dtype.element_ty), mask=offs_n[:, None] < max_seqlen_k)
            tl.atomic_add(dv_store_ptrs, dv_block.to(dv_ptr.dtype.element_ty), mask=offs_n[:, None] < max_seqlen_k)

            # -- update output accumulator --
            alpha = tl.math.exp2(m_i - m_ij)
            acc = (
                acc * alpha[:, None]
            )  # Scale each row of acc by the corresponding elements in alpha
            # v = tl.load(v_ptrs, cache_modifier=".cg")  # v.shape = [BLOCK_N, HEAD_DIM]
            v = tl.load(v_ptrs)
            acc += tl.dot(p.to(v.dtype), v)  # acc.shape = [BLOCK_M, HEAD_DIM]
            # -- update l_i
            l_ij = tl.sum(p, 1)  # rowsum(p)
            l_i = l_i * alpha + l_ij
            # update m_i
            m_i = m_ij.to(m_i.dtype)

            # if (
            #     (l_iter == (tile_iter_end - tile_iter) - 1)
            #     and (iter == tile_iter_end - 1)
            #     and (MASKED_BLOCKS == 2)
            # ):
            #     mask1 = offs_m >= BLOCK_N
            #     m_i = tl.where(mask1, m_i, float("-inf"))
            #     l_i = tl.where(mask1, l_i, 1.0)
            #     mask1 = mask1[:, None]
            #     acc = tl.where(mask1, acc, 0.0)

            # update k/v pointer
            v_ptrs += BLOCK_N * stride_vn
            k_ptrs += BLOCK_N * stride_kn

        # initialize pointer to m and l
        m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # lean output tile epilogue
        if not host_block:
            # Update pointers of partial results Mp[cta], Lp[cta], Op[cta]
            mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
            lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
            # op_ptrs = (
            #     Op
            #     + current_pid * stride_oph  # stride_oph is total_program dimension
            #     + offs_m[:, None] * stride_opm
            #     + offs_k[None, :] * stride_opn
            # )

            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
            tl.debug_barrier()
            # tl.store(locks + current_pid, 1, cache_modifier=".wt")
            # According to streamK gemm, store + cache_modifier won't work universally
            # atomic_xchg is better solution but a less performant variant
            tl.atomic_xchg(locks + current_pid, 1)

        if host_block:  # and finishing_block:
            # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
            # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction

            o_h_offs = (
                q_idx * BLOCK_M * stride_om
                + tile_head_idx * stride_oh
                + offs_m[:, None] * stride_om
                + offs_k[None, :] * stride_on
            )
            o_ptrs = Out + o_h_offs

            if not finishing_block:
                # if host not finishing_block: # another CTA is processing the end of the output tile and store partial results

                last_cta = current_pid + 1
                temp_end_gid = cta_end_tile_gid
                split = 1
                while (split < num_splits) and (temp_end_gid < tile_iter_end):
                    if last_cta < high_load_wgs:
                        if (tile_iter_end - temp_end_gid) < max_tiles_per_wg:
                            temp_end_gid += tile_iter_end - temp_end_gid
                        else:
                            temp_end_gid += max_tiles_per_wg
                    else:
                        if (tile_iter_end - temp_end_gid) < (max_tiles_per_wg - 1):
                            temp_end_gid += tile_iter_end - temp_end_gid
                        else:
                            temp_end_gid += max_tiles_per_wg - 1

                    last_cta += 1
                    split += 1
                # Next, load nonHost partial restult
                for cta in range((current_pid + 1), last_cta):
                    # According to treamK gemm, atomic_cas is universal solution but less performant
                    while tl.atomic_cas(locks + cta, 1, 1) != 1:
                        # while tl.load(locks + cta, cache_modifier=".cv", volatile=True) != 1:
                        pass

                    # Partial results are stored in [nonHost, Host-nonFinishing] layout
                    offs_mplp = cta * BLOCK_M + offs_m
                    mp_ptrs = Mp + offs_mplp
                    lp_ptrs = Lp + offs_mplp
                    op_h_offs = (
                        cta * stride_oph
                        + offs_m[:, None] * stride_opm
                        + offs_k[None, :] * stride_opn
                    )
                    op_ptrs = Op + op_h_offs

                    m_cta = tl.load(mp_ptrs)
                    l_cta = tl.load(lp_ptrs)
                    acc_cta = tl.load(op_ptrs)

                    # m_i is the host CTA's m, m_cta is other nonHost CTA's m
                    m_new = tl.maximum(m_cta, m_i)
                    alpha = tl.math.exp2(m_cta - m_new)
                    alpha1 = tl.math.exp2(m_i - m_new)
                    l_new = alpha * l_cta + alpha1 * l_i
                    acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                    # update m, l
                    m_i = m_new
                    l_i = l_new
            # host CTA write final result to memory
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        # update iter
        iter = iter + (local_iter_end - local_iter)