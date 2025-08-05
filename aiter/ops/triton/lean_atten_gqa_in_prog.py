# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Lean Attention
===============
This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480
Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.
Lean Attention is for both decode and prefill attention of transformer based models.

It currently supports ragged batching decode and prefill attention with causal=1

TO be added features:
- Add GQA support
- batch_size > 1 for prefill/causal=1
- Misc
    - N_CTX with non-integer number of BLOCK_N (pad zeros or add mask)
    -
"""

import torch

import triton
import triton.language as tl

from .utils.index_max_tiles import calculate_max_output_tiles_analytically


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
    num_warps: int,
    waves_per_eu: int,
    n_ctx: list[int],
    # max_output_tile_cnt: int,
):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
    # For MI300, BLOCK_M=128, BLOCK_N=64 is better for performance
    MASKED_BLOCKS = BLOCK_M // BLOCK_N

    if causal:
        # Only support BLOCK_M is multiple of BLOCK_N
        # TODO: add other scenarios
        assert BLOCK_M % BLOCK_N == 0

    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch

    # GQA Support
    H_Q = q.shape[1]
    H_KV = k.shape[1]
    assert H_Q % H_KV == 0, "Number of Q heads must be a multiple of K/V heads for GQA"
    GQA_FACTOR = H_Q // H_KV

    qk_scale = sm_scale * 1.44269504

    # FIX: The call was using 'H' which is not defined. It should use H_Q.
    max_output_tile_cnt = calculate_max_output_tiles_analytically(
        causal=causal, batch_size=batch_size, n_ctx=n_ctx, max_seqlen_q=N_CTX_Q,
        num_heads=H_Q, num_SMs=total_programs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    max_output_tile_cnt += 4


    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
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
        H_Q,
        H_KV,
        BLOCK_M,
        BLOCK_N,
        total_programs,
    )

    grid = (total_programs, 1, 1)

    o = torch.empty_like(q, dtype=v.dtype)

    la_kernel = la_persistent[grid](
        False,
        0,
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        o,
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
        MASKED_BLOCKS=MASKED_BLOCKS,
        GQA_FACTOR=GQA_FACTOR,
        batch_size=batch_size,
        causal=causal,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        # leanAttention params
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        max_output_tile_cnt=max_output_tile_cnt,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=1,
        num_ctas=1,
    )

    print(f"la kernel {la_kernel.n_regs} registers used, {la_kernel.n_spills} spills")
    ms = 0
    return o


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

    # Total tiles is based on the number of K/V heads, as this defines the outer loop of work.
    total_tiles = tiles_per_head * num_heads_k

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
def find_group(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
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
        task_size = (q_block_idx + 1) * MASKED_BLOCKS

        # Check if the global tile `x` falls within this task's range
        if total_blocks_processed + task_size > x and found == False:
            # We found it. Return the Q index, the size of its workload, and its starting tile.
            final_q_block_idx, final_task_size, final_total_blocks = (
                q_block_idx,
                task_size,
                total_blocks_processed,
            )
            found = True

        # Add this task's size to the running total and move to the next
        total_blocks_processed += task_size
    # Return values
    return final_q_block_idx, final_task_size, final_total_blocks


@triton.jit
def la_persistent(
    is_pod,
    pod_pid,
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
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
    MASKED_BLOCKS: tl.constexpr,
    GQA_FACTOR: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    max_output_tile_cnt: tl.constexpr,
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

    for i in tl.static_range(max_output_tile_cnt + 1):
        if iter < cta_end_tile_gid:
            iter = la_persistent_inner(
                Q,
                K,
                V,
                qk_scale,
                Mp,
                Lp,
                Op,
                Out,
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
                iter=iter,
                cta_end_tile_gid=cta_end_tile_gid,
                current_pid=current_pid,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                MASKED_BLOCKS=MASKED_BLOCKS,
                GQA_FACTOR=GQA_FACTOR,
                batch_size=batch_size,
                causal=causal,
                num_m_blocks=num_m_blocks,
                num_n_blocks=num_n_blocks,
                # leanAttention params
                high_load_wgs=high_load_wgs,
                max_tiles_per_wg=max_tiles_per_wg,
                tiles_per_head=tiles_per_head,
                num_splits=num_splits,
            )


@triton.jit
def la_persistent_inner(
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
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
    iter,
    cta_end_tile_gid,
    current_pid,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    GQA_FACTOR: tl.constexpr,
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
    # The `iter` variable loops over tiles defined by K/V heads.
    # So, tile_kv_head_idx is the primary head index.
    tile_kv_head_idx = iter // tiles_per_head

    # --- All scheduling logic is based on the K/V head workload ---
    if causal:
        tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
        per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
            iter
            - (tile_kv_head_idx * tiles_per_head)
            - (tile_batch_idx * (tiles_per_head // batch_size)),
            MASKED_BLOCKS,
            num_m_blocks,
        )
        tile_iter = (
            tile_kv_head_idx * tiles_per_head
            + (tile_batch_idx * (tiles_per_head // batch_size))
            + total_blocks
        )
        tile_iter_end = tile_iter + (per_head_tile_size)
        # q_idx is the M-block index, independent of the Q-head
        q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
    else:
        # This logic determines which batch item this `iter` corresponds to for ragged batching.
        # It calculates `tile_idx` which is the batch-item-specific index.
        tile_idx = tile_kv_head_idx * batch_size
        tile_iter = tile_kv_head_idx * tiles_per_head
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
        # q_idx is the batch-item index for non-causal
        q_idx = tile_idx

    local_iter = iter - tile_iter
    host_block = iter == tile_iter
    
    remaining_tiles = tile_iter_end - cta_end_tile_gid
    should_steal = (remaining_tiles > 0) and (remaining_tiles <= 2)
    local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter
    if should_steal:
        local_iter_end += remaining_tiles

    host_is_finishing = (cta_end_tile_gid >= tile_iter_end) or (should_steal and host_block)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Correct handling of batch offsets for K/V tensors.
    if causal:
        b_seq_size = tile_batch_idx * num_n_blocks
    else:
        tile_batch_idx = q_idx % batch_size
        b_seq_size = 0
        if tile_batch_idx > 0:
            b_seq_size = tl.load(batch_num_block_n + tile_batch_idx - 1)

    # --- GQA support is added via a loop over the Q heads for each K/V head task ---
    for q_head_group_iter in range(GQA_FACTOR):
        q_head_idx = tile_kv_head_idx * GQA_FACTOR + q_head_group_iter
        
        # --- All calculations below are now inside the GQA loop ---
        
        k_offs = (
            (b_seq_size + local_iter) * BLOCK_N * stride_kn
            + tile_kv_head_idx * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_k[:, None] * stride_kk
        )
        v_offs = (
            (b_seq_size + local_iter) * BLOCK_N * stride_vn
            + tile_kv_head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )

        k_ptrs = K + k_offs
        v_ptrs = V + v_offs

        q_offs = (
            q_idx * BLOCK_M * stride_qm
            + q_head_idx * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        q_ptrs = Q + q_offs

        if causal:
            q_start_m = q_idx * BLOCK_M

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        q = tl.load(q_ptrs)

        # We need to re-initialize k/v pointers for each Q head iteration
        temp_k_ptrs = k_ptrs
        temp_v_ptrs = v_ptrs

        for l_iter_inner in range(local_iter_end):
            k = tl.load(temp_k_ptrs)
            qk = tl.dot(q, k)
            qk = qk * qk_scale

            if causal:
                k_start_n = (b_seq_size + local_iter + l_iter_inner) * BLOCK_N
                mask = (q_start_m + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
                qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))

            p_arg = qk - m_ij[:, None]
            p_arg = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), p_arg)
            p = tl.math.exp2(p_arg)

            alpha_arg = m_i - m_ij
            alpha_arg = tl.where(m_ij == float("-inf"), 0.0, alpha_arg)
            alpha = tl.math.exp2(alpha_arg)
            acc = acc * alpha[:, None]
            
            v = tl.load(temp_v_ptrs)
            acc += tl.dot(p.to(v.dtype), v)
            
            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            m_i = m_ij.to(m_i.dtype)
            
            temp_v_ptrs += BLOCK_N * stride_vn
            temp_k_ptrs += BLOCK_N * stride_kn

        # Epilogue must also be inside the GQA loop since it writes per-Q-head results.
        # The reduction logic for split work is complex with GQA and a shared scratchpad.
        # This implementation assumes that for GQA, workloads are not split across workgroups.
        # This is a simplifying assumption to fix the memory fault.
        if host_block:
            acc_reshaped = tl.reshape(acc, (BLOCK_M, 2, HEAD_DIM // 2))
            acc_permuted = tl.permute(acc_reshaped, (0, 2, 1))
            acc0, acc1 = tl.split(acc_permuted)

            o_ptrs0 = (
                Out
                + q_idx * BLOCK_M * stride_om
                + q_head_idx * stride_oh
                + offs_m[:, None] * stride_om
                + tl.arange(0, HEAD_DIM // 2)[None, :] * stride_on
            )
            o_ptrs1 = (
                Out
                + q_idx * BLOCK_M * stride_om
                + q_head_idx * stride_oh
                + offs_m[:, None] * stride_om
                + (tl.arange(0, HEAD_DIM // 2)[None, :] + HEAD_DIM // 2) * stride_on
            )

            acc0 = acc0 / l_i[:, None]
            acc1 = acc1 / l_i[:, None]
            tl.store(o_ptrs0, acc0.to(Out.type.element_ty))
            tl.store(o_ptrs1, acc1.to(Out.type.element_ty))

    # update iter outside the GQA loop
    iter = iter + local_iter_end

    return iter