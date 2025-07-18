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
    H = q.shape[1]

    qk_scale = sm_scale * 1.44269504

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
        H,
        H,
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
        batch_size=batch_size,
        causal=causal,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        # leanAttention params
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        num_stages=1,
        num_ctas=1,
    )

    print(f"la kernel {la_kernel.n_regs} registers used, {la_kernel.n_spills} spills")

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

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

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
    lean_griddimz = num_SMs
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

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

    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)
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
def find_group(x, MASKED_BLOCKS, num_m_blocks:tl.constexpr):
    total_blocks_processed = 0
    final_q_block_idx = 0
    final_task_size = 0
    final_total_blocks = 0
    found = False
    # Iterate through the tasks in the desired ping-pong order
    for i in range(0, num_m_blocks):
        pair_idx = i // 2
        if (i % 2) == 0:
            q_block_idx = pair_idx
        else:
            q_block_idx = num_m_blocks - 1 - pair_idx

        task_size = (q_block_idx + 1) * MASKED_BLOCKS

        if total_blocks_processed + task_size > x and found == False:
            final_q_block_idx, final_task_size, final_total_blocks = q_block_idx, task_size, total_blocks_processed
            found = True

        total_blocks_processed += task_size

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
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,
    stride_oh,
    stride_on,
    stride_oph,
    stride_opm,
    stride_opn,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
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

    while iter < cta_end_tile_gid:
        tile_head_idx = iter // tiles_per_head

        if causal:
            tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
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
            tile_idx = tile_head_idx * batch_size
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

        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)
    acc += tl.dot(qk.to(v.dtype), v)
        if causal:
            b_seq_size = tile_batch_idx * num_n_blocks
        else:
            tile_batch_idx = tile_idx % batch_size
            b_seq_size = 0
            if tile_batch_idx > 0:
                b_seq_size = tl.load(batch_num_block_n + tile_batch_idx - 1)

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

        # ================================================================
        #  MODIFICATION FOR PROFILING:
        #  The original online softmax logic has been replaced with a
        #  simplified (Q @ K.T) @ V to isolate dot product VGPR usage.
        # ================================================================
        
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        q = tl.load(q_ptrs)

        for l_iter in range(local_iter, local_iter_end):
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            
            # Compute Q @ K.T
            qk = tl.dot(q, k)
            
            # Compute (Q @ K.T) @ V
            # This maintains the correct shape for the accumulator.
            acc += tl.dot(qk.to(v.dtype), v)

            # update k/v pointer
            v_ptrs += BLOCK_N * stride_vn
            k_ptrs += BLOCK_N * stride_kn

        # ================================================================
        #  MODIFICATION FOR PROFILING:
        #  The following store prevents the compiler from performing
        #  Dead Code Elimination on the main loop.
        # ================================================================
        
        if local_iter == 0:
            if causal:
                 q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
            else:
                 q_idx = tile_batch_idx
            
            o_h_offs = (
                q_idx * BLOCK_M * stride_om
                + tile_head_idx * stride_oh
                + offs_m[:, None] * stride_om
                + offs_k[None, :] * stride_on
            )
            o_ptrs = Out + o_h_offs
            
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        # update iter
        iter = iter + (local_iter_end - local_iter)
