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


# =================================================================================================
# Host-Side Launcher
# =================================================================================================
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
    """
    Host-side launcher for the Lean Attention kernel.

    This function sets up and launches the Triton kernel for Lean Attention. It handles
    the calculation of grid dimensions, buffer sizes, and other parameters required
    by the kernel, based on the Stream-K algorithm.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        Mp (torch.Tensor): Temporary buffer for partial max values.
        Lp (torch.Tensor): Temporary buffer for partial sum-of-exponentials values.
        Op (torch.Tensor): Temporary buffer for partial output values.
        locks (torch.Tensor): Tensor for synchronization locks between thread blocks.
        batch_num_block_n (torch.Tensor): Tensor containing the number of N-blocks for each item in the batch.
        total_programs (int): The number of persistent programs to launch, typically matching the number of SMs.
        BLOCK_M (int): The block size for the M dimension (query sequence length).
        BLOCK_N (int): The block size for the N dimension (key/value sequence length).
        causal (bool): If True, applies a causal mask for autoregressive attention.
        batch_size (int): The number of sequences in the batch.
        sm_scale (torch.float16): The scale factor for the softmax.
        num_warps (int): The number of warps per thread block.
        waves_per_eu (int): Waves per execution unit, for performance tuning.
    """
    # Shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
    MASKED_BLOCKS = BLOCK_M // BLOCK_N

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

    # Launch the persistent kernel
    la_kernel = la_persistent[grid](
        is_pod=False,
        pod_pid=0,
        Q=q, K=k, V=v,
        qk_scale=qk_scale,
        Mp=Mp, Lp=Lp, Op=Op, Out=o,
        batch_num_block_n=batch_num_block_n,
        locks=locks,
        stride_qm=q.stride(0), stride_qh=q.stride(1), stride_qk=q.stride(2),
        stride_kn=k.stride(0), stride_kh=k.stride(1), stride_kk=k.stride(2),
        stride_vn=v.stride(0), stride_vh=v.stride(1), stride_vk=v.stride(2),
        stride_om=o.stride(0), stride_oh=o.stride(1), stride_on=o.stride(2),
        stride_oph=Op.stride(0), stride_opm=Op.stride(1), stride_opn=Op.stride(2),
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MASKED_BLOCKS=MASKED_BLOCKS,
        batch_size=batch_size,
        causal=causal,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
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


# =================================================================================================
# Host-Side Helper for Work Distribution
# =================================================================================================
def get_num_splits_and_buffer_sizes(
    causal, batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k,
    BLOCK_M, BLOCK_N, num_SMs
):
    """
    Calculates the distribution of work (tiles) among thread blocks (CTAs)
    based on the Stream-K algorithm.
    """
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal: Calculate tiles for the triangular attention matrix.
        for i in range(num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
        tiles_per_head *= batch_size
    else:
        # Decode or Not Causal: Full rectangular attention matrix.
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k
    lean_griddimz = num_SMs
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

    # Find max number of splits required for reduction
    if total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + (num_n_blocks + max_tiles_per_tb - 2) // max_tiles_per_tb
    else:
        even_split = False
        num_splits = 1 + (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)

    high_load_tbs = total_tiles - (max_tiles_per_tb - 1) * lean_griddimz
    num_n_blocks = num_n_blocks // batch_size

    return (
        num_m_blocks, num_n_blocks, high_load_tbs, max_tiles_per_tb,
        tiles_per_head, lean_griddimz, num_splits, even_split
    )


# =================================================================================================
# Triton JIT Helper Functions
# =================================================================================================
@triton.jit
def find_group(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
    """
    Calculates the q_block_idx, task_size, and starting tile offset for causal attention
    using a closed-form mathematical formula.
    """
    tiles_per_pair = (num_m_blocks + 1) * MASKED_BLOCKS
    pair_idx = x // tiles_per_pair
    base_tiles = pair_idx * tiles_per_pair
    x_local = x - base_tiles
    task1_size = (pair_idx + 1) * MASKED_BLOCKS
    is_in_task1 = x_local < task1_size

    q_block_idx = tl.where(is_in_task1, pair_idx, num_m_blocks - 1 - pair_idx)
    task_size = tl.where(is_in_task1, task1_size, (num_m_blocks - pair_idx) * MASKED_BLOCKS)
    total_blocks = tl.where(is_in_task1, base_tiles, base_tiles + task1_size)

    if num_m_blocks % 2 != 0:
        num_pairs = num_m_blocks // 2
        total_tiles_in_pairs = num_pairs * tiles_per_pair
        is_in_middle_block = x >= total_tiles_in_pairs
        middle_q_idx = num_m_blocks // 2
        middle_task_size = (middle_q_idx + 1) * MASKED_BLOCKS
        q_block_idx = tl.where(is_in_middle_block, middle_q_idx, q_block_idx)
        task_size = tl.where(is_in_middle_block, middle_task_size, task_size)
        total_blocks = tl.where(is_in_middle_block, total_tiles_in_pairs, total_blocks)

    return q_block_idx, task_size, total_blocks


@triton.jit
def _attention_step(q, k_ptrs, v_ptrs, m_i, l_i, acc, qk_scale,
                    causal, q_start_m, b_seq_size, l_iter,
                    offs_m, offs_n, BLOCK_M, BLOCK_N, HEAD_DIM):
    """
    Performs one step of the attention calculation for a single BLOCK_N block.
    """
    k = tl.load(k_ptrs)
    qk = tl.dot(q, k) * qk_scale

    if causal:
        k_start_n = (b_seq_size + l_iter) * BLOCK_N
        mask = (q_start_m + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
        qk = tl.where(mask, qk, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))

    # Safe exp2 calculation to avoid NaNs
    p_arg = qk - m_ij[:, None]
    p_arg = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), p_arg)
    p = tl.math.exp2(p_arg)

    # Safe scaling factor calculation
    alpha_arg = m_i - m_ij
    alpha_arg = tl.where(m_ij == float("-inf"), 0.0, alpha_arg)
    alpha = tl.math.exp2(alpha_arg)

    # Update accumulator
    acc = acc * alpha[:, None]
    v = tl.load(v_ptrs)
    acc += tl.dot(p.to(v.dtype), v)

    # Update stats
    l_ij = tl.sum(p, 1)
    l_i = l_i * alpha + l_ij
    m_i = m_ij.to(m_i.dtype)

    return m_i, l_i, acc


@triton.jit
def _write_partial_results(Mp, Lp, Op, m_i, l_i, acc, locks, current_pid,
                           offs_m, offs_k, stride_oph, stride_opm, stride_opn,
                           BLOCK_M, HEAD_DIM):
    """
    For non-host blocks, writes partial results to global memory and signals completion.
    """
    mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
    lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
    op_ptrs = (Op + current_pid * stride_oph +
               offs_m[:, None] * stride_opm + offs_k[None, :] * stride_opn)

    tl.store(mp_ptrs, m_i, cache_modifier=".wt")
    tl.store(lp_ptrs, l_i, cache_modifier=".wt")
    tl.store(op_ptrs, acc, cache_modifier=".wt")
    tl.debug_barrier()
    tl.atomic_xchg(locks + current_pid, 1)


@triton.jit
def _reduce_and_write_final_results(Out, o_ptrs, finishing_block, m_i, l_i, acc,
                                    current_pid, high_load_wgs, max_tiles_per_wg,
                                    num_splits, tile_iter_end, cta_end_tile_gid,
                                    locks, Mp, Lp, Op, offs_m, offs_k,
                                    stride_oph, stride_opm, stride_opn,
                                    BLOCK_M, HEAD_DIM):
    """
    For host blocks, waits for non-hosts, reduces all partial results, and writes the final output.
    """
    if not finishing_block:
        # Determine which other CTAs are working on this same output tile
        last_cta = current_pid + 1
        temp_end_gid = cta_end_tile_gid
        split = 1
        while (split < num_splits) and (temp_end_gid < tile_iter_end):
            if last_cta < high_load_wgs:
                temp_end_gid += max_tiles_per_wg
            else:
                temp_end_gid += max_tiles_per_wg - 1
            last_cta += 1
            split += 1

        # Spin-wait on locks and reduce partial results
        for cta in range(current_pid + 1, last_cta):
            while tl.atomic_cas(locks + cta, 1, 1) != 1:
                pass

            # Load partial results from non-host CTA
            offs_mplp = cta * BLOCK_M + offs_m
            op_h_offs = (cta * stride_oph +
                         offs_m[:, None] * stride_opm + offs_k[None, :] * stride_opn)
            mp_ptrs = Mp + offs_mplp
            lp_ptrs = Lp + offs_mplp
            op_ptrs = Op + op_h_offs

            m_cta = tl.load(mp_ptrs)
            l_cta = tl.load(lp_ptrs)
            acc_cta = tl.load(op_ptrs)

            # Combine partial results with host's results
            m_new = tl.maximum(m_cta, m_i)
            alpha = tl.math.exp2(m_cta - m_new)
            alpha1 = tl.math.exp2(m_i - m_new)
            l_new = alpha * l_cta + alpha1 * l_i
            acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]

            # Update host's running stats
            m_i = m_new
            l_i = l_new

    # Finalize and write output
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(Out.type.element_ty))


# =================================================================================================
# Main Triton Kernel
# =================================================================================================
@triton.jit
def la_persistent(
    is_pod, pod_pid,
    Q, K, V, qk_scale,
    Mp, Lp, Op, Out,
    batch_num_block_n, locks,
    stride_qm, stride_qh, stride_qk,
    stride_kn, stride_kh, stride_kk,
    stride_vn, stride_vh, stride_vk,
    stride_om, stride_oh, stride_on,
    stride_oph, stride_opm, stride_opn,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr, batch_size: tl.constexpr, causal: tl.constexpr,
    num_m_blocks: tl.constexpr, num_n_blocks: tl.constexpr,
    high_load_wgs: tl.constexpr, max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr, num_splits: tl.constexpr
):
    """
    Triton kernel for Lean Attention.
    """
    current_pid = tl.program_id(0) if not is_pod else pod_pid

    # Determine the range of global tiles this CTA is responsible for.
    if current_pid < high_load_wgs:
        iter_start = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter_start + max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (current_pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter_start + (max_tiles_per_wg - 1)

    iter = iter_start
    while iter < cta_end_tile_gid:
        # 1. Work Assignment: Map global tile ID to specific Q/K/V blocks
        tile_head_idx = iter // tiles_per_head
        if causal:
            tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
            per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
                iter - (tile_head_idx * tiles_per_head) - (tile_batch_idx * (tiles_per_head // batch_size)),
                MASKED_BLOCKS, num_m_blocks
            )
            tile_iter = (tile_head_idx * tiles_per_head +
                         (tile_batch_idx * (tiles_per_head // batch_size)) + total_blocks)
            tile_iter_end = tile_iter + per_head_tile_size
            tile_idx = (tile_head_idx * batch_size + tile_batch_idx) * num_m_blocks + per_head_tile_idx
        else:  # Decode case
            tile_idx = tile_head_idx * batch_size
            tile_iter = tile_head_idx * tiles_per_head
            req_size = tl.load(batch_num_block_n)
            tile_iter_end = tile_iter + req_size
            for b in range(1, batch_size):
                next_req_size = tl.load(batch_num_block_n + b)
                local_head_iter = iter % tiles_per_head
                if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                    tile_iter += req_size
                    tile_idx += b
                    tile_iter_end = tile_iter + (next_req_size - req_size)
                req_size = next_req_size

        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter
        host_block = (iter == tile_iter)
        finishing_block = (cta_end_tile_gid >= tile_iter_end)

        # 2. Setup Pointers and Offsets
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, HEAD_DIM)

        if causal:
            b_seq_size = tile_batch_idx * num_n_blocks
            q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
            q_start_m = q_idx * BLOCK_M
        else:  # Decode
            tile_batch_idx = tile_idx % batch_size
            b_seq_size = 0
            if tile_batch_idx > 0:
                b_seq_size = tl.load(batch_num_block_n + tile_batch_idx - 1)
            q_idx = tile_batch_idx
            q_start_m = 0 # Not used in decode

        k_offs = ((b_seq_size + local_iter) * BLOCK_N * stride_kn + tile_head_idx * stride_kh +
                  offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk)
        v_offs = ((b_seq_size + local_iter) * BLOCK_N * stride_vn + tile_head_idx * stride_vh +
                  offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        q_offs = (q_idx * BLOCK_M * stride_qm + tile_head_idx * stride_qh +
                  offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

        k_ptrs = K + k_offs
        v_ptrs = V + v_offs
        q_ptrs = Q + q_offs

        # 3. Core Attention Calculation
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        q = tl.load(q_ptrs)

        for l_iter_offset in range(local_iter_end - local_iter):
            l_iter_current = local_iter + l_iter_offset
            m_i, l_i, acc = _attention_step(
                q, k_ptrs, v_ptrs, m_i, l_i, acc, qk_scale,
                causal, q_start_m, b_seq_size, l_iter_current,
                offs_m, offs_n, BLOCK_M, BLOCK_N, HEAD_DIM
            )
            # Update k/v pointers for next iteration
            v_ptrs += BLOCK_N * stride_vn
            k_ptrs += BLOCK_N * stride_kn

        # 4. Reduction and Output
        if not host_block:
            _write_partial_results(
                Mp, Lp, Op, m_i, l_i, acc, locks, current_pid,
                offs_m, offs_k, stride_oph, stride_opm, stride_opn,
                BLOCK_M, HEAD_DIM
            )
        else:  # host_block
            o_h_offs = (q_idx * BLOCK_M * stride_om + tile_head_idx * stride_oh +
                        offs_m[:, None] * stride_om + offs_k[None, :] * stride_on)
            o_ptrs = Out + o_h_offs
            _reduce_and_write_final_results(
                Out, o_ptrs, finishing_block, m_i, l_i, acc,
                current_pid, high_load_wgs, max_tiles_per_wg,
                num_splits, tile_iter_end, cta_end_tile_gid,
                locks, Mp, Lp, Op, offs_m, offs_k,
                stride_oph, stride_opm, stride_opn,
                BLOCK_M, HEAD_DIM
            )

        # Update main loop iterator
        iter += (local_iter_end - local_iter)

