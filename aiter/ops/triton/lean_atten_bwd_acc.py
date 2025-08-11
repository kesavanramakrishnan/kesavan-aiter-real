import torch
import functools
from typing import Optional
import triton
import triton.language as tl
import aiter.ops.triton.utils.arch_info as arch_info

# Lean attention backward pass (Triton)
# - Computes dQ, dK, dV with a fused kernel per-CTA: dK/dV first, then dQ.
# - dK/dV inner loop operates in the transposed domain (tile K, iterate Q^T)
#   to improve locality and simplify accumulation shapes.
# - Supports causal and non-causal (ragged) sequences via `batch_num_block_n`.
# - Uses softmax LSE `M` from forward for numerical stability when reconstructing P.


@triton.jit
def find_group(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
    """
    Returns a zig-zag grouping index for m-blocks. Currently unused in this file,
    but kept for potential scheduling experiments.
    """
    total_blocks_processed = 0
    final_q_block_idx = 0
    final_task_size = 0
    final_total_blocks = 0
    found = False
    for i in range(0, num_m_blocks):
        pair_idx = i // 2
        if (i % 2) == 0:
            q_block_idx = pair_idx
        else:
            q_block_idx = num_m_blocks - 1 - pair_idx
        task_size = (q_block_idx + 1) * MASKED_BLOCKS
        if total_blocks_processed + task_size > x and not found:
            final_q_block_idx, final_task_size, final_total_blocks = (
                q_block_idx,
                task_size,
                total_blocks_processed,
            )
            found = True
        total_blocks_processed += task_size
    return final_q_block_idx, final_task_size, final_total_blocks


def get_num_splits_and_buffer_sizes_bwd(
    causal: bool,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_heads_k: int,
    BLOCK_M: int,
    BLOCK_N: int,
    num_SMs: int,
):
    """
    Compute a safe set of scheduling parameters for the backward pass.
    Emphasizes robustness for small workloads (avoids div-by-zero) and returns:
      - num_m_blocks, num_n_blocks (per-block counts)
      - high_load_wgs, max_tiles_per_wg (for even tile distribution across CTAs)
      - tiles_per_head, total_programs, num_splits, even_split
    """
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    if max_seqlen_q == 1:
        causal = False

    # Adjust by GQA if used (we're focusing core functionality; treat heads equal)
    max_seqlen_q_eff = max_seqlen_q * num_heads // max(1, num_heads_k)

    # tiles per head
    tiles_per_head = 0
    if causal:
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
        tiles_per_head *= batch_size
    else:
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * max(1, num_heads_k)

    # Grid size
    if total_tiles <= 0:
        # No work: return safe defaults
        return (
            num_m_blocks,
            max(1, num_n_blocks // max(1, batch_size)),
            0,
            1,
            tiles_per_head,
            1,
            1,
            True,
        )

    total_programs = max(1, num_SMs)
    max_tiles_per_tb = (total_tiles + total_programs - 1) // total_programs
    # Splits
    if max_tiles_per_tb <= 1:
        num_splits = 1
        even_split = (total_tiles % total_programs) == 0
    else:
        if (total_tiles % total_programs) == 0:
            even_split = True
            num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
        else:
            even_split = False
            # Denominator guaranteed >= 1 here
            num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1))

    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * total_programs)

    # Needed for causal. This is (per batch n_ctx) // BLOCK_N
    num_n_blocks_causal = num_n_blocks // max(1, batch_size)

    return (
        num_m_blocks,
        num_n_blocks_causal,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    )


@triton.jit
def la_bwd_dkdv_inner(
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DK,
    DV,
    batch_num_block_n,
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_dom,
    stride_doh,
    stride_dok,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_deltab,
    stride_deltah,
    stride_deltam,
    stride_dkn_out,
    stride_dkh_out,
    stride_dkk_out,
    stride_dvn_out,
    stride_dvh_out,
    stride_dvk_out,
    sm_scale,
    N_CTX_Q,
    total_n_blocks_all_batches,
    total_tiles_kv,
    high_load_wgs_kv,
    max_tiles_per_wg_kv,
    H: tl.constexpr,
    B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Compute dK and dV by tiling over K/V-blocks and iterating over Q^T tiles.
    Shapes per tile:
      - K/V tile: (BLOCK_N, HEAD_DIM)
      - Q_T, DO_T: (HEAD_DIM, BLOCK_M)
      - p_T, ds_T: (BLOCK_N, BLOCK_M)
      - Accumulators dV/dK: (BLOCK_N, HEAD_DIM)
    Each CTA owns disjoint KV tiles (no atomics needed) and loops over all Q m-blocks
    for the mapped batch/head.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Compiler hints to help vectorization
    tl.assume(stride_qm > 0)
    tl.assume(stride_qh > 0)
    tl.assume(stride_qk > 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_kk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_dom > 0)
    tl.assume(stride_doh > 0)
    tl.assume(stride_dok > 0)
    tl.assume(stride_mb > 0)
    tl.assume(stride_mh > 0)
    tl.assume(stride_mm > 0)
    tl.assume(stride_deltab > 0)
    tl.assume(stride_deltah > 0)
    tl.assume(stride_deltam > 0)
    tl.assume(stride_dkn_out > 0)
    tl.assume(stride_dkh_out > 0)
    tl.assume(stride_dkk_out > 0)
    tl.assume(stride_dvn_out > 0)
    tl.assume(stride_dvh_out > 0)
    tl.assume(stride_dvk_out > 0)

    # 1/ln(2) for fast exp2 softmax
    RCP_LN2: tl.constexpr = 1.4426950408889634

    # Distribute KV tiles across CTAs with an even remainder split
    if pid < high_load_wgs_kv:
        iter_start_kv = max_tiles_per_wg_kv * pid
        num_to_process_kv = max_tiles_per_wg_kv
    else:
        iter_start_kv = (max_tiles_per_wg_kv - 1) * (pid - high_load_wgs_kv) + high_load_wgs_kv * max_tiles_per_wg_kv
        num_to_process_kv = max_tiles_per_wg_kv - 1

    for i in range(0, num_to_process_kv):
        iter_kv = iter_start_kv + i
        valid_kv = iter_kv < total_tiles_kv

        # Map linear KV tile id -> (head_idx, n_linear) and then to (batch_idx, n_block_in_batch)
        kv_id = iter_kv
        tile_head_idx_raw = kv_id % H
        head_idx = tl.minimum(tile_head_idx_raw, H - 1)
        n_linear = kv_id // H

        batch_idx = 0
        n_block_in_batch = 0
        if CAUSAL:
            # Uniform n-blocks per batch in causal decode
            num_n_blocks_per_batch = total_n_blocks_all_batches // B
            batch_idx = n_linear // num_n_blocks_per_batch
            n_block_in_batch = n_linear % num_n_blocks_per_batch
            b_seq_size = batch_idx * num_n_blocks_per_batch
        else:
            # Ragged non-causal: find which batch this linear n-block falls into via cumulative counts
            prev_running_blocks = 0
            match_prev_blocks = 0
            found = 0
            for b in range(0, B):
                blocks_total = tl.load(batch_num_block_n + b, mask=valid_kv, other=0) if b > 0 else tl.load(batch_num_block_n, mask=valid_kv, other=0)
                is_match = (found == 0) & (n_linear < blocks_total)
                batch_idx = tl.where(is_match, b, batch_idx)
                match_prev_blocks = tl.where(is_match, prev_running_blocks, match_prev_blocks)
                found = tl.where(is_match, 1, found)
                prev_running_blocks = blocks_total
            n_block_in_batch = n_linear - match_prev_blocks
            b_seq_size = 0 if batch_idx == 0 else tl.load(batch_num_block_n + batch_idx - 1, mask=valid_kv, other=0)

        # Load K/V tile for this kv (BLOCK_N, HEAD_DIM)
        k_offs = (
            (b_seq_size + n_block_in_batch) * BLOCK_N * stride_kn
            + head_idx * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_k[None, :] * stride_kk
        )
        v_offs = (
            (b_seq_size + n_block_in_batch) * BLOCK_N * stride_vn
            + head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )
        k_tile = tl.load(K + k_offs, mask=valid_kv, other=0.0)
        v_tile = tl.load(V + v_offs, mask=valid_kv, other=0.0)

        k_start_abs = (b_seq_size + n_block_in_batch) * BLOCK_N

        dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        # Iterate over all Q m-blocks for this batch/head using pointer increments
        num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
        # Base absolute q row start for this batch
        q_start_abs = batch_idx * N_CTX_Q
        # Build base pointers once
        qT_ptrs = (
            Q
            + q_start_abs * stride_qm
            + head_idx * stride_qh
            + offs_k[:, None] * stride_qk
            + offs_m[None, :] * stride_qm
        )
        doT_ptrs = (
            DO
            + q_start_abs * stride_dom
            + head_idx * stride_doh
            + offs_k[:, None] * stride_dok
            + offs_m[None, :] * stride_dom
        )
        m_ptrs = (
            M
            + batch_idx * stride_mb
            + head_idx * stride_mh
            + offs_m * stride_mm
        )
        delta_ptrs = (
            Delta
            + batch_idx * stride_deltab
            + head_idx * stride_deltah
            + offs_m * stride_deltam
        )
        # Alignment hints
        qT_ptrs = tl.multiple_of(qT_ptrs, (16, 1))
        doT_ptrs = tl.multiple_of(doT_ptrs, (16, 1))
        for _ in range(0, num_m_blocks_total):
            q_tile_T = tl.load(qT_ptrs, mask=valid_kv)
            do_tile_T = tl.load(doT_ptrs, mask=valid_kv)
            m_rows = tl.load(m_ptrs, mask=(valid_kv & (offs_m < BLOCK_M)), other=-float("inf"))
            delta_rows = tl.load(delta_ptrs, mask=(valid_kv & (offs_m < BLOCK_M)), other=0.0)

            # Reconstruct logits and probabilities in the transposed domain using exp2
            qk_T = tl.dot(k_tile, q_tile_T) * sm_scale
            p_T = tl.math.exp2(qk_T * RCP_LN2 - m_rows[None, :] * RCP_LN2)

            if CAUSAL:
                mask_T = (q_start_abs + offs_m[None, :]) >= (k_start_abs + offs_n[:, None])
                p_T = tl.where(mask_T & valid_kv, p_T, 0.0)

            dp_T = tl.dot(v_tile, do_tile_T)
            ds_T = p_T * (dp_T - delta_rows[None, :])

            # Accumulate in (BLOCK_N, HEAD_DIM)
            dv_acc += tl.dot(p_T.to(do_tile_T.type.element_ty), tl.trans(do_tile_T))
            dk_acc += tl.dot(ds_T.to(q_tile_T.type.element_ty), tl.trans(q_tile_T))

            # Advance pointers and absolute row start
            q_start_abs += BLOCK_M
            qT_ptrs += BLOCK_M * stride_qm
            doT_ptrs += BLOCK_M * stride_dom
            m_ptrs += BLOCK_M * stride_mm
            delta_ptrs += BLOCK_M * stride_deltam

        # Store accumulated dV and scaled dK for this KV tile
        dv_ptrs_out = (
            DV
            + (b_seq_size + n_block_in_batch) * BLOCK_N * stride_dvn_out
            + head_idx * stride_dvh_out
            + offs_n[:, None] * stride_dvn_out
            + offs_k[None, :] * stride_dvk_out
        )
        dk_ptrs_out = (
            DK
            + (b_seq_size + n_block_in_batch) * BLOCK_N * stride_dkn_out
            + head_idx * stride_dkh_out
            + offs_n[:, None] * stride_dkn_out
            + offs_k[None, :] * stride_dkk_out
        )
        dv_ptrs_out = tl.multiple_of(dv_ptrs_out, (1, 16))
        dk_ptrs_out = tl.multiple_of(dk_ptrs_out, (1, 16))
        tl.store(dv_ptrs_out, dv_acc.to(DV.type.element_ty), mask=valid_kv)
        tl.store(dk_ptrs_out, (dk_acc * sm_scale).to(DK.type.element_ty), mask=valid_kv)


@triton.jit
def la_bwd_dq_inner(
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DQ,
    batch_num_block_n,
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_dom,
    stride_doh,
    stride_dok,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_deltab,
    stride_deltah,
    stride_deltam,
    stride_dqm,
    stride_dqh,
    stride_dqk,
    sm_scale,
    N_CTX_Q,
    total_n_blocks_all_batches,
    total_tiles_q,
    high_load_wgs_q,
    max_tiles_per_wg_q,
    H: tl.constexpr,
    B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Compute dQ by tiling over Q-blocks and streaming through all K/V tiles for
    the mapped batch/head. Reconstructs probabilities via LSE for stability and
    applies causal masking when required.
    """
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Compiler hints
    tl.assume(stride_qm > 0)
    tl.assume(stride_qh > 0)
    tl.assume(stride_qk > 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_kk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_dom > 0)
    tl.assume(stride_doh > 0)
    tl.assume(stride_dok > 0)
    tl.assume(stride_mb > 0)
    tl.assume(stride_mh > 0)
    tl.assume(stride_mm > 0)
    tl.assume(stride_deltab > 0)
    tl.assume(stride_deltah > 0)
    tl.assume(stride_deltam > 0)
    tl.assume(stride_dqm > 0)
    tl.assume(stride_dqh > 0)
    tl.assume(stride_dqk > 0)

    RCP_LN2: tl.constexpr = 1.4426950408889634

    # Distribute Q tiles across CTAs with an even remainder split
    if pid < high_load_wgs_q:
        iter_start_q = max_tiles_per_wg_q * pid
        num_to_process_q = max_tiles_per_wg_q
    else:
        iter_start_q = (max_tiles_per_wg_q - 1) * (pid - high_load_wgs_q) + high_load_wgs_q * max_tiles_per_wg_q
        num_to_process_q = max_tiles_per_wg_q - 1

    # Map tile id -> (batch_idx, head_idx, m_block) helpers
    num_m_blocks_total_q = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    tiles_per_batch = num_m_blocks_total_q * H

    for i in range(0, num_to_process_q):
        iter_q = iter_start_q + i
        valid_q = iter_q < total_tiles_q

        # Map Q tile id to (batch_idx, head_idx, m_block)
        tile_id = iter_q
        batch_idx = tile_id // tiles_per_batch
        rem = tile_id % tiles_per_batch
        head_idx = rem // num_m_blocks_total_q
        m_block = rem % num_m_blocks_total_q

        batch_idx = tl.where(valid_q, batch_idx, 0)
        head_idx = tl.where(valid_q, head_idx, 0)
        m_block = tl.where(valid_q, m_block, 0)

        # Q/DO/M/Delta pointers for this Q tile
        q_start_abs = batch_idx * N_CTX_Q + m_block * BLOCK_M
        q_ptrs = Q + q_start_abs * stride_qm + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        do_ptrs = DO + q_start_abs * stride_dom + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + (m_block * BLOCK_M + offs_m) * stride_mm
        delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + (m_block * BLOCK_M + offs_m) * stride_deltam

        q_tile = tl.load(q_ptrs, mask=valid_q)
        do_tile = tl.load(do_ptrs, mask=valid_q)
        m_rows = tl.load(m_ptrs, mask=(valid_q & (offs_m < BLOCK_M)), other=-float("inf"))
        delta_rows = tl.load(delta_ptrs, mask=(valid_q & (offs_m < BLOCK_M)), other=0.0)

        # Determine this batch's K/V n-block range
        num_n_blocks_per_batch = 0
        b_seq_size_blocks = 0
        if CAUSAL:
            num_n_blocks_per_batch = total_n_blocks_all_batches // B
            b_seq_size_blocks = batch_idx * num_n_blocks_per_batch
        else:
            # Ragged non-causal: derive per-batch block range from cumulative counts
            blocks_prev = 0
            for b in range(0, B):
                blocks_total = tl.load(batch_num_block_n + b, mask=valid_q, other=0) if b > 0 else tl.load(batch_num_block_n, mask=valid_q, other=0)
                if b == batch_idx:
                    num_n_blocks_per_batch = blocks_total - blocks_prev
                    b_seq_size_blocks = blocks_prev
                blocks_prev = blocks_total

        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Build base K/V pointers and absolute K start for causal mask
        k_ptrs = (
            K
            + (b_seq_size_blocks) * BLOCK_N * stride_kn
            + head_idx * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_k[None, :] * stride_kk
        )
        v_ptrs = (
            V
            + (b_seq_size_blocks) * BLOCK_N * stride_vn
            + head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )
        # Alignment hints
        k_ptrs = tl.multiple_of(k_ptrs, (1, 16))
        v_ptrs = tl.multiple_of(v_ptrs, (1, 16))
        k_start_abs = b_seq_size_blocks * BLOCK_N

        # Stream across all K/V tiles for this batch/head
        for _ in range(0, num_n_blocks_per_batch):
            k_tile = tl.load(k_ptrs, mask=valid_q, other=0.0)
            v_tile = tl.load(v_ptrs, mask=valid_q, other=0.0)

            # Reconstruct logits and probabilities using exp2
            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            p = tl.math.exp2(qk * RCP_LN2 - m_rows[:, None] * RCP_LN2)

            if CAUSAL:
                mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :])
                p = tl.where(mask, p, 0.0)

            # Form dS and accumulate contribution to dQ
            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = p * (dp - delta_rows[:, None])
            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

            # Advance pointers and absolute K start
            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            k_start_abs += BLOCK_N

        # Final scale and store dQ tile
        dq_acc = dq_acc * sm_scale
        dq_ptrs_out = (
            DQ
            + q_start_abs * stride_dqm
            + head_idx * stride_dqh
            + offs_m[:, None] * stride_dqm
            + offs_k[None, :] * stride_dqk
        )
        tl.store(dq_ptrs_out, dq_acc.to(DQ.type.element_ty), mask=valid_q)


@triton.jit
def la_bwd_fused_streamk(
    # tensors
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DQ,
    DK,
    DV,
    batch_num_block_n,
    # strides
    stride_qm,
    stride_qh,
    stride_qk,
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_dom,
    stride_doh,
    stride_dok,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_deltab,
    stride_deltah,
    stride_deltam,
    stride_dqm,
    stride_dqh,
    stride_dqk,
    stride_dkn_out,
    stride_dkh_out,
    stride_dkk_out,
    stride_dvn_out,
    stride_dvh_out,
    stride_dvk_out,
    # scalars/scheduling
    sm_scale,
    N_CTX_Q,
    total_n_blocks_all_batches,
    # KV scheduling
    total_tiles_kv,
    high_load_wgs_kv,
    max_tiles_per_wg_kv,
    # Q scheduling
    total_tiles_q,
    high_load_wgs_q,
    max_tiles_per_wg_q,
    H: tl.constexpr,
    B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Fused per-CTA kernel: execute dK/dV first, then dQ, to maximize data reuse and
    minimize launch overhead. Work distribution mirrors the host-side scheduling.
    """
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Phase 1: dK/dV in transposed domain
    la_bwd_dkdv_inner(
        Q,
        K,
        V,
        DO,
        M,
        Delta,
        DK,
        DV,
        batch_num_block_n,
        stride_qm,
        stride_qh,
        stride_qk,
        stride_kn,
        stride_kh,
        stride_kk,
        stride_vn,
        stride_vh,
        stride_vk,
        stride_dom,
        stride_doh,
        stride_dok,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_deltab,
        stride_deltah,
        stride_deltam,
        stride_dkn_out,
        stride_dkh_out,
        stride_dkk_out,
        stride_dvn_out,
        stride_dvh_out,
        stride_dvk_out,
        sm_scale,
        N_CTX_Q,
        total_n_blocks_all_batches,
        total_tiles_kv,
        high_load_wgs_kv,
        max_tiles_per_wg_kv,
        H=H,
        B=B,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=CAUSAL,
    )

    # Phase 2: dQ (Q-streaming)
    la_bwd_dq_inner(
        Q,
        K,
        V,
        DO,
        M,
        Delta,
        DQ,
        batch_num_block_n,
        stride_qm,
        stride_qh,
        stride_qk,
        stride_kn,
        stride_kh,
        stride_kk,
        stride_vn,
        stride_vh,
        stride_vk,
        stride_dom,
        stride_doh,
        stride_dok,
        stride_mb,
        stride_mh,
        stride_mm,
        stride_deltab,
        stride_deltah,
        stride_deltam,
        stride_dqm,
        stride_dqh,
        stride_dqk,
        sm_scale,
        N_CTX_Q,
        total_n_blocks_all_batches,
        total_tiles_q,
        high_load_wgs_q,
        max_tiles_per_wg_q,
        H=H,
        B=B,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=CAUSAL,
    )


def persistent_lean_attention_bwd(
    q: torch.Tensor,     # (B * seq_len_q, H, d)
    k: torch.Tensor,     # (total_seq_len_k, H, d)
    v: torch.Tensor,     # (total_seq_len_k, H, d)
    do: torch.Tensor,    # (B * seq_len_q, H, d)
    o: torch.Tensor,     # (B * seq_len_q, H, d)
    softmax_lse: torch.Tensor,  # (B, H, seq_len_q), natural log-sum-exp per row
    dq: torch.Tensor,    # output (B * seq_len_q, H, d)
    dk: torch.Tensor,    # output (total_seq_len_k, H, d)
    dv: torch.Tensor,    # output (total_seq_len_k, H, d)
    batch_num_block_n: Optional[torch.Tensor],  # (B,) cumulative BLOCK_N per batch (for decode). Pass None for uniform.
    batch_size: int,
    sm_scale: float,
    causal: bool = True,
    config: Optional[dict] = None,
    num_programs: Optional[int] = None,
):
    """
    Host-side launcher for the fused backward pass.
    - Validates shapes, derives scheduling, constructs Delta = sum(do * o) per row.
    - Builds a minimal `batch_num_block_n` if not provided to support ragged non-causal.
    - Launches a single fused kernel where each CTA computes dK/dV then dQ.
    """
    if config is None:
        # Use a minimal default to avoid external JSON dependency during tests
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "num_warps": 4, "waves_per_eu": 1}
    # Resolve total programs (CTAs) preference: explicit arg > config > device
    total_programs_pref = None
    if num_programs is not None:
        total_programs_pref = int(num_programs)
    elif "num_ctas" in config:
        total_programs_pref = int(config["num_ctas"])
    else:
        try:
            total_programs_pref = int(arch_info.get_num_sms())
        except Exception:
            total_programs_pref = int(torch.cuda.get_device_properties(q.device).multi_processor_count)

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]

    assert q.shape == do.shape == o.shape
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM in {8, 16, 32, 64, 128, 256}

    # Layout and dimensions
    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # sum of seqlens across batch
    H = q.shape[1]

    # Scheduling parameters (safe variant)
    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        _even_split,
    ) = get_num_splits_and_buffer_sizes_bwd(
        causal,
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        H,
        BLOCK_M,
        BLOCK_N,
        num_SMs=total_programs_pref,
    )

    grid = (total_programs,)

    # NOTE: `DqOp` and `locks` are not used by the current fused design;
    # retained here only for potential experimentation with host-block reductions.
    DqOp = torch.empty(
        (total_programs, BLOCK_M, HEAD_DIM), device=q.device, dtype=dq.dtype
    )
    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # If no ragged decode, build cumulative n-blocks per batch for non-causal; single-element for B==1
    if batch_num_block_n is None:
        num_n_blocks_total = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
        if not causal:
            if batch_size > 1:
                per_batch_blocks = num_n_blocks_total // batch_size
                batch_num_block_n = (torch.arange(1, batch_size + 1, device=q.device, dtype=torch.int32) * per_batch_blocks)
            else:
                batch_num_block_n = torch.tensor([num_n_blocks_total], device=q.device, dtype=torch.int32)
        else:
            # not used in causal paths; provide a sane placeholder
            batch_num_block_n = torch.tensor([num_n_blocks_total // max(1, batch_size)], device=q.device, dtype=torch.int32)

    # Compute Delta on host: sum(dO * O) per row, arranged as [B, H, Nq]
    Delta = (do * o).sum(dim=-1).view(batch_size, N_CTX_Q, H).permute(0, 2, 1).contiguous()
    # Ensure dk/dv/dq are zero-initialized
    dk.zero_(); dv.zero_(); dq.zero_()

    # --- Fused Stream-K pass: dK/dV then dQ in one launch ---
    # KV scheduling (tiles are KV blocks × heads)
    total_n_blocks_all_batches = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
    total_kv_tiles = total_n_blocks_all_batches * H
    max_tiles_per_wg_kv = (total_kv_tiles + total_programs - 1) // total_programs
    high_load_wgs_kv = total_kv_tiles - ((max_tiles_per_wg_kv - 1) * total_programs)
    # Q scheduling (tiles are Q m-blocks × heads × batches)
    num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    total_tiles_q = batch_size * H * num_m_blocks_total
    max_tiles_per_wg_q = (total_tiles_q + total_programs - 1) // total_programs
    high_load_wgs_q = total_tiles_q - ((max_tiles_per_wg_q - 1) * total_programs)

    la_bwd_fused_streamk[grid](
        q,
        k,
        v,
        do,
        softmax_lse,
        Delta,
        dq,
        dk,
        dv,
        batch_num_block_n,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        sm_scale,
        N_CTX_Q,
        total_n_blocks_all_batches,
        total_kv_tiles,
        high_load_wgs_kv,
        max_tiles_per_wg_kv,
        total_tiles_q,
        high_load_wgs_q,
        max_tiles_per_wg_q,
        H=H,
        B=batch_size,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
        num_warps=config["num_warps"],
        waves_per_eu=config["waves_per_eu"],
        num_stages=1,
        num_ctas=1,
    )

    return
