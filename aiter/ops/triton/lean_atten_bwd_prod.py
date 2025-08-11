import torch
import functools
from typing import Optional
import triton
import triton.language as tl
import aiter.ops.triton.utils.arch_info as arch_info

# from aiter.ops.triton.lean_atten_val import (
#     _get_config,  # reuse config discovery (we avoid external JSON in tests)
#     get_num_splits_and_buffer_sizes,  # reuse scheduling (we implement a safe variant below)
# )


@triton.jit
def find_group(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
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
    Safe variant of forward scheduling to avoid division-by-zero for small workloads.
    Returns:
      num_m_blocks, num_n_blocks, high_load_wgs, max_tiles_per_wg, tiles_per_head,
      total_programs, num_splits, even_split
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
def la_bwd_kv_persistent(
    # tensors
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DK,
    DV,
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
    stride_dkn_out,
    stride_dkh_out,
    stride_dkk_out,
    stride_dvn_out,
    stride_dvh_out,
    stride_dvk_out,
    # scalars/scheduling
    sm_scale,
    batch_num_block_n,
    N_CTX_Q,
    total_n_blocks_all_batches,
    H: tl.constexpr,
    B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    kv_id = pid
    total_kv_tiles = total_n_blocks_all_batches * H
    if kv_id >= total_kv_tiles:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)

    head_idx = kv_id % H
    n_linear = kv_id // H

    # Map n_linear -> (batch_idx, n_block_in_batch)
    batch_idx = 0
    n_block_in_batch = 0
    if CAUSAL:
        num_n_blocks_per_batch = total_n_blocks_all_batches // B
        batch_idx = n_linear // num_n_blocks_per_batch
        n_block_in_batch = n_linear % num_n_blocks_per_batch
        b_seq_size = batch_idx * num_n_blocks_per_batch
    else:
        prev_cum_running = 0
        match_prev_cum = 0
        found = 0
        for b in range(0, B):
            cum = tl.load(batch_num_block_n + b) if b > 0 else tl.load(batch_num_block_n)
            is_match = (found == 0) & (n_linear < cum)
            batch_idx = tl.where(is_match, b, batch_idx)
            match_prev_cum = tl.where(is_match, prev_cum_running, match_prev_cum)
            found = tl.where(is_match, 1, found)
            prev_cum_running = cum
        n_block_in_batch = n_linear - match_prev_cum
        b_seq_size = 0 if batch_idx == 0 else tl.load(batch_num_block_n + batch_idx - 1)

    # Load K/V tile for this kv
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
    k_tile = tl.load(K + k_offs)
    v_tile = tl.load(V + v_offs)

    k_start_abs = (b_seq_size + n_block_in_batch) * BLOCK_N

    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    for m_block in range(0, num_m_blocks_total):
        q_start_abs = batch_idx * N_CTX_Q + m_block * BLOCK_M
        q_ptrs = Q + q_start_abs * stride_qm + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        do_ptrs = DO + q_start_abs * stride_dom + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + (m_block * BLOCK_M + offs_m) * stride_mm
        delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + (m_block * BLOCK_M + offs_m) * stride_deltam

        q_tile = tl.load(q_ptrs)
        do_tile = tl.load(do_ptrs)
        m_rows = tl.load(m_ptrs, mask=(offs_m < BLOCK_M), other=-float("inf"))
        delta_rows = tl.load(delta_ptrs, mask=(offs_m < BLOCK_M), other=0.0)

        qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        p = tl.math.exp(qk - m_rows[:, None])

        if CAUSAL:
            mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :])
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = p * (dp - delta_rows[:, None])

        dv_acc += tl.dot(tl.trans(p).to(do_tile.type.element_ty), do_tile)
        dk_acc += tl.dot(tl.trans(ds).to(q_tile.type.element_ty), q_tile)

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
    tl.store(dv_ptrs_out, dv_acc.to(DV.type.element_ty))
    tl.store(dk_ptrs_out, (dk_acc * sm_scale).to(DK.type.element_ty))


@triton.jit
def la_bwd_q_persistent(
    # tensors
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DQ,
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
    # scalars/scheduling
    sm_scale,
    batch_num_block_n,
    N_CTX_Q,
    total_n_blocks_all_batches,
    H: tl.constexpr,
    B: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    # Map pid -> (batch_idx, head_idx, m_block)
    num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    tiles_per_head = num_m_blocks_total
    tiles_per_batch = tiles_per_head * H

    batch_idx = pid // tiles_per_batch
    rem = pid % tiles_per_batch
    head_idx = rem // tiles_per_head
    m_block = rem % tiles_per_head

    # Guard
    if batch_idx >= B:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Q/DO/M/Delta pointers for this Q tile
    q_start_abs = batch_idx * N_CTX_Q + m_block * BLOCK_M
    q_ptrs = Q + q_start_abs * stride_qm + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    do_ptrs = DO + q_start_abs * stride_dom + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + (m_block * BLOCK_M + offs_m) * stride_mm
    delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + (m_block * BLOCK_M + offs_m) * stride_deltam

    q_tile = tl.load(q_ptrs)
    do_tile = tl.load(do_ptrs)
    m_rows = tl.load(m_ptrs, mask=(offs_m < BLOCK_M), other=-float("inf"))
    delta_rows = tl.load(delta_ptrs, mask=(offs_m < BLOCK_M), other=0.0)

    # Determine this batch's K/V n-block range
    num_n_blocks_per_batch = 0
    b_seq_size_blocks = 0
    if CAUSAL:
        num_n_blocks_per_batch = total_n_blocks_all_batches // B
        b_seq_size_blocks = batch_idx * num_n_blocks_per_batch
    else:
        prev_cum = 0
        for b in range(0, B):
            cum = tl.load(batch_num_block_n + b) if b > 0 else tl.load(batch_num_block_n)
            if b == batch_idx:
                num_n_blocks_per_batch = cum - prev_cum
                b_seq_size_blocks = prev_cum
            prev_cum = cum

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Iterate over all K/V tiles for this batch
    for n_block in range(0, num_n_blocks_per_batch):
        k_offs = (
            (b_seq_size_blocks + n_block) * BLOCK_N * stride_kn
            + head_idx * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_k[None, :] * stride_kk
        )
        v_offs = (
            (b_seq_size_blocks + n_block) * BLOCK_N * stride_vn
            + head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )
        k_tile = tl.load(K + k_offs)
        v_tile = tl.load(V + v_offs)

        # Compute qk and probabilities
        qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        p = tl.math.exp(qk - m_rows[:, None])

        if CAUSAL:
            k_start_abs = (b_seq_size_blocks + n_block) * BLOCK_N
            mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :])
            p = tl.where(mask, p, 0.0)

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = p * (dp - delta_rows[:, None])

        dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

    # Write dQ with sm_scale
    dq_acc = dq_acc * sm_scale
    dq_ptrs_out = (
        DQ
        + q_start_abs * stride_dqm
        + head_idx * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_k[None, :] * stride_dqk
    )
    tl.store(dq_ptrs_out, dq_acc.to(DQ.type.element_ty))


@triton.jit
def la_bwd_q_streamk(
    # tensors
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DQ,
    DqOp,
    locks,
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
    stride_op_ph,
    stride_op_pm,
    stride_op_pk,
    # scalars/scheduling
    sm_scale,
    total_tiles,
    high_load_wgs,
    max_tiles_per_wg,
    tiles_per_head,
    num_splits: tl.constexpr,
    max_output_tile_cnt: tl.constexpr,
    batch_size: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
):
    pid = tl.program_id(0)

    # Determine tile range [iter_start, iter_end) for this CTA
    if pid < high_load_wgs:
        iter_start = max_tiles_per_wg * pid
        cta_end_tile_gid = iter_start + max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter_start + (max_tiles_per_wg - 1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    iter = iter_start
    for _i in tl.static_range(max_output_tile_cnt + 1):
        valid = iter < total_tiles
        if not valid:
            break

        # Map LeanTile id to (tile_head_idx, tile_batch_idx, per_head_tile_idx) and local iter range
        tile_head_idx_raw = iter // tiles_per_head
        tile_head_idx = tl.minimum(tile_head_idx_raw, H - 1)
        # defaults
        tile_batch_idx = 0
        per_head_tile_idx = 0
        tile_iter = 0
        tile_iter_end = 0
        tile_idx = 0

        if CAUSAL and valid:
            per_head_span = tl.maximum(1, tiles_per_head // batch_size)
            tile_batch_idx = (iter % tiles_per_head) // per_head_span
            per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
                iter - (tile_head_idx * tiles_per_head) - (tile_batch_idx * per_head_span),
                MASKED_BLOCKS,
                num_m_blocks,
            )
            tile_iter = tile_head_idx * tiles_per_head + tile_batch_idx * per_head_span + total_blocks
            tile_iter_end = tile_iter + per_head_tile_size
            tile_idx = (tile_head_idx * batch_size + tile_batch_idx) * num_m_blocks + per_head_tile_idx
        elif (not CAUSAL) and valid:
            tile_idx = tile_head_idx * batch_size
            tile_iter = tile_head_idx * tiles_per_head
            # derive per-batch req sizes
            req_size = 0
            prev_size = 0
            local_head_iter = iter % tiles_per_head
            tile_batch_idx = 0
            for b in range(0, batch_size):
                cum = tl.load(batch_num_block_n + b) if b > 0 else tl.load(batch_num_block_n)
                if (local_head_iter < cum) & (local_head_iter >= prev_size):
                    tile_batch_idx = b
                    tile_iter = tile_head_idx * tiles_per_head + prev_size
                    tile_iter_end = tile_iter + (cum - prev_size)
                prev_size = cum
            tile_idx = tile_head_idx * batch_size + tile_batch_idx
        # Compute local iter bounds for this CTA
        local_iter = tl.where(valid, iter - tile_iter, 0)
        local_iter_end = tl.where(valid, tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter, 0)

        # Mask all subsequent loads/computes with `valid`
        host_block = (local_iter == 0) & valid
        finishing_block = tl.where(valid, cta_end_tile_gid >= tile_iter_end, False)

        # Derive memory offsets
        if CAUSAL:
            q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
            b_seq_size = tile_batch_idx * num_n_blocks
        else:
            q_idx = tile_batch_idx
            if tile_batch_idx == 0:
                b_seq_size = 0
            else:
                b_seq_size = tl.load(batch_num_block_n + tile_batch_idx - 1)

        q_offs = (
            q_idx * BLOCK_M * stride_qm
            + tile_head_idx * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        q_ptrs = Q + q_offs
        do_ptrs = DO + (
            q_idx * BLOCK_M * stride_dom
            + tile_head_idx * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_k[None, :] * stride_dok
        )
        m_ptrs = M + (
            tile_batch_idx * stride_mb
            + tile_head_idx * stride_mh
            + (q_idx * BLOCK_M + offs_m) * stride_mm
        )
        delta_ptrs = Delta + (
            tile_batch_idx * stride_deltab
            + tile_head_idx * stride_deltah
            + (q_idx * BLOCK_M + offs_m) * stride_deltam
        )

        q_tile = tl.load(q_ptrs, mask=valid)
        do_tile = tl.load(do_ptrs, mask=valid)
        m_rows = tl.load(m_ptrs, mask=((offs_m < BLOCK_M) & valid), other=-float("inf"))
        delta_rows = tl.load(delta_ptrs, mask=((offs_m < BLOCK_M) & valid), other=0.0)
        
        # Initialize accumulator for this CTA's output tile segment
        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Process all LeanTiles for this output tile segment
        for l_iter in range(local_iter, local_iter_end):
            k_offs = (
                (b_seq_size + l_iter) * BLOCK_N * stride_kn
                + tile_head_idx * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_k[None, :] * stride_kk
            )
            v_offs = (
                (b_seq_size + l_iter) * BLOCK_N * stride_vn
                + tile_head_idx * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_k[None, :] * stride_vk
            )
            k_tile = tl.load(K + k_offs)
            v_tile = tl.load(V + v_offs)

            # Compute p and ds
            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            p = tl.math.exp(qk - m_rows[:, None])
            if CAUSAL:
                k_start_n = (b_seq_size + l_iter) * BLOCK_N
                mask = (q_idx * BLOCK_M + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
                p = tl.where(mask, p, 0.0)
            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = p * (dp - delta_rows[:, None])

            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

        # Non-host: store partials and signal
        if not host_block:
            op_ptrs = (
                DqOp
                + pid * stride_op_ph
                + offs_m[:, None] * stride_op_pm
                + offs_k[None, :] * stride_op_pk
            )
            tl.store(op_ptrs, dq_acc.to(DqOp.type.element_ty), cache_modifier=".wt")
            tl.debug_barrier()
            tl.atomic_xchg(locks + pid, 1)
        else:
            # Compute last_cta covering this tile using num_splits heuristic
            last_cta = pid + 1
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
            # Reduce partials from other CTAs
            for cta in range(pid + 1, last_cta):
                while tl.atomic_cas(locks + cta, 1, 1) != 1:
                    pass
                op_ptrs = (
                    DqOp
                    + cta * stride_op_ph
                    + offs_m[:, None] * stride_op_pm
                    + offs_k[None, :] * stride_op_pk
                )
                dq_acc += tl.load(op_ptrs)
            # Write final dQ (apply sm_scale)
            dq_acc = dq_acc * sm_scale
            dq_ptrs_out = (
                DQ
                + q_idx * BLOCK_M * stride_dqm
                + tile_head_idx * stride_dqh
                + offs_m[:, None] * stride_dqm
                + offs_k[None, :] * stride_dqk
            )
            tl.store(dq_ptrs_out, dq_acc.to(DQ.type.element_ty))

        # Advance iter by the amount processed for this tile segment
        iter = iter + (local_iter_end - local_iter)
        if iter >= cta_end_tile_gid:
            break


@triton.jit
def la_bwd_kv_streamk(
    # tensors
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DK,
    DV,
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
    stride_dkn_out,
    stride_dkh_out,
    stride_dkk_out,
    stride_dvn_out,
    stride_dvh_out,
    stride_dvk_out,
    # scalars/scheduling
    sm_scale,
    batch_num_block_n,
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
    pid = tl.program_id(0)

    # Determine KV tile range [iter_start, iter_end) for this CTA
    if pid < high_load_wgs_kv:
        iter_start = max_tiles_per_wg_kv * pid
        num_to_process = max_tiles_per_wg_kv
    else:
        iter_start = (max_tiles_per_wg_kv - 1) * (pid - high_load_wgs_kv) + high_load_wgs_kv * max_tiles_per_wg_kv
        num_to_process = max_tiles_per_wg_kv - 1

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    offs_m = tl.arange(0, BLOCK_M)

    for i in range(0, num_to_process):
        iter = iter_start + i
        valid = iter < total_tiles_kv

        # Map KV tile id to (head_idx, n_linear) guarded by valid
        kv_id = iter
        tile_head_idx_raw = kv_id % H
        head_idx = tl.minimum(tile_head_idx_raw, H - 1)
        n_linear = kv_id // H

        # Map n_linear -> (batch_idx, n_block_in_batch)
        batch_idx = 0
        n_block_in_batch = 0
        if CAUSAL:
            num_n_blocks_per_batch = total_n_blocks_all_batches // B
            batch_idx = n_linear // num_n_blocks_per_batch
            n_block_in_batch = n_linear % num_n_blocks_per_batch
            b_seq_size = batch_idx * num_n_blocks_per_batch
        else:
            prev_cum_running = 0
            match_prev_cum = 0
            found = 0
            for b in range(0, B):
                cum = tl.load(batch_num_block_n + b, mask=valid, other=0) if b > 0 else tl.load(batch_num_block_n, mask=valid, other=0)
                is_match = (found == 0) & (n_linear < cum)
                batch_idx = tl.where(is_match, b, batch_idx)
                match_prev_cum = tl.where(is_match, prev_cum_running, match_prev_cum)
                found = tl.where(is_match, 1, found)
                prev_cum_running = cum
            n_block_in_batch = n_linear - match_prev_cum
            b_seq_size = 0 if batch_idx == 0 else tl.load(batch_num_block_n + batch_idx - 1, mask=valid, other=0)

        # Load K/V tile for this kv
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
        k_tile = tl.load(K + k_offs, mask=valid, other=0.0)
        v_tile = tl.load(V + v_offs, mask=valid, other=0.0)

        k_start_abs = (b_seq_size + n_block_in_batch) * BLOCK_N

        dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

        num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
        for m_block in range(0, num_m_blocks_total):
            q_start_abs = batch_idx * N_CTX_Q + m_block * BLOCK_M
            q_ptrs = Q + q_start_abs * stride_qm + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
            do_ptrs = DO + q_start_abs * stride_dom + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
            m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + (m_block * BLOCK_M + offs_m) * stride_mm
            delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + (m_block * BLOCK_M + offs_m) * stride_deltam

            q_tile = tl.load(q_ptrs, mask=valid)
            do_tile = tl.load(do_ptrs, mask=valid)
            m_rows = tl.load(m_ptrs, mask=(valid & (offs_m < BLOCK_M)), other=-float("inf"))
            delta_rows = tl.load(delta_ptrs, mask=(valid & (offs_m < BLOCK_M)), other=0.0)

            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            p = tl.math.exp(qk - m_rows[:, None])

            if CAUSAL:
                mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :])
                p = tl.where(mask & valid, p, 0.0)

            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = p * (dp - delta_rows[:, None])

            dv_acc += tl.dot(tl.trans(p).to(do_tile.type.element_ty), do_tile)
            dk_acc += tl.dot(tl.trans(ds).to(q_tile.type.element_ty), q_tile)

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
        tl.store(dv_ptrs_out, dv_acc.to(DV.type.element_ty), mask=valid)
        tl.store(dk_ptrs_out, (dk_acc * sm_scale).to(DK.type.element_ty), mask=valid)


@triton.jit
def la_bwd_q_streamk_tiles(
    # tensors
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DQ,
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
    # scalars/scheduling
    sm_scale,
    total_q_tiles,
    tiles_per_cta,
    N_CTX_Q,
    total_n_blocks_all_batches,
    CAUSAL: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    tiles_per_batch = num_m_blocks_total * H

    for i in range(0, tiles_per_cta):
        tile_id = pid * tiles_per_cta + i
        valid = tile_id < total_q_tiles

        batch_idx = tile_id // tiles_per_batch
        rem = tile_id % tiles_per_batch
        head_idx = rem // num_m_blocks_total
        m_block = rem % num_m_blocks_total

        batch_idx = tl.where(valid, batch_idx, 0)
        head_idx = tl.where(valid, head_idx, 0)
        m_block = tl.where(valid, m_block, 0)

        # Q/DO/M/Delta pointers for this Q tile
        q_start_abs = batch_idx * N_CTX_Q + m_block * BLOCK_M
        q_ptrs = Q + q_start_abs * stride_qm + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        do_ptrs = DO + q_start_abs * stride_dom + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + (m_block * BLOCK_M + offs_m) * stride_mm
        delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + (m_block * BLOCK_M + offs_m) * stride_deltam

        q_tile = tl.load(q_ptrs, mask=valid)
        do_tile = tl.load(do_ptrs, mask=valid)
        m_rows = tl.load(m_ptrs, mask=(valid & (offs_m < BLOCK_M)), other=-float("inf"))
        delta_rows = tl.load(delta_ptrs, mask=(valid & (offs_m < BLOCK_M)), other=0.0)

        # Determine this batch's K/V n-block range
        num_n_blocks_per_batch = 0
        b_seq_size_blocks = 0
        if CAUSAL:
            num_n_blocks_per_batch = total_n_blocks_all_batches // B
            b_seq_size_blocks = batch_idx * num_n_blocks_per_batch
        else:
            prev_cum = 0
            for b in range(0, B):
                cum = tl.load(batch_num_block_n + b) if b > 0 else tl.load(batch_num_block_n)
                if b == batch_idx:
                    num_n_blocks_per_batch = cum - prev_cum
                    b_seq_size_blocks = prev_cum
                prev_cum = cum

        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Iterate over all K/V tiles for this batch
        for n_block in range(0, num_n_blocks_per_batch):
            k_offs = (
                (b_seq_size_blocks + n_block) * BLOCK_N * stride_kn
                + head_idx * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_k[None, :] * stride_kk
            )
            v_offs = (
                (b_seq_size_blocks + n_block) * BLOCK_N * stride_vn
                + head_idx * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_k[None, :] * stride_vk
            )
            k_tile = tl.load(K + k_offs, mask=valid, other=0.0)
            v_tile = tl.load(V + v_offs, mask=valid, other=0.0)

            # Compute qk and probabilities
            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            p = tl.math.exp(qk - m_rows[:, None])

            if CAUSAL:
                k_start_abs = (b_seq_size_blocks + n_block) * BLOCK_N
                mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :])
                p = tl.where(mask, p, 0.0)

            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = p * (dp - delta_rows[:, None])

            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

        # Write dQ with sm_scale
        dq_acc = dq_acc * sm_scale
        dq_ptrs_out = (
            DQ
            + q_start_abs * stride_dqm
            + head_idx * stride_dqh
            + offs_m[:, None] * stride_dqm
            + offs_k[None, :] * stride_dqk
        )
        tl.store(dq_ptrs_out, dq_acc.to(DQ.type.element_ty), mask=valid)


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
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # ----------------- Phase 1: dK/dV KV-StreamK -----------------
    if pid < high_load_wgs_kv:
        iter_start_kv = max_tiles_per_wg_kv * pid
        num_to_process_kv = max_tiles_per_wg_kv
    else:
        iter_start_kv = (max_tiles_per_wg_kv - 1) * (pid - high_load_wgs_kv) + high_load_wgs_kv * max_tiles_per_wg_kv
        num_to_process_kv = max_tiles_per_wg_kv - 1

    for i in range(0, num_to_process_kv):
        iter_kv = iter_start_kv + i
        valid_kv = iter_kv < total_tiles_kv

        # Map KV tile id to (head_idx, n_linear)
        kv_id = iter_kv
        tile_head_idx_raw = kv_id % H
        head_idx = tl.minimum(tile_head_idx_raw, H - 1)
        n_linear = kv_id // H

        # Map n_linear -> (batch_idx, n_block_in_batch)
        batch_idx = 0
        n_block_in_batch = 0
        if CAUSAL:
            num_n_blocks_per_batch = total_n_blocks_all_batches // B
            batch_idx = n_linear // num_n_blocks_per_batch
            n_block_in_batch = n_linear % num_n_blocks_per_batch
            b_seq_size = batch_idx * num_n_blocks_per_batch
        else:
            prev_cum_running = 0
            match_prev_cum = 0
            found = 0
            for b in range(0, B):
                cum = tl.load(batch_num_block_n + b, mask=valid_kv, other=0) if b > 0 else tl.load(batch_num_block_n, mask=valid_kv, other=0)
                is_match = (found == 0) & (n_linear < cum)
                batch_idx = tl.where(is_match, b, batch_idx)
                match_prev_cum = tl.where(is_match, prev_cum_running, match_prev_cum)
                found = tl.where(is_match, 1, found)
                prev_cum_running = cum
            n_block_in_batch = n_linear - match_prev_cum
            b_seq_size = 0 if batch_idx == 0 else tl.load(batch_num_block_n + batch_idx - 1, mask=valid_kv, other=0)

        # Load K/V tile for this kv
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

        num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
        for m_block in range(0, num_m_blocks_total):
            q_start_abs = batch_idx * N_CTX_Q + m_block * BLOCK_M
            q_ptrs = Q + q_start_abs * stride_qm + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
            do_ptrs = DO + q_start_abs * stride_dom + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
            m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + (m_block * BLOCK_M + offs_m) * stride_mm
            delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + (m_block * BLOCK_M + offs_m) * stride_deltam

            q_tile = tl.load(q_ptrs, mask=valid_kv)
            do_tile = tl.load(do_ptrs, mask=valid_kv)
            m_rows = tl.load(m_ptrs, mask=(valid_kv & (offs_m < BLOCK_M)), other=-float("inf"))
            delta_rows = tl.load(delta_ptrs, mask=(valid_kv & (offs_m < BLOCK_M)), other=0.0)

            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            p = tl.math.exp(qk - m_rows[:, None])

            if CAUSAL:
                mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :])
                p = tl.where(mask & valid_kv, p, 0.0)

            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = p * (dp - delta_rows[:, None])

            dv_acc += tl.dot(tl.trans(p).to(do_tile.type.element_ty), do_tile)
            dk_acc += tl.dot(tl.trans(ds).to(q_tile.type.element_ty), q_tile)

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
        tl.store(dv_ptrs_out, dv_acc.to(DV.type.element_ty), mask=valid_kv)
        tl.store(dk_ptrs_out, (dk_acc * sm_scale).to(DK.type.element_ty), mask=valid_kv)

    # ----------------- Phase 2: dQ Q-StreamK -----------------
    if pid < high_load_wgs_q:
        iter_start_q = max_tiles_per_wg_q * pid
        num_to_process_q = max_tiles_per_wg_q
    else:
        iter_start_q = (max_tiles_per_wg_q - 1) * (pid - high_load_wgs_q) + high_load_wgs_q * max_tiles_per_wg_q
        num_to_process_q = max_tiles_per_wg_q - 1

    # Derive Q tile mapping helpers
    num_m_blocks_total_q = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    tiles_per_batch = num_m_blocks_total_q * H

    for i in range(0, num_to_process_q):
        iter_q = iter_start_q + i
        valid_q = iter_q < total_tiles_q

        # Map Q tile id -> (batch_idx, head_idx, m_block)
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
            prev_cum = 0
            for b in range(0, B):
                cum = tl.load(batch_num_block_n + b, mask=valid_q, other=0) if b > 0 else tl.load(batch_num_block_n, mask=valid_q, other=0)
                if b == batch_idx:
                    num_n_blocks_per_batch = cum - prev_cum
                    b_seq_size_blocks = prev_cum
                prev_cum = cum

        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Iterate over all K/V tiles for this batch
        for n_block in range(0, num_n_blocks_per_batch):
            k_offs = (
                (b_seq_size_blocks + n_block) * BLOCK_N * stride_kn
                + head_idx * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_k[None, :] * stride_kk
            )
            v_offs = (
                (b_seq_size_blocks + n_block) * BLOCK_N * stride_vn
                + head_idx * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_k[None, :] * stride_vk
            )
            k_tile = tl.load(K + k_offs, mask=valid_q, other=0.0)
            v_tile = tl.load(V + v_offs, mask=valid_q, other=0.0)

            # Compute qk and probabilities
            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            p = tl.math.exp(qk - m_rows[:, None])

            if CAUSAL:
                k_start_abs = (b_seq_size_blocks + n_block) * BLOCK_N
                mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :])
                p = tl.where(mask, p, 0.0)

            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = p * (dp - delta_rows[:, None])

            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

        # Write dQ with sm_scale
        dq_acc = dq_acc * sm_scale
        dq_ptrs_out = (
            DQ
            + q_start_abs * stride_dqm
            + head_idx * stride_dqh
            + offs_m[:, None] * stride_dqm
            + offs_k[None, :] * stride_dqk
        )
        tl.store(dq_ptrs_out, dq_acc.to(DQ.type.element_ty), mask=valid_q)


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
    Backward pass launcher using a persistent, StreamK-style kernel.
    - Host-block reduction is used for dQ (per output Q tile), mirroring forward.
    - dK/dV are accumulated via atomic adds in the K/V-tile domain (transposed domain).
    - Normalized probabilities P are formed via softmax LSE from the forward pass.
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

    # dQ partial buffer and locks for host-block accumulation
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

    # Compute Delta on host: sum(dO * O) per row
    Delta = (do * o).sum(dim=-1).view(batch_size, N_CTX_Q, H).permute(0, 2, 1).contiguous()
    # Ensure dk/dv are zero-initialized
    dk.zero_(); dv.zero_(); dq.zero_()

    # --- Fused Stream-K pass: dK/dV then dQ in one launch ---
    # KV scheduling
    total_n_blocks_all_batches = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
    total_kv_tiles = total_n_blocks_all_batches * H
    max_tiles_per_wg_kv = (total_kv_tiles + total_programs - 1) // total_programs
    high_load_wgs_kv = total_kv_tiles - ((max_tiles_per_wg_kv - 1) * total_programs)
    # Q scheduling
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


def flash_attn_onekernel_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dbias: Optional[torch.Tensor],
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
    USE_INT64_STRIDES: Optional[bool] = False,
    config: Optional[dict] = None,
):
    """
    Adapter to plug Lean Attention persistent backward into the MHA API used by test_mha_bwd.
    Supports non-varlen, no-dropout cases. Returns delta for parity with reference API.
    """
    # Constraints for this adapter
    if dropout_p and dropout_p != 0.0:
        raise NotImplementedError("Lean production bwd adapter only supports dropout_p == 0.0")
    if (cu_seqlens_q is not None) or (cu_seqlens_k is not None):
        raise NotImplementedError("Lean production bwd adapter expects non-varlen (cu_seqlens_* is None)")

    # Input layout is BSHD (batch, seqlen, heads, dim)
    batch_size, seqlen_q, num_q_heads, head_dim = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape

    # Flatten to [B*Seqlen, H, D]
    q_flat = q.reshape(batch_size * seqlen_q, num_q_heads, head_dim).contiguous()
    k_flat = k.reshape(batch_size * seqlen_k, num_k_heads, head_dim).contiguous()
    v_flat = v.reshape(batch_size * seqlen_k, num_k_heads, head_dim).contiguous()
    o_flat = o.reshape(batch_size * seqlen_q, num_q_heads, head_dim).contiguous()
    do_flat = do.reshape(batch_size * seqlen_q, num_q_heads, head_dim).contiguous()

    dq_flat = dq.reshape(batch_size * seqlen_q, num_q_heads, head_dim).contiguous()
    dk_flat = dk.reshape(batch_size * seqlen_k, num_k_heads, head_dim).contiguous()
    dv_flat = dv.reshape(batch_size * seqlen_k, num_k_heads, head_dim).contiguous()

    # batch_num_block_n: rely on kernel default for uniform non-causal; pass None
    batch_num_block_n = None

    # Launch Lean persistent backward
    persistent_lean_attention_bwd(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        do=do_flat,
        o=o_flat,
        softmax_lse=softmax_lse,
        dq=dq_flat,
        dk=dk_flat,
        dv=dv_flat,
        batch_num_block_n=batch_num_block_n,
        batch_size=batch_size,
        sm_scale=sm_scale,
        causal=causal,
        config=config,
        num_programs=None,
    )

    # Write results back into provided buffers' views
    dq.copy_(dq_flat.view_as(dq))
    dk.copy_(dk_flat.view_as(dk))
    dv.copy_(dv_flat.view_as(dv))

    # Return Delta = sum(do * o) per row, layout (B, H, seqlen_q)
    delta = (do * o).sum(dim=-1).permute(0, 2, 1).contiguous()
    return delta
