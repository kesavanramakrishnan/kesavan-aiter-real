import torch
import functools
from typing import Optional
import triton
import triton.language as tl

# from aiter.ops.triton.lean_atten_val import (
#     _get_config,  # reuse config discovery (we avoid external JSON in tests)
#     get_num_splits_and_buffer_sizes,  # reuse scheduling (we implement a safe variant below)
# )


@triton.jit
def swizzle_by_grouping(pid: tl.int32, GROUP_SIZE: tl.constexpr) -> tl.int32:
    group_id = pid // GROUP_SIZE
    pid_in_group = pid % GROUP_SIZE
    swizzled_pid = group_id + (pid_in_group * (tl.num_programs(0) // GROUP_SIZE))
    return swizzled_pid


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

    total_programs = min(max(1, num_SMs), total_tiles)
    max_tiles_per_tb = (total_tiles + total_programs - 1) // total_programs
    # Splits
    if max_tiles_per_tb <= 1:
        num_splits = 1
        even_split = total_tiles % total_programs == 0
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
def la_bwd_persistent(
    # tensors
    Q,
    K,
    V,
    DO,
    M,      # softmax lse (natural log-sum-exp) shape [B, H, N_CTX_Q]
    Delta,  # row-wise sum(dO * O) shape [B, H, N_CTX_Q]
    DQ,
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
    stride_dqm,
    stride_dqh,
    stride_dqk,
    stride_dkn_out,
    stride_dkh_out,
    stride_dkk_out,
    stride_dvn_out,
    stride_dvh_out,
    stride_dvk_out,
    # host-accum buffers for dQ
    DqOp,            # [total_programs, n_ctx_q, head_dim]
    locks,           # [total_programs]
    stride_op_ph,    # total_programs
    stride_op_pm,    # n_ctx_q
    stride_op_pk,    # head_dim
    # scalars/scheduling
    sm_scale,
    batch_num_block_n,
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
    current_pid = tl.program_id(0)
    # Disable grouping swizzle for small grids to ensure full tile coverage in tests
    # GROUP_SIZE = 12
    # current_pid = swizzle_by_grouping(current_pid, GROUP_SIZE)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (current_pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

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
    tl.assume(stride_dkn_out > 0)
    tl.assume(stride_dkh_out > 0)
    tl.assume(stride_dkk_out > 0)
    tl.assume(stride_dvn_out > 0)
    tl.assume(stride_dvh_out > 0)
    tl.assume(stride_dvk_out > 0)
    tl.assume(stride_op_ph > 0)
    tl.assume(stride_op_pm > 0)
    tl.assume(stride_op_pk > 0)

    while iter < cta_end_tile_gid:
        tile_head_idx = iter // tiles_per_head

        if causal:
            tile_batch_idx = (iter % tiles_per_head) // (tl.maximum(1, tiles_per_head // batch_size))
            per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
                iter - (tile_head_idx * tiles_per_head) - (tile_batch_idx * tl.maximum(1, (tiles_per_head // batch_size))),
                MASKED_BLOCKS,
                num_m_blocks,
            )
            tile_iter = (
                tile_head_idx * tiles_per_head
                + (tile_batch_idx * tl.maximum(1, (tiles_per_head // batch_size)))
                + total_blocks
            )
            tile_iter_end = tile_iter + (per_head_tile_size)
            tile_idx = (tile_head_idx * batch_size + tile_batch_idx) * num_m_blocks + per_head_tile_idx
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

        host_block = iter == tile_iter
        finishing_block = cta_end_tile_gid >= tile_iter_end

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

        # pointers to K/V tiles start of current local_iter
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
        v_ptrs = V + v_offs
        k_ptrs = tl.multiple_of(k_ptrs, (16, 1))
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
        do_offs = (
            q_idx * BLOCK_M * stride_dom
            + tile_head_idx * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_k[None, :] * stride_dok
        )
        do_ptrs = DO + do_offs

        # load q and do
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)

        # per-row softmax lse and delta
        m_ptrs = M + (
            tile_batch_idx * stride_mb
            + tile_head_idx * stride_mh
            + (q_idx * BLOCK_M + offs_m) * stride_mm
        )
        m_rows = tl.load(m_ptrs, mask=(offs_m < BLOCK_M), other=-float("inf"))
        delta_ptrs = Delta + (
            tile_batch_idx * stride_deltab
            + tile_head_idx * stride_deltah
            + (q_idx * BLOCK_M + offs_m) * stride_deltam
        )
        delta_rows = tl.load(delta_ptrs, mask=(offs_m < BLOCK_M), other=0.0)

        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # iterate lean tiles for this output tile
        # For initial correctness: focus on dK/dV coverage. We ignore dQ accumulation here.
        for l_iter in range(local_iter, local_iter_end):
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

            # qk and normalized probabilities p = softmax(qk * sm_scale - M)
            qk = tl.dot(q, k) * sm_scale
            p = tl.math.exp(qk - m_rows[:, None])

            if causal:
                k_start_n = (b_seq_size + l_iter) * BLOCK_N
                mask = (q_idx * BLOCK_M + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
                p = tl.where(mask, p, 0.0)

            # dp and ds
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta_rows[:, None])

            # accumulate dV and dK via atomics per K/V tile first for correctness
            dv_partial = tl.dot(tl.trans(p).to(do.type.element_ty), do)
            dk_partial = tl.dot(tl.trans(ds).to(q.type.element_ty), q)

            # output pointers for DV / DK tiles
            dv_ptrs_out = (
                DV
                + (b_seq_size + l_iter) * BLOCK_N * stride_dvn_out
                + tile_head_idx * stride_dvh_out
                + offs_n[:, None] * stride_dvn_out
                + offs_k[None, :] * stride_dvk_out
            )
            dk_ptrs_out = (
                DK
                + (b_seq_size + l_iter) * BLOCK_N * stride_dkn_out
                + tile_head_idx * stride_dkh_out
                + offs_n[:, None] * stride_dkn_out
                + offs_k[None, :] * stride_dkk_out
            )

            # atomic adds; scale dk by sm_scale
            tl.atomic_add(dv_ptrs_out, dv_partial)
            tl.atomic_add(dk_ptrs_out, dk_partial * sm_scale)

            v_ptrs += BLOCK_N * stride_vn
            k_ptrs += BLOCK_N * stride_kn

        # For now, skip dQ host-block accumulation to validate dK/dV first

        iter = iter + (local_iter_end - local_iter)


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
        num_to_process = max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        num_to_process = max_tiles_per_wg - 1

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Process each assigned LeanTile
    for i in range(0, num_to_process):
        iter = iter_start + i
        valid = iter < total_tiles

        # Map LeanTile id to (tile_head_idx, tile_batch_idx, per_head_tile_idx) and local iter range
        tile_head_idx_raw = iter // tiles_per_head
        tile_head_idx = tl.minimum(tile_head_idx_raw, H - 1)
        # defaults to avoid use before set when invalid
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
        # In Stream-K, each iter processes one k/v block of the current output tile
        # So we limit to a single LeanTile per iter
        local_iter_end = tl.where(valid, local_iter + 1, 0)

        # Mask all subsequent loads/computes with `valid` so we don't use continue
        host_block = (local_iter == 0) & valid
        finishing_block = False  # Will be determined by split computation below when reducing

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
        
        # Initialize accumulator for this output tile segment
        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Process one LeanTile (one K/V block) for this output tile
        l_iter = local_iter
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

        # Host-CTA reduction across splits for the same output tile
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
            # Determine whether this CTA is finishing the tile within its assigned range
            # Compute last_cta covering this tile using num_splits heuristic from forward
            last_cta = pid + 1
            temp_end_gid = iter_start + num_to_process
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

    BLOCK_M = config["BLOCK_SIZE_M"]
    BLOCK_N = config["BLOCK_SIZE_N"]

    assert q.shape == do.shape == o.shape
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]

    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM in {16, 32, 64, 128, 256}

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
        num_SMs=torch.cuda.get_device_properties(q.device).multi_processor_count,
    )

    grid = (total_programs,)

    # dQ partial buffer and locks for host-block accumulation
    DqOp = torch.empty(
        (total_programs, N_CTX_Q, HEAD_DIM), device=q.device, dtype=dq.dtype
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

    # --- New: KV-centric pass for dK/dV with Stream-K equalization (1 KV tile per CTA) ---
    total_n_blocks_all_batches = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
    total_kv_tiles = total_n_blocks_all_batches * H
    grid_kv = (total_kv_tiles,)

    la_bwd_kv_persistent[grid_kv](
        q,
        k,
        v,
        do,
        softmax_lse,
        Delta,
        dk,
        dv,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        sm_scale,
        batch_num_block_n,
        N_CTX_Q,
        total_n_blocks_all_batches,
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

    # Optionally run Q-centric pass for dQ later; for now we leave dq as zeros or compute separately
    # --- Q-centric pass for dQ: use KV-style correctness (one Q tile per CTA, full K/V sweep) ---
    num_m_blocks_total = (N_CTX_Q + BLOCK_M - 1) // BLOCK_M
    total_q_tiles = batch_size * H * num_m_blocks_total
    grid_q = (total_q_tiles,)

    la_bwd_q_persistent[grid_q](
        q,
        k,
        v,
        do,
        softmax_lse,
        Delta,
        dq,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        sm_scale,
        batch_num_block_n,
        N_CTX_Q,
        total_n_blocks_all_batches,
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
