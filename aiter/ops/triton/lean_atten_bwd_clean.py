import torch
import functools
import json
import os
from typing import Optional
import triton
import triton.language as tl
import aiter.ops.triton.utils.arch_info as arch_info

# Lean attention backward pass (Triton)
# - Computes dQ, dK, dV with separate kernels: dK/dV first, then dQ.
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
    for i in tl.static_range(0, num_m_blocks):
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
    batch_size: int,
    H: int,
    N_CTX_Q: int,
    N_CTX_K: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_M_KV: int,
    BLOCK_N_KV: int,
    total_programs: int,
):
    """
    Compute scheduling parameters for both Q and KV kernels in the backward pass.
    - total_n_blocks_all_batches: Total number of blocks over the K sequence length.
    - total_kv_tiles: Total tiles for the KV kernel (num_n_blocks * num_heads).
    - high_load_wgs_kv, max_tiles_per_wg_kv: Scheduling parameters for KV tiles.
    - num_m_blocks_total: Total number of blocks over the Q sequence length.
    - total_tiles_q: Total tiles for the Q kernel (batch_size * num_heads * num_m_blocks).
    - high_load_wgs_q, max_tiles_per_wg_q: Scheduling parameters for Q tiles.
    - total_n_blocks_all_batches_kv: N-blocks for the KV kernel (using BLOCK_N_KV).
    - total_kv_tiles_kv: Total tiles for the KV kernel (using BLOCK_N_KV).
    - high_load_wgs_kv_kv, max_tiles_per_wg_kv_kv: Scheduling for KV tiles (using BLOCK_N_KV).
    """
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

    # KV kernel specific scheduling (potentially smaller tiles)
    num_m_blocks_total_kv = (N_CTX_Q + BLOCK_M_KV - 1) // BLOCK_M_KV
    total_n_blocks_all_batches_kv = (N_CTX_K + BLOCK_N_KV - 1) // BLOCK_N_KV
    total_kv_tiles_kv = total_n_blocks_all_batches_kv * H
    max_tiles_per_wg_kv_kv = (total_kv_tiles_kv + total_programs - 1) // total_programs
    high_load_wgs_kv_kv = total_kv_tiles_kv - ((max_tiles_per_wg_kv_kv - 1) * total_programs)

    return (
        total_n_blocks_all_batches,
        max_tiles_per_wg_kv,
        high_load_wgs_kv,
        num_m_blocks_total,
        total_tiles_q,
        max_tiles_per_wg_q,
        high_load_wgs_q,
        num_m_blocks_total_kv,
        total_n_blocks_all_batches_kv,
        total_kv_tiles_kv,
        max_tiles_per_wg_kv_kv,
        high_load_wgs_kv_kv,
    )


@functools.lru_cache(maxsize=1)
def _load_bwd_tuned_db(db_path: str):
    try:
        with open(db_path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _select_bwd_config(db, causal, B, H, D, NQ, NK):
    candidates = [
        e
        for e in db
        if e.get("key")
        and e.get("config")
        and int(e["key"].get("causal", 0)) == int(causal)
        and e["key"].get("H") == H
        and e["key"].get("D") == D
    ]
    if not candidates:
        return None
    # Prefer exact NQ/NK match when present; else nearest by L1 distance
    exact = [e for e in candidates if int(e["key"].get("NQ", -1)) == NQ and int(e["key"].get("NK", -1)) == NK and int(e["key"].get("B", -1)) == B]
    if exact:
        best = exact[0]
    else:
        best = min(
            candidates,
            key=lambda e: abs(int(e["key"].get("NQ", 0)) - NQ) + abs(int(e["key"].get("NK", 0)) - NK),
        )
    return best.get("config")

@triton.jit
def la_bwd_preprocess(
    o_ptr,
    do_ptr,
    delta_ptr,
    stride_om,
    stride_oh,
    stride_ok,
    stride_deltab,
    stride_deltah,
    stride_deltam,
    N_CTX_Q,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Compute Delta = rowsum(dO * O) per row in [B, H, N_CTX_Q], mirroring flash-attn preprocess.
    Inputs O/DO are laid out as [B*N_CTX_Q, H, HEAD_DIM]; Delta is [B, H, N_CTX_Q].
    Grid: (tl.cdiv(N_CTX_Q, BLOCK_M), B, H)
    """
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    # Compute flattened row index into O/DO: (bid * N_CTX_Q + m)
    row_idx = bid * N_CTX_Q + offs_m

    # Offsets into O/DO: [B*N_CTX_Q, H, K]
    offs_ok = (
        row_idx[:, None] * stride_om
        + hid * stride_oh
        + offs_k[None, :] * stride_ok
    )

    mask_m = offs_m < N_CTX_Q
    mask = mask_m[:, None]

    o = tl.load(o_ptr + offs_ok, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs_ok, mask=mask, other=0.0)

    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)

    offs_delta = (
        bid * stride_deltab
        + hid * stride_deltah
        + offs_m * stride_deltam
    )
    tl.store(delta_ptr + offs_delta, delta, mask=mask_m)

@triton.jit
def la_bwd_dkdv_inner(
    Q, K, V, DO, M, Delta, DK, DV, batch_num_block_n,
    stride_qm, stride_qh, stride_qk,
    stride_kn, stride_kh, stride_kk,
    stride_vn, stride_vh, stride_vk,
    stride_dom, stride_doh, stride_dok,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dkn_out, stride_dkh_out, stride_dkk_out,
    stride_dvn_out, stride_dvh_out, stride_dvk_out,
    sm_scale, N_CTX_Q, SEQLEN_K,
    total_n_blocks_all_batches, total_tiles_kv,
    high_load_wgs_kv, max_tiles_per_wg_kv,
    H: tl.constexpr, B: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, MAX_TILES_PER_WG_KV_CONST: tl.constexpr,
    NUM_M_BLOCKS_TOTAL: tl.constexpr, RAGGED_BATCHING: tl.constexpr,
    PREFETCH_QT: tl.constexpr,
    waves_per_eu: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # fast exp2
    RCP_LN2: tl.constexpr = 1.4426950408889634

    # stride hints
    tl.assume(stride_qm > 0); tl.assume(stride_qh > 0); tl.assume(stride_qk > 0)
    tl.assume(stride_kn > 0); tl.assume(stride_kh > 0); tl.assume(stride_kk > 0)
    tl.assume(stride_vn > 0); tl.assume(stride_vh > 0); tl.assume(stride_vk > 0)
    tl.assume(stride_dom > 0); tl.assume(stride_doh > 0); tl.assume(stride_dok > 0)
    tl.assume(stride_mb > 0); tl.assume(stride_mh > 0); tl.assume(stride_mm > 0)
    tl.assume(stride_deltab > 0); tl.assume(stride_deltah > 0); tl.assume(stride_deltam > 0)
    tl.assume(stride_dkn_out > 0); tl.assume(stride_dkh_out > 0); tl.assume(stride_dkk_out > 0)
    tl.assume(stride_dvn_out > 0); tl.assume(stride_dvh_out > 0); tl.assume(stride_dvk_out > 0)

    # even remainder split of KV tiles
    if pid < high_load_wgs_kv:
        iter_start_kv = max_tiles_per_wg_kv * pid
        num_to_process_kv = max_tiles_per_wg_kv
    else:
        iter_start_kv = (max_tiles_per_wg_kv - 1) * (pid - high_load_wgs_kv) + high_load_wgs_kv * max_tiles_per_wg_kv
        num_to_process_kv = max_tiles_per_wg_kv - 1

    # causal mask alignment: location of the diagonal when SEQLEN_Q != SEQLEN_K
    CAUSAL_SHIFT = N_CTX_Q - SEQLEN_K

    # -------------------- per-CTA KV tiles --------------------
    for i in range(0, num_to_process_kv):
        iter_kv = iter_start_kv + i
        valid_kv = iter_kv < total_tiles_kv
        if valid_kv:
            kv_id = iter_kv
            head_idx = tl.minimum(kv_id % H, H - 1)
            n_linear = kv_id // H

            # map linear n-block -> (batch_idx, block_in_batch) and per-batch prefix size
            if CAUSAL:
                n_per_batch = total_n_blocks_all_batches // B
                batch_idx = n_linear // n_per_batch
                n_block_in_batch = n_linear % n_per_batch
                b_seq_size = batch_idx * n_per_batch
            else:
                if RAGGED_BATCHING:
                    prev_running = 0
                    match_prev = 0
                    found = 0
                    batch_idx = 0
                    for b in range(0, B):
                        cur = tl.load(batch_num_block_n + b, mask=True, other=0) if b > 0 else tl.load(batch_num_block_n, mask=True, other=0)
                        is_match = (found == 0) & (n_linear < cur)
                        batch_idx = tl.where(is_match, b, batch_idx)
                        match_prev = tl.where(is_match, prev_running, match_prev)
                        found = tl.where(is_match, 1, found)
                        prev_running = cur
                    n_block_in_batch = n_linear - match_prev
                    b_seq_size = 0 if batch_idx == 0 else tl.load(batch_num_block_n + batch_idx - 1, mask=True, other=0)
                else:
                    per_batch_blocks = total_n_blocks_all_batches // B
                    batch_idx = n_linear // per_batch_blocks
                    n_block_in_batch = n_linear % per_batch_blocks
                    b_seq_size = batch_idx * per_batch_blocks

            # ----- load K/V tile once -----
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
            k_tile = tl.load(K + k_offs, mask=True, other=0.0)
            # Delay loading V tile to shorten live range; see inner loop
            k_start_abs = (b_seq_size + n_block_in_batch) * BLOCK_N

            # accumulators
            dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

            # ----- set up "transposed-domain" pointers for a single pass over Q/DO/M/Delta -----
            q_base_abs = batch_idx * N_CTX_Q
            qT_ptrs = (
                Q + q_base_abs * stride_qm + head_idx * stride_qh
                + offs_k[:, None] * stride_qk + offs_m[None, :] * stride_qm
            )
            doT_ptrs = (
                DO + q_base_abs * stride_dom + head_idx * stride_doh
                + offs_k[:, None] * stride_dok + offs_m[None, :] * stride_dom
            )
            m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + offs_m * stride_mm
            delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + offs_m * stride_deltam

            qT_ptrs = tl.multiple_of(qT_ptrs, (16, 1))
            doT_ptrs = tl.multiple_of(doT_ptrs, (16, 1))

            # dynamic loop bound (avoid big unroll)
            q_abs = q_base_abs
            rows_limit = (batch_idx + 1) * N_CTX_Q            # end row (exclusive) for this batch

            # Causal pruning: skip Q blocks below the diagonal start for this K tile
            if CAUSAL:
                q_min_start = k_start_abs + CAUSAL_SHIFT
                if q_min_start > q_abs:
                    skip_blocks = (q_min_start - q_abs + BLOCK_M - 1) // BLOCK_M
                    q_abs += skip_blocks * BLOCK_M
                    qT_ptrs += skip_blocks * BLOCK_M * stride_qm
                    doT_ptrs += skip_blocks * BLOCK_M * stride_dom
                    m_ptrs += skip_blocks * BLOCK_M * stride_mm
                    delta_ptrs += skip_blocks * BLOCK_M * stride_deltam
            # If after pruning we are beyond the batch rows, nothing to do
            num_iters_kv = 0
            if q_abs < rows_limit:
                num_iters_kv = (rows_limit - q_abs + BLOCK_M - 1) // BLOCK_M

            # Software prefetch: optionally load first Qᵀ/DOᵀ ahead of compute
            q_tile_T = tl.load(qT_ptrs, mask=True)
            do_tile_T = tl.load(doT_ptrs, mask=True)
            next_qT_ptrs = qT_ptrs + BLOCK_M * stride_qm
            next_doT_ptrs = doT_ptrs + BLOCK_M * stride_dom
            for it in range(0, num_iters_kv):
                # Ensure current tiles are loaded when software prefetch is disabled
                if PREFETCH_QT <= 1:
                    q_tile_T = tl.load(qT_ptrs, mask=True)
                    do_tile_T = tl.load(doT_ptrs, mask=True)
                # per-row LSE stats
                mask_mrows = (q_abs + offs_m) < rows_limit
                m_rows = tl.load(m_ptrs,     mask=mask_mrows, other=-float("inf"))
                delta_rows = tl.load(delta_ptrs, mask=mask_mrows, other=0.0)

                # logits in transposed domain: (BLOCK_N, BLOCK_M)
                qk_T = tl.dot(k_tile, q_tile_T) * sm_scale

                # causal mask (aligned to diagonal when seq_q != seq_k)
                if CAUSAL:
                    mask_T = (q_abs + offs_m[None, :]) >= (k_start_abs + offs_n[:, None] + CAUSAL_SHIFT)
                    qk_T = tl.where(mask_T, qk_T, -float("inf"))

                # probabilities using exp2 and stored m_rows
                # p_T = exp2(qk - m) ; also wipe NaNs (appear when all masked)
                p_T = tl.math.exp2(qk_T * RCP_LN2 - m_rows[None, :] * RCP_LN2)
                p_T = tl.where(p_T == p_T, p_T, 0.0)   # NaN -> 0
                p_T = tl.where(mask_mrows[None, :], p_T, 0.0)

                # --------- accumulate dV (one GEMM)
                # dV += Pᵀ · dO  but in our shapes we already have P in (N, M) and dOᵀ is (K, M)
                dv_acc += tl.dot(p_T.to(do_tile_T.type.element_ty), tl.trans(do_tile_T))

                # --------- accumulate dK (one GEMM)
                # dp_T = V · dOᵀ  (N, K) x (K, M) = (N, M)
                # Use low-precision inputs to enable TC/MFMA with fp32 accumulate
                # Load V tile just-in-time to shorten live range
                v_tile = tl.load(V + v_offs, mask=True, other=0.0)
                dp_T = tl.dot(v_tile, do_tile_T)
                dp_T = tl.where(mask_mrows[None, :], dp_T, 0.0)
                delta_cols = tl.where(mask_mrows, delta_rows, 0.0)
                # ds_T = P ⊙ (dp_T - Δ)
                ds_T = p_T * (dp_T - delta_cols[None, :])
                # dK += ds_T · Qᵀ   (N, M) x (M, K) = (N, K)
                # Cast ds_T to low precision to trigger TC/MFMA; accumulate into fp32
                dk_acc += tl.dot(ds_T.to(q_tile_T.type.element_ty), tl.trans(q_tile_T))

                # advance
                q_abs += BLOCK_M
                qT_ptrs += BLOCK_M * stride_qm
                doT_ptrs += BLOCK_M * stride_dom
                m_ptrs += BLOCK_M * stride_mm
                delta_ptrs += BLOCK_M * stride_deltam

                # Software prefetch for next iteration
                if PREFETCH_QT > 1:
                    do_prefetch = it + 1 < num_iters_kv
                    if do_prefetch:
                        q_tile_T = tl.load(next_qT_ptrs, mask=True)
                        do_tile_T = tl.load(next_doT_ptrs, mask=True)
                        next_qT_ptrs += BLOCK_M * stride_qm
                        next_doT_ptrs += BLOCK_M * stride_dom
                    # else keep last loaded tiles (won't be used)

            # ----- store -----
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
            # Mask tail rows at the end of each batch's K-range to avoid OOB stores
            # Determine number of n-blocks in this batch
            n_blocks_in_batch = 0
            if CAUSAL:
                n_blocks_in_batch = total_n_blocks_all_batches // B
            else:
                if RAGGED_BATCHING:
                    prev_total = tl.where(batch_idx == 0, 0, tl.load(batch_num_block_n + batch_idx - 1, mask=True, other=0))
                    cur_total = tl.load(batch_num_block_n + batch_idx, mask=True, other=0)
                    n_blocks_in_batch = cur_total - prev_total
                else:
                    n_blocks_in_batch = total_n_blocks_all_batches // B
            batch_end_abs = (b_seq_size + n_blocks_in_batch) * BLOCK_N
            mask_n_store = (k_start_abs + offs_n) < batch_end_abs
            store_mask = mask_n_store[:, None]
            tl.store(tl.multiple_of(dv_ptrs_out, (1, 16)), dv_acc.to(DV.type.element_ty), mask=store_mask)
            tl.store(tl.multiple_of(dk_ptrs_out, (1, 16)), (dk_acc * sm_scale).to(DK.type.element_ty), mask=store_mask)


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
    SEQLEN_K,
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
    MAX_N_BLOCKS_PER_BATCH_CONST: tl.constexpr,
    RAGGED_BATCHING: tl.constexpr,
    PREFETCH_KV: tl.constexpr,
    waves_per_eu: tl.constexpr,
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

        rows_in_tile = tl.minimum(BLOCK_M, N_CTX_Q - m_block * BLOCK_M)
        mask_m = (offs_m < rows_in_tile) & valid_q
        full_rows = rows_in_tile == BLOCK_M


        # Q/DO/M/Delta pointers for this Q tile
        q_start_abs = batch_idx * N_CTX_Q + m_block * BLOCK_M
        q_ptrs = Q + q_start_abs * stride_qm + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        do_ptrs = DO + q_start_abs * stride_dom + head_idx * stride_doh + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        m_ptrs = M + batch_idx * stride_mb + head_idx * stride_mh + (m_block * BLOCK_M + offs_m) * stride_mm
        delta_ptrs = Delta + batch_idx * stride_deltab + head_idx * stride_deltah + (m_block * BLOCK_M + offs_m) * stride_deltam
        q_ptrs = tl.multiple_of(q_ptrs, (1, 16))
        do_ptrs = tl.multiple_of(do_ptrs, (1, 16))

        if full_rows:
            q_tile   = tl.load(q_ptrs)
            do_tile  = tl.load(do_ptrs)
            m_rows   = tl.load(m_ptrs)
            delta_rows = tl.load(delta_ptrs)
        else:
            q_tile   = tl.load(q_ptrs,  mask=mask_m[:, None], other=0.0)
            do_tile  = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            m_rows   = tl.load(m_ptrs,  mask=mask_m, other=-float("inf"))
            delta_rows = tl.load(delta_ptrs, mask=mask_m, other=0.0)

        # Determine this batch's K/V n-block range
        num_n_blocks_per_batch = 0
        b_seq_size_blocks = 0
        if CAUSAL:
            num_n_blocks_per_batch = total_n_blocks_all_batches // B
            b_seq_size_blocks = batch_idx * num_n_blocks_per_batch
        else:
            if not RAGGED_BATCHING:
                # Uniform batch sizes
                num_n_blocks_per_batch = total_n_blocks_all_batches // B
                b_seq_size_blocks = batch_idx * num_n_blocks_per_batch
            else:
                # Ragged batch sizes
                blocks_prev = 0
                for b in tl.static_range(0, B):
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
        # Causal pruning: determine [start_n, end_n) bounds for this Q tile
        start_n_blocks = 0
        end_n_blocks = num_n_blocks_per_batch
        if CAUSAL:
            # Diagonal index for the top of this Q tile
            # q_start_abs + m runs [q_start_abs, q_start_abs + BLOCK_M)
            # Valid K indices satisfy: q >= k + (N_CTX_Q - SEQLEN_K)
            # Solve for k: k <= q - (N_CTX_Q - SEQLEN_K)
            # Using the smallest q in the tile gives the tightest start bound
            k_max_for_tile = (q_start_abs) - (N_CTX_Q - SEQLEN_K)
            end_n_blocks = tl.maximum(0, tl.minimum(num_n_blocks_per_batch, (k_max_for_tile - k_start_abs) // BLOCK_N + 1))
            # Using the largest q in the tile for the lower bound
            k_min_for_tile = (q_start_abs + BLOCK_M - 1) - (N_CTX_Q - SEQLEN_K)
            start_n_blocks = tl.maximum(0, tl.minimum(num_n_blocks_per_batch, (k_min_for_tile - k_start_abs - (BLOCK_N - 1)) // BLOCK_N))
            start_n_blocks = tl.minimum(start_n_blocks, end_n_blocks)

        # Dynamic loop to avoid heavy unrolling and register pressure
        # Prefetch first K tile
        k_tile = tl.load(k_ptrs, mask=valid_q, other=0.0)
        for it in range(start_n_blocks, end_n_blocks):
            # Ensure current K tile is loaded when software prefetch is disabled
            if PREFETCH_KV <= 1:
                k_tile = tl.load(k_ptrs, mask=valid_q, other=0.0)
            # Consume prefetched K tile

            # Reconstruct logits and probabilities using exp2
            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            if CAUSAL:
                mask = (q_start_abs + offs_m[:, None]) >= (k_start_abs + offs_n[None, :]) + (N_CTX_Q - SEQLEN_K)
                qk = tl.where(mask, qk, -float("inf"))
            p = tl.math.exp2(qk * RCP_LN2 - m_rows[:, None] * RCP_LN2)
            p = tl.where(p != p, 0, p)

            # Load V tile only when needed for dP to shorten live range
            v_tile = tl.load(v_ptrs, mask=valid_q, other=0.0)
            # Form dS and accumulate contribution to dQ with higher precision
            dp = tl.dot(do_tile.to(tl.float32), tl.trans(v_tile).to(tl.float32))
            ds = p.to(tl.float32) * (dp - delta_rows[:, None])
            # Cast ds to low precision to trigger TC/MFMA; accumulate into fp32 accumulator
            dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

            # Advance pointers and absolute K start; prefetch next K tile if enabled
            k_ptrs += BLOCK_N * stride_kn
            v_ptrs += BLOCK_N * stride_vn
            k_start_abs += BLOCK_N
            if PREFETCH_KV > 1 and (it + 1) < end_n_blocks:
                k_tile = tl.load(k_ptrs, mask=valid_q, other=0.0)

        # Final scale and store dQ tile
        dq_acc = dq_acc * sm_scale
        dq_ptrs_out = (
            DQ
            + q_start_abs * stride_dqm
            + head_idx * stride_dqh
            + offs_m[:, None] * stride_dqm
            + offs_k[None, :] * stride_dqk
        )
        dq_ptrs_out = tl.multiple_of(dq_ptrs_out, (1, 16))
        if full_rows:
            tl.store(dq_ptrs_out, dq_acc.to(DQ.type.element_ty))
        else:
            tl.store(dq_ptrs_out, dq_acc.to(DQ.type.element_ty), mask=mask_m[:, None])


@triton.jit
def la_bwd_kv_streamk(
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
    SEQLEN_K,
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
    MAX_TILES_PER_WG_KV_CONST: tl.constexpr,
    NUM_M_BLOCKS_TOTAL: tl.constexpr,
    RAGGED_BATCHING: tl.constexpr,
    PREFETCH_QT: tl.constexpr,
    waves_per_eu: tl.constexpr,
):
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
        sm_scale, N_CTX_Q, SEQLEN_K,
        total_n_blocks_all_batches, total_tiles_kv,
        high_load_wgs_kv, max_tiles_per_wg_kv,
        H=H,
        B=B,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=CAUSAL,
        MAX_TILES_PER_WG_KV_CONST=MAX_TILES_PER_WG_KV_CONST,
        NUM_M_BLOCKS_TOTAL=NUM_M_BLOCKS_TOTAL,
        RAGGED_BATCHING=RAGGED_BATCHING,
        PREFETCH_QT=PREFETCH_QT,
        waves_per_eu=waves_per_eu,
    )


@triton.jit
def la_bwd_q_streamk(
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
    SEQLEN_K,
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
    MAX_N_BLOCKS_PER_BATCH_CONST: tl.constexpr,
    RAGGED_BATCHING: tl.constexpr,
    PREFETCH_KV: tl.constexpr,
    waves_per_eu: tl.constexpr,
):
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
        SEQLEN_K,
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
        MAX_N_BLOCKS_PER_BATCH_CONST=MAX_N_BLOCKS_PER_BATCH_CONST,
        RAGGED_BATCHING=RAGGED_BATCHING,
        PREFETCH_KV=PREFETCH_KV,
        waves_per_eu=waves_per_eu,
    )


@triton.jit
def la_bwd_fused_streamk(
    Q, K, V, DO, M, Delta, DQ, DK, DV, batch_num_block_n, kv_batch_num_block_n,
    stride_qm, stride_qh, stride_qk,
    stride_kn, stride_kh, stride_kk,
    stride_vn, stride_vh, stride_vk,
    stride_dom, stride_doh, stride_dok,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dqm, stride_dqh, stride_dqk,
    stride_dkn_out, stride_dkh_out, stride_dkk_out,
    stride_dvn_out, stride_dvh_out, stride_dvk_out,
    sm_scale,
    N_CTX_Q, SEQLEN_K,
    total_n_blocks_all_batches, total_tiles_q, high_load_wgs_q, max_tiles_per_wg_q,
    total_n_blocks_all_batches_kv, total_kv_tiles_kv, high_load_wgs_kv_kv, max_tiles_per_wg_kv_kv,
    H: tl.constexpr, B: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M_Q: tl.constexpr, BLOCK_N_Q: tl.constexpr,
    BLOCK_M_KV: tl.constexpr, BLOCK_N_KV: tl.constexpr,
    CAUSAL: tl.constexpr,
    MAX_N_BLOCKS_PER_BATCH_CONST: tl.constexpr,
    MAX_TILES_PER_WG_KV_CONST: tl.constexpr,
    RAGGED_BATCHING: tl.constexpr,
    PREFETCH_QT: tl.constexpr,
    PREFETCH_KV: tl.constexpr,
    waves_per_eu: tl.constexpr,
):
    # Phase 1: dK/dV with KV tiling
    la_bwd_dkdv_inner(
        Q, K, V, DO, M, Delta, DK, DV, kv_batch_num_block_n,
        stride_qm, stride_qh, stride_qk,
        stride_kn, stride_kh, stride_kk,
        stride_vn, stride_vh, stride_vk,
        stride_dom, stride_doh, stride_dok,
        stride_mb, stride_mh, stride_mm,
        stride_deltab, stride_deltah, stride_deltam,
        stride_dkn_out, stride_dkh_out, stride_dkk_out,
        stride_dvn_out, stride_dvh_out, stride_dvk_out,
        sm_scale, N_CTX_Q, SEQLEN_K,
        total_n_blocks_all_batches_kv, total_kv_tiles_kv, high_load_wgs_kv_kv, max_tiles_per_wg_kv_kv,
        H=H, B=B, HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M_KV, BLOCK_N=BLOCK_N_KV,
        CAUSAL=CAUSAL, MAX_TILES_PER_WG_KV_CONST=MAX_TILES_PER_WG_KV_CONST,
        NUM_M_BLOCKS_TOTAL=(N_CTX_Q + BLOCK_M_KV - 1) // BLOCK_M_KV,
        RAGGED_BATCHING=RAGGED_BATCHING,
        PREFETCH_QT=PREFETCH_QT,
        waves_per_eu=waves_per_eu,
    )

    # Phase 2: dQ with Q tiling
    la_bwd_dq_inner(
        Q, K, V, DO, M, Delta, DQ, batch_num_block_n,
        stride_qm, stride_qh, stride_qk,
        stride_kn, stride_kh, stride_kk,
        stride_vn, stride_vh, stride_vk,
        stride_dom, stride_doh, stride_dok,
        stride_mb, stride_mh, stride_mm,
        stride_deltab, stride_deltah, stride_deltam,
        stride_dqm, stride_dqh, stride_dqk,
        sm_scale, N_CTX_Q, SEQLEN_K,
        total_n_blocks_all_batches, total_tiles_q, high_load_wgs_q, max_tiles_per_wg_q,
        H=H, B=B, HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M_Q, BLOCK_N=BLOCK_N_Q,
        CAUSAL=CAUSAL, MAX_N_BLOCKS_PER_BATCH_CONST=MAX_N_BLOCKS_PER_BATCH_CONST,
        RAGGED_BATCHING=RAGGED_BATCHING,
        PREFETCH_KV=PREFETCH_KV,
        waves_per_eu=waves_per_eu,
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
    seqlen_k: int = 0,
    config: Optional[dict] = None,
    num_programs: Optional[int] = 304,
    profile_split: bool = False,
):
    """
    Host-side launcher for the backward pass.
    - Default: fused two-phase kernel (dK/dV then dQ) for performance.
    - If profile_split=True: launches split dK/dV and dQ kernels separately with CUDA timing.
    """
    if config is None:
        # Use a minimal default to avoid external JSON dependency during tests
        config = {
        "split_kernels": True,
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_warps": 4,
        "BLOCK_SIZE_M_KV": 32, "BLOCK_SIZE_N_KV": 128, "num_warps_kv": 2, "waves_per_eu": 1, "num_programs_mult": 3,
        }
    #Optional: override with tuned DB (env AITER_BWD_TUNED_DB)
    try:
        db_path = os.environ.get("AITER_BWD_TUNED_DB")
        if db_path:
            H_tmp = q.shape[1]
            D_tmp = q.shape[-1]
            NQ_tmp = q.shape[0] // batch_size
            NK_tmp = k.shape[0]
            db = _load_bwd_tuned_db(db_path)
            tuned = _select_bwd_config(db, causal, batch_size, H_tmp, D_tmp, NQ_tmp, NK_tmp)
            tuned_np_mult = None
            if tuned:
                tuned_np_mult = tuned.get("num_programs_mult")
                # do not override num_programs from DB; user controls it
                tuned = {k: v for k, v in tuned.items() if k != "num_programs" and k != "num_programs_mult"}
                config.update(tuned)
    except Exception:
        pass
    # Resolve total programs (CTAs) preference: explicit arg > config > device
    total_programs_pref = None
    sm_count = None
    if num_programs is not None:
        total_programs_pref = int(num_programs)
    elif "num_ctas" in config:
        total_programs_pref = int(config["num_ctas"])
    else:
        try:
            sm_count = int(arch_info.get_num_sms())
            total_programs_pref = sm_count
        except Exception:
            sm_count = int(torch.cuda.get_device_properties(q.device).multi_processor_count)
            total_programs_pref = sm_count

    #If enabled, apply tuned grid size from DB multiplier when user hasn't set num_programs explicitly
    try:
        use_tuned_grid = os.environ.get("AITER_BWD_USE_TUNED_GRID", "0") == "1"
        if use_tuned_grid and (num_programs is None):
            # tuned_np_mult set above when DB entry exists
            if 'tuned' in locals():
                tuned_np_mult = locals().get('tuned_np_mult', None)
                if tuned_np_mult and sm_count:
                    total_programs_pref = max(1, int(tuned_np_mult) * int(sm_count))
    except Exception:
        pass

    print(f"config: {config}")

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

    BLOCK_M_KV = int(config.get("BLOCK_SIZE_M_KV", BLOCK_M))
    BLOCK_N_KV = int(config.get("BLOCK_SIZE_N_KV", BLOCK_N))

    (
        total_n_blocks_all_batches,
        max_tiles_per_wg_kv,
        high_load_wgs_kv,
        num_m_blocks_total,
        total_tiles_q,
        max_tiles_per_wg_q,
        high_load_wgs_q,
        num_m_blocks_total_kv,
        total_n_blocks_all_batches_kv,
        total_kv_tiles_kv,
        max_tiles_per_wg_kv_kv,
        high_load_wgs_kv_kv,
    ) = get_num_splits_and_buffer_sizes_bwd(
        batch_size,
        H,
        N_CTX_Q,
        N_CTX_K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_M_KV,
        BLOCK_N_KV,
        total_programs_pref,
    )

    grid = (total_programs_pref,)

    # NOTE: `DqOp` and `locks` are not used by the current design;
    # retained here only for potential experimentation with host-block reductions.
    DqOp = torch.empty(
        (total_programs_pref, BLOCK_M, HEAD_DIM), device=q.device, dtype=dq.dtype
    )
    locks = torch.zeros((total_programs_pref,), device=q.device, dtype=torch.int32)

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
    
    ragged_batching = False
    if not causal and batch_size > 1:
        # If any per-batch size differs, mark as ragged
        sizes = torch.diff(torch.cat((torch.tensor([0], device=batch_num_block_n.device, dtype=batch_num_block_n.dtype),
                                      batch_num_block_n)))
        ragged_batching = (sizes != sizes[0]).any().item()

    # Compute Delta on device: [B, H, Nq]
    Delta = torch.empty((batch_size, H, N_CTX_Q), device=q.device, dtype=torch.float32)
    pre_grid = (
        triton.cdiv(N_CTX_Q, BLOCK_M),
        batch_size,
        H,
    )
    la_bwd_preprocess[pre_grid](
        o,
        do,
        Delta,
        o.stride(0),  # stride_om (B*N_CTX_Q)
        o.stride(1),  # stride_oh (H)
        o.stride(2),  # stride_ok (K)
        Delta.stride(0),  # stride_deltab (B)
        Delta.stride(1),  # stride_deltah (H)
        Delta.stride(2),  # stride_deltam (M)
        N_CTX_Q,
        BLOCK_M=BLOCK_M,
        HEAD_DIM=HEAD_DIM,
    )
    # Outputs are fully written by kernels with proper masking; avoid extra global zeroing

    # --- Launch fused kernel by default ---
    # Build KV-specific batch_num_block_n if non-causal to match KV BLOCK_N_KV mapping
    kv_batch_num_block_n = batch_num_block_n
    if (not causal):
        num_n_blocks_total_kv = (N_CTX_K + BLOCK_N_KV - 1) // BLOCK_N_KV
        if batch_size > 1:
            per_batch_blocks_kv = num_n_blocks_total_kv // batch_size
            kv_batch_num_block_n = (
                torch.arange(1, batch_size + 1, device=q.device, dtype=torch.int32)
                * per_batch_blocks_kv
            )
        else:
            kv_batch_num_block_n = torch.tensor([num_n_blocks_total_kv], device=q.device, dtype=torch.int32)

    # Config-driven prefetch defaults
    prefetch_qt = int(config.get("prefetch_qt", 2))
    prefetch_kv = int(config.get("prefetch_kv", 2))

    # If user didn't specify num_programs/ctas, prefer 2x SMs by default
    if num_programs is None and ("num_ctas" not in config):
        try:
            if sm_count is None:
                sm_count = int(arch_info.get_num_sms())
        except Exception:
            sm_count = int(torch.cuda.get_device_properties(q.device).multi_processor_count)
        total_programs_pref = max(1, 2 * int(sm_count))
        grid = (total_programs_pref,)

    if profile_split:
        # Time split kernels: dK/dV then dQ
        kernel_timing = {
            "dkdv": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0.0,
            },
            "dq": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0.0,
            },
        }
        # dK/dV
        kernel_timing["dkdv"]["start_event"].record()
        kv_kernel = la_bwd_dkdv_inner[grid](
            q, k, v, do, softmax_lse, Delta, dk, dv, kv_batch_num_block_n,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            sm_scale, N_CTX_Q, seqlen_k,
            total_n_blocks_all_batches_kv, total_kv_tiles_kv, high_load_wgs_kv_kv, max_tiles_per_wg_kv_kv,
            H=H, B=batch_size, HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M_KV, BLOCK_N=BLOCK_N_KV,
            CAUSAL=causal, MAX_TILES_PER_WG_KV_CONST=max_tiles_per_wg_kv_kv,
            NUM_M_BLOCKS_TOTAL=num_m_blocks_total_kv,
            RAGGED_BATCHING=ragged_batching,
            PREFETCH_QT=prefetch_qt,
            waves_per_eu=config["waves_per_eu"],
            num_warps=config.get("num_warps", 2),
            num_stages=config.get("num_stages", 2),
        )
        kernel_timing["dkdv"]["end_event"].record()
        torch.cuda.synchronize()
        ms = kernel_timing["dkdv"]["start_event"].elapsed_time(kernel_timing["dkdv"]["end_event"])
        kernel_timing["dkdv"]["ms"] += ms

        # dQ
        kernel_timing["dq"]["start_event"].record()
        q_kernel = la_bwd_dq_inner[grid](
            q, k, v, do, softmax_lse, Delta, dq, batch_num_block_n,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            sm_scale, N_CTX_Q, seqlen_k,
            total_n_blocks_all_batches, total_tiles_q, high_load_wgs_q, max_tiles_per_wg_q,
            H=H, B=batch_size, HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=causal, MAX_N_BLOCKS_PER_BATCH_CONST=(N_CTX_K + BLOCK_N - 1) // BLOCK_N,
            RAGGED_BATCHING=ragged_batching,
            PREFETCH_KV=prefetch_kv,
            waves_per_eu=config["waves_per_eu"],
            num_warps=config.get("num_warps", 2),
            num_stages=config.get("num_stages", 2),
        )
        kernel_timing["dq"]["end_event"].record()
        torch.cuda.synchronize()
        ms = kernel_timing["dq"]["start_event"].elapsed_time(kernel_timing["dq"]["end_event"])
        kernel_timing["dq"]["ms"] += ms

        print(f"la bwd split timing: dkdv={kernel_timing['dkdv']['ms']:.3f} ms, dq={kernel_timing['dq']['ms']:.3f} ms")
        print(f"la bwd split kv kernel {kv_kernel.n_regs} regs, {kv_kernel.n_spills} spills; q kernel {q_kernel.n_regs} regs, {q_kernel.n_spills} spills")
        return
    kernel_timing = {
            "fused": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0.0,
            }
        }
        # dK/dV
    kernel_timing["fused"]["start_event"].record()
    fused_kernel = la_bwd_fused_streamk[grid](
        q, k, v, do, softmax_lse, Delta, dq, dk, dv, batch_num_block_n, kv_batch_num_block_n,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        Delta.stride(0), Delta.stride(1), Delta.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        sm_scale, N_CTX_Q, seqlen_k,
        total_n_blocks_all_batches, total_tiles_q, high_load_wgs_q, max_tiles_per_wg_q,
        total_n_blocks_all_batches_kv, total_kv_tiles_kv, high_load_wgs_kv_kv, max_tiles_per_wg_kv_kv,
        H=H, B=batch_size, HEAD_DIM=HEAD_DIM,
        BLOCK_M_Q=BLOCK_M, BLOCK_N_Q=BLOCK_N,
        BLOCK_M_KV=BLOCK_M_KV, BLOCK_N_KV=BLOCK_N_KV,
        CAUSAL=causal,
        MAX_N_BLOCKS_PER_BATCH_CONST=(N_CTX_K + BLOCK_N - 1) // BLOCK_N,
        MAX_TILES_PER_WG_KV_CONST=max_tiles_per_wg_kv_kv,
        RAGGED_BATCHING=ragged_batching,
        PREFETCH_QT=prefetch_qt,
        PREFETCH_KV=prefetch_kv,
        waves_per_eu=config["waves_per_eu"],
        num_warps=config.get("num_warps", 2),
        num_stages=config.get("num_stages", 2),
    )
    kernel_timing["fused"]["end_event"].record()
    torch.cuda.synchronize()
    ms = kernel_timing["fused"]["start_event"].elapsed_time(kernel_timing["fused"]["end_event"])
    kernel_timing["fused"]["ms"] += ms
    print(f"la bwd fused kernel {fused_kernel.n_regs} regs, {fused_kernel.n_spills} spills, {ms:.3f} ms")
    return
    # Spill guard: fallback to num_stages=1 if needed
    if fused_kernel.n_spills > 0:
        fused_kernel = la_bwd_fused_streamk[grid](
            q, k, v, do, softmax_lse, Delta, dq, dk, dv, batch_num_block_n, kv_batch_num_block_n,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            sm_scale, N_CTX_Q, seqlen_k,
            total_n_blocks_all_batches, total_tiles_q, high_load_wgs_q, max_tiles_per_wg_q,
            total_n_blocks_all_batches_kv, total_kv_tiles_kv, high_load_wgs_kv_kv, max_tiles_per_wg_kv_kv,
            H=H, B=batch_size, HEAD_DIM=HEAD_DIM,
            BLOCK_M_Q=BLOCK_M, BLOCK_N_Q=BLOCK_N,
            BLOCK_M_KV=BLOCK_M_KV, BLOCK_N_KV=BLOCK_N_KV,
            CAUSAL=causal,
            MAX_N_BLOCKS_PER_BATCH_CONST=(N_CTX_K + BLOCK_N - 1) // BLOCK_N,
            MAX_TILES_PER_WG_KV_CONST=max_tiles_per_wg_kv_kv,
            RAGGED_BATCHING=ragged_batching,
            PREFETCH_QT=prefetch_qt,
            PREFETCH_KV=prefetch_kv,
            waves_per_eu=config["waves_per_eu"],
            num_warps=config.get("num_warps", 2),
            num_stages=1,  # Force num_stages=1 for spill guard
        )
    print(f"la bwd fused kernel {fused_kernel.n_regs} regs, {fused_kernel.n_spills} spills")
    return