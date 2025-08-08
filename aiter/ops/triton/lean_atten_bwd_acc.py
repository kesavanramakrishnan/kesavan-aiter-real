import math
import torch
import triton
import triton.language as tl
from aiter.ops.triton.lean_atten_val import (
    get_num_splits_and_buffer_sizes as fwd_get_num_splits_and_buffer_sizes,
    LOG_TWO_E,
)
import aiter.ops.triton.utils.arch_info as arch_info

# -----------------------------------------------------------------------------
# Helpers mirrored from forward pass scheduling (lean_atten_val.py)
# -----------------------------------------------------------------------------

@triton.jit
def _swizzle_by_grouping(pid: tl.int32, GROUP_SIZE: tl.constexpr) -> tl.int32:
    group_id = pid // GROUP_SIZE
    pid_in_group = pid % GROUP_SIZE
    swizzled_pid = group_id + (pid_in_group * (tl.num_programs(0) // GROUP_SIZE))
    return swizzled_pid

@triton.jit
def _find_group(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
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


# -----------------------------------------------------------------------------
# Work-centric persistent backward kernel (lean tiles), forward-style scheduling
# -----------------------------------------------------------------------------

@triton.jit
def la_bwd_persistent(
    Q,
    K,
    V,
    DO,
    M,
    Delta,
    DQ,
    DK,
    DV,
    DQp,
    locks,
    qk_scale,
    # Strides
    stride_qm, stride_qh, stride_qk,
    stride_kn, stride_kh, stride_kk,
    stride_vn, stride_vh, stride_vk,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    # Meta / scheduling
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
    pid = tl.program_id(0)
    GROUP_SIZE = 12
    pid = _swizzle_by_grouping(pid, GROUP_SIZE)

    # Compute global tile range for this CTA
    if pid < high_load_wgs:
        iter = max_tiles_per_wg * pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
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
    tl.assume(stride_dok > 0)
    tl.assume(stride_mm > 0)
    tl.assume(stride_deltam > 0)
    tl.assume(stride_dqm > 0)
    tl.assume(stride_dqk > 0)
    tl.assume(stride_dkn > 0)
    tl.assume(stride_dkk > 0)
    tl.assume(stride_dvn > 0)
    tl.assume(stride_dvk > 0)

    for _ in tl.static_range(1024):
        if iter < cta_end_tile_gid:
            # Map iter to head and (batch) tile
            tiles_per_head_local = tiles_per_head
            tile_head_idx = iter // tiles_per_head_local
            if causal:
                tile_batch_idx = (iter % tiles_per_head_local) // (tiles_per_head_local // batch_size)
                per_head_tile_idx, per_head_tile_size, total_blocks = _find_group(
                    iter - (tile_head_idx * tiles_per_head_local) - (tile_batch_idx * (tiles_per_head_local // batch_size)),
                    MASKED_BLOCKS,
                    num_m_blocks,
                )
                tile_iter = tile_head_idx * tiles_per_head_local + (tile_batch_idx * (tiles_per_head_local // batch_size)) + total_blocks
                tile_iter_end = tile_iter + per_head_tile_size
                tile_idx = (tile_head_idx * batch_size + tile_batch_idx) * num_m_blocks + per_head_tile_idx
            else:
                tile_idx = tile_head_idx * batch_size
                tile_iter = tile_head_idx * tiles_per_head_local
                # For non-causal, ragged batching uses batch_num_block_n externally; here assume uniform for simplicity
                tile_iter_end = tile_iter + tiles_per_head_local // batch_size
                tile_batch_idx = tile_idx % batch_size

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
                b_seq_size = 0

        # Pointers for K/V at current local_iter
            k_offs = (
                (b_seq_size + local_iter) * BLOCK_N * stride_kn
                + tile_head_idx * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_k[None, :] * stride_kk
            )
            v_offs = (
                (b_seq_size + local_iter) * BLOCK_N * stride_vn
                + tile_head_idx * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_k[None, :] * stride_vk
            )
            k_ptrs = K + k_offs
            v_ptrs = V + v_offs

        # Q/DO pointers for this output tile's M block
            if causal:
                q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
                q_start_m = q_idx * BLOCK_M
            else:
                q_idx = tile_batch_idx
                q_start_m = q_idx * BLOCK_M
            q_offs = (
            q_idx * BLOCK_M * stride_qm
            + tile_head_idx * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
            )
            q_ptrs = Q + q_offs

        # Load block-wise constants
            q = tl.load(q_ptrs)
            dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Load M and Delta slices once
            m_ptrs = M + (tile_batch_idx * stride_mb + tile_head_idx * stride_mh + (q_start_m + offs_m) * stride_mm)
            m_tile = tl.load(m_ptrs, mask=(offs_m + q_start_m < num_m_blocks * BLOCK_M), other=-float("inf"))[:, None]
            delta_ptrs = Delta + (tile_batch_idx * stride_deltab + tile_head_idx * stride_deltah + (q_start_m + offs_m) * stride_deltam)
            Di = tl.load(delta_ptrs, mask=(offs_m + q_start_m < num_m_blocks * BLOCK_M), other=0.0)

        # Iterate local lean tiles (each is a K-block)
            for l_iter in range(local_iter, local_iter_end):
                mask_kv = (offs_n[:, None] < num_n_blocks * BLOCK_N) & (offs_k[None, :] < HEAD_DIM)
                k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
                v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

            # Recompute logits and probabilities with exp2 to match forward
                qk = tl.dot(q, tl.trans(k))
                qk = qk * qk_scale
                if causal:
                    k_start_n = (b_seq_size + l_iter) * BLOCK_N
                    cmask = (q_start_m + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
                    qk = tl.where(cmask, qk, float("-inf"))
                p = tl.math.exp2(qk - m_tile)

                do_offs = (
                    tile_batch_idx * stride_dob + tile_head_idx * stride_doh + (q_start_m + offs_m)[:, None] * stride_dom + offs_k[None, :] * stride_dok
                )
                do = tl.load(DO + do_offs)

            # dQ accumulation
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - Di[:, None])
                dq_acc += tl.dot(ds.to(tl.float32), k.to(tl.float32))

            # dV accumulation (transposed domain): dv += P^T @ dO
                kq = tl.dot(k, tl.trans(q))
                kq = kq * qk_scale
                if causal:
                    cmask_T = (q_start_m + offs_m[None, :]) >= ((b_seq_size + l_iter) * BLOCK_N + offs_n[:, None])
                    kq = tl.where(cmask_T, kq, float("-inf"))
                # Broadcast m over columns (q positions): [BLOCK_N, BLOCK_M] - [1, BLOCK_M]
                m_cols = m_tile[:, 0]
                p_T = tl.math.exp2(kq - m_cols[None, :])
                dp_T = tl.dot(v, tl.trans(do))
                ds_T = p_T * (dp_T - Di[None, :])

            # Atomic accumulate into DK/DV for this K-block tile
                dv_out_ptrs = DV + (
                    tile_batch_idx * stride_dvb + tile_head_idx * stride_dvh + ((b_seq_size + l_iter) * BLOCK_N + offs_n)[:, None] * stride_dvn + offs_k[None, :] * stride_dvk
                )
                dk_out_ptrs = DK + (
                    tile_batch_idx * stride_dkb + tile_head_idx * stride_dkh + ((b_seq_size + l_iter) * BLOCK_N + offs_n)[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
                )
                tl.atomic_add(dv_out_ptrs, tl.dot(p_T.to(do.type.element_ty), do), mask=mask_kv)
                tl.atomic_add(dk_out_ptrs, tl.dot(ds_T.to(q.type.element_ty), q) * (qk_scale / LOG_TWO_E), mask=mask_kv)

            # advance K/V pointers to next K-block
                v_ptrs += BLOCK_N * stride_vn
                k_ptrs += BLOCK_N * stride_kn

            # Store/reduce dQ partials
            dq_acc = dq_acc * (qk_scale / LOG_TWO_E)
            if not host_block:
                dq_partial_ptrs = DQp + pid * (BLOCK_M * HEAD_DIM) + offs_m[:, None] * HEAD_DIM + offs_k[None, :]
                tl.store(dq_partial_ptrs, dq_acc, cache_modifier=".wt")
                tl.atomic_xchg(locks + pid, 1)
            else:
                # host reduction across split CTAs if needed
                if not finishing_block:
                    last_cta = pid + 1
                    temp_end_gid = cta_end_tile_gid
                    split = 1
                    while (split < num_splits) and (temp_end_gid < tile_iter_end):
                        if last_cta < high_load_wgs:
                            temp_end_gid += tl.minimum(max_tiles_per_wg, tile_iter_end - temp_end_gid)
                        else:
                            temp_end_gid += tl.minimum(max_tiles_per_wg - 1, tile_iter_end - temp_end_gid)
                        last_cta += 1
                        split += 1

                    for cta in range(pid + 1, last_cta):
                        while tl.atomic_cas(locks + cta, 1, 1) != 1:
                            pass
                        dq_partial_ptrs = DQp + cta * (BLOCK_M * HEAD_DIM) + offs_m[:, None] * HEAD_DIM + offs_k[None, :]
                        dq_part = tl.load(dq_partial_ptrs)
                        dq_acc += dq_part

                # Write final dQ for this output tile
                dq_out_ptrs = DQ + (
                    tile_batch_idx * stride_dqb + tile_head_idx * stride_dqh + (q_start_m + offs_m)[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
                )
                tl.store(dq_out_ptrs, dq_acc.to(DQ.dtype.element_ty))

            iter = iter + (local_iter_end - local_iter)
@triton.jit
def bwd_dkdv_streamk_tile(
    Q, K, V, sm_scale, DO, O, M, Delta,
    DK, DV,
    DKp, DVp, locks,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    batch_idx, k_head_idx, n_block_idx,
    num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # Swizzle to improve locality (match forward grouping)
    GROUP_SIZE = 12
    pid = (pid // GROUP_SIZE) + (pid % GROUP_SIZE) * (tl.num_programs(0) // GROUP_SIZE)

    # Determine this CTA's slice over m-block iterations
    num_m_blocks = tl.cdiv(max_seqlen_q, BLOCK_M)
    tiles_per_cta = tl.cdiv(num_m_blocks, tl.num_programs(0))
    start_m_block = tiles_per_cta * pid
    end_m_block = tl.minimum(num_m_blocks, start_m_block + tiles_per_cta)
    num_participating = tl.cdiv(num_m_blocks, tiles_per_cta)
    if pid >= num_participating:
        return

    # Tile coordinates
    start_n = n_block_idx * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_kv = (offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)

    # Load K, V tile
    k_ptrs = K + batch_idx*stride_kb + k_head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
    v_ptrs = V + batch_idx*stride_vb + k_head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
    k_tile = tl.load(k_ptrs, mask=mask_kv, other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_kv, other=0.0)

    # Accumulators per CTA
    dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # GQA mapping
    q_head_base = k_head_idx * GQA_GROUP_SIZE

    # Iterate over m blocks assigned to this CTA
    for m_block_idx in range(start_m_block, end_m_block):
        start_m = m_block_idx * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < max_seqlen_q

        for g in range(GQA_GROUP_SIZE):
            q_head_idx = q_head_base + g
            Q_ptr = Q + batch_idx*stride_qb + q_head_idx*stride_qh
            DO_ptr = DO + batch_idx*stride_dob + q_head_idx*stride_doh
            M_ptr = M + batch_idx*stride_mb + q_head_idx*stride_mh
            Delta_ptr = Delta + batch_idx*stride_deltab + q_head_idx*stride_deltah

            q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
            do_ptrs = DO_ptr + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
            mask_q = mask_m[:, None] & (offs_d[None, :] < HEAD_DIM)

            q = tl.load(q_ptrs, mask=mask_q, other=0.0)
            do = tl.load(do_ptrs, mask=mask_q, other=0.0)
            m_vals = tl.load(M_ptr + offs_m * stride_mm, mask=mask_m, other=-float('inf'))

            # Transposed domain P^T
            kq = tl.dot(k_tile, tl.trans(q))
            p_T = tl.math.exp(kq * sm_scale - m_vals[None, :])
            p_T = tl.where(m_vals[None, :] == -float('inf'), 0.0, p_T)
            if CAUSAL:
                causal_mask_T = (offs_m[None, :] >= (offs_n[:, None] + max_seqlen_q - max_seqlen_k))
                p_T = tl.where(causal_mask_T, p_T, 0.0)

            # dV and dK accumulation
            dv_acc += tl.dot(p_T.to(do.type.element_ty), do)
            dp_T = tl.dot(v_tile, tl.trans(do))
            Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)
            ds_T = p_T * (dp_T - Di[None, :])
            dk_acc += tl.dot(ds_T.to(q.dtype), q)

    # Write per-CTA partials and signal
    dk_ptrs = DKp + pid * (BLOCK_N*HEAD_DIM) + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
    dv_ptrs = DVp + pid * (BLOCK_N*HEAD_DIM) + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(dk_ptrs, dk_acc, cache_modifier=".wt")
    tl.store(dv_ptrs, dv_acc, cache_modifier=".wt")
    tl.atomic_xchg(locks + pid, 1)

    # Host CTA reduces and writes final
    # Use CTA 0 as host for simplicity
    if pid == 0:
        acc_dk = dk_acc
        acc_dv = dv_acc
        # Reduce over participating CTAs only
        for cta in range(1, num_participating):
            while tl.atomic_cas(locks + cta, 1, 1) != 1:
                pass
            dk_cta_ptrs = DKp + cta * (BLOCK_N*HEAD_DIM) + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
            dv_cta_ptrs = DVp + cta * (BLOCK_N*HEAD_DIM) + offs_n[:, None] * HEAD_DIM + offs_d[None, :]
            dk_cta = tl.load(dk_cta_ptrs)
            dv_cta = tl.load(dv_cta_ptrs)
            acc_dk += dk_cta
            acc_dv += dv_cta
            tl.atomic_xchg(locks + cta, 0)

        # Vectorized half-split store along HEAD_DIM like forward
        offs_half = tl.arange(0, HEAD_DIM // 2)
        dv_out0 = DV + batch_idx*stride_dvb + k_head_idx*stride_dvh + offs_n[:,None]*stride_dvn + offs_half[None, :]*stride_dvd
        dv_out1 = DV + batch_idx*stride_dvb + k_head_idx*stride_dvh + offs_n[:,None]*stride_dvn + (offs_half[None, :]+(HEAD_DIM // 2))*stride_dvd
        dk_out0 = DK + batch_idx*stride_dkb + k_head_idx*stride_dkh + offs_n[:,None]*stride_dkn + offs_half[None, :]*stride_dkd
        dk_out1 = DK + batch_idx*stride_dkb + k_head_idx*stride_dkh + offs_n[:,None]*stride_dkn + (offs_half[None, :]+(HEAD_DIM // 2))*stride_dkd
        # reshape/permutation to avoid slicing
        dv_rs = tl.reshape(acc_dv, (BLOCK_N, 2, HEAD_DIM // 2))
        dv_pm = tl.permute(dv_rs, (0, 2, 1))
        dv0, dv1 = tl.split(dv_pm)
        dk_rs = tl.reshape(acc_dk, (BLOCK_N, 2, HEAD_DIM // 2))
        dk_pm = tl.permute(dk_rs, (0, 2, 1))
        dk0, dk1 = tl.split(dk_pm)
        # masks per row
        mask_row = (offs_n < max_seqlen_k)[:, None]
        tl.store(dv_out0, dv0.to(DV.dtype.element_ty), mask=mask_row)
        tl.store(dv_out1, dv1.to(DV.dtype.element_ty), mask=mask_row)
        tl.store(dk_out0, (dk0 * sm_scale).to(DK.dtype.element_ty), mask=mask_row)
        tl.store(dk_out1, (dk1 * sm_scale).to(DK.dtype.element_ty), mask=mask_row)


@triton.jit
def bwd_dkdv_streamk_persistent(
    Q, K, V, sm_scale, DO, O, M, Delta,
    DK, DV,
    DKp, DVp, locks,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    batch_size, num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    num_m_blocks, num_n_blocks,
    total_tiles, high_load_wgs, max_tiles_per_wg,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PARTICIPATING: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_groups = num_ctas // PARTICIPATING
    group_id = pid // PARTICIPATING
    lane = pid % PARTICIPATING

    # Schedule tiles by group_id (host CTA group per tile), not by pid
    if group_id < high_load_wgs:
        iter_start = max_tiles_per_wg * group_id
        num_tiles_to_process = max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (group_id - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        num_tiles_to_process = max_tiles_per_wg - 1

    tiles_this = num_tiles_to_process
    if iter_start + tiles_this > total_tiles:
        tiles_this = total_tiles - iter_start
    if tiles_this < 0:
        tiles_this = 0
    for i in range(tiles_this):
        tile_gid = iter_start + i
        # Map tile id -> (batch, k_head, n_block)
        tmp = tile_gid
        n_block_idx = tmp % num_n_blocks
        tmp //= num_n_blocks
        k_head_idx = tmp % num_k_heads
        batch_idx = tmp // num_k_heads

        # Split m_blocks across lanes within this group
        num_participating = tl.minimum(PARTICIPATING, num_m_blocks)
        tiles_per_lane = tl.cdiv(num_m_blocks, num_participating)
        start_m_block = tiles_per_lane * lane
        end_m_block = tl.minimum(num_m_blocks, start_m_block + tiles_per_lane)
        if lane < num_participating:

            start_n = n_block_idx * BLOCK_N
            offs_n = start_n + tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, HEAD_DIM)
            mask_kv = (offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)

            k_ptrs = K + batch_idx*stride_kb + k_head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
            v_ptrs = V + batch_idx*stride_vb + k_head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
            k_ptrs = tl.multiple_of(k_ptrs, (16, 1))
            v_ptrs = tl.multiple_of(v_ptrs, (1, 16))
            k_tile = tl.load(k_ptrs, mask=mask_kv, other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_kv, other=0.0)

            dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            q_head_base = k_head_idx * GQA_GROUP_SIZE

            for m_block_idx in range(start_m_block, end_m_block):
                start_m = m_block_idx * BLOCK_M
                offs_m = start_m + tl.arange(0, BLOCK_M)
                mask_m = offs_m < max_seqlen_q
                for g in range(GQA_GROUP_SIZE):
                    q_head_idx = q_head_base + g
                    Q_ptr = Q + batch_idx*stride_qb + q_head_idx*stride_qh
                    DO_ptr = DO + batch_idx*stride_dob + q_head_idx*stride_doh
                    M_ptr = M + batch_idx*stride_mb + q_head_idx*stride_mh
                    Delta_ptr = Delta + batch_idx*stride_deltab + q_head_idx*stride_deltah
                    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
                    do_ptrs = DO_ptr + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
                    mask_q = mask_m[:, None] & (offs_d[None, :] < HEAD_DIM)
                    q_ptrs = tl.multiple_of(q_ptrs, (1, 16))
                    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
                    do = tl.load(do_ptrs, mask=mask_q, other=0.0)
                    m_vals = tl.load(M_ptr + offs_m * stride_mm, mask=mask_m, other=-float('inf'))
                    kq = tl.dot(k_tile, tl.trans(q))
                    p_T = tl.math.exp(kq * sm_scale - m_vals[None, :])
                    p_T = tl.where(m_vals[None, :] == -float('inf'), 0.0, p_T)
                    if CAUSAL:
                        causal_mask_T = (offs_m[None, :] >= (offs_n[:, None] + max_seqlen_q - max_seqlen_k))
                        p_T = tl.where(causal_mask_T, p_T, 0.0)
                    dv_acc += tl.dot(p_T.to(do.type.element_ty), do)
                    dp_T = tl.dot(v_tile, tl.trans(do))
                    Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)
                    ds_T = p_T * (dp_T - Di[None, :])
                    dk_acc += tl.dot(ds_T.to(q.dtype), q)

            # Per-CTA partial store and signal (use local indices)
            local_n = tl.arange(0, BLOCK_N)
            local_d = tl.arange(0, HEAD_DIM)
            dk_ptrs = DKp + pid * (BLOCK_N*HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]
            dv_ptrs = DVp + pid * (BLOCK_N*HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]
            tl.store(dk_ptrs, dk_acc, cache_modifier=".wt")
            tl.store(dv_ptrs, dv_acc, cache_modifier=".wt")
            tl.atomic_xchg(locks + pid, 1)

            # Host CTA reduction for participating CTAs in this group
            if lane == 0:
                acc_dk = dk_acc
                acc_dv = dv_acc
                for l in range(1, num_participating):
                    cta = group_id * PARTICIPATING + l
                    while tl.atomic_cas(locks + cta, 1, 1) != 1:
                        pass
                    dk_cta_ptrs = DKp + cta * (BLOCK_N*HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]
                    dv_cta_ptrs = DVp + cta * (BLOCK_N*HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]
                    dk_cta = tl.load(dk_cta_ptrs)
                    dv_cta = tl.load(dv_cta_ptrs)
                    acc_dk += dk_cta
                    acc_dv += dv_cta
                    tl.atomic_xchg(locks + cta, 0)

                # Vectorized store halves
                offs_half = tl.arange(0, HEAD_DIM // 2)
                dv_out0 = DV + batch_idx*stride_dvb + k_head_idx*stride_dvh + offs_n[:,None]*stride_dvn + offs_half[None, :]*stride_dvd
                dv_out1 = DV + batch_idx*stride_dvb + k_head_idx*stride_dvh + offs_n[:,None]*stride_dvn + (offs_half[None, :]+(HEAD_DIM // 2))*stride_dvd
                dk_out0 = DK + batch_idx*stride_dkb + k_head_idx*stride_dkh + offs_n[:,None]*stride_dkn + offs_half[None, :]*stride_dkd
                dk_out1 = DK + batch_idx*stride_dkb + k_head_idx*stride_dkh + offs_n[:,None]*stride_dkn + (offs_half[None, :]+(HEAD_DIM // 2))*stride_dkd
                dv_rs = tl.reshape(acc_dv, (BLOCK_N, 2, HEAD_DIM // 2))
                dv_pm = tl.permute(dv_rs, (0, 2, 1))
                dv0, dv1 = tl.split(dv_pm)
                dk_rs = tl.reshape(acc_dk, (BLOCK_N, 2, HEAD_DIM // 2))
                dk_pm = tl.permute(dk_rs, (0, 2, 1))
                dk0, dk1 = tl.split(dk_pm)
                mask_row = (offs_n < max_seqlen_k)[:, None]
                tl.store(dv_out0, dv0.to(DV.dtype.element_ty), mask=mask_row)
                tl.store(dv_out1, dv1.to(DV.dtype.element_ty), mask=mask_row)
                tl.store(dk_out0, (dk0 * sm_scale).to(DK.dtype.element_ty), mask=mask_row)
                tl.store(dk_out1, (dk1 * sm_scale).to(DK.dtype.element_ty), mask=mask_row)
            
def bwd_dq_streamk_tile(
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ,
    DQp, locks,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    batch_idx, q_head_idx, m_block_idx,
    num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    # Partition N blocks among CTAs
    num_n_blocks = tl.cdiv(max_seqlen_k, BLOCK_N)
    tiles_per_cta = tl.cdiv(num_n_blocks, tl.num_programs(0))
    start_n_block = tiles_per_cta * pid
    end_n_block = tl.minimum(num_n_blocks, start_n_block + tiles_per_cta)

    # Tile coordinates
    start_m = m_block_idx * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_q = (offs_m[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)

    # Pointers
    k_head_idx = q_head_idx // GQA_GROUP_SIZE

    q_ptrs = Q + batch_idx*stride_qb + q_head_idx*stride_qh + offs_m[:,None]*stride_qm + offs_d[None,:]*stride_qd
    do_ptrs = DO + batch_idx*stride_dob + q_head_idx*stride_doh + offs_m[:,None]*stride_dom + offs_d[None,:]*stride_dod
    m_ptrs = M + batch_idx*stride_mb + q_head_idx*stride_mh + offs_m*stride_mm

    q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
    do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
    m_tile = tl.load(m_ptrs, mask=(offs_m < max_seqlen_q), other=-float('inf'))[:, None]
    Di = tl.load(Delta + batch_idx*stride_deltab + q_head_idx*stride_deltah + offs_m*stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Iterate assigned N blocks
    for n_block in range(start_n_block, end_n_block):
        start_n = n_block * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < max_seqlen_k
        mask_kv = mask_n[:, None] & (offs_d[None, :] < HEAD_DIM)

        k_ptrs = K + batch_idx*stride_kb + k_head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
        v_ptrs = V + batch_idx*stride_vb + k_head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
        k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

        # Compute P = exp(QK^T * sm_scale - M)
        qk = tl.dot(q_tile, tl.trans(k))
        p = tl.math.exp(qk * sm_scale - m_tile)
        p = tl.where(m_tile == -float('inf'), 0.0, p)
        if CAUSAL:
            causal_mask = (offs_m[:, None] >= (offs_n[None, :] + max_seqlen_q - max_seqlen_k))
            p = tl.where(causal_mask, p, 0.0)

        dp = tl.dot(do_tile, tl.trans(v))
        ds = p * (dp - Di[:, None])
        dq_acc += tl.dot(ds.to(k.dtype), k)

    # Write per-CTA partial and signal
    local_m = tl.arange(0, BLOCK_M)
    local_d = tl.arange(0, HEAD_DIM)
    dq_ptrs = DQp + pid * (BLOCK_M*HEAD_DIM) + local_m[:, None] * HEAD_DIM + local_d[None, :]
    tl.store(dq_ptrs, dq_acc, cache_modifier=".wt")
    tl.atomic_xchg(locks + pid, 1)

    # Host reduces and writes
    if pid == 0:
        acc_dq = dq_acc
        for cta in range(1, tl.num_programs(0)):
            while tl.atomic_cas(locks + cta, 1, 1) != 1:
                pass
            dq_cta_ptrs = DQp + cta * (BLOCK_M*HEAD_DIM) + local_m[:, None] * HEAD_DIM + local_d[None, :]
            dq_cta = tl.load(dq_cta_ptrs)
            acc_dq += dq_cta

        dq_out = DQ + batch_idx*stride_dqb + q_head_idx*stride_dqh + offs_m[:,None]*stride_dqm + offs_d[None,:]*stride_dqd
        tl.store(dq_out, (acc_dq * sm_scale).to(DQ.dtype.element_ty), mask=mask_q)


@triton.jit
def bwd_dq_streamk_persistent(
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ,
    DQp, locks,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    batch_size, num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    num_m_blocks, num_n_blocks,
    total_tiles, high_load_wgs, max_tiles_per_wg,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PARTICIPATING: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_groups = num_ctas // PARTICIPATING
    group_id = pid // PARTICIPATING
    lane = pid % PARTICIPATING

    if group_id < high_load_wgs:
        iter_start = max_tiles_per_wg * group_id
        num_tiles_to_process = max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (group_id - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        num_tiles_to_process = max_tiles_per_wg - 1

    # Slice n_blocks across lanes in the group
    num_participating = tl.minimum(PARTICIPATING, num_n_blocks)
    tiles_per_lane = tl.cdiv(num_n_blocks, num_participating)
    start_n_block = tiles_per_lane * lane
    end_n_block = tl.minimum(num_n_blocks, start_n_block + tiles_per_lane)

    tiles_this = num_tiles_to_process
    if iter_start + tiles_this > total_tiles:
        tiles_this = total_tiles - iter_start
    if tiles_this < 0:
        tiles_this = 0
    for i in range(tiles_this):
        tile_gid = iter_start + i
        # Map tile id -> (batch, q_head, m_block)
        tmp = tile_gid
        m_block_idx = tmp % num_m_blocks
        tmp //= num_m_blocks
        q_head_idx = tmp % num_q_heads
        batch_idx = tmp // num_q_heads

        start_m = m_block_idx * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_q = (offs_m[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)
        k_head_idx = q_head_idx // GQA_GROUP_SIZE

        q_ptrs = Q + batch_idx*stride_qb + q_head_idx*stride_qh + offs_m[:,None]*stride_qm + offs_d[None,:]*stride_qd
        do_ptrs = DO + batch_idx*stride_dob + q_head_idx*stride_doh + offs_m[:,None]*stride_dom + offs_d[None,:]*stride_dod
        m_ptrs = M + batch_idx*stride_mb + q_head_idx*stride_mh + offs_m*stride_mm
        q_ptrs = tl.multiple_of(q_ptrs, (1, 16))
        q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
        do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
        m_tile = tl.load(m_ptrs, mask=(offs_m < max_seqlen_q), other=-float('inf'))[:, None]
        Di = tl.load(Delta + batch_idx*stride_deltab + q_head_idx*stride_deltah + offs_m*stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)

        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        if lane < num_participating:
            for n_block in range(start_n_block, end_n_block):
                start_n = n_block * BLOCK_N
                offs_n = start_n + tl.arange(0, BLOCK_N)
                mask_n = offs_n < max_seqlen_k
                mask_kv = mask_n[:, None] & (offs_d[None, :] < HEAD_DIM)

                k_ptrs = K + batch_idx*stride_kb + k_head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
                v_ptrs = V + batch_idx*stride_vb + k_head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
                k_ptrs = tl.multiple_of(k_ptrs, (16, 1))
                v_ptrs = tl.multiple_of(v_ptrs, (1, 16))
                k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
                v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

                qk = tl.dot(q_tile, tl.trans(k))
                p = tl.math.exp(qk * sm_scale - m_tile)
                p = tl.where(m_tile == -float('inf'), 0.0, p)
                if CAUSAL:
                    causal_mask = (offs_m[:, None] >= (offs_n[None, :] + max_seqlen_q - max_seqlen_k))
                    p = tl.where(causal_mask, p, 0.0)

                dp = tl.dot(do_tile, tl.trans(v))
                ds = p * (dp - Di[:, None])
                dq_acc += tl.dot(ds.to(k.dtype), k)

        # Per-CTA partial
        local_m = tl.arange(0, BLOCK_M)
        local_d = tl.arange(0, HEAD_DIM)
        dq_ptrs = DQp + pid * (BLOCK_M*HEAD_DIM) + local_m[:, None] * HEAD_DIM + local_d[None, :]
        tl.store(dq_ptrs, dq_acc, cache_modifier=".wt")
        tl.atomic_xchg(locks + pid, 1)

        if lane == 0:
            acc_dq = dq_acc
            for l in range(1, num_participating):
                cta = group_id * PARTICIPATING + l
                while tl.atomic_cas(locks + cta, 1, 1) != 1:
                    pass
                dq_cta_ptrs = DQp + cta * (BLOCK_M*HEAD_DIM) + local_m[:, None] * HEAD_DIM + local_d[None, :]
                dq_cta = tl.load(dq_cta_ptrs)
                acc_dq += dq_cta
                tl.atomic_xchg(locks + cta, 0)

            offs_half = tl.arange(0, HEAD_DIM // 2)
            dq_out0 = DQ + batch_idx*stride_dqb + q_head_idx*stride_dqh + offs_m[:,None]*stride_dqm + offs_half[None,:]*stride_dqd
            dq_out1 = DQ + batch_idx*stride_dqb + q_head_idx*stride_dqh + offs_m[:,None]*stride_dqm + (offs_half[None,:]+(HEAD_DIM//2))*stride_dqd
            dq_rs = tl.reshape(acc_dq, (BLOCK_M, 2, HEAD_DIM // 2))
            dq_pm = tl.permute(dq_rs, (0, 2, 1))
            dq0, dq1 = tl.split(dq_pm)
            mask_row = (offs_m < max_seqlen_q)[:, None]
            tl.store(dq_out0, (dq0 * sm_scale).to(DQ.dtype.element_ty), mask=mask_row)
            tl.store(dq_out1, (dq1 * sm_scale).to(DQ.dtype.element_ty), mask=mask_row)

# ----------------------------------------------------------------------------
# Triton Kernels (_bwd_dkdv_la_inner, _bwd_dq_la_inner, 
# _bwd_la_persistent_inner, bwd_la_persistent)
#
# No changes are needed in the Triton kernel code from the previous version.
# The following Python code is the updated launcher.
# ----------------------------------------------------------------------------


@triton.jit
def _bwd_dkdv_la_inner(
    dk_acc, dv_acc, Q_ptr, k_tile, v_tile, DO_ptr, M_ptr, Delta_ptr, sm_scale,
    stride_qm, stride_qd, stride_dom, stride_dod, stride_mm, stride_deltam,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    max_seqlen_q, max_seqlen_k,
    start_n, start_m_loop, num_m_steps,
    CAUSAL: tl.constexpr,
):
    """
    Inner loop for dK/dV. Iterates over Q blocks for a given K/V block.
    """
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    curr_m = start_m_loop
    for _ in range(num_m_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < max_seqlen_q

        # Load Q and dO
        q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        do_ptrs = DO_ptr + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
        mask_q = mask_m[:, None] & (offs_d[None, :] < HEAD_DIM)

        q = tl.load(q_ptrs, mask=mask_q, other=0.0)
        do = tl.load(do_ptrs, mask=mask_q, other=0.0)

        # Load softmax stats (log-sum-exp)
        m_vals = tl.load(M_ptr + offs_m * stride_mm, mask=mask_m, other=-float('inf'))

        # Transposed-domain recompute: KQT and operate with P^T
        # kq = K_tile @ Q^T -> [BLOCK_N, BLOCK_M]
        kq = tl.dot(k_tile, tl.trans(q))
        # p_T[j,i] = exp(kq[j,i] * sm_scale - m_vals[i])
        p_T = tl.math.exp(kq * sm_scale - m_vals[None, :])
        p_T = tl.where(m_vals[None, :] == -float('inf'), 0.0, p_T)

        if CAUSAL:
            # causal mask transposed: allow only when q_pos >= k_pos
            causal_mask_T = (offs_m[None, :] >= (offs_n[:, None] + max_seqlen_q - max_seqlen_k))
            p_T = tl.where(causal_mask_T, p_T, 0.0)

        # Load Delta
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)

        # Compute dV: dV += P^T @ dO -> use transposed-domain p_T directly
        dv_acc += tl.dot(p_T.to(do.type.element_ty), do)

        # Compute dS in transposed domain, then dK
        # dp_T = (dO @ V^T)^T = V @ dO^T -> [BLOCK_N, BLOCK_M]
        dp_T = tl.dot(v_tile, tl.trans(do))
        ds_T = p_T * (dp_T - Di[None, :])
        # ds_T = ds_T * sm_scale (scaling applied at store site as before)
        dk_acc += tl.dot(ds_T.to(q.dtype), q)

        curr_m += BLOCK_M

    return dk_acc, dv_acc

@triton.jit
def _bwd_dq_la_inner(
    dq_acc, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
    stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
    max_seqlen_q, max_seqlen_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_m, start_n_loop, num_n_steps,
    CAUSAL: tl.constexpr,
):
    """
    Inner loop for dQ. Iterates over K/V blocks for a given Q block.
    """
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    curr_n = start_n_loop
    for _ in range(num_n_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < max_seqlen_k

        # Load K and V
        k_ptrs = K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        mask_kv = mask_n[:, None] & (offs_d[None, :] < HEAD_DIM)

        k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

        # Compute P = exp(QK^T * sm_scale - M)
        qk = tl.dot(q_tile, tl.trans(k))
        p = tl.math.exp(qk * sm_scale - m_tile)
        p = tl.where(m_tile == -float('inf'), 0.0, p)

        if CAUSAL:
            causal_mask = (offs_m[:, None] >= (offs_n[None, :] + max_seqlen_q - max_seqlen_k))
            p = tl.where(causal_mask, p, 0.0)

        # Compute dP = dO @ V^T
        dp = tl.dot(do_tile, tl.trans(v))

        # Load Delta
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)

        # Compute dS
        ds = p * (dp - Di[:, None])
        # ds = ds * sm_scale

        # Compute dQ
        dq_acc += tl.dot(ds.to(k.dtype), k)

        curr_n += BLOCK_N

    return dq_acc

@triton.jit
def _bwd_la_persistent_inner(
    # Pointers to matrices
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ, DK, DV,
    # Partial results buffers for host-based reduction
    DK_partials, DV_partials, locks_dkdv,
    DQ_partials, locks_dq,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    # Scheduling & Workload parameters
    work_item_id: tl.int32,
    num_dkdv_work_items: tl.int32,
    num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    # Scheduling for host reduction
    work_per_dk_block,
    work_per_dq_block,
    num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2,
    GQA_GROUP_SIZE: tl.constexpr,
    # Meta-parameters
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Processes a single work item for the backward pass.
    A work item can be either a dK/dV task or a dQ task.
    """
    if work_item_id < num_dkdv_work_items:
        # ========== COMPUTE PARTIAL dK and dV ==========
        
        # Decompose the work_item_id to get the dK/dV block and the work within that block.
        dk_block_id = work_item_id // work_per_dk_block
        work_in_dk_block_id = work_item_id % work_per_dk_block

        # Decompose dk_block_id to get batch, head, and n-block indices
        temp_id = dk_block_id
        n_block_idx = temp_id % num_n_blocks_1
        temp_id //= num_n_blocks_1
        k_head_idx = temp_id % num_k_heads
        batch_idx = temp_id // num_k_heads
        
        # Decompose work_in_dk_block_id to get m-block and q-head group indices
        temp_id = work_in_dk_block_id
        m_block_idx = temp_id % num_m_blocks_1
        q_head_group_idx = temp_id // num_m_blocks_1
        
        q_head_idx = k_head_idx * GQA_GROUP_SIZE + q_head_group_idx

        # Make the last worker in the block the host, so all workers are scheduled before it
        is_host = (work_in_dk_block_id == (work_per_dk_block - 1))

        start_m = m_block_idx * BLOCK_M1
        start_n = n_block_idx * BLOCK_N1

        # Accumulators for the partial results, in high precision
        dk_accumulator = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dv_accumulator = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        offs_d = tl.arange(0, HEAD_DIM)
        # Local indices for partial buffers (always 0..BLOCK_N1-1, 0..HEAD_DIM-1)
        local_n = tl.arange(0, BLOCK_N1)
        local_d = tl.arange(0, HEAD_DIM)
        mask_kv = (offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)
        
        k_ptrs = K + batch_idx*stride_kb + k_head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
        v_ptrs = V + batch_idx*stride_vb + k_head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
        k_tile = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_kv, other=0.0)

        Q_ptr = Q + batch_idx * stride_qb + q_head_idx * stride_qh
        DO_ptr = DO + batch_idx * stride_dob + q_head_idx * stride_doh
        M_ptr = M + batch_idx * stride_mb + q_head_idx * stride_mh
        Delta_ptr = Delta + batch_idx * stride_deltab + q_head_idx * stride_deltah

        # Compute this item's contribution to dK and dV
        dk_accumulator, dv_accumulator = _bwd_dkdv_la_inner(
            dk_accumulator, dv_accumulator, Q_ptr, k_tile, v_tile, DO_ptr, M_ptr, Delta_ptr, sm_scale,
            stride_qm, stride_qd, stride_dom, stride_dod, stride_mm, stride_deltam,
            BLOCK_M1, BLOCK_N1, HEAD_DIM, max_seqlen_q, max_seqlen_k,
            start_n, start_m, 1,
            CAUSAL=CAUSAL
        )

        # Host-based reduction logic
        if is_host:
            # Loop through all other work items that contribute to this dK/dV block
            for i in range(0, work_per_dk_block - 1):
                # Wait for the worker CTA to signal that it's done
                while tl.atomic_cas(locks_dkdv + dk_block_id * work_per_dk_block + i, 1, 1) != 1:
                    pass
                
                # Load the partial results from the worker
                worker_work_item_id = dk_block_id * work_per_dk_block + i
                dk_worker_partial_ptr = DK_partials + worker_work_item_id * (BLOCK_N1 * HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]
                dv_worker_partial_ptr = DV_partials + worker_work_item_id * (BLOCK_N1 * HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]

                dk_worker_partial = tl.load(dk_worker_partial_ptr)
                dv_worker_partial = tl.load(dv_worker_partial_ptr)

                # Accumulate the worker's results
                dk_accumulator += dk_worker_partial
                dv_accumulator += dv_worker_partial

            # After accumulating all results, write the final values to global memory
            dv_ptrs_out = DV + batch_idx*stride_dvb + k_head_idx*stride_dvh + offs_n[:,None]*stride_dvn + offs_d[None,:]*stride_dvd
            dk_ptrs_out = DK + batch_idx*stride_dkb + k_head_idx*stride_dkh + offs_n[:,None]*stride_dkn + offs_d[None,:]*stride_dkd
            tl.store(dv_ptrs_out, dv_accumulator.to(DV.dtype.element_ty), mask=mask_kv)
            tl.store(dk_ptrs_out, (dk_accumulator * sm_scale).to(DK.dtype.element_ty), mask=mask_kv)

        else:
            # If this is not the host, write partial results to temporary storage
            # and signal completion.
            dk_partial_ptrs = DK_partials + work_item_id * (BLOCK_N1 * HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]
            dv_partial_ptrs = DV_partials + work_item_id * (BLOCK_N1 * HEAD_DIM) + local_n[:, None] * HEAD_DIM + local_d[None, :]

            tl.store(dk_partial_ptrs, dk_accumulator, cache_modifier=".wt")
            tl.store(dv_partial_ptrs, dv_accumulator, cache_modifier=".wt")
            
            # Signal that this worker has completed its task
            tl.atomic_xchg(locks_dkdv + work_item_id, 1)
    
    else:
        # ========== COMPUTE PARTIAL dQ with host reduction ==========
        local_id = work_item_id - num_dkdv_work_items
        dq_block_id = local_id // work_per_dq_block
        work_in_dq_block_id = local_id % work_per_dq_block

        temp = dq_block_id
        m_block_idx = temp % num_m_blocks_2
        temp //= num_m_blocks_2
        q_head_idx = temp % num_q_heads
        batch_idx = temp // num_q_heads

        k_head_idx = q_head_idx // GQA_GROUP_SIZE
        start_m = m_block_idx * BLOCK_M2
        start_n = work_in_dq_block_id * BLOCK_N2

        dq_accumulator = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_q = (offs_m[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)
        
        q_ptrs = Q + batch_idx*stride_qb + q_head_idx*stride_qh + offs_m[:,None]*stride_qm + offs_d[None,:]*stride_qd
        do_ptrs = DO + batch_idx*stride_dob + q_head_idx*stride_doh + offs_m[:,None]*stride_dom + offs_d[None,:]*stride_dod
        m_ptrs = M + batch_idx*stride_mb + q_head_idx*stride_mh + offs_m*stride_mm
        
        q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
        do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
        m_tile = tl.load(m_ptrs, mask=(offs_m < max_seqlen_q), other=-float('inf'))[:, None]

        K_ptr = K + batch_idx * stride_kb + k_head_idx * stride_kh
        V_ptr = V + batch_idx * stride_vb + k_head_idx * stride_vh
        Delta_ptr = Delta + batch_idx * stride_deltab + q_head_idx * stride_deltah

        dq_accumulator = _bwd_dq_la_inner(
            dq_accumulator, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
            stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
            max_seqlen_q, max_seqlen_k,
            BLOCK_M2, BLOCK_N2, HEAD_DIM,
            start_m, start_n, 1,
            CAUSAL=CAUSAL
        )

        # Host is last work in the block
        is_host = (work_in_dq_block_id == (work_per_dq_block - 1))
        local_m = tl.arange(0, BLOCK_M2)
        local_d = tl.arange(0, HEAD_DIM)

        if is_host:
            for i in range(0, work_per_dq_block - 1):
                while tl.atomic_cas(locks_dq + dq_block_id * work_per_dq_block + i, 1, 1) != 1:
                    pass
                worker_local_item_id = dq_block_id * work_per_dq_block + i
                dq_worker_ptr = DQ_partials + worker_local_item_id * (BLOCK_M2 * HEAD_DIM) + local_m[:, None] * HEAD_DIM + local_d[None, :]
                dq_worker_partial = tl.load(dq_worker_ptr)
                dq_accumulator += dq_worker_partial

            dq_ptrs_out = DQ + batch_idx*stride_dqb + q_head_idx*stride_dqh + offs_m[:,None]*stride_dqm + offs_d[None,:]*stride_dqd
            tl.store(dq_ptrs_out, (dq_accumulator * sm_scale).to(DQ.dtype.element_ty), mask=mask_q)
        else:
            my_local_item_id = local_id
            dq_partial_ptr = DQ_partials + my_local_item_id * (BLOCK_M2 * HEAD_DIM) + local_m[:, None] * HEAD_DIM + local_d[None, :]
            tl.store(dq_partial_ptr, dq_accumulator, cache_modifier=".wt")
            tl.atomic_xchg(locks_dq + dq_block_id * work_per_dq_block + work_in_dq_block_id, 1)


@triton.jit
def bwd_la_persistent(
    # Pointers to matrices
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ, DK, DV,
    # Partial results buffers for host-based reduction
    DK_partials, DV_partials, locks_dkdv,
    DQ_partials, locks_dq,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    # Lean Attention Scheduling Params
    total_work_items,
    high_load_wgs,
    max_tiles_per_wg,
    # Workload split point
    num_dkdv_work_items,
    # Other parameters
    num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    # Scheduling for host reduction
    work_per_dk_block,
    work_per_dq_block,
    num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2,
    GQA_GROUP_SIZE: tl.constexpr,
    # Meta-parameters
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Determine the range of work items for this program
    if pid < high_load_wgs:
        iter_start = max_tiles_per_wg * pid
        num_tiles_to_process = max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        num_tiles_to_process = max_tiles_per_wg - 1

    # Loop and process each assigned work item
    for i in range(num_tiles_to_process):
        work_item_id = iter_start + i
        if work_item_id < total_work_items:
            # Call the inner function to process one tile
            _bwd_la_persistent_inner(
                Q, K, V, sm_scale, DO, O, M, Delta,
                DQ, DK, DV,
                DK_partials, DV_partials, locks_dkdv,
                DQ_partials, locks_dq,
                stride_qb, stride_qh, stride_qm, stride_qd,
                stride_kb, stride_kh, stride_kn, stride_kd,
                stride_vb, stride_vh, stride_vn, stride_vd,
                stride_dob, stride_doh, stride_dom, stride_dod,
                stride_dqb, stride_dqh, stride_dqm, stride_dqd,
                stride_dkb, stride_dkh, stride_dkn, stride_dkd,
                stride_dvb, stride_dvh, stride_dvn, stride_dvd,
                stride_mb, stride_mh, stride_mm,
                stride_deltab, stride_deltah, stride_deltam,
                work_item_id,
                num_dkdv_work_items,
                num_q_heads, num_k_heads,
                max_seqlen_q, max_seqlen_k,
                work_per_dk_block,
                work_per_dq_block,
                num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2,
                GQA_GROUP_SIZE,
                BLOCK_M1, BLOCK_N1,
                BLOCK_M2, BLOCK_N2,
                HEAD_DIM,
                CAUSAL,
            )

# ----------------------------------------------------------------------------
# REVISED: Scheduling function adapted from the forward pass reference
# ----------------------------------------------------------------------------
def get_bwd_scheduling_params(
    batch_size, num_q_heads, num_k_heads, max_seqlen_q, max_seqlen_k,
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_SMs, gqa_group_size
):
    """
    Calculates scheduling parameters for the backward pass, adapted from the
    forward pass reference logic.
    """
    # Calculate the number of blocks for each workload
    num_m_blocks_1 = triton.cdiv(max_seqlen_q, BLOCK_M1)
    num_n_blocks_1 = triton.cdiv(max_seqlen_k, BLOCK_N1)
    num_m_blocks_2 = triton.cdiv(max_seqlen_q, BLOCK_M2)
    num_n_blocks_2 = triton.cdiv(max_seqlen_k, BLOCK_N2)

    # Define the workload for each part of the backward pass
    dkdv_work_items = batch_size * num_k_heads * num_n_blocks_1
    work_per_dk_block = num_m_blocks_1 * gqa_group_size
    # For dQ, each (batch, q_head, m_block) reduces over all n_blocks
    dq_work_items = batch_size * num_q_heads * num_m_blocks_2
    work_per_dq_block = num_n_blocks_2
    
    total_dkdv_work_items = dkdv_work_items * work_per_dk_block
    total_dq_work_items = dq_work_items * work_per_dq_block
    
    # Combine workloads into a single pool of tiles
    total_tiles = total_dkdv_work_items + total_dq_work_items

    if total_tiles == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Determine grid size, matching the reference's use of num_SMs
    num_wgs = num_SMs
    if total_tiles < num_wgs:
        num_wgs = total_tiles
    
    # Max number of tiles per work-group (task block/CTA)
    max_tiles_per_wg = triton.cdiv(total_tiles, num_wgs)

    # Number of work-groups that will have the max number of tiles
    high_load_wgs = total_tiles % num_wgs
    if high_load_wgs == 0 and total_tiles > 0:
        high_load_wgs = num_wgs

    return (
        num_wgs,
        total_tiles,
        max_tiles_per_wg,
        high_load_wgs,
        total_dkdv_work_items, # This is the split point, `num_dkdv_work_items`
        work_per_dk_block,
        work_per_dq_block,
        num_m_blocks_1,
        num_m_blocks_2,
        num_n_blocks_1,
        num_n_blocks_2,
    )

# ----------------------------------------------------------------------------
# REVISED: Python Launcher using the new scheduling function
# ----------------------------------------------------------------------------

def la_backward_persistent(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    sm_scale: float,
    causal: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_sm: int = 108, # Example SM count, should be detected from hardware
):
    """
    Backward pass launcher using a persistent, work-centric kernel with atomic reductions.
    """
    # Config and shapes (use defaults to avoid external JSON dependency)
    sm_count = arch_info.get_num_sms()
    BLOCK_M = 128
    BLOCK_N = 64
    MASKED_BLOCKS = BLOCK_M // BLOCK_N

    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, seqlen_k, num_k_heads, _ = k.shape
    assert num_q_heads == num_k_heads, "GQA/MQA not yet supported in this path"

    # Layout conversions
    q_bhsd = q.transpose(1, 2).contiguous()     # [B, H, S_q, D]
    k_bhsd = k.transpose(1, 2).contiguous()     # [B, H, S_k, D]
    v_bhsd = v.transpose(1, 2).contiguous()
    do_bhsd = do.transpose(1, 2).contiguous()
    m_bhs = softmax_lse.transpose(1, 2).contiguous()  # [B, H, S_q]
    delta_bhs = torch.sum((o.transpose(1, 2).contiguous() * do_bhsd), dim=-1)

    # Flatten batch into the sequence dimension for K/V like forward
    Q_stream = q_bhsd.permute(0, 2, 1, 3).reshape(batch * seqlen_q, num_q_heads, head_sz)
    K_stream = k_bhsd.permute(0, 2, 1, 3).reshape(batch * seqlen_k, num_k_heads, head_sz)
    V_stream = v_bhsd.permute(0, 2, 1, 3).reshape(batch * seqlen_k, num_k_heads, head_sz)

    # Scheduling params (local, safe; avoid forward helper dependency)
    num_m_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks_per_batch = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    num_n_blocks = num_n_blocks_per_batch
    # tiles_per_head: number of lean tiles an output tile needs
    if causal:
        tiles_per_head = 0
        for i in range(num_m_blocks):
            # each q-block i covers (i+1)*BLOCK_M k-columns under causal mask
            tiles_per_head += (( (i + 1) * BLOCK_M ) + BLOCK_N - 1) // BLOCK_N
        tiles_per_head *= batch
    else:
        tiles_per_head = num_m_blocks * num_n_blocks_per_batch
    total_programs = max(1, min(sm_count, tiles_per_head * num_q_heads))
    # Initialize; will be recomputed safely below
    high_load_wgs = 0
    max_tiles_per_wg = 1
    num_splits = 1

    # Guard against small workloads: cap programs to total tiles
    total_tiles = tiles_per_head * num_q_heads
    if total_tiles <= 0:
        dq.zero_(); dk.zero_(); dv.zero_();
        return
    total_programs = max(1, min(total_programs, total_tiles))
    # Recompute per-WG distribution to avoid division-by-zero cases
    max_tiles_per_wg = (total_tiles + total_programs - 1) // total_programs
    high_load_wgs = total_tiles % total_programs
    if high_load_wgs == 0 and total_tiles > 0:
        high_load_wgs = total_programs
    # Safe num_splits (>=1)
    num_splits = max(1, 1 + ((num_n_blocks + max_tiles_per_wg - 2) // max(1, max_tiles_per_wg)))

    grid = (total_programs, 1, 1)
    qk_scale = sm_scale * LOG_TWO_E

    # Allocate dQ partials and locks; DK/DV updated with atomics
    DQp = torch.empty((total_programs, BLOCK_M, head_sz), dtype=torch.float32, device=q.device)
    locks = torch.zeros((total_programs,), dtype=torch.int32, device=q.device)

    # Prepare output buffers in [B, H, S, D]
    dq_bhsd = torch.empty_like(q_bhsd)
    dk_bhsd = torch.zeros_like(k_bhsd)
    dv_bhsd = torch.zeros_like(v_bhsd)

    la_bwd_persistent[grid](
        Q_stream, K_stream, V_stream,
        do_bhsd, m_bhs, delta_bhs,
        dq_bhsd, dk_bhsd, dv_bhsd,
        DQp, locks, qk_scale,
        Q_stream.stride(0), Q_stream.stride(1), Q_stream.stride(2),
        K_stream.stride(0), K_stream.stride(1), K_stream.stride(2),
        V_stream.stride(0), V_stream.stride(1), V_stream.stride(2),
        do_bhsd.stride(0), do_bhsd.stride(1), do_bhsd.stride(2), do_bhsd.stride(3),
        m_bhs.stride(0), m_bhs.stride(1), m_bhs.stride(2),
        delta_bhs.stride(0), delta_bhs.stride(1), delta_bhs.stride(2),
        dq_bhsd.stride(0), dq_bhsd.stride(1), dq_bhsd.stride(2), dq_bhsd.stride(3),
        dk_bhsd.stride(0), dk_bhsd.stride(1), dk_bhsd.stride(2), dk_bhsd.stride(3),
        dv_bhsd.stride(0), dv_bhsd.stride(1), dv_bhsd.stride(2), dv_bhsd.stride(3),
        HEAD_DIM=head_sz,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MASKED_BLOCKS=MASKED_BLOCKS,
        batch_size=batch,
        causal=causal,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
    )

    # Transpose results back to [B, S, H, D]
    dq.copy_(dq_bhsd.transpose(1, 2))
    dk.copy_(dk_bhsd.transpose(1, 2))
    dv.copy_(dv_bhsd.transpose(1, 2))