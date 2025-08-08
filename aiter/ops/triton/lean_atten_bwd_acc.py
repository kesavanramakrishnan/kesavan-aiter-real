import math
import torch
import triton
import triton.language as tl
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
    config = {
        "BLOCK_M1": 32, "BLOCK_N1": 64,
        "BLOCK_M2": 64, "BLOCK_N2": 32,
    }

    batch, _, num_q_heads, head_sz = q.shape
    _, _, num_k_heads, _ = k.shape
    
    gqa_group_size = num_q_heads // num_k_heads
    
    q_bhsd = q.transpose(1, 2).contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()
    o_bhsd = o.transpose(1, 2).contiguous()
    do_bhsd = do.transpose(1, 2).contiguous()
    
    dq_bhsd = torch.zeros_like(q_bhsd)
    dk_bhsd = torch.zeros_like(k_bhsd)
    dv_bhsd = torch.zeros_like(v_bhsd)

    delta = torch.sum(o_bhsd * do_bhsd, dim=-1, keepdim=False)

    # Use a fixed number of work-groups based on SM count, per the reference
    num_wgs = num_sm

    # Persistent dK/dV (Stream-K)
    BLOCK_M1 = config["BLOCK_M1"]; BLOCK_N1 = config["BLOCK_N1"]
    num_m_blocks_1 = (max_seqlen_q + BLOCK_M1 - 1) // BLOCK_M1
    num_n_blocks_1 = (max_seqlen_k + BLOCK_N1 - 1) // BLOCK_N1
    total_tiles_dkdv = batch * num_k_heads * num_n_blocks_1
    # Multi-CTA grid; group PARTICIPATING CTAs per tile like forward
    PARTICIPATING = 4
    total_programs = num_wgs * PARTICIPATING
    grid_dkdv = (total_programs,)
    max_tiles_per_wg_dkdv = triton.cdiv(total_tiles_dkdv, num_wgs)
    high_load_wgs_dkdv = total_tiles_dkdv % num_wgs
    if high_load_wgs_dkdv == 0 and total_tiles_dkdv > 0:
        high_load_wgs_dkdv = num_wgs

    # Per-CTA scratch sized by total programs
    DKp = torch.empty((total_programs, BLOCK_N1, head_sz), dtype=torch.float32, device=q.device)
    DVp = torch.empty((total_programs, BLOCK_N1, head_sz), dtype=torch.float32, device=q.device)
    locks = torch.zeros((total_programs,), dtype=torch.int32, device=q.device)

    bwd_dkdv_streamk_persistent[grid_dkdv](
        q_bhsd, k_bhsd, v_bhsd, sm_scale, do_bhsd, o_bhsd, softmax_lse, delta,
        dk_bhsd, dv_bhsd,
        DKp, DVp, locks,
        q_bhsd.stride(0), q_bhsd.stride(1), q_bhsd.stride(2), q_bhsd.stride(3),
        k_bhsd.stride(0), k_bhsd.stride(1), k_bhsd.stride(2), k_bhsd.stride(3),
        v_bhsd.stride(0), v_bhsd.stride(1), v_bhsd.stride(2), v_bhsd.stride(3),
        do_bhsd.stride(0), do_bhsd.stride(1), do_bhsd.stride(2), do_bhsd.stride(3),
        dk_bhsd.stride(0), dk_bhsd.stride(1), dk_bhsd.stride(2), dk_bhsd.stride(3),
        dv_bhsd.stride(0), dv_bhsd.stride(1), dv_bhsd.stride(2), dv_bhsd.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        batch, num_q_heads, num_k_heads,
        max_seqlen_q, max_seqlen_k,
        num_m_blocks_1, num_n_blocks_1,
        total_tiles_dkdv, high_load_wgs_dkdv, max_tiles_per_wg_dkdv,
        gqa_group_size,
        HEAD_DIM=head_sz,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, PARTICIPATING=PARTICIPATING,
    )

    # Persistent dQ (Stream-K)
    BLOCK_M2 = config["BLOCK_M2"]; BLOCK_N2 = config["BLOCK_N2"]
    num_m_blocks_2 = (max_seqlen_q + BLOCK_M2 - 1) // BLOCK_M2
    num_n_blocks_2 = (max_seqlen_k + BLOCK_N2 - 1) // BLOCK_N2
    total_tiles_dq = batch * num_q_heads * num_m_blocks_2
    grid_dq = (total_programs,)
    max_tiles_per_wg_dq = triton.cdiv(total_tiles_dq, num_wgs)
    high_load_wgs_dq = total_tiles_dq % num_wgs
    if high_load_wgs_dq == 0 and total_tiles_dq > 0:
        high_load_wgs_dq = num_wgs

    DQp = torch.empty((total_programs, BLOCK_M2, head_sz), dtype=torch.float32, device=q.device)
    locks_q = torch.zeros((total_programs,), dtype=torch.int32, device=q.device)

    bwd_dq_streamk_persistent[grid_dq](
        q_bhsd, k_bhsd, v_bhsd, sm_scale, do_bhsd, o_bhsd, softmax_lse, delta,
        dq_bhsd,
        DQp, locks_q,
        q_bhsd.stride(0), q_bhsd.stride(1), q_bhsd.stride(2), q_bhsd.stride(3),
        k_bhsd.stride(0), k_bhsd.stride(1), k_bhsd.stride(2), k_bhsd.stride(3),
        v_bhsd.stride(0), v_bhsd.stride(1), v_bhsd.stride(2), v_bhsd.stride(3),
        do_bhsd.stride(0), do_bhsd.stride(1), do_bhsd.stride(2), do_bhsd.stride(3),
        dq_bhsd.stride(0), dq_bhsd.stride(1), dq_bhsd.stride(2), dq_bhsd.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        batch, num_q_heads, num_k_heads,
        max_seqlen_q, max_seqlen_k,
        num_m_blocks_2, num_n_blocks_2,
        total_tiles_dq, high_load_wgs_dq, max_tiles_per_wg_dq,
        gqa_group_size,
        HEAD_DIM=head_sz,
        CAUSAL=causal,
        BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, PARTICIPATING=PARTICIPATING,
    )

    dq.copy_(dq_bhsd.transpose(1, 2))
    dk.copy_(dk_bhsd.transpose(1, 2))
    dv.copy_(dv_bhsd.transpose(1, 2))