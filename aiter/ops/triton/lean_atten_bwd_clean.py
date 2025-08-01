import math
import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------
# Inner compute kernels for the backward pass (No changes needed here)
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

    k_tile_T = tl.trans(k_tile)

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
        m_vals = tl.load(M_ptr + offs_m * stride_mm, mask=mask_m, other=0.0)

        # Recompute P = exp(QK^T * sm_scale - M)
        qk = tl.dot(q, k_tile_T)
        p = tl.math.exp(qk * sm_scale - m_vals[:, None])

        if CAUSAL:
            causal_mask = (offs_m[:, None] >= (offs_n[None, :] + max_seqlen_q - max_seqlen_k))
            p = tl.where(causal_mask, p, 0.0)

        # Load Delta
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)

        # Compute dV
        dv_acc += tl.dot(tl.trans(p).to(do.type.element_ty), do)

        # Compute dS, then dK
        dp = tl.dot(do, tl.trans(v_tile))
        ds = p * (dp - Di[:, None])
        ds = ds * sm_scale
        dk_acc += tl.dot(tl.trans(ds).to(q.type.element_ty), q)

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

        if CAUSAL:
            causal_mask = (offs_m[:, None] >= (offs_n[None, :] + max_seqlen_q - max_seqlen_k))
            p = tl.where(causal_mask, p, 0.0)

        # Compute dP = dO @ V^T
        dp = tl.dot(do_tile, tl.trans(v))

        # Load Delta
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)

        # Compute dS
        ds = p * (dp - Di[:, None])
        ds = ds * sm_scale

        # Compute dQ
        dq_acc += tl.dot(ds.to(k.dtype), k)

        curr_n += BLOCK_N

    return dq_acc

# ----------------------------------------------------------------------------
# Main Persistent Backward Kernel with GQA and Cross-Attention Support
# ----------------------------------------------------------------------------

@triton.jit
def bwd_la_persistent(
    # Pointers to matrices
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ, DK, DV,
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
    num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2,
    GQA_GROUP_SIZE: tl.constexpr,
    # Meta-parameters
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < high_load_wgs:
        iter_start = max_tiles_per_wg * pid
        num_tiles_to_process = max_tiles_per_wg
    else:
        iter_start = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        num_tiles_to_process = max_tiles_per_wg - 1

    for i in range(num_tiles_to_process):
        work_item_id = iter_start + i
        
        if work_item_id < total_work_items:
            if work_item_id < num_dkdv_work_items:
                # ========== COMPUTE PARTIAL dK and dV ==========
                item_id = work_item_id
                m_block_idx = item_id % num_m_blocks_1
                item_id = item_id // num_m_blocks_1
                n_block_idx = item_id % num_n_blocks_1
                item_id = item_id // num_n_blocks_1
                # This is the Q head index
                q_head_idx = item_id % num_q_heads
                batch_idx = item_id // num_q_heads

                # GQA: Map Q head to K/V head
                k_head_idx = q_head_idx // GQA_GROUP_SIZE

                start_m = m_block_idx * BLOCK_M1
                start_n = n_block_idx * BLOCK_N1

                dk_partial = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
                dv_partial = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

                offs_n = start_n + tl.arange(0, BLOCK_N1)
                offs_d = tl.arange(0, HEAD_DIM)
                mask_kv = (offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)
                # Use k_head_idx for K and V pointers
                k_ptrs = K + batch_idx*stride_kb + k_head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
                v_ptrs = V + batch_idx*stride_vb + k_head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
                k_tile = tl.load(k_ptrs, mask=mask_kv, other=0.0)
                v_tile = tl.load(v_ptrs, mask=mask_kv, other=0.0)

                # Use q_head_idx for Q, dO, M, Delta pointers
                Q_ptr = Q + batch_idx * stride_qb + q_head_idx * stride_qh
                DO_ptr = DO + batch_idx * stride_dob + q_head_idx * stride_doh
                M_ptr = M + batch_idx * stride_mb + q_head_idx * stride_mh
                Delta_ptr = Delta + batch_idx * stride_deltab + q_head_idx * stride_deltah

                dk_partial, dv_partial = _bwd_dkdv_la_inner(
                    dk_partial, dv_partial, Q_ptr, k_tile, v_tile, DO_ptr, M_ptr, Delta_ptr, sm_scale,
                    stride_qm, stride_qd, stride_dom, stride_dod, stride_mm, stride_deltam,
                    BLOCK_M1, BLOCK_N1, HEAD_DIM, max_seqlen_q, max_seqlen_k,
                    start_n, start_m, 1,
                    CAUSAL=CAUSAL
                )

                # Use k_head_idx for atomic add pointers
                dv_ptrs_out = DV + batch_idx*stride_dvb + k_head_idx*stride_dvh + offs_n[:,None]*stride_dvn + offs_d[None,:]*stride_dvd
                dk_ptrs_out = DK + batch_idx*stride_dkb + k_head_idx*stride_dkh + offs_n[:,None]*stride_dkn + offs_d[None,:]*stride_dkd
                tl.atomic_add(dv_ptrs_out, dv_partial, mask=mask_kv)
                tl.atomic_add(dk_ptrs_out, dk_partial, mask=mask_kv)

            else:
                # ========== COMPUTE PARTIAL dQ ==========
                item_id = work_item_id - num_dkdv_work_items
                n_block_idx = item_id % num_n_blocks_2
                item_id = item_id // num_n_blocks_2
                m_block_idx = item_id % num_m_blocks_2
                item_id = item_id // num_m_blocks_2
                q_head_idx = item_id % num_q_heads
                batch_idx = item_id // num_q_heads

                # GQA: Map Q head to K/V head
                k_head_idx = q_head_idx // GQA_GROUP_SIZE

                start_m = m_block_idx * BLOCK_M2
                start_n = n_block_idx * BLOCK_N2

                dq_partial = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

                offs_m = start_m + tl.arange(0, BLOCK_M2)
                offs_d = tl.arange(0, HEAD_DIM)
                mask_q = (offs_m[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)
                q_ptrs = Q + batch_idx*stride_qb + q_head_idx*stride_qh + offs_m[:,None]*stride_qm + offs_d[None,:]*stride_qd
                do_ptrs = DO + batch_idx*stride_dob + q_head_idx*stride_doh + offs_m[:,None]*stride_dom + offs_d[None,:]*stride_dod
                m_ptrs = M + batch_idx*stride_mb + q_head_idx*stride_mh + offs_m*stride_mm
                q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
                do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
                m_tile = tl.load(m_ptrs, mask=(offs_m < max_seqlen_q), other=-float('inf'))[:, None]

                # Use k_head_idx for K, V pointers
                K_ptr = K + batch_idx * stride_kb + k_head_idx * stride_kh
                V_ptr = V + batch_idx * stride_vb + k_head_idx * stride_vh
                # Use q_head_idx for Delta pointer
                Delta_ptr = Delta + batch_idx * stride_deltab + q_head_idx * stride_deltah

                dq_partial = _bwd_dq_la_inner(
                    dq_partial, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
                    stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
                    max_seqlen_q, max_seqlen_k,
                    BLOCK_M2, BLOCK_N2, HEAD_DIM,
                    start_m, start_n, 1,
                    CAUSAL=CAUSAL
                )

                # Use q_head_idx for atomic add pointer
                dq_ptrs_out = DQ + batch_idx*stride_dqb + q_head_idx*stride_dqh + offs_m[:,None]*stride_dqm + offs_d[None,:]*stride_dqd
                tl.atomic_add(dq_ptrs_out, dq_partial, mask=mask_q)

# ----------------------------------------------------------------------------
# Python Launcher for the Persistent Backward Pass
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
    
    # GQA Support
    gqa_group_size = num_q_heads // num_k_heads
    
    q_bhsd = q.transpose(1, 2).contiguous()
    k_bhsd = k.transpose(1, 2).contiguous()
    v_bhsd = v.transpose(1, 2).contiguous()
    o_bhsd = o.transpose(1, 2).contiguous()
    do_bhsd = do.transpose(1, 2).contiguous()
    
    # Create output tensors in the layout expected by the kernel
    dq_bhsd = torch.empty_like(q_bhsd)
    dk_bhsd = torch.empty_like(k_bhsd)
    dv_bhsd = torch.empty_like(v_bhsd)
    
    dq_bhsd.zero_()
    dk_bhsd.zero_()
    dv_bhsd.zero_()

    delta = torch.sum(o_bhsd * do_bhsd, dim=-1, keepdim=False)

    num_m_blocks_1 = triton.cdiv(max_seqlen_q, config["BLOCK_M1"])
    num_n_blocks_1 = triton.cdiv(max_seqlen_k, config["BLOCK_N1"])
    num_m_blocks_2 = triton.cdiv(max_seqlen_q, config["BLOCK_M2"])
    num_n_blocks_2 = triton.cdiv(max_seqlen_k, config["BLOCK_N2"])

    # Define the total workload.
    num_dkdv_work_items = batch * num_q_heads * num_n_blocks_1 * num_m_blocks_1
    num_dq_work_items = batch * num_q_heads * num_m_blocks_2 * num_n_blocks_2
    total_work_items = num_dkdv_work_items + num_dq_work_items

    num_wgs = 108 * 4 
    if total_work_items < num_wgs:
        num_wgs = total_work_items if total_work_items > 0 else 1

    max_tiles_per_wg = triton.cdiv(total_work_items, num_wgs)
    high_load_wgs = total_work_items % num_wgs
    if high_load_wgs == 0 and total_work_items > 0:
        high_load_wgs = num_wgs

    grid = (num_wgs,)

    bwd_la_persistent[grid](
        q_bhsd, k_bhsd, v_bhsd, sm_scale, do_bhsd, o_bhsd, softmax_lse, delta,
        dq_bhsd, dk_bhsd, dv_bhsd,
        q_bhsd.stride(0), q_bhsd.stride(1), q_bhsd.stride(2), q_bhsd.stride(3),
        k_bhsd.stride(0), k_bhsd.stride(1), k_bhsd.stride(2), k_bhsd.stride(3),
        v_bhsd.stride(0), v_bhsd.stride(1), v_bhsd.stride(2), v_bhsd.stride(3),
        do_bhsd.stride(0), do_bhsd.stride(1), do_bhsd.stride(2), do_bhsd.stride(3),
        dq_bhsd.stride(0), dq_bhsd.stride(1), dq_bhsd.stride(2), dq_bhsd.stride(3),
        dk_bhsd.stride(0), dk_bhsd.stride(1), dk_bhsd.stride(2), dk_bhsd.stride(3),
        dv_bhsd.stride(0), dv_bhsd.stride(1), dv_bhsd.stride(2), dv_bhsd.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        total_work_items,
        high_load_wgs,
        max_tiles_per_wg,
        num_dkdv_work_items,
        num_q_heads, num_k_heads,
        max_seqlen_q, max_seqlen_k,
        num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2,
        gqa_group_size,
        HEAD_DIM=head_sz,
        CAUSAL=causal,
        **config,
    )

    dq.copy_(dq_bhsd.transpose(1, 2))
    dk.copy_(dk_bhsd.transpose(1, 2))
    dv.copy_(dv_bhsd.transpose(1, 2))
