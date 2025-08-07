import math
import torch
import triton
import triton.language as tl


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

@triton.jit
def _bwd_la_persistent_inner(
    # Pointers to matrices
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ, DK, DV,
    # Strides
    DK_p, DV_p, Locks,
    # Strides for new buffers
    stride_dkp_item, stride_dkp_n, stride_dkp_d, # Assuming DK_p is contiguous
    stride_dvp_item, stride_dvp_n, stride_dvp_d, # Assuming DV_p is contiguous
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam,
    stride_lockb, stride_lockh, stride_lockn,
    # Scheduling & Workload parameters
    work_item_id: tl.int32,
    num_dkdv_work_items: tl.int32,
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
    """
    Processes a single work item for the backward pass.
    A work item can be either a dK/dV task or a dQ task.
    """
    if work_item_id < num_dkdv_work_items:
        # ========== COMPUTE PARTIAL dK and dV ==========
        item_id = work_item_id
        m_block_idx = item_id % num_m_blocks_1
        item_id = item_id // num_m_blocks_1
        n_block_idx = item_id % num_n_blocks_1
        item_id = item_id // num_n_blocks_1
        q_head_idx = item_id % num_q_heads
        batch_idx = item_id // num_q_heads

        k_head_idx = q_head_idx // GQA_GROUP_SIZE
        start_m = m_block_idx * BLOCK_M1
        start_n = n_block_idx * BLOCK_N1

        dk_partial = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dv_partial = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        offs_n = tl.arange(0, BLOCK_N1)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_kv = (start_n + offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)
        
        dkp_ptr = DK_p + work_item_id * stride_dkp_item + offs_n[:, None] * stride_dkp_n + offs_d[None, :]
        dvp_ptr = DV_p + work_item_id * stride_dvp_item + offs_n[:, None] * stride_dvp_n + offs_d[None, :]
        tl.store(dkp_ptr, dk_partial, mask=mask_kv)
        tl.store(dvp_ptr, dv_partial, mask=mask_kv)
        
        # Atomically increment the counter for this output tile
        lock_ptr = Locks + batch_idx * stride_lockb + k_head_idx * stride_lockh + n_block_idx * stride_lockn
        tl.atomic_add(lock_ptr, 1)

        # 4. HOST-SPECIFIC LOGIC: WAIT AND REDUCE
        is_host = (m_block_idx == 0) and ((q_head_idx % GQA_GROUP_SIZE) == 0)
        if is_host:
            # Wait until all contributors for this output tile have finished
            num_contributors = num_m_blocks_1 * GQA_GROUP_SIZE
            while tl.load(lock_ptr) < num_contributors:
                pass  # Spin-wait

            # Now, perform the reduction
            dk_final = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
            dv_final = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

            for m_contrib_idx in range(num_m_blocks_1):
                for q_head_contrib_offset in range(GQA_GROUP_SIZE):
                    # Reconstruct the work_item_id for each contributor
                    q_head_contrib_idx = k_head_idx * GQA_GROUP_SIZE + q_head_contrib_offset
                    contrib_item_id = ((((batch_idx * num_q_heads + q_head_contrib_idx) \
                                       * num_n_blocks_1) + n_block_idx) \
                                       * num_m_blocks_1) + m_contrib_idx

                    # Load the partial result
                    dkp_contrib_ptr = DK_p + contrib_item_id * stride_dkp_item + offs_n[:, None] * stride_dkp_n + offs_d[None, :]
                    dvp_contrib_ptr = DV_p + contrib_item_id * stride_dvp_item + offs_n[:, None] * stride_dvp_n + offs_d[None, :]
                    
                    dk_p_contrib = tl.load(dkp_contrib_ptr, mask=mask_kv, other=0.0)
                    dv_p_contrib = tl.load(dvp_contrib_ptr, mask=mask_kv, other=0.0)

                    dk_final += dk_p_contrib
                    dv_final += dv_p_contrib

            # 5. HOST WRITES FINAL RESULT
            start_n = n_block_idx * BLOCK_N1
            dv_ptrs_out = DV + batch_idx*stride_dvb + k_head_idx*stride_dvh + (start_n + offs_n[:,None])*stride_dvn + offs_d[None,:]*stride_dvd
            dk_ptrs_out = DK + batch_idx*stride_dkb + k_head_idx*stride_dkh + (start_n + offs_n[:,None])*stride_dkn + offs_d[None,:]*stride_dkd
            tl.store(dk_ptrs_out, dk_final, mask=mask_kv)
            tl.store(dv_ptrs_out, dv_final, mask=mask_kv)
    else:
        # ========== COMPUTE PARTIAL dQ ==========
        item_id = work_item_id - num_dkdv_work_items
        m_block_idx = item_id % num_m_blocks_2
        item_id = item_id // num_m_blocks_2
        n_block_idx = item_id % num_n_blocks_2
        item_id = item_id // num_n_blocks_2
        q_head_idx = item_id % num_q_heads
        batch_idx = item_id // num_q_heads

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

        K_ptr = K + batch_idx * stride_kb + k_head_idx * stride_kh
        V_ptr = V + batch_idx * stride_vb + k_head_idx * stride_vh
        Delta_ptr = Delta + batch_idx * stride_deltab + q_head_idx * stride_deltah

        dq_partial = _bwd_dq_la_inner(
            dq_partial, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
            stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
            max_seqlen_q, max_seqlen_k,
            BLOCK_M2, BLOCK_N2, HEAD_DIM,
            start_m, start_n, 1,
            CAUSAL=CAUSAL
        )

        dq_ptrs_out = DQ + batch_idx*stride_dqb + q_head_idx*stride_dqh + offs_m[:,None]*stride_dqm + offs_d[None,:]*stride_dqd
        tl.atomic_add(dq_ptrs_out, dq_partial, mask=mask_q)


@triton.jit
def bwd_la_persistent(
    # Pointers to matrices
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ, DK, DV,
    # Partials
    DK_p, DV_p, Locks,
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
    stride_lockb, stride_lockh, stride_lockn,
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
                DK_p, DV_p, Locks,
                stride_qb, stride_qh, stride_qm, stride_qd,
                stride_kb, stride_kh, stride_kn, stride_kd,
                stride_vb, stride_vh, stride_vn, stride_vd,
                stride_dob, stride_doh, stride_dom, stride_dod,
                stride_dqb, stride_dqh, stride_dqm, stride_dqd,
                stride_dkb, stride_dkh, stride_dkn, stride_dkd,
                stride_dvb, stride_dvh, stride_dvn, stride_dvd,
                stride_mb, stride_mh, stride_mm,
                stride_deltab, stride_deltah, stride_deltam,
                stride_lockb, stride_lockh, stride_lockn,
                work_item_id,
                num_dkdv_work_items,
                num_q_heads, num_k_heads,
                max_seqlen_q, max_seqlen_k,
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
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2, num_SMs
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
    # Note: Unlike the fwd pass, the bwd pass workload is always a full grid.
    # The `causal` logic from the fwd pass scheduling is not applicable here.
    dkdv_tiles = batch_size * num_q_heads * num_m_blocks_1 * num_n_blocks_1
    dq_tiles = batch_size * num_q_heads * num_m_blocks_2 * num_n_blocks_2
    
    # Combine workloads into a single pool of tiles
    total_tiles = dkdv_tiles + dq_tiles

    if total_tiles == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

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
        dkdv_tiles, # This is the split point, `num_dkdv_work_items`
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
    num_wgs = num_sm * 4

    # REVISED: Calculate scheduling params using the adapted function
    (num_wgs, total_work_items, max_tiles_per_wg, high_load_wgs, num_dkdv_work_items,
     num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2
    ) = get_bwd_scheduling_params(
        batch, num_q_heads, num_k_heads, max_seqlen_q, max_seqlen_k,
        config["BLOCK_M1"], config["BLOCK_N1"], config["BLOCK_M2"], config["BLOCK_N2"], num_wgs
    )
    
    dk_partials = torch.empty(
        (num_dkdv_work_items, config["BLOCK_N1"], head_sz), dtype=torch.float32, device=q.device
    )
    dv_partials = torch.empty_like(dk_partials)

    # One lock/counter for each OUTPUT dK/dV tile
    locks = torch.zeros((batch, num_k_heads, num_n_blocks_1), dtype=torch.int32, device=q.device)
    
    if total_work_items == 0:
        dq.copy_(dq_bhsd.transpose(1, 2))
        dk.copy_(dk_bhsd.transpose(1, 2))
        dv.copy_(dv_bhsd.transpose(1, 2))
        return

    grid = (num_wgs,)
    
    bwd_la_persistent[grid](
        q_bhsd, k_bhsd, v_bhsd, sm_scale, do_bhsd, o_bhsd, softmax_lse, delta,
        dq_bhsd, dk_bhsd, dv_bhsd,
        dk_partials, dv_partials, locks,  # Pass new buffers
        q_bhsd.stride(0), q_bhsd.stride(1), q_bhsd.stride(2), q_bhsd.stride(3),
        k_bhsd.stride(0), k_bhsd.stride(1), k_bhsd.stride(2), k_bhsd.stride(3),
        v_bhsd.stride(0), v_bhsd.stride(1), v_bhsd.stride(2), v_bhsd.stride(3),
        do_bhsd.stride(0), do_bhsd.stride(1), do_bhsd.stride(2), do_bhsd.stride(3),
        dq_bhsd.stride(0), dq_bhsd.stride(1), dq_bhsd.stride(2), dq_bhsd.stride(3),
        dk_bhsd.stride(0), dk_bhsd.stride(1), dk_bhsd.stride(2), dk_bhsd.stride(3),
        dv_bhsd.stride(0), dv_bhsd.stride(1), dv_bhsd.stride(2), dv_bhsd.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        locks.stride(0), locks.stride(1), locks.stride(2),
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