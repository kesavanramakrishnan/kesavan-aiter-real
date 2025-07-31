import math
import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------
# Inner compute kernels for the backward pass (adapted for the new launcher)
# ----------------------------------------------------------------------------

@triton.jit
def _bwd_dkdv_la_inner(
    dk_acc, dv_acc, Q_ptr, k_tile, v_tile, DO_ptr, M_ptr, Delta_ptr, sm_scale,
    stride_qm, stride_qd, stride_dom, stride_dod, stride_deltam,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    max_seqlen_q,
    start_n, start_m_loop, num_m_steps,
    CAUSAL: tl.constexpr,
):
    """
    Inner loop for dK/dV. Iterates over Q blocks for a given K/V block.
    This function is computationally identical to the standard FA2 version.
    """
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Transpose K and V tiles for dot products
    k_tile_T = tl.trans(k_tile)
    v_tile_T = tl.trans(v_tile)

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
        m_vals = tl.load(M_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)
        
        # Compute P = exp(QK^T * sm_scale - M)
        qk = tl.dot(q, k_tile_T)
        # NOTE: p is unnormalized
        p = tl.math.exp(qk * sm_scale - m_vals[:, None])


        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
        
        # Load Delta (row-wise sum of dO * O)
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)

        # Compute dV
        dv_acc += tl.dot(tl.trans(p).to(do.type.element_ty), do)
        
        # Compute dS, then dK
        dp = tl.dot(do, v_tile_T)
        ds = p * (dp - Di[:, None])
        # Apply scaling factor to dS before dot product for numerical stability
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
    This function is computationally identical to the standard FA2 version.
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
        # NOTE: p is unnormalized
        p = tl.math.exp2(qk * sm_scale * 1.44269504 - m_tile)

        
        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        # Compute dP = dO @ V^T
        dp = tl.dot(do_tile, tl.trans(v))
        
        # Load Delta
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)
        
        # Compute dS
        ds = p * (dp - Di[:, None])
        # Apply scaling factor to dS before dot product for numerical stability
        ds = ds * sm_scale
        
        # Compute dQ
        dq_acc += tl.dot(ds, k.to(tl.float32))
        
        curr_n += BLOCK_N
        
    return dq_acc

# ----------------------------------------------------------------------------
# Main Persistent Backward Kernel
# ----------------------------------------------------------------------------

@triton.jit
def bwd_la_persistent(
    # Pointers to matrices
    Q, K, V, sm_scale, DO, O, M, Delta,
    DQ, DK, DV,
    # Partial result buffers and locks
    DKp, DVp, DQp, locks,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_mb, stride_mh, stride_mm,
    stride_deltab, stride_deltah, stride_deltam, # Strides for Delta
    stride_dkp_w, stride_dkp_tile, stride_dkp_block,
    stride_dqp_w, stride_dqp_tile, stride_dqp_block,
    # Lean Attention Scheduling Params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    num_splits: tl.constexpr,
    # Other parameters
    num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    num_m_blocks, num_n_blocks,
    num_dkdv_tiles,
    # Meta-parameters
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < high_load_wgs:
        iter = max_tiles_per_wg * pid
        end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        end_tile_gid = iter + (max_tiles_per_wg - 1)

    while iter < end_tile_gid:
        tile_id = iter
        
        if tile_id < num_dkdv_tiles:
            # ---------- Part 1: compute dK and dV ----------
            n_block_idx = tile_id % num_n_blocks
            head_idx = (tile_id // num_n_blocks) % num_k_heads
            batch_idx = tile_id // (num_n_blocks * num_k_heads)
            
            output_tile_id = batch_idx * num_k_heads * num_n_blocks + head_idx * num_n_blocks + n_block_idx
            host_pid = output_tile_id % high_load_wgs if high_load_wgs > 0 else 0
            is_host = (pid == host_pid)

            start_n = n_block_idx * BLOCK_N1
            dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
            dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

            offs_n = start_n + tl.arange(0, BLOCK_N1)
            offs_d = tl.arange(0, HEAD_DIM)
            mask_kv = (offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)
            
            k_ptrs = K + batch_idx*stride_kb + head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
            v_ptrs = V + batch_idx*stride_vb + head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
            k_tile = tl.load(k_ptrs, mask=mask_kv, other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_kv, other=0.0)

            Q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
            DO_ptr = DO + batch_idx * stride_dob + head_idx * stride_doh
            M_ptr = M + batch_idx * stride_mb + head_idx * stride_mh
            Delta_ptr = Delta + batch_idx * stride_deltab + head_idx * stride_deltah 

            start_m_loop = 0 if not CAUSAL else start_n
            num_m_steps = tl.cdiv(max_seqlen_q - start_m_loop, BLOCK_M1)
            if num_m_steps > 0:
                dk, dv = _bwd_dkdv_la_inner(
                    dk, dv, Q_ptr, k_tile, v_tile, DO_ptr, M_ptr, Delta_ptr, sm_scale,
                    stride_qm, stride_qd, stride_dom, stride_dod, stride_deltam,
                    BLOCK_M1, BLOCK_N1, HEAD_DIM, max_seqlen_q,
                    start_n, start_m_loop, num_m_steps, CAUSAL=CAUSAL
                )
            
            if is_host:
                for i in range(1, num_splits):
                    contributor_pid = (host_pid + i) % high_load_wgs
                    while tl.atomic_cas(locks + contributor_pid, 1, 1) != 1:
                        pass
                    
                    dkp_ptr = DKp + contributor_pid * stride_dkp_w + output_tile_id * stride_dkp_tile
                    dvp_ptr = DVp + contributor_pid * stride_dkp_w + output_tile_id * stride_dkp_tile
                    
                    dk_partial = tl.load(dkp_ptr + tl.arange(0, BLOCK_N1)[:, None] * stride_dkp_block + tl.arange(0, HEAD_DIM)[None, :])
                    dv_partial = tl.load(dvp_ptr + tl.arange(0, BLOCK_N1)[:, None] * stride_dkp_block + tl.arange(0, HEAD_DIM)[None, :])
                    
                    dk += dk_partial
                    dv += dv_partial
                
                dv_ptrs = DV + batch_idx*stride_dvb + head_idx*stride_dvh + offs_n[:,None]*stride_dvn + offs_d[None,:]*stride_dvd
                dk_ptrs = DK + batch_idx*stride_dkb + head_idx*stride_dkh + offs_n[:,None]*stride_dkn + offs_d[None,:]*stride_dkd
                tl.store(dv_ptrs, dv, mask=mask_kv)
                tl.store(dk_ptrs, dk, mask=mask_kv) # Scaling is now done inside
            else:
                dkp_ptr = DKp + pid * stride_dkp_w + output_tile_id * stride_dkp_tile
                dvp_ptr = DVp + pid * stride_dkp_w + output_tile_id * stride_dkp_tile
                
                tl.store(dkp_ptr + tl.arange(0, BLOCK_N1)[:, None] * stride_dkp_block + tl.arange(0, HEAD_DIM)[None, :], dk)
                tl.store(dvp_ptr + tl.arange(0, BLOCK_N1)[:, None] * stride_dkp_block + tl.arange(0, HEAD_DIM)[None, :], dv)
                tl.atomic_xchg(locks + pid, 1)

        else:
            # ---------- Part 2: compute dQ ----------
            dq_tile_id = tile_id - num_dkdv_tiles
            
            m_block_idx = dq_tile_id % num_m_blocks
            head_idx = (dq_tile_id // num_m_blocks) % num_q_heads
            batch_idx = dq_tile_id // (num_m_blocks * num_q_heads)

            output_tile_id = batch_idx * num_q_heads * num_m_blocks + head_idx * num_m_blocks + m_block_idx
            host_pid = output_tile_id % high_load_wgs if high_load_wgs > 0 else 0
            is_host = (pid == host_pid)

            start_m = m_block_idx * BLOCK_M2

            offs_m = start_m + tl.arange(0, BLOCK_M2)
            offs_d = tl.arange(0, HEAD_DIM)
            mask_q = (offs_m[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)

            q_ptrs = Q + batch_idx*stride_qb + head_idx*stride_qh + offs_m[:,None]*stride_qm + offs_d[None,:]*stride_qd
            do_ptrs = DO + batch_idx*stride_dob + head_idx*stride_doh + offs_m[:,None]*stride_dom + offs_d[None,:]*stride_dod
            m_ptrs = M + batch_idx*stride_mb + head_idx*stride_mh + offs_m*stride_mm
            
            q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
            do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
            m_tile = tl.load(m_ptrs, mask=(offs_m < max_seqlen_q), other=-float('inf'))[:, None]
            
            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

            K_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
            V_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
            Delta_ptr = Delta + batch_idx * stride_deltab + head_idx * stride_deltah


            start_n_loop = 0
            end_n_loop = max_seqlen_k if not CAUSAL else min(start_m + BLOCK_M2, max_seqlen_k)
            num_n_steps = tl.cdiv(end_n_loop - start_n_loop, BLOCK_N2)
            if num_n_steps > 0:
                dq = _bwd_dq_la_inner(
                    dq, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
                    stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
                    max_seqlen_q, max_seqlen_k,
                    BLOCK_M2, BLOCK_N2, HEAD_DIM,
                    start_m, start_n_loop, num_n_steps, CAUSAL=CAUSAL
                )

            if is_host:
                for i in range(1, num_splits):
                    contributor_pid = (host_pid + i) % high_load_wgs
                    while tl.atomic_cas(locks + contributor_pid, 1, 1) != 1:
                        pass
                    
                    dqp_ptr = DQp + contributor_pid * stride_dqp_w + output_tile_id * stride_dqp_tile
                    dq_partial = tl.load(dqp_ptr + tl.arange(0, BLOCK_M2)[:, None] * stride_dqp_block + tl.arange(0, HEAD_DIM)[None, :])
                    dq += dq_partial
                
                dq_ptrs = DQ + batch_idx*stride_dqb + head_idx*stride_dqh + offs_m[:,None]*stride_dqm + offs_d[None,:]*stride_dqd
                tl.store(dq_ptrs, dq, mask=mask_q) # Scaling is now done inside
            else:
                dqp_ptr = DQp + pid * stride_dqp_w + output_tile_id * stride_dqp_tile
                tl.store(dqp_ptr + tl.arange(0, BLOCK_M2)[:, None] * stride_dqp_block + tl.arange(0, HEAD_DIM)[None, :], dq)
                tl.atomic_xchg(locks + pid, 1)

        iter += 1


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
    Backward pass launcher using the persistent Lean Attention scheduling style.
    """
    config = {
        "BLOCK_M1": 32, "BLOCK_N1": 64,
        "BLOCK_M2": 64, "BLOCK_N2": 32,
    }
    
    batch, _, num_q_heads, head_sz = q.shape
    _, _, num_k_heads, _ = k.shape
    assert num_q_heads == num_k_heads, "MQA/GQA not supported"

    q, k, v, o, do = [t.transpose(1, 2) for t in (q, k, v, o, do)]
    dq, dk, dv = [t.transpose(1, 2) for t in (dq, dk, dv)]

    delta = torch.sum(o * do, dim=-1, keepdim=False)
    
    num_m_blocks = triton.cdiv(max_seqlen_q, config["BLOCK_M2"])
    num_n_blocks = triton.cdiv(max_seqlen_k, config["BLOCK_N1"])
    
    num_dkdv_tiles = batch * num_k_heads * num_n_blocks
    num_dq_tiles = batch * num_q_heads * num_m_blocks
    total_tiles = num_dkdv_tiles + num_dq_tiles
    
    num_wgs = 1024
    if total_tiles < num_wgs:
        num_wgs = total_tiles
        
    max_tiles_per_wg = math.ceil(total_tiles / num_wgs)
    high_load_wgs = total_tiles % num_wgs
    if high_load_wgs == 0:
        high_load_wgs = num_wgs
    
    num_splits = math.ceil(total_tiles / (num_wgs * max_tiles_per_wg))

    grid = (num_wgs,)

    num_output_dkdv_tiles = batch * num_k_heads * num_n_blocks
    dkp = torch.zeros((num_wgs, num_output_dkdv_tiles, config["BLOCK_N1"], head_sz), device=q.device, dtype=torch.float32)
    dvp = torch.zeros_like(dkp)
    
    num_output_dq_tiles = batch * num_q_heads * num_m_blocks
    dqp = torch.zeros((num_wgs, num_output_dq_tiles, config["BLOCK_M2"], head_sz), device=q.device, dtype=torch.float32)
    locks = torch.zeros((num_wgs,), device=q.device, dtype=torch.int32)

    bwd_la_persistent[grid](
        q, k, v, sm_scale, do, o, softmax_lse, delta,
        dq, dk, dv,
        dkp, dvp, dqp, locks,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        dkp.stride(0), dkp.stride(1), dkp.stride(2),
        dqp.stride(0), dqp.stride(1), dqp.stride(2),
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        num_splits=num_splits,
        num_q_heads=num_q_heads, num_k_heads=num_k_heads,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        num_m_blocks=num_m_blocks, num_n_blocks=num_n_blocks,
        num_dkdv_tiles=num_dkdv_tiles,
        HEAD_DIM=head_sz,
        CAUSAL=causal,
        **config,
    )
    
    dq.transpose_(1, 2)
    dk.transpose_(1, 2)
    dv.transpose_(1, 2)
