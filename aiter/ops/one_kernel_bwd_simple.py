# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Dict
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_preprocess(
    o_ptr,
    do_ptr,
    delta_ptr,
    stride_o_b,
    stride_o_h,
    stride_o_m,
    stride_o_k,
    stride_delta_b,
    stride_delta_h,
    stride_delta_m,
    max_seqlen_q,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Computes the initial delta value, which is the row-wise dot product
    of the output gradient (dO) and the output (O).
    """
    pid_m = tl.program_id(0)  # seqlen
    bid = tl.program_id(1)  # batch
    hid = tl.program_id(2)  # head

    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    # Offset O/DO by batch and head
    offs = (
        bid * stride_o_b
        + hid * stride_o_h
        + offs_m[:, None] * stride_o_m
        + offs_k[None, :] * stride_o_k
    )

    # Create masks
    mask_m = offs_m < max_seqlen_q
    mask_k = offs_k < HEAD_DIM
    mask = mask_m[:, None] & mask_k[None, :]

    # Load blocks of O and dO
    o = tl.load(o_ptr + offs, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs, mask=mask, other=0.0)

    # Compute delta and store
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    offs_delta = (
        bid * stride_delta_b
        + hid * stride_delta_h
        + offs_m * stride_delta_m
    )
    tl.store(delta_ptr + offs_delta, delta, mask=mask_m)


@triton.jit
def _bwd_dkdv_inner(
    dk, dv,
    Q, k, v, DO, M, D,
    sm_scale,
    stride_qm, stride_qk,
    stride_dom, stride_dok,
    stride_deltam,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    seqlen_q,
    seqlen_k,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
):
    """
    Inner loop for computing dK and dV.
    It iterates over blocks of Q.
    """
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Pointers to Q and dO that will be advanced
    qT_ptrs = Q + offs_k[:, None] * stride_qk
    do_ptrs = DO + offs_k[None, :] * stride_dok

    curr_m = start_m
    for _ in range(num_steps):
        offs_m = curr_m + offs_m_base
        mask_m = offs_m < seqlen_q

        qT_ptrs_step = qT_ptrs + offs_m[None, :] * stride_qm
        do_ptrs_step = do_ptrs + offs_m[:, None] * stride_dom
        
        mask_qT = mask_m[None, :] & (offs_k[:, None] < HEAD_DIM)
        mask_do = mask_m[:, None] & (offs_k[None, :] < HEAD_DIM)

        qT = tl.load(qT_ptrs_step, mask=mask_qT, other=0.0)
        
        # Load softmax stats
        m_vals = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        
        # Compute P = exp(QK^T * sm_scale - M)
        qkT = tl.dot(k, qT)
        qkT_scaled = qkT * sm_scale
        pT = tl.math.exp(qkT_scaled - m_vals[None, :])

        if MASK:
            causal_mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(causal_mask, pT, 0.0)
        
        # Compute dV
        do = tl.load(do_ptrs_step, mask=mask_do, other=0.0)
        dv += tl.dot(pT.to(do.type.element_ty), do)

        # Compute dS
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])
        
        # Compute dK
        dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))
        
        curr_m += BLOCK_M
        
    return dk, dv


@triton.jit
def _bwd_dq_inner(
    dq,
    q, K, V, do, m, Delta,
    sm_scale,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_deltam,
    seqlen_q,
    seqlen_k,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    end_n,
    num_steps,
    MASK: tl.constexpr,
):
    """
    Inner loop for computing dQ.
    It iterates over blocks of K and V.
    """
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_k = tl.arange(0, HEAD_DIM)

    # Pointers to K and V that will be advanced
    kT_ptrs = K + offs_k[:, None] * stride_kk
    vT_ptrs = V + offs_k[:, None] * stride_vk
    
    Di = tl.load(Delta + offs_m * stride_deltam, mask=(offs_m < seqlen_q), other=0.0)

    curr_n = start_n
    for _ in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        mask_n = offs_n < end_n
        
        kT_ptrs_step = kT_ptrs + offs_n[None, :] * stride_kn
        vT_ptrs_step = vT_ptrs + offs_n[None, :] * stride_vn
        
        mask_kT = mask_n[None, :] & (offs_k[:, None] < HEAD_DIM)

        kT = tl.load(kT_ptrs_step, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs_step, mask=mask_kT, other=0.0)

        # Compute P = exp(QK^T * sm_scale - M)
        qk = tl.dot(q, kT)
        qk_scaled = qk * sm_scale
        p = tl.math.exp(qk_scaled - m)

        if MASK:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(causal_mask, p, 0.0)
        
        # Compute dS
        dp = tl.dot(do, vT)
        ds = p * (dp - Di[:, None])
        
        # Compute dQ
        dq += tl.dot(ds.to(kT.type.element_ty), tl.trans(kT))

        curr_n += BLOCK_N2
        
    return dq


@triton.jit
def bwd_kernel(
    # Pointers to matrices
    Q, K, V, sm_scale, DO,
    DQ, DK, DV,
    M, Delta,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dob, stride_doh, stride_dom, stride_dod,
    # Other parameters
    num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    # Meta-parameters
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Main backward kernel for flash attention.
    Computes dQ, dK, and dV.
    It is split into two main parts:
    1. Compute dK and dV by iterating over blocks of K/V.
    2. Compute dQ by iterating over blocks of Q.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    bid = tl.program_id(2)
    
    # This kernel is parallelized over batch, head, and sequence dimension.
    # It assumes MHA, where each Q head corresponds to a K/V head.
    # A given block computes one of the following:
    #  - A block of dK/dV for a given head and batch.
    #  - A block of dQ for a given head and batch.
    
    # For simplicity, we use two separate program IDs for the sequence dimension
    # to handle the two parts of the calculation.
    
    # ---------- Part 1: compute dK and dV ----------
    start_n = pid_n * BLOCK_N1
    if start_n < max_seqlen_k:
        # Each thread block processes one head
        hqid = tl.program_id(3)
        hkid = hqid

        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

        offs_n = start_n + tl.arange(0, BLOCK_N1)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_kv = (offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)

        k_ptrs = K + bid * stride_kb + hkid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + bid * stride_vb + hkid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

        Q_ptr = Q + bid * stride_qb + hqid * stride_qh
        DO_ptr = DO + bid * stride_dob + hqid * stride_doh
        M_ptr = M + bid * stride_deltab + hqid * stride_deltah
        Delta_ptr = Delta + bid * stride_deltab + hqid * stride_deltah

        # Iterate over Q blocks
        start_m_loop = 0 if not CAUSAL else start_n
        num_steps = tl.cdiv(max_seqlen_q - start_m_loop, BLOCK_M1)
        
        if num_steps > 0:
            dk, dv = _bwd_dkdv_inner(
                dk, dv, Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, sm_scale,
                stride_qm, stride_qd, stride_dom, stride_dod, stride_deltam,
                BLOCK_M1, BLOCK_N1, HEAD_DIM, max_seqlen_q, max_seqlen_k,
                start_n, start_m_loop, num_steps, MASK=CAUSAL)

        # Write back dK and dV
        dv_ptrs = DV + bid * stride_dvb + hkid * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(dv_ptrs, dv, mask=mask_kv)
        dk_ptrs = DK + bid * stride_dkb + hkid * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(dk_ptrs, dk, mask=mask_kv)

    # ---------- Part 2: compute dQ ----------
    start_m = pid_m * BLOCK_M2
    if start_m < max_seqlen_q:
        # Each thread block processes one head
        hqid = tl.program_id(3)
        hkid = hqid

        offs_m = start_m + tl.arange(0, BLOCK_M2)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_q = (offs_m[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)

        q_ptrs = Q + bid * stride_qb + hqid * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = DO + bid * stride_dob + hqid * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        m_ptrs = M + bid * stride_deltab + hqid * stride_deltah + offs_m * stride_deltam

        q = tl.load(q_ptrs, mask=mask_q, other=0.0)
        do = tl.load(do_ptrs, mask=mask_q, other=0.0)
        m = tl.load(m_ptrs, mask=(offs_m < max_seqlen_q))[:, None]

        K_ptr = K + bid * stride_kb + hkid * stride_kh
        V_ptr = V + bid * stride_vb + hkid * stride_vh
        Delta_ptr = Delta + bid * stride_deltab + hqid * stride_deltah

        dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
        
        # Iterate over K blocks
        start_n_loop = 0
        end_n_loop = max_seqlen_k if not CAUSAL else min(start_m + BLOCK_M2, max_seqlen_k)
        num_steps = tl.cdiv(end_n_loop - start_n_loop, BLOCK_N2)

        if num_steps > 0:
            dq = _bwd_dq_inner(
                dq, q, K_ptr, V_ptr, do, m, Delta_ptr, sm_scale,
                stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk, stride_deltam,
                max_seqlen_q, max_seqlen_k, BLOCK_M2, BLOCK_N2, HEAD_DIM,
                start_m, start_n_loop, end_n_loop, num_steps, MASK=CAUSAL)

        # Write back dQ
        dq_ptrs = DQ + bid * stride_dqb + hqid * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
        dq *= sm_scale
        tl.store(dq_ptrs, dq, mask=mask_q)


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
    sm_scale: float,
    causal: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
):
    """
    Backward pass for scaled dot-product attention.

    Args:
        do: Gradient of the output tensor.
        q, k, v, o: Tensors from the forward pass.
        softmax_lse: Log-sum-exp of the softmax from the forward pass.
        dq, dk, dv: Output tensors for the gradients of Q, K, and V.
        sm_scale: Scaling factor for the dot product.
        causal: If True, applies a causal mask.
        max_seqlen_q, max_seqlen_k: Maximum sequence lengths.
    """
    # Define kernel constants
    config = {
        "preprocess_kernel": {"PRE_BLOCK": 64},
        "onekernel": {
            "BLOCK_M1": 32, "BLOCK_N1": 64,
            "BLOCK_M2": 64, "BLOCK_N2": 32,
        },
    }

    # Tensor shapes
    batch, seqlen_q, num_q_heads, head_sz = q.shape
    _, _, num_k_heads, _ = k.shape

    # This basic implementation assumes Multi-Head Attention (MHA)
    assert num_q_heads == num_k_heads, "MQA/GQA not supported in this basic version"

    # Reshape tensors to [batch, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    o = o.transpose(1, 2)
    do = do.transpose(1, 2)
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)
    dv = dv.transpose(1, 2)
    
    # Strides
    q_strides = q.stride()
    k_strides = k.stride()
    v_strides = v.stride()
    o_strides = o.stride()
    do_strides = do.stride()
    dq_strides = dq.stride()
    dk_strides = dk.stride()
    dv_strides = dv.stride()

    # Pre-computation of Delta
    delta = torch.empty_like(softmax_lse)
    delta_strides = delta.stride()
    
    pre_grid = (
        triton.cdiv(max_seqlen_q, config["preprocess_kernel"]["PRE_BLOCK"]),
        batch,
        num_q_heads,
    )
    _bwd_preprocess[pre_grid](
        o, do, delta,
        *o_strides,
        *delta_strides,
        max_seqlen_q,
        BLOCK_M=config["preprocess_kernel"]["PRE_BLOCK"],
        HEAD_DIM=head_sz,
    )
    
    # Main backward kernel
    # We launch two separate grids for dK/dV and dQ computations
    grid_dk_dv = (
        triton.cdiv(max_seqlen_k, config["onekernel"]["BLOCK_N1"]), 
        1, 
        batch, 
        num_k_heads
    )
    grid_dq = (
        1, 
        triton.cdiv(max_seqlen_q, config["onekernel"]["BLOCK_M2"]), 
        batch, 
        num_q_heads
    )
    
    bwd_kernel[grid_dk_dv, grid_dq](
        q, k, v, sm_scale, do,
        dq, dk, dv,
        softmax_lse, delta,
        *q_strides, *k_strides, *v_strides,
        *dq_strides, *dk_strides, *dv_strides,
        *delta_strides,
        *do_strides,
        num_q_heads, num_k_heads,
        max_seqlen_q, max_seqlen_k,
        HEAD_DIM=head_sz,
        CAUSAL=causal,
        **config["onekernel"],
    )

    # Transpose gradients back to original layout
    dq.transpose_(1, 2)
    dk.transpose_(1, 2)
    dv.transpose_(1, 2)  