# import math
# import torch
# import triton
# import triton.language as tl

# # ----------------------------------------------------------------------------
# # Inner compute kernels for the backward pass (with corrections)
# # ----------------------------------------------------------------------------

# @triton.jit
# def _bwd_dkdv_la_inner(
#     dk_acc, dv_acc, Q_ptr, k_tile, v_tile, DO_ptr, M_ptr, Delta_ptr, sm_scale,
#     stride_qm, stride_qd, stride_dom, stride_dod, stride_mm, stride_deltam,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
#     max_seqlen_q, max_seqlen_k,
#     start_n, start_m_loop, num_m_steps,
#     CAUSAL: tl.constexpr,
# ):
#     """
#     Inner loop for dK/dV. Iterates over Q blocks for a given K/V block.
#     This function is computationally identical to the standard FA2 version.
#     """
#     offs_n = start_n + tl.arange(0, BLOCK_N)
#     offs_d = tl.arange(0, HEAD_DIM)

#     k_tile_T = tl.trans(k_tile)

#     curr_m = start_m_loop
#     for _ in range(num_m_steps):
#         offs_m = curr_m + tl.arange(0, BLOCK_M)
#         mask_m = offs_m < max_seqlen_q

#         # Load Q and dO
#         q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
#         do_ptrs = DO_ptr + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
#         mask_q = mask_m[:, None] & (offs_d[None, :] < HEAD_DIM)

#         q = tl.load(q_ptrs, mask=mask_q, other=0.0)
#         do = tl.load(do_ptrs, mask=mask_q, other=0.0)

#         # Load softmax stats (log-sum-exp)
#         m_vals = tl.load(M_ptr + offs_m * stride_mm, mask=mask_m, other=0.0)

#         # Recompute P = exp(QK^T * sm_scale - M)
#         qk = tl.dot(q, k_tile_T)
#         # FIX: Reverted to tl.math.exp as softmax_lse (M) is the natural log-sum-exp.
#         p = tl.math.exp(qk * sm_scale - m_vals[:, None])

#         if CAUSAL:
#             # Correct causal mask alignment for varying seq lens
#             causal_mask = (offs_m[:, None] >= (offs_n[None, :] + max_seqlen_q - max_seqlen_k))
#             p = tl.where(causal_mask, p, 0.0)

#         # Load Delta (row-wise sum of dO * O) using the correct stride
#         Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)

#         # Compute dV
#         dv_acc += tl.dot(tl.trans(p).to(do.type.element_ty), do)

#         # Compute dS, then dK
#         dp = tl.dot(do, tl.trans(v_tile))
#         ds = p * (dp - Di[:, None])
#         # Apply scaling factor to dS before dot product
#         ds = ds * sm_scale
#         dk_acc += tl.dot(tl.trans(ds).to(q.type.element_ty), q)

#         curr_m += BLOCK_M

#     return dk_acc, dv_acc

# @triton.jit
# def _bwd_dq_la_inner(
#     dq_acc, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
#     stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
#     max_seqlen_q, max_seqlen_k,
#     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
#     start_m, start_n_loop, num_n_steps,
#     CAUSAL: tl.constexpr,
# ):
#     """
#     Inner loop for dQ. Iterates over K/V blocks for a given Q block.
#     This function is computationally identical to the standard FA2 version.
#     """
#     offs_m = start_m + tl.arange(0, BLOCK_M)
#     offs_d = tl.arange(0, HEAD_DIM)

#     curr_n = start_n_loop
#     for _ in range(num_n_steps):
#         offs_n = curr_n + tl.arange(0, BLOCK_N)
#         mask_n = offs_n < max_seqlen_k

#         # Load K and V
#         k_ptrs = K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
#         v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
#         mask_kv = mask_n[:, None] & (offs_d[None, :] < HEAD_DIM)

#         k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
#         v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

#         # Compute P = exp(QK^T * sm_scale - M)
#         qk = tl.dot(q_tile, tl.trans(k))
#         # FIX: Reverted to tl.math.exp for consistency with softmax_lse.
#         p = tl.math.exp(qk * sm_scale - m_tile)

#         if CAUSAL:
#             causal_mask = (offs_m[:, None] >= (offs_n[None, :] + max_seqlen_q - max_seqlen_k))
#             p = tl.where(causal_mask, p, 0.0)

#         # Compute dP = dO @ V^T
#         dp = tl.dot(do_tile, tl.trans(v))

#         # Load Delta
#         Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)

#         # Compute dS
#         ds = p * (dp - Di[:, None])
#         # Apply scaling factor to dS before dot product
#         ds = ds * sm_scale

#         # Compute dQ
#         dq_acc += tl.dot(ds.to(k.dtype), k)

#         curr_n += BLOCK_N

#     return dq_acc

# # ----------------------------------------------------------------------------
# # Main Persistent Backward Kernel with ATOMIC REDUCTION
# # ----------------------------------------------------------------------------

# @triton.jit
# def bwd_la_persistent(
#     # Pointers to matrices
#     Q, K, V, sm_scale, DO, O, M, Delta,
#     DQ, DK, DV,
#     # Strides
#     stride_qb, stride_qh, stride_qm, stride_qd,
#     stride_kb, stride_kh, stride_kn, stride_kd,
#     stride_vb, stride_vh, stride_vn, stride_vd,
#     stride_dob, stride_doh, stride_dom, stride_dod,
#     stride_dqb, stride_dqh, stride_dqm, stride_dqd,
#     stride_dkb, stride_dkh, stride_dkn, stride_dkd,
#     stride_dvb, stride_dvh, stride_dvn, stride_dvd,
#     stride_mb, stride_mh, stride_mm,
#     stride_deltab, stride_deltah, stride_deltam,
#     # Lean Attention Scheduling Params
#     total_work_items,
#     high_load_wgs,
#     max_tiles_per_wg,
#     # Workload split point
#     num_dkdv_work_items,
#     # Other parameters
#     num_q_heads, num_k_heads,
#     max_seqlen_q, max_seqlen_k,
#     num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2,
#     # Meta-parameters
#     BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
#     BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
#     HEAD_DIM: tl.constexpr,
#     CAUSAL: tl.constexpr,
# ):
#     # 1. Persistent scheduling to get the global work-item ID
#     pid = tl.program_id(0)
#     if pid < high_load_wgs:
#         iter_start = max_tiles_per_wg * pid
#         num_tiles_to_process = max_tiles_per_wg
#     else:
#         iter_start = (max_tiles_per_wg - 1) * (pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
#         num_tiles_to_process = max_tiles_per_wg - 1

#     # Each WG processes its assigned range of work items
#     # The loop is structured to avoid the unsupported 'break' statement
#     for i in range(num_tiles_to_process):
#         work_item_id = iter_start + i
        
#         # Guard the entire loop body to avoid processing out-of-bounds work items
#         if work_item_id < total_work_items:
#             # 2. Determine if this work item is for dK/dV or dQ
#             # A "work item" is one Q_block x K_block interaction
#             if work_item_id < num_dkdv_work_items:
#                 # ========== COMPUTE PARTIAL dK and dV ==========
#                 # Deconstruct work_item_id to get (batch, head, m_block, n_block)
#                 item_id = work_item_id
#                 m_block_idx = item_id % num_m_blocks_1
#                 item_id = item_id // num_m_blocks_1
#                 n_block_idx = item_id % num_n_blocks_1
#                 item_id = item_id // num_n_blocks_1
#                 head_idx = item_id % num_k_heads
#                 batch_idx = item_id // num_k_heads

#                 start_m = m_block_idx * BLOCK_M1
#                 start_n = n_block_idx * BLOCK_N1

#                 # Initialize partial result accumulators
#                 dk_partial = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
#                 dv_partial = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

#                 # Load K and V tiles
#                 offs_n = start_n + tl.arange(0, BLOCK_N1)
#                 offs_d = tl.arange(0, HEAD_DIM)
#                 mask_kv = (offs_n[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)
#                 k_ptrs = K + batch_idx*stride_kb + head_idx*stride_kh + offs_n[:,None]*stride_kn + offs_d[None,:]*stride_kd
#                 v_ptrs = V + batch_idx*stride_vb + head_idx*stride_vh + offs_n[:,None]*stride_vn + offs_d[None,:]*stride_vd
#                 k_tile = tl.load(k_ptrs, mask=mask_kv, other=0.0)
#                 v_tile = tl.load(v_ptrs, mask=mask_kv, other=0.0)

#                 # Pointers to head-specific data
#                 Q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
#                 DO_ptr = DO + batch_idx * stride_dob + head_idx * stride_doh
#                 M_ptr = M + batch_idx * stride_mb + head_idx * stride_mh
#                 Delta_ptr = Delta + batch_idx * stride_deltab + head_idx * stride_deltah

#                 # Compute partial gradients for this single (Q_block, K_block) pair
#                 dk_partial, dv_partial = _bwd_dkdv_la_inner(
#                     dk_partial, dv_partial, Q_ptr, k_tile, v_tile, DO_ptr, M_ptr, Delta_ptr, sm_scale,
#                     stride_qm, stride_qd, stride_dom, stride_dod, stride_mm, stride_deltam,
#                     BLOCK_M1, BLOCK_N1, HEAD_DIM, max_seqlen_q, max_seqlen_k,
#                     start_n, start_m, 1, # num_m_steps is 1 because this is one work item
#                     CAUSAL=CAUSAL
#                 )

#                 # 3. ATOMICALLY ADD to global memory
#                 dv_ptrs_out = DV + batch_idx*stride_dvb + head_idx*stride_dvh + offs_n[:,None]*stride_dvn + offs_d[None,:]*stride_dvd
#                 dk_ptrs_out = DK + batch_idx*stride_dkb + head_idx*stride_dkh + offs_n[:,None]*stride_dkn + offs_d[None,:]*stride_dkd
#                 tl.atomic_add(dv_ptrs_out, dv_partial, mask=mask_kv)
#                 tl.atomic_add(dk_ptrs_out, dk_partial, mask=mask_kv)

#             else:
#                 # ========== COMPUTE PARTIAL dQ ==========
#                 item_id = work_item_id - num_dkdv_work_items
#                 n_block_idx = item_id % num_n_blocks_2
#                 item_id = item_id // num_n_blocks_2
#                 m_block_idx = item_id % num_m_blocks_2
#                 item_id = item_id // num_m_blocks_2
#                 head_idx = item_id % num_q_heads
#                 batch_idx = item_id // num_q_heads

#                 start_m = m_block_idx * BLOCK_M2
#                 start_n = n_block_idx * BLOCK_N2

#                 # Initialize partial result accumulator
#                 dq_partial = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

#                 # Load Q, dO, M tiles
#                 offs_m = start_m + tl.arange(0, BLOCK_M2)
#                 offs_d = tl.arange(0, HEAD_DIM)
#                 mask_q = (offs_m[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)
#                 q_ptrs = Q + batch_idx*stride_qb + head_idx*stride_qh + offs_m[:,None]*stride_qm + offs_d[None,:]*stride_qd
#                 do_ptrs = DO + batch_idx*stride_dob + head_idx*stride_doh + offs_m[:,None]*stride_dom + offs_d[None,:]*stride_dod
#                 m_ptrs = M + batch_idx*stride_mb + head_idx*stride_mh + offs_m*stride_mm
#                 q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
#                 do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
#                 m_tile = tl.load(m_ptrs, mask=(offs_m < max_seqlen_q), other=-float('inf'))[:, None]

#                 # Pointers to head-specific data
#                 K_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
#                 V_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
#                 Delta_ptr = Delta + batch_idx * stride_deltab + head_idx * stride_deltah

#                 # Compute partial gradient for this single (Q_block, K_block) pair
#                 dq_partial = _bwd_dq_la_inner(
#                     dq_partial, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
#                     stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
#                     max_seqlen_q, max_seqlen_k,
#                     BLOCK_M2, BLOCK_N2, HEAD_DIM,
#                     start_m, start_n, 1, # num_n_steps is 1 because this is one work item
#                     CAUSAL=CAUSAL
#                 )

#                 # 3. ATOMICALLY ADD to global memory
#                 dq_ptrs_out = DQ + batch_idx*stride_dqb + head_idx*stride_dqh + offs_m[:,None]*stride_dqm + offs_d[None,:]*stride_dqd
#                 tl.atomic_add(dq_ptrs_out, dq_partial, mask=mask_q)

# # ----------------------------------------------------------------------------
# # Python Launcher for the Persistent Backward Pass
# # ----------------------------------------------------------------------------

# def la_backward_persistent(
#     do: torch.Tensor,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     o: torch.Tensor,
#     softmax_lse: torch.Tensor,
#     dq: torch.Tensor,
#     dk: torch.Tensor,
#     dv: torch.Tensor,
#     sm_scale: float,
#     causal: bool,
#     max_seqlen_q: int,
#     max_seqlen_k: int,
# ):
#     """
#     Backward pass launcher using a persistent, work-centric kernel with atomic reductions.
#     """
#     config = {
#         "BLOCK_M1": 32, "BLOCK_N1": 64,
#         "BLOCK_M2": 64, "BLOCK_N2": 32,
#     }

#     batch, _, num_q_heads, head_sz = q.shape
#     _, _, num_k_heads, _ = k.shape
#     assert num_q_heads == num_k_heads, "MQA/GQA not supported yet"

#     # Kernel expects tensors in [batch, heads, seqlen, headdim]
#     # The test script provides tensors in [batch, seqlen, heads, headdim], so we transpose.
#     q_bhsd = q.transpose(1, 2).contiguous()
#     k_bhsd = k.transpose(1, 2).contiguous()
#     v_bhsd = v.transpose(1, 2).contiguous()
#     o_bhsd = o.transpose(1, 2).contiguous()
#     do_bhsd = do.transpose(1, 2).contiguous()
#     # The output gradient tensors also need to be in the kernel's expected layout
#     dq_bhsd = torch.empty_like(q_bhsd)
#     dk_bhsd = torch.empty_like(k_bhsd)
#     dv_bhsd = torch.empty_like(v_bhsd)
    
#     # CRITICAL: Initialize gradient tensors to zero for atomic accumulation.
#     dq_bhsd.zero_()
#     dk_bhsd.zero_()
#     dv_bhsd.zero_()

#     # Pre-computation of Delta (can also be a small separate kernel)
#     delta = torch.sum(o_bhsd * do_bhsd, dim=-1, keepdim=False)

#     num_m_blocks_1 = triton.cdiv(max_seqlen_q, config["BLOCK_M1"])
#     num_n_blocks_1 = triton.cdiv(max_seqlen_k, config["BLOCK_N1"])
#     num_m_blocks_2 = triton.cdiv(max_seqlen_q, config["BLOCK_M2"])
#     num_n_blocks_2 = triton.cdiv(max_seqlen_k, config["BLOCK_N2"])

#     # Define the total workload. A "work item" is one Q_block x K_block interaction.
#     num_dkdv_work_items = batch * num_k_heads * num_n_blocks_1 * num_m_blocks_1
#     num_dq_work_items = batch * num_q_heads * num_m_blocks_2 * num_n_blocks_2
#     total_work_items = num_dkdv_work_items + num_dq_work_items

#     # Heuristic for the number of work-groups (WGs) or CTAs
#     num_wgs = 108 * 4 # Example: 4 waves on an A100
#     if total_work_items < num_wgs:
#         num_wgs = total_work_items if total_work_items > 0 else 1

#     max_tiles_per_wg = triton.cdiv(total_work_items, num_wgs)
#     high_load_wgs = total_work_items % num_wgs
#     if high_load_wgs == 0 and total_work_items > 0:
#         high_load_wgs = num_wgs

#     grid = (num_wgs,)

#     bwd_la_persistent[grid](
#         q_bhsd, k_bhsd, v_bhsd, sm_scale, do_bhsd, o_bhsd, softmax_lse, delta,
#         dq_bhsd, dk_bhsd, dv_bhsd,
#         # Strides for [B, H, S, D] layout
#         q_bhsd.stride(0), q_bhsd.stride(1), q_bhsd.stride(2), q_bhsd.stride(3),
#         k_bhsd.stride(0), k_bhsd.stride(1), k_bhsd.stride(2), k_bhsd.stride(3),
#         v_bhsd.stride(0), v_bhsd.stride(1), v_bhsd.stride(2), v_bhsd.stride(3),
#         do_bhsd.stride(0), do_bhsd.stride(1), do_bhsd.stride(2), do_bhsd.stride(3),
#         dq_bhsd.stride(0), dq_bhsd.stride(1), dq_bhsd.stride(2), dq_bhsd.stride(3),
#         dk_bhsd.stride(0), dk_bhsd.stride(1), dk_bhsd.stride(2), dk_bhsd.stride(3),
#         dv_bhsd.stride(0), dv_bhsd.stride(1), dv_bhsd.stride(2), dv_bhsd.stride(3),
#         softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
#         delta.stride(0), delta.stride(1), delta.stride(2),
#         # Lean Attention Scheduling Params
#         total_work_items,
#         high_load_wgs,
#         max_tiles_per_wg,
#         # Workload split point
#         num_dkdv_work_items,
#         # Other parameters
#         num_q_heads, num_k_heads,
#         max_seqlen_q, max_seqlen_k,
#         num_m_blocks_1, num_m_blocks_2, num_n_blocks_1, num_n_blocks_2,
#         # Meta-parameters
#         HEAD_DIM=head_sz,
#         CAUSAL=causal,
#         **config,
#     )

#     # Transpose the final outputs back to the user-expected [B, S, H, D] layout
#     dq.copy_(dq_bhsd.transpose(1, 2))
#     dk.copy_(dk_bhsd.transpose(1, 2))
#     dv.copy_(dv_bhsd.transpose(1, 2))
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
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k_tile_T = tl.trans(k_tile)
    v_tile_T = tl.trans(v_tile)

    curr_m = start_m_loop
    for _ in range(num_m_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < max_seqlen_q

        q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        do_ptrs = DO_ptr + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
        mask_q = mask_m[:, None] & (offs_d[None, :] < HEAD_DIM)

        q = tl.load(q_ptrs, mask=mask_q, other=0.0)
        do = tl.load(do_ptrs, mask=mask_q, other=0.0)
        m_vals = tl.load(M_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)

        qk = tl.dot(q, k_tile_T)
        p = tl.math.exp(qk * sm_scale - m_vals[:, None])
        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)
        dv_acc += tl.dot(tl.trans(p).to(do.type.element_ty), do)

        dp = tl.dot(do, v_tile_T)
        ds = p * (dp - Di[:, None]) * sm_scale
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
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    curr_n = start_n_loop
    for _ in range(num_n_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < max_seqlen_k

        k_ptrs = K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        mask_kv = mask_n[:, None] & (offs_d[None, :] < HEAD_DIM)

        k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

        qk = tl.dot(q_tile, tl.trans(k))
        p = tl.math.exp2(qk * sm_scale * 1.44269504 - m_tile)
        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        dp = tl.dot(do_tile, tl.trans(v))
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)
        ds = p * (dp - Di[:, None]) * sm_scale

        dq_acc += tl.dot(ds, k.to(tl.float32))
        curr_n += BLOCK_N

    return dq_acc
