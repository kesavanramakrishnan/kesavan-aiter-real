import math
import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------
# Inner compute kernels (largely the same, with minor consistency fixes)
# ----------------------------------------------------------------------------

@triton.jit
def _bwd_dkdv_inner(
    dk_acc, dv_acc, Q_ptr, k_tile, v_tile, DO_ptr, LSE_ptr, Delta_ptr, sm_scale,
    stride_qm, stride_qd, stride_dom, stride_dod, stride_mm, stride_deltam,
    seqlen_q, seqlen_k,
    start_n,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # This CTA is responsible for a single BLOCK_N of K/V
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Loop over Q in blocks of BLOCK_M
    for start_m in range(0, seqlen_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q

        # Load Q and dO blocks
        q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        do_ptrs = DO_ptr + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod)
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        # Recompute P = exp(QK^T * sm_scale - LSE)
        # 1. S = Q @ K^T
        qk = tl.dot(q, tl.trans(k_tile))
        # 2. P = exp(S * sm_scale - LSE)
        lse = tl.load(LSE_ptr + offs_m * stride_mm, mask=mask_m, other=0.0)
        p = tl.exp((qk * sm_scale) - lse[:, None])

        # Apply causal mask if needed
        if CAUSAL:
            # Causal mask is aligned to bottom-right
            causal_mask = (offs_m[:, None] >= (offs_n[None, :] + seqlen_q - seqlen_k))
            p = tl.where(causal_mask, p, 0.0)

        # Accumulate dV = P^T @ dO
        # Transpose of p is not needed, dot product takes care of it.
        dv_acc += tl.dot(tl.trans(p.to(do.dtype)), do)

        # Compute dS = P * (dO @ V^T - Delta)
        delta = tl.load(Delta_ptr + offs_m * stride_deltam, mask=mask_m, other=0.0)
        dp = tl.dot(do, tl.trans(v_tile))
        ds = p * (dp - delta[:, None])

        # Accumulate dK = dS^T @ Q
        dk_acc += tl.dot(tl.trans(ds.to(q.dtype)), q)

    return dk_acc, dv_acc

@triton.jit
def _bwd_dq_inner(
    dq_acc, q_tile, K_ptr, V_ptr, do_tile, lse_tile, Delta_ptr, sm_scale,
    stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
    seqlen_q, seqlen_k,
    start_m,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # This CTA is responsible for a single BLOCK_M of Q
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Loop over K/V in blocks of BLOCK_N
    for start_n in range(0, seqlen_k, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen_k
        
        # Load K and V blocks
        k_ptrs = K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Recompute P = exp(QK^T * sm_scale - LSE)
        qk = tl.dot(q_tile, tl.trans(k))
        p = tl.exp((qk * sm_scale) - lse_tile)
        
        # Apply causal mask if needed
        if CAUSAL:
            causal_mask = (offs_m[:, None] >= (offs_n[None, :] + seqlen_q - seqlen_k))
            p = tl.where(causal_mask, p, 0.0)
            
        # Compute dS = P * (dO @ V^T - Delta)
        delta = tl.load(Delta_ptr + offs_m * stride_deltam, mask=offs_m < seqlen_q, other=0.0)
        dp = tl.dot(do_tile, tl.trans(v))
        ds = p * (dp - delta[:, None])
        
        # Accumulate dQ = dS @ K
        dq_acc += tl.dot(ds.to(k.dtype), k)

    return dq_acc

# ----------------------------------------------------------------------------
# Main "One-Kernel" Backward Pass
# ----------------------------------------------------------------------------

@triton.jit
def la_bwd_one_kernel(
    # Pointers to Tensors
    Q, K, V, sm_scale, DO, LSE, Delta,
    DQ, DK, DV,
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lseb, stride_lseh, stride_lsem,
    stride_deltab, stride_deltah, stride_deltam,
    # Shape and other configs
    num_q_heads, num_k_heads,
    max_seqlen_q, max_seqlen_k,
    # Meta-parameters
    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, # For dK/dV
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, # For dQ
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # This kernel is parallelized over Batch, Head, and N (context length)
    pid_n = tl.program_id(0) # Block index along K/V sequence
    pid_h = tl.program_id(1) # Head index
    pid_b = tl.program_id(2) # Batch index

    # --------- PHASE 1: Calculate dK and dV ---------
    start_n1 = pid_n * BLOCK_N1
    offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)
    offs_d = tl.arange(0, HEAD_DIM)

    # Pointers to K and V blocks
    k_ptrs = K + pid_b*stride_kb + pid_h*stride_kh + offs_n1[:,None]*stride_kn + offs_d[None,:]*stride_kd
    v_ptrs = V + pid_b*stride_vb + pid_h*stride_vh + offs_n1[:,None]*stride_vn + offs_d[None,:]*stride_vd
    
    # Load K and V for this CTA
    mask_kv = (offs_n1[:, None] < max_seqlen_k) & (offs_d[None, :] < HEAD_DIM)
    k_tile = tl.load(k_ptrs, mask=mask_kv, other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_kv, other=0.0)

    # Pointers to the start of Q, dO, LSE, Delta for this batch and head
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    DO_ptr = DO + pid_b * stride_dob + pid_h * stride_doh
    LSE_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    Delta_ptr = Delta + pid_b * stride_deltab + pid_h * stride_deltah

    # Initialize accumulators
    dk_acc = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    dk_acc, dv_acc = _bwd_dkdv_inner(
        dk_acc, dv_acc, Q_ptr, k_tile, v_tile, DO_ptr, LSE_ptr, Delta_ptr, sm_scale,
        stride_qm, stride_qd, stride_dom, stride_dod, stride_lsem, stride_deltam,
        max_seqlen_q, max_seqlen_k, start_n1,
        BLOCK_M1, BLOCK_N1, HEAD_DIM, CAUSAL=CAUSAL
    )
    
    # Write back dK and dV
    dv_ptrs = DV + pid_b*stride_dvb + pid_h*stride_dvh + offs_n1[:,None]*stride_dvn + offs_d[None,:]*stride_dvd
    dk_ptrs = DK + pid_b*stride_dkb + pid_h*stride_dkh + offs_n1[:,None]*stride_dkn + offs_d[None,:]*stride_dkd
    tl.store(dv_ptrs, dv_acc, mask=mask_kv)
    tl.store(dk_ptrs, dk_acc * sm_scale, mask=mask_kv)

    # --------- PHASE 2: Calculate dQ ---------
    # Repurpose the grid: pid_n is now the block index along Q sequence
    start_m2 = pid_n * BLOCK_M2
    offs_m2 = start_m2 + tl.arange(0, BLOCK_M2)

    # Pointers to Q, dO, LSE, Delta blocks
    q_ptrs = Q_ptr + offs_m2[:,None]*stride_qm + offs_d[None,:]*stride_qd
    do_ptrs = DO_ptr + offs_m2[:,None]*stride_dom + offs_d[None,:]*stride_dod
    lse_ptrs = LSE_ptr + offs_m2*stride_lsem
    
    # Load Q, dO, LSE for this CTA
    mask_q = (offs_m2[:, None] < max_seqlen_q) & (offs_d[None, :] < HEAD_DIM)
    q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
    do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
    lse_tile = tl.load(lse_ptrs, mask=offs_m2 < max_seqlen_q, other=0.0)[:, None]

    # Pointers to the start of K and V
    K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    V_ptr = V + pid_b * stride_vb + pid_h * stride_vh

    # Initialize accumulator
    dq_acc = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

    dq_acc = _bwd_dq_inner(
        dq_acc, q_tile, K_ptr, V_ptr, do_tile, lse_tile, Delta_ptr, sm_scale,
        stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
        max_seqlen_q, max_seqlen_k, start_m2,
        BLOCK_M2, BLOCK_N2, HEAD_DIM, CAUSAL=CAUSAL
    )
    
    # Write back dQ
    dq_ptrs = DQ + pid_b*stride_dqb + pid_h*stride_dqh + offs_m2[:,None]*stride_dqm + offs_d[None,:]*stride_dqd
    tl.store(dq_ptrs, dq_acc, mask=mask_q)


# ----------------------------------------------------------------------------
# Launcher for the "One-Kernel" Backward Pass
# ----------------------------------------------------------------------------
def la_backward_one_kernel_launcher(
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
):
    batch, num_q_heads, max_seqlen_q, head_sz = q.shape
    _, num_k_heads, max_seqlen_k, _ = k.shape

    # Pre-computation of Delta (can also be a small separate kernel)
    delta = torch.sum(o * do, dim=-1)

    # Define optimal block sizes
    config = {
        "BLOCK_M1": 64, "BLOCK_N1": 64,
        "BLOCK_M2": 64, "BLOCK_N2": 64,
    }

    # Grid setup: parallelize over N_k, heads, and batch
    # We repurpose the first dimension of the grid for both N and M blocks
    grid_dim_n = triton.cdiv(max_seqlen_k, config['BLOCK_N1'])
    grid_dim_m = triton.cdiv(max_seqlen_q, config['BLOCK_M2'])
    grid_n = max(grid_dim_m, grid_dim_n)

    grid = (grid_n, num_k_heads, batch)

    # Note: Triton requires tensors to be contiguous or have compatible strides.
    # The B,H,S,D format is generally fine.
    
    la_bwd_one_kernel[grid](
        q, k, v, sm_scale, do, softmax_lse, delta,
        dq, dk, dv,
        # Strides for B, H, S, D layout
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        # Shape and other configs
        num_q_heads, num_k_heads,
        max_seqlen_q, max_seqlen_k,
        # Meta-parameters
        HEAD_DIM=head_sz,
        CAUSAL=causal,
        **config
    )

# # SPDX-License-Identifier: MIT
# # Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# """
# Lean Attention
# ===============
# This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480
# Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.
# Lean Attention is for both decode and prefill attention of transformer based models.

# It currently supports ragged batching decode and prefill attention with causal=1

# TO be added features:
# - Add GQA support
# - batch_size > 1 for prefill/causal=1
# - Misc
#     - N_CTX with non-integer number of BLOCK_N (pad zeros or add mask)
#     -
# """

# import torch

# import triton
# import triton.language as tl

# from .utils.index_max_tiles import calculate_max_output_tiles_analytically


# # Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs
# def persistent_lean_attention(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     Mp: torch.Tensor,
#     Lp: torch.Tensor,
#     Op: torch.Tensor,  # (total_programs, n_ctx_q, d)
#     locks: torch.Tensor,
#     batch_num_block_n: torch.Tensor,
#     total_programs: int,
#     BLOCK_M: int,
#     BLOCK_N: int,
#     causal: bool,
#     batch_size: int,
#     sm_scale: torch.float16,
#     num_warps: int,
#     waves_per_eu: int,
#     n_ctx: list[int],
#     # max_output_tile_cnt: int,
# ):
#     # shape constraints
#     HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
#     assert (
#         HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
#     ), "Incompatible Q/K/V Hidden Dimensions"
#     assert HEAD_DIM_K in {16, 32, 64, 128, 256}

#     # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
#     # For MI300, BLOCK_M=128, BLOCK_N=64 is better for performance
#     MASKED_BLOCKS = BLOCK_M // BLOCK_N

#     if causal:
#         # Only support BLOCK_M is multiple of BLOCK_N
#         # TODO: add other scenarios
#         assert BLOCK_M % BLOCK_N == 0

#     N_CTX_Q = q.shape[0] // batch_size
#     N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
#     H = q.shape[1]

#     qk_scale = sm_scale * 1.44269504

#     max_output_tile_cnt = calculate_max_output_tiles_analytically(
#         causal=causal, batch_size=batch_size, n_ctx=n_ctx, max_seqlen_q=N_CTX_Q,
#         num_heads=H, num_SMs=total_programs, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
#     )

#     (
#         num_m_blocks,
#         num_n_blocks,
#         high_load_wgs,
#         max_tiles_per_wg,
#         tiles_per_head,
#         total_programs,
#         num_splits,
#         even_split,
#     ) = get_num_splits_and_buffer_sizes(
#         causal,
#         batch_size,
#         N_CTX_Q,
#         N_CTX_K,
#         H,
#         H,
#         BLOCK_M,
#         BLOCK_N,
#         total_programs,
#     )
#     # print(
#     #    f"high_load_wgs={high_load_wgs}, max_tiles_per_wg={max_tiles_per_wg}, tiles_per_head={tiles_per_head}"
#     # )
#     # print(
#     #    f"total_programs={total_programs}, num_splits={num_splits}, even_split={even_split}"
#     # )
#     # print(f"num_m_blocks={num_m_blocks}, num_n_blocks={num_n_blocks}")

#     grid = (total_programs, 1, 1)

#     o = torch.empty_like(q, dtype=v.dtype)

#     """
#     kernel_timing = {
#         "attn_fwd": {
#             "start_event": torch.cuda.Event(enable_timing=True),
#             "end_event": torch.cuda.Event(enable_timing=True),
#             "ms": 0,
#             "experiments": 0,
#         },
#     }
#     kernel_timing["attn_fwd"]["start_event"].record()
#     """
#     la_kernel = la_persistent[grid](
#         False,
#         0,
#         q,
#         k,
#         v,
#         qk_scale,
#         Mp,
#         Lp,
#         Op,
#         o,
#         batch_num_block_n,
#         locks,
#         q.stride(0),  # N_CTX_Q
#         q.stride(1),  # H
#         q.stride(2),  # Head_Dim
#         k.stride(0),
#         k.stride(1),
#         k.stride(2),
#         v.stride(0),
#         v.stride(1),
#         v.stride(2),
#         o.stride(0),
#         o.stride(1),
#         o.stride(2),
#         Op.stride(0),  # total_programs
#         Op.stride(1),  # n_ctx_q
#         Op.stride(2),  # head_dim
#         HEAD_DIM=HEAD_DIM_K,
#         BLOCK_M=BLOCK_M,
#         BLOCK_N=BLOCK_N,
#         MASKED_BLOCKS=MASKED_BLOCKS,
#         batch_size=batch_size,
#         causal=causal,
#         num_m_blocks=num_m_blocks,
#         num_n_blocks=num_n_blocks,
#         # leanAttention params
#         high_load_wgs=high_load_wgs,
#         max_tiles_per_wg=max_tiles_per_wg,
#         tiles_per_head=tiles_per_head,
#         num_splits=num_splits,
#         max_output_tile_cnt=max_output_tile_cnt,
#         waves_per_eu=waves_per_eu,
#         num_warps=num_warps,
#         num_stages=1,
#         num_ctas=1,
#     )
#     """
#     kernel_timing["attn_fwd"]["end_event"].record()
#     torch.cuda.synchronize()
#     for k in ["attn_fwd"]:
#         ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
#         kernel_timing[k]["ms"] += ms
#     total_ms = kernel_timing["attn_fwd"]["ms"]
#     """
#     print(f"la kernel {la_kernel.n_regs} registers used, {la_kernel.n_spills} spills")
#     ms = 0
#     return o


# def get_num_splits_and_buffer_sizes(
#     causal,
#     batch_size,
#     max_seqlen_q,
#     max_seqlen_k,
#     num_heads,
#     num_heads_k,
#     BLOCK_M,
#     BLOCK_N,
#     num_SMs,
# ):
#     ##### Lean Atteion: Calculate Splits and Tile Sizes #####
#     ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
#     num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
#     num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

#     # TODO: Support Grouped-Query Attention
#     max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

#     # print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
#     # print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
#     # print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
#     # print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")

#     if max_seqlen_q == 1:
#         causal = False

#     tiles_per_head = 0
#     if causal:
#         # Prefill - Causal
#         for i in range(0, num_m_blocks):
#             tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
#         # Does not support ragged batch for causal.
#         tiles_per_head = tiles_per_head * batch_size
#     else:
#         # Decode or Not Causal
#         tiles_per_head = num_m_blocks * num_n_blocks

#     total_tiles = tiles_per_head * num_heads_k  # Total tiles across all heads

#     # StreamK Lean has as many threadblocks as SMs
#     # This should be a function of tile size and number of scratchpad space
#     # LeanAttention assign 3 CTAs per SM (bounded by LDS size)
#     lean_griddimz = num_SMs  # CTA launch grid
#     # if (total_tiles <= 2 * 2 * num_SMs):
#     #    lean_griddimz = min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks)
#     # else:
#     #    lean_griddimz = min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks)

#     # Max number lean tiles per task block (CTA)
#     max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

#     # Find max number of splits
#     num_splits = 0
#     even_split = False
#     if total_tiles % lean_griddimz == 0:
#         even_split = True
#         num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
#     else:
#         even_split = False
#         num_splits = 1 + (
#             (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
#         )

#     # high_load_tbs is the remainder of total_tile / num_cta
#     high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

#     # Needed for causal. This is (per batch n_ctx) // BLOCK_N
#     num_n_blocks = num_n_blocks // batch_size

#     return (
#         num_m_blocks,
#         num_n_blocks,
#         high_load_tbs,
#         max_tiles_per_tb,
#         tiles_per_head,
#         lean_griddimz,
#         num_splits,
#         even_split,
#     )


# @triton.jit
# def find_group(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
#     total_blocks_processed = 0
#     final_q_block_idx = 0
#     final_task_size = 0
#     final_total_blocks = 0
#     found = False
#     # Iterate through the tasks in the desired ping-pong order
#     for i in range(0, num_m_blocks):
#         # Determine the actual Q block index for the current task in the ping-pong sequence
#         pair_idx = i // 2
#         if (i % 2) == 0:
#             # Even tasks are from the top (e.g., 0, 1, 2...)
#             q_block_idx = pair_idx
#         else:
#             # Odd tasks are from the bottom (e.g., N-1, N-2, ...)
#             q_block_idx = num_m_blocks - 1 - pair_idx

#         # Calculate the size of this task's workload (number of K/V blocks to process)
#         task_size = (q_block_idx + 1) * MASKED_BLOCKS

#         # Check if the global tile `x` falls within this task's range
#         if total_blocks_processed + task_size > x and found == False:
#             # We found it. Return the Q index, the size of its workload, and its starting tile.
#             final_q_block_idx, final_task_size, final_total_blocks = (
#                 q_block_idx,
#                 task_size,
#                 total_blocks_processed,
#             )
#             found = True

#         # Add this task's size to the running total and move to the next
#         total_blocks_processed += task_size
#     # Return values
#     return final_q_block_idx, final_task_size, final_total_blocks


# @triton.jit
# def la_persistent(
#     is_pod,
#     pod_pid,
#     Q,
#     K,
#     V,
#     qk_scale,
#     Mp,
#     Lp,
#     Op,
#     Out,
#     batch_num_block_n,
#     locks,
#     stride_qm,  # n_ctx_q
#     stride_qh,  # Head
#     stride_qk,  # head_dim
#     stride_kn,
#     stride_kh,
#     stride_kk,
#     stride_vn,
#     stride_vh,
#     stride_vk,
#     stride_om,  # n_ctx_q
#     stride_oh,  # Head
#     stride_on,  # head_dim
#     stride_oph,  # total_programs
#     stride_opm,  # n_ctx_q
#     stride_opn,  # head_dim
#     HEAD_DIM: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     MASKED_BLOCKS: tl.constexpr,
#     batch_size: tl.constexpr,
#     causal: tl.constexpr,
#     num_m_blocks: tl.constexpr,
#     num_n_blocks: tl.constexpr,
#     # leanAttention params
#     high_load_wgs: tl.constexpr,
#     max_tiles_per_wg: tl.constexpr,
#     tiles_per_head: tl.constexpr,
#     num_splits: tl.constexpr,
#     max_output_tile_cnt: tl.constexpr,
# ):
#     if is_pod:
#         current_pid = pod_pid
#     else:
#         current_pid = tl.program_id(0)

#     if current_pid < high_load_wgs:
#         iter = max_tiles_per_wg * current_pid
#         cta_end_tile_gid = iter + max_tiles_per_wg
#     else:
#         iter = (max_tiles_per_wg - 1) * (
#             current_pid - high_load_wgs
#         ) + high_load_wgs * max_tiles_per_wg
#         cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

#     for i in tl.static_range(max_output_tile_cnt + 1):
#         if iter < cta_end_tile_gid:
#             iter = la_persistent_inner(
#                 Q,
#                 K,
#                 V,
#                 qk_scale,
#                 Mp,
#                 Lp,
#                 Op,
#                 Out,
#                 batch_num_block_n,
#                 locks,
#                 stride_qm,  # n_ctx_q
#                 stride_qh,  # Head
#                 stride_qk,  # head_dim
#                 stride_kn,
#                 stride_kh,
#                 stride_kk,
#                 stride_vn,
#                 stride_vh,
#                 stride_vk,
#                 stride_om,  # n_ctx_q
#                 stride_oh,  # Head
#                 stride_on,  # head_dim
#                 stride_oph,  # total_programs
#                 stride_opm,  # n_ctx_q
#                 stride_opn,  # head_dim
#                 iter=iter,
#                 cta_end_tile_gid=cta_end_tile_gid,
#                 current_pid=current_pid,
#                 HEAD_DIM=HEAD_DIM,
#                 BLOCK_M=BLOCK_M,
#                 BLOCK_N=BLOCK_N,
#                 MASKED_BLOCKS=MASKED_BLOCKS,
#                 batch_size=batch_size,
#                 causal=causal,
#                 num_m_blocks=num_m_blocks,
#                 num_n_blocks=num_n_blocks,
#                 # leanAttention params
#                 high_load_wgs=high_load_wgs,
#                 max_tiles_per_wg=max_tiles_per_wg,
#                 tiles_per_head=tiles_per_head,
#                 num_splits=num_splits,
#             )


# @triton.jit
# def la_persistent_inner(
#     Q,
#     K,
#     V,
#     qk_scale,
#     Mp,
#     Lp,
#     Op,
#     Out,
#     batch_num_block_n,
#     locks,
#     stride_qm,  # n_ctx_q
#     stride_qh,  # Head
#     stride_qk,  # head_dim
#     stride_kn,
#     stride_kh,
#     stride_kk,
#     stride_vn,
#     stride_vh,
#     stride_vk,
#     stride_om,  # n_ctx_q
#     stride_oh,  # Head
#     stride_on,  # head_dim
#     stride_oph,  # total_programs
#     stride_opm,  # n_ctx_q
#     stride_opn,  # head_dim
#     iter,
#     cta_end_tile_gid,
#     current_pid,
#     HEAD_DIM: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     MASKED_BLOCKS: tl.constexpr,
#     batch_size: tl.constexpr,
#     causal: tl.constexpr,
#     num_m_blocks: tl.constexpr,
#     num_n_blocks: tl.constexpr,
#     # leanAttention params
#     high_load_wgs: tl.constexpr,
#     max_tiles_per_wg: tl.constexpr,
#     tiles_per_head: tl.constexpr,
#     num_splits: tl.constexpr,
# ):
#     """
#     if is_pod:
#         current_pid = pod_pid
#     else:
#         current_pid = tl.program_id(0)

#     if current_pid < high_load_wgs:
#         iter = max_tiles_per_wg * current_pid
#         cta_end_tile_gid = iter + max_tiles_per_wg
#     else:
#         iter = (max_tiles_per_wg - 1) * (
#             current_pid - high_load_wgs
#         ) + high_load_wgs * max_tiles_per_wg
#         cta_end_tile_gid = iter + (max_tiles_per_wg - 1)
#     """
#     # Loop context length
#     # while iter < cta_end_tile_gid:
#     # Calculate index of current head output tile
#     # The tiles_per_head is the sum of # BLOCK_N in K/V sequence of all batches
#     tile_head_idx = iter // tiles_per_head

#     # To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
#     # [tile_iter, tile_iter_end) are in the form of global tile id
#     if causal:
#         tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
#         # Does not support ragged batching. All requests in the batch have the same context length (per_head_tile_size)
#         # tiles_per_head: total sum of # BLOCK_N in K/V sequence of all batches
#         # per_head_tile_size: per head # BLOCK_N of each output tile
#         per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
#             iter
#             - (tile_head_idx * tiles_per_head)
#             - (tile_batch_idx * (tiles_per_head // batch_size)),
#             MASKED_BLOCKS,
#             num_m_blocks,
#         )
#         tile_iter = (
#             tile_head_idx * tiles_per_head
#             + (tile_batch_idx * (tiles_per_head // batch_size))
#             + total_blocks
#         )
#         tile_iter_end = tile_iter + (per_head_tile_size)
#         tile_idx = (
#             tile_head_idx * batch_size + tile_batch_idx
#         ) * num_m_blocks + per_head_tile_idx
#     else:
#         tile_idx = (
#             tile_head_idx * batch_size
#         )  # Output tile idx, 1 output tile per head per batch
#         tile_iter = tile_head_idx * tiles_per_head
#         if batch_size == 1:
#             req_size = tiles_per_head
#         else:
#             req_size = tl.load(batch_num_block_n)
#         tile_iter_end = tile_iter + req_size
#         for b in range(1, batch_size):
#             next_req_size = tl.load(batch_num_block_n + b)
#             local_head_iter = iter % tiles_per_head
#             if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
#                 tile_iter = tile_iter + req_size
#                 tile_idx = tile_idx + b
#                 tile_iter_end = tile_iter + (next_req_size - req_size)
#             req_size = next_req_size
#     # Local lean tile ID within a loop of an output tile
#     local_iter = iter - tile_iter

#     host_block = iter == tile_iter
#     stole_last_tile = cta_end_tile_gid == (tile_iter_end - 1)

#     # Determine the number of tiles this WG will process for the current output tile.
#     local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter
#     if stole_last_tile:
#         local_iter_end += 1

#     # 'host_is_finishing' tells the host if it's the sole contributor for this tile,
#     # which means it doesn't need to perform a reduction with other WGs.
#     host_is_finishing = (cta_end_tile_gid >= tile_iter_end) or (
#         stole_last_tile and host_block
#     )

#     # if iter == tile_iter:
#     #     host_block = True
#     # else:
#     #     host_block = False
#     # # finishing_block: the output tile is finished within this block
#     # if cta_end_tile_gid >= tile_iter_end:
#     #     finishing_block = True
#     # else:
#     #     finishing_block = False

#     offs_m = tl.arange(0, BLOCK_M)
#     offs_n = tl.arange(0, BLOCK_N)
#     offs_k = tl.arange(0, HEAD_DIM)

#     if causal:
#         b_seq_size = tile_batch_idx * num_n_blocks
#     else:
#         tile_batch_idx = tile_idx % batch_size
#         b_seq_size = 0
#         if tile_batch_idx > 0:
#             b_seq_size = tl.load(
#                 batch_num_block_n + tile_batch_idx - 1
#             )  # Previous batch size

#     k_offs = (
#         (b_seq_size + local_iter) * BLOCK_N * stride_kn
#         + tile_head_idx * stride_kh
#         + offs_n[None, :] * stride_kn
#         + offs_k[:, None] * stride_kk
#     )
#     v_offs = (
#         (b_seq_size + local_iter) * BLOCK_N * stride_vn
#         + tile_head_idx * stride_vh
#         + offs_n[:, None] * stride_vn
#         + offs_k[None, :] * stride_vk
#     )

#     k_ptrs = K + k_offs
#     k_ptrs = tl.multiple_of(k_ptrs, (16, 1))
#     v_ptrs = V + v_offs
#     v_ptrs = tl.multiple_of(v_ptrs, (1, 16))

#     if causal:
#         q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
#     else:
#         q_idx = tile_batch_idx
#     q_offs = (
#         q_idx * BLOCK_M * stride_qm
#         + tile_head_idx * stride_qh
#         + offs_m[:, None] * stride_qm
#         + offs_k[None, :] * stride_qk
#     )
#     q_ptrs = Q + q_offs
#     q_ptrs = tl.multiple_of(q_ptrs, (1, 16))

#     if causal:
#         q_start_m = q_idx * BLOCK_M

#     m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
#     l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
#     acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

#     q = tl.load(q_ptrs)

#     for l_iter in range(local_iter, local_iter_end):
#         # -- compute qk ----
#         # k = tl.load(k_ptrs, cache_modifier=".cg")
#         k = tl.load(k_ptrs)
#         qk = tl.dot(q, k)
#         qk = qk * qk_scale

#         # Apply the causal mask
#         #    qk = tl.where(mask, qk, float("-inf"))

#         if causal:
#             # Get the starting column index of the current K block
#             k_start_n = (b_seq_size + l_iter) * BLOCK_N
#             # Create mask based on absolute sequence positions
#             mask = (q_start_m + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
#             # Apply the mask
#             qk = tl.where(mask, qk, float("-inf"))

#         m_ij = tl.maximum(m_i, tl.max(qk, 1))

#         # Prevent NaNs in p calculation
#         p_arg = qk - m_ij[:, None]
#         # If m_ij is -inf, the subtraction results in NaN. Force the argument to -inf instead.
#         p_arg = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), p_arg)
#         p = tl.math.exp2(p_arg)

#         # Prevent NaNs in alpha calculation
#         # If m_i and m_ij are both -inf, the subtraction is NaN. Force the argument to 0, so alpha=1.
#         alpha_arg = m_i - m_ij
#         alpha_arg = tl.where(m_ij == float("-inf"), 0.0, alpha_arg)
#         alpha = tl.math.exp2(alpha_arg)
#         acc = (
#             acc * alpha[:, None]
#         )  # Scale each row of acc by the corresponding elements in alpha
#         # v = tl.load(v_ptrs, cache_modifier=".cg")  # v.shape = [BLOCK_N, HEAD_DIM]
#         v = tl.load(v_ptrs)
#         acc += tl.dot(p.to(v.dtype), v)  # acc.shape = [BLOCK_M, HEAD_DIM]
#         # -- update l_i
#         l_ij = tl.sum(p, 1)  # rowsum(p)
#         l_i = l_i * alpha + l_ij
#         # update m_i
#         m_i = m_ij.to(m_i.dtype)
#         # if (
#         #     (l_iter == (tile_iter_end - tile_iter) - 1)
#         #     and (iter == tile_iter_end - 1)
#         #     and (MASKED_BLOCKS == 2)
#         # ):
#         #     mask1 = offs_m >= BLOCK_N
#         #     m_i = tl.where(mask1, m_i, float("-inf"))
#         #     l_i = tl.where(mask1, l_i, 1.0)
#         #     mask1 = mask1[:, None]
#         #     acc = tl.where(mask1, acc, 0.0)
#         # update k/v pointer
#         v_ptrs += BLOCK_N * stride_vn
#         k_ptrs += BLOCK_N * stride_kn

#     # initialize pointer to m and l
#     m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
#     l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
#     # acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

#     # lean output tile epilogue
#     if not host_block:
#         # Update pointers of partial results Mp[cta], Lp[cta], Op[cta]
#         mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
#         lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
#         op_ptrs = (
#             Op
#             + current_pid * stride_oph  # stride_oph is total_program dimension
#             + offs_m[:, None] * stride_opm
#             + offs_k[None, :] * stride_opn
#         )

#         tl.store(mp_ptrs, m_i, cache_modifier=".wt")
#         tl.store(lp_ptrs, l_i, cache_modifier=".wt")
#         tl.store(op_ptrs, acc, cache_modifier=".wt")
#         tl.debug_barrier()
#         # tl.store(locks + current_pid, 1, cache_modifier=".wt")
#         # According to streamK gemm, store + cache_modifier won't work universally
#         # atomic_xchg is better solution but a less performant variant
#         tl.atomic_xchg(locks + current_pid, 1)

#     if host_block:  # and finishing_block:
#         # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
#         # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
#         acc_reshaped = tl.reshape(acc, (BLOCK_M, 2, HEAD_DIM // 2))
#         acc_permuted = tl.permute(acc_reshaped, (0, 2, 1))
#         acc0, acc1 = tl.split(acc_permuted)

#         # o_h_offs = (
#         #     q_idx * BLOCK_M * stride_om
#         #     + tile_head_idx * stride_oh
#         #     + offs_m[:, None] * stride_om
#         #     + offs_k[None, :] * stride_on
#         # )
#         # o_ptrs = Out + o_h_offs

#         if not host_is_finishing:
#             # if host not finishing_block: # another CTA is processing the end of the output tile and store partial results

#             last_cta = current_pid + 1
#             temp_end_gid = cta_end_tile_gid
#             split = 1
#             while (split < num_splits) and (temp_end_gid < tile_iter_end):
#                 if last_cta < high_load_wgs:
#                     if (tile_iter_end - temp_end_gid) < max_tiles_per_wg:
#                         temp_end_gid += tile_iter_end - temp_end_gid
#                     else:
#                         temp_end_gid += max_tiles_per_wg
#                 else:
#                     if (tile_iter_end - temp_end_gid) < (max_tiles_per_wg - 1):
#                         temp_end_gid += tile_iter_end - temp_end_gid
#                     else:
#                         temp_end_gid += max_tiles_per_wg - 1

#                 last_cta += 1
#                 split += 1
#             # Next, load nonHost partial restult
#             for cta in range((current_pid + 1), last_cta):
#                 # According to treamK gemm, atomic_cas is universal solution but less performant
#                 while tl.atomic_cas(locks + cta, 1, 1) != 1:
#                     # while tl.load(locks + cta, cache_modifier=".cv", volatile=True) != 1:
#                     pass

#                 # Partial results are stored in [nonHost, Host-nonFinishing] layout
#                 offs_mplp = cta * BLOCK_M + offs_m
#                 mp_ptrs = Mp + offs_mplp
#                 lp_ptrs = Lp + offs_mplp
#                 """
#                 op_h_offs = (
#                     cta * stride_oph
#                     + offs_m[:, None] * stride_opm
#                     + offs_k[None, :] * stride_opn
#                 )
#                 op_ptrs = Op + op_h_offs
#                 """
#                 op_ptrs0 = (
#                     Op
#                     + cta * stride_oph
#                     + offs_m[:, None] * stride_opm
#                     + tl.arange(0, HEAD_DIM // 2)[None, :] * stride_opn
#                 )
#                 op_ptrs1 = (
#                     Op
#                     + cta * stride_oph
#                     + offs_m[:, None] * stride_opm
#                     + (tl.arange(0, HEAD_DIM // 2)[None, :] + HEAD_DIM // 2)
#                     * stride_opn
#                 )

#                 m_cta = tl.load(mp_ptrs)
#                 l_cta = tl.load(lp_ptrs)
#                 # acc_cta = tl.load(op_ptrs)
#                 acc_cta0 = tl.load(op_ptrs0)
#                 acc_cta1 = tl.load(op_ptrs1)

#                 # m_i is the host CTA's m, m_cta is other nonHost CTA's m
#                 m_new = tl.maximum(m_cta, m_i)
#                 alpha = tl.math.exp2(m_cta - m_new)
#                 alpha1 = tl.math.exp2(m_i - m_new)
#                 l_new = alpha * l_cta + alpha1 * l_i
#                 # acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
#                 acc0 = acc_cta0 * alpha[:, None] + acc0 * alpha1[:, None]
#                 acc1 = acc_cta1 * alpha[:, None] + acc1 * alpha1[:, None]
#                 # update m, l
#                 m_i = m_new
#                 l_i = l_new

#         # host CTA write final result to memory
#         # acc = acc / l_i[:, None]
#         # tl.store(o_ptrs, acc.to(Out.type.element_ty))
#         o_ptrs0 = (
#             Out
#             + q_idx * BLOCK_M * stride_om
#             + tile_head_idx * stride_oh
#             + offs_m[:, None] * stride_om
#             + tl.arange(0, HEAD_DIM // 2)[None, :] * stride_on
#         )
#         o_ptrs1 = (
#             Out
#             + q_idx * BLOCK_M * stride_om
#             + tile_head_idx * stride_oh
#             + offs_m[:, None] * stride_om
#             + (tl.arange(0, HEAD_DIM // 2)[None, :] + HEAD_DIM // 2) * stride_on
#         )

#         acc0 = acc0 / l_i[:, None]
#         acc1 = acc1 / l_i[:, None]
#         tl.store(o_ptrs0, acc0.to(Out.type.element_ty))
#         tl.store(o_ptrs1, acc1.to(Out.type.element_ty))

#     # update iter
#     iter = iter + (local_iter_end - local_iter)

#     return iter