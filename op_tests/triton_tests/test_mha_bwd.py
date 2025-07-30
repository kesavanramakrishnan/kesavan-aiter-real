# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import triton
import triton.language as tl
from aiter.ops.triton.mha import flash_attn_func
from aiter.test_mha_common import attention_ref

# Assuming the new dkdv kernel and its helpers are in this file
from aiter.ops.triton.lean_atten_bwd_updated_proto import (
    _bwd_preprocess,
    la_bwd_dkdv_persistent,
    get_num_splits_and_buffer_sizes,
)

def lean_flash_attn_onekernel_backward(
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
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
    """
    Wrapper for the Lean Attention backward pass dk/dv kernel.
    """
    batch_size, _, num_q_heads, head_dim = q.shape
    _, _, num_k_heads, _ = k.shape

    # Reshape tensors for the kernel: (B, S, H, D) -> (B*H, S, D)
    q_reshaped = q.transpose(1, 2).contiguous().view(batch_size * num_q_heads, max_seqlen_q, head_dim)
    k_reshaped = k.transpose(1, 2).contiguous().view(batch_size * num_k_heads, max_seqlen_k, head_dim)
    v_reshaped = v.transpose(1, 2).contiguous().view(batch_size * num_k_heads, max_seqlen_k, head_dim)
    do_reshaped = do.transpose(1, 2).contiguous().view(batch_size * num_q_heads, max_seqlen_q, head_dim)
    o_reshaped = o.transpose(1, 2).contiguous().view(batch_size * num_q_heads, max_seqlen_q, head_dim)
    
    dk_reshaped = dk.transpose(1, 2).contiguous().view(batch_size * num_k_heads, max_seqlen_k, head_dim)
    dv_reshaped = dv.transpose(1, 2).contiguous().view(batch_size * num_k_heads, max_seqlen_k, head_dim)
    
    lse_reshaped = softmax_lse.view(batch_size * num_q_heads, max_seqlen_q)
    
    # Pre-computation of D = rowsum(dO * O)
    delta = torch.empty_like(lse_reshaped)
    pre_block_m = 128 # A common block size for preprocessing
    pre_grid = (triton.cdiv(max_seqlen_q, pre_block_m), batch_size * num_q_heads)
    
    _bwd_preprocess[pre_grid](
        o_reshaped, do_reshaped, delta,
        o_reshaped.stride(0), o_reshaped.stride(1), o_reshaped.stride(2),
        delta.stride(0), delta.stride(1),
        seqlen_q=max_seqlen_q,
        BLOCK_M=pre_block_m,
        BLOCK_DMODEL=head_dim,
    )
    
    # Get Lean Attention schedule
    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        grid_size,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        causal, batch_size, max_seqlen_q, max_seqlen_k,
        num_q_heads, num_k_heads, BLOCK_M, BLOCK_N, total_programs,
    )
    
    # Allocate tensors for partial results and synchronization
    dKp = torch.empty((grid_size, max_seqlen_k, head_dim), device='cuda', dtype=torch.float32)
    dVp = torch.empty((grid_size, max_seqlen_k, head_dim), device='cuda', dtype=torch.float32)
    locks = torch.zeros(grid_size, dtype=torch.int32, device='cuda')
    
    max_output_tile_cnt = max_tiles_per_wg

    # Launch the dk/dv kernel
    la_bwd_dkdv_persistent[(grid_size,)](
        dk_reshaped, dv_reshaped, q_reshaped, k_reshaped, v_reshaped, do_reshaped, lse_reshaped, delta,
        dKp, dVp, locks,
        dk_reshaped.stride(0), dk_reshaped.stride(1), dk_reshaped.stride(2),
        dv_reshaped.stride(0), dv_reshaped.stride(1), dv_reshaped.stride(2),
        q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2),
        k_reshaped.stride(0), k_reshaped.stride(1), k_reshaped.stride(2),
        v_reshaped.stride(0), v_reshaped.stride(1), v_reshaped.stride(2),
        do_reshaped.stride(0), do_reshaped.stride(1), do_reshaped.stride(2),
        lse_reshaped.stride(0), lse_reshaped.stride(1),
        delta.stride(0), delta.stride(1),
        dKp.stride(0), dKp.stride(1), dKp.stride(2),
        dVp.stride(0), dVp.stride(1), dVp.stride(2),
        sm_scale,
        max_seqlen_q, max_seqlen_k,
        high_load_wgs, max_tiles_per_wg, tiles_per_head, num_m_blocks, num_splits, max_output_tile_cnt,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    # Reshape back to original (B, S, H, D)
    dk_reshaped = dk_reshaped.view(batch_size, num_k_heads, max_seqlen_k, head_dim).transpose(1, 2).contiguous()
    dv_reshaped = dv_reshaped.view(batch_size, num_k_heads, max_seqlen_k, head_dim).transpose(1, 2).contiguous()
    dk.copy_(dk_reshaped)
    dv.copy_(dv_reshaped)

@pytest.mark.parametrize(
    "causal, batch, h, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, waves_per_eu, num_warps ",
    [
        (False, 2, 64, 16, [65536, 65536], 128, 912, torch.float16, 16, 128, 3, 4),
        (False, 1, 64, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 64, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 64, 16, [524288], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 2, 96, 16, [32768, 32768], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 96, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 96, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 96, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 96, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),
        (False, 1, 96, 16, [1048576], 16, 912, torch.float16, 16, 256, 1, 4),
        (False, 1, 128, 16, [32768], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 128, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 128, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 128, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 128, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),
        (
            False, 3, 64, 16, [4096, 32768, 65536], 128, 912,
            torch.float16, 16, 128, 2, 4,
        ),
        (
            False, 8, 64, 16, [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536], 128, 912,
            torch.float16, 16, 128, 2, 4,
        ),
        (
            True, 1, 64, 8192, [8192], 128, 304,
            torch.float16, 128, 64, 1, 4,
        ),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 64, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 16, 16, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 32, 16, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 32, 32, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 64, 16, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 64, 32, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 64, 64, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 16, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 32, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 64, 1, 4),
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 128, 1, 4),
    ],
)
def test_lean_bwd_dkdv_persistent(
    causal, batch, h, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, waves_per_eu, num_warps
):
    torch.manual_seed(20)
    torch.cuda.empty_cache()

    sum_n_ctx = sum(n_ctx)
    
    # For simplicity, this test uses MHA (num_q_heads == num_k_heads)
    num_q_heads = h
    num_k_heads = h

    q = torch.randn((batch, n_ctx_q, num_q_heads, d), device="cuda", dtype=init_dtype, requires_grad=True)
    k = torch.randn((batch, sum_n_ctx, num_k_heads, d), device="cuda", dtype=init_dtype, requires_grad=True)
    v = torch.randn((batch, sum_n_ctx, num_k_heads, d), device="cuda", dtype=init_dtype, requires_grad=True)
    do = torch.randn_like(q)
    
    sm_scale = 1.0 / (d**0.5)

    # --- Forward Pass ---
    with torch.enable_grad():
        triton_out, lse, _ = flash_attn_func(
            q, k, v, dropout_p=0.0, causal=causal, return_lse=True, return_attn_probs=True
        )

    # --- Triton Lean BWD Pass ---
    triton_dq_lean = torch.empty_like(q)
    triton_dk_lean = torch.empty_like(k)
    triton_dv_lean = torch.empty_like(v)
    
    lean_flash_attn_onekernel_backward(
        do, q, k, v, triton_out, lse, 
        triton_dq_lean, triton_dk_lean, triton_dv_lean,
        sm_scale, causal, n_ctx_q, sum_n_ctx, 
        total_programs, BLOCK_M, BLOCK_N
    )

    # --- PyTorch Reference BWD Pass ---
    with torch.enable_grad():
        torch_out, _ = attention_ref(q, k, v, causal=causal)
    
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    # --- Comparison ---
    torch.testing.assert_close(triton_out, torch_out.to(triton_out.dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(triton_dk_lean, torch_dk.to(triton_out.dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(triton_dv_lean, torch_dv.to(triton_out.dtype), atol=1e-2, rtol=1e-2)
