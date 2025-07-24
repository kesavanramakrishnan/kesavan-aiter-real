# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import triton
import triton.language as tl

# Import the backward pass implementation from its source file
# NOTE: Ensure the file 'lean_attn_bwd_full.py' is in the same directory or accessible
from aiter.ops.triton.lean_atten_bwd_proto import persistent_lean_attention_backward

# ==============================================================================
# Reference PyTorch Implementation
# ==============================================================================

def attention_ref(q, k, v, causal=False, sm_scale=1.0):
    """
    A simple, direct implementation of attention in PyTorch to serve as a
    ground truth reference. This version now returns the raw scores needed
    for the LSE calculation.
    """
    # Reshape to have heads and sequence length in the right dimensions
    q_torch = q.transpose(1, 2)
    k_torch = k.transpose(1, 2)
    v_torch = v.transpose(1, 2)
    
    # Scale Q
    q_torch = q_torch * sm_scale
    
    # Matmul Q and K
    scores = torch.matmul(q_torch, k_torch.transpose(-2, -1))
    
    # Apply causal mask if needed
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], device=q.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float('-inf'))
        
    # Softmax
    probs = torch.softmax(scores, dim=-1)
    
    # Matmul with V
    output = torch.matmul(probs, v_torch)
    
    # Transpose back to original shape
    output = output.transpose(1, 2)
    
    return output, scores


# ==============================================================================
# Pytest Test Script
# ==============================================================================

@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(128, 128), (64, 128)])
@pytest.mark.parametrize("NUM_HEADS", [16, 32])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_lean_attention_unified_bwd(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    # --- Test Setup ---
    # Create random tensors for q, k, v and the upstream gradient do
    # For non-GQA, NUM_K_HEADS is the same as NUM_Q_HEADS
    NUM_K_HEADS = NUM_HEADS
    q = torch.randn((BATCH, SEQLEN_Q, NUM_HEADS, HEAD_SZ), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype, requires_grad=True)
    do = torch.randn_like(q)

    sm_scale = 1.0 / (HEAD_SZ ** 0.5)

    # --- Reference Calculation (PyTorch) ---
    # 1. Perform the forward pass with the reference implementation
    torch_out, torch_scores = attention_ref(q, k, v, causal=CAUSAL, sm_scale=sm_scale)
    
    # 2. Compute gradients using PyTorch's autograd
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    # --- Triton Calculation ---
    # 1. Get intermediate tensors (out, lse) from the reference forward pass
    triton_fwd_out = torch_out.detach().clone()
    # Correctly calculate LSE from raw scores
    triton_lse = torch.logsumexp(torch_scores, dim=-1).transpose(1,2)

    # 2. Initialize output tensors for Triton gradients
    triton_dq = torch.empty_like(q)
    triton_dk = torch.empty_like(k)
    triton_dv = torch.empty_like(v)
    
    # 3. Call the Lean Attention backward function
    try:
        persistent_lean_attention_backward(
            q, k, v, triton_fwd_out, do, triton_lse,
            triton_dq, triton_dk, triton_dv,
            total_programs=108, # Example for A100
            BLOCK_M=64,
            BLOCK_N=64,
            causal=CAUSAL,
            sm_scale=sm_scale,
            num_warps=4,
            waves_per_eu=2
        )
    except Exception as e:
        # This might fail if the Triton kernel is not compiled/runnable
        print(f"Kernel launch failed: {e}")
        pytest.fail("Triton kernel execution failed.")

    # --- Comparison ---
    # Compare the results from Triton and PyTorch
    torch.testing.assert_close(triton_dq, torch_dq, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(triton_dk, torch_dk, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(triton_dv, torch_dv, atol=1e-2, rtol=1e-2)

