# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Part 2: The "Tracer" Data Generator and Kernel Runner

This script runs a modified MHA kernel that, instead of computing attention,
writes its work-group ID into the output tensor. This creates a "trace"
of which physical hardware unit processed which logical part of the output,
allowing us to verify the swizzling pattern in practice.
"""

import torch
import triton
import triton.language as tl
from typing import Dict, Any

# Assuming the original mha.py file is in aiter/ops/triton/
# We need the swizzle function from it.
from aiter.ops.triton.mha import swizzle_mha_wid_balanced

# =================================================================
#                   TRACER TRITON KERNEL
# =================================================================

@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def _attn_tracer_kernel(
    out_ptr,
    stride_oz, stride_oh, stride_om, stride_on,
    SEQLEN_Q: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_XCD: tl.constexpr,
):
    """
    A modified kernel that does not compute attention.
    Instead, it writes the program_id (work-group ID) into the output tensor
    at the location this work-group is responsible for.
    """
    # Decompose the work-item ID using the swizzle function
    NUM_BLOCKS = _cdiv_fn(SEQLEN_Q, BLOCK_M)
    original_wid = tl.program_id(0)
    
    # Apply the same swizzling used in the real kernel
    swizzled_wid = swizzle_mha_wid_balanced(original_wid, NUM_Q_HEADS, NUM_BLOCKS, NUM_XCD)
    
    # Decompose the swizzled ID to find the logical (head, block)
    start_m_block = swizzled_wid % NUM_BLOCKS
    off_q_head = (swizzled_wid // NUM_BLOCKS) % NUM_Q_HEADS
    off_z = swizzled_wid // (NUM_Q_HEADS * NUM_BLOCKS)

    # Calculate output pointers
    offs_m = start_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    out_offs = (off_z * stride_oz + 
                off_q_head * stride_oh + 
                offs_m[:, None] * stride_om + 
                offs_d[None, :] * stride_on)
    
    # Create a mask to avoid writing out of bounds
    mask = (offs_m[:, None] < SEQLEN_Q) & (offs_d[None, :] < BLOCK_DMODEL)

    # Write the original work-group ID to the output tensor
    # We cast the ID to the output tensor's dtype
    tracer_val = original_wid.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_offs, tracer_val, mask=mask)


# =================================================================
#                   PYTHON DISPATCHER FOR TRACER
# =================================================================

def flash_attn_tracer_func(mha_config: Dict[str, Any]) -> torch.Tensor:
    """
    Dispatcher that sets up tensors and launches the tracer kernel.
    It returns a tensor where each element's value is the ID of the
    work-group that wrote it.
    """
    # Extract config
    BATCH = mha_config['BATCH']
    SEQLEN_Q = mha_config['SEQLEN_Q']
    NUM_Q_HEADS = mha_config['NUM_Q_HEADS']
    HEAD_SZ = mha_config['HEAD_SZ']
    BLOCK_M = mha_config['BLOCK_M']
    NUM_XCD = mha_config['NUM_XCDS']

    # Create an empty output tensor. The input tensors (Q,K,V) are not needed.
    o_tracer = torch.zeros((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=torch.float16)

    # Get strides for the output tensor
    o_strides = (o_tracer.stride(0), o_tracer.stride(2), o_tracer.stride(1), o_tracer.stride(3))

    # Calculate grid size. This MUST match the grid size of the real kernel.
    grid = lambda META: (BATCH * NUM_Q_HEADS * triton.cdiv(SEQLEN_Q, META["BLOCK_M"]),)
    
    # Launch the tracer kernel
    _attn_tracer_kernel[grid](
        o_tracer,
        *o_strides,
        SEQLEN_Q=SEQLEN_Q,
        NUM_Q_HEADS=NUM_Q_HEADS,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=HEAD_SZ,
        NUM_XCD=NUM_XCD,
        num_warps=4,
        num_stages=1
    )
    
    return o_tracer


if __name__ == '__main__':
    # --- Example Usage ---
    mha_config_example = {
        "BATCH": 1,
        "SEQLEN_Q": 1024,
        "NUM_Q_HEADS": 10,
        "HEAD_SZ": 128,
        "BLOCK_M": 128,
        "NUM_XCDS": 8,
    }

    print("Running tracer kernel to get actual work-group mapping...")
    ground_truth_map = flash_attn_tracer_func(mha_config_example)
    
    print("\nTracer function executed successfully.")
    print(f"Output tensor shape: {ground_truth_map.shape}")
    
    # Extract the map for the first head and batch to inspect
    # Since all values in a block are the same, we just take the first element.
    # The result is a (SEQLEN_Q) tensor where each value is the work-group ID.
    first_head_trace = ground_truth_map[0, :, 0, 0]
    print(f"\nTrace for Head 0 (first {mha_config_example['BLOCK_M']} tokens):")
    print(first_head_trace[:mha_config_example['BLOCK_M']].cpu().numpy())

    # The values should be constant for the entire block, as one work-group
    # handles one block.
    assert torch.all(first_head_trace[:BLOCK_M] == first_head_trace[0])
    print("\nVerified that the first block was written by a single work-group.")
