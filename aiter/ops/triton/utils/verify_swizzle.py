# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Part 3: The Verification and Comparison Script

This script orchestrates the verification process. It:
1. Defines a specific MHA configuration to test.
2. Uses `swizzle_visualizer.py` to plot the THEORETICAL mapping.
3. Uses `mha_tracer.py` to run a kernel and get the ACTUAL mapping.
4. Plots the actual mapping for a side-by-side comparison, allowing for
   a visual confirmation that the swizzling works as designed.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from typing import Dict, Any

# Import the necessary functions from our other scripts
from swizzle_visualizer import (
    visualize_theoretical_mapping, 
    swizzle_mha_wid_balanced_python, 
    get_xcd_id_for_head_balanced,
    cdiv,
    get_text_color
)
from mha_tracer import flash_attn_tracer_func

# =================================================================
#               GROUND-TRUTH VISUALIZATION FUNCTION
# =================================================================

def visualize_traced_mapping(
    mha_config: Dict[str, Any],
    traced_output: torch.Tensor,
    get_xcd_id_fn: Callable[[int, Dict[str, Any]], int],
    title: str = "Actual (Traced) Swizzle Pattern"
):
    """
    Visualizes the actual mapping derived from the tracer kernel's output.
    """
    num_heads = mha_config['NUM_Q_HEADS']
    num_xcds = mha_config['NUM_XCDS']
    block_m = mha_config['BLOCK_M']
    num_blocks = cdiv(mha_config['SEQLEN_Q'], block_m)
    mha_config['NUM_BLOCKS'] = num_blocks

    if num_blocks == 0 or num_heads == 0:
        print("Warning: Cannot visualize with 0 blocks or 0 heads.")
        return
        
    # This matrix will store the XCD ID for each (head, block) pair.
    mapping_matrix = np.zeros((num_heads, num_blocks), dtype=int)
    
    # The traced_output tensor has shape (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ)
    # The value at each position is the work-group ID that wrote it.
    # We only need to sample one point per block to find its ID.
    for h in range(num_heads):
        for b in range(num_blocks):
            # Get the work-group ID for the first token in this block
            work_group_id = int(traced_output[0, b * block_m, h, 0].item())
            
            # Now, use the DECODER to find out which XCD this work-group belongs to
            xcd_id = get_xcd_id_fn(work_group_id, mha_config)
            mapping_matrix[h, b] = xcd_id

    # --- Visualization setup (same as theoretical) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(min(20, num_blocks * 1.2), num_heads * 0.8), constrained_layout=True)
    pastel_colors = sns.color_palette("pastel", num_xcds)
    cmap = ListedColormap(pastel_colors)

    ax.imshow(mapping_matrix, cmap=cmap, vmin=0, vmax=num_xcds - 1)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xticks(np.arange(num_blocks + 1) - .5, minor=True)
    ax.set_yticks(np.arange(num_heads + 1) - .5, minor=True)
    ax.grid(which="minor", color='white', linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Block #", fontsize=12)
    ax.set_ylabel("Head #", fontsize=12)
    ax.set_xticks(np.arange(num_blocks))
    ax.set_xticklabels([f"B{j}" for j in range(num_blocks)])
    ax.set_yticks(np.arange(num_heads))
    ax.set_yticklabels([f"H{j}" for j in range(num_heads)])

    for h_idx in range(num_heads):
        for b_idx in range(num_blocks):
            xcd_id = mapping_matrix[h_idx, b_idx]
            bg_color = cmap(xcd_id / (num_xcds - 1) if num_xcds > 1 else 0.5)
            text_color = get_text_color(bg_color)
            ax.text(b_idx, h_idx, f"XCD\n{xcd_id}", ha="center", va="center", color=text_color, fontsize=9, weight='bold', linespacing=1.5)

    plt.suptitle(f"{num_heads} Heads, {num_blocks} Blocks, {num_xcds} XCDs", fontsize=20, weight='bold')
    plt.show()

# =================================================================
#                            MAIN
# =================================================================

def main():
    """Main function to run the verification process."""
    
    # --- Define the configuration to test ---
    # This config should match what you expect in your real model.
    mha_config_to_test = {
        "BATCH": 1,
        "SEQLEN_Q": 2048,
        "NUM_Q_HEADS": 10,
        "HEAD_SZ": 128,
        "BLOCK_M": 128,
        "NUM_XCDS": 4, # Number of hardware units (e.g., XCDs)
    }

    print("--- Step 1: Visualizing Theoretical Mapping ---")
    visualize_theoretical_mapping(
        mha_config=mha_config_to_test.copy(), # Use a copy to avoid mutation
        swizzle_fn=swizzle_mha_wid_balanced_python,
        get_xcd_id_fn=get_xcd_id_for_head_balanced,
        title="Theoretical (Expected) Swizzle Pattern"
    )

    print("\n--- Step 2: Running Tracer Kernel for Actual Mapping ---")
    traced_output_tensor = flash_attn_tracer_func(mha_config_to_test)
    print("Tracer kernel finished.")

    print("\n--- Step 3: Visualizing Actual (Traced) Mapping ---")
    visualize_traced_mapping(
        mha_config=mha_config_to_test.copy(), # Use a copy
        traced_output=traced_output_tensor,
        get_xcd_id_fn=get_xcd_id_for_head_balanced,
        title="Actual (Traced from Kernel) Swizzle Pattern"
    )
    
    print("\nVerification complete. Compare the two plots to confirm correctness.")


if __name__ == '__main__':
    main()
