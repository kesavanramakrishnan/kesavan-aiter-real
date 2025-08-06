# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Part 1: The "Plugin" Visualization Tool

This script visualizes the THEORETICAL mapping of work-items to hardware
based on a given Python swizzle function. You can plug in different swizzle
functions to see their intended access patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable, Dict, Any
import seaborn as sns
from matplotlib.colors import ListedColormap

# =================================================================
#                        HELPER FUNCTIONS
# =================================================================

def get_text_color(background_color: Tuple[float, float, float, float]) -> str:
    """Determines if text should be black or white based on background luminance."""
    luminance = (0.299 * background_color[0] + 0.587 * background_color[1] + 0.114 * background_color[2])
    return "white" if luminance < 0.5 else "black"

def cdiv(x, y):
    """Ceiling division utility."""
    return (x + y - 1) // y

# =================================================================
#                   EXAMPLE SWIZZLE FUNCTION
# =================================================================

def swizzle_mha_wid_balanced_python(wid: int, config: Dict[str, Any]) -> int:
    """
    Swizzle 1: Groups heads together onto XCDs, handling non-divisible cases.
    This is the Python reference implementation of the swizzle logic.
    """
    NUM_Q_HEADS = config['NUM_Q_HEADS']
    NUM_BLOCKS = config['NUM_BLOCKS']
    NUM_XCDS = config['NUM_XCDS']

    wids_per_batch = NUM_Q_HEADS * NUM_BLOCKS
    off_z = wid // wids_per_batch
    local_wid = wid % wids_per_batch

    heads_per_xcd_short = NUM_Q_HEADS // NUM_XCDS
    heads_per_xcd_tall = heads_per_xcd_short + 1
    num_tall_xcds = NUM_Q_HEADS % NUM_XCDS
    
    wids_per_tall_xcd = heads_per_xcd_tall * NUM_BLOCKS
    wids_in_tall_xcds_total = num_tall_xcds * wids_per_tall_xcd

    if local_wid < wids_in_tall_xcds_total:
        xcd_idx_tall = local_wid // wids_per_tall_xcd
        local_wid_in_xcd_tall = local_wid % wids_per_tall_xcd
        target_q_head_tall = xcd_idx_tall * heads_per_xcd_tall + (local_wid_in_xcd_tall // NUM_BLOCKS)
        local_wid_in_xcd = local_wid_in_xcd_tall
        target_q_head = target_q_head_tall
    else:
        wids_per_short_xcd = heads_per_xcd_short * NUM_BLOCKS
        wid_after_tall = local_wid - wids_in_tall_xcds_total
        xcd_local_idx_short = wid_after_tall // wids_per_short_xcd
        local_wid_in_xcd_short = wid_after_tall % wids_per_short_xcd
        target_q_head_short = (
            (num_tall_xcds * heads_per_xcd_tall) 
            + xcd_local_idx_short * heads_per_xcd_short 
            + (local_wid_in_xcd_short // NUM_BLOCKS)
        )
        local_wid_in_xcd = local_wid_in_xcd_short
        target_q_head = target_q_head_short

    target_start_m = local_wid_in_xcd % NUM_BLOCKS
    swizzled_local_wid = target_q_head * NUM_BLOCKS + target_start_m
    
    return off_z * wids_per_batch + swizzled_local_wid

def get_xcd_id_for_head_balanced(wid: int, config: Dict[str, Any]) -> int:
    """Decoder: Finds the XCD ID for a head-balanced swizzled WID."""
    num_heads = config['NUM_Q_HEADS']
    num_xcds = config['NUM_XCDS']
    num_blocks = config['NUM_BLOCKS']
    swizzled_h = (wid // num_blocks) % num_heads

    heads_per_xcd_short = num_heads // num_xcds
    heads_per_xcd_tall = heads_per_xcd_short + 1
    num_tall_xcds = num_heads % num_xcds
    tall_heads_total = num_tall_xcds * heads_per_xcd_tall

    if heads_per_xcd_short == 0:
        return swizzled_h
    elif swizzled_h < tall_heads_total:
        return swizzled_h // heads_per_xcd_tall
    else:
        heads_after_tall = swizzled_h - tall_heads_total
        return num_tall_xcds + (heads_after_tall // heads_per_xcd_short)

# =================================================================
#                        VISUALIZATION
# =================================================================

def visualize_theoretical_mapping(
    mha_config: Dict[str, Any],
    swizzle_fn: Callable[[int, Dict[str, Any]], int],
    get_xcd_id_fn: Callable[[int, Dict[str, Any]], int],
    title: str = "Theoretical Swizzle Pattern"
):
    """
    Visualizes the theoretical mapping of (Head, Block) pairs to XCDs.
    """
    num_heads = mha_config['NUM_Q_HEADS']
    num_xcds = mha_config['NUM_XCDS']
    
    num_blocks = cdiv(mha_config['SEQLEN_Q'], mha_config['BLOCK_M'])
    mha_config['NUM_BLOCKS'] = num_blocks

    if num_blocks == 0 or num_heads == 0:
        print("Warning: Cannot visualize with 0 blocks or 0 heads.")
        return

    # This matrix will store the XCD ID for each (head, block) pair.
    mapping_matrix = np.zeros((num_heads, num_blocks), dtype=int)

    for h in range(num_heads):
        for b in range(num_blocks):
            original_wid = h * num_blocks + b
            swizzled_wid = swizzle_fn(original_wid, mha_config)
            xcd_id = get_xcd_id_fn(swizzled_wid, mha_config)
            mapping_matrix[h, b] = xcd_id

    # --- Visualization setup ---
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

if __name__ == '__main__':
    # --- Example Usage ---
    mha_config_example = {
        "SEQLEN_Q": 1024,
        "NUM_Q_HEADS": 10,
        "BLOCK_M": 128,
        "NUM_XCDS": 8, # Number of hardware units (e.g., XCDs)
    }

    print("Visualizing theoretical mapping for head-balanced swizzle...")
    visualize_theoretical_mapping(
        mha_config=mha_config_example,
        swizzle_fn=swizzle_mha_wid_balanced_python,
        get_xcd_id_fn=get_xcd_id_for_head_balanced
    )
