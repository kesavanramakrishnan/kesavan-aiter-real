import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable, Optional
import seaborn as sns
from matplotlib.colors import ListedColormap

def get_text_color(background_color: Tuple[float, float, float, float]) -> str:
    """Determines if text should be black or white based on background luminance."""
    luminance = (0.299 * background_color[0] + 0.587 * background_color[1] + 0.114 * background_color[2])
    return "white" if luminance < 0.5 else "black"

def visualize_default_mapping(
    grid_dim: Tuple[int, int],
    num_xcds: int,
    viz_blocks: int
):
    """
    Generates a matrix plot for the default interleaved mapping based on head number.

    Args:
        grid_dim: A tuple (total_blocks, num_heads).
        num_xcds: The number of XCDs to map onto.
        viz_blocks: The number of blocks to display in the visualization.
    """
    total_blocks, num_heads = grid_dim
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. Create the mapping matrix ---
    # The XCD assignment is based purely on the head number (interleaved).
    mapping_matrix = np.zeros((num_heads, viz_blocks), dtype=int)
    for h in range(num_heads):
        for b in range(viz_blocks):
            mapping_matrix[h, b] = h % num_xcds

    # --- 2. Visualize the matrix ---
    fig, ax = plt.subplots(1, 1, figsize=(min(20, viz_blocks * 1.2), num_heads * 0.8), constrained_layout=True)
    title = f"Default Interleaved Mapping ({num_heads} Heads, {num_xcds} XCDs, {total_blocks} Blocks)"
    fig.suptitle(title, fontsize=20, weight='bold')

    # Define a discrete colormap with one color per XCD
    pastel_colors = sns.color_palette("pastel", num_xcds)
    cmap = ListedColormap(pastel_colors)
    
    # Display the matrix as an image
    im = ax.imshow(mapping_matrix, cmap=cmap, vmin=0, vmax=num_xcds - 1)

    # Add gridlines to visually separate each block
    ax.set_xticks(np.arange(viz_blocks + 1) - .5, minor=True)
    ax.set_yticks(np.arange(num_heads + 1) - .5, minor=True)
    ax.grid(which="minor", color='white', linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Add the XCD ID number inside each cell with improved text color
    for h in range(num_heads):
        for b in range(viz_blocks):
            xcd_id = mapping_matrix[h, b]
            bg_color = cmap(xcd_id / (num_xcds - 1) if num_xcds > 1 else 0.5)
            text_color = get_text_color(bg_color)
            ax.text(b, h, str(xcd_id), ha="center", va="center", color=text_color, fontsize=12, weight='bold')

    # Set labels and ticks
    ax.set_xlabel("Block #", fontsize=12)
    ax.set_xticks(np.arange(viz_blocks))
    ax.set_xticklabels([f"B{j}" for j in range(viz_blocks)])

    ax.set_ylabel("Head #", fontsize=12)
    ax.set_yticks(np.arange(num_heads))
    ax.set_yticklabels([f"H{j}" for j in range(num_heads)])
    
    # Add a color bar legend
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, location='right', pad=0.02)
    cbar.set_label('XCD ID', fontsize=12)
    cbar.set_ticks(np.arange(num_xcds))

    plt.show()

# =================================================================
#                  NEW: BEFORE/AFTER COMPARISON UTILITY
# =================================================================

def visualize_mapping_comparison(
    num_heads: int,
    num_blocks: int,
    num_xcds: int,
    swizzle_fn: Callable[[int, int, int], int],
    capacity_per_xcd: Optional[int] = None,
):
    """Visualize default (round-robin) vs user-provided swizzle mapping.

    Each mapping is shown as a matrix where rows correspond to heads and
    columns correspond to blocks (work-groups).  Cell colors indicate the
    XCD ID to which that (head, block) pair is mapped.

    Args:
        num_heads: Total number of heads (matrix rows).
        num_blocks: Total number of blocks per head (matrix columns).
        num_xcds:   Number of available XCDs.
        swizzle_fn: Function mapping (head, block, num_xcds) → xcd_id after
                     swizzling.
        capacity_per_xcd: Maximum #WGs an XCD can accommodate.  If provided,
                     the visualized *num_blocks* will be clamped such that
                     num_heads * num_blocks ≤ capacity_per_xcd * num_xcds.
    """
    # --- handle capacity restriction ---
    if capacity_per_xcd is not None:
        max_total_wgs = capacity_per_xcd * num_xcds
        if num_heads * num_blocks > max_total_wgs:
            num_blocks = max_total_wgs // num_heads
            if num_blocks == 0:
                raise ValueError(
                    "Capacity too small for the given number of heads; decrease num_heads or increase capacity_per_xcd.")

    # Build default (round-robin) and swizzled mapping matrices
    default_map = np.zeros((num_heads, num_blocks), dtype=int)
    swizzled_map = np.zeros_like(default_map)

    for h in range(num_heads):
        for b in range(num_blocks):
            default_map[h, b] = h % num_xcds  # round-robin baseline
            swizzled_map[h, b] = swizzle_fn(h, b, num_xcds)

    # Visualization setup -------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(min(25, num_blocks * 1.2 * 2), num_heads * 0.8), constrained_layout=True)
    pastel_colors = sns.color_palette("pastel", num_xcds)
    cmap = ListedColormap(pastel_colors)

    # Helper to render each subplot
    def _render(ax, data, title: str):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=num_xcds - 1)
        ax.set_title(title, fontsize=16, weight='bold')
        # grid
        ax.set_xticks(np.arange(num_blocks + 1) - .5, minor=True)
        ax.set_yticks(np.arange(num_heads + 1) - .5, minor=True)
        ax.grid(which="minor", color='white', linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        # labels
        ax.set_xlabel("Block #", fontsize=12)
        ax.set_ylabel("Head #", fontsize=12)
        ax.set_xticks(np.arange(num_blocks))
        ax.set_xticklabels([f"B{j}" for j in range(num_blocks)])
        ax.set_yticks(np.arange(num_heads))
        ax.set_yticklabels([f"H{j}" for j in range(num_heads)])
        # cell labels with adaptive text color
        for h in range(num_heads):
            for b in range(num_blocks):
                xcd_id = data[h, b]
                bg_color = cmap(xcd_id / (num_xcds - 1) if num_xcds > 1 else 0.5)
                text_color = get_text_color(bg_color)
                ax.text(b, h, str(xcd_id), ha="center", va="center", color=text_color, fontsize=10, weight='bold')
        return im

    im1 = _render(axes[0], default_map, "Before (Round-Robin)")
    _ = _render(axes[1], swizzled_map, "After (Swizzled)")

    # single shared color bar on the right
    cbar = fig.colorbar(im1, ax=axes, location='right', pad=0.02, shrink=0.8)
    cbar.set_label('XCD ID', fontsize=12)
    cbar.set_ticks(np.arange(num_xcds))
    plt.suptitle(f"Mapping Comparison – {num_heads} Heads, {num_blocks} Blocks, {num_xcds} XCDs", fontsize=20, weight='bold')
    plt.show()
    input("Press Enter to close the plot...")

# =================================================================
#                            --- HOW TO USE ---
# =================================================================

def remap_xcd_swizzle(pid: int, grid_mn: int, num_xcds: int) -> int:
    """
    A direct, pure-Python port of the remap_xcd swizzling function.
    
    This function takes a PID (like a head index) from a round-robin
    distribution and maps it to a new, contiguous PID.
    """
    # Number of PIDs per XCD in the new contiguous arrangement
    pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
    
    # Number of XCDs that will hold an extra PID if the load is uneven
    tall_xcds = grid_mn % num_xcds
    if tall_xcds == 0 and grid_mn > 0:
        tall_xcds = num_xcds
        
    # Deconstruct the original PID into its round-robin components
    original_xcd = pid % num_xcds
    local_pid_offset = pid // num_xcds
    
    # Calculate the new, contiguous PID based on the original mapping
    if original_xcd < tall_xcds:
        # This PID was on a "tall" XCD, which gets more PIDs
        new_pid = original_xcd * pids_per_xcd + local_pid_offset
    else:
        # This PID was on a "short" XCD
        pids_in_tall_group = tall_xcds * pids_per_xcd
        pids_per_short_xcd = pids_per_xcd - 1
        
        # Offset into the short group
        local_xcd_offset = original_xcd - tall_xcds
        
        new_pid = (pids_in_tall_group 
                   + local_xcd_offset * pids_per_short_xcd 
                   + local_pid_offset)
                   
    return new_pid

if __name__ == "__main__":
    # User-defined parameters -------------------------------------------
    NUM_HEADS = 8
    NUM_BLOCKS = 2  # per head
    NUM_XCDS = 2
    CAPACITY_PER_XCD = 8  # visualize up to this many WGs per XCD

    # --- Wrapper for the visualizer ---
    def swizzle_for_viz(head, block, num_xcds):
        # The MHA kernel remaps the head index.
        # So, 'head' is the pid, and 'NUM_HEADS' is the grid_mn.
        remapped_head = remap_xcd_swizzle(head, NUM_HEADS, num_xcds)
        
        # In a contiguous layout, the XCD is found by simple division.
        heads_per_xcd = (NUM_HEADS + num_xcds - 1) // num_xcds
        return remapped_head // heads_per_xcd

    # Call the comparison visualization ---------------------------------
    visualize_mapping_comparison(
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        num_xcds=NUM_XCDS,
        swizzle_fn=swizzle_for_viz,
        capacity_per_xcd=CAPACITY_PER_XCD,
    )