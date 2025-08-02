import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

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
    cmap = plt.colormaps.get('cividis').resampled(num_xcds)
    
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
#                            --- HOW TO USE ---
# =================================================================

# 1. Define your grid parameters as specified
GRID_DIMENSIONS = (128, 8)  # 128 Blocks, 8 Heads
NUM_XCDS = 4

# 2. Define how many blocks to show in the plot to keep it readable
BLOCKS_TO_VISUALIZE = 2

# 3. Call the visualization function
if __name__ == "__main__":
    visualize_default_mapping(
        grid_dim=GRID_DIMENSIONS,
        num_xcds=NUM_XCDS,
        viz_blocks=BLOCKS_TO_VISUALIZE
    )