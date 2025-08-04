#!/usr/bin/env python3
"""
Example swizzle functions for XCD mapping visualization.

This module provides various swizzle functions that can be used with
visualize_mapping_comparison() to see different mapping strategies.
"""

from visualize_swizzling import visualize_mapping_comparison


def round_robin_swizzle(head: int, block: int, num_xcds: int) -> int:
    """Default round-robin assignment (same as baseline)."""
    return head % num_xcds


def block_interleaved_swizzle(head: int, block: int, num_xcds: int) -> int:
    """Interleave based on block number instead of head."""
    return block % num_xcds


def head_block_xor_swizzle(head: int, block: int, num_xcds: int) -> int:
    """Use XOR of head and block for more random-looking distribution."""
    return (head ^ block) % num_xcds


def alternating_head_groups_swizzle(head: int, block: int, num_xcds: int) -> int:
    """Group heads in pairs and alternate XCD assignment."""
    return ((head // 2) + block) % num_xcds


def fibonacci_swizzle(head: int, block: int, num_xcds: int) -> int:
    """Use Fibonacci-like pattern for distribution."""
    return (head + block * 2) % num_xcds


def custom_pattern_swizzle(head: int, block: int, num_xcds: int) -> int:
    """Custom pattern: heads 0-3 go to XCD 0,1,2,3; heads 4-7 go to XCD 2,3,0,1."""
    if head < 4:
        return head % num_xcds
    else:
        return (head + 2) % num_xcds


def pid_remap_swizzle(head: int, block: int, num_xcds: int) -> int:
    """
    Swizzle based on the pid_preprocessing.py remap_xcd logic.
    
    This implements the sophisticated PID remapping that handles uneven
    distribution of work groups across XCDs, similar to the triton.jit
    remap_xcd function.
    """
    # Treat (head, block) as a single PID for remapping
    pid = head * 4 + block  # Assuming 4 blocks per head for this example
    total_pids = 32  # 8 heads * 4 blocks
    
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (total_pids + num_xcds - 1) // num_xcds
    
    # When total_pids cannot divide num_xcds, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as tall_xcds
    tall_xcds = total_pids % num_xcds
    tall_xcds = num_xcds if tall_xcds == 0 else tall_xcds
    
    # Compute current XCD and local pid within the XCD
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        new_pid = xcd * pids_per_xcd + local_pid
    else:
        new_pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    
    # Convert back to XCD ID
    return new_pid % num_xcds


def pid_remap_generic_swizzle(head: int, block: int, num_xcds: int, 
                              num_heads: int = 8, num_blocks: int = 4) -> int:
    """
    Generic version of pid_remap_swizzle that works with any head/block counts.
    
    Args:
        head: Head index
        block: Block index  
        num_xcds: Number of XCDs
        num_heads: Total number of heads
        num_blocks: Total number of blocks per head
    """
    # Treat (head, block) as a single PID for remapping
    pid = head * num_blocks + block
    total_pids = num_heads * num_blocks
    
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (total_pids + num_xcds - 1) // num_xcds
    
    # When total_pids cannot divide num_xcds, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    tall_xcds = total_pids % num_xcds
    tall_xcds = num_xcds if tall_xcds == 0 else tall_xcds
    
    # Compute current XCD and local pid within the XCD
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    
    # Calculate new pid based on the new grouping
    if xcd < tall_xcds:
        new_pid = xcd * pids_per_xcd + local_pid
    else:
        new_pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    
    # Convert back to XCD ID
    return new_pid % num_xcds


if __name__ == "__main__":
    # Example usage with different swizzle functions
    
    # Parameters
    NUM_HEADS = 8
    NUM_BLOCKS = 4
    NUM_XCDS = 4
    CAPACITY_PER_XCD = 16
    
    print("Testing different swizzle functions...")
    
    # Test 1: Block interleaved
    print("\n1. Block Interleaved Swizzle")
    visualize_mapping_comparison(
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        num_xcds=NUM_XCDS,
        swizzle_fn=block_interleaved_swizzle,
        capacity_per_xcd=CAPACITY_PER_XCD,
    )
    
    # Test 2: Head-Block XOR
    print("\n2. Head-Block XOR Swizzle")
    visualize_mapping_comparison(
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        num_xcds=NUM_XCDS,
        swizzle_fn=head_block_xor_swizzle,
        capacity_per_xcd=CAPACITY_PER_XCD,
    )
    
    # Test 3: Alternating head groups
    print("\n3. Alternating Head Groups Swizzle")
    visualize_mapping_comparison(
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        num_xcds=NUM_XCDS,
        swizzle_fn=alternating_head_groups_swizzle,
        capacity_per_xcd=CAPACITY_PER_XCD,
    )
    
    # Test 4: Custom pattern
    print("\n4. Custom Pattern Swizzle")
    visualize_mapping_comparison(
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        num_xcds=NUM_XCDS,
        swizzle_fn=custom_pattern_swizzle,
        capacity_per_xcd=CAPACITY_PER_XCD,
    )
    
    # Test 5: PID Remap Swizzle (from pid_preprocessing.py)
    print("\n5. PID Remap Swizzle")
    visualize_mapping_comparison(
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        num_xcds=NUM_XCDS,
        swizzle_fn=pid_remap_swizzle,
        capacity_per_xcd=CAPACITY_PER_XCD,
    ) 