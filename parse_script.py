#!/usr/bin/env python3
import re
import sys
import collections

def parse_log_file(log_file_path):
    """
    Parses a log file containing Triton kernel traces to validate the
    maximum number of output tiles processed by a single workgroup.
    This version uses a more robust parsing strategy that pieces together
    traces from operand fragments to handle incomplete log output.
    """
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        sys.exit(1)

    # Split the log into blocks for each test configuration.
    config_blocks = re.split(r"Host-side calculation:", content)[1:]

    if not config_blocks:
        print("No test configurations found in the log file.")
        return

    print(f"Found {len(config_blocks)} test configurations to analyze.\n")

    # Regex to find the individual components of our trace: (operand X) Y
    operand_regex = re.compile(r"\(operand (\d+)\)\s*(\d+)")

    for i, block in enumerate(config_blocks):
        print(f"--- Analyzing Configuration {i+1} ---")

        # 1. Extract the host-side calculated value.
        host_calc_match = re.search(r"Max OUTPUT tiles per workgroup \(CTA\) = (\d+)", block)
        if host_calc_match:
            host_max_tiles = int(host_calc_match.group(1))
            print(f"Host-side calculation: {host_max_tiles} max output tiles per WG.")
        else:
            print("Could not find host-side calculation for this block.")
            host_max_tiles = "N/A"

        # 2. Parse kernel traces by reconstructing them from operand fragments.
        workload_per_pid = collections.defaultdict(set)
        
        # Find all operand matches in the current configuration block.
        all_operands = operand_regex.finditer(block)
        
        current_trace = {}
        num_traces_found = 0

        for match in all_operands:
            try:
                operand_idx_str, value_str = match.groups()
            except (ValueError, IndexError):
                continue

            # A new trace starts when we see operand '0'.
            # If the previous trace was complete, process it.
            if operand_idx_str == '0':
                if len(current_trace) == 4:
                    num_traces_found += 1
                    try:
                        pid = int(current_trace['0'])
                        head = int(current_trace['1'])
                        batch = int(current_trace['2'])
                        m_block = int(current_trace['3'])
                        workload_per_pid[pid].add((head, batch, m_block))
                    except (KeyError, ValueError):
                        pass # Incomplete trace, ignore
                
                # Start a new trace
                current_trace = {}

            current_trace[operand_idx_str] = value_str
        
        # Process the very last trace in the block if it's complete
        if len(current_trace) == 4:
            num_traces_found += 1
            try:
                pid = int(current_trace['0'])
                head = int(current_trace['1'])
                batch = int(current_trace['2'])
                m_block = int(current_trace['3'])
                workload_per_pid[pid].add((head, batch, m_block))
            except (KeyError, ValueError):
                pass # Incomplete trace, ignore

        if not workload_per_pid:
            print("Kernel-side validation: No complete kernel traces were found.")
            print("-> This is likely due to the print buffer overflowing.\n")
            continue
            
        # 3. Find the maximum number of unique tiles processed by any single pid.
        kernel_max_tiles = max(len(tiles) for tiles in workload_per_pid.values())

        print(f"Kernel-side validation: {kernel_max_tiles} max output tiles per WG.")
        print(f"({num_traces_found} complete traces parsed across {len(workload_per_pid)} workgroups)")

        # 4. Compare results
        if host_max_tiles != "N/A" and host_max_tiles == kernel_max_tiles:
            print("Result: MATCH\n")
        else:
            print("Result: MISMATCH\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_script.py <path_to_log_file>")
        sys.exit(1)
    log_file = sys.argv[1]
    parse_log_file(log_file)

