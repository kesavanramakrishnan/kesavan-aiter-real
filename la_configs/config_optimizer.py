#!/usr/bin/env python3
"""
Config Optimizer - Selects the best performing configurations from multiple autotune files.

Usage:
    python config_optimizer.py --input file1.json file2.json file3.json --output best_configs.json
    python config_optimizer.py -i file1.json file2.json -o best_configs.json --metric tflops
    python config_optimizer.py -i file1.json file2.json -o best_configs.json --metric latency
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def load_config_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse a JSON config file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        sys.exit(1)


def create_key_hash(config_entry: Dict[str, Any]) -> str:
    """Create a unique hash for a configuration key."""
    key = config_entry["key"]
    return f"{key['causal']}_{key['B']}_{key['H']}_{key['D']}_{key['NQ']}_{key['NK']}"


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any], metric: str) -> bool:
    """Compare two configurations based on the specified metric."""
    if metric == "tflops":
        # Higher tflops is better
        return config1["tflops"] > config2["tflops"]
    elif metric == "latency":
        # Lower ms is better
        return config1["ms"] < config2["ms"]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def select_best_configs(config_files: List[str], metric: str = "tflops") -> List[Dict[str, Any]]:
    """
    Load all config files and select the best configuration for each unique key combination.
    
    Args:
        config_files: List of file paths to JSON config files
        metric: Metric to optimize for ("tflops" or "latency")
    
    Returns:
        List of best configurations
    """
    # Dictionary to store the best config for each key combination
    best_configs = {}
    
    # Process each config file
    for file_path in config_files:
        print(f"Processing {file_path}...")
        configs = load_config_file(file_path)
        
        for config in configs:
            key_hash = create_key_hash(config)
            
            # If this is the first config for this key, or if it's better than the current best
            if key_hash not in best_configs or compare_configs(config, best_configs[key_hash], metric):
                best_configs[key_hash] = config.copy()
                best_configs[key_hash]["source_file"] = file_path
    
    # Convert back to list and sort by key for consistent output
    result = list(best_configs.values())
    result.sort(key=lambda x: (x["key"]["causal"], x["key"]["B"], x["key"]["H"], 
                              x["key"]["D"], x["key"]["NQ"], x["key"]["NK"]))
    
    return result


def save_configs(configs: List[Dict[str, Any]], output_file: str):
    """Save configurations to a JSON file."""
    # Remove the source_file field before saving
    cleaned_configs = []
    for config in configs:
        cleaned_config = config.copy()
        if "source_file" in cleaned_config:
            del cleaned_config["source_file"]
        cleaned_configs.append(cleaned_config)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(cleaned_configs, f, indent=4)
        print(f"Best configurations saved to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")
        sys.exit(1)


def print_summary(configs: List[Dict[str, Any]], metric: str):
    """Print a summary of the selected configurations."""
    print(f"\n=== Summary ===")
    print(f"Total configurations selected: {len(configs)}")
    print(f"Optimization metric: {metric}")
    
    if metric == "tflops":
        avg_tflops = sum(c["tflops"] for c in configs) / len(configs)
        max_tflops = max(c["tflops"] for c in configs)
        min_tflops = min(c["tflops"] for c in configs)
        print(f"Average TFLOPs: {avg_tflops:.2f}")
        print(f"Max TFLOPs: {max_tflops:.2f}")
        print(f"Min TFLOPs: {min_tflops:.2f}")
    else:  # latency
        avg_ms = sum(c["ms"] for c in configs) / len(configs)
        max_ms = max(c["ms"] for c in configs)
        min_ms = min(c["ms"] for c in configs)
        print(f"Average latency (ms): {avg_ms:.4f}")
        print(f"Max latency (ms): {max_ms:.4f}")
        print(f"Min latency (ms): {min_ms:.4f}")
    
    # Count configurations by causal setting
    causal_count = defaultdict(int)
    for config in configs:
        causal = config["key"]["causal"]
        causal_count[causal] += 1
    
    print(f"\nConfigurations by causal setting:")
    for causal, count in sorted(causal_count.items()):
        print(f"  Causal {causal}: {count} configs")


def main():
    parser = argparse.ArgumentParser(
        description="Select the best performing configurations from multiple autotune files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config_optimizer.py -i file1.json file2.json -o best.json
  python config_optimizer.py -i *.json -o best.json --metric latency
  python config_optimizer.py -i configs/*.json -o optimized.json --metric tflops
        """
    )
    
    parser.add_argument(
        "-i", "--input", 
        nargs="+", 
        required=True,
        help="Input JSON config files (can use glob patterns)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Output JSON file for best configurations"
    )
    
    parser.add_argument(
        "--metric", 
        choices=["tflops", "latency"], 
        default="tflops",
        help="Metric to optimize for (default: tflops)"
    )
    
    parser.add_argument(
        "--summary", 
        action="store_true",
        help="Print summary statistics"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns in input files
    input_files = []
    for pattern in args.input:
        if "*" in pattern or "?" in pattern:
            from glob import glob
            input_files.extend(glob(pattern))
        else:
            input_files.append(pattern)
    
    # Remove duplicates and check if files exist
    input_files = list(set(input_files))
    for file_path in input_files:
        if not Path(file_path).exists():
            print(f"Warning: File '{file_path}' does not exist, skipping...")
            input_files.remove(file_path)
    
    if not input_files:
        print("Error: No valid input files found.")
        sys.exit(1)
    
    print(f"Processing {len(input_files)} input files...")
    
    # Select best configurations
    best_configs = select_best_configs(input_files, args.metric)
    
    # Save to output file
    save_configs(best_configs, args.output)
    
    # Print summary if requested
    if args.summary:
        print_summary(best_configs, args.metric)


if __name__ == "__main__":
    main()
