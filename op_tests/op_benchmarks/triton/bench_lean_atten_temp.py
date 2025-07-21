# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
import argparse
from aiter.ops.triton.lean_atten_updated_invest import persistent_lean_attention

# Data type mapping for argparse
str_to_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

## -----------------
## Benchmark Configs
## -----------------

# These configs are derived from your pytest parametrize list
# The format is: (BATCH, N_HEADS, N_CTX_Q, N_CTX_LIST, D_HEAD, CAUSAL)
lean_attention_configs = [
    (2, 64, 16, [65536, 65536], 128, False),
    (1, 64, 16, [131072], 128, False),
    (1, 64, 16, [262144], 64, False),
    (1, 96, 16, [65536], 128, False),
    (1, 96, 16, [131072], 128, False),
    (8, 64, 16, [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536], 128, False),
    (1, 64, 8192, [8192], 128, True),
    (2, 64, 2048, [2048, 2048], 128, True),
    # You can add more configurations here
]

# Create triton.testing.Benchmark objects
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['BATCH', 'H', 'N_CTX_Q', 'AVG_N_CTX', 'D_HEAD'],
        x_vals=[(b, h, nq, sum(nk_list) // len(nk_list), d) for b, h, nq, nk_list, d, _ in lean_attention_configs],
        line_arg='provider',
        line_vals=['Triton'],
        line_names=['Triton'],
        styles=[('blue', '-')],
        ylabel="TFLOPS",
        plot_name="lean-attention-performance",
        args={'causal': False, 'dtype': torch.float16},
    )
)
def benchmark_lean_attention(BATCH, H, N_CTX_Q, D_HEAD, provider, causal, dtype=torch.float16, **kwargs):
    """
    Benchmark function for the persistent_lean_attention kernel.
    """
    torch.manual_seed(20)

    # Find the matching full config from the list to get N_CTX_LIST
    # This is a bit of a workaround to fit the variable-length list into triton's benchmark API
    n_ctx_list = []
    for b_cfg, h_cfg, nq_cfg, nk_list_cfg, d_cfg, causal_cfg in lean_attention_configs:
        if b_cfg == BATCH and h_cfg == H and nq_cfg == N_CTX_Q and d_cfg == D_HEAD:
            n_ctx_list = nk_list_cfg
            causal = causal_cfg # Use the causality from the config
            break

    if not n_ctx_list:
        raise ValueError("Could not find a matching benchmark configuration.")

    # These are kernel-specific tuning parameters, hardcoded for simplicity
    # For a more advanced benchmark, these could also be arguments
    BLOCK_M = 128 if causal else 16
    BLOCK_N = 64 if causal else 128
    num_warps = 4
    waves_per_eu = 1 if causal else 2
    total_programs = 304 if causal else 912

    # Derived parameters
    sum_n_ctx = sum(n_ctx_list)
    sm_scale = 0.5

    # Prepare inputs
    list_num_block_n = [(n_ctx + BLOCK_N - 1) // BLOCK_N for n_ctx in n_ctx_list]
    batch_num_block_n = torch.tensor(list_num_block_n, device="cuda").cumsum(dim=0, dtype=torch.int32)

    q = torch.randn((N_CTX_Q * BATCH, H, D_HEAD), dtype=dtype, device="cuda")
    k = torch.randn((sum_n_ctx, H, D_HEAD), dtype=dtype, device="cuda")
    v = torch.randn((sum_n_ctx, H, D_HEAD), dtype=dtype, device="cuda")

    # Kernel-specific persistent buffers
    Mp = torch.empty((total_programs, BLOCK_M), device="cuda", dtype=torch.float32)
    Lp = torch.empty((total_programs, BLOCK_M), device="cuda", dtype=torch.float32)
    Op = torch.empty((total_programs, BLOCK_M, D_HEAD), device="cuda", dtype=torch.float32)
    locks = torch.zeros((total_programs,), device="cuda", dtype=torch.int32)

    # Define the function to benchmark
    def fn():
        return persistent_lean_attention(
            q, k, v, Mp, Lp, Op, locks, batch_num_block_n,
            total_programs, BLOCK_M, BLOCK_N, causal, BATCH,
            sm_scale, num_warps, waves_per_eu
        )

    # Benchmarking
    ms = triton.testing.do_bench(fn)

    # Performance calculations
    # FLOPS for one attention head: 2 * N_CTX_Q * N_CTX * D_HEAD
    total_flops = sum(2 * H * N_CTX_Q * n_ctx * D_HEAD for n_ctx in n_ctx_list)
    if causal:
        # Causal attention is roughly half the FLOPS
        total_flops *= 0.5

    # Memory IO
    bytes_in = (q.numel() + k.numel() + v.numel()) * q.element_size()
    bytes_out = q.numel() * q.element_size()
    total_bytes = bytes_in + bytes_out

    tflops = total_flops / ms * 1e-9
    bandwidth = total_bytes / ms * 1e-6

    # Report metrics
    # The triton decorator automatically handles plotting 'TFLOPS'
    # You can print other metrics for detailed analysis
    print(f"Config: Batch={BATCH}, Heads={H}, N_CTX_Q={N_CTX_Q}, Avg N_CTX={sum(n_ctx_list)//len(n_ctx_list)}, D_HEAD={D_HEAD}, Causal={causal}")
    print(f"\tWalltime: {ms:.3f} ms")
    print(f"\tTFLOPS: {tflops:.2f}")
    print(f"\tBandwidth: {bandwidth:.2f} GB/s")

    return tflops

## ----------------
## Argument Parser
## ----------------

def get_parser():
    """
    Creates the argument parser for the benchmark script.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark script for the Lean Attention kernel."
    )
    parser.add_argument(
        '-o',
        action="store_true",
        help="Write performance results to a CSV file."
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='fp16',
        choices=['fp16', 'bf16', 'fp32'],
        help="Specify the data type for tensors."
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Run the benchmark
    benchmark_lean_attention.run(
        save_path="." if args.o else None,
        print_data=True
    )