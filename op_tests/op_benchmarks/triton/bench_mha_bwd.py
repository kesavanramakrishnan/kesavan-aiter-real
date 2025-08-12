# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
import triton

# Forward (for producing out and softmax_lse)
from aiter.ops.triton.mha import flash_attn_func

# Backward implementations
from aiter.ops.triton.lean_atten_bwd_prod import (
    flash_attn_onekernel_backward as lean_bwd,
)
from aiter.ops.triton.mha_onekernel_bwd import (
    flash_attn_onekernel_backward as onekernel_bwd,
)


# Minimal benchmark configs: a couple of representative shapes
configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=["CAUSAL", "BATCH", "NUM_HEADS", "SEQLEN", "HEAD_SZ"],
        x_vals=[
            (False, 1, 4, 128, 128),
            (False, 2, 64, 8192, 128),
        ],
        line_arg="provider",
        line_vals=["lean", "onek"],
        line_names=["Lean BWD (ms)", "OneKernel BWD (ms)"],
        ylabel="ms",
        plot_name="attention-bwd-comparison",
        args={},
    )
)


@triton.testing.perf_report(configs)
def bench_attention_bwd(
    CAUSAL: bool,
    BATCH: int,
    NUM_HEADS: int,
    SEQLEN: int,
    HEAD_SZ: int,
    provider: str,
    device: str = "cuda",
):
    warmup = 25
    rep = 100

    # dtype and scale
    dtype = torch.float16
    sm_scale = 1.0 / (HEAD_SZ ** 0.5)

    # Build inputs: bshd layout for forward
    q = torch.randn((BATCH, SEQLEN, NUM_HEADS, HEAD_SZ), dtype=dtype, device=device)
    k = torch.randn((BATCH, SEQLEN, NUM_HEADS, HEAD_SZ), dtype=dtype, device=device)
    v = torch.randn((BATCH, SEQLEN, NUM_HEADS, HEAD_SZ), dtype=dtype, device=device)

    # Run forward once to obtain out and softmax_lse used by both backward paths
    out, softmax_lse = flash_attn_func(
        q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=CAUSAL, return_lse=True
    )

    # Upstream gradient
    do = torch.randn_like(out)

    # Allocate grads
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if provider == "lean":
        # Lean wrapper expects non-varlen, no-dropout path
        fn = lambda: lean_bwd(
            do,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            None,            # dbias
            sm_scale,
            None,            # alibi_slopes
            CAUSAL,
            None,            # cu_seqlens_q
            None,            # cu_seqlens_k
            SEQLEN,
            SEQLEN,
            0.0,             # dropout_p
        )
    elif provider == "onek":
        fn = lambda: onekernel_bwd(
            do,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            None,            # dbias
            sm_scale,
            None,            # alibi_slopes
            CAUSAL,
            None,            # cu_seqlens_q
            None,            # cu_seqlens_k
            SEQLEN,
            SEQLEN,
            0.0,             # dropout_p
        )
    else:
        return None

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


def main():
    bench_attention_bwd.run(save_path=".", print_data=True)


if __name__ == "__main__":
    sys.exit(main())


