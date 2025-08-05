# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import logging
import numpy as np
import triton
import sys

from aiter.ops.triton.mha import (
    flash_attn_func,
    flash_attn_fp8_func,
    flash_attn_varlen_func,
    flash_attn_varlen_fp8_func,
    mha_set_use_fused_bwd_kernel,
    mha_set_use_int64_strides,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False
ATOL_fp8 = 2.5e-1
RTOL_fp8 = 2.5e-1


def pad_rearrange_dropout_mask(
    S_dmask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqlen_q,
    seqlen_k,
    num_q_heads,
):
    batch_size = cu_seqlens_q.numel() - 1

    padded_dropout_mask = torch.ones(
        (batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda"
    )
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
            padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[
                b, h, :, :
            ]

    return padded_dropout_mask


def fp8_assert_close(
    tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5
):
    """Assert tensors are close with tolerance for small percentage of elements"""
    # standard comparison
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / torch.abs(tensor_b.clamp(min=1e-6))

    # calculate elements that exceed tolerance
    abs_check = abs_diff > atol
    rel_check = rel_diff > rtol
    failed_check = torch.logical_and(abs_check, rel_check)

    # calculate percentage of failed elements
    failed_percentage = failed_check.sum().item() / failed_check.numel() * 100

    # if percentage is small enough, test passes
    if failed_percentage <= max_diff_percentage:
        return True

    # Otherwise, provide diagnostic information
    max_abs_idx = torch.argmax(abs_diff).item()
    max_rel_idx = torch.argmax(rel_diff).item()

    flat_to_idx = lambda flat_idx, shape: np.unravel_index(flat_idx, shape)

    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)

    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()

    raise AssertionError(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )

configs = []
for CAUSAL in [True]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["BATCH", "NUM_Q_HEADS", "NUM_K_HEADS", "SEQLEN_Q", "SEQLEN_K", "HEAD_SZ", "DROPOUT", "RETURN_LSE", "RETURN_SOFTMAX", "FP8"],
            x_vals=[
                (2, 64, 64, 8192, 8192, 128, 0.0, False, False, False),
 #               (1, 64, 16, [131072], 128, 304, torch.float16, 16, 64, 2, 4),
 #               (1, 64, 16, [262144], 128, 304, torch.float16, 16, 64, 2, 4),
 #               (1, 64, 16, [524288], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 96, 16, [32768], 128, 304, torch.float16, 16, 64, 2, 4),
 #               (1, 96, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 96, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 96, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 96, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
 #               (1, 96, 16, [1048576], 16, 912, torch.float16, 16, 256, 1, 4),  #
 #               (1, 128, 16, [32768], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 128, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 128, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 128, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
 #               (1, 128, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
 #               (3, 64, 16, [4096, 32768, 65536], 64, 912, torch.float16, 16, 64, 2, 4),
            ],
            line_arg="provider",
            line_vals=["triton"],
            line_names=["Triton"],
            # styles=[('red', '-'), ('blue', '-')],
            ylabel="ms",
            plot_name="mha-attention-",
            args={
                "CAUSAL": CAUSAL,
            },
        )
    )


@triton.testing.perf_report(configs)
def bench_mha(
    provider,
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    FP8: bool,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    dropout_mask = None
    if FP8:
        triton_out = lambda: flash_attn_fp8_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=RETURN_LSE,
            return_attn_probs=RETURN_SOFTMAX,
        )
    else:
        triton_out = lambda: flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=RETURN_LSE,
            return_attn_probs=RETURN_SOFTMAX,
        )

    '''
    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    if DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
        dropout_mask = sd_mask >= 0
        if DEBUG_MODE:
            print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(
                f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}"
            )

    if RETURN_SOFTMAX or RETURN_LSE:
        triton_out = triton_out[0]
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")
    '''

    warmup = 1
    rep = 1

    ms = triton.testing.do_bench(triton_out, warmup=warmup, rep=rep)

    return ms


def main():
    bench_mha.run(save_path=".", print_data=True)


if __name__ == "__main__":
    sys.exit(main())

   
