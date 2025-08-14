# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import pytest
import torch
from bisect import bisect_right
from typing import Union, List
from aiter.ops.triton.lean_atten_lse import (
    _persistent_lean_attention,
    persistent_lean_attention,
    _get_config,
)
import aiter.ops.triton.utils.arch_info as arch_info


def get_lean_attn_inputs(
    batch: int,
    n_ctx_q: int,
    n_ctx: List[int],
    block_n: int,
    h_q: int,
    h_k: int,
    d: int,
    total_programs: int,
    init_dtype: Union[torch.dtype, str],
):
    assert batch == len(n_ctx)
    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")
    # Allocate Tensors
    q = torch.empty((n_ctx_q * batch, h_q, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((sum_n_ctx, h_k, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((sum_n_ctx, h_k, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, n_ctx_q, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        # print(f"s={s}")
        list_num_block_n = [
            (int(str(s).strip()) + block_n - 1) // block_n for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    return q, k, v, Mp, Lp, Op, locks, batch_num_block_n


def reference_attention(q, k, v, n_ctx, n_ctx_q, sm_scale, causal):

    # Calculate Pytorch reference output with GQA mapping
    ref_out = torch.empty_like(q, dtype=q.dtype)
    start = 0
    start_q = 0

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]  # [n_ctx_q, H_Q, d]
        kb = k[start : (start + int(b)), :, :]  # [b, H_K, d]
        vb = v[start : (start + int(b)), :, :]  # [b, H_K, d]

        H_Q = qb.shape[1]
        H_K = kb.shape[1]
        assert H_Q % H_K == 0
        group_size = H_Q // H_K

        out_b = torch.empty_like(qb)
        for hq in range(H_Q):
            hk = hq // group_size
            qh = qb[:, hq, :]  # [n_ctx_q, d]
            kh = kb[:, hk, :]  # [b, d]
            vh = vb[:, hk, :]  # [b, d]
            scores = (qh @ kh.T) * sm_scale  # [n_ctx_q, b]
            if causal:
                M = torch.tril(torch.ones((n_ctx_q, b), device=qh.device, dtype=torch.bool))
                scores = scores.masked_fill(~M, float("-inf"))
            probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)  # [n_ctx_q, b]
            out_h = probs @ vh  # [n_ctx_q, d]
            out_b[:, hq, :] = out_h

        ref_out[start_q : (start_q + int(n_ctx_q)), :, :] = out_b
        start += b
        start_q += n_ctx_q
    return ref_out


def reference_lse(q, k, n_ctx, n_ctx_q, sm_scale, causal):
    # Natural-log sum-exp per row per Q head with GQA mapping
    ref_lse = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    start = 0
    start_q = 0

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]  # [n_ctx_q, H_Q, d]
        kb = k[start : (start + int(b)), :, :]  # [b, H_K, d]

        H_Q = qb.shape[1]
        H_K = kb.shape[1]
        assert H_Q % H_K == 0
        group_size = H_Q // H_K

        for hq in range(H_Q):
            hk = hq // group_size
            qh = qb[:, hq, :]  # [n_ctx_q, d]
            kh = kb[:, hk, :]  # [b, d]
            scores = (qh @ kh.T) * sm_scale  # [n_ctx_q, b]
            if causal:
                M = torch.tril(torch.ones((n_ctx_q, b), device=qh.device, dtype=torch.bool))
                scores = scores.masked_fill(~M, float("-inf"))
            lse_row = torch.logsumexp(scores.float(), dim=-1)  # [n_ctx_q]
            ref_lse[start_q : (start_q + int(n_ctx_q)), hq] = lse_row

        start += b
        start_q += n_ctx_q

    return ref_lse


@pytest.mark.parametrize(
    "causal, batch, h_q, h_k, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, waves_per_eu, num_warps ",
    [
        # (False, 260, 32, 8, 128, [8192] * 260, 128, 304, torch.float16, 128, 64, 1, 4),
        # (False, 256, 32, 8, 128, [8192] * 256, 128, 304, torch.float16, 128, 64, 1, 4)
        (False, 1024, 64, 8, 128, [8192] * 1024, 128, 304, torch.float16, 128, 64, 1, 4),
        # (False, 16, 128, 8, 1, [8192] * 16, 128, 304, torch.float16, 128, 64, 1, 4),
        # # (True, 1, 128, 128, 8192, [8192], 64, 304, torch.float16, 128, 64, 1, 4),
        # #Added from bench_mha_la
        # (False, 1, 64, 64, 128, [16384], 128, 304, torch.float16, 128, 64, 1, 4),
        # (False, 1, 96, 96, 128, [32768], 128, 304, torch.float16, 128, 64, 1, 4),
        # (True, 1, 64, 64, 8192, [8192], 128, 304, torch.float16, 128, 64, 1, 4),
        # (True, 2, 64, 64, 8192, [8192, 8192], 128, 304, torch.float16, 128, 64, 1, 4),
        # (True, 1, 64, 64, 16384, [16384], 128, 304, torch.float16, 128, 64, 1, 4),
        # (True, 4, 32, 32, 2048, [2048]*4, 128, 304, torch.float16, 128, 64, 1, 4),
        # (True, 2, 64, 64, 16384, [16384, 16384], 128, 304, torch.float16, 128, 64, 2, 4),
        # (False, 2, 64, 64, 8192, [8192, 8192], 128, 304, torch.float16, 128, 64, 1, 4),
        # (False, 2, 64, 64, 16384, [16384, 16384], 128, 304, torch.float16, 128, 64, 1, 4),
        # (False, 2, 48, 48, 16384, [16384, 16384], 128, 304, torch.float16, 128, 64, 1, 4),
        # (False, 2, 64, 64, 128, [65536, 65536], 128, 304, torch.float16, 128, 64, 1, 4),
        # (False, 2, 64, 64, 16, [65536, 65536], 128, 912, torch.float16, 16, 128, 3, 4),
        # (False, 1, 64, 64, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        # (False, 1, 64, 64, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        # (False, 1, 64, 64, 16, [524288], 64, 912, torch.float16, 16, 64, 2, 4),
        # (False, 2, 96, 96, 16, [32768, 32768], 128, 912, torch.float16, 16, 128, 2, 4),
        # (False, 1, 96, 96, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
        # (False, 1, 96, 96, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        # (False, 1, 96, 96, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        # (False, 1, 96, 96, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        # (False, 1, 96, 96, 16, [1048576], 16, 912, torch.float16, 16, 256, 1, 4),  #
        # (False, 1, 128, 128, 16, [32768], 128, 912, torch.float16, 16, 128, 2, 4),
        # (False, 1, 128, 128, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
        # (False, 1, 128, 128, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        # (False, 1, 128, 128, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        # (False, 1, 128, 128, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        # (
        #     False,
        #     3,
        #     64,
        #     64,
        #     16,
        #     [4096, 32768, 65536],
        #     128,
        #     912,
        #     torch.float16,
        #     16,
        #     128,
        #     2,
        #     4,
        # ),
        # (
        #     False,
        #     8,
        #     64,
        #     64,
        #     16,
        #     [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536],
        #     128,
        #     912,
        #     torch.float16,
        #     16,
        #     64,
        #     2,
        #     4,
        # ),
        # (
        #     True,
        #     1,
        #     64,
        #     64,
        #     8192,
        #     [8192],
        #     128,
        #     304,
        #     torch.float16,
        #     128,
        #     64,
        #     2,
        #     4,
        # ),  # Causal=1,
        # (True, 2, 64, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 64, 2, 4),
        #These test cases fail:
        # (True, 2, 64, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 64, 2, 4),
        # (True, 1, 64, 64, 4096, [4096], 128, 304, torch.float16, 128, 16, 3, 4),
        # (False, 1, 64, 64, 4096, [4096], 128, 304, torch.float16, 128, 16, 3, 4),
    ],
)
def test_persistent_lean_attention(
    request,
    batch,
    h_q,
    h_k,
    n_ctx_q,
    n_ctx,
    d,
    total_programs,
    init_dtype,
    BLOCK_M,
    BLOCK_N,
    waves_per_eu,
    num_warps,
    causal,
):
    # print("here")
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.manual_seed(20)
    # Long seqlen (>512K) can hit memory access fault. Suspect compiler issue
    # WA with shorter d and longer BLOCK_N
    if any(item > 524288 for item in n_ctx):
        BLOCK_N = 256
        d = 16

    assert batch == len(n_ctx)
    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        # print(f"s={s}")
        list_num_block_n = [
            (int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    sm_scale = 0.5

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        BLOCK_N,
        h_q,
        h_k,
        d,
        total_programs,
        init_dtype,
    )

    # Triton LeanAttention output
    la_out, ms = _persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        causal,
        batch,
        sm_scale,
        num_warps,
        waves_per_eu,
        return_lse=False,
    )

    # Calculate Pytorch reference output and LSE
    ref_out = reference_attention(q, k, v, n_ctx, n_ctx_q, sm_scale, causal)
    # ref_lse = reference_lse(q, k, n_ctx, n_ctx_q, sm_scale, causal)

    # print(f"ref_lse={ref_lse}")
    # print(f"lse={lse}")

    # print(f"la_out={la_out}")
    # print(f"ref_out={ref_out}")

    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
    # torch.testing.assert_close(ref_lse, lse, atol=1e-2, rtol=3e-3)


# NOTE: Tests where the workload < num_sms currently fail.
# You can elicit this behavior by decreasing `h` and `n_ctx`.
# Tests also appear to fail when n_ctx_q != n_ctx when causal=True.
@pytest.mark.skip(
    "Known issue with lean attention causes these tests to fail. La is a WIP."
)
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("h_q", [16])
@pytest.mark.parametrize("h_k", [16])
@pytest.mark.parametrize("n_ctx_q", [8192])
@pytest.mark.parametrize("n_ctx", [[8192]])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("causal", [(True), (False)])
@pytest.mark.parametrize("init_dtype", [torch.float16])
def test_persistent_lean_attention_outer(
    batch,
    h_q,
    h_k,
    n_ctx_q,
    n_ctx,
    d,
    init_dtype,
    causal,
):
    torch.manual_seed(20)

    sm_scale = 0.5
    config = _get_config(
        batch_size=batch,
        causal=causal,
    )
    sm_count = arch_info.get_num_sms()

    # Long seqlen (>512K) can hit memory access fault. Suspect compiler issue
    # WA with shorter d and longer BLOCK_N
    if any(item > 524288 for item in n_ctx):
        config["BLOCK_SIZE_N"] = 256
        d = 16

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        config["BLOCK_SIZE_N"],
        h_q,
        h_k,
        d,
        sm_count,
        init_dtype,
    )

    # Triton LeanAttention output
    # Override heads in config so kernel reads H_Q/H_K from config dict
    config = _get_config(batch_size=batch, causal=causal)
    config["H_Q"], config["H_K"] = h_q, h_k

    la_out = persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        batch,
        sm_scale,
        causal=causal,
        config=config,
    )

    # Calculate Pytorch refence output
    ref_out = reference_attention(q, k, v, n_ctx, n_ctx_q, sm_scale, causal)
    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)


def print_mismatches(ref_out, la_out, atol=1e-8, rtol=1e-5):
    # Check if shapes match first
    if ref_out.shape != la_out.shape:
        print(f"Shape mismatch! Reference: {ref_out.shape}, Actual: {la_out.shape}")
        return

    # Find mismatches using absolute and relative tolerance
    abs_diff = torch.abs(ref_out - la_out)
    rel_diff = abs_diff / (
        torch.abs(ref_out) + 1e-8
    )  # Add small epsilon to avoid division by zero

    mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)

    if not mismatch_mask.any():
        print("Tensors match within tolerances!")
        return

    # Get indices of mismatches
    mismatched_indices = torch.nonzero(mismatch_mask)

    print(f"Found {len(mismatched_indices)} mismatches:")
    for idx in mismatched_indices:
        idx_tuple = tuple(idx.tolist())
        print(f"At index {idx_tuple}:")
        print(f"  Reference: {ref_out[idx_tuple].item()}")
        print(f"  Actual:    {la_out[idx_tuple].item()}")
        print(f"  Abs diff:  {abs_diff[idx_tuple].item()}")
        print(f"  Rel diff:  {rel_diff[idx_tuple].item()}\n")

# (False, 16, 64, 8, 1, [8192] * 16, 128, 304, torch.float16, 128, 64, 1, 4),
def main():
    batch = 16
    causal = False
    h_q = 64
    h_k = 8
    n_ctx_q = 128
    n_ctx = [8192] * batch
    d = 128
    total_programs = 304
    init_dtype = torch.float16
    BLOCK_M = 128
    BLOCK_N = 64
    waves_per_eu = 1
    num_warps = 4
    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")
    print(f"causal={causal}, batch={batch}")
    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        list_num_block_n = [
            (int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    sm_scale = 0.5

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        BLOCK_N,
        h_q,
        h_k,
        d,
        total_programs,
        init_dtype,
    )

    # Triton LeanAttention output
    la_out, ms = _persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        causal,
        batch,
        sm_scale,
        num_warps,
        waves_per_eu,
    )
    # print(f"ms={ms}")

    ref_out = reference_attention(q, k, v, n_ctx, n_ctx_q, sm_scale, causal)

    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    try:
        torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
    except AssertionError:
        print("Assertion failed! Showing mismatches:")
        print_mismatches(ref_out, la_out, atol, rtol)
        raise  # Re-raise the exception after printing mismatches

    # torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(main())