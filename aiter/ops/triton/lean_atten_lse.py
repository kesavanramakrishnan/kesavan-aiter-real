# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Lean Attention
===============
This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480
Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.
Lean Attention is for both decode and prefill attention of transformer based models.

It currently supports ragged batching decode and prefill attention with causal=1

TO be added features:
- Add GQA support
- Misc
    - N_CTX with non-integer number of BLOCK_SIZE_N (pad zeros or add mask)
    -
"""

import torch
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from typing import Optional
from bisect import bisect_right
import triton
import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

LOG_TWO_E = 1.44269504  # log_2(e) value for softmax scaling
# Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs


@functools.lru_cache(maxsize=1024)
def _get_config(
    causal: bool,
    batch_size: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-LEANATTN-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    config = _get_config._config_dict["any"]
    return (
        config.copy()
    )  # return a copy to avoid mutation of stored config in LRU cache


def persistent_lean_attention(
    q: torch.Tensor,  # (B * seq_len_q, H, d)
    k: torch.Tensor,  # (total_seq_len_k, H, d) -> supports ragged batching
    v: torch.Tensor,  # (total_seq_len_k, H, d)
    Mp: torch.Tensor,  # temp buffer to store partial max during sm
    Lp: torch.Tensor,  # temp buffer to store partial se during sm
    Op: torch.Tensor,  # (total_programs, n_ctx_q, d)
    locks: torch.Tensor,  # (H, seq_len_q) -> used to synchronize blocks
    batch_num_block_n: torch.Tensor,  # (B) -> cumulative sum of BLOCK_N
    batch_size: int,
    sm_scale: torch.float16,
    causal: bool = True,  # causal masking
    config: Optional[dict] = None,
    return_lse: bool = False,
):
    """
    Lean Attention kernel.
    """
    _LOGGER.info(
        f"LEAN_ATTEN_PAGED: q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} Mp={tuple(Mp.shape)} Lp={tuple(Lp.shape)}  Op={tuple(Op.shape)}"
    )
    if config is None:
        config = _get_config(causal=causal, batch_size=batch_size)
    sm_count = arch_info.get_num_sms()

    return _persistent_lean_attention(
        q=q,
        k=k,
        v=v,
        Mp=Mp,
        Lp=Lp,
        Op=Op,
        locks=locks,
        batch_num_block_n=batch_num_block_n,
        total_programs=sm_count,
        BLOCK_M=config["BLOCK_SIZE_M"],
        BLOCK_N=config["BLOCK_SIZE_N"],
        causal=causal,
        batch_size=batch_size,
        sm_scale=sm_scale,
        num_warps=config["num_warps"],
        waves_per_eu=config["waves_per_eu"],
        config=config,
        return_lse=return_lse,
    )


# Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs
def _persistent_lean_attention(
    q: torch.Tensor,  # (B * seq_len_q, H, d)
    k: torch.Tensor,  # (total_seq_len_k, H, d) -> supports ragged batching
    v: torch.Tensor,  # (total_seq_len_k, H, d)
    Mp: torch.Tensor,  # temp buffer to store partial max during sm
    Lp: torch.Tensor,  # temp buffer to store partial se during sm
    Op: torch.Tensor,  # (total_programs, n_ctx_q, d) -> stores partial output values
    locks: torch.Tensor,  # (H, seq_len_q) -> used to synchronize blocks
    batch_num_block_n: torch.Tensor,  # (B) -> cumulative sum of BLOCK_N for each item in the batch
    total_programs: int,  # number of thread blocks (CTAs) to launch -> eq to num SMs
    BLOCK_M: int,  # seq_q tile size
    BLOCK_N: int,  # seq_k tile size
    causal: bool,  # causal masking
    batch_size: int,
    sm_scale: torch.float16,  # typically 1 / sqrt(d)
    num_warps: int,
    waves_per_eu: int,
    config: dict = {},
    return_lse: bool = False,
):
    """
    Inner kernel launching function.
    """
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
    # For MI300, BLOCK_M=128, BLOCK_N=64 is better for performance
    MASKED_BLOCKS = BLOCK_M // BLOCK_N

    if causal:
        # Only support BLOCK_M is multiple of BLOCK_N
        # TODO: add other scenarios
        assert BLOCK_M % BLOCK_N == 0

    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
    H = q.shape[1]
    H_K = k.shape[1]
    assert H % H_K == 0, "For GQA, the number of Q heads must be divisible by K/V heads"
    GQA_GROUP_SIZE = H // H_K

    qk_scale = sm_scale * LOG_TWO_E

    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        causal,
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        H_K,
        BLOCK_M,
        BLOCK_N,
        total_programs,
    )

    # print(
    #     f"high_load_wgs={high_load_wgs}, max_tiles_per_wg={max_tiles_per_wg}, tiles_per_head={tiles_per_head}"
    # )
    # print(
    #     f"total_programs={total_programs}, num_splits={num_splits}, even_split={even_split}"
    # )
    # print(f"num_m_blocks={num_m_blocks}, num_n_blocks={num_n_blocks}")

    max_output_tile_cnt = calculate_max_output_tiles_analytically(
        tiles_per_head=tiles_per_head,
        num_m_blocks=num_m_blocks,
        num_wgs=total_programs,
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        causal=causal,
        MASKED_BLOCKS=MASKED_BLOCKS,
        batch_size=batch_size,
        batch_num_block_n=batch_num_block_n,
    )

    max_output_tile_cnt = max_output_tile_cnt + 4

    grid = (total_programs, 1, 1)

    o = torch.empty_like(q, dtype=v.dtype)
    lse = None
    if return_lse:
        lse = torch.empty((q.shape[0], H), dtype=torch.float32, device=q.device)

    """
    kernel_timing = {
        "attn_fwd": {
            "start_event": torch.cuda.Event(enable_timing=True),
            "end_event": torch.cuda.Event(enable_timing=True),
            "ms": 0,
            "experiments": 0,
        },
    }
    kernel_timing["attn_fwd"]["start_event"].record()
    """
    la_kernel = la_persistent[grid](
        False,
        0,
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        o,
        lse if return_lse else torch.empty(1, device=q.device, dtype=torch.float32),
        batch_num_block_n,
        locks,
        q.stride(0),  # N_CTX_Q
        q.stride(1),  # H
        q.stride(2),  # Head_Dim
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        (lse.stride(0) if return_lse else 0),
        (lse.stride(1) if return_lse else 0),
        N_CTX_Q,
        Op.stride(0),  # total_programs
        Op.stride(1),  # n_ctx_q
        Op.stride(2),  # head_dim
        program_count=total_programs,
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MASKED_BLOCKS=MASKED_BLOCKS,
        batch_size=batch_size,
        causal=causal,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        # leanAttention params
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        max_output_tile_cnt=max_output_tile_cnt,
        num_warps=num_warps,
        num_stages=1,
        num_heads_q=H,
        num_heads_k=H_K,
        gqa_group_size=GQA_GROUP_SIZE,
        store_lse=return_lse,
        ln_2=(1.0 / LOG_TWO_E),
        use_64_indexing=((k.stride(0) * N_CTX_K) >= (1 << 31) or (v.stride(0) * N_CTX_K) >= (1 << 31)),
    )
    """
    kernel_timing["attn_fwd"]["end_event"].record()
    torch.cuda.synchronize()
    for k in ["attn_fwd"]:
        ms = kernel_timing[k]["start_event"].elapsed_time(kernel_timing[k]["end_event"])
        kernel_timing[k]["ms"] += ms
    total_ms = kernel_timing["attn_fwd"]["ms"]
    """
    # print(f"la kernel {la_kernel.n_regs} registers used, {la_kernel.n_spills} spills")
    ms = 0
    if return_lse:
        return o, lse, ms
    return o, ms


def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    BLOCK_M,
    BLOCK_N,
    num_SMs,
):
    """
    Calculates parameters for Lean Attention (num CTAs, num_m_blocks, num_n_blocks, etc.))
    """
    ##### Lean Attention: Calculate Splits and Tile Sizes #####
    ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    # print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
    # print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
    # print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    # print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
        # Does not support ragged batch for causal.
        tiles_per_head = tiles_per_head * batch_size
    else:
        # Decode or Not Causal
        tiles_per_head = num_m_blocks * num_n_blocks

    # Schedule by Q heads so every Q head produces an output tile; map to KV via gqa_group_size later
    total_tiles = tiles_per_head * num_heads

    # StreamK Lean has as many threadblocks as SMs
    # This should be a function of tile size and number of scratchpad space
    # LeanAttention assign 3 CTAs per SM (bounded by LDS size)
    if total_tiles <= num_SMs:
        lean_griddimz = min(
            (total_tiles + 1) // 2,
            (32 * total_tiles + num_n_blocks - 1) // num_n_blocks,
        )
    else:
        lean_griddimz = num_SMs  # CTA launch grid
    # if (total_tiles <= 2 * 2 * num_SMs):
    #    lean_griddimz = min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks)
    # else:
    #    lean_griddimz = min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks)

    # Max number lean tiles per task block (CTA)
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

    # Find max number of splits
    num_splits = 0
    even_split = False
    if total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
    else:
        even_split = False
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )

    # high_load_tbs is the remainder of total_tile / num_cta
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    # Needed for causal. This is (per batch n_ctx) // BLOCK_N
    num_n_blocks = num_n_blocks // batch_size

    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
        num_splits,
        even_split,
    )


def calculate_max_output_tiles_analytically(
    tiles_per_head: int,
    num_m_blocks: int,
    num_wgs: int,
    high_load_wgs: int,
    max_tiles_per_wg: int,
    causal: bool,
    MASKED_BLOCKS: int,
    batch_size: int,
    batch_num_block_n,
):
    """
    Calculates the maximum number of output tiles any single workgroup will process
    using a fast, analytical method with binary search.
    """
    if num_wgs == 0:
        return 0

    m_block_boundaries = []
    if causal:
        # Pre-compute the boundaries of each M-block's workload for a single head.
        # This list will be used for binary searches.
        total_blocks = 0
        for i in range(num_m_blocks):
            pair_idx = i // 2
            q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
            task_size = (q_block_idx + 1) * MASKED_BLOCKS
            total_blocks += task_size
            m_block_boundaries.append(total_blocks)

    max_total_output_tiles = 0
    # Loop through each workgroup to find the one that spans the most output tiles.
    for wg_id in range(num_wgs):
        total_output_tiles_for_wg = 0

        # Determine the range of global lean tile indices for this WG
        if wg_id < high_load_wgs:
            start_iter = max_tiles_per_wg * wg_id
            end_iter = start_iter + max_tiles_per_wg
        else:
            start_iter = (max_tiles_per_wg - 1) * (
                wg_id - high_load_wgs
            ) + high_load_wgs * max_tiles_per_wg
            end_iter = start_iter + (max_tiles_per_wg - 1)

        start_head = start_iter // tiles_per_head
        end_head = (end_iter - 1) // tiles_per_head

        # Loop through each head this workgroup touches
        for head_idx in range(start_head, end_head + 1):
            head_start_iter = head_idx * tiles_per_head

            # Find the intersection of the WG's range and the current head's range
            wg_start_in_head = max(start_iter, head_start_iter)
            wg_end_in_head = min(end_iter, head_start_iter + tiles_per_head)

            if not causal:
                # For non-causal, each head is one output tile (match baseline implementation)
                total_output_tiles_for_wg += 1
                continue

            # --- Causal Logic using Binary Search ---
            # Convert to indices relative to the start of the head's workload
            relative_start = wg_start_in_head - head_start_iter
            relative_end = wg_end_in_head - head_start_iter

            # Use binary search to find which M-block the start and end tiles fall into
            start_m_idx = bisect_right(m_block_boundaries, relative_start)
            end_m_idx = bisect_right(m_block_boundaries, relative_end - 1)

            # The number of output tiles is the number of boundaries crossed
            tiles_in_this_head = (end_m_idx - start_m_idx) + 1
            total_output_tiles_for_wg += tiles_in_this_head

        max_total_output_tiles = max(max_total_output_tiles, total_output_tiles_for_wg)

    return max_total_output_tiles


@triton.jit
def find_group(x, MASKED_BLOCKS: tl.constexpr, num_m_blocks: tl.constexpr):
    total_blocks_processed = 0
    final_q_block_idx = 0
    final_task_size = 0
    final_total_blocks = 0
    found = False
    # Iterate through the tasks in the desired ping-pong order
    for i in range(0, num_m_blocks):
        # Determine the actual Q block index for the current task in the ping-pong sequence
        pair_idx = i // 2
        if (i % 2) == 0:
            # Even tasks are from the top (e.g., 0, 1, 2...)
            q_block_idx = pair_idx
        else:
            # Odd tasks are from the bottom (e.g., N-1, N-2, ...)
            q_block_idx = num_m_blocks - 1 - pair_idx

        # Calculate the size of this task's workload (number of K/V blocks to process)
        task_size = (q_block_idx + 1) * MASKED_BLOCKS

        # Check if the global tile `x` falls within this task's range
        if total_blocks_processed + task_size > x and not found:
            # We found it. Return the Q index, the size of its workload, and its starting tile.
            final_q_block_idx, final_task_size, final_total_blocks = (
                q_block_idx,
                task_size,
                total_blocks_processed,
            )
            found = True

        # Add this task's size to the running total and move to the next
        total_blocks_processed += task_size
    # Return values
    return final_q_block_idx, final_task_size, final_total_blocks


@triton.jit
def _attention_inner(
    q,
    k_ptrs,
    v_ptrs,
    stride_vn,
    stride_kn,
    m_i,
    l_i,
    acc,
    qk_scale,
    causal,
    q_start_m,
    b_seq_size,
    offs_m,
    offs_n,
    BLOCK_M,
    BLOCK_N,
    HEAD_DIM,
    local_iter,
    local_iter_end,
    use_64_indexing: tl.constexpr,
):
    """
    Performs attention calculation for an (maybe partial) output tile
    """
    for l_iter in range(local_iter, local_iter_end):
        k = tl.load(k_ptrs)
        qk = tl.dot(q, k) * qk_scale

        if causal:
            # Get the starting column index of the current K block
            k_start_n = (b_seq_size + l_iter) * BLOCK_N
            # Create mask based on absolute sequence positions
            mask = (q_start_m + offs_m[:, None]) >= (k_start_n + offs_n[None, :])
            # Apply the mask
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        if causal:
            # Safe exp2 calculation to avoid NaNs
            p_arg = qk - m_ij[:, None]
            p_arg = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), p_arg)
            p = tl.math.exp2(p_arg)

            # Safe scaling factor calculation
            alpha_arg = m_i - m_ij
            alpha_arg = tl.where(m_ij == float("-inf"), 0.0, alpha_arg)
            alpha = tl.math.exp2(alpha_arg)
        else:
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)  # p.shape = [BLOCK_M, BLOCK_N]
            alpha = tl.math.exp2(m_i - m_ij)

        # Update accumulator
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs)
        acc += tl.dot(p.to(v.dtype), v)

        # Update stats
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        m_i = m_ij.to(m_i.dtype)

        if use_64_indexing:
            BLOCK_N64 = tl.full((), BLOCK_N, tl.int64)
            stride_kn64 = tl.full((), stride_kn, tl.int64)
            stride_vn64 = tl.full((), stride_vn, tl.int64)
            # update k/v pointer
            v_ptrs += BLOCK_N64 * stride_vn64
            k_ptrs += BLOCK_N64 * stride_kn64
        else:
            v_ptrs += BLOCK_N * stride_vn
            k_ptrs += BLOCK_N * stride_kn

    return m_i, l_i, acc


@triton.jit
def la_persistent(
    is_pod,
    pod_pid,
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    LSE,
    batch_num_block_n,
    locks,
    stride_qm,  # n_ctx_q
    stride_qh,  # Head
    stride_qk,  # head_dim
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,  # n_ctx_q
    stride_oh,  # Head
    stride_on,  # head_dim
    stride_lsem,  # n_ctx_q for LSE
    stride_lseh,  # Head for LSE
    n_ctx_q_rows,
    stride_oph,  # total_programs
    stride_opm,  # n_ctx_q
    stride_opn,  # head_dim
    program_count,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    max_output_tile_cnt: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_k: tl.constexpr,
    gqa_group_size: tl.constexpr,
    store_lse: tl.constexpr,
    ln_2,
    use_64_indexing: tl.constexpr,
):
    if is_pod:
        current_pid = pod_pid
    else:
        current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (
            current_pid - high_load_wgs
        ) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

    tl.assume(stride_qm > 0)  # n_ctx_q
    tl.assume(stride_qh > 0)  # Head
    tl.assume(stride_qk > 0)  # head_dim
    tl.assume(stride_kn > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_kk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_om > 0)  # n_ctx_q
    tl.assume(stride_oh > 0)  # Head
    tl.assume(stride_on > 0)  # head_dim
    tl.assume(stride_oph > 0)  # total_programs
    tl.assume(stride_opm > 0)  # n_ctx_q
    tl.assume(stride_opn > 0)  # head_dim

    for i in tl.static_range(max_output_tile_cnt + 1):
        if iter < cta_end_tile_gid:
            iter = la_persistent_inner(
                Q,
                K,
                V,
                qk_scale,
                Mp,
                Lp,
                Op,
                Out,
                LSE,
                batch_num_block_n,
                locks,
                stride_qm,  # n_ctx_q
                stride_qh,  # Head
                stride_qk,  # head_dim
                stride_kn,
                stride_kh,
                stride_kk,
                stride_vn,
                stride_vh,
                stride_vk,
                stride_om,  # n_ctx_q
                stride_oh,  # Head
                stride_on,  # head_dim
                stride_lsem,
                stride_lseh,
                n_ctx_q_rows,
                stride_oph,  # total_programs
                stride_opm,  # n_ctx_q
                stride_opn,  # head_dim
                program_count,
                iter=iter,
                cta_end_tile_gid=cta_end_tile_gid,
                current_pid=current_pid,
                HEAD_DIM=HEAD_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                MASKED_BLOCKS=MASKED_BLOCKS,
                batch_size=batch_size,
                causal=causal,
                num_m_blocks=num_m_blocks,
                num_n_blocks=num_n_blocks,
                # leanAttention params
                high_load_wgs=high_load_wgs,
                max_tiles_per_wg=max_tiles_per_wg,
                tiles_per_head=tiles_per_head,
                num_splits=num_splits,
                num_heads_q=num_heads_q,
                num_heads_k=num_heads_k,
                gqa_group_size=gqa_group_size,
                store_lse=store_lse,
                ln_2=ln_2,
                use_64_indexing=use_64_indexing,
            )


@triton.jit
def la_persistent_inner(
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    LSE,
    batch_num_block_n,
    locks,
    stride_qm,  # n_ctx_q
    stride_qh,  # Head
    stride_qk,  # head_dim
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,  # n_ctx_q
    stride_oh,  # Head
    stride_on,  # head_dim
    stride_lsem,
    stride_lseh,
    n_ctx_q_rows,
    stride_oph,  # total_programs
    stride_opm,  # n_ctx_q
    stride_opn,  # head_dim
    program_count,
    iter,
    cta_end_tile_gid,
    current_pid,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    num_heads_q: tl.constexpr,
    num_heads_k: tl.constexpr,
    gqa_group_size: tl.constexpr,
    store_lse: tl.constexpr,
    ln_2,
    use_64_indexing: tl.constexpr,
):

    tl.assume(stride_qm > 0)  # n_ctx_q
    tl.assume(stride_qh > 0)  # Head
    tl.assume(stride_qk > 0)  # head_dim
    tl.assume(stride_kn > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_kk > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_vk > 0)
    tl.assume(stride_om > 0)  # n_ctx_q
    tl.assume(stride_oh > 0)  # Head
    tl.assume(stride_on > 0)  # head_dim
    tl.assume(stride_oph > 0)  # total_programs
    tl.assume(stride_opm > 0)  # n_ctx_q
    tl.assume(stride_opn > 0)  # head_dim

    # Loop context length
    # while iter < cta_end_tile_gid:
    # Calculate index of current head output tile
    # The tiles_per_head is the sum of # BLOCK_N in K/V sequence of all batches
    tile_head_idx = iter // tiles_per_head
    # To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
    # [tile_iter, tile_iter_end) are in the form of global tile id
    if causal:
        tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
        # Does not support ragged batching. All requests in the batch have the same context length (per_head_tile_size)
        # tiles_per_head: total sum of # BLOCK_N in K/V sequence of all batches
        # per_head_tile_size: per head # BLOCK_N of each output tile
        per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
            iter
            - (tile_head_idx * tiles_per_head)
            - (tile_batch_idx * (tiles_per_head // batch_size)),
            MASKED_BLOCKS,
            num_m_blocks,
        )
        tile_iter = (
            tile_head_idx * tiles_per_head
            + (tile_batch_idx * (tiles_per_head // batch_size))
            + total_blocks
        )
        tile_iter_end = tile_iter + (per_head_tile_size)
        tile_idx = (
            tile_head_idx * batch_size + tile_batch_idx
        ) * num_m_blocks + per_head_tile_idx
    else:
        tile_idx = (
            tile_head_idx * batch_size
        )  # Output tile idx, 1 output tile per head per batch
        tile_iter = tile_head_idx * tiles_per_head
        if batch_size == 1:
            req_size = tiles_per_head
        else:
            req_size = tl.load(batch_num_block_n)
        tile_iter_end = tile_iter + req_size
        for b in range(1, batch_size):
            next_req_size = tl.load(batch_num_block_n + b)
            local_head_iter = iter - (tile_head_idx * tiles_per_head)
            if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                tile_iter = tile_iter + req_size
                tile_idx = tile_idx + b
                tile_iter_end = tile_iter + (next_req_size - req_size)
            req_size = next_req_size
    # Local lean tile ID within a loop of an output tile
    local_iter = iter - tile_iter
    local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

    if iter == tile_iter:
        host_block = True
    else:
        host_block = False
    # finishing_block: the output tile is finished within this block
    if cta_end_tile_gid >= tile_iter_end:
        finishing_block = True
    else:
        finishing_block = False

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    if causal:
        b_seq_size = tile_batch_idx * num_n_blocks
    else:
        tile_batch_idx = tile_idx % batch_size
        b_seq_size = 0
        if tile_batch_idx > 0:
            b_seq_size = tl.load(
                batch_num_block_n + tile_batch_idx - 1
            )  # Previous batch size

    if gqa_group_size == 1:
        tile_khead_idx = tile_head_idx
    else:
        tile_khead_idx = tile_head_idx // gqa_group_size

    # if use_64_indexing:
    BLOCK_N64 = tl.full((), BLOCK_N, tl.int64)
    stride_kn64 = tl.full((), stride_kn, tl.int64)
    stride_vn64 = tl.full((), stride_vn, tl.int64)
    bn64 = tl.full((), b_seq_size, tl.int64) + tl.full((), local_iter, tl.int64)
    k_offs = (
        (bn64 * BLOCK_N64) * stride_kn64
        + tile_khead_idx * stride_kh
        + offs_n[None, :] * stride_kn
        + offs_k[:, None] * stride_kk
    )
    v_offs = (
        (bn64 * BLOCK_N64) * stride_vn64
        + tile_khead_idx * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_k[None, :] * stride_vk
    )
    # else:
    #     bn32 = b_seq_size + local_iter
    #     k_offs = (
    #         (bn32 * BLOCK_N) * stride_kn
    #         + tile_khead_idx * stride_kh
    #         + offs_n[None, :] * stride_kn
    #         + offs_k[:, None] * stride_kk
    #     )
    #     v_offs = (
    #         (bn32 * BLOCK_N) * stride_vn
    #         + tile_khead_idx * stride_vh
    #         + offs_n[:, None] * stride_vn
    #         + offs_k[None, :] * stride_vk
    #     )

    k_ptrs = K + k_offs
    k_ptrs = tl.multiple_of(k_ptrs, (16, 1))
    v_ptrs = V + v_offs
    v_ptrs = tl.multiple_of(v_ptrs, (1, 16))

    if causal:
        q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
        q_start_m = q_idx * BLOCK_M
    else:
        q_idx = tile_batch_idx
        q_start_m = 0

    q_offs = (
        q_idx * BLOCK_M * stride_qm
        + tile_head_idx * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_k[None, :] * stride_qk
    )
    q_ptrs = Q + q_offs
    q_ptrs = tl.multiple_of(q_ptrs, (1, 16))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(q_ptrs)

    m_i, l_i, acc = _attention_inner(
        q,
        k_ptrs,
        v_ptrs,
        stride_vn,
        stride_kn,
        m_i,
        l_i,
        acc,
        qk_scale,
        causal,
        q_start_m,
        b_seq_size,
        offs_m,
        offs_n,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        local_iter,
        local_iter_end,
        use_64_indexing=True,
    )

    # initialize pointer to m and l
    m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # lean output tile epilogue
    # Compute per-row mask to avoid OOB when n_ctx_q < BLOCK_M
    # q_start_m is a global row index across all batches. Convert to per-batch row index.
    batch_base_rows = tile_batch_idx * n_ctx_q_rows
    q_start_m_in_batch = q_start_m - batch_base_rows
    rows_remaining = n_ctx_q_rows - q_start_m_in_batch
    rows_remaining = tl.maximum(0, rows_remaining)
    rows_in_tile = tl.minimum(BLOCK_M, rows_remaining)
    if not causal:
        rows_in_tile = BLOCK_M
    mask_m = offs_m < rows_in_tile
    is_full_tile = rows_in_tile == BLOCK_M

    if not host_block:
        # Update pointers of partial results Mp[cta], Lp[cta], Op[cta]
        mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
        lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
        op_ptrs = (
            Op
            + current_pid * stride_oph  # stride_oph is total_program dimension
            + offs_m[:, None] * stride_opm
            + offs_k[None, :] * stride_opn
        )
        
        if is_full_tile:
            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
        else:
            tl.store(mp_ptrs, m_i, mask=mask_m, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, mask=mask_m, cache_modifier=".wt")
            tl.store(op_ptrs, acc, mask=mask_m[:, None], cache_modifier=".wt")
        tl.debug_barrier()
        # tl.store(locks + current_pid, 1, cache_modifier=".wt")
        # According to streamK gemm, store + cache_modifier won't work universally
        # atomic_xchg is better solution but a less performant variant
        tl.atomic_xchg(locks + current_pid, 1)

    if host_block:  # and finishing_block:
        # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
        # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
        acc_reshaped = tl.reshape(acc, (BLOCK_M, 2, HEAD_DIM // 2))
        acc_permuted = tl.permute(acc_reshaped, (0, 2, 1))
        acc0, acc1 = tl.split(acc_permuted)

        # o_h_offs = (
        #    q_idx * BLOCK_M * stride_om
        #    + tile_head_idx * stride_oh
        #    + offs_m[:, None] * stride_om
        #    + offs_k[None, :] * stride_on
        # )
        # o_ptrs = Out + o_h_offs

        if not finishing_block:
            # if host not finishing_block: # another CTA is processing the end of the output tile and store partial results
            last_cta = current_pid + 1
            temp_end_gid = cta_end_tile_gid
            split = 1
            while (split < num_splits) and (temp_end_gid < tile_iter_end) and (last_cta < program_count):
                if last_cta < high_load_wgs:
                    if (tile_iter_end - temp_end_gid) < max_tiles_per_wg:
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg
                else:
                    if (tile_iter_end - temp_end_gid) < (max_tiles_per_wg - 1):
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg - 1

                last_cta += 1
                split += 1
            # Next, load nonHost partial restult
            for cta in range((current_pid + 1), tl.minimum(last_cta, program_count)):
                # According to treamK gemm, atomic_cas is universal solution but less performant
                while tl.atomic_cas(locks + cta, 1, 1) != 1:
                    # while tl.load(locks + cta, cache_modifier=".cv", volatile=True) != 1:
                    pass

                # Partial results are stored in [nonHost, Host-nonFinishing] layout
                offs_mplp = cta * BLOCK_M + offs_m
                mp_ptrs = Mp + offs_mplp
                lp_ptrs = Lp + offs_mplp
                """
                op_h_offs = (
                    cta * stride_oph
                    + offs_m[:, None] * stride_opm
                    + offs_k[None, :] * stride_opn
                )
                op_ptrs = Op + op_h_offs
                """
                op_ptrs0 = (
                    Op
                    + cta * stride_oph
                    + offs_m[:, None] * stride_opm
                    + tl.arange(0, HEAD_DIM // 2)[None, :] * stride_opn
                )
                op_ptrs1 = (
                    Op
                    + cta * stride_oph
                    + offs_m[:, None] * stride_opm
                    + (tl.arange(0, HEAD_DIM // 2)[None, :] + HEAD_DIM // 2) * stride_opn
                )

                m_cta = tl.load(mp_ptrs)
                l_cta = tl.load(lp_ptrs)
                # acc_cta = tl.load(op_ptrs)
                acc_cta0 = tl.load(op_ptrs0)
                acc_cta1 = tl.load(op_ptrs1)

                # m_i is the host CTA's m, m_cta is other nonHost CTA's m
                m_new = tl.maximum(m_cta, m_i)
                alpha = tl.math.exp2(m_cta - m_new)
                alpha1 = tl.math.exp2(m_i - m_new)
                l_new = alpha * l_cta + alpha1 * l_i
                # acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                acc0 = acc_cta0 * alpha[:, None] + acc0 * alpha1[:, None]
                acc1 = acc_cta1 * alpha[:, None] + acc1 * alpha1[:, None]
                # update m, l
                m_i = m_new
                l_i = l_new

        # host CTA write final result to memory
        # acc = acc / l_i[:, None]
        # tl.store(o_ptrs, acc.to(Out.type.element_ty))
        o_ptrs0 = (
            Out
            + q_idx * BLOCK_M * stride_om
            + tile_head_idx * stride_oh
            + offs_m[:, None] * stride_om
            + tl.arange(0, HEAD_DIM // 2)[None, :] * stride_on
        )
        o_ptrs1 = (
            Out
            + q_idx * BLOCK_M * stride_om
            + tile_head_idx * stride_oh
            + offs_m[:, None] * stride_om
            + (tl.arange(0, HEAD_DIM // 2)[None, :] + HEAD_DIM // 2) * stride_on
        )

        acc0 = acc0 / l_i[:, None]
        acc1 = acc1 / l_i[:, None]
        if is_full_tile:
            tl.store(o_ptrs0, acc0.to(Out.type.element_ty))
            tl.store(o_ptrs1, acc1.to(Out.type.element_ty))
        else:
            tl.store(o_ptrs0, acc0.to(Out.type.element_ty), mask=mask_m[:, None])
            tl.store(o_ptrs1, acc1.to(Out.type.element_ty), mask=mask_m[:, None])

        if store_lse:
            lse_vals = m_i * ln_2 + tl.log(l_i)
            lse_ptrs = (
                LSE
                + (q_idx * BLOCK_M + offs_m) * stride_lsem
                + tile_head_idx * stride_lseh
            )
            tl.store(lse_ptrs, lse_vals, mask=mask_m)

    # update iter
    iter = iter + (local_iter_end - local_iter)

    return iter