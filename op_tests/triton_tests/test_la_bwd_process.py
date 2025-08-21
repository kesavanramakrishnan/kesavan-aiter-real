import pytest
import torch
import triton
import sys
import os

from aiter.ops.triton.lean_atten_bwd_clean import persistent_lean_attention_bwd
from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward
from aiter.ops.triton.mha import flash_attn_func

# Define data types for testing, including float32 on capable hardware
DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    DTYPES.append(torch.float32)

# Define tolerance levels for result comparisons
ATOL = {torch.float16: 1e-2, torch.bfloat16: 2e-2, torch.float32: 1e-4}
RTOL = {torch.float16: 1e-2, torch.bfloat16: 2e-2, torch.float32: 1e-4}

# Benchmark-derived non-varlen configs (BATCH, HQ, HK, N_CTX_Q, N_CTX_K)
# Sourced from nonvarlen_benchmark_configs used in fused-attention bwd benchmarks
BENCH_NONVARLEN_CONFIGS = [
    (16, 16, 16, 1024, 1024),
    (8, 16, 16, 2048, 2048),
    (4, 16, 16, 4096, 4096),
    (2, 16, 16, 8192, 8192),
    # (8, 16, 16, 1024, 4096),
    # (1, 16, 16, 4096, 16384),
    # (2, 48, 48, 1024, 1024),
    # (2, 48, 48, 2048, 1024),
    # (2, 48, 48, 4096, 8192),
    # (2, 48, 48, 8192, 4096),
    
    # (2, 48, 48, 16384, 8192),
    # (8, 16, 16, 1989, 15344),
    # (4, 16, 16, 4097, 163),
    # (2, 16, 16, 8122, 2159),
    # (1, 16, 16, 16281, 7),
    # (2, 48, 48, 1021, 1020),
    # (2, 48, 48, 2001, 2048),
    # (2, 48, 48, 3996, 9639),
    # (2, 48, 48, 8181, 1021),
]


HEAD_SZ_128_BENCH_CONFIGS = [
    (16, 16, 16, 1024, 1024),
    # (1, 16, 16, 512, 512),
    (4, 16, 16, 4096, 4096),
    (2, 16, 16, 8192, 8192),
    (8, 16, 16, 1024, 4096),
    (1, 16, 16, 4096, 16384),
    (2, 48, 48, 1024, 1024),
    (2, 48, 48, 2048, 1024),
    (2, 48, 48, 4096, 8192),
    (2, 48, 48, 8192, 4096),
    # (2, 48, 48, 16384, 8192),
    # (2, 48, 48, 8192, 4096),
    # (2, 48, 48, 16384, 8192),
    # (2, 48, 48, 8192, 16384),
    # (2, 48, 48, 16384, 16384),
]


def _run_la_bwd_process_test(
    BATCH: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    causal: bool,
    dtype: torch.dtype,
    device: str = "cuda",
):
    torch.manual_seed(2024)

    # Define tensor shapes
    q_shape = (BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_SZ)
    k_shape = (BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ)
    v_shape = (BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_SZ)

    # Initialize input tensors
    q = torch.randn(q_shape, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(k_shape, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(v_shape, dtype=dtype, device=device, requires_grad=True)
    sm_scale = HEAD_SZ**-0.5

    # --- PyTorch Reference Implementation ---
    gqa_group_size = NUM_Q_HEADS // NUM_K_HEADS
    k_ref_pt = k.repeat_interleave(gqa_group_size, dim=1)
    v_ref_pt = v.repeat_interleave(gqa_group_size, dim=1)

    # Manual forward pass to get softmax_lse for Triton kernels
    q_ref = q.float()
    k_ref = k.repeat_interleave(gqa_group_size, dim=1).float()
    v_ref = v.repeat_interleave(gqa_group_size, dim=1).float()

    scores = (q_ref @ k_ref.transpose(-2, -1)) * sm_scale
    if causal:
        # The alignment of the causal mask depends on the difference between
        # the query and key sequence lengths.
        # This is the reference implementation of the causal mask.
        mask = torch.ones(SEQLEN_Q, SEQLEN_K, device=device, dtype=torch.bool).tril(
            diagonal=(SEQLEN_K - SEQLEN_Q)
        )
        scores.masked_fill_(~mask, -float("inf"))

    softmax_lse_ref = torch.logsumexp(scores, dim=-1).to(dtype)
    p = torch.softmax(scores, dim=-1)
    p = torch.nan_to_num(p)
    o_ref = (p @ v_ref).to(dtype)

    # Backward pass
    do = torch.randn_like(o_ref)
    o_ref.backward(do)
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad, k.grad, v.grad = None, None, None

    # --- Triton Implementations ---
    # Prepare tensors in BSHD format (batch, seqlen, heads, dim)
    q_bsnh = q.permute(0, 2, 1, 3).contiguous()
    k_bsnh = k.permute(0, 2, 1, 3).contiguous()
    v_bsnh = v.permute(0, 2, 1, 3).contiguous()
    o_bsnh = o_ref.permute(0, 2, 1, 3).contiguous()
    do_bsnh = do.permute(0, 2, 1, 3).contiguous()

    # --- Lean Attention Backward Pass ---
    # Flatten to Lean shapes: q = [B*NQ, H, D], k/v = [NK, H, D] to match DB keys (NK per-seq)
    q_flat = q_bsnh.reshape(BATCH * SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ).contiguous()
    k_flat = k_bsnh.reshape(BATCH * SEQLEN_K, NUM_K_HEADS, HEAD_SZ).contiguous()
    v_flat = v_bsnh.reshape(BATCH * SEQLEN_K, NUM_K_HEADS, HEAD_SZ).contiguous()
    o_flat = o_bsnh.reshape(BATCH * SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ).contiguous()
    do_flat = do_bsnh.reshape(BATCH * SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ).contiguous()

    dq_flat = torch.zeros_like(q_flat)
    dk_flat = torch.zeros_like(k_flat)
    dv_flat = torch.zeros_like(v_flat)

    # Compute batch_num_block_n for non-causal multi-batch decode
    # default BLOCK_N for tests (avoid loading json config)
    BLOCK_N = 64
    num_n_blocks = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N
    batch_num_block_n = None
    if (not causal) and (BATCH > 1):
        batch_num_block_n = (
            torch.arange(1, BATCH + 1, device=device, dtype=torch.int32) * num_n_blocks
        )

    persistent_lean_attention_bwd(
        q=q_flat, k=k_flat, v=v_flat, do=do_flat, o=o_flat,
        softmax_lse=softmax_lse_ref,
        dq=dq_flat, dk=dk_flat, dv=dv_flat,
        batch_num_block_n=batch_num_block_n,
        batch_size=BATCH, sm_scale=sm_scale, causal=causal,
        seqlen_k=SEQLEN_K,
        num_programs=304,
    )

    dq_la_bsnh = dq_flat.view(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ).contiguous()
    dk_la_bsnh = dk_flat.view(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ).contiguous()
    dv_la_bsnh = dv_flat.view(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ).contiguous()

    # --- Flash Attention Backward Pass ---
    dq_flash_bsnh = torch.zeros_like(q_bsnh)
    dk_flash_bsnh = torch.zeros_like(k_bsnh)
    dv_flash_bsnh = torch.zeros_like(v_bsnh)

    flash_attn_onekernel_backward(
        do=do_bsnh, q=q_bsnh, k=k_bsnh, v=v_bsnh, o=o_bsnh,
        softmax_lse=softmax_lse_ref,
        dq=dq_flash_bsnh, dk=dk_flash_bsnh, dv=dv_flash_bsnh,
        dbias=None, sm_scale=sm_scale, alibi_slopes=None, causal=causal,
        cu_seqlens_q=None, cu_seqlens_k=None,
        max_seqlen_q=SEQLEN_Q, max_seqlen_k=SEQLEN_K,
        dropout_p=0.0,
    )

    # Convert outputs back to BHSD format for comparison
    dq_la = dq_la_bsnh.permute(0, 2, 1, 3)
    dk_la = dk_la_bsnh.permute(0, 2, 1, 3)
    dv_la = dv_la_bsnh.permute(0, 2, 1, 3)

    dq_flash = dq_flash_bsnh.permute(0, 2, 1, 3)
    dk_flash = dk_flash_bsnh.permute(0, 2, 1, 3)
    dv_flash = dv_flash_bsnh.permute(0, 2, 1, 3)

    # --- Assertions ---
    atol, rtol = ATOL[dtype], RTOL[dtype]

    # print("\n--- dK Comparison (Lean Attn vs Flash Attn) ---")
    # dk_diff = torch.abs(dk_la - dk_flash)
    # # print(dk_diff)
    # dk_tol = atol + rtol * torch.abs(dk_flash)
    # dk_mismatch_pct = (dk_diff > dk_tol).float().mean().item() * 100
    # print(f"% mismatched (dK): {dk_mismatch_pct:.6f}%")

    # print("\n--- dV Comparison (Lean Attn vs Flash Attn) ---")
    # dv_diff = torch.abs(dv_la - dv_flash)
    # dv_tol = atol + rtol * torch.abs(dv_flash)
    # dv_mismatch_pct = (dv_diff > dv_tol).float().mean().item() * 100
    # print(f"% mismatched (dV): {dv_mismatch_pct:.6f}%")

    # print("\n--- dQ Comparison (Lean Attn vs Flash Attn) ---")
    # dq_diff = torch.abs(dq_la - dq_flash)
    # dq_tol = atol + rtol * torch.abs(dq_flash)
    # dq_mismatch_pct = (dq_diff > dq_tol).float().mean().item() * 100
    # print(f"% mismatched (dQ): {dq_mismatch_pct:.6f}%")
    # print("dq_la", dq_la)
    # print("dq_flash", dq_flash)
    #Compare Lean Attention with Flash Attention
    # print("dq_la", dq_la)
    # print("dq_flash", dq_flash)

    
    # torch.testing.assert_close(dq_la, dq_flash, atol=atol, rtol=rtol, msg="dQ (Lean Attn vs Flash Attn)")
    torch.testing.assert_close(dk_la, dk_ref, atol=atol, rtol=rtol, msg="dK (Lean Attn vs Flash Attn)")
    torch.testing.assert_close(dv_la, dv_ref, atol=atol, rtol=rtol, msg="dV (Lean Attn vs Flash Attn)")

# @pytest.mark.parametrize("BATCH", [1, 2, 4, 8, 16])
# @pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16,16), (48,48)])
# @pytest.mark.parametrize("HEAD_SZ", [64])
# @pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", ([(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (1024, 4096)]))
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dtype", [torch.float16])

@pytest.mark.parametrize("BATCH", [16])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(16,16)])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", ([(1024, 1024)]))
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_la_bwd_vs_flash_bwd(
    BATCH: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    causal: bool,
    dtype: torch.dtype,
    device: str = "cuda",
):
    _run_la_bwd_process_test(
        BATCH,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        HEAD_SZ,
        SEQLEN_Q,
        SEQLEN_K,
        causal,
        dtype,
        device,
    )


@pytest.mark.parametrize(
    "BATCH, NUM_Q_HEADS, NUM_K_HEADS, SEQLEN_Q, SEQLEN_K",
    HEAD_SZ_128_BENCH_CONFIGS,
)
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_la_bwd_vs_flash_bwd_head_sz_128(
    BATCH: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    HEAD_SZ: int,
    causal: bool,
    dtype: torch.dtype,
    device: str = "cuda",
):
    _run_la_bwd_process_test(
        BATCH,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        HEAD_SZ,
        SEQLEN_Q,
        SEQLEN_K,
        causal,
        dtype,
        device,
    )

@pytest.mark.parametrize(
    "BATCH, NUM_Q_HEADS, NUM_K_HEADS, SEQLEN_Q, SEQLEN_K",
    BENCH_NONVARLEN_CONFIGS,
)
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_la_bwd_vs_flash_bwd_bench_nonvarlen(
    BATCH: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    HEAD_SZ: int,
    causal: bool,
    dtype: torch.dtype,
    device: str = "cuda",
):
    _run_la_bwd_process_test(
        BATCH,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        HEAD_SZ,
        SEQLEN_Q,
        SEQLEN_K,
        causal,
        dtype,
        device,
    )


def main_la():
    """Main function for manual testing of individual configurations"""
    import sys
    
    # Default configuration - you can modify these values for testing
    BATCH = 2
    NUM_Q_HEADS = 16
    NUM_K_HEADS = 16
    HEAD_SZ = 128
    SEQLEN_Q = 16384
    SEQLEN_K = 16384
    causal = False
    dtype = torch.float16
    device = "cuda"
    
    print(f"Testing Lean Attention Backward configuration:")
    print(f"  BATCH: {BATCH}")
    print(f"  NUM_Q_HEADS: {NUM_Q_HEADS}")
    print(f"  NUM_K_HEADS: {NUM_K_HEADS}")
    print(f"  HEAD_SZ: {HEAD_SZ}")
    print(f"  SEQLEN_Q: {SEQLEN_Q}")
    print(f"  SEQLEN_K: {SEQLEN_K}")
    print(f"  causal: {causal}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    print("-" * 50)
    
    # Ensure tuned DB is picked up like in bench runs
    os.environ.setdefault("AITER_BWD_TUNED_DB", "la_configs/bwd_long_seq.json")
    os.environ.setdefault("AITER_BWD_USE_TUNED_GRID", "1")

    # Shared deterministic inputs
    def _generate_inputs(B, HQ, HK, D, SQ, SK, dtype, device, seed=2024):
        torch.manual_seed(seed)
        q = torch.randn((B, HQ, SQ, D), dtype=dtype, device=device)
        k = torch.randn((B, HK, SK, D), dtype=dtype, device=device)
        v = torch.randn((B, HK, SK, D), dtype=dtype, device=device)
        o = torch.randn((B, HQ, SQ, D), dtype=dtype, device=device)
        do = torch.randn((B, HQ, SQ, D), dtype=dtype, device=device)
        softmax_lse = torch.randn((B, HQ, SQ), dtype=torch.float32, device=device)
        return q, k, v, o, do, softmax_lse

    q, k, v, o_seed, do_seed, _ = _generate_inputs(
        BATCH, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, SEQLEN_Q, SEQLEN_K, dtype, device, seed=2024
    )
    sm_scale = HEAD_SZ**-0.5

    # Prepare tensors in BSHD format (batch, seqlen, heads, dim)
    q_bsnh = q.permute(0, 2, 1, 3).contiguous()
    k_bsnh = k.permute(0, 2, 1, 3).contiguous()
    v_bsnh = v.permute(0, 2, 1, 3).contiguous()

    # Forward via MHA to mirror bench_full_bwd.py and get o and softmax_lse
    o_bsnh, softmax_lse = flash_attn_func(
        q_bsnh,
        k_bsnh,
        v_bsnh,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        return_lse=True,
        return_attn_probs=False,
        mapping_mode=0,
        use_remap=False,
    )
    do_bsnh = torch.randn_like(o_bsnh)

    # Flatten to [B*Seqlen, H, D] and concatenate K/V across batch
    q_flat = q_bsnh.reshape(BATCH * SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ).contiguous()
    k_flat = k_bsnh.reshape(BATCH * SEQLEN_K, NUM_K_HEADS, HEAD_SZ).contiguous()
    v_flat = v_bsnh.reshape(BATCH * SEQLEN_K, NUM_K_HEADS, HEAD_SZ).contiguous()
    o_flat = o_bsnh.reshape(BATCH * SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ).contiguous()
    do_flat = do_bsnh.reshape(BATCH * SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ).contiguous()

    dq_flat = torch.zeros_like(q_flat)
    dk_flat = torch.zeros_like(k_flat)
    dv_flat = torch.zeros_like(v_flat)

    # Compute batch_num_block_n for non-causal multi-batch decode
    BLOCK_N = 64
    num_n_blocks = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N
    batch_num_block_n = None
    if (not causal) and (BATCH > 1):
        batch_num_block_n = (
            torch.arange(1, BATCH + 1, device=device, dtype=torch.int32) * num_n_blocks
        )

    # Warmup run (not timed)
    print("Warmup Lean Attention Backward...")
    torch.cuda.synchronize()
    persistent_lean_attention_bwd(
        q=q_flat, k=k_flat, v=v_flat, do=do_flat, o=o_flat,
        softmax_lse=softmax_lse,
        dq=dq_flat, dk=dk_flat, dv=dv_flat,
        batch_num_block_n=batch_num_block_n,
        batch_size=BATCH, sm_scale=sm_scale, causal=causal,
        seqlen_k=SEQLEN_K,
    )
    torch.cuda.synchronize()

    print("Running Lean Attention Backward (15 iters)...")
    torch.cuda.synchronize()
    
    try:
        iters = 1
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            persistent_lean_attention_bwd(
                q=q_flat, k=k_flat, v=v_flat, do=do_flat, o=o_flat,
                softmax_lse=softmax_lse,
                dq=dq_flat, dk=dk_flat, dv=dv_flat,
                batch_num_block_n=batch_num_block_n,
                batch_size=BATCH, sm_scale=sm_scale, causal=causal,
                seqlen_k=SEQLEN_K,
            )
        end_evt.record()
        torch.cuda.synchronize()
        avg_ms = start_evt.elapsed_time(end_evt) / iters
        print(f"la bwd avg: {avg_ms:.3f} ms over {iters} iters")
        print("✅ Lean Attention Backward completed successfully!")
        print(f"Output shapes - dq: {dq_flat.shape}, dk: {dk_flat.shape}, dv: {dv_flat.shape}")
        return 0
        
    except Exception as e:
        print(f"❌ Lean Attention Backward failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main_flash():
    """Main function for manual testing Flash Attention one-kernel backward"""
    import sys

    # Default configuration - you can modify these values for testing
    BATCH = 2
    NUM_Q_HEADS = 16
    NUM_K_HEADS = 16
    HEAD_SZ = 128
    SEQLEN_Q = 16384
    SEQLEN_K = 16384
    causal = False
    dtype = torch.float16
    device = "cuda"

    print(f"Testing Flash Attention Backward configuration:")
    print(f"  BATCH: {BATCH}")
    print(f"  NUM_Q_HEADS: {NUM_Q_HEADS}")
    print(f"  NUM_K_HEADS: {NUM_K_HEADS}")
    print(f"  HEAD_SZ: {HEAD_SZ}")
    print(f"  SEQLEN_Q: {SEQLEN_Q}")
    print(f"  SEQLEN_K: {SEQLEN_K}")
    print(f"  causal: {causal}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    print("-" * 50)

    # Use same deterministic inputs as main_la
    def _generate_inputs(B, HQ, HK, D, SQ, SK, dtype, device, seed=2024):
        torch.manual_seed(seed)
        q = torch.randn((B, HQ, SQ, D), dtype=dtype, device=device)
        k = torch.randn((B, HK, SK, D), dtype=dtype, device=device)
        v = torch.randn((B, HK, SK, D), dtype=dtype, device=device)
        o = torch.randn((B, HQ, SQ, D), dtype=dtype, device=device)
        do = torch.randn((B, HQ, SQ, D), dtype=dtype, device=device)
        softmax_lse = torch.randn((B, HQ, SQ), dtype=torch.float32, device=device)
        return q, k, v, o, do, softmax_lse

    q, k, v, _, _, _ = _generate_inputs(
        BATCH, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ, SEQLEN_Q, SEQLEN_K, dtype, device, seed=2024
    )
    sm_scale = HEAD_SZ**-0.5

    # Prepare tensors in BSHD format (batch, seqlen, heads, dim)
    q_bsnh = q.permute(0, 2, 1, 3).contiguous()
    k_bsnh = k.permute(0, 2, 1, 3).contiguous()
    v_bsnh = v.permute(0, 2, 1, 3).contiguous()

    # Forward via MHA to mirror bench_full_bwd.py and get o and softmax_lse
    o_bsnh, softmax_lse = flash_attn_func(
        q_bsnh,
        k_bsnh,
        v_bsnh,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=causal,
        return_lse=True,
        return_attn_probs=False,
        mapping_mode=0,
        use_remap=False,
    )
    do_bsnh = torch.randn_like(o_bsnh)

    dq_bsnh = torch.zeros_like(q_bsnh)
    dk_bsnh = torch.zeros_like(k_bsnh)
    dv_bsnh = torch.zeros_like(v_bsnh)

    # Warmup run (not timed)
    print("Warmup Flash Attention One-Kernel Backward...")
    torch.cuda.synchronize()
    flash_attn_onekernel_backward(
        do=do_bsnh,
        q=q_bsnh,
        k=k_bsnh,
        v=v_bsnh,
        o=o_bsnh,
        softmax_lse=softmax_lse,
        dq=dq_bsnh,
        dk=dk_bsnh,
        dv=dv_bsnh,
        dbias=None,
        sm_scale=sm_scale,
        alibi_slopes=None,
        causal=causal,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=SEQLEN_Q,
        max_seqlen_k=SEQLEN_K,
        dropout_p=0.0,
    )
    torch.cuda.synchronize()

    print("Running Flash Attention One-Kernel Backward (15 iters)...")
    torch.cuda.synchronize()
    
    try:
        iters = 1
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(iters):
            flash_attn_onekernel_backward(
                do=do_bsnh,
                q=q_bsnh,
                k=k_bsnh,
                v=v_bsnh,
                o=o_bsnh,
                softmax_lse=softmax_lse,
                dq=dq_bsnh,
                dk=dk_bsnh,
                dv=dv_bsnh,
                dbias=None,
                sm_scale=sm_scale,
                alibi_slopes=None,
                causal=causal,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=SEQLEN_Q,
                max_seqlen_k=SEQLEN_K,
                dropout_p=0.0,
            )
        end_evt.record()
        torch.cuda.synchronize()
        avg_ms = start_evt.elapsed_time(end_evt) / iters
        print(f"fa2 onekernel bwd avg: {avg_ms:.3f} ms over {iters} iters")
        print("✅ Flash Attention Backward completed successfully!")
        print(
            f"Output shapes - dq: {dq_bsnh.shape}, dk: {dk_bsnh.shape}, dv: {dv_bsnh.shape}"
        )
        return 0
        
    except Exception as e:
        print(f"❌ Flash Attention Backward failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main_la())    
