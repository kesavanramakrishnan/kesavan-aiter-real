import torch
import math
import triton
import sys, os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from aiter.ops.triton.lean_atten_bwd_acc import la_backward_persistent
from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward

# --- 1. Define Test Configurations ---
TEST_CONFIGS = [
    {
        "BATCH": 2, "N_HEADS": 48, "SEQLEN_Q": 64, "SEQLEN_K": 128,
        "HEAD_DIM": 64, "CAUSAL": False, "DTYPE": torch.float16
    },
    {
        "BATCH": 2, "N_HEADS": 32, "SEQLEN_Q": 128, "SEQLEN_K": 128,
        "HEAD_DIM": 64, "CAUSAL": True, "DTYPE": torch.float16
    },
    {
        "BATCH": 1, "N_HEADS": 16, "SEQLEN_Q": 256, "SEQLEN_K": 256,
        "HEAD_DIM": 32, "CAUSAL": False, "DTYPE": torch.float16
    },
    {
        "BATCH": 4, "N_HEADS": 8, "SEQLEN_Q": 512, "SEQLEN_K": 256,
        "HEAD_DIM": 64, "CAUSAL": False, "DTYPE": torch.float16
    },
     {
        "BATCH": 1, "N_HEADS": 48, "SEQLEN_Q": 8192, "SEQLEN_K": 8192,
        "HEAD_DIM": 64, "CAUSAL": False, "DTYPE": torch.float16
    }
]

# --- Helper Functions ---

def ground_truth_backward(q, k, v, do, sm_scale, causal):
    """
    Computes the reference gradients using PyTorch's autograd with a corrected
    causal mask implementation.
    """
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    SEQLEN_Q, SEQLEN_K = q.shape[2], k.shape[2]

    logits = (q_ref @ k_ref.transpose(-2, -1)) * sm_scale
    if causal:
        mask = torch.ones(SEQLEN_Q, SEQLEN_K, device=q.device, dtype=torch.bool).tril(
            diagonal=(SEQLEN_K - SEQLEN_Q)
        )
        logits.masked_fill_(~mask, float('-inf'))

    p = torch.softmax(logits, dim=-1)
    p = torch.nan_to_num(p)  # Handle cases where all logits are -inf
    o_ref = p @ v_ref

    o_ref.backward(gradient=do)

    return q_ref.grad, k_ref.grad, v_ref.grad, o_ref


def run_test_and_benchmark(config: dict):
    """
    Runs a single test case for correctness and performance, comparing Lean
    Attention and One-Kernel Flash Attention.
    """
    BATCH, N_HEADS, SEQLEN_Q, SEQLEN_K, HEAD_DIM, CAUSAL, DTYPE = (
        config['BATCH'], config['N_HEADS'], config['SEQLEN_Q'],
        config['SEQLEN_K'], config['HEAD_DIM'], config['CAUSAL'], config['DTYPE']
    )
    DEVICE = "cuda"
    N_REPS = 50
    torch.manual_seed(0)

    q = torch.randn((BATCH, N_HEADS, SEQLEN_Q, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    k = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    v = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    do = torch.randn_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    print("  Computing and benchmarking reference gradients...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    dq_ref, dk_ref, dv_ref, o_ref = ground_truth_backward(q, k, v, do, sm_scale, CAUSAL)
    end_event.record()
    torch.cuda.synchronize()
    ref_time_ms = start_event.elapsed_time(end_event)

    # --- Prepare Triton Inputs ---
    logits = (q.to(torch.float32) @ k.to(torch.float32).transpose(-2, -1)) * sm_scale
    if CAUSAL:
        mask = torch.ones(SEQLEN_Q, SEQLEN_K, device=DEVICE, dtype=torch.bool).tril(diagonal=(SEQLEN_K - SEQLEN_Q))
        logits.masked_fill_(~mask, float('-inf'))
    softmax_lse_stable = torch.logsumexp(logits, dim=-1).to(DTYPE)

    # --- Benchmark Lean Attention ---
    print("  Warming up and benchmarking Lean Attention kernel...")
    dq_la = torch.empty_like(q)
    dk_la = torch.empty_like(k)
    dv_la = torch.empty_like(v)
    
    # Transpose from BHSD to BSHD for the kernel
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    o_bshd = o_ref.transpose(1, 2).contiguous()
    do_bshd = do.transpose(1, 2).contiguous()
    dq_la_bshd = dq_la.transpose(1, 2).contiguous()
    dk_la_bshd = dk_la.transpose(1, 2).contiguous()
    dv_la_bshd = dv_la.transpose(1, 2).contiguous()
    num_sm = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    la_kernel_call = lambda: la_backward_persistent(
        do_bshd, q_bshd, k_bshd, v_bshd, o_bshd, softmax_lse_stable,
        dq_la_bshd, dk_la_bshd, dv_la_bshd, sm_scale,
        CAUSAL, SEQLEN_Q, SEQLEN_K, num_sm=num_sm
    )
    la_kernel_call() # Warmup
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(N_REPS):
        la_kernel_call()
    end_event.record()
    torch.cuda.synchronize()
    la_time_ms = start_event.elapsed_time(end_event) / N_REPS

    # --- Benchmark Flash Attention (One Kernel) ---
    print("  Warming up and benchmarking One-Kernel Flash Attention...")
    dq_flash_bshd = torch.empty_like(q_bshd)
    dk_flash_bshd = torch.empty_like(k_bshd)
    dv_flash_bshd = torch.empty_like(v_bshd)

    flash_kernel_call = lambda: flash_attn_onekernel_backward(
        do=do_bshd, q=q_bshd, k=k_bshd, v=v_bshd, o=o_bshd,
        softmax_lse=softmax_lse_stable, dq=dq_flash_bshd, dk=dk_flash_bshd, dv=dv_flash_bshd,
        dbias=None, sm_scale=sm_scale, alibi_slopes=None, causal=CAUSAL,
        cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=SEQLEN_Q,
        max_seqlen_k=SEQLEN_K, dropout_p=0.0
    )
    flash_kernel_call() # Warmup
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(N_REPS):
        flash_kernel_call()
    end_event.record()
    torch.cuda.synchronize()
    flash_time_ms = start_event.elapsed_time(end_event) / N_REPS

    # --- Compare Results & Performance ---
    atol, rtol = 1e-2, 1e-2
    dq_la, dk_la, dv_la = dq_la_bshd.transpose(1, 2), dk_la_bshd.transpose(1, 2), dv_la_bshd.transpose(1, 2)
    la_passed = all([
        torch.allclose(dq_la, dq_ref, atol=atol, rtol=rtol),
        torch.allclose(dk_la, dk_ref, atol=atol, rtol=rtol),
        torch.allclose(dv_la, dv_ref, atol=atol, rtol=rtol)
    ])

    dq_flash, dk_flash, dv_flash = dq_flash_bshd.transpose(1, 2), dk_flash_bshd.transpose(1, 2), dv_flash_bshd.transpose(1, 2)
    flash_passed = all([
        torch.allclose(dq_flash, dq_ref, atol=atol, rtol=rtol),
        torch.allclose(dk_flash, dk_ref, atol=atol, rtol=rtol),
        torch.allclose(dv_flash, dv_ref, atol=atol, rtol=rtol)
    ])

    print("\n  --- Correctness & Performance ---")
    print(f"  Correctness | Lean Attention:      {'✅ PASSED' if la_passed else '❌ FAILED'}")
    print(f"  Correctness | One-Kernel Flash:    {'✅ PASSED' if flash_passed else '❌ FAILED'}")
    print("-" * 40)
    print(f"  Performance | PyTorch Reference:   {ref_time_ms:.4f} ms")
    print(f"  Performance | Lean Attention:      {la_time_ms:.4f} ms (Speedup: {ref_time_ms/la_time_ms:.2f}x)")
    print(f"  Performance | One-Kernel Flash:    {flash_time_ms:.4f} ms (Speedup: {ref_time_ms/flash_time_ms:.2f}x)")
    print(f"  Performance | Lean vs. Flash:      {flash_time_ms / la_time_ms:.2f}x")

    return {
        "la_passed": la_passed, "flash_passed": flash_passed, "ref_ms": ref_time_ms,
        "la_ms": la_time_ms, "flash_ms": flash_time_ms
    }

if __name__ == "__main__":
    results = []
    print("🚀 Starting Triton Kernel Test & Benchmark Suite 🚀")
    for i, config in enumerate(TEST_CONFIGS):
        print("\n" + "="*80)
        print(f"📊 Running Test {i+1}/{len(TEST_CONFIGS)}")
        print(f"   Config: {config}")
        print("="*80)
        try:
            res = run_test_and_benchmark(config)
            status = []
            if res['la_passed']: status.append("LA_PASSED")
            else: status.append("LA_FAILED")
            if res['flash_passed']: status.append("FLASH_PASSED")
            else: status.append("FLASH_FAILED")
            results.append({"config": config, "status": " | ".join(status), **res})
        except Exception as e:
            print(f"\n  🔥 Test CRASHED with exception: {e}")
            results.append({"config": config, "status": "🔥 CRASHED", "exception": e})

    print("\n\n" + "#"*80)
    print("📋 Test Suite Summary")
    print("#"*80)
    for i, res in enumerate(results):
        print(f"Test #{i+1}: {res['status']}")
        if "la_passed" in res:
            la_speedup = res['ref_ms'] / res['la_ms']
            flash_speedup = res['ref_ms'] / res['flash_ms']
            la_vs_flash = res['flash_ms'] / res['la_ms']
            print(f"  Lean Attn: {res['la_ms']:.4f} ms ({la_speedup:.2f}x vs Ref)")
            print(f"  Flash Attn: {res['flash_ms']:.4f} ms ({flash_speedup:.2f}x vs Ref)")
            print(f"  --> Lean is {la_vs_flash:.2f}x {'FASTER' if la_vs_flash > 1 else 'SLOWER'} than Flash")
        print(f"  Config: {res['config']}")
    print("#"*80)
