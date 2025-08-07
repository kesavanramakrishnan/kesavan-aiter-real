import torch
import math
import triton
import sys, os

# Assume the la_backward_persistent launcher is in this path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from aiter.ops.triton.lean_atten_bwd_clean import la_backward_persistent

# --- 1. Define Test Configurations ---
# Add or modify dictionaries in this list to test different scenarios.
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
    Computes the reference gradients using PyTorch's autograd.
    """
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    # Re-run the forward pass to build the computation graph
    logits = (q_ref @ k_ref.transpose(-2, -1)) * sm_scale
    if causal:
        mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
        logits.masked_fill_(mask, float('-inf'))

    p = torch.softmax(logits, dim=-1)
    o_ref = p @ v_ref

    # Call backward on the loss
    o_ref.backward(gradient=do)

    return q_ref.grad, k_ref.grad, v_ref.grad


def run_test_and_benchmark(config: dict):
    """
    Runs a single test case for correctness and performance based on the
    provided configuration dictionary. Returns a dictionary with results.
    """
    # --- 1. Unpack Test Parameters ---
    BATCH, N_HEADS, SEQLEN_Q, SEQLEN_K, HEAD_DIM, CAUSAL, DTYPE = (
        config['BATCH'], config['N_HEADS'], config['SEQLEN_Q'],
        config['SEQLEN_K'], config['HEAD_DIM'], config['CAUSAL'], config['DTYPE']
    )
    DEVICE = "cuda"
    N_REPS = 50 # Number of repetitions for stable timing
    torch.manual_seed(0)

    # --- 2. Initialize Inputs ---
    q = torch.randn((BATCH, N_HEADS, SEQLEN_Q, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    k = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    v = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    do = torch.randn_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # --- 3. Compute and Benchmark Reference Gradients ---
    print("  Computing and benchmarking reference gradients...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    dq_ref, dk_ref, dv_ref = ground_truth_backward(q, k, v, do, sm_scale, CAUSAL)
    end_event.record()
    torch.cuda.synchronize()
    ref_time_ms = start_event.elapsed_time(end_event)

    # --- 4. Compute and Benchmark Triton Kernel Gradients ---
    # Prepare kernel inputs
    logits = (q.to(torch.float32) @ k.to(torch.float32).transpose(-2, -1)) * sm_scale
    if CAUSAL:
        mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
        logits.masked_fill_(mask, float('-inf'))

    softmax_lse_stable = torch.logsumexp(logits, dim=-1).to(DTYPE)
    p = torch.softmax(logits, dim=-1).to(DTYPE)
    o_triton = p @ v

    dq_triton = torch.empty_like(q)
    dk_triton = torch.empty_like(k)
    dv_triton = torch.empty_like(v)
    
    # Define a lambda for the kernel call
    kernel_call = lambda: la_backward_persistent(
        do.transpose(1, 2), q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        o_triton.transpose(1, 2), softmax_lse_stable, dq_triton.transpose(1, 2),
        dk_triton.transpose(1, 2), dv_triton.transpose(1, 2), sm_scale,
        CAUSAL, SEQLEN_Q, SEQLEN_K
    )

    print("  Warming up and benchmarking Triton kernel...")
    # Warmup call to handle JIT compilation
    kernel_call()
    torch.cuda.synchronize()

    # Timed calls
    start_event.record()
    for _ in range(N_REPS):
        kernel_call()
    end_event.record()
    torch.cuda.synchronize()
    triton_time_ms = start_event.elapsed_time(end_event) / N_REPS


    # --- 5. Compare Results & Performance ---
    print("\n  --- Correctness & Performance ---")
    atol, rtol = 1e-2, 1e-2
    passed = True

    dq_is_close = torch.allclose(dq_triton, dq_ref, atol=atol, rtol=rtol)
    if not dq_is_close: passed = False
    print(f"  Correctness | dQ allclose: {dq_is_close}")

    dk_is_close = torch.allclose(dk_triton, dk_ref, atol=atol, rtol=rtol)
    if not dk_is_close: passed = False
    print(f"  Correctness | dK allclose: {dk_is_close}")

    dv_is_close = torch.allclose(dv_triton, dv_ref, atol=atol, rtol=rtol)
    if not dv_is_close: passed = False
    print(f"  Correctness | dV allclose: {dv_is_close}")

    print("-" * 40)
    print(f"  Performance | PyTorch Reference: {ref_time_ms:.4f} ms")
    print(f"  Performance | Triton Kernel:     {triton_time_ms:.4f} ms")
    print(f"  Performance | Speedup:           {ref_time_ms / triton_time_ms:.2f}x")

    return {
        "passed": passed,
        "ref_ms": ref_time_ms,
        "triton_ms": triton_time_ms,
        "speedup": ref_time_ms / triton_time_ms
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
            result_data = run_test_and_benchmark(config)
            status = "✅ PASSED" if result_data["passed"] else "❌ FAILED"
            print(f"\n  Correctness Status: {status}")
            results.append({"config": config, "status": status, **result_data})
        except Exception as e:
            print(f"\n  🔥 Test CRASHED with exception: {e}")
            results.append({"config": config, "status": f"🔥 CRASHED", "exception": e})

    # --- Final Summary ---
    print("\n\n" + "#"*80)
    print("📋 Test Suite Summary")
    print("#"*80)
    for i, result in enumerate(results):
        print(f"Test #{i+1}: {result['status']}")
        if "passed" in result:
            print(f"  Perf: {result['speedup']:.2f}x speedup ({result['triton_ms']:.4f} ms vs {result['ref_ms']:.4f} ms)")
        print(f"  Config: {result['config']}")
    print("#"*80)