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
        "BATCH": 1, "N_HEADS": 8, "SEQLEN_Q": 8192, "SEQLEN_K": 8192,
        "HEAD_DIM": 64, "CAUSAL": False, "DTYPE": torch.float16
    },
    {
        "BATCH": 1, "N_HEADS": 8, "SEQLEN_Q": 8192, "SEQLEN_K": 8192,
        "HEAD_DIM": 64, "CAUSAL": True, "DTYPE": torch.float16
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


def run_test(config: dict):
    """
    Runs a single test case based on the provided configuration dictionary.
    Returns True if the test passes, False otherwise.
    """
    # --- 1. Unpack Test Parameters ---
    BATCH, N_HEADS, SEQLEN_Q, SEQLEN_K, HEAD_DIM, CAUSAL, DTYPE = (
        config['BATCH'], config['N_HEADS'], config['SEQLEN_Q'],
        config['SEQLEN_K'], config['HEAD_DIM'], config['CAUSAL'], config['DTYPE']
    )
    DEVICE = "cuda"
    torch.manual_seed(0)

    # --- 2. Initialize Inputs ---
    q = torch.randn((BATCH, N_HEADS, SEQLEN_Q, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    k = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    v = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    do = torch.randn_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # --- 3. Compute Reference Gradients ---
    print("  Computing reference gradients with PyTorch autograd...")
    dq_ref, dk_ref, dv_ref = ground_truth_backward(q, k, v, do, sm_scale, CAUSAL)

    # --- 4. Compute Triton Kernel Gradients ---
    # Your kernel needs 'o' and 'softmax_lse' from the forward pass.
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

    print("  Computing gradients with your Triton kernel...")
    # NOTE: Tensors are transposed to [batch, seqlen, n_heads, head_dim] for the launcher.
    la_backward_persistent(
        do.transpose(1, 2),
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        o_triton.transpose(1, 2),
        softmax_lse_stable,
        dq_triton.transpose(1, 2),
        dk_triton.transpose(1, 2),
        dv_triton.transpose(1, 2),
        sm_scale,
        CAUSAL,
        SEQLEN_Q,
        SEQLEN_K,
    )

    # --- 5. Compare Results ---
    print("\n  --- Comparison Results ---")
    atol, rtol = 1e-2, 1e-2
    passed = True

    # Compare dQ
    dq_is_close = torch.allclose(dq_triton, dq_ref, atol=atol, rtol=rtol)
    if not dq_is_close: passed = False
    print(f"  dQ allclose: {dq_is_close}")
    print(f"    Max difference in dQ: {(dq_triton - dq_ref).abs().max().item():.6f}")

    # Compare dK
    dk_is_close = torch.allclose(dk_triton, dk_ref, atol=atol, rtol=rtol)
    if not dk_is_close: passed = False
    print(f"  dK allclose: {dk_is_close}")
    print(f"    Max difference in dK: {(dk_triton - dk_ref).abs().max().item():.6f}")

    # Compare dV
    dv_is_close = torch.allclose(dv_triton, dv_ref, atol=atol, rtol=rtol)
    if not dv_is_close: passed = False
    print(f"  dV allclose: {dv_is_close}")
    print(f"    Max difference in dV: {(dv_triton - dv_ref).abs().max().item():.6f}")

    return passed


if __name__ == "__main__":
    results = []
    print("🚀 Starting Triton Kernel Test Suite 🚀")

    for i, config in enumerate(TEST_CONFIGS):
        print("\n" + "="*80)
        print(f"📊 Running Test {i+1}/{len(TEST_CONFIGS)}")
        print(f"   Config: {config}")
        print("="*80)
        try:
            passed = run_test(config)
            if passed:
                print("\n  ✅ Test Passed!")
            else:
                print("\n  ❌ Test FAILED.")
            results.append({"config": config, "status": "✅ PASSED" if passed else "❌ FAILED"})
        except Exception as e:
            print(f"\n  🔥 Test CRASHED with exception: {e}")
            results.append({"config": config, "status": f"🔥 CRASHED: {e}"})

    # --- Final Summary ---
    print("\n\n" + "#"*80)
    print("📋 Test Suite Summary")
    print("#"*80)
    for i, result in enumerate(results):
        print(f"Test #{i+1}: {result['status']}")
        print(f"  Config: {result['config']}")
    print("#"*80)