import torch
import math
import triton

# Assume the code from your Canvas (bwd_la_persistent, etc.) is in this file
# or has been imported. For this example, I'll paste the launcher function.
from aiter.ops.triton.lean_atten_bwd_clean import la_backward_persistent


# def flash_attn_la_backward(
#     do: torch.Tensor,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     o: torch.Tensor,
#     softmax_lse: torch.Tensor,
#     dq: torch.Tensor,
#     dk: torch.Tensor,
#     dv: torch.Tensor,
#     sm_scale: float,
#     causal: bool,
#     max_seqlen_q: int,
#     max_seqlen_k: int,
# ):
#     """
#     This is the launcher function from your Canvas.
#     (The full kernel code would be here)
#     """
#     # ... (rest of the launcher code from the Canvas)
#     # For the purpose of this test, we will mock its behavior
#     # In your actual test, you would call your real kernel.
#     pass


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


def run_test():
    # --- 1. Define Test Parameters ---
    BATCH, N_HEADS, SEQLEN_Q, SEQLEN_K, HEAD_DIM = 1, 1, 64, 128, 64
    CAUSAL = True
    DTYPE = torch.float16
    DEVICE = "cuda"

    torch.manual_seed(0)

    # --- 2. Initialize Inputs ---
    q = torch.randn((BATCH, N_HEADS, SEQLEN_Q, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    k = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    v = torch.randn((BATCH, N_HEADS, SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    do = torch.randn_like(q) # Gradient of the output has same shape as Q
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # --- 3. Compute Reference Gradients ---
    print("Computing reference gradients with PyTorch autograd...")
    dq_ref, dk_ref, dv_ref = ground_truth_backward(q, k, v, do, sm_scale, CAUSAL)
    print("Reference gradients computed.")

    # --- 4. Compute Triton Kernel Gradients ---
    # Your kernel needs the output (o) and softmax_lse from the forward pass.
    # We can compute them here.
    logits = (q.to(torch.float32) @ k.to(torch.float32).transpose(-2, -1)) * sm_scale
    if CAUSAL:
        mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
        logits.masked_fill_(mask, float('-inf'))
    
    # THIS IS THE KEY CHANGE: Calculate the true LSE stably
    softmax_lse_stable = torch.logsumexp(logits, dim=-1).to(DTYPE)

    # Calculate the forward output 'o'
    p = torch.softmax(logits, dim=-1).to(DTYPE)
    o_triton = p @ v


    # --- FIX END ---

    # Initialize output tensors for your kernel
    dq_triton = torch.empty_like(q)
    dk_triton = torch.empty_like(k)
    dv_triton = torch.empty_like(v)
    
    print("\nComputing gradients with your Triton kernel...")
    
    # Call your launcher function
    # NOTE: The tensors need to be in [batch, seqlen, n_heads, head_dim] format
    # for the launcher, so we transpose them.
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
    print("Triton kernel execution finished.")

    # --- 5. Compare Results ---
    print("\n--- Comparison Results ---")
    
    # Set tolerance levels for comparison
    atol, rtol = 1e-2, 1e-2

    # Compare dQ
    dq_is_close = torch.allclose(dq_triton, dq_ref, atol=atol, rtol=rtol)
    print(f"dQ allclose: {dq_is_close}")
    print(f"  Max difference in dQ: {(dq_triton - dq_ref).abs().max().item()}")

    # Compare dK
    dk_is_close = torch.allclose(dk_triton, dk_ref, atol=atol, rtol=rtol)
    # print(dk_triton)
    print(f"dK allclose: {dk_is_close}")
    print(f"  Max difference in dK: {(dk_triton - dk_ref).abs().max().item()}")

    # Compare dV
    dv_is_close = torch.allclose(dv_triton, dv_ref, atol=atol, rtol=rtol)
    print(f"dV allclose: {dv_is_close}")
    print(f"  Max difference in dV: {(dv_triton - dv_ref).abs().max().item()}")

    if dq_is_close and dk_is_close and dv_is_close:
        print("\n✅ All gradients match the ground truth!")
    else:
        print("\n❌ One or more gradients do not match the ground truth.")


if __name__ == "__main__":
    # Make sure to have the kernel definitions available in the scope
    # For a real test, you would import them or have them in the same file.
    # Since I cannot run the kernel, I'll skip the call to run_test().
    run_test() 
    print("Test script is ready. You would call run_test() to execute it.")