import pytest
import torch
import triton

from aiter.ops.triton.lean_atten_bwd_acc import persistent_lean_attention_bwd
from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward

# Define data types for testing, including float32 on capable hardware
DTYPES = [torch.float16]
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    DTYPES.append(torch.float32)

# Define tolerance levels for result comparisons
ATOL = {torch.float16: 1e-2, torch.bfloat16: 2e-2, torch.float32: 1e-4}
RTOL = {torch.float16: 1e-2, torch.bfloat16: 2e-2, torch.float32: 1e-4}


@pytest.mark.parametrize("BATCH", [1,])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4)])
@pytest.mark.parametrize("HEAD_SZ", [16])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(64, 64)])
@pytest.mark.parametrize("causal", [False, True])
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
    """
    Compares the backward pass of Lean Attention with a reference Flash Attention
    implementation and PyTorch's native scaled_dot_product_attention.
    """
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

    print("\n--- dK Comparison (Lean Attn vs Flash Attn) ---")
    dk_diff = torch.abs(dk_la - dk_flash)
    print(dk_diff)
    dk_tol = atol + rtol * torch.abs(dk_flash)
    dk_mismatch_pct = (dk_diff > dk_tol).float().mean().item() * 100
    print(f"% mismatched (dK): {dk_mismatch_pct:.6f}%")

    print("\n--- dV Comparison (Lean Attn vs Flash Attn) ---")
    dv_diff = torch.abs(dv_la - dv_flash)
    dv_tol = atol + rtol * torch.abs(dv_flash)
    dv_mismatch_pct = (dv_diff > dv_tol).float().mean().item() * 100
    print(f"% mismatched (dV): {dv_mismatch_pct:.6f}%")

    print("\n--- dQ Comparison (Lean Attn vs Flash Attn) ---")
    dq_diff = torch.abs(dq_la - dq_flash)
    dq_tol = atol + rtol * torch.abs(dq_flash)
    dq_mismatch_pct = (dq_diff > dq_tol).float().mean().item() * 100
    print(f"% mismatched (dQ): {dq_mismatch_pct:.6f}%")

    # Compare Lean Attention with Flash Attention
    torch.testing.assert_close(dq_la, dq_flash, atol=atol, rtol=rtol, msg="dQ (Lean Attn vs Flash Attn)")
    torch.testing.assert_close(dk_la, dk_flash, atol=atol, rtol=rtol, msg="dK (Lean Attn vs Flash Attn)")
    torch.testing.assert_close(dv_la, dv_flash, atol=atol, rtol=rtol, msg="dV (Lean Attn vs Flash Attn)")
