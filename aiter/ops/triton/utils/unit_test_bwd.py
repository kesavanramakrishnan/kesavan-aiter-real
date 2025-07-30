import math
import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------------
#  New row‑major (non‑transposed) implementation for dK / dV
# ----------------------------------------------------------------------------
@triton.jit
def _bwd_dkdv_inner_row(
    dk, dv,
    Q, K, V, DO, M, D, L, # Accept L
    sm_scale,
    stride_qm, stride_qk,
    stride_dom, stride_dok,
    stride_deltam, # Re-use this stride for M, D, and L
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    seqlen_q,
    seqlen_k,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
):
    """Compute dK and dV for a `[BLOCK_N, HEAD_DIM]` tile of K/V."""
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n      = start_n + tl.arange(0, BLOCK_N)
    offs_k      = tl.arange(0, HEAD_DIM)

    # Materialise the K / V tile in SMEM
    stride_kn = HEAD_DIM
    stride_kd = 1
    stride_vn = HEAD_DIM
    stride_vd = 1

    mask_kv = (offs_n[:, None] < seqlen_k) & (offs_k[None, :] < HEAD_DIM)
    k_tile = tl.load(K + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kd,
                      mask=mask_kv, other=0.0)
    v_tile = tl.load(V + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vd,
                      mask=mask_kv, other=0.0)

    # Transpose once for reuse
    k_tile_T = tl.trans(k_tile)
    v_tile_T = tl.trans(v_tile)
    
    dv_acc = tl.zeros((BLOCK_N, HEAD_DIM), dtype=Q.type.element_ty)
    dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), dtype=Q.type.element_ty)

    curr_m = start_m
    for _ in range(num_steps):
        offs_m = curr_m + offs_m_base
        mask_m = offs_m < seqlen_q

        # ---- RESTORED LINES FOR LOADING Q and dO ----
        q_ptrs  = Q  + offs_m[:, None] * stride_qm  + offs_k[None, :] * stride_qk
        do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        mask_q  = mask_m[:, None] & (offs_k[None, :] < HEAD_DIM)
        q  = tl.load(q_ptrs , mask=mask_q, other=0.0)
        do = tl.load(do_ptrs, mask=mask_q, other=0.0)

        # ---- Load M and L ----
        m_vals = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        l_vals = tl.load(L + offs_m * stride_deltam, mask=mask_m, other=1.0) # Denominator

        # ---- P = softmax (NOW CORRECTLY NORMALIZED) ----
        qk = tl.dot(q, k_tile_T)
        p_unnormalized = tl.math.exp(qk * sm_scale - m_vals[:, None])
        
        if MASK:
            causal = offs_m[:, None] >= offs_n[None, :]
            p_unnormalized = tl.where(causal, p_unnormalized, 0.0)
        
        # Normalize to get the true probability matrix P
        p = p_unnormalized / l_vals[:, None]

        # ---- dV ----
        dv_acc += tl.dot(tl.trans(p).to(do.type.element_ty), do)

        # ---- dS / dK ----
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m, other=0.0)
        dp = tl.dot(do, v_tile_T)
        ds = p * (dp - Di[:, None]) # Use the correct, normalized p
        ds = ds * sm_scale
        dk_acc += tl.dot(tl.trans(ds).to(q.type.element_ty), q)

        curr_m += BLOCK_M

    # Store accumulated results back to global memory
    dk_ptrs = dk + offs_n[:, None] * HEAD_DIM + offs_k[None, :]
    dv_ptrs = dv + offs_n[:, None] * HEAD_DIM + offs_k[None, :]
    tl.store(dk_ptrs, dk_acc, mask=mask_kv)
    tl.store(dv_ptrs, dv_acc, mask=mask_kv)


# ----------------------------------------------------------------------------
#  PyTorch reference for validation
# ----------------------------------------------------------------------------

def _ground_truth_dkdv(Q, K, V, DO, sm_scale, causal=False):
    Q_ = Q.detach().clone().requires_grad_(True)
    K_ = K.detach().clone().requires_grad_(True)
    V_ = V.detach().clone().requires_grad_(True)

    logits = (Q_ @ K_.T) * sm_scale
    if causal:
        mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
        logits = logits.masked_fill(mask, float('-inf'))
    P = torch.softmax(logits, dim=-1)
    O = P @ V_
    loss = (O * DO).sum()
    loss.backward()
    return K_.grad.detach(), V_.grad.detach()


# ----------------------------------------------------------------------------
#  Minimal test – compare Triton vs PyTorch autograd
# ----------------------------------------------------------------------------

def _run_once(device="cuda"):
    BLOCK_M, BLOCK_N, HEAD_DIM = 16, 64, 32
    seqlen_q, seqlen_k = 32, 64

    dtype = torch.float32
    torch.manual_seed(0)

    Q  = torch.randn(seqlen_q, HEAD_DIM, device=device, dtype=dtype)
    K  = torch.randn(seqlen_k, HEAD_DIM, device=device, dtype=dtype)
    V  = torch.randn_like(K)
    DO = torch.randn(seqlen_q, HEAD_DIM, device=device, dtype=dtype)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Precompute M, D, and L for the kernel
    logits = (Q @ K.T) * sm_scale
    m_vals = logits.max(dim=-1).values

    # ---- NEW: Calculate the softmax denominator L ----
    p_unnormalized = torch.exp(logits - m_vals[:, None])
    L_vec = p_unnormalized.sum(dim=-1)
    
    # Use the true P for the reference calculation
    P = p_unnormalized / L_vec[:, None]
    O = P @ V
    D_vec = (O * DO).sum(dim=-1)

    M = m_vals.detach()
    D = D_vec.detach()
    L = L_vec.detach() # Pass L to the kernel

    stride_qm, stride_qk  = Q.stride()
    stride_dom, stride_dok = DO.stride()
    stride_deltam = M.stride(0) # M, D, and L have the same stride

    dk_out = torch.zeros(BLOCK_N, HEAD_DIM, device=device, dtype=torch.float32)
    dv_out = torch.zeros_like(dk_out)

    num_steps = math.ceil(seqlen_q / BLOCK_M)
    grid = (1,)

    # Add L to the kernel call
    _bwd_dkdv_inner_row[grid](
        dk_out, dv_out,
        Q, K, V, DO, M, D, L, # Pass L here
        sm_scale,
        stride_qm, stride_qk,
        stride_dom, stride_dok,
        stride_deltam,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
        seqlen_q=seqlen_q, seqlen_k=seqlen_k,
        start_n=0, start_m=0, num_steps=num_steps,
        MASK=False,
    )

    dk_ref, dv_ref = _ground_truth_dkdv(Q, K, V, DO, sm_scale, causal=False)
    dk_ref_block = dk_ref[:BLOCK_N]
    dv_ref_block = dv_ref[:BLOCK_N]

    print("max |Δdk|:", (dk_out - dk_ref_block).abs().max().item())
    print("max |Δdv|:", (dv_out - dv_ref_block).abs().max().item())
    print("dk close:", torch.allclose(dk_out, dk_ref_block, atol=1e-4, rtol=1e-3))
    print("dv close:", torch.allclose(dv_out, dv_ref_block, atol=1e-4, rtol=1e-3))



if __name__ == "__main__":
    _run_once()
