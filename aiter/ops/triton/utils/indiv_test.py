import torch
import triton
import triton.language as tl
import math

# ----------------------------------------------------------------------------
# 1. The Triton JIT function to be tested
# ----------------------------------------------------------------------------
@triton.jit
def _bwd_dq_la_inner(
    dq_acc, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
    stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
    max_seqlen_q, max_seqlen_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    start_m, start_n_loop, num_n_steps,
    CAUSAL: tl.constexpr,
):
    """
    Inner loop for dQ. Iterates over K/V blocks for a given Q block.
    """
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    curr_n = start_n_loop
    for _ in range(num_n_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < max_seqlen_k

        # Load K and V
        k_ptrs = K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        mask_kv = mask_n[:, None] & (offs_d[None, :] < HEAD_DIM)

        k = tl.load(k_ptrs, mask=mask_kv, other=0.0)
        v = tl.load(v_ptrs, mask=mask_kv, other=0.0)

        # Recompute P = exp(QK^T * sm_scale - M)
        qk = tl.dot(q_tile, tl.trans(k))
        p = tl.math.exp2(qk * sm_scale * 1.44269504 - m_tile)
        
        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        # Compute dP = dO @ V^T
        dp = tl.dot(do_tile, tl.trans(v))
        
        # Load Delta
        Di = tl.load(Delta_ptr + offs_m * stride_deltam, mask=(offs_m < max_seqlen_q), other=0.0)
        
        # Compute dS = P * (dP - D_i)
        ds = p * (dp - Di[:, None])
        
        # Apply scaling factor to dS
        ds_scaled = ds * sm_scale
        
        # Compute dQ += dS_scaled @ K
        dq_acc += tl.dot(ds_scaled, k.to(tl.float32))
        
        curr_n += BLOCK_N
        
    return dq_acc


# ----------------------------------------------------------------------------
# 2. A simple launcher kernel to call the JIT function from Python
# ----------------------------------------------------------------------------
@triton.jit
def dq_inner_launcher_kernel(
    # Output
    DQ_ptr,
    # Inputs
    Q_tile_ptr, K_ptr, V_ptr, DO_tile_ptr, M_tile_ptr, Delta_ptr,
    # Strides
    stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
    # Params
    sm_scale, max_seqlen_q, max_seqlen_k,
    start_m, start_n_loop, num_n_steps,
    # Constexpr
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    This kernel's only job is to load the initial data for one block
    and call the inner function we want to test.
    """
    dq_acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    q_ptrs = Q_tile_ptr + (tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    do_ptrs = DO_tile_ptr + (tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    m_ptrs = M_tile_ptr + tl.arange(0, BLOCK_M)[:, None]

    mask_m = offs_m < max_seqlen_q
    mask_q = mask_m[:, None] & (offs_d[None, :] < HEAD_DIM)

    q_tile = tl.load(q_ptrs, mask=mask_q, other=0.0)
    do_tile = tl.load(do_ptrs, mask=mask_q, other=0.0)
    m_tile = tl.load(m_ptrs, mask=mask_m[:, None], other=-float('inf'))

    dq_acc = _bwd_dq_la_inner(
        dq_acc, q_tile, K_ptr, V_ptr, do_tile, m_tile, Delta_ptr, sm_scale,
        stride_kn, stride_kd, stride_vn, stride_vd, stride_deltam,
        max_seqlen_q, max_seqlen_k,
        BLOCK_M, BLOCK_N, HEAD_DIM,
        start_m, start_n_loop, num_n_steps,
        CAUSAL=CAUSAL
    )

    dq_out_ptrs = DQ_ptr + (tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dq_out_ptrs, dq_acc.to(DQ_ptr.type.element_ty), mask=mask_q)


# ----------------------------------------------------------------------------
# 3. PyTorch reference implementation for comparison
# ----------------------------------------------------------------------------
def torch_reference_dq_inner(
    q_tile, K, V, do_tile, m_tile, Delta, sm_scale,
    BLOCK_M, BLOCK_N, HEAD_DIM,
    start_m, start_n_loop, num_n_steps,
    max_seqlen_q, max_seqlen_k,
    causal
):
    dq_acc = torch.zeros_like(q_tile, dtype=torch.float32)
    offs_m = torch.arange(start_m, start_m + BLOCK_M, device=q_tile.device)
    RCP_LN2 = 1.44269504

    for i in range(num_n_steps):
        curr_n = start_n_loop + i * BLOCK_N
        
        k_block = K[curr_n : curr_n + BLOCK_N, :]
        v_block = V[curr_n : curr_n + BLOCK_N, :]
        
        qk = q_tile @ k_block.T
        p = torch.exp2(qk * sm_scale * RCP_LN2 - m_tile)
        
        if causal:
            offs_n = torch.arange(curr_n, curr_n + BLOCK_N, device=q_tile.device)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            p = torch.where(causal_mask, p, 0.0)

        dp = do_tile @ v_block.T
        Di = Delta[start_m : start_m + BLOCK_M]
        ds = p * (dp - Di[:, None])
        ds_scaled = ds * sm_scale
        dq_acc += ds_scaled @ k_block.to(torch.float32)
        
    return dq_acc


# ----------------------------------------------------------------------------
# 4. Main test execution logic
# ----------------------------------------------------------------------------
def _run_single_test(test_name, params):
    # --- Unpack Test Parameters ---
    SEQLEN_Q, SEQLEN_K = params["SEQLEN_Q"], params["SEQLEN_K"]
    HEAD_DIM = params["HEAD_DIM"]
    BLOCK_M, BLOCK_N = params["BLOCK_M"], params["BLOCK_N"]
    CAUSAL = params["CAUSAL"]
    DTYPE = params["DTYPE"]
    DEVICE = "cuda"

    torch.manual_seed(123)

    # --- Initialize Tensors ---
    K = torch.randn((SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    V = torch.randn((SEQLEN_K, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    Delta = torch.randn((SEQLEN_Q,), dtype=torch.float32, device=DEVICE)

    start_m = BLOCK_M # Test a block that isn't the first one
    if SEQLEN_Q <= BLOCK_M: # Adjust if SEQLEN_Q is small
        start_m = 0

    q_tile = torch.randn((BLOCK_M, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    do_tile = torch.randn((BLOCK_M, HEAD_DIM), dtype=DTYPE, device=DEVICE)
    m_tile = torch.randn((BLOCK_M, 1), dtype=torch.float32, device=DEVICE)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Determine loop bounds for the inner function
    start_n_loop = 0
    end_n_loop = SEQLEN_K if not CAUSAL else min(start_m + BLOCK_M, SEQLEN_K)
    num_n_steps = (end_n_loop - start_n_loop + BLOCK_N - 1) // BLOCK_N

    # --- Run Triton Kernel ---
    dq_triton = torch.empty_like(q_tile)
    grid = (1,)
    dq_inner_launcher_kernel[grid](
        DQ_ptr=dq_triton,
        Q_tile_ptr=q_tile, K_ptr=K, V_ptr=V, DO_tile_ptr=do_tile, M_tile_ptr=m_tile, Delta_ptr=Delta,
        stride_kn=K.stride(0), stride_kd=K.stride(1),
        stride_vn=V.stride(0), stride_vd=V.stride(1),
        stride_deltam=Delta.stride(0),
        sm_scale=sm_scale, max_seqlen_q=SEQLEN_Q, max_seqlen_k=SEQLEN_K,
        start_m=start_m, start_n_loop=start_n_loop, num_n_steps=num_n_steps,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM,
        CAUSAL=CAUSAL
    )

    # --- Run PyTorch Reference ---
    dq_ref = torch_reference_dq_inner(
        q_tile.float(), K.float(), V.float(), do_tile.float(), m_tile, Delta, sm_scale,
        BLOCK_M, BLOCK_N, HEAD_DIM,
        start_m, start_n_loop, num_n_steps,
        SEQLEN_Q, SEQLEN_K,
        causal=CAUSAL
    )

    # --- Compare Results ---
    atol = 1e-1 if DTYPE == torch.float16 else 1e-4
    rtol = 1e-2 if DTYPE == torch.float16 else 1e-5
    
    is_close = torch.allclose(dq_triton.float(), dq_ref, atol=atol, rtol=rtol)
    
    # --- Print Summary ---
    result_str = "✅ PASS" if is_close else "❌ FAIL"
    print(f"[{result_str}] {test_name}")
    
    if not is_close:
        max_diff = (dq_triton.float() - dq_ref).abs().max().item()
        print(f"    Max absolute difference: {max_diff:.6f}")
    
    return is_close

def run_all_tests():
    print("--- Running dQ Inner Function Tests ---")
    
    test_configs = [
        { "name": "Base Case (Causal, SQ=SK=256, D=64, FP16)", "params": {"SEQLEN_Q": 256, "SEQLEN_K": 256, "HEAD_DIM": 64, "BLOCK_M": 64, "BLOCK_N": 32, "CAUSAL": True, "DTYPE": torch.float16}},
        { "name": "Non-Causal (SQ=SK=256, D=64, FP16)", "params": {"SEQLEN_Q": 256, "SEQLEN_K": 256, "HEAD_DIM": 64, "BLOCK_M": 64, "BLOCK_N": 32, "CAUSAL": False, "DTYPE": torch.float16}},
        { "name": "Unequal Lengths (SQ > SK, Causal, FP16)", "params": {"SEQLEN_Q": 256, "SEQLEN_K": 128, "HEAD_DIM": 64, "BLOCK_M": 64, "BLOCK_N": 32, "CAUSAL": True, "DTYPE": torch.float16}},
        { "name": "Unequal Lengths (SK > SQ, Causal, FP16)", "params": {"SEQLEN_Q": 128, "SEQLEN_K": 256, "HEAD_DIM": 64, "BLOCK_M": 64, "BLOCK_N": 32, "CAUSAL": True, "DTYPE": torch.float16}},
        { "name": "Large Head Dim (D=128, Causal, FP16)", "params": {"SEQLEN_Q": 256, "SEQLEN_K": 256, "HEAD_DIM": 128, "BLOCK_M": 32, "BLOCK_N": 32, "CAUSAL": True, "DTYPE": torch.float16}},
        { "name": "Edge Case (Non-multiple lengths, Causal, FP16)", "params": {"SEQLEN_Q": 250, "SEQLEN_K": 240, "HEAD_DIM": 64, "BLOCK_M": 64, "BLOCK_N": 32, "CAUSAL": True, "DTYPE": torch.float16}},
        { "name": "Full Precision (Causal, SQ=SK=256, D=64, FP32)", "params": {"SEQLEN_Q": 256, "SEQLEN_K": 256, "HEAD_DIM": 64, "BLOCK_M": 64, "BLOCK_N": 32, "CAUSAL": True, "DTYPE": torch.float32}},
    ]

    passed_count = 0
    for config in test_configs:
        if _run_single_test(config["name"], config["params"]):
            passed_count += 1
            
    print("-" * 40)
    print(f"Summary: {passed_count} / {len(test_configs)} tests passed.")
    print("-" * 40)

if __name__ == "__main__":
    run_all_tests()