import torch
import functools
from typing import Optional
import triton
import triton.language as tl
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.lean_atten_bwd_clean import persistent_lean_attention_bwd




def flash_attn_onekernel_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dbias: torch.Tensor,
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    philox_seed: Optional[int] = 0,
    philox_offset: Optional[int] = 0,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
    USE_INT64_STRIDES: Optional[bool] = False,
    config: Optional[dict] = None,
):
    """
    Drop-in wrapper to match the onekernel backward API. Supports non-varlen, no-dropout paths.
    Reshapes inputs to the layout expected by `persistent_lean_attention_bwd` and invokes it.
    Returns host-computed delta with the same shape as `softmax_lse`.
    """
    # Only non-varlen supported in this lean path
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        raise NotImplementedError("lean_atten_bwd_prod only supports non-varlen inputs in this path")
    # Dropout and alibi not supported in lean path
    if dropout_p and dropout_p > 0.0:
        raise NotImplementedError("lean_atten_bwd_prod expects dropout_p == 0.0")

    # Shapes: q/do/o: [B, Nq, H, D], k/v: [B, Nk, H, D]
    B, Nq, H, D = q.shape
    Nk = k.shape[1]
    # Build lean views: [B*Nq, H, D] and [B*Nk, H, D]
    q_lean = q.reshape(B * Nq, H, D)
    do_lean = do.reshape(B * Nq, H, D)
    o_lean = o.reshape(B * Nq, H, D)
    k_lean = k.reshape(B * Nk, H, D)
    v_lean = v.reshape(B * Nk, H, D)

    # Outputs as views to preserve user buffers and strides
    dq_view = dq.reshape(B * Nq, H, D)
    dk_view = dk.reshape(B * Nk, H, D)
    dv_view = dv.reshape(B * Nk, H, D)

    # Launch lean persistent fused kernel
    persistent_lean_attention_bwd(
        q=q_lean,
        k=k_lean,
        v=v_lean,
        do=do_lean,
        o=o_lean,
        softmax_lse=softmax_lse,
        dq=dq_view,
        dk=dk_view,
        dv=dv_view,
        batch_num_block_n=None,
        batch_size=B,
        sm_scale=sm_scale,
        causal=causal,
        config=config,
        num_programs=None,
    )

    # Return delta to match API (same as softmax_lse shape: [B, H, Nq])
    delta = (do * o).sum(dim=-1)  # [B, Nq, H]
    delta = delta.permute(0, 2, 1).contiguous()  # [B, H, Nq]
    return delta
