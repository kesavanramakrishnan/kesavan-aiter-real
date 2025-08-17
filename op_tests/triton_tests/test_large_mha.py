import argparse
import json
import math
import os
from typing import Tuple

import torch

from aiter.ops.triton.lean_atten_bwd_clean import persistent_lean_attention_bwd
from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp16", "float16", "half"):  # default
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "single"):
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {s}")


@torch.no_grad()
def _compute_lse_streaming(
    q_bhmd: torch.Tensor,
    k_bhdn: torch.Tensor,
    sm_scale: float,
    causal: bool,
    block_n: int,
) -> torch.Tensor:
    """
    Compute LSE = logsumexp(Q @ K^T) row-wise, without materializing logits.
    Shapes:
      q_bhmd: [B, H, NQ, D]
      k_bhdn: [B, H, D, NK]
    Returns:
      lse_bhm: [B, H, NQ] (float32)
    """
    assert q_bhmd.ndim == 4 and k_bhdn.ndim == 4
    B, H, NQ, D = q_bhmd.shape
    _, _, Dk, NK = k_bhdn.shape
    assert D == Dk
    device = q_bhmd.device
    lse = torch.empty((B, H, NQ), device=device, dtype=torch.float32)
    diag = NK - NQ  # for non-varlen causal alignment

    ar_i = torch.arange(NQ, device=device).to(torch.int64)
    for b in range(B):
        for h in range(H):
            m_row = torch.full((NQ,), -float("inf"), device=device, dtype=torch.float32)
            s_row = torch.zeros((NQ,), device=device, dtype=torch.float32)
            q = q_bhmd[b, h].to(torch.float32)  # [NQ, D]
            for ks in range(0, NK, block_n):
                ke = min(ks + block_n, NK)
                n = ke - ks
                k_chunk = k_bhdn[b, h, :, ks:ke].to(torch.float32)  # [D, n]
                logits = (q @ k_chunk) * sm_scale  # [NQ, n]
                if causal:
                    ar_j = (ks + torch.arange(n, device=device)).to(torch.int64)[None, :]
                    mask = (ar_i[:, None] + diag) >= ar_j  # [NQ, n]
                    logits = torch.where(mask, logits, torch.full_like(logits, -float("inf")))

                chunk_max = torch.max(logits, dim=1).values  # [NQ]
                m_new = torch.maximum(m_row, chunk_max)
                # softmax sum update: s_row = s_row*exp(m-m_new) + sum(exp(logits - m_new))
                s_row = s_row * torch.exp(m_row - m_new) + torch.sum(
                    torch.exp(logits - m_new[:, None]), dim=1
                )
                m_row = m_new

            lse[b, h] = torch.log(s_row) + m_row
    return lse


@torch.no_grad()
def _compute_o_streaming(
    q_bhmd: torch.Tensor,
    k_bhdn: torch.Tensor,
    v_bhdn: torch.Tensor,
    lse_bhm: torch.Tensor,
    sm_scale: float,
    causal: bool,
    block_n: int,
) -> torch.Tensor:
    """
    Compute O = softmax(QK^T) V row-wise, using the provided per-row LSE, without
    materializing logits or P.
    Returns O_bhmd: [B, H, NQ, D]
    """
    B, H, NQ, D = q_bhmd.shape
    NK = k_bhdn.shape[-1]
    device = q_bhmd.device
    O = torch.zeros_like(q_bhmd, dtype=q_bhmd.dtype)
    diag = NK - NQ

    ar_i = torch.arange(NQ, device=device).to(torch.int64)
    for b in range(B):
        for h in range(H):
            q = q_bhmd[b, h].to(torch.float32)  # [NQ, D]
            lse_row = lse_bhm[b, h]  # [NQ], float32
            acc = torch.zeros((NQ, D), device=device, dtype=torch.float32)
            for ks in range(0, NK, block_n):
                ke = min(ks + block_n, NK)
                n = ke - ks
                k_chunk = k_bhdn[b, h, :, ks:ke].to(torch.float32)  # [D, n]
                v_chunk = v_bhdn[b, h, :, ks:ke].to(torch.float32)  # [D, n]
                logits = (q @ k_chunk) * sm_scale  # [NQ, n]
                if causal:
                    ar_j = (ks + torch.arange(n, device=device)).to(torch.int64)[None, :]
                    mask = (ar_i[:, None] + diag) >= ar_j
                    logits = torch.where(mask, logits, torch.full_like(logits, -float("inf")))
                # p_chunk = exp(logits - lse)
                p_chunk = torch.exp(logits - lse_row[:, None])  # [NQ, n]
                acc += p_chunk @ v_chunk.T  # [NQ, D]
            O[b, h] = acc.to(O.dtype)
    return O


def _save_outputs_text(path: str, header: dict, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        def dump_tensor(name: str, t: torch.Tensor):
            f.write(f"# tensor {name}\n")
            flat = t.flatten().to(torch.float32).cpu()
            # write in chunks to limit memory overhead
            chunk = 1_000_000
            numel = flat.numel()
            for start in range(0, numel, chunk):
                end = min(start + chunk, numel)
                buf = flat[start:end].tolist()
                f.write("\n".join(f"{x:.8e}" for x in buf) + ("\n" if end < numel else ""))
            f.write(f"\n# end {name}\n")
        dump_tensor("dq", dq)
        dump_tensor("dk", dk)
        dump_tensor("dv", dv)


def _compare_text_files(path_a: str, path_b: str, atol: float, rtol: float) -> int:
    def _iter_values(path):
        with open(path, "r") as f:
            header = json.loads(f.readline())
            name = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("# tensor "):
                    name = line.split("# tensor ", 1)[1]
                    continue
                if line.startswith("# end "):
                    name = None
                    continue
                if name is not None:
                    yield name, float(line)
    it_a = _iter_values(path_a)
    it_b = _iter_values(path_b)
    total = 0
    within = 0
    max_abs = 0.0
    per_tensor = {"dq": {"total": 0, "within": 0, "max_abs": 0.0},
                  "dk": {"total": 0, "within": 0, "max_abs": 0.0},
                  "dv": {"total": 0, "within": 0, "max_abs": 0.0}}
    try:
        while True:
            name_a, val_a = next(it_a)
            name_b, val_b = next(it_b)
            if name_a != name_b:
                print(f"section mismatch: {name_a} vs {name_b}")
                return 2
            err = abs(val_a - val_b)
            ref = abs(val_b)
            ok = err <= (atol + rtol * ref)
            total += 1
            per = per_tensor[name_a]
            per["total"] += 1
            if ok:
                within += 1
                per["within"] += 1
            if err > max_abs:
                max_abs = err
            if err > per["max_abs"]:
                per["max_abs"] = err
    except StopIteration:
        pass
    ratio = within / max(1, total)
    print(f"compare: within tol {within}/{total} = {ratio*100:.6f}% (max_abs={max_abs:.3e})")
    for key in ("dq", "dk", "dv"):
        p = per_tensor[key]
        pr = p["within"] / max(1, p["total"]) * 100.0
        print(f"  {key}: within {p['within']}/{p['total']} = {pr:.6f}% (max_abs={p['max_abs']:.3e})")
    return 0 if within == total else 1


@torch.no_grad()
def run_la(
    B: int,
    HQ: int,
    HK: int,
    NQ: int,
    NK: int,
    D: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    out_path: str,
    lse_block_n: int,
):
    torch.manual_seed(seed)
    q = torch.randn((B, HQ, NQ, D), dtype=dtype, device=device)
    k = torch.randn((B, HK, NK, D), dtype=dtype, device=device)
    v = torch.randn((B, HK, NK, D), dtype=dtype, device=device)

    # For simplicity, require HQ == HK (same as typical fused case)
    if HQ != HK:
        g = HQ // HK
        if HQ % HK != 0:
            raise ValueError("HQ must be a multiple of HK for GQA replication")
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    sm_scale = 1.0 / math.sqrt(D)
    # Shapes for streaming helpers
    q_bhmd = q  # [B, H, NQ, D]
    k_bhdn = k.permute(0, 1, 3, 2).contiguous()  # [B, H, D, NK]
    v_bhdn = v.permute(0, 1, 3, 2).contiguous()  # [B, H, D, NK]

    lse = _compute_lse_streaming(q_bhmd.to(torch.float32), k_bhdn.to(torch.float32), sm_scale, causal, lse_block_n)
    o = _compute_o_streaming(q_bhmd, k_bhdn, v_bhdn, lse, sm_scale, causal, lse_block_n)

    # Random upstream gradient do
    do = torch.randn_like(o)

    # Re-layout for lean attention launcher
    q_flat = q.permute(0, 2, 1, 3).contiguous().view(B * NQ, HQ, D)
    k_flat = k.permute(0, 2, 1, 3).contiguous().view(B * NK, HQ, D)
    v_flat = v.permute(0, 2, 1, 3).contiguous().view(B * NK, HQ, D)
    o_flat = o.permute(0, 2, 1, 3).contiguous().view(B * NQ, HQ, D)
    do_flat = do.permute(0, 2, 1, 3).contiguous().view(B * NQ, HQ, D)

    dq = torch.empty_like(q_flat)
    dk = torch.empty_like(k_flat)
    dv = torch.empty_like(v_flat)

    # For non-causal multi-batch, build batch_num_block_n as needed
    batch_num_block_n = None
    if (not causal) and (B > 1):
        BLOCK_N = 64
        num_n_blocks = (NK + BLOCK_N - 1) // BLOCK_N
        batch_num_block_n = (
            torch.arange(1, B + 1, device=device, dtype=torch.int32) * num_n_blocks
        )

    persistent_lean_attention_bwd(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        do=do_flat,
        o=o_flat,
        softmax_lse=lse.to(torch.float32),
        dq=dq,
        dk=dk,
        dv=dv,
        batch_num_block_n=batch_num_block_n,
        batch_size=B,
        sm_scale=sm_scale,
        causal=causal,
        seqlen_k=NK,
    )

    header = {
        "mode": "la",
        "B": B,
        "H": HQ,
        "NQ": NQ,
        "NK": NK,
        "D": D,
        "causal": int(causal),
        "dtype": str(dtype),
        "order": "dq,dk,dv",
    }
    _save_outputs_text(out_path, header, dq, dk, dv)
    print(f"wrote LA outputs to {out_path}")


@torch.no_grad()
def run_flash(
    B: int,
    HQ: int,
    HK: int,
    NQ: int,
    NK: int,
    D: int,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    out_path: str,
    lse_block_n: int,
):
    torch.manual_seed(seed)
    # We construct in [B, H, NQ/NK, D] for streaming compute, then permute to
    # the kernel-expected [B, NQ/NK, H, D] before launching flash-attn.
    q = torch.randn((B, HQ, NQ, D), dtype=dtype, device=device)  # [B, H, NQ, D]
    k = torch.randn((B, HK, NK, D), dtype=dtype, device=device)  # [B, H, NK, D]
    v = torch.randn((B, HK, NK, D), dtype=dtype, device=device)  # [B, H, NK, D]
    sm_scale = 1.0 / math.sqrt(D)

    # Repeat K/V heads to match Q heads if needed (GQA)
    if HQ != HK:
        g = HQ // HK
        if HQ % HK != 0:
            raise ValueError("HQ must be a multiple of HK for GQA replication")
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    # Compute LSE and O streaming to match LA's inputs
    q_bhmd = q
    k_bhdn = k.permute(0, 1, 3, 2).contiguous()
    v_bhdn = v.permute(0, 1, 3, 2).contiguous()
    lse = _compute_lse_streaming(q_bhmd.to(torch.float32), k_bhdn.to(torch.float32), sm_scale, causal, lse_block_n)
    o = _compute_o_streaming(q_bhmd, k_bhdn, v_bhdn, lse, sm_scale, causal, lse_block_n)
    do = torch.randn_like(o)

    # Layouts expected by one-kernel wrapper (BSHD = [B, seqlen, H, D])
    q_bsnh = q.permute(0, 2, 1, 3).contiguous()
    k_bsnh = k.permute(0, 2, 1, 3).contiguous()
    v_bsnh = v.permute(0, 2, 1, 3).contiguous()
    o_bsnh = o.permute(0, 2, 1, 3).contiguous()
    do_bsnh = do.permute(0, 2, 1, 3).contiguous()

    dq = torch.zeros_like(q_bsnh)
    dk = torch.zeros_like(k_bsnh)
    dv = torch.zeros_like(v_bsnh)

    flash_attn_onekernel_backward(
        do=do_bsnh,
        q=q_bsnh,
        k=k_bsnh,
        v=v_bsnh,
        o=o_bsnh,
        softmax_lse=lse,
        dq=dq,
        dk=dk,
        dv=dv,
        dbias=None,
        sm_scale=sm_scale,
        alibi_slopes=None,
        causal=causal,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=NQ,
        max_seqlen_k=NK,
        dropout_p=0.0,
    )

    header = {
        "mode": "flash",
        "B": B,
        "H": HQ,
        "NQ": NQ,
        "NK": NK,
        "D": D,
        "causal": int(causal),
        "dtype": str(dtype),
        "order": "dq,dk,dv",
    }
    _save_outputs_text(out_path, header, dq, dk, dv)
    print(f"wrote Flash outputs to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Large MHA: run LA or Flash separately and save outputs to text, or compare.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-l", "--lean", action="store_true", help="Run Lean Attention and save outputs")
    mode.add_argument("-f", "--flash", action="store_true", help="Run Flash Attention and save outputs")
    mode.add_argument("-c", "--compare", nargs=2, metavar=("LA_FILE", "FLASH_FILE"), help="Compare two saved output files")

    parser.add_argument("--out", type=str, default="/tmp/mha_outputs.txt", help="Output text file path for -l/-f")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--HQ", type=int, default=16)
    parser.add_argument("--HK", type=int, default=None)
    parser.add_argument("--NQ", type=int, default=4096)
    parser.add_argument("--NK", type=int, default=4096)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--lse_block_n", type=int, default=2048, help="Block size over NK for streaming LSE/O compute")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)

    args = parser.parse_args()

    if args.compare:
        code = _compare_text_files(args.compare[0], args.compare[1], args.atol, args.rtol)
        raise SystemExit(code)

    device = torch.device(args.device)
    dtype = _dtype_from_str(args.dtype)
    HK = args.HQ if args.HK is None else int(args.HK)

    if args.lean:
        run_la(
            B=args.B,
            HQ=args.HQ,
            HK=HK,
            NQ=args.NQ,
            NK=args.NK,
            D=args.D,
            causal=args.causal,
            dtype=dtype,
            device=device,
            seed=args.seed,
            out_path=args.out,
            lse_block_n=args.lse_block_n,
        )
        return
    if args.flash:
        run_flash(
            B=args.B,
            HQ=args.HQ,
            HK=HK,
            NQ=args.NQ,
            NK=args.NK,
            D=args.D,
            causal=args.causal,
            dtype=dtype,
            device=device,
            seed=args.seed,
            out_path=args.out,
            lse_block_n=args.lse_block_n,
        )
        return


if __name__ == "__main__":
    main()


