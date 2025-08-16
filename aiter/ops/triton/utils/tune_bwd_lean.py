import math
import time
import itertools
import argparse
import json
from typing import Dict, List, Tuple

import torch

from aiter.ops.triton.lean_atten_bwd_clean import persistent_lean_attention_bwd


def _default_shape_grid() -> List[Dict]:
	# Scenarios restricted to match benchmark configs
	bench_configs = [
		(16, 16, 16, 1024, 1024),
		(8, 16, 16, 2048, 2048),
		(4, 16, 16, 4096, 4096),
		(2, 16, 16, 8192, 8192),
		(8, 16, 16, 1024, 4096),
		(1, 16, 16, 4096, 16384),
		(2, 48, 48, 1024, 1024),
		(2, 48, 48, 2048, 1024),
		(2, 48, 48, 4096, 8192),
		(2, 48, 48, 8192, 4096),
        (2, 48, 48, 128, 8192),
        (2, 48, 48, 128, 4096),
        (2, 48, 48, 128, 1024),
        (2, 48, 48, 128, 2048),
	]
	scenarios: List[Dict] = []
	# Use 128 for head dimension by default (matches kernel-supported head sizes)
	head_dim = 128
	causal = False
	for (B, HQ, HK, NQ, NK) in bench_configs:
		scenarios.append({
			"batch": B,
			"heads": HQ,
			"head_dim": head_dim,
			"n_ctx_q": NQ,
			"n_ctx_k": NK,
			"causal": causal,
		})
	return scenarios


def _default_config_grid() -> List[Dict]:
    # Tunable space (narrow and stable by default)
    block_m_q = [64, 128]
    block_n_q = [64]  # stable for Q-path for now
    num_warps_q = [2, 4]

    block_m_kv = [32]
    # Try larger N for KV tiling to improve arithmetic intensity
    block_n_kv = [64, 128]
    num_warps_kv = [1, 2, 4]

    waves_per_eu_vals = [1]
    num_programs_mult_vals = [1, 2, 3, 4, 5]
    prefetch_qt_vals = [2]
    prefetch_kv_vals = [2]
    num_stages_vals = [1, 2]  # Add num_stages tuning

    cfgs = []
    for (bmq, bnq, nwq, bmk, bnk, nwk, wpe, np_mult, pf_qt, pf_kv, ns) in itertools.product(
        block_m_q, block_n_q, num_warps_q, block_m_kv, block_n_kv, num_warps_kv,
        waves_per_eu_vals, num_programs_mult_vals, prefetch_qt_vals, prefetch_kv_vals, num_stages_vals
    ):
        cfgs.append({
            "BLOCK_SIZE_M": bmq,
            "BLOCK_SIZE_N": bnq,
            "num_warps": nwq,
            "BLOCK_SIZE_M_KV": bmk,
            "BLOCK_SIZE_N_KV": bnk,
            "num_warps_kv": nwk,
            "waves_per_eu": wpe,
            "num_programs_mult": np_mult,
            # Fused-friendly defaults; can be overridden by CLI/DB
            "fused": True,
            "prefetch_qt": pf_qt,
            "prefetch_kv": pf_kv,
            "num_stages": ns,  # Add num_stages to config
        })
    return cfgs


def allocate_tensors(scn: Dict, dtype: torch.dtype, device: torch.device):
    B = scn["batch"]
    H = scn["heads"]
    D = scn["head_dim"]
    NQ = scn["n_ctx_q"]
    NK = scn["n_ctx_k"]

    # Layouts expected by persistent_lean_attention_bwd
    q = torch.randn((B * NQ, H, D), device=device, dtype=dtype)
    k = torch.randn((NK, H, D), device=device, dtype=dtype)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    o = torch.randn_like(q)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(k)
    # LSE will be computed per scenario in run_once for numerical stability
    softmax_lse = torch.empty((B, H, NQ), device=device, dtype=torch.float32)

    return q, k, v, do, o, softmax_lse, dq, dk, dv


@torch.no_grad()
def run_once(scn: Dict, cfg: Dict, dtype: torch.dtype, device: torch.device, warmup: int, iters: int) -> Tuple[float, float]:
    q, k, v, do, o, lse, dq, dk, dv = allocate_tensors(scn, dtype, device)
    scale = 1.0 / math.sqrt(scn["head_dim"])
    # Derive num_programs from multiplier if present; user can still override externally
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    num_programs = None
    if "num_programs_mult" in cfg:
        try:
            num_programs = max(1, int(cfg["num_programs_mult"]) * int(sm_count))
        except Exception:
            num_programs = None

    # ensure fused by default for this tuner variant
    cfg["fused"] = bool(cfg.get("fused", True))
    # prefetch knobs passed to runtime
    cfg["prefetch_qt"] = int(cfg.get("prefetch_qt", 1))
    cfg["prefetch_kv"] = int(cfg.get("prefetch_kv", 1))

    # ensure deterministic-ish
    torch.cuda.synchronize()

    # Precompute numerically-stable softmax LSE (B, H, NQ)
    B = scn["batch"]; H = scn["heads"]; NQ = scn["n_ctx_q"]; NK = scn["n_ctx_k"]
    # Reshape to (B, H, NQ, D) and (B, H, D, NK)
    q_bhmd = q.view(B, NQ, H, scn["head_dim"]).permute(0, 2, 1, 3).contiguous()
    k_bhdn = k.view(NK, H, scn["head_dim"]).permute(1, 2, 0).contiguous().unsqueeze(0).expand(B, -1, -1, -1)
    logits = torch.matmul(q_bhmd.to(torch.float32), k_bhdn.to(torch.float32)) * scale
    if scn["causal"]:
        mask = torch.ones(NQ, NK, device=device, dtype=torch.bool).tril(diagonal=(NK - NQ))
        logits.masked_fill_(~mask, float('-inf'))
    lse = torch.logsumexp(logits, dim=-1).to(torch.float32)
    # warmup
    for _ in range(max(0, warmup)):
        persistent_lean_attention_bwd(
            q, k, v, do, o, lse, dq, dk, dv,
            batch_num_block_n=None,
            batch_size=scn["batch"],
            sm_scale=scale,
            causal=scn["causal"],
            config=cfg,
            num_programs=num_programs,
            seqlen_k=NK,
        )
    torch.cuda.synchronize()

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        persistent_lean_attention_bwd(
            q, k, v, do, o, lse, dq, dk, dv,
            batch_num_block_n=None,
            batch_size=scn["batch"],
            sm_scale=scale,
            causal=scn["causal"],
            config=cfg,
            num_programs=num_programs,
            seqlen_k=NK,
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / max(1, iters)

    # Optional: very rough flop model for backward attention; use time if unsure
    # Here we report only time in ms and derived TFLOP/s using an approximate formula.
    B = scn["batch"]; H = scn["heads"]; D = scn["head_dim"]; NQ = scn["n_ctx_q"]; NK = scn["n_ctx_k"]
    # Backward approx: 6 * B * H * NQ * NK * D MACs -> 12 flops
    approx_flops = 12.0 * B * H * NQ * NK * D
    tflops = approx_flops / (ms * 1e-3) / 1e12

    return ms, tflops


def format_scn(s: Dict) -> str:
    return f"B={s['batch']} H={s['heads']} D={s['head_dim']} NQ={s['n_ctx_q']} NK={s['n_ctx_k']} causal={int(s['causal'])}"


def main():
    parser = argparse.ArgumentParser(description="Autotune lean attention backward (split kernels)")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"], type=str)
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--limit_scenarios", default=None, type=int, help="Limit number of scenarios for quick runs")
    parser.add_argument("--limit_configs", default=None, type=int, help="Limit number of configs per scenario")
    parser.add_argument("--out", type=str, default="bwd_split_autotune.json", help="Path to write best-config DB as JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    scenarios = _default_shape_grid()
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    cfg_grid = _default_config_grid()
    if args.limit_configs is not None:
        cfg_grid = cfg_grid[: args.limit_configs]

    print(f"Running autotune on {len(scenarios)} scenarios × {len(cfg_grid)} configs")

    results: List[Tuple[Dict, Dict, float, float]] = []  # (scn, cfg, ms, tflops)

    for scn in scenarios:
        best_ms = float("inf"); best_cfg = None; best_tflops = 0.0
        # Filter illegal/unstable configs for causal and known-bad mixes
        legal_cfgs = []
        for cfg in cfg_grid:
            legal = True
            if scn["causal"]:
                if (cfg["BLOCK_SIZE_M"] % cfg["BLOCK_SIZE_N"]) != 0:
                    legal = False
                if (cfg["BLOCK_SIZE_M_KV"] % cfg["BLOCK_SIZE_N_KV"]) != 0:
                    legal = False
                # Prefer BM >= BN in causal to avoid large masked tails
                if cfg["BLOCK_SIZE_M"] < cfg["BLOCK_SIZE_N"]:
                    legal = False
                if cfg["BLOCK_SIZE_M_KV"] < cfg["BLOCK_SIZE_N_KV"]:
                    legal = False
            # Force num_programs_mult default to 2 for fused unless user overrides
            if cfg.get("fused", True) and "num_programs_mult" not in cfg:
                cfg["num_programs_mult"] = 2
            # Heuristic: very small N tiles with high warps are unstable on some backends
            if cfg["BLOCK_SIZE_N"] == 32 and cfg["num_warps"] >= 8:
                legal = False
            if cfg["BLOCK_SIZE_N_KV"] == 32 and cfg["num_warps_kv"] >= 8:
                legal = False
            # Stronger filter: avoid N=32 entirely for now
            if cfg["BLOCK_SIZE_N"] == 32 or cfg["BLOCK_SIZE_N_KV"] == 32:
                legal = False
            # Avoid 8 warps on small tiles to reduce compiler pressure
            if cfg["num_warps"] >= 8 and (cfg["BLOCK_SIZE_M"] <= 64 or cfg["BLOCK_SIZE_N"] <= 64):
                legal = False
            if cfg["num_warps_kv"] >= 8 and (cfg["BLOCK_SIZE_M_KV"] <= 64 or cfg["BLOCK_SIZE_N_KV"] <= 64):
                legal = False
            # Also filter ridiculous tile larger than context
            if cfg["BLOCK_SIZE_M"] > scn["n_ctx_q"] or cfg["BLOCK_SIZE_M_KV"] > scn["n_ctx_q"]:
                legal = False
            if not legal:
                continue
            legal_cfgs.append(cfg)

        for cfg in legal_cfgs:
            try:
                ms, tflops = run_once(scn, cfg, dtype, device, args.warmup, args.iters)
            except RuntimeError as e:
                # Skip failing configs
                print(f"[skip] {format_scn(scn)} cfg={cfg} error={str(e)}")
                continue
            if ms < best_ms:
                best_ms = ms; best_cfg = cfg; best_tflops = tflops

        if best_cfg is not None:
            results.append((scn, best_cfg, best_ms, best_tflops))
            print(f"[best] {format_scn(scn)} -> {best_ms:.3f} ms, {best_tflops:.2f} TF/s, cfg={best_cfg}")
        else:
            print(f"[none] {format_scn(scn)} -> no legal/working config")

    # Summary table
    print("\n==== Summary (best per scenario) ====")
    for scn, cfg, ms, tf in results:
        print(f"{format_scn(scn)} | {ms:.3f} ms | {tf:.2f} TF/s | cfg={cfg}")

    # Write config DB to disk
    db = []
    for scn, cfg, ms, tf in results:
        # Include optional num_programs_mult so runtime can scale CTA count to device
        entry_cfg = dict(cfg)
        if "num_programs_mult" not in entry_cfg:
            entry_cfg["num_programs_mult"] = 1
        db.append({
            "key": {
                "causal": int(scn["causal"]),
                "B": scn["batch"],
                "H": scn["heads"],
                "D": scn["head_dim"],
                "NQ": scn["n_ctx_q"],
                "NK": scn["n_ctx_k"],
            },
            "config": entry_cfg,
            "ms": ms,
            "tflops": tf,
        })
    try:
        with open(args.out, "w") as f:
            json.dump(db, f)
        print(f"\nWrote {len(db)} best-config entries to {args.out}")
        print("To use it, set AITER_BWD_TUNED_DB to this path. Optionally set AITER_BWD_USE_TUNED_GRID=1 to scale CTAs.")
    except Exception as e:
        print(f"Failed to write autotune DB to {args.out}: {e}")


if __name__ == "__main__":
    main()


