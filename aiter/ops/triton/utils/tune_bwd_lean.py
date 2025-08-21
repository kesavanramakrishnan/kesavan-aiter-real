import math
import time
import itertools
import argparse
import json
from typing import Dict, List, Tuple

import torch

from aiter.ops.triton.lean_atten_bwd_clean import persistent_lean_attention_bwd
from aiter.ops.triton.mha_onekernel_bwd import (
    flash_attn_onekernel_backward,
    _get_config as _fa2_get_config,
)


def _default_shape_grid() -> List[Dict]:
	# Scenarios restricted to match benchmark configs
	bench_configs = [
		# (16, 16, 16, 1024, 1024),
		# (8, 16, 16, 2048, 2048),
		# (4, 16, 16, 4096, 4096),
		# (2, 16, 16, 8192, 8192),
		# (2, 48, 48, 1024, 1024),
		# (2, 48, 48, 2048, 1024),
		# (2, 48, 48, 4096, 8192),
		# (2, 48, 48, 8192, 4096),
        # (2, 16, 16, 16384, 16384),
        # (2, 16, 16, 32768, 32768),
        # (2, 16, 16, 16384, 8192),
        # (1, 16, 16, 4096, 16384),
        # (2, 16, 16, 16384, 32768),
        (2, 16, 16, 16384, 16384),
        # (2, 48, 48, 32768, 32768),
        # (2, 48, 48, 16384, 8192),
        # (1, 48, 48, 4096, 16384),
        # (2, 48, 48, 16384, 32768),
        # (2, 48, 48, 128, 8192),
        # (2, 48, 48, 128, 4096),
        # (2, 48, 48, 128, 1024),
        # (2, 48, 48, 128, 2048),
    
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
    block_m_q = [128, 64, 32, 16]
    block_n_q = [64, 32, 16]  # stable for Q-path for now
    num_warps_q = [1, 2, 4, 8]

    block_m_kv = [16, 32, 64, 128]
    # Try larger N for KV tiling to improve arithmetic intensity
    block_n_kv = [64, 128, 16]
    num_warps_kv = [1, 2, 4, 8]

    waves_per_eu_vals = [1, 2]
    num_programs_mult_vals = [2, 3, 4, 5, 6]
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


def _fa2_default_config_grid() -> List[Dict]:
    # Tune core one-kernel params. Keep search small to avoid compile pressure.
    cfgs = []
    BLOCK_M1_vals = [32, 64]
    BLOCK_N1_vals = [128, 64]
    BLOCK_M2_vals = [128, 64]
    BLOCK_N2_vals = [32, 64]
    BLK_SLICE_FACTOR_vals = [2]
    num_warps_vals = [2, 4, 8]
    num_ctas_vals = [1]
    num_stages_vals = [1]
    for bm1 in BLOCK_M1_vals:
        for bn1 in BLOCK_N1_vals:
            for bm2 in BLOCK_M2_vals:
                for bn2 in BLOCK_N2_vals:
                    for slice_f in BLK_SLICE_FACTOR_vals:
                        for nw in num_warps_vals:
                            for nctas in num_ctas_vals:
                                for nst in num_stages_vals:
                                    cfgs.append({
                                        "BLOCK_M1": bm1,
                                        "BLOCK_N1": bn1,
                                        "BLOCK_M2": bm2,
                                        "BLOCK_N2": bn2,
                                        "BLK_SLICE_FACTOR": slice_f,
                                        "num_warps": nw,
                                        "num_ctas": nctas,
                                        "num_stages": nst,
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


@torch.no_grad()
def run_once_fa2(scn: Dict, cfg: Dict, dtype: torch.dtype, device: torch.device, warmup: int, iters: int) -> Tuple[float, float]:
    B = scn["batch"]; Hq = scn["heads"]; Hk = scn["heads"]; D = scn["head_dim"]; NQ = scn["n_ctx_q"]; NK = scn["n_ctx_k"]
    scale = 1.0 / math.sqrt(D)
    # Allocate BSHD layout to match FA2 wrapper
    q = torch.randn((B, NQ, Hq, D), device=device, dtype=dtype)
    k = torch.randn((B, NK, Hk, D), device=device, dtype=dtype)
    v = torch.randn((B, NK, Hk, D), device=device, dtype=dtype)
    o = torch.randn((B, NQ, Hq, D), device=device, dtype=dtype)
    do = torch.randn_like(o)
    lse = torch.randn((B, Hq, NQ), device=device, dtype=torch.float32)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Build a config dict by merging the default backend config and overriding onekernel
    base_cfg = _fa2_get_config()
    fa2_cfg = dict(base_cfg)
    fa2_cfg["onekernel"] = dict(fa2_cfg["onekernel"])
    fa2_cfg["onekernel"].update(cfg)

    torch.cuda.synchronize()
    for _ in range(max(0, warmup)):
        flash_attn_onekernel_backward(
            do, q, k, v, o, lse, dq, dk, dv,
            dbias=None, sm_scale=scale, alibi_slopes=None, causal=scn["causal"],
            cu_seqlens_q=None, cu_seqlens_k=None,
            max_seqlen_q=NQ, max_seqlen_k=NK,
            dropout_p=0.0,
            USE_INT64_STRIDES=False,
            config=fa2_cfg,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        flash_attn_onekernel_backward(
            do, q, k, v, o, lse, dq, dk, dv,
            dbias=None, sm_scale=scale, alibi_slopes=None, causal=scn["causal"],
            cu_seqlens_q=None, cu_seqlens_k=None,
            max_seqlen_q=NQ, max_seqlen_k=NK,
            dropout_p=0.0,
            USE_INT64_STRIDES=False,
            config=fa2_cfg,
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / max(1, iters)
    approx_flops = 12.0 * B * Hq * NQ * NK * D
    tflops = approx_flops / (ms * 1e-3) / 1e12
    return ms, tflops


def format_scn(s: Dict) -> str:
    return f"B={s['batch']} H={s['heads']} D={s['head_dim']} NQ={s['n_ctx_q']} NK={s['n_ctx_k']} causal={int(s['causal'])}"


def main():
    parser = argparse.ArgumentParser(description="Autotune attention backward kernels")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"], type=str)
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--limit_scenarios", default=None, type=int, help="Limit number of scenarios for quick runs")
    parser.add_argument("--limit_configs", default=None, type=int, help="Limit number of configs per scenario")
    parser.add_argument("--out", type=str, default="bwd_split_autotune.json", help="Path to write best-config DB as JSON")
    parser.add_argument("--algo", type=str, choices=["lean", "fa2"], default="lean", help="Which kernel to tune: 'lean' (persistent) or 'fa2' (flash one-kernel)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    scenarios = _default_shape_grid()
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    cfg_grid = _default_config_grid() if args.algo == "lean" else _fa2_default_config_grid()
    if args.limit_configs is not None:
        cfg_grid = cfg_grid[: args.limit_configs]

    print(f"Running autotune on {len(scenarios)} scenarios × {len(cfg_grid)} configs")

    results: List[Tuple[Dict, Dict, float, float]] = []  # (scn, cfg, ms, tflops)

    for scn in scenarios:
        best_ms = float("inf"); best_cfg = None; best_tflops = float('-inf')
        # Filter illegal/unstable configs for causal and known-bad mixes
        legal_cfgs = []
        for cfg in cfg_grid:
            legal = True
            if args.algo == "lean":
                if scn["causal"]:
                    if (cfg["BLOCK_SIZE_M"] % cfg["BLOCK_SIZE_N"]) != 0:
                        legal = False
                    if (cfg["BLOCK_SIZE_M_KV"] % cfg["BLOCK_SIZE_N_KV"]) != 0:
                        legal = False
                    if cfg["BLOCK_SIZE_M"] < cfg["BLOCK_SIZE_N"]:
                        legal = False
                    if cfg["BLOCK_SIZE_M_KV"] < cfg["BLOCK_SIZE_N_KV"]:
                        legal = False
                if cfg.get("fused", True) and "num_programs_mult" not in cfg:
                    cfg["num_programs_mult"] = 2
                if cfg["BLOCK_SIZE_N"] == 32 and cfg["num_warps"] >= 8:
                    legal = False
                if cfg["BLOCK_SIZE_N_KV"] == 32 and cfg["num_warps_kv"] >= 8:
                    legal = False
                if cfg["BLOCK_SIZE_N"] == 32 or cfg["BLOCK_SIZE_N_KV"] == 32:
                    legal = False
                if cfg["num_warps"] >= 8 and (cfg["BLOCK_SIZE_M"] <= 64 or cfg["BLOCK_SIZE_N"] <= 64):
                    legal = False
                if cfg["num_warps_kv"] >= 8 and (cfg["BLOCK_SIZE_M_KV"] <= 64 or cfg["BLOCK_SIZE_N_KV"] <= 64):
                    legal = False
                if cfg["BLOCK_SIZE_M"] > scn["n_ctx_q"] or cfg["BLOCK_SIZE_M_KV"] > scn["n_ctx_q"]:
                    legal = False
            else:
                # FA2 legality: ensure integers and reasonable ranges
                for key in ("BLOCK_M1","BLOCK_N1","BLOCK_M2","BLOCK_N2","BLK_SLICE_FACTOR"):
                    if key not in cfg or int(cfg[key]) <= 0:
                        legal = False
                if cfg.get("num_warps", 4) not in (2,4,8):
                    legal = False
                if cfg.get("num_ctas", 1) not in (1,2):
                    legal = False
                if cfg.get("num_stages", 1) not in (1,2):
                    legal = False
                # Optional: reject absurd blocks
                if cfg["BLOCK_M2"] > scn["n_ctx_q"]:
                    legal = False
            if not legal:
                continue
            legal_cfgs.append(cfg)

        for cfg in legal_cfgs:
            try:
                if args.algo == "lean":
                    ms, tflops = run_once(scn, cfg, dtype, device, args.warmup, args.iters)
                else:
                    ms, tflops = run_once_fa2(scn, cfg, dtype, device, args.warmup, args.iters)
            except RuntimeError as e:
                print(f"[skip] {format_scn(scn)} cfg={cfg} error={str(e)}")
                continue
            if tflops > best_tflops:
                best_tflops = tflops; best_cfg = cfg; best_ms = ms

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
        if args.algo == "lean":
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
        else:
            # For FA2, store only the onekernel sub-config; runtime will merge with base
            entry_cfg = dict(cfg)
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
        if args.algo == "lean":
            print("To use it, set AITER_BWD_TUNED_DB to this path. Optionally set AITER_BWD_USE_TUNED_GRID=1 to scale CTAs.")
        else:
            print("For FA2, merge 'config' entries into the 'onekernel' section of your backend config or feed them via the config parameter.")
    except Exception as e:
        print(f"Failed to write autotune DB to {args.out}: {e}")


if __name__ == "__main__":
    main()


