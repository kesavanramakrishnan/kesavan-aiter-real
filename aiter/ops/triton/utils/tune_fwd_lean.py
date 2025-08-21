import math
import time
import itertools
import argparse
import json
from typing import Dict, List, Tuple

import torch

from aiter.ops.triton.lean_atten_lse import persistent_lean_attention
import aiter.ops.triton.utils.arch_info as arch_info


def _default_shape_grid() -> List[Dict]:
	# Scenarios inspired by benchmark configs
	scenarios: List[Dict] = []
	# Use 128 for head dimension by default
	head_dim = 128
	# Favor prefill-like, large context, causal=True
	bench_configs = [
		(2, 16, 16, 1024, 1024, True),
		(2, 16, 16, 4096, 4096, True),
		(2, 16, 16, 8192, 8192, True),
		(2, 48, 48, 4096, 8192, True),
		(2, 48, 48, 8192, 4096, True),
		(2, 16, 16, 16384, 16384, True),
		(2, 48, 48, 16384, 16384, True),
	]
	for (B, HQ, HK, NQ, NK, causal) in bench_configs:
		scenarios.append({
			"batch": B,
			"heads": HQ,
			"heads_k": HK,
			"head_dim": head_dim,
			"n_ctx_q": NQ,
			"n_ctx_k": NK,
			"causal": causal,
		})
	return scenarios


def _default_config_grid() -> List[Dict]:
	# Tunable space (forward fused persistent)
	block_m_vals = [128, 64, 32]
	block_n_vals = [128, 64]
	num_warps_vals = [2, 4]
	waves_per_eu_vals = [1, 2]

	cfgs = []
	for bm, bn, nw, wpe in itertools.product(block_m_vals, block_n_vals, num_warps_vals, waves_per_eu_vals):
		cfgs.append({
			"BLOCK_SIZE_M": bm,
			"BLOCK_SIZE_N": bn,
			"num_warps": nw,
			"waves_per_eu": wpe,
		})
	return cfgs


def allocate_tensors(scn: Dict, dtype: torch.dtype, device: torch.device, cfg: Dict):
	B = scn["batch"]
	H = scn["heads"]
	D = scn["head_dim"]
	NQ = scn["n_ctx_q"]
	NK = scn["n_ctx_k"]

	# Layouts expected by persistent_lean_attention
	q = torch.randn((B * NQ, H, D), device=device, dtype=dtype)
	k = torch.randn((NK, H, D), device=device, dtype=dtype)
	v = torch.randn_like(k)

	# Determine total programs and BLOCK sizes to size temp buffers
	try:
		total_programs = int(arch_info.get_num_sms())
	except Exception:
		total_programs = torch.cuda.get_device_properties(device).multi_processor_count
	bm = int(cfg["BLOCK_SIZE_M"]) if "BLOCK_SIZE_M" in cfg else 128

	# Temp buffers: Mp/Lp: [total_programs, BLOCK_M], fp32; Op: [total_programs, NQ, D], same dtype as v
	Mp = torch.empty((total_programs, bm), device=device, dtype=torch.float32)
	Lp = torch.empty_like(Mp)
	Op = torch.empty((total_programs, NQ, D), device=device, dtype=dtype)
	locks = torch.zeros((total_programs,), device=device, dtype=torch.int32)

	# For non-causal multi-batch you would pass cumulative n-blocks; we default None in causal or B==1
	batch_num_block_n = None
	return q, k, v, Mp, Lp, Op, locks, batch_num_block_n


@torch.no_grad()
def run_once(scn: Dict, cfg: Dict, dtype: torch.dtype, device: torch.device, warmup: int, iters: int) -> Tuple[float, float]:
	q, k, v, Mp, Lp, Op, locks, batch_num_block_n = allocate_tensors(scn, dtype, device, cfg)
	scale = 1.0 / math.sqrt(scn["head_dim"])
	B = scn["batch"]
	NK = scn["n_ctx_k"]

	# warmup
	for _ in range(max(0, warmup)):
		_ = persistent_lean_attention(
			q=q, k=k, v=v,
			Mp=Mp, Lp=Lp, Op=Op, locks=locks,
			batch_num_block_n=batch_num_block_n,
			batch_size=B,
			sm_scale=scale,
			causal=scn["causal"],
			config=cfg,
			return_lse=False,
		)
		# persistent_lean_attention returns (o, ms) or (o, lse, ms); ignore return here
	torch.cuda.synchronize()

	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	start.record()
	for _ in range(max(1, iters)):
		_ = persistent_lean_attention(
			q=q, k=k, v=v,
			Mp=Mp, Lp=Lp, Op=Op, locks=locks,
			batch_num_block_n=batch_num_block_n,
			batch_size=B,
			sm_scale=scale,
			causal=scn["causal"],
			config=cfg,
			return_lse=False,
		)
	end.record()
	torch.cuda.synchronize()
	ms = start.elapsed_time(end) / max(1, iters)

	# Simple FLOPs model for forward attention: ~2 * B * H * NQ * NK * D matmuls (QK^T and P*V)
	H = scn["heads"]
	D = scn["head_dim"]
	NQ = scn["n_ctx_q"]
	approx_flops = 4.0 * B * H * NQ * NK * D
	tflops = approx_flops / (ms * 1e-3) / 1e12
	return ms, tflops


def format_scn(s: Dict) -> str:
	return f"B={s['batch']} H={s['heads']} D={s['head_dim']} NQ={s['n_ctx_q']} NK={s['n_ctx_k']} causal={int(s['causal'])}"


def main():
	parser = argparse.ArgumentParser(description="Autotune Lean Attention forward (persistent)")
	parser.add_argument("--device", default="cuda", type=str)
	parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"], type=str)
	parser.add_argument("--iters", default=10, type=int)
	parser.add_argument("--warmup", default=10, type=int)
	parser.add_argument("--limit_scenarios", default=None, type=int, help="Limit number of scenarios for quick runs")
	parser.add_argument("--limit_configs", default=None, type=int, help="Limit number of configs per scenario")
	parser.add_argument("--out", type=str, default="fwd_lean_autotune.json", help="Path to write best-config DB as JSON")
	args = parser.parse_args()

	device = torch.device(args.device)
	dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

	scenarios = _default_shape_grid()
	if args.limit_scenarios is not None:
		scenarios = scenarios[: args.limit_scenarios]

	cfg_grid = _default_config_grid()
	if args.limit_configs is not None:
		cfg_grid = cfg_grid[: args.limit_configs]

	print(f"Running forward autotune on {len(scenarios)} scenarios × {len(cfg_grid)} configs")

	results: List[Tuple[Dict, Dict, float, float]] = []  # (scn, cfg, ms, tflops)

	for scn in scenarios:
		best_ms = float("inf"); best_cfg = None; best_tflops = float('-inf')
		# Filter illegal/unstable configs
		legal_cfgs = []
		for cfg in cfg_grid:
			legal = True
			# Causal forward prefers BLOCK_M multiple of BLOCK_N and BM >= BN
			if scn["causal"]:
				if (cfg["BLOCK_SIZE_M"] % cfg["BLOCK_SIZE_N"]) != 0:
					legal = False
				if cfg["BLOCK_SIZE_M"] < cfg["BLOCK_SIZE_N"]:
					legal = False
			# Ensure tiles not larger than sequence
			if cfg["BLOCK_SIZE_M"] > scn["n_ctx_q"]:
				legal = False
			if cfg["BLOCK_SIZE_N"] > scn["n_ctx_k"]:
				legal = False
			if not legal:
				continue
			legal_cfgs.append(cfg)

		for cfg in legal_cfgs:
			try:
				ms, tflops = run_once(scn, cfg, dtype, device, args.warmup, args.iters)
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

	# Summary
	print("\n==== Summary (best per scenario) ====")
	for scn, cfg, ms, tf in results:
		print(f"{format_scn(scn)} | {ms:.3f} ms | {tf:.2f} TF/s | cfg={cfg}")

	# Write config DB to disk
	db = []
	for scn, cfg, ms, tf in results:
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
		print("To use it, consume entries in your own forward config loader or map them into LEANATTN-DEFAULT.json by shape.")
	except Exception as e:
		print(f"Failed to write autotune DB to {args.out}: {e}")


if __name__ == "__main__":
	main()


