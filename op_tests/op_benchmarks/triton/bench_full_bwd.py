import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import triton
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_configs,
    get_available_models,
    print_vgpr,
)
import torch
import sys
import warnings
import argparse

from aiter.ops.triton.mha import (
    flash_attn_func,
    flash_attn_fp8_func,
    flash_attn_varlen_func,
    flash_attn_varlen_fp8_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.test_mha_common import (
    generate_random_padding_mask,
    generate_qkv,
)


def nonvarlen_benchmark_configs():
    configs = [
        # (2, 48, 48, 128, 8192),
        # (2, 48, 48, 128, 4096),
        # (2, 48, 48, 128, 1024),
        # (2, 48, 48, 128, 2048),
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
        (2, 48, 48, 16384, 8192),
        (2, 48, 48, 8192, 16384),
        (2, 16, 16, 16384, 16384),
        (2, 16, 16, 16384, 32768),
        (2, 16, 16, 32768, 16384),
        (2, 16, 16, 32768, 32768),
        
        # (2, 48, 48, 16384, 8192),
        # (8, 16, 16, 1989, 15344),
        # (4, 16, 16, 4097, 163),
        # (2, 16, 16, 8122, 2159),
        # (1, 16, 16, 16281, 7),
        # (2, 48, 48, 1021, 1020),
        # (2, 48, 48, 2001, 2048),
        # (2, 48, 48, 3996, 9639),
        # (2, 48, 48, 8181, 1021),
    ]
    return configs


def varlen_benchmark_configs():
    configs = [
        # (2, 16, 4, 1024, 1024),
        # (8, 16, 2, 2048, 2048),
        # (4, 16, 8, 4096, 4096),
        # (2, 16, 4, 8192, 8192),
        # (2, 16, 8, 16384, 16384),
        # (2, 48, 12, 1024, 1024),
        # (2, 48, 24, 2048, 2048),
        # (2, 48, 8, 4096, 4096),
        # (2, 48, 4, 8192, 8192),
        # (2, 48, 2, 16384, 16384),
        # (2, 64, 32, 1024, 1024),
        # (4, 64, 16, 2048, 2048),
        # (4, 64, 8, 4096, 4096),
        # (4, 64, 32, 8192, 8192),
        (4, 128, 16, 16384, 16384),
    ]
    return configs


def model_benchmark_configs(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    fa_configs = []
    batch_size = args.b if args.b else 1

    for model_name, config in configs.items():
        HQ = config["num_attention_heads"]
        HK = (
            HQ
            if config["num_key_value_heads"] is None
            else config["num_key_value_heads"]
        )
        N_CTX_Q = args.sq if args.sq else 8192
        N_CTX_K = args.sk if args.sk else N_CTX_Q
        HEAD_DIM = config["hidden_size"] // HQ
        fa_configs.append((model_name, batch_size, HQ, HK, N_CTX_Q, N_CTX_K, HEAD_DIM))

    return fa_configs


def pad_rearrange_dropout_mask(
    S_dmask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqlen_q,
    seqlen_k,
    num_q_heads,
):
    batch_size = cu_seqlens_q.numel() - 1

    padded_dropout_mask = torch.ones(
        (batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda"
    )
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
            padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[
                b, h, :, :
            ]

    return padded_dropout_mask


def create_benchmark_configs(custom, args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    mode = args.mode
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = args.causal
    varlen = args.layout == "thd"

    configs = []
    plot_name = f"fused-attention-{mode}-D_HEAD-{head_size}-layout-{args.layout}-fp8-{args.fp8}-causal-{causal}"
    extra_args = {"D_HEAD": head_size, "dtype": dtype, "causal": causal, "mode": mode}

    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()  # Assume this exists
        else:
            x_vals_list = nonvarlen_benchmark_configs()  # Assume this exists

        if args.model:
            x_vals_list = model_benchmark_configs(args)
            x_names = ["model", "BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD"]
            plot_name = f"fused-attention-{mode}-layout-{args.layout}-fp8-{args.fp8}-causal-{causal}"
            extra_args = {"dtype": dtype, "causal": causal, "mode": mode}

    unit = "TFLOPS"
    if args.return_time:
        unit = "ms"
    if args.return_bandwidth:
        unit = "GB/s"
    # unit = "ms"

    if mode == "bwd":
        if getattr(args, "compare_fa2_lean", False):
            # Compare FA2 one-kernel backward vs Lean persistent backward
            line_vals = [f"fa2-onekernel({unit})", f"lean-bwd({unit})"]
        elif args.fused_bwd:
            line_vals = [f"fused-bwd({unit})"]
        else:
            line_vals = [f"fused-bwd({unit})", f"bwd({unit})"]
    else:
        line_vals = [f"fwd({unit})"]

    if args.bench_torch:
        line_vals = [f"Triton({unit})", f"Torch({unit})"]

    if args.test_mode:
        if args.fused_bwd:
            line_vals = [f"fused-bwd({unit})"]
        else:
            line_vals = [f"bwd({unit})"]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-")],
            ylabel=unit,
            plot_name=plot_name,
            args=extra_args,
        )
    )
    return configs


def run_benchmark(custom, args):
    torch.manual_seed(20)
    # Capture original one-kernel bwd to allow safe switching per-call
    from aiter.ops.triton import mha as _mha
    try:
        _orig_onekernel_bwd = _mha.flash_attn_onekernel_backward
    except Exception:
        _orig_onekernel_bwd = None

    @triton.testing.perf_report(create_benchmark_configs(custom, args))
    def bench_mha(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        dtype,
        causal,
        mode,
        provider,
        dropout=0.0,
        model=None,
        sm_scale=None,
        device="cuda",
    ):
        """
        Benchmark or test function for multi-head attention backward pass.
        In test_mode, verifies output matching with non-varlen inputs.
        """
        assert dropout <= 0.0, "Dropout not supported in this benchmark."
        requires_grad = mode == "bwd" or args.test_mode
        return_lse = True
        return_attn_probs = False
        varlen = args.layout == "thd"

        global _USE_FUSED_BWD

        fused_backward = "fused-bwd" in provider
        if getattr(args, "compare_fa2_lean", False):
            # Override provider mapping by label:
            # - "fa2-onekernel" => use FA2 one-kernel backward (no lean)
            # - "lean-bwd" => monkey-patch bwd to Lean
            provider_is_fa2 = provider.startswith("fa2-onekernel")
            provider_is_lean = provider.startswith("lean-bwd")
            # Ensure we don't toggle fused kernel; we want the standard one-kernel bwd path
            fused_backward = False

        # Conditionally switch to Lean Attention persistent bwd (only non-varlen)
        if (args.lean or (getattr(args, "compare_fa2_lean", False) and provider_is_lean)) and (mode == "bwd") and (not varlen):
            try:
                from aiter.ops.triton.lean_atten_bwd_clean import (
                    persistent_lean_attention_bwd as _lean_bwd,
                )

                def _lean_flash_attn_onekernel_backward(
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
                    alibi_slopes: torch.Tensor,
                    causal: bool,
                    cu_seqlens_q: torch.Tensor,
                    cu_seqlens_k: torch.Tensor,
                    max_seqlen_q: int,
                    max_seqlen_k: int,
                    dropout_p: float,
                    philox_seed: int = 0,
                    philox_offset: int = 0,
                    descale_q: torch.Tensor = None,
                    descale_k: torch.Tensor = None,
                    descale_v: torch.Tensor = None,
                    descale_do: torch.Tensor = None,
                    USE_INT64_STRIDES: bool = False,
                    config: dict = None,
                ):
                    # Only non-varlen supported in this lean path
                    if cu_seqlens_q is not None or cu_seqlens_k is not None:
                        raise NotImplementedError(
                            "lean_atten_bwd_clean path only supports non-varlen"
                        )
                    # Dropout not supported in lean path
                    if dropout_p and dropout_p > 0.0:
                        raise NotImplementedError(
                            "lean_atten_bwd_clean path expects dropout_p == 0.0"
                        )

                    # q/do/o: [B, Nq, H, D], k/v: [B, Nk, H, D]
                    B, Nq, H, D = q.shape
                    Nk = k.shape[1]

                    # Build lean views
                    q_lean = q.reshape(B * Nq, H, D)
                    do_lean = do.reshape(B * Nq, H, D)
                    o_lean = o.reshape(B * Nq, H, D)
                    k_lean = k.reshape(B * Nk, H, D)
                    v_lean = v.reshape(B * Nk, H, D)

                    dq_view = dq.reshape(B * Nq, H, D)
                    dk_view = dk.reshape(B * Nk, H, D)
                    dv_view = dv.reshape(B * Nk, H, D)

                    _lean_bwd(
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
                        seqlen_k=N_CTX_K,
                    )
                    print("Sum of DQ: ", dq_view.sum())
                    print("Sum of DK: ", dk_view.sum())
                    print("Sum of DV: ", dv_view.sum())

                    # Return delta to match API (same as softmax_lse shape: [B, H, Nq])
                    delta = (do * o).sum(dim=-1)
                    delta = delta.permute(0, 2, 1).contiguous()
                    return delta

                if _orig_onekernel_bwd is not None:
                    _mha.flash_attn_onekernel_backward = _lean_flash_attn_onekernel_backward
            except Exception as e:
                raise RuntimeError(f"Failed to enable Lean backward (-l/--lean): {e}")
        else:
            # Restore original one-kernel bwd for varlen or when lean not selected
            if _orig_onekernel_bwd is not None:
                _mha.flash_attn_onekernel_backward = _orig_onekernel_bwd

        mha_set_use_fused_bwd_kernel(fused_backward)

        # Default softmax scale to match standard attention
        if sm_scale is None:
            sm_scale = 1.0 / (D_HEAD**0.5)

        # Test mode: run tests from op_tests with specified shapes
        if args.test_mode:
            import op_tests.triton_tests.test_mha as test_mha

            print(
                f"Testing kernel implementation <{provider}> against Torch with shape:"
            )
            print(
                f"BATCH={BATCH}, HQ={HQ}, HK={HK}, N_CTX_Q={N_CTX_Q}, N_CTX_K={N_CTX_K}, D_HEAD={D_HEAD}"
            )
            if varlen:
                test_mha.test_mha(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    True,
                    True,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Forward test passed!")
                test_mha.test_mha_backward_varlen(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Backward test passed!")
            else:
                test_mha.test_mha_varlen(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    True,
                    True,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Forward test passed!")
                test_mha.test_mha_backward(
                    BATCH,
                    N_CTX_Q,
                    N_CTX_K,
                    HQ,
                    HK,
                    D_HEAD,
                    dropout,
                    causal,
                    args.fp8,
                    dtype,
                )
                print("Backward test passed!")

            return 0

        # Generate base inputs
        q = torch.randn((BATCH, N_CTX_Q, HQ, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, N_CTX_K, HK, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, N_CTX_K, HK, D_HEAD), device=device, dtype=dtype)
        q.requires_grad = requires_grad
        k.requires_grad = requires_grad
        v.requires_grad = requires_grad

        # FLOPS calculation variables
        flops_per_matmul = 0

        # Input preparation
        if varlen:
            query_padding_mask = generate_random_padding_mask(
                N_CTX_Q, BATCH, device, mode="full" if args.equal_seqlens else "random"
            )
            key_padding_mask = generate_random_padding_mask(
                N_CTX_K, BATCH, device, mode="full" if args.equal_seqlens else "random"
            )
            (
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                q,
                k,
                v,
                _,
                _,
                _,
            ) = generate_qkv(
                q, k, v, query_padding_mask, key_padding_mask, kvpacked=False
            )
            q_unpad.requires_grad = True
            k_unpad.requires_grad = True
            v_unpad.requires_grad = True

            q_input, k_input, v_input = q_unpad, k_unpad, v_unpad

            num_contexts = len(cu_seqlens_q) - 1
            for i in range(num_contexts):
                seqlen_q = (cu_seqlens_q[i + 1] - cu_seqlens_q[i]).item()
                seqlen_k = (cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item()
                if causal:
                    valid_out_elements = (
                        ((seqlen_k**2 + seqlen_k) / 2)
                        if seqlen_q > seqlen_k
                        else (seqlen_q * seqlen_k - ((seqlen_q**2 - seqlen_q) / 2))
                    )
                    flops_per_matmul += valid_out_elements * HQ * D_HEAD * 2
                else:
                    flops_per_matmul += seqlen_q * seqlen_k * HQ * D_HEAD * 2
        else:
            q_input, k_input, v_input = q, k, v

            if causal:
                valid_out_elements = (
                    ((N_CTX_K**2 + N_CTX_K) / 2)
                    if N_CTX_Q > N_CTX_K
                    else (N_CTX_Q * N_CTX_K - ((N_CTX_Q**2 - N_CTX_Q) / 2))
                )
                flops_per_matmul = 2.0 * BATCH * HQ * valid_out_elements * D_HEAD
            else:
                flops_per_matmul = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * D_HEAD

        # Benchmark mode
        MAPPING_AITER = 0
        MAPPING_HEAD_FIRST_K7 = 1
        MAPPING_HEAD_FIRST_M = 2
        MAPPING_TRITON_FA = 3
        
        if varlen:
            if args.fp8:

                def fn():
                    return flash_attn_varlen_fp8_func(
                        q_input,
                        k_input,
                        v_input,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                    )

            else:

                def fn():
                    return flash_attn_varlen_func(
                        q_input,
                        k_input,
                        v_input,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                    )

        else:
            if args.fp8:

                def fn():
                    return flash_attn_fp8_func(
                        q_input,
                        k_input,
                        v_input,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                    )

            else:

                def fn():
                    return flash_attn_func(
                        q_input,
                        k_input,
                        v_input,
                        dropout_p=dropout,
                        softmax_scale=sm_scale,
                        causal=causal,
                        return_lse=return_lse,
                        return_attn_probs=return_attn_probs,
                        mapping_mode=args.mapping_mode, 
                        use_remap=args.use_remap,
                    )

        if mode == "bwd":
            with torch.enable_grad():
                triton_out = fn()[0]
                d_out = torch.randn_like(triton_out)

                def fn():
                    grads = torch.autograd.grad(
                        triton_out,
                        (q_input, k_input, v_input),
                        d_out,
                        retain_graph=True,
                    )
                    print("Sum of DQ grads: ", grads[0].sum())
                    print("Sum of DK grads: ", grads[1].sum())
                    print("Sum of DV grads: ", grads[2].sum())
                    return grads

        ms = triton.testing.do_bench(fn)

        total_flops = 2 * flops_per_matmul
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

        input_bytes = q.element_size()
        output_bytes = q.element_size()
        if varlen:
            total_num_tokens_q = cu_seqlens_q[-1].item()
            total_num_tokens_k = cu_seqlens_k[-1].item()
        else:
            total_num_tokens_q = BATCH * N_CTX_Q
            total_num_tokens_k = BATCH * N_CTX_K
        mem = (
            total_num_tokens_q * HQ * D_HEAD * input_bytes
            + 2 * total_num_tokens_k * HK * D_HEAD * input_bytes
            + total_num_tokens_q * HQ * D_HEAD * output_bytes
        )
        # return ms
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / ms * 1e-9
        else:  # GB/s
            return mem / ms * 1e-3

    bench_mha.run(save_path=None, print_data=True, show_plots=False)


def supported_layouts():
    layouts = (
        "bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]. "
        "thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]. "
    )
    return layouts


# argparse lacks support for boolean argument type (sigh...)
def str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():

    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-model_configs",
        type=str,
        default="utils/model_configs.json",
        help="Model config json file.",
    )
    available_models = get_available_models()  # Dynamically load model names
    model_help = (
        "Model name to benchmark. Select from: ["
        + ", ".join(available_models)
        + "]. Use 'all' to benchmark all models. Provide model family (the part before -) to benchmark all models in that family. One can provide multiple as -model \"llama3,mistral_7B\""
    )
    parser.add_argument("-model", type=str, default="", help=model_help)
    parser.add_argument(
        "-mode", type=str, default="fwd", help="fwd:forward kernel, bwd:backward kernel"
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="If specified, uses equal sequence lengths with thd layout, i.e t = b * sq",
    )
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-causal", type=str2bool, default=None)
    parser.add_argument("-fp8", action="store_true", default=False)
    parser.add_argument("-quantize_p", action="store_true", default=False)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument("-bench_torch", action="store_true", default=False)
    parser.add_argument("-fused_bwd", action="store_true", default=False)
    parser.add_argument(
        "--compare_fa2_lean",
        action="store_true",
        default=False,
        help="Compare FA2 fused backward vs Lean persistent backward on the same plot",
    )
    parser.add_argument(
        "-l",
        "--lean",
        action="store_true",
        default=False,
        help="Use Lean Attention persistent backward (non-varlen, no-dropout)."
    )
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument(
        "-return_all",
        action="store_true",
        default=False,
        help="Prints TFLOPS, walltime, bandwidth.",
    )
    parser.add_argument(
        "-test_mode",
        action="store_true",
        default=False,
        help="Tests correctness of the Triton provider comparing the output to the Torch sdpa.",
    )
    # prints TFLOPS without setting the following
    parser.add_argument(
        "-return_time", action="store_true", default=False, help="Prints only walltime."
    )
    parser.add_argument(
        "-return_bandwidth",
        action="store_true",
        default=False,
        help="Prints only memory bandwidth.",
    )

    parser.add_argument("-layout", type=str, default=None, help=supported_layouts())

    parser.add_argument(
        "-persistent",
        nargs="?",
        const="fixed",
        choices=["fixed", "dynamic"],
        default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.",
    )
    parser.add_argument(
        "-mapping_mode", 
        type=int, 
        default=0, 
        choices=[0, 1, 2, 3],
        help="Mapping mode: 0=aiter_fa, 1=head_first_k7, 2=head_first_m, 3=triton_fa"
    )
    parser.add_argument(
        "-no_remap",
        action="store_false",
        dest="use_remap",
        help="Disable remap functionality (only applies to aiter mode)"
    )

    return parser.parse_args()


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main():
    args = parse_args()

    if args.model:
        if args.causal is None:  # User didn't specify -causal
            args.causal = True
        if args.layout is None:  # User didn't specify -layout
            args.layout = "thd"
        print(
            f"Note: using -model config defaults: causal={True}, layout={'thd'}. This is the most common real life scenario, but can be overridden with -causal and -layout flags."
        )
    else:
        # the defaults for causal and varlen when not using the -model
        if args.causal is None:  # User didn't specify -causal
            args.causal = False
        if args.layout is None:  # User didn't specify -layout
            args.layout = "bshd"

    custom_config = False

    assert (
        args.layout == "thd" or not args.equal_seqlens or args.model
    ), "Equal sequence lengths arg must be used with the thd layout or a model config."
    if args.hq or args.hk or args.d:
        custom_config = True
        assert (
            args.b and args.hq and args.sq and args.d
        ), "If custom config is specified, please provide \
                all of batch, number of Q heads, Q sequence length \
                and head size."

    if args.model:
        assert not (
            args.hq or args.hk or args.d
        ), "Specifying model fixes hq, hk and d already. Do not provide them!"

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    assert (
        args.layout in supported_layouts()
    ), f"{args.layout} is not in supported layouts: {supported_layouts()}."

    if args.layout == "thd" and args.equal_seqlens:
        warnings.warn(
            "Using 'thd' layout with equal_seqlen=True incurs an extra sequence length lookup cost "
            "compared to 'bshd' layout. Consider using 'bshd' for better performance.",
            category=RuntimeWarning,
        )

    if args.print_vgpr:
        assert not args.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(custom_config, args)

        print_vgpr(fun, "fused-attention")
        return 0

    run_benchmark(custom_config, args)


if __name__ == "__main__":
    import sys

    sys.exit(main())