# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter.test_common import perftest
from aiter import dtypes
from batched_gemm_bf16_common import kernels_list
from aiter.utility.mp_tuner import mp_tuner
import argparse


def checkClose(a, b, rtol=1e-3, atol=0.01):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        return True
    else:
        percent = (a[mask]).numel() / a.numel()
        if percent > 0.01:
            return False
        else:
            return True


def run_torch(x, weight, bias=None, dtype=dtypes.bf16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=dtypes.bf16, device="cuda")
    for b in range(B):
        b_out = F.linear(x[b, :, :].to(dtypes.fp32), weight[b, :, :].to(dtypes.fp32))
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


def get_untuned_batched_gemm_list(untuned_batched_gemm_file):
    assert os.path.exists(
        untuned_batched_gemm_file
    ), f"Not exist bf16_untuned_batched_gemm.csv file: {untuned_batched_gemm_file}"
    untunedf = pd.read_csv(untuned_batched_gemm_file)
    return untunedf


def get_tuned_batched_gemm_list(tuned_batched_gemm_file):
    if os.path.exists(tuned_batched_gemm_file):
        tunedf = pd.read_csv(tuned_batched_gemm_file)
    else:
        tunedf = pd.DataFrame(
            columns=["B", "M", "N", "K", "kernelId", "splitK", "us", "kernelName"]
        )
    return tunedf


def run_batched_gemm(x, weight, out, kernel_id, splitK=0):
    aiter.batched_gemm_bf16_tune(x, weight, out, kernel_id, splitK)
    return out


def generate_data(b, m, n, k, device="cuda"):
    x = torch.randint(-20, 20, (b, m, k), dtype=dtypes.bf16, device=device)
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtypes.bf16, device=device)
    out = torch.empty(b, m, n, dtype=dtypes.bf16, device=device)
    return x, weight, out


def tune_batched_gemm_list(untunedf, tunedf, issorted=False, useSplitK=False, mp_num=1):
    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    task = []
    tasks_data = []
    for i in range(len(untunedf)):
        B = untunedf.loc[i, "B"]
        M = untunedf.loc[i, "M"]
        N = untunedf.loc[i, "N"]
        K = untunedf.loc[i, "K"]
        kernels_num = len(kernels_list)
        if tunedf[
            (tunedf["B"] == B)
            & (tunedf["M"] == M)
            & (tunedf["N"] == N)
            & (tunedf["K"] == K)
            & (tunedf["cu_num"] == cu_num)
        ].empty:
            print(f"tuning B:{B}, M:{M}, N:{N}, K:{K}")
            # kernelId, splitK, time = tune_batched_gemm(B, M, N, K, useSplitK)
            total_kernel_nums = 0
            for i in range(kernels_num):
                kernel = kernels_list[i]
                maxsplitK = (
                    aiter.compute_batched_gemm_SplitK(
                        B, M, N, K, kernel.MPerBLOCK, kernel.NPerBLOCK, kernel.KPerBLOCK
                    )
                    if useSplitK
                    else 0
                )
                for splitK in range(maxsplitK + 1):
                    info = ((cu_num, B, M, N, K), i, splitK)
                    task.append(
                        (
                            info,
                            generate_data,
                            (B, M, N, K),
                            run_batched_gemm,
                            ([0, 1, 2], i, splitK),
                            {},
                            run_torch,
                            ([0, 1],),
                            {},
                            None,
                            1e-2,
                            1e-2,
                        )
                    )
                    total_kernel_nums = total_kernel_nums + 1

            tasks_data.append((total_kernel_nums, ()))
        else:
            print(f"B:{B}, M:{M}, N:{N}, K{K} is in tuned batched_gemm, skip!!!")
            print()
            print()
    if task:
        shape_grouped = False
        ret = mp_tuner(task, tasks_data, mp_num, False, shape_grouped)
        for el in ret:
            info, time, err_ratio = el
            (cu_num, B, M, N, K), kernelId, splitK = info
            kernelName = (
                "None"
                if kernelId == -1 or time == "nan"
                else kernels_list[kernelId].name
            )
            temp = pd.DataFrame(
                {
                    "B": [B],
                    "M": [M],
                    "N": [N],
                    "K": [K],
                    "cu_num": [cu_num],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                }
            )
            tunedf = pd.concat([tunedf, temp], ignore_index=True)

    if issorted:
        tunedf = tunedf.sort_values(by=["B", "M", "N", "K"])
    print("Totall tuning result:")
    print(tunedf)
    return tunedf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK batched_gemm bf16 kernel",
    )

    parser.add_argument(
        "-i",
        "--untune_file",
        default="aiter/configs/bf16_untuned_batched_gemm.csv",
        required=False,
        help="input",
    )

    parser.add_argument(
        "-o",
        "--tune_file",
        default="aiter/configs/bf16_tuned_batched_gemm.csv",
        required=False,
        help="output: tuning result store this file",
    )

    parser.add_argument(
        "-k", "--splitK", action="store_true", required=False, help="Use splitK kernels"
    )

    parser.add_argument(
        "--sort",
        action="store_true",
        required=False,
        help="Arranged according to the B M N K size",
    )

    parser.add_argument(
        "--mp",
        type=int,
        default=torch.cuda.device_count(),
        help="Tuning on multiple GPUs using multiple processes",
    )
    args = parser.parse_args()
    untunedf = get_untuned_batched_gemm_list(args.untune_file)
    tunedf = get_tuned_batched_gemm_list(args.tune_file)
    tunedf = tune_batched_gemm_list(untunedf, tunedf, args.sort, args.splitK, args.mp)
    tunedf.to_csv(args.tune_file, index=False)
