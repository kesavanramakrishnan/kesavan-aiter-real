#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor gemm_a4w4_asm(torch::Tensor& A,       // A:[M, K/2] f4x2
                            torch::Tensor& B,       // B:[N, K/2] f4x2
                            torch::Tensor& A_scale, // A_scale:[M, K/32] e8m0 paded
                            torch::Tensor& B_scale, // B_scale:[N, K/32] e8m0 paded
                            torch::Tensor& out,     // Out:[M, N] bf16
                            std::string& kernelName,
                            std::optional<torch::Tensor>& bias, // bias:[M, N] f32
                            std::optional<float> alpha      = 1.0,
                            std::optional<float> beta       = 0.0,
                            std::optional<bool> bpreshuffle = true,
                            std::optional<int> log2_k_split = std::nullopt);
