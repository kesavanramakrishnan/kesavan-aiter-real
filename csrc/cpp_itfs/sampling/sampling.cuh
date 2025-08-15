/*
 * Copyright (C) 2024-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "hip/hip_runtime.h"

#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#include <hipcub/block/block_adjacent_difference.hpp>
#include <hipcub/block/block_reduce.hpp>
#include <hipcub/block/block_scan.hpp>
#include <limits>
#include <numeric>
#include <tuple>

#include "vec_dtypes.cuh"

namespace aiter {

namespace sampling {

using namespace hipcub;

constexpr uint32_t BLOCK_THREADS = 1024;

constexpr BlockScanAlgorithm SCAN_ALGO     = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

template <typename T>
struct ValueCount
{
    T value;
    int count;

    __device__ ValueCount operator+(const ValueCount& other) const
    {
        return {value + other.value, count + other.count};
    }
    __device__ ValueCount& operator+=(const ValueCount& other)
    {
        value += other.value;
        count += other.count;
        return *this;
    }
};

struct BoolDiffOp
{
    __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const
    {
        return lhs != rhs;
    }
};

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y)
{
    return (x + y - 1) / y;
}

template <uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage
{
    union
    {
        float deterministic_scan[BLOCK_THREADS / 32];
        typename BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
        typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
        typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
        typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
            reduce_value_count;
        typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
    } block_prim;
    struct
    {
        int32_t sampled_id;
        int32_t last_valid_id;
        float max_val;
        union
        {
            float value;
            ValueCount<float> pair;
        } block_aggregate;
    };
};

template <typename T>
__device__ __forceinline__ T infinity()
{
    return __builtin_huge_valf();
}

/*!
 * \brief Deterministic inclusive scan implementation, use Belloch scan algorithm.
 * \note This implementation is slower than the hipcub::BlockScan, but it is deterministic.
 */
template <uint32_t VEC_SIZE,
          uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ void DeterministicInclusiveSum(
    const float* in_data,
    float* out_data,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage)
{
    float* smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
    float thread_data[VEC_SIZE];
    float thread_sum = 0;
#pragma unroll
    for(uint32_t i = 0; i < VEC_SIZE; ++i)
    {
        thread_sum += in_data[i];
        thread_data[i] = thread_sum;
    }

    float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
    for(uint32_t offset = 1; offset < 32; offset *= 2)
    {
        float tmp = __shfl_up(thread_exclusive_prefix_sum, offset);
        if((threadIdx.x + 1) % (offset * 2) == 0)
        {
            thread_exclusive_prefix_sum += tmp;
        }
    }

    float warp_sum = __shfl(thread_exclusive_prefix_sum, threadIdx.x | 0xffffffff);
    if(threadIdx.x % 32 == 31)
    {
        thread_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for(uint32_t offset = 16; offset >= 1; offset /= 2)
    {
        float tmp = __shfl_xor(thread_exclusive_prefix_sum, offset);
        if((threadIdx.x + 1) % (offset * 2) == 0)
        {
            thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
        }
        if((threadIdx.x + 1) % (offset * 2) == offset)
        {
            thread_exclusive_prefix_sum = tmp;
        }
    }

    smem_prefix_sum[threadIdx.x / 32] = warp_sum;
    __syncthreads();

    if(threadIdx.x < 32)
    {
        float warp_exclusive_prefix_sum =
            (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
        for(uint32_t offset = 1; offset < 32; offset *= 2)
        {
            float tmp = __shfl_up(warp_exclusive_prefix_sum, offset);
            if((threadIdx.x + 1) % (offset * 2) == 0)
            {
                warp_exclusive_prefix_sum += tmp;
            }
        }

        if(threadIdx.x % 32 == 31)
        {
            warp_exclusive_prefix_sum = 0;
        }

#pragma unroll
        for(uint32_t offset = 16; offset >= 1; offset /= 2)
        {
            float tmp = __shfl_xor(warp_exclusive_prefix_sum, offset);
            if((threadIdx.x + 1) % (offset * 2) == 0)
            {
                warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
            }
            if((threadIdx.x + 1) % (offset * 2) == offset)
            {
                warp_exclusive_prefix_sum = tmp;
            }
        }
        if(threadIdx.x < BLOCK_THREADS / 32)
        {
            smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
        }
    }
    __syncthreads();

#pragma unroll
    for(uint32_t i = 0; i < VEC_SIZE; ++i)
    {
        out_data[i] =
            smem_prefix_sum[threadIdx.x / 32] + thread_exclusive_prefix_sum + thread_data[i];
    }
}

template <uint32_t VEC_SIZE,
          uint32_t BLOCK_THREADS,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          typename TempStorage>
__device__ __forceinline__ float
GetMaxValue(float* in_data, uint32_t row_idx, uint32_t d, TempStorage& temp_storage)
{
    const uint32_t tx = threadIdx.x;
    vec_t<float, VEC_SIZE> in_data_vec;

    float max_val = 0;
    for(uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
    {
        in_data_vec.fill(0);
        if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
        {
            in_data_vec.cast_load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
        }
        float in_data_[VEC_SIZE];
#pragma unroll
        for(uint32_t j = 0; j < VEC_SIZE; ++j)
        {
            in_data_[j] = in_data_vec[j];
        }
        max_val =
            max(max_val,
                BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                    .Reduce(in_data_, hipcub::Max()));
        __syncthreads();
    }
    if(tx == 0)
    {
        temp_storage.max_val = max_val;
    }
    __syncthreads();
    return temp_storage.max_val;
}

template <uint32_t VEC_SIZE,
          uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          bool DETERMINISTIC,
          typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i,
    uint32_t d,
    Predicate pred,
    float u,
    vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage)
{
    const uint32_t tx = threadIdx.x;
    float prob_greater_than_threshold[VEC_SIZE];
    float inclusive_cdf[VEC_SIZE];
    bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
    for(uint32_t j = 0; j < VEC_SIZE; ++j)
    {
        prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
        valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
    }
    float aggregate_local =
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
            .Sum(prob_greater_than_threshold);
    if(tx == 0)
    {
        temp_storage->block_aggregate.value = aggregate_local;
    }
    __syncthreads();
    aggregate_local = temp_storage->block_aggregate.value;

    if(aggregate + aggregate_local > u)
    {
        if constexpr(DETERMINISTIC)
        {
            DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
                prob_greater_than_threshold, inclusive_cdf, temp_storage);
        }
        else
        {
            BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
                .InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

            __syncthreads();
        }

#pragma unroll
        for(uint32_t j = 0; j < VEC_SIZE; ++j)
        {
            greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
        }

        bool greater_than_u_diff[VEC_SIZE];

        BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
            .SubtractLeft(greater_than_u, greater_than_u_diff, BoolDiffOp{});

        __syncthreads();

#pragma unroll
        for(uint32_t j = 0; j < VEC_SIZE; ++j)
        {
            if(greater_than_u_diff[j])
            {
                atomicMin(&(temp_storage->sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
            }
        }
        __syncthreads();
    }

    // update the last valid index
    int valid_index[VEC_SIZE];
#pragma unroll
    for(uint32_t j = 0; j < VEC_SIZE; ++j)
    {
        if(valid[j])
        {
            valid_index[j] = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
        }
        else
        {
            valid_index[j] = -1;
        }
    }
    int max_valid_index =
        BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
            .Reduce(valid_index, hipcub::Max());
    if(tx == 0 && max_valid_index != -1)
    {
        temp_storage->last_valid_id = max_valid_index;
    }
    __syncthreads();
    aggregate += aggregate_local;
}

template <uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          uint32_t VEC_SIZE,
          bool DETERMINISTIC,
          typename DType,
          typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs,
                                           IdType* output,
                                           IdType* indices,
                                           float* top_p_arr,
                                           float top_p_val,
                                           uint32_t d,
                                           uint64_t philox_seed,
                                           uint64_t philox_offset)
{
    const uint32_t batch_size = gridDim.x;
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    hiprandStatePhilox4_32_10_t state;
    hiprand_init(philox_seed, bx, philox_offset, &state);
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
    float top_p            = (top_p_arr == nullptr) ? top_p_val : top_p_arr[row_idx];

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto& temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
            smem_sampling);

    vec_t<float, VEC_SIZE> probs_vec;
    float aggregate;
    float q    = 1;
    double low = 0, high = 1.f;
    int sampled_id;
    do
    {
        temp_storage.sampled_id = d;
        __syncthreads();
        float u   = hiprand_uniform(&state) * q;
        aggregate = 0;
#pragma unroll 2
        for(uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
        {
            probs_vec.fill(0);
            if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
            {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            DeviceSamplingFromProb<VEC_SIZE,
                                   BLOCK_THREADS,
                                   SCAN_ALGORITHM,
                                   REDUCE_ALGORITHM,
                                   DETERMINISTIC>(
                i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
            if(aggregate > u)
            {
                break;
            }
        }
        __syncthreads();
        sampled_id = temp_storage.sampled_id;
        if(sampled_id == d)
        {
            // NOTE(Zihao): this would happen when u is very close to 1
            // and the sum of probabilities is smaller than u
            // In this case, we use the last valid index as the sampled id
            sampled_id = temp_storage.last_valid_id;
        }
        double pivot_0 = probs[row_idx * d + sampled_id];
        double pivot_1 = (pivot_0 + high) / 2;

        float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
#pragma unroll 2
        for(uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
        {
            probs_vec.fill(0);
            if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
            {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
            for(uint32_t j = 0; j < VEC_SIZE; ++j)
            {
                probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
                probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;
            }

            aggregate_gt_pivot_0 +=
                BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                    .Sum(probs_gt_pivot_0);
            if(tx == 0)
            {
                temp_storage.block_aggregate.value = aggregate_gt_pivot_0;
            }
            __syncthreads();
            aggregate_gt_pivot_0 = temp_storage.block_aggregate.value;

            aggregate_gt_pivot_1 +=
                BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                    .Sum(probs_gt_pivot_1);
            if(tx == 0)
            {
                temp_storage.block_aggregate.value = aggregate_gt_pivot_1;
            }
            __syncthreads();
            aggregate_gt_pivot_1 = temp_storage.block_aggregate.value;
        }
        if(aggregate_gt_pivot_0 < top_p)
        {
            // case 1: pivot_0 accepted
            break;
        }
        if(aggregate_gt_pivot_1 < top_p)
        {
            // case 2: pivot_0 rejected, pivot_1 accepted
            low  = pivot_0;
            high = pivot_1;
            q    = aggregate_gt_pivot_0;
        }
        else
        {
            // case 3: pivot_0 rejected, pivot_1 rejected
            low = pivot_1;
            q   = aggregate_gt_pivot_1;
        }
    } while(low < high);
    __syncthreads();
    if(tx == 0)
    {
        output[bx] = sampled_id;
    }
}

template <uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          uint32_t VEC_SIZE,
          bool DETERMINISTIC,
          typename DType,
          typename IdType>
__global__ void TopKTopPSamplingFromProbKernel(DType* probs,
                                               IdType* top_k_arr,
                                               float* top_p_arr,
                                               IdType* output,
                                               IdType* indices,
                                               IdType top_k_val,
                                               float top_p_val,
                                               uint32_t d,
                                               uint64_t philox_seed,
                                               uint64_t philox_offset)
{
    const uint32_t batch_size = gridDim.x;
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    hiprandStatePhilox4_32_10_t state;
    hiprand_init(philox_seed, bx, philox_offset, &state);
    const uint32_t row_idx = indices == nullptr ? bx : indices[bx];
    const uint32_t k       = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];
    const float p          = top_p_arr == nullptr ? top_p_val : top_p_arr[row_idx];

    extern __shared__ __align__(
        alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto& temp_storage =
        reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
            smem_sampling);

    vec_t<float, VEC_SIZE> probs_vec;
    float aggregate;
    float q    = 1;
    double low = 0, high = 1.f;
    int sampled_id;
    do
    {
        temp_storage.sampled_id = d;
        __syncthreads();
        float u   = hiprand_uniform(&state) * q;
        aggregate = 0;
#pragma unroll 2
        for(uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
        {
            probs_vec.fill(0);
            if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
            {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            DeviceSamplingFromProb<VEC_SIZE,
                                   BLOCK_THREADS,
                                   SCAN_ALGORITHM,
                                   REDUCE_ALGORITHM,
                                   DETERMINISTIC>(
                i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
            if(aggregate > u)
            {
                break;
            }
        }
        __syncthreads();
        sampled_id = temp_storage.sampled_id;
        if(sampled_id == d)
        {
            // NOTE(Zihao): this would happen when u is very close to 1
            // and the sum of probabilities is smaller than u
            // In this case, we use the last valid index as the sampled id
            sampled_id = temp_storage.last_valid_id;
        }
        double pivot_0 = probs[row_idx * d + sampled_id];
        double pivot_1 = (pivot_0 + high) / 2;

        ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
#pragma unroll 2
        for(uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
        {
            probs_vec.fill(0);
            if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
            {
                probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
            }

            ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
            for(uint32_t j = 0; j < VEC_SIZE; ++j)
            {
                probs_gt_pivot_0[j] = {
                    (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
                    (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
                probs_gt_pivot_1[j] = {
                    (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
                    (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
            }

            aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS>(
                                        temp_storage.block_prim.reduce_value_count)
                                        .Sum(probs_gt_pivot_0);
            if(tx == 0)
            {
                temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
            }
            __syncthreads();
            aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

            aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS>(
                                        temp_storage.block_prim.reduce_value_count)
                                        .Sum(probs_gt_pivot_1);
            if(tx == 0)
            {
                temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
            }
            __syncthreads();
            aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
        }
        if(aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p)
        {
            // case 1: pivot_0 accepted
            break;
        }
        if(aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p)
        {
            // case 2: pivot_0 rejected, pivot_1 accepted
            low  = pivot_0;
            high = pivot_1;
            q    = aggregate_gt_pivot_0.value;
        }
        else
        {
            // case 3: pivot_0 rejected, pivot_1 rejected
            low = pivot_1;
            q   = aggregate_gt_pivot_1.value;
        }
    } while(low < high);
    __syncthreads();
    if(tx == 0)
    {
        output[bx] = sampled_id;
    }
}

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct RenormTempStorage
{
    union
    {
        typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
        typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
        typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
            reduce_value_count;
    } block_prim;
    struct
    {
        float max_val;
        float min_val;
        union
        {
            struct
            {
                float values[2];
            };
            struct
            {
                int counts[2];
            };
            struct
            {
                ValueCount<float> pairs[2];
            };
        } block_aggregate;
    };
};

template <uint32_t BLOCK_THREADS,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          uint32_t VEC_SIZE,
          typename DType,
          typename IdType>
__global__ void TopKRenormProbKernel(
    DType* probs, DType* renormed_prob, IdType* top_k_arr, uint32_t top_k_val, uint32_t d)
{
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    const uint32_t row_idx = bx;
    uint32_t k             = top_k_arr == nullptr ? top_k_val : top_k_arr[bx];
    double pivot = -infinity<float>(), normalizer = 1;
    vec_t<float, VEC_SIZE> probs_vec;
    if(k < d)
    {
        extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>))
            uint8_t smem_renorm[];
        auto& temp_storage =
            reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
        temp_storage.max_val = 0;

        float max_val = GetMaxValue<VEC_SIZE,
                                    BLOCK_THREADS,
                                    REDUCE_ALGORITHM,
                                    RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(
            probs, row_idx, d, temp_storage);

        double low = 0, high = max_val;
        float min_gt_low, max_le_high;
        float sum_low = 1;
        // f(x) = len(nonzero(probs > x)), f(x) is non-increasing
        // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
        // loop invariant:
        // - f(low) >= k, f(high) < k
        // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
        // stopping condition: min_gt_low == max_le_high
        // - f(low) >= k, f(min_gt_low) == f(max_le_high) == f(high) < k
        do
        {
            double pivot_0 = (high + 2 * low) / 3;
            double pivot_1 = (2 * high + low) / 3;

            ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
            min_gt_low  = high;
            max_le_high = low;
#pragma unroll 2
            for(uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
            {
                probs_vec.fill(0);
                if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
                {
                    probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE +
                                        tx * VEC_SIZE);
                }
                ValueCount<float> probs_gt_pivot_0_pair[VEC_SIZE], probs_gt_pivot_1_pair[VEC_SIZE];
#pragma unroll
                for(uint32_t j = 0; j < VEC_SIZE; ++j)
                {
                    probs_gt_pivot_0_pair[j] = {
                        (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
                        (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
                    probs_gt_pivot_1_pair[j] = {
                        (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
                        (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};

                    if(probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)
                    {
                        min_gt_low = min(min_gt_low, probs_vec[j]);
                    }
                    if(probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)
                    {
                        max_le_high = max(max_le_high, probs_vec[j]);
                    }
                }

                aggregate_gt_pivot_0 +=
                    BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                        temp_storage.block_prim.reduce_value_count)
                        .Sum(probs_gt_pivot_0_pair);
                __syncthreads();

                aggregate_gt_pivot_1 +=
                    BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                        temp_storage.block_prim.reduce_value_count)
                        .Sum(probs_gt_pivot_1_pair);
                __syncthreads();
            }
            min_gt_low =
                BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                    .Reduce(min_gt_low, hipcub::Min());
            __syncthreads();
            max_le_high =
                BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                    .Reduce(max_le_high, hipcub::Max());
            if(tx == 0)
            {
                temp_storage.block_aggregate.pairs[0] = aggregate_gt_pivot_0;
                temp_storage.block_aggregate.pairs[1] = aggregate_gt_pivot_1;
                temp_storage.min_val                  = min_gt_low;
                temp_storage.max_val                  = max_le_high;
            }
            __syncthreads();
            aggregate_gt_pivot_0 = temp_storage.block_aggregate.pairs[0];
            aggregate_gt_pivot_1 = temp_storage.block_aggregate.pairs[1];
            min_gt_low           = temp_storage.min_val;
            max_le_high          = temp_storage.max_val;

            if(aggregate_gt_pivot_1.count >= k)
            {
                low     = pivot_1;
                sum_low = float(aggregate_gt_pivot_1.value);
            }
            else if(aggregate_gt_pivot_0.count >= k)
            {
                low     = pivot_0;
                high    = min(pivot_1, max_le_high);
                sum_low = float(aggregate_gt_pivot_0.value);
            }
            else
            {
                high = min(pivot_0, max_le_high);
            }
        } while(min_gt_low != max_le_high);

        normalizer = __frcp_rn(max(sum_low, 1e-8));
        pivot      = low;
    }

    // normalize
#pragma unroll 2
    for(uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
    {
        probs_vec.fill(0);
        if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
        {
            probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
#pragma unroll
        for(uint32_t j = 0; j < VEC_SIZE; ++j)
        {
            probs_vec[j] = (probs_vec[j] > pivot) ? probs_vec[j] * normalizer : 0;
        }
        if((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
        {
            probs_vec.store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE +
                            tx * VEC_SIZE);
        }
    }
}

} // namespace sampling

} // namespace aiter