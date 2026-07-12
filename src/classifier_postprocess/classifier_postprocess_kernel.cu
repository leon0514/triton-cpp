/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "classifier_postprocess/classifier_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <cub/cub.cuh>

namespace classifier_postprocess
{

static __device__ __forceinline__ float read_input(
    const void *input, bool input_is_half, int idx)
{
    if (input_is_half)
    {
        return __half2float(reinterpret_cast<const __half *>(input)[idx]);
    }
    return reinterpret_cast<const float *>(input)[idx];
}

// 每个 block 处理一个 batch：计算 softmax 概率（或原样拷贝）
__global__ void softmax_kernel(
    const void *input,
    bool input_is_half,
    float *probs,
    int total_images,
    int num_classes,
    bool apply_softmax)
{
    int b = blockIdx.x;
    if (b >= total_images)
    {
        return;
    }

    const int tid = threadIdx.x;
    __shared__ float s_reduce[256];

    const int base = b * num_classes;

    // 1. 求最大值（数值稳定性）
    float local_max = -FLT_MAX;
    for (int c = tid; c < num_classes; c += blockDim.x)
    {
        float v = read_input(input, input_is_half, base + c);
        local_max = fmaxf(local_max, v);
    }
    s_reduce[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_reduce[tid] = fmaxf(s_reduce[tid], s_reduce[tid + s]);
        }
        __syncthreads();
    }
    float max_val = s_reduce[0];

    // 2. 求 exp 之和
    float local_sum = 0.0f;
    if (apply_softmax)
    {
        for (int c = tid; c < num_classes; c += blockDim.x)
        {
            float v = read_input(input, input_is_half, base + c);
            local_sum += expf(v - max_val);
        }
    }
    s_reduce[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_reduce[tid] += s_reduce[tid + s];
        }
        __syncthreads();
    }
    float sum_val = s_reduce[0];

    // 3. 写回概率
    for (int c = tid; c < num_classes; c += blockDim.x)
    {
        float v = read_input(input, input_is_half, base + c);
        float p = apply_softmax ? expf(v - max_val) / sum_val : v;
        probs[base + c] = p;
    }
}

// 初始化排序 value：class index
__global__ void init_sort_values_kernel(
    int total_images,
    int num_classes,
    int *sort_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_classes;
    if (idx >= total)
    {
        return;
    }
    sort_values[idx] = idx % num_classes;
}

// 从排序后的 key/value 中提取 top-k
__global__ void extract_topk_kernel(
    const float *sorted_keys,
    const int *sorted_values,
    int total_images,
    int num_classes,
    int top_k,
    float *scores,
    int *classes)
{
    int b = blockIdx.x;
    int k = threadIdx.x;
    if (b >= total_images || k >= top_k)
    {
        return;
    }

    int idx = b * num_classes + k;
    scores[b * top_k + k] = sorted_keys[idx];
    classes[b * top_k + k] = sorted_values[idx];
}

size_t get_segmented_sort_temp_storage_bytes(
    int total_elements, int num_segments)
{
    size_t bytes = 0;
    float *d_keys_in = nullptr;
    float *d_keys_out = nullptr;
    int *d_values_in = nullptr;
    int *d_values_out = nullptr;
    int *d_offsets = nullptr;

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        total_elements, num_segments,
        d_offsets, d_offsets + 1,
        0, sizeof(float) * 8);

    return bytes;
}

void classifier_postprocess_gpu(
    const void *input,
    float *d_probs,
    int *d_sort_values,
    float *d_sorted_keys,
    int *d_sorted_values,
    int *d_classes,
    float *d_scores,
    bool input_is_half,
    int total_images,
    int num_classes,
    int top_k,
    bool apply_softmax,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_bytes,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_classes <= 0)
    {
        return;
    }

    const int block = 256;

    // 1. softmax（或原样拷贝）
    softmax_kernel<<<total_images, block, 0, stream>>>(
        input, input_is_half, d_probs, total_images, num_classes, apply_softmax);
    checkRuntime(cudaPeekAtLastError());

    // 2. 初始化 value 为 class index
    int total_elements = total_images * num_classes;
    int grid_values = (total_elements + block - 1) / block;
    init_sort_values_kernel<<<grid_values, block, 0, stream>>>(
        total_images, num_classes, d_sort_values);
    checkRuntime(cudaPeekAtLastError());

    // 3. CUB 分段排序：每段一个 batch，按概率降序
    checkRuntime(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_temp, cub_temp_bytes,
        d_probs, d_sorted_keys,
        d_sort_values, d_sorted_values,
        total_elements, total_images,
        d_sort_offsets, d_sort_offsets + 1,
        0, sizeof(float) * 8,
        stream));

    // 4. 提取 top-k
    extract_topk_kernel<<<total_images, top_k, 0, stream>>>(
        d_sorted_keys, d_sorted_values,
        total_images, num_classes, top_k,
        d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());
}

} // namespace classifier_postprocess
