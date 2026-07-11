/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_postprocess/yolo26_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace yolo26_postprocess
{

__global__ void init_candidates_kernel(
    int total_images,
    int num_predictions,
    Candidate *candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_predictions;
    if (idx >= total)
        return;

    candidates[idx].score    = -FLT_MAX;
    candidates[idx].class_id = -1;
    candidates[idx].batch_idx = idx / num_predictions;
}

template <typename T>
__global__ void filter_kernel(
    const T *input,
    int total_images,
    int num_predictions,
    float conf_thresh,
    Candidate *candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_predictions;
    if (idx >= total)
        return;

    int batch_idx = idx / num_predictions;
    int pred_idx  = idx % num_predictions;

    int offset = (batch_idx * num_predictions + pred_idx) * 6;
    float x1    = static_cast<float>(input[offset + 0]);
    float y1    = static_cast<float>(input[offset + 1]);
    float x2    = static_cast<float>(input[offset + 2]);
    float y2    = static_cast<float>(input[offset + 3]);
    float score = static_cast<float>(input[offset + 4]);
    int class_id = __float2int_rn(static_cast<float>(input[offset + 5]));

    if (score < conf_thresh)
        return;

    int pos = atomicAdd(counts + batch_idx, 1);
    if (pos >= num_predictions)
        return;

    Candidate cand;
    cand.x1       = x1;
    cand.y1       = y1;
    cand.x2       = x2;
    cand.y2       = y2;
    cand.score    = score;
    cand.class_id = class_id;
    cand.batch_idx = batch_idx;

    candidates[batch_idx * num_predictions + pos] = cand;
}

struct CandidateScoreGreater
{
    __device__ __forceinline__ bool operator()(const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score;
    }
};

__global__ void init_output_kernel(
    int total_images,
    int max_detections,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * max_detections;
    if (idx >= total)
        return;

    boxes[idx * 4 + 0] = 0.0f;
    boxes[idx * 4 + 1] = 0.0f;
    boxes[idx * 4 + 2] = 0.0f;
    boxes[idx * 4 + 3] = 0.0f;
    scores[idx]        = 0.0f;
    classes[idx]       = -1;

    if (idx < total_images)
        num_dets[idx] = 0;
}

__global__ void write_topk_kernel(
    const Candidate *candidates,
    const int *counts,
    int total_images,
    int num_predictions,
    int max_detections,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_images)
        return;

    int count = counts[b];
    if (count > num_predictions)
        count = num_predictions;

    int keep = min(count, max_detections);
    const Candidate *cand = candidates + b * num_predictions;

    for (int i = 0; i < keep; ++i)
    {
        const Candidate &c = cand[i];
        int out_idx = b * max_detections + i;
        boxes[out_idx * 4 + 0] = c.x1;
        boxes[out_idx * 4 + 1] = c.y1;
        boxes[out_idx * 4 + 2] = c.x2;
        boxes[out_idx * 4 + 3] = c.y2;
        scores[out_idx]        = c.score;
        classes[out_idx]       = c.class_id;
    }

    num_dets[b] = keep;
}

void yolo26_postprocess_gpu(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_predictions,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_predictions <= 0)
        return;

    const int block = 256;
    const int total_cand = total_images * num_predictions;
    const int grid_cand = (total_cand + block - 1) / block;

    // 1. 清零计数并初始化候选缓冲区（未填充位置 score=-FLT_MAX，确保排序沉底）
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));
    init_candidates_kernel<<<grid_cand, block, 0, stream>>>(
        total_images, num_predictions, d_candidates);
    checkRuntime(cudaPeekAtLastError());

    // 2. 置信度过滤
    const int total_work = total_images * num_predictions;
    const int grid_filter = (total_work + block - 1) / block;
    if (input_is_half)
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            total_images, num_predictions, conf_thresh,
            d_candidates, d_counts);
    }
    else
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const float *>(input),
            total_images, num_predictions, conf_thresh,
            d_candidates, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());

    // 3. 每张图按 score 降序排序
    for (int b = 0; b < total_images; ++b)
    {
        thrust::device_ptr<Candidate> ptr(d_candidates + b * num_predictions);
        thrust::sort(ptr, ptr + num_predictions, CandidateScoreGreater());
    }

    // 4. 清零输出缓冲区
    const int total_out = total_images * max_detections;
    const int grid_out = (total_out + block - 1) / block;
    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());

    // 5. 写前 max_detections 个结果
    const int grid_batch = (total_images + block - 1) / block;
    write_topk_kernel<<<grid_batch, block, 0, stream>>>(
        d_candidates, d_counts,
        total_images, num_predictions, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());
}

} // namespace yolo26_postprocess
