/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "rfdetr_postprocess/rfdetr_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace rfdetr_postprocess
{

// RF-DETR ONNX 输出 labels 维度为 91：
//   - dim 0 为背景/无目标槽位
//   - dim 1..90 对应官方 COCO ID 1..90
// COCO 90 个 ID 中有 10 个空 ID，由 config.pbtxt 中的 skip_coco_ids 指定。
// 后处理跳过空类别，输出 class_id 为 names 文件中的 0-based 索引。

__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__global__ void init_candidates_kernel(
    int total_images,
    int num_queries,
    Candidate *candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_queries;
    if (idx >= total)
        return;

    candidates[idx].score     = -FLT_MAX;
    candidates[idx].class_id  = -1;
    candidates[idx].batch_idx = idx / num_queries;
}

template <typename T>
__global__ void filter_kernel(
    const T *dets,
    const T *labels,
    int total_images,
    int num_queries,
    float input_width,
    float input_height,
    float conf_thresh,
    const int *coco_id_to_index,
    Candidate *candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_queries;
    if (idx >= total)
        return;

    int batch_idx = idx / num_queries;
    int query_idx = idx % num_queries;

    // dets 读取: [batch, num_queries, 4] 归一化 cxcywh
    int det_offset = (batch_idx * num_queries + query_idx) * 4;
    float cx = static_cast<float>(dets[det_offset + 0]);
    float cy = static_cast<float>(dets[det_offset + 1]);
    float w  = static_cast<float>(dets[det_offset + 2]);
    float h  = static_cast<float>(dets[det_offset + 3]);

    // labels 读取: [batch, num_queries, 91]，dim 0 为背景，dim 1..90 为 COCO ID 1~90
    int label_offset = (batch_idx * num_queries + query_idx) * 91;

    // 对有效 COCO ID 做 sigmoid，取最大值与索引
    float best_score  = 0.0f;
    int best_class_id = -1;  // names 文件中的 0-based 索引
    #pragma unroll 4
    for (int coco_id = 1; coco_id <= 90; ++coco_id)
    {
        int class_idx = coco_id_to_index[coco_id];
        if (class_idx < 0)
            continue;

        float logit = static_cast<float>(labels[label_offset + coco_id]);
        float score = sigmoid(logit);
        if (score > best_score)
        {
            best_score  = score;
            best_class_id = class_idx;
        }
    }

    if (best_class_id < 0)
        return;

    if (best_score < conf_thresh)
        return;

    // 反归一化到模型输入坐标系，并转换为 xyxy
    float x1 = (cx - 0.5f * w) * input_width;
    float y1 = (cy - 0.5f * h) * input_height;
    float x2 = (cx + 0.5f * w) * input_width;
    float y2 = (cy + 0.5f * h) * input_height;

    int pos = atomicAdd(counts + batch_idx, 1);
    if (pos >= num_queries)
        return;

    Candidate cand;
    cand.x1       = x1;
    cand.y1       = y1;
    cand.x2       = x2;
    cand.y2       = y2;
    cand.score    = best_score;
    // 输出 names 文件中的 0-based 索引
    cand.class_id = best_class_id;
    cand.batch_idx = batch_idx;

    candidates[batch_idx * num_queries + pos] = cand;
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
    int num_queries,
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
    if (count > num_queries)
        count = num_queries;

    int keep = min(count, max_detections);
    const Candidate *cand = candidates + b * num_queries;

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

void rfdetr_postprocess_gpu(
    const void *dets,
    const void *labels,
    bool input_is_half,
    int total_images,
    int num_queries,
    float input_width,
    float input_height,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    const int *d_coco_id_to_index,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_queries <= 0)
        return;

    const int block = 256;
    const int total_cand = total_images * num_queries;
    const int grid_cand = (total_cand + block - 1) / block;

    // 1. 清零计数并初始化候选缓冲区
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));
    init_candidates_kernel<<<grid_cand, block, 0, stream>>>(
        total_images, num_queries, d_candidates);
    checkRuntime(cudaPeekAtLastError());

    // 2. 置信度过滤、类别选择、坐标转换
    const int total_work = total_images * num_queries;
    const int grid_filter = (total_work + block - 1) / block;
    if (input_is_half)
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const half *>(dets),
            reinterpret_cast<const half *>(labels),
            total_images, num_queries,
            input_width, input_height,
            conf_thresh,
            d_coco_id_to_index,
            d_candidates, d_counts);
    }
    else
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const float *>(dets),
            reinterpret_cast<const float *>(labels),
            total_images, num_queries,
            input_width, input_height,
            conf_thresh,
            d_coco_id_to_index,
            d_candidates, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());

    // 3. 每张图按 score 降序排序
    for (int b = 0; b < total_images; ++b)
    {
        thrust::device_ptr<Candidate> ptr(d_candidates + b * num_queries);
        thrust::sort(ptr, ptr + num_queries, CandidateScoreGreater());
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
        total_images, num_queries, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());
}

} // namespace rfdetr_postprocess
