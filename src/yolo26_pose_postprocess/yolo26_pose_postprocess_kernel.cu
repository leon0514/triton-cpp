/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_pose_postprocess/yolo26_pose_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <cub/cub.cuh>

namespace yolo26_pose_postprocess
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
    candidates[idx].kpt_offset = 0;
}

template <typename T>
__global__ void filter_kernel(
    const T *input,
    int total_images,
    int num_predictions,
    int num_keypoints,
    int keypoint_dim,
    float conf_thresh,
    Candidate *candidates,
    float *keypoints,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_predictions;
    if (idx >= total)
        return;

    int batch_idx = idx / num_predictions;
    int pred_idx  = idx % num_predictions;

    int kpt_stride = num_keypoints * keypoint_dim;
    int num_fields = 6 + kpt_stride;
    int offset = (batch_idx * num_predictions + pred_idx) * num_fields;

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

    int kpt_base_offset = (batch_idx * num_predictions + pos) * kpt_stride;

    for (int k = 0; k < kpt_stride; ++k)
    {
        keypoints[kpt_base_offset + k] = static_cast<float>(input[offset + 6 + k]);
    }

    Candidate cand;
    cand.x1        = x1;
    cand.y1        = y1;
    cand.x2        = x2;
    cand.y2        = y2;
    cand.score     = score;
    cand.class_id  = class_id;
    cand.batch_idx = batch_idx;
    cand.kpt_offset = kpt_base_offset;

    candidates[batch_idx * num_predictions + pos] = cand;
}

struct CandidateScoreGreater
{
    __device__ __forceinline__ bool operator()(const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score;
    }
};

// 从候选框中提取 score/key 并复制 value，供 CUB 分段排序使用
__global__ void prepare_sort_kernel(
    const Candidate *candidates,
    int total,
    float *keys_out,
    Candidate *values_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    keys_out[idx]   = candidates[idx].score;
    values_out[idx] = candidates[idx];
}

__global__ void init_output_kernel(
    int total_images,
    int max_detections,
    int num_keypoints,
    int keypoint_dim,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes,
    float *output_keypoints)
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

    float *kpt = output_keypoints + idx * num_keypoints * keypoint_dim;
    for (int k = 0; k < num_keypoints * keypoint_dim; ++k)
    {
        kpt[k] = 0.0f;
    }

    if (idx < total_images)
        num_dets[idx] = 0;
}

__global__ void write_topk_kernel(
    const Candidate *candidates,
    const int *counts,
    int total_images,
    int num_predictions,
    int max_detections,
    int num_keypoints,
    int keypoint_dim,
    const float *keypoints,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes,
    float *output_keypoints)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_images)
        return;

    int count = counts[b];
    if (count > num_predictions)
        count = num_predictions;

    int keep = min(count, max_detections);
    const Candidate *cand = candidates + b * num_predictions;

    int kpt_stride = num_keypoints * keypoint_dim;

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

        const float *src_kpt = keypoints + c.kpt_offset;
        float *dst_kpt       = output_keypoints + out_idx * kpt_stride;
        for (int k = 0; k < kpt_stride; ++k)
        {
            dst_kpt[k] = src_kpt[k];
        }
    }

    num_dets[b] = keep;
}

size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments)
{
    size_t bytes = 0;
    float *d_keys_in = nullptr;
    float *d_keys_out = nullptr;
    Candidate *d_candidates_in = nullptr;
    Candidate *d_candidates_out = nullptr;
    int *d_offsets = nullptr;

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, bytes,
        d_keys_in, d_keys_out,
        d_candidates_in, d_candidates_out,
        total_candidates, num_segments,
        d_offsets, d_offsets + 1,
        0, sizeof(float) * 8);

    return bytes;
}

void yolo26_pose_postprocess_gpu(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_predictions,
    int num_keypoints,
    int keypoint_dim,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    float *d_keypoints,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_output_keypoints,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
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

    // 2. 置信度过滤 + 关键点读取
    const int total_work = total_images * num_predictions;
    const int grid_filter = (total_work + block - 1) / block;
    if (input_is_half)
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            total_images, num_predictions, num_keypoints, keypoint_dim,
            conf_thresh, d_candidates, d_keypoints, d_counts);
    }
    else
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const float *>(input),
            total_images, num_predictions, num_keypoints, keypoint_dim,
            conf_thresh, d_candidates, d_keypoints, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());

    // 3. 使用 CUB DeviceSegmentedRadixSort 在 GPU 上对所有 batch 并行排序
    int sort_total = total_images * num_predictions;
    int grid_sort = (sort_total + block - 1) / block;
    prepare_sort_kernel<<<grid_sort, block, 0, stream>>>(
        d_candidates, sort_total,
        d_sort_keys_in, d_sort_candidates_in);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_temp, cub_temp_storage_bytes,
        d_sort_keys_in, d_sort_keys_out,
        d_sort_candidates_in, d_sort_candidates_out,
        sort_total,
        total_images,
        d_sort_offsets, d_sort_offsets + 1,
        0, sizeof(float) * 8,
        stream));

    // 将排序后的候选框复制回 d_candidates，供后续 top-K 写入使用
    checkRuntime(cudaMemcpyAsync(
        d_candidates, d_sort_candidates_out,
        sort_total * sizeof(Candidate),
        cudaMemcpyDeviceToDevice, stream));

    // 4. 清零输出缓冲区
    const int total_out = total_images * max_detections;
    const int grid_out = (total_out + block - 1) / block;
    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections, num_keypoints, keypoint_dim,
        d_num_dets, d_boxes, d_scores, d_classes, d_output_keypoints);
    checkRuntime(cudaPeekAtLastError());

    // 5. 写前 max_detections 个结果
    const int grid_batch = (total_images + block - 1) / block;
    write_topk_kernel<<<grid_batch, block, 0, stream>>>(
        d_candidates, d_counts,
        total_images, num_predictions, max_detections,
        num_keypoints, keypoint_dim, d_keypoints,
        d_num_dets, d_boxes, d_scores, d_classes, d_output_keypoints);
    checkRuntime(cudaPeekAtLastError());
}

} // namespace yolo26_pose_postprocess
