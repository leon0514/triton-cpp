/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_pose_postprocess/yolo11_pose_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

namespace yolo11_pose_postprocess
{

static __device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float iou(
    float x1, float y1, float x2, float y2,
    float x1_, float y1_, float x2_, float y2_)
{
    float ix1 = fmaxf(x1, x1_);
    float iy1 = fmaxf(y1, y1_);
    float ix2 = fminf(x2, x2_);
    float iy2 = fminf(y2, y2_);

    float inter_w = fmaxf(0.0f, ix2 - ix1);
    float inter_h = fmaxf(0.0f, iy2 - iy1);
    float inter   = inter_w * inter_h;

    float area1 = (x2 - x1) * (y2 - y1);
    float area2 = (x2_ - x1_) * (y2_ - y1_);
    float uni   = area1 + area2 - inter + 1e-6f;

    return inter / uni;
}

template <typename T>
static __device__ __forceinline__ float read_input(
    const T *input,
    int batch_idx,
    int anchor_idx,
    int channel_idx,
    int num_anchors,
    int num_channels,
    bool anchors_first)
{
    int idx;
    if (anchors_first)
    {
        idx = batch_idx * num_anchors * num_channels +
              anchor_idx * num_channels +
              channel_idx;
    }
    else
    {
        idx = batch_idx * num_channels * num_anchors +
              channel_idx * num_anchors +
              anchor_idx;
    }
    return static_cast<float>(input[idx]);
}

template <typename T>
__global__ void decode_filter_kernel(
    const T *input,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_keypoints,
    int keypoint_dim,
    bool anchors_first,
    bool apply_sigmoid,
    bool apply_sigmoid_keypoints,
    float conf_thresh,
    int max_candidates,
    Candidate *candidates,
    float *keypoints,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_anchors;
    if (idx >= total)
        return;

    int batch_idx  = idx / num_anchors;
    int anchor_idx = idx % num_anchors;
    int num_channels = 4 + num_classes + num_keypoints * keypoint_dim;

    float cx = read_input(input, batch_idx, anchor_idx, 0, num_anchors, num_channels, anchors_first);
    float cy = read_input(input, batch_idx, anchor_idx, 1, num_anchors, num_channels, anchors_first);
    float w  = read_input(input, batch_idx, anchor_idx, 2, num_anchors, num_channels, anchors_first);
    float h  = read_input(input, batch_idx, anchor_idx, 3, num_anchors, num_channels, anchors_first);

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    float max_logit = -FLT_MAX;
    int class_id    = 0;

    for (int c = 0; c < num_classes; ++c)
    {
        float logit = read_input(
            input, batch_idx, anchor_idx, 4 + c,
            num_anchors, num_channels, anchors_first);
        if (apply_sigmoid)
        {
            logit = sigmoid(logit);
        }
        if (logit > max_logit)
        {
            max_logit = logit;
            class_id  = c;
        }
    }

    float score = max_logit;
    if (score < conf_thresh)
        return;

    int pos = atomicAdd(counts + batch_idx, 1);
    if (pos >= max_candidates)
        return;

    int kpt_base_offset = (batch_idx * max_candidates + pos) * num_keypoints * keypoint_dim;

    for (int k = 0; k < num_keypoints; ++k)
    {
        int src_channel = 4 + num_classes + k * keypoint_dim;
        int dst_offset  = kpt_base_offset + k * keypoint_dim;

        for (int d = 0; d < keypoint_dim; ++d)
        {
            float val = read_input(
                input, batch_idx, anchor_idx, src_channel + d,
                num_anchors, num_channels, anchors_first);

            // 最后一个维度默认为 visibility/confidence，可按需做 sigmoid
            if (d == keypoint_dim - 1 && apply_sigmoid_keypoints)
            {
                val = sigmoid(val);
            }

            keypoints[dst_offset + d] = val;
        }
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

    candidates[batch_idx * max_candidates + pos] = cand;
}

struct CandidateScoreGreater
{
    __device__ __forceinline__ bool operator()(
        const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score;
    }
};

__global__ void nms_kernel(
    const Candidate *candidates,
    const int *counts,
    int total_images,
    int max_candidates,
    int max_detections,
    int num_keypoints,
    int keypoint_dim,
    float iou_thresh,
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
    if (count <= 0)
    {
        num_dets[b] = 0;
        return;
    }

    const Candidate *cand = candidates + b * max_candidates;

    float *boxes_b   = boxes + b * max_detections * 4;
    float *scores_b  = scores + b * max_detections;
    int *classes_b   = classes + b * max_detections;
    float *kpts_b    = output_keypoints + b * max_detections * num_keypoints * keypoint_dim;

    int kept = 0;
    for (int i = 0; i < count && kept < max_detections; ++i)
    {
        const Candidate &c = cand[i];

        bool keep = true;
        for (int j = 0; j < kept; ++j)
        {
            if (classes_b[j] != c.class_id)
                continue;

            float iou_val = iou(
                c.x1, c.y1, c.x2, c.y2,
                boxes_b[j * 4 + 0],
                boxes_b[j * 4 + 1],
                boxes_b[j * 4 + 2],
                boxes_b[j * 4 + 3]);

            if (iou_val > iou_thresh)
            {
                keep = false;
                break;
            }
        }

        if (keep)
        {
            boxes_b[kept * 4 + 0] = c.x1;
            boxes_b[kept * 4 + 1] = c.y1;
            boxes_b[kept * 4 + 2] = c.x2;
            boxes_b[kept * 4 + 3] = c.y2;
            scores_b[kept]        = c.score;
            classes_b[kept]       = c.class_id;

            const float *src_kpt = keypoints + c.kpt_offset;
            float *dst_kpt       = kpts_b + kept * num_keypoints * keypoint_dim;
            for (int k = 0; k < num_keypoints * keypoint_dim; ++k)
            {
                dst_kpt[k] = src_kpt[k];
            }

            ++kept;
        }
    }

    num_dets[b] = kept;
}

__global__ void init_candidates_kernel(
    int total_images,
    int max_candidates,
    Candidate *candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * max_candidates;
    if (idx >= total)
        return;

    candidates[idx].score     = -FLT_MAX;
    candidates[idx].class_id  = -1;
    candidates[idx].batch_idx = idx / max_candidates;
    candidates[idx].kpt_offset = 0;
}

__global__ void cap_counts_kernel(
    int total_images,
    int max_candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_images)
        return;

    if (counts[idx] > max_candidates)
        counts[idx] = max_candidates;
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

void yolo11_pose_postprocess_gpu(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_keypoints,
    int keypoint_dim,
    bool anchors_first,
    bool apply_sigmoid,
    bool apply_sigmoid_keypoints,
    float conf_thresh,
    float iou_thresh,
    int max_detections,
    int max_candidates,
    int *d_counts,
    Candidate *d_candidates,
    float *d_keypoints,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_output_keypoints,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_anchors <= 0)
        return;

    // 初始化候选计数、候选框缓冲区和输出缓冲区
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));

    int total_cand = total_images * max_candidates;
    int block_init = 256;
    int grid_cand  = (total_cand + block_init - 1) / block_init;
    init_candidates_kernel<<<grid_cand, block_init, 0, stream>>>(
        total_images, max_candidates, d_candidates);
    checkRuntime(cudaPeekAtLastError());

    int total_out = total_images * max_detections;
    int grid_out  = (total_out + block_init - 1) / block_init;
    init_output_kernel<<<grid_out, block_init, 0, stream>>>(
        total_images, max_detections, num_keypoints, keypoint_dim,
        d_num_dets, d_boxes, d_scores, d_classes, d_output_keypoints);
    checkRuntime(cudaPeekAtLastError());

    // decode + confidence 过滤
    int total_anchors = total_images * num_anchors;
    int block_dec = 256;
    int grid_dec  = (total_anchors + block_dec - 1) / block_dec;

    if (input_is_half)
    {
        decode_filter_kernel<<<grid_dec, block_dec, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            total_images, num_anchors, num_classes, num_keypoints, keypoint_dim,
            anchors_first, apply_sigmoid, apply_sigmoid_keypoints,
            conf_thresh, max_candidates,
            d_candidates, d_keypoints, d_counts);
    }
    else
    {
        decode_filter_kernel<<<grid_dec, block_dec, 0, stream>>>(
            reinterpret_cast<const float *>(input),
            total_images, num_anchors, num_classes, num_keypoints, keypoint_dim,
            anchors_first, apply_sigmoid, apply_sigmoid_keypoints,
            conf_thresh, max_candidates,
            d_candidates, d_keypoints, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());

    // 防止候选数超过 max_candidates
    int grid_cap = (total_images + block_init - 1) / block_init;
    cap_counts_kernel<<<grid_cap, block_init, 0, stream>>>(
        total_images, max_candidates, d_counts);
    checkRuntime(cudaPeekAtLastError());

    // 将每张图的候选框按 score 降序排序
    for (int b = 0; b < total_images; ++b)
    {
        thrust::device_ptr<Candidate> ptr(d_candidates + b * max_candidates);
        thrust::sort(ptr, ptr + max_candidates, CandidateScoreGreater());
    }

    // NMS，每张图一个线程（候选数通常几百，完全够用）
    int block_nms = 1;
    int grid_nms  = total_images;
    nms_kernel<<<grid_nms, block_nms, 0, stream>>>(
        d_candidates, d_counts,
        total_images, max_candidates, max_detections,
        num_keypoints, keypoint_dim,
        iou_thresh, d_keypoints,
        d_num_dets, d_boxes, d_scores, d_classes, d_output_keypoints);
    checkRuntime(cudaPeekAtLastError());
}

} // namespace yolo11_pose_postprocess
