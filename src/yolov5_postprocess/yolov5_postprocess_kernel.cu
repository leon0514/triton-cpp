/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolov5_postprocess/yolov5_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <cub/cub.cuh>

namespace yolov5_postprocess
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
    bool anchors_first,
    bool apply_sigmoid,
    bool has_objectness,
    float conf_thresh,
    int max_candidates,
    float *sort_keys,
    Candidate *sort_candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_anchors;
    if (idx >= total)
        return;

    int batch_idx  = idx / num_anchors;
    int anchor_idx = idx % num_anchors;
    int num_channels = num_classes + (has_objectness ? 5 : 4);
    int cls_offset = has_objectness ? 5 : 4;

    // 先算 score（objectness + 类别 argmax）并做置信度过滤：
    // 被阈值丢弃的 anchor（通常占绝大多数）不再读取框坐标。
    float obj_conf = 1.0f;
    if (has_objectness)
    {
        float obj_logit = read_input(input, batch_idx, anchor_idx, 4,
                                     num_anchors, num_channels, anchors_first);
        obj_conf = apply_sigmoid ? sigmoid(obj_logit) : obj_logit;
    }

    if (obj_conf < conf_thresh)
        return;

    float max_logit = -FLT_MAX;
    int class_id    = 0;

    for (int c = 0; c < num_classes; ++c)
    {
        float logit = read_input(
            input, batch_idx, anchor_idx, cls_offset + c,
            num_anchors, num_channels, anchors_first);
        if (logit > max_logit)
        {
            max_logit = logit;
            class_id  = c;
        }
    }

    float cls_conf = apply_sigmoid ? sigmoid(max_logit) : max_logit;
    float score = obj_conf * cls_conf;

    if (score < conf_thresh)
        return;

    int pos = atomicAdd(counts + batch_idx, 1);
    if (pos >= max_candidates)
        return;

    float cx = read_input(input, batch_idx, anchor_idx, 0, num_anchors, num_channels, anchors_first);
    float cy = read_input(input, batch_idx, anchor_idx, 1, num_anchors, num_channels, anchors_first);
    float w  = read_input(input, batch_idx, anchor_idx, 2, num_anchors, num_channels, anchors_first);
    float h  = read_input(input, batch_idx, anchor_idx, 3, num_anchors, num_channels, anchors_first);

    Candidate cand;
    cand.x1       = cx - w * 0.5f;
    cand.y1       = cy - h * 0.5f;
    cand.x2       = cx + w * 0.5f;
    cand.y2       = cy + h * 0.5f;
    cand.score    = score;
    cand.class_id = class_id;
    cand.batch_idx = batch_idx;

    // 直接写入 CUB 排序输入缓冲区，省掉中间 candidates 缓冲和 prepare_sort 拷贝
    int out_idx = batch_idx * max_candidates + pos;
    sort_keys[out_idx]       = score;
    sort_candidates[out_idx] = cand;
}

__global__ void nms_kernel(
    const Candidate *candidates,
    const int *counts,
    int total_images,
    int max_candidates,
    int max_detections,
    float iou_thresh,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_images)
        return;

    // counts 可能因 atomicAdd 超过 max_candidates（超过部分未写入缓冲区），这里封顶
    int count = min(counts[b], max_candidates);
    if (count <= 0)
    {
        num_dets[b] = 0;
        return;
    }

    const Candidate *cand = candidates + b * max_candidates;

    float *boxes_b   = boxes + b * max_detections * 4;
    float *scores_b  = scores + b * max_detections;
    int *classes_b   = classes + b * max_detections;

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
            ++kept;
        }
    }

    num_dets[b] = kept;
}

// 由 counts 计算 CUB 分段排序的 [begin, end) 偏移：
// 每段只覆盖实际候选（段间空隙不参与排序，CUB 保证不改动空隙元素），
// 避免对 max_candidates 固定窗口做全量排序，也省去候选缓冲区的初始化。
__global__ void compute_segment_offsets_kernel(
    const int *counts,
    int total_images,
    int max_candidates,
    int *begin_offsets,
    int *end_offsets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_images)
        return;

    int begin = idx * max_candidates;
    begin_offsets[idx] = begin;
    end_offsets[idx]   = begin + min(counts[idx], max_candidates);
}

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

size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments)
{
    size_t bytes = 0;
    float *d_keys_in = nullptr;
    float *d_keys_out = nullptr;
    Candidate *d_candidates_in = nullptr;
    Candidate *d_candidates_out = nullptr;
    int *d_begin_offsets = nullptr;
    int *d_end_offsets = nullptr;

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, bytes,
        d_keys_in, d_keys_out,
        d_candidates_in, d_candidates_out,
        total_candidates, num_segments,
        d_begin_offsets, d_end_offsets,
        0, sizeof(float) * 8);

    return bytes;
}

void yolov5_postprocess_gpu(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_anchors,
    int num_classes,
    bool anchors_first,
    bool apply_sigmoid,
    bool has_objectness,
    float conf_thresh,
    float iou_thresh,
    int max_detections,
    int max_candidates,
    int *d_counts,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_begin_offsets,
    int *d_sort_end_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_anchors <= 0)
        return;

    const int block = 256;

    // 初始化候选计数和输出缓冲区
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));

    int total_out = total_images * max_detections;
    int grid_out  = (total_out + block - 1) / block;
    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());

    // decode + confidence 过滤，候选直接写入 CUB 排序输入缓冲区
    int total_anchors = total_images * num_anchors;
    int grid_dec  = (total_anchors + block - 1) / block;

    if (input_is_half)
    {
        decode_filter_kernel<<<grid_dec, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            total_images, num_anchors, num_classes, anchors_first, apply_sigmoid, has_objectness,
            conf_thresh, max_candidates,
            d_sort_keys_in, d_sort_candidates_in, d_counts);
    }
    else
    {
        decode_filter_kernel<<<grid_dec, block, 0, stream>>>(
            reinterpret_cast<const float *>(input),
            total_images, num_anchors, num_classes, anchors_first, apply_sigmoid, has_objectness,
            conf_thresh, max_candidates,
            d_sort_keys_in, d_sort_candidates_in, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());

    // 按实际候选数计算分段偏移，CUB 只排有效候选（段间空隙不参与排序）
    int grid_seg = (total_images + block - 1) / block;
    compute_segment_offsets_kernel<<<grid_seg, block, 0, stream>>>(
        d_counts, total_images, max_candidates,
        d_sort_begin_offsets, d_sort_end_offsets);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_temp, cub_temp_storage_bytes,
        d_sort_keys_in, d_sort_keys_out,
        d_sort_candidates_in, d_sort_candidates_out,
        total_images * max_candidates,
        total_images,
        d_sort_begin_offsets, d_sort_end_offsets,
        0, sizeof(float) * 8,
        stream));

    // NMS 直接读取排序输出（每张图一个线程，候选数通常几十到几百，完全够用）
    nms_kernel<<<total_images, 1, 0, stream>>>(
        d_sort_candidates_out, d_counts,
        total_images, max_candidates, max_detections,
        iou_thresh,
        d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());
}

} // namespace yolov5_postprocess
