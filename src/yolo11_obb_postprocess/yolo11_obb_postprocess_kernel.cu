/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_obb_postprocess/yolo11_obb_postprocess_kernel.hpp"
#include "rotated_iou.cuh"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>

namespace yolo11_obb_postprocess
{

// ------------------------------------------------------------------
// 常量 / 工具
// ------------------------------------------------------------------

static inline __host__ __device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// 读取 half/float 输入
__device__ __forceinline__ float read_input(const void *input, bool is_half, int idx)
{
    if (is_half)
    {
        return __half2float(reinterpret_cast<const half *>(input)[idx]);
    }
    return reinterpret_cast<const float *>(input)[idx];
}

// 输入索引：channels_first [batch, channels, anchors]
__device__ __forceinline__ int idx_ch_first(int batch_idx, int ch, int anchor, int total_channels, int num_anchors)
{
    return batch_idx * total_channels * num_anchors + ch * num_anchors + anchor;
}

// 输入索引：anchors_first [batch, anchors, channels]
__device__ __forceinline__ int idx_anchor_first(int batch_idx, int anchor, int ch, int num_anchors, int total_channels)
{
    return batch_idx * num_anchors * total_channels + anchor * total_channels + ch;
}

__device__ __forceinline__ float get_value(
    const void *input,
    bool is_half,
    bool anchors_first,
    int batch_idx,
    int anchor,
    int ch,
    int total_channels,
    int num_anchors)
{
    int idx = anchors_first
                  ? idx_anchor_first(batch_idx, anchor, ch, num_anchors, total_channels)
                  : idx_ch_first(batch_idx, ch, anchor, total_channels, num_anchors);
    return read_input(input, is_half, idx);
}

// ------------------------------------------------------------------
// 初始化候选框缓冲区（避免未初始化脏数据参与排序）
// ------------------------------------------------------------------
__global__ void init_candidates_kernel(
    Candidate *d_candidates,
    int total_images,
    int max_candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * max_candidates;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x)
    {
        Candidate cand;
        cand.cx        = 0.0f;
        cand.cy        = 0.0f;
        cand.w         = 0.0f;
        cand.h         = 0.0f;
        cand.angle     = 0.0f;
        cand.score     = -1.0f;         // 无效候选，NMS 会跳过
        cand.class_id  = 0;
        cand.batch_idx = i / max_candidates;
        d_candidates[i] = cand;
    }
}

// ------------------------------------------------------------------
// Decode + 置信度过滤 kernel
// ------------------------------------------------------------------
__global__ void decode_filter_kernel(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_anchors,
    int num_classes,
    bool anchors_first,
    bool apply_sigmoid,
    float conf_thresh,
    int max_candidates,
    int *d_counts,
    Candidate *d_candidates)
{
    const int total_channels = 5 + num_classes;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_images * num_anchors;
         index += blockDim.x * gridDim.x)
    {
        int batch_idx = index / num_anchors;
        int anchor    = index % num_anchors;

        // 原版 Ultralytics ONNX 输出顺序：box(4) + cls(num_classes) + angle(1)
        // 几何量已完成 grid/stride 解码，类别与角度均已通过 Sigmoid/解码
        float cx = get_value(input, input_is_half, anchors_first, batch_idx, anchor, 0,
                             total_channels, num_anchors);
        float cy = get_value(input, input_is_half, anchors_first, batch_idx, anchor, 1,
                             total_channels, num_anchors);
        float w  = get_value(input, input_is_half, anchors_first, batch_idx, anchor, 2,
                             total_channels, num_anchors);
        float h  = get_value(input, input_is_half, anchors_first, batch_idx, anchor, 3,
                             total_channels, num_anchors);

        // 找最大类别分数（原版导出已做 Sigmoid）
        float best_score = 0.0f;
        int best_class   = 0;
        for (int c = 0; c < num_classes; ++c)
        {
            float score = get_value(input, input_is_half, anchors_first, batch_idx, anchor,
                                    4 + c, total_channels, num_anchors);
            if (apply_sigmoid)
            {
                score = sigmoid(score);
            }
            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }

        // 角度通道：原版 Ultralytics ONNX 导出后已是 (sigmoid(raw)-0.25)*π
        float angle = get_value(input, input_is_half, anchors_first, batch_idx, anchor,
                                4 + num_classes, total_channels, num_anchors);

        if (best_score < conf_thresh)
        {
            continue;
        }

        // 限制候选数量
        int *count_ptr = d_counts + batch_idx;
        int old        = atomicAdd(count_ptr, 1);
        if (old >= max_candidates)
        {
            // 还原计数值
            atomicSub(count_ptr, 1);
            continue;
        }

        Candidate cand;
        cand.cx       = cx;
        cand.cy       = cy;
        cand.w        = w;
        cand.h        = h;
        cand.angle    = angle;
        cand.score    = best_score;
        cand.class_id = best_class;
        cand.batch_idx = batch_idx;

        d_candidates[batch_idx * max_candidates + old] = cand;
    }
}

// ------------------------------------------------------------------
// 从候选框中提取 score/key 并复制 value，供 CUB 分段排序使用
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// 初始化输出缓冲区：未使用位置清零，避免客户端读到脏数据
// ------------------------------------------------------------------
__global__ void init_output_kernel(
    int total_images,
    int max_detections,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * max_detections;
    if (idx >= total)
        return;

    d_boxes[idx * 5 + 0] = 0.0f;
    d_boxes[idx * 5 + 1] = 0.0f;
    d_boxes[idx * 5 + 2] = 0.0f;
    d_boxes[idx * 5 + 3] = 0.0f;
    d_boxes[idx * 5 + 4] = 0.0f;
    d_scores[idx]        = 0.0f;
    d_classes[idx]       = -1;

    if (idx < total_images)
        d_num_dets[idx] = 0;
}

// ------------------------------------------------------------------
// NMS kernel：每图一个线程，候选框已按 score 降序
// ------------------------------------------------------------------
__global__ void nms_kernel(
    int total_images,
    Candidate *d_candidates,
    const int *d_counts,
    int max_candidates,
    float iou_thresh,
    int max_detections,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes)
{
    for (int batch_idx = blockIdx.x * blockDim.x + threadIdx.x; batch_idx < total_images;
         batch_idx += blockDim.x * gridDim.x)
    {
        int count = d_counts[batch_idx];
        count     = min(count, max_candidates);

        int keep_count = 0;
        for (int i = 0; i < count && keep_count < max_detections; ++i)
        {
            const Candidate &ci = d_candidates[batch_idx * max_candidates + i];
            if (ci.score < 0.0f)
                continue; // 已被抑制

            // 保留该框
            int out_idx = batch_idx * max_detections + keep_count;
            d_boxes[out_idx * 5 + 0]   = ci.cx;
            d_boxes[out_idx * 5 + 1]   = ci.cy;
            d_boxes[out_idx * 5 + 2]   = ci.w;
            d_boxes[out_idx * 5 + 3]   = ci.h;
            d_boxes[out_idx * 5 + 4]   = ci.angle;
            d_scores[out_idx]          = ci.score;
            d_classes[out_idx]         = ci.class_id;
            keep_count++;

            // 抑制后续同类候选（使用精确旋转 IoU）
            for (int j = i + 1; j < count; ++j)
            {
                Candidate &cj = d_candidates[batch_idx * max_candidates + j];
                if (cj.score < 0.0f || cj.class_id != ci.class_id)
                    continue;

                float iou = rotated_iou(
                    ci.cx, ci.cy, ci.w, ci.h, ci.angle,
                    cj.cx, cj.cy, cj.w, cj.h, cj.angle);
                if (iou > iou_thresh)
                {
                    cj.score = -1.0f; // 标记为抑制
                }
            }
        }
        d_num_dets[batch_idx] = keep_count;
    }
}

// ------------------------------------------------------------------
// 查询 CUB DeviceSegmentedRadixSort 所需临时存储字节数
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// 对外主函数
// ------------------------------------------------------------------
void yolo11_obb_postprocess_gpu(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_anchors,
    int num_classes,
    bool anchors_first,
    bool apply_sigmoid,
    float conf_thresh,
    float iou_thresh,
    int max_detections,
    int max_candidates,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream)
{
    const int total_work = total_images * num_anchors;
    int block_size       = 256;
    int grid_size        = (total_work + block_size - 1) / block_size;
    grid_size            = min(grid_size, 65536);

    // 1. 初始化候选框缓冲区
    int init_total = total_images * max_candidates;
    int init_grid  = (init_total + block_size - 1) / block_size;
    init_grid      = min(init_grid, 65536);
    init_candidates_kernel<<<init_grid, block_size, 0, stream>>>(
        d_candidates, total_images, max_candidates);
    checkRuntime(cudaPeekAtLastError());

    // 2. 清零计数
    cudaMemsetAsync(d_counts, 0, sizeof(int) * total_images, stream);

    // 3. 解码与过滤
    decode_filter_kernel<<<grid_size, block_size, 0, stream>>>(
        input,
        input_is_half,
        total_images,
        num_anchors,
        num_classes,
        anchors_first,
        apply_sigmoid,
        conf_thresh,
        max_candidates,
        d_counts,
        d_candidates);
    checkRuntime(cudaPeekAtLastError());

    // 4. 使用 CUB DeviceSegmentedRadixSort 在 GPU 上对所有 batch 并行排序
    int sort_total = total_images * max_candidates;
    int grid_sort  = (sort_total + block_size - 1) / block_size;
    prepare_sort_kernel<<<grid_sort, block_size, 0, stream>>>(
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

    // 将排序后的候选框复制回 d_candidates，供 NMS 使用
    checkRuntime(cudaMemcpyAsync(
        d_candidates, d_sort_candidates_out,
        sort_total * sizeof(Candidate),
        cudaMemcpyDeviceToDevice, stream));

    // 5. 清零输出缓冲区
    int out_total = total_images * max_detections;
    int out_grid  = (out_total + block_size - 1) / block_size;
    out_grid      = min(out_grid, 65536);
    init_output_kernel<<<out_grid, block_size, 0, stream>>>(
        total_images, max_detections, d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());

    // 6. NMS
    int nms_grid = (total_images + block_size - 1) / block_size;
    nms_grid     = min(nms_grid, 65536);
    nms_kernel<<<nms_grid, block_size, 0, stream>>>(
        total_images,
        d_candidates,
        d_counts,
        max_candidates,
        iou_thresh,
        max_detections,
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes);
    checkRuntime(cudaPeekAtLastError());

    // 不在这里同步，由调用者通过 cudaEvent 控制，避免阻塞 CPU
}

} // namespace yolo11_obb_postprocess
