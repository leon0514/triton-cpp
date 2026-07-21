/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __RFDETR_SEG_POSTPROCESS_KERNEL_HPP__
#define __RFDETR_SEG_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace rfdetr_seg_postprocess
{

// 单个候选框，保留对应 query 索引以便后续读取 mask
struct Candidate
{
    float x1       = 0.0f;
    float y1       = 0.0f;
    float x2       = 0.0f;
    float y2       = 0.0f;
    float score    = 0.0f;
    int class_id   = 0;
    int batch_idx  = 0;
    int query_idx  = 0;
};

/**
 * @brief 对 RF-DETR-seg 模型原始输出执行 decode + 置信度过滤 + Top-K + mask 生成。
 *
 * 输入排布：
 *   - dets:   [batch, num_queries, 4]   归一化 cxcywh
 *   - labels: [batch, num_queries, 91]  dim 0 为背景，dim 1..90 为 COCO ID 1~90
 *   - masks:  [batch, num_queries, mask_h, mask_w]  mask logits
 *
 * 输出 mask 说明：
 *   对每个保留的检测框，从对应 query 的模型 mask 中裁剪框内区域，并最近邻采样为固定
 *   160x160，按行优先展开并顺序拼接到 detection_masks 中。每个 mask 在 1D buffer 中的
 *   起始偏移写入 detection_mask_offsets，固定形状 [160, 160] 写入 detection_mask_shapes。
 *   1D buffer 的总容量为 max_detections * 160 * 160。
 */
size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments);

void rfdetr_seg_postprocess_gpu(
    const void *dets,
    const void *labels,
    const void *masks,
    bool input_is_half,
    int total_images,
    int num_queries,
    float input_width,
    float input_height,
    int mask_height,
    int mask_width,
    bool parse_masks,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    int *d_det_to_query_idx,
    float *d_detection_masks,
    int *d_mask_offsets,
    int *d_mask_shapes,
    const int *d_coco_id_to_index,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream);

} // namespace rfdetr_seg_postprocess

#endif // __RFDETR_SEG_POSTPROCESS_KERNEL_HPP__
