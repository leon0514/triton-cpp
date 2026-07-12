/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_POSTPROCESS_KERNEL_HPP__
#define __YOLO11_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace yolo11_postprocess
{

// 单个候选框
struct Candidate
{
    float x1       = 0.0f;
    float y1       = 0.0f;
    float x2       = 0.0f;
    float y2       = 0.0f;
    float score    = 0.0f;
    int class_id   = 0;
    int batch_idx  = 0;
};

/**
 * @brief 对 YOLO11 模型原始输出执行 decode + confidence 过滤 + NMS。
 *
 * 支持两种输入排布：
 *   - channels_first: [batch, 4+num_classes, num_anchors]
 *   - anchors_first:  [batch, num_anchors, 4+num_classes]
 *
 * @param input            输入数据指针（device）
 * @param input_is_half    输入是否为 FP16（否则为 FP32）
 * @param total_images     总图像数（所有 request 的 batch_size 之和）
 * @param num_anchors      anchor 数量（如 8400）
 * @param num_classes      类别数（如 80）
 * @param anchors_first    true 表示输入排布为 [batch, anchors, 4+classes]
 * @param apply_sigmoid    是否对 class score 再应用一次 sigmoid（模型已做则填 false）
 * @param conf_thresh      置信度阈值
 * @param iou_thresh       NMS IoU 阈值
 * @param max_detections   每张图最多保留的检测框数
 * @param max_candidates   每张图最多进入 NMS 的候选框数
 * @param d_counts         每张图过滤后的候选数（device int[total_images]，输出）
 * @param d_candidates     候选框缓冲区（device Candidate[total_images * max_candidates]，输出）
 * @param d_num_dets       每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes          检测框输出缓冲区（device float[total_images * max_detections * 4]）
 * @param d_scores         分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes        类别输出缓冲区（device int[total_images * max_detections]）
 * @param d_sort_keys_in   CUB 排序输入 key 缓冲区（device float[total_images * max_candidates]）
 * @param d_sort_keys_out  CUB 排序输出 key 缓冲区（device float[total_images * max_candidates]）
 * @param d_sort_candidates_in  CUB 排序输入 value 缓冲区（device Candidate[total_images * max_candidates]）
 * @param d_sort_candidates_out CUB 排序输出 value 缓冲区（device Candidate[total_images * max_candidates]）
 * @param d_sort_offsets   分段排序偏移（device int[total_images + 1]）
 * @param d_cub_temp       CUB 临时存储（可为 nullptr 当大小为 0）
 * @param cub_temp_storage_bytes CUB 临时存储字节数
 * @param stream           CUDA 流
 */
// 查询 CUB DeviceSegmentedRadixSort 所需临时存储字节数
size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments);

void yolo11_postprocess_gpu(
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
    cudaStream_t stream);

} // namespace yolo11_postprocess

#endif // __YOLO11_POSTPROCESS_KERNEL_HPP__
