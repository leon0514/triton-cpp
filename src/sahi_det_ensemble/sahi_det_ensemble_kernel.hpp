/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __SAHI_DET_ENSEMBLE_KERNEL_HPP__
#define __SAHI_DET_ENSEMBLE_KERNEL_HPP__

#include <cuda_runtime.h>

namespace sahi_det_ensemble
{

/**
 * @brief 置信度过滤 + 偏移校正 + 裁剪 一体化 kernel。
 * 输入来自 detector 输出 [num_slices, max_dets, ...]，直接处理并压缩。
 *
 * det_num_dets: [num_slices] 每张切片的检测数
 * det_boxes:     [num_slices, max_dets, 4]
 * det_scores:    [num_slices, max_dets]
 * det_classes:   [num_slices, max_dets]
 * slice_offsets: [num_slices, 4] [ox, oy, w, h]
 *
 * 输出压缩后的有效检测（已偏移+裁剪+过滤）。
 */
void filter_and_offset(
    const int *det_num_dets, const float *det_boxes,
    const float *det_scores, const int *det_classes,
    const int *slice_offsets,
    int num_slices, int max_dets,
    float conf_threshold, int img_w, int img_h,
    float *out_boxes, float *out_scores, int *out_classes,
    int *out_slice_idx, int *d_out_count,
    int max_output, cudaStream_t stream);

/**
 * @brief 逐类 NMS（GPU 版）。
 * 输入已按分数从高到低排列。
 */
void nms_per_class(
    const float *boxes, const float *scores, const int *classes,
    int N, float iou_threshold, int num_classes,
    int *keep, int *d_num_kept,
    int *d_class_offsets, int *d_counters,
    int *d_flags, cudaStream_t stream);

/**
 * @brief 将检测器输出 [n, actual_num_dets, 4] 按行拷贝到 workspace [n, max_dets, 4]。
 * 检测器输出步长 = actual_num_dets * 4，workspace 步长 = max_dets * 4，两个步长不同，
 * 不能用 flat memcpy。此 kernel 逐行 pitched copy。
 */
void strided_copy_boxes(
    const float *src, float *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream);

void strided_copy_scores(
    const float *src, float *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream);

void strided_copy_classes(
    const int *src, int *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream);

void strided_copy_keypoints(
    const float *src, float *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream);

/**
 * @brief Top-K 选择 + 填充到固定维度输出。
 */
void topk_pad(
    const float *boxes, const float *scores, const int *classes,
    int N, int max_detections,
    float *out_boxes, float *out_scores, int *out_classes,
    int *d_num_dets, cudaStream_t stream);

} // namespace sahi_det_ensemble

#endif
