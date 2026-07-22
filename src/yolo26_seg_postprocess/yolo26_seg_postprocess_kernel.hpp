/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_SEG_POSTPROCESS_KERNEL_HPP__
#define __YOLO26_SEG_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>

namespace yolo26_seg_postprocess
{

    // YOLO26 one-to-one (NMS-free) seg 头输出的单个候选
    // 输入格式: [batch, num_predictions, 6 + num_masks]
    // 前 6 列: [x1, y1, x2, y2, score, class]
    // 后 num_masks 列: mask 系数
    struct Candidate
    {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        float score = 0.0f;
        int class_id = 0;
        int batch_idx = 0;
        int pred_idx = 0; // 在 output0 中的预测索引，用于读取 mask 系数
    };

    static_assert(sizeof(Candidate) == 32, "Candidate must stay 32-byte aligned");

    /**
     * @brief 对 YOLO26 one-to-one (NMS-free) seg 输出做置信度过滤、top-K 排序与 mask 生成。
     *
     * 输入格式:
     *   - output0: [batch, num_predictions, 6 + num_masks]
     *   - output1: [batch, num_masks, proto_h, proto_w]
     *
     * 输出 mask 说明：
     *   对每个保留的检测框，按 prototype 分辨率计算 mask，将裁剪区域内的 mask 最近邻
     *   采样为固定 160x160 并顺序拼接到 detection_masks 中。每个 mask 在 1D buffer 中的
     *   起始偏移写入 mask_offsets，固定形状 [160, 160] 写入 mask_shapes。
     */
    size_t get_segmented_sort_temp_storage_bytes(
        int total_candidates, int num_segments);

    void yolo26_seg_postprocess_gpu(
        const void *input,
        const void *mask_protos,
        bool input_is_half,
        int total_images,
        int num_predictions,
        int num_masks,
        int proto_h,
        int proto_w,
        int input_width,
        int input_height,
        float conf_thresh,
        int max_detections,
        int *d_counts,
        Candidate *d_candidates,
        int *d_num_dets,
        float *d_boxes,
        float *d_scores,
        int *d_classes,
        float *d_detection_masks,
        int *d_mask_offsets,
        int *d_mask_shapes,
        float *d_sort_keys_in,
        float *d_sort_keys_out,
        Candidate *d_sort_candidates_in,
        Candidate *d_sort_candidates_out,
        int *d_sort_offsets,
        void *d_cub_temp,
        size_t cub_temp_storage_bytes,
        cudaStream_t stream);

    /**
     * @brief 融合 kernel：系数读取 + matmul + crop + sigmoid 一步完成。
     *
     * 在调用本函数前，必须先调用 yolo26_seg_postprocess_gpu 完成过滤 + top-K + 坐标写入。
     *
     * 仅在 crop 区域内逐像素计算 dot(mask_weights, proto[:,y,x])，
     * 无中间 raw_masks buffer，对中小尺寸检测框显著减少无效计算。
     */
    void yolo26_seg_compute_masks_gpu(
        const void *input,
        const void *mask_protos,
        bool input_is_half,
        int total_images,
        int num_predictions,
        int num_masks,
        int proto_h,
        int proto_w,
        int input_width,
        int input_height,
        const int *d_num_dets,
        const float *d_boxes,
        const Candidate *d_candidates,
        int max_detections,
        float *d_detection_masks,
        int *d_mask_offsets,
        int *d_mask_shapes,
        cudaStream_t stream);

} // namespace yolo26_seg_postprocess

#endif // __YOLO26_SEG_POSTPROCESS_KERNEL_HPP__
