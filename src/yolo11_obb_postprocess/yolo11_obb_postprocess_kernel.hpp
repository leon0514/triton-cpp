/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_OBB_POSTPROCESS_KERNEL_HPP__
#define __YOLO11_OBB_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace yolo11_obb_postprocess
{

// 单个 OBB 候选框
struct Candidate
{
    float cx       = 0.0f;
    float cy       = 0.0f;
    float w        = 0.0f;
    float h        = 0.0f;
    float angle    = 0.0f;
    float score    = 0.0f;
    int class_id   = 0;
    int batch_idx  = 0;
};

/**
 * @brief 对 YOLO11-OBB 模型输出执行 decode + confidence 过滤 + NMS。
 *
 * 假设输入已经完成 grid/stride 解码，几何通道直接为 [cx, cy, w, h, angle]。
 * 若模型输出为原始 DFL 距离或归一化坐标，请在外部（如 TensorRT ONNX 解析阶段）
 * 完成解码后再传入本后端。
 *
 * 支持两种输入排布：
 *   - channels_first: [batch, 5+num_classes, num_anchors]
 *   - anchors_first:  [batch, num_anchors, 5+num_classes]
 *
 * 输出框格式为 [cx, cy, w, h, angle]（angle 为弧度）。
 * NMS 使用精确的旋转 IoU（Skew IoU），基于 Sutherland-Hodgman 多边形裁剪。
 *
 * @param input            输入数据指针（device）
 * @param input_is_half    输入是否为 FP16（否则为 FP32）
 * @param total_images     总图像数（所有 request 的 batch_size 之和）
 * @param num_anchors      anchor 数量（如 8400）
 * @param num_classes      类别数（DOTA 为 15）
 * @param anchors_first    true 表示输入排布为 [batch, anchors, 5+classes]
 * @param apply_sigmoid    是否对 class score 再应用一次 sigmoid（模型已做则填 false）
 * @param conf_thresh      置信度阈值
 * @param iou_thresh       NMS IoU 阈值
 * @param max_detections   每张图最多保留的检测框数
 * @param max_candidates   每张图最多进入 NMS 的候选框数
 * @param d_counts         每张图过滤后的候选数（device int[total_images]，输出）
 * @param d_candidates     候选框缓冲区（device Candidate[total_images * max_candidates]，输出）
 * @param d_num_dets       每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes          检测框输出缓冲区（device float[total_images * max_detections * 5]）
 * @param d_scores         分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes        类别输出缓冲区（device int[total_images * max_detections]）
 * @param stream           CUDA 流
 */
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
    cudaStream_t stream);

} // namespace yolo11_obb_postprocess

#endif // __YOLO11_OBB_POSTPROCESS_KERNEL_HPP__
