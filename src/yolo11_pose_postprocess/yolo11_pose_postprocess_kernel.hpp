/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_POSE_POSTPROCESS_KERNEL_HPP__
#define __YOLO11_POSE_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace yolo11_pose_postprocess
{

// 单个候选框，关键点位移信息保存在独立的全局显存缓冲区中。
struct Candidate
{
    float x1      = 0.0f;
    float y1      = 0.0f;
    float x2      = 0.0f;
    float y2      = 0.0f;
    float score   = 0.0f;
    int class_id  = 0;
    int batch_idx = 0;
    int kpt_offset = 0;  // 在 d_keypoints 中的偏移，指向该候选框的关键点
};

/**
 * @brief 对 YOLO11-pose 模型原始输出执行 decode + confidence 过滤 + NMS。
 *
 * 支持两种输入排布：
 *   - channels_first: [batch, 4+num_classes+num_keypoints*keypoint_dim, num_anchors]
 *   - anchors_first:  [batch, num_anchors, 4+num_classes+num_keypoints*keypoint_dim]
 *
 * @param input                 输入数据指针（device）
 * @param input_is_half         输入是否为 FP16（否则为 FP32）
 * @param total_images          总图像数（所有 request 的 batch_size 之和）
 * @param num_anchors           anchor 数量（如 8400）
 * @param num_classes           类别数（YOLO11-pose 通常为 1）
 * @param num_keypoints         关键点数量（如 17）
 * @param keypoint_dim          每个关键点的维度（通常为 3：x, y, visibility）
 * @param anchors_first         true 表示输入排布为 [batch, anchors, channels]
 * @param apply_sigmoid         是否对 class score 再应用一次 sigmoid（模型已做则填 false）
 * @param apply_sigmoid_keypoints 是否对关键点 visibility 再应用一次 sigmoid
 * @param conf_thresh           置信度阈值
 * @param iou_thresh            NMS IoU 阈值
 * @param max_detections        每张图最多保留的检测框数
 * @param max_candidates        每张图最多进入 NMS 的候选框数
 * @param d_counts              每张图过滤后的候选数（device int[total_images]，输出）
 * @param d_candidates          候选框缓冲区（device Candidate[total_images * max_candidates]，输出）
 * @param d_keypoints           候选框关键点缓冲区（device float[total_images * max_candidates * num_keypoints * keypoint_dim]，输出）
 * @param d_num_dets            每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes               检测框输出缓冲区（device float[total_images * max_detections * 4]）
 * @param d_scores              分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes             类别输出缓冲区（device int[total_images * max_detections]）
 * @param d_output_keypoints    关键点输出缓冲区（device float[total_images * max_detections * num_keypoints * keypoint_dim]）
 * @param stream                CUDA 流
 */
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
    cudaStream_t stream);

} // namespace yolo11_pose_postprocess

#endif // __YOLO11_POSE_POSTPROCESS_KERNEL_HPP__
