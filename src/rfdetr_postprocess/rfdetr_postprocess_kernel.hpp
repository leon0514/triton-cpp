/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __RFDETR_POSTPROCESS_KERNEL_HPP__
#define __RFDETR_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>

namespace rfdetr_postprocess
{

// RF-DETR 单个候选框
struct Candidate
{
    float x1      = 0.0f;
    float y1      = 0.0f;
    float x2      = 0.0f;
    float y2      = 0.0f;
    float score   = 0.0f;
    int class_id  = 0;
    int batch_idx = 0;
};

/**
 * @brief 对 RF-DETR 模型输出做后处理。
 *
 * 输入:
 *   - dets:   [batch, num_queries, 4]  归一化 cxcywh 框
 *   - labels: [batch, num_queries, 91] 类别 logits（最后一维为背景）
 *
 * 处理:
 *   - 去掉 labels 最后一维背景
 *   - 对前 90 维做 sigmoid，取最大值作为置信度/类别
 *   - 跳过 COCO 空 ID（12, 26, 29, 30, 45, 66, 68, 69, 71, 83）
 *   - 将归一化 cxcywh 转换为模型输入坐标系下的 xyxy
 *   - 按置信度排序，保留前 max_detections 个
 *
 * @param dets           dets 数据指针（device）
 * @param labels         labels 数据指针（device）
 * @param input_is_half  输入是否为 FP16（否则为 FP32）
 * @param total_images   总图像数
 * @param num_queries    每图 query 数（RF-DETR 通常为 300）
 * @param input_width    模型输入宽度（用于反归一化）
 * @param input_height   模型输入高度（用于反归一化）
 * @param conf_thresh    置信度阈值
 * @param max_detections 每张图最多保留的检测框数
 * @param d_counts       每张图过滤后的候选数（device int[total_images]，中间缓冲区）
 * @param d_candidates   候选框缓冲区（device Candidate[total_images * num_queries]）
 * @param d_num_dets     每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes        检测框输出缓冲区（device float[total_images * max_detections * 4]）
 * @param d_scores       分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes      类别输出缓冲区（device int[total_images * max_detections]）
 * @param stream         CUDA 流
 */
void rfdetr_postprocess_gpu(
    const void *dets,
    const void *labels,
    bool input_is_half,
    int total_images,
    int num_queries,
    float input_width,
    float input_height,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    cudaStream_t stream);

} // namespace rfdetr_postprocess

#endif // __RFDETR_POSTPROCESS_KERNEL_HPP__
