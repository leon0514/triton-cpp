/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __MAP_BOXES_HPP__
#define __MAP_BOXES_HPP__

#include <cuda_runtime.h>

/**
 * @brief 将检测框从预处理后的模型输入坐标系（如 640x640 letterbox）映射回原图坐标系。
 *
 * 使用 d2i（dst -> image）仿射矩阵，每个 batch 图像对应一个 6 元素矩阵。
 *
 * @param boxes          [total_images, max_detections, 4] 检测框，in-place 修改。
 * @param d2i            [total_images, 6] d2i 矩阵，device 指针。
 * @param total_images   图像数量。
 * @param max_detections 每张图最大检测框数。
 * @param stream         CUDA 流。
 */
void map_boxes_to_image(
    float *boxes,
    const float *d2i,
    int total_images,
    int max_detections,
    cudaStream_t stream);

/**
 * @brief 将关键点坐标从预处理后的模型输入坐标系映射回原图坐标系。
 *
 * @param keypoints      [total_images, max_detections, num_keypoints, keypoint_dim] 关键点坐标，in-place 修改前两个元素。
 * @param d2i            [total_images, 6] d2i 矩阵，device 指针。
 * @param total_images   图像数量。
 * @param max_detections 每张图最大检测框数。
 * @param num_keypoints  关键点数量。
 * @param keypoint_dim   每个关键点的维度数（如 2：x,y；3：x,y,visibility）。
 * @param stream         CUDA 流。
 */
void map_keypoints_to_image(
    float *keypoints,
    const float *d2i,
    int total_images,
    int max_detections,
    int num_keypoints,
    int keypoint_dim,
    cudaStream_t stream);

/**
 * @brief 将 OBB 旋转框的顶点坐标从预处理后的模型输入坐标系映射回原图坐标系。
 *
 * @param boxes          [total_images, max_detections, 5] 或 [..., 4+angle] 的 box，只映射 x,y 位置。
 * @param d2i            [total_images, 6] d2i 矩阵，device 指针。
 * @param total_images   图像数量。
 * @param max_detections 每张图最大检测框数。
 * @param box_dims       每个框的维度数（如 5：x,y,w,h,angle）。
 * @param stream         CUDA 流。
 *
 * 注意：只映射前两个元素（中心点 x,y），w/h/angle 保持不变。
 */
void map_obb_to_image(
    float *boxes,
    const float *d2i,
    int total_images,
    int max_detections,
    int box_dims,
    cudaStream_t stream);

#endif // __MAP_BOXES_HPP__
