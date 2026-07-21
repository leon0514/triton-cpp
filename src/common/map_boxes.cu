/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "common/map_boxes.hpp"

#include <cuda_runtime.h>

namespace
{

static __global__ void map_boxes_kernel(
    float *boxes,
    const float *d2i,
    int total_images,
    int max_detections)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_images * max_detections;
    if (idx >= total)
    {
        return;
    }

    const int img_idx = idx / max_detections;
    const float *m = d2i + img_idx * 6;
    float *box = boxes + idx * 4;

    const float x1 = box[0];
    const float y1 = box[1];
    const float x2 = box[2];
    const float y2 = box[3];

    box[0] = m[0] * x1 + m[1] * y1 + m[2];
    box[1] = m[3] * x1 + m[4] * y1 + m[5];
    box[2] = m[0] * x2 + m[1] * y2 + m[2];
    box[3] = m[3] * x2 + m[4] * y2 + m[5];
}

static __global__ void map_keypoints_kernel(
    float *keypoints,
    const float *d2i,
    int total_images,
    int max_detections,
    int num_keypoints,
    int keypoint_dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_images * max_detections * num_keypoints;
    if (idx >= total)
    {
        return;
    }

    const int img_idx = idx / (max_detections * num_keypoints);
    const float *m = d2i + img_idx * 6;
    float *kp = keypoints + idx * keypoint_dim;

    const float x = kp[0];
    const float y = kp[1];
    kp[0] = m[0] * x + m[1] * y + m[2];
    kp[1] = m[3] * x + m[4] * y + m[5];
}

static __global__ void map_obb_kernel(
    float *boxes,
    const float *d2i,
    int total_images,
    int max_detections,
    int box_dims)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = total_images * max_detections;
    if (idx >= total)
    {
        return;
    }

    const int img_idx = idx / max_detections;
    const float *m = d2i + img_idx * 6;
    float *box = boxes + idx * box_dims;

    const float x = box[0];
    const float y = box[1];
    box[0] = m[0] * x + m[1] * y + m[2];
    box[1] = m[3] * x + m[4] * y + m[5];
}

} // namespace

void map_boxes_to_image(
    float *boxes,
    const float *d2i,
    int total_images,
    int max_detections,
    cudaStream_t stream)
{
    if (total_images <= 0 || max_detections <= 0)
    {
        return;
    }
    const int total = total_images * max_detections;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    map_boxes_kernel<<<grid, block, 0, stream>>>(boxes, d2i, total_images, max_detections);
}

void map_keypoints_to_image(
    float *keypoints,
    const float *d2i,
    int total_images,
    int max_detections,
    int num_keypoints,
    int keypoint_dim,
    cudaStream_t stream)
{
    if (total_images <= 0 || max_detections <= 0 || num_keypoints <= 0 || keypoint_dim <= 0)
    {
        return;
    }
    const int total = total_images * max_detections * num_keypoints;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    map_keypoints_kernel<<<grid, block, 0, stream>>>(
        keypoints, d2i, total_images, max_detections, num_keypoints, keypoint_dim);
}

void map_obb_to_image(
    float *boxes,
    const float *d2i,
    int total_images,
    int max_detections,
    int box_dims,
    cudaStream_t stream)
{
    if (total_images <= 0 || max_detections <= 0 || box_dims <= 0)
    {
        return;
    }
    const int total = total_images * max_detections;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    map_obb_kernel<<<grid, block, 0, stream>>>(boxes, d2i, total_images, max_detections, box_dims);
}