/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "sahi_preprocess/slice_impl.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace sahi
{

static __global__ void slice_kernel(
    const uint8_t *__restrict__ image,
    uint8_t *__restrict__ outs,
    const int width,
    const int height,
    const int slice_width,
    const int slice_height,
    const int slice_num_h,
    const int slice_num_v,
    const int *__restrict__ slice_offsets)
{
    const int slice_idx = blockIdx.z;

    const int start_x = slice_offsets[slice_idx * 4 + 0];
    const int start_y = slice_offsets[slice_idx * 4 + 1];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= slice_width || y >= slice_height)
    {
        return;
    }

    const int dx = start_x + x;
    const int dy = start_y + y;

    if (dx >= width || dy >= height)
    {
        return;
    }

    const int src_index = (dy * width + dx) * 3;
    const int dst_index = (slice_idx * slice_width * slice_height + y * slice_width + x) * 3;

    outs[dst_index + 0] = image[src_index + 0];
    outs[dst_index + 1] = image[src_index + 1];
    outs[dst_index + 2] = image[src_index + 2];
}

void slice_plane(
    const uint8_t *image,
    uint8_t *outs,
    const int *slice_offsets,
    const int width,
    const int height,
    const int slice_width,
    const int slice_height,
    const int slice_num_h,
    const int slice_num_v,
    cudaStream_t stream)
{
    const int slice_total = slice_num_h * slice_num_v;

    dim3 block(32, 32);
    dim3 grid(
        (slice_width + block.x - 1) / block.x,
        (slice_height + block.y - 1) / block.y,
        slice_total);

    slice_kernel<<<grid, block, 0, stream>>>(
        image, outs,
        width, height,
        slice_width, slice_height,
        slice_num_h, slice_num_v,
        slice_offsets);
}

} // namespace sahi
