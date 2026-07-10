/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "preprocess/preprocess_kernel.hpp"
#include "common/check.hpp"
#include <cstdint>

namespace preprocess
{

// 双线性插值读取 BGR 像素，返回 BGR 三个通道值
static __device__ void sample_bilinear_bgr(
    const uint8_t *src,
    int src_width,
    int src_height,
    int src_line_size,
    float sx,
    float sy,
    const float *fill_value,
    float &b,
    float &g,
    float &r)
{
    if (sx < -0.5f || sy < -0.5f || sx >= src_width - 0.5f || sy >= src_height - 0.5f)
    {
        b = fill_value[0];
        g = fill_value[1];
        r = fill_value[2];
        return;
    }

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = sx - x0;
    float fy = sy - y0;
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;

    x0 = max(x0, 0);
    y0 = max(y0, 0);
    x1 = min(x1, src_width - 1);
    y1 = min(y1, src_height - 1);

    const uint8_t *row0 = src + y0 * src_line_size;
    const uint8_t *row1 = src + y1 * src_line_size;

    float w00 = fx1 * fy1;
    float w01 = fx * fy1;
    float w10 = fx1 * fy;
    float w11 = fx * fy;

    b = row0[x0 * 3 + 0] * w00 + row0[x1 * 3 + 0] * w01 +
        row1[x0 * 3 + 0] * w10 + row1[x1 * 3 + 0] * w11;

    g = row0[x0 * 3 + 1] * w00 + row0[x1 * 3 + 1] * w01 +
        row1[x0 * 3 + 1] * w10 + row1[x1 * 3 + 1] * w11;

    r = row0[x0 * 3 + 2] * w00 + row0[x1 * 3 + 2] * w01 +
        row1[x0 * 3 + 2] * w10 + row1[x1 * 3 + 2] * w11;
}

// 最近邻插值
static __device__ void sample_nearest_bgr(
    const uint8_t *src,
    int src_width,
    int src_height,
    int src_line_size,
    float sx,
    float sy,
    const float *fill_value,
    float &b,
    float &g,
    float &r)
{
    int x = (int)roundf(sx);
    int y = (int)roundf(sy);

    if (x < 0 || y < 0 || x >= src_width || y >= src_height)
    {
        b = fill_value[0];
        g = fill_value[1];
        r = fill_value[2];
        return;
    }

    const uint8_t *pixel = src + y * src_line_size + x * 3;
    b = pixel[0];
    g = pixel[1];
    r = pixel[2];
}

// 归一化 + 通道交换
static __device__ void normalize_and_swap(
    float b,
    float g,
    float r,
    const float *mean,
    const float *std_inv,
    NormType norm_type,
    float alpha,
    float beta,
    ChannelType channel_type,
    float &out_ch0,
    float &out_ch1,
    float &out_ch2)
{
    float c0 = b;
    float c1 = g;
    float c2 = r;

    if (channel_type == ChannelType::SwapRB)
    {
        c0 = r; // R
        c2 = b; // B
    }

    if (norm_type == NormType::MeanStd)
    {
        out_ch0 = (c0 * alpha - mean[0]) * std_inv[0];
        out_ch1 = (c1 * alpha - mean[1]) * std_inv[1];
        out_ch2 = (c2 * alpha - mean[2]) * std_inv[2];
    }
    else if (norm_type == NormType::AlphaBeta)
    {
        out_ch0 = c0 * alpha + beta;
        out_ch1 = c1 * alpha + beta;
        out_ch2 = c2 * alpha + beta;
    }
    else
    {
        out_ch0 = c0;
        out_ch1 = c1;
        out_ch2 = c2;
    }
}

template <typename _T>
__global__ void warp_affine_batched_kernel(
    const BatchItem *items,
    int dst_width,
    int dst_height,
    _T *dst_batch,
    int dst_area,
    const float *mean,
    const float *std_inv,
    NormType norm_type,
    float alpha,
    float beta,
    ChannelType channel_type,
    InterpMethod interp,
    const float *fill_value)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    int n  = blockIdx.z;

    if (dx >= dst_width || dy >= dst_height)
        return;

    const BatchItem &item = items[n];

    // 使用 d2i 矩阵，将目标坐标反推到源坐标
    const float *d2i = item.d2i;
    float sx = d2i[0] * dx + d2i[1] * dy + d2i[2];
    float sy = d2i[3] * dx + d2i[4] * dy + d2i[5];

    float b = 0, g = 0, r = 0;
    if (interp == InterpMethod::Nearest)
    {
        sample_nearest_bgr(
            item.src_ptr,
            item.src_width,
            item.src_height,
            item.src_line_size,
            sx, sy,
            fill_value,
            b, g, r);
    }
    else
    {
        sample_bilinear_bgr(
            item.src_ptr,
            item.src_width,
            item.src_height,
            item.src_line_size,
            sx, sy,
            fill_value,
            b, g, r);
    }

    float out_ch0, out_ch1, out_ch2;
    normalize_and_swap(
        b, g, r,
        mean, std_inv,
        norm_type, alpha, beta,
        channel_type,
        out_ch0, out_ch1, out_ch2);

    _T *out_n = dst_batch + n * 3 * dst_area;
    int idx   = dy * dst_width + dx;
    out_n[0 * dst_area + idx] = static_cast<_T>(out_ch0);
    out_n[1 * dst_area + idx] = static_cast<_T>(out_ch1);
    out_n[2 * dst_area + idx] = static_cast<_T>(out_ch2);
}

template <typename _T>
void warp_affine_batched(
    int num_images,
    const BatchItem *items,
    int dst_width,
    int dst_height,
    _T *dst_batch,
    int dst_area,
    const float *mean,
    const float *std_inv,
    NormType norm_type,
    float alpha,
    float beta,
    ChannelType channel_type,
    InterpMethod interp,
    const float *fill_value,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (dst_width + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y,
        num_images);

    warp_affine_batched_kernel<<<
        grid, block, 0, stream>>>(
        items,
        dst_width,
        dst_height,
        dst_batch,
        dst_area,
        mean,
        std_inv,
        norm_type,
        alpha,
        beta,
        channel_type,
        interp,
        fill_value);

    checkRuntime(cudaPeekAtLastError());
}

// 显式实例化
template void warp_affine_batched<float>(
    int num_images,
    const BatchItem *items,
    int dst_width,
    int dst_height,
    float *dst_batch,
    int dst_area,
    const float *mean,
    const float *std_inv,
    NormType norm_type,
    float alpha,
    float beta,
    ChannelType channel_type,
    InterpMethod interp,
    const float *fill_value,
    cudaStream_t stream);

template void warp_affine_batched<half>(
    int num_images,
    const BatchItem *items,
    int dst_width,
    int dst_height,
    half *dst_batch,
    int dst_area,
    const float *mean,
    const float *std_inv,
    NormType norm_type,
    float alpha,
    float beta,
    ChannelType channel_type,
    InterpMethod interp,
    const float *fill_value,
    cudaStream_t stream);

void copy_batch_items_to_device(
    const BatchItem *host_items,
    BatchItem *device_items,
    int num_images,
    cudaStream_t stream)
{
    checkRuntime(cudaMemcpyAsync(
        device_items,
        host_items,
        sizeof(BatchItem) * num_images,
        cudaMemcpyHostToDevice,
        stream));
}

} // namespace preprocess
