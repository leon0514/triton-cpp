/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __PREPROCESS_KERNEL_HPP__
#define __PREPROCESS_KERNEL_HPP__

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace preprocess
{

// 归一化类型，与 common/norm.hpp 对齐
enum class NormType : int
{
    None      = 0,
    MeanStd   = 1,
    AlphaBeta = 2
};

// 通道交换类型
enum class ChannelType : int
{
    None   = 0,
    SwapRB = 1
};

// 插值方法
enum class InterpMethod : int
{
    Bilinear = 0,
    Nearest  = 1
};

// 每个 batch 项的仿射矩阵与源图信息
struct BatchItem
{
    const uint8_t *src_ptr   = nullptr; // 源图像 GPU 指针
    int src_width            = 0;       // 源图像宽度
    int src_height           = 0;       // 源图像高度
    int src_line_size        = 0;       // 源图像每行字节数（步长）
    float d2i[6];                       // dst -> image 逆变换矩阵（6 floats）
};

/**
 * @brief 在单个 CUDA Kernel 中完成：BGR/RGB resize/letterbox/roi + HWC->NCHW + normalize + fp32/fp16
 *
 * @tparam _T 输出数据类型，支持 float / half
 * @param num_images     batch 内图像数量
 * @param items          每个图像的源指针、d2i 矩阵等信息（device 指针）
 * @param dst_width      目标宽度
 * @param dst_height     目标高度
 * @param dst_batch      输出 NCHW 张量（batch*3*dst_height*dst_width）
 * @param dst_area       dst_width * dst_height
 * @param mean           mean 数组（device 常量指针，长度 3）
 * @param std_inv        1/std 数组（device 常量指针，长度 3）
 * @param norm_type      归一化类型
 * @param alpha          AlphaBeta 模式下的 alpha
 * @param beta           AlphaBeta 模式下的 beta
 * @param channel_type   通道交换类型
 * @param interp         插值方法
 * @param fill_value     letterbox/越界填充色 BGR（device 常量指针，长度 3）
 * @param stream         CUDA 流，必须使用非默认流以支持 Triton 并发
 */
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
    cudaStream_t stream);

// 显式实例化声明
extern template void warp_affine_batched<float>(
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

extern template void warp_affine_batched<half>(
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

/**
 * @brief 将 host 端的 BatchItem 数组异步拷贝到 device
 */
void copy_batch_items_to_device(
    const BatchItem *host_items,
    BatchItem *device_items,
    int num_images,
    cudaStream_t stream);

} // namespace preprocess

#endif // __PREPROCESS_KERNEL_HPP__
