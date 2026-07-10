/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "preprocess/preprocess_impl.hpp"
#include "common/check.hpp"

#include <cstring>
#include <algorithm>

namespace preprocess
{

static void compute_d2i_matrix(
    ResizeType resize_type,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    float d2i[6],
    int roi_x = 0,
    int roi_y = 0,
    int roi_w = 0,
    int roi_h = 0)
{
    bool use_roi = (roi_w > 0 && roi_h > 0);

    if (use_roi)
    {
        affine::CropResizeMatrix mat;
        mat.compute(
            {roi_w, roi_h},
            {dst_width, dst_height},
            {roi_x, roi_y});
        memcpy(d2i, mat.d2i, sizeof(mat.d2i));
    }
    else if (resize_type == ResizeType::LetterBox)
    {
        affine::LetterBoxMatrix mat;
        mat.compute({src_width, src_height}, {dst_width, dst_height});
        memcpy(d2i, mat.d2i, sizeof(mat.d2i));
    }
    else
    {
        affine::ResizeMatrix mat;
        mat.compute({src_width, src_height}, {dst_width, dst_height});
        memcpy(d2i, mat.d2i, sizeof(mat.d2i));
    }
}

Preprocess::Preprocess(const PreprocessConfig &config) : config_(config)
{
    // 分配 device 常量内存
    checkRuntime(cudaMalloc(&d_mean_, sizeof(float) * 3));
    checkRuntime(cudaMalloc(&d_std_inv_, sizeof(float) * 3));
    checkRuntime(cudaMalloc(&d_fill_value_, sizeof(float) * 3));

    prepare_device_constants();

    // 根据最大 batch size 预分配 host 侧 BatchItem 工作空间
    int max_batch = config_.max_batch_size > 0 ? config_.max_batch_size : 16;
    host_items_.reserve(max_batch);
}

Preprocess::~Preprocess()
{
    if (d_mean_)
    {
        cudaFree(d_mean_);
        d_mean_ = nullptr;
    }
    if (d_std_inv_)
    {
        cudaFree(d_std_inv_);
        d_std_inv_ = nullptr;
    }
    if (d_fill_value_)
    {
        cudaFree(d_fill_value_);
        d_fill_value_ = nullptr;
    }
}

void Preprocess::prepare_device_constants()
{
    float mean[3]    = {0.0f, 0.0f, 0.0f};
    float std_inv[3] = {1.0f, 1.0f, 1.0f};

    if (config_.norm.type == norm_image::NormType::MeanStd)
    {
        for (int i = 0; i < 3; ++i)
        {
            mean[i]    = config_.norm.mean[i];
            std_inv[i] = 1.0f / config_.norm.std[i];
        }
    }

    checkRuntime(cudaMemcpy(d_mean_, mean, sizeof(float) * 3, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(d_std_inv_, std_inv, sizeof(float) * 3, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(d_fill_value_, config_.fill_value, sizeof(float) * 3, cudaMemcpyHostToDevice));
}

void Preprocess::forward(
    const ImageDesc *images,
    int num_images,
    void *dst_buffer,
    float *transform_buffer,
    cudaStream_t stream,
    BatchItem *external_d_items)
{
    if (num_images <= 0)
        return;

    // 准备 host 侧 BatchItem
    host_items_.resize(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        const ImageDesc &img = images[i];
        BatchItem &item      = host_items_[i];

        item.src_ptr    = img.data;
        item.src_width  = img.width;
        item.src_height = img.height;
        item.src_line_size = img.line_size > 0 ? img.line_size : img.width * 3;

        compute_d2i_matrix(
            config_.resize_type,
            img.width,
            img.height,
            config_.target_width,
            config_.target_height,
            item.d2i,
            img.roi_x,
            img.roi_y,
            img.roi_width,
            img.roi_height);

        // 输出 transform metadata（d2i 矩阵）到 device workspace
        if (transform_buffer != nullptr)
        {
            checkRuntime(cudaMemcpyAsync(
                transform_buffer + i * 6, item.d2i, sizeof(float) * 6,
                cudaMemcpyHostToDevice, stream));
        }
    }

    // 分配 / 复用 device 侧 BatchItem 缓冲区
    BatchItem *d_items = external_d_items != nullptr ? external_d_items : items_memory_.gpu(num_images);
    copy_batch_items_to_device(host_items_.data(), d_items, num_images, stream);

    int dst_area = config_.target_width * config_.target_height;

    // 根据输出类型调用核函数
    if (config_.output_type == OutputType::FP16)
    {
        half *dst = reinterpret_cast<half *>(dst_buffer);
        warp_affine_batched<half>(
            num_images,
            d_items,
            config_.target_width,
            config_.target_height,
            dst,
            dst_area,
            d_mean_,
            d_std_inv_,
            static_cast<NormType>(config_.norm.type),
            config_.norm.alpha,
            config_.norm.beta,
            static_cast<ChannelType>(config_.norm.channel_type),
            config_.interp,
            d_fill_value_,
            stream);
    }
    else
    {
        float *dst = reinterpret_cast<float *>(dst_buffer);
        warp_affine_batched<float>(
            num_images,
            d_items,
            config_.target_width,
            config_.target_height,
            dst,
            dst_area,
            d_mean_,
            d_std_inv_,
            static_cast<NormType>(config_.norm.type),
            config_.norm.alpha,
            config_.norm.beta,
            static_cast<ChannelType>(config_.norm.channel_type),
            config_.interp,
            d_fill_value_,
            stream);
    }
}

} // namespace preprocess
