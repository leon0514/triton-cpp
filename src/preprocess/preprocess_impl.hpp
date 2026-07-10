/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __PREPROCESS_IMPL_HPP__
#define __PREPROCESS_IMPL_HPP__

#include "common/affine.hpp"
#include "common/memory.hpp"
#include "common/norm.hpp"
#include "preprocess/preprocess_kernel.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <memory>
#include <string>
#include <vector>

namespace preprocess
{

// 支持的 resize 方式
enum class ResizeType : int
{
    DirectResize = 0,
    LetterBox    = 1
};

// 输出数据类型
enum class OutputType : int
{
    FP32 = 0,
    FP16 = 1
};

// 单张输入图像的描述
struct ImageDesc
{
    const uint8_t *data        = nullptr; // 输入图像数据指针（GPU 或 CPU，由调用者保证最终为 GPU）
    int width                  = 0;
    int height                 = 0;
    int line_size              = 0;       // 每行字节数，0 表示 width*3

    // ROI 裁剪（可选），width/height 均大于 0 时生效
    int roi_x                  = 0;
    int roi_y                  = 0;
    int roi_width              = 0;
    int roi_height             = 0;
};

// 预处理配置
struct PreprocessConfig
{
    int target_width  = 640;
    int target_height = 640;

    ResizeType resize_type = ResizeType::LetterBox;
    OutputType output_type = OutputType::FP32;
    InterpMethod interp    = InterpMethod::Bilinear;

    norm_image::Norm norm;

    // letterbox / 越界填充色，BGR 顺序
    float fill_value[3] = {114.0f, 114.0f, 114.0f};

    // 是否输出 transform metadata（每个图像 6 个 float：d2i 矩阵）
    bool output_transform = true;

    // 最大 batch size，用于预分配工作空间
    int max_batch_size = 16;
};

class Preprocess
{
  public:
    Preprocess(const PreprocessConfig &config);
    ~Preprocess();

    // 禁止拷贝
    Preprocess(const Preprocess &) = delete;
    Preprocess &operator=(const Preprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行预处理
     *
     * @param images       输入图像数组（host 侧描述，data 必须为 device 指针）
     * @param num_images   图像数量（动态 batch）
     * @param dst_buffer   输出 GPU 缓冲区，大小需 >= num_images*3*H*W*sizeof(dtype)
     * @param transform_buffer 可选，输出每个图像的 d2i 矩阵，大小需 >= num_images*6*sizeof(float)
     * @param stream       CUDA 流
     */
    void forward(
        const ImageDesc *images,
        int num_images,
        void *dst_buffer,
        float *transform_buffer,
        cudaStream_t stream,
        BatchItem *external_d_items = nullptr);

    inline const PreprocessConfig &config() const { return config_; }
    inline int target_width() const { return config_.target_width; }
    inline int target_height() const { return config_.target_height; }
    inline int dst_area() const { return config_.target_width * config_.target_height; }
    inline OutputType output_type() const { return config_.output_type; }
    inline size_t output_bytes_per_image() const
    {
        return 3 * dst_area() * (config_.output_type == OutputType::FP16 ? sizeof(half) : sizeof(float));
    }

  private:
    void prepare_device_constants();

    PreprocessConfig config_;

    // device 侧常量内存（暂用 global memory）
    float *d_mean_        = nullptr;
    float *d_std_inv_     = nullptr;
    float *d_fill_value_  = nullptr;

    // host 侧工作缓冲区
    std::vector<BatchItem> host_items_;

    // device 侧 BatchItem 缓冲区
    tensor::Memory<BatchItem> items_memory_;

    // device 侧临时输出 buffer（当调用者未提供足够大的 dst_buffer 时使用，实际建议由 Triton 分配）
    tensor::Memory<uint8_t> workspace_;
};

} // namespace preprocess

#endif // __PREPROCESS_IMPL_HPP__
