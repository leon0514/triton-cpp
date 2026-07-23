/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __SAHI_SLICE_IMPL_HPP__
#define __SAHI_SLICE_IMPL_HPP__

#include "common/memory.hpp"

#include <cuda_runtime.h>
#include <vector>

namespace sahi
{

// 在 CUDA 文件 slice_kernel.cu 中定义的核函数包装
void slice_plane(
    const uint8_t *image,
    uint8_t *outs,
    const int *slice_offsets,
    int width,
    int height,
    int slice_width,
    int slice_height,
    int slice_num_h,
    int slice_num_v,
    cudaStream_t stream);

// SAHI 切片配置
struct SliceImageConfig
{
    int slice_width  = 640;
    int slice_height = 640;

    float overlap_width_ratio  = 0.2f;
    float overlap_height_ratio = 0.2f;

    // 预分配最大切片数量，避免频繁重新分配 workspace
    int max_slices = 64;

    // 启用自动切片：根据图像分辨率自动确定 slice_width/slice_height/overlap
    // 启用后，手动设置的 slice_width/slice_height 将被忽略
    bool auto_slice = false;
};

// 切片结果描述
struct SliceResult
{
    int slice_num    = 0;
    int slice_width  = 0;
    int slice_height = 0;

    // 所有切片图像拼接后的 device 缓冲区，布局：
    // [slice_num, slice_height, slice_width, 3]
    uint8_t *d_output_images = nullptr;

    // 每个切片在原图中的偏移和尺寸，device 缓冲区，布局：
    // [slice_num, 4] = [x, y, w, h]
    int *d_slice_offsets = nullptr;
};

class SliceImage
{
  public:
    explicit SliceImage(const SliceImageConfig &config);
    ~SliceImage() = default;

    // 禁止拷贝
    SliceImage(const SliceImage &) = delete;
    SliceImage &operator=(const SliceImage &) = delete;

    /**
     * @brief 在 device 图像上执行 SAHI 切片。
     *
     * @param d_image   输入图像 device 指针，布局 [H, W, 3]，BGR HWC
     * @param width     图像宽度
     * @param height    图像高度
     * @param stream    CUDA 流
     * @return 切片结果，结果内指针仅在 SliceImage 对象存活期间有效
     */
    const SliceResult &slice(
        const uint8_t *d_image,
        int width,
        int height,
        cudaStream_t stream);

    inline const SliceImageConfig &config() const { return config_; }

  private:
    SliceImageConfig config_;

    tensor::Memory<uint8_t> output_images_;
    tensor::Memory<int> d_slice_offsets_;

    std::vector<int> h_slice_offsets_;
    SliceResult result_;
};

} // namespace sahi

#endif // __SAHI_SLICE_IMPL_HPP__
