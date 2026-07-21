/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "sahi_preprocess/slice_impl.hpp"
#include "common/check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace sahi
{

static int calculateNumCuts(int dimension, int subDimension, float overlapRatio)
{
    float step = subDimension * (1.0f - overlapRatio);
    if (step <= 0.0f)
    {
        return 1;
    }

    float cuts = static_cast<float>(dimension - subDimension) / step;
    if (std::fabs(cuts - std::round(cuts)) < 0.0001f)
    {
        cuts = std::round(cuts);
    }

    int numCuts = static_cast<int>(std::ceil(cuts));
    return numCuts + 1;
}

static int calc_resolution_factor(int resolution)
{
    int expo = 0;
    while (std::pow(2, expo) < resolution)
    {
        ++expo;
    }
    return expo - 1;
}

static std::string calc_aspect_ratio_orientation(int width, int height)
{
    if (width < height)
    {
        return "vertical";
    }
    else if (width > height)
    {
        return "horizontal";
    }
    return "square";
}

static std::tuple<int, int, float, float> calc_ratio_and_slice(
    const std::string &orientation,
    int slide = 1,
    float ratio = 0.1f)
{
    int slice_row = 1;
    int slice_col = 1;
    float overlap_height_ratio = 0.0f;
    float overlap_width_ratio = 0.0f;

    if (orientation == "vertical")
    {
        slice_row = slide;
        slice_col = slide * 2;
        overlap_height_ratio = ratio;
        overlap_width_ratio = ratio;
    }
    else if (orientation == "horizontal")
    {
        slice_row = slide * 2;
        slice_col = slide;
        overlap_height_ratio = ratio;
        overlap_width_ratio = ratio;
    }
    else if (orientation == "square")
    {
        slice_row = slide;
        slice_col = slide;
        overlap_height_ratio = ratio;
        overlap_width_ratio = ratio;
    }

    return std::make_tuple(slice_row, slice_col, overlap_height_ratio, overlap_width_ratio);
}

static std::tuple<int, int, float, float> calc_slice_and_overlap_params(
    const std::string &resolution,
    int width,
    int height,
    const std::string &orientation)
{
    int split_row = 1;
    int split_col = 1;
    float overlap_height_ratio = 0.0f;
    float overlap_width_ratio = 0.0f;

    if (resolution == "medium")
    {
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) =
            calc_ratio_and_slice(orientation, 1, 0.8f);
    }
    else if (resolution == "high")
    {
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) =
            calc_ratio_and_slice(orientation, 2, 0.4f);
    }
    else if (resolution == "ultra-high")
    {
        std::tie(split_row, split_col, overlap_height_ratio, overlap_width_ratio) =
            calc_ratio_and_slice(orientation, 4, 0.4f);
    }
    else
    {
        split_col = 1;
        split_row = 1;
        overlap_width_ratio = 0.0f;
        overlap_height_ratio = 0.0f;
    }

    int slice_height = height / split_col;
    int slice_width = width / split_row;

    return std::make_tuple(slice_width, slice_height, overlap_width_ratio, overlap_height_ratio);
}

static std::tuple<int, int, float, float> get_resolution_selector(
    const std::string &resolution,
    int width,
    int height)
{
    std::string orientation = calc_aspect_ratio_orientation(width, height);
    return calc_slice_and_overlap_params(resolution, width, height, orientation);
}

static std::tuple<int, int, float, float> get_auto_slice_params(int width, int height)
{
    int resolution = height * width;
    int factor = calc_resolution_factor(resolution);

    if (factor < 18)
    {
        return get_resolution_selector("low", width, height);
    }
    else if (factor < 21)
    {
        return get_resolution_selector("medium", width, height);
    }
    else if (factor < 24)
    {
        return get_resolution_selector("high", width, height);
    }
    return get_resolution_selector("ultra-high", width, height);
}

SliceImage::SliceImage(const SliceImageConfig &config) : config_(config) {}

const SliceResult &SliceImage::slice(
    const uint8_t *d_image,
    int width,
    int height,
    cudaStream_t stream)
{
    int slice_width = config_.slice_width;
    int slice_height = config_.slice_height;
    float overlap_width_ratio = config_.overlap_width_ratio;
    float overlap_height_ratio = config_.overlap_height_ratio;

    int slice_num_h = calculateNumCuts(width, slice_width, overlap_width_ratio);
    int slice_num_v = calculateNumCuts(height, slice_height, overlap_height_ratio);
    int slice_num = slice_num_h * slice_num_v;

    if (slice_num > config_.max_slices)
    {
        fprintf(stderr,
            "[sahi_preprocess] warning: computed slice_num (%d) exceeds max_slices (%d), "
            "consider increasing max_slices\n",
            slice_num, config_.max_slices);
    }

    printf("[sahi_preprocess] slice_width=%d slice_height=%d "
           "overlap_width_ratio=%f overlap_height_ratio=%f "
           "slice_num_h=%d slice_num_v=%d slice_num=%d\n",
           slice_width, slice_height,
           overlap_width_ratio, overlap_height_ratio,
           slice_num_h, slice_num_v, slice_num);

    size_t output_img_size = 3ULL * slice_width * slice_height;
    size_t total_output_bytes = slice_num * output_img_size;

    // 预分配并复用 workspace
    output_images_.gpu(total_output_bytes);
    d_slice_offsets_.gpu(slice_num * 4);

    // 输入已在 device 上（由 backend 保证），直接使用调用者提供的指针
    const uint8_t *input_device = d_image;

    // 填充输出缓冲区为灰色（114 为 SAHI 默认填充值）
    checkRuntime(cudaMemsetAsync(
        output_images_.gpu(), 114, output_images_.gpu_bytes(), stream));

    // 计算切片起始坐标（host 侧）
    int overlap_width_pixel = static_cast<int>(slice_width * overlap_width_ratio);
    int overlap_height_pixel = static_cast<int>(slice_height * overlap_height_ratio);

    h_slice_offsets_.resize(slice_num * 4);
    for (int i = 0; i < slice_num_h; ++i)
    {
        for (int j = 0; j < slice_num_v; ++j)
        {
            int index = (i * slice_num_v + j);
            if (index >= slice_num)
            {
                break;
            }

            int x = std::min(width - slice_width, std::max(0, i * (slice_width - overlap_width_pixel)));
            int y = std::min(height - slice_height, std::max(0, j * (slice_height - overlap_height_pixel)));

            h_slice_offsets_[index * 4 + 0] = x;
            h_slice_offsets_[index * 4 + 1] = y;
            h_slice_offsets_[index * 4 + 2] = slice_width;
            h_slice_offsets_[index * 4 + 3] = slice_height;
        }
    }

    // 将偏移量拷贝到 device
    checkRuntime(cudaMemcpyAsync(
        d_slice_offsets_.gpu(), h_slice_offsets_.data(),
        slice_num * 4 * sizeof(int),
        cudaMemcpyHostToDevice, stream));

    // 启动 CUDA 切片核函数
    slice_plane(
        input_device,
        output_images_.gpu(),
        d_slice_offsets_.gpu(),
        width, height,
        slice_width, slice_height,
        slice_num_h, slice_num_v,
        stream);

    // 填充结果结构
    result_.slice_num = slice_num;
    result_.slice_width = slice_width;
    result_.slice_height = slice_height;
    result_.d_output_images = output_images_.gpu();
    result_.d_slice_offsets = d_slice_offsets_.gpu();

    return result_;
}

} // namespace sahi
