/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __CLASSIFIER_POSTPROCESS_KERNEL_HPP__
#define __CLASSIFIER_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace classifier_postprocess
{

/**
 * @brief 对分类模型输出执行 softmax + top-k。
 *
 * 输入排布：[total_images, num_classes]。
 * 输出：
 *   - d_classes: [total_images, top_k]，每行前 top_k 的类别索引
 *   - d_scores:  [total_images, top_k]，对应的概率/分数
 *
 * @param input            输入数据指针（device）
 * @param d_probs          softmax 概率中间缓冲区（device，大小 total_images*num_classes）
 * @param d_sort_values    排序用 value 初始值（class index，大小 total_images*num_classes）
 * @param d_sorted_keys    CUB 排序输出 key 缓冲区（大小 total_images*num_classes）
 * @param d_sorted_values  CUB 排序输出 value 缓冲区（大小 total_images*num_classes）
 * @param d_classes        top-k 类别输出缓冲区（device，大小 total_images*top_k）
 * @param d_scores         top-k 分数输出缓冲区（device，大小 total_images*top_k）
 * @param input_is_half    输入是否为 FP16（否则为 FP32）
 * @param total_images     总图像数
 * @param num_classes      类别数
 * @param top_k            每图保留的 top-k 数量
 * @param apply_softmax    是否对输入应用 softmax
 * @param d_sort_offsets   分段排序偏移（device int[total_images + 1]）
 * @param d_cub_temp       CUB 临时存储
 * @param cub_temp_bytes   CUB 临时存储字节数
 * @param stream           CUDA 流
 */
size_t get_segmented_sort_temp_storage_bytes(
    int total_elements, int num_segments);

void classifier_postprocess_gpu(
    const void *input,
    float *d_probs,
    int *d_sort_values,
    float *d_sorted_keys,
    int *d_sorted_values,
    int *d_classes,
    float *d_scores,
    bool input_is_half,
    int total_images,
    int num_classes,
    int top_k,
    bool apply_softmax,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_bytes,
    cudaStream_t stream);

} // namespace classifier_postprocess

#endif // __CLASSIFIER_POSTPROCESS_KERNEL_HPP__
