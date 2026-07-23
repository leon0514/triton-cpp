/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "classifier_postprocess/classifier_postprocess_impl.hpp"
#include "common/logging.hpp"
#include <cstdio>

namespace classifier_postprocess
{

ClassifierPostprocess::ClassifierPostprocess(const ClassifierPostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;
    const int num_classes = config_.num_classes;

    probs_workspace_.gpu(max_batch * num_classes);
    sort_values_workspace_.gpu(max_batch * num_classes);
    sorted_keys_workspace_.gpu(max_batch * num_classes);
    sorted_values_workspace_.gpu(max_batch * num_classes);

    classes_workspace_.gpu(max_batch * config_.top_k);
    scores_workspace_.gpu(max_batch * config_.top_k);

    // 分段排序偏移：每段长度为 num_classes
    sort_offsets_workspace_.gpu(max_batch + 1);
    std::vector<int> h_offsets(max_batch + 1);
    for (int i = 0; i <= max_batch; ++i)
    {
        h_offsets[i] = i * num_classes;
    }
    cudaMemcpy(sort_offsets_workspace_.gpu(), h_offsets.data(),
               (max_batch + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // 查询 CUB 临时存储大小
    cub_temp_storage_bytes_ = get_segmented_sort_temp_storage_bytes(
        max_batch * num_classes, max_batch);
    if (cub_temp_storage_bytes_ > 0)
    {
        cub_temp_storage_workspace_.gpu(cub_temp_storage_bytes_);
    }
}

void ClassifierPostprocess::forward(
    const void *input,
    bool input_is_half,
    int total_images,
    cudaStream_t stream)
{
    if (total_images <= 0 || config_.num_classes <= 0)
    {
        return;
    }

    const int num_classes = config_.num_classes;
    const int top_k = config_.top_k;

    float *d_probs = probs_workspace_.gpu(total_images * num_classes);
    int *d_sort_values = sort_values_workspace_.gpu(total_images * num_classes);
    float *d_sorted_keys = sorted_keys_workspace_.gpu(total_images * num_classes);
    int *d_sorted_values = sorted_values_workspace_.gpu(total_images * num_classes);

    int *d_classes = classes_workspace_.gpu(total_images * top_k);
    float *d_scores = scores_workspace_.gpu(total_images * top_k);

    int *d_sort_offsets = sort_offsets_workspace_.gpu(total_images + 1);
    uint8_t *d_cub_temp = cub_temp_storage_bytes_ > 0
                              ? cub_temp_storage_workspace_.gpu()
                              : nullptr;

    LOG_INFO(
            "[classifier_postprocess] num_classes=%d top_k=%d apply_softmax=%d\n",
            num_classes, top_k, (int)config_.apply_softmax);
    fflush(stderr);

    classifier_postprocess_gpu(
        input,
        d_probs,
        d_sort_values,
        d_sorted_keys,
        d_sorted_values,
        d_classes,
        d_scores,
        input_is_half,
        total_images,
        num_classes,
        top_k,
        config_.apply_softmax,
        d_sort_offsets,
        d_cub_temp,
        cub_temp_storage_bytes_,
        stream);
}

} // namespace classifier_postprocess
