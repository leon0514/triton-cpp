/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "rfdetr_postprocess/rfdetr_postprocess_impl.hpp"
#include <cstdio>

namespace rfdetr_postprocess
{

RfDetrPostprocess::RfDetrPostprocess(const RfDetrPostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配输出 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 4);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);
}

void RfDetrPostprocess::forward(
    const void *dets,
    const void *labels,
    bool input_is_half,
    int total_images,
    int num_queries,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_queries <= 0)
        return;

    int *d_counts      = counts_memory_.gpu(total_images);
    Candidate *d_cands = candidates_memory_.gpu(total_images * num_queries);

    int *d_num_dets = num_detections_workspace_.gpu(total_images);
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);

    fprintf(stderr, "[rfdetr_postprocess] conf_thresh=%f input_size=%.0fx%.0f max_detections=%d\n",
            config_.confidence_threshold, config_.input_width, config_.input_height, config_.max_detections);
    fflush(stderr);

    rfdetr_postprocess_gpu(
        dets,
        labels,
        input_is_half,
        total_images,
        num_queries,
        config_.input_width,
        config_.input_height,
        config_.confidence_threshold,
        config_.max_detections,
        d_counts,
        d_cands,
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes,
        stream);
}

} // namespace rfdetr_postprocess
