/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_postprocess/yolo11_postprocess_impl.hpp"
#include <cstdio>

namespace yolo11_postprocess
{

Yolo11Postprocess::Yolo11Postprocess(const Yolo11PostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;

    counts_memory_.gpu(max_batch);
    candidates_memory_.gpu(max_batch * config_.max_candidates);

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 4);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);
}

void Yolo11Postprocess::forward(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_anchors,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_anchors <= 0)
        return;

    int *d_counts = counts_memory_.gpu(total_images);
    Candidate *d_candidates = candidates_memory_.gpu(total_images * config_.max_candidates);

    int *d_num_dets = num_detections_workspace_.gpu(total_images);
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);

    fprintf(stderr, "[yolo11_postprocess] apply_sigmoid=%d conf_thresh=%f iou_thresh=%f\n",
            (int)config_.apply_sigmoid, config_.confidence_threshold, config_.iou_threshold);
    fflush(stderr);
    yolo11_postprocess_gpu(
        input,
        input_is_half,
        total_images,
        num_anchors,
        config_.num_classes,
        config_.anchors_first,
        config_.apply_sigmoid,
        config_.confidence_threshold,
        config_.iou_threshold,
        config_.max_detections,
        config_.max_candidates,
        d_counts,
        d_candidates,
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes,
        stream);
}

} // namespace yolo11_postprocess
