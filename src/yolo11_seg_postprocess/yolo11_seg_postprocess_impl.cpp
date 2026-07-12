/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_seg_postprocess/yolo11_seg_postprocess_impl.hpp"
#include <cstdio>

namespace yolo11_seg_postprocess
{

Yolo11SegPostprocess::Yolo11SegPostprocess(const Yolo11SegPostprocessConfig &config)
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

    size_t mask_pixels = static_cast<size_t>(max_batch) * config_.max_detections *
                         config_.proto_height * config_.proto_width;
    detection_masks_workspace_.gpu(mask_pixels);
    mask_offsets_workspace_.gpu(max_batch * config_.max_detections);
    mask_shapes_workspace_.gpu(max_batch * config_.max_detections * 2);
}

void Yolo11SegPostprocess::forward(
    const void *input,
    const void *mask_protos,
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

    size_t mask_pixels = static_cast<size_t>(total_images) * config_.max_detections *
                         config_.proto_height * config_.proto_width;
    float *d_detection_masks = detection_masks_workspace_.gpu(mask_pixels);
    int *d_mask_offsets = mask_offsets_workspace_.gpu(total_images * config_.max_detections);
    int *d_mask_shapes = mask_shapes_workspace_.gpu(total_images * config_.max_detections * 2);

    fprintf(stderr,
            "[yolo11_seg_postprocess] apply_sigmoid=%d conf_thresh=%f iou_thresh=%f "
            "num_masks=%d proto=%dx%d input=%dx%d\n",
            (int)config_.apply_sigmoid, config_.confidence_threshold, config_.iou_threshold,
            config_.num_masks, config_.proto_height, config_.proto_width,
            config_.input_width, config_.input_height);
    fflush(stderr);

    yolo11_seg_postprocess_gpu(
        input,
        mask_protos,
        input_is_half,
        total_images,
        num_anchors,
        config_.num_classes,
        config_.num_masks,
        config_.proto_height,
        config_.proto_width,
        config_.input_width,
        config_.input_height,
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
        d_detection_masks,
        d_mask_offsets,
        d_mask_shapes,
        stream);
}

} // namespace yolo11_seg_postprocess
