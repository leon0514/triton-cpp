/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_pose_postprocess/yolo11_pose_postprocess_impl.hpp"
#include <cstdio>

namespace yolo11_pose_postprocess
{

Yolo11PosePostprocess::Yolo11PosePostprocess(const Yolo11PosePostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;
    const int num_channels = 4 + config_.num_classes +
                             config_.num_keypoints * config_.keypoint_dim;

    counts_memory_.gpu(max_batch);
    candidates_memory_.gpu(max_batch * config_.max_candidates);
    keypoints_memory_.gpu(max_batch * config_.max_candidates *
                          config_.num_keypoints * config_.keypoint_dim);

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 4);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);
    keypoints_workspace_.gpu(max_batch * config_.max_detections *
                             config_.num_keypoints * config_.keypoint_dim);

    fprintf(stderr,
            "[yolo11_pose_postprocess] num_classes=%d num_keypoints=%d keypoint_dim=%d "
            "channels=%d apply_sigmoid=%d apply_sigmoid_keypoints=%d conf_thresh=%f iou_thresh=%f\n",
            config_.num_classes, config_.num_keypoints, config_.keypoint_dim,
            num_channels,
            (int)config_.apply_sigmoid, (int)config_.apply_sigmoid_keypoints,
            config_.confidence_threshold, config_.iou_threshold);
    fflush(stderr);
}

void Yolo11PosePostprocess::forward(
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
    float *d_keypoints = keypoints_memory_.gpu(
        total_images * config_.max_candidates *
        config_.num_keypoints * config_.keypoint_dim);

    int *d_num_dets = num_detections_workspace_.gpu(total_images);
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);
    float *d_output_keypoints = keypoints_workspace_.gpu(
        total_images * config_.max_detections *
        config_.num_keypoints * config_.keypoint_dim);

    yolo11_pose_postprocess_gpu(
        input,
        input_is_half,
        total_images,
        num_anchors,
        config_.num_classes,
        config_.num_keypoints,
        config_.keypoint_dim,
        config_.anchors_first,
        config_.apply_sigmoid,
        config_.apply_sigmoid_keypoints,
        config_.confidence_threshold,
        config_.iou_threshold,
        config_.max_detections,
        config_.max_candidates,
        d_counts,
        d_candidates,
        d_keypoints,
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes,
        d_output_keypoints,
        stream);
}

} // namespace yolo11_pose_postprocess
