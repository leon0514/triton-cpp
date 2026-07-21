/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_pose_postprocess/yolo11_pose_postprocess_impl.hpp"
#include "common/map_boxes.hpp"
#include <cstdio>

namespace yolo11_pose_postprocess
{

Yolo11PosePostprocess::Yolo11PosePostprocess(const Yolo11PosePostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;
    const int max_candidates = config_.max_candidates;
    const int total_candidates = max_batch * max_candidates;
    const int num_channels = 4 + config_.num_classes +
                             config_.num_keypoints * config_.keypoint_dim;

    counts_memory_.gpu(max_batch);
    keypoints_memory_.gpu(max_batch * config_.max_candidates *
                          config_.num_keypoints * config_.keypoint_dim);

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 4);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);
    keypoints_workspace_.gpu(max_batch * config_.max_detections *
                             config_.num_keypoints * config_.keypoint_dim);

    // CUB 分段排序工作区：一次性申请，推理时复用
    sort_keys_in_workspace_.gpu(total_candidates);
    sort_keys_out_workspace_.gpu(total_candidates);
    sort_candidates_in_workspace_.gpu(total_candidates);
    sort_candidates_out_workspace_.gpu(total_candidates);
    // begin/end 两个偏移数组，具体值每次执行时由 compute_segment_offsets_kernel 填写
    sort_offsets_workspace_.gpu(2 * max_batch);

    // 查询 CUB 临时存储大小（函数实现在 .cu 中，避免在 .cpp 中 include cub）
    cub_sort_temp_storage_bytes_ = get_segmented_sort_temp_storage_bytes(
        total_candidates, max_batch);
    if (cub_sort_temp_storage_bytes_ > 0)
    {
        cub_sort_temp_storage_workspace_.gpu(cub_sort_temp_storage_bytes_);
    }

    // fprintf(stderr,
    //         "[yolo11_pose_postprocess] num_classes=%d num_keypoints=%d keypoint_dim=%d "
    //         "channels=%d apply_sigmoid=%d apply_sigmoid_keypoints=%d conf_thresh=%f iou_thresh=%f\n",
    //         config_.num_classes, config_.num_keypoints, config_.keypoint_dim,
    //         num_channels,
    //         (int)config_.apply_sigmoid, (int)config_.apply_sigmoid_keypoints,
    //         config_.confidence_threshold, config_.iou_threshold);
    // fflush(stderr);
}

void Yolo11PosePostprocess::forward(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_anchors,
    cudaStream_t stream,
    const float *d2i)
{
    if (total_images <= 0 || num_anchors <= 0)
        return;

    int *d_counts = counts_memory_.gpu(total_images);
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

    float *d_sort_keys_in = sort_keys_in_workspace_.gpu(total_images * config_.max_candidates);
    float *d_sort_keys_out = sort_keys_out_workspace_.gpu(total_images * config_.max_candidates);
    Candidate *d_sort_candidates_in = sort_candidates_in_workspace_.gpu(total_images * config_.max_candidates);
    Candidate *d_sort_candidates_out = sort_candidates_out_workspace_.gpu(total_images * config_.max_candidates);
    int *d_sort_begin_offsets = sort_offsets_workspace_.gpu(2 * total_images);
    int *d_sort_end_offsets   = d_sort_begin_offsets + total_images;
    uint8_t *d_cub_temp = cub_sort_temp_storage_bytes_ > 0
                              ? cub_sort_temp_storage_workspace_.gpu()
                              : nullptr;

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
        d_keypoints,
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes,
        d_output_keypoints,
        d_sort_keys_in,
        d_sort_keys_out,
        d_sort_candidates_in,
        d_sort_candidates_out,
        d_sort_begin_offsets,
        d_sort_end_offsets,
        d_cub_temp,
        cub_sort_temp_storage_bytes_,
        stream);

    // 将检测框和关键点从模型输入坐标系映射回原图坐标系
    if (d2i != nullptr)
    {
        map_boxes_to_image(
            d_boxes, d2i, total_images, config_.max_detections, stream);
        map_keypoints_to_image(
            d_output_keypoints, d2i, total_images, config_.max_detections,
            config_.num_keypoints, config_.keypoint_dim, stream);
    }
}

} // namespace yolo11_pose_postprocess
