/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_pose_postprocess/yolo26_pose_postprocess_impl.hpp"
#include "common/map_boxes.hpp"
#include "yolo26_pose_postprocess/triton_config.hpp"
#include "common/check.hpp"
#include "common/logging.hpp"
#include <cstdio>

namespace yolo26_pose_postprocess
{

Yolo26PosePostprocess::Yolo26PosePostprocess(const Yolo26PosePostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配输出 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;
    const int num_predictions = config_.max_detections;
    const int total_candidates = max_batch * num_predictions;
    const int kpt_elements = num_predictions * config_.num_keypoints * config_.keypoint_dim;

    counts_memory_.gpu(max_batch);
    keypoints_memory_.gpu(max_batch * kpt_elements);
    candidates_memory_.gpu(total_candidates);

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
    sort_offsets_workspace_.gpu(max_batch + 1);

    // 预填分段偏移：每段长度为 num_predictions
    std::vector<int> h_offsets(max_batch + 1);
    for (int i = 0; i <= max_batch; ++i)
    {
        h_offsets[i] = i * num_predictions;
    }
    cudaMemcpy(sort_offsets_workspace_.gpu(), h_offsets.data(),
               (max_batch + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // 查询 CUB 临时存储大小（函数实现在 .cu 中，避免在 .cpp 中 include cub）
    cub_sort_temp_storage_bytes_ = get_segmented_sort_temp_storage_bytes(
        total_candidates, max_batch);
    if (cub_sort_temp_storage_bytes_ > 0)
    {
        cub_sort_temp_storage_workspace_.gpu(cub_sort_temp_storage_bytes_);
    }
}

void Yolo26PosePostprocess::forward(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_predictions,
    cudaStream_t stream,
    const float *d2i)
{
    if (total_images <= 0 || num_predictions <= 0)
        return;

    const int kpt_stride = config_.num_keypoints * config_.keypoint_dim;

    int *d_counts = counts_memory_.gpu(total_images);
    float *d_keypoints = keypoints_memory_.gpu(
        total_images * num_predictions * kpt_stride);
    Candidate *d_cands = candidates_memory_.gpu(total_images * num_predictions);

    int *d_num_dets = num_detections_workspace_.gpu(total_images);
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);
    float *d_output_keypoints = keypoints_workspace_.gpu(
        total_images * config_.max_detections * kpt_stride);

    float *d_sort_keys_in = sort_keys_in_workspace_.gpu(total_images * num_predictions);
    float *d_sort_keys_out = sort_keys_out_workspace_.gpu(total_images * num_predictions);
    Candidate *d_sort_candidates_in = sort_candidates_in_workspace_.gpu(total_images * num_predictions);
    Candidate *d_sort_candidates_out = sort_candidates_out_workspace_.gpu(total_images * num_predictions);
    int *d_sort_offsets = sort_offsets_workspace_.gpu(total_images + 1);

    // 根据实际 num_predictions 填充分段偏移
    std::vector<int> h_offsets(total_images + 1);
    for (int i = 0; i <= total_images; ++i)
    {
        h_offsets[i] = i * num_predictions;
    }
    checkRuntime(cudaMemcpyAsync(d_sort_offsets, h_offsets.data(),
                                 (total_images + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice, stream));

    // 若实际规模超过构造时预分配，重新查询/申请 CUB 临时存储
    const int max_batch = config_.max_batch_size;
    const int total_candidates = total_images * num_predictions;
    const int preallocated_total = max_batch * config_.max_detections;
    if (total_candidates > preallocated_total)
    {
        cub_sort_temp_storage_bytes_ = get_segmented_sort_temp_storage_bytes(
            total_candidates, total_images);
        if (cub_sort_temp_storage_bytes_ > 0)
        {
            cub_sort_temp_storage_workspace_.gpu(cub_sort_temp_storage_bytes_);
        }
    }
    uint8_t *d_cub_temp = cub_sort_temp_storage_bytes_ > 0
                              ? cub_sort_temp_storage_workspace_.gpu()
                              : nullptr;

    LOG_INFO("[yolo26_pose_postprocess] num_keypoints=%d keypoint_dim=%d "
            "conf_thresh=%f max_detections=%d\n",
            config_.num_keypoints, config_.keypoint_dim,
            config_.confidence_threshold, config_.max_detections);
    fflush(stderr);

    yolo26_pose_postprocess_gpu(
        input,
        input_is_half,
        total_images,
        num_predictions,
        config_.num_keypoints,
        config_.keypoint_dim,
        config_.confidence_threshold,
        config_.max_detections,
        d_counts,
        d_keypoints,
        d_cands,
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes,
        d_output_keypoints,
        d_sort_keys_in,
        d_sort_keys_out,
        d_sort_candidates_in,
        d_sort_candidates_out,
        d_sort_offsets,
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

} // namespace yolo26_pose_postprocess
