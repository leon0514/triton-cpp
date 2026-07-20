/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_obb_postprocess/yolo11_obb_postprocess_impl.hpp"
#include <cstdio>

namespace yolo11_obb_postprocess
{

Yolo11ObbPostprocess::Yolo11ObbPostprocess(const Yolo11ObbPostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;
    const int max_candidates = config_.max_candidates;
    const int total_candidates = max_batch * max_candidates;

    counts_memory_.gpu(max_batch);
    candidates_memory_.gpu(total_candidates);

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 5);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);

    // CUB 分段排序工作区：一次性申请，推理时复用
    sort_keys_in_workspace_.gpu(total_candidates);
    sort_keys_out_workspace_.gpu(total_candidates);
    sort_candidates_in_workspace_.gpu(total_candidates);
    sort_candidates_out_workspace_.gpu(total_candidates);
    sort_offsets_workspace_.gpu(max_batch + 1);

    // 预填分段偏移：每段长度为 max_candidates
    std::vector<int> h_offsets(max_batch + 1);
    for (int i = 0; i <= max_batch; ++i)
    {
        h_offsets[i] = i * max_candidates;
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

void Yolo11ObbPostprocess::forward(
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
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 5);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);

    float *d_sort_keys_in = sort_keys_in_workspace_.gpu(total_images * config_.max_candidates);
    float *d_sort_keys_out = sort_keys_out_workspace_.gpu(total_images * config_.max_candidates);
    Candidate *d_sort_candidates_in = sort_candidates_in_workspace_.gpu(total_images * config_.max_candidates);
    Candidate *d_sort_candidates_out = sort_candidates_out_workspace_.gpu(total_images * config_.max_candidates);
    int *d_sort_offsets = sort_offsets_workspace_.gpu(total_images + 1);
    uint8_t *d_cub_temp = cub_sort_temp_storage_bytes_ > 0
                              ? cub_sort_temp_storage_workspace_.gpu()
                              : nullptr;

    // fprintf(stderr,
    //         "[yolo11_obb_postprocess] apply_sigmoid=%d conf_thresh=%f iou_thresh=%f "
    //         "num_classes=%d\n",
    //         (int)config_.apply_sigmoid, config_.confidence_threshold,
    //         config_.iou_threshold, config_.num_classes);
    // fflush(stderr);

    yolo11_obb_postprocess_gpu(
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
        d_sort_keys_in,
        d_sort_keys_out,
        d_sort_candidates_in,
        d_sort_candidates_out,
        d_sort_offsets,
        d_cub_temp,
        cub_sort_temp_storage_bytes_,
        stream);
}

} // namespace yolo11_obb_postprocess
