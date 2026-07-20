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
    const int max_candidates = config_.max_candidates;
    const int total_candidates = max_batch * max_candidates;

    counts_memory_.gpu(max_batch);

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 4);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);

    size_t mask_pixels = static_cast<size_t>(max_batch) * config_.max_detections *
                         config_.proto_height * config_.proto_width;
    detection_masks_workspace_.gpu(mask_pixels);
    mask_offsets_workspace_.gpu(max_batch * config_.max_detections);
    mask_shapes_workspace_.gpu(max_batch * config_.max_detections * 2);

    // NMS 阶段需要的检测->候选映射
    det_to_cand_idx_workspace_.gpu(max_batch * config_.max_detections);

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

    int *d_num_dets = num_detections_workspace_.gpu(total_images);
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);

    size_t mask_pixels = static_cast<size_t>(total_images) * config_.max_detections *
                         config_.proto_height * config_.proto_width;
    float *d_detection_masks = detection_masks_workspace_.gpu(mask_pixels);
    int *d_mask_offsets = mask_offsets_workspace_.gpu(total_images * config_.max_detections);
    int *d_mask_shapes = mask_shapes_workspace_.gpu(total_images * config_.max_detections * 2);

    int *d_det_to_cand_idx = det_to_cand_idx_workspace_.gpu(total_images * config_.max_detections);

    float *d_sort_keys_in = sort_keys_in_workspace_.gpu(total_images * config_.max_candidates);
    float *d_sort_keys_out = sort_keys_out_workspace_.gpu(total_images * config_.max_candidates);
    Candidate *d_sort_candidates_in = sort_candidates_in_workspace_.gpu(total_images * config_.max_candidates);
    Candidate *d_sort_candidates_out = sort_candidates_out_workspace_.gpu(total_images * config_.max_candidates);
    int *d_sort_begin_offsets = sort_offsets_workspace_.gpu(2 * total_images);
    int *d_sort_end_offsets   = d_sort_begin_offsets + total_images;
    uint8_t *d_cub_temp = cub_sort_temp_storage_bytes_ > 0
                              ? cub_sort_temp_storage_workspace_.gpu()
                              : nullptr;

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
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes,
        d_detection_masks,
        d_mask_offsets,
        d_mask_shapes,
        d_det_to_cand_idx,
        d_sort_keys_in,
        d_sort_keys_out,
        d_sort_candidates_in,
        d_sort_candidates_out,
        d_sort_begin_offsets,
        d_sort_end_offsets,
        d_cub_temp,
        cub_sort_temp_storage_bytes_,
        stream);
}

} // namespace yolo11_seg_postprocess
