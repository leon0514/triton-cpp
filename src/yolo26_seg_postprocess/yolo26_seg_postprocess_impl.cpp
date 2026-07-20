/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_seg_postprocess/yolo26_seg_postprocess_impl.hpp"
#include "common/check.hpp"
#include <cstdio>

namespace yolo26_seg_postprocess
{

Yolo26SegPostprocess::Yolo26SegPostprocess(const Yolo26SegPostprocessConfig &config)
    : config_(config)
{
    const int max_batch = config_.max_batch_size;
    const int num_predictions = config_.max_detections;
    const int total_candidates = max_batch * num_predictions;

    counts_memory_.gpu(max_batch);
    candidates_memory_.gpu(total_candidates);

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 4);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);

    size_t mask_pixels = static_cast<size_t>(max_batch) * config_.max_detections *
                         config_.proto_height * config_.proto_width;
    detection_masks_workspace_.gpu(mask_pixels);
    mask_offsets_workspace_.gpu(max_batch * config_.max_detections);
    mask_shapes_workspace_.gpu(max_batch * config_.max_detections * 2);

    sort_keys_in_workspace_.gpu(total_candidates);
    sort_keys_out_workspace_.gpu(total_candidates);
    sort_candidates_in_workspace_.gpu(total_candidates);
    sort_candidates_out_workspace_.gpu(total_candidates);
    sort_offsets_workspace_.gpu(max_batch + 1);

    std::vector<int> h_offsets(max_batch + 1);
    for (int i = 0; i <= max_batch; ++i)
    {
        h_offsets[i] = i * num_predictions;
    }
    cudaMemcpy(sort_offsets_workspace_.gpu(), h_offsets.data(),
               (max_batch + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cub_sort_temp_storage_bytes_ = get_segmented_sort_temp_storage_bytes(
        total_candidates, max_batch);
    if (cub_sort_temp_storage_bytes_ > 0)
    {
        cub_sort_temp_storage_workspace_.gpu(cub_sort_temp_storage_bytes_);
    }
}

void Yolo26SegPostprocess::forward(
    const void *input,
    const void *mask_protos,
    bool input_is_half,
    int total_images,
    int num_predictions,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_predictions <= 0)
        return;

    int *d_counts = counts_memory_.gpu(total_images);
    Candidate *d_cands = candidates_memory_.gpu(total_images * num_predictions);

    int *d_num_dets = num_detections_workspace_.gpu(total_images);
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);

    size_t mask_pixels = static_cast<size_t>(total_images) * config_.max_detections *
                         config_.proto_height * config_.proto_width;
    float *d_detection_masks = detection_masks_workspace_.gpu(mask_pixels);
    int *d_mask_offsets = mask_offsets_workspace_.gpu(total_images * config_.max_detections);
    int *d_mask_shapes = mask_shapes_workspace_.gpu(total_images * config_.max_detections * 2);

    float *d_sort_keys_in = sort_keys_in_workspace_.gpu(total_images * num_predictions);
    float *d_sort_keys_out = sort_keys_out_workspace_.gpu(total_images * num_predictions);
    Candidate *d_sort_candidates_in = sort_candidates_in_workspace_.gpu(total_images * num_predictions);
    Candidate *d_sort_candidates_out = sort_candidates_out_workspace_.gpu(total_images * num_predictions);
    int *d_sort_offsets = sort_offsets_workspace_.gpu(total_images + 1);

    std::vector<int> h_offsets(total_images + 1);
    for (int i = 0; i <= total_images; ++i)
    {
        h_offsets[i] = i * num_predictions;
    }
    checkRuntime(cudaMemcpyAsync(d_sort_offsets, h_offsets.data(),
                                 (total_images + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice, stream));

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

    fprintf(stderr,
            "[yolo26_seg_postprocess] num_masks=%d proto=%dx%d input=%dx%d "
            "conf_thresh=%f max_detections=%d\n",
            config_.num_masks, config_.proto_height, config_.proto_width,
            config_.input_width, config_.input_height,
            config_.confidence_threshold, config_.max_detections);
    fflush(stderr);

    yolo26_seg_postprocess_gpu(
        input,
        mask_protos,
        input_is_half,
        total_images,
        num_predictions,
        config_.num_masks,
        config_.proto_height,
        config_.proto_width,
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
        d_detection_masks,
        d_mask_offsets,
        d_mask_shapes,
        d_sort_keys_in,
        d_sort_keys_out,
        d_sort_candidates_in,
        d_sort_candidates_out,
        d_sort_offsets,
        d_cub_temp,
        cub_sort_temp_storage_bytes_,
        stream);
}

} // namespace yolo26_seg_postprocess
