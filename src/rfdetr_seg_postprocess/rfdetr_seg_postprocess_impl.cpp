/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "rfdetr_seg_postprocess/rfdetr_seg_postprocess_impl.hpp"
#include "common/map_boxes.hpp"
#include "common/check.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace rfdetr_seg_postprocess
{

        RfDetrSegPostprocess::RfDetrSegPostprocess(const RfDetrSegPostprocessConfig &config)
        : config_(config)
    {
        const int max_batch = config_.max_batch_size;
        const int max_detections = config_.max_detections;

        // COCO ID -> names 文件索引映射
        int host_coco_id_to_index[91];
        for (int i = 0; i <= 90; ++i)
            host_coco_id_to_index[i] = -1;

        int index = 0;
        for (int coco_id = 1; coco_id <= 90; ++coco_id)
        {
            if (std::find(config_.skip_coco_ids.begin(), config_.skip_coco_ids.end(), coco_id) !=
                config_.skip_coco_ids.end())
            {
                continue;
            }
            host_coco_id_to_index[coco_id] = index++;
        }

        coco_id_to_index_workspace_.gpu(91);
        checkRuntime(cudaMemcpy(coco_id_to_index_workspace_.gpu(), host_coco_id_to_index,
                                91 * sizeof(int), cudaMemcpyHostToDevice));

        num_detections_workspace_.gpu(max_batch);
        boxes_workspace_.gpu(max_batch * max_detections * 4);
        scores_workspace_.gpu(max_batch * max_detections);
        classes_workspace_.gpu(max_batch * max_detections);
        det_to_query_idx_workspace_.gpu(max_batch * max_detections);

        // mask workspace 仅预分配一份固定槽位：每检测一个 slot = 160 * 160
        detection_masks_workspace_.gpu(max_batch * max_detections * config_.mask_output_resolution * config_.mask_output_resolution);
        mask_offsets_workspace_.gpu(max_batch * max_detections);
        mask_shapes_workspace_.gpu(max_batch * max_detections * 2);

        counts_memory_.gpu(max_batch);
        candidates_memory_.gpu(max_batch * config_.num_queries);

        const int num_queries = config_.num_queries;
        sort_keys_in_workspace_.gpu(max_batch * num_queries);
        sort_keys_out_workspace_.gpu(max_batch * num_queries);
        sort_candidates_in_workspace_.gpu(max_batch * num_queries);
        sort_candidates_out_workspace_.gpu(max_batch * num_queries);
        sort_offsets_workspace_.gpu(max_batch + 1);

        std::vector<int> h_offsets(max_batch + 1);
        for (int i = 0; i <= max_batch; ++i)
        {
            h_offsets[i] = i * num_queries;
        }
        checkRuntime(cudaMemcpy(sort_offsets_workspace_.gpu(), h_offsets.data(),
                                (max_batch + 1) * sizeof(int), cudaMemcpyHostToDevice));

        cub_sort_temp_storage_bytes_ = get_segmented_sort_temp_storage_bytes(
            max_batch * num_queries, max_batch);
        if (cub_sort_temp_storage_bytes_ > 0)
        {
            cub_sort_temp_storage_workspace_.gpu(cub_sort_temp_storage_bytes_);
        }
    }

    void RfDetrSegPostprocess::forward(
        const void *dets,
        const void *labels,
        const void *masks,
        bool input_is_half,
        int total_images,
        int num_queries,
        int mask_height,
        int mask_width,
        cudaStream_t stream,
        const float *d2i)
    {
        if (total_images <= 0 || num_queries <= 0)
            return;

        int *d_counts = counts_memory_.gpu(total_images);
        Candidate *d_cands = candidates_memory_.gpu(total_images * num_queries);

        int *d_num_dets = num_detections_workspace_.gpu(total_images);
        float *d_boxes = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
        float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
        int *d_classes = classes_workspace_.gpu(total_images * config_.max_detections);
        int *d_det_to_query_idx = det_to_query_idx_workspace_.gpu(total_images * config_.max_detections);

        float *d_detection_masks = nullptr;
        if (config_.return_masks)
        {
            d_detection_masks = detection_masks_workspace_.gpu(
                total_images * config_.max_detections * config_.mask_output_resolution * config_.mask_output_resolution);
        }
        int *d_mask_offsets = mask_offsets_workspace_.gpu(total_images * config_.max_detections);
        int *d_mask_shapes = mask_shapes_workspace_.gpu(total_images * config_.max_detections * 2);

        float *d_sort_keys_in = sort_keys_in_workspace_.gpu(total_images * num_queries);
        float *d_sort_keys_out = sort_keys_out_workspace_.gpu(total_images * num_queries);
        Candidate *d_sort_candidates_in = sort_candidates_in_workspace_.gpu(total_images * num_queries);
        Candidate *d_sort_candidates_out = sort_candidates_out_workspace_.gpu(total_images * num_queries);
        int *d_sort_offsets = sort_offsets_workspace_.gpu(total_images + 1);
        uint8_t *d_cub_temp = cub_sort_temp_storage_bytes_ > 0
                                  ? cub_sort_temp_storage_workspace_.gpu()
                                  : nullptr;

        if (num_queries != config_.num_queries)
        {
            std::vector<int> h_offsets(total_images + 1);
            for (int i = 0; i <= total_images; ++i)
            {
                h_offsets[i] = i * num_queries;
            }
            checkRuntime(cudaMemcpy(d_sort_offsets, h_offsets.data(),
                                    (total_images + 1) * sizeof(int),
                                    cudaMemcpyHostToDevice));
        }

        rfdetr_seg_postprocess_gpu(
            dets,
            labels,
            masks,
            input_is_half,
            total_images,
            num_queries,
            config_.input_width,
            config_.input_height,
            mask_height,
            mask_width,
            config_.mask_output_resolution,
            config_.return_masks,
            config_.confidence_threshold,
            config_.max_detections,
            d_counts,
            d_cands,
            d_num_dets,
            d_boxes,
            d_scores,
            d_classes,
            d_det_to_query_idx,
            d_detection_masks,
            d_mask_offsets,
            d_mask_shapes,
            coco_id_to_index_workspace_.gpu(),
            d_sort_keys_in,
            d_sort_keys_out,
            d_sort_candidates_in,
            d_sort_candidates_out,
            d_sort_offsets,
            d_cub_temp,
            cub_sort_temp_storage_bytes_,
            stream);

        // 将检测框从模型输入坐标系映射回原图坐标系
        if (d2i != nullptr)
        {
            map_boxes_to_image(
                d_boxes, d2i, total_images, config_.max_detections, stream);
        }
    }

} // namespace rfdetr_seg_postprocess
