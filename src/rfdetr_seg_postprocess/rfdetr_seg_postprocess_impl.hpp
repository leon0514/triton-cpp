/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __RFDETR_SEG_POSTPROCESS_IMPL_HPP__
#define __RFDETR_SEG_POSTPROCESS_IMPL_HPP__

#include "common/map_boxes.hpp"
#include "common/memory.hpp"
#include "rfdetr_seg_postprocess/rfdetr_seg_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace rfdetr_seg_postprocess
{

struct RfDetrSegPostprocessConfig
{
    float confidence_threshold = 0.25f;
    int max_detections         = 300;
    int max_batch_size         = 16;
    int num_queries            = 100;
    float input_width          = 384.0f;
    float input_height         = 384.0f;
    bool return_masks          = true;
    int mask_output_resolution       = 160;  // 输出 mask 的宽高（正方形）
    std::vector<int> skip_coco_ids = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83};
};

class RfDetrSegPostprocess
{
  public:
    explicit RfDetrSegPostprocess(const RfDetrSegPostprocessConfig &config);
    ~RfDetrSegPostprocess() = default;

    RfDetrSegPostprocess(const RfDetrSegPostprocess &) = delete;
    RfDetrSegPostprocess &operator=(const RfDetrSegPostprocess &) = delete;

    void forward(
        const void *dets,
        const void *labels,
        const void *masks,
        bool input_is_half,
        int total_images,
        int num_queries,
        int mask_height,
        int mask_width,
        cudaStream_t stream,
        const float *d2i = nullptr);

    inline const RfDetrSegPostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }
    inline const int *coco_id_to_index_gpu() const { return coco_id_to_index_workspace_.gpu(); }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }
    inline int *det_to_query_idx_gpu() const { return det_to_query_idx_workspace_.gpu(); }
    inline float *detection_masks_gpu() const { return detection_masks_workspace_.gpu(); }
    inline int *mask_offsets_gpu() const { return mask_offsets_workspace_.gpu(); }
    inline int *mask_shapes_gpu() const { return mask_shapes_workspace_.gpu(); }

  private:
    RfDetrSegPostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<rfdetr_seg_postprocess::Candidate> candidates_memory_;
    tensor::Memory<int> coco_id_to_index_workspace_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;
    tensor::Memory<int> det_to_query_idx_workspace_;

    tensor::Memory<float> detection_masks_workspace_;
    tensor::Memory<int> mask_offsets_workspace_;
    tensor::Memory<int> mask_shapes_workspace_;

    tensor::Memory<float> sort_keys_in_workspace_;
    tensor::Memory<float> sort_keys_out_workspace_;
    tensor::Memory<Candidate> sort_candidates_in_workspace_;
    tensor::Memory<Candidate> sort_candidates_out_workspace_;
    tensor::Memory<int> sort_offsets_workspace_;
    tensor::Memory<uint8_t> cub_sort_temp_storage_workspace_;
    size_t cub_sort_temp_storage_bytes_ = 0;
};

} // namespace rfdetr_seg_postprocess

#endif // __RFDETR_SEG_POSTPROCESS_IMPL_HPP__
