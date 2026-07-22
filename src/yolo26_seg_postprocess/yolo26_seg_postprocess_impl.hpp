/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_SEG_POSTPROCESS_IMPL_HPP__
#define __YOLO26_SEG_POSTPROCESS_IMPL_HPP__

#include "common/map_boxes.hpp"
#include "common/memory.hpp"
#include "yolo26_seg_postprocess/yolo26_seg_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace yolo26_seg_postprocess
{

    // 后处理配置
    struct Yolo26SegPostprocessConfig
    {
        int num_masks = 32;
        int proto_height = 160;
        int proto_width = 160;
        int input_width = 640;
        int input_height = 640;
        float confidence_threshold = 0.25f;
        int max_detections = 300;
        int max_batch_size = 16;
    };

    class Yolo26SegPostprocess
    {
    public:
        explicit Yolo26SegPostprocess(const Yolo26SegPostprocessConfig &config);
        ~Yolo26SegPostprocess();

        Yolo26SegPostprocess(const Yolo26SegPostprocess &) = delete;
        Yolo26SegPostprocess &operator=(const Yolo26SegPostprocess &) = delete;

        /**
         * @brief 在指定 CUDA 流上执行 YOLO26-seg 后处理。
         *
         * 输出缓冲区为实例初始化时预分配的 GPU workspace。
         */
        void forward(
            const void *input,
            const void *mask_protos,
            bool input_is_half,
            int total_images,
            int num_predictions,
            cudaStream_t stream,
            const float *d2i = nullptr);

        inline const Yolo26SegPostprocessConfig &config() const { return config_; }
        inline int max_detections() const { return config_.max_detections; }
        inline int proto_height() const { return config_.proto_height; }
        inline int proto_width() const { return config_.proto_width; }

        inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
        inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
        inline float *scores_gpu() const { return scores_workspace_.gpu(); }
        inline int *classes_gpu() const { return classes_workspace_.gpu(); }
        inline float *masks_gpu() const { return detection_masks_workspace_.gpu(); }
        inline int *mask_offsets_gpu() const { return mask_offsets_workspace_.gpu(); }
        inline int *mask_shapes_gpu() const { return mask_shapes_workspace_.gpu(); }

    private:
        Yolo26SegPostprocessConfig config_;

        tensor::Memory<int> counts_memory_;
        tensor::Memory<yolo26_seg_postprocess::Candidate> candidates_memory_;

        tensor::Memory<int> num_detections_workspace_;

        tensor::Memory<float> boxes_workspace_;
        tensor::Memory<float> scores_workspace_;
        tensor::Memory<int> classes_workspace_;

        tensor::Memory<float> detection_masks_workspace_;
        tensor::Memory<int> mask_offsets_workspace_;
        tensor::Memory<int> mask_shapes_workspace_;

        // CUB DeviceSegmentedRadixSort 工作区
        tensor::Memory<float> sort_keys_in_workspace_;
        tensor::Memory<float> sort_keys_out_workspace_;
        tensor::Memory<Candidate> sort_candidates_in_workspace_;
        tensor::Memory<Candidate> sort_candidates_out_workspace_;
        tensor::Memory<int> sort_offsets_workspace_;
        tensor::Memory<uint8_t> cub_sort_temp_storage_workspace_;
        size_t cub_sort_temp_storage_bytes_ = 0;
    };

} // namespace yolo26_seg_postprocess

#endif // __YOLO26_SEG_POSTPROCESS_IMPL_HPP__
