/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __RFDETR_POSTPROCESS_IMPL_HPP__
#define __RFDETR_POSTPROCESS_IMPL_HPP__

#include "common/memory.hpp"
#include "rfdetr_postprocess/rfdetr_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace rfdetr_postprocess
{

// 后处理配置
struct RfDetrPostprocessConfig
{
    float confidence_threshold = 0.25f;
    int max_detections         = 300;
    int max_batch_size         = 16;
    float input_width          = 640.0f;
    float input_height         = 640.0f;
};

class RfDetrPostprocess
{
  public:
    explicit RfDetrPostprocess(const RfDetrPostprocessConfig &config);
    ~RfDetrPostprocess() = default;

    RfDetrPostprocess(const RfDetrPostprocess &) = delete;
    RfDetrPostprocess &operator=(const RfDetrPostprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行 RF-DETR 后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace。
     */
    void forward(
        const void *dets,
        const void *labels,
        bool input_is_half,
        int total_images,
        int num_queries,
        cudaStream_t stream);

    inline const RfDetrPostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }

  private:
    RfDetrPostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<rfdetr_postprocess::Candidate> candidates_memory_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;
};

} // namespace rfdetr_postprocess

#endif // __RFDETR_POSTPROCESS_IMPL_HPP__
