/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_POSTPROCESS_IMPL_HPP__
#define __YOLO26_POSTPROCESS_IMPL_HPP__

#include "common/memory.hpp"
#include "yolo26_postprocess/yolo26_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace yolo26_postprocess
{

// 后处理配置
struct Yolo26PostprocessConfig
{
    float confidence_threshold = 0.25f;
    int max_detections         = 300;
    int max_batch_size         = 16;
};

class Yolo26Postprocess
{
  public:
    explicit Yolo26Postprocess(const Yolo26PostprocessConfig &config);
    ~Yolo26Postprocess() = default;

    Yolo26Postprocess(const Yolo26Postprocess &) = delete;
    Yolo26Postprocess &operator=(const Yolo26Postprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行 YOLO26 后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace。
     */
    void forward(
        const void *input,
        bool input_is_half,
        int total_images,
        int num_predictions,
        cudaStream_t stream);

    inline const Yolo26PostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }

  private:
    Yolo26PostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<yolo26_postprocess::Candidate> candidates_memory_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;
};

} // namespace yolo26_postprocess

#endif // __YOLO26_POSTPROCESS_IMPL_HPP__
