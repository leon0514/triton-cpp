/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_OBB_POSTPROCESS_IMPL_HPP__
#define __YOLO26_OBB_POSTPROCESS_IMPL_HPP__

#include "common/memory.hpp"
#include "yolo26_obb_postprocess/yolo26_obb_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace yolo26_obb_postprocess
{

// 后处理配置
struct Yolo26ObbPostprocessConfig
{
    float confidence_threshold = 0.25f;
    int max_detections         = 300;
    int max_batch_size         = 16;
};

class Yolo26ObbPostprocess
{
  public:
    explicit Yolo26ObbPostprocess(const Yolo26ObbPostprocessConfig &config);
    ~Yolo26ObbPostprocess() = default;

    Yolo26ObbPostprocess(const Yolo26ObbPostprocess &) = delete;
    Yolo26ObbPostprocess &operator=(const Yolo26ObbPostprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行 YOLO26-OBB 后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace。
     */
    void forward(
        const void *input,
        bool input_is_half,
        int total_images,
        int num_predictions,
        cudaStream_t stream);

    inline const Yolo26ObbPostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }

  private:
    Yolo26ObbPostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<yolo26_obb_postprocess::Candidate> candidates_memory_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;

    // CUB DeviceSegmentedRadixSort 工作区（避免 thrust + 主机同步）
    tensor::Memory<float> sort_keys_in_workspace_;
    tensor::Memory<float> sort_keys_out_workspace_;
    tensor::Memory<Candidate> sort_candidates_in_workspace_;
    tensor::Memory<Candidate> sort_candidates_out_workspace_;
    tensor::Memory<int> sort_offsets_workspace_;
    tensor::Memory<uint8_t> cub_sort_temp_storage_workspace_;
    size_t cub_sort_temp_storage_bytes_ = 0;
};

} // namespace yolo26_obb_postprocess

#endif // __YOLO26_OBB_POSTPROCESS_IMPL_HPP__
