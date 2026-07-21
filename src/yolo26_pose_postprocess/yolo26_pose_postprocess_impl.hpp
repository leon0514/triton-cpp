/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_POSE_POSTPROCESS_IMPL_HPP__
#define __YOLO26_POSE_POSTPROCESS_IMPL_HPP__

#include "common/map_boxes.hpp"
#include "common/memory.hpp"
#include "yolo26_pose_postprocess/yolo26_pose_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace yolo26_pose_postprocess
{

// 后处理配置
struct Yolo26PosePostprocessConfig
{
    int num_keypoints        = 17;
    int keypoint_dim         = 3;
    float confidence_threshold = 0.25f;
    int max_detections       = 300;
    int max_batch_size       = 16;
};

class Yolo26PosePostprocess
{
  public:
    explicit Yolo26PosePostprocess(const Yolo26PosePostprocessConfig &config);
    ~Yolo26PosePostprocess() = default;

    Yolo26PosePostprocess(const Yolo26PosePostprocess &) = delete;
    Yolo26PosePostprocess &operator=(const Yolo26PosePostprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行 YOLO26-pose 后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace。
     *
     * @param d2i              [total_images, 6] 仿射矩阵（device），可选；非空时
     *                         将检测框和关键点映射回原图坐标系。
     */
    void forward(
        const void *input,
        bool input_is_half,
        int total_images,
        int num_predictions,
        cudaStream_t stream,
        const float *d2i = nullptr);

    inline const Yolo26PosePostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }
    inline int num_keypoints() const { return config_.num_keypoints; }
    inline int keypoint_dim() const { return config_.keypoint_dim; }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }
    inline float *keypoints_gpu() const { return keypoints_workspace_.gpu(); }

  private:
    Yolo26PosePostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<float> keypoints_memory_;
    tensor::Memory<yolo26_pose_postprocess::Candidate> candidates_memory_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;
    tensor::Memory<float> keypoints_workspace_;

    // CUB DeviceSegmentedRadixSort 工作区（避免 thrust + 主机同步）
    tensor::Memory<float> sort_keys_in_workspace_;
    tensor::Memory<float> sort_keys_out_workspace_;
    tensor::Memory<Candidate> sort_candidates_in_workspace_;
    tensor::Memory<Candidate> sort_candidates_out_workspace_;
    tensor::Memory<int> sort_offsets_workspace_;
    tensor::Memory<uint8_t> cub_sort_temp_storage_workspace_;
    size_t cub_sort_temp_storage_bytes_ = 0;
};

} // namespace yolo26_pose_postprocess

#endif // __YOLO26_POSE_POSTPROCESS_IMPL_HPP__
