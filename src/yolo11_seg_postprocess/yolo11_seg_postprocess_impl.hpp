/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_SEG_POSTPROCESS_IMPL_HPP__
#define __YOLO11_SEG_POSTPROCESS_IMPL_HPP__

#include "common/memory.hpp"
#include "yolo11_seg_postprocess/yolo11_seg_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace yolo11_seg_postprocess
{

// 后处理配置
struct Yolo11SegPostprocessConfig
{
    int num_classes          = 80;
    int num_masks            = 32;
    int proto_height         = 160;
    int proto_width          = 160;
    int input_width          = 640;
    int input_height         = 640;
    float confidence_threshold = 0.25f;
    float iou_threshold      = 0.45f;
    int max_detections       = 300;
    int max_candidates       = 1000;
    int max_batch_size       = 16;

    // 模型是否已经对 class score 做过 sigmoid（Ultralytics 默认导出已做）
    bool apply_sigmoid       = false;

    // 模型输出排布：
    // false -> [batch, 4+num_classes+num_masks, num_anchors]
    // true  -> [batch, num_anchors, 4+num_classes+num_masks]
    bool anchors_first = false;
};

class Yolo11SegPostprocess
{
  public:
    explicit Yolo11SegPostprocess(const Yolo11SegPostprocessConfig &config);
    ~Yolo11SegPostprocess() = default;

    Yolo11SegPostprocess(const Yolo11SegPostprocess &) = delete;
    Yolo11SegPostprocess &operator=(const Yolo11SegPostprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行 YOLO11-seg 后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace，调用者通过
     * num_detections_gpu() / boxes_gpu() / scores_gpu() / classes_gpu() /
     * detection_masks_gpu() / mask_offsets_gpu() / mask_shapes_gpu()
     * 获取结果指针并拷贝到 response buffer。
     *
     * @param input            模型 output0 数据指针（device）
     * @param mask_protos      模型 output1 prototype mask 数据指针（device）
     * @param input_is_half    输入是否为 FP16
     * @param total_images     总图像数（动态 batch 之和）
     * @param num_anchors      anchor 数量
     * @param stream           CUDA 流
     */
    void forward(
        const void *input,
        const void *mask_protos,
        bool input_is_half,
        int total_images,
        int num_anchors,
        cudaStream_t stream);

    inline const Yolo11SegPostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }
    inline int proto_height() const { return config_.proto_height; }
    inline int proto_width() const { return config_.proto_width; }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }
    inline float *detection_masks_gpu() const { return detection_masks_workspace_.gpu(); }
    inline int *mask_offsets_gpu() const { return mask_offsets_workspace_.gpu(); }
    inline int *mask_shapes_gpu() const { return mask_shapes_workspace_.gpu(); }

  private:
    Yolo11SegPostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<yolo11_seg_postprocess::Candidate> candidates_memory_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;
    tensor::Memory<float> detection_masks_workspace_;
    tensor::Memory<int> mask_offsets_workspace_;
    tensor::Memory<int> mask_shapes_workspace_;
};

} // namespace yolo11_seg_postprocess

#endif // __YOLO11_SEG_POSTPROCESS_IMPL_HPP__
