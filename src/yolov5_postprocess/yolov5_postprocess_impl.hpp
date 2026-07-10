/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLOV5_POSTPROCESS_IMPL_HPP__
#define __YOLOV5_POSTPROCESS_IMPL_HPP__

#include "common/memory.hpp"
#include "yolov5_postprocess/yolov5_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace yolov5_postprocess
{

// 后处理配置
struct Yolov5PostprocessConfig
{
    int num_classes          = 80;
    float confidence_threshold = 0.25f;
    float iou_threshold      = 0.45f;
    int max_detections       = 300;
    int max_candidates       = 1000;
    int max_batch_size       = 16;

    // 模型是否已经对 objectness / class score 做过 sigmoid。
    // Ultralytics 新版 YOLOv5 导出通常已做 sigmoid，此时填 false；
    // 原始训练输出（logits）则填 true。
    bool apply_sigmoid       = false;

    // 模型输出排布：
    // false -> [batch, C, num_anchors]
    // true  -> [batch, num_anchors, C]
    bool anchors_first = false;

    // YOLOv5 经典导出是否包含 objectness 分支：
    // true  -> C = 5 + num_classes（cx, cy, w, h, obj_conf, cls...）
    // false -> C = 4 + num_classes（与 YOLO11 相同）
    bool has_objectness = true;
};

class Yolov5Postprocess
{
  public:
    explicit Yolov5Postprocess(const Yolov5PostprocessConfig &config);
    ~Yolov5Postprocess() = default;

    Yolov5Postprocess(const Yolov5Postprocess &) = delete;
    Yolov5Postprocess &operator=(const Yolov5Postprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行 YOLOv5 后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace，调用者通过
     * num_detections_gpu() / boxes_gpu() / scores_gpu() / classes_gpu()
     * 获取结果指针并拷贝到 response buffer。
     *
     * @param input            模型输出数据指针（device）
     * @param input_is_half    输入是否为 FP16
     * @param total_images     总图像数（动态 batch 之和）
     * @param num_anchors      anchor 数量
     * @param stream           CUDA 流
     */
    void forward(
        const void *input,
        bool input_is_half,
        int total_images,
        int num_anchors,
        cudaStream_t stream);

    inline const Yolov5PostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }

  private:
    Yolov5PostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<yolov5_postprocess::Candidate> candidates_memory_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;
};

} // namespace yolov5_postprocess

#endif // __YOLOV5_POSTPROCESS_IMPL_HPP__
