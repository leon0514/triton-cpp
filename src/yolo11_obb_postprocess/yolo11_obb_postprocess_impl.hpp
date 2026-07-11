/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_OBB_POSTPROCESS_IMPL_HPP__
#define __YOLO11_OBB_POSTPROCESS_IMPL_HPP__

#include "common/memory.hpp"
#include "yolo11_obb_postprocess/yolo11_obb_postprocess_kernel.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace yolo11_obb_postprocess
{

// 后处理配置
struct Yolo11ObbPostprocessConfig
{
    int num_classes             = 15;    // DOTA 默认 15 类
    float confidence_threshold  = 0.25f;
    float iou_threshold         = 0.45f;
    int max_detections          = 300;
    int max_candidates          = 1000;
    int max_batch_size          = 16;

    // 模型是否已经对 class score 做过 sigmoid（Ultralytics 默认导出已做）
    bool apply_sigmoid          = false;

    // 模型输出排布：
    // false -> [batch, 5+num_classes, num_anchors]
    // true  -> [batch, num_anchors, 5+num_classes]
    bool anchors_first          = false;
};

class Yolo11ObbPostprocess
{
  public:
    explicit Yolo11ObbPostprocess(const Yolo11ObbPostprocessConfig &config);
    ~Yolo11ObbPostprocess() = default;

    Yolo11ObbPostprocess(const Yolo11ObbPostprocess &) = delete;
    Yolo11ObbPostprocess &operator=(const Yolo11ObbPostprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行 YOLO11-OBB 后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace，调用者通过
     * num_detections_gpu() / boxes_gpu() / scores_gpu() / classes_gpu()
     * 获取结果指针并拷贝到 response buffer。
     *
     * 输出框格式为 [cx, cy, w, h, angle]（angle 为弧度）。
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

    inline const Yolo11ObbPostprocessConfig &config() const { return config_; }
    inline int max_detections() const { return config_.max_detections; }

    inline int *num_detections_gpu() const { return num_detections_workspace_.gpu(); }
    inline float *boxes_gpu() const { return boxes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }
    inline int *classes_gpu() const { return classes_workspace_.gpu(); }

  private:
    Yolo11ObbPostprocessConfig config_;

    tensor::Memory<int> counts_memory_;
    tensor::Memory<yolo11_obb_postprocess::Candidate> candidates_memory_;

    tensor::Memory<int> num_detections_workspace_;
    tensor::Memory<float> boxes_workspace_;
    tensor::Memory<float> scores_workspace_;
    tensor::Memory<int> classes_workspace_;
};

} // namespace yolo11_obb_postprocess

#endif // __YOLO11_OBB_POSTPROCESS_IMPL_HPP__
