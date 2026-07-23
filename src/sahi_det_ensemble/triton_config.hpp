/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __SAHI_DET_ENSEMBLE_TRITON_CONFIG_HPP__
#define __SAHI_DET_ENSEMBLE_TRITON_CONFIG_HPP__

#include <triton/core/tritonbackend.h>
#include <string>

namespace sahi_det_ensemble_backend
{

// SAHI + Detection Ensemble 配置
enum class OutputType { DET, POSE, SEG, OBB };

struct EnsembleConfig
{
    // 输出类型：det | pose | seg | obb
    OutputType output_type = OutputType::DET;

    // 检测模型名称（ensemble 模型，包含 preprocess + inference + postprocess）
    std::string detector_model = "YOLO11_DET_PRE_ENSEMBLE";

    // 检测参数
    float confidence_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_detections = 300;

    // Box 维度：4 = AABB (x1,y1,x2,y2)，5 = OBB (cx,cy,w,h,angle)
    int box_dim = 4;

    // 分块批量大小（一次给 detector 发送的最大切片数）
    int chunk_size = 16;

    // 类别数（用于 NMS 分组）
    int num_classes = 80;

    // Pose 关键点数（仅 pose 模式）
    int num_keypoints = 17;

    // Seg 掩码输出尺寸（仅 seg 模式）
    int mask_output_size = 160;

    // SAHI 切片配置
    int slice_width = 640;
    int slice_height = 640;
    float overlap_width_ratio = 0.2f;
    float overlap_height_ratio = 0.2f;
    int max_slices = 64;
};

TRITONSERVER_Error *ParseEnsembleConfig(
    TRITONSERVER_Message *model_config_message,
    EnsembleConfig &config);

} // namespace sahi_det_ensemble_backend

#endif // __SAHI_DET_ENSEMBLE_TRITON_CONFIG_HPP__
