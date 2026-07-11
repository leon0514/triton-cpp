/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_POSE_TRITON_CONFIG_HPP__
#define __YOLO11_POSE_TRITON_CONFIG_HPP__

#include "yolo11_pose_postprocess/yolo11_pose_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace yolo11_pose_postprocess_backend
{

TRITONSERVER_Error *ParseYolo11PosePostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo11_pose_postprocess::Yolo11PosePostprocessConfig &config);

} // namespace yolo11_pose_postprocess_backend

#endif // __YOLO11_POSE_TRITON_CONFIG_HPP__
