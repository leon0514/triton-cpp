/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_POSE_POSTPROCESS_TRITON_CONFIG_HPP__
#define __YOLO26_POSE_POSTPROCESS_TRITON_CONFIG_HPP__

#include "yolo26_pose_postprocess/yolo26_pose_postprocess_impl.hpp"
#include <triton/core/tritonserver.h>

namespace yolo26_pose_postprocess_backend
{

TRITONSERVER_Error *ParseYolo26PosePostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo26_pose_postprocess::Yolo26PosePostprocessConfig &config);

} // namespace yolo26_pose_postprocess_backend

#endif // __YOLO26_POSE_POSTPROCESS_TRITON_CONFIG_HPP__
