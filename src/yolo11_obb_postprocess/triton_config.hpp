/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_OBB_POSTPROCESS_TRITON_CONFIG_HPP__
#define __YOLO11_OBB_POSTPROCESS_TRITON_CONFIG_HPP__

#include "yolo11_obb_postprocess/yolo11_obb_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace yolo11_obb_postprocess_backend
{

TRITONSERVER_Error *ParseYolo11ObbPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo11_obb_postprocess::Yolo11ObbPostprocessConfig &config);

} // namespace yolo11_obb_postprocess_backend

#endif // __YOLO11_OBB_POSTPROCESS_TRITON_CONFIG_HPP__
