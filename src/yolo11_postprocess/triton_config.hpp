/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_TRITON_CONFIG_HPP__
#define __YOLO11_TRITON_CONFIG_HPP__

#include "yolo11_postprocess/yolo11_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace yolo11_postprocess_backend
{

TRITONSERVER_Error *ParseYolo11PostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo11_postprocess::Yolo11PostprocessConfig &config);

} // namespace yolo11_postprocess_backend

#endif // __YOLO11_TRITON_CONFIG_HPP__
