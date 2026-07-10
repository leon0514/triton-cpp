/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLOV5_TRITON_CONFIG_HPP__
#define __YOLOV5_TRITON_CONFIG_HPP__

#include "yolov5_postprocess/yolov5_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace yolov5_postprocess_backend
{

TRITONSERVER_Error *ParseYolov5PostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolov5_postprocess::Yolov5PostprocessConfig &config);

} // namespace yolov5_postprocess_backend

#endif // __YOLOV5_TRITON_CONFIG_HPP__
