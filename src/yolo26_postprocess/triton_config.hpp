/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_TRITON_CONFIG_HPP__
#define __YOLO26_TRITON_CONFIG_HPP__

#include "yolo26_postprocess/yolo26_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace yolo26_postprocess_backend
{

TRITONSERVER_Error *ParseYolo26PostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo26_postprocess::Yolo26PostprocessConfig &config);

} // namespace yolo26_postprocess_backend

#endif // __YOLO26_TRITON_CONFIG_HPP__
