/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_OBB_POSTPROCESS_TRITON_CONFIG_HPP__
#define __YOLO26_OBB_POSTPROCESS_TRITON_CONFIG_HPP__

#include "yolo26_obb_postprocess/yolo26_obb_postprocess_impl.hpp"
#include <triton/core/tritonserver.h>

namespace yolo26_obb_postprocess_backend
{

TRITONSERVER_Error *ParseYolo26ObbPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo26_obb_postprocess::Yolo26ObbPostprocessConfig &config);

} // namespace yolo26_obb_postprocess_backend

#endif // __YOLO26_OBB_POSTPROCESS_TRITON_CONFIG_HPP__
