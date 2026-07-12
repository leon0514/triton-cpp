/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_SEG_TRITON_CONFIG_HPP__
#define __YOLO11_SEG_TRITON_CONFIG_HPP__

#include "yolo11_seg_postprocess/yolo11_seg_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace yolo11_seg_postprocess_backend
{

TRITONSERVER_Error *ParseYolo11SegPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo11_seg_postprocess::Yolo11SegPostprocessConfig &config);

} // namespace yolo11_seg_postprocess_backend

#endif // __YOLO11_SEG_TRITON_CONFIG_HPP__
