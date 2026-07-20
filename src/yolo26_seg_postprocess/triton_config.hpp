/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_SEG_POSTPROCESS_TRITON_CONFIG_HPP__
#define __YOLO26_SEG_POSTPROCESS_TRITON_CONFIG_HPP__

#include "yolo26_seg_postprocess/yolo26_seg_postprocess_impl.hpp"
#include <triton/core/tritonserver.h>

namespace yolo26_seg_postprocess_backend
{

TRITONSERVER_Error *ParseYolo26SegPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo26_seg_postprocess::Yolo26SegPostprocessConfig &config);

} // namespace yolo26_seg_postprocess_backend

#endif // __YOLO26_SEG_POSTPROCESS_TRITON_CONFIG_HPP__
