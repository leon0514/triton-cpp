/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __RFDETR_SEG_TRITON_CONFIG_HPP__
#define __RFDETR_SEG_TRITON_CONFIG_HPP__

#include "rfdetr_seg_postprocess/rfdetr_seg_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace rfdetr_seg_postprocess_backend
{

TRITONSERVER_Error *ParseRfDetrSegPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    rfdetr_seg_postprocess::RfDetrSegPostprocessConfig &config);

} // namespace rfdetr_seg_postprocess_backend

#endif // __RFDETR_SEG_TRITON_CONFIG_HPP__
