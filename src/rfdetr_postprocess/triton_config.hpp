/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __RFDETR_TRITON_CONFIG_HPP__
#define __RFDETR_TRITON_CONFIG_HPP__

#include "rfdetr_postprocess/rfdetr_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace rfdetr_postprocess_backend
{

TRITONSERVER_Error *ParseRfDetrPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    rfdetr_postprocess::RfDetrPostprocessConfig &config);

} // namespace rfdetr_postprocess_backend

#endif // __RFDETR_TRITON_CONFIG_HPP__
