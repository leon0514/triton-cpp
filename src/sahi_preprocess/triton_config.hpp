/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __SAHI_TRITON_CONFIG_HPP__
#define __SAHI_TRITON_CONFIG_HPP__

#include "sahi_preprocess/slice_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace sahi_backend
{

TRITONSERVER_Error *ParseSahiConfig(
    TRITONSERVER_Message *model_config_message,
    sahi::SliceImageConfig &config);

} // namespace sahi_backend

#endif // __SAHI_TRITON_CONFIG_HPP__
