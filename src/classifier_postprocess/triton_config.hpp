/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __CLASSIFIER_TRITON_CONFIG_HPP__
#define __CLASSIFIER_TRITON_CONFIG_HPP__

#include "classifier_postprocess/classifier_postprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace classifier_postprocess_backend
{

TRITONSERVER_Error *ParseClassifierPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    classifier_postprocess::ClassifierPostprocessConfig &config);

} // namespace classifier_postprocess_backend

#endif // __CLASSIFIER_TRITON_CONFIG_HPP__
