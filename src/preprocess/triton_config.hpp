/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __TRITON_CONFIG_HPP__
#define __TRITON_CONFIG_HPP__

#include "preprocess/preprocess_impl.hpp"
#include <triton/core/tritonbackend.h>

namespace preprocess_backend
{

/**
 * @brief 将 Triton 模型配置（JSON 格式）解析为 PreprocessConfig。
 *
 * @param model_config_message TRITONBACKEND_ModelConfig 返回的消息对象
 * @return 解析后的预处理配置
 * @throws TRITONSERVER_Error* 解析失败时返回错误
 */
TRITONSERVER_Error *ParsePreprocessConfig(
    TRITONSERVER_Message *model_config_message,
    preprocess::PreprocessConfig &config);

} // namespace preprocess_backend

#endif // __TRITON_CONFIG_HPP__
