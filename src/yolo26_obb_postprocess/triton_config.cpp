/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_obb_postprocess/triton_config.hpp"
#include "common/json.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace yolo26_obb_postprocess_backend
{

#define RETURN_TRITON_ERROR(CODE, MSG) \
    return TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_##CODE, (MSG))

#define RETURN_IF_ERROR(X)                  \
    do                                      \
    {                                       \
        TRITONSERVER_Error *err__ = (X);    \
        if (err__ != nullptr)               \
        {                                   \
            return err__;                   \
        }                                   \
    } while (false)

static std::string GetStringParameter(const nlohmann::json &parameters, const std::string &key)
{
    auto it = parameters.find(key);
    if (it == parameters.end())
    {
        return "";
    }

    const auto &param = it.value();
    if (param.contains("string_value"))
    {
        return param["string_value"].get<std::string>();
    }

    return "";
}

TRITONSERVER_Error *ParseYolo26ObbPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo26_obb_postprocess::Yolo26ObbPostprocessConfig &config)
{
    const char *buffer;
    size_t byte_size;
    RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
        model_config_message, &buffer, &byte_size));

    try
    {
        nlohmann::json model_config = nlohmann::json::parse(
            std::string(buffer, byte_size));

        const auto &parameters = model_config.value("parameters", nlohmann::json::object());

        std::string conf = GetStringParameter(parameters, "confidence_threshold");
        if (!conf.empty())
        {
            config.confidence_threshold = std::stof(conf);
        }

        std::string max_dets = GetStringParameter(parameters, "max_detections");
        if (!max_dets.empty())
        {
            config.max_detections = std::stoi(max_dets);
        }

        std::string max_batch = GetStringParameter(parameters, "max_batch_size");
        if (!max_batch.empty())
        {
            config.max_batch_size = std::stoi(max_batch);
        }
        else
        {
            auto mb_it = model_config.find("max_batch_size");
            if (mb_it != model_config.end() && mb_it->is_number())
            {
                config.max_batch_size = mb_it->get<int>();
            }
        }

        if (config.confidence_threshold < 0.0f || config.confidence_threshold > 1.0f)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "confidence_threshold must be in [0, 1]");
        }
        if (config.max_detections <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "max_detections must be positive");
        }

        return nullptr;
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, e.what());
    }
}

} // namespace yolo26_obb_postprocess_backend
