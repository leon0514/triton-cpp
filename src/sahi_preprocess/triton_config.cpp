/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "sahi_preprocess/triton_config.hpp"
#include "common/json.hpp"

#include <algorithm>
#include <sstream>
#include <string>

namespace sahi_backend
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

TRITONSERVER_Error *ParseSahiConfig(
    TRITONSERVER_Message *model_config_message,
    sahi::SliceImageConfig &config)
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

        std::string slice_width = GetStringParameter(parameters, "slice_width");
        std::string slice_height = GetStringParameter(parameters, "slice_height");
        std::string overlap_width = GetStringParameter(parameters, "overlap_width_ratio");
        std::string overlap_height = GetStringParameter(parameters, "overlap_height_ratio");
        std::string max_slices = GetStringParameter(parameters, "max_slices");

        if (!slice_width.empty())
        {
            config.slice_width = std::stoi(slice_width);
        }
        if (!slice_height.empty())
        {
            config.slice_height = std::stoi(slice_height);
        }
        if (!overlap_width.empty())
        {
            config.overlap_width_ratio = std::stof(overlap_width);
        }
        if (!overlap_height.empty())
        {
            config.overlap_height_ratio = std::stof(overlap_height);
        }
        if (!max_slices.empty())
        {
            config.max_slices = std::stoi(max_slices);
        }

        std::string auto_slice = GetStringParameter(parameters, "auto_slice");
        if (!auto_slice.empty())
        {
            config.auto_slice = (auto_slice == "true" || auto_slice == "1");
        }

        // auto_slice 模式下 slice_width/height 由图像分辨率自动确定
        if (!config.auto_slice)
        {
            if (config.slice_width <= 0 || config.slice_height <= 0)
            {
                RETURN_TRITON_ERROR(INVALID_ARG, "slice_width and slice_height must be positive");
            }
        }
        if (config.max_slices <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "max_slices must be positive");
        }

        return nullptr;
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, e.what());
    }
}

} // namespace sahi_backend
