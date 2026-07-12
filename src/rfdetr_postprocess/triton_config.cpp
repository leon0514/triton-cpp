/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "rfdetr_postprocess/triton_config.hpp"
#include "common/json.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace rfdetr_postprocess_backend
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

static std::vector<int> ParseIntArray(const std::string &s)
{
    std::vector<int> result;
    if (s.empty())
        return result;

    // 支持 "[1, 2, 3]" 或 "1,2,3" 或 "[1,2,3]"
    std::string cleaned;
    for (char c : s)
    {
        if (std::isdigit(c) || c == '-' || c == ',' || c == ' ')
            cleaned.push_back(c);
    }

    std::stringstream ss(cleaned);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        std::string trimmed;
        for (char c : token)
        {
            if (!std::isspace(static_cast<unsigned char>(c)))
                trimmed.push_back(c);
        }
        if (!trimmed.empty())
            result.push_back(std::stoi(trimmed));
    }
    return result;
}

TRITONSERVER_Error *ParseRfDetrPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    rfdetr_postprocess::RfDetrPostprocessConfig &config)
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

        std::string num_queries = GetStringParameter(parameters, "num_queries");
        if (!num_queries.empty())
        {
            config.num_queries = std::stoi(num_queries);
        }

        std::string input_width = GetStringParameter(parameters, "input_width");
        if (!input_width.empty())
        {
            config.input_width = std::stof(input_width);
        }

        std::string input_height = GetStringParameter(parameters, "input_height");
        if (!input_height.empty())
        {
            config.input_height = std::stof(input_height);
        }

        std::string skip_ids = GetStringParameter(parameters, "skip_coco_ids");
        if (!skip_ids.empty())
        {
            config.skip_coco_ids = ParseIntArray(skip_ids);
        }

        if (config.confidence_threshold < 0.0f || config.confidence_threshold > 1.0f)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "confidence_threshold must be in [0, 1]");
        }
        if (config.max_detections <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "max_detections must be positive");
        }
        if (config.input_width <= 0.0f || config.input_height <= 0.0f)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "input_width and input_height must be positive");
        }
        if (config.num_queries <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "num_queries must be positive");
        }

        return nullptr;
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, e.what());
    }
}

} // namespace rfdetr_postprocess_backend
