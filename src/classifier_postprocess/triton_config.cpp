/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "classifier_postprocess/triton_config.hpp"
#include "common/json.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace classifier_postprocess_backend
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

TRITONSERVER_Error *ParseClassifierPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    classifier_postprocess::ClassifierPostprocessConfig &config)
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

        std::string num_classes = GetStringParameter(parameters, "num_classes");
        if (!num_classes.empty())
        {
            config.num_classes = std::stoi(num_classes);
        }

        std::string top_k = GetStringParameter(parameters, "top_k");
        if (!top_k.empty())
        {
            config.top_k = std::stoi(top_k);
        }

        std::string apply_softmax = GetStringParameter(parameters, "apply_softmax");
        if (!apply_softmax.empty())
        {
            config.apply_softmax = (apply_softmax == "true" || apply_softmax == "1");
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

        if (config.num_classes <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "num_classes must be positive");
        }
        if (config.top_k <= 0 || config.top_k > config.num_classes)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "top_k must be in [1, num_classes]");
        }
        if (config.max_batch_size <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "max_batch_size must be positive");
        }

        return nullptr;
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, e.what());
    }
}

} // namespace classifier_postprocess_backend
