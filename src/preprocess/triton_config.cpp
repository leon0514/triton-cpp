/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "preprocess/triton_config.hpp"
#include "common/json.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace preprocess_backend
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

static std::vector<float> ParseFloatArray(const std::string &text)
{
    std::vector<float> result;
    std::stringstream ss(text);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        token.erase(std::remove(token.begin(), token.end(), '['), token.end());
        token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
        token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
        if (!token.empty())
        {
            result.push_back(std::stof(token));
        }
    }

    return result;
}

static preprocess::ResizeType ParseResizeType(const std::string &value)
{
    if (value == "direct" || value == "resize")
    {
        return preprocess::ResizeType::DirectResize;
    }
    return preprocess::ResizeType::LetterBox;
}

static preprocess::OutputType ParseOutputType(const std::string &value)
{
    if (value == "FP16" || value == "fp16")
    {
        return preprocess::OutputType::FP16;
    }
    return preprocess::OutputType::FP32;
}

static norm_image::ChannelType ParseChannelType(const std::string &value)
{
    if (value == "swap_rb" || value == "rgb" || value == "RGB")
    {
        return norm_image::ChannelType::SwapRB;
    }
    return norm_image::ChannelType::None;
}

static TRITONSERVER_Error *
ParseNormalization(const nlohmann::json &parameters, preprocess::PreprocessConfig &config)
{
    std::string norm_type = GetStringParameter(parameters, "norm_type");

    if (norm_type == "mean_std")
    {
        std::string mean_text = GetStringParameter(parameters, "mean");
        std::string std_text  = GetStringParameter(parameters, "std");

        std::vector<float> mean = ParseFloatArray(mean_text);
        std::vector<float> std  = ParseFloatArray(std_text);

        if (mean.size() != 3 || std.size() != 3)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "mean and std must be arrays of length 3");
        }

        float alpha = 1.0f / 255.0f;
        std::string alpha_text = GetStringParameter(parameters, "alpha");
        if (!alpha_text.empty())
        {
            alpha = std::stof(alpha_text);
        }

        norm_image::ChannelType ct = ParseChannelType(GetStringParameter(parameters, "channel_type"));
        config.norm = norm_image::Norm::mean_std(mean.data(), std.data(), alpha, ct);
    }
    else if (norm_type == "alpha_beta")
    {
        float alpha = 1.0f;
        float beta  = 0.0f;

        std::string alpha_text = GetStringParameter(parameters, "alpha");
        std::string beta_text  = GetStringParameter(parameters, "beta");
        if (!alpha_text.empty())
        {
            alpha = std::stof(alpha_text);
        }
        if (!beta_text.empty())
        {
            beta = std::stof(beta_text);
        }

        norm_image::ChannelType ct = ParseChannelType(GetStringParameter(parameters, "channel_type"));
        config.norm = norm_image::Norm::alpha_beta(alpha, beta, ct);
    }
    else
    {
        config.norm = norm_image::Norm::None();
    }

    return nullptr;
}

TRITONSERVER_Error *ParsePreprocessConfig(
    TRITONSERVER_Message *model_config_message,
    preprocess::PreprocessConfig &config)
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

        config.target_width  = std::stoi(GetStringParameter(parameters, "target_width"));
        config.target_height = std::stoi(GetStringParameter(parameters, "target_height"));

        if (config.target_width <= 0 || config.target_height <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "target_width and target_height must be positive");
        }

        config.resize_type = ParseResizeType(GetStringParameter(parameters, "resize_type"));
        config.output_type = ParseOutputType(GetStringParameter(parameters, "output_type"));

        std::string fill_text = GetStringParameter(parameters, "fill_value");
        if (!fill_text.empty())
        {
            std::vector<float> fill = ParseFloatArray(fill_text);
            if (fill.size() == 3)
            {
                for (int i = 0; i < 3; ++i)
                {
                    config.fill_value[i] = fill[i];
                }
            }
        }

        std::string output_transform = GetStringParameter(parameters, "output_transform");
        config.output_transform = (output_transform == "true" || output_transform == "1");

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

        return ParseNormalization(parameters, config);
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, e.what());
    }
}

} // namespace preprocess_backend
