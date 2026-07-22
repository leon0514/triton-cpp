/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "sahi_det_ensemble/triton_config.hpp"
#include "common/json.hpp"

#include <string>

namespace sahi_det_ensemble_backend
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
        return "";

    const auto &param = it.value();
    if (param.contains("string_value"))
        return param["string_value"].get<std::string>();

    return "";
}

TRITONSERVER_Error *ParseEnsembleConfig(
    TRITONSERVER_Message *model_config_message,
    EnsembleConfig &config)
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

        // 检测模型名称
        std::string det_model = GetStringParameter(parameters, "detector_model");
        if (!det_model.empty())
            config.detector_model = det_model;

        // 检测参数
        std::string conf = GetStringParameter(parameters, "conf_threshold");
        if (!conf.empty())
            config.confidence_threshold = std::stof(conf);

        std::string iou = GetStringParameter(parameters, "iou_threshold");
        if (!iou.empty())
            config.iou_threshold = std::stof(iou);

        std::string max_dets = GetStringParameter(parameters, "max_detections");
        if (!max_dets.empty())
            config.max_detections = std::stoi(max_dets);

        std::string chunk = GetStringParameter(parameters, "chunk_size");
        if (!chunk.empty())
            config.chunk_size = std::stoi(chunk);

        std::string num_cls = GetStringParameter(parameters, "num_classes");
        if (!num_cls.empty())
            config.num_classes = std::stoi(num_cls);

        // SAHI 切片参数
        std::string sw = GetStringParameter(parameters, "slice_width");
        if (!sw.empty())
            config.slice_width = std::stoi(sw);

        std::string sh = GetStringParameter(parameters, "slice_height");
        if (!sh.empty())
            config.slice_height = std::stoi(sh);

        std::string ow = GetStringParameter(parameters, "overlap_width_ratio");
        if (!ow.empty())
            config.overlap_width_ratio = std::stof(ow);

        std::string oh = GetStringParameter(parameters, "overlap_height_ratio");
        if (!oh.empty())
            config.overlap_height_ratio = std::stof(oh);

        std::string ms = GetStringParameter(parameters, "max_slices");
        if (!ms.empty())
            config.max_slices = std::stoi(ms);

        // 安全检查
        if (config.confidence_threshold < 0.0f || config.confidence_threshold > 1.0f)
            RETURN_TRITON_ERROR(INVALID_ARG, "conf_threshold must be in [0, 1]");
        if (config.iou_threshold < 0.0f || config.iou_threshold > 1.0f)
            RETURN_TRITON_ERROR(INVALID_ARG, "iou_threshold must be in [0, 1]");
        if (config.max_detections <= 0)
            RETURN_TRITON_ERROR(INVALID_ARG, "max_detections must be > 0");
        if (config.num_classes <= 0)
            RETURN_TRITON_ERROR(INVALID_ARG, "num_classes must be > 0");
        if (config.chunk_size <= 0)
            RETURN_TRITON_ERROR(INVALID_ARG, "chunk_size must be > 0");
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INTERNAL, e.what());
    }

    return nullptr;
}

} // namespace sahi_det_ensemble_backend
