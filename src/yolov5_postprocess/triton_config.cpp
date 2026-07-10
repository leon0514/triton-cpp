/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolov5_postprocess/triton_config.hpp"
#include "common/json.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace yolov5_postprocess_backend
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

TRITONSERVER_Error *ParseYolov5PostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolov5_postprocess::Yolov5PostprocessConfig &config)
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

        std::string conf = GetStringParameter(parameters, "confidence_threshold");
        if (!conf.empty())
        {
            config.confidence_threshold = std::stof(conf);
        }

        std::string iou = GetStringParameter(parameters, "iou_threshold");
        if (!iou.empty())
        {
            config.iou_threshold = std::stof(iou);
        }

        std::string max_dets = GetStringParameter(parameters, "max_detections");
        if (!max_dets.empty())
        {
            config.max_detections = std::stoi(max_dets);
        }

        std::string max_cand = GetStringParameter(parameters, "max_candidates");
        if (!max_cand.empty())
        {
            config.max_candidates = std::stoi(max_cand);
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

        std::string fmt = GetStringParameter(parameters, "output_format");
        if (fmt == "anchor_first" || fmt == "anchors_first")
        {
            config.anchors_first = true;
        }
        else
        {
            config.anchors_first = false;
        }

        std::string score_act = GetStringParameter(parameters, "score_activation");
        config.apply_sigmoid = (score_act == "sigmoid");

        std::string obj = GetStringParameter(parameters, "has_objectness");
        if (obj == "false" || obj == "0")
        {
            config.has_objectness = false;
        }
        else if (obj == "true" || obj == "1" || obj.empty())
        {
            // 默认按 YOLOv5 经典格式处理，包含 objectness 分支。
            config.has_objectness = true;
        }
        else
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "has_objectness must be 'true' or 'false'");
        }

        if (config.num_classes <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "num_classes must be positive");
        }
        if (config.confidence_threshold < 0.0f || config.confidence_threshold > 1.0f)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "confidence_threshold must be in [0, 1]");
        }
        if (config.iou_threshold < 0.0f || config.iou_threshold > 1.0f)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "iou_threshold must be in [0, 1]");
        }
        if (config.max_detections <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "max_detections must be positive");
        }
        if (config.max_candidates <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "max_candidates must be positive");
        }

        return nullptr;
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, e.what());
    }
}

} // namespace yolov5_postprocess_backend
