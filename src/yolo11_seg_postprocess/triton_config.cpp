/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_seg_postprocess/triton_config.hpp"
#include "common/json.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace yolo11_seg_postprocess_backend
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

TRITONSERVER_Error *ParseYolo11SegPostprocessConfig(
    TRITONSERVER_Message *model_config_message,
    yolo11_seg_postprocess::Yolo11SegPostprocessConfig &config)
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

        std::string num_masks = GetStringParameter(parameters, "num_masks");
        if (!num_masks.empty())
        {
            config.num_masks = std::stoi(num_masks);
        }

        std::string proto_height = GetStringParameter(parameters, "proto_height");
        if (!proto_height.empty())
        {
            config.proto_height = std::stoi(proto_height);
        }

        std::string proto_width = GetStringParameter(parameters, "proto_width");
        if (!proto_width.empty())
        {
            config.proto_width = std::stoi(proto_width);
        }

        std::string mask_output_resolution = GetStringParameter(parameters, "mask_output_resolution");
        if (!mask_output_resolution.empty())
        {
            config.mask_output_resolution = std::stoi(mask_output_resolution);
        }
        else
        {
            config.mask_output_resolution = config.proto_height;  // 默认与 proto 尺寸一致
        }

        std::string input_width = GetStringParameter(parameters, "input_width");
        if (!input_width.empty())
        {
            config.input_width = std::stoi(input_width);
        }

        std::string input_height = GetStringParameter(parameters, "input_height");
        if (!input_height.empty())
        {
            config.input_height = std::stoi(input_height);
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

        if (config.num_classes <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "num_classes must be positive");
        }
        if (config.num_masks <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "num_masks must be positive");
        }
        if (config.proto_height <= 0 || config.proto_width <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "proto_height and proto_width must be positive");
        }
        if (config.input_width <= 0 || config.input_height <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "input_width and input_height must be positive");
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

} // namespace yolo11_seg_postprocess_backend
