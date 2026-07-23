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

        // 输出类型
        std::string out_type = GetStringParameter(parameters, "output_type");
        if (out_type == "pose")
            config.output_type = OutputType::POSE;
        else if (out_type == "seg")
            config.output_type = OutputType::SEG;
        else if (out_type == "obb")
            config.output_type = OutputType::OBB;
        else
            config.output_type = OutputType::DET;

        // Box 维度：OBB 模式为 5 (cx,cy,w,h,angle)，其余为 4 (x1,y1,x2,y2)
        if (config.output_type == OutputType::OBB)
            config.box_dim = 5;

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

        std::string num_kpts = GetStringParameter(parameters, "num_keypoints");
        if (!num_kpts.empty())
            config.num_keypoints = std::stoi(num_kpts);

        std::string mask_sz = GetStringParameter(parameters, "mask_output_resolution");
        if (!mask_sz.empty())
            config.mask_output_resolution = std::stoi(mask_sz);

        // SAHI 切片参数完全由 SAHI_PREPROCESS 控制，ensemble 无需配置

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

        // 校验 seg 模式 detection_masks 输出 dims 与 mask_output_resolution 一致
        if (config.output_type == OutputType::SEG)
        {
            const auto &outputs = model_config.value("output", nlohmann::json::array());
            for (const auto &o : outputs)
            {
                if (o.value("name", "") == "detection_masks")
                {
                    const auto &dims = o.value("dims", nlohmann::json::array());
                    if (dims.size() >= 2)
                    {
                        int config_dim = dims[1].get<int>();
                        int expected = config.max_detections * config.mask_output_resolution * config.mask_output_resolution;
                        if (config_dim != expected)
                        {
                            char buf[256];
                            snprintf(buf, sizeof(buf),
                                "detection_masks dims[1] mismatch: config has %d, "
                                "expected max_detections(%d) × mask_output_resolution(%d)² = %d",
                                config_dim, config.max_detections, config.mask_output_resolution, expected);
                            RETURN_TRITON_ERROR(INVALID_ARG, buf);
                        }
                    }
                    break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        RETURN_TRITON_ERROR(INTERNAL, e.what());
    }

    return nullptr;
}

} // namespace sahi_det_ensemble_backend
