/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "classifier_postprocess/classifier_postprocess_impl.hpp"
#include "classifier_postprocess/triton_config.hpp"
#include "common/device.hpp"
#include "common/memory.hpp"

#include <triton/core/tritonbackend.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#define BACKEND_NAME "classifier_postprocess"

namespace classifier_postprocess_backend
{

#define RETURN_IF_ERROR(X)                  \
    do                                      \
    {                                       \
        TRITONSERVER_Error *err__ = (X);    \
        if (err__ != nullptr)               \
        {                                   \
            return err__;                   \
        }                                   \
    } while (false)

#define RETURN_TRITON_ERROR(CODE, MSG) \
    return TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_##CODE, (MSG))

// ===================== State Management =====================

struct ModelState
{
    TRITONBACKEND_Model *triton_model = nullptr;
    classifier_postprocess::ClassifierPostprocessConfig config;

    explicit ModelState(TRITONBACKEND_Model *model) : triton_model(model) {}

    TRITONSERVER_Error *LoadConfig()
    {
        TRITONSERVER_Message *config_message;
        RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));
        TRITONSERVER_Error *err = ParseClassifierPostprocessConfig(config_message, config);
        TRITONSERVER_MessageDelete(config_message);
        return err;
    }
};

struct ModelInstanceState
{
    TRITONBACKEND_ModelInstance *triton_instance = nullptr;
    int device_id = 0;
    std::unique_ptr<classifier_postprocess::ClassifierPostprocess> postprocessor;
    cudaStream_t stream = nullptr;
    tensor::Memory<uint8_t> input_workspace_;

    explicit ModelInstanceState(TRITONBACKEND_ModelInstance *instance)
        : triton_instance(instance)
    {
        TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id);
    }

    ~ModelInstanceState()
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
    }

    TRITONSERVER_Error *Init(ModelState *model_state)
    {
        AutoDevice auto_device(device_id);

        cudaError_t cuerr = cudaStreamCreate(&stream);
        if (cuerr != cudaSuccess)
        {
            RETURN_TRITON_ERROR(INTERNAL, cudaGetErrorString(cuerr));
        }

        postprocessor = std::make_unique<classifier_postprocess::ClassifierPostprocess>(
            model_state->config);

        const auto &cfg = model_state->config;
        size_t max_input_bytes = static_cast<size_t>(cfg.max_batch_size) *
                                 cfg.num_classes * sizeof(float);
        input_workspace_.gpu(max_input_bytes);

        return nullptr;
    }
};

// ===================== Request Helpers =====================

struct RequestInfo
{
    TRITONBACKEND_Request *request = nullptr;
    TRITONBACKEND_Response *response = nullptr;

    int batch_size = 1;
    uint64_t total_input_bytes = 0;
    int image_offset = 0;

    TRITONSERVER_DataType input_datatype = TRITONSERVER_TYPE_FP32;
    bool input_on_device = false;
    int64_t input_mem_type_id = 0;
    const void *input_base = nullptr;

    void *classes_buffer = nullptr;
    void *scores_buffer = nullptr;

    TRITONSERVER_MemoryType classes_mem_type = TRITONSERVER_MEMORY_CPU;
    TRITONSERVER_MemoryType scores_mem_type = TRITONSERVER_MEMORY_CPU;

    int64_t classes_mem_type_id = 0;
    int64_t scores_mem_type_id = 0;
};

class ResponseGuard
{
  public:
    explicit ResponseGuard(const std::vector<RequestInfo> &infos) : infos_(infos) {}

    ~ResponseGuard()
    {
        if (committed_)
        {
            return;
        }

        TRITONSERVER_Error_Code error_code = TRITONSERVER_ERROR_INTERNAL;
        const char *error_message = "Unknown internal error";

        if (error_ != nullptr)
        {
            error_code    = TRITONSERVER_ErrorCode(error_);
            error_message = TRITONSERVER_ErrorMessage(error_);
        }

        for (const auto &info : infos_)
        {
            TRITONSERVER_Error *cloned_error =
                TRITONSERVER_ErrorNew(error_code, error_message);

            if (info.response == nullptr)
            {
                TRITONBACKEND_Response *response = nullptr;
                TRITONSERVER_Error *new_err =
                    TRITONBACKEND_ResponseNew(&response, info.request);

                if (new_err == nullptr)
                {
                    TRITONSERVER_Error *send_err = TRITONBACKEND_ResponseSend(
                        response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, cloned_error);
                    if (send_err != nullptr)
                    {
                        TRITONSERVER_ErrorDelete(send_err);
                    }
                    TRITONSERVER_ErrorDelete(cloned_error);
                }
                else
                {
                    TRITONSERVER_ErrorDelete(new_err);
                    TRITONSERVER_ErrorDelete(cloned_error);
                }
            }
            else
            {
                TRITONSERVER_Error *send_err = TRITONBACKEND_ResponseSend(
                    info.response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, cloned_error);
                if (send_err != nullptr)
                {
                    TRITONSERVER_ErrorDelete(send_err);
                }
                TRITONSERVER_ErrorDelete(cloned_error);
            }
        }

        if (error_ != nullptr)
        {
            TRITONSERVER_ErrorDelete(error_);
        }
    }

    void SetError(TRITONSERVER_Error *error)
    {
        if (error == nullptr)
        {
            return;
        }

        if (error_ == nullptr)
        {
            error_ = error;
        }
        else
        {
            TRITONSERVER_ErrorDelete(error);
        }
    }

    void Commit() { committed_ = true; }

  private:
    const std::vector<RequestInfo> &infos_;
    TRITONSERVER_Error *error_ = nullptr;
    bool committed_ = false;
};

static TRITONSERVER_Error *
ExtractModelOutputFromRequest(
    TRITONBACKEND_Request *request,
    RequestInfo &info,
    int num_classes)
{
    info.request = request;

    uint32_t input_count;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));
    if (input_count != 1)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Only one input tensor per request is supported");
    }

    TRITONBACKEND_Input *input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, 0, &input));

    const char *input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t *input_shape;
    uint32_t input_dims_count;
    uint32_t input_buffer_count;
    uint64_t input_byte_size;

    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape,
        &input_dims_count, &input_byte_size, &input_buffer_count));

    if (input_datatype != TRITONSERVER_TYPE_FP32 &&
        input_datatype != TRITONSERVER_TYPE_FP16)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input data type must be FP32 or FP16");
    }

    if (input_dims_count != 2)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input must be 2-D [batch, num_classes] tensor");
    }

    int n = static_cast<int>(input_shape[0]);
    int classes = static_cast<int>(input_shape[1]);
    if (n <= 0 || classes != num_classes)
    {
        RETURN_TRITON_ERROR(INVALID_ARG,
            "Input shape must be [batch, num_classes] and num_classes match config");
    }

    int total_elements = n * classes;
    size_t expected_bytes = total_elements *
        (input_datatype == TRITONSERVER_TYPE_FP16 ? sizeof(uint16_t) : sizeof(float));
    if (input_byte_size != expected_bytes)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input byte size mismatch");
    }

    if (input_buffer_count != 1)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input buffer count must be 1");
    }

    const void *buffer;
    TRITONSERVER_MemoryType mem_type;
    int64_t mem_type_id;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, 0, &buffer, &input_byte_size, &mem_type, &mem_type_id));

    info.batch_size        = n;
    info.total_input_bytes = input_byte_size;
    info.input_base        = buffer;
    info.input_on_device   = (mem_type == TRITONSERVER_MEMORY_GPU);
    info.input_mem_type_id = mem_type_id;
    info.input_datatype    = input_datatype;

    return nullptr;
}

// ===================== Response Helpers =====================

static TRITONSERVER_Error *
AllocateOutput(
    TRITONBACKEND_Response *response,
    const char *name,
    TRITONSERVER_DataType dtype,
    const int64_t *shape,
    uint32_t dims,
    size_t byte_size,
    void **buffer,
    TRITONSERVER_MemoryType *memory_type,
    int64_t *memory_type_id)
{
    TRITONBACKEND_Output *output;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
        response, &output, name, dtype, shape, dims));

    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
        output, buffer, byte_size, memory_type, memory_type_id));

    return nullptr;
}

static TRITONSERVER_Error *
CopyOutputToResponse(
    void *response_buffer,
    const void *workspace_buffer,
    size_t byte_size,
    TRITONSERVER_MemoryType memory_type,
    cudaStream_t stream)
{
    cudaMemcpyKind kind = (memory_type == TRITONSERVER_MEMORY_GPU)
                              ? cudaMemcpyDeviceToDevice
                              : cudaMemcpyDeviceToHost;

    cudaError_t err = cudaMemcpyAsync(response_buffer, workspace_buffer, byte_size, kind, stream);
    if (err != cudaSuccess)
    {
        RETURN_TRITON_ERROR(INTERNAL, cudaGetErrorString(err));
    }

    return nullptr;
}

// ===================== Execute =====================

extern "C" {

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance *instance,
    TRITONBACKEND_Request **requests,
    const uint32_t request_count)
{
    ModelInstanceState *instance_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void **>(&instance_state)));

    int device_id;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
    AutoDevice auto_device(device_id);

    cudaStream_t stream = instance_state->stream;
    classifier_postprocess::ClassifierPostprocess *postprocessor = instance_state->postprocessor.get();
    const auto &config = postprocessor->config();
    const int top_k = postprocessor->top_k();

    // 1. 提取所有 request 信息
    std::vector<RequestInfo> infos;
    infos.reserve(request_count);

    for (uint32_t r = 0; r < request_count; ++r)
    {
        RequestInfo info;
        TRITONSERVER_Error *err = ExtractModelOutputFromRequest(
            requests[r], info, config.num_classes);

        if (err != nullptr)
        {
            TRITONBACKEND_Response *response = nullptr;
            TRITONSERVER_Error *new_err = TRITONBACKEND_ResponseNew(&response, requests[r]);

            if (new_err == nullptr)
            {
                TRITONSERVER_Error *send_err =
                    TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
                if (send_err != nullptr)
                {
                    TRITONSERVER_ErrorDelete(send_err);
                }
                TRITONSERVER_ErrorDelete(err);
            }
            else
            {
                TRITONSERVER_ErrorDelete(new_err);
                TRITONSERVER_ErrorDelete(err);
            }
            continue;
        }

        infos.push_back(std::move(info));
    }

    const int request_num = static_cast<int>(infos.size());
    if (request_num == 0)
    {
        return nullptr;
    }

    ResponseGuard guard(infos);

    #define GUARDED_RETURN_IF_ERROR(X)                      \
        do                                                  \
        {                                                   \
            TRITONSERVER_Error *err__ = (X);                \
            if (err__ != nullptr)                           \
            {                                               \
                guard.SetError(err__);                      \
                return nullptr;                             \
            }                                               \
        } while (false)

    // 2. 为有效 request 创建 response
    for (int i = 0; i < request_num; ++i)
    {
        GUARDED_RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&infos[i].response, infos[i].request));

        infos[i].classes_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].classes_mem_type_id = device_id;
        infos[i].scores_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].scores_mem_type_id = device_id;
    }

    // 3. 计算整体 batch 规模
    int total_images = 0;
    uint64_t total_input_bytes = 0;

    for (auto &info : infos)
    {
        info.image_offset = total_images;
        total_images += info.batch_size;
        total_input_bytes += info.total_input_bytes;
    }

    // 4. 使用实例预分配的输入 workspace
    uint8_t *input_base_ptr = instance_state->input_workspace_.gpu(total_input_bytes);
    if (total_input_bytes > 0 && input_base_ptr == nullptr)
    {
        GUARDED_RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to allocate input device workspace"));
    }

    // 5. 拷贝所有输入到连续 device workspace
    uint64_t input_offset = 0;
    for (const auto &info : infos)
    {
        uint8_t *dst = input_base_ptr + input_offset;
        cudaError_t err = CopyBufferToDevice(
            dst, info.input_base, info.total_input_bytes,
            info.input_on_device, static_cast<int>(info.input_mem_type_id),
            device_id, stream);
        if (err != cudaSuccess)
        {
            GUARDED_RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
        }
        input_offset += info.total_input_bytes;
    }

    bool input_is_half = (infos[0].input_datatype == TRITONSERVER_TYPE_FP16);
    const void *d_input = input_base_ptr;

    // 6. 执行后处理
    postprocessor->forward(d_input, input_is_half, total_images, stream);

    int *d_classes = postprocessor->classes_gpu();
    float *d_scores = postprocessor->scores_gpu();

    // 7. 分配输出 buffer 并拷贝结果
    for (auto &info : infos)
    {
        const int offset = info.image_offset;
        const int batch_size = info.batch_size;

        const int64_t classes_shape[2] = {batch_size, top_k};
        const int64_t scores_shape[2] = {batch_size, top_k};

        const size_t classes_bytes = batch_size * top_k * sizeof(int);
        const size_t scores_bytes = batch_size * top_k * sizeof(float);

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            info.response, "classes", TRITONSERVER_TYPE_INT32,
            classes_shape, 2, classes_bytes,
            &info.classes_buffer, &info.classes_mem_type,
            &info.classes_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            info.response, "scores", TRITONSERVER_TYPE_FP32,
            scores_shape, 2, scores_bytes,
            &info.scores_buffer, &info.scores_mem_type,
            &info.scores_mem_type_id));

        GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
            info.classes_buffer,
            d_classes + offset * top_k,
            classes_bytes,
            info.classes_mem_type,
            stream));

        GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
            info.scores_buffer,
            d_scores + offset * top_k,
            scores_bytes,
            info.scores_mem_type,
            stream));
    }

    // 8. 同步流并发送响应
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess)
    {
        GUARDED_RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(sync_err)));
    }

    guard.Commit();
    for (const auto &info : infos)
    {
        TRITONSERVER_Error *send_err = TRITONBACKEND_ResponseSend(
            info.response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
        if (send_err != nullptr)
        {
            TRITONSERVER_ErrorDelete(send_err);
        }
    }

    return nullptr;
}

// ===================== Backend Lifecycle =====================

TRITONSERVER_Error *
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model)
{
    const char *cid;
    RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cid));
    std::string name(cid);

    auto *model_state = new ModelState(model);
    RETURN_IF_ERROR(model_state->LoadConfig());
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void *>(model_state)));

    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model)
{
    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
    delete reinterpret_cast<ModelState *>(vstate);
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance)
{
    TRITONBACKEND_Model *model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
    ModelState *model_state = reinterpret_cast<ModelState *>(vstate);

    auto *instance_state = new ModelInstanceState(instance);
    RETURN_IF_ERROR(instance_state->Init(model_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void *>(instance_state)));

    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance)
{
    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    delete reinterpret_cast<ModelInstanceState *>(vstate);
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend)
{
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_Finalize(TRITONBACKEND_Backend *)
{
    return nullptr;
}

} // extern "C"

} // namespace classifier_postprocess_backend
