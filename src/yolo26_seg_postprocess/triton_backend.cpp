/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_seg_postprocess/yolo26_seg_postprocess_impl.hpp"
#include "yolo26_seg_postprocess/triton_config.hpp"
#include "common/device.hpp"

#include <triton/core/tritonbackend.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define BACKEND_NAME "yolo26_seg_postprocess"

namespace yolo26_seg_postprocess_backend
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
    yolo26_seg_postprocess::Yolo26SegPostprocessConfig config;

    explicit ModelState(TRITONBACKEND_Model *model) : triton_model(model) {}

    TRITONSERVER_Error *LoadConfig()
    {
        TRITONSERVER_Message *config_message;
        RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));
        TRITONSERVER_Error *err = ParseYolo26SegPostprocessConfig(config_message, config);
        TRITONSERVER_MessageDelete(config_message);
        return err;
    }
};

// ===================== Async Response Completion =====================

struct CompletionTask
{
    std::vector<TRITONBACKEND_Response *> responses;
    cudaEvent_t event = nullptr;
};

class CompletionQueue
{
  public:
    CompletionQueue() = default;

    void SetDeviceId(int device_id)
    {
        device_id_ = device_id;
    }

    void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (shutdown_)
                return;
            shutdown_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable())
        {
            worker_.join();
        }
    }

    ~CompletionQueue()
    {
        Stop();
    }

    void Push(CompletionTask task)
    {
        EnsureStarted();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            queue_.push(std::move(task));
        }
        cv_.notify_one();
    }

  private:
    void EnsureStarted()
    {
        std::lock_guard<std::mutex> lock(start_mutex_);
        if (!worker_.joinable())
        {
            worker_ = std::thread(&CompletionQueue::Run, this);
        }
    }

    void Run()
    {
        AutoDevice auto_device(device_id_);

        while (true)
        {
            CompletionTask task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });
                if (shutdown_ && queue_.empty())
                {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop();
            }

            if (task.event != nullptr)
            {
                cudaEventSynchronize(task.event);
                cudaEventDestroy(task.event);
            }

            for (auto *response : task.responses)
            {
                if (response != nullptr)
                {
                    TRITONSERVER_Error *send_err = TRITONBACKEND_ResponseSend(
                        response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
                    if (send_err != nullptr)
                    {
                        TRITONSERVER_ErrorDelete(send_err);
                    }
                }
            }
        }
    }

    int device_id_ = 0;
    std::mutex start_mutex_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<CompletionTask> queue_;
    bool shutdown_ = false;
    std::thread worker_;
};

struct ModelInstanceState
{
    TRITONBACKEND_ModelInstance *triton_instance = nullptr;
    int device_id = 0;
    std::unique_ptr<yolo26_seg_postprocess::Yolo26SegPostprocess> postprocessor;
    cudaStream_t stream = nullptr;
    tensor::Memory<uint8_t> output0_workspace_;
    tensor::Memory<uint8_t> output1_workspace_;
    tensor::Memory<float> transform_workspace_;
    CompletionQueue completion_queue;
    int *h_num_dets_pinned = nullptr;
    size_t pinned_capacity = 0;

    explicit ModelInstanceState(TRITONBACKEND_ModelInstance *instance)
        : triton_instance(instance)
    {
        TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id);
        completion_queue.SetDeviceId(device_id);
    }

    ~ModelInstanceState()
    {
        completion_queue.Stop();

        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }

        if (h_num_dets_pinned != nullptr)
        {
            cudaFreeHost(h_num_dets_pinned);
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

        postprocessor = std::make_unique<yolo26_seg_postprocess::Yolo26SegPostprocess>(
            model_state->config);

        const auto &cfg = model_state->config;
        size_t max_output0_elements = static_cast<size_t>(cfg.max_batch_size) *
                                      cfg.max_detections * (6 + cfg.num_masks);
        output0_workspace_.gpu(max_output0_elements * sizeof(float));

        size_t max_output1_elements = static_cast<size_t>(cfg.max_batch_size) *
                                      cfg.num_masks * cfg.proto_height * cfg.proto_width;
        output1_workspace_.gpu(max_output1_elements * sizeof(float));

        pinned_capacity = cfg.max_batch_size;
        cudaError_t pinned_err = cudaMallocHost(
            reinterpret_cast<void **>(&h_num_dets_pinned),
            pinned_capacity * sizeof(int));
        if (pinned_err != cudaSuccess)
        {
            h_num_dets_pinned = nullptr;
            pinned_capacity = 0;
            RETURN_TRITON_ERROR(INTERNAL, cudaGetErrorString(pinned_err));
        }

        return nullptr;
    }
};

// ===================== Request Helpers =====================

struct RequestInfo
{
    TRITONBACKEND_Request *request = nullptr;
    TRITONBACKEND_Response *response = nullptr;

    int batch_size = 1;
    int num_predictions = 0;
    int actual_num_dets = 0;
    uint64_t total_output0_bytes = 0;
    uint64_t total_output1_bytes = 0;
    int image_offset = 0;

    TRITONSERVER_DataType output0_datatype = TRITONSERVER_TYPE_FP32;
    TRITONSERVER_DataType output1_datatype = TRITONSERVER_TYPE_FP32;
    bool output0_on_device = false;
    int64_t output0_mem_type_id = 0;
    bool output1_on_device = false;
    int64_t output1_mem_type_id = 0;
    const void *output0_base = nullptr;
    const void *output1_base = nullptr;

    bool transform_on_device = false;
    int64_t transform_mem_type_id = 0;
    const void *transform_base = nullptr;
    uint64_t total_transform_bytes = 0;

    void *num_dets_buffer = nullptr;
    void *boxes_buffer = nullptr;
    void *scores_buffer = nullptr;
    void *classes_buffer = nullptr;
    void *detection_masks_buffer = nullptr;
    void *mask_offsets_buffer = nullptr;
    void *mask_shapes_buffer = nullptr;

    TRITONSERVER_MemoryType num_dets_mem_type = TRITONSERVER_MEMORY_CPU;
    TRITONSERVER_MemoryType boxes_mem_type = TRITONSERVER_MEMORY_CPU;
    TRITONSERVER_MemoryType scores_mem_type = TRITONSERVER_MEMORY_CPU;
    TRITONSERVER_MemoryType classes_mem_type = TRITONSERVER_MEMORY_CPU;
    TRITONSERVER_MemoryType detection_masks_mem_type = TRITONSERVER_MEMORY_CPU;
    TRITONSERVER_MemoryType mask_offsets_mem_type = TRITONSERVER_MEMORY_CPU;
    TRITONSERVER_MemoryType mask_shapes_mem_type = TRITONSERVER_MEMORY_CPU;

    int64_t num_dets_mem_type_id = 0;
    int64_t boxes_mem_type_id = 0;
    int64_t scores_mem_type_id = 0;
    int64_t classes_mem_type_id = 0;
    int64_t detection_masks_mem_type_id = 0;
    int64_t mask_offsets_mem_type_id = 0;
    int64_t mask_shapes_mem_type_id = 0;
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
                        response,
                        TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                        cloned_error);
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
                    info.response,
                    TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    cloned_error);
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
ExtractModelOutputsFromRequest(
    TRITONBACKEND_Request *request,
    RequestInfo &info,
    int num_masks)
{
    info.request = request;

    uint32_t input_count;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));
    if (input_count != 3)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Exactly three input tensors per request are required: model_output, mask_protos and transform_metadata");
    }

    // output0
    TRITONBACKEND_Input *output0_input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "model_output", &output0_input));

    const char *output0_name;
    TRITONSERVER_DataType output0_datatype;
    const int64_t *output0_shape;
    uint32_t output0_dims_count;
    uint32_t output0_buffer_count;
    uint64_t output0_byte_size;

    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        output0_input, &output0_name, &output0_datatype, &output0_shape,
        &output0_dims_count, &output0_byte_size, &output0_buffer_count));

    if (output0_datatype != TRITONSERVER_TYPE_FP32 &&
        output0_datatype != TRITONSERVER_TYPE_FP16)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output0 data type must be FP32 or FP16");
    }

    if (output0_dims_count != 3)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output0 must be 3-D [N, P, 6+num_masks] tensor");
    }

    int n = static_cast<int>(output0_shape[0]);
    int num_predictions = static_cast<int>(output0_shape[1]);
    int feat = static_cast<int>(output0_shape[2]);
    int expected_feat = 6 + num_masks;

    if (feat != expected_feat)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output0 last dim must equal 6 + num_masks");
    }

    if (n <= 0 || num_predictions <= 0)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output0 dimensions must be positive");
    }

    int total_elements = n * num_predictions * expected_feat;
    size_t expected_bytes = total_elements *
        (output0_datatype == TRITONSERVER_TYPE_FP16 ? sizeof(uint16_t) : sizeof(float));
    if (output0_byte_size != expected_bytes)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output0 byte size mismatch");
    }

    if (output0_buffer_count != 1)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output0 buffer count must be 1");
    }

    const void *output0_buffer;
    TRITONSERVER_MemoryType output0_mem_type;
    int64_t output0_mem_type_id;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        output0_input, 0, &output0_buffer, &output0_byte_size,
        &output0_mem_type, &output0_mem_type_id));

    info.batch_size = n;
    info.num_predictions = num_predictions;
    info.total_output0_bytes = output0_byte_size;
    info.output0_base = output0_buffer;
    info.output0_on_device = (output0_mem_type == TRITONSERVER_MEMORY_GPU);
    info.output0_mem_type_id = output0_mem_type_id;
    info.output0_datatype = output0_datatype;

    // output1 (prototypes)
    TRITONBACKEND_Input *output1_input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "mask_protos", &output1_input));

    const char *output1_name;
    TRITONSERVER_DataType output1_datatype;
    const int64_t *output1_shape;
    uint32_t output1_dims_count;
    uint32_t output1_buffer_count;
    uint64_t output1_byte_size;

    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        output1_input, &output1_name, &output1_datatype, &output1_shape,
        &output1_dims_count, &output1_byte_size, &output1_buffer_count));

    if (output1_datatype != TRITONSERVER_TYPE_FP32 &&
        output1_datatype != TRITONSERVER_TYPE_FP16)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output1 data type must be FP32 or FP16");
    }

    if (output1_dims_count != 4)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output1 must be 4-D [N, num_masks, H, W] tensor");
    }

    if (static_cast<int>(output1_shape[0]) != n)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output1 batch size must match output0");
    }

    if (static_cast<int>(output1_shape[1]) != num_masks)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output1 channel dim must equal num_masks");
    }

    int proto_h = static_cast<int>(output1_shape[2]);
    int proto_w = static_cast<int>(output1_shape[3]);
    if (proto_h <= 0 || proto_w <= 0)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output1 spatial dimensions must be positive");
    }

    int total_proto_elements = n * num_masks * proto_h * proto_w;
    size_t expected_proto_bytes = total_proto_elements *
        (output1_datatype == TRITONSERVER_TYPE_FP16 ? sizeof(uint16_t) : sizeof(float));
    if (output1_byte_size != expected_proto_bytes)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output1 byte size mismatch");
    }

    if (output1_buffer_count != 1)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "output1 buffer count must be 1");
    }

    const void *output1_buffer;
    TRITONSERVER_MemoryType output1_mem_type;
    int64_t output1_mem_type_id;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        output1_input, 0, &output1_buffer, &output1_byte_size,
        &output1_mem_type, &output1_mem_type_id));

    info.total_output1_bytes = output1_byte_size;
    info.output1_base = output1_buffer;
    info.output1_on_device = (output1_mem_type == TRITONSERVER_MEMORY_GPU);
    info.output1_mem_type_id = output1_mem_type_id;
    info.output1_datatype = output1_datatype;

    // transform_metadata
    TRITONBACKEND_Input *transform_input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "transform_metadata", &transform_input));

    const char *transform_name;
    TRITONSERVER_DataType transform_datatype;
    const int64_t *transform_shape;
    uint32_t transform_dims_count;
    uint32_t transform_buffer_count;
    uint64_t transform_byte_size;

    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        transform_input, &transform_name, &transform_datatype, &transform_shape,
        &transform_dims_count, &transform_byte_size, &transform_buffer_count));

    if (transform_datatype != TRITONSERVER_TYPE_FP32)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "transform_metadata data type must be FP32");
    }

    if (transform_dims_count != 2 || transform_shape[0] != n || transform_shape[1] != 6)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "transform_metadata must be [N, 6]");
    }

    if (transform_byte_size != static_cast<size_t>(n) * 6 * sizeof(float))
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "transform_metadata byte size mismatch");
    }

    if (transform_buffer_count != 1)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "transform_metadata buffer count must be 1");
    }

    const void *transform_buffer;
    TRITONSERVER_MemoryType transform_mem_type;
    int64_t transform_mem_type_id;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        transform_input, 0, &transform_buffer, &transform_byte_size,
        &transform_mem_type, &transform_mem_type_id));

    info.total_transform_bytes = transform_byte_size;
    info.transform_base = transform_buffer;
    info.transform_on_device = (transform_mem_type == TRITONSERVER_MEMORY_GPU);
    info.transform_mem_type_id = transform_mem_type_id;

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
    yolo26_seg_postprocess::Yolo26SegPostprocess *postprocessor = instance_state->postprocessor.get();
    const auto &config = postprocessor->config();
    const int max_detections = postprocessor->max_detections();
    constexpr int kMaskOutputSize = 160;

    // 1. 提取所有 request 信息
    std::vector<RequestInfo> infos;
    infos.reserve(request_count);

    for (uint32_t r = 0; r < request_count; ++r)
    {
        RequestInfo info;
        TRITONSERVER_Error *err = ExtractModelOutputsFromRequest(
            requests[r], info, config.num_masks);

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

        infos[i].num_dets_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].num_dets_mem_type_id = device_id;
        infos[i].boxes_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].boxes_mem_type_id = device_id;
        infos[i].scores_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].scores_mem_type_id = device_id;
        infos[i].classes_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].classes_mem_type_id = device_id;
        infos[i].detection_masks_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].detection_masks_mem_type_id = device_id;
        infos[i].mask_offsets_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].mask_offsets_mem_type_id = device_id;
        infos[i].mask_shapes_mem_type = TRITONSERVER_MEMORY_GPU;
        infos[i].mask_shapes_mem_type_id = device_id;
    }

    // 3. 计算整体 batch 规模并统一 num_predictions
    int total_images = 0;
    uint64_t total_output0_bytes = 0;
    uint64_t total_output1_bytes = 0;
    uint64_t total_transform_bytes = 0;
    int num_predictions = infos[0].num_predictions;

    for (auto &info : infos)
    {
        info.image_offset = total_images;
        total_images += info.batch_size;
        total_output0_bytes += info.total_output0_bytes;
        total_output1_bytes += info.total_output1_bytes;
        total_transform_bytes += info.total_transform_bytes;

        if (info.num_predictions != num_predictions)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "All requests must have the same num_predictions"));
            return nullptr;
        }
    }

    // 4. 准备 device 输入：
    //    单个 request 且两个输入都已在 GPU 上时（ensemble 链中上游模型的输出即为此情形），
    //    直接透传指针零拷贝；否则统一拷贝到实例预分配的连续 workspace。
    const void *d_output0 = nullptr;
    const void *d_output1 = nullptr;
    bool input_is_half = false;

    if (request_num == 1 && infos[0].output0_on_device && infos[0].output1_on_device &&
        infos[0].output0_mem_type_id == device_id && infos[0].output1_mem_type_id == device_id)
    {
        d_output0      = infos[0].output0_base;
        d_output1      = infos[0].output1_base;
        input_is_half  = (infos[0].output0_datatype == TRITONSERVER_TYPE_FP16);
    }
    else
    {
        uint8_t *output0_base_ptr = instance_state->output0_workspace_.gpu(total_output0_bytes);
        if (total_output0_bytes > 0 && output0_base_ptr == nullptr)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, "Failed to allocate output0 device workspace"));
            return nullptr;
        }

        uint8_t *output1_base_ptr = instance_state->output1_workspace_.gpu(total_output1_bytes);
        if (total_output1_bytes > 0 && output1_base_ptr == nullptr)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, "Failed to allocate output1 device workspace"));
            return nullptr;
        }

        // 5. 拷贝所有输入到连续 device workspace
        uint64_t output0_offset = 0;
        uint64_t output1_offset = 0;
        for (const auto &info : infos)
        {
            uint8_t *dst0 = output0_base_ptr + output0_offset;
            cudaError_t err0 = CopyBufferToDevice(
                dst0, info.output0_base, info.total_output0_bytes,
                info.output0_on_device, static_cast<int>(info.output0_mem_type_id),
                device_id, stream);
            if (err0 != cudaSuccess)
            {
                guard.SetError(TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err0)));
                return nullptr;
            }
            output0_offset += info.total_output0_bytes;

            uint8_t *dst1 = output1_base_ptr + output1_offset;
            cudaError_t err1 = CopyBufferToDevice(
                dst1, info.output1_base, info.total_output1_bytes,
                info.output1_on_device, static_cast<int>(info.output1_mem_type_id),
                device_id, stream);
            if (err1 != cudaSuccess)
            {
                guard.SetError(TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err1)));
                return nullptr;
            }
            output1_offset += info.total_output1_bytes;
        }

        input_is_half = (infos[0].output0_datatype == TRITONSERVER_TYPE_FP16);
        d_output0     = output0_base_ptr;
        d_output1     = output1_base_ptr;
    }

    // 6. 准备 transform_metadata device buffer
    float *d_transform = nullptr;
    {
        float *transform_base_ptr = instance_state->transform_workspace_.gpu(total_transform_bytes / sizeof(float));
        if (total_transform_bytes > 0 && transform_base_ptr == nullptr)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, "Failed to allocate transform device workspace"));
            return nullptr;
        }

        uint64_t transform_offset = 0;
        for (const auto &info : infos)
        {
            const size_t bytes = info.total_transform_bytes;
            if (bytes == 0)
            {
                continue;
            }
            float *dst = transform_base_ptr + transform_offset;
            cudaError_t err = CopyBufferToDevice(
                dst, info.transform_base, bytes,
                info.transform_on_device, static_cast<int>(info.transform_mem_type_id),
                device_id, stream);
            if (err != cudaSuccess)
            {
                guard.SetError(TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
                return nullptr;
            }
            transform_offset += info.batch_size * 6;
        }
        d_transform = transform_base_ptr;
    }

    // 7. 执行后处理
    postprocessor->forward(
        d_output0,
        d_output1,
        input_is_half,
        total_images,
        num_predictions,
        stream,
        d_transform);

    int *d_num_dets = postprocessor->num_detections_gpu();
    float *d_boxes = postprocessor->boxes_gpu();
    float *d_scores = postprocessor->scores_gpu();
    int *d_classes = postprocessor->classes_gpu();
    float *d_detection_masks = postprocessor->masks_gpu();
    int *d_mask_offsets = postprocessor->mask_offsets_gpu();
    int *d_mask_shapes = postprocessor->mask_shapes_gpu();

    // 7. 把每个样本的实际 num_dets 拷贝到主机，用于动态输出形状
    if (static_cast<size_t>(total_images) > instance_state->pinned_capacity)
    {
        int *new_pinned = nullptr;
        cudaError_t realloc_err = cudaMallocHost(
            reinterpret_cast<void **>(&new_pinned),
            static_cast<size_t>(total_images) * sizeof(int));
        if (realloc_err != cudaSuccess)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(realloc_err)));
            return nullptr;
        }
        if (instance_state->h_num_dets_pinned != nullptr)
        {
            cudaFreeHost(instance_state->h_num_dets_pinned);
        }
        instance_state->h_num_dets_pinned = new_pinned;
        instance_state->pinned_capacity = static_cast<size_t>(total_images);
    }

    cudaError_t num_dets_err = cudaMemcpyAsync(
        instance_state->h_num_dets_pinned, d_num_dets, total_images * sizeof(int),
        cudaMemcpyDeviceToHost, stream);
    if (num_dets_err != cudaSuccess)
    {
        guard.SetError(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(num_dets_err)));
        return nullptr;
    }
    num_dets_err = cudaStreamSynchronize(stream);
    if (num_dets_err != cudaSuccess)
    {
        guard.SetError(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(num_dets_err)));
        return nullptr;
    }

    int *h_num_dets = instance_state->h_num_dets_pinned;

    // 8. 根据实际 num_dets 分配动态输出 buffer
    for (int i = 0; i < request_num; ++i)
    {
        int offset = infos[i].image_offset;
        int actual_num_dets = 0;
        for (int b = 0; b < infos[i].batch_size; ++b)
        {
            actual_num_dets = std::max(actual_num_dets, h_num_dets[offset + b]);
        }
        if (actual_num_dets > max_detections)
        {
            actual_num_dets = max_detections;
        }
        if (actual_num_dets < 0)
        {
            actual_num_dets = 0;
        }
        infos[i].actual_num_dets = actual_num_dets;

        const int64_t num_dets_shape[2] = {infos[i].batch_size, 1};
        const int64_t boxes_shape[3] = {infos[i].batch_size, actual_num_dets, 4};
        const int64_t scores_shape[2] = {infos[i].batch_size, actual_num_dets};
        const int64_t classes_shape[2] = {infos[i].batch_size, actual_num_dets};
        const int64_t detection_masks_shape[2] = {
            infos[i].batch_size, actual_num_dets * kMaskOutputSize * kMaskOutputSize};
        const int64_t mask_offsets_shape[2] = {infos[i].batch_size, actual_num_dets};
        const int64_t mask_shapes_shape[3] = {infos[i].batch_size, actual_num_dets, 2};

        const size_t num_dets_bytes = infos[i].batch_size * sizeof(int);
        const size_t boxes_bytes = infos[i].batch_size * actual_num_dets * 4 * sizeof(float);
        const size_t scores_bytes = infos[i].batch_size * actual_num_dets * sizeof(float);
        const size_t classes_bytes = infos[i].batch_size * actual_num_dets * sizeof(int);
        const size_t detection_masks_bytes = infos[i].batch_size * actual_num_dets *
                                             kMaskOutputSize * kMaskOutputSize * sizeof(float);
        const size_t mask_offsets_bytes = infos[i].batch_size * actual_num_dets * sizeof(int);
        const size_t mask_shapes_bytes = infos[i].batch_size * actual_num_dets * 2 * sizeof(int);

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "num_dets", TRITONSERVER_TYPE_INT32,
            num_dets_shape, 2, num_dets_bytes,
            &infos[i].num_dets_buffer, &infos[i].num_dets_mem_type,
            &infos[i].num_dets_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "detection_boxes", TRITONSERVER_TYPE_FP32,
            boxes_shape, 3, boxes_bytes,
            &infos[i].boxes_buffer, &infos[i].boxes_mem_type,
            &infos[i].boxes_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "detection_scores", TRITONSERVER_TYPE_FP32,
            scores_shape, 2, scores_bytes,
            &infos[i].scores_buffer, &infos[i].scores_mem_type,
            &infos[i].scores_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "detection_classes", TRITONSERVER_TYPE_INT32,
            classes_shape, 2, classes_bytes,
            &infos[i].classes_buffer, &infos[i].classes_mem_type,
            &infos[i].classes_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "detection_masks", TRITONSERVER_TYPE_FP32,
            detection_masks_shape, 2, detection_masks_bytes,
            &infos[i].detection_masks_buffer, &infos[i].detection_masks_mem_type,
            &infos[i].detection_masks_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "mask_offsets", TRITONSERVER_TYPE_INT32,
            mask_offsets_shape, 2, mask_offsets_bytes,
            &infos[i].mask_offsets_buffer, &infos[i].mask_offsets_mem_type,
            &infos[i].mask_offsets_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "mask_shapes", TRITONSERVER_TYPE_INT32,
            mask_shapes_shape, 3, mask_shapes_bytes,
            &infos[i].mask_shapes_buffer, &infos[i].mask_shapes_mem_type,
            &infos[i].mask_shapes_mem_type_id));
    }

    // 9. 将结果从 workspace 分发到各 response 的 output buffer
    for (const auto &info : infos)
    {
        const int offset = info.image_offset;
        const int actual_num_dets = info.actual_num_dets;

        GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
            info.num_dets_buffer,
            d_num_dets + offset,
            info.batch_size * sizeof(int),
            info.num_dets_mem_type,
            stream));

        if (actual_num_dets > 0)
        {
            GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                info.boxes_buffer,
                d_boxes + offset * max_detections * 4,
                info.batch_size * actual_num_dets * 4 * sizeof(float),
                info.boxes_mem_type,
                stream));

            GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                info.scores_buffer,
                d_scores + offset * max_detections,
                info.batch_size * actual_num_dets * sizeof(float),
                info.scores_mem_type,
                stream));

            GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                info.classes_buffer,
                d_classes + offset * max_detections,
                info.batch_size * actual_num_dets * sizeof(int),
                info.classes_mem_type,
                stream));

            GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                info.detection_masks_buffer,
                d_detection_masks + offset * max_detections * kMaskOutputSize * kMaskOutputSize,
                info.batch_size * actual_num_dets * kMaskOutputSize * kMaskOutputSize * sizeof(float),
                info.detection_masks_mem_type,
                stream));

            GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                info.mask_shapes_buffer,
                d_mask_shapes + offset * max_detections * 2,
                info.batch_size * actual_num_dets * 2 * sizeof(int),
                info.mask_shapes_mem_type,
                stream));
        }

        // mask_offsets 需要重新计算为相对偏移：每个检测框在 sample 内 mask buffer 中的位置
        if (actual_num_dets > 0 && info.mask_offsets_buffer != nullptr)
        {
            if (info.mask_offsets_mem_type == TRITONSERVER_MEMORY_CPU)
            {
                for (int b = 0; b < info.batch_size; ++b)
                {
                    int *dst = static_cast<int *>(info.mask_offsets_buffer) +
                               b * actual_num_dets;
                    for (int d = 0; d < actual_num_dets; ++d)
                    {
                        dst[d] = d * kMaskOutputSize * kMaskOutputSize;
                    }
                }
            }
            else
            {
                // GPU buffer：通过 cudaMemcpyAsync 从临时 host buffer 拷贝
                std::vector<int> host_offsets(info.batch_size * actual_num_dets);
                for (int b = 0; b < info.batch_size; ++b)
                {
                    for (int d = 0; d < actual_num_dets; ++d)
                    {
                        host_offsets[b * actual_num_dets + d] = d * kMaskOutputSize * kMaskOutputSize;
                    }
                }
                cudaError_t merr = cudaMemcpyAsync(
                    info.mask_offsets_buffer, host_offsets.data(),
                    host_offsets.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
                if (merr != cudaSuccess)
                {
                    guard.SetError(TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(merr)));
                    return nullptr;
                }
            }
        }
    }

    // 10. 记录 CUDA event，将响应发送交给后台线程
    cudaEvent_t event;
    cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess)
    {
        guard.SetError(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
        return nullptr;
    }
    cudaEventRecord(event, stream);

    CompletionTask task;
    task.event = event;
    task.responses.reserve(request_num);
    for (auto &info : infos)
    {
        task.responses.push_back(info.response);
    }

    instance_state->completion_queue.Push(std::move(task));

    guard.Commit();
    return nullptr;

    #undef GUARDED_RETURN_IF_ERROR
}

} // extern "C"

// ===================== Backend C API =====================

extern "C" {

TRITONSERVER_Error *
TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend)
{
    const char *cname;
    RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));

    if (std::string(cname) != BACKEND_NAME)
    {
        RETURN_TRITON_ERROR(INTERNAL, "Unexpected backend name");
    }

    uint32_t api_version_major, api_version_minor;
    RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

    if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
        (api_version_minor < TRITONBACKEND_API_VERSION_MINOR))
    {
        RETURN_TRITON_ERROR(INTERNAL, "Triton backend API version mismatch");
    }

    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_Finalize(TRITONBACKEND_Backend *)
{
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model)
{
    ModelState *model_state = new ModelState(model);
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

    void *vmodelstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
    ModelState *model_state = reinterpret_cast<ModelState *>(vmodelstate);

    ModelInstanceState *instance_state = new ModelInstanceState(instance);
    TRITONSERVER_Error *init_err = instance_state->Init(model_state);
    if (init_err != nullptr)
    {
        delete instance_state;
        return init_err;
    }

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

} // extern "C"

} // namespace yolo26_seg_postprocess_backend
