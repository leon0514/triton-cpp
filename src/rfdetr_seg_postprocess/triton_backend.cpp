/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "rfdetr_seg_postprocess/rfdetr_seg_postprocess_impl.hpp"
#include "rfdetr_seg_postprocess/triton_config.hpp"
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

#define BACKEND_NAME "rfdetr_seg_postprocess"

namespace rfdetr_seg_postprocess_backend
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
    rfdetr_seg_postprocess::RfDetrSegPostprocessConfig config;

    explicit ModelState(TRITONBACKEND_Model *model) : triton_model(model) {}

    TRITONSERVER_Error *LoadConfig()
    {
        TRITONSERVER_Message *config_message;
        RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));
        TRITONSERVER_Error *err = ParseRfDetrSegPostprocessConfig(config_message, config);
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
    std::unique_ptr<rfdetr_seg_postprocess::RfDetrSegPostprocess> postprocessor;
    cudaStream_t stream = nullptr;
    tensor::Memory<uint8_t> dets_workspace_;
    tensor::Memory<uint8_t> labels_workspace_;
    tensor::Memory<uint8_t> masks_workspace_;
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

        postprocessor = std::make_unique<rfdetr_seg_postprocess::RfDetrSegPostprocess>(
            model_state->config);

        const auto &cfg = model_state->config;
        size_t max_dets_elements   = static_cast<size_t>(cfg.max_batch_size) * cfg.num_queries * 4;
        size_t max_labels_elements = static_cast<size_t>(cfg.max_batch_size) * cfg.num_queries * 91;
        size_t max_masks_elements  = static_cast<size_t>(cfg.max_batch_size) * cfg.num_queries * 160 * 160;
        dets_workspace_.gpu(max_dets_elements * sizeof(float));
        labels_workspace_.gpu(max_labels_elements * sizeof(float));
        masks_workspace_.gpu(max_masks_elements * sizeof(float));

        pinned_capacity = static_cast<size_t>(cfg.max_batch_size);
        cudaError_t pinned_err = cudaHostAlloc(
            &h_num_dets_pinned, pinned_capacity * sizeof(int), cudaHostAllocDefault);
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

struct InputBufferInfo
{
    const void *base = nullptr;
    uint64_t byte_size = 0;
    bool on_device = false;
    int64_t mem_type_id = 0;
    TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_FP32;
    int batch_size = 1;
    int dim0 = 0;
    int dim1 = 0;
    int dim2 = 0;
    int dim3 = 0;
};

struct RequestInfo
{
    TRITONBACKEND_Request *request = nullptr;
    TRITONBACKEND_Response *response = nullptr;

    int batch_size = 1;
    int num_queries = 0;
    int mask_height = 0;
    int mask_width = 0;
    int actual_num_dets = 0;

    InputBufferInfo dets_info;
    InputBufferInfo labels_info;
    InputBufferInfo masks_info;
    InputBufferInfo transform_info;

    int image_offset = 0;

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
ExtractInputByName(
    TRITONBACKEND_Request *request,
    const char *expected_name,
    int expected_dims,
    InputBufferInfo &info)
{
    TRITONBACKEND_Input *input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, expected_name, &input));

    const char *input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t *input_shape;
    uint32_t input_dims_count;
    uint32_t input_buffer_count;
    uint64_t input_byte_size;

    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape,
        &input_dims_count, &input_byte_size, &input_buffer_count));

    if (std::string(input_name) != expected_name)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, ("Input name must be " + std::string(expected_name)).c_str());
    }

    if (input_datatype != TRITONSERVER_TYPE_FP32 &&
        input_datatype != TRITONSERVER_TYPE_FP16)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input data type must be FP32 or FP16");
    }

    if (input_dims_count != expected_dims)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input dimension count mismatch");
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

    info.base        = buffer;
    info.byte_size   = input_byte_size;
    info.on_device   = (mem_type == TRITONSERVER_MEMORY_GPU);
    info.mem_type_id = mem_type_id;
    info.datatype    = input_datatype;

    return nullptr;
}

static TRITONSERVER_Error *
ExtractModelOutputsFromRequest(
    TRITONBACKEND_Request *request,
    RequestInfo &info)
{
    info.request = request;

    uint32_t input_count;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));
    if (input_count != 4)
    {
        RETURN_TRITON_ERROR(INVALID_ARG,
            "RF-DETR seg postprocess requires four inputs: dets, labels, masks and transform_metadata");
    }

    RETURN_IF_ERROR(ExtractInputByName(request, "dets", 3, info.dets_info));
    RETURN_IF_ERROR(ExtractInputByName(request, "labels", 3, info.labels_info));
    RETURN_IF_ERROR(ExtractInputByName(request, "masks", 4, info.masks_info));

    // transform_metadata
    {
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

        if (transform_dims_count != 2)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "transform_metadata must be 2-D");
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

        info.transform_info.base        = transform_buffer;
        info.transform_info.byte_size   = transform_byte_size;
        info.transform_info.on_device   = (transform_mem_type == TRITONSERVER_MEMORY_GPU);
        info.transform_info.mem_type_id = transform_mem_type_id;
        info.transform_info.datatype    = transform_datatype;
        info.transform_info.dim0        = static_cast<int>(transform_shape[0]);
        info.transform_info.dim1        = static_cast<int>(transform_shape[1]);
    }

    // dets shape [N, Q, 4]
    TRITONBACKEND_Input *dets_input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "dets", &dets_input));
    const char *dets_name;
    TRITONSERVER_DataType dets_dtype;
    const int64_t *dets_shape;
    uint32_t dets_dims;
    uint32_t dets_buffer_count;
    uint64_t dets_byte_size;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        dets_input, &dets_name, &dets_dtype, &dets_shape,
        &dets_dims, &dets_byte_size, &dets_buffer_count));

    int n_dets = static_cast<int>(dets_shape[0]);
    int q_dets = static_cast<int>(dets_shape[1]);
    int f_dets = static_cast<int>(dets_shape[2]);
    if (f_dets != 4)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "dets last dim must equal 4");
    }

    // labels shape [N, Q, 91]
    TRITONBACKEND_Input *labels_input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "labels", &labels_input));
    const char *labels_name;
    TRITONSERVER_DataType labels_dtype;
    const int64_t *labels_shape;
    uint32_t labels_dims;
    uint32_t labels_buffer_count;
    uint64_t labels_byte_size;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        labels_input, &labels_name, &labels_dtype, &labels_shape,
        &labels_dims, &labels_byte_size, &labels_buffer_count));

    int n_labels = static_cast<int>(labels_shape[0]);
    int q_labels = static_cast<int>(labels_shape[1]);
    int f_labels = static_cast<int>(labels_shape[2]);
    if (f_labels != 91)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "labels last dim must equal 91");
    }

    // masks shape [N, Q, H, W]
    TRITONBACKEND_Input *masks_input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "masks", &masks_input));
    const char *masks_name;
    TRITONSERVER_DataType masks_dtype;
    const int64_t *masks_shape;
    uint32_t masks_dims;
    uint32_t masks_buffer_count;
    uint64_t masks_byte_size;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        masks_input, &masks_name, &masks_dtype, &masks_shape,
        &masks_dims, &masks_byte_size, &masks_buffer_count));

    int n_masks = static_cast<int>(masks_shape[0]);
    int q_masks = static_cast<int>(masks_shape[1]);
    int h_masks = static_cast<int>(masks_shape[2]);
    int w_masks = static_cast<int>(masks_shape[3]);

    if (n_dets != n_labels || n_dets != n_masks || q_dets != q_labels || q_dets != q_masks)
    {
        RETURN_TRITON_ERROR(INVALID_ARG,
            "dets, labels and masks must have the same batch and query dimensions");
    }

    if (n_dets <= 0 || q_dets <= 0 || h_masks <= 0 || w_masks <= 0)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input dimensions must be positive");
    }

    if (info.transform_info.dim0 != n_dets || info.transform_info.dim1 != 6)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "transform_metadata must be [N, 6] matching dets batch");
    }

    if (info.transform_info.byte_size != static_cast<size_t>(n_dets) * 6 * sizeof(float))
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "transform_metadata byte size mismatch");
    }

    size_t expected_dets_bytes = static_cast<size_t>(n_dets) * q_dets * 4 *
        (dets_dtype == TRITONSERVER_TYPE_FP16 ? sizeof(uint16_t) : sizeof(float));
    if (dets_byte_size != expected_dets_bytes)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "dets byte size mismatch");
    }

    size_t expected_labels_bytes = static_cast<size_t>(n_labels) * q_labels * 91 *
        (labels_dtype == TRITONSERVER_TYPE_FP16 ? sizeof(uint16_t) : sizeof(float));
    if (labels_byte_size != expected_labels_bytes)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "labels byte size mismatch");
    }

    size_t expected_masks_bytes = static_cast<size_t>(n_masks) * q_masks * h_masks * w_masks *
        (masks_dtype == TRITONSERVER_TYPE_FP16 ? sizeof(uint16_t) : sizeof(float));
    if (masks_byte_size != expected_masks_bytes)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "masks byte size mismatch");
    }

    if (dets_dtype != labels_dtype || dets_dtype != masks_dtype)
    {
        RETURN_TRITON_ERROR(INVALID_ARG,
            "dets, labels and masks must have the same data type");
    }

    info.batch_size   = n_dets;
    info.num_queries  = q_dets;
    info.mask_height  = h_masks;
    info.mask_width   = w_masks;

    info.dets_info.dim0 = n_dets;
    info.dets_info.dim1 = q_dets;
    info.dets_info.dim2 = f_dets;

    info.labels_info.dim0 = n_labels;
    info.labels_info.dim1 = q_labels;
    info.labels_info.dim2 = f_labels;

    info.masks_info.dim0 = n_masks;
    info.masks_info.dim1 = q_masks;
    info.masks_info.dim2 = h_masks;
    info.masks_info.dim3 = w_masks;

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
    if (byte_size == 0 || response_buffer == nullptr || workspace_buffer == nullptr)
    {
        return nullptr;
    }

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
    rfdetr_seg_postprocess::RfDetrSegPostprocess *postprocessor = instance_state->postprocessor.get();
    const auto &config = postprocessor->config();
    const int max_detections = postprocessor->max_detections();
    constexpr int kMaskOutputSize = 160;

    std::vector<RequestInfo> infos;
    infos.reserve(request_count);

    for (uint32_t r = 0; r < request_count; ++r)
    {
        RequestInfo info;
        TRITONSERVER_Error *err = ExtractModelOutputsFromRequest(requests[r], info);

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

    for (int i = 0; i < request_num; ++i)
    {
        GUARDED_RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&infos[i].response, infos[i].request));

        infos[i].num_dets_mem_type    = TRITONSERVER_MEMORY_GPU;
        infos[i].num_dets_mem_type_id = device_id;
        infos[i].boxes_mem_type       = TRITONSERVER_MEMORY_GPU;
        infos[i].boxes_mem_type_id    = device_id;
        infos[i].scores_mem_type      = TRITONSERVER_MEMORY_GPU;
        infos[i].scores_mem_type_id   = device_id;
        infos[i].classes_mem_type     = TRITONSERVER_MEMORY_GPU;
        infos[i].classes_mem_type_id  = device_id;
        infos[i].detection_masks_mem_type    = TRITONSERVER_MEMORY_GPU;
        infos[i].detection_masks_mem_type_id = device_id;
        infos[i].mask_offsets_mem_type       = TRITONSERVER_MEMORY_GPU;
        infos[i].mask_offsets_mem_type_id    = device_id;
        infos[i].mask_shapes_mem_type        = TRITONSERVER_MEMORY_GPU;
        infos[i].mask_shapes_mem_type_id     = device_id;
    }

    int total_images = 0;
    uint64_t total_dets_bytes = 0;
    uint64_t total_labels_bytes = 0;
    uint64_t total_masks_bytes = 0;
    uint64_t total_transform_bytes = 0;
    int num_queries = infos[0].num_queries;
    int mask_height = infos[0].mask_height;
    int mask_width  = infos[0].mask_width;

    for (auto &info : infos)
    {
        info.image_offset = total_images;
        total_images += info.batch_size;
        total_dets_bytes += info.dets_info.byte_size;
        total_labels_bytes += info.labels_info.byte_size;
        total_masks_bytes += info.masks_info.byte_size;
        total_transform_bytes += info.transform_info.byte_size;

        if (info.num_queries != num_queries)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "All requests must have the same num_queries"));
            return nullptr;
        }
        if (info.mask_height != mask_height || info.mask_width != mask_width)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "All requests must have the same mask resolution"));
            return nullptr;
        }
    }

    uint8_t *dets_base_ptr = instance_state->dets_workspace_.gpu(total_dets_bytes);
    uint8_t *labels_base_ptr = instance_state->labels_workspace_.gpu(total_labels_bytes);
    uint8_t *masks_base_ptr = instance_state->masks_workspace_.gpu(total_masks_bytes);
    if ((total_dets_bytes > 0 && dets_base_ptr == nullptr) ||
        (total_labels_bytes > 0 && labels_base_ptr == nullptr) ||
        (total_masks_bytes > 0 && masks_base_ptr == nullptr))
    {
        guard.SetError(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL, "Failed to allocate input device workspace"));
        return nullptr;
    }

    uint64_t dets_offset = 0;
    uint64_t labels_offset = 0;
    uint64_t masks_offset = 0;
    for (const auto &info : infos)
    {
        uint8_t *dst_dets = dets_base_ptr + dets_offset;
        cudaError_t err = CopyBufferToDevice(
            dst_dets, info.dets_info.base, info.dets_info.byte_size,
            info.dets_info.on_device, static_cast<int>(info.dets_info.mem_type_id),
            device_id, stream);
        if (err != cudaSuccess)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
            return nullptr;
        }
        dets_offset += info.dets_info.byte_size;

        uint8_t *dst_labels = labels_base_ptr + labels_offset;
        err = CopyBufferToDevice(
            dst_labels, info.labels_info.base, info.labels_info.byte_size,
            info.labels_info.on_device, static_cast<int>(info.labels_info.mem_type_id),
            device_id, stream);
        if (err != cudaSuccess)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
            return nullptr;
        }
        labels_offset += info.labels_info.byte_size;

        uint8_t *dst_masks = masks_base_ptr + masks_offset;
        err = CopyBufferToDevice(
            dst_masks, info.masks_info.base, info.masks_info.byte_size,
            info.masks_info.on_device, static_cast<int>(info.masks_info.mem_type_id),
            device_id, stream);
        if (err != cudaSuccess)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
            return nullptr;
        }
        masks_offset += info.masks_info.byte_size;
    }

    bool input_is_half = (infos[0].dets_info.datatype == TRITONSERVER_TYPE_FP16);

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
            const size_t bytes = info.transform_info.byte_size;
            if (bytes == 0)
            {
                continue;
            }
            float *dst = transform_base_ptr + transform_offset;
            cudaError_t err = CopyBufferToDevice(
                dst, info.transform_info.base, bytes,
                info.transform_info.on_device, static_cast<int>(info.transform_info.mem_type_id),
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
        dets_base_ptr,
        labels_base_ptr,
        masks_base_ptr,
        input_is_half,
        total_images,
        num_queries,
        mask_height,
        mask_width,
        stream,
        d_transform);

    int *d_num_dets = postprocessor->num_detections_gpu();
    float *d_boxes  = postprocessor->boxes_gpu();
    float *d_scores = postprocessor->scores_gpu();
    int *d_classes  = postprocessor->classes_gpu();
    float *d_detection_masks = postprocessor->detection_masks_gpu();
    int *d_mask_offsets      = postprocessor->mask_offsets_gpu();
    int *d_mask_shapes       = postprocessor->mask_shapes_gpu();

    if (total_images > static_cast<int>(instance_state->pinned_capacity))
    {
        if (instance_state->h_num_dets_pinned != nullptr)
        {
            cudaError_t free_err = cudaFreeHost(instance_state->h_num_dets_pinned);
            if (free_err != cudaSuccess)
            {
                guard.SetError(TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(free_err)));
                return nullptr;
            }
            instance_state->h_num_dets_pinned = nullptr;
            instance_state->pinned_capacity = 0;
        }

        cudaError_t alloc_err = cudaHostAlloc(
            &instance_state->h_num_dets_pinned,
            total_images * sizeof(int),
            cudaHostAllocDefault);
        if (alloc_err != cudaSuccess)
        {
            instance_state->h_num_dets_pinned = nullptr;
            instance_state->pinned_capacity = 0;
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(alloc_err)));
            return nullptr;
        }
        instance_state->pinned_capacity = total_images;
    }

    int *h_num_dets = instance_state->h_num_dets_pinned;
    cudaError_t num_dets_err = cudaMemcpyAsync(
        h_num_dets, d_num_dets, total_images * sizeof(int),
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

    const int slot_size = kMaskOutputSize * kMaskOutputSize;

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
        const int64_t boxes_shape[3]    = {infos[i].batch_size, actual_num_dets, 4};
        const int64_t scores_shape[2]   = {infos[i].batch_size, actual_num_dets};
        const int64_t classes_shape[2]  = {infos[i].batch_size, actual_num_dets};
        const bool return_masks = config.return_masks;
        const int64_t detection_masks_shape[2] = {infos[i].batch_size,
                                                  return_masks ? actual_num_dets * slot_size : 0};
        const int64_t mask_offsets_shape[2]    = {infos[i].batch_size, actual_num_dets};
        const int64_t mask_shapes_shape[3]     = {infos[i].batch_size, actual_num_dets, 2};

        const size_t num_dets_bytes = infos[i].batch_size * sizeof(int);
        const size_t boxes_bytes    = infos[i].batch_size * actual_num_dets * 4 * sizeof(float);
        const size_t scores_bytes   = infos[i].batch_size * actual_num_dets * sizeof(float);
        const size_t classes_bytes  = infos[i].batch_size * actual_num_dets * sizeof(int);
        const size_t detection_masks_bytes = return_masks
            ? infos[i].batch_size * actual_num_dets * slot_size * sizeof(float)
            : 0;
        const size_t mask_offsets_bytes    = infos[i].batch_size * actual_num_dets * sizeof(int);
        const size_t mask_shapes_bytes     = infos[i].batch_size * actual_num_dets * 2 * sizeof(int);

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
            for (int b = 0; b < info.batch_size; ++b)
            {
                GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                    static_cast<float *>(info.boxes_buffer) + b * actual_num_dets * 4,
                    d_boxes + (offset + b) * max_detections * 4,
                    actual_num_dets * 4 * sizeof(float),
                    info.boxes_mem_type,
                    stream));

                GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                    static_cast<float *>(info.scores_buffer) + b * actual_num_dets,
                    d_scores + (offset + b) * max_detections,
                    actual_num_dets * sizeof(float),
                    info.scores_mem_type,
                    stream));

                GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                    static_cast<int *>(info.classes_buffer) + b * actual_num_dets,
                    d_classes + (offset + b) * max_detections,
                    actual_num_dets * sizeof(int),
                    info.classes_mem_type,
                    stream));

                if (config.return_masks)
                {
                    GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                        static_cast<float *>(info.detection_masks_buffer) + b * actual_num_dets * slot_size,
                        d_detection_masks + (offset + b) * max_detections * slot_size,
                        actual_num_dets * slot_size * sizeof(float),
                        info.detection_masks_mem_type,
                        stream));
                }

                GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                    static_cast<int *>(info.mask_offsets_buffer) + b * actual_num_dets,
                    d_mask_offsets + (offset + b) * max_detections,
                    actual_num_dets * sizeof(int),
                    info.mask_offsets_mem_type,
                    stream));

                GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                    static_cast<int *>(info.mask_shapes_buffer) + b * actual_num_dets * 2,
                    d_mask_shapes + (offset + b) * max_detections * 2,
                    actual_num_dets * 2 * sizeof(int),
                    info.mask_shapes_mem_type,
                    stream));
            }
        }
    }

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

} // namespace rfdetr_seg_postprocess_backend
