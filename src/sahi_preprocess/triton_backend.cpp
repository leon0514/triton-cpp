/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "sahi_preprocess/slice_impl.hpp"
#include "sahi_preprocess/triton_config.hpp"
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

#define BACKEND_NAME "sahi_preprocess"

namespace sahi_backend
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
    sahi::SliceImageConfig config;

    explicit ModelState(TRITONBACKEND_Model *model) : triton_model(model) {}

    TRITONSERVER_Error *LoadConfig()
    {
        TRITONSERVER_Message *config_message;
        RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));
        TRITONSERVER_Error *err = ParseSahiConfig(config_message, config);
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
            {
                return;
            }
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
    std::unique_ptr<sahi::SliceImage> slicer;
    cudaStream_t stream = nullptr;
    CompletionQueue completion_queue;

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
    }

    TRITONSERVER_Error *Init(ModelState *model_state)
    {
        AutoDevice auto_device(device_id);

        cudaError_t cuerr = cudaStreamCreate(&stream);
        if (cuerr != cudaSuccess)
        {
            RETURN_TRITON_ERROR(INTERNAL, cudaGetErrorString(cuerr));
        }

        slicer = std::make_unique<sahi::SliceImage>(model_state->config);
        return nullptr;
    }
};

// ===================== Request Helpers =====================

struct RequestInfo
{
    TRITONBACKEND_Request *request = nullptr;
    TRITONBACKEND_Response *response = nullptr;

    int img_width = 0;
    int img_height = 0;
    uint64_t total_input_bytes = 0;

    bool input_on_device = false;
    int64_t input_mem_type_id = 0;
    const uint8_t *input_base = nullptr;

    // 当输入不在当前 device 时，保存 host->device 的 staging buffer
    tensor::Memory<uint8_t> input_workspace;
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
            error_code = TRITONSERVER_ErrorCode(error_);
            error_message = TRITONSERVER_ErrorMessage(error_);
        }

        for (const auto &info : infos_)
        {
            TRITONSERVER_Error *cloned_error =
                TRITONSERVER_ErrorNew(error_code, error_message);

            if (info.response == nullptr)
            {
                TRITONBACKEND_Response *response = nullptr;
                TRITONSERVER_Error *new_err = TRITONBACKEND_ResponseNew(&response, info.request);
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
ExtractImageFromRequest(TRITONBACKEND_Request *request, RequestInfo &info)
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

    if (input_datatype != TRITONSERVER_TYPE_UINT8)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input data type must be UINT8");
    }

    if (input_dims_count != 3)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input must be 3-D [H, W, 3] tensor");
    }

    int h = static_cast<int>(input_shape[0]);
    int w = static_cast<int>(input_shape[1]);
    int c = static_cast<int>(input_shape[2]);

    if (c != 3)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input must have 3 channels");
    }
    if (h <= 0 || w <= 0)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input dimensions must be positive");
    }

    const uint64_t total_bytes = static_cast<uint64_t>(h) * w * c;
    if (input_byte_size != total_bytes)
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

    info.img_height = h;
    info.img_width = w;
    info.total_input_bytes = total_bytes;
    info.input_base = static_cast<const uint8_t *>(buffer);
    info.input_on_device = (mem_type == TRITONSERVER_MEMORY_GPU);
    info.input_mem_type_id = mem_type_id;

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
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, name, dtype, shape, dims));

    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, buffer, byte_size, memory_type, memory_type_id));

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
    sahi::SliceImage *slicer = instance_state->slicer.get();

    // 1. 提取所有 request 信息
    std::vector<RequestInfo> infos;
    infos.reserve(request_count);

    for (uint32_t r = 0; r < request_count; ++r)
    {
        RequestInfo info;
        TRITONSERVER_Error *err = ExtractImageFromRequest(requests[r], info);
        if (err != nullptr)
        {
            TRITONBACKEND_Response *response = nullptr;
            TRITONSERVER_Error *new_err = TRITONBACKEND_ResponseNew(&response, requests[r]);
            if (new_err == nullptr)
            {
                TRITONSERVER_Error *send_err = TRITONBACKEND_ResponseSend(
                    response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
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

    // 2. 为每个 request 准备 device 输入（跨卡时拷贝到当前 device）
    for (auto &info : infos)
    {
        if (info.input_on_device && info.input_mem_type_id == device_id)
        {
            // 零拷贝：直接复用输入指针
            continue;
        }

        // 需要拷贝到当前 device，每个 request 独立 staging buffer
        uint8_t *dst = info.input_workspace.gpu(info.total_input_bytes);
        if (dst == nullptr)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, "Failed to allocate input staging buffer"));
            return nullptr;
        }

        cudaError_t err = CopyBufferToDevice(
            dst, info.input_base, info.total_input_bytes,
            info.input_on_device, static_cast<int>(info.input_mem_type_id),
            device_id, stream);
        if (err != cudaSuccess)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
            return nullptr;
        }
        info.input_base = dst;
    }

    // 3. 对每个 request 执行切片，并创建 response / 输出 buffer
    for (auto &info : infos)
    {
        const sahi::SliceResult &result = slicer->slice(
            info.input_base, info.img_width, info.img_height, stream);

        // 同步流以获取 slice 结果
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(sync_err)));
            return nullptr;
        }

        const int slice_num = result.slice_num;
        const int slice_width = result.slice_width;
        const int slice_height = result.slice_height;

        GUARDED_RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&info.response, info.request));

        const int64_t sliced_images_shape[4] = {slice_num, slice_height, slice_width, 3};
        const int64_t slice_offsets_shape[2] = {slice_num, 4};
        const size_t sliced_images_bytes = static_cast<size_t>(slice_num) * slice_height * slice_width * 3;
        const size_t slice_offsets_bytes = static_cast<size_t>(slice_num) * 4 * sizeof(int);

        TRITONSERVER_MemoryType sliced_images_mem_type = TRITONSERVER_MEMORY_GPU;
        int64_t sliced_images_mem_type_id = device_id;
        TRITONSERVER_MemoryType slice_offsets_mem_type = TRITONSERVER_MEMORY_GPU;
        int64_t slice_offsets_mem_type_id = device_id;

        void *sliced_images_buffer = nullptr;
        void *slice_offsets_buffer = nullptr;

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            info.response, "sliced_images", TRITONSERVER_TYPE_UINT8,
            sliced_images_shape, 4, sliced_images_bytes,
            &sliced_images_buffer, &sliced_images_mem_type, &sliced_images_mem_type_id));

        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            info.response, "slice_offsets", TRITONSERVER_TYPE_INT32,
            slice_offsets_shape, 2, slice_offsets_bytes,
            &slice_offsets_buffer, &slice_offsets_mem_type, &slice_offsets_mem_type_id));

        GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
            sliced_images_buffer,
            result.d_output_images,
            sliced_images_bytes,
            sliced_images_mem_type,
            stream));

        GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
            slice_offsets_buffer,
            result.d_slice_offsets,
            slice_offsets_bytes,
            slice_offsets_mem_type,
            stream));
    }

    // 4. 记录 CUDA event，后台线程发送 response
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

} // namespace sahi_backend
