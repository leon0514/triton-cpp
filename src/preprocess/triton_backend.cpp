/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "preprocess/preprocess_impl.hpp"
#include "preprocess/triton_config.hpp"
#include "common/device.hpp"

#include <triton/core/tritonbackend.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define BACKEND_NAME "preprocess"

namespace preprocess_backend
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
    preprocess::PreprocessConfig config;

    explicit ModelState(TRITONBACKEND_Model *model) : triton_model(model) {}

    TRITONSERVER_Error *LoadConfig()
    {
        TRITONSERVER_Message *config_message;
        RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));
        TRITONSERVER_Error *err = ParsePreprocessConfig(config_message, config);
        TRITONSERVER_MessageDelete(config_message);
        return err;
    }
};

// ===================== Async Response Completion =====================

/**
 * @brief 可复用 device workspace 池。
 *
 * 每个 Execute 可能需要一个 device staging buffer（host 输入、BatchItem 等）。
 * 使用固定大小的池可以避免每次 Execute 都调用 cudaMalloc，同时支持
 * 后台完成线程尚未结束时新的 Execute 进入。
 */
template <typename T>
class WorkspacePool
{
  public:
    void seed(size_t count, size_t initial_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        buffers_.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            auto buffer = std::make_unique<tensor::Memory<T>>();
            buffer->gpu(initial_count);
            buffers_.push_back(std::move(buffer));
            free_indices_.push(i);
        }
    }

    // 返回一个可用 buffer 的索引，并保证其元素容量 >= min_count。
    size_t acquire(size_t min_count)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_indices_.empty())
        {
            size_t idx = buffers_.size();
            auto buffer = std::make_unique<tensor::Memory<T>>();
            buffer->gpu(min_count);
            buffers_.push_back(std::move(buffer));
            return idx;
        }

        size_t idx = free_indices_.front();
        free_indices_.pop();
        if (buffers_[idx]->gpu_size() < min_count)
        {
            buffers_[idx]->gpu(min_count);
        }
        return idx;
    }

    tensor::Memory<T> share(size_t idx)
    {
        tensor::Memory<T> m;
        m.set_shared_memory(*buffers_[idx]);
        return m;
    }

    void release(size_t idx)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_indices_.push(idx);
    }

  private:
    std::mutex mutex_;
    // 使用 unique_ptr 存储 buffer 对象，避免 vector 扩容时移动对象本身
    // 导致任何潜在的内部地址/引用失效。
    std::vector<std::unique_ptr<tensor::Memory<T>>> buffers_;
    std::queue<size_t> free_indices_;
};

/**
 * @brief 异步响应任务。
 *
 * 持有本次 Execute 所需的 device workspace 所有权，以及待发送的 response。
 * 后台线程等待 CUDA event 完成后发送 response，任务析构时自动释放 workspace。
 * 这样 Execute 函数无需同步 CUDA stream 即可返回，恢复 Triton 异步调度性能。
 */
struct CompletionTask
{
    std::vector<TRITONBACKEND_Response *> responses;
    cudaEvent_t event = nullptr;

    // 持有 host->device 输入 staging buffer，确保 GPU 使用期间内存不被释放或覆盖。
    tensor::Memory<uint8_t> input_workspace;
    WorkspacePool<uint8_t> *input_pool = nullptr;
    size_t input_pool_index = static_cast<size_t>(-1);

    // 持有预处理输出/变换矩阵的 device workspace，供 CopyOutputToResponse 使用。
    tensor::Memory<uint8_t> output_workspace;
    WorkspacePool<uint8_t> *output_pool = nullptr;
    size_t output_pool_index = static_cast<size_t>(-1);

    tensor::Memory<float> transform_workspace;
    WorkspacePool<float> *transform_pool = nullptr;
    size_t transform_pool_index = static_cast<size_t>(-1);

    // 持有 BatchItem device buffer，确保 kernel 完成前不被覆盖。
    tensor::Memory<preprocess::BatchItem> items_workspace;
    WorkspacePool<preprocess::BatchItem> *items_pool = nullptr;
    size_t items_pool_index = static_cast<size_t>(-1);
};

class CompletionQueue
{
  public:
    CompletionQueue() = default;

    void SetDeviceId(int device_id)
    {
        device_id_ = device_id;
    }

    // 显式停止后台线程并等待其退出。
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
        // 后台线程需要切换到该实例对应的 CUDA device，
        // 因为 cudaEvent 是与创建它的 device 绑定的。
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

            // 仅阻塞后台完成线程，等待本次 batch 的 GPU 工作结束，
            // 随后发送响应。Triton Execute 线程已提前返回。
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

            // 释放各 workspace 到池中，供后续 Execute 复用。
            if (task.input_pool != nullptr && task.input_pool_index != static_cast<size_t>(-1))
            {
                task.input_pool->release(task.input_pool_index);
            }
            if (task.output_pool != nullptr && task.output_pool_index != static_cast<size_t>(-1))
            {
                task.output_pool->release(task.output_pool_index);
            }
            if (task.transform_pool != nullptr && task.transform_pool_index != static_cast<size_t>(-1))
            {
                task.transform_pool->release(task.transform_pool_index);
            }
            if (task.items_pool != nullptr && task.items_pool_index != static_cast<size_t>(-1))
            {
                task.items_pool->release(task.items_pool_index);
            }

            // task 离开作用域，workspace 引用计数归零后释放。
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

/**
 * @brief 模型实例状态。
 *
 * 注意：析构时必须先显式停止 completion_queue（join 后台线程），
 * 然后再销毁 stream，避免后台线程仍在引用该流。
 */
struct ModelInstanceState
{
    TRITONBACKEND_ModelInstance *triton_instance = nullptr;
    int device_id = 0;
    std::unique_ptr<preprocess::Preprocess> preprocessor;
    cudaStream_t stream = nullptr;
    CompletionQueue completion_queue;
    WorkspacePool<uint8_t> input_pool;
    WorkspacePool<uint8_t> output_pool;
    WorkspacePool<float> transform_pool;
    WorkspacePool<preprocess::BatchItem> items_pool;

    explicit ModelInstanceState(TRITONBACKEND_ModelInstance *instance)
        : triton_instance(instance)
    {
        TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id);
        completion_queue.SetDeviceId(device_id);
    }

    ~ModelInstanceState()
    {
        // 先停止后台线程，确保其不再使用 stream 上的 event。
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

        preprocessor = std::make_unique<preprocess::Preprocess>(model_state->config);

        // 预分配 2 个 host->device 输入 staging buffer。
        const auto &config = preprocessor->config();
        const int max_batch = config.max_batch_size > 0 ? config.max_batch_size : 16;
        const size_t initial_input_bytes = static_cast<size_t>(max_batch) *
                                           config.target_width * config.target_height * 3;
        input_pool.seed(2, initial_input_bytes);
        items_pool.seed(2, static_cast<size_t>(max_batch));
        output_pool.seed(2, static_cast<size_t>(max_batch) * preprocessor->output_bytes_per_image());
        transform_pool.seed(2, static_cast<size_t>(max_batch) * 6);

        return nullptr;
    }
};

// ===================== Request Helpers =====================

struct RequestInfo
{
    TRITONBACKEND_Request *request = nullptr;
    TRITONBACKEND_Response *response = nullptr;

    // 该 request 中包含的图像数量（Triton 动态 batch 后 N >= 1）
    int batch_size = 1;

    // 单张图像尺寸
    int img_height = 0;
    int img_width  = 0;

    // 该 request 的总输入字节数（batch_size * H * W * 3）
    uint64_t total_input_bytes = 0;

    // 该 request 在整体 batch 中的起始图像索引
    int image_offset = 0;

    bool input_on_device = false;
    int64_t input_mem_type_id = 0;

    // 输入 buffer（host 或 device）
    const uint8_t *input_base = nullptr;

    // 输出 buffer 及其内存类型
    void *output_buffer = nullptr;
    TRITONSERVER_MemoryType output_mem_type = TRITONSERVER_MEMORY_CPU;
    int64_t output_mem_type_id = 0;

    // transform buffer 及其内存类型
    void *transform_buffer = nullptr;
    TRITONSERVER_MemoryType transform_mem_type = TRITONSERVER_MEMORY_CPU;
    int64_t transform_mem_type_id = 0;
};

/**
 * @brief RAII 守卫，确保 Execute 异常退出时所有 Request 都能收到 Response。
 *
 * Triton 要求每个进入 ModelInstanceExecute 的 Request 必须有且仅有一次 ResponseSend。
 * 该守卫在析构时检查是否已提交（Commit）。若未提交，说明执行路径中途失败，
 * 它会为 infos 中所有 Request 创建或发送带错误码的 Response，防止客户端挂起。
 */
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

        // 为每个失败请求克隆独立的错误实例，避免将同一个 error_ 多次传入
        // ResponseSend 导致 use-after-free / double-free。
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
                    // Triton 不接管错误对象所有权，必须显式释放。
                    TRITONSERVER_ErrorDelete(cloned_error);
                }
                else
                {
                    // Response 都创建失败了，只能释放错误对象，该请求仍会挂起，
                    // 但这是 Triton 内部资源耗尽，已无法做更多。
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
                // Triton 不接管错误对象所有权，必须显式释放。
                TRITONSERVER_ErrorDelete(cloned_error);
            }
        }

        // 所有响应发送完毕后，释放原始 error_。
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
            // 只保留第一个错误，后续错误释放避免泄漏
            TRITONSERVER_ErrorDelete(error);
        }
    }

    void Commit() { committed_ = true; }

  private:
    const std::vector<RequestInfo> &infos_;
    TRITONSERVER_Error *error_ = nullptr;
    bool committed_ = false;
};

/**
 * @brief 从单个 request 提取图像信息。
 *
 * 支持 Triton 动态 Batch 契约：
 * - 配置文件声明单张图维度 [H, W, 3]
 * - 底层实际接收 [N, H, W, 3]（N 为 batch size，N=1 时表示未合并）
 */
static TRITONSERVER_Error *
ExtractImageFromRequest(
    TRITONBACKEND_Request *request,
    RequestInfo &info)
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

    if (input_dims_count != 3 && input_dims_count != 4)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input must be 3-D [H,W,C] or 4-D [N,H,W,C] tensor");
    }

    int n = 1;
    int h, w, c;
    if (input_dims_count == 3)
    {
        h = static_cast<int>(input_shape[0]);
        w = static_cast<int>(input_shape[1]);
        c = static_cast<int>(input_shape[2]);
    }
    else
    {
        n = static_cast<int>(input_shape[0]);
        h = static_cast<int>(input_shape[1]);
        w = static_cast<int>(input_shape[2]);
        c = static_cast<int>(input_shape[3]);
    }

    if (c != 3)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input must have 3 channels");
    }

    if (n <= 0 || h <= 0 || w <= 0)
    {
        RETURN_TRITON_ERROR(INVALID_ARG, "Input dimensions must be positive");
    }

    const uint64_t img_bytes = static_cast<uint64_t>(h * w * c);
    const uint64_t total_bytes = img_bytes * n;
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

    info.batch_size        = n;
    info.img_height        = h;
    info.img_width         = w;
    info.total_input_bytes = total_bytes;
    info.input_base        = static_cast<const uint8_t *>(buffer);
    info.input_on_device   = (mem_type == TRITONSERVER_MEMORY_GPU);
    info.input_mem_type_id = mem_type_id;

    return nullptr;
}

// ===================== Response Helpers =====================

static TRITONSERVER_DataType OutputDtype(preprocess::OutputType type)
{
    return (type == preprocess::OutputType::FP16)
               ? TRITONSERVER_TYPE_FP16
               : TRITONSERVER_TYPE_FP32;
}

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
// Triton loads this entrypoint by name, so it must have C linkage.

extern "C" {

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance *instance,
    TRITONBACKEND_Request **requests,
    const uint32_t request_count)
{
    ModelInstanceState *instance_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void **>(&instance_state)));

    int device_id;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
    AutoDevice auto_device(device_id);

    cudaStream_t stream                  = instance_state->stream;
    preprocess::Preprocess *preprocessor = instance_state->preprocessor.get();
    const auto &config                   = preprocessor->config();

    const size_t bytes_per_image = preprocessor->output_bytes_per_image();

    // 1. 提取所有 request 信息；解析失败的直接在此发送错误响应，不加入 infos。
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
                // Triton 不接管 err 所有权，发送后必须显式释放。
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
                // Response 创建失败，释放 new_err 和原始的 err，避免泄漏。
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

    // RAII 守卫：确保 Execute 异常退出时，infos 中所有 Request 都能收到 Response。
    // 必须在 infos 之后声明，以保证先析构 guard（发送失败响应），再析构 infos。
    ResponseGuard guard(infos);

    // 局部宏：出错时设置 guard 错误并返回。guard 析构会统一发送失败响应。
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

    // 2. 为有效 request 创建 response，并提前分配输出 buffer。
    //    当 max_batch_size > 0 时，Triton 会在配置文件的声明维度前隐式追加 batch 维。
    //    因此运行时输出 shape 必须为 4-D [N, 3, H, W] 和 2-D [N, 6]，
    //    其中 N 是当前 request 的实际 batch_size，绝不能硬编码为 1。
    const TRITONSERVER_DataType output_dtype = OutputDtype(config.output_type);

    for (int i = 0; i < request_num; ++i)
    {
        GUARDED_RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&infos[i].response, infos[i].request));

        // 优先申请 GPU 显存，提升链式调用吞吐量；无法满足时 Triton 会自动回退。
        infos[i].output_mem_type    = TRITONSERVER_MEMORY_GPU;
        infos[i].output_mem_type_id = device_id;
        infos[i].transform_mem_type    = TRITONSERVER_MEMORY_GPU;
        infos[i].transform_mem_type_id = device_id;

        const int64_t output_shape[4]    = {
            infos[i].batch_size,
            3,
            preprocessor->target_height(),
            preprocessor->target_width()};
        const int64_t transform_shape[2] = {infos[i].batch_size, 6};

        const size_t output_byte_size = infos[i].batch_size * bytes_per_image;
        GUARDED_RETURN_IF_ERROR(AllocateOutput(
            infos[i].response, "preprocessed_output", output_dtype,
            output_shape, 4, output_byte_size,
            &infos[i].output_buffer, &infos[i].output_mem_type, &infos[i].output_mem_type_id));

        if (config.output_transform)
        {
            const size_t transform_byte_size = infos[i].batch_size * 6 * sizeof(float);
            GUARDED_RETURN_IF_ERROR(AllocateOutput(
                infos[i].response, "transform_metadata", TRITONSERVER_TYPE_FP32,
                transform_shape, 2, transform_byte_size,
                &infos[i].transform_buffer, &infos[i].transform_mem_type, &infos[i].transform_mem_type_id));
        }

    }

    // 3. 计算整体 batch 规模和各 workspace 大小。
    int total_images = 0;
    uint64_t total_host_input_bytes = 0;

    for (auto &info : infos)
    {
        info.image_offset = total_images;
        total_images += info.batch_size;

        if (!info.input_on_device)
        {
            total_host_input_bytes += info.total_input_bytes;
        }
    }

    if (total_images == 0)
    {
        // 无有效图像，直接发送空响应（防御性分支，正常情况不会进入）
        for (auto &info : infos)
        {
            TRITONBACKEND_ResponseSend(
                info.response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
        }
        guard.Commit();
        return nullptr;
    }

    // 4. 从池中获取 host->device 输入 staging buffer。
    tensor::Memory<uint8_t> input_workspace;
    size_t input_pool_index = static_cast<size_t>(-1);
    uint8_t *input_base_ptr = nullptr;

    if (total_host_input_bytes > 0)
    {
        input_pool_index = instance_state->input_pool.acquire(total_host_input_bytes);
        input_workspace  = instance_state->input_pool.share(input_pool_index);
        input_base_ptr   = input_workspace.gpu();
        if (input_base_ptr == nullptr)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, "Failed to acquire input staging buffer"));
            return nullptr;
        }
    }

    // 5. 从池中获取 BatchItem / 输出 / transform 的 device workspace。
    size_t items_pool_index = static_cast<size_t>(-1);
    tensor::Memory<preprocess::BatchItem> items_workspace;
    preprocess::BatchItem *d_items_base = nullptr;

    size_t output_pool_index = static_cast<size_t>(-1);
    tensor::Memory<uint8_t> output_workspace;
    uint8_t *output_base = nullptr;

    size_t transform_pool_index = static_cast<size_t>(-1);
    tensor::Memory<float> transform_workspace;
    float *transform_base = nullptr;

    if (total_images > 0)
    {
        items_pool_index = instance_state->items_pool.acquire(total_images);
        items_workspace  = instance_state->items_pool.share(items_pool_index);
        d_items_base     = items_workspace.gpu();
        if (d_items_base == nullptr)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, "Failed to acquire BatchItem buffer"));
            return nullptr;
        }

        output_pool_index = instance_state->output_pool.acquire(total_images * bytes_per_image);
        output_workspace  = instance_state->output_pool.share(output_pool_index);
        output_base       = output_workspace.gpu();
        if (output_base == nullptr)
        {
            guard.SetError(TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL, "Failed to acquire output workspace"));
            return nullptr;
        }

        if (config.output_transform)
        {
            transform_pool_index = instance_state->transform_pool.acquire(total_images * 6);
            transform_workspace  = instance_state->transform_pool.share(transform_pool_index);
            transform_base       = transform_workspace.gpu();
            if (transform_base == nullptr)
            {
                guard.SetError(TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL, "Failed to acquire transform workspace"));
                return nullptr;
            }
        }
    }

    // 6. 逐个 request 处理：
    //    - 输入已在当前 GPU 上时直接复用（零拷贝）。
    //    - host 输入或位于其他 GPU 的输入，统一拷贝到共享的 device staging buffer
    //      的不同 offset，避免跨卡访问。
    //    - 输出先写入共享的 device workspace，再异步拷贝到 response buffer，
    //      这样同时支持 Triton 分配的 GPU/CPU 输出 buffer（HTTP 响应通常为 CPU）。
    uint64_t input_offset  = 0;
    int items_offset       = 0;
    int output_item_offset = 0;
    for (const auto &info : infos)
    {
        const uint64_t img_bytes = static_cast<uint64_t>(info.img_height * info.img_width * 3);

        std::vector<preprocess::ImageDesc> req_images;
        req_images.reserve(info.batch_size);

        const bool need_device_copy = !info.input_on_device ||
            (info.input_on_device && info.input_mem_type_id != device_id);

        if (need_device_copy)
        {
            uint8_t *dst = input_base_ptr + input_offset;
            cudaError_t cuerr = CopyBufferToDevice(
                dst, info.input_base, info.total_input_bytes,
                info.input_on_device, static_cast<int>(info.input_mem_type_id),
                device_id, stream);
            if (cuerr != cudaSuccess)
            {
                guard.SetError(TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(cuerr)));
                return nullptr;
            }

            for (int n = 0; n < info.batch_size; ++n)
            {
                preprocess::ImageDesc image;
                image.data      = dst + n * img_bytes;
                image.width     = info.img_width;
                image.height    = info.img_height;
                image.line_size = info.img_width * 3;
                req_images.push_back(image);
            }

            input_offset += info.total_input_bytes;
        }
        else
        {
            for (int n = 0; n < info.batch_size; ++n)
            {
                preprocess::ImageDesc image;
                image.data      = info.input_base + n * img_bytes;
                image.width     = info.img_width;
                image.height    = info.img_height;
                image.line_size = info.img_width * 3;
                req_images.push_back(image);
            }
        }

        preprocessor->forward(
            req_images.data(),
            info.batch_size,
            output_base + output_item_offset * bytes_per_image,
            config.output_transform ? transform_base + output_item_offset * 6 : nullptr,
            stream,
            d_items_base + items_offset);

        items_offset += info.batch_size;
        output_item_offset += info.batch_size;
    }

    // 7. 将结果从 workspace 异步拷贝到各 response 的 output buffer。
    for (const auto &info : infos)
    {
        const int offset      = info.image_offset;
        const size_t out_size = info.batch_size * bytes_per_image;

        GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
            info.output_buffer,
            output_base + offset * bytes_per_image,
            out_size,
            info.output_mem_type,
            stream));

        if (config.output_transform)
        {
            const size_t trans_size = info.batch_size * 6 * sizeof(float);
            GUARDED_RETURN_IF_ERROR(CopyOutputToResponse(
                info.transform_buffer,
                transform_base + offset * 6,
                trans_size,
                info.transform_mem_type,
                stream));
        }
    }

    // 8. 记录 CUDA event，将响应发送交给后台线程，Execute 立即返回。
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

    // 将 workspace 所有权转移给后台任务，确保 GPU 完成前内存不被释放或覆盖。
    task.input_workspace       = std::move(input_workspace);
    task.input_pool            = &instance_state->input_pool;
    task.input_pool_index      = input_pool_index;
    task.output_workspace      = std::move(output_workspace);
    task.output_pool           = &instance_state->output_pool;
    task.output_pool_index     = output_pool_index;
    task.transform_workspace   = std::move(transform_workspace);
    task.transform_pool        = &instance_state->transform_pool;
    task.transform_pool_index  = transform_pool_index;
    task.items_workspace       = std::move(items_workspace);
    task.items_pool            = &instance_state->items_pool;
    task.items_pool_index      = items_pool_index;

    instance_state->completion_queue.Push(std::move(task));

    // 任务已成功提交给后台线程，主线程可以安全返回。
    // 此时标记 guard 已提交，析构时不再重复发送响应。
    guard.Commit();
    return nullptr;

    #undef GUARDED_RETURN_IF_ERROR
}

} // extern "C"

// ===================== Backend C API =====================
// Triton loads these entrypoints by name, so they must have C linkage.

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

} // namespace preprocess_backend
