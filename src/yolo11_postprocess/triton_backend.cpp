/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_postprocess/yolo11_postprocess_impl.hpp"
#include "yolo11_postprocess/triton_config.hpp"
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

#define BACKEND_NAME "yolo11_postprocess"

namespace yolo11_postprocess_backend
{

#define RETURN_IF_ERROR(X)               \
    do                                   \
    {                                    \
        TRITONSERVER_Error *err__ = (X); \
        if (err__ != nullptr)            \
        {                                \
            return err__;                \
        }                                \
    } while (false)

#define RETURN_TRITON_ERROR(CODE, MSG) \
    return TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_##CODE, (MSG))

    // ===================== State Management =====================

    struct ModelState
    {
        TRITONBACKEND_Model *triton_model = nullptr;
        yolo11_postprocess::Yolo11PostprocessConfig config;

        explicit ModelState(TRITONBACKEND_Model *model) : triton_model(model) {}

        TRITONSERVER_Error *LoadConfig()
        {
            TRITONSERVER_Message *config_message;
            RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));
            TRITONSERVER_Error *err = ParseYolo11PostprocessConfig(config_message, config);
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
            AutoDevice auto_device(device_id_);

            while (true)
            {
                CompletionTask task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]
                             { return shutdown_ || !queue_.empty(); });
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
        std::unique_ptr<yolo11_postprocess::Yolo11Postprocess> postprocessor;
        cudaStream_t stream = nullptr;
        tensor::Memory<uint8_t> input_workspace_;
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
            // 先停止后台线程，确保其不再使用 stream 上的 event。
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

            postprocessor = std::make_unique<yolo11_postprocess::Yolo11Postprocess>(
                model_state->config);

            // 预分配输入 workspace：max_batch * 8400 * (4+num_classes) * FP32
            const auto &cfg = model_state->config;
            size_t max_input_elements = static_cast<size_t>(cfg.max_batch_size) *
                                        8400 * (cfg.num_classes + 4);
            input_workspace_.gpu(max_input_elements * sizeof(float));

            pinned_capacity = static_cast<size_t>(cfg.max_batch_size);
            cuerr = cudaMallocHost(&h_num_dets_pinned, pinned_capacity * sizeof(int));
            if (cuerr != cudaSuccess)
            {
                RETURN_TRITON_ERROR(INTERNAL, cudaGetErrorString(cuerr));
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
        int num_anchors = 0;
        int actual_num_dets = 0;
        uint64_t total_input_bytes = 0;
        int image_offset = 0;

        TRITONSERVER_DataType input_datatype = TRITONSERVER_TYPE_FP32;
        bool input_on_device = false;
        int64_t input_mem_type_id = 0;
        const void *input_base = nullptr;

        bool transform_on_device = false;
        int64_t transform_mem_type_id = 0;
        const void *transform_base = nullptr;

        void *num_dets_buffer = nullptr;
        void *boxes_buffer = nullptr;
        void *scores_buffer = nullptr;
        void *classes_buffer = nullptr;

        TRITONSERVER_MemoryType num_dets_mem_type = TRITONSERVER_MEMORY_CPU;
        TRITONSERVER_MemoryType boxes_mem_type = TRITONSERVER_MEMORY_CPU;
        TRITONSERVER_MemoryType scores_mem_type = TRITONSERVER_MEMORY_CPU;
        TRITONSERVER_MemoryType classes_mem_type = TRITONSERVER_MEMORY_CPU;

        int64_t num_dets_mem_type_id = 0;
        int64_t boxes_mem_type_id = 0;
        int64_t scores_mem_type_id = 0;
        int64_t classes_mem_type_id = 0;
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
    ExtractInputsFromRequest(
        TRITONBACKEND_Request *request,
        RequestInfo &info,
        int num_classes,
        bool anchors_first)
    {
        info.request = request;

        uint32_t input_count;
        RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));
        if (input_count != 2)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "Exactly two input tensors per request are required: model_output and transform_metadata");
        }

        // 1. model_output
        TRITONBACKEND_Input *model_input;
        RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "model_output", &model_input));

        const char *input_name;
        TRITONSERVER_DataType input_datatype;
        const int64_t *input_shape;
        uint32_t input_dims_count;
        uint32_t input_buffer_count;
        uint64_t input_byte_size;

        RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
            model_input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));

        if (input_datatype != TRITONSERVER_TYPE_FP32 &&
            input_datatype != TRITONSERVER_TYPE_FP16)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "model_output data type must be FP32 or FP16");
        }

        if (input_dims_count != 3)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "model_output must be 3-D [N, C, A] or [N, A, C] tensor");
        }

        int n = static_cast<int>(input_shape[0]);
        int dim1 = static_cast<int>(input_shape[1]);
        int dim2 = static_cast<int>(input_shape[2]);
        int expected_channels = num_classes + 4;

        if (anchors_first)
        {
            if (dim2 != expected_channels)
            {
                RETURN_TRITON_ERROR(INVALID_ARG, "model_output last dim must equal 4 + num_classes");
            }
            info.num_anchors = dim1;
        }
        else
        {
            if (dim1 != expected_channels)
            {
                RETURN_TRITON_ERROR(INVALID_ARG, "model_output channel dim must equal 4 + num_classes");
            }
            info.num_anchors = dim2;
        }

        if (n <= 0 || info.num_anchors <= 0)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "model_output dimensions must be positive");
        }

        int num_channels = expected_channels;
        int total_elements = n * info.num_anchors * num_channels;
        size_t expected_bytes = total_elements *
                                (input_datatype == TRITONSERVER_TYPE_FP16 ? sizeof(uint16_t) : sizeof(float));
        if (input_byte_size != expected_bytes)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "model_output byte size mismatch");
        }

        if (input_buffer_count != 1)
        {
            RETURN_TRITON_ERROR(INVALID_ARG, "model_output buffer count must be 1");
        }

        const void *buffer;
        TRITONSERVER_MemoryType mem_type;
        int64_t mem_type_id;
        RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
            model_input, 0, &buffer, &input_byte_size, &mem_type, &mem_type_id));

        info.batch_size = n;
        info.total_input_bytes = input_byte_size;
        info.input_base = buffer;
        info.input_on_device = (mem_type == TRITONSERVER_MEMORY_GPU);
        info.input_mem_type_id = mem_type_id;
        info.input_datatype = input_datatype;

        // 2. transform_metadata
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

        info.transform_base = transform_buffer;
        info.transform_on_device = (transform_mem_type == TRITONSERVER_MEMORY_GPU);
        info.transform_mem_type_id = transform_mem_type_id;

        return nullptr;
    }

    // ===================== Response Helpers =====================

    static TRITONSERVER_DataType OutputDtypeForFloat()
    {
        return TRITONSERVER_TYPE_FP32;
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

    // 2-D strided copy for workspace layouts where the source has a larger
    // pitch than the destination (e.g. workspace [N, max_dets, 4] with
    // stride max_dets*4, but output only needs [N, actual_dets, 4]).
    static TRITONSERVER_Error *
    CopyOutputStrided2D(
        void *dst,
        const void *src,
        size_t width_bytes,
        size_t src_pitch_bytes,
        size_t dst_pitch_bytes,
        size_t height,
        TRITONSERVER_MemoryType memory_type,
        cudaStream_t stream)
    {
        cudaMemcpyKind kind = (memory_type == TRITONSERVER_MEMORY_GPU)
                                  ? cudaMemcpyDeviceToDevice
                                  : cudaMemcpyDeviceToHost;

        cudaError_t err = cudaMemcpy2DAsync(
            dst, dst_pitch_bytes, src, src_pitch_bytes,
            width_bytes, height, kind, stream);
        if (err != cudaSuccess)
        {
            RETURN_TRITON_ERROR(INTERNAL, cudaGetErrorString(err));
        }

        return nullptr;
    }

    // ===================== Execute =====================

    extern "C"
    {

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
            yolo11_postprocess::Yolo11Postprocess *postprocessor = instance_state->postprocessor.get();
            const auto &config = postprocessor->config();
            const int max_detections = postprocessor->max_detections();

            // 1. 提取所有 request 信息
            std::vector<RequestInfo> infos;
            infos.reserve(request_count);

            for (uint32_t r = 0; r < request_count; ++r)
            {
                RequestInfo info;
                TRITONSERVER_Error *err = ExtractInputsFromRequest(
                    requests[r], info, config.num_classes, config.anchors_first);

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

#define GUARDED_RETURN_IF_ERROR(X)       \
    do                                   \
    {                                    \
        TRITONSERVER_Error *err__ = (X); \
        if (err__ != nullptr)            \
        {                                \
            guard.SetError(err__);       \
            return nullptr;              \
        }                                \
    } while (false)

            // 2. 为有效 request 创建 response（输出 buffer 在得到实际 num_dets 后再分配）
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
            }

            // 3. 计算整体 batch 规模并统一 num_anchors
            int total_images = 0;
            uint64_t total_input_bytes = 0;
            int num_anchors = infos[0].num_anchors;

            for (auto &info : infos)
            {
                info.image_offset = total_images;
                total_images += info.batch_size;
                total_input_bytes += info.total_input_bytes;

                if (info.num_anchors != num_anchors)
                {
                    guard.SetError(TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        "All requests must have the same num_anchors"));
                    return nullptr;
                }
            }

            // 4. 准备 device 输入：
            //    单个 request 且输入已在当前 GPU 上时（ensemble 链中上游模型的输出即为此情形），
            //    直接透传指针零拷贝；否则统一拷贝到实例预分配的连续 workspace。
            const void *d_input = nullptr;
            bool input_is_half = false;

            if (request_num == 1 && infos[0].input_on_device &&
                infos[0].input_mem_type_id == device_id)
            {
                d_input = infos[0].input_base;
                input_is_half = (infos[0].input_datatype == TRITONSERVER_TYPE_FP16);
            }
            else
            {
                uint8_t *input_base_ptr = instance_state->input_workspace_.gpu(total_input_bytes);
                if (total_input_bytes > 0 && input_base_ptr == nullptr)
                {
                    guard.SetError(TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL, "Failed to allocate input device workspace"));
                    return nullptr;
                }

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
                        guard.SetError(TRITONSERVER_ErrorNew(
                            TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(err)));
                        return nullptr;
                    }
                    input_offset += info.total_input_bytes;
                }

                input_is_half = (infos[0].input_datatype == TRITONSERVER_TYPE_FP16);
                d_input = input_base_ptr;
            }

            // 5. 准备 transform_metadata device buffer
            float *d_transform = nullptr;
            {
                const size_t transform_bytes = static_cast<size_t>(total_images) * 6 * sizeof(float);
                float *transform_base_ptr = instance_state->transform_workspace_.gpu(transform_bytes / sizeof(float));
                if (transform_bytes > 0 && transform_base_ptr == nullptr)
                {
                    guard.SetError(TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL, "Failed to allocate transform device workspace"));
                    return nullptr;
                }

                uint64_t transform_offset = 0;
                for (const auto &info : infos)
                {
                    const size_t bytes = static_cast<size_t>(info.batch_size) * 6 * sizeof(float);
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

            // 6. 执行后处理（输出写入实例预分配的 GPU workspace）
            postprocessor->forward(
                d_input,
                input_is_half,
                total_images,
                num_anchors,
                stream,
                d_transform);

            int *d_num_dets = postprocessor->num_detections_gpu();
            float *d_boxes = postprocessor->boxes_gpu();
            float *d_scores = postprocessor->scores_gpu();
            int *d_classes = postprocessor->classes_gpu();

            // 7. 把每个样本的实际 num_dets 拷贝到锁页内存
            if (static_cast<size_t>(total_images) > instance_state->pinned_capacity)
            {
                if (instance_state->h_num_dets_pinned != nullptr)
                {
                    cudaFreeHost(instance_state->h_num_dets_pinned);
                }
                instance_state->pinned_capacity = total_images;
                cudaError_t cuerr = cudaMallocHost(&instance_state->h_num_dets_pinned, total_images * sizeof(int));
                if (cuerr != cudaSuccess)
                {
                    guard.SetError(TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL, "Failed to reallocate pinned memory"));
                    return nullptr;
                }
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

                const size_t num_dets_bytes = infos[i].batch_size * sizeof(int);
                const size_t boxes_bytes = infos[i].batch_size * actual_num_dets * 4 * sizeof(float);
                const size_t scores_bytes = infos[i].batch_size * actual_num_dets * sizeof(float);
                const size_t classes_bytes = infos[i].batch_size * actual_num_dets * sizeof(int);

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
                    // Workspace uses [total_images, max_detections, ...] strided layout.
                    // Output buffer is contiguous [batch_size, actual_num_dets, ...].
                    // Use 2-D pitched copy to skip padding between images in workspace.
                    GUARDED_RETURN_IF_ERROR(CopyOutputStrided2D(
                        info.boxes_buffer,
                        d_boxes + offset * max_detections * 4,
                        actual_num_dets * 4 * sizeof(float), // width per image
                        max_detections * 4 * sizeof(float),  // src pitch (workspace stride)
                        actual_num_dets * 4 * sizeof(float), // dst pitch (output stride)
                        info.batch_size,                     // height (num images)
                        info.boxes_mem_type,
                        stream));

                    GUARDED_RETURN_IF_ERROR(CopyOutputStrided2D(
                        info.scores_buffer,
                        d_scores + offset * max_detections,
                        actual_num_dets * sizeof(float),
                        max_detections * sizeof(float),
                        actual_num_dets * sizeof(float),
                        info.batch_size,
                        info.scores_mem_type,
                        stream));

                    GUARDED_RETURN_IF_ERROR(CopyOutputStrided2D(
                        info.classes_buffer,
                        d_classes + offset * max_detections,
                        actual_num_dets * sizeof(int),
                        max_detections * sizeof(int),
                        actual_num_dets * sizeof(int),
                        info.batch_size,
                        info.classes_mem_type,
                        stream));
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

            // 输入/输出 workspace 均由 ModelInstanceState 持有并复用，
            // 只需等待 event 完成即可发送响应。
            instance_state->completion_queue.Push(std::move(task));

            guard.Commit();
            return nullptr;

#undef GUARDED_RETURN_IF_ERROR
        }

    } // extern "C"

    // ===================== Backend C API =====================

    extern "C"
    {

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

} // namespace yolo11_postprocess_backend
