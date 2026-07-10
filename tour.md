# Triton 自定义 Backend 编写教程

> 本教程基于 `triton-cpp` 项目（包含 `preprocess`、`yolo11_postprocess`、`yolov5_postprocess` 三个后端）的实战经验，从零开始讲解如何为 NVIDIA Triton Inference Server 编写一个带 CUDA 加速的自定义后端。

---

## 目录

1. [目标读者与前置知识](#1-目标读者与前置知识)
2. [Triton Backend 是什么](#2-triton-backend-是什么)
3. [Backend 生命周期与 C API](#3-backend-生命周期与-c-api)
4. [最小可运行 Backend 示例](#4-最小可运行-backend-示例)
5. [核心机制详解](#5-核心机制详解)
   - 5.1 [状态管理：ModelState 与 ModelInstanceState](#51-状态管理-modelstate-与-modelinstancestate)
   - 5.2 [执行入口：ModelInstanceExecute](#52-执行入口-modelinstanceexecute)
   - 5.3 [Request/Response 与 Tensor](#53-requestresponse-与-tensor)
   - 5.4 [内存管理与异步响应](#54-内存管理与异步响应)
6. [CUDA 集成最佳实践](#6-cuda-集成最佳实践)
7. [config.pbtxt 与参数解析](#7-configpbtxt-与参数解析)
8. [实战：preprocess backend](#8-实战-preprocess-backend)
9. [实战：yolo11_postprocess backend](#9-实战-yolo11_postprocess-backend)
10. [实战：yolov5_postprocess backend](#10-实战-yolov5_postprocess-backend)
11. [构建与部署](#11-构建与部署)
12. [Python 客户端封装](#12-python-客户端封装)
13. [常见错误与调试](#13-常见错误与调试)
14. [最佳实践清单](#14-最佳实践清单)

---

## 1. 目标读者与前置知识

本教程假设你已具备：

- C++17 编程基础
- CUDA 编程基础（kernel、stream、event、内存拷贝）
- 对 Triton Inference Server 的基本了解（model repository、config.pbtxt）
- Linux 命令行与 CMake 基础

完成本教程后，你将能够：

- 理解 Triton Backend C API 的完整生命周期
- 编写支持 GPU 的自定义后端
- 正确处理动态 batch、异步响应、内存管理
- 将后端部署到 Triton 模型仓库

---

## 2. Triton Backend 是什么

Triton Inference Server 通过 **Backend** 来支持不同的推理框架和自定义计算逻辑。常见后端包括：

- `tensorrt_plan`：TensorRT 模型
- `onnxruntime`：ONNX 模型
- `pytorch`：PyTorch 模型
- 自定义后端：任何你能用 C/C++ 实现的逻辑

自定义后端本质上是一个共享库（`.so`），实现了 Triton 定义的 C API。Triton 在加载模型时会调用这些函数，将请求（request）交给后端处理，后端将结果打包成响应（response）返回给客户端。

对于计算机视觉流水线，常见需求包括：

- **预处理 backend**：将客户端发来的原始图像（UINT8，HWC）转换为模型输入（FP32/FP16，CHW），并记录仿射变换矩阵
- **后处理 backend**：将模型输出（如 YOLO 的 `[batch, C, anchors]`）解码为检测框、分数、类别

本项目就是围绕这两种 backend 展开的。

---

## 3. Backend 生命周期与 C API

Triton Backend C API 定义了后端的完整生命周期，每个 backend 必须实现以下函数：

```cpp
// 后端级生命周期
TRITONSERVER_Error *TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend);
TRITONSERVER_Error *TRITONBACKEND_Finalize(TRITONBACKEND_Backend *backend);

// 模型级生命周期
TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model);
TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model);

// 实例级生命周期
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance);
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance);

// 执行入口
TRITONSERVER_Error *TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance *instance,
    TRITONBACKEND_Request **requests,
    const uint32_t request_count);
```

这些函数都在 `extern "C"` 块中声明，Triton 通过符号名查找它们。

### 3.1 生命周期流程

```
Triton 启动
    │
    ▼
TRITONBACKEND_Initialize          ← 每个 backend 一次
    │
    ▼
加载模型 A
    │
    ▼
TRITONBACKEND_ModelInitialize     ← 每个模型一次
    │
    ▼
为模型 A 创建实例（instance）
    │
    ▼
TRITONBACKEND_ModelInstanceInitialize  ← 每个实例一次
    │
    ▼
接收请求 ──▶ TRITONBACKEND_ModelInstanceExecute  ← 每次推理调用
    │
    ▼
模型 A 卸载
    │
    ▼
TRITONBACKEND_ModelInstanceFinalize
TRITONBACKEND_ModelFinalize
    │
    ▼
Triton 关闭
    │
    ▼
TRITONBACKEND_Finalize
```

关键点：

- **ModelState**：在 `ModelInitialize` 中创建，保存模型级配置（如 `num_classes`、`iou_threshold`），所有实例共享
- **ModelInstanceState**：在 `ModelInstanceInitialize` 中创建，每个 GPU 实例一份，保存 CUDA stream、workspace、后处理对象等
- 一个模型可以有多个 instance（多 GPU 或多 stream），Triton 会为每个 instance 独立调用 `ModelInstanceInitialize`

---

## 4. 最小可运行 Backend 示例

下面是一个什么都不做的 "passthrough" backend，帮助理解最小结构：

```cpp
// minimal_backend.cpp
#include <triton/core/tritonbackend.h>

#define RETURN_IF_ERROR(X)                  \
    do                                      \
    {                                       \
        TRITONSERVER_Error *err__ = (X);    \
        if (err__ != nullptr) return err__; \
    } while (false)

#define RETURN_TRITON_ERROR(CODE, MSG) \
    return TRITONSERVER_ErrorNew(      \
        TRITONSERVER_Error_Code::TRITONSERVER_ERROR_##CODE, (MSG))

#define BACKEND_NAME "minimal"

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
    return nullptr;  // 返回 nullptr 表示成功
}

TRITONSERVER_Error *TRITONBACKEND_Finalize(TRITONBACKEND_Backend *)
{
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model)
{
    // 这里可以读取 config.pbtxt
    return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *)
{
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *)
{
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *)
{
    return nullptr;
}

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance *instance,
    TRITONBACKEND_Request **requests,
    const uint32_t request_count)
{
    // 每个 request 都必须有一个 response
    for (uint32_t r = 0; r < request_count; ++r)
    {
        TRITONBACKEND_Response *response = nullptr;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, requests[r]));
        TRITONSERVER_Error *send_err = TRITONBACKEND_ResponseSend(
            response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
        if (send_err != nullptr)
        {
            TRITONSERVER_ErrorDelete(send_err);
        }
    }
    return nullptr;
}

} // extern "C"
```

### 4.1 CMakeLists.txt 配置

```cmake
cmake_minimum_required(VERSION 3.18)
project(minimal_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_path(TRITON_BACKEND_HEADER_DIR
    NAMES triton/core/tritonbackend.h
    PATHS /opt/tritonserver/include /opt/tritonserver/include/triton
    NO_DEFAULT_PATH)

include_directories(${TRITON_BACKEND_HEADER_DIR})

add_library(triton_minimal SHARED minimal_backend.cpp)
set_target_properties(triton_minimal PROPERTIES
    OUTPUT_NAME "triton_minimal"
    POSITION_INDEPENDENT_CODE ON)
```

### 4.2 config.pbtxt

```protobuf
name: "minimal"
backend: "minimal"
max_batch_size: 0

input [
  { name: "input", data_type: TYPE_FP32, dims: [1] }
]
output [
  { name: "output", data_type: TYPE_FP32, dims: [1] }
]
```

### 4.3 部署

```bash
mkdir -p /opt/tritonserver/backends/minimal
cp build/libtriton_minimal.so /opt/tritonserver/backends/minimal/
mkdir -p model_repository/minimal/1
```

> **注意**：`backend` 字段值必须与 `BACKEND_NAME` 宏一致，Triton 会按这个名字去 backends 目录找 `.so` 文件。

---

## 5. 核心机制详解

### 5.1 状态管理：ModelState 与 ModelInstanceState

#### ModelState

```cpp
struct ModelState
{
    TRITONBACKEND_Model *triton_model = nullptr;
    MyConfig config;

    explicit ModelState(TRITONBACKEND_Model *model) : triton_model(model) {}

    TRITONSERVER_Error *LoadConfig()
    {
        TRITONSERVER_Message *config_message;
        RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(triton_model, 1, &config_message));
        TRITONSERVER_Error *err = ParseConfig(config_message, config);
        TRITONSERVER_MessageDelete(config_message);
        return err;
    }
};
```

在 `TRITONBACKEND_ModelInitialize` 中：

```cpp
TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model)
{
    ModelState *model_state = new ModelState(model);
    RETURN_IF_ERROR(model_state->LoadConfig());
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
        model, reinterpret_cast<void *>(model_state)));
    return nullptr;
}
```

#### ModelInstanceState

```cpp
struct ModelInstanceState
{
    TRITONBACKEND_ModelInstance *triton_instance = nullptr;
    int device_id = 0;
    cudaStream_t stream = nullptr;

    explicit ModelInstanceState(TRITONBACKEND_ModelInstance *instance)
        : triton_instance(instance)
    {
        TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id);
    }

    ~ModelInstanceState()
    {
        if (stream != nullptr)
            cudaStreamDestroy(stream);
    }

    TRITONSERVER_Error *Init(ModelState *model_state)
    {
        AutoDevice auto_device(device_id);
        cudaStreamCreate(&stream);
        // 预分配 workspace、初始化处理对象...
        return nullptr;
    }
};
```

在 `TRITONBACKEND_ModelInstanceInitialize` 中：

```cpp
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
```

### 5.2 执行入口：ModelInstanceExecute

`TRITONBACKEND_ModelInstanceExecute` 是后端最核心的函数。Triton 会把一个或多个 request 一起交给它处理，后端需要：

1. 解析每个 request 的输入
2. 创建 response 并分配输出 buffer
3. 执行实际计算（CPU/GPU）
4. 发送 response

```cpp
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance *instance,
    TRITONBACKEND_Request **requests,
    const uint32_t request_count)
{
    // 获取实例状态
    ModelInstanceState *instance_state;
    TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void **>(&instance_state));

    // 切换 CUDA device
    int device_id;
    TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id);
    AutoDevice auto_device(device_id);

    // 处理每个 request...
    for (uint32_t r = 0; r < request_count; ++r)
    {
        // 解析输入
        // 创建 response
        // 分配输出
        // 计算
        // 发送 response
    }

    return nullptr;
}
```

### 5.3 Request/Response 与 Tensor

#### 读取输入

```cpp
TRITONBACKEND_Input *input;
TRITONBACKEND_RequestInputByIndex(request, 0, &input);

const char *input_name;
TRITONSERVER_DataType input_datatype;
const int64_t *input_shape;
uint32_t input_dims_count;
uint32_t input_buffer_count;
uint64_t input_byte_size;

TRITONBACKEND_InputProperties(
    input, &input_name, &input_datatype, &input_shape,
    &input_dims_count, &input_byte_size, &input_buffer_count);

const void *buffer;
TRITONSERVER_MemoryType mem_type;
int64_t mem_type_id;
TRITONBACKEND_InputBuffer(
    input, 0, &buffer, &input_byte_size, &mem_type, &mem_type_id);
```

#### 分配输出

```cpp
TRITONBACKEND_Output *output;
TRITONBACKEND_ResponseOutput(
    response, &output, "output0", TRITONSERVER_TYPE_FP32, shape, dims);

void *output_buffer;
TRITONSERVER_MemoryType output_mem_type;
int64_t output_mem_type_id;
TRITONBACKEND_OutputBuffer(
    output, &output_buffer, byte_size, &output_mem_type, &output_mem_type_id);
```

#### 发送响应

```cpp
TRITONBACKEND_ResponseSend(
    response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
```

如果有错误：

```cpp
TRITONSERVER_Error *err = TRITONSERVER_ErrorNew(
    TRITONSERVER_ERROR_INVALID_ARG, "bad input shape");
TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
TRITONSERVER_ErrorDelete(err);
```

> **关键**：`ResponseSend` **不会** 接管 `TRITONSERVER_Error` 的所有权，发送后必须手动 `TRITONSERVER_ErrorDelete(err)`。

### 5.4 内存管理与异步响应

#### 问题场景

如果 `Execute` 函数直接调用 `cudaStreamSynchronize`，Triton 调度线程会被阻塞，无法并发处理其他请求。最佳实践是：

1. 把所有 GPU 操作提交到 CUDA stream
2. 记录一个 CUDA event
3. 启动后台线程等待 event
4. `Execute` 立即返回
5. 后台线程在 GPU 完成后发送 response

#### 实现模式

```cpp
struct CompletionTask
{
    std::vector<TRITONBACKEND_Response *> responses;
    cudaEvent_t event = nullptr;
};

class CompletionQueue
{
public:
    void Push(CompletionTask task)
    {
        EnsureStarted();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            queue_.push(std::move(task));
        }
        cv_.notify_one();
    }

    void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    void Run()
    {
        AutoDevice auto_device(device_id_);
        while (true)
        {
            CompletionTask task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });
                if (shutdown_ && queue_.empty()) return;
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
                    auto *err = TRITONBACKEND_ResponseSend(
                        response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
                    if (err != nullptr) TRITONSERVER_ErrorDelete(err);
                }
            }
        }
    }

    int device_id_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<CompletionTask> queue_;
    bool shutdown_ = false;
    std::thread worker_;
};
```

在 `Execute` 中：

```cpp
// 提交所有 GPU 工作后
cudaEvent_t event;
cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
cudaEventRecord(event, stream);

CompletionTask task;
task.event = event;
for (auto &info : infos)
    task.responses.push_back(info.response);

instance_state->completion_queue.Push(std::move(task));
```

#### ResponseGuard：异常安全

Triton 要求每个 request 必须有且仅有一次 `ResponseSend`。如果在执行过程中途出错，必须确保所有 request 的 response 都被发送。

```cpp
class ResponseGuard
{
public:
    explicit ResponseGuard(const std::vector<RequestInfo> &infos) : infos_(infos) {}

    ~ResponseGuard()
    {
        if (committed_) return;

        // 用统一的错误码和信息，为每个未完成的 response 创建独立错误对象
        auto code = error_ ? TRITONSERVER_ErrorCode(error_) : TRITONSERVER_ERROR_INTERNAL;
        const char *msg = error_ ? TRITONSERVER_ErrorMessage(error_) : "Unknown error";

        for (const auto &info : infos_)
        {
            auto *cloned = TRITONSERVER_ErrorNew(code, msg);
            auto *response = info.response ? info.response : CreateResponseForRequest(info.request);
            auto *send_err = TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, cloned);
            if (send_err) TRITONSERVER_ErrorDelete(send_err);
            TRITONSERVER_ErrorDelete(cloned);
        }

        if (error_) TRITONSERVER_ErrorDelete(error_);
    }

    void SetError(TRITONSERVER_Error *error)
    {
        if (error_ == nullptr) error_ = error;
        else TRITONSERVER_ErrorDelete(error);
    }

    void Commit() { committed_ = true; }

private:
    const std::vector<RequestInfo> &infos_;
    TRITONSERVER_Error *error_ = nullptr;
    bool committed_ = false;
};
```

使用方式：

```cpp
ResponseGuard guard(infos);

// 各种可能失败的操作...
if (something_wrong)
{
    guard.SetError(TRITONSERVER_ErrorNew(...));
    return nullptr;
}

// 一切成功，提交后台任务
guard.Commit();
```

> **注意**：`guard` 必须在 `infos` 之后声明，这样析构时 `infos` 还在生命周期内。

---

## 6. CUDA 集成最佳实践

### 6.1 AutoDevice

Triton 实例可能运行在不同 GPU 上，`TRITONBACKEND_ModelInstanceDeviceId` 返回实例所属的 device id。执行任何 CUDA 操作前都要切到对应设备：

```cpp
class AutoDevice
{
public:
    explicit AutoDevice(int device_id)
    {
        cudaGetDevice(&prev_device_);
        if (prev_device_ != device_id)
            cudaSetDevice(device_id);
    }

    ~AutoDevice()
    {
        int current;
        cudaGetDevice(&current);
        if (current != prev_device_)
            cudaSetDevice(prev_device_);
    }

private:
    int prev_device_ = 0;
};
```

### 6.2 Stream

每个 instance 应该有自己的 CUDA stream：

```cpp
cudaStreamCreate(&stream);
```

所有 kernel、memcpy、event 都使用这个 stream，避免阻塞默认流。

### 6.3 内存池

频繁 `cudaMalloc`/`cudaFree` 会带来同步开销。项目中使用 `tensor::Memory<T>` 封装了一个带引用计数的内存池，容量不足时自动扩容：

```cpp
tensor::Memory<float> workspace_;

// 预分配
workspace_.gpu(max_elements);

// 运行时按需扩容
float *d_ptr = workspace_.gpu(needed_elements);
```

### 6.4 输入连续化

Triton 的多个 request 可能各自位于不同 GPU 或 CPU 内存。为了单次 kernel launch，通常需要把它们拷贝到一个连续的 device workspace：

```cpp
uint8_t *input_base_ptr = instance_state->input_workspace_.gpu(total_input_bytes);

uint64_t offset = 0;
for (const auto &info : infos)
{
    cudaMemcpyKind kind = info.input_on_device
                              ? cudaMemcpyDeviceToDevice
                              : cudaMemcpyHostToDevice;
    cudaMemcpyAsync(input_base_ptr + offset, info.input_base,
                    info.total_input_bytes, kind, stream);
    offset += info.total_input_bytes;
}
```

---

## 7. config.pbtxt 与参数解析

### 7.1 典型配置

```protobuf
name: "yolov5_postprocess"
backend: "yolov5_postprocess"
max_batch_size: 16

input [
  {
    name: "model_output"
    data_type: TYPE_FP32
    dims: [85, 8400]
  }
]

output [
  { name: "num_dets",         data_type: TYPE_INT32, dims: [1] },
  { name: "detection_boxes",  data_type: TYPE_FP32, dims: [300, 4] },
  { name: "detection_scores", data_type: TYPE_FP32, dims: [300] },
  { name: "detection_classes", data_type: TYPE_INT32, dims: [300] }
]

parameters: {
  key: "num_classes"
  value: { string_value: "80" }
}
parameters: {
  key: "confidence_threshold"
  value: { string_value: "0.25" }
}

instance_group [
  { count: 1, kind: KIND_GPU, gpus: [0] }
]
```

### 7.2 参数解析代码

Triton 提供 `TRITONBACKEND_ModelConfig` 获取序列化后的 JSON，然后自行解析：

```cpp
TRITONSERVER_Error *ParseConfig(
    TRITONSERVER_Message *model_config_message,
    MyConfig &config)
{
    const char *buffer;
    size_t byte_size;
    RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
        model_config_message, &buffer, &byte_size));

    nlohmann::json model_config = nlohmann::json::parse(
        std::string(buffer, byte_size));

    const auto &parameters = model_config.value("parameters", nlohmann::json::object());

    auto it = parameters.find("num_classes");
    if (it != parameters.end())
    {
        config.num_classes = std::stoi(
            it->value()["string_value"].get<std::string>());
    }

    // 更多参数...

    return nullptr;
}
```

### 7.3 参数统一用 string_value

Triton 的 `parameters` 只支持 `string_value`，所以即使参数是数字，也要写成 `"80"`，后端再用 `std::stoi`/`std::stof` 转换。

---

## 8. 实战：preprocess backend

`preprocess` 后端将客户端发来的 `raw_image`（UINT8 HWC）转换为模型输入（FP32/FP16 CHW），并输出 `transform_metadata`（d2i 仿射逆矩阵，用于后处理将坐标映射回原图）。

### 8.1 核心流程

1. 读取输入：原始图像，动态 shape `[N, H, W, 3]`
2. 根据配置计算目标尺寸、归一化参数、resize 方式
3. 使用 CUDA kernel 对每个 batch 图像做仿射变换 + 归一化
4. 输出：
   - `preprocessed_output`：`[N, 3, target_height, target_width]`
   - `transform_metadata`：`[N, 6]`

### 8.2 关键点

- 使用 3D CUDA grid：`x/y` 覆盖目标像素，`z` 覆盖总图像数
- 单次 kernel launch 处理所有图像
- 动态 batch：`config.pbtxt` 中声明单图维度，运行时按实际 `N` 分配输出

### 8.3 仿射变换

letterbox resize 需要计算 `src -> dst` 的仿射矩阵，并保存其逆矩阵 `d2i`，后处理用它将网络坐标映射回原图。

```cpp
// 典型 d2i 矩阵（2x3 按行优先展开为 6 个 float）
float d2i[6] = { ... };
```

---

## 9. 实战：yolo11_postprocess backend

`yolo11_postprocess` 将 YOLO11 模型输出解码为检测框。

### 9.1 输入输出

- 输入：`[N, 84, 8400]` 或 `[N, 8400, 84]`（FP32/FP16）
  - 84 = 4 box + 80 class
- 输出：
  - `num_dets`：`[N, 1]`
  - `detection_boxes`：`[N, 300, 4]`（x1, y1, x2, y2）
  - `detection_scores`：`[N, 300]`
  - `detection_classes`：`[N, 300]`

### 9.2 CUDA 算法步骤

1. **decode_filter_kernel**：每个线程处理一个 `(batch, anchor)`
   - 读取 `cx, cy, w, h` 并转换为 `x1, y1, x2, y2`
   - 在 80 个 class logit 中找最大值和 class_id
   - 若 `max_logit < conf_thresh` 则丢弃
   - 否则写入候选缓冲区

2. **cap_counts_kernel**：限制候选数不超过 `max_candidates`

3. **thrust::sort**：按 score 降序排序每图候选

4. **nms_kernel**：每图一个线程，按 score 从高到低做 class-aware NMS

### 9.3 配置参数

```protobuf
parameters: {
  key: "output_format"
  value: { string_value: "channel_first" }  # 或 "anchor_first"
}
parameters: {
  key: "score_activation"
  value: { string_value: "none" }           # 或 "sigmoid"
}
```

---

## 10. 实战：yolov5_postprocess backend

`yolov5_postprocess` 与 `yolo11_postprocess` 结构几乎一致，但额外支持 YOLOv5 经典导出格式中的 **objectness** 分支。

### 10.1 输入输出

- 输入：
  - 含 objectness：`[N, 85, 8400]` 或 `[N, 8400, 85]`
    - 85 = 4 box + 1 obj_conf + 80 class
  - 不含 objectness：`[N, 84, 8400]` 或 `[N, 8400, 84]`（与 YOLO11 相同）
- 输出：与 YOLO11 完全一致

### 10.2 新增配置

```protobuf
parameters: {
  key: "has_objectness"
  value: { string_value: "true" }   # YOLOv5 经典格式
}
```

### 10.3 分数计算

```cpp
float obj_conf = has_objectness ? read_input(..., 4) : 1.0f;
if (apply_sigmoid) obj_conf = sigmoid(obj_conf);

float max_cls_logit = max over class logits;
float cls_conf = apply_sigmoid ? sigmoid(max_cls_logit) : max_cls_logit;

float score = obj_conf * cls_conf;
```

---

## 11. 构建与部署

### 11.1 构建

**必须使用 Triton SDK 镜像**，运行时镜像没有后端开发头文件。

```bash
cd triton-cpp

docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:25.01-py3-sdk \
    bash workspace/build.sh
```

### 11.2 一个容易踩的坑

> ⚠️ **不要在容器内的 `/workspace` 目录下用相对路径执行 `bash workspace/build.sh`**。
>
> 因为 `workspace/build.sh` 内部通过 `$(dirname "$0")` 计算项目根目录。如果你在 `/workspace` 下执行 `bash workspace/build.sh`，`$0` 是 `workspace/build.sh`，`dirname` 得到 `workspace`，于是 build 目录被设为 `/workspace/workspace/build`，CMake 会去找 `/workspace/workspace/CMakeLists.txt`，导致报错：
>
> ```
> CMake Error: The source directory "/workspace/workspace" does not appear to contain CMakeLists.txt.
> ```
>
> **正确做法**：从项目根目录执行 `bash workspace/build.sh`（此时项目根目录就是当前目录），或者在容器内切换到 `/workspace` 后执行 `bash build.sh`（将 build.sh 复制到根目录时）。更稳妥的改法是使用 `realpath`：
>
> ```bash
> SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
> ```

### 11.3 部署

#### 方式一：作为独立 backend

```bash
mkdir -p /opt/tritonserver/backends/yolov5_postprocess
cp build/libtriton_yolov5_postprocess.so /opt/tritonserver/backends/yolov5_postprocess/
cp -r workspace/models/yolov5_postprocess <model_repository>/
```

#### 方式二：放在模型版本目录

```bash
mkdir -p <model_repository>/yolov5_postprocess/1
cp build/libtriton_yolov5_postprocess.so <model_repository>/yolov5_postprocess/1/
cp workspace/models/yolov5_postprocess/config.pbtxt <model_repository>/yolov5_postprocess/
```

### 11.4 Ensemble 配置

实际生产环境通常用 ensemble 串联 preprocess、模型、postprocess：

```protobuf
name: "yolov5_ensemble"
platform: "ensemble"
max_batch_size: 16

ensemble_scheduling {
  step [
    {
      model_name: "preprocess"
      input_map { key: "raw_image" value: "raw_image" }
      output_map { key: "preprocessed_output" value: "preprocessed_output" }
      output_map { key: "transform_metadata" value: "transform_metadata" }
    },
    {
      model_name: "yolov5"
      input_map { key: "images" value: "preprocessed_output" }
      output_map { key: "output0" value: "output0" }
    },
    {
      model_name: "yolov5_postprocess"
      input_map { key: "model_output" value: "output0" }
      output_map { key: "num_dets" value: "num_dets" }
      output_map { key: "detection_boxes" value: "detection_boxes" }
      output_map { key: "detection_scores" value: "detection_scores" }
      output_map { key: "detection_classes" value: "detection_classes" }
    }
  ]
}
```

---

## 12. Python 客户端封装

项目提供了 `workspace/triton_client` 这一 Python 客户端封装，统一支持
gRPC、HTTP 和系统共享内存（SHM）三种调用方式。所有脚本均在 `workspace`
目录下执行。

### 12.1 目录与文件

```
workspace/triton_client/
├── __init__.py        # 导出 TritonClient、TritonClientError
├── client.py          # 核心实现
├── example.py         # 三种协议调用示例
├── test_client.py     # 单元测试 + 集成测试
├── benchmark.py       # 时间/性能测试
└── README.md
```

### 12.2 基本用法

```python
import numpy as np
from PIL import Image
from triton_client import TritonClient

img = Image.open("images/bus.jpg").convert("RGB")
img_np = np.array(img)[np.newaxis, ...]

outputs = [
    "num_dets",
    "detection_boxes",
    "detection_scores",
    "detection_classes",
    "transform_metadata",
]

# gRPC / HTTP
with TritonClient("localhost:48001", protocol="grpc") as client:
    result = client.infer(
        model_name="yolov5_ensemble",
        inputs={"raw_image": img_np},
        outputs=outputs,
    )

# 共享内存需要预先指定输出 shape / dtype
output_specs = {
    "num_dets": ([1, 1], "int32"),
    "detection_boxes": ([1, 300, 4], "float32"),
    "detection_scores": ([1, 300], "float32"),
    "detection_classes": ([1, 300], "int32"),
    "transform_metadata": ([1, 6], "float32"),
}

with TritonClient("localhost:48000", protocol="shm") as client:
    result = client.infer(
        model_name="yolov5_ensemble",
        inputs={"raw_image": img_np},
        outputs=outputs,
        output_specs=output_specs,
    )
```

### 12.3 关键特性

- 同一套 `infer()` API 适配三种协议。
- `inputs` 为 `dict[str, np.ndarray]`，`outputs` 为输出名列表。
- SHM 模式自动创建 / 复用 / 销毁共享内存区域。
- SHM 模式下返回的输出数组会从共享内存拷贝一份，关闭 client 后仍可安全使用。
- 支持 `is_server_ready`、`is_model_ready`、`get_model_metadata` 等辅助方法。

### 12.4 运行脚本

```bash
# 示例
python3 triton_client/example.py --protocol grpc
python3 triton_client/example.py --protocol http
python3 triton_client/example.py --protocol shm

# 测试
python3 triton_client/test_client.py
python3 triton_client/test_client.py --integration

# 性能对比
python3 triton_client/benchmark.py --protocol grpc --count 100 --warmup 10
python3 triton_client/benchmark.py --protocol http --count 100 --warmup 10
python3 triton_client/benchmark.py --protocol shm --count 100 --warmup 10
```

---

## 13. 常见错误与调试

### 13.1 "Unexpected backend name"

`BACKEND_NAME` 宏与 `config.pbtxt` 中的 `backend` 字段不一致。例如：

```cpp
#define BACKEND_NAME "yolov5_postprocess"
```

```protobuf
backend: "yolov5_postprocess"
```

### 13.2 "Input must be 3-D [N, C, A] or [N, A, C] tensor"

动态 batch 下，Triton 会在声明维度前追加 batch 维。如果 `config.pbtxt` 中 `max_batch_size > 0`，声明 `dims: [85, 8400]`，运行时实际 shape 是 `[N, 85, 8400]`。后端代码必须按 3-D 处理。

### 13.3 Response 泄漏 / Triton 卡住

每个 request 必须有且仅有一次 `ResponseSend`。如果中途 `return` 导致某些 request 没有 response，Triton 会挂起或报错。使用 `ResponseGuard` 可以解决这个问题。

### 13.4 CUDA 设备不一致

Triton 实例可能绑定到不同 GPU。创建 stream、分配显存、启动 kernel 前都要 `AutoDevice auto_device(device_id)`。后台完成线程也要切到对应设备。

### 13.5 输入 byte size 不匹配

检查 `TRITONBACKEND_InputProperties` 返回的 shape 和 byte_size 是否与你预期的一致。FP16 时每个元素 2 字节，FP32 时 4 字节。

### 13.6 thrust::sort 在 stream 上同步

`thrust::sort` 默认使用同步语义，但会按传入的 device pointer 推断执行策略。在 stream 上异步排序需要显式指定 `thrust::cuda::par.on(stream)`。本项目目前每张图单独 `thrust::sort`，在 Execute 路径中调用，虽然会引入同步，但候选数通常只有几百，开销可控。

---

## 14. 最佳实践清单

- [ ] `BACKEND_NAME` 与 `config.pbtxt` 的 `backend` 字段一致
- [ ] 所有 C API 函数在 `extern "C"` 中
- [ ] 正确管理 `TRITONSERVER_Error` 所有权：发送后必须 `ErrorDelete`
- [ ] 每个 request 必须有且仅有一次 `ResponseSend`
- [ ] 使用 `ResponseGuard` 处理异常路径
- [ ] 每个 instance 有独立的 CUDA stream
- [ ] 执行 CUDA 操作前 `AutoDevice` 切到实例所在 GPU
- [ ] 后台线程也要在正确的 GPU 上等待 event
- [ ] 尽量使用异步 CUDA + event + 后台线程，避免 `Execute` 阻塞
- [ ] 预分配 workspace，避免热路径上的 `cudaMalloc`
- [ ] 动态 batch 下输出 buffer 按实际 `batch_size` 分配
- [ ] 输入 buffer 可能是 CPU 或 GPU，正确选择 `cudaMemcpyKind`
- [ ] 多 request 时把输入拷贝到连续 device workspace 再单次 launch kernel
- [ ] 参数统一用 `string_value`，后端自行转换为数值类型
- [ ] 用 SDK 镜像编译，运行时镜像没有 `tritonbackend.h`
- [ ] 注意 `build.sh` 的相对路径问题，建议使用绝对路径或 `realpath`

---

## 结语

编写 Triton 自定义 backend 的核心是：**理解生命周期、管理好 Request/Response、正确使用 CUDA 异步执行**。本项目的 `preprocess`、`yolo11_postprocess`、`yolov5_postprocess` 三个后端展示了完整模式，新增一个后端时通常只需：

1. 复制一个已有后端目录
2. 修改 namespace、类名、backend name
3. 修改 CUDA kernel 算法
4. 更新 `CMakeLists.txt`
5. 添加对应的 `workspace/models/<name>/config.pbtxt`

按照这个模式，你可以快速扩展出新的预处理、后处理或任意自定义计算后端。
