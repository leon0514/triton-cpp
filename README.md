# 图像预处理  triton preprocess cuda backend
1. 支持配置目标图像宽高
原因： 基础功能，必须支持不同模型对分辨率的要求。
2. 支持配置 std \ mean 归一化
原因： 几乎所有主流深度学习网络（如 ResNet、YOLO、Transformer 等）都需要该步骤。
3. 支持配置 resize 方式 (Letter box \ 直接 resize)
原因： 分类/分割模型常用直接 Resize，检测模型常用 Letter box，两类主流任务均需兼容。
4. 支持动态 Batch
原因： Triton 吞吐量优化的核心机制，必须在 CUDA Kernel 中支持并发处理多图。
5. ROI 裁剪与预处理一体化
原因： 在“检测+识别”的双阶段任务（如人脸、车牌）中，此功能可减少大量不必要的显存拷贝和单独裁剪算子的开销，性能提升显著。
6. 零拷贝、CUDA IPC / CPU 共享内存
原因： CUDA 后端的灵魂所在。如果不使用共享内存和零拷贝，CUDA 预处理带来的性能优势会被高昂的进程间通信（IPC）拷贝开销完全抵消。
7. 支持通道排布转换与 RGB/BGR 转换
原因： 绝大多数输入为 HWC (RGB/BGR)，而模型推理需要 NCHW (RGB 或 BGR)。将 HWC->CHW 转换与 RGB↔BGR 转换融合在同一个 CUDA 缩放 Kernel 中，效率最高。
8. 支持多数据类型输出 (FP32 \ FP16)
原因： TensorRT 常用 FP16 推理，直接输出 FP16 可以避免推理前再做一次类型转换。
9. 支持输出变换元数据 (Transform Metadata)
原因： 目标检测等任务在获取模型输出后，必须依赖这些参数（如缩放比例和 Padding 偏移量）将坐标还原到原图。若不输出，后处理将无法正常进行。


# TRITON 注意事项
在 Triton 中实现和部署自定义的 C++ CUDA 预处理后端（Custom Backend），除了算法本身，还需要特别注意与 Triton 架构对齐的系统级细节。
以下是开发和部署过程中需要重点注意的事项：
1. CUDA 流 (Stream) 的绑定与异步执行
规避默认流： 切勿在 CUDA Kernel 中使用默认流（stream 0 或 nullptr）。默认流是同步的，会导致 Triton 内的所有并发实例（Model Instances）序列化执行，严重降低服务吞吐量。
正确做法： 通过 Triton API 获取当前实例分配的 CUDA Stream，并在启动 Kernel 和进行内存拷贝（如 cudaMemcpyAsync）时传入该 Stream：
2. 显存管理：避免使用原生 cudaMalloc
内存碎片与耗时： 在推理请求中直接调用 cudaMalloc 和 cudaFree 会导致严重的显存碎片，且显存申请是同步阻塞操作，会大幅拉高延迟。
正确做法：
所有的输出张量（Output Tensors），必须使用 Triton 提供的 API（如 TRITONBACKEND_OutputBuffer）申请显存。这能让 Triton 自动应用其内部的显存池，并支持零拷贝。
如果预处理内部需要临时辅助显存，建议在模型实例初始化（TRITONBACKEND_ModelInstanceInitialize）时一次性申请好（Workspace），在推理时复用该显存，或者使用 Triton 提供的 CudaMemoryManager。
3. 输入数据位置（Host / Device）的兼容性
内存类型判断： 客户端发送的数据可能在 CPU（System Memory），也可能通过共享内存或 GPU 显存（CUDA Device Memory）传输。
正确做法： 预处理后端必须在运行时查询输入张量的内存类型：
TRITONSERVER_MemoryType memory_type;
int64_t memory_type_id;
TRITONBACKEND_InputProperties(input, ..., &memory_type, &memory_type_id, ...);
如果输入在 CPU，需要先通过 cudaMemcpyAsync 异步拷贝到 GPU 后，再调用预处理 Kernel；如果已经在 GPU，则直接进行预处理。
4. 容错处理：绝不能调用 exit() 或 throw
进程崩溃风险： 预处理后端是作为动态链接库（.so）加载到 Triton 进程中的。任何未捕获的 C++ 异常、assert 失败或直接调用 exit() 都会导致整个 Triton 服务进程崩溃，影响同服务器上的其他模型。
正确做法：
使用 Triton 的错误处理宏（如 RETURN_IF_TRITONSERVER_ERROR）。
遇到格式错误、输入图片损坏等异常时，应仅向当前请求返回错误状态（通过 TRITONBACKEND_ResponseSend 传递错误码），确保服务整体的稳定性。
5. 动态 Batch 的请求对齐 (Request Packing)
批处理逻辑： Triton 的动态 Batch 会将多个独立的客户端请求（Requests）组合成一个 Batch 传入后端。
正确做法：
在 Execute 函数中，需要遍历该 Batch 内的所有 Requests。
必须将多张图的指针、尺寸等信息，整合后一次性送入 CUDA Kernel，实现单次 Kernel 启动处理整个 Batch，而不是对 Batch 内的图片循环启动 Kernel。
6. 配置对齐与版本兼容性
配置校验： 预处理后端的输入输出张量名称、数据类型必须在 config.pbtxt 中严格声明。
正确做法： 在 TRITONBACKEND_ModelInitialize 阶段，解析并校验 config.pbtxt，确保配置的参数（如 mean/std 数组长度是否为 3）符合预期。如果校验失败，应拒绝加载模型并给出明确报错，防止运行时出错。


# 构建与部署

## 目录结构
```
.
├── CMakeLists.txt                  # 根目录构建配置
├── build.sh                        # 构建脚本
├── src/
│   ├── common/                     # 基础工具代码（memory、norm、affine、json 等）
│   └── preprocess/                 # 预处理后端源码
│       ├── preprocess_kernel.cu    # CUDA 核函数
│       ├── preprocess_impl.*       # 预处理封装
│       ├── triton_config.*         # Triton 配置解析
│       └── triton_backend.cpp      # Triton 后端 C API
└── workspace/models/preprocess/    # Triton 模型配置示例
    ├── config.pbtxt
    └── 1/
```

## 编译

**必须使用 Triton SDK 镜像进行编译。** 运行时镜像（如 `nvcr.io/nvidia/tritonserver:25.01-py3`）不包含后端开发头文件 `triton/core/tritonbackend.h`，无法编译自定义后端。

推荐使用官方 SDK 镜像，通过 `-v` 挂载本地源码：

```bash
# 进入项目根目录
cd triton-cpp

# 启动 SDK 容器并挂载当前目录到 /workspace
# 注意：必须是 -py3-sdk 结尾的 SDK 镜像，而不是 -py3 运行时镜像
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:25.01-py3-sdk \
    bash build.sh
```

构建产物位于 `build/libtriton_preprocess.so`。

## 部署

将构建产物和模型配置拷贝到 Triton 模型仓库的 backend 目录：

```bash
# 方式一：作为独立 backend
mkdir -p /opt/tritonserver/backends/preprocess
cp build/libtriton_preprocess.so /opt/tritonserver/backends/preprocess/
cp -r workspace/models/preprocess <model_repository>/

# 方式二：放在模型版本目录下（backend 名称为 preprocess）
mkdir -p <model_repository>/preprocess/1
cp build/libtriton_preprocess.so <model_repository>/preprocess/1/
cp workspace/models/preprocess/config.pbtxt <model_repository>/preprocess/
```

启动 Triton：

```bash
tritonserver --model-repository <model_repository>
```

## 配置参数说明

`workspace/models/preprocess/config.pbtxt` 中通过 `parameters` 字段配置预处理行为：

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `target_width` | int | 目标宽度 | `640` |
| `target_height` | int | 目标高度 | `640` |
| `resize_type` | string | `direct` 直接缩放 / `letterbox`  letterbox | `letterbox` |
| `output_type` | string | `FP32` / `FP16` | `FP32` |
| `norm_type` | string | `none` / `mean_std` / `alpha_beta` | `mean_std` |
| `mean` | float[3] | mean_std 归一化的 mean | `[0.485, 0.456, 0.406]` |
| `std` | float[3] | mean_std 归一化的 std | `[0.229, 0.224, 0.225]` |
| `alpha` | float | mean_std 的缩放因子 / alpha_beta 的 alpha | `0.00392156862745098` |
| `beta` | float | alpha_beta 的 beta | `0.0` |
| `channel_type` | string | `none` / `swap_rb`（输入 BGR，输出 RGB） | `swap_rb` |
| `fill_value` | float[3] | letterbox 填充色 BGR | `[114.0, 114.0, 114.0]` |
| `output_transform` | bool | 是否输出 d2i 变换矩阵 | `true` |

## 输入输出

配置文件遵循 Triton 动态 Batch 契约（`max_batch_size > 0`），声明的是**单张图像**的维度：

- 输入：`raw_image`，`TYPE_UINT8`，声明形状 `[H, W, 3]`，运行时可能为 `[N, H, W, 3]`，BGR 顺序。
- 输出：`preprocessed_output`，`TYPE_FP32/FP16`，声明形状 `[3, target_height, target_width]`，运行时实际返回 `[N, 3, target_height, target_width]`。
- 输出：`transform_metadata`，`TYPE_FP32`，声明形状 `[6]`，运行时实际返回 `[N, 6]`，为 `d2i` 仿射逆变换矩阵，用于将网络坐标映射回原图坐标。

其中 `N` 是当前 request 的实际 batch size，由后端根据输入动态确定，**绝不能硬编码为 1**。

## 实现要点

### 1. 动态 Batch 与单次 Kernel Launch
`config.pbtxt` 中设置 `max_batch_size: 16`，输入/输出 dims 声明单张图维度。

`TRITONBACKEND_ModelInstanceExecute` 收集所有 request 的图像（支持每个 request 内部再带 batch 维度 `[N, H, W, 3]`），通过 3D CUDA Grid（x/y 覆盖目标像素，z 覆盖总图像数）单次启动 `warp_affine_batched_kernel`，避免 per-image 循环启动 kernel 的开销。

由于 `max_batch_size > 0`，Triton 会在配置声明维度前隐式追加 batch 维。因此向单个 Response 分配输出时，必须使用运行时 4-D 形状 `[N, 3, H, W]` 和 2-D 形状 `[N, 6]`，其中 `N` 取当前 request 的 `batch_size`，确保声明形状与 buffer 字节数严格对齐。

### 2. Host 输入数据安全
对于仍在 Host 的输入，不再复用单个 device buffer，而是为每次 Execute 分配一块独立的连续 device workspace，并按顺序将每张图拷贝到不同 offset。配合 `CompletionTask` 持有 workspace 所有权，保证 GPU 使用期间不会被后续 Execute 覆盖或提前释放。

### 3. 异步响应恢复 Triton 流水线
`Execute` 函数不再调用 `cudaStreamSynchronize` 阻塞调度线程。GPU 完成后通过 `cudaEvent` 通知后台完成线程，由该线程负责调用 `TRITONBACKEND_ResponseSend`。`Execute` 提交任务后立即返回，使 Triton 可以并发调度下一个 batch。

### 4. Workspace 生命周期管理
每次 Execute 的 `input_workspace`、`output_workspace`、`transform_workspace` 随 `CompletionTask` 移动到后台线程，待 `cudaEventSynchronize` 确认 GPU 完成后，任务析构自动释放 device 内存，避免跨 batch 的数据竞争。

### 5. 异常路径下的 Request 防挂起（RAII ResponseGuard）
Triton 要求每个进入 `ModelInstanceExecute` 的 Request 必须有且仅有一次 `ResponseSend`。任何中途错误（如显存分配失败、输出 buffer 分配失败、CUDA 拷贝失败）都不得直接 `return` 退出。

`triton_backend.cpp` 中实现了 `ResponseGuard` RAII 守卫：
- 在 `infos` 之后声明，析构顺序上先于 `infos`
- 正常路径成功提交后台任务后调用 `Commit()`，析构时不再动作
- 异常路径下，`guard.SetError()` 记录错误并 `return nullptr`，触发 `ResponseGuard` 析构
- 析构时遍历所有未发送或已创建但未成功完成的 Response
- **关键**：`TRITONBACKEND_ResponseSend` 会接管 error 所有权，不能将同一个 error 指针重复传给多个 ResponseSend。`ResponseGuard` 在循环内部通过 `TRITONSERVER_ErrorCode` / `TRITONSERVER_ErrorMessage` 为每个请求**克隆独立的错误对象**，分别传入 ResponseSend；循环结束后再安全释放原始 `error_`
