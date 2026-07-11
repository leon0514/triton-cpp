# 构建与部署

## 编译

**必须使用 Triton SDK 镜像进行编译。** 运行时镜像（如 `nvcr.io/nvidia/tritonserver:25.01-py3`）不包含后端开发头文件 `triton/core/tritonbackend.h`，无法编译自定义后端。

推荐使用官方 SDK 镜像，通过 `-v` 挂载本地源码：

```bash
# 进入项目根目录
cd triton-cpp

# 启动 SDK 容器并挂载当前目录到 /workspace
docker run --rm -it --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    nvcr.io/nvidia/tritonserver:25.01-py3 \
    bash build.sh
```

构建产物位于 `build/libtriton_preprocess.so`、`build/libtriton_yolo11_postprocess.so`、`build/libtriton_yolo11_pose_postprocess.so`、`build/libtriton_yolov5_postprocess.so`，分别对应四个后端。

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
