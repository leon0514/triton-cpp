# Triton 客户端封装

`triton_client` 是对官方 `tritonclient` 的轻量级统一封装，同一套 Python API 即可切换 gRPC、HTTP 以及基于 HTTP 的系统共享内存（SHM）三种推理协议。它隐藏了不同协议在输入构造、输出解析、共享内存注册等方面的差异，适用于需要快速对比协议性能或在生产环境做稳定推理调用的场景。

## 功能特性

- **统一 API**：`TritonClient.infer()` 统一支持 `grpc` / `http` / `shm`。
- **零拷贝共享内存**：`shm` 模式通过系统共享内存传输输入/输出张量，显著降低大输入或大 batch 的延迟。
- **自动资源回收**：支持 `with` 上下文管理器，退出时自动注销并销毁共享内存区域。
- **输入 SHM 复用**：相同 shape 的输入会复用已创建的共享内存，跨请求无需反复 `register/unregister`。
- **输出安全拷贝**：SHM 模式下返回的 `ndarray` 会先拷贝到普通内存，关闭 client 后仍可安全使用。
- **动态输出友好**：返回完整输出张量，可按照后端返回的 `num_dets` 对检测结果进行切片。

## 目录结构

```
workspace/triton_client/
├── __init__.py        # 导出 TritonClient、TritonClientError
├── client.py          # 核心封装实现
├── example.py         # 三协议调用示例
├── test_client.py     # 单元测试 + 集成测试
├── benchmark.py       # 时间/性能测试
└── README.md
```

## 依赖

```bash
pip install numpy pillow tritonclient[all]
```

或按协议按需安装：

```bash
pip install tritonclient[grpc]
pip install tritonclient[http]
```

> 共享内存功能包含在 `tritonclient[http]` 中，无需额外依赖。

## 协议对比

| 协议 | 底层实现 | 特点 | 适用场景 |
|------|----------|------|----------|
| `grpc` | gRPC | 吞吐高、延迟低，Triton 默认推荐 | 生产高并发推理 |
| `http` | REST | 调试方便、可读性好 | 快速验证、轻量调用 |
| `shm` | HTTP + 系统共享内存 | 避免输入/输出数据在进程间拷贝 | 大图像、高分辨率 mask、性能敏感 |

## 快速开始

以下脚本默认在 `workspace` 目录下执行。

```python
import numpy as np
from PIL import Image
from triton_client import TritonClient

img = Image.open("images/bus.jpg").convert("RGB")
img_np = np.array(img)[np.newaxis, ...]  # [1, H, W, 3]

outputs = [
    "num_dets",
    "detection_boxes",
    "detection_scores",
    "detection_classes",
    "transform_metadata",
]

# gRPC
with TritonClient("localhost:48001", protocol="grpc") as client:
    result = client.infer(
        model_name="YOLOV5_DET_PRE_ENSEMBLE",
        inputs={"raw_image": img_np},
        outputs=outputs,
    )
    print(result["num_dets"])

# HTTP
with TritonClient("localhost:48000", protocol="http") as client:
    result = client.infer(
        model_name="YOLOV5_DET_PRE_ENSEMBLE",
        inputs={"raw_image": img_np},
        outputs=outputs,
    )

# 共享内存（需要预先指定输出 shape 和 dtype）
output_specs = {
    "num_dets": ([1, 1], "int32"),
    "detection_boxes": ([1, 300, 4], "float32"),
    "detection_scores": ([1, 300], "float32"),
    "detection_classes": ([1, 300], "int32"),
    "transform_metadata": ([1, 6], "float32"),
}

with TritonClient("localhost:48000", protocol="shm") as client:
    result = client.infer(
        model_name="YOLOV5_DET_PRE_ENSEMBLE",
        inputs={"raw_image": img_np},
        outputs=outputs,
        output_specs=output_specs,
    )
```

## 动态输出处理

本项目的检测后处理后端均返回**动态数量**的检测结果：`num_dets` 表示当前图片实际检测到的目标数，其余输出张量按最大检测数分配。读取时建议按 `num_dets` 切片：

```python
num_dets = int(result["num_dets"][0, 0])
boxes = result["detection_boxes"][0, :num_dets]   # [N, 4]
scores = result["detection_scores"][0, :num_dets]  # [N]
classes = result["detection_classes"][0, :num_dets] # [N]
```

分割模型还会返回 `detection_masks`、`mask_offsets`、`mask_shapes`，可按 `num_dets` 逐目标取出对应 mask：

```python
masks = result["detection_masks"][0]
offsets = result["mask_offsets"][0]
shapes = result["mask_shapes"][0]

for i in range(num_dets):
    h, w = shapes[i]
    off = offsets[i]
    if h > 0 and w > 0 and off >= 0:
        mask = masks[off:off + h * w].reshape(h, w)
```

## API 说明

### `TritonClient(url, protocol="http", verbose=False)`

- `url`：服务端地址，例如 `localhost:48001`。
- `protocol`：协议类型，可选 `"grpc"`、`"http"`、`"shm"`。
- `verbose`：是否打印底层请求日志。

支持上下文管理器，`with` 块结束时会自动释放共享内存并关闭连接。

### `infer(model_name, inputs, outputs, output_specs=None)`

执行推理并返回 `dict[str, np.ndarray]`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_name` | `str` | 模型或 ensemble 名称 |
| `inputs` | `dict[str, np.ndarray]` | 输入张量，键为输入名 |
| `outputs` | `list[str]` | 需要返回的输出张量名 |
| `output_specs` | `dict[str, (shape, dtype)]` | **SHM 模式必填**，预分配输出缓冲区 |

`output_specs` 示例：

```python
{
    "num_dets": ([1, 1], "int32"),
    "detection_boxes": ([1, 300, 4], "float32"),
}
```

### 其他方法

- `is_server_ready()` / `is_server_live()`：服务端状态。
- `is_model_ready(model_name)`：模型状态。
- `get_model_metadata(model_name)`：获取简化后的模型元数据，返回 `{"name": ..., "inputs": [...], "outputs": [...]}`。

## 共享内存（SHM）说明

### 为什么需要 `output_specs`

SHM 模式下，输出张量需要先在客户端分配好系统共享内存并注册到 Triton 服务端，因此必须提前知道每个输出的 shape 和 dtype。`output_specs` 就是用来描述这些静态最大输出的。

### 共享内存生命周期

1. `TritonClient` 构造时创建 HTTP 客户端。
2. 第一次 `infer()` 时：
   - 为每个输出按 `output_specs` 创建 SHM 并 `register`。
   - 为每个输入创建 SHM 并 `register`。
3. 后续请求若输入 shape 不变则复用输入 SHM；输出 SHM 在 `output_specs` 不变时一直复用。
4. `close()` / `__exit__` 时统一 `unregister` 并销毁所有 SHM。

### 多输入支持

SHM 模式同样支持多个输入张量，每个输入会独立创建/复用一个共享内存区域：

```python
inputs = {
    "input_0": arr0,
    "input_1": arr1,
}
```

## 运行示例

在 `workspace` 目录下执行：

```bash
python3 triton_client/example.py --protocol grpc
python3 triton_client/example.py --protocol http
python3 triton_client/example.py --protocol shm
```

也支持指定模型和图片：

```bash
python3 triton_client/example.py --protocol grpc --model YOLO11_DET_PRE_ENSEMBLE --image images/zidane.jpg
```

## 测试

```bash
# 单元测试（无需 Triton 服务）
python3 triton_client/test_client.py

# 集成测试（需要本地 Triton 服务正在运行）
python3 triton_client/test_client.py --integration
```

集成测试默认连接 `localhost:48001` (gRPC) 和 `localhost:48000` (HTTP/SHM)，并调用 `YOLOV5_DET_PRE_ENSEMBLE` 模型，可在 `test_client.py` 中按需修改。

## 性能测试

```bash
python3 triton_client/benchmark.py --protocol grpc --count 100 --warmup 10
python3 triton_client/benchmark.py --protocol http --count 100 --warmup 10
python3 triton_client/benchmark.py --protocol shm --count 100 --warmup 10
```

输出示例：

```
========== Benchmark Results ==========
Protocol:       shm
Requests:       100
Total time:     1.864 s
Throughput:     53.65 req/s
Mean latency:   18.64 ms
Median latency: 18.52 ms
P90 latency:    19.34 ms
P95 latency:    19.75 ms
P99 latency:    21.03 ms
Min latency:    17.89 ms
Max latency:    21.56 ms
```

## 注意事项

- `shm` 模式必须提供 `output_specs`，因为输出共享内存缓冲区需要在推理前分配好。
- 所有协议均返回普通的 `numpy.ndarray`。SHM 模式下会在返回前把数据从共享内存拷贝出来，因此关闭 client 后仍可安全使用结果。
- 多输入场景在 gRPC / HTTP / SHM 下均完全支持。
- 当输入 shape 发生变化时，SHM 模式会自动重新分配该输入对应的共享内存区域，其余输入不受影响。
