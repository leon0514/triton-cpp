# Triton 客户端封装

对官方 `tritonclient` 的轻量级统一封装，同一套 API 支持三种推理协议：

- `grpc`
- `http`
- `shm`（基于 HTTP 的系统共享内存）

## 目录结构

```
workspace/triton_client/
├── __init__.py        # 导出 TritonClient、TritonClientError
├── client.py          # 核心实现
├── example.py         # 三种协议调用示例
├── test_client.py     # 单元测试 + 集成测试
├── benchmark.py       # 时间/性能测试
└── README.md
```

## 依赖

```bash
pip install numpy pillow tritonclient[all]
```

或仅安装需要的协议：

```bash
pip install tritonclient[grpc]
pip install tritonclient[http]
```

## 基本用法

以下脚本均在 `workspace` 目录下执行。

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
        model_name="yolov5_ensemble",
        inputs={"raw_image": img_np},
        outputs=outputs,
    )
    print(result["num_dets"])

# HTTP
with TritonClient("localhost:48000", protocol="http") as client:
    result = client.infer(
        model_name="yolov5_ensemble",
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
        model_name="yolov5_ensemble",
        inputs={"raw_image": img_np},
        outputs=outputs,
        output_specs=output_specs,
    )
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
| `inputs` | `dict[str, np.ndarray]` | 输入张量 |
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
- `get_model_metadata(model_name)`：获取简化后的模型元数据。

## 运行示例

在 `workspace` 目录下执行：

```bash
python3 triton_client/example.py --protocol grpc
python3 triton_client/example.py --protocol http
python3 triton_client/example.py --protocol shm
```

## 测试

```bash
# 单元测试（无需 Triton 服务）
python3 triton_client/test_client.py

# 集成测试（需要本地 Triton 服务正在运行）
python3 triton_client/test_client.py --integration
```

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
- 多输入场景在 gRPC / HTTP 下完全支持；SHM 模式下同样支持多个输入张量，每个输入会独立创建共享内存区域。
