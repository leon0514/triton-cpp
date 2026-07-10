# Triton 客户端封装

对官方 `tritonclient` 的轻量级统一封装，同一套 API 支持三种推理协议：

- `grpc`
- `http`
- `shm`（基于 HTTP 的系统共享内存）

以下所有脚本均设计为在 `workspace` 目录下执行。

## 用法

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

## 说明

- `inputs` 是 `dict[str, np.ndarray]`，key 为输入张量名，value 为 numpy 数组。
- `outputs` 是需要返回的输出张量名列表。
- `shm` 模式必须提供 `output_specs`，因为输出共享内存缓冲区需要在推理前分配好。
- 所有协议均返回普通的 `numpy.ndarray`。SHM 模式下会在返回前把数据从共享内存拷贝出来，因此关闭 client 后仍可安全使用结果。
- `get_model_metadata(model_name)` 可获取简化后的模型元数据，三种协议均支持。

## 示例、测试与性能对比

在 `workspace` 目录下执行：

```bash
# 三种协议示例
python3 triton_client/example.py --protocol grpc
python3 triton_client/example.py --protocol http
python3 triton_client/example.py --protocol shm

# 单元测试（无需 Triton 服务）
python3 triton_client/test_client.py

# 集成测试（需要本地 Triton 服务正在运行）
python3 triton_client/test_client.py --integration

# 时间/性能测试（需要本地 Triton 服务正在运行）
python3 triton_client/benchmark.py --protocol grpc --count 100 --warmup 10
python3 triton_client/benchmark.py --protocol http --count 100 --warmup 10
python3 triton_client/benchmark.py --protocol shm --count 100 --warmup 10
```
