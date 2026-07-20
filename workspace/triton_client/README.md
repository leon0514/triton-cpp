# Triton Client Wrapper

A concise Python wrapper around the official Triton client SDK that exposes the
same `infer()` API for HTTP, gRPC and system shared memory (SHM) inference.

## Installation

```bash
pip install tritonclient[all] numpy
```

## Usage

```python
import numpy as np
from triton_client import TritonClient

inputs = {"raw_image": np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)}
outputs = ["num_dets", "detection_boxes", "detection_scores", "detection_classes"]

# gRPC
with TritonClient("localhost:8001", protocol="grpc") as client:
    result = client.infer("yolo", inputs, outputs)

# HTTP
with TritonClient("localhost:8000", protocol="http") as client:
    result = client.infer("yolo", inputs, outputs)

# Shared memory: output_specs describes every requested output tensor.
output_specs = {
    "num_dets": ([1, 1], "INT32"),
    "detection_boxes": ([1, 300, 4], "FP32"),
    "detection_scores": ([1, 300], "FP32"),
    "detection_classes": ([1, 300], "INT32"),
}
with TritonClient("localhost:8001", protocol="shm") as client:
    result = client.infer("yolo", inputs, outputs, output_specs=output_specs)
```

## API

### `TritonClient(url, *, protocol="grpc", **kwargs)`

Factory that returns one of:

- `TritonHttpClient`
- `TritonGrpcClient`
- `TritonSharedMemoryClient`

All clients support the context-manager protocol.

### `infer(model_name, inputs, outputs, *, model_version="", output_specs=None, **kwargs)`

Run inference and return a dict mapping output names to `np.ndarray`.

- `inputs`: `dict[str, np.ndarray]`
- `outputs`: `list[str]`
- `output_specs`: required for SHM, `dict[str, (shape, dtype)]` where `dtype`
  is a Triton datatype string such as `"FP32"`, `"INT32"`, `"UINT8"`, etc.

### Shared memory control protocol

`TritonSharedMemoryClient` uses gRPC by default to register/unregister SHM
regions with the server. Pass `control_protocol="http"` to switch the control
channel to HTTP:

```python
with TritonClient(
    "localhost:8000", protocol="shm", control_protocol="http"
) as client:
    result = client.infer(...)
```

## Notes

- The wrapper only deals with the Triton client itself. Thread pools,
  client pools and stream orchestration belong in the application layer.
- SHM regions are cached inside each `TritonSharedMemoryClient` instance and
  reused across requests with the same tensor name. They are released when
  the client is closed.
