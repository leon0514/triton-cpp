# Inference Business Layer

This directory contains application-level utilities built on top of the core
`triton_client` wrapper. These components are intentionally separated from the
core client because they depend on specific business requirements (camera
count, FPS, threading model, etc.).

## Modules

### `labels.py`

Fetch label lists from the Triton `CUSTOM_LABELS` Python backend model.

```python
from inference.labels import get_labels
from triton_client import TritonClient

with TritonClient("localhost:48000", protocol="http") as client:
    labels = get_labels(client, "YOLOV5_DET_PRE_ENSEMBLE")
```

Command line:

```bash
python3 inference/labels.py --url localhost:48000 YOLOV5_DET_PRE_ENSEMBLE
python3 inference/labels.py --url localhost:48001 --protocol grpc YOLOV5_DET_PRE_ENSEMBLE
```

### `client_pool.py`

A fixed-size pool of reusable `TritonClient` instances. Use this when many
worker threads need to perform inference but you want to limit the number of
underlying clients (and therefore the number of SHM files when using
`protocol="shm"`).

```python
from inference.client_pool import TritonClientPool

with TritonClientPool(
    "localhost:48001", protocol="shm", size=8
) as pool:
    with pool.acquire() as client:
        result = client.infer(...)
```

## Client Pool Size

- `size` should match your desired inference concurrency. A common starting
  point is the number of Triton model instances or 2× CPU cores, whichever is
  smaller.
- When `protocol="shm"`, the number of shared-memory files equals
  `size × (input_tensors + output_tensors)`. Keep `size` as small as your
  throughput allows.
