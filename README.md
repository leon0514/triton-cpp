# Triton 自定义后端部署指南

本项目实现了多个 GPU 加速的 Triton Inference Server 自定义后端：

- `YOLO_640_LETTERBOX_PREPROCESS`：图像预处理（resize、letterbox、归一化、仿射变换矩阵）。
- `YOLOV5_DET_PRE_POSTPROCESS`：YOLOv5 检测后处理。
- `YOLO11_DET_PRE_POSTPROCESS` / `YOLO11_OBB_PRE_POSTPROCESS` / `YOLO11_POSE_PRE_POSTPROCESS` / `YOLO11_SEG_PRE_POSTPROCESS`：YOLO11 系列检测 / OBB / 姿态 / 分割后处理。
- `YOLO26_DET_PRE_POSTPROCESS` / `YOLO26_OBB_PRE_POSTPROCESS` / `YOLO26_POSE_PRE_POSTPROCESS` / `YOLO26_SEG_PRE_POSTPROCESS`：YOLO26系列检测 / OBB / 姿态 / 分割后处理。
- `RFDETR_DET_PRE_POSTPROCESS`：RF-DETR 后处理。
- `RFDETR_SEG_PRE_POSTPROCESS`：RF-DETR 分割后处理（支持 `return_masks` 开关）。
- `sahi_ensemble`：SAHI（Slicing Aided Hyper Inference）集成后端，支持 det / pose / seg 三种输出类型。

模型仓库位于 `workspace/models/`，所有自定义后端 `.so` 都嵌在对应模型的 `1/` 目录下，通过 `docker-compose.yml` 一键启动。

---

## 目录

1. [环境要求](#环境要求)
2. [编译](#编译)
3. [启动服务](#启动服务)
4. [模型操作](#模型操作)
5. [模型配置](#模型配置)
6. [模型转换](#模型转换)
7. [测试](#测试)

---

## 环境要求

- NVIDIA Docker / Docker Compose
- NVIDIA Container Toolkit（`--gpus all` 需要）
- 镜像：`nvcr.io/nvidia/tritonserver:25.01-py3`
- CUDA architectures：`80;86;89`（可在 `build.sh` 中通过 `CUDA_ARCHS` 覆盖）

---

## 编译

### 1. 使用 Docker 编译（推荐）

项目根目录下执行：

```bash
cd triton-cpp

docker run --rm --gpus all \
  -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tritonserver:25.01-py3 \
  bash /workspace/build.sh
```

说明：
- 运行时镜像已包含 Triton backend 头文件和 CUDA toolkit，可直接编译。
- 编译产物输出到 `build/` 目录：
  ```
  build/libtriton_preprocess.so
  build/libtriton_yolo11_postprocess.so
  build/libtriton_yolo11_obb_postprocess.so
  build/libtriton_yolo11_pose_postprocess.so
  build/libtriton_yolo11_seg_postprocess.so
  build/libtriton_yolov5_postprocess.so
  build/libtriton_yolo26_postprocess.so
  build/libtriton_rfdetr_postprocess.so
  build/libtriton_rfdetr_seg_postprocess.so
  build/libtriton_sahi_ensemble.so
  ```

### 2. 复制产物到模型仓库

```bash
cp build/libtriton_preprocess.so workspace/models/YOLO_640_LETTERBOX_PREPROCESS/1/
cp build/libtriton_preprocess.so workspace/models/RFDETR_512_DIRECT_PREPROCESS/1/
cp build/libtriton_yolo11_postprocess.so workspace/models/YOLO11_DET_PRE_POSTPROCESS/1/
cp build/libtriton_yolo11_obb_postprocess.so workspace/models/YOLO11_OBB_PRE_POSTPROCESS/1/
cp build/libtriton_yolo11_pose_postprocess.so workspace/models/YOLO11_POSE_PRE_POSTPROCESS/1/
cp build/libtriton_yolo11_seg_postprocess.so workspace/models/YOLO11_SEG_PRE_POSTPROCESS/1/
cp build/libtriton_yolov5_postprocess.so workspace/models/YOLOV5_DET_PRE_POSTPROCESS/1/
cp build/libtriton_yolo26_postprocess.so workspace/models/YOLO26_DET_PRE_POSTPROCESS/1/
cp build/libtriton_yolo26_pose_postprocess.so workspace/models/YOLO26_POSE_PRE_POSTPROCESS/1/
cp build/libtriton_yolo26_obb_postprocess.so workspace/models/YOLO26_OBB_PRE_POSTPROCESS/1/
cp build/libtriton_yolo26_seg_postprocess.so workspace/models/YOLO26_SEG_PRE_POSTPROCESS/1/
cp build/libtriton_rfdetr_postprocess.so workspace/models/RFDETR_DET_PRE_POSTPROCESS/1/
cp build/libtriton_rfdetr_seg_postprocess.so workspace/models/RFDETR_SEG_PRE_POSTPROCESS/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO11_DET_ENSEMBLE/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO11_POSE_ENSEMBLE/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO11_SEG_ENSEMBLE/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO11_OBB_ENSEMBLE/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO26_DET_ENSEMBLE/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO26_POSE_ENSEMBLE/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO26_SEG_ENSEMBLE/1/
cp build/libtriton_sahi_ensemble.so workspace/models/SAHI_YOLO26_OBB_ENSEMBLE/1/
```

### 3. 自定义 CUDA 架构

```bash
CUDA_ARCHS="80;86;89" ./build.sh
```

或在容器内：

```bash
CUDA_ARCHS="80;86;89" bash /workspace/build.sh
```

---

## 启动服务

### 1. 启动 Triton Inference Server

```bash
cd triton-cpp
docker compose up -d
```

默认端口映射：
- HTTP：`48000 -> 8000`
- gRPC：`48001 -> 8001`
- Metrics：`48002 -> 8002`

检查服务是否就绪：

```bash
curl -s http://localhost:48000/v2/health/ready
```

返回空且 HTTP 200 表示就绪。

查看模型状态：

```bash
docker logs triton-backends --tail 80
```

### 2. 按需加载模型（可选）

默认情况下 Triton 会加载模型仓库里的全部模型。如果希望只加载部分模型，推荐通过 `workspace/models_to_load.txt` 配置，每行一个模型名，支持 `#` 注释，加载很多模型时更直观：

```text
# workspace/models_to_load.txt
CLASSIFIER_CLASSIFY_PRE_ENSEMBLE
YOLO11_DET_PRE_ENSEMBLE
YOLO11_POSE_PRE_ENSEMBLE
YOLO11_SEG_PRE_ENSEMBLE
```

然后在项目根目录的 `.env` 中指定该文件：

```bash
TRITON_MODELS_FILE=/models_to_load.txt
```

配置后重启容器即可生效：

```bash
docker compose down
docker compose up -d
```

如果模型数量很少，也可以直接在 `.env` 中用逗号分隔：

```bash
TRITON_MODELS=CLASSIFIER_CLASSIFY_PRE_ENSEMBLE,YOLO11_DET_PRE_ENSEMBLE
```

说明：
- `.env` 文件会被 `docker-compose.yml` 自动加载。
- `TRITON_MODELS_FILE` 优先级高于 `TRITON_MODELS`。
- 配置列表中的 ensemble 模型及其依赖（预处理、推理、后处理）会自动加载。
- `CUSTOM_LABELS` 模型是前端显示类别名依赖的公共服务，启用按需加载时会被自动追加到加载列表，无需手动配置。
- 未加载的模型在前端 `/api/models` 中会显示为 `ready: false`，无法被选择推理。
- 启用按需加载后，仍可通过 Triton Model Repository API 手动加载 / 卸载 / 更新模型，详见 [模型操作](#模型操作)。
- 留空或不配置上述变量时，行为与之前一致：加载所有模型。

### 3. 启动可视化前端（可选）

```bash
cd triton-cpp/triton-display
docker compose up -d --build
```

浏览器访问：`http://localhost:8088`

---

## 模型操作

当启用按需加载（`TRITON_MODELS_FILE` 或 `TRITON_MODELS` 非空）时，Triton 会运行在 `--model-control-mode=explicit` 模式下，此时可以通过 **Model Repository API** 在运行时手动管理模型，无需重启容器。

### 1. 查看模型状态

```bash
curl -s -X POST http://localhost:48000/v2/repository/index \
  -H 'Content-Type: application/json' \
  -d '{}' | python3 -m json.tool
```

返回示例：

```json
[
  { "name": "CLASSIFIER_CLASSIFY_PRE_ENSEMBLE", "version": "1", "state": "READY" },
  { "name": "YOLO11_POSE_PRE_ENSEMBLE" }
]
```

有 `state: READY` 的表示已加载，其余表示尚未加载。

### 2. 手动加载模型

```bash
curl -s -X POST http://localhost:48000/v2/repository/models/YOLO11_POSE_PRE_ENSEMBLE/load \
  -H 'Content-Type: application/json' \
  -d '{}'
```

加载后检查就绪状态：

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  http://localhost:48000/v2/models/YOLO11_POSE_PRE_ENSEMBLE/ready
# 200
```

### 3. 手动卸载模型

```bash
curl -s -X POST http://localhost:48000/v2/repository/models/YOLO11_POSE_PRE_ENSEMBLE/unload \
  -H 'Content-Type: application/json' \
  -d '{}'
```

卸载后该模型即不可推理，GPU 显存会被释放。

### 4. 更新模型（热重载）

当替换了某个模型的文件（例如更新了 `workspace/models/CLASSIFIER_CLASSIFY_PRE/1/model.plan`）后，需要先卸载再加载，使 Triton 重新读取最新文件：

```bash
MODEL=YOLO11_POSE_PRE_ENSEMBLE

curl -s -X POST http://localhost:48000/v2/repository/models/$MODEL/unload \
  -H 'Content-Type: application/json' -d '{}'

curl -s -X POST http://localhost:48000/v2/repository/models/$MODEL/load \
  -H 'Content-Type: application/json' -d '{}'
```

> 注意：直接覆盖模型仓库中的文件不会自动生效，必须执行 unload + load。

### 5. 完全手动管理

如果你希望启动时不加载任何模型，全部通过 API 手动控制，可以在 `workspace/models_to_load.txt` 中只保留 `CUSTOM_LABELS`：

```text
CUSTOM_LABELS
```

这样 Triton 启动后仅加载 `CUSTOM_LABELS` 公共服务，其它模型按需通过上述 API 加载。

---

## 模型配置

模型仓库结构如下（每个模型一个目录，`config.pbtxt` 描述输入输出）：

```
workspace/models/
├── YOLO_640_LETTERBOX_PREPROCESS/
│   ├── config.pbtxt
│   └── 1/libtriton_preprocess.so
├── YOLO11_DET_PRE/
│   ├── config.pbtxt
│   └── 1/model.plan
├── YOLO11_DET_PRE_POSTPROCESS/
│   ├── config.pbtxt
│   └── 1/libtriton_yolo11_postprocess.so
├── YOLO11_DET_PRE_ENSEMBLE/config.pbtxt
├── CUSTOM_LABELS/
│   ├── config.pbtxt
│   └── 1/model.py
└── ...
```

### 1. 预处理配置（`YOLO_640_LETTERBOX_PREPROCESS/config.pbtxt`）

```protobuf
name: "YOLO_640_LETTERBOX_PREPROCESS"
backend: "preprocess"
max_batch_size: 16

input [
  {
    name: "raw_image"
    data_type: TYPE_UINT8
    dims: [-1, -1, 3]
  }
]

output [
  { name: "preprocessed_output", data_type: TYPE_FP32, dims: [3, 640, 640] },
  { name: "transform_metadata", data_type: TYPE_FP32, dims: [6] }
]

parameters: {
  key: "target_width"    value: { string_value: "640" }
}
parameters: {
  key: "target_height"   value: { string_value: "640" }
}
parameters: {
  key: "resize_type"     value: { string_value: "letterbox" }
}
parameters: {
  key: "output_type"     value: { string_value: "FP32" }
}
parameters: {
  key: "norm_type"       value: { string_value: "alpha_beta" }
}
parameters: {
  key: "alpha"           value: { string_value: "0.00392156862745098" }
}
parameters: {
  key: "beta"            value: { string_value: "0.0" }
}
parameters: {
  key: "channel_type"    value: { string_value: "none" }
}
parameters: {
  key: "fill_value"      value: { string_value: "[114.0, 114.0, 114.0]" }
}
parameters: {
  key: "output_transform" value: { string_value: "true" }
}
```

参数说明：

| 参数 | 类型 | 说明 |
|------|------|------|
| `target_width` / `target_height` | int | 目标尺寸 |
| `resize_type` | string | `direct` 直接缩放 / `letterbox` 等比缩放并填充 |
| `output_type` | string | `FP32` / `FP16` |
| `norm_type` | string | `none` / `mean_std` / `alpha_beta` |
| `alpha` / `beta` | float | `alpha_beta` 归一化参数：`x * alpha + beta` |
| `mean` / `std` | float[3] | `mean_std` 归一化参数 |
| `channel_type` | string | `none` / `swap_rb`（BGR -> RGB） |
| `fill_value` | float[3] | letterbox 填充色（BGR） |
| `output_transform` | bool | 是否输出 `d2i` 逆变换矩阵 |

### 2. 检测后处理配置（`YOLO11_DET_PRE_POSTPROCESS/config.pbtxt`）

```protobuf
name: "YOLO11_DET_PRE_POSTPROCESS"
backend: "yolo11_postprocess"
max_batch_size: 16

input [
  { name: "model_output", data_type: TYPE_FP32, dims: [84, 8400] }
]

output [
  { name: "num_dets",          data_type: TYPE_INT32, dims: [1] },
  { name: "detection_boxes",   data_type: TYPE_FP32, dims: [-1, 4] },
  { name: "detection_scores",  data_type: TYPE_FP32, dims: [-1] },
  { name: "detection_classes", data_type: TYPE_INT32, dims: [-1] }
]

parameters: {
  key: "num_classes"          value: { string_value: "80" }
}
parameters: {
  key: "confidence_threshold" value: { string_value: "0.25" }
}
parameters: {
  key: "iou_threshold"        value: { string_value: "0.45" }
}
parameters: {
  key: "max_detections"       value: { string_value: "300" }
}
parameters: {
  key: "max_candidates"       value: { string_value: "8400" }
}
parameters: {
  key: "output_format"        value: { string_value: "channel_first" }
}
parameters: {
  key: "score_activation"     value: { string_value: "none" }
}
```

参数说明：

| 参数 | 说明 |
|------|------|
| `num_classes` | 类别数 |
| `confidence_threshold` | 置信度阈值 |
| `iou_threshold` | NMS IoU 阈值 |
| `max_detections` | 每图最多保留检测数 |
| `max_candidates` | 进入 NMS 的候选数上限 |
| `output_format` | `channel_first`（`[C, N]`）或 `anchor_first`（`[N, C]`） |
| `score_activation` | `none` / `sigmoid` |

### 3. 各后处理模型输出维度对照

| 模型 | 输入名 | 输入维度 | 输出 |
|------|--------|----------|------|
| `YOLO11_DET_PRE_POSTPROCESS` | `model_output` | `[84, 8400]` | `num_dets`, `detection_boxes[-1,4]`, `detection_scores[-1]`, `detection_classes[-1]` |
| `YOLO11_OBB_PRE_POSTPROCESS` | `model_output` | `[20, 8400]` | 同上，但 `detection_boxes[-1,5]`（含 angle） |
| `YOLO11_POSE_PRE_POSTPROCESS` | `model_output` | `[56, 8400]` | 同上 + `detection_keypoints[-1,17,3]` |
| `YOLO11_SEG_PRE_POSTPROCESS` | `model_output`, `mask_protos` | `[116, 8400]`, `[32, 160, 160]` | 同上 + `detection_masks[-1,-1]`, `mask_shapes[-1,2]` |
| `YOLOV5_DET_PRE_POSTPROCESS` | `model_output` | `[25200, 85]` | 同 `YOLO11_DET_PRE_POSTPROCESS` |
| `YOLO26_DET_PRE_POSTPROCESS` | `model_output` | `[300, 6]` | 同 `YOLO11_DET_PRE_POSTPROCESS` |
| `YOLO26_OBB_PRE_POSTPROCESS` | `model_output` | `[300, X]` | 同 `YOLO11_OBB_PRE_POSTPROCESS` |
| `YOLO26_POSE_PRE_POSTPROCESS` | `model_output`, `transform_metadata` | `[300, 57]`, `[6]` | 同 `YOLO11_POSE_PRE_POSTPROCESS` |
| `YOLO26_SEG_PRE_POSTPROCESS` | `model_output`, `mask_protos`, `transform_metadata` | `[300, X]`, `[32,160,160]`, `[6]` | 同 `YOLO11_SEG_PRE_POSTPROCESS` |
| `RFDETR_DET_PRE_POSTPROCESS` | `dets`, `labels` | `[300, 4]`, `[300, 91]` | 同 `YOLO11_DET_PRE_POSTPROCESS` |

### 4. Ensemble 配置示例（`YOLO11_DET_PRE_ENSEMBLE/config.pbtxt`）

```protobuf
name: "YOLO11_DET_PRE_ENSEMBLE"
platform: "ensemble"
max_batch_size: 16

input [
  { name: "raw_image", data_type: TYPE_UINT8, dims: [-1, -1, 3] }
]

output [
  { name: "num_dets",          data_type: TYPE_INT32, dims: [1] },
  { name: "detection_boxes",   data_type: TYPE_FP32, dims: [-1, 4] },
  { name: "detection_scores",  data_type: TYPE_FP32, dims: [-1] },
  { name: "detection_classes", data_type: TYPE_INT32, dims: [-1] }
]

ensemble_scheduling {
  step [
    {
      model_name: "YOLO_640_LETTERBOX_PREPROCESS"
      input_map { key: "raw_image" value: "raw_image" }
      output_map { key: "preprocessed_output" value: "preprocessed_output" }
      output_map { key: "transform_metadata" value: "transform_metadata" }
    },
    {
      model_name: "YOLO11_DET_PRE"
      input_map { key: "images" value: "preprocessed_output" }
      output_map { key: "output0" value: "output0" }
    },
    {
      model_name: "YOLO11_DET_PRE_POSTPROCESS"
      input_map { key: "model_output" value: "output0" }
      output_map { key: "num_dets" value: "num_dets" }
      output_map { key: "detection_boxes" value: "detection_boxes" }
      output_map { key: "detection_scores" value: "detection_scores" }
      output_map { key: "detection_classes" value: "detection_classes" }
    }
  ]
}
```

### 5. SAHI Ensemble 配置

SAHI（Slicing Aided Hyper Inference）通过将大图切分为多个小块分别推理，再合并结果，大幅提升高分辨率图像上的小目标检测效果。

本项目通过 `sahi_ensemble` 后端统一实现，通过 `output_type` 参数支持三种输出类型：

| `output_type` | 说明 | 对应 Ensemble 示例 |
|---|---|---|
| `det` | 目标检测 | `SAHI_YOLO11_DET_ENSEMBLE` / `SAHI_YOLO26_DET_ENSEMBLE` |
| `pose` | 姿态估计 | `SAHI_YOLO11_POSE_ENSEMBLE` / `SAHI_YOLO26_POSE_ENSEMBLE` |
| `seg` | 实例分割 | `SAHI_YOLO11_SEG_ENSEMBLE` / `SAHI_YOLO26_SEG_ENSEMBLE` |
| `obb` | 旋转框检测 | `SAHI_YOLO11_OBB_ENSEMBLE` / `SAHI_YOLO26_OBB_ENSEMBLE` |

**工作流程（全部 GPU 加速）：**

1. 接收 `raw_image [H, W, 3]`
2. 同步调用 `SAHI_PREPROCESS` → 切分为多个 `slice_width × slice_height` 小块 + 每个小块的偏移量
3. 分块（按 `chunk_size`）调用指定的 `detector_model` 进行推理
4. CUDA kernel：置信度过滤 + 偏移校正 + 裁剪（一体化）
5. CUDA 逐类 NMS（合并跨切片的重叠框）
6. Top-K 排序 + 填充固定输出

**配置示例（det 类型）：**

```protobuf
name: "SAHI_YOLO11_DET_ENSEMBLE"
backend: "sahi_ensemble"
max_batch_size: 0

input [
  { name: "raw_image", data_type: TYPE_UINT8, dims: [-1, -1, -1, 3] }
]

output [
  { name: "num_dets",          data_type: TYPE_INT32, dims: [1] },
  { name: "detection_boxes",   data_type: TYPE_FP32, dims: [300, 4] },
  { name: "detection_scores",  data_type: TYPE_FP32, dims: [300] },
  { name: "detection_classes", data_type: TYPE_INT32, dims: [300] }
]

parameters: {
  key: "output_type"     value: { string_value: "det" }
}
parameters: {
  key: "detector_model"  value: { string_value: "YOLO11_DET_PRE_ENSEMBLE" }
}
parameters: {
  key: "conf_threshold"  value: { string_value: "0.25" }
}
parameters: {
  key: "iou_threshold"   value: { string_value: "0.45" }
}
parameters: {
  key: "max_detections"  value: { string_value: "300" }
}
parameters: {
  key: "chunk_size"      value: { string_value: "16" }
}
parameters: {
  key: "num_classes"     value: { string_value: "80" }
}
parameters: {
  key: "slice_width"     value: { string_value: "640" }
}
parameters: {
  key: "slice_height"    value: { string_value: "640" }
}
```

**pose 类型额外输出：**

```protobuf
output [
  { name: "num_dets",            data_type: TYPE_INT32, dims: [1] },
  { name: "detection_boxes",     data_type: TYPE_FP32, dims: [300, 4] },
  { name: "detection_scores",    data_type: TYPE_FP32, dims: [300] },
  { name: "detection_classes",   data_type: TYPE_INT32, dims: [300] },
  { name: "detection_keypoints", data_type: TYPE_FP32, dims: [300, 17, 3] }
]

parameters: {
  key: "output_type"     value: { string_value: "pose" }
}
parameters: {
  key: "detector_model"  value: { string_value: "YOLO11_POSE_PRE_ENSEMBLE" }
}
parameters: {
  key: "num_keypoints"   value: { string_value: "17" }
}
parameters: {
  key: "num_classes"     value: { string_value: "1" }
}
```

**seg 类型额外输出：**

```protobuf
output [
  ...
  { name: "detection_masks",   data_type: TYPE_FP32, dims: [300, 25600] },
  { name: "mask_shapes",       data_type: TYPE_INT32, dims: [300, 2] }
]

parameters: {
  key: "output_type"             value: { string_value: "seg" }
}
parameters: {
  key: "detector_model"          value: { string_value: "YOLO11_SEG_PRE_ENSEMBLE" }
}
parameters: {
  key: "mask_output_resolution"  value: { string_value: "160" }
}
parameters: {
  key: "num_classes"             value: { string_value: "80" }
}
```

**obb 类型额外输出：**

```protobuf
output [
  { name: "num_dets", data_type: TYPE_INT32, dims: [1] },
  { name: "detection_boxes", data_type: TYPE_FP32, dims: [300, 5] },
  { name: "detection_scores", data_type: TYPE_FP32, dims: [300] },
  { name: "detection_classes", data_type: TYPE_INT32, dims: [300] }
]

parameters: {
  key: "output_type"     value: { string_value: "obb" }
}
parameters: {
  key: "detector_model"  value: { string_value: "YOLO11_OBB_PRE_ENSEMBLE" }
}
parameters: {
  key: "num_classes"     value: { string_value: "15" }
}
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_type` | string | `det` | 输出类型：`det` / `pose` / `seg` |
| `detector_model` | string | `YOLO11_DET_PRE_ENSEMBLE` | 底层的检测 ensemble 模型名 |
| `conf_threshold` | float | `0.25` | SAHI 合并阶段的置信度阈值 |
| `iou_threshold` | float | `0.45` | SAHI 合并阶段的 NMS IoU 阈值 |
| `max_detections` | int | `300` | 最终最多保留的检测数 |
| `chunk_size` | int | `16` | 每次发给 detector 的最大切片数（batch size） |
| `num_classes` | int | `80` | 类别数（pose 模式下通常为 `1`） |
| `num_keypoints` | int | `17` | 关键点数（仅 pose 模式） |
| `mask_output_resolution` | int | `160` | 分割 mask 输出边长（仅 seg 模式，输出为 N×N） |
| `slice_width` | int | `640` | SAHI 切片宽度 |
| `slice_height` | int | `640` | SAHI 切片高度 |
| `overlap_width_ratio` | float | `0.2` | 切片水平重叠比例 |
| `overlap_height_ratio` | float | `0.2` | 切片垂直重叠比例 |
| `max_slices` | int | `64` | 最大切片数（防止异常大图 OOM） |

> **注意**：SAHI ensemble 的 `max_batch_size` 必须为 `0`，输入 dims 为 `[-1, -1, -1, 3]`（带 batch 占位）。`detector_model` 指定的底层模型必须有 `max_batch_size > 0` 以支持分块批量推理。

### 6. 标签服务配置（`CUSTOM_LABELS/config.pbtxt`）

```protobuf
name: "CUSTOM_LABELS"
backend: "python"
max_batch_size: 0

input [
  { name: "model_name", data_type: TYPE_STRING, dims: [1] }
]
output [
  { name: "labels", data_type: TYPE_STRING, dims: [-1] }
]

parameters: {
  key: "names_directory"
  value: { string_value: "/models/CUSTOM_LABELS/names" }
}
```

标签文件放在 `workspace/models/CUSTOM_LABELS/names/<model_name>.txt`，每行一个类别名。

---

## 模型转换

通用流程：**PyTorch `.pt` -> ONNX -> TensorRT `.plan`**。

> **推荐在 Triton Docker 容器内执行导出**，这样无需在宿主机安装 TensorRT / CUDA 工具链。下面各小节只给出 `trtexec` 命令；若使用 Docker，请把模型目录挂载进去，例如：
>
> ```bash
> docker run --rm --gpus all \
>   -v "$(pwd)/workspace/models/<model_name>/1":/models \
>   nvcr.io/nvidia/tritonserver:25.01-py3 \
>   /usr/src/tensorrt/bin/trtexec \
>     --onnx=/models/model.onnx \
>     --saveEngine=/models/model.plan \
>     --fp16 \
>     --minShapes=images:1x3x640x640 \
>     --optShapes=images:1x3x640x640 \
>     --maxShapes=images:16x3x640x640
> ```
>
> 具体 `--minShapes/--optShapes/--maxShapes` 等参数见下面各模型小节。
>
> 也可以直接进入容器交互式执行（便于批量转换）：
>
> ```bash
> docker run --rm --gpus all -it \
>   -v "$(pwd)/workspace/models":/models \
>   nvcr.io/nvidia/tritonserver:25.01-py3 bash
> cd /models/<model_name>/1
> /usr/src/tensorrt/bin/trtexec ...
> ```

### 1. YOLO11 检测

```bash
cd workspace/models/YOLO11_DET_PRE/1

# 1) PT -> ONNX
python3 - <<PY
from ultralytics import YOLO
model = YOLO("YOLO11_DET_PREn.pt")
model.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=True)
PY

# 2) ONNX -> TensorRT plan
trtexec --onnx=YOLO11_DET_PREn.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:16x3x640x640
```

### 2. YOLO11-Pose

```bash
cd workspace/models/YOLO11_POSE_PRE/1

python3 - <<PY
from ultralytics import YOLO
model = YOLO("YOLO11_DET_PREn-pose.pt")
model.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=True)
PY

trtexec --onnx=YOLO11_DET_PREn-pose.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:16x3x640x640
```

### 3. YOLO11-OBB

```bash
cd workspace/models/YOLO11_OBB_PRE/1

python3 - <<PY
from ultralytics import YOLO
model = YOLO("YOLO11_DET_PREn-obb.pt")
model.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=True)
PY

trtexec --onnx=YOLO11_DET_PREn-obb.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:16x3x640x640
```

### 4. YOLO11-Seg

```bash
cd workspace/models/YOLO11_SEG_PRE/1

python3 - <<PY
from ultralytics import YOLO
model = YOLO("YOLO11_DET_PREs-seg.pt")
model.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=True)
PY

trtexec --onnx=YOLO11_DET_PREs-seg.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:16x3x640x640
```

### 5. YOLOv5

```bash
cd workspace/models/YOLOV5_DET_PRE/1

python3 - <<PY
from ultralytics import YOLO
model = YOLO("YOLOV5_DET_PREs.pt")
model.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=True)
PY

# YOLOv5 默认导出为 anchor_first [batch, 25200, 85]
trtexec --onnx=YOLOV5_DET_PREs.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:16x3x640x640
```

后处理 `config.pbtxt` 需配合：

```protobuf
input [
  { name: "model_output", data_type: TYPE_FP32, dims: [25200, 85] }
]
parameters: {
  key: "output_format"    value: { string_value: "anchor_first" }
}
parameters: {
  key: "has_objectness"   value: { string_value: "true" }
}
parameters: {
  key: "max_candidates"   value: { string_value: "25200" }
}
```

### 6. YOLO26

```bash
cd workspace/models/YOLO26_DET_PRE/1

python3 - <<PY
from ultralytics import YOLO
model = YOLO("YOLO26_DET_PREs.pt")
model.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=True)
PY

trtexec --onnx=YOLO26_DET_PREs.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:16x3x640x640
```

YOLO26 输出为 `[batch, 300, 6]`，后处理无需 NMS。

### 7. RF-DETR Seg

```bash
cd workspace/models/RFDETR_SEG_PRE/1

# ONNX -> TensorRT plan（input 384x384，max batch 16，FP16）
trtexec --onnx=model.onnx \
  --saveEngine=model.plan \
  --minShapes=input:1x3x384x384 \
  --optShapes=input:8x3x384x384 \
  --maxShapes=input:16x3x384x384 \
  --fp16 \
  --skipInference
```

`config.pbtxt` 已配置为 `platform: "tensorrt_plan"`，输出 `dets`/`labels`/`masks`。

### 8. RF-DETR

```bash
cd workspace/models/RFDETR_DET_PRE/1

# 转出onnx具体见官方仓库，目前支持的是不带seg的目标检测版本
# 注意：RF-DETR ONNX 有两个输出 dets [batch,300,4] 和 labels [batch,300,91]
trtexec --onnx=RFDETR_DET_PRE-small.sim.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=images:1x3x512x512 \
  --optShapes=images:1x3x512x512 \
  --maxShapes=images:16x3x512x512
```

RF-DETR 使用独立的预处理 `RFDETR_512_DIRECT_PREPROCESS`（512x512、ImageNet 归一化、swap_rb）。

### 转换后检查清单

1. `.plan` 文件放在对应模型的 `1/` 目录。
2. `config.pbtxt` 中的 `dims` 与 ONNX 输出严格一致。
3. `output_format`（`channel_first` / `anchor_first`）与 ONNX 排布一致。
4. `score_activation` 与导出时是否做 sigmoid 一致（Ultralytics 默认导出通常已做 sigmoid，填 `none`）。
5. 若修改了输入分辨率，同步修改 `YOLO_640_LETTERBOX_PREPROCESS` 的 `target_width/target_height` 和后处理 `input_width/input_height`。

---

## 测试

### 1. 健康检查

```bash
curl -s http://localhost:48000/v2/health/ready
curl -s http://localhost:48000/v2/health/live
```

### 2. 运行 Python 测试

`triton_client` 目录下保留了基于封装客户端的示例和测试脚本：

```bash
cd workspace/triton_client

# 单元测试（无需 Triton 服务）
python3 test_client.py

# 集成测试（需要本地 Triton 服务正在运行）
python3 test_client.py --integration

# 三协议调用示例
python3 example.py --protocol grpc
python3 example.py --protocol http
python3 example.py --protocol shm
```

更多客户端用法详见 [`workspace/triton_client/README.md`](workspace/triton_client/README.md)。

更完整的模型推理与可视化示例（含 gRPC/HTTP/SHM 三协议）见 [`workspace/examples/README.md`](workspace/examples/README.md)。

### 3. 可视化前端

浏览器打开 `http://localhost:8088`，选择 ensemble 模型并上传图片。

---

## 常见问题

**Q：构建时报错找不到 `tritonbackend.h`？**  
A：确保使用 `nvcr.io/nvidia/tritonserver:25.01-py3` 镜像，并在容器内执行 `bash /workspace/build.sh`。

**Q：模型启动时报错 `output shape mismatch`？**  
A：检查 `config.pbtxt` 中的 `dims` 是否与 TensorRT plan 的实际输出一致，特别注意 `channel_first` 与 `anchor_first`。

**Q：分割模型 mask 看不到或锯齿严重？**  
A：确认已重新构建并部署 `triton-display` 容器；浏览器需强制刷新（Ctrl+F5）。
