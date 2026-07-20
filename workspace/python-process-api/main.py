import ctypes
import time
import tritonserver
import cv2
import numpy as np

import os
os.environ["TRITONBACKEND_PYTHON_ENABLE_STATS"] = "1"
# 开启 Python 后端详细日志
os.environ["PYTORCH_BACKEND_LOG_LEVEL"] = "VERBOSE" 
os.environ["TRITONSERVER_LOG_VERBOSE"] = "3"

server = tritonserver.Server(
    model_repository="/models",
    exit_on_error=True,
    # 与 triton_entrypoint.sh 的 --model-control-mode=explicit --load-model 等效：
    # 只加载目标 ensemble 链（preprocess/TRT/postprocess 会作为依赖自动加载），
    # 避免全量加载整个仓库（显存翻倍、启动慢，也绕开关闭时卸载超时的模型）
    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    startup_models=["YOLO11_DET_PRE_ENSEMBLE"],
)


server.start(wait_until_ready=True, timeout=120)


# list models 
models = server._model_index()

print("Available models:")
for model in models:
    print(f"- {model.name} (version: {model.version}, state: {model.state})")

# test inference for YOLO26 Ensemble model
model_name = "YOLO11_DET_PRE_ENSEMBLE"
image = cv2.imread("/images/bus.jpg")
if image is None:
    raise FileNotFoundError("bus.jpg 读取失败：请确认文件存在于当前工作目录")

image_batch = np.expand_dims(image, axis=0)  # Add batch dimension

model = server.model(model_name)

start = time.time()
for _ in range(1000):
    request = model.create_request()
    # 让 Triton 把输出分配到 CPU 内存，保证下方 tensor_to_numpy 两条路径都可用
    request.output_memory_type = tritonserver.MemoryType.CPU
    request.inputs["raw_image"] = image_batch

    response = model.infer(request)

    response_obj = next(iter(response))

    # print(f"Response for model {model_name}:")
    # for output_name, output_data in response_obj.outputs.items():
    #     print(f"- {output_name}: {output_data.shape}")
end = time.time()
print(f"Average inference time for 1000 runs: {(end - start) / 1000:.4f} seconds per run")
print(f"Average FPS: {1000 / (end - start):.2f} frames per second")

# ---- 解析输出并绘制到图上 ----
DTYPE_MAP = {"FP32": np.float32, "INT32": np.int32, "UINT8": np.uint8}


def tensor_to_numpy(t):
    """tritonserver.Tensor -> numpy。优先 DLPack，失败则按 data_ptr 直接读（CPU 内存）。"""
    try:
        return np.from_dlpack(t)
    except Exception:
        dtype = DTYPE_MAP[str(t.data_type).rsplit(".", 1)[-1]]
        nbytes = int(np.prod(t.shape)) * np.dtype(dtype).itemsize
        buf = (ctypes.c_char * nbytes).from_address(t.data_ptr)
        return np.frombuffer(buf, dtype=dtype).reshape(t.shape).copy()


def draw_detections(image, boxes, scores, classes, d2i, labels, out_path="result.jpg"):
    """boxes 为 640x640 letterbox 坐标，用 d2i（dst->image 逆仿射矩阵）映射回原图后绘制。"""
    h, w = image.shape[:2]
    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        X1 = d2i[0] * x1 + d2i[1] * y1 + d2i[2]
        Y1 = d2i[3] * x1 + d2i[4] * y1 + d2i[5]
        X2 = d2i[0] * x2 + d2i[1] * y2 + d2i[2]
        Y2 = d2i[3] * x2 + d2i[4] * y2 + d2i[5]
        p1 = (int(np.clip(X1, 0, w - 1)), int(np.clip(Y1, 0, h - 1)))
        p2 = (int(np.clip(X2, 0, w - 1)), int(np.clip(Y2, 0, h - 1)))

        cls = int(cls)
        name = labels[cls] if labels and cls < len(labels) else str(cls)
        text = f"{name} {score:.2f}"
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), (0, 255, 0), -1)
        cv2.putText(image, text, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        print(f"  {text}: box=({p1[0]}, {p1[1]}, {p2[0]}, {p2[1]})")
    cv2.imwrite(out_path, image)
    print(f"saved: {out_path}")


outputs = response_obj.outputs
num = int(tensor_to_numpy(outputs["num_dets"])[0, 0])
boxes = tensor_to_numpy(outputs["detection_boxes"])[0, :num]
scores = tensor_to_numpy(outputs["detection_scores"])[0, :num]
classes = tensor_to_numpy(outputs["detection_classes"])[0, :num]
d2i = tensor_to_numpy(outputs["transform_metadata"])[0]

labels_path = "/models/CUSTOM_LABELS/names/YOLO26_DET_PRE_ENSEMBLE.txt"
labels = open(labels_path).read().splitlines() if os.path.exists(labels_path) else None

print(f"num_dets = {num}")
draw_detections(image, boxes, scores, classes, d2i, labels)

# 优雅关闭 server。个别模型卸载可能超过 exit_timeout（默认 30s）而强制退出，
# 关闭阶段的错误不影响已拿到的推理结果，直接忽略
try:
    server.stop()
except tritonserver.InternalError:
    pass