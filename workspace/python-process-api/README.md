# Triton Python Process API 示例

使用 `tritonserver` Python 包（in-process C API）直接在进程内加载模型仓库并推理，
无需启动独立的 tritonserver 服务进程，也不占用 HTTP/gRPC 端口。

当前示例（`main.py`）：加载 `YOLO26_DET_PRE_ENSEMBLE`（预处理 → TensorRT → 后处理的
ensemble 链），对 `bus.jpg` 推理，解析检测结果并绘制保存为 `result.jpg`。

## 运行环境

- 必须在 Triton 容器内运行（`tritonserver` Python 包随 NGC 镜像提供）：
  `nvcr.io/nvidia/tritonserver:25.01-py3`
- 模型仓库挂载到 `/models`（即本仓库的 `workspace/models`）
- 需要 `opencv-python-headless`（镜像默认不带 cv2）
- 需要一块空闲显存足够的 GPU（配置中 `gpus: [0]` 指容器内可见的第一块卡）

示例：

```bash
docker run --rm --gpus '"device=0"' \
  -v "$PWD/workspace/models:/models:ro" \
  -v "$PWD/workspace/python-process-api:/app" \
  -v "$PWD/workspace/images/bus.jpg:/app/bus.jpg:ro" \
  nvcr.io/nvidia/tritonserver:25.01-py3 \
  bash -c "pip install -q opencv-python-headless && cd /app && python3 main.py"
```

## 输出说明

ensemble 输出 5 个字段：

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| `num_dets` | `[1, 1]` | 实际检测框数量 N |
| `detection_boxes` | `[1, N, 4]` | 检测框 `[x1, y1, x2, y2]`，**640×640 letterbox 坐标系** |
| `detection_scores` | `[1, N]` | 置信度 |
| `detection_classes` | `[1, N]` | COCO 类别 id（类别名见 `/models/CUSTOM_LABELS/names/<模型名>.txt`） |
| `transform_metadata` | `[1, 6]` | letterbox 的 d2i 逆仿射矩阵（dst → 原图） |

`detection_boxes` 在 letterbox 坐标系，需要用 `transform_metadata`（d2i 矩阵）映射回原图：

```
x_orig = d2i[0] * x + d2i[1] * y + d2i[2]
y_orig = d2i[3] * x + d2i[4] * y + d2i[5]
```

`main.py` 中的 `draw_detections()` 演示了完整映射与绘制流程。

## 关键注意事项（踩坑记录）

1. **按需加载模型**：in-process 默认在 `start()` 时全量加载整个 `/models` 仓库
   （显存翻倍、启动慢，且仓库内无关模型可能在关闭时卸载超时）。示例使用
   `ModelControlMode.EXPLICIT + startup_models=[...]`，等价于
   `triton_entrypoint.sh` 的 `--model-control-mode=explicit --load-model=X`，
   ensemble 依赖的子模型会自动加载。
2. **请求绑定创建它的模型**：必须先 `model = server.model(name)` 再
   `request = model.create_request()`。`model.infer(request)` 不会切换请求
   所属的模型——用错模型创建的请求会打到错误的模型上（不报错但结果不对）。
3. **等待就绪**：`server.start(wait_until_ready=True, timeout=120)`，否则模型
   还没加载完就 infer 会报 not ready。
4. **优雅关闭**：结尾调用 `server.stop()`，否则仓库中的 Python 后端模型
   （CUSTOM_LABELS）的 stub 子进程会打印 `Non-graceful termination detected`。
   个别模型卸载可能超过 `exit_timeout`（默认 30s）触发
   `InternalError: Exit timeout expired`，发生在推理完成后，可安全忽略。
5. **输入校验**：`cv2.imread` 失败返回 `None`，`np.expand_dims(None, 0)` 会变成
   object 数组，被序列化为 BYTES 类型，报
   `input 'raw_image' data-type is 'BYTES', but model expects 'UINT8'`。
   报错前先确认图片路径（`main.py` 已加判空）。
6. **输出转 numpy**：优先 `np.from_dlpack(tensor)`，失败时回退按 `data_ptr`
   直读（请求已设置 `output_memory_type=MemoryType.CPU` 保证 CPU 内存）。
7. **日志**：`os.environ["TRITONSERVER_LOG_VERBOSE"]` 对 in-process 无效；
   详细日志应使用 `tritonserver.Server(log_verbose=3)`。

## 文件

- `main.py` — 完整示例：按需加载 → 推理 → 解析 → d2i 映射 → 绘制 → 优雅关闭
