#!/bin/bash
set -e

# Convert yolo11s-seg.pt -> ONNX -> TensorRT plan

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PT_FILE="${SCRIPT_DIR}/yolo11s-seg.pt"
ONNX_FILE="${SCRIPT_DIR}/yolo11s-seg.onnx"
PLAN_FILE="${SCRIPT_DIR}/model.plan"

echo "[convert] PT -> ONNX ..."
python3 - <<PY
from ultralytics import YOLO
model = YOLO("${PT_FILE}")
model.export(format="onnx", imgsz=640, half=False, simplify=True, dynamic=True)
PY

echo "[convert] ONNX -> TensorRT plan ..."
trtexec --onnx="${ONNX_FILE}" \
    --saveEngine="${PLAN_FILE}" \
    --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640

echo "[convert] Done: ${PLAN_FILE}"
ls -lh "${PLAN_FILE}"
