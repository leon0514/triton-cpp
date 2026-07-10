#!/bin/bash
# Convert YOLOv5 ONNX to TensorRT .plan using trtexec inside Docker.
#
# Usage:
#   cd <project_root>
#   bash workspace/convert_yolov5_plan.sh
#
# The script mounts the project root to /workspace inside the container
# and runs trtexec with fixed 640x640 shape and dynamic batch [1, 16].

set -e

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

ONNX_HOST="${PROJECT_ROOT}/workspace/models/yolov5/1/model.onnx"
PLAN_HOST="${PROJECT_ROOT}/workspace/models/yolov5/1/model.plan"

# NOTE: Use the Triton *runtime* image here, not the SDK image.
# The runtime image contains trtexec; the SDK image does not.
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/tritonserver:25.01-py3}"
BATCH_MIN="${BATCH_MIN:-1}"
BATCH_OPT="${BATCH_OPT:-16}"
BATCH_MAX="${BATCH_MAX:-16}"
FP16_FLAG="${FP16_FLAG:---fp16}"

if [ ! -f "$ONNX_HOST" ]; then
    echo "ERROR: ONNX model not found: $ONNX_HOST" >&2
    echo "Please export YOLOv5 ONNX first, e.g.:" >&2
    echo "  cd /tmp && git clone https://github.com/ultralytics/yolov5.git" >&2
    echo "  cd yolov5 && python export.py --weights yolov5s.pt --img 640 --batch 16 --include onnx --dynamic" >&2
    exit 1
fi

# Paths inside the container (project root is mounted to /workspace)
ONNX_CONTAINER="/workspace/workspace/models/yolov5/1/model.onnx"
PLAN_CONTAINER="/workspace/workspace/models/yolov5/1/model.plan"

echo "Converting YOLOv5 ONNX to TensorRT plan ..."
echo "  Host ONNX:   $ONNX_HOST"
echo "  Host plan:   $PLAN_HOST"
echo "  Docker image: $DOCKER_IMAGE"
echo "  Batch shapes: $BATCH_MIN / $BATCH_OPT / $BATCH_MAX"
echo "  FP16:        $FP16_FLAG"

# Remove old plan if exists
rm -f "$PLAN_HOST"

# trtexec may not be in PATH in some Triton images; try common locations.
TRTEXEC_PATHS=(
    "trtexec"
    "/usr/src/tensorrt/bin/trtexec"
    "/usr/local/tensorrt/bin/trtexec"
    "/opt/tensorrt/bin/trtexec"
    "/usr/local/bin/trtexec"
)

TRTEXEC_CMD=""
for path in "${TRTEXEC_PATHS[@]}"; do
    if docker run --rm "$DOCKER_IMAGE" bash -c "command -v $path || test -x $path" >/dev/null 2>&1; then
        TRTEXEC_CMD="$path"
        break
    fi
done

if [ -z "$TRTEXEC_CMD" ]; then
    echo "ERROR: trtexec not found in container. Searched: ${TRTEXEC_PATHS[*]}" >&2
    echo "Please use an image that includes trtexec, e.g.:" >&2
    echo "  nvcr.io/nvidia/tensorrt:25.01-py3" >&2
    echo "  nvcr.io/nvidia/tritonserver:25.01-py3 (usually has it under /usr/src/tensorrt/bin)" >&2
    echo "You can override the image with: DOCKER_IMAGE=<image> bash $0" >&2
    exit 1
fi

echo "Found trtexec: $TRTEXEC_CMD"

docker run --rm -it --gpus all \
    -v "${PROJECT_ROOT}:/workspace" \
    -w /workspace \
    "$DOCKER_IMAGE" \
    "$TRTEXEC_CMD" \
        --onnx="$ONNX_CONTAINER" \
        --saveEngine="$PLAN_CONTAINER" \
        --minShapes="images:${BATCH_MIN}x3x640x640" \
        --optShapes="images:${BATCH_OPT}x3x640x640" \
        --maxShapes="images:${BATCH_MAX}x3x640x640" \
        $FP16_FLAG

if [ -f "$PLAN_HOST" ]; then
    echo ""
    echo "SUCCESS: TensorRT plan saved to $PLAN_HOST"
    ls -lh "$PLAN_HOST"
else
    echo "ERROR: Plan file not created" >&2
    exit 1
fi
