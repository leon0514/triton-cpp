#!/bin/bash
# Build script for Triton custom backends (preprocess, yolo11_postprocess, yolo11_pose_postprocess, yolo11_obb_postprocess, yolo11_seg_postprocess, yolov5_postprocess, yolo26_postprocess, rfdetr_postprocess, classifier_postprocess)
# Recommended to run inside nvcr.io/nvidia/tritonserver:25.01-py3

set -e

SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"

# Default CUDA architectures for Ampere/Ada
CUDA_ARCHS="${CUDA_ARCHS:-"80;86;89"}"

# Auto-detect Triton installation if not explicitly set.
# The runtime image installs Triton under /opt/tritonserver.
if [ -z "${TRITON_DIR}" ]; then
    for candidate in /opt/tritonserver /usr /usr/local /workspace/install; do
        if [ -f "${candidate}/include/triton/core/tritonbackend.h" ]; then
            TRITON_DIR="${candidate}"
            break
        fi
    done
fi

# Default Triton installation path inside the runtime image
TRITON_DIR="${TRITON_DIR:-/opt/tritonserver}"

if [ ! -f "${TRITON_DIR}/include/triton/core/tritonbackend.h" ]; then
    echo "[build.sh] ERROR: Cannot find Triton backend headers at ${TRITON_DIR}/include/triton/core/tritonbackend.h" >&2
    echo "[build.sh] Searching common locations for tritonbackend.h ..." >&2
    find /workspace /opt /usr -name "tritonbackend.h" -print 2>/dev/null | head -20 >&2 || true
    echo "[build.sh] Listing /workspace/install if present:" >&2
    ls -la /workspace/install 2>/dev/null >&2 || true
    echo "[build.sh] Listing /opt/tritonserver if present:" >&2
    ls -la /opt/tritonserver 2>/dev/null >&2 || true

    echo "[build.sh] This usually means you are using the wrong Triton container image." >&2
    echo "[build.sh] Please use the runtime image: nvcr.io/nvidia/tritonserver:25.01-py3" >&2
    echo "[build.sh] Or set TRITON_DIR to a Triton installation that contains include/triton/core/tritonbackend.h" >&2
    exit 1
fi

# Ensure cmake is available. The runtime image may not have it pre-installed.
if ! command -v cmake &> /dev/null; then
    echo "[build.sh] cmake not found, trying to install via pip..."
    python3 -m pip install --user cmake || pip3 install --user cmake || pip install --user cmake

    # Update PATH to include the user-level binary directory
    export PATH="${HOME}/.local/bin:${PATH}"

    if ! command -v cmake &> /dev/null; then
        echo "[build.sh] ERROR: cmake installation failed or is not in PATH." >&2
        echo "[build.sh] Please install cmake manually, e.g.:" >&2
        echo "  apt-get update && apt-get install -y cmake" >&2
        echo "  or" >&2
        echo "  pip install cmake" >&2
        exit 1
    fi
fi

echo "[build.sh] Using cmake: $(command -v cmake)"
echo "[build.sh] Using TRITON_DIR: ${TRITON_DIR}"
echo "[build.sh] Using CUDA_ARCHITECTURES: ${CUDA_ARCHS}"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DTRITON_DIR="${TRITON_DIR}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"

echo "Build complete. Shared libraries in ${BUILD_DIR}:"
ls -1 ${BUILD_DIR}/libtriton_*.so 2>/dev/null || true
echo "Copy the required .so to <model>/1/libtriton_<backend>.so or <backend_dir>/<backend>/libtriton_<backend>.so"
