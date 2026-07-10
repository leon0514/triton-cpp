#!/bin/bash
# Helper to find trtexec inside a Docker image.
# Usage: bash find_trtexec.sh [image]
IMAGE="${1:-nvcr.io/nvidia/tritonserver:25.01-py3}"
echo "Searching for trtexec in $IMAGE ..."
docker run --rm --gpus all "$IMAGE" bash -c '
    echo "PATH=$PATH"
    for p in trtexec /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt/bin/trtexec /opt/tensorrt/bin/trtexec; do
        if command -v "$p" >/dev/null 2>&1; then
            echo "FOUND (in PATH): $p -> $(command -v $p)"
        elif [ -x "$p" ]; then
            echo "FOUND (executable): $p"
        fi
    done
    echo "--- all tensorrt bin dirs ---"
    find /usr /opt -name trtexec -type f 2>/dev/null | head -20 || true
'
