#!/usr/bin/env python3
"""
Convert YOLOv5 ONNX model to TensorRT .plan for Triton Inference Server.

Run this script inside a Triton/TensorRT Docker container, e.g.:
    docker run --rm -it --gpus all \
        -v $(pwd):/workspace \
        -w /workspace \
        nvcr.io/nvidia/tritonserver:25.01-py3-sdk \
        python3 workspace/convert_yolov5_to_trt.py

Or with the runtime image (also contains TensorRT):
    docker run --rm -it --gpus all \
        -v $(pwd):/workspace \
        -w /workspace \
        nvcr.io/nvidia/tritonserver:25.01-py3 \
        python3 workspace/convert_yolov5_to_trt.py
"""

import argparse
import os
import sys


def convert_with_tensorrt_api(onnx_path, plan_path, fp16=False, workspace_mb=1024):
    """Convert ONNX to TensorRT engine using Python API."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: ONNX parse failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_mb * (1 << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build engine for the explicit batch shape from ONNX.
    print(f"Building TensorRT engine (fp16={fp16}) ...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("ERROR: Engine build failed")
        sys.exit(1)

    os.makedirs(os.path.dirname(plan_path) or ".", exist_ok=True)
    with open(plan_path, "wb") as f:
        f.write(engine_bytes)

    print(f"Saved TensorRT engine to {plan_path}")


def convert_with_trtexec(onnx_path, plan_path, fp16=False, workspace_mb=1024):
    """Fallback: convert ONNX to TensorRT engine using trtexec CLI."""
    import subprocess

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        f"--workspace={workspace_mb}",
        "--explicitBatch",
    ]
    if fp16:
        cmd.append("--fp16")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("ERROR: trtexec failed")
        sys.exit(1)

    print(f"Saved TensorRT engine to {plan_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv5 ONNX to TensorRT plan"
    )
    parser.add_argument(
        "--onnx",
        default="workspace/models/yolov5/1/model.onnx",
        help="Path to input ONNX model",
    )
    parser.add_argument(
        "--plan",
        default="workspace/models/yolov5/1/model.plan",
        help="Path to output TensorRT plan",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Build FP16 engine"
    )
    parser.add_argument(
        "--workspace", type=int, default=1024, help="Workspace size in MB"
    )
    parser.add_argument(
        "--use-trtexec",
        action="store_true",
        help="Force use of trtexec CLI instead of TensorRT Python API",
    )
    args = parser.parse_args()

    if args.use_trtexec:
        convert_with_trtexec(args.onnx, args.plan, args.fp16, args.workspace)
    else:
        try:
            convert_with_tensorrt_api(
                args.onnx, args.plan, args.fp16, args.workspace
            )
        except ImportError:
            print("tensorrt Python API not available, falling back to trtexec")
            convert_with_trtexec(
                args.onnx, args.plan, args.fp16, args.workspace
            )


if __name__ == "__main__":
    main()
