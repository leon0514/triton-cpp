#!/usr/bin/env python3
"""Example: invoke a Triton ensemble with gRPC, HTTP or shared memory.

Usage:
    python3 triton_client/example.py --protocol grpc
    python3 triton_client/example.py --protocol http
    python3 triton_client/example.py --protocol shm
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from triton_client import TritonClient


def main():
    parser = argparse.ArgumentParser(description="Triton client wrapper example")
    parser.add_argument(
        "--protocol",
        choices=["grpc", "http", "shm"],
        default="grpc",
        help="Inference protocol",
    )
    parser.add_argument(
        "--model",
        default="YOLOV5_DET_PRE_ENSEMBLE",
        help="Model or ensemble name",
    )
    parser.add_argument(
        "--image",
        default="images/bus.jpg",
        help="Input image path (default: images/bus.jpg, relative to workspace dir)",
    )
    parser.add_argument(
        "--grpc-url",
        default="localhost:48001",
        help="gRPC server URL",
    )
    parser.add_argument(
        "--http-url",
        default="localhost:48000",
        help="HTTP/SHM server URL",
    )
    args = parser.parse_args()

    url = args.grpc_url if args.protocol == "grpc" else args.http_url

    image_path = Path(args.image)
    if not image_path.exists() and args.image == "images/bus.jpg":
        image_path = Path("workspace") / args.image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)[np.newaxis, ...]  # [1, H, W, 3]

    outputs = [
        "num_dets",
        "detection_boxes",
        "detection_scores",
        "detection_classes",
        "transform_metadata",
    ]

    # Shared memory requires explicit output buffer specs.
    output_specs = {
        "num_dets": ([1, 1], "int32"),
        "detection_boxes": ([1, 300, 4], "float32"),
        "detection_scores": ([1, 300], "float32"),
        "detection_classes": ([1, 300], "int32"),
        "transform_metadata": ([1, 6], "float32"),
    }

    print(f"Protocol: {args.protocol}")
    print(f"URL:      {url}")
    print(f"Model:    {args.model}")
    print(f"Image:    {args.image}")

    with TritonClient(url=url, protocol=args.protocol) as client:
        print(f"Server ready: {client.is_server_ready()}")
        print(f"Model ready:  {client.is_model_ready(args.model)}")

        infer_kwargs = {}
        if args.protocol == "shm":
            infer_kwargs["output_specs"] = output_specs

        result = client.infer(
            model_name=args.model,
            inputs={"raw_image": img_np},
            outputs=outputs,
            **infer_kwargs,
        )

    print(f"num_dets: {result['num_dets']}")
    print(f"top 5 boxes:  {result['detection_boxes'][0, :5]}")
    print(f"top 5 scores: {result['detection_scores'][0, :5]}")
    print(f"top 5 classes:{result['detection_classes'][0, :5]}")


if __name__ == "__main__":
    main()
