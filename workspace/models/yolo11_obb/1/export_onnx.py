from ultralytics import YOLO
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n-obb.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=True,
        half=False,
        simplify=True,
        device=args.device,
    )
    print(f"ONNX exported to: {out}")


if __name__ == "__main__":
    main()
