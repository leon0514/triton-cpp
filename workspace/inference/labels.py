"""Fetch model labels from the Triton ``CUSTOM_LABELS`` Python backend model."""

import argparse
import sys
from pathlib import Path
from typing import List

# Allow running this script directly from the package directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from triton_client import TritonClient

LABELS_MODEL_NAME = "CUSTOM_LABELS"


def get_labels(client: TritonClient, model_name: str) -> List[str]:
    """Return the label list for ``model_name`` from the ``CUSTOM_LABELS`` model.

    Args:
        client: An open ``TritonClient`` instance (any protocol).
        model_name: Name of the ensemble model whose labels should be fetched.

    Returns:
        List of label strings.
    """
    model_name_arr = np.array([model_name.encode("utf-8")], dtype=object)
    result = client.infer(
        LABELS_MODEL_NAME,
        inputs={"model_name": model_name_arr},
        outputs=["labels"],
    )
    labels = result["labels"]
    return [label.decode("utf-8") for label in labels]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch labels for a Triton ensemble model."
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--protocol",
        choices=["http", "grpc"],
        default="http",
        help="Protocol to use (default: http)",
    )
    parser.add_argument(
        "model_name",
        help="Ensemble model name, e.g. YOLOV5_DET_PRE_ENSEMBLE",
    )
    args = parser.parse_args()

    try:
        with TritonClient(args.url, protocol=args.protocol) as client:
            labels = get_labels(client, args.model_name)
            for label in labels:
                print(label)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
