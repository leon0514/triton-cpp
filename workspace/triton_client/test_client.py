#!/usr/bin/env python3
"""Tests for triton_client.

Run without a server:
    python3 triton_client/test_client.py

Run with a local Triton server:
    python3 triton_client/test_client.py --integration
"""

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

# Make sure we test the local package when run from the workspace directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from triton_client import TritonClient, TritonClientError
from triton_client.client import _triton_type


class TestTypeConversion(unittest.TestCase):
    def test_triton_type(self):
        self.assertEqual(_triton_type(np.uint8), "UINT8")
        self.assertEqual(_triton_type(np.int32), "INT32")
        self.assertEqual(_triton_type(np.float32), "FP32")
        self.assertEqual(_triton_type("float16"), "FP16")
        self.assertEqual(_triton_type("int64"), "INT64")
        self.assertRaises(TritonClientError, _triton_type, "not_a_dtype")


class TestClientConstruction(unittest.TestCase):
    def test_invalid_protocol(self):
        with self.assertRaises(TritonClientError):
            TritonClient("localhost:8000", protocol="tcp")

    def test_context_manager(self):
        # Construction itself does not contact the server.
        with TritonClient("localhost:48000", protocol="http") as client:
            self.assertFalse(client._closed)
        self.assertTrue(client._closed)


class TestIntegration(unittest.TestCase):
    """Requires a running Triton server."""

    GRPC_URL = "localhost:48001"
    HTTP_URL = "localhost:48000"
    MODEL = "YOLOV5_DET_PRE_ENSEMBLE"
    IMAGE = "images/bus.jpg"

    def _load_image(self):
        from PIL import Image

        image_path = Path(self.IMAGE)
        if not image_path.exists():
            image_path = Path("workspace") / self.IMAGE
        img = Image.open(image_path).convert("RGB")
        return np.array(img)[np.newaxis, ...]

    def _output_specs(self):
        return {
            "num_dets": ([1, 1], "int32"),
            "detection_boxes": ([1, 300, 4], "float32"),
            "detection_scores": ([1, 300], "float32"),
            "detection_classes": ([1, 300], "int32"),
            "transform_metadata": ([1, 6], "float32"),
        }

    def _infer(self, protocol):
        url = self.GRPC_URL if protocol == "grpc" else self.HTTP_URL
        img_np = self._load_image()
        outputs = list(self._output_specs().keys())

        with TritonClient(url=url, protocol=protocol) as client:
            self.assertTrue(client.is_server_ready())
            self.assertTrue(client.is_model_ready(self.MODEL))

            kwargs = {}
            if protocol == "shm":
                kwargs["output_specs"] = self._output_specs()
            return client.infer(
                model_name=self.MODEL,
                inputs={"raw_image": img_np},
                outputs=outputs,
                **kwargs,
            )

    def test_grpc(self):
        result = self._infer("grpc")
        self.assertIn("num_dets", result)
        self.assertGreater(result["num_dets"].flat[0], 0)

    def test_http(self):
        result = self._infer("http")
        self.assertIn("num_dets", result)
        self.assertGreater(result["num_dets"].flat[0], 0)

    def test_shm(self):
        result = self._infer("shm")
        self.assertIn("num_dets", result)
        self.assertGreater(result["num_dets"].flat[0], 0)

    def test_shm_reuse(self):
        """Shared memory regions should be reused across requests."""
        url = self.HTTP_URL
        img_np = self._load_image()
        outputs = list(self._output_specs().keys())

        with TritonClient(url=url, protocol="shm") as client:
            for _ in range(3):
                result = client.infer(
                    model_name=self.MODEL,
                    inputs={"raw_image": img_np},
                    outputs=outputs,
                    output_specs=self._output_specs(),
                )
            self.assertGreater(result["num_dets"].flat[0], 0)

    def test_metadata(self):
        with TritonClient(self.HTTP_URL, protocol="http") as client:
            meta = client.get_model_metadata(self.MODEL)
            self.assertEqual(meta["name"], self.MODEL)
            self.assertTrue(any(t["name"] == "num_dets" for t in meta["outputs"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests against a live Triton server",
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTypeConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestClientConstruction))

    if args.integration:
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
