"""Tests for label fetching utilities.

Unit tests do not require a running Triton server. Integration tests are run
only when ``--integration`` is passed.
"""

import sys
import unittest
from pathlib import Path

# Allow running this script directly from the package directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from inference.labels import LABELS_MODEL_NAME, get_labels
from triton_client import TritonClient
from triton_client.errors import ProtocolError


class TestLabelsUnit(unittest.TestCase):
    def test_model_name_constant(self):
        self.assertEqual(LABELS_MODEL_NAME, "CUSTOM_LABELS")

    def test_input_array_encoding(self):
        arr = np.array(["YOLOV5_DET_PRE_ENSEMBLE".encode("utf-8")], dtype=object)
        self.assertEqual(arr.shape, (1,))
        self.assertIsInstance(arr[0], bytes)


class TestLabelsIntegration(unittest.TestCase):
    """Run only when a Triton server is available."""

    def test_fetch_labels(self):
        with TritonClient("localhost:48000", protocol="http") as client:
            labels = get_labels(client, "YOLOV5_DET_PRE_ENSEMBLE")
            self.assertIsInstance(labels, list)
            if labels:
                self.assertIsInstance(labels[0], str)


class TestLabelsConnectionFailure(unittest.TestCase):
    def test_raises_protocol_error_on_connection_failure(self):
        with TritonClient("localhost:59999", protocol="http") as client:
            with self.assertRaises(ProtocolError):
                get_labels(client, "YOLOV5_DET_PRE_ENSEMBLE")


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestLabelsUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelsConnectionFailure))

    if "--integration" in sys.argv:
        sys.argv.remove("--integration")
        suite.addTests(loader.loadTestsFromTestCase(TestLabelsIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    main()
