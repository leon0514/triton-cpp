"""Abstract protocol interface for Triton clients."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .errors import ProtocolError


def numpy_to_triton_dtype(array: np.ndarray) -> str:
    """Return the Triton datatype string for a numpy array."""
    dtype = array.dtype
    if dtype == np.float16:
        return "FP16"
    if dtype == np.float32:
        return "FP32"
    if dtype == np.float64:
        return "FP64"
    if dtype == np.int8:
        return "INT8"
    if dtype == np.int16:
        return "INT16"
    if dtype == np.int32:
        return "INT32"
    if dtype == np.int64:
        return "INT64"
    if dtype == np.uint8:
        return "UINT8"
    if dtype == np.uint16:
        return "UINT16"
    if dtype == np.uint32:
        return "UINT32"
    if dtype == np.uint64:
        return "UINT64"
    if dtype == np.bool_:
        return "BOOL"
    if dtype == object:
        return "BYTES"
    raise ProtocolError(f"Unsupported numpy dtype for Triton: {dtype}")


class TritonClientProtocol(ABC):
    """Common interface implemented by HTTP, gRPC and SHM clients."""

    @abstractmethod
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: List[str],
        *,
        model_version: str = "",
        output_specs: Optional[Dict[str, Tuple[List[int], str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """Run inference on ``model_name`` and return named output arrays.

        Args:
            model_name: Name of the Triton model/ensemble.
            inputs: Mapping from input tensor name to ``np.ndarray``.
            outputs: List of output tensor names to fetch.
            model_version: Optional model version. Empty string means latest.
            output_specs: Required by the SHM client; mapping from output name
                to ``(shape, dtype)`` so that output shared memory regions can
                be allocated before inference.
            **kwargs: Extra protocol-specific options forwarded to the client.

        Returns:
            Mapping from output tensor name to ``np.ndarray``. The arrays are
            owned by the caller and remain valid after the client is closed.
        """

    @abstractmethod
    def is_server_ready(self) -> bool:
        """Return ``True`` if the server is ready."""

    @abstractmethod
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """Return ``True`` if the model is ready."""

    @abstractmethod
    def get_model_metadata(self, model_name: str, model_version: str = "") -> dict:
        """Return model metadata as a dict."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources."""

    def __enter__(self) -> "TritonClientProtocol":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
