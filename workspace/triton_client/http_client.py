"""HTTP client wrapper for Triton Inference Server."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from .base import TritonClientProtocol, numpy_to_triton_dtype
from .errors import ProtocolError


class TritonHttpClient(TritonClientProtocol):
    """Synchronous HTTP client wrapper."""

    def __init__(self, url: str, *, verbose: bool = False, **kwargs: Any) -> None:
        """Create an HTTP client.

        Args:
            url: ``host:port`` of the Triton HTTP endpoint.
            verbose: Enable verbose client output.
            **kwargs: Forwarded to ``httpclient.InferenceServerClient``.
        """
        self._url = url
        self._client = httpclient.InferenceServerClient(url, verbose=verbose, **kwargs)

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
        infer_inputs = [
            self._build_input(name, array) for name, array in inputs.items()
        ]
        infer_outputs = [httpclient.InferRequestedOutput(name) for name in outputs]

        try:
            response = self._client.infer(
                model_name,
                infer_inputs,
                model_version=model_version,
                outputs=infer_outputs,
                **kwargs,
            )
        except (InferenceServerException, ConnectionError) as exc:
            raise ProtocolError(f"HTTP inference failed: {exc}") from exc

        return {name: response.as_numpy(name) for name in outputs}

    def is_server_ready(self) -> bool:
        try:
            return self._client.is_server_ready()
        except (InferenceServerException, ConnectionError) as exc:
            raise ProtocolError(f"Failed to check server readiness: {exc}") from exc

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        try:
            return self._client.is_model_ready(model_name, model_version=model_version)
        except (InferenceServerException, ConnectionError) as exc:
            raise ProtocolError(f"Failed to check model readiness: {exc}") from exc

    def get_model_metadata(self, model_name: str, model_version: str = "") -> dict:
        try:
            return self._client.get_model_metadata(
                model_name, model_version=model_version
            )
        except (InferenceServerException, ConnectionError) as exc:
            raise ProtocolError(f"Failed to get model metadata: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def _build_input(self, name: str, array: np.ndarray) -> httpclient.InferInput:
        input_tensor = httpclient.InferInput(
            name, list(array.shape), numpy_to_triton_dtype(array)
        )
        input_tensor.set_data_from_numpy(array)
        return input_tensor
