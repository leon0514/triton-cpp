"""System shared memory wrapper and SHM-backed Triton client."""

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonclient.utils.shared_memory as shm
from tritonclient.utils import InferenceServerException

from .base import TritonClientProtocol, numpy_to_triton_dtype
from .errors import ProtocolError, SharedMemoryError


def _triton_dtype_to_numpy(dtype: str) -> np.dtype:
    """Convert a Triton datatype string to a numpy dtype."""
    mapping = {
        "FP16": np.float16,
        "FP32": np.float32,
        "FP64": np.float64,
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
        "BOOL": np.bool_,
    }
    if dtype not in mapping:
        raise SharedMemoryError(f"Unsupported Triton dtype: {dtype}")
    return np.dtype(mapping[dtype])


def _byte_size(shape: List[int], dtype: str) -> int:
    """Return the byte size of a tensor with the given shape and Triton dtype."""
    return int(np.prod(shape)) * _triton_dtype_to_numpy(dtype).itemsize


class SharedMemoryRegion:
    """Thin wrapper around a Triton system shared memory region.

    The region is lazily created on first use and must be explicitly destroyed
    when it is no longer needed.
    """

    def __init__(self, name: str, key: str, byte_size: int) -> None:
        self.name = name
        self.key = key
        self.byte_size = byte_size
        self._handle: Optional[Any] = None

    def create(self) -> "SharedMemoryRegion":
        """Create the underlying shared memory region."""
        if self._handle is not None:
            return self
        try:
            self._handle = shm.create_shared_memory_region(
                self.name, self.key, self.byte_size
            )
        except Exception as exc:
            raise SharedMemoryError(
                f"Failed to create shared memory region '{self.name}': {exc}"
            ) from exc
        return self

    def destroy(self) -> None:
        """Destroy the underlying shared memory region."""
        if self._handle is None:
            return
        try:
            shm.destroy_shared_memory_region(self._handle)
        except Exception as exc:
            raise SharedMemoryError(
                f"Failed to destroy shared memory region '{self.name}': {exc}"
            ) from exc
        finally:
            self._handle = None

    def write(self, array: np.ndarray, offset: int = 0) -> None:
        """Copy ``array`` into the shared memory region at ``offset``."""
        if self._handle is None:
            raise SharedMemoryError(
                f"Shared memory region '{self.name}' has not been created"
            )
        try:
            shm.set_shared_memory_region(self._handle, [array], offset=offset)
        except Exception as exc:
            raise SharedMemoryError(
                f"Failed to write shared memory region '{self.name}': {exc}"
            ) from exc

    def read(self, shape: List[int], dtype: str, offset: int = 0) -> np.ndarray:
        """Return a copy of the data stored in the region."""
        if self._handle is None:
            raise SharedMemoryError(
                f"Shared memory region '{self.name}' has not been created"
            )
        try:
            array = shm.get_contents_as_numpy(
                self._handle, _triton_dtype_to_numpy(dtype), shape, offset=offset
            )
        except Exception as exc:
            raise SharedMemoryError(
                f"Failed to read shared memory region '{self.name}': {exc}"
            ) from exc
        # Return a copy so the caller owns the data independently of the SHM region.
        return np.copy(array)


class TritonSharedMemoryClient(TritonClientProtocol):
    """Triton client that communicates input/output tensors via system SHM.

    A lightweight HTTP or gRPC client is used internally to register/unregister
    the shared memory regions with the server and to trigger inference.
    """

    def __init__(
        self,
        url: str,
        *,
        control_protocol: str = "grpc",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create an SHM-backed client.

        Args:
            url: ``host:port`` of the Triton endpoint.
            control_protocol: Underlying control protocol, either ``"grpc"`` or ``"http"``.
            verbose: Enable verbose client output.
            **kwargs: Forwarded to the underlying client.
        """
        self._url = url
        self._protocol = control_protocol.lower()
        self._input_regions: Dict[str, SharedMemoryRegion] = {}
        self._output_regions: Dict[str, SharedMemoryRegion] = {}
        if self._protocol == "grpc":
            self._client = grpcclient.InferenceServerClient(
                url, verbose=verbose, **kwargs
            )
            self._module = grpcclient
        elif self._protocol == "http":
            self._client = httpclient.InferenceServerClient(
                url, verbose=verbose, **kwargs
            )
            self._module = httpclient
        else:
            raise ProtocolError(
                f"Unsupported SHM control protocol '{control_protocol}', use grpc or http"
            )

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
        if output_specs is None:
            raise SharedMemoryError(
                "output_specs is required for shared memory inference"
            )

        missing = set(outputs) - set(output_specs)
        if missing:
            raise SharedMemoryError(
                f"Missing output_specs for outputs: {sorted(missing)}"
            )

        input_regions = {
            name: self._get_or_create_input_region(name, array)
            for name, array in inputs.items()
        }
        output_regions = {
            name: self._get_or_create_output_region(name, output_specs[name])
            for name in outputs
        }

        for name, array in inputs.items():
            input_regions[name].write(array)

        infer_inputs = [
            self._build_shm_input(name, array, input_regions[name])
            for name, array in inputs.items()
        ]
        infer_outputs = [
            self._build_shm_output(name, output_specs[name], output_regions[name])
            for name in outputs
        ]

        try:
            response = self._client.infer(
                model_name,
                infer_inputs,
                model_version=model_version,
                outputs=infer_outputs,
                **kwargs,
            )
        except (InferenceServerException, ConnectionError) as exc:
            raise ProtocolError(f"SHM inference failed: {exc}") from exc

        # Touch the response so the server completes the write before we read.
        _ = response.get_response()

        return {
            name: output_regions[name].read(*output_specs[name]) for name in outputs
        }

    def is_server_ready(self) -> bool:
        try:
            return self._client.is_server_ready()
        except InferenceServerException as exc:
            raise ProtocolError(f"Failed to check server readiness: {exc}") from exc

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        try:
            return self._client.is_model_ready(model_name, model_version=model_version)
        except InferenceServerException as exc:
            raise ProtocolError(f"Failed to check model readiness: {exc}") from exc

    def get_model_metadata(self, model_name: str, model_version: str = "") -> dict:
        try:
            return self._client.get_model_metadata(
                model_name, model_version=model_version
            )
        except InferenceServerException as exc:
            raise ProtocolError(f"Failed to get model metadata: {exc}") from exc

    def close(self) -> None:
        for region in self._input_regions.values():
            self._unregister_region(region)
            region.destroy()
        self._input_regions.clear()
        for region in self._output_regions.values():
            self._unregister_region(region)
            region.destroy()
        self._output_regions.clear()
        self._client.close()

    def _get_or_create_input_region(
        self, name: str, array: np.ndarray
    ) -> SharedMemoryRegion:
        """Return a cached input region or create a new one large enough."""
        byte_size = array.nbytes
        region = self._input_regions.get(name)
        if region is not None and region.byte_size >= byte_size:
            return region
        if region is not None:
            self._unregister_region(region)
            region.destroy()
        shm_name, shm_key = self._unique_names(f"input_{name}")
        region = SharedMemoryRegion(shm_name, shm_key, byte_size)
        region.create()
        self._register_region(region)
        self._input_regions[name] = region
        return region

    def _get_or_create_output_region(
        self, name: str, spec: Tuple[List[int], str]
    ) -> SharedMemoryRegion:
        """Return a cached output region or create a new one large enough."""
        shape, dtype = spec
        byte_size = _byte_size(shape, dtype)
        region = self._output_regions.get(name)
        if region is not None and region.byte_size >= byte_size:
            return region
        if region is not None:
            self._unregister_region(region)
            region.destroy()
        shm_name, shm_key = self._unique_names(f"output_{name}")
        region = SharedMemoryRegion(shm_name, shm_key, byte_size)
        region.create()
        self._register_region(region)
        self._output_regions[name] = region
        return region

    def _register_region(self, region: SharedMemoryRegion) -> None:
        try:
            self._client.register_system_shared_memory(
                region.name, region.key, region.byte_size
            )
        except InferenceServerException as exc:
            raise SharedMemoryError(
                f"Failed to register shared memory region '{region.name}': {exc}"
            ) from exc

    def _unregister_region(self, region: SharedMemoryRegion) -> None:
        try:
            self._client.unregister_system_shared_memory(region.name)
        except Exception:
            # Ignore cleanup errors so that destroy still runs.
            pass

    def _build_shm_input(
        self, name: str, array: np.ndarray, region: SharedMemoryRegion
    ) -> Any:
        input_tensor = self._module.InferInput(
            name, list(array.shape), numpy_to_triton_dtype(array)
        )
        # Tell the server the actual tensor byte size, which may be smaller
        # than the cached shared memory region capacity.
        input_tensor.set_shared_memory(region.name, array.nbytes)
        return input_tensor

    def _build_shm_output(
        self,
        name: str,
        spec: Tuple[List[int], str],
        region: SharedMemoryRegion,
    ) -> Any:
        shape, dtype = spec
        output_tensor = self._module.InferRequestedOutput(name)
        # Tell the server the actual tensor byte size, which may be smaller
        # than the cached shared memory region capacity.
        output_tensor.set_shared_memory(region.name, _byte_size(shape, dtype))
        return output_tensor

    def _unique_names(self, prefix: str) -> Tuple[str, str]:
        """Generate unique Triton SHM name and POSIX SHM key."""
        token = f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        return f"triton_{token}", f"/triton_{token}"
