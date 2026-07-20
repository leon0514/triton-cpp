"""Triton inference client wrapper supporting gRPC, HTTP and shared memory."""

from typing import Any

from .base import TritonClientProtocol, numpy_to_triton_dtype
from .errors import ProtocolError, SharedMemoryError, TritonClientError
from .grpc_client import TritonGrpcClient
from .http_client import TritonHttpClient
from .shm_client import SharedMemoryRegion, TritonSharedMemoryClient


class TritonClient:
    """Factory for creating a Triton client over HTTP, gRPC or shared memory.

    Examples:
        >>> with TritonClient("localhost:8001", protocol="grpc") as client:
        ...     result = client.infer("model", {"input": array}, ["output"])

        >>> with TritonClient("localhost:8000", protocol="http") as client:
        ...     result = client.infer("model", {"input": array}, ["output"])

        >>> specs = {"output": ([1, 3, 224, 224], "FP32")}
        >>> with TritonClient("localhost:8001", protocol="shm") as client:
        ...     result = client.infer(
        ...         "model", {"input": array}, ["output"], output_specs=specs
        ...     )
    """

    _CLIENTS = {
        "http": TritonHttpClient,
        "grpc": TritonGrpcClient,
        "shm": TritonSharedMemoryClient,
    }

    def __new__(
        cls,
        url: str,
        *,
        protocol: str = "grpc",
        **kwargs: Any,
    ) -> TritonClientProtocol:
        protocol = protocol.lower()
        if protocol not in cls._CLIENTS:
            raise ProtocolError(
                f"Unsupported protocol '{protocol}'. "
                f"Choose from {list(cls._CLIENTS)}."
            )
        return cls._CLIENTS[protocol](url, **kwargs)


__all__ = [
    "TritonClient",
    "TritonClientProtocol",
    "TritonHttpClient",
    "TritonGrpcClient",
    "TritonSharedMemoryClient",
    "SharedMemoryRegion",
    "numpy_to_triton_dtype",
    "TritonClientError",
    "ProtocolError",
    "SharedMemoryError",
]
