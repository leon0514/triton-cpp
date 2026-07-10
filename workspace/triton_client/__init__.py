"""Triton inference client wrapper supporting gRPC, HTTP and shared memory."""

from .client import TritonClient, TritonClientError

__all__ = ["TritonClient", "TritonClientError"]
