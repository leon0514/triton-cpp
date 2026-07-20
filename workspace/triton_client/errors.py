"""Exceptions raised by the Triton client wrapper."""


class TritonClientError(Exception):
    """Base exception for all Triton client wrapper errors."""


class ProtocolError(TritonClientError):
    """Raised when a protocol-specific operation fails."""


class SharedMemoryError(TritonClientError):
    """Raised when shared memory allocation or access fails."""
