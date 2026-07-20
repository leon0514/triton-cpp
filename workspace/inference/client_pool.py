"""A fixed-size pool of reusable Triton clients.

When many worker threads (e.g. one per video stream) each create their own
``TritonClient`` with ``protocol="shm"``, the number of shared-memory files
grows with the number of threads. A pool limits the number of underlying
clients -- and therefore the number of SHM regions -- to a fixed size.
"""

import sys
import threading
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

# Allow running this script directly from the package directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_client import TritonClient
from triton_client.protocol import TritonClientProtocol


class TritonClientPool:
    """Fixed-size pool of reusable Triton clients.

    Args:
        url: Triton server URL.
        protocol: ``"http"``, ``"grpc"`` or ``"shm"``.
        size: Number of clients kept in the pool. Each client owns its own
            set of shared-memory regions when ``protocol="shm"``.
        **kwargs: Forwarded to ``TritonClient``.
    """

    def __init__(
        self,
        url: str,
        *,
        protocol: str = "grpc",
        size: int = 4,
        **kwargs: Any,
    ) -> None:
        if size <= 0:
            raise ValueError("pool size must be positive")
        self._url = url
        self._protocol = protocol
        self._kwargs = kwargs
        self._size = size
        self._clients: deque[TritonClientProtocol] = deque()
        self._lock = threading.Lock()
        self._closed = False
        for _ in range(size):
            self._clients.append(self._create_client())

    def _create_client(self) -> TritonClientProtocol:
        return TritonClient(self._url, protocol=self._protocol, **self._kwargs)

    @contextmanager
    def acquire(self) -> Iterator[TritonClientProtocol]:
        """Acquire a client from the pool and return it when done.

        Example:
            >>> with pool.acquire() as client:
            ...     result = client.infer(...)
        """
        client = self._acquire()
        try:
            yield client
        finally:
            self._release(client)

    def _acquire(self) -> TritonClientProtocol:
        with self._lock:
            if self._closed:
                raise RuntimeError("client pool is closed")
            if self._clients:
                return self._clients.popleft()
            # Pool exhausted: create an extra client on demand instead of blocking.
            return self._create_client()

    def _release(self, client: TritonClientProtocol) -> None:
        with self._lock:
            if self._closed:
                client.close()
                return
            self._clients.append(client)

    def close(self) -> None:
        """Close all clients in the pool."""
        with self._lock:
            self._closed = True
            while self._clients:
                self._clients.popleft().close()

    def __enter__(self) -> "TritonClientPool":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
