#!/usr/bin/env python3
"""Minimal Triton client wrapper with gRPC / HTTP / shared-memory support.

Examples
--------
# gRPC
client = TritonClient(url="localhost:48001", protocol="grpc")
result = client.infer(
    model_name="yolo11_ensemble",
    inputs={"raw_image": img_np},
    outputs=["num_dets", "detection_boxes"],
)

# HTTP
client = TritonClient(url="localhost:48000", protocol="http")
result = client.infer(
    model_name="yolo11_ensemble",
    inputs={"raw_image": img_np},
    outputs=["num_dets", "detection_boxes"],
)

# System shared memory (over HTTP)
client = TritonClient(url="localhost:48000", protocol="shm")
result = client.infer(
    model_name="yolo11_ensemble",
    inputs={"raw_image": img_np},
    outputs=["num_dets", "detection_boxes"],
    output_specs={
        "num_dets": ([1, 1], "int32"),
        "detection_boxes": ([1, 300, 4], "float32"),
    },
)
"""

from __future__ import annotations

import os
import uuid
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

# Optional Triton client imports are performed lazily so that the module can be
# imported in environments where only one of grpc/http is available.


def _triton_type(np_dtype: Union[str, np.dtype]) -> str:
    """Convert a numpy dtype to a Triton datatype string."""
    mapping = {
        np.bool_: "BOOL",
        np.uint8: "UINT8",
        np.uint16: "UINT16",
        np.uint32: "UINT32",
        np.uint64: "UINT64",
        np.int8: "INT8",
        np.int16: "INT16",
        np.int32: "INT32",
        np.int64: "INT64",
        np.float16: "FP16",
        np.float32: "FP32",
        np.float64: "FP64",
        np.object_: "BYTES",
    }
    try:
        dt = np.dtype(np_dtype)
    except TypeError as exc:
        raise TritonClientError(f"Invalid numpy dtype: {np_dtype}") from exc
    # np.dtype(np.str_) is not directly in mapping; treat generic object/str as BYTES.
    try:
        return mapping[dt.type]
    except KeyError:
        if dt.kind in ("U", "S", "O"):
            return "BYTES"
        raise TritonClientError(f"Unsupported numpy dtype: {dt}")


class TritonClientError(Exception):
    """Raised when a Triton client operation fails."""


class TritonClient:
    """Unified Triton inference client.

    Parameters
    ----------
    url : str
        Server URL. For gRPC use ``host:port`` (e.g. ``localhost:48001``).
        For HTTP/SHM use ``host:port`` (e.g. ``localhost:48000``).
    protocol : {"grpc", "http", "shm"}
        Communication protocol. ``shm`` uses HTTP underneath and transfers
        input/output tensors through system shared memory.
    verbose : bool, optional
        Enable verbose logging in the underlying Triton client.
    """

    _VALID_PROTOCOLS = {"grpc", "http", "shm"}

    def __init__(
        self,
        url: str,
        protocol: str = "http",
        verbose: bool = False,
    ):
        protocol = protocol.lower()
        if protocol not in self._VALID_PROTOCOLS:
            raise TritonClientError(
                f"Invalid protocol '{protocol}'. "
                f"Supported: {', '.join(sorted(self._VALID_PROTOCOLS))}"
            )

        self.url = url
        self.protocol = protocol
        self.verbose = verbose
        self._client: Any = None
        self._closed = False

        # Shared-memory state (only used when protocol == "shm").
        self._input_shm: Dict[str, Any] = {}  # tensor name -> shm handle
        self._input_shm_names: Dict[str, str] = {}  # tensor name -> shm name
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._output_shm: Dict[str, Any] = {}  # tensor name -> shm handle
        self._output_shm_names: Dict[str, str] = {}  # tensor name -> shm name
        self._output_specs: Dict[str, Tuple[Sequence[int], Union[str, np.dtype]]] = {}

        self._load_client()

    # ------------------------------------------------------------------
    # Construction / cleanup
    # ------------------------------------------------------------------
    def _load_client(self) -> None:
        if self.protocol == "grpc":
            import tritonclient.grpc as grpcclient

            self._client = grpcclient.InferenceServerClient(
                url=self.url, verbose=self.verbose
            )
        else:
            import tritonclient.http as httpclient

            self._client = httpclient.InferenceServerClient(
                url=self.url, verbose=self.verbose
            )

    def close(self) -> None:
        """Release shared memory and close the client."""
        if self._closed:
            return
        self._cleanup_shared_memory()
        self._client = None
        self._closed = True

    def __enter__(self) -> "TritonClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Server / model status
    # ------------------------------------------------------------------
    def is_server_ready(self) -> bool:
        """Return True if the server is ready."""
        return self._client.is_server_ready()

    def is_server_live(self) -> bool:
        """Return True if the server is live."""
        return self._client.is_server_live()

    def is_model_ready(self, model_name: str) -> bool:
        """Return True if the model is ready."""
        return self._client.is_model_ready(model_name)

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """Return simplified model metadata.

        Returns
        -------
        dict
            ``{"name": str, "inputs": [...], "outputs": [...]}`` where each
            tensor is ``{"name": str, "datatype": str, "shape": List[int]}``.
        """
        raw = self._client.get_model_metadata(model_name)
        if self.protocol == "grpc":
            return {
                "name": raw.name,
                "inputs": [
                    {
                        "name": t.name,
                        "datatype": t.datatype,
                        "shape": list(t.shape),
                    }
                    for t in raw.inputs
                ],
                "outputs": [
                    {
                        "name": t.name,
                        "datatype": t.datatype,
                        "shape": list(t.shape),
                    }
                    for t in raw.outputs
                ],
            }
        # HTTP / SHM both use the HTTP client.
        return {
            "name": raw.get("name", model_name),
            "inputs": [
                {
                    "name": t["name"],
                    "datatype": t["datatype"],
                    "shape": list(t["shape"]),
                }
                for t in raw.get("inputs", [])
            ],
            "outputs": [
                {
                    "name": t["name"],
                    "datatype": t["datatype"],
                    "shape": list(t["shape"]),
                }
                for t in raw.get("outputs", [])
            ],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Sequence[str],
        output_specs: Optional[
            Dict[str, Tuple[Sequence[int], Union[str, np.dtype]]]
        ] = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference and return output tensors as a name -> ndarray mapping.

        Parameters
        ----------
        model_name : str
            Name of the model/ensemble to invoke.
        inputs : dict
            Mapping from input tensor name to numpy array.
        outputs : sequence of str
            Names of the output tensors to fetch.
        output_specs : dict, optional
            Required for ``protocol="shm"``. Maps output name to
            ``(shape, dtype)``. Example: ``{"num_dets": ([1, 1], "int32")}``.

        Returns
        -------
        dict
            Mapping from output tensor name to numpy array.
        """
        if self._closed:
            raise TritonClientError("Client has been closed.")

        if not inputs:
            raise TritonClientError("At least one input tensor is required.")

        if self.protocol == "shm":
            return self._infer_shm(model_name, inputs, outputs, output_specs)
        if self.protocol == "grpc":
            return self._infer_grpc(model_name, inputs, outputs)
        return self._infer_http(model_name, inputs, outputs)

    # ------------------------------------------------------------------
    # gRPC / HTTP helpers
    # ------------------------------------------------------------------
    def _infer_grpc(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        import tritonclient.grpc as grpcclient

        triton_inputs = [
            self._make_infer_input(grpcclient.InferInput, name, arr)
            for name, arr in inputs.items()
        ]
        triton_outputs = [
            grpcclient.InferRequestedOutput(name) for name in outputs
        ]

        response = self._client.infer(
            model_name, triton_inputs, outputs=triton_outputs
        )
        return {name: response.as_numpy(name) for name in outputs}

    def _infer_http(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        import tritonclient.http as httpclient

        triton_inputs = [
            self._make_infer_input(httpclient.InferInput, name, arr)
            for name, arr in inputs.items()
        ]
        triton_outputs = [
            httpclient.InferRequestedOutput(name) for name in outputs
        ]

        response = self._client.infer(
            model_name, triton_inputs, outputs=triton_outputs
        )
        return {name: response.as_numpy(name) for name in outputs}

    @staticmethod
    def _make_infer_input(
        infer_input_cls, name: str, arr: np.ndarray
    ) -> Any:
        """Create a Triton InferInput from a numpy array."""
        inp = infer_input_cls(name, list(arr.shape), _triton_type(arr.dtype))
        inp.set_data_from_numpy(arr)
        return inp

    # ------------------------------------------------------------------
    # Shared memory helpers
    # ------------------------------------------------------------------
    def _cleanup_input_shm(self) -> None:
        """Unregister and destroy all input shared memory regions."""
        if self._client is None:
            return
        try:
            from tritonclient.utils.shared_memory import (
                destroy_shared_memory_region,
            )

            for name, handle in list(self._input_shm.items()):
                shm_name = self._input_shm_names.get(name, name)
                try:
                    self._client.unregister_system_shared_memory(shm_name)
                except Exception as err:
                    warnings.warn(
                        f"Failed to unregister input SHM {shm_name}: {err}"
                    )
                try:
                    destroy_shared_memory_region(handle)
                except Exception as err:
                    warnings.warn(
                        f"Failed to destroy input SHM {shm_name}: {err}"
                    )
        except ImportError:
            pass
        finally:
            self._input_shm.clear()
            self._input_shm_names.clear()
            self._input_shapes.clear()

    def _cleanup_output_shm(self) -> None:
        """Unregister and destroy all output shared memory regions."""
        if self._client is None:
            return
        try:
            from tritonclient.utils.shared_memory import (
                destroy_shared_memory_region,
            )

            for name, handle in list(self._output_shm.items()):
                shm_name = self._output_shm_names.get(name, name)
                try:
                    self._client.unregister_system_shared_memory(shm_name)
                except Exception as err:
                    warnings.warn(
                        f"Failed to unregister output SHM {shm_name}: {err}"
                    )
                try:
                    destroy_shared_memory_region(handle)
                except Exception as err:
                    warnings.warn(
                        f"Failed to destroy output SHM {shm_name}: {err}"
                    )
        except ImportError:
            pass
        finally:
            self._output_shm.clear()
            self._output_shm_names.clear()
            self._output_specs.clear()

    def _cleanup_shared_memory(self) -> None:
        """Unregister and destroy all shared memory regions."""
        self._cleanup_input_shm()
        self._cleanup_output_shm()

    def _ensure_input_shm(
        self, name: str, arr: np.ndarray
    ) -> Tuple[str, int]:
        """Create or recreate a single input shared memory region if needed."""
        from tritonclient.utils.shared_memory import (
            create_shared_memory_region,
            set_shared_memory_region,
        )

        shape = tuple(arr.shape)
        byte_size = int(arr.nbytes)

        # Reuse existing region if shape and byte size match.
        if name in self._input_shm and self._input_shapes.get(name) == shape:
            try:
                set_shared_memory_region(self._input_shm[name], [arr])
                return self._input_shm_names[name], byte_size
            except Exception:
                # Region exists but is too small or stale; recreate just this one.
                pass
        else:
            # Only clean up this input's region, leave others intact.
            old_handle = self._input_shm.pop(name, None)
            old_name = self._input_shm_names.pop(name, None)
            self._input_shapes.pop(name, None)
            if old_handle is not None and self._client is not None:
                try:
                    from tritonclient.utils.shared_memory import (
                        destroy_shared_memory_region,
                    )

                    if old_name is not None:
                        try:
                            self._client.unregister_system_shared_memory(old_name)
                        except Exception as err:
                            warnings.warn(
                                f"Failed to unregister input SHM {old_name}: {err}"
                            )
                    destroy_shared_memory_region(old_handle)
                except Exception as err:
                    warnings.warn(f"Failed to destroy input SHM {old_name}: {err}")

        unique = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        shm_name = f"triton_cli_input_{name}_{unique}"
        handle = create_shared_memory_region(shm_name, shm_name, byte_size)
        set_shared_memory_region(handle, [arr])
        self._client.register_system_shared_memory(shm_name, shm_name, byte_size)

        self._input_shm[name] = handle
        self._input_shm_names[name] = shm_name
        self._input_shapes[name] = shape
        return shm_name, byte_size

    def _ensure_output_shm(
        self,
        specs: Dict[str, Tuple[Sequence[int], Union[str, np.dtype]]],
    ) -> Dict[str, Tuple[int, np.dtype]]:
        """Create shared memory regions for outputs if they changed."""
        from tritonclient.utils.shared_memory import create_shared_memory_region

        if specs == self._output_specs and self._output_shm:
            return {
                name: (
                    int(np.prod(shape)) * np.dtype(dtype).itemsize,
                    np.dtype(dtype),
                )
                for name, (shape, dtype) in specs.items()
            }

        self._cleanup_output_shm()
        self._output_specs = {k: v for k, v in specs.items()}

        result: Dict[str, Tuple[int, np.dtype]] = {}
        unique = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        for name, (shape, dtype) in specs.items():
            dt = np.dtype(dtype)
            byte_size = int(np.prod(shape)) * dt.itemsize
            shm_name = f"triton_cli_output_{name}_{unique}"
            handle = create_shared_memory_region(shm_name, shm_name, byte_size)
            self._client.register_system_shared_memory(
                shm_name, shm_name, byte_size
            )
            self._output_shm[name] = handle
            self._output_shm_names[name] = shm_name
            result[name] = (byte_size, dt)
        return result

    def _infer_shm(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: Sequence[str],
        output_specs: Optional[
            Dict[str, Tuple[Sequence[int], Union[str, np.dtype]]]
        ],
    ) -> Dict[str, np.ndarray]:
        import tritonclient.http as httpclient
        from tritonclient.utils.shared_memory import get_contents_as_numpy

        if output_specs is None:
            raise TritonClientError(
                "output_specs is required for shared-memory inference. "
                "Example: {'num_dets': ([1, 1], 'int32')}"
            )

        missing = set(outputs) - set(output_specs.keys())
        if missing:
            raise TritonClientError(
                f"Missing output_specs for: {', '.join(sorted(missing))}"
            )

        out_infos = self._ensure_output_shm(output_specs)

        triton_inputs = []
        for input_name, input_arr in inputs.items():
            input_shm_name, input_byte_size = self._ensure_input_shm(
                input_name, input_arr
            )
            inp = httpclient.InferInput(
                input_name,
                list(input_arr.shape),
                _triton_type(input_arr.dtype),
            )
            inp.set_shared_memory(input_shm_name, input_byte_size)
            triton_inputs.append(inp)

        triton_outputs = []
        for name in outputs:
            byte_size, _ = out_infos[name]
            out = httpclient.InferRequestedOutput(name)
            shm_name = self._output_shm_names[name]
            out.set_shared_memory(shm_name, byte_size)
            triton_outputs.append(out)

        self._client.infer(model_name, triton_inputs, outputs=triton_outputs)

        result: Dict[str, np.ndarray] = {}
        for name in outputs:
            handle = self._output_shm[name]
            shape, dtype = output_specs[name]
            arr = get_contents_as_numpy(handle, np.dtype(dtype), shape)
            # Return a copy so the caller can safely use the result after the
            # client (and its shared memory regions) is closed.
            result[name] = arr.copy()
        return result
