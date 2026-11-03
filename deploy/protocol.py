"""MessagePack messages with numeric NumPy arrays, shared by server and clients.

This is a versioned 4RC protocol, not the msgpack-numpy or OpenPI wire format.
Only this module, numpy and msgpack are needed on the client.
"""
from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_ARRAY_KEYS = {"__ndarray__", "dtype", "shape", "data"}


class ProtocolError(ValueError):
    """A message cannot be represented by the deployment protocol."""


def _numeric_dtype(value: Any) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.kind not in "biuf" or dtype.itemsize not in (1, 2, 4, 8):
        raise ProtocolError("Only bool, integer and real floating-point arrays are supported")
    return dtype


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        dtype = _numeric_dtype(value.dtype)
        if value.ndim > 8 or value.nbytes > MAX_MESSAGE_BYTES:
            raise ProtocolError("Array exceeds the protocol size or dimension limit")
        return {
            "__ndarray__": True,
            "dtype": dtype.str,
            "shape": list(value.shape),
            "data": value.tobytes(order="C"),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise ProtocolError(f"Unsupported value type: {type(value).__name__}")


def _decode(value: dict) -> Any:
    if "__ndarray__" not in value:
        return value
    if set(value) != _ARRAY_KEYS or value["__ndarray__"] is not True:
        raise ProtocolError("Malformed ndarray record")
    if not isinstance(value["dtype"], str):
        raise ProtocolError("Array dtype must be a string")
    dtype = _numeric_dtype(value["dtype"])
    shape, data = value["shape"], value["data"]
    if (
        not isinstance(shape, list) or len(shape) > 8
        or any(type(size) is not int or size < 0 for size in shape)
        or not isinstance(data, bytes)
    ):
        raise ProtocolError("Invalid array shape or data")
    size = math.prod(shape) * dtype.itemsize
    if size != len(data) or size > MAX_MESSAGE_BYTES:
        raise ProtocolError("Array shape and data length disagree")
    # Copy so torch.from_numpy and clients receive writable storage.
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def pack_message(message: dict) -> bytes:
    """Encode a message dictionary; arrays keep their shape and numeric dtype."""
    if not isinstance(message, dict) or any(not isinstance(key, str) for key in message):
        raise ProtocolError("A message must be a dictionary with string keys")
    try:
        payload = msgpack.packb(message, default=_encode, use_bin_type=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(str(exc)) from exc
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Message exceeds 16 MiB")
    return payload


def unpack_message(payload: bytes) -> dict:
    """Decode one binary message without pickle or Python object deserialization."""
    if not isinstance(payload, bytes):
        raise ProtocolError("Send a binary MessagePack message, not WebSocket text")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError("Message exceeds 16 MiB")
    try:
        message = msgpack.unpackb(
            payload, raw=False, strict_map_key=True, object_hook=_decode,
            max_bin_len=MAX_MESSAGE_BYTES, max_str_len=MAX_MESSAGE_BYTES,
            max_array_len=4096, max_map_len=128, max_ext_len=0,
        )
    except (TypeError, ValueError, OverflowError, msgpack.UnpackException) as exc:
        raise ProtocolError(str(exc)) from exc
    if not isinstance(message, dict) or any(not isinstance(key, str) for key in message):
        raise ProtocolError("A message must be a dictionary with string keys")
    return message
