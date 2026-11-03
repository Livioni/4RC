import msgpack
import numpy as np
import pytest

from deploy.protocol import MAX_MESSAGE_BYTES, ProtocolError, pack_message, unpack_message


@pytest.mark.parametrize("array", [
    np.arange(60, dtype=np.uint8).reshape(4, 5, 3)[:, ::2],
    np.array([[1, 2]], dtype=">f4"),
    np.array([True, False]),
    np.array(2.5, dtype=np.float32),
    np.empty((0, 2), dtype=np.int64),
])
def test_array_roundtrip_preserves_shape_dtype_and_values(array):
    result = unpack_message(pack_message({"array": array, "nested": {"index": np.int64(3)}}))
    assert result["array"].dtype == array.dtype
    assert result["array"].shape == array.shape
    assert result["array"].flags.writeable
    np.testing.assert_array_equal(result["array"], array)
    assert result["nested"] == {"index": 3}


@pytest.mark.parametrize("array", [np.array([object()]), np.array(["text"]), np.array([1j])])
def test_non_numeric_arrays_are_rejected(array):
    with pytest.raises(ProtocolError):
        pack_message({"array": array})


@pytest.mark.parametrize("record", [
    {"__ndarray__": True, "dtype": "O", "shape": [1], "data": b"12345678"},
    {"__ndarray__": True, "dtype": "<f4", "shape": [10], "data": b"1234"},
    {"__ndarray__": True, "dtype": "u1", "shape": [-1], "data": b""},
    {"__ndarray__": True, "dtype": "u1", "shape": [True], "data": b"x"},
    {"__ndarray__": True, "dtype": "u1", "shape": [1] * 9, "data": b"x"},
    {"__ndarray__": True, "dtype": "u1", "shape": [1], "data": "x"},
    {"__ndarray__": False},
])
def test_forged_array_records_are_rejected(record):
    payload = msgpack.packb({"array": record}, use_bin_type=True)
    with pytest.raises(ProtocolError):
        unpack_message(payload)


@pytest.mark.parametrize("payload", ["text", b"\xc1", msgpack.packb([1, 2]), msgpack.packb({1: 2})])
def test_invalid_messages_are_rejected(payload):
    with pytest.raises(ProtocolError):
        unpack_message(payload)


def test_message_size_limit():
    with pytest.raises(ProtocolError, match="16 MiB"):
        unpack_message(b"x" * (MAX_MESSAGE_BYTES + 1))
    with pytest.raises(ProtocolError, match="16 MiB"):
        pack_message({"data": b"x" * MAX_MESSAGE_BYTES})
