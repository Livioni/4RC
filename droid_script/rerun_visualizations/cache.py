"""Numeric, versioned replay caches shared by inference and the Rerun viewer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np


SCHEMA_VERSION = 1


def camera_matrices(value: np.ndarray, count: int) -> np.ndarray:
    """The Stage1 decoder returns [N,3,4]; caches expose homogeneous [N,4,4]."""
    value = np.asarray(value)
    if value.shape == (count, 3, 4):
        matrices = np.tile(np.eye(4, dtype=value.dtype), (count, 1, 1))
        matrices[:, :3] = value
        value = matrices
    if (value.shape != (count, 4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError("Expected finite camera-to-base transforms [N,3,4] or [N,4,4]")
    return value


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_npz(path: Path, **arrays) -> None:
    """Replace only complete files, without object arrays or pickle."""
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Replay caches cannot contain object arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_metadata(directory: Path, metadata: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, suffix=".json",
                                     encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(directory / "metadata.json")


def save_prediction(directory: Path, prediction, frame_indices, metadata: dict) -> dict:
    """Publish predictions before IK so an interrupted solve can reuse inference."""
    directory.mkdir(parents=True, exist_ok=True)
    if prediction.clouds is None or prediction.camera_to_base is None:
        raise ValueError("Caching requires point clouds and predicted camera poses")
    lengths = [len(cloud["points"]) for cloud in prediction.clouds]
    arrays = {f"tcp_{name}": value for name, value in prediction.tcp.items()}
    arrays.update(frame_indices=np.asarray(frame_indices, dtype=np.int64),
                  camera_to_base=camera_matrices(prediction.camera_to_base, len(frame_indices)),
                  cloud_offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64))
    for key in ("points", "colors", "confidence"):
        arrays[f"cloud_{key}"] = np.concatenate([cloud[key] for cloud in prediction.clouds])
    write_npz(directory / "prediction.npz", **arrays)
    metadata = dict(metadata, schema_version=SCHEMA_VERSION,
                    windows=prediction.windows, source_windows=prediction.source_windows,
                    files={"prediction.npz": file_digest(directory / "prediction.npz")})
    metadata.pop("ik", None)
    write_metadata(directory, metadata)
    return metadata


def save_robot_states(directory: Path, states: dict, metadata: dict, ik_metadata: dict) -> dict:
    write_npz(directory / "robot_states.npz", **states)
    metadata = dict(metadata, ik=ik_metadata,
                    files={**metadata["files"],
                           "robot_states.npz": file_digest(directory / "robot_states.npz")})
    write_metadata(directory, metadata)
    return metadata


def _array(data, name: str, shape: tuple, *, finite: bool = True) -> np.ndarray:
    value = data[name]
    if (value.shape != shape or value.dtype.kind not in "fibu"
            or (finite and not np.isfinite(value).all())):
        raise ValueError(f"Invalid cached {name}: expected finite numeric {shape}, got {value.shape}")
    return value


def load_cache(directory: Path, *, require_robot: bool = True):
    directory = directory.expanduser().resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    rate = metadata.get("frame_rate_hz")
    if (metadata.get("schema_version") != SCHEMA_VERSION or isinstance(rate, bool)
            or not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0):
        raise ValueError("Unsupported replay cache version or frame rate")
    filenames = ["prediction.npz"] + (["robot_states.npz"] if require_robot else [])
    for filename in filenames:
        expected = metadata.get("files", {}).get(filename)
        if not expected:
            raise ValueError(f"Cache has no complete {filename}; generate/recompute IK first")
        if file_digest(directory / filename) != expected:
            raise ValueError(f"Cache checksum mismatch: {filename}; regenerate this cache")
    with np.load(directory / "prediction.npz", allow_pickle=False) as data:
        indices = data["frame_indices"]
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 2
                or indices[0] < 0 or np.any(np.diff(indices) != 1)):
            raise ValueError("Cache frame indices must be contiguous original episode indices")
        count = len(indices)
        if count != metadata["num_frames"]:
            raise ValueError("Cache frame count disagrees with metadata")
        if "timestamps_seconds" in metadata:
            times = np.asarray(metadata["timestamps_seconds"], dtype=np.float64)
            if (times.shape != (count,) or not np.isfinite(times).all()
                    or times[0] < 0 or np.any(np.diff(times) <= 0)):
                raise ValueError("Invalid cached timestamps_seconds")
        tcp = {name: _array(data, f"tcp_{name}", shape) for name, shape in (
            ("position", (count, 1, 3)), ("rotation", (count, 1, 3, 3)),
            ("gripper", (count, 1)), ("confidence", (count, 1)))}
        if np.any((tcp["gripper"] < 0) | (tcp["gripper"] > 1)):
            raise ValueError("Cached gripper_open must be in [0,1]")
        cameras = camera_matrices(data["camera_to_base"], count)
        offsets = _array(data, "cloud_offsets", (count + 1,))
        if offsets.dtype.kind not in "iu" or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("Invalid cached cloud offsets")
        total = int(offsets[-1])
        cloud_arrays = {key: _array(data, f"cloud_{key}", shape) for key, shape in (
            ("points", (total, 3)), ("colors", (total, 3)), ("confidence", (total,)))}
        clouds = [{key: value[offsets[i]:offsets[i + 1]] for key, value in cloud_arrays.items()}
                  for i in range(count)]
    prediction = SimpleNamespace(tcp=tcp, camera_to_base=cameras, clouds=clouds,
                                 windows=metadata["windows"], source_windows=metadata["source_windows"])
    states = None
    if require_robot:
        with np.load(directory / "robot_states.npz", allow_pickle=False) as data:
            states = {name: _array(data, name, shape) for name, shape in (
                ("frame_indices", (count,)), ("joints", (count, 7)),
                ("gripper_open", (count,)), ("fk_tcp_poses", (count, 4, 4)),
                ("success", (count,)), ("position_error_m", (count,)),
                ("rotation_error_rad", (count,)), ("solve_seconds", (count,)))}
        if states["success"].dtype != np.bool_ or not np.array_equal(states["frame_indices"], indices):
            raise ValueError("Robot state timeline or success flags disagree with prediction cache")
        if not np.allclose(states["gripper_open"], tcp["gripper"][:, 0]):
            raise ValueError("Robot gripper states disagree with prediction cache")
    return metadata, prediction, indices, states


def validate_episode_timing(metadata, episode, indices) -> None:
    """Refuse to silently replay a cache against a different source timeline."""
    if not np.isclose(metadata["frame_rate_hz"], episode.frame_rate):
        raise ValueError("Cached frame rate disagrees with the source episode")
    if "timestamps_seconds" in metadata:
        expected = [episode.time_seconds(int(i)) for i in indices]
        if not np.allclose(metadata["timestamps_seconds"], expected, rtol=0, atol=1e-9):
            raise ValueError("Cached timestamps disagree with the source episode")
