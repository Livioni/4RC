"""Bounded-memory DROID metadata indexing. No image decoding during indexing."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

import numpy as np

LOGGER = logging.getLogger(__name__)
FPS = 15.0
INDEX_VERSION = 1
DEFAULT_INDEX = "droid_script/cache/droid.sqlite"
DEFAULT_TRAIN = "droid_script/splits/train_set.txt"
DEFAULT_VAL = "droid_script/splits/val_set.txt"


@contextmanager
def index_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ranges_from_mask(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def rotation_matrix(rpy):
    r, p, y = np.moveaxis(rpy, -1, 0)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                     sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                     -sp, cp*sr, cp*cr), axis=-1).reshape(-1, 3, 3)


def camera_record(episode, camera, metadata, depth_metadata, linear_speed, angular_speed):
    camera = str(camera)
    n = int(metadata["frame_count"])
    if n < 2:
        raise ValueError("Fewer than two frames")
    k = np.load(episode / "intrinsic" / f"{camera}.npy", allow_pickle=False)
    ext = np.load(episode / "extrinsic" / f"{camera}.npy", allow_pickle=False)
    state = np.load(episode / "TCP" / camera / "state.npy", mmap_mode="r", allow_pickle=False)
    tcp_meta = read_json(episode / "TCP" / camera / "metadata.json")
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError(f"Invalid intrinsic: {camera}")
    if (ext.shape != (4, 4) or not np.isfinite(ext).all()
            or not np.allclose(ext[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(ext[:3, :3].T @ ext[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(ext[:3, :3]), 1, atol=1e-4)):
        raise ValueError(f"Invalid world-to-camera matrix: {camera}")
    if state.shape != (n, 7) or not np.isfinite(state).all():
        raise ValueError(f"Invalid TCP shape or values: {camera}")
    if np.any((state[:, 6] < -1e-6) | (state[:, 6] > 1 + 1e-6)):
        raise ValueError(f"Gripper must be continuous in [0,1]: {camera}")
    expected = {
        "position_unit": "meter", "rotation_unit": "radian", "camera_id": camera,
        "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
        "coordinate_frame": "OpenCV camera (+x right, +y down, +z forward)",
        "extrinsics": "world_to_camera, applied directly to world TCP pose",
    }
    if any(tcp_meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unsupported TCP coordinate convention: {camera}")
    dmeta = depth_metadata["cameras"][camera]
    if (depth_metadata.get("units") != "millimeters"
            or (dmeta["height"], dmeta["width"], dmeta["frames"]) != (180, 320, n)):
        raise ValueError(f"Expected aligned 320x180 millimeter depth: {camera}")
    for kind in ("images", "depths"):
        for frame in (0, n - 1):
            path = episode / kind / camera / f"{frame:06d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
    bad = np.zeros(n - 1, dtype=bool)
    if linear_speed is not None:
        bad |= np.linalg.norm(np.diff(state[:, :3].astype(np.float64), axis=0), axis=-1) * FPS > linear_speed
    if angular_speed is not None:
        rotations = rotation_matrix(state[:, 3:6].astype(np.float64))
        relative = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
        angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
        bad |= angle * FPS > angular_speed
    cuts = [0, *(np.flatnonzero(bad) + 1).tolist(), n]
    segments = list(zip(cuts[:-1], cuts[1:]))
    pixels = state[:, :3] @ k.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels[:, :2] / pixels[:, 2:3]
    visible = (np.isfinite(uv).all(-1) & (state[:, 2] > 1e-6)
               & (uv[:, 0] >= 0) & (uv[:, 0] < 320) & (uv[:, 1] >= 0) & (uv[:, 1] < 180))
    texts = []
    for key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        values = metadata.get(key, [])
        if isinstance(values, str):
            values = [values]
        for text in values:
            if isinstance(text, str) and text.strip() and text.strip() not in texts:
                texts.append(text.strip())
    xyz = state[:, :3].astype(np.float64)
    return {"intrinsics": k.tolist(), "extrinsics": ext.tolist(), "segments": segments,
            "visible": ranges_from_mask(visible), "instructions": texts,
            "mean": xyz.mean(0).tolist(), "second": np.square(xyz).mean(0).tolist(),
            "invalid_transitions": int(bad.sum())}


def index_metadata(path):
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as connection:
        return json.loads(connection.execute("SELECT value FROM meta WHERE key='index'").fetchone()[0])


def ensure_index(root, path=DEFAULT_INDEX, *, rebuild=False, max_episodes=None,
                 max_tcp_linear_speed=3.0, max_tcp_angular_speed=4 * math.pi):
    root, path = Path(root).expanduser().resolve(), Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    for threshold in (max_tcp_linear_speed, max_tcp_angular_speed):
        if threshold is not None and (not math.isfinite(threshold) or threshold <= 0):
            raise ValueError("TCP speed thresholds must be positive or None")
    signature = {"version": INDEX_VERSION, "root": str(root), "fps": FPS,
                 "max_tcp_linear_speed": max_tcp_linear_speed,
                 "max_tcp_angular_speed": max_tcp_angular_speed,
                 "max_episodes": max_episodes}
    with index_lock(path):
        if path.is_file() and not rebuild:
            meta = index_metadata(path)
            if meta["signature"] != signature:
                raise ValueError(f"Index settings changed: rebuild {path} or use a separate index")
            return path
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        if temporary.exists():
            temporary.unlink()
        connection = sqlite3.connect(temporary)
        started = time.monotonic()
        try:
            connection.executescript("""
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE cameras(id INTEGER PRIMARY KEY, episode TEXT NOT NULL,
                    camera TEXT NOT NULL, frames INTEGER NOT NULL, data TEXT NOT NULL,
                    UNIQUE(episode,camera));
                CREATE TABLE excluded(episode TEXT PRIMARY KEY, reason TEXT NOT NULL);
                CREATE TABLE profiles(key TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE profile_samples(profile TEXT, camera_id INTEGER, starts TEXT,
                    PRIMARY KEY(profile,camera_id));
            """)
            with os.scandir(root) as entries:
                names = sorted(entry.name for entry in entries if entry.is_dir())
            if max_episodes is not None:
                names = names[:max_episodes]
            accepted = rejected = 0
            for number, name in enumerate(names, 1):
                episode = root / name
                try:
                    metadata = read_json(episode / "metadata.json")
                    depth_metadata = read_json(episode / "depths" / "metadata.json")
                    cameras = sorted(depth_metadata["cameras"])
                    if len(cameras) != 2:
                        raise ValueError(f"Expected two external cameras, got {cameras}")
                    records = [camera_record(episode, cam, metadata, depth_metadata,
                                             max_tcp_linear_speed, max_tcp_angular_speed) for cam in cameras]
                    for camera, record in zip(cameras, records):
                        connection.execute("INSERT INTO cameras(episode,camera,frames,data) VALUES(?,?,?,?)",
                                           (name, camera, int(metadata["frame_count"]), json.dumps(record)))
                    accepted += 1
                except (OSError, ValueError, KeyError, TypeError) as error:
                    connection.execute("INSERT INTO excluded VALUES(?,?)", (name, str(error)))
                    rejected += 1
                if number % 500 == 0 or number == len(names):
                    connection.commit()
                    LOGGER.info("Index %d/%d: %d accepted, %d excluded, %.1fs", number, len(names),
                                accepted, rejected, time.monotonic() - started)
            if not accepted:
                raise ValueError("No valid DROID episodes found")
            meta = {"signature": signature, "uuid": uuid.uuid4().hex,
                    "episodes": accepted, "excluded": rejected}
            connection.execute("INSERT INTO meta VALUES('index',?)", (json.dumps(meta),))
            connection.commit()
            connection.close()
            temporary.replace(path)
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
    return path


def load_splits(root, train_file=DEFAULT_TRAIN, val_file=DEFAULT_VAL):
    result = {}
    for split, path in (("train", train_file), ("validation", val_file)):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run python -m droid_script.prepare_droid_dataset")
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate episode in {path}")
        for name in names:
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Expected one relative episode directory name, got {name!r}")
            if not (Path(root) / name).is_dir():
                raise FileNotFoundError(f"Split episode does not exist: {name}")
        result[split] = names
    overlap = set(result["train"]) & set(result["validation"])
    if overlap:
        raise ValueError(f"Train/validation overlap: {sorted(overlap)[:5]}")
    return result


def write_splits(root, index_path, train_file=DEFAULT_TRAIN, val_file=DEFAULT_VAL,
                 *, seed=42, validation_fraction=0.1, overwrite=False):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    destinations = [Path(train_file), Path(val_file)]
    if destinations[0].resolve() == destinations[1].resolve():
        raise ValueError("Train and validation files must differ")
    if any(path.exists() for path in destinations) and not overwrite:
        if not all(path.exists() for path in destinations):
            raise ValueError("Only one split file exists; provide both or explicitly overwrite")
        return load_splits(root, train_file, val_file)
    with sqlite3.connect(f"file:{Path(index_path).resolve()}?mode=ro", uri=True) as connection:
        names = [row[0] for row in connection.execute("SELECT DISTINCT episode FROM cameras ORDER BY episode")]
    result = {"train": [], "validation": []}
    for name in names:
        fraction = int(hashlib.sha256(f"{seed}/{name}".encode()).hexdigest()[:16], 16) / 2**64
        result["validation" if fraction < validation_fraction else "train"].append(name)
    for split, path in zip(("train", "validation"), destinations):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text("".join(name + "\n" for name in result[split]), encoding="utf-8")
        temporary.replace(path)
    return result
