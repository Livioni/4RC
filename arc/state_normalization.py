"""Real RoboTwin joint/gripper state IO and train-anchor-only normalization."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

STATE_NORM_FILENAME = "state_norm.json"
STATE_LAYOUT = [f"{arm}.{name}" for arm in ("left", "right")
                for name in (*[f"joint_{i}" for i in range(6)], "gripper")]
JOINT_INDICES = [i for i in range(14) if i not in (6, 13)]


def read_robot_states(episode_path, num_frames=None):
    path = Path(episode_path) / "robot_state.npy"
    if not path.is_file():
        raise ValueError(f"V2 requires real current robot state: missing {path}")
    state = np.load(path, mmap_mode="r", allow_pickle=False)
    if state.ndim != 2 or state.shape[1] != 14 or (num_frames is not None and len(state) != num_frames):
        raise ValueError(f"Expected frame-aligned robot_state.npy [T,14]: {path}, got {state.shape}")
    return state


def validate_robot_states(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 14 or not np.isfinite(values).all():
        raise ValueError("Current real robot states must be finite [N,14]")
    if np.any((values[:, [6, 13]] < 0) | (values[:, [6, 13]] > 1)):
        raise ValueError("Current real robot state grippers must be in [0,1]")


def state_parameters(q01, q99, minimum, maximum):
    q01, q99, minimum, maximum = [np.asarray(x, dtype=np.float64) for x in (q01, q99, minimum, maximum)]
    if any(x.shape != (1, 12) or not np.isfinite(x).all() for x in (q01, q99, minimum, maximum)):
        raise ValueError("State joint statistics must be finite [1,12]")
    if np.any((minimum > q01) | (q01 > q99) | (q99 > maximum)):
        raise ValueError("Invalid state quantile bounds")
    fallback = q99 - q01 < 1e-4
    low, high = np.where(fallback, minimum, q01), np.where(fallback, maximum, q99)
    offset, scale = np.full((1, 14), 0.5), np.full((1, 14), 0.5)
    offset[:, JOINT_INDICES] = (low + high) / 2
    scale[:, JOINT_INDICES] = np.maximum(high - low, 1e-3) / 2
    return offset, scale


def validate_state_norm(record):
    expected = {"version": 1, "layout": STATE_LAYOUT, "statistics_scope": "train_valid_current_frames",
                "joint_indices": JOINT_INDICES, "minimum_span": 1e-3, "fallback_span": 1e-4}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Invalid state normalization {key}")
    if not isinstance(record.get("count"), int) or record["count"] < 1:
        raise ValueError("State statistics require valid training frames")
    offset, scale = state_parameters(*(record[key] for key in ("q01", "q99", "min", "max")))
    for name, expected in (("offset", offset), ("scale", scale)):
        actual = np.asarray(record[name], dtype=np.float32)
        if actual.shape != (1, 14) or not np.isfinite(actual).all() or not np.array_equal(actual, expected.astype(np.float32)):
            raise ValueError(f"State normalization {name} disagrees with quantile/gripper rules")
    return record


def compute_state_norm(mixture, scratch_dir):
    """Exact quantiles per joint, bounded RAM; each eligible current frame once."""
    count, sources = 0, []
    minimum, maximum = np.full(12, np.inf), np.full(12, -np.inf)
    with TemporaryDirectory(prefix=".state-norm-", dir=scratch_dir) as tmp:
        paths = [Path(tmp) / f"joint-{i}.bin" for i in range(12)]
        with ExitStack() as stack:
            handles = [stack.enter_context(path.open("wb")) for path in paths]
            for source in mixture.sources:
                if source.weight <= 0:
                    continue
                if source.dataset.split != "train":
                    raise ValueError("State normalization may only use the training split")
                source_count = 0
                for values in source.dataset.iter_current_robot_states():
                    validate_robot_states(values)
                    joints = np.asarray(values, dtype=np.float32)[:, JOINT_INDICES]
                    if not len(joints):
                        continue
                    minimum, maximum = np.minimum(minimum, joints.min(0)), np.maximum(maximum, joints.max(0))
                    count += len(joints)
                    source_count += len(joints)
                    for column, handle in enumerate(handles):
                        np.ascontiguousarray(joints[:, column]).tofile(handle)
                sources.append({"name": source.name, "count": source_count, "selection": source.dataset.manifest()})
        if not count:
            raise ValueError("No valid training current frames for state normalization")
        ranks = np.array([0.01, 0.99]) * (count - 1)
        lower, upper = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
        quantiles = np.empty((2, 12))
        for column, path in enumerate(paths):
            values = np.memmap(path, mode="r+", dtype=np.float32, shape=(count,))
            values.partition(np.unique(np.r_[lower, upper]))
            lo, hi = values[lower].astype(np.float64), values[upper].astype(np.float64)
            quantiles[:, column] = lo + (hi - lo) * (ranks - lower)
            del values
    offset, scale = state_parameters(quantiles[:1], quantiles[1:], minimum[None], maximum[None])
    return validate_state_norm({"version": 1, "layout": STATE_LAYOUT, "joint_indices": JOINT_INDICES,
        "statistics_scope": "train_valid_current_frames", "minimum_span": 1e-3, "fallback_span": 1e-4,
        "count": count, "sources": sources, "q01": quantiles[:1].tolist(), "q99": quantiles[1:].tolist(),
        "min": minimum[None].tolist(), "max": maximum[None].tolist(),
        "offset": offset.tolist(), "scale": scale.tolist()})


def save_state_norm(directory, record):
    validate_state_norm(record)
    path = Path(directory) / STATE_NORM_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint_state_norm(checkpoint):
    path = Path(checkpoint).expanduser()
    directory = path if path.is_dir() else path.parent
    record = json.loads((directory / STATE_NORM_FILENAME).read_text())
    return validate_state_norm(record)


def assert_state_norm_matches(policy, record):
    validate_state_norm(record)
    for name in ("offset", "scale"):
        actual = getattr(policy, f"state_{name}").detach().float().cpu()
        if not torch.equal(actual, torch.tensor(record[name], dtype=torch.float32)):
            raise ValueError(f"Checkpoint state_{name} differs from {STATE_NORM_FILENAME}")


def initialize_state_normalization(policy, dataset, config, accelerator):
    from accelerate.utils import broadcast_object_list

    result = [None]
    if accelerator.is_main_process:
        try:
            root = Path(config["output_dir"])
            root.mkdir(parents=True, exist_ok=True)
            record = load_checkpoint_state_norm(config["resume"]) if config.get("resume") else compute_state_norm(dataset, root)
            save_state_norm(root, record)
            result[0] = {"record": record}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    broadcast_object_list(result)
    if "error" in result[0]:
        raise ValueError(result[0]["error"])
    record = result[0]["record"]
    if config.get("resume"):
        assert_state_norm_matches(policy, record)
    else:
        policy.set_state_stats(record["offset"], record["scale"])
    return record
