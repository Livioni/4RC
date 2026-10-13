"""Shared Stage2 quantile statistics, portable artifacts, and initialization."""
from __future__ import annotations

from contextlib import ExitStack
import json
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from arc.action import future_actions_in_current_camera

LOGGER = logging.getLogger(__name__)
NORM_FILENAME = "action_norm.json"
NORM_METHOD = "quantile"
NORM_VERSION = 1
QUANTILE_EPSILON = 1e-6
STAT_NAMES = ("mean", "std", "q01", "q99")
ACTION_COMPONENTS = ["x", "y", "z", "rotation6d_0", "rotation6d_1", "rotation6d_2",
                     "rotation6d_3", "rotation6d_4", "rotation6d_5", "gripper"]


class ActionNormalizationError(ValueError):
    """Invalid action shapes, normalization parameters, or checkpoint metadata."""


def _quantile_parameters(action, q01, q99, epsilon):
    is_tensor = torch.is_tensor(action)
    action = action.float() if is_tensor and action.is_floating_point() else action
    if not is_tensor:
        action = np.asarray(action)
    if action.ndim < 2 or min(action.shape[-2:]) < 1:
        raise ActionNormalizationError(f"Action must have shape [...,H,D], got {action.shape}")
    if is_tensor:
        if not action.is_floating_point():
            raise ActionNormalizationError("Action must use a floating dtype")
        q01 = torch.as_tensor(q01, dtype=torch.float32, device=action.device)
        q99 = torch.as_tensor(q99, dtype=torch.float32, device=action.device)
    else:
        if not np.issubdtype(action.dtype, np.floating):
            raise ActionNormalizationError("Action must use a floating dtype")
        q01, q99 = np.asarray(q01, dtype=np.float32), np.asarray(q99, dtype=np.float32)
    if q01.ndim != 2 or q01.shape != q99.shape or q01.shape[-1] != action.shape[-1]:
        raise ActionNormalizationError("q01/q99 must have matching [1,D] or [H,D] shapes")
    if q01.shape[0] != 1:
        if q01.shape[0] < action.shape[-2]:
            raise ActionNormalizationError("Action horizon exceeds quantile statistics horizon")
        q01, q99 = q01[:action.shape[-2]], q99[:action.shape[-2]]
    # Epsilon is applied exactly once, to q99 before subtraction; no clipping.
    scale = (q99 + float(epsilon)) - q01
    finite = torch.isfinite if is_tensor else np.isfinite
    if not bool(finite(q01).all() & finite(scale).all() & (q99 >= q01).all() & (scale > 0).all()):
        raise ActionNormalizationError("Quantiles must be finite with positive effective q99-q01")
    return action, q01, scale


def quantile_normalize(action, q01, q99, *, epsilon=QUANTILE_EPSILON):
    """Map raw [...,H,D] actions to nominal [-1,1], preserving outliers.

    Shared [1,D] statistics broadcast over any horizon. Torch tensors remain
    on their device and preserve gradients; tensor arithmetic uses float32.
    """
    action, q01, scale = _quantile_parameters(action, q01, q99, epsilon)
    return 2.0 * ((action - q01) / scale) - 1.0


def quantile_unnormalize(action, q01, q99, *, epsilon=QUANTILE_EPSILON):
    """Invert quantile_normalize before decoding positions, rotations or grippers."""
    action, q01, scale = _quantile_parameters(action, q01, q99, epsilon)
    return ((action + 1.0) / 2.0) * scale + q01


def configure_action_normalization(config):
    """New runs have exactly one action normalization format."""
    config.setdefault("action_normalization", NORM_METHOD)
    config.setdefault("action_normalization_version", NORM_VERSION)
    # Fitting the complete dataset on rank zero can exceed the default process
    # group timeout. This also covers index/profile construction before fitting.
    config.setdefault("distributed_timeout_seconds", 86400)
    timeout = config["distributed_timeout_seconds"]
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("distributed_timeout_seconds must be finite and positive")
    require_quantile_config(config)


def require_quantile_config(config):
    if (config.get("action_normalization") != NORM_METHOD
            or config.get("action_normalization_version") != NORM_VERSION):
        raise ActionNormalizationError(
            "Stage2 requires quantile normalization (version 1). "
            "Old mean/std Stage2 checkpoints are unsupported; start a new run from Stage1."
        )


def action_layout(num_arms):
    arms = ["left", "right"] if num_arms == 2 else [f"arm_{i}" for i in range(num_arms)]
    return [f"{arm}.{component}" for arm in arms for component in ACTION_COMPONENTS]


def iter_future_action_rows(states, extrinsics, anchors, segment_ends, horizon, *, chunk_size=8192):
    """Yield every real (anchor, future offset) label as float32 [N,A*10].

    This uses the same coordinate/rotation/gripper conversion as sampled labels.
    Extrinsics may be a static camera matrix or a per-frame camera trajectory.
    No padded rows or image reads are involved.
    """
    anchors = np.asarray(anchors, dtype=np.int64)
    segment_ends = np.asarray(segment_ends, dtype=np.int64)
    if anchors.shape != segment_ends.shape or horizon < 1 or chunk_size < 1:
        raise ValueError("Invalid action statistics window specification")
    static_camera = np.ndim(extrinsics) == 2
    for offset in range(1, horizon + 1):
        selected = anchors[anchors + offset < segment_ends]
        for begin in range(0, len(selected), chunk_size):
            current_indices = selected[begin:begin + chunk_size]
            future_indices = current_indices + offset
            state = torch.tensor(np.asarray(states[future_indices]), dtype=torch.float32)
            if static_camera:
                current = torch.tensor(np.asarray(extrinsics), dtype=torch.float32)
                future = current.expand(len(future_indices), -1, -1)
            else:
                current = torch.tensor(np.asarray(extrinsics[current_indices]), dtype=torch.float32)
                future = torch.tensor(np.asarray(extrinsics[future_indices]), dtype=torch.float32)
            yield future_actions_in_current_camera(state, future, current).flatten(-2).numpy()


def validate_action_norm(record, num_arms):
    expected = {"version": NORM_VERSION, "method": NORM_METHOD, "epsilon": QUANTILE_EPSILON,
                "horizon_shared": True, "num_arms": num_arms, "action_dim": num_arms * 10,
                "layout": action_layout(num_arms), "statistics_scope": "train_valid_window_actions"}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ActionNormalizationError(f"Invalid action normalization {key}: expected {value!r}")
    if not isinstance(record.get("count"), int) or record["count"] < 1:
        raise ActionNormalizationError("Action normalization requires a positive sample count")
    stats = {}
    for name in STAT_NAMES:
        try:
            value = np.asarray(record[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionNormalizationError(f"Invalid action normalization {name}") from exc
        if value.shape != (1, num_arms * 10) or not np.isfinite(value).all():
            raise ActionNormalizationError(f"{name} must contain finite [1,{num_arms * 10}] statistics")
        stats[name] = value
    if (stats["std"] < 0).any() or (stats["q99"] < stats["q01"]).any():
        raise ActionNormalizationError("Invalid standard deviation or reversed quantiles")
    scale = (stats["q99"].astype(np.float32) + np.float32(QUANTILE_EPSILON)) - stats["q01"].astype(np.float32)
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ActionNormalizationError("Float32 effective q99-q01 must be finite and positive")
    return stats


def compute_action_norm(mixture, num_arms, scratch_dir):
    """Exact pooled quantiles with bounded RAM and one temporary file per column.

    Every valid window action has equal weight, regardless of sampler weights.
    In-place partition on contiguous memory-mapped columns avoids a full RAM copy.
    """
    dimension = num_arms * 10
    count = 0
    mean, m2 = np.zeros(dimension), np.zeros(dimension)
    sources = []
    last_progress = time.monotonic()
    with TemporaryDirectory(prefix=".action-norm-", dir=scratch_dir) as temporary:
        paths = [Path(temporary) / f"column-{i}.bin" for i in range(dimension)]
        with ExitStack() as stack:
            handles = [stack.enter_context(path.open("wb")) for path in paths]
            for source in mixture.sources:
                if source.weight <= 0:
                    continue
                dataset = source.dataset
                if dataset.split != "train":
                    raise ActionNormalizationError("Quantile fitting may only use the training split")
                source_count = 0
                LOGGER.info("Fit action quantiles: source=%s, streams=%d", source.name, len(dataset))
                for values in dataset.iter_action_targets():
                    values = np.asarray(values, dtype=np.float32)
                    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
                        raise ActionNormalizationError(f"Invalid action labels in source {source.name}")
                    size = len(values)
                    if not size:
                        continue
                    batch = values.astype(np.float64)
                    batch_mean = batch.mean(axis=0)
                    delta = batch_mean - mean
                    new_count = count + size
                    m2 += np.square(batch - batch_mean).sum(axis=0) + delta**2 * (count * size / new_count)
                    mean += delta * (size / new_count)
                    count = new_count
                    source_count += size
                    for column, handle in enumerate(handles):
                        np.ascontiguousarray(values[:, column]).tofile(handle)
                    if time.monotonic() - last_progress >= 30:
                        LOGGER.info("Fit action quantiles: %d labels, %.2f GiB temporary columns",
                                    count, count * dimension * 4 / 1024**3)
                        last_progress = time.monotonic()
                LOGGER.info("Fit action quantiles: source=%s, valid window actions=%d", source.name, source_count)
                sources.append({"name": source.name, "count": source_count,
                                "history_frames": dataset.history_frames,
                                "prediction_horizon": dataset.prediction_horizon,
                                "selection": dataset.manifest()})
        if count == 0:
            raise ActionNormalizationError("No valid training actions for quantile normalization")
        ranks = np.array([0.01, 0.99]) * (count - 1)
        lower, upper = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
        quantiles = np.empty((2, dimension), dtype=np.float64)
        for column, path in enumerate(paths):
            LOGGER.info("Select exact action quantiles: dimension %d/%d", column + 1, dimension)
            values = np.memmap(path, mode="r+", dtype=np.float32, shape=(count,))
            values.partition(np.unique(np.r_[lower, upper]))
            lo, hi = values[lower].astype(np.float64), values[upper].astype(np.float64)
            quantiles[:, column] = lo + (hi - lo) * (ranks - lower)
            del values
    record = {
        "version": NORM_VERSION, "method": NORM_METHOD, "epsilon": QUANTILE_EPSILON,
        "horizon_shared": True, "num_arms": num_arms, "action_dim": dimension,
        "layout": action_layout(num_arms), "statistics_scope": "train_valid_window_actions",
        "count": count, "sources": sources,
        "mean": mean[None].tolist(), "std": np.sqrt(np.maximum(m2 / count, 0))[None].tolist(),
        "q01": quantiles[0:1].tolist(), "q99": quantiles[1:2].tolist(),
    }
    validate_action_norm(record, num_arms)
    return record


def save_action_norm(directory, record):
    validate_action_norm(record, record["num_arms"])
    path = Path(directory) / NORM_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_checkpoint_action_norm(checkpoint, num_arms):
    path = Path(checkpoint).expanduser()
    directory = path if path.is_dir() else path.parent
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise ActionNormalizationError(f"Quantile Stage2 checkpoint requires {config_path}")
    require_quantile_config(json.loads(config_path.read_text(encoding="utf-8")))
    path = directory / NORM_FILENAME
    if not path.is_file():
        raise ActionNormalizationError(f"Quantile Stage2 checkpoint requires {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_action_norm(record, num_arms)
    return record


def assert_action_norm_matches(policy, record):
    stats = validate_action_norm(record, policy.num_arms)
    for name in ("q01", "q99"):
        actual = getattr(policy, f"action_{name}").detach().float().cpu()
        expected = torch.tensor(stats[name], dtype=torch.float32)
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            raise ActionNormalizationError(f"Checkpoint action_{name} differs from {NORM_FILENAME}")


def initialize_action_normalization(policy, dataset, config, accelerator):
    """Fit only on rank zero, or reuse a checkpoint; publish one identical record."""
    from accelerate.utils import broadcast_object_list

    checkpoint = config.get("resume") or config.get("stage2_checkpoint")
    result = [None]
    if accelerator.is_main_process:
        try:
            root = Path(config["output_dir"])
            root.mkdir(parents=True, exist_ok=True)
            if checkpoint:
                record = load_checkpoint_action_norm(checkpoint, policy.num_arms)
                assert_action_norm_matches(policy, record)
            else:
                record = compute_action_norm(dataset, policy.num_arms, root)
            save_action_norm(root, record)
            result[0] = {"record": record}
        except Exception as exc:
            # Broadcast failures too, so other ranks do not wait for missing data.
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    broadcast_object_list(result)
    if "error" in result[0]:
        raise ActionNormalizationError(result[0]["error"])
    record = result[0]["record"]
    if checkpoint:
        assert_action_norm_matches(policy, record)
    else:
        policy.set_action_quantile_stats(record["q01"], record["q99"])
    LOGGER.info("Action normalization: %s, %d dimensions, %d valid training window actions%s",
                NORM_METHOD, record["action_dim"], record["count"], " (reused checkpoint)" if checkpoint else "")
    return record
