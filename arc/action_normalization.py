"""Componentwise Stage2 action statistics, portable artifacts, and initialization."""
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
NORM_METHOD = "componentwise"
NORM_VERSION = 2
POSITION_FALLBACK_SPAN = 1e-4
POSITION_MIN_SPAN = 1e-3
STAT_NAMES = ("mean", "std", "q01", "q99")
ACTION_COMPONENTS = ["x", "y", "z", "rotation6d_0", "rotation6d_1", "rotation6d_2",
                     "rotation6d_3", "rotation6d_4", "rotation6d_5", "gripper"]


class ActionNormalizationError(ValueError):
    """Invalid action shapes, normalization parameters, or checkpoint metadata."""


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
    require_action_normalization_config(config)


def require_action_normalization_config(config):
    pair = (config.get("action_normalization"), config.get("action_normalization_version"))
    if pair != (NORM_METHOD, NORM_VERSION):
        raise ActionNormalizationError(
            "Stage2 requires componentwise action normalization v2; "
            "start new componentwise runs from Stage1."
        )


def assert_stage2_contract(config, saved):
    """Reject semantic changes before constructing or loading a Stage2 model."""
    for key in ("language_conditioning", "action_normalization", "action_normalization_version"):
        if config.get(key) != saved.get(key):
            raise ValueError(f"Cannot load Stage2 with changed {key}; start from Stage1")


def componentwise_parameters(q01, q99, minimum, maximum):
    """XYZ affine parameters; rotation columns and gripper retain their units."""
    q01, q99, minimum, maximum = [np.asarray(v, dtype=np.float64) for v in (q01, q99, minimum, maximum)]
    if (q01.ndim != 2 or q01.shape[0] != 1 or q01.shape[1] not in (10, 20)
            or any(v.shape != q01.shape or not np.isfinite(v).all() for v in (q01, q99, minimum, maximum))
            or (minimum > q01).any() or (q01 > q99).any() or (q99 > maximum).any()):
        raise ActionNormalizationError("Invalid componentwise action statistics")
    fallback = q99 - q01 < POSITION_FALLBACK_SPAN
    low, high = np.where(fallback, minimum, q01), np.where(fallback, maximum, q99)
    position = np.arange(q01.shape[1]) % 10 < 3
    offset, scale = np.zeros_like(q01), np.ones_like(q01)
    offset[:, position] = (low[:, position] + high[:, position]) / 2
    scale[:, position] = np.maximum(high[:, position] - low[:, position], POSITION_MIN_SPAN) / 2
    return offset, scale


def _affine_parameters(action, offset, scale):
    if torch.is_tensor(action):
        if not action.is_floating_point():
            raise ActionNormalizationError("Actions must use a floating dtype")
        action = action.float()
        offset = torch.as_tensor(offset, dtype=torch.float32, device=action.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=action.device)
        finite = torch.isfinite
    else:
        action = np.asarray(action)
        if not np.issubdtype(action.dtype, np.floating):
            raise ActionNormalizationError("Actions must use a floating dtype")
        offset, scale = np.asarray(offset, dtype=np.float32), np.asarray(scale, dtype=np.float32)
        finite = np.isfinite
    if (action.ndim < 2 or offset.shape != (1, action.shape[-1]) or scale.shape != offset.shape
            or not bool(finite(offset).all() & finite(scale).all() & (scale > 0).all())):
        raise ActionNormalizationError("Expected finite affine [1,D] parameters with positive scales")
    return action, offset, scale


def componentwise_normalize(action, offset, scale):
    action, offset, scale = _affine_parameters(action, offset, scale)
    return (action - offset) / scale


def componentwise_unnormalize(action, offset, scale):
    action, offset, scale = _affine_parameters(action, offset, scale)
    return action * scale + offset


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
    require_action_normalization_config({"action_normalization": record.get("method"),
                                         "action_normalization_version": record.get("version")})
    expected = {"horizon_shared": True, "num_arms": num_arms, "action_dim": num_arms * 10,
                "layout": action_layout(num_arms), "statistics_scope": "train_valid_window_actions"}
    expected.update(position_fallback_span=POSITION_FALLBACK_SPAN, position_min_span=POSITION_MIN_SPAN,
                    component_rules=["quantile_xyz", "identity_rotation6d", "identity_gripper"])
    for key, value in expected.items():
        if record.get(key) != value:
            raise ActionNormalizationError(f"Invalid action normalization {key}: expected {value!r}")
    if not isinstance(record.get("count"), int) or record["count"] < 1:
        raise ActionNormalizationError("Action normalization requires a positive sample count")
    stats = {}
    for name in (*STAT_NAMES, "min", "max", "offset", "scale"):
        try:
            value = np.asarray(record[name], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionNormalizationError(f"Invalid action normalization {name}") from exc
        if value.shape != (1, num_arms * 10) or not np.isfinite(value).all():
            raise ActionNormalizationError(f"{name} must contain finite [1,{num_arms * 10}] statistics")
        stats[name] = value
    if (stats["std"] < 0).any() or (stats["q99"] < stats["q01"]).any():
        raise ActionNormalizationError("Invalid standard deviation or reversed quantiles")
    offset, scale = componentwise_parameters(stats["q01"], stats["q99"], stats["min"], stats["max"])
    for name, value in (("offset", offset), ("scale", scale)):
        if not np.array_equal(stats[name].astype(np.float32), value.astype(np.float32)):
            raise ActionNormalizationError(f"Saved {name} disagrees with componentwise rules")
    scale = scale.astype(np.float32)
    if not np.isfinite(offset.astype(np.float32)).all():
        raise ActionNormalizationError("Float32 offset must be finite")
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ActionNormalizationError("Float32 normalization scale must be finite and positive")
    return stats


def compute_action_norm(mixture, num_arms, scratch_dir):
    """Exact XYZ-only quantiles with bounded RAM and one temporary file per XYZ axis.

    Every valid window action has equal weight, regardless of sampler weights.
    In-place partition on contiguous memory-mapped columns avoids a full RAM copy.
    """
    dimension = num_arms * 10
    position_indices = np.flatnonzero(np.arange(dimension) % 10 < 3)
    statistics_dim = len(position_indices)
    layout = action_layout(num_arms)
    minimum, maximum = np.full(statistics_dim, np.inf), np.full(statistics_dim, -np.inf)
    count = 0
    mean, m2 = np.zeros(statistics_dim), np.zeros(statistics_dim)
    sources = []
    last_progress = time.monotonic()
    with TemporaryDirectory(prefix=".action-norm-", dir=scratch_dir) as temporary:
        paths = [Path(temporary) / f"xyz-{i}.bin" for i in position_indices]
        with ExitStack() as stack:
            handles = [stack.enter_context(path.open("wb")) for path in paths]
            for source in mixture.sources:
                if source.weight <= 0:
                    continue
                dataset = source.dataset
                if dataset.split != "train":
                    raise ActionNormalizationError("Quantile fitting may only use the training split")
                source_count = 0
                LOGGER.info("Fit XYZ normalization: source=%s, streams=%d, position dimensions=%d",
                            source.name, len(dataset), statistics_dim)
                for values in dataset.iter_action_targets():
                    values = np.asarray(values, dtype=np.float32)
                    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
                        raise ActionNormalizationError(f"Invalid action labels in source {source.name}")
                    size = len(values)
                    if not size:
                        continue
                    values = values[:, position_indices]
                    minimum = np.minimum(minimum, values.min(axis=0))
                    maximum = np.maximum(maximum, values.max(axis=0))
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
                        LOGGER.info("Fit XYZ normalization: %d labels, %.2f GiB temporary XYZ columns",
                                    count, count * statistics_dim * 4 / 1024**3)
                        last_progress = time.monotonic()
                LOGGER.info("Fit XYZ normalization: source=%s, valid window actions=%d", source.name, source_count)
                sources.append({"name": source.name, "count": source_count,
                                "history_frames": dataset.history_frames,
                                "prediction_horizon": dataset.prediction_horizon,
                                "selection": dataset.manifest()})
        if count == 0:
            raise ActionNormalizationError("No valid training actions for XYZ normalization")
        ranks = np.array([0.01, 0.99]) * (count - 1)
        lower, upper = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
        quantiles = np.empty((2, statistics_dim), dtype=np.float64)
        for column, path in enumerate(paths):
            LOGGER.info("Select exact XYZ quantiles: %s (%d/%d)",
                        layout[position_indices[column]], column + 1, statistics_dim)
            values = np.memmap(path, mode="r+", dtype=np.float32, shape=(count,))
            values.partition(np.unique(np.r_[lower, upper]))
            lo, hi = values[lower].astype(np.float64), values[upper].astype(np.float64)
            quantiles[:, column] = lo + (hi - lo) * (ranks - lower)
            del values
    def action_columns(values):
        # Keep the action-aligned artifact layout. Non-XYZ slots are unused zeros,
        # not measured statistics; their actual affine transform stays (0, 1).
        expanded = np.zeros((*values.shape[:-1], dimension), dtype=np.float64)
        expanded[..., position_indices] = values
        return expanded

    mean, m2 = action_columns(mean), action_columns(m2)
    minimum, maximum = action_columns(minimum), action_columns(maximum)
    quantiles = action_columns(quantiles)
    record = {
        "version": NORM_VERSION, "method": NORM_METHOD,
        "horizon_shared": True, "num_arms": num_arms, "action_dim": dimension,
        "layout": action_layout(num_arms), "statistics_scope": "train_valid_window_actions",
        "count": count, "sources": sources,
        "mean": mean[None].tolist(), "std": np.sqrt(np.maximum(m2 / count, 0))[None].tolist(),
        "q01": quantiles[0:1].tolist(), "q99": quantiles[1:2].tolist(),
    }
    offset, scale = componentwise_parameters(quantiles[0:1], quantiles[1:2], minimum[None], maximum[None])
    record.update(min=minimum[None].tolist(), max=maximum[None].tolist(),
                  offset=offset.tolist(), scale=scale.tolist(),
                  position_fallback_span=POSITION_FALLBACK_SPAN, position_min_span=POSITION_MIN_SPAN,
                  component_rules=["quantile_xyz", "identity_rotation6d", "identity_gripper"])
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
        raise ActionNormalizationError(f"Componentwise Stage2 checkpoint requires {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require_action_normalization_config(config)
    path = directory / NORM_FILENAME
    if not path.is_file():
        raise ActionNormalizationError(f"Componentwise Stage2 checkpoint requires {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("method"), record.get("version")) != (config["action_normalization"], config["action_normalization_version"]):
        raise ActionNormalizationError("Checkpoint config and action_norm.json versions disagree")
    validate_action_norm(record, num_arms)
    return record


def assert_action_norm_matches(policy, record):
    stats = validate_action_norm(record, policy.num_arms)
    for name in ("offset", "scale"):
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
        policy.set_action_component_stats(record["offset"], record["scale"])
    LOGGER.info("Action normalization: %s, %d XYZ dimensions scaled; rotation6D/gripper unchanged; "
                "%d valid training window actions%s", record["method"], policy.num_arms * 3,
                record["count"], " (reused checkpoint)" if checkpoint else "")
    return record
