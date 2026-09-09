"""Metric-space TCP errors and mergeable depth statistics (no scale alignment)."""

from __future__ import annotations

import numpy as np
import torch

from arc.rotation import rpy_to_matrix, so3_geodesic_angle


DEPTH_RANGES = {"within_3m": 3.0, "all_valid": None}
DEPTH_STAT_NAMES = (
    "valid_pixels", "abs_sum", "abs_rel_sum", "sq_sum", "sq_rel_sum",
    "log_sq_sum", "log10_abs_sum", "delta1_count", "delta2_count",
    "delta3_count", "nonpositive_prediction_count", "nonfinite_prediction_count",
)
ARMS = ("left", "right", "both")


def rotation_error_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.rad2deg(so3_geodesic_angle(
        torch.as_tensor(prediction, dtype=torch.float64),
        torch.as_tensor(target, dtype=torch.float64),
    ).numpy())


def tcp_error_arrays(
    prediction: dict[str, np.ndarray], states: np.ndarray,
    frame_indices: np.ndarray, frame_rate: float,
) -> dict[str, np.ndarray]:
    """All arrays have [time, arm, ...]; endpoints have exactly one time slot."""
    states = np.asarray(states, dtype=np.float64)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if states.shape != (len(indices), 2, 7) or len(indices) < 2:
        raise ValueError("Expected at least two TCP frames with shape [T,2,7]")
    if not np.isfinite(states).all() or not np.isin(states[..., 6], (0, 1)).all():
        raise ValueError("TCP truth must be finite with binary gripper labels")
    dt = np.diff(indices) / frame_rate
    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("Frame times must be finite and strictly increasing")
    position = np.asarray(prediction["position"], dtype=np.float64)
    rotation = np.asarray(prediction["rotation"], dtype=np.float64)
    gripper = np.asarray(prediction["gripper"], dtype=np.float64)
    for name, value, shape in (
        ("position", position, (len(indices), 2, 3)),
        ("rotation", rotation, (len(indices), 2, 3, 3)),
        ("gripper", gripper, (len(indices), 2)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid predicted TCP {name}")
    target_rotation = rpy_to_matrix(torch.from_numpy(states[..., 3:6])).numpy()
    delta = (position - states[..., :3]) * 1000.0
    position_error = np.linalg.norm(delta, axis=-1)
    angle = rotation_error_deg(rotation, target_rotation)
    displacement = np.linalg.norm(np.diff(delta, axis=0), axis=-1)
    pred_relative = rotation[:-1].swapaxes(-1, -2) @ rotation[1:]
    gt_relative = target_rotation[:-1].swapaxes(-1, -2) @ target_rotation[1:]
    relative_angle = rotation_error_deg(pred_relative, gt_relative)
    return {
        "position_mm": position_error,
        "axis_abs_mm": np.abs(delta),
        "rotation_deg": angle,
        "gripper_pred": gripper >= 0.5,
        "gripper_gt": states[..., 6] >= 0.5,
        "displacement_mm": displacement,
        "velocity_mm_s": displacement / dt[:, None],
        "relative_rotation_deg": relative_angle,
        "relative_rotation_rate_deg_s": relative_angle / dt[:, None],
        "endpoint_position_mm": position_error[-1:],
        "endpoint_rotation_deg": angle[-1:],
    }


def _distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def summarize_tcp(errors: dict[str, np.ndarray], arm: str) -> dict:
    arm_index = ARMS.index(arm)

    def select(key: str) -> np.ndarray:
        value = errors[key]
        return value if arm == "both" else value[:, arm_index]

    position = select("position_mm")
    angle = select("rotation_deg")
    result = {
        "sample_count": int(position.size),
        "pair_count": int(select("displacement_mm").size),
        "endpoint_count": int(select("endpoint_position_mm").size),
        **_distribution(position, "position_mm"),
        **_distribution(angle, "rotation_deg"),
    }
    for axis_index, axis in enumerate("xyz"):
        result[f"{axis}_mae_mm"] = float(select("axis_abs_mm")[..., axis_index].mean())
    pred, target = select("gripper_pred").astype(bool), select("gripper_gt").astype(bool)
    tp, fp = int((pred & target).sum()), int((pred & ~target).sum())
    fn, tn = int((~pred & target).sum()), int((~pred & ~target).sum())
    result.update({
        "gripper_tp_count": tp, "gripper_fp_count": fp,
        "gripper_fn_count": fn, "gripper_tn_count": tn,
        "gripper_accuracy": (tp + tn) / target.size,
        "gripper_precision": tp / (tp + fp) if tp + fp else None,
        "gripper_recall": tp / (tp + fn) if tp + fn else None,
        "gripper_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    })
    for key in (
        "displacement_mm", "velocity_mm_s", "relative_rotation_deg",
        "relative_rotation_rate_deg_s", "endpoint_position_mm", "endpoint_rotation_deg",
    ):
        result[f"{key}_mean"] = float(select(key).mean())
        result[f"{key}_rmse"] = float(np.sqrt(np.square(select(key)).mean()))
    for threshold in (10, 20, 50):
        result[f"position_le_{threshold}mm"] = float((position <= threshold).mean())
    for threshold in (5, 10, 15):
        result[f"rotation_le_{threshold}deg"] = float((angle <= threshold).mean())
    result["pose_le_20mm_10deg"] = float(((position <= 20) & (angle <= 10)).mean())
    return result


def depth_frame_statistics(
    prediction: np.ndarray, target: np.ndarray, max_depth: float | None,
) -> np.ndarray:
    """Float64 sufficient statistics; the mask depends only on ground truth.

    Nonfinite predictions are counted and make the eventual depth result invalid.
    Finite nonpositive predictions retain their raw linear error, fail delta,
    and use a 1e-5 m lower bound for logarithms.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Depth prediction/truth must have identical [H,W] shapes")
    mask = np.isfinite(target) & (target > 0)
    if max_depth is not None:
        mask &= target <= max_depth
    p, g = prediction[mask], target[mask]
    stats = np.zeros(len(DEPTH_STAT_NAMES), dtype=np.float64)
    stats[0] = len(g)
    stats[-2] = np.count_nonzero(np.isfinite(p) & (p <= 0))
    stats[-1] = np.count_nonzero(~np.isfinite(p))
    if not len(g) or stats[-1]:
        return stats
    error = p - g
    safe_p = np.maximum(p, 1e-5)
    log_error = np.log(safe_p) - np.log(g)
    ratio = np.maximum(safe_p / g, g / safe_p)
    stats[1:7] = (
        np.abs(error).sum(), (np.abs(error) / g).sum(), np.square(error).sum(),
        (np.square(error) / g).sum(), np.square(log_error).sum(),
        np.abs(np.log10(safe_p) - np.log10(g)).sum(),
    )
    stats[7:10] = [np.count_nonzero((p > 0) & (ratio < 1.25**k)) for k in (1, 2, 3)]
    return stats


def summarize_depth(frame_stats: np.ndarray) -> dict:
    frame_stats = np.asarray(frame_stats, dtype=np.float64).reshape(-1, len(DEPTH_STAT_NAMES))
    total = frame_stats.sum(axis=0)
    n = int(total[0])
    result = {
        "valid_pixel_count": n,
        "valid_frame_count": int(np.count_nonzero(frame_stats[:, 0])),
        "empty_frame_count": int(np.count_nonzero(frame_stats[:, 0] == 0)),
        "nonpositive_prediction_count": int(total[-2]),
        "nonfinite_prediction_count": int(total[-1]),
    }
    names = ("mae_m", "abs_rel", "sq_rel_m", "rmse_m", "rmse_log", "log10", "delta1", "delta2", "delta3")
    if not n or total[-1]:
        return {**result, **dict.fromkeys(names)}
    values = (
        total[1] / n, total[2] / n, total[4] / n, np.sqrt(total[3] / n),
        np.sqrt(total[5] / n), total[6] / n, *(total[7:10] / n),
    )
    return {**result, **{key: float(value) for key, value in zip(names, values)}}


def macro_metrics(metrics: list[dict]) -> dict:
    """Equal group means; counts add, undefined metric entries are omitted."""
    if not metrics:
        return {}
    result = {}
    for key in metrics[0]:
        values = [m[key] for m in metrics if m.get(key) is not None]
        result[key] = (
            int(sum(values)) if key.endswith("_count") else
            float(np.mean(values)) if values else None
        )
    return result
