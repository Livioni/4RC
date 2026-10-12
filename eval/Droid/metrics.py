"""Absolute DROID metrics, aligned TCP trajectories, and projected 2D tracking."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


DEPTH_RANGES = {"within_3m": 3.0, "all_valid": None}
DEPTH_COLUMNS = (
    "count", "abs_error", "squared_error", "abs_rel", "sq_rel", "log_squared_error",
    "log10_error", "log_error", "delta1", "delta2", "delta3", "nonpositive_count",
)
RAY_COLUMNS = ("count", "l1", "direction_l1", "direction_angle_deg", "origin_l2_m")
LOG_EPS = 1e-5
TCP_TRAJECTORY_KEYS = ("tcp_ate_sim3_m", "tcp_rpe_t_sim3_m", "tcp_rpe_r_sim3_deg")
TCP_TRACKING_THRESHOLDS = (1, 2, 4, 8, 16)
TCP_TRACKING_SAMPLE_KEYS = ("tcp_tracking_error_px", "tcp_tracking_error_px_256",
                            "tcp_tracking_projectable", "tcp_tracking_pred_in_view")


def tcp_tracking_definitions() -> dict:
    return {
        "source": "raw camera-coordinate 3D TCP projected with the same GT K as GT TCP; no Sim(3), predicted K, or clipping",
        "image_size_wh": [320, 180],
        "mask": "GT geometrically in RGB AND complete GT state valid AND original frame index greater than the initial query frame",
        "query_frame_included": False,
        "prediction_out_of_view": "retained; finite positive-Z projections use their actual pixel distance even outside RGB",
        "unprojectable_prediction": "nonfinite or nonpositive-Z projection counts as incorrect at all thresholds and remains in their denominator; EPE excludes it with a separate count",
        "epe": "2D Euclidean pixel distance; mean/RMSE/median/P95 over projectable eligible frames, including outside-image predictions",
        "thresholds_px": list(TCP_TRACKING_THRESHOLDS),
        "points_within_threshold": "fraction of all eligible GT frames with pixel distance strictly less than threshold; does not filter on predicted visibility or confidence",
        "average_pts_within_thresh": "arithmetic mean of the five threshold accuracies",
        "normalized_256": "scale x differences by 256/320 and y differences by 256/180 before evaluating the same thresholds",
        "in_view_fraction": "fraction of eligible GT frames whose prediction projects inside the 320x180 RGB with positive Z",
        "aggregation": "camera_macro gives each defined episode/camera metric equal weight; frame_micro pools eligible samples without pairing cameras/episodes",
        "visibility_scope": "geometric in-image visibility only; no occlusion labels or occlusion predictions, so no TAP-Vid AJ/OA is reported",
        "reference": "https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/evaluation_datasets.py",
    }


def tcp_tracking_samples(arrays) -> dict[str, np.ndarray]:
    """Compact 2D errors over GT-eligible post-query frames, independent of metric_mask."""
    indices = np.asarray(arrays["frame_indices"])
    gt_uv, pred_uv = (np.asarray(arrays[key], dtype=np.float64) for key in ("gt_tcp_uv", "pred_tcp_uv"))
    position = np.asarray(arrays["tcp_position"], dtype=np.float64)
    gt_visible, state_valid, pred_visible = (
        np.asarray(arrays[key], dtype=bool) for key in ("gt_visible", "gt_state_valid", "pred_visible"))
    n = len(indices)
    if (indices.shape != (n,) or gt_uv.shape != (n, 2) or pred_uv.shape != (n, 2) or position.shape != (n, 3)
            or any(values.shape != (n,) for values in (gt_visible, state_valid, pred_visible))):
        raise ValueError("Invalid TCP 2D tracking array shapes")
    if not np.issubdtype(indices.dtype, np.integer) or np.any(np.diff(indices) <= 0):
        raise ValueError("Tracking frame indices must be increasing integers")
    eligible = gt_visible & state_valid & (indices > indices[0] if n else np.zeros(0, dtype=bool))
    if not np.isfinite(gt_uv[eligible]).all():
        raise ValueError("Nonfinite GT projection on eligible tracking frames")
    projectable = (np.isfinite(position).all(-1) & (position[:, 2] > 0) & np.isfinite(pred_uv).all(-1))[eligible]
    error = np.full(int(eligible.sum()), np.nan, dtype=np.float64)
    error_256 = error.copy()
    difference = pred_uv[eligible][projectable] - gt_uv[eligible][projectable]
    error[projectable] = np.linalg.norm(difference, axis=-1)
    error_256[projectable] = np.linalg.norm(difference * [256 / 320, 256 / 180], axis=-1)
    if not np.isfinite(error[projectable]).all() or not np.isfinite(error_256[projectable]).all():
        raise ValueError("Nonfinite pixel distance on projectable tracking frames")
    return {"tcp_tracking_error_px": error, "tcp_tracking_error_px_256": error_256,
            "tcp_tracking_projectable": projectable,
            "tcp_tracking_pred_in_view": pred_visible[eligible] & projectable,
            "tcp_tracking_frame_indices": indices[eligible]}


def summarize_tcp_tracking(samples: dict[str, np.ndarray]) -> dict:
    """PCK denominators include every eligible frame, even unprojectable predictions."""
    error = np.asarray(samples["tcp_tracking_error_px"], dtype=np.float64)
    normalized = np.asarray(samples["tcp_tracking_error_px_256"], dtype=np.float64)
    projectable = np.asarray(samples["tcp_tracking_projectable"], dtype=bool)
    in_view = np.asarray(samples["tcp_tracking_pred_in_view"], dtype=bool)
    n = len(error)
    if any(values.shape != (n,) for values in (error, normalized, projectable, in_view)):
        raise ValueError("Invalid compact tracking sample shapes")
    if not np.isfinite(error[projectable]).all() or not np.isfinite(normalized[projectable]).all():
        raise ValueError("Nonfinite scored pixel tracking distance")
    result = {"tcp_tracking_frame_count": n,
              "tcp_tracking_projectable_frame_count": int(projectable.sum()),
              "tcp_tracking_unprojectable_frame_count": int((~projectable).sum()),
              "tcp_tracking_prediction_in_view_frame_count": int(in_view.sum()),
              "tcp_tracking_prediction_out_of_view_frame_count": int((~in_view).sum()),
              "tcp_tracking_in_view_fraction": float(in_view.mean()) if n else None}
    for values, suffix in ((error, ""), (normalized, "_256")):
        distances = values[projectable]
        for label, value in (
            ("mean", distances.mean() if len(distances) else None),
            ("rmse", np.sqrt(np.square(distances).mean()) if len(distances) else None),
            ("median", np.median(distances) if len(distances) else None),
            ("p95", np.percentile(distances, 95) if len(distances) else None),
        ):
            result[f"tcp_tracking_epe_px{suffix}_{label}"] = float(value) if value is not None else None
        rates = []
        for threshold in TCP_TRACKING_THRESHOLDS:
            correct = int(np.count_nonzero(projectable & (values < threshold)))
            rate = correct / n if n else None
            key = f"tcp_tracking_pts_within_{threshold}px{suffix}"
            result[key], result[f"{key}_count"] = rate, correct
            rates.append(rate)
        result[f"tcp_tracking_average_pts_within_thresh{suffix}"] = float(np.mean(rates)) if n else None
    return result


def tcp_trajectory_definitions(delta_frames: int = 1) -> dict:
    return {
        "alignment": "one position-based Umeyama Sim(3) fit per complete episode/camera, using metric_mask frames",
        "coordinate_frame": "fixed OpenCV camera; TCP poses map TCP-local coordinates into this camera",
        "pose_transform": "aligned_position = scale * R * prediction + t; aligned_orientation = R * prediction_orientation",
        "ate": "L2 distance between aligned and GT positions; primary ATE statistic is RMSE, unit meter",
        "rpe": "E_ij = inverse(inverse(Q_i) @ Q_j) @ (inverse(P_i) @ P_j); Q=GT, P=aligned prediction",
        "rpe_t": "L2 norm of translation(E_ij), unit meter; full SE(3) relative pose, not position differences alone",
        "rpe_r": "SO(3) geodesic angle of rotation(E_ij), unit degree",
        "rpe_delta_frames": delta_frames,
        "pairs": "all overlapping original-frame pairs with frame_j - frame_i = delta and both metric_mask endpoints valid; never across cameras/episodes",
        "degenerate_policy": "fewer than 3 valid poses, covariance rank < 2, or nonpositive scale: null trajectory metrics; absolute metrics retained",
        "aggregation": "camera_macro averages defined per-camera statistics; frame_micro pools ATE frames and RPE pairs after separate per-camera fits",
    }


def fit_sim3(prediction: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit p_target = scale * R * p_prediction + t, without reflections.

    A line or a stationary trajectory cannot determine a unique global rotation.
    Reject these cases instead of selecting an arbitrary rotation for the RPE.
    """
    p, g = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != g.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Sim(3) expects corresponding [N,3] positions")
    if len(p) < 3:
        raise ValueError("insufficient_valid_frames")
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("nonfinite_alignment_positions")
    p_mean, g_mean = p.mean(axis=0), g.mean(axis=0)
    p_centered, g_centered = p - p_mean, g - g_mean
    covariance = g_centered.T @ p_centered / len(p)
    u, singular, vh = np.linalg.svd(covariance)
    # Relative tolerance also accepts small but well-conditioned metric motions.
    if singular[0] <= 0 or singular[1] <= max(singular[0] * 1e-12, np.finfo(float).tiny):
        raise ValueError("degenerate_alignment_covariance")
    signs = np.ones(3)
    if np.linalg.det(u @ vh) < 0:
        signs[-1] = -1
    rotation = (u * signs) @ vh
    variance = np.square(p_centered).sum() / len(p)
    scale = float(np.dot(singular, signs) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("nonpositive_alignment_scale")
    translation = g_mean - scale * rotation @ p_mean
    return scale, rotation, translation


def tcp_trajectory_errors(position: np.ndarray, rotation: np.ndarray, target: np.ndarray,
                          frame_indices: np.ndarray, mask: np.ndarray,
                          delta_frames: int = 1) -> tuple[dict[str, np.ndarray], dict, np.ndarray]:
    """Compact ATE/RPE samples, a global alignment record, and original RPE frame pairs."""
    p, r = np.asarray(position, dtype=np.float64), np.asarray(rotation, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    indices, valid = np.asarray(frame_indices), np.asarray(mask, dtype=bool)
    n = len(p)
    if (p.shape != (n, 3) or r.shape != (n, 3, 3) or gt.shape != (n, 7)
            or indices.shape != (n,) or valid.shape != (n,)):
        raise ValueError("Invalid TCP trajectory array shapes")
    if (not np.issubdtype(indices.dtype, np.integer) or np.any(np.diff(indices) <= 0)
            or isinstance(delta_frames, bool) or not isinstance(delta_frames, (int, np.integer)) or delta_frames < 1):
        raise ValueError("Frame indices must be increasing integers and RPE delta a positive integer")
    if not all(np.isfinite(values[valid]).all() for values in (p, r, gt[:, :6])):
        raise ValueError("Nonfinite scored TCP trajectory poses")
    errors = {key: np.empty(0, dtype=np.float64) for key in TCP_TRAJECTORY_KEYS}
    alignment = {"status": "undefined", "fit_frame_count": int(valid.sum()),
                 "rpe_delta_frames": int(delta_frames), "scale": None, "rotation": None, "translation_m": None}
    empty_pairs = np.empty((0, 2), dtype=np.int64)
    try:
        scale, global_rotation, translation = fit_sim3(p[valid], gt[valid, :3])
    except (ValueError, np.linalg.LinAlgError) as error:
        alignment["reason"] = str(error)
        return errors, alignment, empty_pairs
    alignment.update(status="success", scale=scale, rotation=global_rotation.tolist(), translation_m=translation.tolist())
    aligned_p = scale * (p @ global_rotation.T) + translation
    aligned_r = global_rotation @ r
    errors["tcp_ate_sim3_m"] = np.linalg.norm(aligned_p[valid] - gt[valid, :3], axis=-1)

    # Pair original frame numbers, rather than adjacent entries of the filtered trajectory.
    start = np.flatnonzero(valid)
    end = np.searchsorted(indices, indices[start] + delta_frames)
    in_range = end < n
    start, end = start[in_range], end[in_range]
    paired = valid[end] & (indices[end] - indices[start] == delta_frames)
    start, end = start[paired], end[paired]
    gt_r = rpy_to_matrix(gt[:, 3:6])
    pred_start_inverse = np.swapaxes(aligned_r[start], -1, -2)
    gt_start_inverse = np.swapaxes(gt_r[start], -1, -2)
    pred_delta_t = np.einsum("nij,nj->ni", pred_start_inverse, aligned_p[end] - aligned_p[start])
    gt_delta_t = np.einsum("nij,nj->ni", gt_start_inverse, gt[end, :3] - gt[start, :3])
    # Multiplication by inverse(GT delta rotation) preserves this vector's norm.
    errors["tcp_rpe_t_sim3_m"] = np.linalg.norm(pred_delta_t - gt_delta_t, axis=-1)
    errors["tcp_rpe_r_sim3_deg"] = rotation_error_deg(pred_start_inverse @ aligned_r[end], gt_start_inverse @ gt_r[end])
    return errors, alignment, np.column_stack((indices[start], indices[end]))


def summarize_tcp_trajectory(errors: dict[str, np.ndarray], alignments: list[dict]) -> dict:
    """Pool aligned frame/pair errors; counters describe alignment coverage separately."""
    result = {
        "tcp_sim3_alignment_success_count": sum(row["status"] == "success" for row in alignments),
        "tcp_sim3_alignment_undefined_count": sum(row["status"] != "success" for row in alignments),
        "tcp_sim3_candidate_frame_count": sum(row["fit_frame_count"] for row in alignments),
        "tcp_ate_sim3_frame_count": len(errors["tcp_ate_sim3_m"]),
        "tcp_rpe_sim3_pair_count": len(errors["tcp_rpe_t_sim3_m"]),
    }
    for key in TCP_TRAJECTORY_KEYS:
        values = np.asarray(errors[key], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite TCP trajectory errors: {key}")
        for label, value in (
            ("mean", values.mean() if len(values) else None),
            ("rmse", np.sqrt(np.square(values).mean()) if len(values) else None),
            ("median", np.median(values) if len(values) else None),
            ("p95", np.percentile(values, 95) if len(values) else None),
        ):
            result[f"{key}_{label}"] = float(value) if value is not None else None
    return result


def project_positions(positions: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera XYZ into the unpadded 320x180 RGB; visibility is geometric."""
    positions = np.asarray(positions, dtype=np.float64)
    homogeneous = positions @ np.asarray(intrinsics, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        uv = homogeneous[..., :2] / homogeneous[..., 2:3]
    visible = (np.isfinite(positions).all(-1) & (positions[..., 2] > 0)
               & np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] < 320)
               & (uv[..., 1] >= 0) & (uv[..., 1] < 180))
    return uv, visible


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Fixed-axis XYZ, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = np.moveaxis(np.asarray(rpy, dtype=np.float64), -1, 0)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                     sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                     -sp, cp*sr, cp*cr), axis=-1).reshape(*np.shape(r), 3, 3)


def rotation_error_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """SO(3) angle via atan2, stable at identity and 180 degrees."""
    relative = np.swapaxes(prediction, -1, -2) @ target
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    skew = np.stack((relative[..., 2, 1] - relative[..., 1, 2],
                     relative[..., 0, 2] - relative[..., 2, 0],
                     relative[..., 1, 0] - relative[..., 0, 1]), axis=-1)
    return np.rad2deg(np.arctan2(np.linalg.norm(skew, axis=-1) / 2, cosine))


def depth_statistics(prediction: np.ndarray, target: np.ndarray, max_depth: float | None) -> np.ndarray:
    """Sufficient statistics over GT-valid pixels; negative predictions stay in errors."""
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"Depth shape mismatch: {prediction.shape} vs {target.shape}")
    mask = np.isfinite(target) & (target > 0)
    if max_depth is not None:
        mask &= target <= max_depth
    p, g = prediction[mask], target[mask]
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite depth prediction at a GT-valid pixel")
    error = p - g
    safe_p = np.maximum(p, LOG_EPS)
    log_error = np.log(safe_p) - np.log(g)
    ratio = np.maximum(safe_p / g, g / safe_p)
    return np.asarray([
        len(g), np.abs(error).sum(), np.square(error).sum(), (np.abs(error) / g).sum(),
        (np.square(error) / g).sum(), np.square(log_error).sum(),
        np.abs(np.log10(safe_p) - np.log10(g)).sum(), log_error.sum(),
        *[np.count_nonzero((p > 0) & (ratio < 1.25**k)) for k in (1, 2, 3)],
        np.count_nonzero(p <= 0),
    ], dtype=np.float64)


def summarize_depth(statistics: np.ndarray) -> dict:
    statistics = np.asarray(statistics, dtype=np.float64).reshape(-1, len(DEPTH_COLUMNS))
    total = statistics.sum(axis=0)
    n = int(total[0])
    names = ("mae_m", "rmse_m", "abs_rel", "sq_rel_m", "rmse_log", "log10", "silog",
             "delta1", "delta2", "delta3")
    result = {"valid_pixel_count": n, "valid_frame_count": int(np.count_nonzero(statistics[:, 0])),
              "nonpositive_prediction_count": int(total[-1])}
    if not n:
        return {**result, **dict.fromkeys(names)}
    mean = total / n
    return {**result, "mae_m": float(mean[1]), "rmse_m": math.sqrt(mean[2]),
            "abs_rel": float(mean[3]), "sq_rel_m": float(mean[4]),
            "rmse_log": math.sqrt(mean[5]), "log10": float(mean[6]),
            "silog": 100 * math.sqrt(max(float(mean[5] - mean[7]**2), 0)),
            **{f"delta{k}": float(mean[7 + k]) for k in (1, 2, 3)}}


def ray_target(intrinsics: np.ndarray, world_to_camera: np.ndarray,
               height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Match arc.datasets.utils.compute_gt_ray_map and nearest original-mask resize."""
    k = np.asarray(intrinsics, dtype=np.float64).copy()
    k[:2, 2] += 1  # Camera rays are predicted in padded 322x182 coordinates.
    x = np.linspace(1 / width, 1 - 1 / width, width)
    y = np.linspace(1 / height, 1 - 1 / height, height)
    xx, yy = np.meshgrid(x, y)
    canonical = np.stack(((xx * 322 - k[0, 2]) / k[0, 0],
                          (yy * 182 - k[1, 2]) / k[1, 1], np.ones_like(xx)), axis=-1)
    c2w = np.linalg.inv(world_to_camera)
    direction = canonical @ c2w[:3, :3].T
    origin = np.broadcast_to(c2w[:3, 3], direction.shape)
    # torch.interpolate(mode='nearest') samples floor(output_index * input_size / output_size).
    rows = np.floor(np.arange(height) * 182 / height).astype(int)
    cols = np.floor(np.arange(width) * 322 / width).astype(int)
    valid = ((rows[:, None] >= 1) & (rows[:, None] < 181)
             & (cols[None, :] >= 1) & (cols[None, :] < 321))
    return np.concatenate((direction, origin), axis=-1), valid


def ray_statistics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape or mask.shape != target.shape[:2]:
        raise ValueError("Invalid ray prediction/target/mask shape")
    p, g = np.asarray(prediction, dtype=np.float64)[mask], target[mask]
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite ray prediction in the RGB region")
    error = np.abs(p - g)
    pd, gd = p[:, :3], g[:, :3]
    norm = np.linalg.norm(pd, axis=-1)
    cosine = np.sum(pd * gd, axis=-1) / np.maximum(norm * np.linalg.norm(gd, axis=-1), 1e-12)
    # A zero predicted direction is invalid geometry and counts as a 180-degree error.
    angle = np.where(norm > 1e-12, np.rad2deg(np.arccos(np.clip(cosine, -1, 1))), 180)
    return np.asarray([len(p), error.mean(axis=-1).sum(), error[:, :3].mean(axis=-1).sum(),
                       angle.sum(), np.linalg.norm(p[:, 3:] - g[:, 3:], axis=-1).sum()])


def summarize_ray(statistics: np.ndarray) -> dict:
    total = np.asarray(statistics, dtype=np.float64).reshape(-1, len(RAY_COLUMNS)).sum(axis=0)
    names = ("l1", "direction_l1", "direction_angle_deg", "origin_l2_m")
    return {"valid_ray_count": int(total[0]),
            **{name: float(value / total[0]) if total[0] else None
               for name, value in zip(names, total[1:])}}


def frame_errors(position: np.ndarray, rotation: np.ndarray, gripper: np.ndarray,
                 confidence: np.ndarray, camera_to_world: np.ndarray, predicted_k: np.ndarray,
                 pose_encoding: np.ndarray, target: np.ndarray, gt_k: np.ndarray,
                 world_to_camera: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame TCP, absolute camera pose and intrinsics errors (all in GT conventions)."""
    gt_rotation = rpy_to_matrix(target[:, 3:6])
    position_error = np.linalg.norm(position - target[:, :3], axis=-1)
    rotation_error = rotation_error_deg(rotation, gt_rotation)
    pred_uv, _ = project_positions(position, gt_k)
    gt_uv, _ = project_positions(target[:, :3], gt_k)
    pred_w2c_rotation = np.swapaxes(camera_to_world[:, :3, :3], -1, -2)
    pred_w2c_translation = -(pred_w2c_rotation @ camera_to_world[:, :3, 3, None])[..., 0]
    gt_c2w = np.linalg.inv(world_to_camera)
    base_position = np.einsum("tij,tj->ti", camera_to_world[:, :3, :3], position) + camera_to_world[:, :3, 3]
    gt_base_position = target[:, :3] @ gt_c2w[:3, :3].T + gt_c2w[:3, 3]
    base_rotation = camera_to_world[:, :3, :3] @ rotation
    gt_base_rotation = gt_c2w[:3, :3] @ gt_rotation
    result = {
        "tcp_position_m": position_error, "tcp_rotation_deg": rotation_error,
        "tcp_reprojection_px": np.linalg.norm(pred_uv - gt_uv, axis=-1),
        "tcp_base_position_m": np.linalg.norm(base_position - gt_base_position, axis=-1),
        "tcp_base_rotation_deg": rotation_error_deg(base_rotation, gt_base_rotation),
        "tcp_gripper_error": gripper - target[:, 6],
        "tcp_gripper_accuracy": ((gripper >= 0.5) == (target[:, 6] >= 0.5)).astype(float),
        "tcp_confidence": confidence,
        "camera_translation_m": np.linalg.norm(pred_w2c_translation - world_to_camera[:3, 3], axis=-1),
        "camera_rotation_deg": rotation_error_deg(pred_w2c_rotation, world_to_camera[:3, :3]),
        "camera_center_m": np.linalg.norm(camera_to_world[:, :3, 3] - gt_c2w[:3, 3], axis=-1),
    }
    for axis, error in zip("xyz", np.abs(position - target[:, :3]).T):
        result[f"tcp_{axis}_abs_m"] = error
    for millimeters, degrees in ((10, 5), (20, 10), (50, 15)):
        result[f"tcp_pose_le_{millimeters}mm_{degrees}deg"] = (
            (position_error <= millimeters / 1000) & (rotation_error <= degrees)).astype(float)
    for name, row, col in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)):
        error = np.abs(predicted_k[:, row, col] - gt_k[row, col])
        result[f"camera_{name}_abs_px"] = error
        if name in ("fx", "fy"):
            result[f"camera_{name}_abs_rel"] = error / gt_k[row, col]
    for label, dimension, focal, column in (("h", 182, gt_k[1, 1], 7), ("w", 322, gt_k[0, 0], 8)):
        gt_fov = 2 * np.arctan(dimension / (2 * focal))
        result[f"camera_fov_{label}_abs_deg"] = np.rad2deg(np.abs(pose_encoding[:, column] - gt_fov))
    return result


def summarize_errors(errors: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    result = {"valid_frame_count": int(np.count_nonzero(mask))}
    distribution_keys = {"tcp_position_m", "tcp_rotation_deg", "tcp_base_position_m",
                         "tcp_base_rotation_deg", "tcp_reprojection_px", "camera_translation_m",
                         "camera_rotation_deg", "camera_center_m"}
    for name, values in errors.items():
        selected = np.asarray(values, dtype=np.float64)[mask]
        if not np.isfinite(selected).all():
            raise ValueError(f"Nonfinite scored values: {name}")
        if name == "tcp_gripper_error":
            result["tcp_gripper_mae"] = float(np.abs(selected).mean()) if len(selected) else None
            result["tcp_gripper_rmse"] = float(np.sqrt(np.square(selected).mean())) if len(selected) else None
        elif name in distribution_keys:
            for label, value in (
                ("mean", selected.mean() if len(selected) else None),
                ("rmse", np.sqrt(np.square(selected).mean()) if len(selected) else None),
                ("median", np.median(selected) if len(selected) else None),
                ("p95", np.percentile(selected, 95) if len(selected) else None),
            ):
                result[f"{name}_{label}"] = float(value) if value is not None else None
        else:
            result[name] = float(selected.mean()) if len(selected) else None
    return result


def camera_macro(metrics: list[dict]) -> dict:
    """Equal weight for each camera stream; counts are summed, undefined values omitted."""
    collected = defaultdict(list)
    for row in metrics:
        for key, value in row.items():
            collected[key]  # Preserve metrics that are undefined for every camera.
            if value is not None:
                collected[key].append(value)
    return {key: (int(sum(values)) if key.endswith("_count") else float(np.mean(values))) if values else None
            for key, values in collected.items()}
