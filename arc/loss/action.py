"""Flow-matching objective and per-trajectory action metric sums."""
from __future__ import annotations

import math
import torch

from arc.action import safe_rotation_6d_to_matrix
from arc.rotation import so3_geodesic_angle


def flow_matching_loss(predictions, *, position_weight=1.0, rotation_weight=1.0, gripper_weight=1.0):
    error = (
        predictions["action_velocity"].float() - predictions["action_target_velocity"].float()
    ).square().unflatten(-1, (2, 10))
    position, rotation, gripper = error[..., :3].mean(), error[..., 3:9].mean(), error[..., 9].mean()
    objective = position_weight * position + rotation_weight * rotation + gripper_weight * gripper
    return {
        "objective": objective,
        "position": position.detach(), "rotation": rotation.detach(), "gripper": gripper.detach(),
    }


def action_metric_sums(prediction, target):
    """Sums and counts, so final metrics do not depend on evaluation batch size."""
    valid = prediction["success"].bool()
    count = valid.sum().float()
    p = prediction["action_position"][valid]
    rotation = prediction["action_rotation"][valid]
    gripper = prediction["action_gripper"][valid]
    gt = target[valid].float()
    position_error = (p - gt[..., :3]).norm(dim=-1)
    angle = so3_geodesic_angle(rotation, safe_rotation_6d_to_matrix(gt[..., 3:9]))
    gt_open = gt[..., 9] >= 0
    pred_open = gripper == 1
    return {
        "count": count,
        "failures": (~valid).sum().float(),
        "position_ade_m": position_error.mean(dim=(1, 2)).sum(),
        "position_fde_m": position_error[:, -1].mean(dim=1).sum(),
        "rotation_deg": (angle.mean(dim=(1, 2)) * (180 / math.pi)).sum(),
        "gripper_accuracy": (pred_open == gt_open).float().mean(dim=(1, 2)).sum(),
        "gripper_tp": (pred_open & gt_open).sum().float(),
        "gripper_fp": (pred_open & ~gt_open).sum().float(),
        "gripper_fn": (~pred_open & gt_open).sum().float(),
    }


def finalize_action_metrics(sums):
    count = sums["count"]
    result = {
        key: (sums[key] / count if count else float("nan"))
        for key in ("position_ade_m", "position_fde_m", "rotation_deg", "gripper_accuracy")
    }
    denominator = 2 * sums["gripper_tp"] + sums["gripper_fp"] + sums["gripper_fn"]
    result["gripper_f1"] = 2 * sums["gripper_tp"] / denominator if denominator else 0.0
    result["valid_trajectories"] = count
    result["recovery_failure_rate"] = sums["failures"] / max(count + sums["failures"], 1)
    return result

