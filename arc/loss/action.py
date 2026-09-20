"""Flow-matching objective and per-trajectory action metric sums."""
from __future__ import annotations

import math
import torch

from arc.action import safe_rotation_6d_to_matrix
from arc.rotation import so3_geodesic_angle


def flow_matching_loss(predictions, *, position_weight=1.0, rotation_weight=1.0, gripper_weight=1.0):
    valid = predictions.get("future_action_valid")
    velocity = predictions["action_velocity"].float()
    target_velocity = predictions["action_target_velocity"].float()
    if valid is None:
        valid = torch.ones(velocity.shape[:2], dtype=torch.bool, device=velocity.device)
    valid = valid.bool()
    error = (
        velocity.masked_fill(~valid[..., None], 0) - target_velocity.masked_fill(~valid[..., None], 0)
    ).square().unflatten(-1, (velocity.shape[-1] // 10, 10))
    def masked_mean(value):
        per_step = value.reshape(*valid.shape, -1).mean(-1)
        return (per_step.sum(1) / valid.sum(1)).mean()
    position = masked_mean(error[..., :3])
    rotation = masked_mean(error[..., 3:9])
    gripper = masked_mean(error[..., 9])
    objective = position_weight * position + rotation_weight * rotation + gripper_weight * gripper
    return {
        "objective": objective,
        "position": position.detach(), "rotation": rotation.detach(), "gripper": gripper.detach(),
    }


def action_metric_sums(prediction, target, future_valid=None):
    """Sums and counts, so final metrics do not depend on evaluation batch size."""
    if future_valid is None:
        future_valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    valid = prediction["success"].bool()
    count = valid.sum().float()
    p = prediction["action_position"][valid]
    rotation = prediction["action_rotation"][valid]
    gripper = prediction["action_gripper"][valid]
    gt = target[valid].float()
    mask = future_valid[valid].bool()[..., None]
    def trajectory_mean(value):
        return (value.masked_fill(~mask, 0).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * target.shape[-2])).sum()
    last = torch.where(future_valid[valid], torch.arange(target.shape[1], device=target.device), -1).amax(1)
    position_error = (p - gt[..., :3]).norm(dim=-1)
    angle = so3_geodesic_angle(rotation, safe_rotation_6d_to_matrix(gt[..., 3:9]))
    gt_open = gt[..., 9] >= 0
    pred_open = gripper == 1
    return {
        "count": count,
        "failures": (~valid).sum().float(),
        "position_ade_m": trajectory_mean(position_error),
        "position_fde_m": position_error[torch.arange(len(last), device=target.device), last].mean(dim=1).sum(),
        "rotation_deg": trajectory_mean(angle) * (180 / math.pi),
        "gripper_accuracy": trajectory_mean((pred_open == gt_open).float()),
        "gripper_mae": trajectory_mean((prediction.get("action_gripper_open", prediction["action_gripper"].float())[valid] - (gt[..., 9] + 1) / 2).abs()),
        "gripper_tp": (pred_open & gt_open & mask).sum().float(),
        "gripper_fp": (pred_open & ~gt_open & mask).sum().float(),
        "gripper_fn": (~pred_open & gt_open & mask).sum().float(),
    }


def finalize_action_metrics(sums):
    count = sums["count"]
    result = {
        key: (sums[key] / count if count else float("nan"))
        for key in ("position_ade_m", "position_fde_m", "rotation_deg", "gripper_accuracy", "gripper_mae")
    }
    denominator = 2 * sums["gripper_tp"] + sums["gripper_fp"] + sums["gripper_fn"]
    result["gripper_f1"] = 2 * sums["gripper_tp"] / denominator if denominator else 0.0
    result["valid_trajectories"] = count
    result["recovery_failure_rate"] = sums["failures"] / max(count + sums["failures"], 1)
    return result

