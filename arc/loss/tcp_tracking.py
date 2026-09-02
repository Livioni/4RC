"""4RC-style uncertainty and temporal objective for two TCP trajectories."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from arc.rotation import rpy_to_matrix, so3_geodesic_angle


class TCPTrackingLoss(nn.Module):
    def __init__(
        self,
        *,
        point_scale: float = 0.1,
        virtual_point_radius: float = 0.03,
        rotation_weight: float = 0.5,
        temporal_weight: float = 0.2,
        gripper_weight: float = 0.2,
        velocity_scale: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 0.2,
        confidence_max: float = 20.0,
    ) -> None:
        super().__init__()
        offsets = torch.cat(
            (
                torch.zeros(1, 3),
                torch.eye(3) * virtual_point_radius,
                -torch.eye(3) * virtual_point_radius,
            ),
            dim=0,
        )
        self.register_buffer("virtual_offsets", offsets, persistent=False)
        self.point_scale = point_scale
        self.rotation_weight = rotation_weight
        self.temporal_weight = temporal_weight
        self.gripper_weight = gripper_weight
        self.velocity_scale = velocity_scale
        self.gamma = gamma
        self.alpha = alpha
        self.confidence_max = confidence_max

    def _rigid_points(
        self, position: torch.Tensor, rotation: torch.Tensor
    ) -> torch.Tensor:
        offsets = self.virtual_offsets.to(dtype=position.dtype)
        rotated = torch.einsum("...ij,kj->...ki", rotation, offsets)
        return position.unsqueeze(-2) + rotated

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = torch.where(mask, values, torch.zeros_like(values))
        return masked.sum() / mask.sum().clamp_min(1)

    def forward(self, predictions: dict, batch: dict) -> dict[str, torch.Tensor]:
        target_state = batch["tcp_state"].float()
        frame_times = batch["frame_times"].float()
        target_position = target_state[..., :3]
        target_rotation = rpy_to_matrix(target_state[..., 3:6])
        target_gripper = target_state[..., 6].clamp(0.0, 1.0)

        pred_position = predictions["tcp_position"].float()
        pred_rotation = predictions["tcp_rotation"].float()
        query_mask = batch["tcp_query_valid"].bool()
        trajectory_mask = query_mask[:, None].expand_as(target_gripper)

        confidence = predictions["tcp_confidence"].float().clamp(
            min=1e-6, max=self.confidence_max
        )
        pred_points = self._rigid_points(pred_position, pred_rotation)
        target_points = self._rigid_points(target_position, target_rotation)

        query_target_points = target_points[:, :1]
        pred_displacement = pred_points - query_target_points
        target_displacement = target_points - query_target_points
        point_error = F.smooth_l1_loss(
            pred_displacement / self.point_scale,
            target_displacement / self.point_scale,
            reduction="none",
        ).mean(dim=(-1, -2))
        rotation_error = so3_geodesic_angle(pred_rotation, target_rotation) / math.pi
        pose_error = point_error + self.rotation_weight * rotation_error
        pose_terms = (
            self.gamma * pose_error * confidence - self.alpha * confidence.log()
        )
        pose_loss = self._masked_mean(pose_terms, trajectory_mask)

        if target_state.shape[1] > 1:
            dt = frame_times[:, 1:] - frame_times[:, :-1]
            direction = torch.where(dt < 0, -torch.ones_like(dt), torch.ones_like(dt))
            dt = direction * dt.abs().clamp_min(1e-6)
            pred_velocity = (
                pred_points[:, 1:] - pred_points[:, :-1]
            ) / dt[..., None, None, None]
            target_velocity = (
                target_points[:, 1:] - target_points[:, :-1]
            ) / dt[..., None, None, None]
            velocity_error = F.smooth_l1_loss(
                pred_velocity / self.velocity_scale,
                target_velocity / self.velocity_scale,
                reduction="none",
            ).mean(dim=(-1, -2))
            pair_confidence = torch.minimum(confidence[:, 1:], confidence[:, :-1])
            temporal_terms = (
                self.gamma * velocity_error * pair_confidence
                - self.alpha * pair_confidence.log()
            )
            temporal_mask = query_mask[:, None].expand_as(velocity_error)
            temporal_loss = self._masked_mean(temporal_terms, temporal_mask)
        else:
            temporal_loss = pred_position.sum() * 0.0

        valid_gripper = target_gripper[trajectory_mask]
        positive = valid_gripper.sum()
        negative = valid_gripper.new_tensor(valid_gripper.numel()) - positive
        pos_weight = torch.where(
            (positive > 0) & (negative > 0),
            negative / positive.clamp_min(1.0),
            torch.ones_like(positive),
        )
        gripper_terms = F.binary_cross_entropy_with_logits(
            predictions["tcp_gripper_logit"].float(),
            target_gripper,
            pos_weight=pos_weight,
            reduction="none",
        )
        gripper_loss = self._masked_mean(gripper_terms, trajectory_mask)
        objective = (
            pose_loss
            + self.temporal_weight * temporal_loss
            + self.gripper_weight * gripper_loss
        )

        position_error = torch.linalg.vector_norm(
            pred_position - target_position, dim=-1
        )
        position_metric = self._masked_mean(position_error, trajectory_mask)
        rotation_metric = self._masked_mean(
            so3_geodesic_angle(pred_rotation, target_rotation), trajectory_mask
        )
        gripper_accuracy = (
            (predictions["tcp_gripper_logit"] >= 0)
            == (target_gripper >= 0.5)
        ).float()
        gripper_metric = self._masked_mean(gripper_accuracy, trajectory_mask)
        return {
            "objective": objective,
            "loss_tcp_pose": pose_loss.detach(),
            "loss_tcp_temporal": temporal_loss.detach(),
            "loss_tcp_gripper": gripper_loss.detach(),
            "metric_tcp_position_m": position_metric.detach(),
            "metric_tcp_rotation_deg": (rotation_metric * (180.0 / math.pi)).detach(),
            "metric_tcp_gripper_accuracy": gripper_metric.detach(),
        }
