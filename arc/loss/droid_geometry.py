"""Metric robot-base rays and explicitly supervised absolute camera poses."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from arc.action import homogeneous
from arc.datasets.utils import compute_gt_ray_map, resize_ray_valid_mask
from arc.loss import GeometryLoss
from arc.models.arc.utils.transform import mat_to_quat, quat_to_mat
from arc.rotation import so3_geodesic_angle


def prepare_geometry_batch(batch, predictions, *, normalize=False):
    if normalize:
        raise ValueError("DROID absolute camera supervision requires metric geometry")
    result = dict(batch)
    result["extrinsics"] = homogeneous(batch["extrinsics"].float())
    result["depth"] = batch["depth"].float()
    result["scale"] = result["depth"].new_ones(result["depth"].shape[0])
    ray_height, ray_width = predictions["ray"].shape[-3:-1]
    height, width = batch["images"].shape[-2:]
    result["ray_map"] = compute_gt_ray_map(result["extrinsics"], batch["intrinsics"].float(),
                                            ray_height, ray_width, height, width)
    result["ray_valid_mask"] = resize_ray_valid_mask(batch["original_mask"], ray_height, ray_width)
    return result


def absolute_camera_loss(prediction, batch, *, translation_weight=1., rotation_weight=1., fov_weight=0.1):
    encoding = prediction["pose_enc"].float()
    target_w2c = homogeneous(batch["extrinsics"].float())
    target_c2w = torch.linalg.inv(target_w2c)
    target_q = mat_to_quat(target_c2w[..., :3, :3])
    q = encoding[..., 3:7]
    identity = torch.zeros_like(q)
    identity[..., 3] = 1
    q = F.normalize(torch.where(q.norm(dim=-1, keepdim=True) > 1e-8, q, identity), dim=-1)
    c2w_rotation = quat_to_mat(q)
    w2c_rotation = c2w_rotation.transpose(-1, -2)
    w2c_translation = -(w2c_rotation @ encoding[..., :3, None]).squeeze(-1)
    translation = F.smooth_l1_loss(w2c_translation, target_w2c[..., :3, 3])
    rotation = torch.minimum((q - target_q).square().sum(-1), (q + target_q).square().sum(-1)).mean()
    height, width = batch["images"].shape[-2:]
    k = batch["intrinsics"].float()
    target_fov = torch.stack((2 * torch.atan(height / (2 * k[..., 1, 1])),
                              2 * torch.atan(width / (2 * k[..., 0, 0]))), dim=-1)
    fov = F.smooth_l1_loss(encoding[..., 7:9], target_fov)
    return {
        "objective": translation_weight * translation + rotation_weight * rotation + fov_weight * fov,
        "loss_camera_translation": translation.detach(), "loss_camera_rotation": rotation.detach(),
        "loss_camera_fov": fov.detach(),
        "metric_camera_translation_m": (w2c_translation - target_w2c[..., :3, 3]).norm(dim=-1).mean().detach(),
        "metric_camera_rotation_deg": (so3_geodesic_angle(w2c_rotation, target_w2c[..., :3, :3]).mean() * 180 / math.pi).detach(),
    }


class DroidGeometryLoss(GeometryLoss):
    def __init__(self, *, camera_loss_weight=1., camera_translation_weight=1.,
                 camera_rotation_weight=1., camera_fov_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.camera_loss_weight = camera_loss_weight
        self.camera_options = dict(translation_weight=camera_translation_weight,
                                   rotation_weight=camera_rotation_weight, fov_weight=camera_fov_weight)

    def forward(self, prediction, batch):
        with torch.autocast(device_type=batch["images"].device.type, enabled=False):
            losses = super().forward(prediction, batch)
            camera = absolute_camera_loss(prediction, batch, **self.camera_options)
            losses["objective"] = losses["objective"] + self.camera_loss_weight * camera.pop("objective")
            losses.update(camera)
        return losses
