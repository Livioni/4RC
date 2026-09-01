"""Differentiable rotation conversions used by TCP tracking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert fixed-axis XYZ roll/pitch/yaw angles to rotation matrices.

    RoboTwin stores TCP orientation as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """
    if rpy.shape[-1] != 3:
        raise ValueError(f"Expected rpy [...,3], got {tuple(rpy.shape)}")
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()

    return torch.stack(
        (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        ),
        dim=-1,
    ).reshape(*rpy.shape[:-1], 3, 3)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert the continuous two-column 6D representation to SO(3)."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f"Expected 6D rotation [...,6], got {tuple(rotation_6d.shape)}"
        )
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(first, dim=-1, eps=1e-6)
    projection = (b1 * second).sum(-1, keepdim=True) * b1
    b2 = F.normalize(second - projection, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def so3_geodesic_angle(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return a numerically stable geodesic rotation error in radians."""
    if prediction.shape[-2:] != (3, 3) or target.shape[-2:] != (3, 3):
        raise ValueError("SO(3) inputs must end in [3,3]")
    relative = prediction.transpose(-1, -2) @ target
    cosine = (
        (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sine, cosine)
