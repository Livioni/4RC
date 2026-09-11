"""Action geometry shared by the dataset, policy and offline evaluation."""
from __future__ import annotations

import math
import torch
from arc.rotation import rpy_to_matrix


def homogeneous(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] == (4, 4):
        return matrix
    if matrix.shape[-2:] != (3, 4):
        raise ValueError("Expected extrinsics [...,3,4] or [...,4,4]")
    row = matrix.new_zeros(*matrix.shape[:-2], 1, 4)
    row[..., 0, 3] = 1
    return torch.cat((matrix, row), dim=-2)


def future_actions_in_current_camera(
    state: torch.Tensor,
    future_w2c: torch.Tensor,
    current_w2c: torch.Tensor,
) -> torch.Tensor:
    """Convert [N,2,7] measured TCP states into [N,2,10] action targets."""
    state = state.float()
    current_w2c = current_w2c.to(device=state.device, dtype=torch.float32)
    future_w2c = future_w2c.to(device=state.device, dtype=torch.float32)
    transform = homogeneous(current_w2c) @ torch.linalg.inv(homogeneous(future_w2c))
    rotation, translation = transform[..., :3, :3], transform[..., :3, 3]
    position = torch.einsum("nij,naj->nai", rotation, state[..., :3]) + translation[:, None]
    orientation = rotation[:, None] @ rpy_to_matrix(state[..., 3:6])
    rotation6d = torch.cat((orientation[..., :, 0], orientation[..., :, 1]), dim=-1)
    return torch.cat((position, rotation6d, 2 * state[..., 6:7] - 1), dim=-1)


def project_tcp(
    positions: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    padding: tuple[int, int, int, int] = (1, 1, 6, 6),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project [...,2,3] TCPs using matching [...,3,3] padded-image intrinsics.

    Invalid centres are masked, never clamped onto visible image borders.
    """
    xyz = positions.float()
    pixels = torch.einsum("...ij,...aj->...ai", intrinsics.float(), xyz)
    denom = pixels[..., 2]
    valid = (
        torch.isfinite(xyz).all(-1)
        & torch.isfinite(pixels).all(-1)
        & (xyz[..., 2] > 1e-6)
        & (denom.abs() > 1e-6)
    )
    uv = pixels[..., :2] / torch.where(valid, denom, torch.ones_like(denom))[..., None]
    left, right, top, bottom = padding
    valid &= (
        (uv[..., 0] >= left) & (uv[..., 0] < image_width - right)
        & (uv[..., 1] >= top) & (uv[..., 1] < image_height - bottom)
    )
    return torch.where(valid[..., None], uv, torch.zeros_like(uv)), valid


def safe_rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Two-column SO(3) projection with a deterministic degenerate fallback."""
    import torch.nn.functional as F
    first, second = value.float()[..., :3], value.float()[..., 3:]
    fallback = torch.zeros_like(first)
    fallback[..., 0] = 1
    first = torch.where(first.norm(dim=-1, keepdim=True) > 1e-6, first, fallback)
    b1 = F.normalize(first, dim=-1)
    second = second - (second * b1).sum(-1, keepdim=True) * b1
    axis = F.one_hot(b1.abs().argmin(-1), num_classes=3).to(b1)
    fallback = axis - (axis * b1).sum(-1, keepdim=True) * b1
    second = torch.where(second.norm(dim=-1, keepdim=True) > 1e-6, second, fallback)
    b2 = F.normalize(second, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-1)


def sinusoidal(value: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 2 or dim % 2:
        raise ValueError("Sinusoidal dimension must be positive and even")
    freq = torch.exp(
        -math.log(10000) * torch.arange(dim // 2, device=value.device, dtype=torch.float32)
        / (dim // 2)
    )
    phase = value.float()[..., None] * freq
    return torch.cat((phase.sin(), phase.cos()), dim=-1)

