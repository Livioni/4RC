"""Ground-truth geometry construction for 4RC training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected (..., 3, 4) or (..., 4, 4), got {extrinsics.shape}")
    bottom = torch.zeros(*extrinsics.shape[:-2], 1, 4, dtype=extrinsics.dtype, device=extrinsics.device)
    bottom[..., 0, 3] = 1
    return torch.cat((extrinsics, bottom), dim=-2)


def normalize_geometry_to_first_camera(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each clip to its first camera and unit mean point distance.

    Returns normalized depth, normalized relative w2c extrinsics, and the
    per-clip scale. Depth is OpenCV z-depth, not Euclidean ray distance.
    """
    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,S,H,W], got {depth.shape}")
    batch, sequence, height, width = depth.shape
    if valid_mask.shape != depth.shape:
        raise ValueError("valid_mask must have the same shape as depth")

    w2c = _homogeneous(extrinsics_w2c)
    first_c2w = torch.linalg.inv(w2c[:, 0])
    relative_w2c = w2c @ first_c2w[:, None]
    relative_c2w = torch.linalg.inv(relative_w2c)

    y, x = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype, device=depth.device),
        torch.arange(width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    fx = intrinsics[..., 0, 0, None, None]
    fy = intrinsics[..., 1, 1, None, None]
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    camera_points = torch.stack(
        (
            (x - cx) * depth / fx,
            (y - cy) * depth / fy,
            depth,
        ),
        dim=-1,
    )
    world_points = torch.einsum(
        "bsij,bshwj->bshwi", relative_c2w[..., :3, :3], camera_points
    )
    world_points = world_points + relative_c2w[..., None, None, :3, 3]
    distances = torch.linalg.vector_norm(world_points, dim=-1)
    valid = valid_mask & torch.isfinite(distances)
    counts = valid.flatten(1).sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("A clip contains no valid depth values")
    scale = (distances * valid).flatten(1).sum(dim=1) / counts.to(depth.dtype)
    scale = scale.clamp_min(eps)

    normalized_depth = depth / scale[:, None, None, None]
    normalized_w2c = relative_w2c.clone()
    normalized_w2c[..., :3, 3] /= scale[:, None, None]
    return normalized_depth, normalized_w2c[..., :3, :], scale


def geometry_to_first_camera(
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
    *,
    normalize: bool = False,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express geometry in the first camera frame, preserving metric scale by default.

    When ``normalize=True``, depth and camera translations are additionally
    divided by the per-clip mean point distance. The returned scale is the
    applied divisor, so it is one for metric (non-normalized) geometry.
    """
    if normalize:
        return normalize_geometry_to_first_camera(
            depth,
            valid_mask,
            intrinsics,
            extrinsics_w2c,
            eps=eps,
        )

    if depth.ndim != 4:
        raise ValueError(f"Expected depth [B,S,H,W], got {depth.shape}")
    if valid_mask.shape != depth.shape:
        raise ValueError("valid_mask must have the same shape as depth")

    w2c = _homogeneous(extrinsics_w2c)
    first_c2w = torch.linalg.inv(w2c[:, 0])
    relative_w2c = w2c @ first_c2w[:, None]
    scale = torch.ones(depth.shape[0], dtype=depth.dtype, device=depth.device)
    return depth, relative_w2c[..., :3, :], scale


def compute_gt_ray_map(
    extrinsics_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    ray_height: int,
    ray_width: int,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Build the DA3 camera-ray representation: [world direction, origin]."""
    c2w = torch.linalg.inv(_homogeneous(extrinsics_w2c))
    dtype, device = c2w.dtype, c2w.device
    dx = 1.0 / ray_width
    dy = 1.0 / ray_height
    x = torch.linspace(dx, 1.0 - dx, ray_width, dtype=dtype, device=device)
    y = torch.linspace(dy, 1.0 - dy, ray_height, dtype=dtype, device=device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    fx = intrinsics[..., 0, 0] / image_width
    fy = intrinsics[..., 1, 1] / image_height
    cx = intrinsics[..., 0, 2] / image_width
    cy = intrinsics[..., 1, 2] / image_height
    x_norm = (x_grid[None, None] - cx[..., None, None]) / fx[..., None, None]
    y_norm = (y_grid[None, None] - cy[..., None, None]) / fy[..., None, None]
    canonical = torch.stack((x_norm, y_norm, torch.ones_like(x_norm)), dim=-1)
    directions = torch.einsum("bsij,bshwj->bshwi", c2w[..., :3, :3], canonical)
    origins = c2w[..., None, None, :3, 3].expand_as(directions)
    return torch.cat((directions, origins), dim=-1)


def resize_ray_valid_mask(original_mask: torch.Tensor, ray_height: int, ray_width: int) -> torch.Tensor:
    """Map the unpadded image region to the native ray-head resolution."""
    batch, sequence, height, width = original_mask.shape
    resized = F.interpolate(
        original_mask.reshape(batch * sequence, 1, height, width).float(),
        size=(ray_height, ray_width),
        mode="nearest",
    )
    return resized.reshape(batch, sequence, ray_height, ray_width) > 0.5
