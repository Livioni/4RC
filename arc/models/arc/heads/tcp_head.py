"""Image-conditioned sparse TCP query and trajectory prediction heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from arc.rotation import rotation_6d_to_matrix


class TCPVisualQueryEncoder(nn.Module):
    """Sample local first-frame patch features around two TCP image points."""

    NUM_ARMS = 2

    def __init__(
        self,
        embed_dim: int = 1536,
        patch_size: int = 14,
        window_size: int = 3,
        adapter_dim: int = 256,
    ) -> None:
        super().__init__()
        if window_size < 1 or window_size % 2 != 1:
            raise ValueError("window_size must be a positive odd integer")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.window_tokens = window_size * window_size
        self.norm = nn.LayerNorm(embed_dim)
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, embed_dim),
        )
        self.arm_embedding = nn.Parameter(torch.zeros(self.NUM_ARMS, embed_dim))
        self.offset_embedding = nn.Parameter(
            torch.zeros(self.window_tokens, embed_dim)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.normal_(self.arm_embedding, std=0.02)
        nn.init.normal_(self.offset_embedding, std=0.02)

    def _offsets(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        radius = self.window_size // 2
        values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        offset_y, offset_x = torch.meshgrid(values, values, indexing="ij")
        return torch.stack((offset_x, offset_y), dim=-1).reshape(-1, 2)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        query_points: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = patch_tokens.shape
        grid_height = image_height // self.patch_size
        grid_width = image_width // self.patch_size

        points = query_points.to(device=patch_tokens.device, dtype=torch.float32)
        center_x = points[..., 0] / self.patch_size - 0.5
        center_y = points[..., 1] / self.patch_size - 0.5
        centers = torch.stack((center_x, center_y), dim=-1)
        offsets = self._offsets(device=points.device, dtype=points.dtype)
        sample_xy = centers[:, :, None, :] + offsets[None, None, :, :]
        sample_x = sample_xy[..., 0].clamp(0, grid_width - 1)
        sample_y = sample_xy[..., 1].clamp(0, grid_height - 1)
        if grid_width > 1:
            normalized_x = sample_x.mul(2.0 / (grid_width - 1)).sub(1.0)
        else:
            normalized_x = torch.zeros_like(sample_x)
        if grid_height > 1:
            normalized_y = sample_y.mul(2.0 / (grid_height - 1)).sub(1.0)
        else:
            normalized_y = torch.zeros_like(sample_y)
        sampling_grid = torch.stack((normalized_x, normalized_y), dim=-1)
        sampling_grid = sampling_grid.reshape(
            batch, self.NUM_ARMS * self.window_tokens, 1, 2
        )

        feature_map = patch_tokens.transpose(1, 2).reshape(
            batch, channels, grid_height, grid_width
        )
        sampled = F.grid_sample(
            feature_map.float(),
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
            batch, self.NUM_ARMS, self.window_tokens, channels
        )
        sampled = sampled.to(dtype=patch_tokens.dtype)
        sampled = sampled + self.adapter(self.norm(sampled))
        sampled = sampled + self.arm_embedding.to(sampled)[None, :, None, :]
        sampled = sampled + self.offset_embedding.to(sampled)[None, None, :, :]

        base_x = torch.floor(points[..., 0] / self.patch_size)
        base_y = torch.floor(points[..., 1] / self.patch_size)
        base_yx = torch.stack((base_y, base_x), dim=-1)
        offset_yx = offsets.flip(-1)
        positions = base_yx[:, :, None, :] + offset_yx[None, None, :, :]
        positions[..., 0].clamp_(0, grid_height - 1)
        positions[..., 1].clamp_(0, grid_width - 1)
        return (
            sampled.reshape(batch, self.NUM_ARMS * self.window_tokens, channels),
            positions.to(dtype=torch.long).reshape(
                batch, self.NUM_ARMS * self.window_tokens, 2
            ),
        )


class TCPTrackHead(nn.Module):
    """Pool local motion tokens and regress absolute dual-arm TCP trajectories."""

    NUM_ARMS = 2

    def __init__(
        self,
        embed_dim: int = 1536,
        hidden_dim: int = 512,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        if window_size < 1 or window_size % 2 != 1:
            raise ValueError("window_size must be a positive odd integer")
        self.window_size = window_size
        self.window_tokens = window_size * window_size
        self.level_logits = nn.Parameter(torch.zeros(4))
        self.window_norm = nn.LayerNorm(embed_dim)
        self.window_score = nn.Linear(embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 11),
        )
        self.register_buffer("position_mean", torch.zeros(self.NUM_ARMS, 3))
        self.register_buffer("position_std", torch.ones(self.NUM_ARMS, 3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.window_score.weight)
        nn.init.zeros_(self.window_score.bias)
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[3:9].copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )

    def set_position_stats(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        if mean.shape != (self.NUM_ARMS, 3) or std.shape != (self.NUM_ARMS, 3):
            raise ValueError("TCP position statistics must both have shape [2,3]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("TCP position statistics must be finite")
        if torch.any(std <= 0):
            raise ValueError("TCP position standard deviations must be positive")
        self.position_mean.copy_(mean.to(self.position_mean))
        self.position_std.copy_(std.to(self.position_std).clamp_min(1e-3))

    def forward(self, levels: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pooled_levels = []
        for level in levels:
            queries = level[:, :, 1:].reshape(
                *level.shape[:2], self.NUM_ARMS, self.window_tokens, level.shape[-1]
            )
            scores = self.window_score(self.window_norm(queries)).squeeze(-1)
            weights = scores.softmax(dim=-1)
            pooled_levels.append((queries * weights[..., None]).sum(dim=-2))

        level_weights = self.level_logits.softmax(dim=0)
        fused = sum(
            weight * value for weight, value in zip(level_weights, pooled_levels)
        )
        raw = self.mlp(self.norm(fused))

        with torch.autocast(device_type=raw.device.type, enabled=False):
            raw_float = raw.float()
            mean = self.position_mean.float()[None, None]
            std = self.position_std.float()[None, None]
            position = mean + std * raw_float[..., :3]
            rotation = rotation_6d_to_matrix(raw_float[..., 3:9])
            gripper_logit = raw_float[..., 9]
            confidence = 1.0 + torch.exp(raw_float[..., 10].clamp(max=10.0))
        return {
            "tcp_position": position,
            "tcp_rotation": rotation,
            "tcp_gripper_logit": gripper_logit,
            "tcp_confidence": confidence,
        }
