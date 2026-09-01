"""Sparse TCP query encoding and trajectory prediction heads."""

from __future__ import annotations

import torch
from torch import nn

from arc.models.arc.heads.head_act import inverse_log_transform
from arc.rotation import rotation_6d_to_matrix, rpy_to_matrix


class TCPQueryEncoder(nn.Module):
    """Encode two first-frame TCP states without assuming visible pixels."""

    def __init__(self, embed_dim: int = 1536, hidden_dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.arm_embedding = nn.Parameter(torch.zeros(2, embed_dim))
        nn.init.normal_(self.arm_embedding, std=0.02)

    def forward(self, tcp_state: torch.Tensor) -> torch.Tensor:
        if tcp_state.ndim != 3 or tcp_state.shape[-2:] != (2, 7):
            raise ValueError(
                f"Expected first-frame TCP state [B,2,7], got {tuple(tcp_state.shape)}"
            )
        xyz = tcp_state[..., :3]
        rpy = tcp_state[..., 3:6]
        gripper = tcp_state[..., 6:7]
        features = torch.cat((xyz, rpy.sin(), rpy.cos(), gripper), dim=-1)
        return self.mlp(features) + self.arm_embedding.unsqueeze(0)


class TCPTrackHead(nn.Module):
    """Fuse four 4RC motion levels and regress two rigid TCP trajectories."""

    def __init__(self, embed_dim: int = 1536, hidden_dim: int = 512) -> None:
        super().__init__()
        self.level_logits = nn.Parameter(torch.zeros(4))
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 11),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[3:9].copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )

    def forward(
        self,
        levels: list[torch.Tensor],
        query_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(levels) != 4:
            raise ValueError(f"Expected four motion levels, got {len(levels)}")
        if query_state.ndim != 3 or query_state.shape[-2:] != (2, 7):
            raise ValueError(f"Expected query_state [B,2,7], got {query_state.shape}")
        tcp_levels = []
        for level in levels:
            if level.ndim != 4 or level.shape[2] != 3:
                raise ValueError(
                    "Sparse motion levels must be [B,S,3,C] (time,left,right)"
                )
            tcp_levels.append(level[:, :, 1:])
        weights = self.level_logits.softmax(dim=0)
        fused = sum(weight * value for weight, value in zip(weights, tcp_levels))
        raw = self.mlp(self.norm(fused))

        delta_position = inverse_log_transform(raw[..., :3])
        relative_rotation = rotation_6d_to_matrix(raw[..., 3:9])
        query_state = query_state.to(device=raw.device, dtype=raw.dtype)
        query_position = query_state[:, None, :, :3]
        query_rotation = rpy_to_matrix(query_state[:, None, :, 3:6])
        return {
            "tcp_delta_position": delta_position,
            "tcp_position": query_position + delta_position,
            "tcp_relative_rotation": relative_rotation,
            "tcp_rotation": relative_rotation @ query_rotation,
            "tcp_gripper_logit": raw[..., 9],
            "tcp_confidence": 1.0 + torch.exp(raw[..., 10].clamp(max=10.0)),
        }
