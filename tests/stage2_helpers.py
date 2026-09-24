"""Small executable reconstruction model for stage-two integration tests."""
import torch
from torch import nn
from torch.nn import functional as F

from arc.action import safe_rotation_6d_to_matrix
from arc.models.arc.heads.tcp_head import TCPVisualQueryEncoder


class TinyLanguage(nn.Module):
    output_dim = 24

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 24)
        self.requires_grad_(False)

    def forward(self, instructions):
        ids = torch.tensor([
            [sum(text.encode()) % 64, len(text) % 64, 1] for text in instructions
        ], device=self.embedding.weight.device)
        return self.embedding(ids), torch.ones_like(ids, dtype=torch.bool)


class TinyTCPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 22)
        self.register_buffer("position_mean", torch.tensor([[-0.2, 0, 1.0], [0.2, 0, 1.0]]))
        self.register_buffer("position_std", torch.ones(2, 3) * 0.01)

    def forward(self, features):
        raw = self.linear(features).unflatten(-1, (2, 11))
        rotation = raw[..., 3:9] + raw.new_tensor([1, 0, 0, 0, 1, 0])
        return {
            "tcp_position": self.position_mean + self.position_std * raw[..., :3],
            "tcp_rotation": safe_rotation_6d_to_matrix(rotation),
            "tcp_gripper_logit": raw[..., 9],
            "tcp_confidence": 1 + raw[..., 10].sigmoid(),
        }


class TinyArc(nn.Module):
    def __init__(self, tcp_query_window_size=3):
        super().__init__()
        self.backbone = nn.Conv2d(3, 32, kernel_size=14, stride=14)
        self.head = nn.Conv2d(32, 7, 1)
        self.motion_decoder = nn.Linear(16, 16)
        self.tcp_visual_query_encoder = TCPVisualQueryEncoder(
            embed_dim=16, adapter_dim=8, window_size=tcp_query_window_size,
        )
        self.tcp_track_head = TinyTCPHead()
        self.calls = 0

    def forward(self, views, tcp_query_points, **kwargs):
        self.calls += 1
        images = torch.stack([view["img"] for view in views], dim=1)
        batch, frames, _, height, width = images.shape
        feature = self.backbone(images.flatten(0, 1))
        patches = feature.flatten(2).transpose(1, 2).unflatten(0, (batch, frames))
        sampled, _ = self.tcp_visual_query_encoder(
            patches[:, 0, :, 16:], tcp_query_points,
            image_height=height, image_width=width,
        )
        motion = self.motion_decoder(patches[..., 16:].mean(2) + sampled.mean(1)[:, None])
        output = self.tcp_track_head(motion)
        geometry = self.head(feature)
        depth = F.interpolate(F.softplus(geometry[:, :1]), size=(height, width)).reshape(batch, frames, height, width)
        ray = geometry[:, 1:].permute(0, 2, 3, 1).unflatten(0, (batch, frames))
        output.update(
            depth=depth, depth_conf=torch.ones_like(depth), ray=ray,
            ray_conf=torch.ones_like(ray[..., 0]),
            backbone_features=[(patches, patches.mean(2), patches.mean(2))] * 4,
        )
        return output


def tiny_batch(batch_size=1, frames=8, horizon=16):
    height = width = 42
    images = torch.randn(batch_size, frames, 3, height, width)
    intrinsics = torch.tensor([[30., 0, 21], [0, 30, 21], [0, 0, 1]]).expand(batch_size, frames, -1, -1).clone()
    centres = torch.tensor([[15., 21.], [27., 21.]]).expand(batch_size, frames, -1, -1).clone()
    state = torch.zeros(batch_size, frames, 2, 7)
    state[..., :3] = torch.tensor([[-0.2, 0, 1.0], [0.2, 0, 1.0]])
    future = torch.zeros(batch_size, horizon, 2, 10)
    future[..., :3] = state[:, -1:, :, :3]
    future[..., 3:9] = torch.tensor([1, 0, 0, 0, 1, 0])
    future[..., 9] = -1
    return {
        "images": images, "intrinsics": intrinsics, "tcp_query_points": centres[:, 0],
        "history_tcp_query_points": centres,
        "history_tcp_valid": torch.ones(batch_size, frames, 2, dtype=torch.bool),
        "frame_times": torch.arange(frames).expand(batch_size, -1) / 15,
        "future_step_indices": torch.arange(1, horizon + 1).expand(batch_size, -1),
        "future_actions": future, "instruction": ["lift the cup"] * batch_size,
        "shuffled_instruction": ["open the drawer"] * batch_size,
        "depth": torch.ones(batch_size, frames, height, width),
        "valid_mask": torch.ones(batch_size, frames, height, width, dtype=torch.bool),
        "original_mask": torch.ones(batch_size, frames, height, width, dtype=torch.bool),
        "extrinsics": torch.eye(4)[:3].expand(batch_size, frames, -1, -1).clone(),
        "tcp_state": state, "tcp_query_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        "padding": torch.tensor([1, 1, 6, 6]).expand(batch_size, -1),
        "source_size": torch.tensor([30, 40]).expand(batch_size, -1),
    }

