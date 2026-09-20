"""Small CPU backbones exercising the production TCP/camera/action heads."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from arc.models.arc.arc import Arc
from arc.models.arc.heads.tcp_head import TCPTrackHead, TCPVisualQueryEncoder
from arc.models.arc.heads.cam_dec import CameraDec
from arc.models.arc.heads.motiondecoder import MotionDecoder


class SmallArc(nn.Module):
    configure_trainable_modules = Arc.configure_trainable_modules

    def __init__(self, tcp_query_window_size=3, num_arms=1):
        super().__init__()
        self.num_arms = num_arms
        self.backbone = nn.Conv2d(3, 32, 14, stride=14)
        self.head = nn.Conv2d(32, 7, 1)
        self.cam_dec = CameraDec(dim_in=32)
        self.motion_decoder = MotionDecoder(embed_dim=16, depth=1, num_heads=4, use_adaln=True)
        self.track_head = nn.Linear(16, 4)
        self.tcp_visual_query_encoder = TCPVisualQueryEncoder(embed_dim=16, adapter_dim=8, num_arms=num_arms)
        self.tcp_track_head = TCPTrackHead(embed_dim=16, hidden_dim=16, num_arms=num_arms)
        with torch.no_grad():
            self.cam_dec.fc_qvec.bias.copy_(torch.tensor([0., 0, 0, 1.]))
            self.cam_dec.fc_fov[0].bias.fill_(1.)

    def set_tcp_position_stats(self, mean, std):
        self.tcp_track_head.set_position_stats(mean, std)

    def forward(self, views, tcp_query_points, decode_camera=True, return_backbone_features=False, **kwargs):
        images = torch.stack([view["img"] for view in views], dim=1)
        batch, frames, _, height, width = images.shape
        feature = self.backbone(images.flatten(0, 1))
        patches = feature.flatten(2).transpose(1, 2).unflatten(0, (batch, frames))
        pooled = patches.mean(2)
        query, positions = self.tcp_visual_query_encoder(patches[:, 0, :, 16:], tcp_query_points,
                                                        image_height=height, image_width=width)
        tokens = torch.cat((pooled[..., None, :], pooled[..., None, :], patches), dim=2)[..., 16:]
        level = self.motion_decoder(tokens, images=images, patch_start_idx=2,
                                    query_tokens=query, query_positions=positions)
        result = self.tcp_track_head([level] * 4)
        geometry = self.head(feature)
        depth = F.interpolate(F.softplus(geometry[:, :1]), size=(height, width)).reshape(batch, frames, height, width)
        ray = geometry[:, 1:].permute(0, 2, 3, 1).unflatten(0, (batch, frames))
        result.update(depth=depth, depth_conf=torch.ones_like(depth), ray=ray,
                      ray_conf=torch.ones_like(ray[..., 0]))
        if decode_camera:
            result["pose_enc"] = self.cam_dec(pooled)
        if return_backbone_features:
            result["backbone_features"] = [(patches, pooled, pooled)] * 4
        return result


class SmallLanguage(nn.Module):
    output_dim = 24

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(64, 24)
        self.requires_grad_(False)

    def forward(self, instructions):
        ids = torch.tensor([[sum(text.encode()) % 64, len(text) % 64] for text in instructions],
                           device=self.embedding.weight.device)
        return self.embedding(ids), torch.ones_like(ids, dtype=torch.bool)


def write_episode(root, name, frames=24):
    episode = Path(root) / name
    episode.mkdir(parents=True)
    (episode / "metadata.json").write_text(json.dumps({"frame_count": frames,
        "language_instruction": [f"move {name}"], "language_instruction_2": [f"please move {name}"]}))
    cameras = ("cam_a", "cam_b")
    (episode / "depths").mkdir()
    (episode / "depths" / "metadata.json").write_text(json.dumps({"units": "millimeters", "cameras": {
        cam: {"frames": frames, "height": 180, "width": 320} for cam in cameras}}))
    for j, cam in enumerate(cameras):
        for folder in ("images", "depths", "TCP"):
            (episode / folder / cam).mkdir(parents=True, exist_ok=True)
        for folder in ("intrinsic", "extrinsic"):
            (episode / folder).mkdir(exist_ok=True)
        k = np.array([[140., 0, 160], [0, 140, 90], [0, 0, 1]], dtype=np.float32)
        angle = .3 + j * .4
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        ext[:3, 3] = [0.1 + j * .05, -.05, .2]
        state = np.zeros((frames, 7), dtype=np.float32)
        xyz = np.stack((np.arange(frames) * .001, np.zeros(frames), np.ones(frames)), axis=-1)
        state[:, :3] = xyz @ ext[:3, :3].T + ext[:3, 3]
        state[:, 5] = angle
        state[:, 6] = np.linspace(0, 1, frames)
        np.save(episode / "intrinsic" / f"{cam}.npy", k)
        np.save(episode / "extrinsic" / f"{cam}.npy", ext)
        np.save(episode / "TCP" / cam / "state.npy", state)
        (episode / "TCP" / cam / "metadata.json").write_text(json.dumps({
            "position_unit": "meter", "rotation_unit": "radian", "camera_id": cam,
            "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
            "coordinate_frame": "OpenCV camera (+x right, +y down, +z forward)",
            "extrinsics": "world_to_camera, applied directly to world TCP pose"}))
        for frame in range(frames):
            Image.fromarray(np.full((180, 320, 3), (frame * 5 + j * 80) % 255, dtype=np.uint8)).save(
                episode / "images" / cam / f"{frame:06d}.png")
            Image.fromarray(np.full((180, 320), 1200, dtype=np.uint16)).save(
                episode / "depths" / cam / f"{frame:06d}.png")
    return episode
