"""Frozen, shared DINOv2 encoding of the current left/right wrist RGBs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FrozenDINOv2Encoder(nn.Module):
    patch_size = 14
    image_height = 196
    image_width = 266

    def __init__(
        self, model_name="facebook/dinov2-giant", *, pretrained=True,
        model_config=None, cache_dir=None, encoder=None,
    ):
        super().__init__()
        if encoder is None:
            from transformers import Dinov2Config, Dinov2Model
            if pretrained:
                encoder = Dinov2Model.from_pretrained(model_name, cache_dir=cache_dir)
            else:
                if model_config is None:
                    raise ValueError("A saved wrist_encoder_config is required to restore wrist weights")
                encoder = Dinov2Model(Dinov2Config.from_dict(model_config))
        if encoder.config.patch_size != self.patch_size:
            raise ValueError("The wrist encoder requires DINOv2 with patch size 14")
        self.encoder = encoder
        self.output_dim = encoder.config.hidden_size
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None], persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 5 or tuple(images.shape[1:]) != (2, 3, 240, 320):
            raise ValueError("wrist_images must be [B,2,3,240,320], ordered left/right")
        if (
            not images.is_floating_point() or not torch.isfinite(images).all()
            or (images < 0).any() or (images > 1).any()
        ):
            raise ValueError("wrist_images must be finite RGB floats in [0,1]")
        batch = images.shape[0]
        pixels = F.interpolate(
            images.flatten(0, 1).float(), size=(192, 256), mode="bicubic",
            align_corners=False, antialias=True,
        )
        pixels = F.pad(pixels, (5, 5, 2, 2), mode="reflect")
        pixels = (pixels - self.image_mean) / self.image_std
        # Dinov2Model returns final LayerNorm features with a leading CLS token.
        patches = self.encoder(pixel_values=pixels).last_hidden_state[:, 1:]
        count = (self.image_height // self.patch_size) * (self.image_width // self.patch_size)
        if patches.shape[1:] != (count, self.output_dim):
            raise ValueError("Unexpected DINOv2 wrist patch shape")
        return patches.reshape(batch, 2, count, self.output_dim)
