"""Anchor-frame wrist RGB loading shared by training and window inference."""
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms import functional as TVF


def load_wrist_images(episode_path, frame_index, *, transform=None):
    images = []
    for view in ("left_view", "right_view"):
        path = Path(episode_path) / "images" / view / f"{int(frame_index):06d}.png"
        with Image.open(path) as source:
            image = source.convert("RGB")
            if image.size != (320, 240):
                raise ValueError(f"Expected 320x240 wrist RGB, got {image.size}: {path}")
            if transform is not None:
                image = transform(image)
            images.append(TVF.pil_to_tensor(image).float().div_(255))
    return torch.stack(images)
