"""Frame-aligned dual-arm real-robot clips for place_pen_in_cups."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF

from .robotwin import RoboTwin4RC


@dataclass(frozen=True)
class PenEpisode:
    task: str
    name: str
    path: Path
    num_frames: int  # Number of decimated frames.
    frame_rate: float
    valid_segments: tuple[tuple[int, int], ...]
    invalid_transition_count: int

    @property
    def max_valid_segment_frames(self):
        return max(end - start for start, end in self.valid_segments)


class PenCups4RC(RoboTwin4RC):
    """Use decimated frame indices internally; map to source rows on IO."""

    def _segments_for(self, episode_or_frames):
        if isinstance(episode_or_frames, PenEpisode):
            return episode_or_frames.num_frames, episode_or_frames.valid_segments
        return super()._segments_for(episode_or_frames)

    def __init__(self, root, *, view="third_person", min_views=2, max_views=18,
                 min_interval=1, max_interval=5, reverse_probability=0.5,
                 frame_rate=30.0, max_depth=3.0, max_tcp_linear_speed=3.0,
                 max_tcp_angular_speed=4 * math.pi, seed=42, augment=True,
                 color_augment=True, max_episodes=None, tasks=None, frame_stride=1,
                 depth_subdir="depths"):
        if view != "third_person" or frame_stride < 1 or not isinstance(frame_stride, int):
            raise ValueError("PenCups requires third_person and a positive integer frame_stride")
        if depth_subdir not in ("depths", "depths_completed"):
            raise ValueError("depth_subdir must be depths or depths_completed")
        if not 1 <= min_views <= max_views or not 1 <= min_interval <= max_interval:
            raise ValueError("Invalid clip sampling limits")
        if not 0 <= reverse_probability <= 1:
            raise ValueError("Invalid reverse_probability")
        self.root, self.view = Path(root), view
        self.min_views, self.max_views = min_views, max_views
        self.min_interval, self.max_interval = min_interval, max_interval
        self.reverse_probability, self.frame_stride = reverse_probability, frame_stride
        self.depth_subdir = depth_subdir
        self.frame_rate = frame_rate
        self.max_depth = max_depth
        self.max_tcp_linear_speed, self.max_tcp_angular_speed = max_tcp_linear_speed, max_tcp_angular_speed
        self.seed, self.augment, self.color_augment, self.epoch = seed, augment, color_augment, 0
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if tasks is not None and tasks != ["place_pen_in_cups"] and tasks != ("place_pen_in_cups",):
            raise ValueError("Only place_pen_in_cups is available")
        self.episodes = []
        positions = []
        for path in sorted(self.root.glob("place_pen_in_cups__episode_*")):
            meta_path = path / "metadata.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text())
            n = int(meta["frame_count"])
            sampled = np.arange(0, n, frame_stride)
            if len(sampled) < min_views:
                continue
            required = [path / "images" / view / "000000.png", path / self.depth_subdir / view / "000000.png",
                        path / "intrinsic" / f"{view}.npy", path / "extrinsic" / f"{view}.npy",
                        path / "TCP" / view / "state.npy", path / "timestamps.npy"]
            if not all(p.is_file() for p in required):
                raise FileNotFoundError(f"Incomplete episode {path}: {[str(p) for p in required if not p.is_file()]}")
            states = np.load(path / "TCP" / view / "state.npy", mmap_mode="r")
            times = np.load(path / "timestamps.npy", mmap_mode="r")
            if states.shape != (n, 2, 7) or times.shape != (n,) or not np.isfinite(states).all() or not np.all(np.diff(times) > 0):
                raise ValueError(f"Invalid aligned TCP/timestamps in {path}")
            if not np.isfinite(times).all() or not np.isfinite(states[:, :, 6]).all() or np.any((states[:, :, 6] < 0) | (states[:, :, 6] > 1)):
                raise ValueError(f"Invalid timestamps/gripper in {path}")
            if frame_rate is not None and (not np.isfinite(frame_rate) or frame_rate <= 0):
                raise ValueError("frame_rate must be positive")
            effective_rate = float(frame_rate if frame_rate is not None else meta["fps"])
            invalid = np.zeros(len(sampled) - 1, dtype=bool)
            selected = np.asarray(states[sampled])
            dt = np.full(len(sampled) - 1, frame_stride / effective_rate)
            if max_tcp_linear_speed is not None:
                invalid |= (np.linalg.norm(np.diff(selected[:, :, :3], axis=0), axis=-1) / dt[:, None] > max_tcp_linear_speed).any(1)
            if max_tcp_angular_speed is not None:
                rot = self._rpy_to_matrix(selected[:, :, 3:6].astype(np.float64))
                relative = np.einsum("taji,tajk->taik", rot[:-1], rot[1:])
                angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
                invalid |= (angle / dt[:, None] > max_tcp_angular_speed).any(1)
            boundaries = (np.flatnonzero(invalid) + 1).tolist()
            segments = tuple((a, b) for a, b in zip([0, *boundaries], [*boundaries, len(sampled)])
                             if b - a >= 1 + (min_views - 1) * min_interval)
            if not segments:
                continue
            self.episodes.append(PenEpisode(meta["task"], path.name, path, len(sampled),
                                            effective_rate / frame_stride, segments, len(boundaries)))
            positions.append(selected[:, :, :3])
            if max_episodes is not None and len(self.episodes) >= max_episodes:
                break
        if not self.episodes:
            raise RuntimeError(f"No valid pen/cups episodes in {self.root}")
        xyz = np.concatenate(positions).astype(np.float64)
        self.tcp_position_mean = torch.tensor(xyz.mean(0), dtype=torch.float32)
        self.tcp_position_std = torch.tensor(np.maximum(xyz.std(0), 1e-3), dtype=torch.float32)
        self.invalid_transition_count = sum(e.invalid_transition_count for e in self.episodes)
        self.segmented_episode_count = sum(e.invalid_transition_count > 0 for e in self.episodes)

    def __getitem__(self, index):
        requested, sample_seed = None, None
        if isinstance(index, tuple):
            index, requested, *rest = index
            sample_seed = rest[0] if rest else None
        episode = self.episodes[int(index)]
        rng = self._rng(index, sample_seed)
        indices, interval = self.sample_frame_indices(episode, rng, requested)
        rows = indices * self.frame_stride
        params = self._sample_augmentation(rng, color_augment=self.color_augment) if self.augment else None
        k = np.load(episode.path / "intrinsic" / f"{self.view}.npy").astype(np.float32)
        k[0, 2] += self.PAD_LEFT
        k[1, 2] += self.PAD_TOP
        extrinsic = np.load(episode.path / "extrinsic" / f"{self.view}.npy").astype(np.float32)
        tcp = np.asarray(np.load(episode.path / "TCP" / self.view / "state.npy", mmap_mode="r")[rows]).copy()
        times = rows.astype(np.float64) / (episode.frame_rate * self.frame_stride)
        images, depths, masks, originals = [], [], [], []
        for row in rows:
            with Image.open(episode.path / "images" / self.view / f"{row:06d}.png") as source:
                rgb = source.convert("RGB")
                if rgb.size != (320, 240):
                    raise ValueError(f"Invalid RGB size in {episode.path}, frame {row}")
                if params:
                    rgb = self._apply_augmentation(rgb, params)
                image = TVF.pil_to_tensor(rgb).float() / 255
            with Image.open(episode.path / self.depth_subdir / self.view / f"{row:06d}.png") as source:
                depth = torch.from_numpy(np.asarray(source, dtype=np.float32) / 1000)
            valid = torch.isfinite(depth) & (depth > 0)
            if self.max_depth is not None:
                valid &= depth <= self.max_depth
            depth = torch.where(valid, depth, 0)
            images.append(F.pad(image, (1, 1, 6, 6), mode="reflect") * 2 - 1)
            depths.append(F.pad(depth, (1, 1, 6, 6)))
            masks.append(F.pad(valid, (1, 1, 6, 6)))
            original = torch.zeros((252, 322), dtype=torch.bool)
            original[6:246, 1:321] = True
            originals.append(original)
        xyz = tcp[0, :, :3]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (k[:2, :2] @ (xyz[:, :2] / xyz[:, 2:3]).T).T + k[:2, 2]
        visible = np.isfinite(uv).all(1) & (xyz[:, 2] > 0) & (uv[:, 0] >= 1) & (uv[:, 0] < 321) & (uv[:, 1] >= 6) & (uv[:, 1] < 246)
        uv = np.nan_to_num(uv, nan=0, posinf=0, neginf=0)
        uv[:, 0] = np.clip(uv[:, 0], 1, 320)
        uv[:, 1] = np.clip(uv[:, 1], 6, 245)
        result = dict(images=torch.stack(images), depth=torch.stack(depths), valid_mask=torch.stack(masks),
                      original_mask=torch.stack(originals), intrinsics=torch.from_numpy(k).expand(len(rows), -1, -1).clone(),
                      extrinsics=torch.from_numpy(extrinsic).expand(len(rows), -1, -1).clone(),
                      frame_indices=torch.tensor(rows.copy()), frame_times=torch.tensor(times, dtype=torch.float32),
                      frame_rate=episode.frame_rate, tcp_state=torch.tensor(tcp),
                      tcp_query_points=torch.tensor(uv, dtype=torch.float32), tcp_query_valid=torch.tensor(visible),
                      interval=interval, task=episode.task, episode=episode.name,
                      source_size=torch.tensor([240, 320]), padding=torch.tensor([1, 1, 6, 6]))
        result.update(self._extra_sample_fields(episode, rows, params))
        return result
