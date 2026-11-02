"""Stage-two action windows for the decimated pen/cups recordings."""
from __future__ import annotations

from functools import partial
import numpy as np
import torch

from .pen_cups import PenCups4RC
from .robotwin_action import RoboTwinActionDataset
from .utils.wrist import load_wrist_images
from arc.state_normalization import validate_robot_states


class PenCupsActionDataset(RoboTwinActionDataset, PenCups4RC):
    def _initialize_base(self, root, **options):
        PenCups4RC.__init__(self, root, **options)

    def _episode_instructions(self, episode):
        return self._read_json(episode.path / "metadata.json").get("language_instruction", [])

    def _read_intrinsics(self, episode):
        return np.load(episode.path / "intrinsic" / f"{self.view}.npy")

    def _read_extrinsics(self, episode):
        matrix = np.load(episode.path / "extrinsic" / f"{self.view}.npy")
        return np.broadcast_to(matrix, (episode.num_frames, *matrix.shape))

    def _read_states(self, episode):
        return np.asarray(np.load(episode.path / "TCP" / self.view / "state.npy", mmap_mode="r")[::self.frame_stride])

    def _read_robot_states(self, episode):
        state = np.asarray(np.load(episode.path / "observations" / "joint_position.npy", mmap_mode="r")[::self.frame_stride]).copy()
        if state.shape != (episode.num_frames, 14):
            raise ValueError(f"Invalid joint state in {episode.path}")
        state[:, [6, 13]] = np.clip(state[:, [6, 13]] / 0.10, 0, 1)
        validate_robot_states(state)
        return state

    def _sample_anchor(self, sample):
        return int(sample["frame_indices"][-1]) // self.frame_stride

    def _extra_sample_fields(self, episode, frame_indices, augmentation):
        row = int(frame_indices[-1])
        transform = None if augmentation is None else partial(self._apply_augmentation, params=augmentation)
        for side in ("left_view", "right_view"):
            path = episode.path / "images" / side / f"{row:06d}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Stage2 requires wrist image: {path}")
        state = self._read_robot_states(episode)[row // self.frame_stride]
        return {"wrist_images": load_wrist_images(episode.path, row, transform=transform),
                "current_robot_state": torch.tensor(state, dtype=torch.float32)}
