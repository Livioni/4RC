"""Causal observation/action windows with deterministic episode holdouts."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from arc.action import future_actions_in_current_camera, project_tcp
from .robotwin import RoboTwin4RC
from .mixture import DatasetSource, WeightedDatasetMixture


class NoActionEpisodes(RuntimeError):
    pass


class RoboTwinActionDataset(RoboTwin4RC):
    def __init__(
        self, root, *, history_frames=8, prediction_horizon=16, split="train",
        validation_fraction=0.1, split_key="robotwin", **options,
    ):
        if history_frames < 1 or prediction_horizon < 1:
            raise ValueError("History and prediction horizons must be positive")
        if split not in ("train", "validation") or not 0 <= validation_fraction < 1:
            raise ValueError("Invalid action dataset split")
        self.history_frames = int(history_frames)
        self.prediction_horizon = int(prediction_horizon)
        self.split = split
        super().__init__(
            root, min_views=history_frames, max_views=history_frames,
            min_interval=1, max_interval=1, reverse_probability=0,
            **options,
        )
        kept, starts, instructions, moments = [], [], [], []
        for episode in self.episodes:
            key = f"{split_key}/{episode.task}/{episode.name}"
            fraction = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64
            is_validation = fraction < validation_fraction
            if is_validation != (split == "validation"):
                continue
            texts = self._read_json(episode.path / "metadata.json").get("instructions", [])
            if not isinstance(texts, list):
                raise ValueError(f"Expected an instruction list in {episode.path}")
            texts = tuple(text.strip() for text in texts if isinstance(text, str) and text.strip())
            if not texts:
                continue
            total_frames = history_frames + 1  # At least one real future label.
            ranges = [
                np.arange(start, end - total_frames + 1, dtype=np.int64)
                for start, end in episode.valid_segments if end - start >= total_frames
            ]
            if not ranges:
                continue
            possible = np.concatenate(ranges)
            states = self._read_states(episode)
            intrinsics = np.load(episode.path / "intrinsics" / f"{self.view}.npy")
            pixels = np.einsum("ij,taj->tai", intrinsics, states[..., :3])
            with np.errstate(divide="ignore", invalid="ignore"):
                uv = pixels[..., :2] / pixels[..., 2:3]
            visible = (
                np.isfinite(uv).all(-1) & (states[..., 2] > 1e-6)
                & (uv[..., 0] >= 0) & (uv[..., 0] < self.SOURCE_WIDTH)
                & (uv[..., 1] >= 0) & (uv[..., 1] < self.SOURCE_HEIGHT)
            ).any(-1)
            valid_windows = np.convolve(visible.astype(np.int32), np.ones(history_frames), mode="valid") > 0
            possible = possible[valid_windows[possible]]
            if not len(possible):
                continue
            extrinsics = np.load(episode.path / "extrinsics" / f"{self.view}.npy")
            if not np.isfinite(extrinsics).all():
                raise ValueError(f"Invalid camera extrinsics in {episode.path}")
            moments.append(self._action_moments(states, extrinsics, possible, episode.valid_segments))
            kept.append(episode)
            starts.append(possible)
            instructions.append(texts)
        if not kept:
            raise NoActionEpisodes(f"No {split} action episodes below {self.root}")
        self.episodes, self.starts, self.instructions = kept, starts, instructions
        self._episode_indices = {episode.path: i for i, episode in enumerate(kept)}
        # The sampler draws episodes uniformly within a source. Statistics use
        # the same distribution over episodes, anchors and future offsets.
        mean = np.mean([item[0] for item in moments], axis=0)
        second = np.mean([item[1] for item in moments], axis=0)
        self.tcp_position_mean = torch.tensor(mean, dtype=torch.float32)
        self.tcp_position_std = torch.tensor(
            np.maximum(second - mean**2, 1e-6) ** 0.5, dtype=torch.float32,
        )
        # These adapter statistics belong ONLY to the new action head.
        self.invalid_transition_count = sum(e.invalid_transition_count for e in kept)
        self.segmented_episode_count = sum(e.invalid_transition_count > 0 for e in kept)
        task_text = {episode.task: texts[0] for episode, texts in zip(kept, instructions)}
        self.shuffled_instructions = [
            next((text for task, text in sorted(task_text.items()) if task != episode.task), "")
            for episode in kept
        ]

    def _read_states(self, episode):
        return np.stack([
            np.load(episode.path / self.tcp_directory / name, mmap_mode="r")
            for name in ("left_state.npy", "right_state.npy")
        ], axis=1)

    def _action_moments(self, states, extrinsics, starts, segments):
        anchors = starts + self.history_frames - 1
        ends = np.array([next(end for start, end in segments if start <= anchor < end)
                         for anchor in anchors])
        lengths = np.minimum(ends - anchors - 1, self.prediction_horizon)
        total = np.zeros((len(anchors), 2, 3), dtype=np.float64)
        square = np.zeros_like(total)
        for offset in range(1, self.prediction_horizon + 1):
            valid = offset <= lengths
            selected = anchors[valid]
            current = extrinsics[selected].astype(np.float64)
            future = extrinsics[selected + offset].astype(np.float64)
            rotation = current[:, :3, :3] @ future[:, :3, :3].transpose(0, 2, 1)
            translation = current[:, :3, 3] - np.einsum("nij,nj->ni", rotation, future[:, :3, 3])
            position = np.einsum(
                "nij,naj->nai", rotation, states[selected + offset, :, :3],
            ) + translation[:, None]
            total[valid] += position
            square[valid] += np.square(position)
        # Match uniform anchor sampling; each anchor averages only real offsets.
        return (total / lengths[:, None, None]).mean(0), (square / lengths[:, None, None]).mean(0)

    def _rng(self, index, sample_seed=None):
        # Sampler seeds already encode the epoch. Persistent workers must not
        # depend on a stale copied dataset.epoch when a run is resumed.
        if sample_seed is not None:
            return np.random.default_rng(np.random.SeedSequence([self.seed, int(index), int(sample_seed)]))
        return super()._rng(index, sample_seed)

    def eligible_indices(self, num_views):
        if int(num_views) != self.history_frames:
            return np.empty(0, dtype=np.int64)
        return np.arange(len(self.episodes), dtype=np.int64)

    def sample_frame_indices(self, episode_or_frames, rng, num_views=None):
        if num_views is not None and int(num_views) != self.history_frames:
            raise ValueError("Action windows have a fixed configured history length")
        index = self._episode_indices[episode_or_frames.path]
        choices = self.starts[index]
        start = int(choices[int(rng.integers(len(choices)))])
        return start + np.arange(self.history_frames, dtype=np.int64), 1

    def __getitem__(self, index):
        episode_index = int(index[0] if isinstance(index, tuple) else index)
        sample = super().__getitem__(index)
        episode = self.episodes[episode_index]
        future_indices = int(sample["frame_indices"][-1]) + np.arange(1, self.prediction_horizon + 1)
        anchor = int(sample["frame_indices"][-1])
        segment_end = next(end for start, end in episode.valid_segments if start <= anchor < end)
        future_valid = future_indices < segment_end
        # Repeat the last valid state only for storage; masked positions carry no supervision.
        padded_indices = np.minimum(future_indices, segment_end - 1)
        future_state = torch.from_numpy(self._read_states(episode)[padded_indices].copy()).float()
        extrinsics = np.load(episode.path / "extrinsics" / f"{self.view}.npy", mmap_mode="r")
        targets = future_actions_in_current_camera(
            future_state, torch.tensor(extrinsics[padded_indices]),
            sample["extrinsics"][-1],
        )
        centres, valid = project_tcp(
            sample["tcp_state"][..., :3], sample["intrinsics"],
            image_height=self.PADDED_HEIGHT, image_width=self.PADDED_WIDTH,
        )
        texts = self.instructions[episode_index]
        if self.split == "validation":
            instruction = texts[0]
        else:
            sample_seed = int(index[2]) if isinstance(index, tuple) and len(index) == 3 else None
            epoch_seed = self.epoch if sample_seed is None else 0
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch_seed, sample_seed or 0, episode_index, 431]))
            instruction = texts[int(rng.integers(len(texts)))]
        sample.update(
            history_tcp_query_points=centres, history_tcp_valid=valid,
            future_actions=targets,
            future_action_valid=torch.from_numpy(future_valid.copy()),
            future_frame_times=torch.tensor(future_indices / episode.frame_rate, dtype=torch.float32),
            instruction=instruction, shuffled_instruction=self.shuffled_instructions[episode_index],
        )
        return sample

    def manifest(self):
        return [
            {"task": e.task, "episode": e.name, "path": str(e.path), "windows": len(starts)}
            for e, starts in zip(self.episodes, self.starts)
        ]


def build_action_dataset(config: dict[str, Any], split="train") -> WeightedDatasetMixture | None:
    sources = []
    common_keys = (
        "view", "frame_rate", "max_depth", "max_tcp_linear_speed",
        "max_tcp_angular_speed", "seed", "max_episodes",
    )
    for source in config["data_sources"]:
        if source.get("type", "robotwin") != "robotwin":
            raise ValueError("Stage two currently supports RoboTwin action labels")
        if source.get("weight", 1.0) <= 0:
            continue
        options = {key: config[key] for key in common_keys if key in config}
        source_options = dict(source.get("options", {}))
        root = source_options.pop("root")
        options.update(source_options)
        if config.get("max_episodes") is not None:
            options["max_episodes"] = config["max_episodes"]
        options["augment"] = bool(config.get("augment", True)) and split == "train"
        try:
            dataset = RoboTwinActionDataset(
                root, history_frames=config["history_frames"],
                prediction_horizon=config["prediction_horizon"], split=split,
                validation_fraction=config.get("validation_fraction", 0.1),
                split_key=source["name"], **options,
            )
        except NoActionEpisodes:
            if split == "validation":
                continue
            raise
        sources.append(DatasetSource(source["name"], "robotwin_action", dataset, source.get("weight", 1.0)))
    return WeightedDatasetMixture(sources) if sources else None

