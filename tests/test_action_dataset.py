import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from arc.datasets.robotwin import RoboTwin4RC
from arc.datasets.robotwin_action import RoboTwinActionDataset
from arc.datasets import DatasetSource, WeightedDatasetMixture, WeightedMultiSourceBatchSampler, collate_training_samples


def write_episode(root, task="lift", name="episode_0", frames=32):
    episode = root / task / name
    for directory in ("images/third_views", "depths/third_views", "TCP_third", "intrinsics", "extrinsics"):
        (episode / directory).mkdir(parents=True)
    for i in range(frames):
        Image.fromarray(np.full((240, 320, 3), i, dtype=np.uint8)).save(episode / "images/third_views" / f"{i:06d}.png")
        Image.fromarray(np.full((240, 320), 1000, dtype=np.uint16)).save(episode / "depths/third_views" / f"{i:06d}.png")
    np.save(episode / "intrinsics/third_views.npy", np.array([[100., 0, 160], [0, 100, 120], [0, 0, 1]], np.float32))
    np.save(episode / "extrinsics/third_views.npy", np.repeat(np.eye(4, dtype=np.float32)[None, :3], frames, axis=0))
    for arm, x in (("left", -0.2), ("right", 0.2)):
        state = np.zeros((frames, 7), dtype=np.float32)
        state[:, 0] = x + np.arange(frames) * 0.002
        state[:, 2] = 1
        state[:, 6] = (np.arange(frames) >= 16).astype(float)
        np.save(episode / "TCP_third" / f"{arm}_state.npy", state)
    (episode / "metadata.json").write_text(json.dumps({
        "frequency_hz": 15, "instructions": [f"Perform {task}.", f"Please perform {task}."],
    }))
    (episode / "TCP_third/metadata.json").write_text(json.dumps({
        "shape": [frames, 7], "columns": ["x","y","z","roll","pitch","yaw","gripper_open"],
        "position_unit": "meter", "rotation_unit": "radian",
        "rpy_convention": RoboTwin4RC.RPY_CONVENTION,
        "coordinate_frame": f"third_views {RoboTwin4RC.CAMERA_SUFFIX}", "camera": "third_views",
        "gripper": {"type":"binary","open":1,"closed":0,"source_threshold":0.5},
    }))
    return episode


def test_future_images_not_read_and_targets_aligned(tmp_path, monkeypatch):
    write_episode(tmp_path)
    dataset = RoboTwinActionDataset(tmp_path, validation_fraction=0, augment=False)
    dataset.starts[0] = np.array([0])
    opened = []
    original = Image.open
    def checked_open(path, *args, **kwargs):
        opened.append(Path(path))
        assert int(Path(path).stem) < 8, "A future image was read"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", checked_open)
    sample = dataset.get_sample(0, 8, 42)
    assert len(opened) == 16  # Eight RGB and eight depth images.
    assert sample["images"].shape == (8, 3, 252, 322)
    assert sample["future_actions"].shape == (16, 2, 10)
    torch.testing.assert_close(sample["future_frame_times"], torch.arange(8, 24) / 15)
    torch.testing.assert_close(sample["future_actions"][:, 0, 0], -0.2 + torch.arange(8, 24) * 0.002)
    assert sample["future_actions"][:8, :, 9].eq(-1).all()
    assert sample["future_actions"][8:, :, 9].eq(1).all()
    torch.testing.assert_close(sample["history_tcp_query_points"][0, 0], torch.tensor([141., 126.]))


def test_fixed_batches_and_epoch_independent_request_seed(tmp_path):
    write_episode(tmp_path, task="lift")
    write_episode(tmp_path, task="place")
    dataset = RoboTwinActionDataset(tmp_path, validation_fraction=0, augment=True)
    dataset.set_epoch(1)
    first = dataset.get_sample(0, 8, 123)
    dataset.set_epoch(17)
    resumed = dataset.get_sample(0, 8, 123)
    for key in ("images", "frame_indices", "future_actions", "history_tcp_query_points"):
        torch.testing.assert_close(first[key], resumed[key])
    assert first["instruction"] == resumed["instruction"]
    mixture = WeightedDatasetMixture([DatasetSource("synthetic", "robotwin_action", dataset, 1.)])
    for batch_size in (1, 2):
        sampler = WeightedMultiSourceBatchSampler(
            mixture, images_per_batch=8 * batch_size, scene_counts=(batch_size,), batches_per_epoch=1,
        )
        batch = collate_training_samples([mixture[request] for request in next(iter(sampler))])
        assert batch["images"].shape[:2] == (batch_size, 8)
        assert batch["history_tcp_query_points"].shape == (batch_size, 8, 2, 2)


def test_episode_split_is_disjoint(tmp_path):
    names = {}
    for i in range(100):
        name = f"episode_{i}"
        fraction = int(hashlib.sha256(f"robotwin/lift/{name}".encode()).hexdigest()[:16], 16) / 2**64
        names.setdefault(fraction < 0.5, name)
        if len(names) == 2:
            break
    for name in names.values():
        write_episode(tmp_path, name=name)
    train = RoboTwinActionDataset(tmp_path, split="train", validation_fraction=0.5, augment=False)
    validation = RoboTwinActionDataset(tmp_path, split="validation", validation_fraction=0.5, augment=False)
    assert {e.path for e in train.episodes}.isdisjoint({e.path for e in validation.episodes})
    assert len(train) == len(validation) == 1


def test_windows_do_not_cross_discontinuities(tmp_path):
    episode = write_episode(tmp_path, frames=64)
    state = np.load(episode / "TCP_third/left_state.npy")
    state[32:, 0] += 0.6  # Invalid physical jump, still inside the image.
    np.save(episode / "TCP_third/left_state.npy", state)
    dataset = RoboTwinActionDataset(tmp_path, validation_fraction=0, augment=False)
    assert dataset.segmented_episode_count == 1
    for start in dataset.starts[0]:
        assert start + 24 <= 32 or start >= 32

