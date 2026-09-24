import json
import pickle
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from arc.datasets import collate_training_samples, DatasetSource, WeightedDatasetMixture, WeightedMultiSourceBatchSampler
from arc.datasets.droid import DroidDataset
from arc.datasets.droid_index import ensure_index, write_splits, load_splits


def test_index_and_split_reuse_without_decoding(droid_files, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Indexing decoded an image")
    monkeypatch.setattr(Image, "open", forbidden)
    ensure_index(droid_files["root"], droid_files["index_path"], rebuild=True)
    train, val = tmp_path / "generated_train.txt", tmp_path / "generated_val.txt"
    args = (droid_files["root"], droid_files["index_path"], train, val)
    first = write_splits(*args, seed=42)
    second = write_splits(*args, seed=43)  # Existing TXT is authoritative.
    assert first == second
    assert not (set(first["train"]) & set(first["validation"]))
    assert set(first["train"] + first["validation"]) == {"train_a", "train_b", "val_a"}
    train.write_text("train_a\n")
    val.write_text("train_a\n")
    with pytest.raises(ValueError, match="overlap"):
        load_splits(droid_files["root"], train, val)


def test_monocular_geometry_fixed_fps_and_arm(droid_files):
    dataset = DroidDataset(**droid_files, augment=False, reverse_probability=1, min_interval=2, max_interval=2)
    sample = dataset.get_sample(0, 3, 100)
    assert sample["images"].shape == (3, 3, 182, 322)
    assert sample["tcp_state"].shape == (3, 1, 7)
    assert sample["tcp_query_points"].shape == (1, 2)
    assert (sample["frame_indices"].diff() == -2).all()
    torch.testing.assert_close(sample["frame_times"], sample["frame_indices"].float() / 15)
    torch.testing.assert_close(sample["intrinsics"][0, :2, 2], torch.tensor([161., 91.]))
    assert sample["depth"][:, 1:-1, 1:-1].eq(1.2).all()
    assert not sample["valid_mask"][:, 0].any()
    other = dataset.get_sample(1, 3, 100)
    assert sample["episode"] == other["episode"] and sample["camera_id"] != other["camera_id"]
    assert not torch.allclose(sample["extrinsics"], other["extrinsics"])
    dataset.set_epoch(10)
    torch.testing.assert_close(sample["images"], dataset.get_sample(0, 3, 100)["images"])


def test_action_only_reads_history_and_masks_short_future(droid_files, monkeypatch):
    dataset = DroidDataset(**droid_files, stage=2, augment=False)
    record = dataset._record(0)
    record["starts"] = [(15, 16)]  # Frames 15..22, one real future at 23.
    opened, original = [], Image.open
    def tracked(path, *args, **kwargs):
        opened.append(Path(path))
        assert 15 <= int(Path(path).stem) <= 22
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Image, "open", tracked)
    sample = dataset.get_sample(0, 8, 42)
    assert len(opened) == 16
    assert sample["future_actions"].shape == (16, 1, 10)
    assert sample["future_action_valid"].sum() == 1
    torch.testing.assert_close(sample["future_actions"], sample["future_actions"][:1].expand(16, -1, -1))
    torch.testing.assert_close(sample["future_step_indices"], torch.arange(1, 17))
    assert "future_frame_times" not in sample
    assert sample["history_tcp_valid"][0].all()
    assert sample["instruction"] != sample["shuffled_instruction"]


def test_profile_cache_workers_and_split_isolation(droid_files, monkeypatch):
    train = DroidDataset(**droid_files, stage=2, cache_size=1)
    val = DroidDataset(**droid_files, stage=2, split="validation")
    train_names = {train._record(i)["episode"] for i in range(len(train))}
    val_names = {val._record(i)["episode"] for i in range(len(val))}
    assert train_names.isdisjoint(val_names)
    assert len(train._cache) == 1
    monkeypatch.setattr(DroidDataset, "_build_profile", lambda *a: pytest.fail("Rebuilt cached statistics"))
    cached = DroidDataset(**droid_files, stage=2)
    torch.testing.assert_close(cached.tcp_position_std, train.tcp_position_std)
    restored = pickle.loads(pickle.dumps(cached))
    assert restored._connection is None and not restored._cache
    mixture = WeightedDatasetMixture([DatasetSource("droid", "droid", cached, 1.)])
    sampler = WeightedMultiSourceBatchSampler(mixture, images_per_batch=8, scene_counts=(1,), batches_per_epoch=2)
    first = list(DataLoader(mixture, batch_sampler=sampler, num_workers=0, collate_fn=collate_training_samples))
    second = list(DataLoader(mixture, batch_sampler=sampler, num_workers=2, multiprocessing_context="spawn", collate_fn=collate_training_samples))
    for a, b in zip(first, second):
        torch.testing.assert_close(a["images"], b["images"])
        assert a["instruction"] == b["instruction"]


def test_discontinuities_and_future_stats(droid_files):
    root = Path(droid_files["root"])
    for camera in ("cam_a", "cam_b"):
        path = root / "train_a" / "TCP" / camera / "state.npy"
        state = np.load(path)
        state[12:, 0] += .3
        np.save(path, state)
    ensure_index(root, droid_files["index_path"], rebuild=True)
    dataset = DroidDataset(**droid_files, stage=2)
    assert dataset.segmented_episode_count == 1
    for seed in range(20):
        sample = dataset.get_sample(0, 8, seed)
        indices = sample["frame_indices"]
        assert indices[-1] < 12 or indices[0] >= 12
    # Compute the exact episode/camera -> anchor -> real future distribution.
    means = []
    for i in range(len(dataset)):
        record = dataset._record(i)
        per_anchor = []
        for a, b in record["starts"]:
            for start in range(a, b):
                anchor = start + 7
                end = next(e for s, e in record["segments"] if s <= anchor < e)
                per_anchor.append(record["state"][anchor+1:min(anchor+17, end), :3].mean(0))
        means.append(np.mean(per_anchor, axis=0))
    torch.testing.assert_close(dataset.tcp_position_mean[0], torch.tensor(np.mean(means, 0)), atol=1e-6, rtol=1e-6)
