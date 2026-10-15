"""Lazy monocular DROID clips for Stage1 reconstruction."""
from __future__ import annotations

from collections import OrderedDict
import json
import logging
import math
from pathlib import Path
import os
import sqlite3

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from arc.action import project_tcp
from arc.datasets.robotwin import RoboTwin4RC
from arc.datasets.droid_index import (DEFAULT_INDEX, DEFAULT_TRAIN, DEFAULT_VAL, FPS, INDEX_VERSION,
                                digest, index_lock, index_metadata, load_splits)

LOGGER = logging.getLogger(__name__)


def select_range(ranges, rng):
    offset = int(rng.integers(sum(end - start for start, end in ranges)))
    for start, end in ranges:
        if offset < end - start:
            return start + offset
        offset -= end - start
    raise RuntimeError("Empty sample ranges")


class DroidDataset(torch.utils.data.Dataset):
    num_arms = 1
    padding = (1, 1, 1, 1)
    SOURCE_HEIGHT, SOURCE_WIDTH = 180, 320
    PADDED_HEIGHT, PADDED_WIDTH = 182, 322

    def __init__(self, root, *, index_path=DEFAULT_INDEX, train_set=DEFAULT_TRAIN, val_set=DEFAULT_VAL,
                 split="train", min_views=2, max_views=18, min_interval=1, max_interval=5,
                 reverse_probability=0.5, frame_rate=15,
                 max_depth=3.0, seed=42, augment=True, color_augment=True, max_episodes=None, cache_size=8,
                 max_tcp_linear_speed=3.0, max_tcp_angular_speed=4 * math.pi, **unused):
        if unused:
            raise TypeError(f"Unknown DROID options: {sorted(unused)}")
        if split not in ("train", "validation"):
            raise ValueError("Expected train/validation split")
        if float(frame_rate) != FPS:
            raise ValueError("This DROID export uses fixed 15 fps")
        if not 1 <= min_views <= max_views or not 1 <= min_interval <= max_interval:
            raise ValueError("Invalid frame count or interval range")
        if not 0 <= reverse_probability <= 1 or cache_size < 1:
            raise ValueError("Invalid reversal or cache size")
        if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= 0):
            raise ValueError("max_depth must be positive or None")
        if max_episodes is not None and max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        self.root = Path(root).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.train_set, self.val_set = str(train_set), str(val_set)
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Missing {self.index_path}; run python -m droid_script.prepare_droid_dataset first")
        meta = index_metadata(self.index_path)
        if meta["signature"]["root"] != str(self.root) or meta["signature"]["version"] != INDEX_VERSION:
            raise ValueError("DROID index root/version mismatch; rebuild the index on this machine")
        for key, value in (("max_tcp_linear_speed", max_tcp_linear_speed),
                           ("max_tcp_angular_speed", max_tcp_angular_speed)):
            if meta["signature"].get(key) != value:
                raise ValueError(f"Index {key} differs from config; rebuild with matching thresholds")
        splits = load_splits(self.root, train_set, val_set)
        names = sorted(splits[split])
        if max_episodes is not None:
            names = names[:max_episodes]
        self.split = split
        self.seed, self.epoch = int(seed), 0
        self.min_views = min_views
        self.max_views = max_views
        self.min_interval, self.max_interval = min_interval, max_interval
        self.reverse_probability = reverse_probability
        self.max_depth = max_depth
        self.augment = augment and split == "train"
        self.color_augment = color_augment
        self.cache_size = int(cache_size)
        self._cache, self._connection, self._pid = OrderedDict(), None, None
        self.profile = digest({"index": meta["uuid"], "episodes": names, "stage": 1,
                               "min_views": self.min_views, "min_interval": self.min_interval})
        with index_lock(self.index_path), sqlite3.connect(self.index_path) as connection:
            cached = connection.execute("SELECT data FROM profiles WHERE key=?", (self.profile,)).fetchone()
            if cached:
                profile = json.loads(cached[0])
            else:
                profile = self._build_profile(connection, set(names))
                connection.execute("INSERT INTO profiles VALUES(?,?)", (self.profile, json.dumps(profile)))
        self.ids = np.asarray(profile["ids"], dtype=np.int64)
        self.lengths = np.asarray(profile["lengths"], dtype=np.int32)
        self.tcp_position_mean = torch.tensor(profile["mean"], dtype=torch.float32)[None]
        self.tcp_position_std = torch.tensor(profile["std"], dtype=torch.float32)[None]
        self.invalid_transition_count = profile["invalid_transitions"]
        self.segmented_episode_count = profile["segmented_episodes"]
        self.instruction_pool = tuple(profile["instruction_pool"])
        LOGGER.info("DROID Stage1 %s: %d camera streams, profile %s", split, len(self), self.profile[:12])

    def _build_profile(self, connection, names):
        ids, lengths, texts, found, segmented = [], [], set(), set(), set()
        total, second = np.zeros(3), np.zeros(3)
        invalid = 0
        connection.execute("CREATE TEMP TABLE selected_episodes(name TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO selected_episodes VALUES(?)", ((name,) for name in names))
        for camera_id, episode, camera, frames, data in connection.execute(
                "SELECT c.* FROM cameras c JOIN selected_episodes s ON c.episode=s.name ORDER BY c.id"):
            found.add(episode)
            record = json.loads(data)
            length = max(end - start for start, end in record["segments"])
            if length < 1 + (self.min_views - 1) * self.min_interval:
                continue
            mean, moment = np.asarray(record["mean"]), np.asarray(record["second"])
            ids.append(camera_id)
            lengths.append(length)
            total += mean
            second += moment
            invalid += record["invalid_transitions"]
            if record["invalid_transitions"]:
                segmented.add(episode)
            texts.update(record["instructions"])
            if len(ids) % 2000 == 0:
                LOGGER.info("DROID Stage1 %s statistics: %d streams", self.split, len(ids))
        if names - found:
            raise ValueError(f"Split episodes absent from index (rebuild or edit TXT): {sorted(names - found)[:5]}")
        if not ids:
            raise ValueError(f"No eligible DROID Stage1 {self.split} windows")
        mean = total / len(ids)
        std = np.sqrt(np.maximum(second / len(ids) - mean ** 2, 1e-6))
        return {"ids": ids, "lengths": lengths, "mean": mean.tolist(), "std": std.tolist(),
                "invalid_transitions": invalid, "segmented_episodes": len(segmented),
                "instruction_pool": sorted(texts)[:256]}

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def eligible_indices(self, num_views):
        if not self.min_views <= num_views <= self.max_views:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.lengths >= 1 + (num_views - 1) * self.min_interval)

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_connection=None, _pid=None, _cache=OrderedDict())
        return state

    def _record(self, index):
        if self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
            self._cache, self._pid = OrderedDict(), os.getpid()
        camera_id = int(self.ids[index])
        if camera_id not in self._cache:
            episode, camera, frames, data = self._connection.execute(
                "SELECT episode,camera,frames,data FROM cameras WHERE id=?", (camera_id,)).fetchone()
            record = json.loads(data)
            record.update(episode=episode, camera=camera, frames=frames)
            record["state"] = np.load(self.root / episode / "TCP" / camera / "state.npy", mmap_mode="r", allow_pickle=False)
            self._cache[camera_id] = record
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(camera_id)
        return self._cache[camera_id]

    def _indices(self, record, views, rng):
        if not self.min_views <= views <= self.max_views:
            raise ValueError("Requested clip length is outside configured limits")
        choices = [interval for interval in range(self.min_interval, self.max_interval + 1)
                   if any(end - start > (views - 1) * interval for start, end in record["segments"])]
        if not choices:
            raise ValueError("No segment can provide the requested clip")
        interval = int(rng.choice(choices))
        ranges = [(start, end - (views - 1) * interval) for start, end in record["segments"]
                  if end - start > (views - 1) * interval]
        indices = select_range(ranges, rng) + np.arange(views) * interval
        if rng.random() < self.reverse_probability:
            indices = indices[::-1].copy()
        return indices, interval

    def get_sample(self, index, num_views, sample_seed):
        # The sampler seed already encodes the epoch, including with persistent workers.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index), int(sample_seed)]))
        record = self._record(index)
        indices, interval = self._indices(record, int(num_views), rng)
        episode, camera = record["episode"], record["camera"]
        k = torch.tensor(record["intrinsics"], dtype=torch.float32)
        k[:2, 2] += 1
        ext = torch.tensor(record["extrinsics"], dtype=torch.float32)
        state = torch.from_numpy(np.asarray(record["state"][indices]).copy()).float()[:, None]
        augmentation = RoboTwin4RC._sample_augmentation(rng, color_augment=self.color_augment) if self.augment else None
        images, depths, masks, originals = [], [], [], []
        for frame in indices.tolist():
            rgb = self.root / episode / "images" / camera / f"{frame:06d}.png"
            depth = self.root / episode / "depths" / camera / f"{frame:06d}.png"
            try:
                with Image.open(rgb) as file:
                    image = file.convert("RGB")
                    if image.size != (320, 180):
                        raise ValueError(f"Expected 320x180 RGB: {rgb}")
                    if augmentation is not None:
                        image = RoboTwin4RC._apply_augmentation(image, augmentation)
                    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1
                with Image.open(depth) as file:
                    array = np.asarray(file, dtype=np.float32) / 1000
                if array.shape != (180, 320):
                    raise ValueError(f"Expected 320x180 depth: {depth}")
            except (OSError, ValueError) as error:
                raise RuntimeError(f"DROID read failed: episode={episode}, camera={camera}, frame={frame}: {error}") from error
            depth_tensor = torch.from_numpy(array)
            valid = torch.isfinite(depth_tensor) & (depth_tensor > 0)
            if self.max_depth is not None:
                valid &= depth_tensor <= self.max_depth
            images.append(F.pad(tensor, self.padding, mode="reflect"))
            depths.append(F.pad(torch.where(valid, depth_tensor, 0), self.padding))
            masks.append(F.pad(valid, self.padding, value=False))
            originals.append(F.pad(torch.ones_like(valid), self.padding, value=False))
        intrinsics = k.expand(len(indices), -1, -1).clone()
        centres, visible = project_tcp(state[..., :3], intrinsics, image_height=182, image_width=322, padding=self.padding)
        sample = {"images": torch.stack(images), "depth": torch.stack(depths), "valid_mask": torch.stack(masks),
                  "original_mask": torch.stack(originals), "intrinsics": intrinsics,
                  "extrinsics": ext.expand(len(indices), -1, -1).clone(),
                  "frame_indices": torch.tensor(indices.copy(), dtype=torch.long),
                  "frame_times": torch.tensor(indices / FPS, dtype=torch.float32), "frame_rate": FPS,
                  "tcp_state": state, "tcp_query_points": centres[0], "tcp_query_valid": visible[0],
                  "interval": interval, "task": "droid", "episode": episode, "camera_id": camera,
                  "source_size": torch.tensor([180, 320]), "padding": torch.tensor(self.padding)}
        return sample

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.get_sample(*index)
        return self.get_sample(index, self.min_views, self.epoch)

    def manifest(self):
        return {"split": self.split, "train_set": self.train_set, "val_set": self.val_set,
                "profile": self.profile, "camera_streams": len(self), "fps": FPS}

def _adapter(spec, config, *, split="train"):
    common = ("index_path", "train_set", "val_set", "min_views", "max_views", "min_interval", "max_interval",
              "reverse_probability", "frame_rate", "max_depth",
              "seed", "augment", "max_episodes", "cache_size", "max_tcp_linear_speed", "max_tcp_angular_speed")
    options = {key: config[key] for key in common if key in config}
    options.update(spec.get("options", {}))
    if config.get("max_episodes") is not None:
        options["max_episodes"] = config["max_episodes"]
    return DroidDataset(split=split, **options)
