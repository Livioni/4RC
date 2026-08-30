"""RoboTwin clip loader using the temporal sampling protocol from 4RC."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, Sampler, default_collate
from torchvision.transforms import functional as TVF


@dataclass(frozen=True, slots=True)
class RoboTwinEpisode:
    task: str
    name: str
    path: Path
    num_frames: int


class RoboTwin4RC(Dataset[dict[str, Any]]):
    """One temporally ordered ``head_view`` clip per episode and epoch.

    RoboTwin frames stay at their native 320x240 resolution. They are only
    padded to 322x252 so both dimensions are divisible by 4RC's patch size 14.
    """

    SOURCE_HEIGHT = 240
    SOURCE_WIDTH = 320
    PAD_LEFT = 1
    PAD_RIGHT = 1
    PAD_TOP = 6
    PAD_BOTTOM = 6
    PADDED_HEIGHT = 252
    PADDED_WIDTH = 322

    def __init__(
        self,
        root: str | Path,
        *,
        view: str = "head_view",
        min_views: int = 2,
        max_views: int = 18,
        min_interval: int = 1,
        max_interval: int = 5,
        seed: int = 42,
        augment: bool = True,
        max_episodes: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.view = view
        self.min_views = min_views
        self.max_views = max_views
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.seed = seed
        self.augment = augment
        self.epoch = 0

        if not self.root.is_dir():
            raise FileNotFoundError(f"RoboTwin root does not exist: {self.root}")
        if min_views < 2 or max_views < min_views:
            raise ValueError("Expected 2 <= min_views <= max_views")
        if min_interval < 1 or max_interval < min_interval:
            raise ValueError("Expected 1 <= min_interval <= max_interval")

        episodes: list[RoboTwinEpisode] = []
        for episode_path in sorted(self.root.glob("*/*")):
            if not episode_path.is_dir():
                continue
            rgb_dir = episode_path / "images" / view
            depth_dir = episode_path / "depths" / view
            intrinsics_file = episode_path / "intrinsics" / f"{view}.npy"
            extrinsics_file = episode_path / "extrinsics" / f"{view}.npy"
            if not (
                rgb_dir.is_dir()
                and depth_dir.is_dir()
                and intrinsics_file.is_file()
                and extrinsics_file.is_file()
            ):
                continue
            extrinsics = np.load(extrinsics_file, mmap_mode="r")
            num_frames = int(extrinsics.shape[0]) if extrinsics.ndim == 3 else 0
            if num_frames < min_views:
                continue
            first_rgb = rgb_dir / "000000.png"
            last_rgb = rgb_dir / f"{num_frames - 1:06d}.png"
            first_depth = depth_dir / "000000.png"
            last_depth = depth_dir / f"{num_frames - 1:06d}.png"
            if not all(path.is_file() for path in (first_rgb, last_rgb, first_depth, last_depth)):
                continue
            episodes.append(
                RoboTwinEpisode(
                    task=episode_path.parent.name,
                    name=episode_path.name,
                    path=episode_path,
                    num_frames=num_frames,
                )
            )

        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        if not episodes:
            raise RuntimeError(f"No valid RoboTwin episodes found below {self.root}")
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int, sample_seed: int | None = None) -> np.random.Generator:
        seed_parts = [self.seed, self.epoch, int(index)]
        if sample_seed is not None:
            seed_parts.append(int(sample_seed))
        seed_sequence = np.random.SeedSequence(seed_parts)
        return np.random.default_rng(seed_sequence)

    def sample_frame_indices(
        self,
        num_frames: int,
        rng: np.random.Generator,
        num_views: int | None = None,
    ) -> tuple[np.ndarray, int]:
        max_views = min(self.max_views, num_frames)
        min_views = min(self.min_views, max_views)
        fixed_num_views = num_views is not None
        if not fixed_num_views:
            num_views = int(rng.integers(min_views, max_views + 1))
        else:
            num_views = int(num_views)
            if not min_views <= num_views <= max_views:
                raise ValueError(
                    f"Cannot sample {num_views} views from {num_frames} frames "
                    f"with configured range [{self.min_views}, {self.max_views}]"
                )
        max_interval = min(self.max_interval, (num_frames - 1) // (num_views - 1))
        if max_interval < self.min_interval:
            if fixed_num_views:
                raise ValueError(
                    f"No valid interval for {num_views} views from {num_frames} frames"
                )
            num_views = min_views
            max_interval = min(self.max_interval, (num_frames - 1) // (num_views - 1))
        if max_interval < self.min_interval:
            raise ValueError(
                f"No valid interval for {num_views} views from {num_frames} frames"
            )
        interval = int(rng.integers(self.min_interval, max_interval + 1))
        last_offset = (num_views - 1) * interval
        start = int(rng.integers(0, num_frames - last_offset))
        return start + np.arange(num_views, dtype=np.int64) * interval, interval

    @staticmethod
    def _sample_augmentation(rng: np.random.Generator) -> dict[str, Any]:
        return {
            "blur": bool(rng.random() < 0.2),
            "blur_radius": float(rng.uniform(0.1, 2.0)),
            "jitter": bool(rng.random() < 0.1),
            "brightness": float(rng.uniform(0.8, 1.2)),
            "contrast": float(rng.uniform(0.8, 1.2)),
            "saturation": float(rng.uniform(0.8, 1.2)),
            "hue": float(rng.uniform(-0.05, 0.05)),
            "jitter_order": rng.permutation(4).tolist(),
            "grayscale": bool(rng.random() < 0.05),
        }

    @staticmethod
    def _apply_augmentation(image: Image.Image, params: dict[str, Any]) -> Image.Image:
        if params["blur"]:
            image = image.filter(ImageFilter.GaussianBlur(radius=params["blur_radius"]))
        if params["jitter"]:
            functions = (
                lambda x: TVF.adjust_brightness(x, params["brightness"]),
                lambda x: TVF.adjust_contrast(x, params["contrast"]),
                lambda x: TVF.adjust_saturation(x, params["saturation"]),
                lambda x: TVF.adjust_hue(x, params["hue"]),
            )
            for operation in params["jitter_order"]:
                image = functions[operation](image)
        if params["grayscale"]:
            image = TVF.rgb_to_grayscale(image, num_output_channels=3)
        return image

    def __getitem__(
        self, index: int | tuple[int, int] | tuple[int, int, int]
    ) -> dict[str, Any]:
        requested_views = None
        sample_seed = None
        if isinstance(index, tuple):
            if len(index) not in (2, 3):
                raise ValueError(
                    "Tuple indices must be (episode, num_views[, sample_seed])"
                )
            index, requested_views, *optional_seed = index
            sample_seed = optional_seed[0] if optional_seed else None
        index = int(index)
        episode = self.episodes[index]
        rng = self._rng(index, sample_seed)
        frame_indices, interval = self.sample_frame_indices(
            episode.num_frames, rng, requested_views
        )
        augmentation = self._sample_augmentation(rng) if self.augment else None

        intrinsics = np.load(episode.path / "intrinsics" / f"{self.view}.npy").astype(
            np.float32, copy=True
        )
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Invalid intrinsics in {episode.path}: {intrinsics.shape}")
        intrinsics[0, 2] += self.PAD_LEFT
        intrinsics[1, 2] += self.PAD_TOP
        extrinsics_all = np.load(
            episode.path / "extrinsics" / f"{self.view}.npy", mmap_mode="r"
        )

        images: list[torch.Tensor] = []
        depths: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        original_masks: list[torch.Tensor] = []
        for frame_index in frame_indices.tolist():
            rgb_path = episode.path / "images" / self.view / f"{frame_index:06d}.png"
            depth_path = episode.path / "depths" / self.view / f"{frame_index:06d}.png"
            with Image.open(rgb_path) as image_file:
                image = image_file.convert("RGB")
                if image.size != (self.SOURCE_WIDTH, self.SOURCE_HEIGHT):
                    raise ValueError(f"Expected 320x240 RoboTwin RGB, got {image.size}: {rgb_path}")
                if augmentation is not None:
                    image = self._apply_augmentation(image, augmentation)
                image_tensor = TVF.pil_to_tensor(image).float().div_(255.0)

            with Image.open(depth_path) as depth_file:
                depth_array = np.asarray(depth_file, dtype=np.float32)
            if depth_array.shape != (self.SOURCE_HEIGHT, self.SOURCE_WIDTH):
                raise ValueError(
                    f"Expected 240x320 RoboTwin depth, got {depth_array.shape}: {depth_path}"
                )
            depth_tensor = torch.from_numpy(depth_array / 1000.0)
            valid_mask = torch.isfinite(depth_tensor) & (depth_tensor > 0)
            depth_tensor = torch.where(valid_mask, depth_tensor, torch.zeros_like(depth_tensor))

            image_tensor = F.pad(
                image_tensor,
                (self.PAD_LEFT, self.PAD_RIGHT, self.PAD_TOP, self.PAD_BOTTOM),
                mode="reflect",
            )
            depth_tensor = F.pad(
                depth_tensor,
                (self.PAD_LEFT, self.PAD_RIGHT, self.PAD_TOP, self.PAD_BOTTOM),
                value=0,
            )
            valid_mask = F.pad(
                valid_mask,
                (self.PAD_LEFT, self.PAD_RIGHT, self.PAD_TOP, self.PAD_BOTTOM),
                value=False,
            )
            original_mask = torch.zeros_like(valid_mask)
            original_mask[
                self.PAD_TOP : self.PAD_TOP + self.SOURCE_HEIGHT,
                self.PAD_LEFT : self.PAD_LEFT + self.SOURCE_WIDTH,
            ] = True

            images.append(image_tensor.mul(2.0).sub(1.0))
            depths.append(depth_tensor)
            valid_masks.append(valid_mask)
            original_masks.append(original_mask)

        return {
            "images": torch.stack(images),
            "depth": torch.stack(depths),
            "valid_mask": torch.stack(valid_masks),
            "original_mask": torch.stack(original_masks),
            "intrinsics": torch.from_numpy(intrinsics).expand(len(images), -1, -1).clone(),
            "extrinsics": torch.from_numpy(np.asarray(extrinsics_all[frame_indices]).copy()),
            "frame_indices": torch.from_numpy(frame_indices),
            "interval": interval,
            "task": episode.task,
            "episode": episode.name,
            "source_size": torch.tensor([self.SOURCE_HEIGHT, self.SOURCE_WIDTH]),
            "padding": torch.tensor(
                [self.PAD_LEFT, self.PAD_RIGHT, self.PAD_TOP, self.PAD_BOTTOM]
            ),
        }


def collate_clips(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack clips after ensuring every scene in this batch has equal length."""
    if not samples:
        raise ValueError("Cannot collate an empty RoboTwin batch")
    sequence_lengths = {sample["images"].shape[0] for sample in samples}
    if len(sequence_lengths) != 1:
        raise ValueError(
            f"All clips in a batch must have equal length, got {sequence_lengths}"
        )
    return default_collate(samples)


def collate_single_clip(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Backward-compatible collate helper restricted to one clip."""
    if len(samples) != 1:
        raise ValueError("RoboTwin4RC requires batch_size=1 because clip length is variable")
    return collate_clips(samples)


def views_from_batch(batch: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
    """Convert a batched clip to the view-major input expected by ``Arc``."""
    images = batch["images"]
    return [{"img": images[:, view_index]} for view_index in range(images.shape[1])]


class EpochRandomSampler(Sampler[int]):
    """A shuffled order determined solely by ``seed`` and ``epoch``.

    This makes mid-epoch resume exact: recreating an epoch and skipping the
    completed batches yields the same remaining episode order.
    """

    def __init__(self, data_source: Dataset, seed: int = 42) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


class FixedImageBatchSampler(Sampler[list[tuple[int, int, int]]]):
    """DAN-style dynamic scene batches with a fixed image budget per device.

    For 18 images and scene counts (1, 2, 3, 6, 9), a yielded DataLoader batch
    contains 1x18, 2x9, 3x6, 6x3, or 9x2 scene-view layouts. Each tuple passed
    to the dataset is (episode_index, views_per_scene, deterministic_draw_seed).
    """

    def __init__(
        self,
        dataset: RoboTwin4RC,
        *,
        images_per_batch: int = 18,
        scene_counts: tuple[int, ...] | list[int] = (1, 2, 3, 6, 9),
        batches_per_epoch: int | None = None,
        recent_buffer_size: int = 10_000,
        seed: int = 42,
    ) -> None:
        if images_per_batch < 2:
            raise ValueError("images_per_batch must be at least 2")
        if not scene_counts:
            raise ValueError("scene_counts cannot be empty")
        if recent_buffer_size < 1:
            raise ValueError("recent_buffer_size must be positive")

        self.dataset = dataset
        self.images_per_batch = int(images_per_batch)
        self.scene_counts = tuple(int(value) for value in scene_counts)
        self.batches_per_epoch = (
            len(dataset) if batches_per_epoch is None else int(batches_per_epoch)
        )
        if self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive")
        self.recent_buffer_size = min(int(recent_buffer_size), len(dataset))
        self.seed = int(seed)
        self.epoch = 0

        combinations: list[tuple[int, int, np.ndarray]] = []
        for scene_count in self.scene_counts:
            if scene_count < 1 or self.images_per_batch % scene_count:
                raise ValueError(
                    f"scene_count={scene_count} must divide "
                    f"images_per_batch={self.images_per_batch}"
                )
            views_per_scene = self.images_per_batch // scene_count
            if not dataset.min_views <= views_per_scene <= dataset.max_views:
                continue
            eligible = np.asarray(
                [
                    index
                    for index, episode in enumerate(dataset.episodes)
                    if episode.num_frames >= views_per_scene
                ],
                dtype=np.int64,
            )
            if len(eligible) >= scene_count:
                combinations.append((scene_count, views_per_scene, eligible))

        if not combinations:
            raise ValueError(
                "No requested scene×view combination is feasible for this dataset"
            )
        self._combinations = combinations
        self.active_combinations = tuple(
            (scene_count, views_per_scene)
            for scene_count, views_per_scene, _ in combinations
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch])
        )
        recent_queue: deque[int] = deque()
        recent_counts = np.zeros(len(self.dataset), dtype=np.int64)
        max_seed = np.iinfo(np.int64).max

        for _ in range(self.batches_per_epoch):
            combination_index = int(rng.integers(len(self._combinations)))
            scene_count, views_per_scene, eligible = self._combinations[
                combination_index
            ]
            fresh = eligible[recent_counts[eligible] == 0]
            candidates = fresh if len(fresh) >= scene_count else eligible
            selected = rng.choice(candidates, size=scene_count, replace=False)
            draw_seeds = rng.integers(
                0, max_seed, size=scene_count, dtype=np.int64
            )

            for episode_index in selected.tolist():
                while len(recent_queue) >= self.recent_buffer_size:
                    expired = recent_queue.popleft()
                    recent_counts[expired] -= 1
                recent_queue.append(episode_index)
                recent_counts[episode_index] += 1

            yield [
                (int(episode_index), views_per_scene, int(draw_seed))
                for episode_index, draw_seed in zip(selected, draw_seeds)
            ]


def _unproject_depth_to_world(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project depth maps and return world points plus camera-to-world poses."""
    if depth.ndim != 3:
        raise ValueError(f"Expected depth [S,H,W], got {tuple(depth.shape)}")
    if intrinsics.shape != (depth.shape[0], 3, 3):
        raise ValueError(
            f"Expected intrinsics [S,3,3], got {tuple(intrinsics.shape)}"
        )
    if extrinsics_w2c.shape not in {
        (depth.shape[0], 3, 4),
        (depth.shape[0], 4, 4),
    }:
        raise ValueError(
            "Expected OpenCV world-to-camera extrinsics [S,3,4] or [S,4,4], "
            f"got {tuple(extrinsics_w2c.shape)}"
        )

    depth = depth.detach().cpu().float()
    intrinsics = intrinsics.detach().cpu().float()
    extrinsics_w2c = extrinsics_w2c.detach().cpu().float()
    sequence, height, width = depth.shape

    if extrinsics_w2c.shape[-2:] == (3, 4):
        bottom = torch.zeros(sequence, 1, 4, dtype=extrinsics_w2c.dtype)
        bottom[:, 0, 3] = 1
        extrinsics_w2c = torch.cat((extrinsics_w2c, bottom), dim=1)
    cameras_to_world = torch.linalg.inv(extrinsics_w2c)

    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype),
        torch.arange(width, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth
    x = (columns[None] - intrinsics[:, 0, 2, None, None]) * z
    x = x / intrinsics[:, 0, 0, None, None]
    y = (rows[None] - intrinsics[:, 1, 2, None, None]) * z
    y = y / intrinsics[:, 1, 1, None, None]
    camera_points = torch.stack((x, y, z), dim=-1)
    world_points = torch.einsum(
        "sij,shwj->shwi", cameras_to_world[:, :3, :3], camera_points
    )
    world_points += cameras_to_world[:, None, None, :3, 3]
    return world_points, cameras_to_world


def build_robotwin_glb_scene(
    sample: dict[str, Any],
    *,
    show_cameras: bool = True,
    max_points_per_view: int | None = 50_000,
    camera_size: float = 0.05,
    point_seed: int = 0,
):
    """Build a colored GT point-cloud scene from one unbatched RoboTwin sample.

    The input is the dictionary returned by RoboTwin4RC.__getitem__. Depth is
    back-projected with the padded intrinsics and OpenCV world-to-camera
    extrinsics. Set max_points_per_view=None to retain every valid pixel.
    """
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "GLB export requires trimesh; install project dependencies"
        ) from error

    required = {"images", "depth", "valid_mask", "intrinsics", "extrinsics"}
    missing = sorted(required.difference(sample))
    if missing:
        raise KeyError(f"RoboTwin sample is missing fields: {missing}")
    if max_points_per_view is not None and max_points_per_view <= 0:
        raise ValueError("max_points_per_view must be positive or None")
    if camera_size <= 0:
        raise ValueError("camera_size must be positive")

    images = sample["images"].detach().cpu().float()
    depth = sample["depth"].detach().cpu().float()
    valid_mask = sample["valid_mask"].detach().cpu().bool()
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected images [S,3,H,W], got {tuple(images.shape)}")
    if depth.shape != valid_mask.shape or images.shape[0] != depth.shape[0]:
        raise ValueError("images, depth and valid_mask must have matching views")
    if images.shape[-2:] != depth.shape[-2:]:
        raise ValueError("images and depth must have the same spatial resolution")

    colors = (
        images.add(1.0)
        .mul(127.5)
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    world_points, cameras_to_world = _unproject_depth_to_world(
        depth, sample["intrinsics"], sample["extrinsics"]
    )
    world_points = world_points.numpy()
    cameras_to_world = cameras_to_world.numpy()

    scene = trimesh.Scene()
    points_added = 0
    for view_index in range(depth.shape[0]):
        mask = valid_mask[view_index].numpy()
        mask &= np.isfinite(world_points[view_index]).all(axis=-1)
        flat_indices = np.flatnonzero(mask.reshape(-1))
        if max_points_per_view is not None and len(flat_indices) > max_points_per_view:
            generator = np.random.default_rng(point_seed + view_index)
            flat_indices = generator.choice(
                flat_indices, size=max_points_per_view, replace=False
            )
        if len(flat_indices) == 0:
            continue

        points = world_points[view_index].reshape(-1, 3)[flat_indices]
        point_colors = colors[view_index].reshape(-1, 3)[flat_indices]
        cloud = trimesh.PointCloud(points, colors=point_colors)
        scene.add_geometry(cloud, node_name=f"points_view_{view_index:03d}")
        points_added += len(points)

    if points_added == 0:
        raise ValueError("The sample contains no valid finite depth points")

    if show_cameras:
        from arc.dust3r.viz import CAM_COLORS, add_scene_cam

        intrinsics = sample["intrinsics"].detach().cpu().numpy()
        for view_index, pose_c2w in enumerate(cameras_to_world):
            focal = float(
                np.sqrt(
                    intrinsics[view_index, 0, 0]
                    * intrinsics[view_index, 1, 1]
                )
            )
            add_scene_cam(
                scene,
                pose_c2w,
                CAM_COLORS[view_index % len(CAM_COLORS)],
                image=colors[view_index],
                focal=focal,
                screen_width=camera_size,
            )
    return scene


def export_robotwin_sample_to_glb(
    sample: dict[str, Any],
    output_path: str | Path,
    **scene_kwargs: Any,
) -> Path:
    """Export one unbatched RoboTwin sample to a binary glTF file."""
    output_path = Path(output_path).expanduser()
    if output_path.suffix.lower() != ".glb":
        output_path = output_path.with_suffix(".glb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = build_robotwin_glb_scene(sample, **scene_kwargs)
    scene.export(file_obj=str(output_path), file_type="glb")
    return output_path


def visualize_scene(
    dataset: RoboTwin4RC,
    index: int = 0,
    output_path: str | Path = "robotwin.glb",
    **scene_kwargs: Any,
) -> Path:
    """Load one dataset item and export its GT RGB-D reconstruction as GLB."""
    return export_robotwin_sample_to_glb(
        dataset[index], output_path=output_path, **scene_kwargs
    )


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Export RoboTwin ground-truth RGB-D frames as a colored GLB point cloud"
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/RoboTwin"))
    parser.add_argument("--output", type=Path, default=Path("robotwin.glb"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points-per-view", type=int, default=50_000)
    parser.add_argument("--camera-size", type=float, default=0.05)
    parser.add_argument("--no-cameras", action="store_true")
    args = parser.parse_args()

    dataset = RoboTwin4RC(
        args.data_root,
        min_views=args.num_views,
        max_views=args.num_views,
        seed=args.seed,
        augment=False,
    )
    sample = dataset[args.index]
    output_path = export_robotwin_sample_to_glb(
        sample,
        args.output,
        show_cameras=not args.no_cameras,
        max_points_per_view=args.max_points_per_view,
        camera_size=args.camera_size,
        point_seed=args.seed,
    )
    frame_indices = sample["frame_indices"].tolist()
    print(
        f"Exported {sample['task']}/{sample['episode']} frames {frame_indices} "
        f"to {output_path.resolve()}"
    )


if __name__ == "__main__":
    _main()
