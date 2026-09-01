#!/usr/bin/env python3
"""Run geometry-only 4RC inference on RoboTwin RGB frames and export a GLB.

Example:
    python geometry_inference.py \
        --model checkpoints/RoboTwin-Geometry/model.safetensors \
        --input demo_outputs/robotwin_random \
        --output glb_outputs/geometry_scene_random.glb

The geometry-only checkpoint predicts depth and camera rays.  The script uses
the rays to recover camera parameters, unprojects every frame into the common
world coordinate system, and exports the fused colored point cloud.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.transforms import functional as TVF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SOURCE_WIDTH = 320
SOURCE_HEIGHT = 240
PAD_LEFT = 1
PAD_RIGHT = 1
PAD_TOP = 6
PAD_BOTTOM = 6
PADDED_WIDTH = 322
PADDED_HEIGHT = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer geometry from RoboTwin head-view RGB frames and export one "
            "static fused colored point cloud as GLB."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the geometry-only model.safetensors checkpoint",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing naturally ordered 320x240 RGB frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .glb file",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cuda, cuda:0, or cpu (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Inference autocast dtype (default: bfloat16 on supported CUDA)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Uniformly sample at most this many frames; 0 processes all frames",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=2.5,
        help="Discard points below this per-frame confidence percentile",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=50_000,
        help="Randomly retain at most this many points per frame; 0 keeps all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic point subsampling",
    )
    return parser.parse_args()


def _natural_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def collect_rgb_paths(input_dir: Path, max_frames: int = 0) -> list[Path]:
    input_dir = input_dir.expanduser()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"RGB input directory does not exist: {input_dir}")
    if max_frames < 0 or max_frames == 1:
        raise ValueError("--max-frames must be 0 or at least 2")

    paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=_natural_sort_key,
    )
    if len(paths) < 2:
        raise ValueError(
            f"Expected at least 2 PNG/JPEG RGB frames in {input_dir}, found {len(paths)}"
        )

    if max_frames and len(paths) > max_frames:
        indices = np.linspace(0, len(paths) - 1, max_frames, dtype=np.int64)
        paths = [paths[int(index)] for index in indices]
    return paths


def load_robotwin_views(
    paths: list[Path],
) -> tuple[list[dict[str, torch.Tensor]], list[np.ndarray]]:
    """Load native RoboTwin RGB and reproduce the geometry training padding."""
    views: list[dict[str, torch.Tensor]] = []
    colors: list[np.ndarray] = []

    for index, path in enumerate(paths):
        with Image.open(path) as image_file:
            image = ImageOps.exif_transpose(image_file).convert("RGB")
            if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise ValueError(
                    f"Expected every RGB frame to be {SOURCE_WIDTH}x{SOURCE_HEIGHT}; "
                    f"got {image.width}x{image.height}: {path}"
                )
            color = np.asarray(image, dtype=np.uint8).copy()
            image_tensor = TVF.pil_to_tensor(image).float().div_(255.0)

        image_tensor = F.pad(
            image_tensor,
            (PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM),
            mode="reflect",
        )
        image_tensor = image_tensor.mul_(2.0).sub_(1.0)
        if image_tensor.shape[-2:] != (PADDED_HEIGHT, PADDED_WIDTH):
            raise RuntimeError(
                f"Internal padding error: got {tuple(image_tensor.shape[-2:])}, "
                f"expected {(PADDED_HEIGHT, PADDED_WIDTH)}"
            )

        views.append(
            {
                "img": image_tensor.unsqueeze(0),
                "true_shape": torch.tensor([[PADDED_HEIGHT, PADDED_WIDTH]]),
                "idx": index,
                "instance": str(index),
            }
        )
        colors.append(color)
    return views, colors


def _remove_training_prefix(key: str) -> str:
    for prefix in ("model.", "module."):
        if key.startswith(prefix):
            return _remove_training_prefix(key[len(prefix) :])
    return key


def _is_allowed_missing_key(key: str) -> bool:
    if key.startswith(("tcp_query_encoder.", "tcp_track_head.")):
        return True
    # DualDPT reuses one LayerNorm instance in all four auxiliary projection
    # levels. safetensors stores the shared tensor once (under level 0), so
    # load_state_dict reports the other state-dict aliases as missing even
    # though the shared module has already received the checkpoint value.
    return re.fullmatch(
        r"head\.scratch\.output_conv2_aux\.[1-3]\.2\.(weight|bias)", key
    ) is not None


def load_geometry_model(model_path: Path, device: torch.device):
    model_path = model_path.expanduser()
    if model_path.name.endswith(".incomplete"):
        raise ValueError(
            f"Checkpoint download is incomplete: {model_path}. "
            "Wait for model.safetensors to finish downloading."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    if model_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint, got: {model_path}")

    from safetensors import SafetensorError
    from safetensors.torch import load_file

    from arc.models.arc.arc import Arc

    try:
        state_dict = load_file(str(model_path), device="cpu")
    except SafetensorError as error:
        raise ValueError(
            f"Could not read {model_path} as a complete safetensors checkpoint: {error}"
        ) from error

    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = _remove_training_prefix(key)
        if normalized_key in normalized_state_dict:
            raise ValueError(
                f"Checkpoint contains duplicate parameter after prefix removal: {normalized_key}"
            )
        normalized_state_dict[normalized_key] = value
    del state_dict

    model = Arc()
    try:
        missing, unexpected = model.load_state_dict(normalized_state_dict, strict=False)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint tensor shapes are incompatible with Arc: {error}") from error
    del normalized_state_dict
    gc.collect()

    incompatible_missing = [key for key in missing if not _is_allowed_missing_key(key)]
    if incompatible_missing or unexpected:
        details = []
        if incompatible_missing:
            details.append(f"missing keys: {incompatible_missing[:12]}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected[:12]}")
        raise ValueError("Incompatible geometry checkpoint; " + "; ".join(details))

    model = model.to(device).eval()
    if missing:
        print(
            f"Loaded geometry-only checkpoint; accepted {len(missing)} expected "
            "missing TCP/shared-alias keys from the modified Arc class."
        )
    return model


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested}) but is not available")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]
    if device.type != "cuda" and dtype != torch.float32:
        raise ValueError(f"{requested} inference is only supported on CUDA by this script")
    return dtype


def infer_geometry(
    model,
    views: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return depth, world points, confidence, and profiling for one clip."""
    device_views = [
        {**view, "img": view["img"].to(device, non_blocking=True)} for view in views
    ]
    autocast_context = (
        contextlib.nullcontext()
        if dtype == torch.float32
        else torch.autocast(device_type=device.type, dtype=dtype)
    )

    try:
        with torch.inference_mode():
            with autocast_context:
                raw_predictions, profiling = model(
                    device_views,
                    profiling=True,
                    force_no_output_conversion=True,
                    inference_track=False,
                    decode_camera=False,
                    decode_motion=False,
                    return_aux_pyramid=False,
                    ref_view_strategy="first",
                )

            height, width = raw_predictions["depth"].shape[-2:]
            model._process_ray_pose_estimation(raw_predictions, height, width)

            depth = raw_predictions["depth"][0].detach().float().cpu().numpy()
            confidence = (
                raw_predictions["depth_conf"][0].detach().float().cpu().numpy()
            )
            extrinsics_w2c = (
                raw_predictions["extrinsics"][0].detach().float().cpu().numpy()
            )
            intrinsics = (
                raw_predictions["intrinsics"][0].detach().float().cpu().numpy()
            )
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA ran out of memory while jointly processing {len(views)} frames. "
            "All frames must share one forward pass to remain in a common coordinate "
            "system; rerun with an explicit --max-frames value."
        ) from error

    from arc.models.arc.utils.geometry import unproject_depth_map_to_point_map

    world_points, _ = unproject_depth_map_to_point_map(
        depth[..., None], extrinsics_w2c, intrinsics
    )
    return depth, world_points, confidence, profiling


def fuse_point_cloud(
    depth: np.ndarray,
    world_points: np.ndarray,
    confidence: np.ndarray,
    colors: list[np.ndarray],
    *,
    confidence_percentile: float,
    max_points_per_frame: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if not 0.0 <= confidence_percentile < 100.0:
        raise ValueError("--confidence-percentile must be in [0, 100)")
    if max_points_per_frame < 0:
        raise ValueError("--max-points-per-frame cannot be negative")
    if not (
        depth.shape == confidence.shape
        and world_points.shape == depth.shape + (3,)
        and len(colors) == depth.shape[0]
    ):
        raise ValueError("Depth, points, confidence, and RGB frame shapes do not match")

    fused_points: list[np.ndarray] = []
    fused_colors: list[np.ndarray] = []
    frame_counts: list[int] = []

    row_slice = slice(PAD_TOP, PAD_TOP + SOURCE_HEIGHT)
    column_slice = slice(PAD_LEFT, PAD_LEFT + SOURCE_WIDTH)
    for frame_index, color in enumerate(colors):
        frame_depth = depth[frame_index, row_slice, column_slice]
        frame_points = world_points[frame_index, row_slice, column_slice]
        frame_confidence = confidence[frame_index, row_slice, column_slice]
        if color.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
            raise ValueError(
                f"Unexpected RGB shape for frame {frame_index}: {color.shape}"
            )

        valid = (
            np.isfinite(frame_depth)
            & (frame_depth > 0)
            & np.isfinite(frame_confidence)
            & np.isfinite(frame_points).all(axis=-1)
        )
        if not np.any(valid):
            frame_counts.append(0)
            continue

        threshold = np.percentile(
            frame_confidence[valid], confidence_percentile
        )
        valid &= frame_confidence >= threshold
        flat_indices = np.flatnonzero(valid.reshape(-1))
        if max_points_per_frame and len(flat_indices) > max_points_per_frame:
            generator = np.random.default_rng(seed + frame_index)
            flat_indices = generator.choice(
                flat_indices, size=max_points_per_frame, replace=False
            )

        fused_points.append(frame_points.reshape(-1, 3)[flat_indices])
        fused_colors.append(color.reshape(-1, 3)[flat_indices])
        frame_counts.append(len(flat_indices))

    if not fused_points:
        raise ValueError("No finite, positive-depth geometry remains after filtering")
    return (
        np.concatenate(fused_points, axis=0).astype(np.float32, copy=False),
        np.concatenate(fused_colors, axis=0).astype(np.uint8, copy=False),
        frame_counts,
    )


def export_glb(points: np.ndarray, colors: np.ndarray, output_path: Path) -> Path:
    try:
        import trimesh
    except ImportError as error:
        raise ImportError(
            "GLB export requires trimesh; install the project requirements"
        ) from error

    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError(
            f"Expected matching points/colors [N,3], got {points.shape} and {colors.shape}"
        )
    if not len(points):
        raise ValueError("Cannot export an empty point cloud")

    output_path = output_path.expanduser()
    if output_path.suffix.lower() != ".glb":
        output_path = output_path.with_suffix(".glb")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.PointCloud(points, colors=colors),
        node_name="fused_geometry",
        geom_name="fused_geometry",
    )
    scene.export(file_obj=str(output_path), file_type="glb")
    return output_path


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    paths = collect_rgb_paths(args.input, args.max_frames)

    print(f"Processing {len(paths)} RoboTwin RGB frames on {device} ({dtype})...")
    views, colors = load_robotwin_views(paths)
    model = load_geometry_model(args.model, device)
    depth, world_points, confidence, profiling = infer_geometry(
        model, views, device, dtype
    )
    points, point_colors, frame_counts = fuse_point_cloud(
        depth,
        world_points,
        confidence,
        colors,
        confidence_percentile=args.confidence_percentile,
        max_points_per_frame=args.max_points_per_frame,
        seed=args.seed,
    )
    output_path = export_glb(points, point_colors, args.output)

    elapsed = float(profiling.get("total_time", float("nan")))
    skipped_frames = sum(count == 0 for count in frame_counts)
    print(f"Inference time: {elapsed:.2f}s")
    print(
        f"Exported {len(points):,} colored points from {len(paths) - skipped_frames}/"
        f"{len(paths)} frames to {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
