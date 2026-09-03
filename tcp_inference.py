#!/usr/bin/env python3
"""Infer and visualize image-conditioned RoboTwin TCP trajectories.

The model takes a RoboTwin RGB clip plus left/right TCP image points in the
first frame. Query points use the original 320x240 pixel coordinate system and
are converted to the padded 4RC input coordinate system internally.

Example:

    conda run -n 4rc python tcp_inference.py \
        --input datasets/RoboTwin/<task>/<episode> \
        --tcp-query-points 120 140 205 138

    python tcp_inference.py \
        --input datasets/RoboTwin_random_subset/beat_block_hammer/episode_0000005 \
        --interactive \
        --frame-indices 15 20 25 30 35 40 45 50 55

"""

from __future__ import annotations

import argparse
import contextlib
import gc
import html
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from geometry_inference import (
    PAD_LEFT,
    PAD_TOP,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    load_robotwin_views,
    resolve_device,
    resolve_dtype,
)


DEFAULT_MODEL = Path("checkpoints/RoboTwin-TCP-Tracking/model.safetensors")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ARM_NAMES = ("left", "right")
ARM_COLORS = ((255, 96, 64), (45, 180, 255))
TCP_GROUND_TRUTH_DIRS = {
    "third_views": "TCP_third",
    "head_view": "TCP_head",
}
TRAIN_MAX_FRAMES = 18
TRAIN_MIN_INTERVAL = 1
TRAIN_MAX_INTERVAL = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run conditional RoboTwin TCP tracking and visualize every selected "
            "frame's geometry and two TCP poses in an interactive viser player."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "RoboTwin episode directory, or a directory containing naturally "
            "ordered 320x240 RGB frames"
        ),
    )
    parser.add_argument(
        "--view",
        default="third_views",
        help="Image view below an episode's images directory",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Image-conditioned RoboTwin-TCP-Tracking model.safetensors checkpoint",
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--tcp-query-points",
        type=float,
        nargs=4,
        metavar=("LEFT_X", "LEFT_Y", "RIGHT_X", "RIGHT_Y"),
        help="First-frame left/right TCP points in original 320x240 pixels",
    )
    query_group.add_argument(
        "--tcp-query-points-file",
        type=Path,
        help=".npy/.npz/.json query points with shape [2,2]",
    )
    query_group.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Open a web page to click the first-frame left/right TCP points, "
            "then run inference and embed the viser player"
        ),
    )
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        "--max-frames",
        type=int,
        default=18,
        help="Take at most this many frames from a fixed-interval clip; 0 uses the remainder",
    )
    frame_group.add_argument(
        "--frame-indices",
        type=int,
        nargs="+",
        help=(
            "Exact episode frame numbers in inference order, for example "
            "--frame-indices 10 15 20 25"
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        help="First numeric frame for automatic clip selection (default: first available)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Frame-number interval for automatic clip selection",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Inference autocast dtype",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=2.5,
        help="Initial per-frame point-cloud confidence percentile",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100_000,
        help="Randomly retain at most this many points per frame; 0 keeps all",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.0016,
        help="viser point size in world units",
    )
    parser.add_argument("--host", default="127.0.0.1", help="viser bind address")
    parser.add_argument("--port", type=int, default=8020, help="viser port")
    parser.add_argument(
        "--ui-host",
        default="127.0.0.1",
        help="Interactive selection-page bind address",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=7860,
        help="Interactive selection-page port",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optionally save the first-frame state and metadata as JSON",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Print/save results without starting viser",
    )
    return parser.parse_args()


def _natural_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def resolve_rgb_dir(input_path: Path, view: str) -> Path:
    """Resolve either an episode root or a direct RGB directory."""
    input_path = input_path.expanduser()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    episode_rgb_dir = input_path / "images" / view
    if episode_rgb_dir.is_dir():
        return episode_rgb_dir
    return input_path


def load_ground_truth_query_points(
    input_path: Path, view: str, frame_index: int
) -> np.ndarray:
    """Project the selected frame's RoboTwin TCP truth to original RGB pixels."""
    input_path = input_path.expanduser()
    episode_path = input_path
    episode_rgb_dir = episode_path / "images" / view
    if not episode_rgb_dir.is_dir():
        if input_path.name == view and input_path.parent.name == "images":
            episode_path = input_path.parent.parent
        else:
            raise FileNotFoundError(
                "Ground-truth TCP selection requires a RoboTwin episode directory "
                f"(or its images/{view} directory), got: {input_path}"
            )

    tcp_directory = TCP_GROUND_TRUTH_DIRS.get(view)
    if tcp_directory is None:
        supported = ", ".join(sorted(TCP_GROUND_TRUTH_DIRS))
        raise ValueError(
            f"No ground-truth TCP directory mapping for view {view!r}; "
            f"supported views: {supported}"
        )

    intrinsics_path = episode_path / "intrinsics" / f"{view}.npy"
    tcp_paths = [
        episode_path / tcp_directory / f"{arm_name}_state.npy"
        for arm_name in ARM_NAMES
    ]
    missing = [
        str(path)
        for path in (intrinsics_path, *tcp_paths)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Ground-truth TCP files are missing: " + ", ".join(missing)
        )

    intrinsics = np.load(intrinsics_path, allow_pickle=False)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(
            f"Expected finite intrinsics [3,3] in {intrinsics_path}, "
            f"got {intrinsics.shape}"
        )

    positions = []
    for arm_name, tcp_path in zip(ARM_NAMES, tcp_paths):
        states = np.load(tcp_path, mmap_mode="r", allow_pickle=False)
        if states.ndim != 2 or states.shape[1] < 3:
            raise ValueError(
                f"Expected {arm_name} TCP states [frames,>=3] in {tcp_path}, "
                f"got {states.shape}"
            )
        if not 0 <= frame_index < states.shape[0]:
            raise IndexError(
                f"Frame {frame_index} is outside {tcp_path} with "
                f"{states.shape[0]} frames"
            )
        positions.append(np.asarray(states[frame_index, :3], dtype=np.float32))
    xyz = np.stack(positions, axis=0)
    z = xyz[:, 2]
    if not np.isfinite(xyz).all() or np.any(z <= 0):
        raise ValueError(
            f"Ground-truth TCP must be finite and in front of the camera: {xyz.tolist()}"
        )

    u = intrinsics[0, 0] * xyz[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * xyz[:, 1] / z + intrinsics[1, 2]
    return _validate_query_points(
        np.stack((u, v), axis=-1),
        f"projected ground truth for frame {frame_index}",
    )


def collect_rgb_paths(
    input_dir: Path,
    max_frames: int,
    frame_indices: list[int] | None = None,
    *,
    start_frame: int | None = None,
    frame_interval: int = 1,
) -> list[Path]:
    """Collect an explicit sequence or a contiguous fixed-interval clip."""
    if max_frames < 0 or max_frames == 1:
        raise ValueError("--max-frames must be 0 or at least 2")
    if start_frame is not None and start_frame < 0:
        raise ValueError("--start-frame cannot be negative")
    if frame_interval < 1:
        raise ValueError("--frame-interval must be positive")

    all_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=_natural_sort_key,
    )
    paths_by_index: dict[int, Path] = {}
    for path in all_paths:
        try:
            index = _frame_index(path)
        except ValueError:
            continue
        if index in paths_by_index:
            raise ValueError(
                f"Multiple images in {input_dir} resolve to frame index {index}"
            )
        paths_by_index[index] = path

    if frame_indices is not None:
        if len(frame_indices) < 2:
            raise ValueError("--frame-indices requires at least 2 frame numbers")
        if len(set(frame_indices)) != len(frame_indices):
            raise ValueError("--frame-indices cannot contain duplicate frame numbers")
        if any(index < 0 for index in frame_indices):
            raise ValueError("--frame-indices cannot contain negative frame numbers")
        missing = [index for index in frame_indices if index not in paths_by_index]
        if missing:
            raise FileNotFoundError(
                f"Requested frame indices are missing from {input_dir}: {missing}"
            )
        return [paths_by_index[index] for index in frame_indices]

    if len(paths_by_index) < 2:
        raise ValueError(
            f"Expected at least 2 numerically named PNG/JPEG frames in {input_dir}, "
            f"found {len(paths_by_index)}"
        )
    first = min(paths_by_index) if start_frame is None else start_frame
    last = max(paths_by_index)
    requested = list(range(first, last + 1, frame_interval))
    if max_frames:
        requested = requested[:max_frames]
    if len(requested) < 2:
        raise ValueError(
            f"Fewer than 2 frames remain from start={first}, interval={frame_interval}"
        )
    missing = [index for index in requested if index not in paths_by_index]
    if missing:
        preview = missing[:12]
        suffix = "..." if len(missing) > len(preview) else ""
        raise FileNotFoundError(
            f"Fixed-interval clip is missing frames in {input_dir}: {preview}{suffix}"
        )
    return [paths_by_index[index] for index in requested]


def warn_if_out_of_training_distribution(paths: list[Path]) -> None:
    indices = np.asarray([_frame_index(path) for path in paths], dtype=np.int64)
    warnings: list[str] = []
    if len(indices) > TRAIN_MAX_FRAMES:
        warnings.append(
            f"{len(indices)} frames exceeds the training maximum {TRAIN_MAX_FRAMES}"
        )
    gaps = np.diff(indices)
    magnitudes = np.abs(gaps)
    if not (np.all(gaps > 0) or np.all(gaps < 0)):
        warnings.append("frame order is not strictly forward or strictly reversed")
    if len(set(magnitudes.tolist())) > 1:
        warnings.append(f"frame intervals are non-uniform: {magnitudes.tolist()}")
    if np.any(
        (magnitudes < TRAIN_MIN_INTERVAL) | (magnitudes > TRAIN_MAX_INTERVAL)
    ):
        warnings.append(
            f"frame intervals fall outside the training range "
            f"[{TRAIN_MIN_INTERVAL},{TRAIN_MAX_INTERVAL}]"
        )
    if warnings:
        print("Warning: clip is outside the TCP training distribution: " + "; ".join(warnings))


def _frame_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(
            f"Cannot infer the RoboTwin frame index from image name {path.name!r}; "
            "pass --tcp-query-points or --tcp-query-points-file explicitly"
        )
    return int(match.group(1))


def _validate_query_points(points: Any, source: str) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.shape != (2, 2):
        raise ValueError(
            f"TCP query points from {source} must have shape [2,2] "
            f"(left/right x/y), got {points_array.shape}"
        )
    if not np.isfinite(points_array).all():
        raise ValueError(f"TCP query points from {source} contain NaN/Inf")
    inside = (
        (points_array[:, 0] >= 0)
        & (points_array[:, 0] < SOURCE_WIDTH)
        & (points_array[:, 1] >= 0)
        & (points_array[:, 1] < SOURCE_HEIGHT)
    )
    if not inside.all():
        raise ValueError(
            f"TCP query points from {source} must lie inside "
            f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}: {points_array.tolist()}"
        )
    return points_array


def _load_query_points_file(path: Path) -> np.ndarray:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TCP query-points file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "tcp_query_points" in archive:
                points = archive["tcp_query_points"]
            elif len(archive.files) == 1:
                points = archive[archive.files[0]]
            else:
                raise ValueError(
                    f"{path} must contain 'tcp_query_points' when it has multiple keys"
                )
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "tcp_query_points" in payload:
            points = payload["tcp_query_points"]
        elif isinstance(payload, dict) and all(name in payload for name in ARM_NAMES):
            points = [payload[name] for name in ARM_NAMES]
        else:
            points = payload
    else:
        raise ValueError("--tcp-query-points-file must be .npy, .npz, or .json")
    points_array = np.asarray(points)
    if points_array.ndim == 3:
        points_array = points_array[0]
    return _validate_query_points(points_array, str(path))


def resolve_tcp_query_points(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.tcp_query_points is not None:
        points = np.asarray(args.tcp_query_points, dtype=np.float32).reshape(2, 2)
        return _validate_query_points(points, "--tcp-query-points"), "explicit CLI"
    assert args.tcp_query_points_file is not None
    return (
        _load_query_points_file(args.tcp_query_points_file),
        str(args.tcp_query_points_file),
    )


def add_interactive_query_point(
    points: list[list[float]], click_index: tuple[int, int] | list[int]
) -> list[list[float]]:
    """Append one original-image click in left-then-right arm order."""
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.size == 0:
        points_array = points_array.reshape(0, 2)
    if points_array.ndim != 2 or points_array.shape[1:] != (2,):
        raise ValueError(
            f"Interactive TCP points must have shape [N,2], got {points_array.shape}"
        )
    if len(points_array) >= len(ARM_NAMES):
        return points_array.tolist()

    click = np.asarray(click_index, dtype=np.float32)
    if click.shape != (2,) or not np.isfinite(click).all():
        raise ValueError(f"Invalid image click coordinate: {click_index!r}")
    x, y = click.tolist()
    if not (0 <= x < SOURCE_WIDTH and 0 <= y < SOURCE_HEIGHT):
        raise ValueError(
            f"Image click must lie inside {SOURCE_WIDTH}x{SOURCE_HEIGHT}, got {(x, y)}"
        )
    return [*points_array.tolist(), [x, y]]


def render_query_overlay(
    image: np.ndarray, points: list[list[float]]
) -> np.ndarray:
    """Draw the selected left/right TCP pixels without changing image size."""
    image_array = np.asarray(image, dtype=np.uint8)
    if image_array.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError(
            f"Expected an RGB image with shape {(SOURCE_HEIGHT, SOURCE_WIDTH, 3)}, "
            f"got {image_array.shape}"
        )
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.size == 0:
        return image_array.copy()
    if points_array.ndim != 2 or points_array.shape[1:] != (2,):
        raise ValueError(
            f"Interactive TCP points must have shape [N,2], got {points_array.shape}"
        )

    canvas = Image.fromarray(image_array.copy())
    draw = ImageDraw.Draw(canvas)
    radius = 6
    for arm_index, point in enumerate(points_array[: len(ARM_NAMES)]):
        x, y = (int(round(float(value))) for value in point)
        color = ARM_COLORS[arm_index]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=3,
        )
        draw.line((x - radius - 2, y, x + radius + 2, y), fill=color, width=2)
        draw.line((x, y - radius - 2, x, y + radius + 2), fill=color, width=2)
        label = "L" if arm_index == 0 else "R"
        label_x = min(max(x + radius + 3, 1), SOURCE_WIDTH - 12)
        label_y = min(max(y - radius - 7, 1), SOURCE_HEIGHT - 12)
        draw.rectangle(
            (label_x - 1, label_y - 1, label_x + 9, label_y + 10), fill=(0, 0, 0)
        )
        draw.text((label_x, label_y), label, fill=color)
    return np.asarray(canvas, dtype=np.uint8)


def _interactive_selection_status(points: list[list[float]]) -> str:
    if not points:
        return "请在图中点击 **左臂 TCP**。"
    left = points[0]
    if len(points) == 1:
        return (
            f"左臂已选：`({left[0]:.1f}, {left[1]:.1f})`。"
            "现在请点击 **右臂 TCP**。"
        )
    right = points[1]
    return (
        f"左臂：`({left[0]:.1f}, {left[1]:.1f})`  "
        f"右臂：`({right[0]:.1f}, {right[1]:.1f})`  "
        "两点已就绪，可以开始推理；如需修改请先重置。"
    )


def _remove_training_prefix(key: str) -> str:
    for prefix in ("model.", "module."):
        if key.startswith(prefix):
            return _remove_training_prefix(key[len(prefix) :])
    return key


def _is_shared_alias_key(key: str) -> bool:
    return re.fullmatch(
        r"head\.scratch\.output_conv2_aux\.[1-3]\.2\.(weight|bias)", key
    ) is not None


def load_tcp_model(model_path: Path, device: torch.device):
    model_path = model_path.expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {model_path}")
    if model_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint, got: {model_path}")

    from safetensors import SafetensorError
    from safetensors.torch import load_file

    from arc.models.arc.arc import Arc

    try:
        saved_state = load_file(str(model_path), device="cpu")
    except SafetensorError as error:
        raise ValueError(f"Could not read checkpoint {model_path}: {error}") from error

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in saved_state.items():
        normalized_key = _remove_training_prefix(key)
        if normalized_key in state_dict:
            raise ValueError(
                f"Duplicate checkpoint parameter after prefix removal: {normalized_key}"
            )
        state_dict[normalized_key] = value
    del saved_state

    tcp_prefixes = ("tcp_visual_query_encoder.", "tcp_track_head.")
    if not all(any(key.startswith(prefix) for key in state_dict) for prefix in tcp_prefixes):
        raise ValueError(
            f"{model_path} is not an image-conditioned TCP checkpoint"
        )
    offset_key = "tcp_visual_query_encoder.offset_embedding"
    window_tokens = int(state_dict[offset_key].shape[0])
    window_size = math.isqrt(window_tokens)
    if window_size * window_size != window_tokens:
        raise ValueError(f"Invalid TCP query window token count: {window_tokens}")

    model = Arc(tcp_query_window_size=window_size)
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint tensor shapes are incompatible with Arc: {error}") from error
    del state_dict
    gc.collect()

    incompatible_missing = [key for key in missing if not _is_shared_alias_key(key)]
    if incompatible_missing or unexpected:
        details = []
        if incompatible_missing:
            details.append(f"missing keys: {incompatible_missing[:12]}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected[:12]}")
        raise ValueError("Incompatible TCP checkpoint; " + "; ".join(details))
    return model.to(device).eval()


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Inverse of RoboTwin's Rz(yaw) @ Ry(pitch) @ Rx(roll) convention."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [...,3,3], got {rotation.shape}")
    horizontal = np.sqrt(rotation[..., 0, 0] ** 2 + rotation[..., 1, 0] ** 2)
    singular = horizontal < 1e-7

    roll = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = np.arctan2(-rotation[..., 2, 0], horizontal)
    yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
    singular_roll = np.arctan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    roll = np.where(singular, singular_roll, roll)
    yaw = np.where(singular, 0.0, yaw)
    return np.stack((roll, pitch, yaw), axis=-1).astype(np.float32)


def infer_tcp_and_geometry(
    model,
    views: list[dict[str, torch.Tensor]],
    query_points: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, Any],
]:
    device_views = [
        {**view, "img": view["img"].to(device, non_blocking=True)} for view in views
    ]
    padded_points = query_points + np.asarray([PAD_LEFT, PAD_TOP], dtype=np.float32)
    query_tensor = torch.from_numpy(padded_points).unsqueeze(0).to(device)
    autocast_context = (
        contextlib.nullcontext()
        if dtype == torch.float32
        else torch.autocast(device_type=device.type, dtype=dtype)
    )

    try:
        with torch.inference_mode():
            with autocast_context:
                predictions, profiling = model(
                    device_views,
                    profiling=True,
                    force_no_output_conversion=True,
                    inference_track=False,
                    decode_camera=False,
                    decode_motion=False,
                    tcp_query_points=query_tensor,
                    decode_tcp=True,
                    return_aux_pyramid=False,
                    ref_view_strategy="first",
                )

            height, width = predictions["depth"].shape[-2:]
            model._process_ray_pose_estimation(predictions, height, width)
            depth = predictions["depth"][0].detach().float().cpu().numpy()
            confidence = predictions["depth_conf"][0].detach().float().cpu().numpy()
            extrinsics = predictions["extrinsics"][0].detach().float().cpu().numpy()
            intrinsics = predictions["intrinsics"][0].detach().float().cpu().numpy()
            tcp = {
                "position": predictions["tcp_position"][0].detach().float().cpu().numpy(),
                "rotation": predictions["tcp_rotation"][0].detach().float().cpu().numpy(),
                "gripper": predictions["tcp_gripper_logit"][0]
                .sigmoid()
                .detach()
                .float()
                .cpu()
                .numpy(),
                "confidence": predictions["tcp_confidence"][0]
                .detach()
                .float()
                .cpu()
                .numpy(),
            }
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA ran out of memory while processing {len(views)} frames; "
            "reduce --max-frames"
        ) from error

    from arc.models.arc.utils.geometry import unproject_depth_map_to_point_map

    world_points, _ = unproject_depth_map_to_point_map(
        depth[..., None], extrinsics, intrinsics
    )
    return depth, world_points, confidence, tcp, extrinsics, profiling


def prepare_frame_point_clouds(
    depth: np.ndarray,
    world_points: np.ndarray,
    confidence: np.ndarray,
    colors: list[np.ndarray],
    *,
    max_points_per_frame: int,
    seed: int = 0,
) -> list[dict[str, np.ndarray]]:
    """Crop padding and cache each frame's valid points sorted by confidence."""
    if max_points_per_frame < 0:
        raise ValueError("--max-points cannot be negative")
    if not (
        depth.shape == confidence.shape
        and world_points.shape == depth.shape + (3,)
        and len(colors) == depth.shape[0]
    ):
        raise ValueError("Depth, points, confidence, and RGB frame shapes do not match")

    row_slice = slice(PAD_TOP, PAD_TOP + SOURCE_HEIGHT)
    column_slice = slice(PAD_LEFT, PAD_LEFT + SOURCE_WIDTH)
    frame_clouds: list[dict[str, np.ndarray]] = []
    for frame_slot, color in enumerate(colors):
        frame_depth = depth[frame_slot, row_slice, column_slice]
        frame_points = world_points[frame_slot, row_slice, column_slice]
        frame_confidence = confidence[frame_slot, row_slice, column_slice]
        if color.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
            raise ValueError(
                f"Unexpected RGB shape for frame slot {frame_slot}: {color.shape}"
            )

        valid = (
            np.isfinite(frame_depth)
            & (frame_depth > 0)
            & np.isfinite(frame_confidence)
            & np.isfinite(frame_points).all(axis=-1)
        )
        flat_indices = np.flatnonzero(valid.reshape(-1))
        if not len(flat_indices):
            raise ValueError(f"Frame slot {frame_slot} has no valid geometry")
        if max_points_per_frame and len(flat_indices) > max_points_per_frame:
            generator = np.random.default_rng(seed + frame_slot)
            flat_indices = generator.choice(
                flat_indices, size=max_points_per_frame, replace=False
            )

        points = frame_points.reshape(-1, 3)[flat_indices]
        point_colors = color.reshape(-1, 3)[flat_indices]
        point_confidence = frame_confidence.reshape(-1)[flat_indices]
        order = np.argsort(point_confidence, kind="stable")[::-1]
        frame_clouds.append(
            {
                "points": points[order].astype(np.float32, copy=False),
                "colors": point_colors[order].astype(np.uint8, copy=False),
                "confidence": point_confidence[order].astype(np.float32, copy=False),
            }
        )
    return frame_clouds


def build_first_frame_result(
    tcp: dict[str, np.ndarray],
    *,
    query_points: np.ndarray,
    query_source: str,
    image_path: Path,
    inference_seconds: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    position = tcp["position"][0]
    rotation = tcp["rotation"][0]
    rpy = matrix_to_rpy(rotation)
    gripper = tcp["gripper"][0]
    confidence = tcp["confidence"][0]

    arms: dict[str, Any] = {}
    for arm_index, arm_name in enumerate(ARM_NAMES):
        arms[arm_name] = {
            "xyz_m": position[arm_index].tolist(),
            "rpy_rad": rpy[arm_index].tolist(),
            "rpy_deg": np.rad2deg(rpy[arm_index]).tolist(),
            "gripper": float(gripper[arm_index]),
            "confidence": float(confidence[arm_index]),
        }

    result = {
        "frame": str(image_path),
        "query_source": query_source,
        "note": "Image-conditioned prediction from first-frame TCP pixel queries.",
        "tcp_query_points": query_points.tolist(),
        "predicted_first_frame": arms,
        "inference_seconds": inference_seconds,
    }
    return result, position, rotation


def _filtered_cloud(
    frame_cloud: dict[str, np.ndarray],
    confidence_percentile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not 0.0 <= confidence_percentile <= 100.0:
        raise ValueError("Confidence percentile must be in [0,100]")
    total = len(frame_cloud["points"])
    keep_count = max(
        1, int(math.ceil(total * (100.0 - confidence_percentile) / 100.0))
    )
    threshold = float(frame_cloud["confidence"][keep_count - 1])
    return (
        frame_cloud["points"][:keep_count],
        frame_cloud["colors"][:keep_count],
        threshold,
    )


def _format_frame_panel(
    tcp: dict[str, np.ndarray],
    *,
    frame_slot: int,
    frame_index: int,
    image_path: Path,
    num_frames: int,
    confidence_percentile: float,
    visible_points: int,
    confidence_threshold: float,
) -> str:
    rpy = matrix_to_rpy(tcp["rotation"][frame_slot])
    lines = [
        f"# Frame {frame_slot + 1}/{num_frames}",
        f"- **Episode frame:** `{frame_index}`",
        f"- **Image:** `{image_path.name}`",
        (
            f"- **Geometry:** `{visible_points:,}` points, top "
            f"`{100.0 - confidence_percentile:.1f}%` "
            f"(confidence ≥ `{confidence_threshold:.4f}`)"
        ),
    ]
    for arm_index, arm_name in enumerate(ARM_NAMES):
        xyz_text = ", ".join(
            f"{value:+.4f}" for value in tcp["position"][frame_slot, arm_index]
        )
        rpy_text = ", ".join(f"{value:+.4f}" for value in rpy[arm_index])
        lines.extend(
            (
                f"## {arm_name.title()} TCP",
                f"- **xyz (m):** `{xyz_text}`",
                f"- **rpy (rad):** `{rpy_text}`",
                (
                    f"- **gripper:** "
                    f"`{tcp['gripper'][frame_slot, arm_index]:.4f}`"
                ),
                (
                    f"- **confidence:** "
                    f"`{tcp['confidence'][frame_slot, arm_index]:.4f}`"
                ),
            )
        )
    return "\n".join(lines)


def compute_default_third_camera_view(
    extrinsic_w2c: np.ndarray,
    scene_center: np.ndarray,
    scene_extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place the viewer just behind an OpenCV camera, looking along its +Z axis."""
    extrinsic = np.asarray(extrinsic_w2c, dtype=np.float64)
    center = np.asarray(scene_center, dtype=np.float64)
    if extrinsic.shape not in ((3, 4), (4, 4)):
        raise ValueError(
            f"Expected a [3,4] or [4,4] world-to-camera matrix, got {extrinsic.shape}"
        )
    if center.shape != (3,):
        raise ValueError(f"Expected a scene center [3], got {center.shape}")
    if not np.isfinite(extrinsic).all() or not np.isfinite(center).all():
        raise ValueError("Camera extrinsic and scene center must be finite")
    if not math.isfinite(scene_extent) or scene_extent <= 0:
        raise ValueError("Scene extent must be finite and positive")

    rotation_w2c = extrinsic[:3, :3]
    translation_w2c = extrinsic[:3, 3]
    rotation_c2w = rotation_w2c.T
    camera_center = -rotation_c2w @ translation_w2c
    forward = rotation_c2w[:, 2]
    up = -rotation_c2w[:, 1]
    forward_norm = np.linalg.norm(forward)
    up_norm = np.linalg.norm(up)
    if forward_norm < 1e-8 or up_norm < 1e-8:
        raise ValueError("Camera extrinsic has a degenerate rotation")
    forward /= forward_norm
    up /= up_norm

    backoff = float(np.clip(scene_extent * 0.15, 0.05, 0.25))
    target_distance = float(np.dot(center - camera_center, forward))
    target_distance = max(target_distance, scene_extent * 0.5, 0.1)
    viewer_position = camera_center - forward * backoff
    look_at = camera_center + forward * target_distance
    return (
        viewer_position.astype(np.float32),
        look_at.astype(np.float32),
        up.astype(np.float32),
    )


def start_visualization(
    frame_clouds: list[dict[str, np.ndarray]],
    tcp: dict[str, np.ndarray],
    frame_indices: list[int],
    image_paths: list[Path],
    *,
    host: str,
    port: int,
    point_size: float,
    confidence_percentile: float,
    reference_extrinsic_w2c: np.ndarray | None = None,
):
    try:
        import viser
        import viser.transforms as tf
    except ImportError as error:
        raise ImportError(
            "viser is required for visualization; install requirements.txt"
        ) from error

    num_frames = len(frame_clouds)
    if not (
        num_frames
        == len(frame_indices)
        == len(image_paths)
        == tcp["position"].shape[0]
        == tcp["rotation"].shape[0]
        == tcp["gripper"].shape[0]
        == tcp["confidence"].shape[0]
    ):
        raise ValueError("Geometry, TCP, image, and frame-index lengths do not match")

    bounds_samples: list[np.ndarray] = []
    for frame_cloud in frame_clouds:
        high_confidence_count = max(1, int(len(frame_cloud["points"]) * 0.95))
        high_confidence_points = frame_cloud["points"][:high_confidence_count]
        sample_step = max(1, len(high_confidence_points) // 10_000)
        bounds_samples.append(high_confidence_points[::sample_step])
    bounds_samples.append(tcp["position"].reshape(-1, 3))
    bounds_points = np.concatenate(bounds_samples, axis=0)
    bounds_min = np.percentile(bounds_points, 1.0, axis=0)
    bounds_max = np.percentile(bounds_points, 99.0, axis=0)
    center = (bounds_min + bounds_max) * 0.5
    extent = max(float(np.max(bounds_max - bounds_min)), 0.1)
    axes_length = float(np.clip(extent * 0.08, 0.025, 0.12))

    default_camera_position = center + extent * np.array(
        [0.0, 0.0, -1.2], dtype=np.float32
    )
    default_camera_look_at = center
    default_camera_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if reference_extrinsic_w2c is not None:
        try:
            (
                default_camera_position,
                default_camera_look_at,
                default_camera_up,
            ) = compute_default_third_camera_view(
                reference_extrinsic_w2c, center, extent
            )
        except ValueError as error:
            print(
                "Warning: could not initialize behind the third camera; "
                f"using the scene fallback view: {error}"
            )

    server = viser.ViserServer(host=host, port=port)
    server.gui.set_panel_label("Geometry + TCP")
    server.gui.configure_theme(
        control_layout="floating", control_width="large", show_logo=False
    )
    server.scene.set_up_direction(tuple(default_camera_up.tolist()))
    server.scene.world_axes.visible = True

    with server.gui.add_folder("Playback", expand_by_default=True):
        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )
        gui_previous = server.gui.add_button("Previous")
        gui_next = server.gui.add_button("Next")
        gui_playing = server.gui.add_checkbox("Play", False)
        gui_fps = server.gui.add_slider(
            "FPS", min=0.25, max=30.0, step=0.25, initial_value=5.0
        )

    with server.gui.add_folder("Geometry", expand_by_default=True):
        gui_confidence = server.gui.add_slider(
            "Confidence percentile",
            min=0.0,
            max=99.0,
            step=0.5,
            initial_value=float(confidence_percentile),
        )
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=1e-5,
            max=0.01,
            step=1e-5,
            initial_value=point_size,
        )
        gui_show_points = server.gui.add_checkbox("Show point cloud", True)

    with server.gui.add_folder("TCP", expand_by_default=True):
        gui_show_tcp = server.gui.add_checkbox("Show TCP spheres + axes", True)
        gui_show_labels = server.gui.add_checkbox("Show TCP labels", True)

    initial_points, _, initial_threshold = _filtered_cloud(
        frame_clouds[0], confidence_percentile
    )
    gui_info = server.gui.add_markdown(
        _format_frame_panel(
            tcp,
            frame_slot=0,
            frame_index=frame_indices[0],
            image_path=image_paths[0],
            num_frames=num_frames,
            confidence_percentile=confidence_percentile,
            visible_points=len(initial_points),
            confidence_threshold=initial_threshold,
        )
    )

    frame_handles: list[dict[str, Any]] = []
    for frame_slot, frame_cloud in enumerate(frame_clouds):
        filtered_points, filtered_colors, _ = _filtered_cloud(
            frame_cloud, confidence_percentile
        )
        root_node = server.scene.add_frame(
            f"/frames/t{frame_slot}", show_axes=False, visible=frame_slot == 0
        )
        point_node = server.scene.add_point_cloud(
            f"/frames/t{frame_slot}/geometry",
            points=filtered_points,
            colors=filtered_colors,
            point_size=point_size,
            point_shape="rounded",
            visible=frame_slot == 0,
        )

        tcp_frames = []
        tcp_spheres = []
        tcp_labels = []
        for arm_index, (arm_name, arm_color) in enumerate(
            zip(ARM_NAMES, ARM_COLORS)
        ):
            quaternion = tf.SO3.from_matrix(
                tcp["rotation"][frame_slot, arm_index]
            ).wxyz
            tcp_frame = server.scene.add_frame(
                f"/frames/t{frame_slot}/tcp/{arm_name}",
                wxyz=quaternion,
                position=tcp["position"][frame_slot, arm_index],
                axes_length=axes_length,
                axes_radius=axes_length * 0.035,
                origin_radius=axes_length * 0.075,
                origin_color=arm_color,
                visible=frame_slot == 0,
            )
            tcp_sphere = server.scene.add_icosphere(
                f"/frames/t{frame_slot}/tcp/{arm_name}/xyz",
                radius=axes_length * 0.11,
                color=arm_color,
                position=(0.0, 0.0, 0.0),
                visible=frame_slot == 0,
            )
            tcp_label = server.scene.add_label(
                f"/frames/t{frame_slot}/tcp/{arm_name}/label",
                text=(
                    f"{arm_name}: "
                    f"gripper={tcp['gripper'][frame_slot, arm_index]:.3f}"
                ),
                position=(0.0, 0.0, axes_length * 1.25),
                anchor="bottom-center",
                visible=frame_slot == 0,
            )
            tcp_frames.append(tcp_frame)
            tcp_spheres.append(tcp_sphere)
            tcp_labels.append(tcp_label)

        frame_handles.append(
            {
                "root": root_node,
                "points": point_node,
                "tcp_frames": tcp_frames,
                "tcp_spheres": tcp_spheres,
                "tcp_labels": tcp_labels,
                "filter_percentile": float(confidence_percentile),
            }
        )

    def _apply_confidence_filter(frame_slot: int) -> tuple[int, float]:
        frame_handle = frame_handles[frame_slot]
        percentile = float(gui_confidence.value)
        points, colors, threshold = _filtered_cloud(
            frame_clouds[frame_slot], percentile
        )
        if frame_handle["filter_percentile"] != percentile:
            frame_handle["points"].points = points
            frame_handle["points"].colors = colors
            frame_handle["filter_percentile"] = percentile
        return len(points), threshold

    def _update_panel(frame_slot: int, visible_points: int, threshold: float) -> None:
        gui_info.content = _format_frame_panel(
            tcp,
            frame_slot=frame_slot,
            frame_index=frame_indices[frame_slot],
            image_path=image_paths[frame_slot],
            num_frames=num_frames,
            confidence_percentile=float(gui_confidence.value),
            visible_points=visible_points,
            confidence_threshold=threshold,
        )

    def _update_visibility() -> None:
        current = int(gui_frame.value)
        with server.atomic():
            for frame_slot, handles in enumerate(frame_handles):
                active = frame_slot == current
                handles["root"].visible = active
                handles["points"].visible = active and bool(gui_show_points.value)
                for tcp_frame, tcp_sphere, tcp_label in zip(
                    handles["tcp_frames"],
                    handles["tcp_spheres"],
                    handles["tcp_labels"],
                ):
                    tcp_frame.visible = active and bool(gui_show_tcp.value)
                    tcp_sphere.visible = active and bool(gui_show_tcp.value)
                    tcp_label.visible = (
                        active
                        and bool(gui_show_tcp.value)
                        and bool(gui_show_labels.value)
                    )
        server.flush()

    @gui_previous.on_click
    def _previous_frame(_) -> None:
        gui_frame.value = (int(gui_frame.value) - 1) % num_frames

    @gui_next.on_click
    def _next_frame(_) -> None:
        gui_frame.value = (int(gui_frame.value) + 1) % num_frames

    @gui_playing.on_update
    def _toggle_playing(_) -> None:
        playing = bool(gui_playing.value)
        gui_frame.disabled = playing
        gui_previous.disabled = playing
        gui_next.disabled = playing

    @gui_frame.on_update
    def _select_frame(_) -> None:
        current = int(gui_frame.value)
        visible_points, threshold = _apply_confidence_filter(current)
        _update_panel(current, visible_points, threshold)
        _update_visibility()

    @gui_confidence.on_update
    def _change_confidence(_) -> None:
        current = int(gui_frame.value)
        visible_points, threshold = _apply_confidence_filter(current)
        _update_panel(current, visible_points, threshold)
        server.flush()

    @gui_point_size.on_update
    def _change_point_size(_) -> None:
        with server.atomic():
            for handles in frame_handles:
                handles["points"].point_size = float(gui_point_size.value)
        server.flush()

    @gui_show_points.on_update
    def _toggle_points(_) -> None:
        _update_visibility()

    @gui_show_tcp.on_update
    def _toggle_tcp(_) -> None:
        _update_visibility()

    @gui_show_labels.on_update
    def _toggle_labels(_) -> None:
        _update_visibility()

    @server.on_client_connect
    def _set_initial_camera(client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = tuple(default_camera_position.tolist())
            client.camera.look_at = tuple(default_camera_look_at.tolist())
            client.camera.up_direction = tuple(default_camera_up.tolist())
        client.flush()

    playback_stop_event = threading.Event()
    setattr(server, "_tcp_playback_stop_event", playback_stop_event)

    def _playback_loop() -> None:
        while not playback_stop_event.is_set():
            if bool(gui_playing.value):
                gui_frame.value = (int(gui_frame.value) + 1) % num_frames
            playback_stop_event.wait(timeout=1.0 / float(gui_fps.value))

    playback_thread = threading.Thread(target=_playback_loop, daemon=True)
    playback_thread.start()
    _update_visibility()
    return server


def stop_visualization(server: Any | None) -> None:
    """Stop a TCP viser server and its playback worker."""
    if server is None:
        return
    stop_event = getattr(server, "_tcp_playback_stop_event", None)
    if stop_event is not None:
        stop_event.set()
    server.stop()


def start_interactive_page(
    *,
    args: argparse.Namespace,
    paths: list[Path],
    selected_indices: list[int],
    views: list[dict[str, torch.Tensor]],
    colors: list[np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Launch click-to-query UI and embed the resulting viser player."""
    try:
        import gradio as gr
    except ImportError as error:
        raise ImportError(
            "gradio is required for --interactive; install requirements.txt"
        ) from error

    base_image = colors[0].copy()
    runtime: dict[str, Any] = {"model": None, "viser": None}
    runtime_lock = threading.Lock()

    try:
        ground_truth_points = load_ground_truth_query_points(
            args.input, args.view, selected_indices[0]
        )
        ground_truth_message = (
            "已找到首帧左右臂真值 TCP，可点击“使用首帧真值 TCP”"
            "自动选择。"
        )
    except (OSError, ValueError, IndexError) as error:
        ground_truth_points = None
        ground_truth_message = f"当前输入无法使用真值 TCP：`{error}`"

    def _select_point(
        points: list[list[float]], event: gr.SelectData
    ) -> tuple[np.ndarray, list[list[float]], str, str, Any]:
        updated = add_interactive_query_point(points, event.index)
        return (
            render_query_overlay(base_image, updated),
            updated,
            "interactive first-frame clicks",
            _interactive_selection_status(updated),
            gr.update(interactive=len(updated) == len(ARM_NAMES)),
        )

    # Gradio inspects this concrete type to inject SelectData. The source file
    # uses postponed annotations while gradio is intentionally imported lazily.
    _select_point.__annotations__["event"] = gr.SelectData

    def _reset_points() -> tuple[np.ndarray, list[list[float]], str, str, Any]:
        return (
            base_image.copy(),
            [],
            "interactive first-frame clicks",
            _interactive_selection_status([]),
            gr.update(interactive=False),
        )

    def _use_ground_truth() -> tuple[
        np.ndarray, list[list[float]], str, str, Any
    ]:
        if ground_truth_points is None:
            raise ValueError("Ground-truth TCP points are unavailable")
        points = ground_truth_points.tolist()
        return (
            render_query_overlay(base_image, points),
            points,
            "projected first-frame ground-truth TCP",
            "已使用首帧 **真值 TCP**。  " + _interactive_selection_status(points),
            gr.update(interactive=True),
        )

    def _run_inference(
        points: list[list[float]],
        query_source: str,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[dict[str, Any], str, str]:
        query_points = _validate_query_points(points, query_source)
        with runtime_lock:
            progress(0.05, desc="加载 TCP 模型")
            if runtime["model"] is None:
                runtime["model"] = load_tcp_model(args.model, device)

            progress(0.20, desc=f"联合推理 {len(paths)} 帧")
            (
                depth,
                world_points,
                confidence,
                tcp,
                extrinsics,
                profiling,
            ) = infer_tcp_and_geometry(
                runtime["model"], views, query_points, device, dtype
            )
            progress(0.72, desc="生成逐帧点云")
            frame_clouds = prepare_frame_point_clouds(
                depth,
                world_points,
                confidence,
                colors,
                max_points_per_frame=args.max_points,
                seed=0,
            )
            elapsed = float(profiling.get("total_time", math.nan))
            result, _, _ = build_first_frame_result(
                tcp,
                query_points=query_points,
                query_source=query_source,
                image_path=paths[0],
                inference_seconds=elapsed,
            )
            result["selected_frame_indices"] = selected_indices
            result["selected_images"] = [str(path) for path in paths]

            if args.save_json is not None:
                output_path = args.save_json.expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

            progress(0.88, desc="启动 Viser")
            stop_visualization(runtime["viser"])
            runtime["viser"] = start_visualization(
                frame_clouds,
                tcp,
                selected_indices,
                paths,
                host=args.host,
                port=args.port,
                point_size=args.point_size,
                confidence_percentile=args.confidence_percentile,
                reference_extrinsic_w2c=extrinsics[0],
            )
            viser_port = runtime["viser"].get_port()
            display_host = (
                "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
            )
            viewer_url = f"http://{display_host}:{viser_port}"
            safe_url = html.escape(viewer_url, quote=True)
            viewer_html = (
                '<div style="display:flex;flex-direction:column;gap:0.75rem">'
                f'<a href="{safe_url}" target="_blank" rel="noopener">'
                "在新窗口打开 Viser</a>"
                f'<iframe src="{safe_url}" title="4RC TCP Viser" '
                'style="width:100%;height:720px;border:1px solid #ddd;'
                'border-radius:8px" allowfullscreen></iframe></div>'
            )
            point_counts = [
                len(frame_cloud["points"]) for frame_cloud in frame_clouds
            ]
            status = (
                f"推理完成：模型前向 `{elapsed:.3f}s`；逐帧有效点数 "
                f"`{point_counts}`；Viser：[{viewer_url}]({viewer_url})"
            )
            progress(1.0, desc="完成")
            return result, status, viewer_html

    with gr.Blocks(title="4RC 双臂 TCP 推理") as demo:
        gr.Markdown(
            "# 4RC 双臂 TCP 交互推理\n"
            f"首帧：`{paths[0]}`。按顺序点击 **左臂 TCP**、**右臂 TCP**，"
            "然后运行整段 clip 推理。坐标使用原始 320×240 图像。"
        )
        selected_points = gr.State([])
        query_source = gr.State("interactive first-frame clicks")
        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(
                    value=base_image,
                    label="第一帧（点击选择 TCP）",
                    type="numpy",
                    interactive=False,
                    format="png",
                    show_label=False,
                    buttons=[],
                    container=False,
                    width=SOURCE_WIDTH,
                    height=SOURCE_HEIGHT,
                    min_width=SOURCE_WIDTH,
                    elem_id="tcp-query-image",
                )
                selection_status = gr.Markdown(_interactive_selection_status([]))
                gr.Markdown(ground_truth_message)
                with gr.Row():
                    ground_truth_button = gr.Button(
                        "使用首帧真值 TCP",
                        interactive=ground_truth_points is not None,
                    )
                    reset_button = gr.Button("重置选点")
                    infer_button = gr.Button(
                        "开始推理", variant="primary", interactive=False
                    )
            with gr.Column(scale=1):
                inference_status = gr.Markdown("尚未开始推理。")
                result_json = gr.JSON(label="首帧 TCP 结果")

        gr.Markdown("## Viser 逐帧结果")
        viewer = gr.HTML(
            "<p>推理完成后，这里会显示逐帧几何与左右臂 TCP 的 "
            "Viser 播放器。</p>"
        )
        selection_outputs = (
            query_image,
            selected_points,
            query_source,
            selection_status,
            infer_button,
        )
        query_image.select(
            _select_point,
            inputs=selected_points,
            outputs=selection_outputs,
            queue=False,
        )
        ground_truth_button.click(
            _use_ground_truth,
            outputs=selection_outputs,
            queue=False,
        )
        reset_button.click(
            _reset_points,
            outputs=selection_outputs,
            queue=False,
        )
        infer_button.click(
            _run_inference,
            inputs=(selected_points, query_source),
            outputs=(result_json, inference_status, viewer),
            concurrency_limit=1,
        )

    display_ui_host = (
        "127.0.0.1" if args.ui_host in {"0.0.0.0", "::"} else args.ui_host
    )
    ui_url = f"http://{display_ui_host}:{args.ui_port}"
    print(f"Interactive TCP selection: {ui_url}")
    try:
        demo.queue(default_concurrency_limit=1).launch(
            server_name=args.ui_host,
            server_port=args.ui_port,
            show_error=True,
        )
    finally:
        stop_visualization(runtime["viser"])
        runtime["model"] = None
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_percentile <= 99.0:
        raise ValueError("--confidence-percentile must be in [0,99]")
    if args.point_size <= 0.0:
        raise ValueError("--point-size must be positive")
    if args.frame_indices is not None and (
        args.start_frame is not None or args.frame_interval != 1
    ):
        raise ValueError(
            "--start-frame/--frame-interval cannot be combined with --frame-indices"
        )
    if args.interactive and args.no_visualize:
        raise ValueError("--interactive cannot be combined with --no-visualize")
    if args.interactive and args.ui_port == args.port:
        raise ValueError("--ui-port and --port must be different in interactive mode")
    if not 1 <= args.ui_port <= 65535:
        raise ValueError("--ui-port must be in [1,65535]")
    rgb_dir = resolve_rgb_dir(args.input, args.view)
    paths = collect_rgb_paths(
        rgb_dir,
        args.max_frames,
        args.frame_indices,
        start_frame=args.start_frame,
        frame_interval=args.frame_interval,
    )
    warn_if_out_of_training_distribution(paths)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if args.view != "third_views":
        print("Warning: the visual-query TCP model was trained on third_views images.")
    selected_indices = [_frame_index(path) for path in paths]
    print(f"Processing {len(paths)} frames on {device} ({dtype}).")
    print(f"Selected frame indices: {selected_indices}")

    views, colors = load_robotwin_views(paths)
    if args.interactive:
        start_interactive_page(
            args=args,
            paths=paths,
            selected_indices=selected_indices,
            views=views,
            colors=colors,
            device=device,
            dtype=dtype,
        )
        return

    query_points, query_source = resolve_tcp_query_points(args)
    print(f"Initial TCP query points ({query_source}): {query_points.tolist()}")
    model = load_tcp_model(args.model, device)
    (
        depth,
        world_points,
        confidence,
        tcp,
        extrinsics,
        profiling,
    ) = infer_tcp_and_geometry(
        model, views, query_points, device, dtype
    )

    frame_clouds = prepare_frame_point_clouds(
        depth,
        world_points,
        confidence,
        colors,
        max_points_per_frame=args.max_points,
        seed=0,
    )
    elapsed = float(profiling.get("total_time", math.nan))
    result, _, _ = build_first_frame_result(
        tcp,
        query_points=query_points,
        query_source=query_source,
        image_path=paths[0],
        inference_seconds=elapsed,
    )
    result["selected_frame_indices"] = selected_indices
    result["selected_images"] = [str(path) for path in paths]

    print(json.dumps(result, indent=2, ensure_ascii=False))
    point_counts = [len(frame_cloud["points"]) for frame_cloud in frame_clouds]
    print(f"Per-frame valid point counts: {point_counts}")
    if args.save_json is not None:
        output_path = args.save_json.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Saved TCP state to {output_path.resolve()}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.no_visualize:
        return

    server = start_visualization(
        frame_clouds,
        tcp,
        selected_indices,
        paths,
        host=args.host,
        port=args.port,
        point_size=args.point_size,
        reference_extrinsic_w2c=extrinsics[0],
        confidence_percentile=args.confidence_percentile,
    )
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Visualization: http://{display_host}:{server.get_port()}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop_visualization(server)


if __name__ == "__main__":
    main()
