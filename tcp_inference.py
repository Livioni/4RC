#!/usr/bin/env python3
"""Infer and visualize a RoboTwin TCP trajectory with per-frame geometry.

``RoboTwin-TCP`` is a conditional tracker: it requires the two first-frame TCP
states as a query and predicts their trajectory from an RGB clip.  It does not
contain an RGB-to-initial-pose head.  Pointing ``--input`` at a RoboTwin episode
uses ``images/head_view`` and automatically reads the matching first-frame query
from ``TCP_head`` (or legacy ``TCP``).  For copied images, preserve their numeric
filenames and pass the original episode with ``--tcp-dir``.

Example (standard RoboTwin episode layout):

    conda run -n 4rc python tcp_inference.py \
        --input datasets/RoboTwin/<task>/<episode> \
        --frame-indices 10 15 20 25

Example (plain image directory and an explicit [2, 7] query):

    conda run -n 4rc python tcp_inference.py \
        --input demo/rgb \
        --tcp-query-file demo/first_tcp_state.npy

Each arm uses ``[x, y, z, roll, pitch, yaw, gripper]``.  Position is in metres,
RPY is in radians, and RoboTwin uses ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geometry_inference import (
    PAD_LEFT,
    PAD_TOP,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    load_robotwin_views,
    resolve_device,
    resolve_dtype,
)


DEFAULT_MODEL = Path("checkpoints/RoboTwin-TCP/model.safetensors")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ARM_NAMES = ("left", "right")
ARM_COLORS = ((255, 96, 64), (45, 180, 255))
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
        default="head_view",
        help="Image view below an episode's images directory",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="RoboTwin-TCP model.safetensors checkpoint",
    )
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "--tcp-query-file",
        type=Path,
        help=(
            "First-frame TCP query as .npy/.npz/.json with shape [2,7]. "
            "If omitted, the script auto-discovers RoboTwin episode TCP files"
        ),
    )
    query_group.add_argument(
        "--tcp-query",
        type=float,
        nargs=14,
        metavar=(
            "LX", "LY", "LZ", "LR", "LP", "LYAW", "LG",
            "RX", "RY", "RZ", "RR", "RP", "RYAW", "RG",
        ),
        help="Explicit left state followed by right state (14 scalars)",
    )
    query_group.add_argument(
        "--tcp-dir",
        type=Path,
        help=(
            "Original episode root or TCP directory containing left_state.npy "
            "and right_state.npy, useful when --input contains copied images"
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
            "pass --tcp-query-file or --tcp-query explicitly"
        )
    return int(match.group(1))


def _validate_query(state: Any, source: str) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.shape != (2, 7):
        raise ValueError(
            f"First-frame TCP query from {source} must have shape [2,7] "
            f"(left/right x xyz/rpy/gripper), got {state_array.shape}"
        )
    if not np.isfinite(state_array).all():
        raise ValueError(f"First-frame TCP query from {source} contains NaN/Inf")
    return state_array


def _load_query_file(path: Path) -> np.ndarray:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TCP query file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        state = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "tcp_state" in archive:
                state = archive["tcp_state"]
            elif len(archive.files) == 1:
                state = archive[archive.files[0]]
            else:
                raise ValueError(
                    f"{path} must contain a 'tcp_state' array when it has multiple keys"
                )
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "tcp_state" in payload:
            state = payload["tcp_state"]
        elif isinstance(payload, dict) and all(name in payload for name in ARM_NAMES):
            state = [payload[name] for name in ARM_NAMES]
        else:
            state = payload
    else:
        raise ValueError("--tcp-query-file must be .npy, .npz, or .json")

    state_array = np.asarray(state)
    if state_array.ndim == 3:
        state_array = state_array[0]
    return _validate_query(state_array, str(path))


def _tcp_dir_names(view: str) -> tuple[str, ...]:
    view_to_tcp = {
        "head_view": "TCP_head",
        "third_views": "TCP_third",
    }
    preferred = view_to_tcp.get(view)
    return (preferred, "TCP") if preferred is not None else ("TCP",)


def _has_tcp_states(path: Path) -> bool:
    return (
        (path / "left_state.npy").is_file()
        and (path / "right_state.npy").is_file()
    )


def _find_episode_tcp_dir(input_dir: Path, view: str) -> Path | None:
    """Find the view-matched TCP directory above an episode/image directory."""
    resolved = input_dir.expanduser().resolve()
    for candidate_root in (resolved, *resolved.parents[:4]):
        if _has_tcp_states(candidate_root):
            return candidate_root
        for name in _tcp_dir_names(view):
            tcp_dir = candidate_root / name
            if _has_tcp_states(tcp_dir):
                return tcp_dir
    return None


def resolve_tcp_query(
    args: argparse.Namespace, first_image: Path
) -> tuple[np.ndarray, str]:
    if args.tcp_query is not None:
        return (
            _validate_query(np.asarray(args.tcp_query).reshape(2, 7), "--tcp-query"),
            "explicit --tcp-query",
        )
    if args.tcp_query_file is not None:
        return _load_query_file(args.tcp_query_file), str(args.tcp_query_file)

    if args.tcp_dir is not None:
        tcp_dir = _find_episode_tcp_dir(args.tcp_dir, args.view)
        if tcp_dir is None:
            raise FileNotFoundError(
                "Could not find left_state.npy/right_state.npy from "
                f"--tcp-dir {args.tcp_dir}"
            )
    else:
        tcp_dir = _find_episode_tcp_dir(args.input, args.view)
    if tcp_dir is not None:
        index = _frame_index(first_image)
        left = np.load(tcp_dir / "left_state.npy", mmap_mode="r", allow_pickle=False)
        right = np.load(tcp_dir / "right_state.npy", mmap_mode="r", allow_pickle=False)
        valid_shapes = (
            left.ndim == 2
            and right.ndim == 2
            and left.shape[1:] == (7,)
            and right.shape[1:] == (7,)
        )
        if not valid_shapes:
            raise ValueError(
                f"Expected left/right TCP arrays [T,7] below {tcp_dir}; "
                f"got {left.shape} and {right.shape}"
            )
        if index >= len(left) or index >= len(right):
            raise IndexError(
                f"Frame {index} is outside TCP arrays with lengths "
                f"{len(left)} and {len(right)}"
            )
        state = np.stack((left[index], right[index]), axis=0)
        return _validate_query(state, str(tcp_dir)), f"{tcp_dir} (frame {index})"

    for name in ("tcp_state.npy", "first_tcp_state.npy", "tcp_state.json"):
        candidate = args.input / name
        if candidate.is_file():
            return _load_query_file(candidate), str(candidate)

    raise ValueError(
        "RoboTwin-TCP cannot infer an initial TCP pose from RGB alone. No matching "
        "episode TCP_head/TCP state files were found. Point --input at the original "
        "episode, pass --tcp-dir for copied images, or supply --tcp-query-file."
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

    tcp_prefixes = ("tcp_query_encoder.", "tcp_track_head.")
    if not all(any(key.startswith(prefix) for key in state_dict) for prefix in tcp_prefixes):
        raise ValueError(
            f"{model_path} does not contain both TCP query encoder and tracking head"
        )

    model = Arc()
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
    query_state: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    device_views = [
        {**view, "img": view["img"].to(device, non_blocking=True)} for view in views
    ]
    query_tensor = torch.from_numpy(query_state).unsqueeze(0).to(device)
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
                    tcp_query_state=query_tensor,
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
    return depth, world_points, confidence, tcp, profiling


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
    query_state: np.ndarray,
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
        "note": "Conditional prediction: the initial TCP query is an input to RoboTwin-TCP.",
        "query_tcp_state": query_state.tolist(),
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

    server = viser.ViserServer(host=host, port=port)
    server.gui.set_panel_label("Geometry + TCP")
    server.gui.configure_theme(
        control_layout="floating", control_width="large", show_logo=False
    )
    server.scene.set_up_direction((0.0, -1.0, 0.0))
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
            client.camera.look_at = tuple(center.tolist())
            client.camera.position = tuple(
                (
                    center
                    + extent * np.array([1.2, -1.2, -1.2], dtype=np.float32)
                ).tolist()
            )
        client.flush()

    def _playback_loop() -> None:
        while True:
            if bool(gui_playing.value):
                gui_frame.value = (int(gui_frame.value) + 1) % num_frames
            time.sleep(1.0 / float(gui_fps.value))

    playback_thread = threading.Thread(target=_playback_loop, daemon=True)
    playback_thread.start()
    _update_visibility()
    return server


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
    rgb_dir = resolve_rgb_dir(args.input, args.view)
    paths = collect_rgb_paths(
        rgb_dir,
        args.max_frames,
        args.frame_indices,
        start_frame=args.start_frame,
        frame_interval=args.frame_interval,
    )
    warn_if_out_of_training_distribution(paths)
    query_state, query_source = resolve_tcp_query(args, paths[0])
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if args.view != "head_view":
        print("Warning: RoboTwin-TCP was trained on head_view images.")
    if not np.all((0.0 <= query_state[:, 6]) & (query_state[:, 6] <= 1.0)):
        print("Warning: query gripper values are outside the training range [0,1].")
    selected_indices = [_frame_index(path) for path in paths]
    print(f"Processing {len(paths)} frames on {device} ({dtype}).")
    print(f"Selected frame indices: {selected_indices}")
    print(f"Initial TCP query source: {query_source}")

    views, colors = load_robotwin_views(paths)
    model = load_tcp_model(args.model, device)
    depth, world_points, confidence, tcp, profiling = infer_tcp_and_geometry(
        model, views, query_state, device, dtype
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
        query_state=query_state,
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
        confidence_percentile=args.confidence_percentile,
    )
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"Visualization: http://{display_host}:{server.get_port()}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
