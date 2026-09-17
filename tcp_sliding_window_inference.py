#!/usr/bin/env python3
"""Infer a complete RoboTwin episode with overlapping TCP windows.

This entry point includes its own preprocessing, model loading and visualization.
It requires the repository arc package and installed third-party dependencies.
Edit the configuration block below or override it with command-line options.

Each window shares one boundary frame with the next window. The final metric
TCP positions from one window are projected into that shared image to become
the next window's left/right visual query points.

Examples:

    python tcp_sliding_window_inference.py \
        --input datasets/RoboTwin/<task>/<episode> \
        --output outputs/tcp_episode.json

    python tcp_sliding_window_inference.py \
        --input datasets/eval_sets/place_dual_shoes/episode_0000092 \
        --output datasets/eval_sets/place_dual_shoes/episode_0000092/pred_tcp_episode.json \
        --interactive \
        --visualize
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import re
from dataclasses import dataclass
import html
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import functional as TVF

# ==================== 用户配置（命令行参数可覆盖） ====================
# 路径相对于启动时的工作目录；模型必须含 TCP visual query / tracking 权重。
DEFAULT_MODEL = Path("checkpoints/RoboTwin-Stage1/model.safetensors")
# 填入完整 episode 目录后可直接运行；None 表示必须传 --input。
# 目录需包含 images/<view>/、intrinsics/<view>.npy 和 metadata.json。
DEFAULT_INPUT: Path | None = None
DEFAULT_OUTPUT: Path | None = None  # None：保存到输入目录 / DEFAULT_OUTPUT_FILENAME
DEFAULT_OUTPUT_FILENAME = "tcp_episode.json"  # 完整双臂轨迹 JSON，同名文件会覆盖
DEFAULT_VIEW = "third_views"  # 模型训练视角；原始 RGB 必须为 320×240
DEFAULT_WINDOW_SIZE = 9  # 每窗帧数：2～18，相邻窗口共享一帧
DEFAULT_BOUNDARY_MERGE = "previous"  # previous / next / average
DEFAULT_DEVICE = "auto"  # auto / cuda / cuda:0 / cpu
DEFAULT_DTYPE = "auto"  # auto / float32 / float16 / bfloat16

# 首帧选点：原始图像像素，顺序为 (左 x, 左 y, 右 x, 右 y)。
# 以下三项最多启用一项；都不启用时，从首帧真值投影取得选点。
DEFAULT_TCP_QUERY_POINTS: tuple[float, float, float, float] | None = None
DEFAULT_TCP_QUERY_POINTS_FILE: Path | None = None  # .npy / .npz / .json，形状 [2,2]
DEFAULT_INTERACTIVE = False  # Gradio 交互选点，完成后展示 Viser
DEFAULT_VISUALIZE = False  # 非交互推理完成后打开 Viser 服务
DEFAULT_CONFIDENCE_PERCENTILE = 2.5  # 点云置信度过滤百分位 [0,99]
DEFAULT_MAX_POINTS = 100_000  # 每帧最多显示点数；0 保留全部
DEFAULT_POINT_SIZE = 0.003  # Viser 点大小
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8020
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 7860

# JSON 每帧包含左右臂 xyz_m、rpy_rad、rpy_deg、夹爪概率/开合及置信度。
# xyz 为每帧相机坐标（x 向右、y 向下、z 向前），单位米；RPY 为固定轴 XYZ。
GRIPPER_OPEN_THRESHOLD = 0.5  # 概率 >= 此值时 gripper_open=True

# ==================== 模型/数据约定（应与训练保持一致） ====================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ARM_NAMES = ("left", "right")
ARM_COLORS = ((255, 96, 64), (45, 180, 255))
TCP_GROUND_TRUTH_DIRS = {"third_views": "TCP_third", "head_view": "TCP_head"}
TRAIN_MAX_FRAMES = 18
SOURCE_WIDTH, SOURCE_HEIGHT = 320, 240
PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM = 1, 1, 6, 6
PADDED_WIDTH, PADDED_HEIGHT = 322, 252


RPY_CONVENTION = "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))"
CAMERA_SUFFIX = "OpenCV camera (+x right, +y down, +z forward)"
BOUNDARY_POLICIES = ("previous", "next", "average")


# ==================== 内置加载、推理与可视化逻辑 ====================

def _natural_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
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
    return project_tcp_positions_to_query_points(
        np.stack(positions, axis=0),
        intrinsics,
        source=f"ground-truth TCP at frame {frame_index}",
    )


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


def project_tcp_positions_to_query_points(
    positions: Any,
    intrinsics: Any,
    *,
    source: str,
) -> np.ndarray:
    """Project left/right camera-frame TCP xyz to original-image pixels.

    The visual TCP query encoder consumes two image points, while the TCP head
    predicts metric positions. Sliding-window inference uses this conversion on
    the shared boundary frame to feed one window's final prediction into the
    next window.
    """
    xyz = np.asarray(positions, dtype=np.float32)
    camera_matrix = np.asarray(intrinsics, dtype=np.float32)
    if xyz.shape != (len(ARM_NAMES), 3):
        raise ValueError(
            f"TCP positions from {source} must have shape [2,3], got {xyz.shape}"
        )
    if camera_matrix.shape != (3, 3):
        raise ValueError(
            f"Camera intrinsics for {source} must have shape [3,3], "
            f"got {camera_matrix.shape}"
        )
    if not np.isfinite(camera_matrix).all():
        raise ValueError(f"Camera intrinsics for {source} contain NaN/Inf")
    if not np.isfinite(xyz).all():
        raise ValueError(f"TCP positions from {source} contain NaN/Inf: {xyz.tolist()}")

    z = xyz[:, 2]
    invalid_depth = np.flatnonzero(z <= 0)
    if len(invalid_depth):
        arm_index = int(invalid_depth[0])
        raise ValueError(
            f"Cannot project {ARM_NAMES[arm_index]} TCP from {source}: "
            f"expected z > 0, got xyz={xyz[arm_index].tolist()}"
        )

    homogeneous = xyz @ camera_matrix.T
    projection_depth = homogeneous[:, 2]
    invalid_projection_depth = np.flatnonzero(
        ~np.isfinite(projection_depth) | (np.abs(projection_depth) < 1e-8)
    )
    if len(invalid_projection_depth):
        arm_index = int(invalid_projection_depth[0])
        raise ValueError(
            f"Cannot project {ARM_NAMES[arm_index]} TCP from {source}: invalid "
            f"homogeneous depth {float(projection_depth[arm_index])}"
        )
    pixels = homogeneous[:, :2] / projection_depth[:, None]

    for arm_index, arm_name in enumerate(ARM_NAMES):
        u, v = (float(value) for value in pixels[arm_index])
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError(
                f"Cannot project {arm_name} TCP from {source}: projected pixel "
                f"is not finite: {[u, v]}"
            )
        if not (0.0 <= u < SOURCE_WIDTH and 0.0 <= v < SOURCE_HEIGHT):
            raise ValueError(
                f"Cannot project {arm_name} TCP from {source}: xyz="
                f"{xyz[arm_index].tolist()} projects outside "
                f"{SOURCE_WIDTH}x{SOURCE_HEIGHT} at pixel={[u, v]}"
            )
    return _validate_query_points(pixels, f"projection of {source}")


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


def _run_tcp_model(
    model,
    views: list[dict[str, torch.Tensor]],
    query_points: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Run the shared model forward used by TCP-only and geometry inference."""
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
    except torch.cuda.OutOfMemoryError as error:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA ran out of memory while processing {len(views)} frames; "
            "reduce the clip or sliding-window length"
        ) from error
    return predictions, profiling


def _extract_tcp_predictions(
    predictions: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    return {
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


def infer_tcp_trajectory(
    model,
    views: list[dict[str, torch.Tensor]],
    query_points: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Infer TCP state without camera recovery or point-cloud materialization."""
    predictions, profiling = _run_tcp_model(
        model, views, query_points, device, dtype
    )
    tcp = _extract_tcp_predictions(predictions)
    del predictions
    return tcp, profiling


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
    predictions, profiling = _run_tcp_model(
        model, views, query_points, device, dtype
    )
    with torch.inference_mode():
        height, width = predictions["depth"].shape[-2:]
        model._process_ray_pose_estimation(predictions, height, width)
        depth = predictions["depth"][0].detach().float().cpu().numpy()
        confidence = predictions["depth_conf"][0].detach().float().cpu().numpy()
        extrinsics = predictions["extrinsics"][0].detach().float().cpu().numpy()
        intrinsics = predictions["intrinsics"][0].detach().float().cpu().numpy()
        tcp = _extract_tcp_predictions(predictions)
    del predictions

    from arc.models.arc.utils.geometry import unproject_depth_map_to_point_map

    world_points, _ = unproject_depth_map_to_point_map(
        depth[..., None], extrinsics, intrinsics
    )
    return depth, world_points, confidence, tcp, extrinsics, profiling


def infer_tcp_and_depth(
    model,
    views: list[dict[str, torch.Tensor]],
    query_points: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Extract TCP and padded metric depth from one forward, without pose recovery."""
    predictions, profiling = _run_tcp_model(
        model, views, query_points, device, dtype
    )
    tcp = _extract_tcp_predictions(predictions)
    depth = predictions["depth"][0].detach().float().cpu().numpy()
    del predictions
    return tcp, depth, profiling


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
    marker_scale = float(np.clip(extent * 0.08, 0.025, 0.12))
    tcp_axes_length = marker_scale * 0.55
    axes_radius = marker_scale * 0.018
    origin_radius = marker_scale * 0.05

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
    world_axes = server.scene.world_axes
    world_axes.axes_length = tcp_axes_length * 2.0
    world_axes.axes_radius = axes_radius
    world_axes.origin_radius = origin_radius
    world_axes.visible = True

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
                axes_length=tcp_axes_length,
                axes_radius=axes_radius,
                origin_radius=origin_radius,
                origin_color=arm_color,
                visible=frame_slot == 0,
            )
            tcp_sphere = server.scene.add_icosphere(
                f"/frames/t{frame_slot}/tcp/{arm_name}/xyz",
                radius=marker_scale * 0.11,
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
                position=(0.0, 0.0, marker_scale * 1.25),
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


@dataclass(slots=True)
class EpisodeInputs:
    episode_path: Path
    image_paths: list[Path]
    frame_indices: list[int]
    intrinsics: np.ndarray
    frame_rate: float


@dataclass(slots=True)
class SlidingWindowPrediction:
    tcp: dict[str, np.ndarray]
    window_records: list[dict[str, Any]]
    source_windows: list[list[int]]
    frame_clouds: list[dict[str, np.ndarray]] | None
    reference_extrinsic_w2c: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer every frame in a RoboTwin episode with overlapping TCP "
            "windows and save the complete dual-arm trajectory as JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        required=DEFAULT_INPUT is None,
        help="Complete RoboTwin episode directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            f"Destination JSON trajectory; defaults to <input>/{DEFAULT_OUTPUT_FILENAME} "
            "(replaces an existing file)"
        ),
    )
    parser.add_argument(
        "--view",
        default=DEFAULT_VIEW,
        help="Image view below the episode's images directory",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Image-conditioned RoboTwin TCP checkpoint",
    )
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "--tcp-query-points",
        type=float,
        nargs=4,
        metavar=("LEFT_X", "LEFT_Y", "RIGHT_X", "RIGHT_Y"),
        help="Override first-frame left/right query points in original pixels",
    )
    query_group.add_argument(
        "--tcp-query-points-file",
        type=Path,
        help="Override first-frame query with a .npy/.npz/.json [2,2] file",
    )
    query_group.add_argument(
        "--interactive",
        action="store_true",
        help="Select first-frame TCP pixels in Gradio, then show the Viser result",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Frames per window; neighboring windows overlap by one frame",
    )
    parser.add_argument(
        "--boundary-merge",
        choices=BOUNDARY_POLICIES,
        default=DEFAULT_BOUNDARY_MERGE,
        help="Which TCP prediction to emit for shared boundary frames",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device such as cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default=DEFAULT_DTYPE,
        help="Inference autocast dtype",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_VISUALIZE,
        help="Start Viser after non-interactive JSON inference",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=DEFAULT_CONFIDENCE_PERCENTILE,
        help="Initial per-frame point-cloud confidence percentile",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help="Randomly retain at most this many points per frame; 0 keeps all",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=DEFAULT_POINT_SIZE,
        help="Viser point size in world units",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Viser bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Viser port")
    parser.add_argument(
        "--ui-host",
        default=DEFAULT_UI_HOST,
        help="Interactive selection-page bind address",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=DEFAULT_UI_PORT,
        help="Interactive selection-page port",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.expanduser() / DEFAULT_OUTPUT_FILENAME
    # 显式 CLI 选点方式优先于顶部默认配置，避免互斥选项的默认值冲突。
    if not (args.interactive or args.tcp_query_points is not None or args.tcp_query_points_file is not None):
        configured = sum((DEFAULT_INTERACTIVE, DEFAULT_TCP_QUERY_POINTS is not None,
                          DEFAULT_TCP_QUERY_POINTS_FILE is not None))
        if configured > 1:
            parser.error("顶部首帧选点配置最多只能启用一项")
        args.interactive = DEFAULT_INTERACTIVE
        args.tcp_query_points = DEFAULT_TCP_QUERY_POINTS
        args.tcp_query_points_file = DEFAULT_TCP_QUERY_POINTS_FILE
    if not 0.0 <= GRIPPER_OPEN_THRESHOLD <= 1.0:
        parser.error("GRIPPER_OPEN_THRESHOLD 必须在 [0,1] 内")
    return args


def build_sliding_windows(
    num_frames: int, window_size: int
) -> list[tuple[int, int]]:
    """Return half-open windows with exactly one shared boundary frame."""
    if num_frames < 2:
        raise ValueError("Sliding-window TCP inference requires at least 2 frames")
    if not 2 <= window_size <= TRAIN_MAX_FRAMES:
        raise ValueError(
            f"--window-size must be between 2 and {TRAIN_MAX_FRAMES}"
        )

    windows: list[tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + window_size, num_frames)
        if end - start < 2:
            raise RuntimeError("Internal window construction produced one frame")
        windows.append((start, end))
        if end == num_frames:
            break
        start = end - 1
    return windows


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def load_episode_inputs(input_path: Path, view: str) -> EpisodeInputs:
    """Validate and load the calibration/timeline for one complete episode."""
    episode_path = input_path.expanduser()
    if not episode_path.is_dir():
        raise FileNotFoundError(f"Episode directory does not exist: {episode_path}")
    rgb_dir = episode_path / "images" / view
    if not rgb_dir.is_dir():
        raise FileNotFoundError(
            f"Expected a complete episode with images/{view}: {episode_path}"
        )
    image_paths = collect_rgb_paths(rgb_dir, max_frames=0)
    frame_indices = [_frame_index(path) for path in image_paths]

    intrinsics_path = episode_path / "intrinsics" / f"{view}.npy"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"Camera intrinsics are missing: {intrinsics_path}")
    intrinsics = np.load(intrinsics_path, allow_pickle=False).astype(
        np.float32, copy=False
    )
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(
            f"Expected finite camera intrinsics [3,3] in {intrinsics_path}, "
            f"got {intrinsics.shape}"
        )

    metadata_path = episode_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Episode metadata is missing: {metadata_path}")
    frame_rate = _read_json_object(metadata_path).get("frequency_hz")
    if (
        not isinstance(frame_rate, (int, float))
        or not math.isfinite(float(frame_rate))
        or float(frame_rate) <= 0
    ):
        raise ValueError(
            f"Expected positive frequency_hz in {metadata_path}, got {frame_rate!r}"
        )
    return EpisodeInputs(
        episode_path=episode_path,
        image_paths=image_paths,
        frame_indices=frame_indices,
        intrinsics=intrinsics,
        frame_rate=float(frame_rate),
    )


def resolve_initial_query_points(
    args: argparse.Namespace, first_frame_index: int
) -> tuple[np.ndarray, str]:
    if args.tcp_query_points is not None:
        points = np.asarray(args.tcp_query_points, dtype=np.float32).reshape(2, 2)
        return _validate_query_points(points, "--tcp-query-points"), "explicit CLI"
    if args.tcp_query_points_file is not None:
        return (
            _load_query_points_file(args.tcp_query_points_file),
            str(args.tcp_query_points_file),
        )
    return (
        load_ground_truth_query_points(args.input, args.view, first_frame_index),
        "projected first-frame ground-truth TCP",
    )


def _rotation_midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    key_rotations = Rotation.from_matrix(np.stack((first, second), axis=0))
    midpoint = Slerp([0.0, 1.0], key_rotations)([0.5]).as_matrix()[0]
    return midpoint.astype(np.float32)


def merge_boundary_predictions(
    previous: dict[str, np.ndarray],
    next_prediction: dict[str, np.ndarray],
    policy: str,
) -> dict[str, np.ndarray]:
    """Merge two [arms,...] predictions for one shared frame."""
    if policy not in BOUNDARY_POLICIES:
        raise ValueError(f"Unknown boundary merge policy: {policy!r}")
    selected = previous if policy == "previous" else next_prediction
    if policy != "average":
        return {key: np.asarray(value).copy() for key, value in selected.items()}

    rotations = np.stack(
        [
            _rotation_midpoint(
                previous["rotation"][arm_index],
                next_prediction["rotation"][arm_index],
            )
            for arm_index in range(len(ARM_NAMES))
        ],
        axis=0,
    )
    return {
        "position": (
            0.5 * (previous["position"] + next_prediction["position"])
        ).astype(np.float32),
        "rotation": rotations,
        "gripper": (
            0.5 * (previous["gripper"] + next_prediction["gripper"])
        ).astype(np.float32),
        "confidence": (
            0.5 * (previous["confidence"] + next_prediction["confidence"])
        ).astype(np.float32),
    }


def _validate_window_prediction(
    tcp: dict[str, np.ndarray], num_frames: int, window_index: int
) -> None:
    expected_shapes = {
        "position": (num_frames, len(ARM_NAMES), 3),
        "rotation": (num_frames, len(ARM_NAMES), 3, 3),
        "gripper": (num_frames, len(ARM_NAMES)),
        "confidence": (num_frames, len(ARM_NAMES)),
    }
    for key, expected in expected_shapes.items():
        value = tcp.get(key)
        if value is None or value.shape != expected:
            actual = None if value is None else value.shape
            raise ValueError(
                f"Window {window_index} produced {key} shape {actual}, "
                f"expected {expected}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Window {window_index} produced NaN/Inf in {key}")


def _frame_prediction(
    values: dict[str, list[np.ndarray]], frame_slot: int
) -> dict[str, np.ndarray]:
    return {key: entries[frame_slot] for key, entries in values.items()}


def infer_episode_sliding_windows(
    model: Any,
    episode: EpisodeInputs,
    initial_query_points: np.ndarray,
    initial_query_source: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int,
    boundary_merge: str,
    keep_geometry: bool,
    max_points_per_frame: int,
    progress_callback: Callable[[float, str], None] | None = None,
    depth_frame_callback: Callable[[int, np.ndarray], None] | None = None,
) -> SlidingWindowPrediction:
    """Run all windows and return one merged prediction per episode frame.

    If supplied, depth_frame_callback receives each original frame index and
    its finalized, padded metric depth exactly once, in timeline order. It
    shares the TCP boundary policy and must not mutate the provided array.
    """
    windows = build_sliding_windows(len(episode.image_paths), window_size)
    query_points = _validate_query_points(
        initial_query_points, initial_query_source
    ).copy()
    query_source = initial_query_source
    accumulated: dict[str, list[np.ndarray]] = {
        "position": [],
        "rotation": [],
        "gripper": [],
        "confidence": [],
    }
    source_windows: list[list[int]] = []
    window_records: list[dict[str, Any]] = []
    frame_clouds: list[dict[str, np.ndarray]] | None = [] if keep_geometry else None
    reference_extrinsic: np.ndarray | None = None
    pending_depth: np.ndarray | None = None

    for window_index, (start, end) in enumerate(windows):
        window_paths = episode.image_paths[start:end]
        window_frame_indices = episode.frame_indices[start:end]
        if progress_callback is not None:
            progress_callback(
                window_index / len(windows),
                f"Inferring window {window_index + 1}/{len(windows)}: "
                f"frames {window_frame_indices[0]}-{window_frame_indices[-1]}",
            )
        views, colors = load_robotwin_views(window_paths)
        query_for_record = query_points.copy()
        if keep_geometry:
            (
                depth,
                world_points,
                geometry_confidence,
                tcp,
                extrinsics,
                profiling,
            ) = infer_tcp_and_geometry(model, views, query_points, device, dtype)
            clouds = prepare_frame_point_clouds(
                depth,
                world_points,
                geometry_confidence,
                colors,
                max_points_per_frame=max_points_per_frame,
                seed=start,
            )
            assert frame_clouds is not None
            frame_clouds.extend(clouds if window_index == 0 else clouds[1:])
            if reference_extrinsic is None:
                reference_extrinsic = extrinsics[0].copy()
        elif depth_frame_callback is not None:
            tcp, depth, profiling = infer_tcp_and_depth(
                model, views, query_points, device, dtype
            )
        else:
            tcp, profiling = infer_tcp_trajectory(
                model, views, query_points, device, dtype
            )

        _validate_window_prediction(tcp, end - start, window_index)
        elapsed_value = float(profiling.get("total_time", math.nan))
        elapsed = elapsed_value if math.isfinite(elapsed_value) else None
        window_records.append(
            {
                "window_index": window_index,
                "frame_indices": window_frame_indices,
                "query_source": query_source,
                "tcp_query_points_px": query_for_record.tolist(),
                "inference_seconds": elapsed,
            }
        )

        if window_index == 0:
            for key in accumulated:
                accumulated[key].extend(value.copy() for value in tcp[key])
            source_windows.extend([[window_index] for _ in range(end - start)])
        else:
            if start != len(accumulated["position"]) - 1:
                raise RuntimeError("Sliding windows do not share exactly one frame")
            previous = _frame_prediction(accumulated, start)
            next_prediction = {key: tcp[key][0] for key in accumulated}
            merged = merge_boundary_predictions(
                previous, next_prediction, boundary_merge
            )
            for key in accumulated:
                accumulated[key][start] = merged[key]
                accumulated[key].extend(value.copy() for value in tcp[key][1:])
            source_windows[start].append(window_index)
            source_windows.extend(
                [[window_index] for _ in range(1, end - start)]
            )

        if depth_frame_callback is not None:
            if depth.ndim != 3 or depth.shape[0] != end - start:
                raise ValueError(f"Invalid window depth shape: {depth.shape}")
            first_new = 0
            if pending_depth is not None:
                if pending_depth.shape != depth[0].shape:
                    raise ValueError("Depth dimensions changed between windows")
                boundary_depth = pending_depth
                if boundary_merge == "next":
                    boundary_depth = depth[0]
                elif boundary_merge == "average":
                    boundary_depth = 0.5 * (pending_depth + depth[0])
                depth_frame_callback(window_frame_indices[0], boundary_depth)
                first_new = 1
            final_window = end == len(episode.image_paths)
            finalized_end = len(depth) if final_window else len(depth) - 1
            for local_index in range(first_new, finalized_end):
                depth_frame_callback(window_frame_indices[local_index], depth[local_index])
            pending_depth = None if final_window else depth[-1].copy()
            del depth

        if end < len(episode.image_paths):
            boundary_frame = window_frame_indices[-1]
            query_points = project_tcp_positions_to_query_points(
                tcp["position"][-1],
                episode.intrinsics,
                source=(
                    f"prediction at frame {boundary_frame} from window "
                    f"{window_index}"
                ),
            )
            query_source = (
                f"window {window_index} final prediction at frame "
                f"{boundary_frame}, projected with episode intrinsics"
            )

        del views, colors, tcp

    merged_tcp = {
        key: np.stack(values, axis=0).astype(np.float32, copy=False)
        for key, values in accumulated.items()
    }
    if len(merged_tcp["position"]) != len(episode.image_paths):
        raise RuntimeError("Merged TCP trajectory does not cover the complete episode")
    if frame_clouds is not None and len(frame_clouds) != len(episode.image_paths):
        raise RuntimeError("Merged geometry does not cover the complete episode")
    if progress_callback is not None:
        progress_callback(1.0, "Sliding-window inference complete")
    return SlidingWindowPrediction(
        tcp=merged_tcp,
        window_records=window_records,
        source_windows=source_windows,
        frame_clouds=frame_clouds,
        reference_extrinsic_w2c=reference_extrinsic,
    )


def build_json_result(
    episode: EpisodeInputs,
    prediction: SlidingWindowPrediction,
    *,
    model_path: Path,
    view: str,
    window_size: int,
    boundary_merge: str,
    initial_query_points: np.ndarray,
    initial_query_source: str,
) -> dict[str, Any]:
    rpy = matrix_to_rpy(prediction.tcp["rotation"])
    frames: list[dict[str, Any]] = []
    for frame_slot, (frame_index, image_path) in enumerate(
        zip(episode.frame_indices, episode.image_paths)
    ):
        arms: dict[str, Any] = {}
        for arm_index, arm_name in enumerate(ARM_NAMES):
            gripper_probability = float(
                prediction.tcp["gripper"][frame_slot, arm_index]
            )
            arms[arm_name] = {
                "xyz_m": prediction.tcp["position"][
                    frame_slot, arm_index
                ].tolist(),
                "rpy_rad": rpy[frame_slot, arm_index].tolist(),
                "rpy_deg": np.rad2deg(rpy[frame_slot, arm_index]).tolist(),
                "gripper_probability": gripper_probability,
                "gripper_open": gripper_probability >= GRIPPER_OPEN_THRESHOLD,
                "confidence": float(
                    prediction.tcp["confidence"][frame_slot, arm_index]
                ),
            }
        frames.append(
            {
                "frame_index": frame_index,
                "time_seconds": frame_index / episode.frame_rate,
                "image": str(image_path),
                "source_windows": prediction.source_windows[frame_slot],
                **arms,
            }
        )

    return {
        "schema_version": 1,
        "episode": str(episode.episode_path),
        "view": view,
        "model": str(model_path.expanduser()),
        "frame_rate_hz": episode.frame_rate,
        "num_frames": len(frames),
        "coordinate_frame": f"per-frame {view} {CAMERA_SUFFIX}",
        "position_unit": "meter",
        "rotation_unit": "radian",
        "rpy_convention": RPY_CONVENTION,
        "gripper": {
            "prediction": "probability",
            "binary_threshold": GRIPPER_OPEN_THRESHOLD,
        },
        "windowing": {
            "window_size": window_size,
            "stride": window_size - 1,
            "boundary_merge": boundary_merge,
            "num_windows": len(prediction.window_records),
        },
        "initial_query": {
            "frame_index": episode.frame_indices[0],
            "source": initial_query_source,
            "tcp_query_points_px": initial_query_points.tolist(),
        },
        "windows": prediction.window_records,
        "frames": frames,
    }


def write_json_atomic(result: dict[str, Any], output_path: Path) -> Path:
    """Replace the destination only after a complete JSON file is durable."""
    destination = output_path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                result,
                temporary,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination.resolve()


def _run_complete_episode(
    args: argparse.Namespace,
    episode: EpisodeInputs,
    model: Any,
    query_points: np.ndarray,
    query_source: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    keep_geometry: bool,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[dict[str, Any], SlidingWindowPrediction, Path]:
    prediction = infer_episode_sliding_windows(
        model,
        episode,
        query_points,
        query_source,
        device=device,
        dtype=dtype,
        window_size=args.window_size,
        boundary_merge=args.boundary_merge,
        keep_geometry=keep_geometry,
        max_points_per_frame=args.max_points,
        progress_callback=progress_callback,
    )
    result = build_json_result(
        episode,
        prediction,
        model_path=args.model,
        view=args.view,
        window_size=args.window_size,
        boundary_merge=args.boundary_merge,
        initial_query_points=query_points,
        initial_query_source=query_source,
    )
    saved_path = write_json_atomic(result, args.output)
    return result, prediction, saved_path


def _interactive_status(points: list[list[float]]) -> str:
    if not points:
        return "请在图中点击 **左臂 TCP**。"
    if len(points) == 1:
        return "左臂已选择；现在请点击 **右臂 TCP**。"
    return "左右臂 TCP 均已选择，可以开始完整 episode 推理。"


def start_interactive_page(
    args: argparse.Namespace,
    episode: EpisodeInputs,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    try:
        import gradio as gr
    except ImportError as error:
        raise ImportError(
            "gradio is required for --interactive; install requirements.txt"
        ) from error

    _, first_colors = load_robotwin_views([episode.image_paths[0]])
    base_image = first_colors[0]
    ground_truth_points = load_ground_truth_query_points(
        episode.episode_path, args.view, episode.frame_indices[0]
    )
    runtime: dict[str, Any] = {"model": None, "viser": None}
    runtime_lock = threading.Lock()

    def _select_point(
        points: list[list[float]], event: gr.SelectData
    ) -> tuple[np.ndarray, list[list[float]], str, str, Any]:
        updated = add_interactive_query_point(points, event.index)
        return (
            render_query_overlay(base_image, updated),
            updated,
            "interactive first-frame clicks",
            _interactive_status(updated),
            gr.update(interactive=len(updated) == len(ARM_NAMES)),
        )

    _select_point.__annotations__["event"] = gr.SelectData

    def _reset() -> tuple[np.ndarray, list[list[float]], str, str, Any]:
        return (
            base_image.copy(),
            [],
            "interactive first-frame clicks",
            _interactive_status([]),
            gr.update(interactive=False),
        )

    def _use_ground_truth() -> tuple[
        np.ndarray, list[list[float]], str, str, Any
    ]:
        points = ground_truth_points.tolist()
        return (
            render_query_overlay(base_image, points),
            points,
            "projected first-frame ground-truth TCP",
            "已使用首帧真值 TCP。" + _interactive_status(points),
            gr.update(interactive=True),
        )

    def _run(
        points: list[list[float]],
        query_source: str,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[dict[str, Any], str, str]:
        query_points = _validate_query_points(points, query_source)
        with runtime_lock:
            progress(0.02, desc="加载 TCP 模型")
            if runtime["model"] is None:
                runtime["model"] = load_tcp_model(args.model, device)

            def _report(fraction: float, message: str) -> None:
                progress(0.08 + 0.78 * fraction, desc=message)

            result, prediction, saved_path = _run_complete_episode(
                args,
                episode,
                runtime["model"],
                query_points,
                query_source,
                device,
                dtype,
                keep_geometry=True,
                progress_callback=_report,
            )
            assert prediction.frame_clouds is not None
            progress(0.90, desc="启动 Viser")
            stop_visualization(runtime["viser"])
            runtime["viser"] = start_visualization(
                prediction.frame_clouds,
                prediction.tcp,
                episode.frame_indices,
                episode.image_paths,
                host=args.host,
                port=args.port,
                point_size=args.point_size,
                confidence_percentile=args.confidence_percentile,
                reference_extrinsic_w2c=prediction.reference_extrinsic_w2c,
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
            progress(1.0, desc="完成")
            return (
                result,
                f"已保存 {len(episode.image_paths)} 帧到 `{saved_path}`；"
                f"Viser：[{viewer_url}]({viewer_url})",
                viewer_html,
            )

    with gr.Blocks(title="4RC TCP Sliding-Window Inference") as demo:
        gr.Markdown(
            "# 完整 Episode 双臂 TCP 推理\n"
            f"首帧：`{episode.image_paths[0]}`。依次点击左、右 TCP，"
            "或直接使用首帧真值，然后运行 sliding-window 推理。"
        )
        selected_points = gr.State([])
        query_source = gr.State("interactive first-frame clicks")
        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(
                    value=base_image,
                    label="首帧（点击选择左右臂 TCP）",
                    type="numpy",
                    interactive=False,
                    format="png",
                    buttons=[],
                )
                selection_status = gr.Markdown(_interactive_status([]))
                with gr.Row():
                    ground_truth_button = gr.Button("使用首帧真值 TCP")
                    reset_button = gr.Button("重置选点")
                    infer_button = gr.Button(
                        "推理完整 Episode", variant="primary", interactive=False
                    )
            with gr.Column(scale=1):
                inference_status = gr.Markdown("尚未开始推理。")
                result_json = gr.JSON(label="完整 TCP JSON")

        gr.Markdown("## Viser 逐帧结果")
        viewer = gr.HTML("<p>推理完成后显示逐帧几何与 TCP。</p>")
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
            _use_ground_truth, outputs=selection_outputs, queue=False
        )
        reset_button.click(_reset, outputs=selection_outputs, queue=False)
        infer_button.click(
            _run,
            inputs=(selected_points, query_source),
            outputs=(result_json, inference_status, viewer),
            concurrency_limit=1,
        )

    display_ui_host = (
        "127.0.0.1" if args.ui_host in {"0.0.0.0", "::"} else args.ui_host
    )
    print(f"Interactive TCP selection: http://{display_ui_host}:{args.ui_port}")
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


def validate_args(args: argparse.Namespace) -> None:
    if not 2 <= args.window_size <= TRAIN_MAX_FRAMES:
        raise ValueError(
            f"--window-size must be between 2 and {TRAIN_MAX_FRAMES}"
        )
    if not 0.0 <= args.confidence_percentile <= 99.0:
        raise ValueError("--confidence-percentile must be in [0,99]")
    if args.max_points < 0:
        raise ValueError("--max-points cannot be negative")
    if args.point_size <= 0.0:
        raise ValueError("--point-size must be positive")
    if not 1 <= args.port <= 65535 or not 1 <= args.ui_port <= 65535:
        raise ValueError("--port and --ui-port must be in [1,65535]")
    if args.interactive and args.port == args.ui_port:
        raise ValueError("--port and --ui-port must differ in interactive mode")


def main() -> None:
    args = parse_args()
    validate_args(args)
    episode = load_episode_inputs(args.input, args.view)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.view != "third_views":
        print("Warning: the visual-query TCP model was trained on third_views images.")
    print(
        f"Episode has {len(episode.image_paths)} frames; using "
        f"window_size={args.window_size} on {device} ({dtype})."
    )

    if args.interactive:
        start_interactive_page(args, episode, device, dtype)
        return

    query_points, query_source = resolve_initial_query_points(
        args, episode.frame_indices[0]
    )
    print(f"Initial TCP query ({query_source}): {query_points.tolist()}")
    model = load_tcp_model(args.model, device)
    try:
        result, prediction, saved_path = _run_complete_episode(
            args,
            episode,
            model,
            query_points,
            query_source,
            device,
            dtype,
            keep_geometry=args.visualize,
            progress_callback=lambda fraction, message: print(message),
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(
        f"Saved {result['num_frames']} frame predictions from "
        f"{result['windowing']['num_windows']} windows to {saved_path}"
    )

    if not args.visualize:
        return
    assert prediction.frame_clouds is not None
    server = start_visualization(
        prediction.frame_clouds,
        prediction.tcp,
        episode.frame_indices,
        episode.image_paths,
        host=args.host,
        port=args.port,
        point_size=args.point_size,
        confidence_percentile=args.confidence_percentile,
        reference_extrinsic_w2c=prediction.reference_extrinsic_w2c,
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
