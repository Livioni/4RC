#!/usr/bin/env python3
"""Standalone RoboTwin Stage2 reconstruction, action prediction and playback.

python stage2_sliding_window_inference.py --input datasets/RoboTwin/TASK/EPISODE
python stage2_sliding_window_inference.py --input datasets/RoboTwin/TASK/EPISODE --interactive

Only the first window uses selected/GT query points. Later windows propagate
recovered TCPs. Geometry uses predicted metric depth and episode calibration.
See README_STAGE2_INFERENCE_CN.md for coordinate and output conventions.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F


DEFAULT_CHECKPOINT = Path("checkpoints/RoboTwin-Stage2/180000")
WIDTH, HEIGHT = 320, 240
PADDING = (1, 1, 6, 6)
ARMS = ("left", "right")
ARM_COLORS = ((255, 100, 55), (45, 175, 255))
GT_COLORS = ((255, 210, 70), (80, 255, 180))
TCP_DIRS = {"third_views": "TCP_third", "head_view": "TCP_head"}


@dataclass
class EpisodeInputs:
    path: Path
    view: str
    image_paths: list[Path]
    frame_indices: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray  # Selected frames, homogeneous OpenCV world-to-camera.
    frequency_hz: float
    instruction: str


def homogeneous(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-2:] == (4, 4):
        return value.copy()
    if value.shape[-2:] != (3, 4):
        raise ValueError("Camera extrinsics must end in [3,4] or [4,4]")
    result = np.zeros((*value.shape[:-2], 4, 4), dtype=np.float32)
    result[..., :3, :] = value
    result[..., 3, 3] = 1
    return result


def load_episode(input_path: Path, view: str, instruction: str | None = None) -> EpisodeInputs:
    path = input_path.expanduser().resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    rate = metadata.get("frequency_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("metadata.json requires a finite positive frequency_hz")
    if instruction is None:
        texts = metadata.get("instructions", [])
        instruction = next((s.strip() for s in texts if isinstance(s, str) and s.strip()), "") if isinstance(texts, list) else ""
    if not instruction.strip():
        raise ValueError("Provide --instruction or a non-empty metadata instructions list")
    indexed = {}
    for image_path in (path / "images" / view).iterdir():
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not image_path.is_file():
            continue
        match = re.search(r"(\d+)$", image_path.stem)
        if match is None:
            raise ValueError(f"RGB filename needs a trailing frame number: {image_path}")
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate RGB frame number {index}")
        indexed[index] = image_path
    indices = np.array(sorted(indexed), dtype=np.int64)
    if not len(indices) or np.any(np.diff(indices) != 1):
        raise ValueError("Episode RGB frames must be non-empty and contiguous")
    intrinsics = np.load(path / "intrinsics" / f"{view}.npy", allow_pickle=False).astype(np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or np.linalg.det(intrinsics) == 0:
        raise ValueError("Expected finite, invertible intrinsics [3,3]")
    extrinsics = homogeneous(np.load(path / "extrinsics" / f"{view}.npy", allow_pickle=False))
    if extrinsics.ndim != 3 or len(extrinsics) <= indices[-1]:
        raise ValueError("Camera extrinsics do not cover all RGB frame indices")
    extrinsics = extrinsics[indices]
    if not np.isfinite(extrinsics).all() or np.any(np.abs(np.linalg.det(extrinsics)) < 1e-8):
        raise ValueError("Invalid camera extrinsics")
    return EpisodeInputs(path, view, [indexed[i] for i in indices], indices, intrinsics,
                         extrinsics, float(rate), instruction.strip())


def build_windows(num_frames: int, history_frames: int = 8, stride: int = 7) -> list[tuple[int, int]]:
    if history_frames < 2 or not 1 <= stride < history_frames:
        raise ValueError("window stride must be between 1 and history_frames - 1")
    if num_frames < history_frames:
        raise ValueError(f"Need at least {history_frames} RGB frames, found {num_frames}")
    starts = list(range(0, num_frames - history_frames + 1, stride))
    if starts[-1] != num_frames - history_frames:
        starts.append(num_frames - history_frames)
    return [(s, s + history_frames) for s in starts]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (WIDTH, HEIGHT):
            raise ValueError(f"Expected {WIDTH}x{HEIGHT} RGB, got {image.size}: {path}")
        return np.asarray(image).copy()


def prepare_images(paths: list[Path], intrinsics: np.ndarray, query_points: np.ndarray):
    """Return [1,K,3,252,322] images and padded-pixel calibration/queries."""
    rgb = np.stack([load_rgb(path) for path in paths])
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255
    images = F.pad(images, PADDING, mode="reflect").mul_(2).sub_(1).unsqueeze(0)
    padded_k = torch.tensor(intrinsics, dtype=torch.float32).clone()
    padded_k[0, 2] += PADDING[0]
    padded_k[1, 2] += PADDING[2]
    queries = torch.from_numpy(validate_query_points(query_points)).unsqueeze(0)
    queries = queries + queries.new_tensor([PADDING[0], PADDING[2]])
    return images, padded_k.expand(1, len(paths), 3, 3).clone(), queries


def validate_query_points(points: Any) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.shape != (2, 2) or not np.isfinite(value).all():
        raise ValueError("Select two finite pixel points, in left/right TCP order")
    if np.any(value < 0) or np.any(value[:, 0] >= WIDTH) or np.any(value[:, 1] >= HEIGHT):
        raise ValueError("TCP query points fall outside the original RGB image")
    return value.copy()


def project_queries(positions: np.ndarray, intrinsics: np.ndarray, *, allow_outside: bool = False) -> np.ndarray:
    xyz = np.asarray(positions, dtype=np.float32)
    if xyz.shape != (2, 3) or not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 1e-6):
        raise ValueError("Cannot propagate TCP queries: non-finite or behind-camera TCP")
    pixels = xyz @ intrinsics.T
    if np.any(np.abs(pixels[:, 2]) <= 1e-6):
        raise ValueError("Cannot project TCP queries: zero projection denominator")
    uv = pixels[:, :2] / pixels[:, 2:3]
    if not np.isfinite(uv).all():
        raise ValueError("Cannot project TCP queries: non-finite projection")
    return uv.copy() if allow_outside else validate_query_points(uv)


def load_tcp_truth(episode: EpisodeInputs) -> np.ndarray | None:
    directory = TCP_DIRS.get(episode.view)
    if directory is None:
        return None
    paths = [episode.path / directory / f"{arm}_state.npy" for arm in ARMS]
    if not all(path.is_file() for path in paths):
        return None
    states = [np.load(path, allow_pickle=False).astype(np.float32) for path in paths]
    if any(s.ndim != 2 or s.shape[1] != 7 for s in states) or len(states[0]) != len(states[1]):
        raise ValueError("TCP truth must contain matching left/right [T,7] arrays")
    return np.stack(states, axis=1)


def initial_truth_queries(episode: EpisodeInputs) -> np.ndarray:
    truth = load_tcp_truth(episode)
    if truth is None or len(truth) <= episode.frame_indices[0]:
        raise ValueError("First-frame TCP truth is unavailable; use --interactive to select TCPs")
    return project_queries(truth[episode.frame_indices[0], :, :3], episode.intrinsics)


def resolve_device_dtype(device_name: str, dtype_name: str) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type not in {"cuda", "cpu"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Device unavailable or unsupported: {device}")
    if dtype_name == "auto":
        with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, dtype_name)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference requires float32")
    return device, dtype


def load_stage2_policy(checkpoint: Path, device: torch.device, *, t5_model: str | None = None):
    from arc.models.arc.arc import Arc
    from arc.models.arc.arc_action import TCPActionPolicy
    from safetensors.torch import load_model

    checkpoint = checkpoint.expanduser()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("training_stage") != 2 or config.get("normalize_geometry", False):
        raise ValueError("A metric-geometry Stage2 checkpoint is required")
    policy = TCPActionPolicy(
        Arc(tcp_query_window_size=config.get("tcp_query_window_size", 3)),
        t5_model=t5_model or config["t5_model"], text_max_length=config["text_max_length"],
        dim=config["action_dim"], depth=config["action_depth"], heads=config["action_heads"],
        prediction_horizon=config["prediction_horizon"], time_unit_seconds=config["time_unit_seconds"],
    )
    # load_model restores omitted aliases of the geometry head's shared norms.
    load_model(policy, str(checkpoint / "model.safetensors"), strict=True, device="cpu")
    if (not torch.isfinite(policy.action_position_mean).all()
            or not torch.isfinite(policy.action_position_std).all()
            or (policy.action_position_std <= 0).any()):
        raise ValueError("Checkpoint has invalid action normalization statistics")
    return policy.eval().requires_grad_(False).to(device), config


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


@torch.inference_mode()
def infer_stage2_window(policy, images: torch.Tensor, intrinsics: torch.Tensor,
                        query_points: torch.Tensor, frame_times: torch.Tensor, instruction: str,
                        *, frequency_hz: float, steps: int = 8, seed: int = 42,
                        device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32) -> dict:
    """Infer one window, reusing reconstruction features for the action condition.

    Input RGB and queries use padded pixels. Returned arrays have no batch axis;
    historical TCPs are in their individual cameras, actions in the last camera.
    No historical/future TCP truth or future RGB is accepted by this function.
    """
    if not instruction.strip() or not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Instruction and positive finite frequency are required")
    images, intrinsics, query_points, frame_times = [
        value.to(device) for value in (images, intrinsics, query_points, frame_times)
    ]
    if images.shape[0] != 1:
        raise ValueError("Standalone window inference expects batch size one")
    generator = torch.Generator(device=device).manual_seed(seed)
    future_times = frame_times[:, -1:] + torch.arange(1, policy.prediction_horizon + 1, device=device)[None] / frequency_hz
    context = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
    with context:
        reconstruction, features = policy.reconstruct(images, query_points)
        condition = policy.make_condition(images, intrinsics, frame_times, future_times,
                                          [instruction], reconstruction, features)
        actions = policy.sample_condition(condition, steps=steps, generator=generator)
    return {
        "depth": _numpy(reconstruction["depth"][0]),
        "depth_confidence": _numpy(reconstruction["depth_conf"][0]),
        "history_position": _numpy(reconstruction["tcp_position"][0]),
        "history_rotation": _numpy(reconstruction["tcp_rotation"][0]),
        "history_gripper_probability": _numpy(reconstruction["tcp_gripper_logit"][0].float().sigmoid()),
        "history_confidence": _numpy(reconstruction["tcp_confidence"][0]),
        "history_valid": condition.history_valid[0].cpu().numpy(),
        "success": bool(actions["success"][0]),
        "action_position": _numpy(actions["action_position"][0]),
        "action_rotation": _numpy(actions["action_rotation"][0]),
        "action_gripper": actions["action_gripper"][0].cpu().numpy(),
        "action_gripper_score": _numpy(actions["action_gripper_score"][0]),
        "future_frame_times": _numpy(actions["future_frame_times"][0]),
    }


def transform_history(position: np.ndarray, rotation: np.ndarray, extrinsics: np.ndarray):
    relative = extrinsics[-1:] @ np.linalg.inv(extrinsics)
    positions = np.einsum("kij,kaj->kai", relative[:, :3, :3], position) + relative[:, None, :3, 3]
    rotations = relative[:, None, :3, :3] @ rotation
    return positions.astype(np.float32), rotations.astype(np.float32)


def future_truth(episode: EpisodeInputs, truth: np.ndarray | None, anchor: int, horizon: int) -> dict:
    from arc.action import future_actions_in_current_camera, safe_rotation_6d_to_matrix

    valid = np.zeros(horizon, dtype=bool)
    positions = np.full((horizon, 2, 3), np.nan, dtype=np.float32)
    rotations = np.full((horizon, 2, 3, 3), np.nan, dtype=np.float32)
    grippers = np.full((horizon, 2), -1, dtype=np.int64)
    if truth is not None:
        offsets = np.arange(anchor + 1, min(anchor + horizon + 1, len(episode.image_paths)))
        offsets = offsets[episode.frame_indices[offsets] < len(truth)]
        if len(offsets):
            state = truth[episode.frame_indices[offsets]]
            finite = np.isfinite(state).all(axis=(1, 2)) & np.isin(state[..., 6], [0, 1]).all(axis=1)
            offsets, state = offsets[finite], state[finite]
            if len(offsets):
                actions = future_actions_in_current_camera(torch.from_numpy(state),
                    torch.from_numpy(episode.extrinsics[offsets]), torch.from_numpy(episode.extrinsics[anchor]))
                slots = offsets - anchor - 1
                valid[slots] = True
                positions[slots] = actions[..., :3].numpy()
                rotations[slots] = safe_rotation_6d_to_matrix(actions[..., 3:9]).numpy()
                grippers[slots] = (actions[..., 9].numpy() >= 0).astype(np.int64)
    return {"valid": valid, "position": positions, "rotation": rotations, "gripper": grippers}


def trajectory_metrics(prediction: dict, truth: dict) -> dict:
    valid = truth["valid"]
    if not prediction["success"] or not valid.any():
        return {"valid_steps": int(valid.sum()), "position_ade_m": None, "position_fde_m": None}
    error = np.linalg.norm(prediction["action_position"][valid] - truth["position"][valid], axis=-1)
    return {"valid_steps": int(valid.sum()), "position_ade_m": float(error.mean()),
            "position_fde_m": float(error[-1].mean())}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_result(result: dict, output: Path) -> None:
    temporary = output / "predictions.json.tmp"
    temporary.write_text(json.dumps(json_safe({key: value for key, value in result.items() if key != "_geometry"}),
                                    ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(output / "predictions.json")


def infer_episode_sliding_windows(policy, episode: EpisodeInputs, initial_queries: np.ndarray,
                                  output: Path, *, config: dict, query_source: str,
                                  stride: int = 7, steps: int | None = None, seed: int = 42,
                                  device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32,
                                  max_windows: int | None = None, keep_geometry: bool = True,
                                  progress: Callable[[float, str], None] | None = None) -> dict:
    """Keep geometry in RAM for playback; save only trajectory JSON, including failures."""
    windows = build_windows(len(episode.image_paths), config["history_frames"], stride)
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    scheduled = windows if max_windows is None else windows[:max_windows]
    queries = validate_query_points(initial_queries)
    projected_queries = queries.copy()
    query_clipped = np.zeros(2, dtype=bool)
    steps = config.get("sampling_steps", 8) if steps is None else steps
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    truth = load_tcp_truth(episode)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "format_version": 1, "episode": str(episode.path), "view": episode.view,
        "instruction": episode.instruction, "frequency_hz": episode.frequency_hz,
        "history_frames": config["history_frames"], "prediction_horizon": policy.prediction_horizon,
        "window_stride": stride, "sampling_steps": steps, "seed": seed,
        "coordinate_system": "Each window: last observation OpenCV camera (+x right, +y down, +z forward), metres",
        "geometry_source": "predicted metric depth with episode intrinsics/extrinsics",
        "gripper_convention": "1=open, 0=closed; future score >=0 means open (score is not a probability)",
        "planned_windows": len(windows), "complete": False, "windows": [], "_geometry": [],
    }
    save_result(result, output)
    try:
        for number, (start, end) in enumerate(scheduled):
            message = f"Window {number + 1}/{len(scheduled)}: frames {episode.frame_indices[start]}–{episode.frame_indices[end - 1]}"
            print(message, flush=True)
            if progress:
                progress(number / len(scheduled), message)
            started = time.perf_counter()
            images, intrinsics, query_tensor = prepare_images(episode.image_paths[start:end], episode.intrinsics, queries)
            frame_times = torch.tensor(episode.frame_indices[start:end] / episode.frequency_hz, dtype=torch.float32)[None]
            prediction = infer_stage2_window(policy, images, intrinsics, query_tensor, frame_times,
                episode.instruction, frequency_hz=episode.frequency_hz, steps=steps, seed=seed + number,
                device=device, dtype=dtype)
            camera_positions = prediction["history_position"]
            position, rotation = transform_history(camera_positions, prediction["history_rotation"], episode.extrinsics[start:end])
            gt = future_truth(episode, truth, end - 1, policy.prediction_horizon)
            # Copy cropped arrays to release padded storage; never write NPY/NPZ.
            depth = prediction.pop("depth")
            confidence = prediction.pop("depth_confidence")
            if keep_geometry:
                result["_geometry"].append({
                    "depth": depth[:, PADDING[2]:PADDING[2] + HEIGHT, PADDING[0]:PADDING[0] + WIDTH].copy(),
                    "confidence": confidence[:, PADDING[2]:PADDING[2] + HEIGHT, PADDING[0]:PADDING[0] + WIDTH].copy(),
                })
            del depth, confidence
            record = {
                "window_index": number, "start": start, "end": end,
                "frame_indices": episode.frame_indices[start:end], "anchor_frame": int(episode.frame_indices[end - 1]),
                "future_frame_indices": episode.frame_indices[end - 1] + np.arange(1, policy.prediction_horizon + 1),
                "query_points_px": queries.copy(), "query_source": query_source,
                "query_points_projected_px": projected_queries.copy(),
                "query_points_clipped": query_clipped.copy(),
                "history_position": position, "history_rotation": rotation,
                **{key: value for key, value in prediction.items() if key not in {"history_position", "history_rotation"}},
                "ground_truth": gt, "metrics": trajectory_metrics(prediction, gt),
                "inference_seconds": time.perf_counter() - started,
            }
            result["windows"].append(record)
            save_result(result, output)
            print(f"  {record['inference_seconds']:.2f}s; success={prediction['success']}; ADE={record['metrics']['position_ade_m']}", flush=True)
            if number + 1 < len(scheduled):
                next_start = scheduled[number + 1][0]
                projected_queries = project_queries(
                    camera_positions[next_start - start], episode.intrinsics, allow_outside=True,
                )
                # Match training's initial query fallback. This only changes the
                # visual sampling point, not the recovered 3D TCP or history mask.
                queries = np.clip(projected_queries, [0, 0], [WIDTH - 1, HEIGHT - 1]).astype(np.float32)
                query_clipped = np.any(queries != projected_queries, axis=-1)
                if query_clipped.any():
                    arms = ", ".join(arm for arm, clipped in zip(ARMS, query_clipped) if clipped)
                    print(f"  Warning: next-window {arms} TCP queries clipped to image bounds: "
                          f"{projected_queries.tolist()} -> {queries.tolist()}", flush=True)
                query_source = f"window {number} recovered TCP at frame {episode.frame_indices[next_start]}"
        result["complete"] = len(scheduled) == len(windows)
        if not result["complete"]:
            result["stop_reason"] = "max_windows limit"
        save_result(result, output)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        save_result(result, output)
        raise
    if progress:
        progress(1.0, "Inference complete")
    return result


def make_point_cloud(depth: np.ndarray, confidence: np.ndarray, rgb: np.ndarray,
                     intrinsics: np.ndarray, frame_w2c: np.ndarray, anchor_w2c: np.ndarray,
                     *, confidence_percentile: float = 2.5, max_points: int = 100_000,
                     max_depth: float | None = 3.0, seed: int = 42):
    if not 0 <= confidence_percentile <= 99 or max_points < 0:
        raise ValueError("Invalid point cloud filtering options")
    rows, columns = np.indices(depth.shape, dtype=np.float32)
    rays = np.stack((columns, rows, np.ones_like(rows)), axis=-1) @ np.linalg.inv(intrinsics).T
    points = rays * depth[..., None]
    transform = anchor_w2c @ np.linalg.inv(frame_w2c)
    points = points @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & np.isfinite(points).all(-1)
    if max_depth is not None:
        valid &= depth <= max_depth
    if valid.any():
        valid &= confidence >= np.percentile(confidence[valid], confidence_percentile)
    indices = np.flatnonzero(valid)
    if max_points and len(indices) > max_points:
        indices = np.random.default_rng(seed).choice(indices, max_points, replace=False)
    return points.reshape(-1, 3)[indices].astype(np.float32), rgb.reshape(-1, 3)[indices]


def build_playback_frames(result: dict) -> list[tuple[int, int]]:
    """Use the first reconstruction covering each frame, including overlap/tail."""
    windows = result["windows"]
    geometry = result.get("_geometry", [])
    if not windows or len(geometry) != len(windows):
        raise ValueError("Playback requires in-memory geometry for every completed window")
    slots = {}
    for index, record in enumerate(windows):
        for frame in range(record["start"], record["end"]):
            slots.setdefault(frame, (index, frame - record["start"]))
    frames = sorted(slots)
    if frames != list(range(len(frames))):
        raise ValueError("Playback windows must cover contiguous episode frames starting at zero")
    return [slots[frame] for frame in frames]


class Stage2Viewer:
    """Play episode frames in order using in-memory window geometry."""

    def __init__(self, result: dict, episode: EpisodeInputs, *, host="127.0.0.1", port=8020,
                 point_size=0.003, fps=5.0, confidence_percentile=2.5, max_points=100_000, max_depth=3.0):
        import viser
        import viser.transforms as tf

        self.tf = tf
        self.result, self.episode = result, episode
        self.max_points, self.max_depth = max_points, max_depth
        self.frame_slots = build_playback_frames(result)
        if not math.isfinite(fps) or not 0.25 <= fps <= 30:
            raise ValueError("FPS must be between 0.25 and 30")
        if not math.isfinite(point_size) or point_size <= 0:
            raise ValueError("Point size must be finite and positive")
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("-y")
        self.server.gui.set_panel_label("4RC Stage2: reconstruction + future TCP")
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._cloud_handle = None
        self._nodes = []
        gui = self.server.gui
        with gui.add_folder("Playback", expand_by_default=True):
            self.frame = gui.add_slider("Frame", min=0, max=max(1, len(self.frame_slots) - 1), step=1, initial_value=0,
                                        disabled=len(self.frame_slots) == 1)
            self.previous = gui.add_button("Previous")
            self.next = gui.add_button("Next")
            self.play = gui.add_checkbox("Play", False)
            self.fps = gui.add_slider("FPS", min=0.25, max=30, step=0.25, initial_value=fps)
        with gui.add_folder("Geometry", expand_by_default=True):
            self.point_size = gui.add_slider("Point size", min=min(1e-5, point_size), max=max(0.01, point_size),
                                             step=1e-5, initial_value=point_size)
            self.confidence = gui.add_slider("Confidence percentile", min=0, max=99, step=0.5,
                                             initial_value=confidence_percentile)
            self.show_cloud = gui.add_checkbox("Point cloud", True)
        with gui.add_folder("TCP", expand_by_default=True):
            self.future = gui.add_slider("Future steps", min=1, max=result["prediction_horizon"], step=1,
                                         initial_value=result["prediction_horizon"])
            self.show_history = gui.add_checkbox("Recovered history", True)
            self.show_prediction = gui.add_checkbox("Predicted future", True)
            self.show_gt = gui.add_checkbox("Ground truth (dashed)", True)
            self.show_axes = gui.add_checkbox("TCP orientation axes", True)
        gui.add_markdown(f"**Instruction:** {episode.instruction}\n\nLeft: orange; right: blue. GT: yellow / green.")
        self.info = gui.add_markdown("")
        self.rgb_handle = gui.add_image(load_rgb(episode.image_paths[0]), label="Historical RGB")
        for control in (self.frame, self.future, self.show_cloud, self.show_history,
                        self.show_prediction, self.show_gt, self.show_axes, self.confidence):
            control.on_update(lambda _: self.refresh())
        self.previous.on_click(lambda _: self.step_frame(-1))
        self.next.on_click(lambda _: self.step_frame(1))
        self.play.on_update(lambda _: self._toggle_playback())
        self.point_size.on_update(lambda _: self._update_point_size())
        self.refresh()

        @self.server.on_client_connect
        def initialize_camera(client):
            client.camera.position = (0.0, -0.1, -0.5)
            client.camera.look_at = (0.0, 0.0, 1.0)
            client.camera.up_direction = (0.0, -1.0, 0.0)

        from viser_video_export import EpisodeVideoExport

        self.video_export = EpisodeVideoExport(
            self.server, frame_count=len(self.frame_slots), frame=self.frame, fps=self.fps,
            controls=[self.frame, self.previous, self.next, self.play, self.fps,
                      self.point_size, self.confidence, self.show_cloud, self.future,
                      self.show_history, self.show_prediction, self.show_gt, self.show_axes],
            render_frame=self.refresh, lock=self._lock, stop_event=self._closed,
            filename="stage2_episode.mp4",
        )
        self._thread = threading.Thread(target=self._playback, daemon=True)
        self._thread.start()

    def step_frame(self, direction=1):
        with self._lock:
            if not self.video_export.busy.is_set():
                self.frame.value = (int(self.frame.value) + direction) % len(self.frame_slots)

    def _toggle_playback(self):
        self.frame.disabled = bool(self.play.value) or len(self.frame_slots) == 1
        self.previous.disabled = self.next.disabled = bool(self.play.value)

    def _update_point_size(self):
        with self._lock:
            if self._cloud_handle is not None:
                self._cloud_handle.point_size = float(self.point_size.value)
        self.server.flush()

    def _playback(self):
        deadline = time.monotonic()
        while not self._closed.wait(0.01):
            now = time.monotonic()
            if not self.play.value:
                deadline = now
            elif now >= deadline:
                self.step_frame()
                deadline = now + 1.0 / float(self.fps.value)

    def _trajectory(self, name, positions, colors, *, dashed=False, width=3):
        positions = np.asarray(positions, dtype=np.float32)
        for arm, color in enumerate(colors):
            p = positions[:, arm]
            finite = np.isfinite(p).all(-1)
            if finite.any():
                self._nodes.append(self.server.scene.add_point_cloud(f"/{name}/{arm}/points", points=p[finite],
                    colors=np.broadcast_to(np.array(color, dtype=np.uint8), (int(finite.sum()), 3)).copy(), point_size=0.006))
            segments = np.stack((p[:-1], p[1:]), axis=1)
            segments = segments[finite[:-1] & finite[1:]]
            if dashed and len(segments):
                delta = segments[:, 1] - segments[:, 0]
                segments = np.concatenate([np.stack((segments[:, 0] + delta * lo, segments[:, 0] + delta * hi), 1)
                                           for lo, hi in ((0, 0.2), (0.4, 0.6), (0.8, 1))])
            if len(segments):
                self._nodes.append(self.server.scene.add_line_segments(f"/{name}/{arm}/lines", points=segments,
                    colors=np.broadcast_to(np.array(color, dtype=np.uint8), segments.shape).copy(), line_width=width))

    def _axes(self, name, positions, rotations):
        for arm in range(2):
            if np.isfinite(positions[arm]).all() and np.isfinite(rotations[arm]).all():
                self._nodes.append(self.server.scene.add_frame(f"/{name}/{arm}", position=positions[arm],
                    wxyz=self.tf.SO3.from_matrix(rotations[arm]).wxyz, axes_length=0.035, axes_radius=0.001))

    def refresh(self, frame_slot=None):
        with self._lock, self.server.atomic():
            if self._closed.is_set():
                return
            index, local = self.frame_slots[int(self.frame.value) if frame_slot is None else frame_slot]
            record = self.result["windows"][index]
            geometry = self.result["_geometry"][index]
            for node in self._nodes:
                node.remove()
            self._nodes.clear()
            self._cloud_handle = None
            frame = record["start"] + local
            rgb = load_rgb(self.episode.image_paths[frame])
            self.rgb_handle.image = rgb
            if self.show_cloud.value:
                points, colors = make_point_cloud(geometry["depth"][local], geometry["confidence"][local],
                    rgb, self.episode.intrinsics, self.episode.extrinsics[frame], self.episode.extrinsics[record["end"] - 1],
                    confidence_percentile=self.confidence.value, max_points=self.max_points, max_depth=self.max_depth)
                self._cloud_handle = self.server.scene.add_point_cloud("/cloud", points=points, colors=colors,
                    point_size=float(self.point_size.value), point_shape="rounded")
                self._nodes.append(self._cloud_handle)
            hp = np.asarray(record["history_position"], dtype=np.float32)
            hr = np.asarray(record["history_rotation"], dtype=np.float32)
            ap = np.asarray(record["action_position"], dtype=np.float32)
            ar = np.asarray(record["action_rotation"], dtype=np.float32)
            n = self.future.value
            if self.show_history.value:
                self._trajectory("history", hp, ARM_COLORS, width=1)
                if self.show_axes.value:
                    self._axes("history_axes", hp[local], hr[local])
            if self.show_prediction.value and record["success"]:
                self._trajectory("prediction", np.concatenate((hp[-1:], ap[:n])), ARM_COLORS, width=4)
                if self.show_axes.value:
                    self._axes("future_axes", ap[n - 1], ar[n - 1])
            gt = record["ground_truth"]
            if self.show_gt.value:
                gp = np.array(gt["position"], dtype=np.float32)[:n]
                gp[~np.asarray(gt["valid"])[:n]] = np.nan
                self._trajectory("truth", gp, GT_COLORS, dashed=True, width=2)
            grip = np.asarray(record["action_gripper"])[n - 1]
            states = ", ".join(f"{arm}: {'open' if value == 1 else 'closed' if value == 0 else 'invalid'}" for arm, value in zip(ARMS, grip))
            self.info.content = (f"**Frame {self.episode.frame_indices[frame]}** · window {index} · history {record['frame_indices'][0]}–{record['anchor_frame']} "
                f"· future +{n}\n\nSuccess: **{record['success']}** · {states}\n\n"
                f"GT steps: {record['metrics']['valid_steps']} · ADE (m): {record['metrics']['position_ade_m']}\n\n"
                "Coordinates: last historical camera; geometry is historical, trajectories are future predictions.")

    def close(self):
        self._closed.set()
        self._thread.join(timeout=2)
        # Wait for any ongoing refresh before releasing the previous run's cache.
        with self._lock:
            self.result = {}
            self._nodes.clear()
            self._cloud_handle = None
        self.server.stop()


def draw_queries(rgb: np.ndarray, points: list) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=ARM_COLORS[index], width=2)
        draw.text((x + 7, y), ARMS[index], fill=ARM_COLORS[index])
    return np.asarray(image)


def start_interactive_page(args, episode: EpisodeInputs, device, dtype):
    import gradio as gr

    rgb = load_rgb(episode.image_paths[0])
    try:
        gt_queries = initial_truth_queries(episode)
    except (ValueError, FileNotFoundError):
        gt_queries = None
    runtime = {"policy": None, "config": None, "viewer": None}

    def select(points, event: gr.SelectData):
        points = [] if len(points) == 2 else list(points)
        points.append([float(event.index[0]), float(event.index[1])])
        return draw_queries(rgb, points), points, "interactive first-frame clicks", gr.update(interactive=len(points) == 2)

    select.__annotations__["event"] = gr.SelectData

    def use_truth():
        points = gt_queries.tolist()
        return draw_queries(rgb, points), points, "projected first-frame ground-truth TCP", gr.update(interactive=True)

    def reset():
        return rgb, [], "interactive first-frame clicks", gr.update(interactive=False)

    def run(points, source, instruction, progress=gr.Progress()):
        queries = validate_query_points(points)
        if not instruction.strip():
            raise gr.Error("任务指令不能为空")
        episode.instruction = instruction.strip()
        if runtime["viewer"] is not None:
            runtime["viewer"].close()
            runtime["viewer"] = None
        progress(0, desc="加载 Stage2 模型")
        if runtime["policy"] is None:
            runtime["policy"], runtime["config"] = load_stage2_policy(args.checkpoint, device, t5_model=args.t5_model)
        result = run_episode(args, episode, runtime["policy"], runtime["config"], queries, source, device, dtype,
                             progress=lambda fraction, message: progress(fraction, desc=message))
        if args.headless:
            return f"已保存 {len(result['windows'])} 个窗口到 {args.output_dir}", ""
        runtime["viewer"] = create_viewer(args, result, episode)
        host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        url = html.escape(f"http://{host}:{runtime['viewer'].server.get_port()}", quote=True)
        return f"已保存 {len(result['windows'])} 个窗口到 {args.output_dir}", (
            f'<a href="{url}" target="_blank">打开 Viser</a>'
            f'<iframe src="{url}" title="Stage2 viewer" style="width:100%;height:740px;border:0"></iframe>')

    with gr.Blocks(title="4RC Stage2 episode inference") as demo:
        gr.Markdown("# Stage2 点云重建与未来轨迹\n依次点击首帧的左、右 TCP，或使用首帧真值投影。后续窗口自动传递恢复的 TCP。")
        points, source = gr.State([]), gr.State("interactive first-frame clicks")
        image = gr.Image(rgb, type="numpy", interactive=False, label="先选左 TCP，再选右 TCP", buttons=[])
        instruction = gr.Textbox(value=episode.instruction, label="任务指令")
        with gr.Row():
            truth_button = gr.Button("使用首帧真值投影", interactive=gt_queries is not None)
            reset_button = gr.Button("重置选点")
            run_button = gr.Button("推理完整 episode", variant="primary", interactive=False)
        status, viewer = gr.Markdown("尚未运行"), gr.HTML()
        selection_outputs = [image, points, source, run_button]
        image.select(select, inputs=[points], outputs=selection_outputs, queue=False)
        truth_button.click(use_truth, outputs=selection_outputs, queue=False)
        reset_button.click(reset, outputs=selection_outputs, queue=False)
        run_button.click(run, inputs=[points, source, instruction], outputs=[status, viewer], concurrency_limit=1)
    try:
        demo.queue(default_concurrency_limit=1).launch(server_name=args.ui_host, server_port=args.ui_port, show_error=True)
    finally:
        if runtime["viewer"] is not None:
            runtime["viewer"].close()


def run_episode(args, episode, policy, config, queries, source, device, dtype, progress=None):
    return infer_episode_sliding_windows(policy, episode, queries, args.output_dir, config=config,
        query_source=source, stride=args.window_stride, steps=args.sampling_steps, seed=args.seed,
        device=device, dtype=dtype, max_windows=args.max_windows, keep_geometry=not args.headless, progress=progress)


def create_viewer(args, result, episode):
    return Stage2Viewer(result, episode, host=args.host, port=args.port,
        point_size=args.point_size, fps=args.fps, confidence_percentile=args.confidence_percentile,
        max_points=args.max_points, max_depth=args.max_depth or None)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, required=True, help="RoboTwin episode directory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--view", default="third_views")
    parser.add_argument("--instruction", help="Override metadata's first non-empty instruction")
    parser.add_argument("--window-stride", type=int, default=7)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t5-model", help="Local T5 directory or pretrained identifier")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--interactive", action="store_true", help="Gradio first-frame selection and instruction editor")
    parser.add_argument("--headless", action="store_true", help="Save outputs without opening Viser")
    parser.add_argument("--max-windows", type=int, help="Debug limit; output is marked incomplete if truncated")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", type=int, default=7860)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--fps", type=float, default=5.0, help="Initial playback frames per second (0.25–30)")
    parser.add_argument("--confidence-percentile", type=float, default=2.5)
    parser.add_argument("--max-points", type=int, default=100_000, help="Per displayed frame; 0 keeps all points")
    parser.add_argument("--max-depth", type=float, default=3.0, help="Display depth cap in metres; 0 disables")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("outputs/stage2_inference") / args.input.resolve().parent.name / args.input.resolve().name
    args.output_dir = args.output_dir.expanduser().resolve()
    if not 0 <= args.confidence_percentile <= 99 or args.max_points < 0:
        parser.error("Invalid point cloud filtering settings")
    if not math.isfinite(args.point_size) or args.point_size <= 0 or not math.isfinite(args.max_depth) or args.max_depth < 0:
        parser.error("point-size must be positive and max-depth non-negative (both finite)")
    if not math.isfinite(args.fps) or not 0.25 <= args.fps <= 30:
        parser.error("fps must be between 0.25 and 30")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("sampling-steps must be positive")
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("max-windows must be positive")
    return args


def main():
    args = parse_args()
    episode = load_episode(args.input, args.view, args.instruction)
    config = json.loads((args.checkpoint.expanduser() / "config.json").read_text())
    build_windows(len(episode.image_paths), config["history_frames"], args.window_stride)
    device, dtype = resolve_device_dtype(args.device, args.dtype)
    print(f"Device: {device}; dtype: {dtype}; output: {args.output_dir}", flush=True)
    if args.interactive:
        start_interactive_page(args, episode, device, dtype)
        return
    queries = initial_truth_queries(episode)
    policy, config = load_stage2_policy(args.checkpoint, device, t5_model=args.t5_model)
    result = run_episode(args, episode, policy, config, queries, "projected first-frame ground-truth TCP", device, dtype)
    print(f"Saved {len(result['windows'])} windows: {args.output_dir / 'predictions.json'}", flush=True)
    if not args.headless:
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        viewer = create_viewer(args, result, episode)
        try:
            print(f"Viser port: {viewer.server.get_port()}; Ctrl-C to stop", flush=True)
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()


if __name__ == "__main__":
    main()
