#!/usr/bin/env python3
"""Single-camera, single-arm DROID Stage 1 episode inference.

Run from the repository root (activate the 4rc environment first)::

    python droid_script/infer_4rc_stage1.py --input datasets/droid_episodes/<episode>
    python droid_script/infer_4rc_stage1.py --input datasets/droid_episodes/<episode> --interactive
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from droid_script.checkpoints import weight_file
from tcp_sliding_window_inference import (
    _frame_index, _remove_training_prefix, build_sliding_windows,
    collect_rgb_paths, matrix_to_rpy, resolve_device, resolve_dtype, write_json_atomic,
)


DEFAULT_MODEL = Path("checkpoints/Droid-Stage1/checkpoint-250000/model.safetensors")
DEFAULT_OUTPUT_ROOT = Path("outputs/droid/stage1_inference")
SOURCE_WIDTH, SOURCE_HEIGHT = 320, 180
PADDED_WIDTH, PADDED_HEIGHT = 322, 182
FRAME_RATE = 15


@dataclass
class Episode:
    path: Path
    camera: str
    image_paths: list[Path]
    intrinsics: np.ndarray

    def clip(self, start_frame: int, max_frames: int = 0) -> list[Path]:
        if not 0 <= start_frame < len(self.image_paths):
            raise ValueError(f"Start frame {start_frame} outside episode [0,{len(self.image_paths) - 1}]")
        if max_frames < 0 or max_frames == 1:
            raise ValueError("--max-frames must be 0 (all remaining frames) or at least 2")
        paths = self.image_paths[start_frame:]
        if max_frames:
            paths = paths[:max_frames]
        if len(paths) < 2:
            raise ValueError(f"Camera {self.camera}: fewer than 2 frames remain from frame {start_frame}")
        return paths


@dataclass
class Prediction:
    tcp: dict[str, np.ndarray]
    windows: list[dict[str, Any]]
    source_windows: list[list[int]]
    clouds: list[dict[str, np.ndarray]] | None
    camera_to_base: np.ndarray | None


def load_episode(path: Path, camera: str | None = None) -> Episode:
    path = path.expanduser().resolve()
    image_root = path / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Expected a DROID episode with images/: {path}")
    cameras = sorted(p.name for p in image_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not cameras:
        raise ValueError(f"No camera directories found in {image_root}")
    camera = camera or cameras[0]
    if camera not in cameras:
        raise ValueError(f"Unknown camera {camera!r}; available cameras: {', '.join(cameras)}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    count = metadata.get("frame_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"Expected integer frame_count >= 2 in {path / 'metadata.json'}")
    paths = collect_rgb_paths(image_root / camera, max_frames=0, start_frame=0)
    if len(paths) != count:
        raise ValueError(f"Camera {camera}: metadata declares {count} frames, found {len(paths)}")
    k_path = path / "intrinsic" / f"{camera}.npy"
    k = np.load(k_path, allow_pickle=False).astype(np.float32)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError(f"Expected finite pinhole intrinsic matrix [3,3]: {k_path}")
    return Episode(path, camera, paths, k)


def validate_query(point: Any, source: str = "TCP query") -> np.ndarray:
    point = np.asarray(point, dtype=np.float32)
    if point.shape == (2,):
        point = point[None]
    if point.shape != (1, 2) or not np.isfinite(point).all():
        raise ValueError(f"{source}: expected one finite (x,y) point, got {point.tolist()}")
    x, y = point[0]
    if not (0 <= x < SOURCE_WIDTH and 0 <= y < SOURCE_HEIGHT):
        raise ValueError(f"{source}: TCP pixel {point[0].tolist()} outside 320x180 image")
    return point


def project_query(position: Any, intrinsics: np.ndarray, source: str) -> np.ndarray:
    xyz = np.asarray(position, dtype=np.float32)
    if xyz.shape != (1, 3) or not np.isfinite(xyz).all() or xyz[0, 2] <= 0:
        raise ValueError(f"{source}: expected finite camera TCP [1,3] with z > 0, got {xyz.tolist()}")
    homogeneous = xyz @ intrinsics.T
    return validate_query(homogeneous[:, :2] / homogeneous[:, 2:3], source)


def ground_truth_queries(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    path = episode.path / "TCP" / episode.camera / "state.npy"
    if not path.is_file():
        raise FileNotFoundError(f"No TCP labels at {path}; use --tcp-query-point X Y or --interactive")
    state = np.load(path, allow_pickle=False, mmap_mode="r")
    if state.shape != (len(episode.image_paths), 7):
        raise ValueError(f"Expected TCP state [{len(episode.image_paths)},7] in {path}, got {state.shape}")
    xyz = np.asarray(state[:, :3], dtype=np.float32)
    homogeneous = xyz @ episode.intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    valid = (np.isfinite(xyz).all(-1) & (xyz[:, 2] > 0) & np.isfinite(uv).all(-1)
             & (uv[:, 0] >= 0) & (uv[:, 0] < SOURCE_WIDTH)
             & (uv[:, 1] >= 0) & (uv[:, 1] < SOURCE_HEIGHT))
    return uv, valid


def resolve_initial_query(episode: Episode, start_frame: int | None,
                          point: Any = None) -> tuple[int, np.ndarray, str]:
    if point is not None:
        start = 0 if start_frame is None else start_frame
        episode.clip(start)
        return start, validate_query(point, "--tcp-query-point"), "manual pixel"
    uv, valid = ground_truth_queries(episode)
    if start_frame is None:
        candidates = np.flatnonzero(valid[:-1])
        if not len(candidates):
            raise ValueError(f"Camera {episode.camera}: no visible GT TCP with at least 2 remaining frames; use --interactive")
        start_frame = int(candidates[0])
    episode.clip(start_frame)
    return start_frame, validate_query(uv[start_frame], f"Camera {episode.camera}, frame {start_frame} GT"), "projected GT TCP"


def load_views(paths: list[Path]) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    views, colors = [], []
    for path in paths:
        with Image.open(path) as image:
            if image.size != (SOURCE_WIDTH, SOURCE_HEIGHT):
                raise ValueError(f"Expected native 320x180 RGB, got {image.size}: {path}")
            rgb = np.array(image.convert("RGB"))
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(127.5).sub_(1)
        tensor = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
        views.append({"img": tensor[None], "true_shape": torch.tensor([[182, 322]]),
                      "idx": _frame_index(path), "instance": str(path)})
        colors.append(rgb)
    return views, colors


def restore_inference_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Strict restoration, allowing only omitted copies of genuinely tied weights."""
    state = dict(state)
    for name in ("position_mean", "position_std"):
        value = state.get(f"tcp_track_head.{name}")
        if value is None or value.shape != (1, 3) or not torch.isfinite(value).all():
            raise ValueError(f"DROID checkpoint requires finite tcp_track_head.{name} [1,3]")
        if name == "position_std" and torch.any(value <= 0):
            raise ValueError("TCP position_std must be positive")
    groups: dict[tuple, list[str]] = {}
    for name, tensor in model.state_dict().items():
        identity = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tensor.stride())
        groups.setdefault(identity, []).append(name)
    for names in groups.values():
        source = next((name for name in names if name in state), None)
        if source is not None:
            for name in names:
                state.setdefault(name, state[source])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Incompatible DROID Stage 1 checkpoint: {error}") from error


def load_model(checkpoint: Path, device: torch.device):
    from arc.models.arc.arc import Arc

    path = weight_file(checkpoint)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        saved = load_file(str(path), device="cpu")
    else:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved = saved.get("state_dict", saved)
    state = {}
    for name, value in saved.items():
        normalized = _remove_training_prefix(name)
        if normalized in state:
            raise ValueError(f"Duplicate checkpoint parameter: {normalized}")
        state[normalized] = value
    del saved
    arm = state.get("tcp_visual_query_encoder.arm_embedding")
    offsets = state.get("tcp_visual_query_encoder.offset_embedding")
    if arm is None or arm.ndim != 2 or arm.shape[0] != 1:
        raise ValueError("Expected a trained single-arm DROID Stage 1 checkpoint")
    if offsets is None or offsets.ndim != 2:
        raise ValueError("Checkpoint is missing TCP visual query offset embeddings")
    window = math.isqrt(offsets.shape[0])
    if window < 1 or window % 2 != 1 or window * window != offsets.shape[0]:
        raise ValueError("Invalid TCP query neighborhood in checkpoint")
    model = Arc(num_arms=1, tcp_query_window_size=window)
    restore_inference_weights(model, state)
    del state
    gc.collect()
    return model.to(device).eval()


def tcp_to_base(tcp: dict[str, np.ndarray], camera_to_base: np.ndarray) -> dict[str, np.ndarray]:
    """Transform both position and orientation, retaining camera-frame JSON separately."""
    result = dict(tcp)
    rotation = camera_to_base[:, :3, :3]
    result["position"] = np.einsum("tij,taj->tai", rotation, tcp["position"]) + camera_to_base[:, None, :3, 3]
    result["rotation"] = rotation[:, None] @ tcp["rotation"]
    return result


def frame_clouds(depth: np.ndarray, confidence: np.ndarray, colors: list[np.ndarray],
                 c2w: np.ndarray, intrinsics: np.ndarray, max_points: int,
                 seed: int) -> list[dict[str, np.ndarray]]:
    yy, xx = np.mgrid[1:SOURCE_HEIGHT + 1, 1:SOURCE_WIDTH + 1]
    clouds = []
    for index, color in enumerate(colors):
        z = depth[index, 1:-1, 1:-1]
        conf = confidence[index, 1:-1, 1:-1]
        k = intrinsics[index]
        camera_points = np.stack(((xx - k[0, 2]) * z / k[0, 0],
                                  (yy - k[1, 2]) * z / k[1, 1], z), axis=-1)
        points = camera_points @ c2w[index, :3, :3].T + c2w[index, :3, 3]
        valid = np.isfinite(points).all(-1) & np.isfinite(conf) & (z > 0)
        selected = np.flatnonzero(valid.ravel())
        if max_points and len(selected) > max_points:
            selected = np.random.default_rng(seed + index).choice(selected, max_points, replace=False)
        clouds.append({"points": points.reshape(-1, 3)[selected].astype(np.float32),
                       "colors": color.reshape(-1, 3)[selected],
                       "confidence": conf.ravel()[selected].astype(np.float32)})
    return clouds


def infer_window(model: Any, paths: list[Path], query: np.ndarray, device: torch.device,
                 dtype: torch.dtype, keep_geometry: bool, max_points: int):
    views, colors = load_views(paths)
    views = [{**view, "img": view["img"].to(device)} for view in views]
    query_tensor = torch.from_numpy(query + np.float32(1))[None].to(device)
    autocast = contextlib.nullcontext() if dtype == torch.float32 else torch.autocast(device_type=device.type, dtype=dtype)
    with torch.inference_mode(), autocast:
        pred = model(views, force_no_output_conversion=True, inference_track=False,
                     decode_camera=keep_geometry, decode_motion=False, decode_tcp=True,
                     tcp_query_points=query_tensor, return_aux_pyramid=False, ref_view_strategy="first")
    def array(value):
        return value[0].detach().float().cpu().numpy()
    tcp = {"position": array(pred["tcp_position"]), "rotation": array(pred["tcp_rotation"]),
           "gripper": array(pred["tcp_gripper_logit"].float().sigmoid()),
           "confidence": array(pred["tcp_confidence"])}
    expected = {"position": (len(paths), 1, 3), "rotation": (len(paths), 1, 3, 3),
                "gripper": (len(paths), 1), "confidence": (len(paths), 1)}
    for key, value in tcp.items():
        if value.shape != expected[key] or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} prediction: expected finite {expected[key]}, got {value.shape}")
    clouds, c2w = None, None
    if keep_geometry:
        from arc.models.arc.utils.transform import pose_encoding_to_extri_intri
        with torch.inference_mode():
            cameras, k = pose_encoding_to_extri_intri(pred["pose_enc"].float(), (PADDED_HEIGHT, PADDED_WIDTH))
        c2w, k = array(cameras), array(k)
        if not np.isfinite(c2w).all() or not np.isfinite(k).all():
            raise ValueError("Camera decoder returned non-finite absolute camera parameters")
        clouds = frame_clouds(array(pred["depth"]), array(pred["depth_conf"]), colors,
                              c2w, k, max_points, _frame_index(paths[0]))
    return tcp, clouds, c2w


def infer_episode(model: Any, episode: Episode, paths: list[Path], query: np.ndarray,
                  query_source: str, *, device: torch.device, dtype: torch.dtype,
                  window_size: int = 9, keep_geometry: bool = False, max_points: int = 100_000,
                  progress: Callable[[float, str], None] | None = None) -> Prediction:
    windows = build_sliding_windows(len(paths), window_size)
    query = validate_query(query).copy()
    values = {key: [] for key in ("position", "rotation", "gripper", "confidence")}
    records, source_windows, cameras = [], [], []
    clouds = [] if keep_geometry else None
    for wi, (start, end) in enumerate(windows):
        frame_ids = [_frame_index(path) for path in paths[start:end]]
        context = f"Camera {episode.camera}, window {wi + 1}/{len(windows)}, frames {frame_ids[0]}-{frame_ids[-1]}"
        if progress:
            progress(wi / len(windows), context)
        began = time.monotonic()
        try:
            tcp, window_clouds, c2w = infer_window(model, paths[start:end], query, device, dtype, keep_geometry, max_points)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(f"{context}: CUDA out of memory; reduce --window-size") from error
        except (ValueError, OSError, RuntimeError) as error:
            raise RuntimeError(f"{context}: {error}") from error
        records.append({"window_index": wi, "frame_indices": frame_ids,
                        "query_source": query_source, "tcp_query_points_px": query.tolist(),
                        "inference_seconds": time.monotonic() - began})
        skip = int(wi > 0)
        if skip:
            source_windows[-1].append(wi)
        source_windows.extend([[wi] for _ in frame_ids[skip:]])
        for key in values:
            values[key].append(tcp[key][skip:])
        if keep_geometry:
            clouds.extend(window_clouds[skip:])
            cameras.append(c2w[skip:])
        if end < len(paths):
            query_source = f"window {wi} prediction at frame {frame_ids[-1]}"
            query = project_query(tcp["position"][-1], episode.intrinsics, f"Camera {episode.camera}, {query_source}")
    if progress:
        progress(1.0, "Sliding-window inference complete")
    return Prediction({key: np.concatenate(parts) for key, parts in values.items()}, records,
                      source_windows, clouds, np.concatenate(cameras) if keep_geometry else None)


def build_result(episode: Episode, paths: list[Path], prediction: Prediction,
                 query: np.ndarray, query_source: str, model: Path, window_size: int) -> dict:
    indices = [_frame_index(path) for path in paths]
    rpy = matrix_to_rpy(prediction.tcp["rotation"])
    frames = []
    for slot, (index, path) in enumerate(zip(indices, paths)):
        frames.append({"frame_index": index, "time_seconds": index / FRAME_RATE,
                       "image": str(path), "source_windows": prediction.source_windows[slot],
                       "tcp": {"xyz_m": prediction.tcp["position"][slot, 0].tolist(),
                               "rpy_rad": rpy[slot, 0].tolist(), "rpy_deg": np.rad2deg(rpy[slot, 0]).tolist(),
                               "gripper_open": float(prediction.tcp["gripper"][slot, 0]),
                               "confidence": float(prediction.tcp["confidence"][slot, 0])}})
    return {"schema_version": 1, "dataset": "droid", "episode": str(episode.path),
            "camera_id": episode.camera, "model": str(weight_file(model).resolve()), "num_arms": 1,
            "frame_rate_hz": FRAME_RATE, "episode_num_frames": len(episode.image_paths),
            "num_frames": len(frames), "start_frame": indices[0], "end_frame": indices[-1],
            "skipped_prefix_frames": list(range(indices[0])),
            "unprocessed_suffix_frames": list(range(indices[-1] + 1, len(episode.image_paths))),
            "coordinate_frame": "OpenCV camera (+x right, +y down, +z forward)",
            "position_unit": "meter", "rotation_unit": "radian",
            "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
            "gripper": {"encoding": "continuous", "range": [0, 1], "field": "tcp.gripper_open"},
            "windowing": {"window_size": window_size, "stride": window_size - 1,
                          "boundary_merge": "previous", "num_windows": len(prediction.windows)},
            "initial_query": {"frame_index": indices[0], "source": query_source,
                              "tcp_query_points_px": validate_query(query).tolist()},
            "windows": prediction.windows, "frames": frames}


def run_and_save(args, episode, model, start, query, query_source, device, dtype,
                 *, keep_geometry=False, progress=None):
    paths = episode.clip(start, args.max_frames)
    prediction = infer_episode(model, episode, paths, query, query_source, device=device,
                               dtype=dtype, window_size=args.window_size, keep_geometry=keep_geometry,
                               max_points=args.max_points, progress=progress)
    result = build_result(episode, paths, prediction, query, query_source, args.model, args.window_size)
    output = args.output or DEFAULT_OUTPUT_ROOT / episode.path.name / episode.camera / "tcp_episode.json"
    saved = write_json_atomic(result, output)
    return result, prediction, paths, saved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="One complete DROID episode directory")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="DROID Stage 1 checkpoint directory or weights")
    parser.add_argument("--camera", help="Camera serial; initial selection in interactive mode; defaults to first sorted camera")
    parser.add_argument("--output", type=Path, help="JSON destination (replaced only after successful inference)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--interactive", action="store_true", help="Select one TCP in a browser, then show Viser")
    group.add_argument("--tcp-query-point", nargs=2, type=float, metavar=("X", "Y"), help="Original 320x180 image pixel")
    parser.add_argument("--start-frame", type=int, help="Explicit start; otherwise first visible GT TCP, or 0 for manual query")
    parser.add_argument("--max-frames", type=int, default=0, help="0: all remaining frames")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--visualize", action="store_true", help="Show Viser after non-interactive inference")
    parser.add_argument("--show-pred-trajectory", action=argparse.BooleanOptionalAction, default=True,
                        help="Initially show the predicted TCP trajectory in Viser (orange)")
    parser.add_argument("--show-gt-trajectory", action=argparse.BooleanOptionalAction, default=True,
                        help="Initially show the GT TCP trajectory in Viser when labels/extrinsics are available (green)")
    parser.add_argument("--max-points", type=int, default=100_000, help="Per-frame point cap; 0 keeps all")
    parser.add_argument("--confidence-percentile", type=float, default=2.5)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--ui-host", default="127.0.0.1", help="Gradio bind address")
    parser.add_argument("--ui-port", type=int, default=7860)
    args = parser.parse_args(argv)
    if not 2 <= args.window_size <= 18:
        parser.error("--window-size must be between 2 and 18")
    if args.max_frames < 0 or args.max_frames == 1:
        parser.error("--max-frames must be 0 or at least 2")
    if args.start_frame is not None and args.start_frame < 0:
        parser.error("--start-frame cannot be negative")
    if args.max_points < 0 or not 0 <= args.confidence_percentile <= 99 or not math.isfinite(args.point_size) or args.point_size <= 0:
        parser.error("Invalid point-cloud display settings")
    if not all(1 <= port <= 65535 for port in (args.port, args.ui_port)):
        parser.error("Ports must be in [1,65535]")
    if args.interactive and args.port == args.ui_port:
        parser.error("Gradio and Viser ports must differ")
    return args


def main(argv=None):
    args = parse_args(argv)
    episode = load_episode(args.input, args.camera)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    weight_file(args.model)
    print(f"Episode: {episode.path.name}; camera: {episode.camera}; frames: {len(episode.image_paths)}", flush=True)
    print(f"Checkpoint: {args.model}; device: {device}; dtype: {dtype}", flush=True)
    if args.interactive:
        from droid_script.stage1_viewer import start_interactive
        start_interactive(args, episode, device, dtype)
        return
    start, query, source = resolve_initial_query(episode, args.start_frame, args.tcp_query_point)
    episode.clip(start, args.max_frames)
    print(f"Start frame: {start}; skipped prefix: {start} frames; query: {query.tolist()} ({source})", flush=True)
    model = load_model(args.model, device)
    try:
        result, prediction, paths, saved = run_and_save(
            args, episode, model, start, query, source, device, dtype,
            keep_geometry=args.visualize, progress=lambda fraction, message: print(message, flush=True))
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Saved {result['num_frames']} frames ({result['start_frame']}-{result['end_frame']}) to {saved}", flush=True)
    if args.visualize:
        from droid_script.stage1_viewer import start_viewer
        viewer = start_viewer(args, episode, prediction, paths)
        try:
            while not viewer.stopped.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            viewer.stop()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
