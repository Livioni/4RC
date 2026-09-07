#!/usr/bin/env python3
"""Infer a complete RoboTwin episode with overlapping TCP windows.

Each window shares one boundary frame with the next window. The final metric
TCP positions from one window are projected into that shared image to become
the next window's left/right visual query points.

Examples:

    python tcp_sliding_window_inference.py \
        --input datasets/RoboTwin/<task>/<episode> \
        --output outputs/tcp_episode.json

    python tcp_sliding_window_inference.py \
        --input datasets/RoboTwin_random_subset/beat_block_hammer/episode_0000000 \
        --output outputs/tcp_episode.json \
        --interactive
"""

from __future__ import annotations

import argparse
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

from geometry_inference import load_robotwin_views, resolve_device, resolve_dtype
from tcp_inference import (
    ARM_NAMES,
    DEFAULT_MODEL,
    TRAIN_MAX_FRAMES,
    _frame_index,
    _load_query_points_file,
    _validate_query_points,
    add_interactive_query_point,
    collect_rgb_paths,
    infer_tcp_and_geometry,
    infer_tcp_trajectory,
    load_ground_truth_query_points,
    load_tcp_model,
    matrix_to_rpy,
    prepare_frame_point_clouds,
    project_tcp_positions_to_query_points,
    render_query_overlay,
    start_visualization,
    stop_visualization,
)


RPY_CONVENTION = "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))"
CAMERA_SUFFIX = "OpenCV camera (+x right, +y down, +z forward)"
BOUNDARY_POLICIES = ("previous", "next", "average")


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
        required=True,
        help="Complete RoboTwin episode directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON trajectory",
    )
    parser.add_argument(
        "--view",
        default="third_views",
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
        default=9,
        help="Frames per window; neighboring windows overlap by one frame",
    )
    parser.add_argument(
        "--boundary-merge",
        choices=BOUNDARY_POLICIES,
        default="previous",
        help="Which TCP prediction to emit for shared boundary frames",
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
        "--visualize",
        action="store_true",
        help="Start Viser after non-interactive JSON inference",
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
        help="Viser point size in world units",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address")
    parser.add_argument("--port", type=int, default=8020, help="Viser port")
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
    return parser.parse_args()


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
) -> SlidingWindowPrediction:
    """Run all windows and return one merged prediction per episode frame."""
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
                "gripper_open": gripper_probability >= 0.5,
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
            "binary_threshold": 0.5,
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
