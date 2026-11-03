#!/usr/bin/env python3
"""Run Stage2 on a cobot_magic pen/cups episode and view point clouds + TCPs.

Example:
    python cobot_magic/viser_stage2_inference.py \
        --input datasets/TCP4D_Pens_in_cups/place_pen_in_cups__episode_0

The Viser point cloud comes from the model's metric depth prediction. It is
shown together with recovered historical TCPs, predicted future TCPs and the
episode's ground-truth TCPs. All geometry in a window is expressed in its last
observation's OpenCV camera frame.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from arc.state_normalization import validate_robot_states
from stage2_sliding_window_inference import (
    EpisodeInputs,
    Stage2Viewer,
    build_windows,
    homogeneous,
    infer_episode_sliding_windows,
    initial_truth_queries,
    load_stage2_policy,
    resolve_device_dtype,
    start_interactive_page,
)


DEFAULT_INPUT = Path("datasets/TCP4D_Pens_in_cups/place_pen_in_cups__episode_0")
DEFAULT_CHECKPOINT = Path("outputs/TCP4D-pen-cups-stage2-post-train/final_checkpoint")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _indexed_images(directory: Path) -> tuple[list[Path], np.ndarray]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {directory}")
    indexed: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = re.fullmatch(r"(\d+)", path.stem)
        if match is None:
            raise ValueError(f"RGB filename must be an integer frame number: {path}")
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate RGB frame {index}: {directory}")
        indexed[index] = path
    indices = np.asarray(sorted(indexed), dtype=np.int64)
    if not len(indices) or not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"RGB frames must be contiguous and start at 000000: {directory}")
    return [indexed[int(index)] for index in indices], indices


def load_cobot_episode(input_path: Path, view: str = "third_person",
                       instruction: str | None = None) -> EpisodeInputs:
    """Adapt the cobot_magic real-robot layout to the shared Stage2 runner."""
    path = input_path.expanduser().resolve()
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "cobot_magic_robolab_dual_arm_v1":
        raise ValueError(f"Unsupported cobot_magic episode format: {metadata.get('format')!r}")
    if metadata.get("complete") is not True:
        raise ValueError(f"Episode is not marked complete: {path}")
    if view != metadata.get("camera_id", "third_person"):
        raise ValueError(f"Episode camera is {metadata.get('camera_id')!r}, not {view!r}")

    rate = metadata.get("fps")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("metadata.json requires a finite positive fps")
    if instruction is None:
        texts = metadata.get("language_instruction", [])
        instruction = next((text.strip() for text in texts
                            if isinstance(text, str) and text.strip()), "") if isinstance(texts, list) else ""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Provide --instruction or a non-empty language_instruction")

    image_paths, frame_indices = _indexed_images(path / "images" / view)
    frame_count = metadata.get("frame_count")
    if isinstance(frame_count, bool) or frame_count != len(image_paths):
        raise ValueError(f"metadata frame_count={frame_count!r}, but found {len(image_paths)} RGB frames")

    intrinsics = np.load(path / "intrinsic" / f"{view}.npy", allow_pickle=False).astype(np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or abs(np.linalg.det(intrinsics)) < 1e-8:
        raise ValueError("Expected finite, invertible intrinsic matrix [3,3]")
    extrinsic = homogeneous(np.load(path / "extrinsic" / f"{view}.npy", allow_pickle=False))
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all() or abs(np.linalg.det(extrinsic)) < 1e-8:
        raise ValueError("Expected a finite, invertible base-to-camera extrinsic [3,4] or [4,4]")
    extrinsics = np.broadcast_to(extrinsic, (len(image_paths), 4, 4)).copy()

    robot_states = np.load(path / "observations" / "joint_position.npy", allow_pickle=False).astype(np.float32)
    if robot_states.shape != (len(image_paths), 14):
        raise ValueError(f"Expected observations/joint_position.npy [{len(image_paths)},14], got {robot_states.shape}")
    robot_states = robot_states.copy()
    # cobot_magic stores the total two-finger opening in metres. Stage2 was
    # trained with a [0,1] opening, using 10 cm as fully open.
    robot_states[:, [6, 13]] = np.clip(robot_states[:, [6, 13]] / 0.10, 0.0, 1.0)
    validate_robot_states(robot_states)

    tcp_truth = np.load(path / "TCP" / view / "state.npy", allow_pickle=False).astype(np.float32)
    if tcp_truth.shape != (len(image_paths), 2, 7) or not np.isfinite(tcp_truth).all():
        raise ValueError(f"Expected finite TCP/{view}/state.npy [{len(image_paths)},2,7], got {tcp_truth.shape}")
    if np.any((tcp_truth[..., 6] < 0) | (tcp_truth[..., 6] > 1)):
        raise ValueError("TCP gripper labels must be in [0,1]")

    for side in ("left_view", "right_view"):
        wrist_paths, wrist_indices = _indexed_images(path / "images" / side)
        if len(wrist_paths) != len(image_paths) or not np.array_equal(wrist_indices, frame_indices):
            raise ValueError(f"{side} frames are not aligned with {view}")

    return EpisodeInputs(
        path=path,
        view=view,
        image_paths=image_paths,
        frame_indices=frame_indices,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        frequency_hz=float(rate),
        instruction=instruction.strip(),
        robot_states=robot_states,
        tcp_truth=tcp_truth,
    )


def create_viewer(args: argparse.Namespace, result: dict, episode: EpisodeInputs) -> Stage2Viewer:
    return Stage2Viewer(
        result,
        episode,
        host=args.host,
        port=args.port,
        point_size=args.point_size,
        fps=args.fps,
        confidence_percentile=args.confidence_percentile,
        max_points=args.max_points,
        max_depth=args.max_depth or None,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="cobot_magic episode directory")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Complete metric Stage2 checkpoint directory")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/cobot_magic_stage2/<episode>")
    parser.add_argument("--view", default="third_person")
    parser.add_argument("--instruction", help="Override metadata language_instruction")
    parser.add_argument("--window-stride", type=int, default=1,
                        help="Run a causal prediction every N frames")
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t5-model", help="Local T5 directory or pretrained identifier")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--interactive", action="store_true",
                        help="Use Gradio to edit the instruction or select the first left/right TCP")
    parser.add_argument("--headless", action="store_true", help="Save predictions without starting Viser")
    parser.add_argument("--max-windows", type=int, help="Debug limit; output will be marked incomplete")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020, help="Viser port")
    parser.add_argument("--ui-host", default="127.0.0.1")
    parser.add_argument("--ui-port", "--server-port", "--server_port", dest="ui_port", type=int, default=7860)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--fps", type=float, default=5.0, help="Viser playback FPS (0.25-30)")
    parser.add_argument("--confidence-percentile", type=float, default=2.5)
    parser.add_argument("--max-points", type=int, default=100_000, help="Per frame; 0 keeps every valid point")
    parser.add_argument("--max-depth", type=float, default=3.0, help="Display depth cap in metres; 0 disables")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path("outputs/cobot_magic_stage2") / args.input.expanduser().resolve().name
    args.output_dir = args.output_dir.expanduser().resolve()
    if not 0 <= args.confidence_percentile <= 99 or args.max_points < 0:
        parser.error("Invalid point-cloud filtering settings")
    if not math.isfinite(args.point_size) or args.point_size <= 0:
        parser.error("point-size must be finite and positive")
    if not math.isfinite(args.max_depth) or args.max_depth < 0:
        parser.error("max-depth must be finite and non-negative")
    if not math.isfinite(args.fps) or not 0.25 <= args.fps <= 30:
        parser.error("fps must be between 0.25 and 30")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("sampling-steps must be positive")
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("max-windows must be positive")
    return args


def run_episode(args, episode, policy, config, queries, source, device, dtype, progress=None):
    return infer_episode_sliding_windows(
        policy,
        episode,
        queries,
        args.output_dir,
        config=config,
        query_source=source,
        stride=args.window_stride,
        steps=args.sampling_steps,
        seed=args.seed,
        device=device,
        dtype=dtype,
        max_windows=args.max_windows,
        keep_geometry=not args.headless,
        progress=progress,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    episode = load_cobot_episode(args.input, args.view, args.instruction)
    checkpoint = args.checkpoint.expanduser()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    build_windows(len(episode.image_paths), config["history_frames"], args.window_stride)
    device, dtype = resolve_device_dtype(args.device, args.dtype)
    print(f"Episode: {episode.path}", flush=True)
    print(f"Device: {device}; dtype: {dtype}; output: {args.output_dir}", flush=True)

    if args.interactive:
        start_interactive_page(
            args,
            episode,
            device,
            dtype,
            run_callback=run_episode,
            create_viewer_callback=create_viewer,
        )
        return

    queries = initial_truth_queries(episode)
    policy, config = load_stage2_policy(checkpoint, device, t5_model=args.t5_model)
    result = run_episode(
        args, episode, policy, config, queries,
        "projected first-frame cobot_magic ground-truth TCP", device, dtype,
    )
    print(f"Saved {len(result['windows'])} windows: {args.output_dir / 'predictions.json'}", flush=True)
    if args.headless:
        return

    del policy
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()
    viewer = create_viewer(args, result, episode)
    try:
        print(f"Viser: http://127.0.0.1:{viewer.server.get_port()} (Ctrl-C to stop)", flush=True)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
