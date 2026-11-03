#!/usr/bin/env python3
"""Stage1 9-frame sliding-window inference for cobot_magic pen/cups data.

Despite the historical filename, this script loads only a Stage1 checkpoint.
Every full window contains 9 frames and advances by 8 frames, so neighboring
windows share exactly one boundary frame. The final shorter tail window follows
the same behavior as ``tcp_sliding_window_inference.py``.

Example:
    python cobot_magic/viser_stage2_inference.py
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

from tcp_sliding_window_inference import (
    BOUNDARY_POLICIES,
    EpisodeInputs,
    build_json_result,
    build_sliding_windows,
    infer_episode_sliding_windows,
    load_tcp_model,
    project_tcp_positions_to_query_points,
    resolve_device,
    resolve_dtype,
    start_visualization,
    stop_visualization,
    write_json_atomic,
)


DEFAULT_INPUT = Path("datasets/TCP4D_Pens_in_cups/place_pen_in_cups__episode_0")
DEFAULT_MODEL = Path("outputs/TCP4D-pen-cups-stage1/final_checkpoint/model.safetensors")
DEFAULT_WINDOW_SIZE = 9
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _load_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {metadata_path}: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected a JSON object: {metadata_path}")
    if metadata.get("format") != "cobot_magic_robolab_dual_arm_v1":
        raise ValueError(f"Unsupported cobot_magic format: {metadata.get('format')!r}")
    if metadata.get("complete") is not True:
        raise ValueError(f"Episode is not marked complete: {path}")
    return metadata


def _collect_images(directory: Path, max_frames: int) -> tuple[list[Path], list[int]]:
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
    indices = sorted(indexed)
    if not indices or indices != list(range(len(indices))):
        raise ValueError(f"RGB frames must be contiguous from 000000: {directory}")
    if max_frames:
        indices = indices[:max_frames]
    if len(indices) < 2:
        raise ValueError("Stage1 sliding-window inference requires at least two frames")
    return [indexed[index] for index in indices], indices


def load_cobot_episode(input_path: Path, view: str, max_frames: int = 0) -> EpisodeInputs:
    """Load the cobot_magic layout into the shared Stage1 episode structure."""
    path = input_path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    metadata = _load_metadata(path)
    if view != metadata.get("camera_id", "third_person"):
        raise ValueError(f"Episode camera is {metadata.get('camera_id')!r}, not {view!r}")
    rate = metadata.get("fps")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("metadata.json requires a finite positive fps")
    image_paths, frame_indices = _collect_images(path / "images" / view, max_frames)
    if not max_frames and metadata.get("frame_count") != len(image_paths):
        raise ValueError(
            f"metadata frame_count={metadata.get('frame_count')!r}, found {len(image_paths)} RGB frames"
        )
    intrinsics_path = path / "intrinsic" / f"{view}.npy"
    intrinsics = np.load(intrinsics_path, allow_pickle=False).astype(np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or abs(np.linalg.det(intrinsics)) < 1e-8:
        raise ValueError(f"Expected finite invertible intrinsics [3,3]: {intrinsics_path}")
    return EpisodeInputs(
        episode_path=path,
        image_paths=image_paths,
        frame_indices=frame_indices,
        intrinsics=intrinsics,
        frame_rate=float(rate),
    )


def load_initial_query(episode: EpisodeInputs, view: str,
                       explicit: list[float] | None) -> tuple[np.ndarray, str]:
    if explicit is not None:
        points = np.asarray(explicit, dtype=np.float32).reshape(2, 2)
        source = "explicit --tcp-query-points"
    else:
        tcp_path = episode.episode_path / "TCP" / view / "state.npy"
        tcp = np.load(tcp_path, mmap_mode="r", allow_pickle=False)
        frame_index = episode.frame_indices[0]
        if tcp.ndim != 3 or tcp.shape[1:] != (2, 7) or frame_index >= len(tcp):
            raise ValueError(f"Expected TCP truth [T,2,7] covering frame {frame_index}: {tcp_path}")
        points = project_tcp_positions_to_query_points(
            np.asarray(tcp[frame_index, :, :3], dtype=np.float32),
            episode.intrinsics,
            source=f"cobot_magic ground-truth TCP at frame {frame_index}",
        )
        source = "projected first-frame cobot_magic ground-truth TCP"
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Initial TCP query must be finite [2,2] in left/right order")
    if np.any(points < 0) or np.any(points[:, 0] >= 320) or np.any(points[:, 1] >= 240):
        raise ValueError(f"Initial TCP query lies outside 320x240: {points.tolist()}")
    return points, source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Stage1 image-conditioned TCP checkpoint (.safetensors)")
    parser.add_argument("--output", type=Path,
                        help="Default: outputs/cobot_magic_stage1/<episode>/tcp_episode.json")
    parser.add_argument("--view", default="third_person")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                        help="Frames per Stage1 window; stride is window-size minus one")
    parser.add_argument("--boundary-merge", choices=BOUNDARY_POLICIES, default="previous")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Debug prefix length; 0 processes the complete episode")
    parser.add_argument("--tcp-query-points", type=float, nargs=4,
                        metavar=("LEFT_X", "LEFT_Y", "RIGHT_X", "RIGHT_Y"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--headless", action="store_true", help="Save JSON without starting Viser")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--confidence-percentile", type=float, default=2.5)
    parser.add_argument("--max-points", type=int, default=100_000, help="Maximum points retained per frame; 0 keeps all")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = (Path("outputs/cobot_magic_stage1") / args.input.expanduser().resolve().name
                       / "tcp_episode.json")
    args.output = args.output.expanduser().resolve()
    if args.window_size != DEFAULT_WINDOW_SIZE:
        parser.error("this cobot_magic Stage1 entry point requires --window-size 9")
    if args.max_frames == 1 or args.max_frames < 0:
        parser.error("max-frames must be 0 or at least 2")
    if not 0 <= args.confidence_percentile <= 99:
        parser.error("confidence-percentile must be in [0,99]")
    if args.max_points < 0 or not math.isfinite(args.point_size) or args.point_size <= 0:
        parser.error("max-points must be non-negative and point-size must be finite and positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1,65535]")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    episode = load_cobot_episode(args.input, args.view, args.max_frames)
    windows = build_sliding_windows(len(episode.image_paths), args.window_size)
    queries, query_source = load_initial_query(episode, args.view, args.tcp_query_points)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(
        f"Stage1 only: {len(episode.image_paths)} frames, {len(windows)} windows, "
        f"window=9, stride=8, overlap=1; device={device}, dtype={dtype}",
        flush=True,
    )
    print(f"Windows: {windows}", flush=True)
    print(f"Initial TCP query ({query_source}): {queries.tolist()}", flush=True)

    model = load_tcp_model(args.model, device)
    try:
        prediction = infer_episode_sliding_windows(
            model,
            episode,
            queries,
            query_source,
            device=device,
            dtype=dtype,
            window_size=args.window_size,
            boundary_merge=args.boundary_merge,
            keep_geometry=not args.headless,
            max_points_per_frame=args.max_points,
            progress_callback=lambda _fraction, message: print(message, flush=True),
        )
        result = build_json_result(
            episode,
            prediction,
            model_path=args.model,
            view=args.view,
            window_size=args.window_size,
            boundary_merge=args.boundary_merge,
            initial_query_points=queries,
            initial_query_source=query_source,
        )
        output = write_json_atomic(result, args.output)
    finally:
        del model
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    print(f"Saved {result['num_frames']} Stage1 TCP frames to {output}", flush=True)
    if args.headless:
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
    print(f"Viser: http://{display_host}:{server.get_port()} (Ctrl-C to stop)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop_visualization(server)


if __name__ == "__main__":
    main()
