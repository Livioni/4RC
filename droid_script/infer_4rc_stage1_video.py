#!/usr/bin/env python3
"""Run a single-arm DROID Stage 1 checkpoint on a video or DROID/RoboLab episode.

    python droid_script/infer_4rc_stage1_video.py --input video.mp4 --interactive
    python droid_script/infer_4rc_stage1_video.py --input video.mp4 \
        --tcp-query-point 160 90 --visualize

Query pixels refer to the resized 320x180 RGB, before the training-time padding.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch

from droid_script import infer_4rc_stage1 as stage1


DEFAULT_OUTPUT_ROOT = Path("outputs/droid/stage1_video")


@dataclass
class VideoEpisode(stage1.Episode):
    fps: float
    original_size: tuple[int, int]
    output_dir: Path


def load_video(path: Path, output_root: Path = DEFAULT_OUTPUT_ROOT,
               fps_override: float | None = None) -> VideoEpisode:
    """Decode once to a persistent cache; never write beside the input video."""
    import cv2

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    if fps_override is not None and (not math.isfinite(fps_override) or fps_override <= 0):
        raise ValueError("--fps must be finite and positive")
    stat = path.stat()
    signature = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "resize": [stage1.SOURCE_WIDTH, stage1.SOURCE_HEIGHT], "version": 1}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    output_dir = output_root.expanduser().resolve() / f"{path.stem}-{key}"
    cache = output_dir / "frames"
    manifest = cache / "video.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [cache / f"{i:08d}.png" for i in range(metadata["num_frames"])]
        if metadata["signature"] != signature or len(paths) < 2 or not all(p.is_file() for p in paths):
            raise ValueError(f"Incomplete video cache: {cache}; remove this cache directory and retry")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".decode-", dir=output_dir))
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot decode video: {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                fps = None
            count, original_size = 0, None
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                size = (bgr.shape[1], bgr.shape[0])
                if original_size is None:
                    original_size = size
                elif size != original_size:
                    raise ValueError("Video resolution changes between frames")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = np.asarray(Image.fromarray(rgb).resize(
                    (stage1.SOURCE_WIDTH, stage1.SOURCE_HEIGHT), Image.Resampling.LANCZOS))
                Image.fromarray(rgb).save(temporary / f"{count:08d}.png")
                count += 1
            if count < 2:
                raise ValueError(f"Expected at least 2 decoded video frames, got {count}: {path}")
            metadata = {"signature": signature, "num_frames": count,
                        "fps": fps, "original_size": list(original_size)}
            (temporary / "video.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            try:
                temporary.rename(cache)
            except OSError:
                # Another process may have finished decoding the same video.
                if not manifest.is_file() or json.loads(manifest.read_text()) != metadata:
                    raise
        finally:
            capture.release()
            if temporary.exists():
                shutil.rmtree(temporary)
        paths = [cache / f"{i:08d}.png" for i in range(metadata["num_frames"])]
    fps = fps_override if fps_override is not None else metadata["fps"]
    if fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video has no valid frame rate; specify --fps explicitly")
    return VideoEpisode(path, "video", paths, None, fps, tuple(metadata["original_size"]), output_dir,
                        dataset="in_the_wild_video", frame_rate=fps)


def run_and_save(args, episode, model, start, query, query_source, device, dtype,
                 *, keep_geometry=False, progress=None):
    paths = episode.clip(start, args.max_frames)
    query = stage1.validate_query(query)
    prediction = stage1.infer_episode(
        model, episode, paths, query, query_source, device=device, dtype=dtype,
        window_size=args.window_size, keep_geometry=keep_geometry,
        max_points=args.max_points, progress=progress)
    result = stage1.build_result(episode, paths, prediction, query, query_source,
                                args.model, args.window_size)
    result.update(dataset="in_the_wild_video", video=str(episode.path), frame_rate_hz=episode.fps,
                  original_image_size=list(episode.original_size), inference_image_size=[320, 180],
                  preprocessing="resize full frame to 320x180, then reflect-pad 1 pixel on every side",
                  intrinsics_source="predicted camera decoder; no input calibration",
                  visualization_coordinate_frame="model-predicted world (DROID robot-base convention)",
                  geometry_note="Metric scale and world frame are model predictions, not calibrated to this video.")
    for frame in result["frames"]:
        frame["time_seconds"] = frame["frame_index"] / episode.fps
    result["initial_query"]["pixel_space"] = "resized 320x180 RGB, before padding"
    result["initial_query"]["original_video_pixels"] = (
        query * (np.array(episode.original_size, dtype=np.float32) / [320, 180])).tolist()
    output = args.output or episode.output_dir / "tcp_video.json"
    saved = stage1.write_json_atomic(result, output)
    return result, prediction, paths, saved


def start_viewer(args, episode, prediction, paths):
    from droid_script.stage1_viewer import start_viewer as start
    return start(args, episode, prediction, paths, frame_rate=episode.fps,
                 scene_frame="model-predicted world", load_ground_truth=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Video file or calibrated DROID/RoboLab episode directory")
    parser.add_argument("--camera", help="Camera name for an episode directory, e.g. third_person")
    parser.add_argument("--model", type=Path, default=stage1.DEFAULT_MODEL, help="Single-arm DROID Stage 1 weights or directory")
    parser.add_argument("--output", type=Path, help="Destination JSON, replaced only after successful inference")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Video files only: decoded frame cache and default result root")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--interactive", action="store_true", help="Click one TCP in Gradio, then display Viser")
    selection.add_argument("--tcp-query-point", type=float, nargs=2, metavar=("X", "Y"), help="Pixel in resized 320x180 RGB")
    parser.add_argument("--start-frame", type=int, help="Episode: first visible GT TCP; video: frame 0")
    parser.add_argument("--max-frames", type=int, default=0, help="0: all remaining frames; otherwise a consecutive clip")
    parser.add_argument("--window-size", type=int, default=9, help="2 to 18 frames; windows share one boundary frame")
    parser.add_argument("--fps", type=float, help="Override video FPS for timestamps/playback; does not resample frames")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--visualize", action="store_true", help="Start Viser after CLI inference; implicit with --interactive")
    parser.add_argument("--show-pred-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-gt-trajectory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-points", type=int, default=50_000, help="Point cap per frame; 0 keeps all")
    parser.add_argument("--confidence-percentile", type=float, default=2.5)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--ui-host", default="127.0.0.1", help="Gradio bind address")
    parser.add_argument("--ui-port", type=int, default=7860)
    args = parser.parse_args(argv)
    if args.input.expanduser().is_dir():
        if args.fps is not None:
            parser.error("--fps only applies to video files; episode playback uses metadata/timestamps.npy")
    else:
        if not args.interactive and args.tcp_query_point is None:
            parser.error("Video files require --interactive or --tcp-query-point X Y")
        if args.camera is not None:
            parser.error("--camera requires an episode directory")
        args.show_gt_trajectory = False
        args.start_frame = 0 if args.start_frame is None else args.start_frame
    if not 2 <= args.window_size <= 18:
        parser.error("--window-size must be between 2 and 18")
    if (args.start_frame is not None and args.start_frame < 0) or args.max_frames < 0 or args.max_frames == 1:
        parser.error("--start-frame must be nonnegative; --max-frames must be 0 or at least 2")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if (args.max_points < 0 or not 0 <= args.confidence_percentile <= 99
            or not math.isfinite(args.point_size) or args.point_size <= 0):
        parser.error("Invalid point-cloud display settings")
    if not all(1 <= port <= 65535 for port in (args.port, args.ui_port)):
        parser.error("Ports must be in [1,65535]")
    if args.interactive and args.port == args.ui_port:
        parser.error("Gradio and Viser ports must differ")
    if args.output is not None and args.output.suffix.lower() != ".json":
        parser.error("--output must have a .json extension")
    if args.tcp_query_point is not None:
        try:
            stage1.validate_query(args.tcp_query_point, "--tcp-query-point")
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.input.expanduser().is_dir():
        return stage1.run(args)
    stage1.weight_file(args.model)
    device = stage1.resolve_device(args.device)
    dtype = stage1.resolve_dtype(args.dtype, device)
    episode = load_video(args.input, args.output_root, args.fps)
    episode.clip(args.start_frame, args.max_frames)
    print(f"Video: {episode.path}; {len(episode.image_paths)} frames, {episode.fps:g} fps; "
          f"{episode.original_size} -> (320, 180)", flush=True)
    print(f"Checkpoint: {args.model}; device: {device}; dtype: {dtype}", flush=True)
    if not math.isclose(episode.fps, 15, rel_tol=0.01):
        print("DROID Stage 1 was trained at 15 fps; this run preserves the video's source frame rate.", flush=True)
    if args.interactive:
        from droid_script.stage1_video_viewer import start_interactive
        start_interactive(args, episode, device, dtype)
        return
    query = stage1.validate_query(args.tcp_query_point)
    model = stage1.load_model(args.model, device)
    try:
        result, prediction, paths, saved = run_and_save(
            args, episode, model, args.start_frame, query, "manual pixel", device, dtype,
            keep_geometry=args.visualize, progress=lambda fraction, message: print(message, flush=True))
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Saved {result['num_frames']} frames to {saved}", flush=True)
    if args.visualize:
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
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        raise SystemExit(str(error)) from error
