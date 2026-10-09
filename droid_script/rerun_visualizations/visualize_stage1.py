#!/usr/bin/env python3
"""Replay cached prediction IK or infer one DROID camera in Rerun Web.

Run in the 4rc environment:
    python droid_script/rerun_visualizations/visualize_stage1.py --prediction <cache_directory>
    python droid_script/rerun_visualizations/visualize_stage1.py --input <episode> --camera <serial>
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

import rerun as rr
import torch

from droid_script import infer_4rc_stage1 as inference
from droid_script.rerun_visualizations.data import load_ground_truth
from droid_script.rerun_visualizations.cache import file_digest, load_cache, validate_episode_timing
from droid_script.rerun_visualizations.robot import prepare_robot
from droid_script.rerun_visualizations.viewer import (
    log_replay, make_blueprint, start_web_viewer, wait_for_web_viewer,
)


DEFAULT_URDF = REPO_ROOT / "embodiments/franka-panda-robotiq-2f85/panda_robotiq_2f85.urdf"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="DROID or RoboLab episode directory (legacy inference mode)")
    source.add_argument("--prediction", type=Path, help="Replay cache from infer_stage1_ik.py; no inference or IK")
    parser.add_argument("--episode-root", type=Path, help="Override cached episode location after moving a dataset")
    parser.add_argument("--camera", help="Camera serial; default is the first sorted images/ directory")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / inference.DEFAULT_MODEL)
    parser.add_argument("--urdf", type=Path, help="Default: bundled URDF, or the cached URDF in replay mode")
    parser.add_argument("--start-frame", type=int, help="Original frame index; default is first visible GT TCP")
    parser.add_argument("--tcp-query-point", nargs=2, type=float, metavar=("X", "Y"), help="Manual query in original 320x180 RGB")
    parser.add_argument("--max-frames", type=int, default=0, help="0: all remaining frames; otherwise at least 2")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-points", type=int, default=100_000, help="Per frame, per cloud; 0: all valid points")
    parser.add_argument("--confidence-percentile", type=float, default=2.5, help="Discard prediction points below this confidence percentile")
    parser.add_argument("--point-size", type=float, default=0.003, help="Point diameter in metres")
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--renderer", choices=("webgl", "webgpu"), default="webgl",
                        help="Browser rendering backend; WebGL is the tested default")
    parser.add_argument("--output", type=Path, help="Export .rrd and exit instead of starting the Web viewer")
    args = parser.parse_args(argv)
    if not 2 <= args.window_size <= 18:
        parser.error("--window-size must be between 2 and 18")
    if args.max_frames < 0 or args.max_frames == 1 or args.max_points < 0:
        parser.error("--max-frames must be 0 or >=2; --max-points must be >=0")
    if args.start_frame is not None and args.start_frame < 0:
        parser.error("--start-frame must be >=0")
    if not math.isfinite(args.confidence_percentile) or not 0 <= args.confidence_percentile <= 99:
        parser.error("--confidence-percentile must be finite and within [0,99]")
    if not math.isfinite(args.point_size) or args.point_size <= 0:
        parser.error("--point-size must be finite and positive")
    if not (math.isfinite(args.min_depth_m) and math.isfinite(args.max_depth_m)
            and 0 <= args.min_depth_m < args.max_depth_m):
        parser.error("Require finite 0 <= --min-depth-m < --max-depth-m")
    if not all(1 <= port <= 65535 for port in (args.web_port, args.grpc_port)):
        parser.error("Ports must be between 1 and 65535")
    if args.web_port == args.grpc_port:
        parser.error("--web-port and --grpc-port must differ")
    if args.output is not None and args.output.suffix.lower() != ".rrd":
        parser.error("--output must end in .rrd")
    if args.prediction is not None and (args.camera is not None or args.start_frame is not None
                                       or args.tcp_query_point is not None or args.max_frames):
        parser.error("Cached playback fixes camera, query and frame interval; generate a new cache to change them")
    if args.episode_root is not None and args.prediction is None:
        parser.error("--episode-root requires --prediction")
    return args


def run(args) -> None:
    if args.prediction is not None:
        return run_cached(args)
    args.urdf = args.urdf or DEFAULT_URDF
    episode = inference.load_episode(args.input, args.camera)
    start, query, query_source = inference.resolve_initial_query(episode, args.start_frame, args.tcp_query_point)
    paths = episode.clip(start, args.max_frames)
    ground_truth = load_ground_truth(episode, paths)
    device = inference.resolve_device(args.device)
    dtype = inference.resolve_dtype(args.dtype, device)
    inference.weight_file(args.model)
    print(f"Episode: {episode.path.name}; camera: {episode.camera}; frames: {start}–{start + len(paths) - 1}", flush=True)
    print(f"Initial query: {query.tolist()} ({query_source}); skipped prefix: {start} frames", flush=True)
    print(f"Checkpoint: {args.model}; device: {device}; dtype: {dtype}", flush=True)
    recording = None
    try:
        with tempfile.TemporaryDirectory(prefix="4rc_rerun_meshes_") as temporary_directory:
            tree = prepare_robot(args.urdf, Path(temporary_directory))
            recording = rr.RecordingStream(f"4rc_{episode.dataset}_stage1_{episode.camera}")
            blueprint = make_blueprint(episode.camera, frame_rate=episode.frame_rate)
            if args.output is None:
                # Initialize the SDK's server before CUDA inference. The Web
                # page is immediately accessible while the model is loading.
                start_web_viewer(recording, args)
                recording.send_blueprint(blueprint)
                recording.set_time("frame", sequence=start)
                recording.set_time("episode_time", duration=episode.time_seconds(start))
                # Static text takes precedence over temporal text in Rerun.
                # Use the first replay time so per-frame information replaces it.
                recording.log("info", rr.TextDocument("Loading Stage1 model…", media_type="text/markdown"))
            print("Ground truth and URDF validated. Loading Stage1 model…", flush=True)
            model = inference.load_model(args.model, device)
            try:
                prediction = inference.infer_episode(
                    model, episode, paths, query, query_source, device=device, dtype=dtype,
                    window_size=args.window_size, keep_geometry=True, max_points=args.max_points,
                    progress=lambda fraction, message: print(message, flush=True),
                )
            finally:
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if args.output is not None:
                output = args.output.expanduser().resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                recording.save(output)
                recording.send_blueprint(blueprint)
            log_replay(recording, args, episode, paths, prediction, ground_truth, tree, query, query_source)
        # Rerun now owns the asset blobs and replay data. Release Python arrays
        # and the temporary mesh directory while keeping its servers alive.
        del tree, prediction, ground_truth
        gc.collect()
        if args.output is not None:
            print(f"Saved {len(paths)} frames to {output}", flush=True)
        else:
            wait_for_web_viewer()
    finally:
        if recording is not None:
            recording.disconnect()


def run_cached(args) -> None:
    """Replay numeric caches without loading a checkpoint or importing cuRobo."""
    import json
    import numpy as np

    metadata, prediction, indices, states = load_cache(args.prediction)
    episode = inference.load_episode(args.episode_root or Path(metadata["episode"]), metadata["camera_id"])
    if len(episode.image_paths) != metadata["episode_num_frames"] or indices[-1] >= len(episode.image_paths):
        raise ValueError("Cached frames disagree with the source episode")
    validate_episode_timing(metadata, episode, indices)
    paths = [episode.image_paths[int(frame)] for frame in indices]
    ground_truth = load_ground_truth(episode, paths)
    urdf = (args.urdf or Path(metadata["urdf"])).expanduser().resolve()
    if file_digest(urdf) != metadata["urdf_sha256"]:
        raise ValueError("Replay URDF differs from the IK model; regenerate IK for the new URDF")
    tcp_metadata = json.loads((episode.path / "TCP" / episode.camera / "metadata.json").read_text())
    tcp_offset = np.asarray(tcp_metadata.get("tcp_offset_in_robotiq_base_m"), dtype=np.float64)
    if (tcp_offset.shape != (3,) or not np.isfinite(tcp_offset).all()
            or not np.allclose(tcp_offset, metadata["ik"]["tcp_offset_in_robotiq_base_m"], atol=1e-9)):
        raise ValueError("Source TCP work point differs from the cached IK work point")
    args.model = metadata["model"]
    query = np.asarray(metadata["initial_query"], dtype=np.float32)
    recording = rr.RecordingStream(f"4rc_{episode.dataset}_stage1_{episode.camera}")
    try:
        blueprint = make_blueprint(episode.camera, predicted_robot=True, frame_rate=episode.frame_rate)
        if args.output is None:
            start_web_viewer(recording, args)
        else:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            recording.save(output)
        recording.send_blueprint(blueprint)
        with tempfile.TemporaryDirectory(prefix="4rc_rerun_meshes_") as temporary:
            directory = Path(temporary)
            tree = prepare_robot(urdf, directory)
            predicted_tree = prepare_robot(urdf, directory, prefix="prediction")
            log_replay(recording, args, episode, paths, prediction, ground_truth, tree, query,
                       metadata["query_source"], predicted_tree=predicted_tree, robot_states=states)
        print(f"Replayed {len(indices)} cached frames; IK success: {int(states['success'].sum())}/{len(indices)}",
              flush=True)
        del tree, predicted_tree, prediction, ground_truth, states
        gc.collect()
        if args.output is not None:
            print(f"Saved {len(paths)} frames to {output}", flush=True)
        else:
            wait_for_web_viewer()
    finally:
        recording.disconnect()


def main(argv=None) -> None:
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        print("\nStopped Rerun Web viewer.", flush=True)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
