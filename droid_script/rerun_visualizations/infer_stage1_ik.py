#!/usr/bin/env python3
"""Save Stage1 geometry and sequential cuRobo IK for independent Rerun replay."""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from droid_script import infer_4rc_stage1 as inference
from droid_script.rerun_visualizations.cache import (
    file_digest, load_cache, save_prediction, save_robot_states,
)
from droid_script.rerun_visualizations.data import load_ground_truth
from droid_script.rerun_visualizations.ik import read_tcp_offset, solve_trajectory


DEFAULT_URDF = REPO_ROOT / "embodiments/franka-panda-robotiq-2f85/panda_robotiq_2f85.urdf"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Extracted DROID episode")
    source.add_argument("--reuse-prediction", type=Path, help="Recompute IK from cached predictions")
    parser.add_argument("--output", type=Path, help="Cache directory; reuse defaults to the source cache")
    parser.add_argument("--camera", help="Camera serial; default: first sorted camera")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / inference.DEFAULT_MODEL)
    parser.add_argument("--urdf", type=Path, help="Default: bundled Panda/Robotiq, or the cached URDF when reusing")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--tcp-query-point", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--device", default="auto", help="CUDA device for inference and IK")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--ik-num-seeds", type=int, default=32)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.005, help="Metres")
    parser.add_argument("--ik-rotation-tolerance", type=float, default=0.05, help="Radians")
    parser.add_argument("--ik-no-cuda-graph", action="store_true")
    args = parser.parse_args(argv)
    if not 2 <= args.window_size <= 18:
        parser.error("--window-size must be between 2 and 18")
    if args.max_frames < 0 or args.max_frames == 1 or args.max_points < 0:
        parser.error("--max-frames must be 0 or >=2; --max-points must be >=0")
    if args.start_frame is not None and args.start_frame < 0:
        parser.error("--start-frame must be >=0")
    if args.ik_num_seeds < 1:
        parser.error("--ik-num-seeds must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.ik_position_tolerance, args.ik_rotation_tolerance)):
        parser.error("IK tolerances must be finite and positive")
    if args.reuse_prediction is not None and (args.camera is not None or args.start_frame is not None
                                              or args.tcp_query_point is not None or args.max_frames):
        parser.error("Reused predictions fix the camera, query and frame interval")
    return args


def run(args):
    device = inference.resolve_device(args.device)
    if device.type != "cuda":
        raise ValueError("Prediction robot IK requires CUDA; use --device cuda:0")
    if args.reuse_prediction is not None:
        metadata, prediction, indices, _ = load_cache(args.reuse_prediction, require_robot=False)
        episode = inference.load_episode(Path(metadata["episode"]), metadata["camera_id"])
        if len(episode.image_paths) != metadata["episode_num_frames"] or indices[-1] >= len(episode.image_paths):
            raise ValueError("Cached frames disagree with the source episode")
        paths = [episode.image_paths[int(frame)] for frame in indices]
        output = (args.output or args.reuse_prediction).expanduser().resolve()
        urdf = (args.urdf or Path(metadata["urdf"])).expanduser().resolve()
        if args.urdf is None and file_digest(urdf) != metadata["urdf_sha256"]:
            raise ValueError("Cached URDF changed; explicitly pass --urdf to recompute with a new model")
    else:
        episode = inference.load_episode(args.input, args.camera)
        start, query, query_source = inference.resolve_initial_query(
            episode, args.start_frame, args.tcp_query_point)
        paths = episode.clip(start, args.max_frames)
        ground_truth = load_ground_truth(episode, paths)
        indices = ground_truth.frame_indices
        urdf = (args.urdf or DEFAULT_URDF).expanduser().resolve()
        # Validate the work point and URDF before the expensive inference pass.
        read_tcp_offset(episode.path, episode.camera)
        urdf_hash = file_digest(urdf)
        model_path = inference.weight_file(args.model).resolve()
        output = (args.output or REPO_ROOT / "outputs/droid/rerun" / episode.path.name
                  / episode.camera).expanduser().resolve()
        dtype = inference.resolve_dtype(args.dtype, device)
        print(f"Episode: {episode.path.name}; camera: {episode.camera}; "
              f"frames: {indices[0]}–{indices[-1]}; device: {device}; dtype: {dtype}", flush=True)
        model = inference.load_model(args.model, device)
        try:
            prediction = inference.infer_episode(
                model, episode, paths, query, query_source, device=device, dtype=dtype,
                window_size=args.window_size, keep_geometry=True, max_points=args.max_points,
                progress=lambda fraction, message: print(message, flush=True))
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        metadata = dict(episode=str(episode.path), camera_id=episode.camera,
                        episode_num_frames=len(episode.image_paths), num_frames=len(paths),
                        frame_rate_hz=15, model=str(model_path), window_size=args.window_size,
                        initial_query=query.tolist(), query_source=query_source,
                        urdf=str(urdf), urdf_sha256=urdf_hash, max_points=args.max_points,
                        coordinate_frame="robot_base", tcp_prediction_frame="OpenCV camera")
    ground_truth = load_ground_truth(episode, paths)
    offset = read_tcp_offset(episode.path, episode.camera)
    metadata.update(urdf=str(urdf), urdf_sha256=file_digest(urdf))
    metadata = save_prediction(output, prediction, indices, metadata)
    print(f"Predictions saved to {output}. Solving sequential IK…", flush=True)
    tcp_base = inference.tcp_to_base(prediction.tcp, prediction.camera_to_base)
    poses = np.tile(np.eye(4), (len(indices), 1, 1))
    poses[:, :3, 3] = tcp_base["position"][:, 0]
    poses[:, :3, :3] = tcp_base["rotation"][:, 0]

    def progress(slot, count, states):
        if slot == 0 or (slot + 1) % 25 == 0 or slot + 1 == count:
            print(f"IK {slot + 1}/{count} (frame {indices[slot]}): "
                  f"{'OK' if states['success'][slot] else 'HOLD'}; "
                  f"position={states['position_error_m'][slot] * 1000:.2f} mm, "
                  f"rotation={states['rotation_error_rad'][slot]:.4f} rad", flush=True)

    states, ik_metadata = solve_trajectory(
        poses, ground_truth.joints[0], tcp_base["gripper"][:, 0], indices, urdf, offset,
        device=device, num_seeds=args.ik_num_seeds,
        position_tolerance=args.ik_position_tolerance,
        rotation_tolerance=args.ik_rotation_tolerance,
        use_cuda_graph=not args.ik_no_cuda_graph, progress=progress)
    save_robot_states(output, states, metadata, ik_metadata)
    print(f"Saved replay cache: {output}\n"
          f"IK success: {int(states['success'].sum())}/{len(indices)}; "
          f"held frames: {int((~states['success']).sum())}\n"
          "Replay with:\n"
          f"  conda run --no-capture-output -n 4rc python "
          f"droid_script/rerun_visualizations/visualize_stage1.py --prediction '{output}'", flush=True)


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        print("\nStopped. A complete prediction cache can be reused with --reuse-prediction.", flush=True)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
