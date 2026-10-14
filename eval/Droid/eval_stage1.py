#!/usr/bin/env python3
"""Evaluate DROID Stage1 depth, rays, absolute cameras and TCP per camera stream.

    python eval/Droid/eval_stage1.py
    python eval/Droid/eval_stage1.py --max-episodes 1 --max-frames 18
    torchrun --standalone --nproc_per_node=4 eval/Droid/eval_stage1.py --resume

See eval/Droid/README_CN.md for visibility masks, metric definitions and outputs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from zipfile import BadZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import torch

from droid_script import infer_4rc_stage1 as stage1
from eval.Droid.metrics import (
    DEPTH_COLUMNS, DEPTH_RANGES, RAY_COLUMNS, TCP_TRACKING_SAMPLE_KEYS, TCP_TRAJECTORY_KEYS,
    camera_macro, depth_statistics, frame_errors,
    project_positions, ray_statistics, ray_target, summarize_depth, summarize_errors, summarize_ray,
    summarize_tcp_trajectory, tcp_trajectory_definitions, tcp_trajectory_errors,
    summarize_tcp_tracking, tcp_tracking_definitions, tcp_tracking_samples,
)


SCHEMA_VERSION = 3
MASK_POLICY = "GT-visible AND prediction-visible AND finite GT state (absolute and Sim(3) metric families)"


def build_parser(*, dataset="droid", description=__doc__):
    """Shared CLI; only DROID uses a validation split file."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / f"datasets/{dataset}_eval_sets")
    if dataset == "droid":
        parser.add_argument("--val-set", type=Path, default=REPO_ROOT / "droid_script/splits/eval_set.txt")
    else:
        parser.set_defaults(val_set=None)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "checkpoints/Droid-Stage1/checkpoint-500000")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / f"outputs/{dataset}/stage1_eval/checkpoint-500000")
    parser.add_argument("--window-size", type=int, default=9, help="Sliding windows share one frame; previous prediction wins")
    parser.add_argument("--device", default="cuda", help="auto/cuda/cuda:0/cpu; torchrun selects LOCAL_RANK")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--episodes", nargs="+", help=("Exact episode names, restricted to --val-set" if dataset == "droid"
                                                     else "Exact episode directory names within --data-root"))
    parser.add_argument("--cameras", nargs="+", help="Only these camera serials; default: all cameras in each episode")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 means all; limit selected episodes for a smoke run")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all; limit frames after the first visible TCP")
    parser.add_argument("--rpe-delta-frames", type=int, default=1, help="Original frame-index separation for TCP RPE pairs")
    parser.add_argument("--recompute-tcp-metrics-from", type=Path,
                        help="Recompute TCP Sim(3) and projected 2D tracking metrics from cached predictions, without model inference; use a new output directory")
    parser.add_argument("--resume", action="store_true", help="Reuse matching complete/skipped camera results; retry failures")
    parser.add_argument("--dry-run", action="store_true", help="Discover validation cameras without loading the model or writing results")
    return parser


def parse_args(argv=None, *, parser=None):
    parser = build_parser() if parser is None else parser
    args = parser.parse_args(argv)
    if not 2 <= args.window_size <= 18:
        parser.error("--window-size must be between 2 and 18")
    if args.max_episodes < 0 or args.max_frames < 0 or args.max_frames == 1:
        parser.error("--max-episodes must be >= 0; --max-frames must be 0 or >= 2")
    if args.rpe_delta_frames < 1:
        parser.error("--rpe-delta-frames must be >= 1")
    for key in ("data_root", "val_set", "model", "output_dir"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.recompute_tcp_metrics_from is not None:
        args.recompute_tcp_metrics_from = args.recompute_tcp_metrics_from.expanduser().resolve()
        if args.resume or args.dry_run or args.episodes or args.cameras or args.max_episodes or args.max_frames:
            parser.error("Offline TCP recomputation uses all source results; do not combine it with resume, dry-run or selection options")
        if args.output_dir == args.recompute_tcp_metrics_from:
            parser.error("Offline TCP recomputation requires a different --output-dir")
    return args


def discover_jobs(args) -> list[dict]:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"DROID root does not exist: {args.data_root}")
    names = [line.split("#", 1)[0].strip() for line in args.val_set.read_text().splitlines()]
    names = [name for name in names if name]
    if len(names) != len(set(names)) or any(Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError("Validation split must contain unique episode directory names")
    if args.episodes:
        missing = set(args.episodes) - set(names)
        if missing:
            raise ValueError(f"Requested episodes outside validation split: {sorted(missing)}")
        names = [name for name in names if name in set(args.episodes)]
    if args.max_episodes:
        names = names[:args.max_episodes]
    jobs = []
    for name in names:
        path = args.data_root / name
        try:
            cameras = {p.name for p in (path / "images").iterdir() if p.is_dir() and not p.name.startswith(".")}
            depth_meta = path / "depths/metadata.json"
            if depth_meta.is_file():
                cameras.update(json.loads(depth_meta.read_text())["cameras"])
            if not cameras:
                raise ValueError("No camera streams found")
            for camera in sorted(cameras):
                if Path(camera).name != camera or camera in (".", ".."):
                    raise ValueError(f"Invalid camera name: {camera!r}")
                if args.cameras and camera not in args.cameras:
                    continue
                jobs.append({"episode": name, "camera": camera, "path": str(path)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Missing/unreadable validation episodes remain visible in failure coverage.
            jobs.append({"episode": name, "camera": "__discovery__", "path": str(path), "discovery_error": str(error)})
    if not jobs:
        raise ValueError("No validation cameras selected")
    return jobs


def file_identity(path: Path) -> dict:
    try:
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError as error:
        return {"path": str(path), "error": str(error)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_config(args, dtype, jobs=None) -> dict:
    sources = (Path(__file__).resolve(), Path(__file__).resolve().with_name("metrics.py"),
               REPO_ROOT / "droid_script/infer_4rc_stage1.py", REPO_ROOT / "tcp_sliding_window_inference.py",
               REPO_ROOT / "arc/models/arc/arc.py", REPO_ROOT / "arc/models/arc/utils/transform.py")
    return {"schema_version": SCHEMA_VERSION, "data_root": str(args.data_root),
            "val_set": str(args.val_set) if args.val_set is not None else None,
            "val_set_sha256": hashlib.sha256(args.val_set.read_bytes()).hexdigest() if args.val_set is not None else None,
            "model": file_identity(stage1.weight_file(args.model)), "window_size": args.window_size,
            "max_frames": args.max_frames, "max_episodes": args.max_episodes,
            "episodes": sorted(set(args.episodes or [])), "cameras": sorted(set(args.cameras or [])),
            "dtype": str(dtype), "torch_version": torch.__version__, "seed": 42,
            "mask_policy": MASK_POLICY, "boundary_merge": "previous", "depth_ranges_m": DEPTH_RANGES,
            "tcp_trajectory_metrics": tcp_trajectory_definitions(args.rpe_delta_frames),
            "tcp_tracking_metrics": tcp_tracking_definitions(),
            "implementation_sha256": digest({str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in sources})}


def camera_fingerprint(job: dict, config: dict) -> str:
    path, camera = Path(job["path"]), job["camera"]
    paths = [path / "metadata.json", path / "depths/metadata.json", path / "intrinsic" / f"{camera}.npy",
             path / "extrinsic" / f"{camera}.npy", path / "TCP" / camera / "state.npy",
             path / "TCP" / camera / "metadata.json"]
    for kind in ("images", "depths"):
        directory = path / kind / camera
        paths.append(directory)
        if directory.is_dir():
            paths.extend(sorted(p for p in directory.iterdir() if p.is_file()))
    return digest({"config": config, "job": job, "files": [file_identity(p) for p in paths]})


def save_arrays(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def read_labels(episode) -> tuple[np.ndarray, np.ndarray]:
    """DROID only; no joints/URDF/index required for evaluation."""
    if episode.dataset != "droid":
        raise ValueError("This evaluator expects DROID episodes")
    path = episode.path / "TCP" / episode.camera
    metadata = json.loads((path / "metadata.json").read_text())
    expected = {"position_unit": "meter", "rotation_unit": "radian", "camera_id": episode.camera,
                "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
                "coordinate_frame": "OpenCV camera (+x right, +y down, +z forward)",
                "extrinsics": "world_to_camera, applied directly to world TCP pose"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unsupported TCP {key}: {metadata.get(key)!r}")
    return read_label_arrays(episode)


def read_label_arrays(episode) -> tuple[np.ndarray, np.ndarray]:
    """Common camera-coordinate TCP, rigid extrinsic and optical-axis depth checks."""
    path = episode.path / "TCP" / episode.camera
    states = np.load(path / "state.npy", allow_pickle=False, mmap_mode="r")
    if states.shape != (len(episode.image_paths), 7):
        raise ValueError(f"Expected TCP [{len(episode.image_paths)},7], got {states.shape}")
    extrinsic = np.load(episode.path / "extrinsic" / f"{episode.camera}.npy", allow_pickle=False).astype(np.float64)
    if (extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all()
            or not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3), atol=2e-4)
            or not np.isclose(np.linalg.det(extrinsic[:3, :3]), 1, atol=2e-4)):
        raise ValueError("Expected a rigid, absolute robot-base-to-camera extrinsic [4,4]")
    depth_metadata = json.loads((episode.path / "depths/metadata.json").read_text())
    if depth_metadata.get("units") != "millimeters":
        raise ValueError("Expected millimeter depth PNGs")
    camera_meta = depth_metadata["cameras"][episode.camera]
    if (camera_meta["height"], camera_meta["width"], camera_meta["frames"]) != (180, 320, len(states)):
        raise ValueError("Depth/RGB camera frame count or resolution mismatch")
    if depth_metadata.get("depth_type", "distance_to_image_plane") != "distance_to_image_plane":
        raise ValueError("Expected optical-axis depth, not distance along a ray")
    return states, extrinsic


def infer_window(model, paths, query, device, dtype) -> dict:
    """Use Stage1's preprocessing/forward flags, retaining geometry for streaming metrics."""
    from arc.models.arc.utils.transform import pose_encoding_to_extri_intri

    views, _ = stage1.load_views(paths)
    views = [{**view, "img": view["img"].to(device)} for view in views]
    query_tensor = torch.from_numpy(stage1.validate_query(query) + np.float32(1))[None].to(device)
    autocast = contextlib.nullcontext() if dtype == torch.float32 else torch.autocast(device_type=device.type, dtype=dtype)
    with torch.inference_mode(), autocast:
        prediction = model(views, force_no_output_conversion=True, inference_track=False, decode_camera=True,
                           decode_motion=False, decode_tcp=True, tcp_query_points=query_tensor,
                           return_aux_pyramid=False, ref_view_strategy="first")
        c2w, k = pose_encoding_to_extri_intri(prediction["pose_enc"].float(), (182, 322))
    def array(tensor):
        return tensor[0].detach().float().cpu().numpy()
    result = {key: array(prediction[key]) for key in (
        "tcp_position", "tcp_rotation", "tcp_confidence", "depth", "ray", "pose_enc")}
    result["tcp_gripper"] = array(prediction["tcp_gripper_logit"].float().sigmoid())
    result["camera_to_world"] = np.tile(np.eye(4, dtype=np.float32), (len(paths), 1, 1))
    result["camera_to_world"][:, :3] = array(c2w)
    result["intrinsics"] = array(k)
    result["intrinsics"][:, :2, 2] -= 1  # Return K in unpadded RGB pixels.
    expected = {"tcp_position": (len(paths), 1, 3), "tcp_rotation": (len(paths), 1, 3, 3),
                "tcp_gripper": (len(paths), 1), "tcp_confidence": (len(paths), 1),
                "depth": (len(paths), 182, 322), "pose_enc": (len(paths), 9)}
    for key, shape in expected.items():
        if result[key].shape != shape:
            raise ValueError(f"Expected {key} {shape}, got {result[key].shape}")
    if result["ray"].ndim != 4 or result["ray"].shape[0] != len(paths) or result["ray"].shape[-1] != 6:
        raise ValueError(f"Expected ray [T,H,W,6], got {result['ray'].shape}")
    for key in ("tcp_position", "tcp_rotation", "tcp_gripper", "tcp_confidence", "camera_to_world", "intrinsics", "pose_enc"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite {key} prediction")
    return result


def compute_tcp_trajectory(arrays, delta_frames: int) -> tuple[dict, dict, dict]:
    errors, alignment, pairs = tcp_trajectory_errors(
        arrays["tcp_position"], arrays["tcp_rotation"], arrays["gt_tcp_state"],
        arrays["frame_indices"], arrays["metric_mask"], delta_frames)
    extra = {f"trajectory_error_{key}": value for key, value in errors.items()}
    extra["tcp_rpe_sim3_frame_pairs"] = pairs
    extra["tcp_ate_sim3_frame_indices"] = (np.asarray(arrays["frame_indices"])[arrays["metric_mask"]]
                                           if alignment["status"] == "success" else np.empty(0, dtype=np.int64))
    return extra, summarize_tcp_trajectory(errors, [alignment]), alignment


def evaluate_camera(model, episode, states, extrinsic, args, device, dtype, progress=None) -> tuple[dict, dict]:
    gt_uv, gt_visible = project_positions(states[:, :3], episode.intrinsics)
    candidates = np.flatnonzero(gt_visible[:-1])
    if not len(candidates):
        return {"status": "skipped", "reason": "no_visible_gt_tcp_with_two_remaining_frames",
                "episode_frame_count": len(states), "gt_visible_total_count": int(gt_visible.sum()),
                "inferred_frame_count": 0, "scored_frame_count": 0}, {}
    first = int(candidates[0])
    paths = episode.clip(first, args.max_frames)
    frame_indices = np.asarray([int(p.stem) for p in paths], dtype=np.int64)
    target = np.asarray(states[frame_indices], dtype=np.float64)
    visible = gt_visible[frame_indices]
    gt_state_valid = np.isfinite(target).all(-1) & (target[:, 6] >= 0) & (target[:, 6] <= 1)
    n = len(paths)
    arrays = {"frame_indices": frame_indices, "gt_tcp_state": target, "gt_visible": visible,
              "gt_state_valid": gt_state_valid, "pred_visible": np.zeros(n, dtype=bool),
              "metric_mask": np.zeros(n, dtype=bool), "source_window": np.full(n, -1, dtype=np.int64),
              "tcp_position": np.zeros((n, 3), dtype=np.float32), "tcp_rotation": np.zeros((n, 3, 3), dtype=np.float32),
              "tcp_gripper": np.zeros(n, dtype=np.float32), "tcp_confidence": np.zeros(n, dtype=np.float32),
              "camera_to_world": np.zeros((n, 4, 4), dtype=np.float32), "intrinsics": np.zeros((n, 3, 3), dtype=np.float32),
              "pose_enc": np.zeros((n, 9), dtype=np.float32), "pred_tcp_uv": np.full((n, 2), np.nan),
              "gt_tcp_uv": gt_uv[frame_indices], "ray_stats": np.zeros((n, len(RAY_COLUMNS)))}
    for name in DEPTH_RANGES:
        arrays[f"depth_stats_{name}"] = np.zeros((n, len(DEPTH_COLUMNS)))
    query = stage1.validate_query(gt_uv[first]).copy()
    initial_query = query.copy()
    query_source = {"kind": "initial_gt", "frame_index": first}
    query_is_fallback = False
    records, ray_targets = [], {}
    windows = stage1.build_sliding_windows(n, args.window_size)
    for wi, (start, end) in enumerate(windows):
        if progress:
            progress(f"window {wi + 1}/{len(windows)} frames {frame_indices[start]}-{frame_indices[end - 1]}")
        began = time.monotonic()
        prediction = infer_window(model, paths[start:end], query, device, dtype)
        pred_uv, pred_visible = project_positions(prediction["tcp_position"][:, 0], episode.intrinsics)
        window_mask = visible[start:end] & gt_state_valid[start:end] & pred_visible
        skip = int(wi > 0)  # The shared frame's prediction and score mask both come from the previous window.
        dest = slice(start + skip, end)
        arrays["source_window"][dest] = wi
        arrays["pred_visible"][dest] = pred_visible[skip:]
        arrays["pred_tcp_uv"][dest] = pred_uv[skip:]
        arrays["metric_mask"][dest] = window_mask[skip:]
        for key in ("tcp_position", "tcp_rotation", "tcp_gripper", "tcp_confidence"):
            arrays[key][dest] = prediction[key][skip:, 0]
        for key in ("camera_to_world", "intrinsics", "pose_enc"):
            arrays[key][dest] = prediction[key][skip:]
        ray_shape = prediction["ray"].shape[1:3]
        if ray_shape not in ray_targets:
            ray_targets[ray_shape] = ray_target(episode.intrinsics, extrinsic, *ray_shape)
        for local in range(skip, end - start):
            slot = start + local
            if not window_mask[local]:
                continue  # No depth/camera/TCP/ray metric from this frame.
            depth_path = episode.path / "depths" / episode.camera / paths[slot].with_suffix(".png").name
            with Image.open(depth_path) as image:
                raw_depth = np.asarray(image)
                if raw_depth.shape != (180, 320) or raw_depth.dtype != np.uint16:
                    raise ValueError(f"Expected 320x180 uint16 depth: {depth_path}")
                depth = raw_depth.astype(np.float64) / 1000
            cropped = prediction["depth"][local, 1:-1, 1:-1]
            for name, limit in DEPTH_RANGES.items():
                arrays[f"depth_stats_{name}"][slot] = depth_statistics(cropped, depth, limit)
            ray_gt, ray_mask = ray_targets[ray_shape]
            arrays["ray_stats"][slot] = ray_statistics(prediction["ray"][local], ray_gt, ray_mask)
        records.append({"window_index": wi, "frame_indices": frame_indices[start:end].tolist(),
                        "tcp_query_points_px": query.tolist(), "query_source": query_source,
                        "query_is_fallback": query_is_fallback, "boundary_pred_visible": bool(pred_visible[-1]),
                        "retained_scored_frame_count": int(window_mask[skip:].sum()),
                        "inference_and_metrics_seconds": time.monotonic() - began})
        if end < n:
            valid_slots = np.flatnonzero(pred_visible)
            if len(valid_slots):
                last = int(valid_slots[-1])
                query = stage1.validate_query(pred_uv[last]).copy()
                query_source = {"kind": "prediction", "window_index": wi, "frame_index": int(frame_indices[start + last])}
            # If every prediction is off image, retain the last valid query from an earlier window (or initial GT).
            query_is_fallback = not bool(pred_visible[-1])
        del prediction
    mask = arrays["metric_mask"]
    errors = frame_errors(arrays["tcp_position"], arrays["tcp_rotation"], arrays["tcp_gripper"],
                          arrays["tcp_confidence"], arrays["camera_to_world"], arrays["intrinsics"],
                          arrays["pose_enc"], target, episode.intrinsics, extrinsic)
    arrays.update({f"error_{key}": value for key, value in errors.items()})
    metrics = summarize_errors(errors, mask)
    trajectory_arrays, trajectory_metrics, alignment = compute_tcp_trajectory(arrays, args.rpe_delta_frames)
    arrays.update(trajectory_arrays)
    metrics.update(trajectory_metrics)
    tracking_arrays = tcp_tracking_samples(arrays)
    arrays.update(tracking_arrays)
    result = {"status": "complete", "episode_frame_count": len(states), "start_frame": first,
              "end_frame": int(frame_indices[-1]), "skipped_prefix_frame_count": first,
              "unprocessed_suffix_frame_count": len(states) - int(frame_indices[-1]) - 1,
              "inferred_frame_count": n, "scored_frame_count": int(mask.sum()),
              "scored_fraction_of_inferred": float(mask.mean()), "gt_visible_total_count": int(gt_visible.sum()),
              "gt_out_of_view_frame_count": int((~visible).sum()),
              "prediction_out_of_view_frame_count": int((~arrays["pred_visible"]).sum()),
              "invalid_gt_state_frame_count": int((~gt_state_valid).sum()),
              "excluded_frame_count": int((~mask).sum()),
              "fallback_window_count": sum(record["query_is_fallback"] for record in records),
              "initial_query": {"frame_index": first, "source": "projected GT TCP", "tcp_query_points_px": initial_query.tolist()},
              "windows": records, "frame_metrics": metrics, "tcp_trajectory_alignment": alignment,
              "tracking_metrics": summarize_tcp_tracking(tracking_arrays),
              "depth_metrics": {name: summarize_depth(arrays[f"depth_stats_{name}"]) for name in DEPTH_RANGES},
              "ray_metrics": summarize_ray(arrays["ray_stats"])}
    return result, arrays


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(args, jobs, config, source_output_dir: Path | None = None) -> dict:
    records, pooled_errors, pooled_masks = [], {}, []
    trajectory_errors = {key: [] for key in TCP_TRAJECTORY_KEYS}
    trajectory_alignments = []
    tracking_samples = {key: [] for key in TCP_TRACKING_SAMPLE_KEYS}
    depth_stats = {name: [] for name in DEPTH_RANGES}
    ray_stats = []
    for index, job in enumerate(jobs, 1):
        directory = (source_output_dir or args.output_dir) / "cameras" / job["episode"] / job["camera"]
        record = json.loads((directory / "metrics.json").read_text())
        records.append(record)
        if record["status"] == "complete":
            frames_path = directory / "frames.npz"
            if source_output_dir is not None and not frames_path.is_file() and record.get("source_frames_npz"):
                frames_path = Path(record["source_frames_npz"])
                if not frames_path.is_absolute():
                    frames_path = directory / frames_path
            with np.load(frames_path, allow_pickle=False) as arrays:
                mask = arrays["metric_mask"].copy()
                pooled_masks.append(mask)
                if source_output_dir is not None:
                    extra, metrics, alignment = compute_tcp_trajectory(arrays, args.rpe_delta_frames)
                    record["frame_metrics"].update(metrics)
                    record["tcp_trajectory_alignment"] = alignment
                    record["source_frames_npz"] = str(frames_path.resolve())
                    save_arrays(args.output_dir / "cameras" / job["episode"] / job["camera"] / "trajectory_errors.npz", extra)
                    tracking = tcp_tracking_samples(arrays)
                    record["tracking_metrics"] = summarize_tcp_tracking(tracking)
                    save_arrays(args.output_dir / "cameras" / job["episode"] / job["camera"] / "tracking_errors.npz", tracking)
                else:
                    extra = arrays
                    tracking = arrays
                trajectory_alignments.append(record["tcp_trajectory_alignment"])
                for key in TCP_TRAJECTORY_KEYS:
                    trajectory_errors[key].append(extra[f"trajectory_error_{key}"].copy())
                for key in TCP_TRACKING_SAMPLE_KEYS:
                    tracking_samples[key].append(tracking[key].copy())
                for key in arrays.files:
                    if key.startswith("error_"):
                        pooled_errors.setdefault(key[6:], []).append(arrays[key].copy())
                for name in DEPTH_RANGES:
                    depth_stats[name].append(arrays[f"depth_stats_{name}"].copy())
                ray_stats.append(arrays["ray_stats"].copy())
        if source_output_dir is not None:
            stage1.write_json_atomic(record, args.output_dir / "cameras" / job["episode"] / job["camera"] / "metrics.json")
            if index % 200 == 0 or index == len(jobs):
                print(f"TCP metrics recomputed: {index}/{len(jobs)} camera streams", flush=True)
    complete = [r for r in records if r["status"] == "complete"]
    def concatenate(parts, columns):
        return np.concatenate(parts) if parts else np.empty((0, columns))
    global_rows = []
    for aggregation in ("camera_macro", "frame_micro"):
        metrics = (camera_macro([r["frame_metrics"] for r in complete]) if aggregation == "camera_macro"
                   else summarize_errors({k: np.concatenate(v) for k, v in pooled_errors.items()},
                                         np.concatenate(pooled_masks) if pooled_masks else np.empty(0, dtype=bool)))
        if aggregation == "frame_micro":
            metrics.update(summarize_tcp_trajectory(
                {key: np.concatenate(parts) if parts else np.empty(0) for key, parts in trajectory_errors.items()},
                trajectory_alignments))
        global_rows.append({"aggregation": aggregation, **metrics})
    depth_rows, global_depth = [], []
    for name in DEPTH_RANGES:
        for record in records:
            depth_rows.append({"episode": record["episode"], "camera": record["camera"], "status": record["status"],
                               "depth_range": name, **record.get("depth_metrics", {}).get(name, {})})
        for aggregation in ("camera_macro", "pixel_micro"):
            metrics = (camera_macro([r["depth_metrics"][name] for r in complete]) if aggregation == "camera_macro"
                       else summarize_depth(concatenate(depth_stats[name], len(DEPTH_COLUMNS))))
            global_depth.append({"depth_range": name, "aggregation": aggregation, **metrics})
    global_ray = [{"aggregation": "camera_macro", **camera_macro([r["ray_metrics"] for r in complete])},
                  {"aggregation": "ray_micro", **summarize_ray(concatenate(ray_stats, len(RAY_COLUMNS)))}]
    global_tracking = [
        {"aggregation": "camera_macro", **camera_macro([r["tracking_metrics"] for r in complete])},
        {"aggregation": "frame_micro", **summarize_tcp_tracking(
            {key: np.concatenate(parts) if parts else np.empty(0) for key, parts in tracking_samples.items()})},
    ]
    base_keys = ("episode", "camera", "status", "reason", "episode_frame_count", "start_frame", "end_frame",
                 "skipped_prefix_frame_count", "unprocessed_suffix_frame_count", "inferred_frame_count",
                 "scored_frame_count", "scored_fraction_of_inferred", "excluded_frame_count", "gt_visible_total_count",
                 "gt_out_of_view_frame_count", "prediction_out_of_view_frame_count", "invalid_gt_state_frame_count",
                 "fallback_window_count")
    camera_rows = [{**{key: r[key] for key in base_keys if key in r}, **r.get("frame_metrics", {}),
                    **{f"ray_{key}": value for key, value in r.get("ray_metrics", {}).items()}} for r in records]
    tracking_rows = [{**{key: r[key] for key in base_keys if key in r}, **r.get("tracking_metrics", {})}
                     for r in records]
    for filename, rows in (("camera_metrics.csv", camera_rows), ("depth_camera_metrics.csv", depth_rows),
                           ("global_frame_metrics.csv", global_rows), ("global_depth_metrics.csv", global_depth),
                           ("global_ray_metrics.csv", global_ray), ("tracking_camera_metrics.csv", tracking_rows),
                           ("global_tracking_metrics.csv", global_tracking)):
        write_csv(args.output_dir / filename, rows)
    coverage = {"selected_episode_count": len({job["episode"] for job in jobs}),
                "camera_stream_count": sum(job["camera"] != "__discovery__" for job in jobs),
                "episode_discovery_failure_count": sum(job["camera"] == "__discovery__" for job in jobs),
                "complete_camera_count": len(complete), "failed_camera_count": sum(r["status"] == "failed" for r in records),
                "skipped_camera_count": sum(r["status"] == "skipped" for r in records),
                "scored_camera_count": sum(r.get("scored_frame_count", 0) > 0 for r in records)}
    count_keys = ("episode_frame_count", "skipped_prefix_frame_count", "inferred_frame_count", "scored_frame_count",
                  "excluded_frame_count", "gt_out_of_view_frame_count", "prediction_out_of_view_frame_count", "fallback_window_count")
    coverage.update({key: sum(r.get(key, 0) for r in records) for key in count_keys})
    summary = {"config": config, "coverage": coverage, "frame_metrics": global_rows,
               "depth_metrics": global_depth, "ray_metrics": global_ray,
               "tracking_metrics": global_tracking,
               "failures_and_skips": [{k: r[k] for k in ("episode", "camera", "status", "reason") if k in r}
                                      for r in records if r["status"] != "complete"],
               "metric_definitions": {"mask": MASK_POLICY, "alignment": "absolute metrics: none; TCP trajectory metrics: per-camera global Sim(3)",
                   "tcp_trajectory": tcp_trajectory_definitions(args.rpe_delta_frames),
                   "tcp_tracking": tcp_tracking_definitions(),
                   "camera_translation": "L2 error of world-to-camera translation; camera_center uses inverse pose translation",
                   "tcp_base": "predicted TCP transformed with predicted camera; GT TCP transformed with GT camera",
                   "rotation": "SO(3) geodesic degrees; RPY target uses Rz(yaw) @ Ry(pitch) @ Rx(roll)",
                   "gripper": "continuous [0,1], accuracy threshold=0.5", "initial_frame_included": True,
                   "visibility": "geometric projection with GT K into 320x180 RGB; not an occlusion test",
                   "depth_log_epsilon_m": 1e-5, "silog": "100 * sqrt(mean(log_error^2) - mean(log_error)^2)",
                   "depth_delta": "max(pred/GT, GT/pred) < 1.25**k; nonpositive predictions fail",
                   "depth_stat_columns": list(DEPTH_COLUMNS), "ray_stat_columns": list(RAY_COLUMNS),
                   "macro": "equal camera-stream weight; counts sum; null metrics excluded",
                   "micro": "pool frames/pixels/rays before computing metrics, RMSE and percentiles",
                   "failure_policy": "only complete camera streams contribute metrics; failed and skipped streams remain in coverage"}}
    stage1.write_json_atomic(summary, args.output_dir / "summary.json")
    return summary


def recompute_tcp_metrics(args) -> dict:
    """Export aligned TCP and projected tracking metrics from saved predictions."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Offline TCP recomputation runs in one process; do not use torchrun")
    source = args.recompute_tcp_metrics_from
    original = json.loads((source / "summary.json").read_text())
    with (source / "camera_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    jobs, seen = [], set()
    for row in rows:
        name, camera = row["episode"], row["camera"]
        if any(not part or Path(part).name != part or part in (".", "..") for part in (name, camera)):
            raise ValueError("Invalid source episode/camera path")
        if (name, camera) in seen:
            raise ValueError("Duplicate source episode/camera")
        seen.add((name, camera))
        jobs.append({"episode": name, "camera": camera})
    if not jobs:
        raise ValueError("Source evaluation has no camera results")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Offline TCP recomputation requires an empty --output-dir")
    config = {"schema_version": SCHEMA_VERSION, "mode": "recompute_tcp_metrics",
              "source_output_dir": str(source), "source_config": original["config"],
              "source_summary_sha256": hashlib.sha256((source / "summary.json").read_bytes()).hexdigest(),
              "tcp_trajectory_metrics": tcp_trajectory_definitions(args.rpe_delta_frames),
              "tcp_tracking_metrics": tcp_tracking_definitions(),
              "metrics_implementation_sha256": hashlib.sha256(Path(__file__).with_name("metrics.py").read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1.write_json_atomic(config, args.output_dir / "run.json")
    print(f"Recomputing TCP trajectory and 2D tracking metrics from {source}; {len(jobs)} camera results; RPE delta={args.rpe_delta_frames}", flush=True)
    summary = summarize_run(args, jobs, config, source_output_dir=source)
    print(f"Saved summary: {args.output_dir / 'summary.json'}", flush=True)
    return summary


def run(args, *, discover_jobs_fn=None, read_labels_fn=None,
        make_config_fn=None, camera_fingerprint_fn=None) -> dict | None:
    """Run the shared evaluator with optional dataset adapters; DROID is the default."""
    discover_jobs_fn = discover_jobs if discover_jobs_fn is None else discover_jobs_fn
    read_labels_fn = read_labels if read_labels_fn is None else read_labels_fn
    make_config_fn = make_config if make_config_fn is None else make_config_fn
    camera_fingerprint_fn = camera_fingerprint if camera_fingerprint_fn is None else camera_fingerprint_fn
    if args.recompute_tcp_metrics_from is not None:
        return recompute_tcp_metrics(args)
    jobs = discover_jobs_fn(args)
    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if args.dry_run:
        if rank == 0:
            print(f"Validation episodes: {len({j['episode'] for j in jobs})}; camera streams: "
                  f"{sum(j['camera'] != '__discovery__' for j in jobs)}; discovery failures: "
                  f"{sum('discovery_error' in j for j in jobs)}")
            for job in jobs[:10]:
                print(json.dumps(job, ensure_ascii=False))
        return None
    if world_size > 1:
        if args.device not in ("auto", "cuda"):
            raise ValueError("torchrun requires --device auto or cuda")
        device = stage1.resolve_device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    else:
        device = stage1.resolve_device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    dtype = stage1.resolve_dtype(args.dtype, device)
    torch.manual_seed(42)
    config = make_config_fn(args, dtype, jobs)
    # Initialize before loading weights so an early error propagates to other ranks.
    if world_size > 1:
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=24))
    model = None
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = args.output_dir / "run.json"
        if rank == 0:
            if manifest.is_file() and json.loads(manifest.read_text()) != config:
                raise ValueError("Evaluation configuration changed; use a new --output-dir")
            stage1.write_json_atomic(config, manifest)
        if world_size > 1:
            torch.distributed.barrier()
        selected = jobs[rank::world_size]
        print(f"[rank {rank}] {len(selected)}/{len(jobs)} camera jobs on {device}, {dtype}; window={args.window_size}", flush=True)
        for index, job in enumerate(selected, 1):
            directory = args.output_dir / "cameras" / job["episode"] / job["camera"]
            result_path = directory / "metrics.json"
            tag = f"[rank {rank} {index}/{len(selected)}] {job['episode']}/{job['camera']}"
            began = time.monotonic()
            fingerprint = None
            try:
                if "discovery_error" in job:
                    raise ValueError(job["discovery_error"])
                fingerprint = camera_fingerprint_fn(job, config)
                if args.resume and result_path.is_file():
                    try:
                        saved = json.loads(result_path.read_text())
                        if saved.get("fingerprint") == fingerprint and saved.get("status") in ("complete", "skipped"):
                            if saved["status"] == "skipped":
                                print(f"{tag}: cached skip", flush=True)
                                continue
                            with np.load(directory / "frames.npz", allow_pickle=False) as cached:
                                # Touch every array so corrupt/incomplete ZIP members trigger a retry.
                                for key in cached.files:
                                    cached[key]
                            print(f"{tag}: cached", flush=True)
                            continue
                    except (OSError, ValueError, EOFError, BadZipFile):
                        pass
                episode = stage1.load_episode(Path(job["path"]), job["camera"])
                states, extrinsic = read_labels_fn(episode)
                _, visible = project_positions(states[:, :3], episode.intrinsics)
                if np.any(visible[:-1]) and model is None:
                    print(f"{tag}: loading {args.model}", flush=True)
                    model = stage1.load_model(args.model, device)
                result, arrays = evaluate_camera(model, episode, states, extrinsic, args, device, dtype,
                                                 progress=lambda message: print(f"{tag}: {message}", flush=True))
                if arrays:
                    save_arrays(directory / "frames.npz", arrays)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                result = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            result.update(episode=job["episode"], camera=job["camera"], fingerprint=fingerprint,
                          elapsed_seconds=time.monotonic() - began)
            stage1.write_json_atomic(result, result_path)
            print(f"{tag}: {result['status']}, scored={result.get('scored_frame_count', 0)}/"
                  f"{result.get('inferred_frame_count', 0)}, {result['elapsed_seconds']:.1f}s"
                  + (f"; {result['reason']}" if "reason" in result else ""), flush=True)
        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            summary = summarize_run(args, jobs, config)
            print(json.dumps(summary["coverage"], ensure_ascii=False), flush=True)
            print(f"Saved summary: {args.output_dir / 'summary.json'}", flush=True)
            return summary
        return None
    finally:
        del model
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    run(parse_args())
