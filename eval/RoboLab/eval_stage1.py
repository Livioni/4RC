#!/usr/bin/env python3
"""Evaluate a DROID Stage1 checkpoint on all RoboLab task/episode camera streams.

    python eval/RoboLab/eval_stage1.py --dry-run
    python eval/RoboLab/eval_stage1.py --max-episodes 1 --max-frames 18
    torchrun --standalone --nproc_per_node=4 eval/RoboLab/eval_stage1.py --resume

Metrics, inference, masks, aggregation and output schemas are shared with DROID.
See eval/RoboLab/README_CN.md for data conventions and usage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.Droid import eval_stage1 as droid


def parse_args(argv=None):
    return droid.parse_args(argv, parser=droid.build_parser(dataset="robolab", description=__doc__))


def discover_jobs(args) -> list[dict]:
    """Select every task/episode, independent of the robot policy's task outcome."""
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"RoboLab root does not exist: {args.data_root}")
    episodes = {}
    for task in sorted(args.data_root.iterdir()):
        if not task.is_dir() or task.name.startswith("."):
            continue
        for path in sorted(task.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            if path.name in episodes:
                raise ValueError(f"Duplicate episode name {path.name!r}: {episodes[path.name]} and {path}")
            episodes[path.name] = path
    if args.episodes:
        missing = set(args.episodes) - episodes.keys()
        if missing:
            raise ValueError(f"Requested episodes not found in RoboLab root: {sorted(missing)}")
    paths = [path for name, path in episodes.items() if not args.episodes or name in args.episodes]
    if args.max_episodes:
        paths = paths[:args.max_episodes]
    jobs = []
    for path in paths:
        base = {"episode": path.name, "task": path.parent.name, "path": str(path)}
        try:
            cameras = {p.name for p in (path / "images").iterdir()
                       if p.is_dir() and not p.name.startswith(".")}
            depth_metadata = path / "depths/metadata.json"
            if depth_metadata.is_file():
                declared = json.loads(depth_metadata.read_text())["cameras"]
                if not isinstance(declared, dict):
                    raise ValueError("Depth cameras must be a mapping")
                cameras.update(declared)
            if not cameras:
                raise ValueError("No camera streams found")
            if any(not camera or Path(camera).name != camera or camera in (".", "..") for camera in cameras):
                raise ValueError("Invalid camera name in depth metadata")
            for camera in sorted(cameras):
                if not args.cameras or camera in args.cameras:
                    jobs.append({**base, "camera": camera})
        except (OSError, ValueError, KeyError, TypeError) as error:
            jobs.append({**base, "camera": "__discovery__", "discovery_error": str(error)})
    if not jobs:
        raise ValueError("No RoboLab cameras selected")
    return jobs


def read_labels(episode):
    """Validate RoboLab metadata, then use the same numerical labels as DROID."""
    if episode.dataset != "robolab" or episode.metadata.get("format") != "robolab_droid_like_v2":
        raise ValueError("This evaluator expects robolab_droid_like_v2 episodes")
    if episode.metadata.get("extrinsics") != "base_to_camera":
        raise ValueError("RoboLab extrinsics must map robot base to OpenCV camera")
    metadata = json.loads((episode.path / "TCP" / episode.camera / "metadata.json").read_text())
    expected = {
        "columns": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"],
        "position_unit": "meter", "rotation_unit": "radian",
        "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
        "tcp_definition_id": "droid_generate_tcp_v2",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Unsupported TCP {key}: {metadata.get(key)!r}")
    if metadata.get("coordinate_frame") not in (
            "OpenCV (+x right, +y down, +z forward)",
            "OpenCV camera (+x right, +y down, +z forward)"):
        raise ValueError(f"Unsupported TCP coordinate_frame: {metadata.get('coordinate_frame')!r}")
    # TCP state already stores openness. observations/gripper_position.npy stores
    # closedness and must not replace this column or cause another inversion.
    return droid.read_label_arrays(episode)


def make_config(args, dtype, jobs) -> dict:
    config = droid.make_config(args, dtype, jobs)
    del config["val_set"], config["val_set_sha256"]
    config.update(dataset="robolab", episode_selection="all_task_outcomes", jobs=jobs)
    config["implementation_sha256"] = droid.digest({
        "shared": config["implementation_sha256"],
        "robolab": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    return config


def camera_fingerprint(job, config) -> str:
    return droid.digest({
        "shared": droid.camera_fingerprint(job, config),
        "timestamps": droid.file_identity(Path(job["path"]) / "timestamps.npy"),
    })


def run(args):
    return droid.run(args, discover_jobs_fn=discover_jobs, read_labels_fn=read_labels,
                     make_config_fn=make_config, camera_fingerprint_fn=camera_fingerprint)


if __name__ == "__main__":
    run(parse_args())
