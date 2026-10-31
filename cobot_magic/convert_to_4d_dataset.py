#!/usr/bin/env python3
"""Convert real ALOHA HDF5 episodes, including wrist RGB, into dual-arm RoboLab-like data."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from cobot_magic.geometry import (
    ARMS, CAMERA, CAMERA_FRAME, DEFAULT_CALIBRATION, DEFAULT_URDF, FORMAT, ROOT,
    RPY_CONVENTION, SOURCE_CAMERA, STATE_COLUMNS, WRISTS, RobotModel, file_digest,
    poses_to_states, read_calibration, rigid_transform,
)


RGB_CAMERAS = {CAMERA: SOURCE_CAMERA, "left_view": "cam_left_wrist", "right_view": "cam_right_wrist"}


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def discover_sources(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        files, root = [path], path.parent
    else:
        files = sorted(p for p in path.rglob("*")
                       if p.is_file() and p.suffix.lower() in (".hdf5", ".h5"))
        root = path
    if not files or any(p.suffix.lower() not in (".hdf5", ".h5") for p in files):
        raise ValueError(f"No HDF5 episodes found: {path}")
    groups = defaultdict(list)
    for file in files:
        groups[file.parent.relative_to(root).as_posix()].append(file)
    return root, dict(groups)


def source_rgb_cameras(source, count):
    cameras = {}
    for camera, source_camera in RGB_CAMERAS.items():
        key = f"observations/images/{source_camera}"
        if key not in source:
            if camera == CAMERA:
                raise ValueError(f"{source.filename}: missing {key}")
            continue
        if source[key].shape != (count, 480, 640, 3) or source[key].dtype != np.uint8:
            raise ValueError(f"{source.filename}: expected {key} uint8 RGB [{count},480,640,3]")
        cameras[camera] = source_camera
    return cameras


def inspect_source(path):
    with h5py.File(path, "r") as source:
        rgb_path, depth_path = (f"observations/{kind}/{SOURCE_CAMERA}"
                                for kind in ("images", "images_depth"))
        for name in ("observations/qpos", "action", rgb_path, depth_path):
            if name not in source:
                raise ValueError(f"{path}: missing {name}")
        count = source["observations/qpos"].shape[0]
        if count < 1:
            raise ValueError(f"Empty episode: {path}")
        rgb_cameras = source_rgb_cameras(source, count)
        for name in ("observations/qpos", "action", "observations/qvel", "observations/effort"):
            if name in source and (source[name].shape != (count, 14)
                                   or source[name].dtype.kind != "f"):
                raise ValueError(f"{path}: expected floating {name} [{count},14]")
        if (source[rgb_path].shape != (count, 480, 640, 3)
                or source[rgb_path].dtype != np.uint8
                or source[depth_path].shape != (count, 480, 640)
                or source[depth_path].dtype != np.uint16):
            raise ValueError(f"{path}: expected aligned 640x480 uint8 RGB and uint16 depth")
        for key, expected in (("depth_unit", "mm"), ("rgb_semantics", "rgb8_rectified"),
                              ("depth_alignment", "factory_calibrated_d2c_to_rectified_color")):
            if key in source.attrs and json_value(source.attrs[key]) != expected:
                raise ValueError(f"{path}: unsupported {key}={source.attrs[key]!r}")
        return {"frames": count, "rgb_cameras": rgb_cameras,
                "needs_fps": (f"observations/camera_timestamps/{SOURCE_CAMERA}" not in source
                              and "capture_fps_target" not in source.attrs)}


def safe_name(value):
    if not isinstance(value, str) or not value or value in (".", "..") or Path(value).name != value:
        raise ValueError(f"Invalid task/episode name: {value!r}")
    return value


def default_task_name(root, group):
    name = root.name if group == "." else group.replace("/", "__")
    return re.sub(r"[^\w.-]+", "_", name).strip("._") or "task"


def prompt(text, default=None):
    while True:
        value = input(text + (f" [{default}]" if default is not None else "") + ": ").strip()
        if value or default is not None:
            return value or str(default)
        print("此项不能为空。", flush=True)


def collect_metadata(root, groups, infos, path, *, instruction=None, fps=None):
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"directories": {}}
    records = manifest["directories"]
    interactive = sys.stdin.isatty()
    for group in groups:
        record = dict(records.get(group, {}))
        if instruction is not None:
            record["instruction"] = instruction.strip()
        if not record.get("instruction"):
            if not interactive:
                raise ValueError(f"目录 {group!r} 缺少 instruction；请在终端交互运行，"
                                 "或使用 --instruction / 已填写的 --metadata-file")
            print(f"\n源目录: {root / group}（{len(groups[group])} 个 episode）", flush=True)
            record["task"] = prompt("任务名", record.get("task", default_task_name(root, group)))
            record["instruction"] = prompt("Instruction（本目录共享）")
            record["status"] = prompt("轨迹结果 success/failure", record.get("status", "success"))
        record.setdefault("task", default_task_name(root, group))
        record.setdefault("status", "success")
        safe_name(record["task"])
        if not isinstance(record["instruction"], str) or not record["instruction"].strip():
            raise ValueError(f"Instruction must be a nonempty string: {group}")
        if record["status"] not in ("success", "failure"):
            raise ValueError(f"Invalid status for {group}: {record['status']}")
        if fps is not None:
            record["fps"] = fps
        if any(infos[p]["needs_fps"] for p in groups[group]) and "fps" not in record:
            if not interactive:
                raise ValueError(f"{group}: missing sampling rate; fill fps or pass --fps")
            record["fps"] = float(prompt("缺少时间戳与采集 FPS，请填写帧率", 30))
        if "fps" in record and (not np.isfinite(float(record["fps"])) or float(record["fps"]) <= 0):
            raise ValueError(f"Expected positive finite fps for {group}")
        records[group] = record
        manifest.update(version=1, source_root=str(root))
        # Persist each answer immediately; a later interruption does not lose it.
        write_json(path, manifest)
    return records


def read_timing(source, count, record):
    key = f"observations/camera_timestamps/{SOURCE_CAMERA}"
    nominal = source.attrs.get("capture_fps_target", record.get("fps"))
    if key in source:
        raw = np.asarray(source[key], dtype=np.float64)
        if raw.shape != (count,) or not np.isfinite(raw).all() or np.any(np.diff(raw) <= 0):
            raise ValueError("Camera timestamps must be finite, strictly increasing [T]")
        times = raw - raw[0]
        if nominal is None:
            nominal = 1 / np.median(np.diff(raw)) if count > 1 else record.get("fps", 30)
        clock = "source camera clock, episode-relative (source origin preserved)"
    else:
        if nominal is None:
            raise ValueError("Missing camera timestamps and FPS")
        raw = None
        times = np.arange(count, dtype=np.float64) / float(nominal)
        clock = "synthesized frame_index / fps"
    nominal = float(nominal)
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError("Expected finite positive sampling FPS")
    return times, raw, nominal, clock


def save_array(directory, relative, array):
    target = directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    # Replacement also makes recalibration safe when the staging copy uses hardlinks.
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.save(stream, array, allow_pickle=False)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, target)


def calibration_metadata(path):
    return {"source": str(path), "sha256": file_digest(path),
            "source_intrinsic_size_wh": [322, 252], "removed_padding_left_top": [1, 6],
            "extrinsic_convention": "world_to_camera; world is footprint; matrix used directly",
            "source_image_size_wh": [640, 480], "rgb_resize": "INTER_AREA",
            "depth_resize": "INTER_NEAREST", "depth_already_aligned": True}


def validate_episode(path):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("complete") is not True:
        raise ValueError(f"Not a complete {FORMAT} episode: {path}")
    count = meta["frame_count"]
    if not isinstance(count, int) or count < 1 or (meta["width"], meta["height"]) != (320, 240):
        raise ValueError("Invalid episode frame count or image size")
    if meta.get("base_frame") != "footprint" or meta.get("extrinsics") != "base_to_camera":
        raise ValueError("Expected footprint-to-OpenCV-camera labels")
    times = np.load(path / "timestamps.npy", allow_pickle=False)
    if (times.shape != (count,) or not np.isfinite(times).all()
            or not np.isclose(times[0], 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Invalid episode timestamps")
    states = np.load(path / f"TCP/{CAMERA}/state.npy", allow_pickle=False)
    poses = rigid_transform(np.load(path / f"TCP/{CAMERA}/pose.npy", allow_pickle=False), "TCP poses")
    if (states.shape != (count, 2, 7) or poses.shape != (count, 2, 4, 4)
            or not np.isfinite(states).all() or np.any((states[..., 6] < 0) | (states[..., 6] > 1))):
        raise ValueError("Invalid dual-arm TCP states/poses")
    rotation = Rotation.from_euler("xyz", states[..., 3:6].reshape(-1, 3)).as_matrix()
    if (not np.allclose(states[..., :3], poses[..., :3, 3], atol=2e-6)
            or not np.allclose(rotation.reshape(count, 2, 3, 3), poses[..., :3, :3], atol=2e-6)):
        raise ValueError("TCP states and full poses disagree")
    tcp_meta = json.loads((path / f"TCP/{CAMERA}/metadata.json").read_text(encoding="utf-8"))
    if (tcp_meta.get("arm_order") != list(ARMS) or tcp_meta.get("columns") != STATE_COLUMNS
            or tcp_meta.get("rpy_convention") != RPY_CONVENTION
            or tcp_meta.get("coordinate_frame") != CAMERA_FRAME):
        raise ValueError("Unsupported TCP metadata")
    for relative in ("observations/joint_position.npy", "action/joint_position.npy"):
        values = np.load(path / relative, allow_pickle=False)
        if values.shape != (count, 14) or not np.isfinite(values).all():
            raise ValueError(f"Invalid source-order numeric array: {relative}")
    intrinsic = np.load(path / f"intrinsic/{CAMERA}.npy", allow_pickle=False)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1])):
        raise ValueError("Invalid intrinsic")
    extrinsic = rigid_transform(np.load(path / f"extrinsic/{CAMERA}.npy", allow_pickle=False), "extrinsic")
    if extrinsic.shape != (4, 4):
        raise ValueError("Expected static camera extrinsic [4,4]")
    base_states = np.load(path / "observations/cartesian_position.npy", allow_pickle=False)
    base_poses = np.linalg.inv(extrinsic) @ poses
    if (base_states.shape != (count, 2, 6) or not np.isfinite(base_states).all()
            or not np.allclose(base_states[..., :3], base_poses[..., :3, 3], atol=2e-6)
            or not np.allclose(Rotation.from_euler("xyz", base_states[..., 3:].reshape(-1, 3))
                               .as_matrix().reshape(count, 2, 3, 3), base_poses[..., :3, :3], atol=2e-6)):
        raise ValueError("Base-frame Cartesian observations disagree with camera TCP poses")
    depth_meta = json.loads((path / "depths/metadata.json").read_text(encoding="utf-8"))
    if (depth_meta.get("units") != "millimeters" or depth_meta.get("dtype") != "uint16"
            or depth_meta.get("depth_type") != "distance_to_image_plane"
            or depth_meta.get("invalid_value") != 0
            or depth_meta.get("cameras", {}).get(CAMERA) != {"frames": count, "width": 320, "height": 240}):
        raise ValueError("Invalid depth metadata")
    expected_names = {f"{i:06d}.png" for i in range(count)}
    rgb_cameras = meta.get("rgb_cameras", {CAMERA: SOURCE_CAMERA})
    if (not isinstance(rgb_cameras, dict) or rgb_cameras.get(CAMERA) != SOURCE_CAMERA
            or any(RGB_CAMERAS.get(camera) != source_camera
                   for camera, source_camera in rgb_cameras.items())):
        raise ValueError("Invalid RGB camera metadata")
    # Incremental wrist extraction deliberately preserves the original metadata.
    rgb_cameras = {**rgb_cameras, **{camera: source_camera for camera, source_camera in RGB_CAMERAS.items()
                                   if (path / "images" / camera).is_dir()}}
    valid_depth = 0
    for kind, camera in [("images", camera) for camera in rgb_cameras] + [("depths", CAMERA)]:
        directory = path / kind / camera
        if not directory.is_dir() or {p.name for p in directory.iterdir()} != expected_names:
            raise ValueError(f"Non-contiguous or mismatched {kind}/{camera} frames")
        for i in range(count):
            frame = cv2.imread(str(directory / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            shape = (240, 320, 3) if kind == "images" else (240, 320)
            dtype = np.uint8 if kind == "images" else np.uint16
            if frame is None or frame.shape != shape or frame.dtype != dtype:
                raise ValueError(f"Invalid {kind}/{camera} frame {i}")
            if kind == "depths":
                valid_depth += np.count_nonzero(frame)
    xyz = poses[..., :3, 3]
    pixels = xyz @ intrinsic.T
    uv = np.divide(pixels[..., :2], pixels[..., 2:], out=np.full_like(pixels[..., :2], np.nan),
                   where=pixels[..., 2:] != 0)
    visible = ((xyz[..., 2] > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < 320)
               & (uv[..., 1] >= 0) & (uv[..., 1] < 240))
    return {"frames": count, "rgb_cameras": list(rgb_cameras), "valid_depth_pixels": int(valid_depth),
            "tcp_in_image_frames": dict(zip(ARMS, visible.sum(axis=0).tolist()))}


def convert_episode(source_path, destination, record, model, offsets, intrinsic, extrinsic,
                    calibration_path):
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        with h5py.File(source_path, "r") as source:
            qpos, action = np.asarray(source["observations/qpos"]), np.asarray(source["action"])
            if not np.isfinite(action).all():
                raise ValueError("Nonfinite source actions")
            count = len(qpos)
            rgb_cameras = source_rgb_cameras(source, count)
            times, raw_times, fps, clock = read_timing(source, count, record)
            openness = np.clip(qpos[:, [6, 13]] / 0.10, 0, 1)
            base_poses = model.tcp_poses(qpos, offsets)
            camera_poses = extrinsic @ base_poses
            states = poses_to_states(camera_poses, openness).astype(np.float32)
            save_array(temporary, "observations/joint_position.npy", qpos)
            save_array(temporary, "observations/gripper_position.npy", qpos[:, [6, 13]])
            save_array(temporary, "observations/cartesian_position.npy",
                       poses_to_states(base_poses, openness)[..., :6].astype(np.float32))
            save_array(temporary, "action/joint_position.npy", action)
            save_array(temporary, "action/gripper_position.npy", action[:, [6, 13]])
            for src, target in (("observations/qvel", "observations/joint_velocity.npy"),
                                ("observations/effort", "observations/effort.npy"),
                                ("base_action", "action/base_action.npy")):
                if src in source:
                    values = np.asarray(source[src])
                    if values.shape[0] != count or not np.isfinite(values).all():
                        raise ValueError(f"Invalid numeric source: {src}")
                    save_array(temporary, target, values)
            if raw_times is not None:
                save_array(temporary, "observations/camera_timestamp.npy", raw_times)
            save_array(temporary, "timestamps.npy", times)
            save_array(temporary, f"intrinsic/{CAMERA}.npy", intrinsic)
            save_array(temporary, f"extrinsic/{CAMERA}.npy", extrinsic)
            save_array(temporary, f"TCP/{CAMERA}/state.npy", states)
            save_array(temporary, f"TCP/{CAMERA}/pose.npy", camera_poses)
            tcp_meta = {
                "shape": list(states.shape), "pose_shape": list(camera_poses.shape),
                "arm_order": list(ARMS), "columns": STATE_COLUMNS,
                "position_unit": "meter", "rotation_unit": "radian",
                "coordinate_frame": CAMERA_FRAME, "rpy_convention": RPY_CONVENTION,
                "pose_convention": "T_camera_tcp; maps TCP coordinates into OpenCV camera",
                "tcp_definition_id": "aloha_fixed_fingertip_center_v1",
                "tcp_definition": "fixed midpoint of area centroids of closed inner fingertip contact faces; orientation equals link6",
                "tcp_reference_links": list(WRISTS), "tcp_offsets_in_link6_m": offsets.tolist(),
            }
            write_json(temporary / f"TCP/{CAMERA}/metadata.json", tcp_meta)
            write_json(temporary / "depths/metadata.json", {
                "units": "millimeters", "dtype": "uint16", "invalid_value": 0,
                "depth_type": "distance_to_image_plane", "frame_alignment": "index",
                "cameras": {CAMERA: {"frames": count, "width": 320, "height": 240}},
            })
            for camera in rgb_cameras:
                (temporary / "images" / camera).mkdir(parents=True)
            (temporary / "depths" / CAMERA).mkdir(parents=True)
            for i in range(count):
                for camera, source_camera in rgb_cameras.items():
                    rgb = cv2.resize(source[f"observations/images/{source_camera}"][i],
                                     (320, 240), interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(temporary / "images" / camera / f"{i:06d}.png"),
                                       cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"Failed writing images/{camera} frame {i}")
                depth = cv2.resize(source[f"observations/images_depth/{SOURCE_CAMERA}"][i],
                                   (320, 240), interpolation=cv2.INTER_NEAREST)
                if not cv2.imwrite(str(temporary / "depths" / CAMERA / f"{i:06d}.png"), depth):
                    raise OSError(f"Failed writing depths/{CAMERA} frame {i}")
                if i == 0 or (i + 1) % 100 == 0 or i + 1 == count:
                    print(f"{source_path.name}: {i + 1}/{count} 帧", flush=True)
            layout = [f"{p}_joint{i}" for p in ("fl", "fr") for i in range(1, 8)]
            layout[6], layout[13] = "left_gripper_width_m", "right_gripper_width_m"
            meta = {
                "format": FORMAT, "task": record["task"], "episode": destination.name,
                "language_instruction": [record["instruction"]], "status": record["status"],
                "complete": True, "camera_id": CAMERA, "source_camera": SOURCE_CAMERA,
                "rgb_cameras": rgb_cameras,
                "width": 320, "height": 240, "fps": fps, "frame_count": count,
                "frame_alignment": "same source HDF5 row for RGB, depth, state and action; no resampling",
                "timestamp_unit": "seconds", "timestamp_clock": clock,
                "source_timestamp_origin": float(raw_times[0]) if raw_times is not None else None,
                "base_frame": "footprint", "extrinsics": "base_to_camera", "camera_frame": CAMERA_FRAME,
                "arm_order": list(ARMS), "qpos_layout": layout,
                "arm_joint_names": [[f"{p}_joint{i}" for i in range(1, 7)] for p in ("fl", "fr")],
                "arm_joint_unit": "radian", "gripper_position_unit": "meter; total two-finger width",
                "gripper_open": "clip(source_width_m / 0.10, 0, 1)",
                "gripper_replay": "joint7=+max(width,0)/2; joint8=-max(width,0)/2; no URDF limit clamp",
                "cartesian_position": "[T,2,6] fixed TCP xyz/rpy in footprint; meters/radians",
                "rpy_convention": RPY_CONVENTION,
                "action_semantics": "source 14D action preserved unchanged; no temporal shift or inferred control semantics",
                "urdf": str(model.path), "urdf_sha256": file_digest(model.path),
                "unobserved_joint_values": {j.name: model.joint_defaults.get(j.name, 0.0)
                                            for j in model.joints if j.kind != "fixed"
                                            and not j.name.startswith(("fl_joint", "fr_joint"))},
                "calibration": calibration_metadata(calibration_path),
                "source": {"path": str(source_path), "bytes": source_path.stat().st_size,
                           "mtime_ns": source_path.stat().st_mtime_ns,
                           "attributes": json_value(dict(source.attrs))},
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(temporary / "metadata.json", meta)
        report = validate_episode(temporary)
        meta["validation"] = report
        write_json(temporary / "metadata.json", meta)
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def converted_episodes(path):
    path = Path(path).expanduser().resolve()
    if (path / "metadata.json").is_file():
        return [path]
    episodes = []
    for child in path.glob("*"):
        if (child / "metadata.json").is_file():
            episodes.append(child)
        elif child.is_dir():
            episodes.extend(p.parent for p in child.glob("*/metadata.json"))
    if not episodes:
        raise ValueError(f"No converted episodes found under {path}")
    return sorted(episodes)


def inspect_wrist_append(path):
    """Plan an append without modifying any episode files or metadata."""
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    count = meta["frame_count"]
    if (meta.get("format") != FORMAT or meta.get("complete") is not True
            or not isinstance(count, int) or count < 1
            or (meta["width"], meta["height"]) != (320, 240)):
        raise ValueError(f"Unsupported converted episode: {path}")
    source_path = Path(meta["source"]["path"]).expanduser()
    with h5py.File(source_path, "r") as source:
        qpos = np.load(path / "observations/joint_position.npy", allow_pickle=False)
        if (qpos.shape != (count, 14)
                or not np.array_equal(qpos, source["observations/qpos"])):
            raise ValueError(f"Source joints/frame alignment mismatch: {path}")
        timestamp_path = path / "observations/camera_timestamp.npy"
        if timestamp_path.is_file():
            key = f"observations/camera_timestamps/{SOURCE_CAMERA}"
            if key not in source or not np.array_equal(
                    np.load(timestamp_path, allow_pickle=False), source[key]):
                raise ValueError(f"Source camera timestamps mismatch: {path}")
        cameras = source_rgb_cameras(source, count)
        for camera in ("left_view", "right_view"):
            if camera not in cameras:
                raise ValueError(f"{source_path}: missing wrist RGB {RGB_CAMERAS[camera]}")
    expected = {f"{i:06d}.png" for i in range(count)}
    existing = {}
    for camera in ("left_view", "right_view"):
        directory = path / "images" / camera
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Expected wrist image directory: {directory}")
        names = {p.name for p in directory.glob("*.png")}
        if names - expected:
            raise ValueError(f"Unexpected wrist frame names: {directory}")
        for name in names:
            frame = cv2.imread(str(directory / name), cv2.IMREAD_UNCHANGED)
            if frame is None or frame.shape != (240, 320, 3) or frame.dtype != np.uint8:
                raise ValueError(f"Invalid existing wrist frame; will not overwrite: {directory / name}")
        existing[camera] = len(names)
    return {"episode": str(path), "source": str(source_path), "frames": count,
            "existing": existing, "missing": {camera: count - n for camera, n in existing.items()}}


def write_png_exclusive(target, image):
    """Publish a complete PNG atomically; an existing destination always wins."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Failed encoding PNG: {target}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded.tobytes())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_wrist_rgb(path):
    """Resume wrist RGB extraction, leaving every existing file unchanged."""
    path = Path(path)
    plan = inspect_wrist_append(path)
    written, skipped = {}, {}
    with h5py.File(plan["source"], "r") as source:
        for camera in ("left_view", "right_view"):
            directory = path / "images" / camera
            directory.mkdir(parents=True, exist_ok=True)
            images = source[f"observations/images/{RGB_CAMERAS[camera]}"]
            written[camera] = skipped[camera] = 0
            for i in range(plan["frames"]):
                target = directory / f"{i:06d}.png"
                if os.path.lexists(target):
                    skipped[camera] += 1
                    continue
                rgb = cv2.resize(images[i], (320, 240), interpolation=cv2.INTER_AREA)
                if write_png_exclusive(target, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    written[camera] += 1
                else:
                    skipped[camera] += 1
            print(f"{path.name}: {camera} 新增 {written[camera]}，跳过 {skipped[camera]}", flush=True)
    return {"episode": str(path), "frames": plan["frames"], "written": written, "skipped": skipped}


def recompute_episode(path, calibration_path, *, urdf=None, offset_left=None,
                      offset_right=None, joint_defaults=None):
    """Rebuild labels from stored measured joints, reusing RGB-D via hardlinks.

    Validate a staged episode before swapping directories; restore the original
    on a failed publish. Neither source HDF5 nor task instructions are needed.
    """
    validate_episode(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    tcp_meta = json.loads((path / f"TCP/{CAMERA}/metadata.json").read_text(encoding="utf-8"))
    model_path = Path(urdf or meta["urdf"]).expanduser()
    if not model_path.is_file() and urdf is None:
        model_path = DEFAULT_URDF
    defaults = {**meta["unobserved_joint_values"], **(joint_defaults or {})}
    model = RobotModel(model_path, joint_defaults=defaults)
    if urdf is not None and file_digest(model.path) != meta["urdf_sha256"]:
        offsets = model.fingertip_offsets()
    else:
        if file_digest(model.path) != meta["urdf_sha256"]:
            raise ValueError("Recorded URDF changed; pass --urdf explicitly to recompute using a new model")
        offsets = np.array(tcp_meta["tcp_offsets_in_link6_m"], dtype=np.float64)
    for arm, value in enumerate((offset_left, offset_right)):
        if value is not None:
            offsets[arm] = value
    if offsets.shape != (2, 3) or not np.isfinite(offsets).all():
        raise ValueError("Expected finite TCP offsets [2,3]")
    intrinsic, extrinsic = read_calibration(calibration_path)
    qpos = np.load(path / "observations/joint_position.npy", allow_pickle=False)
    openness = np.clip(qpos[:, [6, 13]] / 0.10, 0, 1)
    base_poses = model.tcp_poses(qpos, offsets)
    camera_poses = extrinsic @ base_poses
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.recompute.", dir=path.parent))
    backup = temporary.with_name(temporary.name + ".previous")
    try:
        # PNGs can be large. Hardlinks preserve their bytes/inodes without re-encoding.
        shutil.copytree(path, temporary, dirs_exist_ok=True, copy_function=os.link)
        save_array(temporary, f"intrinsic/{CAMERA}.npy", intrinsic)
        save_array(temporary, f"extrinsic/{CAMERA}.npy", extrinsic)
        save_array(temporary, f"TCP/{CAMERA}/state.npy", poses_to_states(camera_poses, openness).astype(np.float32))
        save_array(temporary, f"TCP/{CAMERA}/pose.npy", camera_poses)
        save_array(temporary, "observations/cartesian_position.npy",
                   poses_to_states(base_poses, openness)[..., :6].astype(np.float32))
        tcp_meta["tcp_offsets_in_link6_m"] = offsets.tolist()
        write_json(temporary / f"TCP/{CAMERA}/metadata.json", tcp_meta)
        meta.update(calibration=calibration_metadata(calibration_path), urdf=str(model.path),
                    urdf_sha256=file_digest(model.path),
                    recalculated_utc=datetime.now(timezone.utc).isoformat(),
                    unobserved_joint_values={j.name: model.joint_defaults.get(j.name, 0.0)
                                             for j in model.joints if j.kind != "fixed"
                                             and not j.name.startswith(("fl_joint", "fr_joint"))})
        write_json(temporary / "metadata.json", meta)
        report = validate_episode(temporary)
        meta["validation"] = report
        write_json(temporary / "metadata.json", meta)
        try:
            path.rename(backup)
            temporary.rename(path)
        except BaseException:
            if backup.exists() and not path.exists():
                backup.rename(path)
            raise
        shutil.rmtree(backup)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=argparse.SUPPRESS,
                        help="HDF5 source (default: datasets/4_cams); in existing-data modes, "
                             "the existing dataset, task directory or episode to process")
    parser.add_argument("--output", type=Path, default=argparse.SUPPRESS,
                        help="Conversion destination (default: datasets/4_cams_robolab); "
                             "also accepted as the existing-data path")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--urdf", type=Path,
                        help="Default: bundled URDF for conversion; recorded URDF for recalculation")
    parser.add_argument("--metadata-file", type=Path, help="Default: <output>/task_metadata.json")
    parser.add_argument("--instruction", help="Optional unattended instruction for all selected directories")
    parser.add_argument("--fps", type=float, help="Fallback FPS when source timestamps and capture FPS are absent")
    parser.add_argument("--tcp-offset-left", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--tcp-offset-right", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--joint-default", action="append", default=[], metavar="NAME=VALUE",
                        help="Override an unobserved joint; repeat as necessary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect conversion or wrist append without writing or prompting")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true",
                      help="Validate converted episodes selected by --input or --output")
    mode.add_argument("--recompute-tcp", action="store_true",
                      help="Update episodes selected by --input or --output in place with the current calibration; "
                           "reuse RGB-D/instructions")
    mode.add_argument("--append-wrist-rgb", action="store_true",
                      help="Add missing wrist PNGs to existing episodes using recorded source HDF5 paths; "
                           "never overwrite any existing file or metadata")
    args = parser.parse_args(argv)
    if args.dry_run and (args.recompute_tcp or args.validate_only):
        parser.error("--dry-run supports conversion and --append-wrist-rgb only")
    # Preserve whether a path was explicitly supplied before applying conversion defaults.
    input_path, output_path = getattr(args, "input", None), getattr(args, "output", None)
    if args.recompute_tcp or args.validate_only or args.append_wrist_rgb:
        if (input_path is not None and output_path is not None
                and input_path.expanduser().resolve() != output_path.expanduser().resolve()):
            parser.error("Existing-data modes update or check existing data in place; "
                         "use --input or --output to select one existing path, not two different paths")
        output_path = input_path or output_path
    args.input = input_path or ROOT / "datasets/4_cams"
    args.output = output_path or ROOT / "datasets/4_cams_robolab"
    return args


def run(args):
    args.output = args.output.expanduser().resolve()
    if args.append_wrist_rgb:
        episodes = converted_episodes(args.output)
        # Check every source and alignment before creating the first file.
        plans = [inspect_wrist_append(episode) for episode in episodes]
        if args.dry_run:
            print(json.dumps(plans, ensure_ascii=False, indent=2))
            return
        reports = [append_wrist_rgb(episode) for episode in episodes]
        print(json.dumps({"episodes": len(reports),
                          "written": sum(sum(r["written"].values()) for r in reports),
                          "skipped": sum(sum(r["skipped"].values()) for r in reports)},
                         ensure_ascii=False), flush=True)
        return
    if args.validate_only:
        for episode in converted_episodes(args.output):
            print(episode, json.dumps(validate_episode(episode), ensure_ascii=False), flush=True)
        return
    defaults = {}
    for item in args.joint_default:
        name, value = item.split("=", 1)
        defaults[name] = float(value)
    args.calibration = args.calibration.expanduser().resolve()
    if args.recompute_tcp:
        for episode in converted_episodes(args.output):
            report = recompute_episode(episode, args.calibration, urdf=args.urdf,
                                       offset_left=args.tcp_offset_left, offset_right=args.tcp_offset_right,
                                       joint_defaults=defaults)
            print(f"已更新标定和双臂 TCP: {episode}\n{json.dumps(report, ensure_ascii=False)}", flush=True)
        return
    root, groups = discover_sources(args.input)
    infos = {p: inspect_source(p) for paths in groups.values() for p in paths}
    if args.dry_run:
        print(json.dumps({"source_root": str(root), "directories": {
            group: [{"source": str(p), **infos[p]} for p in paths]
            for group, paths in groups.items()}}, ensure_ascii=False, indent=2))
        return
    model = RobotModel(args.urdf or DEFAULT_URDF, joint_defaults=defaults)
    intrinsic, extrinsic = read_calibration(args.calibration)
    if args.tcp_offset_left is not None and args.tcp_offset_right is not None:
        offsets = np.array([args.tcp_offset_left, args.tcp_offset_right])
    else:
        offsets = model.fingertip_offsets()
        for i, override in enumerate((args.tcp_offset_left, args.tcp_offset_right)):
            if override is not None:
                offsets[i] = override
    if not np.isfinite(offsets).all():
        raise ValueError("TCP offsets must be finite")
    print(f"TCP offsets in link6 (m): {offsets.tolist()}", flush=True)
    records = collect_metadata(root, groups, infos,
                               (args.metadata_file or args.output / "task_metadata.json").expanduser().resolve(),
                               instruction=args.instruction, fps=args.fps)
    jobs = []
    for group, paths in groups.items():
        for path in paths:
            record = records[group]
            name = safe_name(f"{record['task']}__{path.stem}")
            jobs.append((path, args.output / record["task"] / name, record))
    destinations = [dest for _, dest, _ in jobs]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate output episode names; use distinct task names in task_metadata.json")
    for dest in destinations:
        if dest.exists():
            raise FileExistsError(f"Output already exists; choose a new --output: {dest}")
    failures = []
    for source, destination, record in jobs:
        try:
            report = convert_episode(source, destination, record, model, offsets,
                                     intrinsic, extrinsic, args.calibration)
            print(f"完成: {destination}\n{json.dumps(report, ensure_ascii=False)}", flush=True)
        except (ValueError, OSError, KeyError) as error:
            failures.append((str(source), str(error)))
            print(f"失败: {source}: {error}", file=sys.stderr, flush=True)
    if failures:
        raise ValueError(f"{len(failures)}/{len(jobs)} episodes failed: {failures}")


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        raise SystemExit("\n转换已停止；已填写的信息保存在 task_metadata.json。")
    except (ValueError, OSError, KeyError, TypeError, EOFError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
