"""Ground-truth RGB-D and measured robot states in the DROID base frame."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class GroundTruth:
    frame_indices: np.ndarray
    depth_paths: list[Path]
    camera_to_base: np.ndarray
    tcp_camera: np.ndarray
    tcp_poses: np.ndarray
    joints: np.ndarray
    timestamps: np.ndarray
    gripper_closed_radians: float = 0.8


def poses_from_states(states: np.ndarray) -> np.ndarray:
    """Fixed-axis XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    poses = np.tile(np.eye(4), (len(states), 1, 1))
    poses[:, :3, 3] = states[:, :3]
    roll, pitch, yaw = states[:, 3:6].T
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    poses[:, :3, :3] = np.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp, cp * sr, cp * cr,
    ], axis=-1).reshape(-1, 3, 3)
    return poses


def load_ground_truth(episode, paths: list[Path]) -> GroundTruth:
    """Validate all selected replay inputs before loading the neural network."""
    count = len(episode.image_paths)
    indices = np.asarray([int(path.stem) for path in paths], dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= count) or np.any(np.diff(indices) != 1):
        raise ValueError("DROID RGB filenames must use contiguous original frame indices")
    base_to_camera = np.asarray(np.load(
        episode.path / "extrinsic" / f"{episode.camera}.npy", allow_pickle=False), dtype=np.float64)
    if (base_to_camera.shape != (4, 4) or not np.isfinite(base_to_camera).all()
            or not np.allclose(base_to_camera[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(base_to_camera[:3, :3].T @ base_to_camera[:3, :3], np.eye(3), atol=2e-4)
            or not np.isclose(np.linalg.det(base_to_camera[:3, :3]), 1, atol=2e-4)):
        raise ValueError("Expected a finite rigid base-to-camera extrinsic [4,4]")
    camera_to_base = np.linalg.inv(base_to_camera)
    tcp_path = episode.path / "TCP" / episode.camera / "state.npy"
    states = np.asarray(np.load(tcp_path, allow_pickle=False), dtype=np.float64)
    if states.shape != (count, 7):
        raise ValueError(f"Expected TCP state [{count},7] in {tcp_path}, got {states.shape}")
    tcp_metadata_path = tcp_path.with_name("metadata.json")
    if tcp_metadata_path.is_file():
        metadata = json.loads(tcp_metadata_path.read_text(encoding="utf-8"))
        expected = {
            "columns": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"],
            "position_unit": "meter", "rotation_unit": "radian",
            "coordinate_frame": "OpenCV camera (+x right, +y down, +z forward)",
            "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
        }
        for key, value in expected.items():
            if key == "coordinate_frame" and metadata.get(key) in (
                    value, "OpenCV (+x right, +y down, +z forward)"):
                continue
            if metadata.get(key) != value:
                raise ValueError(f"Unsupported {key} in {tcp_metadata_path}")
    states = states[indices]
    if not np.isfinite(states[:, 6]).all() or np.any((states[:, 6] < 0) | (states[:, 6] > 1)):
        raise ValueError(f"Expected finite gripper_open in [0,1]: {tcp_path}")
    joints_path = episode.path / "observations" / "joint_position.npy"
    joints = np.asarray(np.load(joints_path, allow_pickle=False), dtype=np.float64)
    if joints.shape != (count, 7) or not np.isfinite(joints[indices]).all():
        raise ValueError(f"Expected finite measured joints [{count},7] in {joints_path}")
    depth_metadata = episode.path / "depths" / "metadata.json"
    if depth_metadata.is_file():
        depth_info = json.loads(depth_metadata.read_text(encoding="utf-8"))
        units = depth_info.get("units", "millimeters")
        if str(units).lower() not in ("mm", "millimeter", "millimeters"):
            raise ValueError(f"Expected millimeter depth PNGs, got units={units!r}")
        if depth_info.get("depth_type", "distance_to_image_plane") != "distance_to_image_plane":
            raise ValueError("Expected optical-axis depth (distance_to_image_plane), not ray distance")
    depth_paths = [episode.path / "depths" / episode.camera / path.name for path in paths]
    for rgb_path, depth_path in zip(paths, depth_paths):
        with Image.open(rgb_path) as image:
            if image.size != (320, 180):
                raise ValueError(f"Expected 320x180 RGB: {rgb_path}")
        with Image.open(depth_path) as image:
            if image.size != (320, 180) or np.asarray(image).dtype != np.uint16:
                raise ValueError(f"Expected 320x180 uint16 millimeter depth: {depth_path}")
    tcp_poses = camera_to_base @ poses_from_states(states)
    tcp_poses[~np.isfinite(states[:, :6]).all(axis=-1)] = np.nan
    if episode.dataset == "robolab":
        from .robot import ARM_JOINT_NAMES
        if episode.metadata.get("arm_joint_names") != list(ARM_JOINT_NAMES):
            raise ValueError("RoboLab arm_joint_names must follow Panda joint1–joint7 order")
    times = np.asarray([episode.time_seconds(int(i)) for i in indices])
    # RoboLab normalizes measured finger_joint by pi/4; DROID uses 0.8 rad.
    closed_radians = np.pi / 4 if episode.dataset == "robolab" else 0.8
    return GroundTruth(indices, depth_paths, camera_to_base, states, tcp_poses, joints[indices],
                       times, closed_radians)


def backproject_rgbd(rgb: np.ndarray, depth_mm: np.ndarray, intrinsic: np.ndarray,
                    camera_to_base: np.ndarray, *, min_depth: float, max_depth: float,
                    max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    z = depth_mm.astype(np.float32) * 0.001
    yy, xx = np.indices(z.shape)
    selected = np.flatnonzero(np.isfinite(z) & (z > min_depth) & (z <= max_depth))
    if max_points and len(selected) > max_points:
        selected = np.random.default_rng(seed).choice(selected, max_points, replace=False)
    pixels = np.stack([xx.ravel()[selected], yy.ravel()[selected], np.ones(len(selected))], axis=-1)
    camera_points = (pixels @ np.linalg.inv(intrinsic).T) * z.ravel()[selected, None]
    points = camera_points @ camera_to_base[:3, :3].T + camera_to_base[:3, 3]
    return points.astype(np.float32), rgb.reshape(-1, 3)[selected]


def trajectory_segments(positions: np.ndarray) -> list[np.ndarray]:
    """Keep gaps in invalid GT poses instead of drawing across them."""
    valid = np.isfinite(positions).all(axis=-1)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return [positions[start:end] for start, end in edges.reshape(-1, 2) if end - start >= 2]


def project_pixel(position: np.ndarray, intrinsic: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(position).all() or position[2] <= 0:
        return None
    pixel = intrinsic @ position
    pixel = pixel[:2] / pixel[2]
    return pixel if 0 <= pixel[0] < 320 and 0 <= pixel[1] < 180 else None
