"""URDF kinematics and calibrated RGB-D geometry, independent of Rerun/CUDA."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf"
DEFAULT_CALIBRATION = ROOT / "cobot_magic/4cam.yaml"
FORMAT = "cobot_magic_robolab_dual_arm_v1"
CAMERA = "third_person"
SOURCE_CAMERA = "cam_third_person"
ARMS = ("left", "right")
WRISTS = ("fl_link6", "fr_link6")
STATE_COLUMNS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"]
RPY_CONVENTION = "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))"
CAMERA_FRAME = "OpenCV camera (+x right, +y down, +z forward)"
AXIS_COLORS = [[235, 65, 65], [65, 210, 80], [65, 130, 255]]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rigid_transform(value, name="transform") -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if (value.shape[-2:] != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(np.swapaxes(value[..., :3, :3], -1, -2) @ value[..., :3, :3],
                               np.eye(3), atol=2e-4)
            or not np.allclose(np.linalg.det(value[..., :3, :3]), 1, atol=2e-4)):
        raise ValueError(f"Expected rigid {name} [...,4,4]")
    return value


def read_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    camera = yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"][
        "observation.images.cam_third_person"]
    convention = camera.get("extrinsic_convention")
    if convention not in (None, "world_to_camera", "base_to_camera", "w2c"):
        raise ValueError(f"Expected footprint/world-to-camera calibration, got {convention!r}")
    extrinsic = rigid_transform(camera["extrinsic"], "footprint-to-camera extrinsic")
    if extrinsic.shape != (4, 4):
        raise ValueError("Expected one static extrinsic [4,4]")
    intrinsic = np.array(camera["intrinsic"], dtype=np.float64)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1])):
        raise ValueError("Expected finite pinhole intrinsic [3,3]")
    # YAML K is already for 320x240 + left/right 1px and top/bottom 6px.
    intrinsic[:2, 2] -= [1, 6]
    return intrinsic, extrinsic


def resolve_mesh(urdf: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        package, relative = filename[len("package://"):].split("/", 1)
        for root in urdf.parents:
            candidate = root / relative if root.name == package else root / package / relative
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Cannot resolve {filename} relative to {urdf}")
    path = (urdf.parent / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def origin_matrix(node) -> np.ndarray:
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class RobotModel:
    """Named front-arm joints; other movable joints use explicit defaults."""

    def __init__(self, path=DEFAULT_URDF, *, joint_defaults=None):
        self.path = Path(path).expanduser().resolve()
        self.xml = ET.parse(self.path).getroot()
        self.links = {link.get("name"): link for link in self.xml.findall("link")}
        self.joints = []
        for node in self.xml.findall("joint"):
            kind = node.get("type")
            if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                raise ValueError(f"Unsupported joint type: {kind}")
            axis_node = node.find("axis")
            axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None
                                 else "1 0 0", sep=" ")
            if kind != "fixed":
                if np.linalg.norm(axis) < 1e-9:
                    raise ValueError(f"Zero joint axis: {node.get('name')}")
                axis /= np.linalg.norm(axis)
            self.joints.append(Joint(node.get("name"), kind, node.find("parent").get("link"),
                                     node.find("child").get("link"),
                                     origin_matrix(node.find("origin")), axis))
        roots = set(self.links) - {j.child for j in self.joints}
        if roots != {"footprint"}:
            raise ValueError(f"Expected URDF root footprint, got {roots}")
        pending, ordered, known = self.joints[:], [], {"footprint"}
        while pending:
            ready = [j for j in pending if j.parent in known]
            if not ready:
                raise ValueError("Disconnected/cyclic URDF joint tree")
            for j in ready:
                known.add(j.child)
                ordered.append(j)
                pending.remove(j)
        self.joints = ordered
        names = {j.name for j in self.joints}
        measured = {f"{p}_joint{i}" for p in ("fl", "fr") for i in range(1, 9)}
        if not measured.issubset(names):
            raise ValueError(f"Missing front-arm joints: {sorted(measured - names)}")
        self.joint_defaults = dict(joint_defaults or {})
        if set(self.joint_defaults) - names or set(self.joint_defaults) & measured:
            raise ValueError("Joint defaults may only name unobserved URDF joints")
        if not all(np.isfinite(v) for v in self.joint_defaults.values()):
            raise ValueError("Joint defaults must be finite")

    def joint_values(self, qpos) -> dict[str, np.ndarray]:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != 14 or not np.isfinite(qpos).all():
            raise ValueError("Expected finite qpos [T,14]: left six + width, right six + width")
        values = {j.name: np.full(len(qpos), self.joint_defaults.get(j.name, 0.0))
                  for j in self.joints}
        for prefix, start in (("fl", 0), ("fr", 7)):
            for i in range(6):
                values[f"{prefix}_joint{i + 1}"] = qpos[:, start + i]
            width = np.maximum(qpos[:, start + 6], 0.0)
            values[f"{prefix}_joint7"] = width / 2
            values[f"{prefix}_joint8"] = -width / 2
        return values

    def forward(self, qpos) -> dict[str, np.ndarray]:
        values = self.joint_values(qpos)
        transforms = {"footprint": np.tile(np.eye(4), (len(qpos), 1, 1))}
        for joint in self.joints:
            motion = np.tile(np.eye(4), (len(qpos), 1, 1))
            if joint.kind in ("revolute", "continuous"):
                motion[:, :3, :3] = Rotation.from_rotvec(
                    values[joint.name][:, None] * joint.axis).as_matrix()
            elif joint.kind == "prismatic":
                motion[:, :3, 3] = values[joint.name][:, None] * joint.axis
            transforms[joint.child] = transforms[joint.parent] @ joint.origin @ motion
        return transforms

    def fingertip_offsets(self) -> np.ndarray:
        """Area centroid of opposing contact faces at zero finger displacement.

        In link6 the closed contact plane is x=0. Filter *all triangle vertices*
        within 0.1mm of that plane and inward normals within cos(theta)>0.999.
        This excludes the small disconnected export artifacts in link8.dae.
        """
        import trimesh

        zero = self.forward(np.zeros((1, 14)))
        offsets = []
        for prefix, wrist in zip(("fl", "fr"), WRISTS):
            contacts = []
            for finger, inward in ((7, 1), (8, -1)):
                link_name = f"{prefix}_link{finger}"
                weighted, total = np.zeros(3), 0.0
                for visual in self.links[link_name].findall("visual"):
                    mesh_node = visual.find("geometry/mesh")
                    if mesh_node is None:
                        continue
                    scene = trimesh.load(resolve_mesh(self.path, mesh_node.get("filename")),
                                         force="scene", process=False)
                    scale = np.eye(4)
                    scale[:3, :3] = np.diag(np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" "))
                    link_tf = (np.linalg.inv(zero[wrist][0]) @ zero[link_name][0]
                               @ origin_matrix(visual.find("origin")) @ scale)
                    for node in scene.graph.nodes_geometry:
                        scene_tf, geometry = scene.graph[node]
                        mesh = scene.geometry[geometry].copy()
                        mesh.apply_transform(link_tf @ scene_tf)
                        select = ((np.abs(mesh.triangles[:, :, 0]).max(axis=1) < 1e-4)
                                  & (mesh.face_normals[:, 0] * inward > 0.999))
                        areas = mesh.area_faces[select]
                        weighted += (mesh.triangles_center[select] * areas[:, None]).sum(axis=0)
                        total += areas.sum()
                if total <= 1e-10:
                    raise ValueError(f"No fingertip contact surface for {link_name}; provide TCP offsets")
                contacts.append(weighted / total)
            offsets.append(np.mean(contacts, axis=0))
        return np.asarray(offsets)

    def tcp_poses(self, qpos, offsets) -> np.ndarray:
        transforms = self.forward(qpos)
        local = np.tile(np.eye(4), (2, 1, 1))
        local[:, :3, 3] = np.asarray(offsets)
        return np.stack([transforms[wrist] for wrist in WRISTS], axis=1) @ local


def poses_to_states(poses, gripper_open) -> np.ndarray:
    poses = np.asarray(poses)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        angles = Rotation.from_matrix(poses[..., :3, :3].reshape(-1, 3, 3)).as_euler("xyz")
    angles = angles.reshape(*poses.shape[:-2], 3)
    return np.concatenate([poses[..., :3, 3], angles, np.asarray(gripper_open)[..., None]], axis=-1)


def backproject(rgb, depth_mm, intrinsic, camera_to_base, *, min_depth=0.1,
                max_depth=3.0, max_points=100_000):
    z = depth_mm.astype(np.float64) / 1000
    yy, xx = np.indices(z.shape)
    ids = np.flatnonzero((z > min_depth) & (z <= max_depth) & np.isfinite(z))
    if max_points and len(ids) > max_points:
        ids = ids[np.linspace(0, len(ids) - 1, max_points, dtype=int)]
    pixels = np.stack([xx.ravel()[ids], yy.ravel()[ids], np.ones(len(ids))], axis=-1)
    points = (pixels @ np.linalg.inv(intrinsic).T) * z.ravel()[ids, None]
    points = points @ camera_to_base[:3, :3].T + camera_to_base[:3, 3]
    return points.astype(np.float32), rgb.reshape(-1, 3)[ids]
