"""Sequential cuRobo v2 IK for the dataset's fixed Robotiq TCP work point."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import ARM_JOINT_NAMES


TCP_LINK = "four_rc_tcp"
BASE_LINK = "panda_link0"
CUROBO_COMMIT = "78fd485fa82d9b9a063fb4985e371814587e666a"


def read_tcp_offset(episode: Path, camera: str) -> np.ndarray:
    path = episode / "TCP" / camera / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    offset = np.asarray(metadata.get("tcp_offset_in_robotiq_base_m"), dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError(f"Missing finite tcp_offset_in_robotiq_base_m [3] in {path}")
    return offset


def prepare_ik_urdf(source: Path, destination: Path, tcp_offset: np.ndarray) -> np.ndarray:
    """Extract the arm chain and append the label work point, without visual meshes."""
    document = ET.parse(source)
    root = document.getroot()
    parents = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
    chain = []
    link_name = "robotiq_arg2f_base_link"
    seen = set()
    while link_name != BASE_LINK:
        if link_name in seen or link_name not in parents:
            raise ValueError("Expected a Panda chain from panda_link0 to robotiq_arg2f_base_link")
        seen.add(link_name)
        joint = parents[link_name]
        chain.append(joint)
        link_name = joint.find("parent").get("link")
    chain.reverse()
    active = [joint.get("name") for joint in chain if joint.get("type") != "fixed"]
    if active != list(ARM_JOINT_NAMES):
        raise ValueError(f"Expected seven Panda arm joints in URDF order, got {active}")
    links = {link.get("name"): link for link in root.findall("link")}
    output = ET.Element("robot", name="four_rc_ik")
    for name in [BASE_LINK] + [joint.find("child").get("link") for joint in chain]:
        link = deepcopy(links[name])
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                link.remove(element)
        output.append(link)
    for joint in chain:
        output.append(deepcopy(joint))
    ET.SubElement(output, "link", name=TCP_LINK)
    joint = ET.SubElement(output, "joint", name="four_rc_tcp_joint", type="fixed")
    ET.SubElement(joint, "parent", link="robotiq_arg2f_base_link")
    ET.SubElement(joint, "child", link=TCP_LINK)
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, tcp_offset)), rpy="0 0 0")
    ET.ElementTree(output).write(destination, encoding="utf-8", xml_declaration=True)
    return np.asarray([[float(joint.find("limit").get(key)) for key in ("lower", "upper")]
                       for joint in chain if joint.get("type") != "fixed"])


def pose_errors(poses: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.linalg.norm(poses[..., :3, 3] - goal[:3, 3], axis=-1)
    rotation = Rotation.from_matrix(poses[..., :3, :3].reshape(-1, 3, 3))
    angle = (Rotation.from_matrix(goal[:3, :3]).inv() * rotation).magnitude()
    return position.reshape(-1), angle


def solve_trajectory(target_poses: np.ndarray, initial_joints: np.ndarray, gripper_open: np.ndarray,
                     frame_indices: np.ndarray, urdf: Path, tcp_offset: np.ndarray, *,
                     device="cuda:0", num_seeds=32, position_tolerance=0.005,
                     rotation_tolerance=0.05, use_cuda_graph=True, progress=None):
    """Warm-start even the first predicted pose; hold the last valid arm on failure."""
    import torch
    try:
        import curobo
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose
    except ImportError as error:
        raise RuntimeError("cuRobo v2 is required in 4rc; see rerun_visualizations/README_CN.md") from error

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("cuRobo IK requires an available CUDA device")
    target_poses = np.asarray(target_poses, dtype=np.float64)
    initial_joints = np.asarray(initial_joints, dtype=np.float64)
    count = len(frame_indices)
    if (target_poses.shape != (count, 4, 4) or not np.isfinite(target_poses).all()
            or initial_joints.shape != (7,) or not np.isfinite(initial_joints).all()
            or np.asarray(gripper_open).shape != (count,)):
        raise ValueError("Expected finite TCP poses [N,4,4], initial joints [7], and gripper [N]")
    states = dict(frame_indices=np.asarray(frame_indices, dtype=np.int64),
                  joints=np.empty((count, 7)), gripper_open=np.asarray(gripper_open),
                  fk_tcp_poses=np.empty((count, 4, 4)), success=np.zeros(count, dtype=bool),
                  position_error_m=np.empty(count), rotation_error_rad=np.empty(count),
                  solve_seconds=np.empty(count))
    tensor_args = dict(device=device, dtype=torch.float32)
    with tempfile.TemporaryDirectory(prefix="4rc_ik_") as temporary:
        model_path = Path(temporary) / "robot.urdf"
        limits = prepare_ik_urdf(urdf, model_path, tcp_offset)
        if np.any(initial_joints < limits[:, 0]) or np.any(initial_joints > limits[:, 1]):
            raise ValueError("Initial measured joints violate URDF joint limits")
        config = InverseKinematicsCfg.create(
            robot={"kinematics": {"urdf_path": str(model_path), "base_link": BASE_LINK,
                                   "tool_frames": [TCP_LINK]}},
            device_cfg=DeviceCfg(device=device), num_seeds=num_seeds,
            position_tolerance=position_tolerance, orientation_tolerance=rotation_tolerance,
            self_collision_check=False, load_collision_spheres=False,
            use_cuda_graph=use_cuda_graph, random_seed=123,
        )
        solver = InverseKinematics(config)
        if set(solver.joint_names) != set(ARM_JOINT_NAMES):
            raise ValueError(f"Unexpected cuRobo arm joints: {solver.joint_names}")
        solver_order = [ARM_JOINT_NAMES.index(name) for name in solver.joint_names]
        arm_order = [solver.joint_names.index(name) for name in ARM_JOINT_NAMES]

        def fk(joints):
            js = JointState.from_position(joints, joint_names=solver.joint_names)
            pose = solver.compute_kinematics(js).tool_poses.get_link_pose(TCP_LINK)
            xyz = pose.position.detach().cpu().numpy().reshape(-1, 3)
            wxyz = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)
            matrices = np.tile(np.eye(4), (len(xyz), 1, 1))
            matrices[:, :3, 3] = xyz
            matrices[:, :3, :3] = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]]).as_matrix()
            return matrices

        previous = torch.tensor(initial_joints[solver_order][None], **tensor_args)
        initial_fk = fk(previous)[0]
        try:
            for slot, goal_matrix in enumerate(target_poses):
                began = time.monotonic()
                xyzw = Rotation.from_matrix(goal_matrix[:3, :3]).as_quat()
                pose = Pose(position=torch.tensor(goal_matrix[None, :3, 3], **tensor_args),
                            quaternion=torch.tensor(xyzw[None, [3, 0, 1, 2]], **tensor_args))
                result = solver.solve_pose(
                    GoalToolPose.from_poses({TCP_LINK: pose}, num_goalset=1),
                    current_state=JointState.from_position(previous.clone(), joint_names=solver.joint_names),
                    return_seeds=num_seeds,
                )
                candidates = result.solution.reshape(-1, 7)
                arm_candidates = candidates[:, arm_order].detach().cpu().numpy()
                valid = result.success.detach().cpu().numpy().reshape(-1).copy()
                valid &= np.isfinite(arm_candidates).all(axis=1)
                valid &= ((arm_candidates >= limits[:, 0]) & (arm_candidates <= limits[:, 1])).all(axis=1)
                candidate_ids = np.flatnonzero(valid)
                if len(candidate_ids):
                    candidate_poses = fk(candidates[candidate_ids])
                    pos_error, rot_error = pose_errors(candidate_poses, goal_matrix)
                    valid_ids = np.flatnonzero((pos_error <= position_tolerance)
                                               & (rot_error <= rotation_tolerance))
                    if len(valid_ids):
                        ids = candidate_ids[valid_ids]
                        distance = (candidates[ids] - previous).square().sum(dim=-1)
                        best = int(ids[int(distance.argmin())])
                        previous = candidates[best:best + 1].clone()
                        states["success"][slot] = True
                applied_pose = fk(previous)[0]
                states["joints"][slot] = previous[0, arm_order].detach().cpu().numpy()
                states["fk_tcp_poses"][slot] = applied_pose
                pos_error, rot_error = pose_errors(applied_pose, goal_matrix)
                states["position_error_m"][slot] = pos_error[0]
                states["rotation_error_rad"][slot] = rot_error[0]
                states["solve_seconds"][slot] = time.monotonic() - began
                if progress:
                    progress(slot, count, states)
        finally:
            solver.destroy()
    return states, dict(curobo_version=curobo.__version__, curobo_commit=CUROBO_COMMIT,
                        joint_names=list(ARM_JOINT_NAMES), initial_joints=initial_joints.tolist(),
                        initial_fk_tcp_pose=initial_fk.tolist(), tcp_offset_in_robotiq_base_m=tcp_offset.tolist(),
                        num_seeds=num_seeds, position_tolerance_m=position_tolerance,
                        rotation_tolerance_rad=rotation_tolerance, use_cuda_graph=use_cuda_graph,
                        successful_frames=int(states["success"].sum()), total_frames=count,
                        failure_policy="hold_last_valid_arm; gripper_uses_prediction",
                        initialization="measured start-frame joints used only as first IK seed",
                        collision_check=False)
