"""Franka/Robotiq mesh preparation and measured joint replay, without Assimp."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rerun as rr
import trimesh


BASE_FRAME = "robot_base"
ROBOT_FRAME_PREFIX = "gt_robot/"
ROBOT_ENTITY = "gt_robot_model"
PRED_ROBOT_ENTITY = "pred_robot_model"
TIME_TIMELINE = "episode_time"
ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def read_dae_mesh(path: Path) -> trimesh.Scene:
    """Read the bundled triangle DAEs in their original URDF mesh coordinates.

    These legacy files have duplicate IDs and incomplete asset metadata. Resolve
    sources within each geometry, retain indexed normals, and deliberately do
    not rotate Z_UP into Y_UP or apply an implicit COLLADA unit conversion: the
    URDF already declares the mesh scale and link-frame orientation.
    """
    root = ET.parse(path).getroot()
    for node in root.findall(".//c:visual_scene//c:node", NS):
        if any(child.tag.rsplit("}", 1)[-1] in ("matrix", "translate", "rotate", "scale") for child in node):
            raise ValueError(f"DAE scene transforms are unsupported by the bundled-mesh converter: {path}")
    instance_ids = {item.get("url", "").lstrip("#") for item in root.findall(".//c:instance_geometry", NS)}
    scene = trimesh.Scene()
    for geometry in root.findall("./c:library_geometries/c:geometry", NS):
        if instance_ids and geometry.get("id") not in instance_ids:
            continue
        mesh = geometry.find("c:mesh", NS)
        if mesh is None:
            raise ValueError(f"Missing triangle mesh in {path}")
        sources = {}
        for source in mesh.findall("c:source", NS):
            array = source.find("c:float_array", NS)
            accessor = source.find("c:technique_common/c:accessor", NS)
            if array is None or accessor is None:
                raise ValueError(f"Missing source accessor in {path}")
            stride = int(accessor.get("stride", "1"))
            offset = int(accessor.get("offset", "0"))
            count = int(accessor.get("count", "0"))
            values = np.fromstring(array.text or "", sep=" ", dtype=np.float64)
            sources[source.get("id")] = values[offset:offset + count * stride].reshape(count, stride)
        vertices = {}
        for item in mesh.findall("c:vertices", NS):
            position_input = next((i for i in item.findall("c:input", NS) if i.get("semantic") == "POSITION"), None)
            if position_input is None:
                raise ValueError(f"Missing vertex POSITION input in {path}")
            vertices[item.get("id")] = position_input.get("source", "").lstrip("#")
        if any(child.tag.rsplit("}", 1)[-1] in ("polylist", "polygons", "trifans", "tristrips") for child in mesh):
            raise ValueError(f"Expected triangulated bundled DAE: {path}")
        for primitive_index, triangles in enumerate(mesh.findall("c:triangles", NS)):
            inputs = triangles.findall("c:input", NS)
            width = max(int(item.get("offset", "0")) for item in inputs) + 1
            index_nodes = triangles.findall("c:p", NS)
            indices = np.concatenate([np.fromstring(item.text or "", sep=" ", dtype=np.int64) for item in index_nodes])
            if indices.size != int(triangles.get("count", "0")) * 3 * width:
                raise ValueError(f"Invalid DAE triangle indices in {path}")
            indices = indices.reshape(-1, width)
            vertex_input = next(i for i in inputs if i.get("semantic") == "VERTEX")
            position_source = vertices[vertex_input.get("source", "").lstrip("#")]
            points = sources[position_source][indices[:, int(vertex_input.get("offset", "0"))], :3]
            normals = None
            normal_input = next((i for i in inputs if i.get("semantic") == "NORMAL"), None)
            if normal_input is not None:
                normals = sources[normal_input.get("source", "").lstrip("#")][indices[:, int(normal_input.get("offset", "0"))], :3]
            triangle_mesh = trimesh.Trimesh(
                vertices=points, faces=np.arange(len(points)).reshape(-1, 3),
                vertex_normals=normals, process=False,
            )
            scene.add_geometry(triangle_mesh, geom_name=f"{geometry.get('id')}_{primitive_index}")
    if not scene.geometry:
        raise ValueError(f"No triangle geometry in {path}")
    return scene


def prepare_robot(urdf_path: Path, temporary_directory: Path, *,
                  prefix: str = "ground_truth") -> rr.urdf.UrdfTree:
    """Prepare temporary GLB assets; never modify the supplied embodiment."""
    source = urdf_path.expanduser().resolve()
    document = ET.parse(source)
    converted = {}
    for link in document.getroot().findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        for mesh in link.findall("./visual/geometry/mesh"):
            filename = mesh.get("filename", "")
            if not filename or filename.startswith("package://"):
                raise ValueError(f"Expected self-contained URDF mesh paths: {filename!r}")
            asset = (source.parent / filename).resolve()
            if not asset.is_file():
                raise FileNotFoundError(f"Missing URDF mesh: {asset}")
            if asset.suffix.lower() == ".dae":
                if asset not in converted:
                    destination = temporary_directory / f"robotiq_{len(converted)}.glb"
                    destination.write_bytes(read_dae_mesh(asset).export(file_type="glb"))
                    converted[asset] = destination
                asset = converted[asset]
            mesh.set("filename", str(asset))
    prepared = temporary_directory / f"{prefix}_visual_robot.urdf"
    document.write(prepared, encoding="utf-8", xml_declaration=True)
    tree = rr.urdf.UrdfTree.from_file_path(
        prepared, entity_path_prefix=PRED_ROBOT_ENTITY if prefix == "prediction" else ROBOT_ENTITY,
        frame_prefix="pred_robot/" if prefix == "prediction" else ROBOT_FRAME_PREFIX,
        static_transform_entity_path=f"{prefix}/robot/static_transforms",
    )
    joint_names = {joint.name for joint in tree.joints()}
    if not set((*ARM_JOINT_NAMES, "finger_joint")).issubset(joint_names):
        raise ValueError("Expected Panda joints 1-7 and Robotiq finger_joint in --urdf")
    for joint in tree.joints():
        if joint.mimic is not None and joint.mimic.joint != "finger_joint":
            raise ValueError(f"Unsupported mimic source: {joint.mimic.joint!r}")
        if (joint.joint_type != "fixed" and joint.name not in (*ARM_JOINT_NAMES, "finger_joint")
                and joint.mimic is None):
            raise ValueError(f"No measured replay value for joint {joint.name!r}")
    return tree


def joint_values(joint, joints: np.ndarray, gripper_open: np.ndarray, *, closed_radians=0.8) -> np.ndarray:
    if joint.name in ARM_JOINT_NAMES:
        return joints[:, ARM_JOINT_NAMES.index(joint.name)]
    closing = (1.0 - gripper_open) * closed_radians
    if joint.name == "finger_joint":
        return closing
    if joint.mimic is not None:
        return closing * joint.mimic.multiplier + joint.mimic.offset
    return np.zeros(len(joints))


def log_robot(recording: rr.RecordingStream, tree: rr.urdf.UrdfTree, ground_truth=None, *,
              prefix="ground_truth", frame_indices=None, joints=None, gripper_open=None,
              timestamps=None, closed_radians=0.8) -> None:
    if ground_truth is not None:
        frame_indices, joints = ground_truth.frame_indices, ground_truth.joints
        gripper_open = ground_truth.tcp_camera[:, 6]
        timestamps = ground_truth.timestamps
        closed_radians = ground_truth.gripper_closed_radians
    frame_prefix = "pred_robot/" if prefix == "prediction" else ROBOT_FRAME_PREFIX
    recording.send_chunks(tree.stream(include_joint_transforms=False))
    recording.log(
        f"{prefix}/robot/root_transform",
        rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3),
                       parent_frame=BASE_FRAME, child_frame=frame_prefix + tree.root_link().name),
        static=True,
    )
    indexes = [rr.TimeColumn("frame", sequence=frame_indices),
               rr.TimeColumn(TIME_TIMELINE, duration=frame_indices / 15.0 if timestamps is None else timestamps)]
    for joint in tree.joints():
        path = f"{prefix}/robot/joints/{joint.name}"
        if joint.joint_type == "fixed":
            recording.log(path, joint.compute_transform(0, clamp=False), static=True)
        else:
            values = joint_values(joint, joints, gripper_open, closed_radians=closed_radians)
            # Negative mimic multipliers are intentional even when the original
            # Robotiq URDF's mimic joint limits are non-negative.
            recording.send_columns(path, indexes=indexes,
                                   columns=joint.compute_transform_columns(values, clamp=False))
