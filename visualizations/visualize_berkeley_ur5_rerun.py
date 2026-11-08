#!/usr/bin/env python3
"""Replay Berkeley AUTOLAB UR5 TCP4D RGB-D, robot and TCP in Rerun Web."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import time
from urllib.parse import quote
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint.components import LoopMode, PlayState
from scipy.spatial.transform import Rotation


DEFAULT_INPUT = Path(
    "/home/shengyu/Documents/PENG/datasets/Berkeley_AUTOLAB_UR5/"
    "TCP4D_Berkeley_AUTOLAB_UR5/train_set"
)
DEFAULT_CALIBALL = Path(__file__).resolve().parents[2] / "CalibAll"
BASE_FRAME = "ur5_base"
CAMERA_FRAME = "third_person_opencv"
IMAGE_FRAME = "third_person_image"
ROBOT_ENTITY = "robot_model"
ROBOT_PREFIX = "ur5/"
TIMELINE = "episode_time"
TCP_COLOR = [245, 145, 45]
AXIS_COLORS = [[235, 65, 65], [65, 210, 80], [65, 130, 255]]
ARM_JOINTS = [f"{name}_joint" for name in (
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_episode(path, episode):
    path = path.expanduser().resolve()
    if (path / "metadata.json").is_file():
        return path
    if (path / "train_set").is_dir():
        path /= "train_set"
    name = f"berkeley_autolab_ur5__episode_{int(episode)}" if episode.isdecimal() else episode
    if Path(name).name != name:
        raise ValueError("--episode must be an episode number or directory name")
    selected = path / name
    if not (selected / "metadata.json").is_file():
        raise FileNotFoundError(f"Episode metadata not found: {selected / 'metadata.json'}")
    return selected


def require_rigid(value, name):
    if (value.shape[-2:] != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(value[..., :3, :3].swapaxes(-1, -2) @ value[..., :3, :3],
                               np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(value[..., :3, :3]), 1, atol=1e-5)):
        raise ValueError(f"Invalid rigid transform: {name}")


def tcp_reference(tcp_meta):
    """Return the TCP's reference link and fixed link-to-TCP transform."""
    definition = tcp_meta.get("tcp_definition_id")
    tip = np.eye(4)
    if definition == "robotiq_2f85_closed_contact_face_midpoint_v2":
        reference = "robotiq_arg2f_base_link"
        if tcp_meta.get("tcp_orientation_reference") != reference:
            raise ValueError(f"Unsupported TCP orientation reference for {definition!r}")
        offset = np.asarray(tcp_meta["tcp_position_in_gripper_base_m"], dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Expected finite tcp_position_in_gripper_base_m with shape (3,)")
        tip[:3, 3] = offset
        return reference, tip
    raise ValueError(f"Unsupported TCP definition: {definition!r}")


def load_episode(path, args):
    meta = read_json(path / "metadata.json")
    if meta.get("format") != "berkeley_autolab_ur5_tcp4d_v1":
        raise ValueError("Expected berkeley_autolab_ur5_tcp4d_v1 episode")
    if not all(meta.get(key) for key in ("complete", "rgb_tcp_complete", "depth_complete")):
        raise ValueError("Episode conversion is incomplete")
    camera, count = meta["camera_id"], meta["frame_count"]
    if meta.get("timestamp_unit") != "seconds" or not meta.get("extrinsics", "").startswith("base_to_camera"):
        raise ValueError("Expected seconds timestamps and base-to-camera extrinsic")
    depth_meta = read_json(path / "depths/metadata.json")
    if depth_meta.get("units") != "millimeters" or depth_meta.get("depth_type") != "distance_to_image_plane":
        raise ValueError("Expected optical-axis depth in millimeters")
    tcp_meta = read_json(path / f"TCP/{camera}/metadata.json")
    tcp_reference(tcp_meta)
    data = {"meta": meta, "path": path, "tcp_meta": tcp_meta}
    for key, filename, shape in (
        ("intrinsic", f"intrinsic/{camera}.npy", (3, 3)),
        ("extrinsic", f"extrinsic/{camera}.npy", (4, 4)),
        ("poses", f"TCP/{camera}/pose.npy", (count, 1, 4, 4)),
        ("states", f"TCP/{camera}/state.npy", (count, 1, 7)),
        ("qpos", "observations/joint_position.npy", (count, 7)),
        ("timestamps", "timestamps.npy", (count,)),
    ):
        value = np.load(path / filename, allow_pickle=False)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Expected finite {filename} with shape {shape}; got {value.shape}")
        data[key] = value
    require_rigid(data["extrinsic"], "base-to-camera extrinsic")
    require_rigid(data["poses"], "TCP poses")
    k = data["intrinsic"]
    if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1]) or abs(np.linalg.det(k)) < 1e-9:
        raise ValueError("Invalid pinhole intrinsics")
    if count < 1 or np.any(np.diff(data["timestamps"]) <= 0) or meta["fps"] <= 0:
        raise ValueError("Expected nonempty episode, increasing timestamps and positive FPS")
    if not np.allclose(data["poses"][:, 0, :3, 3], data["states"][:, 0, :3], atol=1e-6):
        raise ValueError("TCP state positions disagree with pose matrices")
    if not np.allclose(Rotation.from_euler("xyz", data["states"][:, 0, 3:6]).as_matrix(),
                       data["poses"][:, 0, :3, :3], atol=1e-6):
        raise ValueError("TCP state rotations disagree with pose matrices")
    indices = np.arange(args.start_frame, count, args.stride, dtype=np.int64)
    if args.max_frames:
        indices = indices[:args.max_frames]
    if not len(indices):
        raise ValueError(f"No selected frames; episode has {count} frames")
    for i in indices:
        for folder in ("images", "depths"):
            filename = path / folder / camera / f"{i:06d}.png"
            if not filename.is_file():
                raise FileNotFoundError(filename)
    data["indices"] = indices
    data["camera_to_base"] = np.linalg.inv(data["extrinsic"])
    data["poses_base"] = data["camera_to_base"] @ data["poses"][:, 0]
    return data


def read_rgbd(data, frame):
    meta, path = data["meta"], data["path"]
    filename = f"{meta['camera_id']}/{frame:06d}.png"
    with Image.open(path / "images" / filename) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(path / "depths" / filename) as image:
        depth = np.asarray(image)
    shape = (meta["height"], meta["width"])
    # Pillow versions may decode 16-bit grayscale PNG into int32.
    if (rgb.shape != (*shape, 3) or depth.shape != shape or depth.dtype.kind not in "ui"
            or depth.min() < 0 or depth.max() > 65535 or depth.dtype.itemsize < 2):
        raise ValueError(f"Invalid RGB/depth dimensions or depth dtype at frame {frame}")
    return rgb, depth.astype(np.uint16, copy=False)


def backproject(rgb, depth, intrinsic, camera_to_base, *, min_depth, max_depth, max_points):
    z = depth.astype(np.float64) / 1000.0
    valid = np.flatnonzero((z > min_depth) & (z <= max_depth))
    if max_points and len(valid) > max_points:
        valid = valid[np.linspace(0, len(valid) - 1, max_points, dtype=int)]
    v, u = np.unravel_index(valid, depth.shape)
    rays = np.column_stack((u, v, np.ones(len(valid)))) @ np.linalg.inv(intrinsic).T
    camera_points = rays * z.ravel()[valid, None]
    points = camera_points @ camera_to_base[:3, :3].T + camera_to_base[:3, 3]
    return points.astype(np.float32), rgb.reshape(-1, 3)[valid]


def origin_matrix(node):
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def joint_transforms(root, qpos):
    """Resolve measured angles and URDF mimic joints, including their signs."""
    values = dict(zip(ARM_JOINTS, qpos[:, :6].T))
    values["finger_joint"] = qpos[:, 6]
    result = {}
    for joint in root.findall("joint"):
        name, kind = joint.get("name"), joint.get("type")
        motion = np.broadcast_to(np.eye(4), (len(qpos), 4, 4)).copy()
        if kind != "fixed":
            mimic = joint.find("mimic")
            angle = (values[mimic.get("joint")] * float(mimic.get("multiplier", 1))
                     + float(mimic.get("offset", 0))) if mimic is not None else values[name]
            axis = joint.find("axis")
            axis = np.fromstring(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ")
            axis /= np.linalg.norm(axis)
            if kind in ("revolute", "continuous"):
                motion[:, :3, :3] = Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
            else:
                raise ValueError(f"Unexpected UR5/Robotiq joint type: {kind}")
        result[name] = origin_matrix(joint.find("origin")) @ motion
    return result


def forward_to(root, transforms, tip):
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain = []
    while tip in parents:
        joint = parents[tip]
        chain.append(joint.get("name"))
        tip = joint.find("parent").get("link")
    result = np.eye(4)
    for name in reversed(chain):
        result = result @ transforms[name]
    return result


def tcp_poses_from_joints(root, qpos, tcp_meta):
    reference, tip = tcp_reference(tcp_meta)
    closed = qpos.copy()
    closed[:, 6] = tcp_meta["gripper_closed_q"]
    return forward_to(root, joint_transforms(root, closed), reference) @ tip


def prepare_robot(data, caliball, directory):
    """Use conversion URDF snapshots, resolving visual meshes from CalibAll."""
    import trimesh

    provenance = data["path"].parent.parent / "provenance"
    hashes = read_json(provenance / "conversion.json")["files"]
    root = ET.Element("robot", name="berkeley_ur5e_robotiq")
    converted = {}
    for filename in ("ur5e.urdf", "robotiq_arg2f_85_model.urdf"):
        snapshot = provenance / filename
        sources = [Path(p) for p in hashes if Path(p).name == filename]
        if len(sources) != 1 or hashlib.sha256(snapshot.read_bytes()).hexdigest() != hashes[str(sources[0])]:
            raise ValueError(f"URDF snapshot differs from conversion provenance: {snapshot}")
        document = ET.parse(snapshot).getroot()
        for link in document.findall("link"):
            for collision in list(link.findall("collision")):
                link.remove(collision)
            for mesh in link.findall("visual/geometry/mesh"):
                package, relative = mesh.get("filename").removeprefix("package://").split("/", 1)
                family = "universal_robots" if package == "ur_description" else "robotiq"
                source = caliball / "third_party/urdf" / family / package / relative
                if not source.is_file():
                    # The original recorded project may live outside this checkout's parent.
                    source = sources[0].parent.parent / relative
                if not source.is_file():
                    raise FileNotFoundError(f"Robot mesh missing: {source}; set --caliball-root or --no-robot")
                if source not in converted:
                    target = directory / f"mesh_{len(converted):03d}.glb"
                    scene = trimesh.load(source, force="scene", process=False)
                    target.write_bytes(scene.export(file_type="glb"))
                    converted[source] = target
                mesh.set("filename", str(converted[source]))
        for node in document:
            if node.tag in ("link", "joint", "material"):
                root.append(node)
    mount = ET.SubElement(root, "joint", name="robotiq_mount", type="fixed")
    ET.SubElement(mount, "parent", link="tool0")
    ET.SubElement(mount, "child", link="robotiq_arg2f_base_link")
    tcp_meta = data["tcp_meta"]
    yaw = np.deg2rad(tcp_meta["gripper_mount_yaw_deg"])
    ET.SubElement(mount, "origin", xyz="0 0 0", rpy=f"0 0 {yaw:.17g}")
    expected = data["extrinsic"] @ tcp_poses_from_joints(root, data["qpos"], tcp_meta)
    error = float(np.max(np.abs(expected - data["poses"][:, 0])))
    if not np.allclose(expected, data["poses"][:, 0], atol=2e-6, rtol=0):
        raise ValueError(f"URDF FK disagrees with saved TCP: max matrix error={error:g}")
    print(f"TCP definition: {tcp_meta['tcp_definition_id']}\n"
          f"UR5e + Robotiq: FK validated for {len(data['qpos'])} frames (max error {error:.2g})", flush=True)
    path = directory / "visual_robot.urdf"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    tree = rr.urdf.UrdfTree.from_file_path(
        path, entity_path_prefix=ROBOT_ENTITY, frame_prefix=ROBOT_PREFIX,
        static_transform_entity_path="world/robot/static_transforms")
    return root, tree


def log_robot(recording, data, root, tree):
    recording.send_chunks(tree.stream(include_joint_transforms=False))
    recording.log("world/robot/root", rr.Transform3D(
        translation=[0, 0, 0], parent_frame=BASE_FRAME,
        child_frame=ROBOT_PREFIX + tree.root_link().name), static=True)
    indices = data["indices"]
    indexes = [rr.TimeColumn("frame", sequence=indices),
               rr.TimeColumn(TIMELINE, duration=data["timestamps"][indices])]
    transforms = joint_transforms(root, data["qpos"][indices])
    for joint in root.findall("joint"):
        name = joint.get("name")
        poses = transforms[name]
        parent = ROBOT_PREFIX + joint.find("parent").get("link")
        child = ROBOT_PREFIX + joint.find("child").get("link")
        path = f"world/robot/joints/{name}"
        if np.all(poses == poses[0]):
            recording.log(path, rr.Transform3D(
                translation=poses[0, :3, 3], mat3x3=poses[0, :3, :3],
                parent_frame=parent, child_frame=child), static=True)
        else:
            recording.send_columns(path, indexes=indexes, columns=rr.Transform3D.columns(
                translation=poses[:, :3, 3], mat3x3=poses[:, :3, :3],
                parent_frame=[parent] * len(poses), child_frame=[child] * len(poses)))


def make_blueprint(meta):
    bounds = rrb.VisualBounds2D(x_range=[0, meta["width"]], y_range=[0, meta["height"]])
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin="world", name="RGB-D point cloud · UR5e · TCP",
                    contents=["world/**", f"{ROBOT_ENTITY}/**"],
                    eye_controls=rrb.EyeControls3D(position=[1.3, 1.3, 1.2],
                                                 look_target=[0, 0, 0.15], eye_up=[0, 0, 1])),
                rrb.Spatial2DView(
                    origin="world/camera_frustum", name="Calibrated camera · cloud + robot + TCP",
                    contents=["world/camera_frustum", "world/point_cloud", "world/tcp/**", f"{ROBOT_ENTITY}/**"],
                    visual_bounds=bounds), column_shares=[1.5, 1]),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="rgb", name="Recorded RGB · orange=TCP", visual_bounds=bounds),
                rrb.Spatial2DView(origin="depth", name="Optical-axis depth · metres"),
                rrb.TextDocumentView(origin="info", name="Instruction / TCP pose"), column_shares=[1, 1, 1]),
            row_shares=[3, 1.4]),
        rrb.TimePanel(timeline=TIMELINE, expanded=True, fps=meta["fps"],
                      play_state=PlayState.Paused, loop_mode=LoopMode.All),
        auto_views=False, collapse_panels=True)


def log_scene(recording, args, data):
    meta, indices = data["meta"], data["indices"]
    intrinsic, c2b = data["intrinsic"], data["camera_to_base"]
    recording.log("world", rr.CoordinateFrame(BASE_FRAME), rr.ViewCoordinates.FLU, static=True)
    recording.log("world/base_axes", rr.CoordinateFrame(BASE_FRAME), rr.Arrows3D(
        origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.15, colors=AXIS_COLORS,
        labels=["base X", "base Y", "base Z"], radii=0.002), static=True)
    recording.log("world/camera_pose", rr.Transform3D(
        translation=c2b[:3, 3], mat3x3=c2b[:3, :3],
        parent_frame=BASE_FRAME, child_frame=CAMERA_FRAME), static=True)
    recording.log("world/camera_frustum", rr.CoordinateFrame(IMAGE_FRAME), rr.Pinhole(
        image_from_camera=intrinsic, width=meta["width"], height=meta["height"],
        camera_xyz=rr.ViewCoordinates.RDF, parent_frame=CAMERA_FRAME,
        child_frame=IMAGE_FRAME, image_plane_distance=0.15), static=True)
    recording.log("world/point_cloud", rr.CoordinateFrame(BASE_FRAME), static=True)
    recording.log("world/tcp/trajectory", rr.CoordinateFrame(BASE_FRAME), rr.LineStrips3D(
        [data["poses_base"][indices, :3, 3]], colors=TCP_COLOR, radii=0.002), static=True)
    recording.log("world/tcp/axes", rr.CoordinateFrame("tcp"), rr.Arrows3D(
        origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.06, colors=AXIS_COLORS, radii=0.0015), static=True)
    recording.log("world/tcp/point", rr.CoordinateFrame("tcp"), rr.Points3D(
        [[0, 0, 0]], colors=TCP_COLOR, radii=0.006, labels=["UR5 TCP"], show_labels=True), static=True)
    for slot, i in enumerate(indices):
        recording.set_time("frame", sequence=int(i))
        recording.set_time(TIMELINE, duration=float(data["timestamps"][i]))
        rgb, depth = read_rgbd(data, i)
        points, colors = backproject(rgb, depth, intrinsic, c2b,
                                    min_depth=args.min_depth_m, max_depth=args.max_depth_m,
                                    max_points=args.max_points)
        recording.log("world/point_cloud", rr.Points3D(points, colors=colors, radii=args.point_size / 2))
        recording.log("rgb", rr.Image(rgb))
        recording.log("depth", rr.DepthImage(depth, meter=1000,
                      depth_range=[args.min_depth_m * 1000, args.max_depth_m * 1000]))
        pose = data["poses_base"][i]
        recording.log("world/tcp/pose", rr.Transform3D(
            translation=pose[:3, 3], mat3x3=pose[:3, :3], parent_frame=BASE_FRAME, child_frame="tcp"))
        xyz = data["poses"][i, 0, :3, 3]
        uvw = intrinsic @ xyz
        pixel = uvw[:2] / uvw[2] if xyz[2] > 0 else np.array([-1, -1])
        if 0 <= pixel[0] < meta["width"] and 0 <= pixel[1] < meta["height"]:
            recording.log("rgb/tcp", rr.Points2D(
                [pixel], colors=TCP_COLOR, radii=4, labels=["UR5 TCP"], show_labels=True))
        else:
            recording.log("rgb/tcp", rr.Clear(recursive=True))
        state = data["states"][i, 0]
        vector = lambda v: "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"
        instruction = "\n\n".join(meta.get("language_instruction", [meta["task"]]))
        recording.log("info", rr.TextDocument(
            f"## {meta['episode']}\n\n{instruction}\n\n"
            f"Frame **{i}/{meta['frame_count'] - 1}** · **{data['timestamps'][i]:.3f} s** · {meta['fps']:g} Hz\n\n"
            f"Camera xyz: {vector(state[:3])} m\n\nRPY: {vector(state[3:6])} rad\n\n"
            f"Base xyz: {vector(pose[:3, 3])} m\n\nGripper open: **{state[6]:.3f}** (1=open)\n\n"
            f"Cloud: **{len(points):,}** points; {args.min_depth_m:g}–{args.max_depth_m:g} m.\n\n"
            f"Robot model: **{'disabled' if args.no_robot else 'UR5e + Robotiq 2F-85'}**.\n\n"
            "3D uses URDF base_link in metres; camera uses OpenCV (+X right, +Y down, +Z forward).\n\n"
            f"TCP: {data['tcp_meta']['tcp_definition']}\n\n"
            "TCP is independent of current gripper opening.\n\n"
            f"Source status: {meta['status']}", media_type="text/markdown"))
        if slot == 0 or (slot + 1) % 50 == 0 or slot + 1 == len(indices):
            print(f"Logged {slot + 1}/{len(indices)} frames (source frame {i})", flush=True)
    recording.flush()


def choose_port(preferred, exclude=()):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        if preferred not in exclude:
            try:
                candidate.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    return choose_port(0, exclude) if port in exclude else port


def start_web(recording, args):
    grpc_port = choose_port(args.grpc_port)
    web_port = choose_port(args.web_port, exclude=(grpc_port,))
    uri = recording.serve_grpc(grpc_port=grpc_port, server_memory_limit="75%")
    rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
    print(f"Rerun Web viewer: http://localhost:{web_port}/?url={quote(uri, safe=':/')}&renderer={args.renderer}", flush=True)
    print("远程访问时在本机转发两个端口：\n"
          f"  ssh -N -L {web_port}:127.0.0.1:{web_port} "
          f"-L {grpc_port}:127.0.0.1:{grpc_port} <user>@<server>", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="train_set, dataset root or episode directory")
    parser.add_argument("--episode", default="0", help="Episode number or directory name; ignored for direct episode input")
    parser.add_argument("--caliball-root", type=Path, default=DEFAULT_CALIBALL, help="CalibAll checkout containing UR5e/Robotiq meshes")
    parser.add_argument("--no-robot", action="store_true", help="Display RGB-D and TCP without loading robot meshes")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0: all remaining selected frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=100_000, help="Per frame; 0: unlimited")
    parser.add_argument("--min-depth-m", type=float, default=0.10)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--point-size", type=float, default=0.003, help="Point diameter in metres")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--renderer", choices=("webgl", "webgpu"), default="webgl")
    parser.add_argument("--output", type=Path, help="Export self-contained .rrd and exit")
    args = parser.parse_args(argv)
    if min(args.start_frame, args.max_frames, args.max_points) < 0 or args.stride < 1:
        parser.error("Require start-frame/max-frames/max-points >= 0 and stride >= 1")
    if (not np.isfinite([args.min_depth_m, args.max_depth_m, args.point_size]).all()
            or not 0 <= args.min_depth_m < args.max_depth_m or args.point_size <= 0):
        parser.error("Require finite 0 <= min-depth-m < max-depth-m and point-size > 0")
    if not all(1 <= p <= 65535 for p in (args.web_port, args.grpc_port)) or args.web_port == args.grpc_port:
        parser.error("Web/gRPC ports must be distinct and between 1 and 65535")
    if args.output is not None and args.output.suffix.lower() != ".rrd":
        parser.error("--output must end in .rrd")
    return args


def run(args):
    episode = resolve_episode(args.input, args.episode)
    data = load_episode(episode, args)
    output = args.output.expanduser().resolve() if args.output is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Recording already exists: {output}")
    print(f"Episode: {episode}\nSelected {len(data['indices'])}/{data['meta']['frame_count']} frames", flush=True)
    with tempfile.TemporaryDirectory(prefix="berkeley_ur5_rerun_") as temporary:
        robot = None if args.no_robot else prepare_robot(data, args.caliball_root.expanduser().resolve(), Path(temporary))
        recording = rr.RecordingStream("berkeley_autolab_ur5")
        try:
            if output is None:
                start_web(recording, args)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                recording.save(output)
            recording.send_blueprint(make_blueprint(data["meta"]))
            if robot is not None:
                log_robot(recording, data, *robot)
            log_scene(recording, args, data)
            if output is not None:
                print(f"Saved: {output}", flush=True)
            else:
                print("回放已就绪；打开上方链接，在时间轴播放或拖动，Ctrl+C 停止服务。", flush=True)
                while True:
                    time.sleep(1)
        finally:
            recording.disconnect()


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        print("\nRerun 已停止。", flush=True)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
