#!/usr/bin/env python3
"""Replay a converted dual-arm episode in Rerun Web, or export a standalone RRD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
from urllib.parse import quote
import xml.etree.ElementTree as ET

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint.components import LoopMode, PlayState
import trimesh

from cobot_magic.convert_to_4d_dataset import validate_episode
from cobot_magic.geometry import (
    ARMS, AXIS_COLORS, CAMERA, DEFAULT_URDF, RobotModel, backproject, file_digest, resolve_mesh,
)
from cobot_magic.refine_depth_to_urdf import CORRECTED_DIRECTORY, validate_corrected_depths


BASE_FRAME = "footprint"
CAMERA_FRAME = "third_person_opencv"
ROBOT_PREFIX = "aloha/"
ROBOT_ENTITY = "robot_model"
TIMELINE = "episode_time"
COLORS = ([245, 145, 45], [60, 160, 245])


def format_vector(values):
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def prepare_robot(model, directory):
    document = ET.parse(model.path)
    converted = {}
    for link in document.getroot().findall("link"):
        for collision in list(link.findall("collision")):
            link.remove(collision)
        for mesh in link.findall("visual/geometry/mesh"):
            source = resolve_mesh(model.path, mesh.get("filename"))
            if source.suffix.lower() == ".dae":
                if source not in converted:
                    target = directory / f"mesh_{len(converted):03d}.glb"
                    scene = trimesh.load(source, force="scene", process=False)
                    target.write_bytes(scene.export(file_type="glb"))
                    converted[source] = target
                source = converted[source]
            mesh.set("filename", str(source))
    path = directory / "visual_robot.urdf"
    document.write(path, encoding="utf-8", xml_declaration=True)
    return rr.urdf.UrdfTree.from_file_path(
        path, entity_path_prefix=ROBOT_ENTITY, frame_prefix=ROBOT_PREFIX,
        static_transform_entity_path="world/robot/static_transforms",
    )


def make_blueprint(fps):
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(
                origin="world", name="RGB-D point cloud · ALOHA · dual TCP",
                contents=["world/**", f"{ROBOT_ENTITY}/**"],
                eye_controls=rrb.EyeControls3D(position=[1.6, 1.5, 1.7],
                                             look_target=[0.25, 0, 0.85], eye_up=[0, 0, 1])),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="rgb", name="RGB · orange=left, blue=right"),
                rrb.Spatial2DView(origin="depth", name="Optical-axis depth · metres"),
                rrb.TextDocumentView(origin="info", name="Instruction / TCP pose"),
                column_shares=[1, 1, 1]),
            row_shares=[3, 1.4]),
        rrb.TimePanel(timeline=TIMELINE, expanded=True, fps=fps,
                      play_state=PlayState.Paused, loop_mode=LoopMode.All),
        auto_views=False, collapse_panels=True,
    )


def choose_port(preferred, exclude=()):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        if preferred not in exclude:
            try:
                candidate.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
        candidate.bind(("127.0.0.1", 0))
        selected = candidate.getsockname()[1]
    return choose_port(0, exclude) if selected in exclude else selected


def start_web(recording, args):
    grpc_port = choose_port(args.grpc_port)
    web_port = choose_port(args.web_port, exclude=(grpc_port,))
    uri = recording.serve_grpc(grpc_port=grpc_port, server_memory_limit="75%")
    rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
    print(f"Rerun Web viewer: http://localhost:{web_port}/?url={quote(uri, safe=':/')}&renderer={args.renderer}",
          flush=True)
    print("远程访问时在本机转发两个端口：\n"
          f"  ssh -N -L {web_port}:127.0.0.1:{web_port} "
          f"-L {grpc_port}:127.0.0.1:{grpc_port} <user>@<server>", flush=True)


def log_robot(recording, model, tree, qpos, indices, timestamps):
    recording.send_chunks(tree.stream(include_joint_transforms=False))
    recording.log("world/robot/root", rr.Transform3D(
        translation=[0, 0, 0], mat3x3=np.eye(3), parent_frame=BASE_FRAME,
        child_frame=ROBOT_PREFIX + tree.root_link().name), static=True)
    values = model.joint_values(qpos[indices])
    specs = {joint.name: joint for joint in model.joints}
    indexes = [rr.TimeColumn("frame", sequence=indices),
               rr.TimeColumn(TIMELINE, duration=timestamps[indices])]
    for joint in tree.joints():
        path = f"world/robot/joints/{joint.name}"
        angles = values[joint.name]
        if joint.joint_type == "prismatic":
            # URDF axes live in the joint frame: T = origin @ translate(axis*q).
            # Rerun 0.35 instead adds axis*q in the parent frame, which makes
            # ALOHA's fingers retract along wrist Z rather than open along X.
            spec = specs[joint.name]
            direction = spec.origin[:3, :3] @ spec.axis
            translations = spec.origin[:3, 3] + angles[:, None] * direction
            parent_frame = ROBOT_PREFIX + spec.parent
            child_frame = ROBOT_PREFIX + spec.child
            if np.all(angles == angles[0]):
                recording.log(path, rr.Transform3D(
                    translation=translations[0], mat3x3=spec.origin[:3, :3],
                    parent_frame=parent_frame, child_frame=child_frame), static=True)
            else:
                recording.send_columns(path, indexes=indexes, columns=rr.Transform3D.columns(
                    translation=translations,
                    mat3x3=np.broadcast_to(spec.origin[:3, :3], (len(angles), 3, 3)),
                    parent_frame=[parent_frame] * len(angles),
                    child_frame=[child_frame] * len(angles)))
            continue
        if joint.joint_type == "fixed" or np.all(angles == angles[0]):
            recording.log(path, joint.compute_transform(float(angles[0]), clamp=False), static=True)
        else:
            recording.send_columns(path, indexes=indexes,
                                   columns=joint.compute_transform_columns(angles, clamp=False))


def log_scene(recording, args, meta, indices, model, tree):
    path = args.input
    depth_directory = CORRECTED_DIRECTORY if args.depth_source == "corrected" else "depths"
    intrinsic = np.load(path / f"intrinsic/{CAMERA}.npy", allow_pickle=False)
    base_to_camera = np.load(path / f"extrinsic/{CAMERA}.npy", allow_pickle=False)
    camera_to_base = np.linalg.inv(base_to_camera)
    poses_camera = np.load(path / f"TCP/{CAMERA}/pose.npy", allow_pickle=False)
    poses_base = camera_to_base @ poses_camera
    states = np.load(path / f"TCP/{CAMERA}/state.npy", allow_pickle=False)
    qpos = np.load(path / "observations/joint_position.npy", allow_pickle=False)
    timestamps = np.load(path / "timestamps.npy", allow_pickle=False)
    recording.log("world", rr.CoordinateFrame(BASE_FRAME), rr.ViewCoordinates.FLU, static=True)
    recording.log("world/base_axes", rr.CoordinateFrame(BASE_FRAME), rr.Arrows3D(
        origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.15, colors=AXIS_COLORS,
        labels=["base X", "base Y", "base Z"], radii=0.002), static=True)
    recording.log("world/camera_pose", rr.Transform3D(
        translation=camera_to_base[:3, 3], mat3x3=camera_to_base[:3, :3],
        parent_frame=BASE_FRAME, child_frame=CAMERA_FRAME), static=True)
    recording.log("world/camera_frustum", rr.Pinhole(
        image_from_camera=intrinsic, width=320, height=240, camera_xyz=rr.ViewCoordinates.RDF,
        parent_frame=CAMERA_FRAME, child_frame="third_person_image", image_plane_distance=0.15), static=True)
    recording.log("world/point_cloud", rr.CoordinateFrame(BASE_FRAME), static=True)
    log_robot(recording, model, tree, qpos, indices, timestamps)
    for arm, name in enumerate(ARMS):
        recording.log(f"world/tcp/{name}/trajectory", rr.CoordinateFrame(BASE_FRAME),
                      rr.LineStrips3D([poses_base[indices, arm, :3, 3]], colors=COLORS[arm], radii=0.002),
                      static=True)
        frame = f"tcp_{name}"
        recording.log(f"world/tcp/{name}/axes", rr.CoordinateFrame(frame), rr.Arrows3D(
            origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.06,
            colors=AXIS_COLORS, radii=0.0015), static=True)
        recording.log(f"world/tcp/{name}/point", rr.CoordinateFrame(frame), rr.Points3D(
            [[0, 0, 0]], colors=COLORS[arm], radii=0.006, labels=[f"{name} TCP"], show_labels=True), static=True)
    for slot, i in enumerate(indices):
        recording.set_time("frame", sequence=int(i))
        recording.set_time(TIMELINE, duration=float(timestamps[i]))
        rgb = cv2.cvtColor(cv2.imread(str(path / f"images/{CAMERA}/{i:06d}.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(path / depth_directory / CAMERA / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        points, colors = backproject(rgb, depth, intrinsic, camera_to_base,
                                     min_depth=args.min_depth_m, max_depth=args.max_depth_m,
                                     max_points=args.max_points)
        recording.log("world/point_cloud", rr.Points3D(points, colors=colors, radii=args.point_size / 2))
        recording.log("rgb", rr.Image(rgb))
        # Colormap limits use the image's raw units (mm); meter only controls
        # physical depth readout/backprojection. Passing metres here saturates it.
        recording.log("depth", rr.DepthImage(depth, meter=1000,
                                             depth_range=[args.min_depth_m * 1000, args.max_depth_m * 1000]))
        for arm, name in enumerate(ARMS):
            pose = poses_base[i, arm]
            recording.log(f"world/tcp/{name}/pose", rr.Transform3D(
                translation=pose[:3, 3], mat3x3=pose[:3, :3],
                parent_frame=BASE_FRAME, child_frame=f"tcp_{name}"))
            xyz = poses_camera[i, arm, :3, 3]
            uv = intrinsic @ xyz
            pixel = uv[:2] / uv[2] if xyz[2] > 0 else np.array([-1, -1])
            if 0 <= pixel[0] < 320 and 0 <= pixel[1] < 240:
                recording.log(f"rgb/tcp/{name}", rr.Points2D(
                    [pixel], colors=COLORS[arm], radii=4, labels=[f"{name} TCP"], show_labels=True))
            else:
                recording.log(f"rgb/tcp/{name}", rr.Clear(recursive=True))
        pose_text = "\n\n".join(
            f"**{name}:** camera xyz={format_vector(states[i, a, :3])} m; "
            f"rpy={format_vector(states[i, a, 3:6])} rad; open={states[i, a, 6]:.3f}"
            for a, name in enumerate(ARMS))
        recording.log("info", rr.TextDocument(
            f"## {meta['task']}\n\n{meta['language_instruction'][0]}\n\n"
            f"Status: **{meta['status']}** · frame **{i}/{meta['frame_count'] - 1}** · "
            f"**{timestamps[i]:.3f} s**\n\n{pose_text}\n\n"
            f"3D: footprint (+X forward, +Y left, +Z up), metres.\n\n"
            f"Orange=left; blue=right. Cloud: {len(points):,} points.\n\n"
            f"Depth source: **{args.depth_source}**.\n\n"
            "URDF: measured front arms; other joints use recorded defaults.\n\n"
            "TCP: fixed closed-fingertip center, link6 orientation.", media_type="text/markdown"))
        if slot == 0 or (slot + 1) % 50 == 0 or slot + 1 == len(indices):
            print(f"Logged {slot + 1}/{len(indices)} frames (source frame {i})", flush=True)
    recording.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Converted episode directory")
    parser.add_argument("--urdf", type=Path, help="Override URDF location; content must match conversion")
    parser.add_argument("--depth-source", choices=("raw", "corrected"), default="raw",
                        help="Use original depths or the validated depths_corrected folder")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0: all remaining frames")
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
    if args.start_frame < 0 or args.max_frames < 0 or args.stride < 1 or args.max_points < 0:
        parser.error("Require start-frame/max-frames/max-points >=0 and stride >=1")
    if (not np.isfinite([args.min_depth_m, args.max_depth_m, args.point_size]).all()
            or not 0 <= args.min_depth_m < args.max_depth_m or args.point_size <= 0):
        parser.error("Require finite 0 <= min-depth-m < max-depth-m and point-size >0")
    if not all(1 <= p <= 65535 for p in (args.web_port, args.grpc_port)) or args.web_port == args.grpc_port:
        parser.error("Web/gRPC ports must be distinct and between 1 and 65535")
    if args.output is not None and args.output.suffix.lower() != ".rrd":
        parser.error("--output must end in .rrd")
    return args


def run(args):
    args.input = args.input.expanduser().resolve()
    report = validate_episode(args.input)
    meta = json.loads((args.input / "metadata.json").read_text(encoding="utf-8"))
    urdf = args.urdf or Path(meta["urdf"])
    if not urdf.expanduser().is_file() and args.urdf is None:
        urdf = DEFAULT_URDF
    model = RobotModel(urdf, joint_defaults=meta["unobserved_joint_values"])
    if file_digest(model.path) != meta["urdf_sha256"]:
        raise ValueError("URDF content differs from conversion; specify the original --urdf")
    # Catch a changed workpoint or joint mapping before serving an inconsistent robot.
    tcp_meta = json.loads((args.input / f"TCP/{CAMERA}/metadata.json").read_text(encoding="utf-8"))
    qpos = np.load(args.input / "observations/joint_position.npy", allow_pickle=False)
    extrinsic = np.load(args.input / f"extrinsic/{CAMERA}.npy", allow_pickle=False)
    poses = np.load(args.input / f"TCP/{CAMERA}/pose.npy", allow_pickle=False)
    if not np.allclose(extrinsic @ model.tcp_poses(qpos, tcp_meta["tcp_offsets_in_link6_m"]), poses, atol=2e-6):
        raise ValueError("TCP poses disagree with measured URDF forward kinematics")
    if args.depth_source == "corrected":
        validate_corrected_depths(args.input, urdf=model.path)
    count = meta["frame_count"]
    indices = np.arange(args.start_frame, count, args.stride, dtype=np.int64)
    if args.max_frames:
        indices = indices[:args.max_frames]
    if not len(indices):
        raise ValueError(f"No selected frames; episode has {count} frames")
    output = args.output.expanduser().resolve() if args.output is not None else None
    if output is not None and output.exists():
        raise FileExistsError(f"Recording already exists: {output}")
    print(f"Validated {report['frames']} frames; depth={args.depth_source}; "
          f"replaying {len(indices)}; preparing URDF assets…", flush=True)
    recording = rr.RecordingStream("cobot_magic_dual_arm")
    try:
        if output is None:
            start_web(recording, args)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            recording.save(output)
        recording.send_blueprint(make_blueprint(meta["fps"]))
        with tempfile.TemporaryDirectory(prefix="cobot_magic_rerun_") as temporary:
            tree = prepare_robot(model, Path(temporary))
            log_scene(recording, args, meta, indices, model, tree)
        if output is not None:
            print(f"Saved: {output}", flush=True)
        else:
            print("回放已就绪；打开上方链接，Ctrl+C 停止服务。", flush=True)
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
