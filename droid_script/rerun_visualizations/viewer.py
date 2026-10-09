"""Synchronized prediction/ground-truth Rerun recording and Web serving."""
from __future__ import annotations

from pathlib import Path
import socket
import time
from urllib.parse import quote

import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb
from rerun.blueprint.components import LoopMode, PlayState

from .data import backproject_rgbd, project_pixel, trajectory_segments
from .robot import BASE_FRAME, ROBOT_ENTITY, PRED_ROBOT_ENTITY, TIME_TIMELINE, log_robot


PRED_COLOR = [255, 155, 55]
GT_COLOR = [55, 205, 120]
AXIS_COLORS = [[235, 65, 65], [65, 210, 80], [65, 130, 255]]


def make_blueprint(camera: str, *, predicted_robot: bool = False) -> rrb.Blueprint:
    def spatial(name, contents):
        return rrb.Spatial3DView(
            origin="world", name=name, contents=contents,
            eye_controls=rrb.EyeControls3D(position=[1.35, 1.35, 1.10],
                                         look_target=[0.48, 0.0, 0.15], eye_up=[0, 0, 1]),
        )
    comparison = rrb.Horizontal(
        spatial("Prediction · point cloud + IK URDF + TCP" if predicted_robot else "Prediction · point cloud + TCP",
                ["world/**", "prediction/**", f"{PRED_ROBOT_ENTITY}/**"]),
        spatial("Ground truth · point cloud + URDF + TCP",
                ["world/**", "ground_truth/**", f"{ROBOT_ENTITY}/**"]),
        column_shares=[1, 1],
    )
    lower = rrb.Horizontal(
        rrb.Spatial2DView(origin="rgb", name=f"RGB · {camera} · orange=prediction, green=GT",
                          visual_bounds=rrb.VisualBounds2D(x_range=[0, 320], y_range=[0, 180])),
        rrb.TextDocumentView(origin="info", name="Frame / inference"), column_shares=[2, 1],
    )
    return rrb.Blueprint(
        rrb.Vertical(comparison, lower, row_shares=[3, 1]),
        rrb.TimePanel(timeline=TIME_TIMELINE, expanded=True, fps=15,
                      play_state=PlayState.Paused, loop_mode=LoopMode.All),
        auto_views=False, collapse_panels=True,
    )


def choose_port(preferred: int, *, exclude: tuple[int, ...] = ()) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        if preferred not in exclude:
            try:
                candidate.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
        candidate.bind(("127.0.0.1", 0))
        port = int(candidate.getsockname()[1])
    return choose_port(0, exclude=exclude) if port in exclude else port


def start_web_viewer(recording: rr.RecordingStream, args) -> str:
    grpc_port = choose_port(args.grpc_port)
    web_port = choose_port(args.web_port, exclude=(grpc_port,))
    if (grpc_port, web_port) != (args.grpc_port, args.web_port):
        print(f"Using available ports: Web={web_port}, gRPC={grpc_port}", flush=True)
    uri = recording.serve_grpc(grpc_port=grpc_port, server_memory_limit="75%")
    rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
    url = f"http://localhost:{web_port}/?url={quote(uri, safe=':/')}&renderer={args.renderer}"
    print(f"Rerun Web viewer: {url}", flush=True)
    print("Remote access: run on your own computer:\n"
          f"  ssh -N -L {web_port}:127.0.0.1:{web_port} "
          f"-L {grpc_port}:127.0.0.1:{grpc_port} <user>@<server>", flush=True)
    return url


def log_tcp(recording, prefix: str, position: np.ndarray, rotation: np.ndarray,
            color: list[int], label: str) -> None:
    path = f"{prefix}/tcp"
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        recording.log(path, rr.Clear(recursive=True))
        return
    frame = f"{prefix}_tcp"
    recording.log(path, rr.Transform3D(translation=position, mat3x3=rotation,
                                      parent_frame=BASE_FRAME, child_frame=frame))
    recording.log(f"{path}/axes", rr.CoordinateFrame(frame),
                  rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.05,
                              colors=AXIS_COLORS, radii=0.001))
    recording.log(f"{path}/label", rr.CoordinateFrame(frame),
                  rr.Points3D([[0, 0, 0]], colors=color, radii=0.005,
                              labels=[label], show_labels=True))


def log_rgb_marker(recording, name: str, position, intrinsic, color) -> None:
    pixel = project_pixel(position, intrinsic)
    path = f"rgb/{name}"
    if pixel is None:
        recording.log(path, rr.Clear(recursive=True))
    else:
        recording.log(path, rr.Points2D([pixel], radii=3, colors=color,
                                       labels=[name], show_labels=True))


def filtered_prediction_cloud(cloud, camera_to_base, args):
    points, colors, confidence = cloud["points"], cloud["colors"], cloud["confidence"]
    if not len(points):
        return points, colors
    camera_points = (points - camera_to_base[:3, 3]) @ camera_to_base[:3, :3]
    valid = (np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
             & (camera_points[:, 2] > args.min_depth_m) & (camera_points[:, 2] <= args.max_depth_m))
    if np.any(valid):
        threshold = np.percentile(confidence[valid], args.confidence_percentile)
        valid &= confidence >= threshold
    selected = np.flatnonzero(valid)
    if args.max_points and len(selected) > args.max_points:
        selected = selected[np.linspace(0, len(selected) - 1, args.max_points, dtype=int)]
    return points[selected], colors[selected]


def log_replay(recording, args, episode, paths: list[Path], prediction,
               ground_truth, tree, query: np.ndarray, query_source: str, *,
               predicted_tree=None, robot_states=None) -> None:
    from droid_script.infer_4rc_stage1 import tcp_to_base

    if prediction.clouds is None or prediction.camera_to_base is None:
        raise ValueError("Rerun playback requires geometry and absolute camera predictions")
    tcp = tcp_to_base(prediction.tcp, prediction.camera_to_base)
    recording.log("world", rr.CoordinateFrame(BASE_FRAME), rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    recording.log("world/origin", rr.CoordinateFrame(BASE_FRAME),
                  rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.10,
                                             colors=AXIS_COLORS, radii=0.001), static=True)
    for prefix, color, positions in (
            ("prediction", PRED_COLOR, tcp["position"][:, 0]),
            ("ground_truth", GT_COLOR, ground_truth.tcp_poses[:, :3, 3])):
        recording.log(prefix, rr.CoordinateFrame(BASE_FRAME), rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        strips = trajectory_segments(positions)
        if strips:
            recording.log(f"{prefix}/trajectory", rr.CoordinateFrame(BASE_FRAME),
                          rr.LineStrips3D(strips, colors=color, radii=0.0015), static=True)
        # Named coordinate frames are per entity in Rerun 0.35; they do not
        # inherit from the parent path. Both clouds already contain base XYZ.
        recording.log(f"{prefix}/point_cloud", rr.CoordinateFrame(BASE_FRAME), static=True)
    log_robot(recording, tree, ground_truth)
    if predicted_tree is not None:
        log_robot(recording, predicted_tree, prefix="prediction",
                  frame_indices=robot_states["frame_indices"], joints=robot_states["joints"],
                  gripper_open=robot_states["gripper_open"])
    for slot, (frame, rgb_path, depth_path) in enumerate(zip(ground_truth.frame_indices, paths, ground_truth.depth_paths)):
        recording.set_time("frame", sequence=int(frame))
        recording.set_time(TIME_TIMELINE, duration=float(frame) / 15.0)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(depth_path) as image:
            depth = np.asarray(image)
        gt_points, gt_colors = backproject_rgbd(
            rgb, depth, episode.intrinsics, ground_truth.camera_to_base,
            min_depth=args.min_depth_m, max_depth=args.max_depth_m,
            max_points=args.max_points, seed=int(frame),
        )
        pred_points, pred_colors = filtered_prediction_cloud(
            prediction.clouds[slot], prediction.camera_to_base[slot], args)
        for prefix, points, colors in (("prediction", pred_points, pred_colors),
                                       ("ground_truth", gt_points, gt_colors)):
            recording.log(f"{prefix}/point_cloud", rr.Points3D(points, colors=colors, radii=args.point_size / 2))
        log_tcp(recording, "prediction", tcp["position"][slot, 0], tcp["rotation"][slot, 0],
                PRED_COLOR, f"Pred TCP · open={tcp['gripper'][slot, 0]:.3f}")
        ik_info = ""
        if robot_states is not None:
            succeeded = bool(robot_states["success"][slot])
            actual = robot_states["fk_tcp_poses"][slot]
            error_mm = robot_states["position_error_m"][slot] * 1000
            error_rad = robot_states["rotation_error_rad"][slot]
            color = [90, 175, 255] if succeeded else [255, 65, 65]
            log_tcp(recording, "prediction/ik", actual[:3, 3], actual[:3, :3], color,
                    f"IK {'OK' if succeeded else 'HOLD'} · {error_mm:.1f} mm")
            recording.log("prediction/ik/residual", rr.CoordinateFrame(BASE_FRAME),
                          rr.LineStrips3D([np.stack([actual[:3, 3], tcp['position'][slot, 0]])],
                                          colors=color, radii=0.0015))
            ik_info = (f"**IK: {'OK' if succeeded else 'FAILED — holding last valid arm'}**. "
                       f"TCP residual: **{error_mm:.2f} mm / {error_rad:.4f} rad**. "
                       "Blue=FK TCP; red=held FK TCP. Gripper uses prediction.\n\n")
        gt_pose = ground_truth.tcp_poses[slot]
        log_tcp(recording, "ground_truth", gt_pose[:3, 3], gt_pose[:3, :3],
                GT_COLOR, f"GT TCP · open={ground_truth.tcp_camera[slot, 6]:.3f}")
        recording.log("rgb", rr.Image(rgb))
        log_rgb_marker(recording, "prediction", prediction.tcp["position"][slot, 0], episode.intrinsics, PRED_COLOR)
        log_rgb_marker(recording, "ground_truth", ground_truth.tcp_camera[slot, :3], episode.intrinsics, GT_COLOR)
        if slot == 0:
            recording.log("rgb/initial_query", rr.Points2D(query, colors=[255, 255, 255], radii=5,
                                                           labels=["initial query"], show_labels=True))
        else:
            recording.log("rgb/initial_query", rr.Clear(recursive=True))
        recording.log("info", rr.TextDocument(
            f"## {episode.path.name}\n\n"
            f"Camera: **{episode.camera}** · frame **{int(frame)}** · {frame / 15:.3f} s\n\n"
            f"{ik_info}"
            f"Replay: {int(ground_truth.frame_indices[0])}–{int(ground_truth.frame_indices[-1])} at 15 Hz. "
            f"Skipped prefix: {int(ground_truth.frame_indices[0])} frames.\n\n"
            f"**Orange:** predicted TCP · **Green:** GT TCP. Both scenes use robot-base coordinates (metres).\n\n"
            f"Points: prediction **{len(pred_points):,}**, GT **{len(gt_points):,}**.\n\n"
            f"Predicted camera pose + depth; calibrated GT RGB-D; measured robot joints.\n\n"
            f"TCP confidence: {prediction.tcp['confidence'][slot, 0]:.3f} (raw score).\n\n"
            f"Initial query: {query_source}, pixel {query[0].round(2).tolist()}.\n\n"
            f"Checkpoint: `{args.model}`\n\n"
            "Use the time panel to play/pause or scrub both scenes together.", media_type="text/markdown"))
        if slot == 0 or (slot + 1) % 25 == 0 or slot + 1 == len(paths):
            print(f"Logged replay {slot + 1}/{len(paths)} (original frame {int(frame)})", flush=True)
    recording.flush()


def wait_for_web_viewer() -> None:
    print("Replay ready. Open the URL above; press Ctrl+C to stop the Web viewer.", flush=True)
    while True:
        time.sleep(1)
