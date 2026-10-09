"""Optional Gradio and Viser frontends for DROID Stage 1 inference."""
from __future__ import annotations

import html
from pathlib import Path
import threading
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw
import torch

from droid_script import infer_4rc_stage1 as inference


PRED_COLOR = (255, 90, 40)
GT_COLOR = (50, 220, 100)


def load_gt_trajectory(episode, paths) -> np.ndarray:
    """Read selected camera-frame labels and undo the dataset's world-to-camera pose."""
    state_path = episode.path / "TCP" / episode.camera / "state.npy"
    state = np.load(state_path, allow_pickle=False, mmap_mode="r")
    if state.shape != (len(episode.image_paths), 7):
        raise ValueError(f"Expected TCP state [{len(episode.image_paths)},7] in {state_path}, got {state.shape}")
    ext_path = episode.path / "extrinsic" / f"{episode.camera}.npy"
    w2c = np.load(ext_path, allow_pickle=False).astype(np.float32)
    if (w2c.shape != (4, 4) or not np.isfinite(w2c).all()
            or not np.allclose(w2c[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(w2c[:3, :3].T @ w2c[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(w2c[:3, :3]), 1, atol=1e-4)):
        raise ValueError(f"Expected rigid world-to-camera matrix [4,4]: {ext_path}")
    positions = np.array(state[[inference._frame_index(path) for path in paths], :3], dtype=np.float32)
    # Keep gaps in place, and retain finite labels even when outside the camera view.
    positions[~np.isfinite(positions).all(-1)] = np.nan
    return (positions - w2c[:3, 3]) @ w2c[:3, :3]


def add_trajectory(scene, name, positions, color):
    """Draw the full selected clip without bridging invalid GT frames."""
    positions = np.asarray(positions, dtype=np.float32)
    valid = np.isfinite(positions).all(-1)
    nodes = []
    if valid.any():
        nodes.append(scene.add_point_cloud(f"{name}/points", points=positions[valid],
                                           colors=color, point_size=0.004))
    segments = np.stack((positions[:-1], positions[1:]), axis=1)[valid[:-1] & valid[1:]]
    if len(segments):
        nodes.append(scene.add_line_segments(f"{name}/lines", points=segments, colors=color, line_width=3))
    return nodes


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (320, 180):
            raise ValueError(f"Expected native 320x180 RGB: {path}")
        return np.array(image.convert("RGB"))


def overlay(image: np.ndarray, point: list | None) -> np.ndarray:
    canvas = Image.fromarray(image)
    if point:
        x, y = inference.validate_query(point)[0]
        draw = ImageDraw.Draw(canvas)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 90, 40), width=2)
        draw.text((max(0, min(x + 7, 288)), max(0, y - 12)), "TCP", fill=(255, 90, 40))
    return np.array(canvas)


class Viewer:
    def __init__(self, server, stopped, worker):
        self.server, self.stopped, self.worker = server, stopped, worker

    def stop(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.worker.join(timeout=2)
        self.server.stop()


def start_viewer(args, episode, prediction, paths, *, frame_rate=None,
                 scene_frame="robot base coordinates", load_ground_truth=True) -> Viewer:
    try:
        import viser
        import viser.transforms as tf
    except ImportError as error:
        raise ImportError("Viser is required: python -m pip install viser") from error
    if prediction.clouds is None or prediction.camera_to_base is None:
        raise ValueError("Viewer requires predicted depth and camera geometry")
    frame_rate = episode.frame_rate if frame_rate is None else frame_rate
    tcp = inference.tcp_to_base(prediction.tcp, prediction.camera_to_base)
    gt_positions, gt_error = None, ""
    try:
        if not load_ground_truth:
            raise ValueError("Input video has no calibrated GT trajectory")
        gt_positions = load_gt_trajectory(episode, paths)
        if not np.isfinite(gt_positions).all(-1).any():
            raise ValueError("No finite GT TCP positions in the selected clip")
    except (OSError, ValueError) as error:
        gt_positions, gt_error = None, str(error)
    server = viser.ViserServer(host=args.host, port=args.port)
    stopped = threading.Event()
    lock = threading.RLock()
    server.gui.set_panel_label(f"{episode.dataset.upper()} · {episode.camera}")
    server.scene.set_up_direction("+z")
    with server.gui.add_folder("Playback"):
        frame = server.gui.add_slider("Frame slot", min=0, max=len(paths) - 1, step=1, initial_value=0)
        previous = server.gui.add_button("Previous")
        following = server.gui.add_button("Next")
        playing = server.gui.add_checkbox("Play", initial_value=False)
        fps = server.gui.add_slider("FPS", min=0.1, max=max(60, frame_rate), step=0.1, initial_value=frame_rate)
    with server.gui.add_folder(f"Geometry ({scene_frame})"):
        percentile = server.gui.add_slider("Confidence percentile", min=0, max=99, step=0.5,
                                          initial_value=args.confidence_percentile)
        point_size = server.gui.add_slider("Point size", min=0.00001, max=max(0.02, args.point_size),
                                          step=0.00001, initial_value=args.point_size)
        show_cloud = server.gui.add_checkbox("Show predicted point cloud", initial_value=True)
        show_tcp = server.gui.add_checkbox("Show predicted TCP pose", initial_value=True)
    with server.gui.add_folder(f"TCP trajectories ({scene_frame})"):
        show_pred_trajectory = server.gui.add_checkbox("Show predicted trajectory (orange)",
                                                       initial_value=args.show_pred_trajectory)
        show_gt_trajectory = server.gui.add_checkbox("Show GT trajectory (green)",
                                                     initial_value=args.show_gt_trajectory and gt_positions is not None,
                                                     disabled=gt_positions is None)
        server.gui.add_markdown("Full selected clip. Orange: prediction; green: GT.\n\n"
                                "Prediction uses predicted camera poses; GT uses dataset calibration."
                                if load_ground_truth else
                                "Orange: predicted TCP trajectory. Scene uses model-predicted coordinates; "
                                "no real robot-base calibration is available for this video.")
        if gt_error:
            server.gui.add_markdown(f"GT trajectory unavailable: {gt_error}")
    rgb = server.gui.add_image(read_rgb(paths[0]), label="RGB")
    info = server.gui.add_markdown("")
    cloud_node = server.scene.add_point_cloud("/geometry", points=np.zeros((1, 3), dtype=np.float32),
                                              colors=np.zeros((1, 3), dtype=np.uint8), point_size=args.point_size)
    tcp_node = server.scene.add_frame("/tcp", axes_length=0.05, axes_radius=0.001,
                                      origin_radius=0.004, origin_color=PRED_COLOR)
    label = server.scene.add_label("/tcp/label", "Pred TCP", position=(0, 0, 0.065))
    pred_trajectory = add_trajectory(server.scene, "/trajectories/pred", tcp["position"][:, 0], PRED_COLOR)
    gt_trajectory = []
    gt_node = gt_label = None
    if gt_positions is not None:
        gt_trajectory = add_trajectory(server.scene, "/trajectories/gt", gt_positions, GT_COLOR)
        gt_node = server.scene.add_icosphere("/tcp_gt", radius=0.006, color=GT_COLOR)
        gt_label = server.scene.add_label("/tcp_gt/label", "GT TCP", position=(0, 0, 0.02))
    first_cloud = prediction.clouds[0]["points"]
    center = np.median(first_cloud, axis=0) if len(first_cloud) else tcp["position"][0, 0]
    extent = (max(float(np.max(np.percentile(first_cloud, 95, axis=0) - np.percentile(first_cloud, 5, axis=0))), 0.3)
              if len(first_cloud) else 0.5)
    camera_rotation = prediction.camera_to_base[0, :3, :3]

    @server.on_client_connect
    def connect(client):
        client.camera.position = tuple(center - camera_rotation[:, 2] * extent * 1.5)
        client.camera.look_at = tuple(center)
        client.camera.up_direction = tuple(-camera_rotation[:, 1])

    def render():
        if stopped.is_set():
            return
        with lock, server.atomic():
            slot = int(frame.value)
            cloud = prediction.clouds[slot]
            confidence = cloud["confidence"]
            mask = confidence >= np.percentile(confidence, percentile.value) if len(confidence) else np.zeros(0, dtype=bool)
            cloud_node.points = cloud["points"][mask]
            cloud_node.colors = cloud["colors"][mask]
            cloud_node.point_size = point_size.value
            cloud_node.visible = show_cloud.value
            tcp_node.position = tcp["position"][slot, 0]
            tcp_node.wxyz = tf.SO3.from_matrix(tcp["rotation"][slot, 0]).wxyz
            tcp_node.visible = show_tcp.value
            label.text = f"Pred TCP · opening={tcp['gripper'][slot, 0]:.3f}"
            label.visible = show_tcp.value
            for node in pred_trajectory:
                node.visible = show_pred_trajectory.value
            for node in gt_trajectory:
                node.visible = show_gt_trajectory.value
            if gt_node is not None:
                valid_gt = bool(np.isfinite(gt_positions[slot]).all())
                if valid_gt:
                    gt_node.position = gt_positions[slot]
                gt_node.visible = gt_label.visible = show_gt_trajectory.value and valid_gt
            rgb.image = read_rgb(paths[slot])
            source_frame = inference._frame_index(paths[slot])
            xyz = prediction.tcp["position"][slot, 0]
            info.content = (f"**{episode.path.name} / {episode.camera}**\n\n"
                            f"Frame **{source_frame}** · {episode.time_seconds(source_frame):.3f} s · {slot + 1}/{len(paths)}\n\n"
                            f"Pred TCP camera XYZ (m): `{np.round(xyz, 4).tolist()}`\n\n"
                            f"Gripper opening: **{tcp['gripper'][slot, 0]:.4f}** · "
                            f"confidence: **{tcp['confidence'][slot, 0]:.3f}**\n\n"
                            f"Visible predicted cloud points: {int(mask.sum())}. Scene: {scene_frame}.")

    for control in (frame, percentile, point_size, show_cloud, show_tcp, show_pred_trajectory, show_gt_trajectory):
        control.on_update(lambda _: render())
    previous.on_click(lambda _: setattr(frame, "value", (int(frame.value) - 1) % len(paths)))
    following.on_click(lambda _: setattr(frame, "value", (int(frame.value) + 1) % len(paths)))

    def playback():
        while not stopped.wait(1.0 / max(0.1, float(fps.value))):
            if playing.value:
                frame.value = (int(frame.value) + 1) % len(paths)

    render()
    worker = threading.Thread(target=playback, name="droid-stage1-playback", daemon=True)
    worker.start()
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Viser: http://{host}:{server.get_port()}", flush=True)
    return Viewer(server, stopped, worker)


def selection_for_frame(episode, frame: int, uv, valid):
    """Frame changes always discard the old click; GT is optional."""
    episode.clip(frame)
    can_use_gt = uv is not None and bool(valid[frame])
    return read_rgb(episode.image_paths[frame]), [], can_use_gt


def build_interactive(args, episode, device, dtype):
    """Build without launching so startup and selection can be tested independently."""
    try:
        import gradio as gr
        import viser  # Fail before a long inference if visualization is unavailable.
    except ImportError as error:
        raise ImportError('Interactive mode requires: python -m pip install "gradio==6.12.0" viser') from error
    cameras = sorted(p.name for p in (episode.path / "images").iterdir()
                     if p.is_dir() and not p.name.startswith("."))
    contexts = {}
    runtime = {"model": None, "viewer": None}
    lock = threading.RLock()

    def camera_context(camera):
        with lock:
            if camera not in contexts:
                selected_episode = episode if camera == episode.camera else inference.load_episode(episode.path, camera)
                uv, valid, gt_error = None, None, ""
                try:
                    uv, valid = inference.ground_truth_queries(selected_episode)
                except (OSError, ValueError) as error:
                    gt_error = str(error)
                contexts[camera] = selected_episode, uv, valid, gt_error
            return contexts[camera]

    def initial_frame(selected_episode, valid):
        if args.start_frame is not None:
            frame = args.start_frame
        elif valid is not None and valid[:-1].any():
            frame = int(np.flatnonzero(valid[:-1])[0])
        else:
            frame = 0
        selected_episode.clip(frame)
        return frame

    _, uv, valid, gt_error = camera_context(episode.camera)
    initial = initial_frame(episode, valid)
    base, _, gt_available = selection_for_frame(episode, initial, uv, valid)

    def clear(camera, frame):
        selected_episode, uv, valid, _ = camera_context(camera)
        frame = int(frame)
        image, points, gt_available = selection_for_frame(selected_episode, frame, uv, valid)
        status = f"相机 {camera} · 起始帧 {frame}；跳过前 {frame} 帧。请点击一个 TCP。"
        if not gt_available:
            status += " 当前帧无有效 GT，请手动选点。"
        return image, points, "interactive click", None, status, gr.update(interactive=False), gr.update(interactive=gt_available)

    def switch_camera(camera):
        with lock:
            selected_episode, _, valid, gt_error = camera_context(camera)
            frame = initial_frame(selected_episode, valid)
            selection = clear(camera, frame)
            if runtime["viewer"] is not None:
                runtime["viewer"].stop()
                runtime["viewer"] = None
            return (gr.update(value=frame, maximum=len(selected_episode.image_paths) - 2), *selection,
                    f"GT 选点不可用：{gt_error}" if gt_error else "",
                    None, None, f"当前相机 {camera}，尚未开始推理。",
                    "<p>推理后显示当前相机的 Viser 点云与 TCP。</p>")

    def select(camera, frame, event: gr.SelectData):
        selected_episode, _, _, _ = camera_context(camera)
        frame = int(frame)
        selected_episode.clip(frame)
        point = inference.validate_query(event.index, "interactive click").tolist()
        return (overlay(read_rgb(selected_episode.image_paths[frame]), point[0]), point,
                "interactive click", [camera, frame], "已选中 TCP，可开始推理。", gr.update(interactive=True))

    select.__annotations__["event"] = gr.SelectData

    def use_gt(camera, frame):
        selected_episode, uv, valid, _ = camera_context(camera)
        frame = int(frame)
        selected_episode.clip(frame)
        if uv is None or not valid[frame]:
            raise gr.Error("当前相机和帧没有有效 GT TCP，请手动选点。")
        point = inference.validate_query(uv[frame]).tolist()
        return (overlay(read_rgb(selected_episode.image_paths[frame]), point[0]), point,
                "projected GT TCP", [camera, frame], "已使用当前帧 GT TCP。", gr.update(interactive=True))

    def run(camera, frame, points, source, selection_context, request: gr.Request, progress=gr.Progress()):
        with lock:
            try:
                frame = int(frame)
                if selection_context != [camera, frame]:
                    raise ValueError("相机或起始帧已改变，请重新选择 TCP。")
                selected_episode, _, _, _ = camera_context(camera)
                query = inference.validate_query(points)
                selected_episode.clip(frame, args.max_frames)
                progress(0, desc="加载 DROID Stage 1 checkpoint")
                if runtime["model"] is None:
                    runtime["model"] = inference.load_model(args.model, device)
                result, prediction, paths, saved = inference.run_and_save(
                    args, selected_episode, runtime["model"], frame, query, source, device, dtype,
                    keep_geometry=True, progress=lambda fraction, text: progress(0.1 + 0.85 * fraction, desc=text))
                if runtime["viewer"] is not None:
                    runtime["viewer"].stop()
                    runtime["viewer"] = None
                runtime["viewer"] = start_viewer(args, selected_episode, prediction, paths)
                port = runtime["viewer"].server.get_port()
                host = args.host
                if host in ("0.0.0.0", "::"):
                    host = urlsplit(str(request.request.url)).hostname or "127.0.0.1"
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                url = html.escape(f"http://{host}:{port}", quote=True)
                viewer_html = (f'<p><a href="{url}" target="_blank" rel="noopener">Viser（端口 {port}）</a></p>'
                               f'<iframe src="{url}" title="DROID Viser" '
                               'style="width:100%;height:700px;border:0"></iframe>')
                progress(1, desc="完成")
                return (result, str(saved), f"相机 {camera}：已保存 {len(paths)} 帧（{result['start_frame']}–{result['end_frame']}）；"
                        f"跳过前 {frame} 帧。输出：`{saved}`", viewer_html)
            except Exception as error:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise gr.Error(str(error)) from error

    run.__annotations__["request"] = gr.Request

    with gr.Blocks(title=f"{episode.dataset.upper()} Stage 1 TCP Inference") as demo:
        gr.Markdown(f"# {episode.dataset.upper()} Stage 1 单臂推理\nEpisode：`{episode.path.name}`\n\n"
                    "选择相机和起始帧，点击一个 TCP 或使用 GT，然后推理到结尾。切换相机或起始帧会清除选点。")
        camera = gr.Dropdown(choices=cameras, value=episode.camera, label="相机", interactive=True)
        gt_note = gr.Markdown(f"GT 选点不可用：{gt_error}" if gt_error else "")
        points = gr.State([])
        source = gr.State("interactive click")
        selection_context = gr.State(None)
        with gr.Row():
            with gr.Column():
                start = gr.Slider(minimum=0, maximum=len(episode.image_paths) - 2, step=1,
                                  value=initial, label="起始帧（原始帧号）")
                picture = gr.Image(value=base, type="numpy", interactive=False, label="点击一个 TCP", buttons=[])
                status = gr.Markdown(f"起始帧 {initial}；跳过前 {initial} 帧。请点击一个 TCP。")
                with gr.Row():
                    gt = gr.Button("使用当前帧 GT TCP", interactive=gt_available)
                    reset = gr.Button("重置选点")
                    execute = gr.Button("运行推理", variant="primary", interactive=False)
            with gr.Column():
                inference_status = gr.Markdown("尚未开始推理。")
                download = gr.File(label="下载 TCP JSON")
                output = gr.JSON(label="单臂轨迹 JSON")
        viewer = gr.HTML("<p>推理后显示 Viser 点云与 TCP。</p>")
        selection_outputs = [picture, points, source, selection_context, status, execute]
        camera.change(switch_camera, inputs=[camera],
                      outputs=[start, *selection_outputs, gt, gt_note, output, download, inference_status, viewer],
                      concurrency_id="droid_inference", concurrency_limit=1)
        start.input(clear, inputs=[camera, start], outputs=selection_outputs + [gt], queue=False)
        reset.click(clear, inputs=[camera, start], outputs=selection_outputs + [gt], queue=False)
        picture.select(select, inputs=[camera, start], outputs=selection_outputs, queue=False)
        gt.click(use_gt, inputs=[camera, start], outputs=selection_outputs, queue=False)
        execute.click(run, inputs=[camera, start, points, source, selection_context],
                      outputs=[output, download, inference_status, viewer],
                      concurrency_id="droid_inference", concurrency_limit=1)

    def close():
        with lock:
            if runtime["viewer"] is not None:
                runtime["viewer"].stop()
                runtime["viewer"] = None
            runtime["model"] = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return demo, close


def start_interactive(args, episode, device, dtype):
    demo, close = build_interactive(args, episode, device, dtype)
    try:
        demo.queue(default_concurrency_limit=1).launch(server_name=args.ui_host, server_port=args.ui_port, show_error=True)
    finally:
        demo.close()
        close()
