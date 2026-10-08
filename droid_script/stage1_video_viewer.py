"""Interactive video-frame/TCP selection for uncalibrated DROID Stage 1 inference."""
from __future__ import annotations

import html
import threading
from urllib.parse import urlsplit

import torch

from droid_script import infer_4rc_stage1_video as inference
from droid_script.stage1_viewer import overlay, read_rgb


def build_interactive(args, episode, device, dtype):
    try:
        import gradio as gr
        import viser  # Check before loading the checkpoint.
    except ImportError as error:
        raise ImportError('Interactive mode requires: python -m pip install "gradio==6.12.0" viser') from error
    episode.clip(args.start_frame, args.max_frames)
    runtime = {"model": None, "viewer": None}
    lock = threading.RLock()

    def clear(frame):
        frame = int(frame)
        episode.clip(frame)
        return (read_rgb(episode.image_paths[frame]), [], None,
                f"起始帧 {frame}（{frame / episode.fps:.3f} s），请点击一个 TCP。",
                gr.update(interactive=False))

    def select(frame, event: gr.SelectData):
        frame = int(frame)
        episode.clip(frame)
        point = inference.stage1.validate_query(event.index, "interactive click").tolist()
        return (overlay(read_rgb(episode.image_paths[frame]), point[0]), point, frame,
                f"已选择帧 {frame} 的 TCP：{point[0]}（320×180 像素）。", gr.update(interactive=True))

    select.__annotations__["event"] = gr.SelectData

    def run(frame, points, selection_frame, request: gr.Request, progress=gr.Progress()):
        with lock:
            try:
                frame = int(frame)
                if selection_frame != frame:
                    raise ValueError("起始帧已改变，请重新选择 TCP。")
                query = inference.stage1.validate_query(points)
                episode.clip(frame, args.max_frames)
                progress(0, desc="加载 DROID Stage 1 checkpoint")
                if runtime["model"] is None:
                    runtime["model"] = inference.stage1.load_model(args.model, device)
                result, prediction, paths, saved = inference.run_and_save(
                    args, episode, runtime["model"], frame, query, "interactive click", device, dtype,
                    keep_geometry=True, progress=lambda fraction, message: progress(0.1 + 0.85 * fraction, desc=message))
                if runtime["viewer"] is not None:
                    runtime["viewer"].stop()
                    runtime["viewer"] = None
                runtime["viewer"] = inference.start_viewer(args, episode, prediction, paths)
                port = runtime["viewer"].server.get_port()
                host = args.host
                if host in ("0.0.0.0", "::"):
                    host = urlsplit(str(request.request.url)).hostname if request is not None else None
                    host = host or "127.0.0.1"
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                url = html.escape(f"http://{host}:{port}", quote=True)
                viewer_html = (f'<p><a href="{url}" target="_blank" rel="noopener">打开 Viser（端口 {port}）</a></p>'
                               f'<iframe src="{url}" title="Video TCP Viser" '
                               'style="width:100%;height:700px;border:0"></iframe>')
                progress(1, desc="完成")
                return (result, str(saved),
                        f"已保存 {len(paths)} 帧（{result['start_frame']}–{result['end_frame']}）：`{saved}`", viewer_html)
            except Exception as error:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise gr.Error(str(error)) from error

    run.__annotations__["request"] = gr.Request

    with gr.Blocks(title="DROID Stage 1 · In-the-wild video") as demo:
        gr.Markdown(f"# DROID Stage 1 视频推理\n视频：`{episode.path.name}` · "
                    f"{len(episode.image_paths)} 帧 · {episode.fps:g} fps\n\n"
                    "选择起始帧，在图像中点击一个 TCP（夹爪工作点），然后运行推理。"
                    "切换起始帧会清除选点。图像已缩放到 320×180。")
        points = gr.State([])
        selection_frame = gr.State(None)
        with gr.Row():
            with gr.Column():
                frame = gr.Slider(minimum=0, maximum=len(episode.image_paths) - 2, step=1,
                                  value=args.start_frame, label="起始帧（原始帧号）")
                picture = gr.Image(value=read_rgb(episode.image_paths[args.start_frame]),
                                   type="numpy", interactive=False, label="点击一个 TCP", buttons=[])
                status = gr.Markdown("请点击一个 TCP。")
                with gr.Row():
                    reset = gr.Button("重置选点")
                    execute = gr.Button("运行推理", variant="primary", interactive=False)
            with gr.Column():
                inference_status = gr.Markdown("尚未开始推理。")
                download = gr.File(label="下载 TCP JSON")
                output = gr.JSON(label="单臂轨迹 JSON")
        viewer = gr.HTML("<p>推理完成后显示 Viser 点云、TCP 位姿和轨迹。</p>")
        selections = [picture, points, selection_frame, status, execute]
        frame.input(clear, inputs=[frame], outputs=selections, queue=False)
        reset.click(clear, inputs=[frame], outputs=selections, queue=False)
        picture.select(select, inputs=[frame], outputs=selections, queue=False)
        execute.click(run, inputs=[frame, points, selection_frame],
                      outputs=[output, download, inference_status, viewer],
                      concurrency_id="droid_video_inference", concurrency_limit=1)

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
        demo.queue(default_concurrency_limit=1).launch(
            server_name=args.ui_host, server_port=args.ui_port, show_error=True)
    finally:
        demo.close()
        close()
