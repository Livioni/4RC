"""Shared, browser-rendered episode video export for the inference viewers."""
from __future__ import annotations

from pathlib import Path
import queue
import tempfile
import threading


def _render(client, **kwargs):
    """Bound waiting even on older Viser versions without a render timeout."""
    result = queue.Queue(maxsize=1)

    def capture():
        try:
            result.put(client.get_render(**kwargs))
        except Exception as error:
            result.put(error)

    threading.Thread(target=capture, daemon=True).start()
    try:
        image = result.get(timeout=60)
    except queue.Empty:
        raise TimeoutError("Browser render timed out; keep the Viser tab connected and visible.") from None
    if isinstance(image, Exception):
        raise image
    return image


class EpisodeVideoExport:
    """Serialize export with viewer updates; retain the original playback state."""

    def __init__(self, server, *, frame_count, frame, fps, controls, render_frame,
                 lock, stop_event, filename):
        self.busy = threading.Event()
        self._click_lock = threading.Lock()
        self.server = server
        self.frame_count, self.frame, self.fps = frame_count, frame, fps
        self.controls = controls
        self.render_frame, self.lock = render_frame, lock
        self.stop_event, self.filename = stop_event, filename
        with server.gui.add_folder("Video export"):
            self.button = server.gui.add_button("Export episode MP4 (current view)")
            self.status = server.gui.add_markdown("Uses current FPS and camera view. Keep this tab open during export.")
        self.button.on_click(self._export)

    def _export(self, event):
        client = event.client
        if client is None or not self._click_lock.acquire(blocking=False):
            return
        self.busy.set()
        self.button.disabled = True
        try:
            with self.lock:
                slot = int(self.frame.value)
                disabled = [control.disabled for control in self.controls]
                try:
                    for control in self.controls:
                        control.disabled = True
                    import cv2

                    camera = client.camera
                    # Preserve viewport aspect and cap large browser windows at 1080p.
                    height = int(getattr(camera, "image_height", 720))
                    width = int(getattr(camera, "image_width", round(height * camera.aspect)))
                    scale = min(1.0, 1920 / max(width, 1), 1080 / max(height, 1))
                    width = max(2, int(width * scale) // 2 * 2)
                    height = max(2, int(height * scale) // 2 * 2)
                    view = dict(wxyz=tuple(camera.wxyz), position=tuple(camera.position),
                                fov=float(camera.fov), width=width, height=height,
                                transport_format="jpeg")
                    fps = float(self.fps.value)
                    with tempfile.TemporaryDirectory(prefix="viser-video-") as directory:
                        path = Path(directory) / "episode.mp4"
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        try:
                            if not writer.isOpened():
                                raise RuntimeError("Cannot open MP4 encoder; check the OpenCV/FFmpeg installation.")
                            for index in range(self.frame_count):
                                if self.stop_event.is_set() or client.client_id not in self.server.get_clients():
                                    raise RuntimeError("Export stopped: viewer closed or browser disconnected.")
                                self.status.content = f"Exporting frame {index + 1}/{self.frame_count}…"
                                self.render_frame(index)
                                self.server.flush()
                                image = _render(client, **view)
                                writer.write(cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR))
                        finally:
                            writer.release()
                        client.send_file_download(self.filename, path.read_bytes())
                    self.status.content = f"Export complete: {self.frame_count} frames at {fps:g} FPS. Use the download notification."
                finally:
                    try:
                        if not self.stop_event.is_set():
                            self.render_frame(slot)
                            self.server.flush()
                    finally:
                        for control, state in zip(self.controls, disabled):
                            control.disabled = state
        except Exception as error:
            self.status.content = f"Video export failed: {error}"
        finally:
            self.busy.clear()
            self.button.disabled = False
            self._click_lock.release()
