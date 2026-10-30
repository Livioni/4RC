#!/usr/bin/env python3
"""Convert real-robot HDF5 episodes to one LeRobot v2.1 dataset, then verify it.

Convert a file or recursively convert every .hdf5/.h5 file in a directory:
    python cobot_magic/convert_real_hdf5_to_lerobot_v21.py \
        --input datasets/4_cams/test_new --output datasets/test_new_lerobot_v21

Files are naturally sorted and stored as separate episodes in one dataset.
Use --num-episodes N to convert only the first N files (default: all files).
Images keep each camera's source resolution unless --width/--height is set.
All episodes must have matching cameras, features and FPS. Depth is optional;
RGB-only recordings do not need depth or calibration attributes. RGB values
are preserved as supplied; inputs without rgb_semantics are assumed RGB.

Dependencies: numpy, h5py, opencv-python, datasets, pyarrow, av, and ffmpeg.
The converter does not import LeRobot, so an installed v2.0/v3.0 package
cannot silently change the requested v2.1 output format.

Repeat the source/output checks without converting again:
    python cobot_magic/convert_real_hdf5_to_lerobot_v21.py \
        --input datasets/4_cams/test_new --output datasets/test_new_lerobot_v21 \
        --validate-only

For an additional official-reader check, use an environment with
lerobot==0.3.3 (CODEBASE_VERSION=v2.1) and add --check-lerobot-reader.
For datasets==3.0.1, use pyarrow==17.0.0; pyarrow 25 is incompatible with
that old datasets version when reading Array2D through the Python formatter.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import av
import cv2
import datasets
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000
NUMERIC_SOURCES = {
    "observation.state": "observations/qpos",
    "action": "action",
    "observation.velocity": "observations/qvel",
    "observation.effort": "observations/effort",
    "action.base": "base_action",
}
FORMAT_REFERENCE = (
    "https://github.com/huggingface/lerobot/blob/v0.3.3/"
    "src/lerobot/datasets/lerobot_dataset.py"
)


def write_json(path: Path, value, *, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                  indent=None if compact else 2,
                  separators=(",", ":") if compact else None)
        stream.write("\n")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def discover_sources(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in (".hdf5", ".h5"):
        return [path]
    if path.is_dir():
        files = [p for p in path.rglob("*")
                 if p.is_file() and p.suffix.lower() in (".hdf5", ".h5")]
        files.sort(key=lambda p: [(1, int(part)) if part.isdigit() else (0, part.lower())
                                 for part in re.split(r"(\d+)", p.relative_to(path).as_posix())])
        if files:
            return files
    raise ValueError(f"No HDF5 episodes found: {path}")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")


def read_jsonl(path):
    return list(iter_jsonl(path))


def iter_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def inspect_source(source: h5py.File):
    for key in ("observations/qpos", "action", "observations/images"):
        if key not in source:
            raise ValueError(f"Missing required HDF5 entry: {key}")
    count = len(source["action"])
    if count < 1:
        raise ValueError("Cannot convert an empty episode")
    cameras = sorted(source["observations/images"])
    if not cameras:
        raise ValueError("At least one RGB camera is required")
    depth_cameras = set(source.get("observations/images_depth", {}))
    if depth_cameras - set(cameras):
        raise ValueError("Depth camera names must have matching RGB cameras")
    entries = {}

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            if obj.ndim == 0 or len(obj) != count:
                raise ValueError(f"Frame count mismatch: {name}: {obj.shape}")
            entries[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

    source.visititems(collect)
    known = set(NUMERIC_SOURCES.values())
    for camera in cameras:
        rgb = source[f"observations/images/{camera}"]
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected raw uint8 RGB [N,H,W,3]: {camera}")
        if camera in depth_cameras:
            depth = source[f"observations/images_depth/{camera}"]
            if depth.shape != rgb.shape[:3] or depth.dtype != np.uint16:
                raise ValueError(f"Expected aligned uint16 depth [N,H,W]: {camera}")
        timestamp = source.get(f"observations/camera_timestamps/{camera}")
        if timestamp is not None and (timestamp.ndim != 1 or timestamp.dtype != np.float64):
            raise ValueError(f"Expected float64 camera timestamps [N]: {camera}")
        for prefix in ("images", "images_depth", "camera_timestamps"):
            known.add(f"observations/{prefix}/{camera}")
    unknown = set(entries) - known
    if unknown:
        raise ValueError(f"Unmapped datasets; add an explicit mapping first: {sorted(unknown)}")
    attrs = {key: json_value(value) for key, value in source.attrs.items()}
    # Avoid interpreting BGR or unaligned/non-metric depth as calibrated RGB-D.
    if attrs.get("rgb_semantics", "rgb8") not in ("rgb8", "rgb8_rectified"):
        raise ValueError("Expected source rgb_semantics=rgb8 or rgb8_rectified")
    if depth_cameras and attrs.get("rgb_semantics") != "rgb8_rectified":
        raise ValueError("Expected source rgb_semantics=rgb8_rectified for RGB-D input")
    if depth_cameras and attrs.get("depth_unit") != "mm":
        raise ValueError("Expected source depth_unit=mm")
    if depth_cameras and attrs.get("depth_alignment") != "factory_calibrated_d2c_to_rectified_color":
        raise ValueError("Expected source depth aligned to rectified color")
    return count, cameras, entries, attrs


def output_camera_sizes(source, cameras, width=None, height=None):
    sizes = {}
    for camera in cameras:
        source_height, source_width = source[f"observations/images/{camera}"].shape[1:3]
        camera_width = source_width if width is None else width
        camera_height = source_height if height is None else height
        if min(camera_width, camera_height) < 2 or camera_width % 2 or camera_height % 2:
            raise ValueError(f"{camera}: H.264 yuv420p requires positive even width and height; "
                             "provide --width/--height to override the source dimensions")
        sizes[camera] = (camera_width, camera_height)
    return sizes


def build_features(source, camera_sizes):
    features = {}
    for key, path in NUMERIC_SOURCES.items():
        if path in source:
            obj = source[path]
            if obj.ndim != 2 or obj.dtype != np.float32:
                raise ValueError(f"Expected float32 [N,D]: {path}")
            features[key] = {"dtype": "float32", "shape": [obj.shape[1]],
                             "names": [f"component_{i}" for i in range(obj.shape[1])]}
    for camera, (width, height) in camera_sizes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
        }
        if f"observations/images_depth/{camera}" in source:
            features[f"observation.depth.{camera}"] = {
                "dtype": "uint16", "shape": [height, width],
                "names": ["height", "width"],
            }
        if f"observations/camera_timestamps/{camera}" in source:
            features[f"observation.camera_timestamps.{camera}"] = {
                "dtype": "float64", "shape": [1], "names": None,
            }
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[key] = {"dtype": "float32" if key == "timestamp" else "int64",
                         "shape": [1], "names": None}
    hf_features = {}
    for key, feature in features.items():
        dtype, shape = feature["dtype"], feature["shape"]
        if dtype == "video":
            continue
        if shape == [1]:
            hf_features[key] = datasets.Value(dtype)
        elif len(shape) == 1:
            hf_features[key] = datasets.Sequence(datasets.Value(dtype), length=shape[0])
        else:
            hf_features[key] = datasets.Array2D(shape=tuple(shape), dtype=dtype)
    return features, datasets.Features(hf_features)


class RunningStats:
    """Merge population moments in float64, including per-pixel depth stats."""

    def __init__(self):
        self.samples = 0
        self.frames = 0

    def update(self, values, *, image=False):
        self.frames += len(values)
        if image:
            values = values.reshape(-1, 3).astype(np.float64) / 255.0
        else:
            values = values.astype(np.float64)
            if values.ndim == 1:
                values = values[:, None]
        n = len(values)
        mean = values.mean(axis=0)
        m2 = np.square(values - mean).sum(axis=0)
        lower, upper = values.min(axis=0), values.max(axis=0)
        if self.samples == 0:
            self.mean, self.m2 = mean, m2
            self.lower, self.upper = lower, upper
        else:
            delta = mean - self.mean
            total = self.samples + n
            self.mean += delta * n / total
            self.m2 += m2 + np.square(delta) * self.samples * n / total
            self.lower = np.minimum(self.lower, lower)
            self.upper = np.maximum(self.upper, upper)
        self.samples += n

    def export(self, *, image=False):
        result = {"min": self.lower, "max": self.upper, "mean": self.mean,
                  "std": np.sqrt(np.maximum(self.m2 / self.samples, 0.0))}
        if image:
            result = {key: value.reshape(3, 1, 1) for key, value in result.items()}
        return {**{key: value.tolist() for key, value in result.items()},
                "count": [self.frames]}


class VideoWriter:
    def __init__(self, path, width, height, fps, crf):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "2",
            "-g", "12", "-movflags", "+faststart", "-y", str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)

    def __enter__(self):
        return self

    def write(self, rgb):
        self.process.stdin.write(rgb.tobytes())

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is not None:
                self.process.terminate()
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait(timeout=60)
            if code and exc_type is None:
                self.log.seek(0)
                raise RuntimeError(self.log.read().decode("utf-8", errors="replace"))
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


def arrow_column(values, dtype):
    if isinstance(dtype, pa.ExtensionType):
        # Construct Array2D storage without expanding millions of pixels to Python ints.
        n, height, width = values.shape
        rows = pa.ListArray.from_arrays(
            pa.array(np.arange(n * height + 1, dtype=np.int32) * width),
            pa.array(values.reshape(-1)),
        )
        frames = pa.ListArray.from_arrays(
            pa.array(np.arange(n + 1, dtype=np.int32) * height), rows,
        )
        return pa.ExtensionArray.from_storage(dtype, frames)
    return pa.array(values.tolist(), type=dtype)


def calibration_metadata(source, camera_sizes):
    original = json.loads(source.attrs.get("d2c_calibration_json", "{}"))
    result = {}
    for camera, (width, height) in camera_sizes.items():
        if camera not in original:
            continue
        src_h, src_w = source[f"observations/images/{camera}"].shape[1:3]
        sx, sy = width / src_w, height / src_h
        fx, fy, cx, cy = original[camera]["color_intrinsics"]
        # OpenCV resize maps output pixel centers to source pixel centers.
        resize = np.array([[sx, 0, (sx - 1) / 2],
                           [0, sy, (sy - 1) / 2], [0, 0, 1]])
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        result[camera] = {
            "source_size_wh": [src_w, src_h], "output_size_wh": [width, height],
            "source_pixel_to_output_pixel": resize.tolist(),
            "rectified_color_intrinsics_3x3": (resize @ intrinsic).tolist(),
            "source_factory_calibration": original[camera],
            "output_distortion": [0.0] * 5,
            "note": "Factory distortion is retained for provenance only. "
                    "Robot/world camera extrinsics are not present in the source.",
        }
        if f"observations/images_depth/{camera}" in source:
            result[camera].update({
                "rectified_depth_intrinsics_3x3": (resize @ intrinsic).tolist(),
                "depth_unit": "mm", "depth_zero_is_invalid": True,
                "depth_semantics": str(source.attrs.get("depth_semantics", "")),
                "depth_note": "Depth is already aligned to rectified color; do not apply D2C again.",
            })
    return result


def batch_values(source, start, end, camera_sizes, fps, *,
                 episode_index=0, index_offset=0):
    values = {key: source[path][start:end] for key, path in NUMERIC_SOURCES.items()
              if path in source}
    indices = np.arange(start, end, dtype=np.int64)
    values.update(timestamp=(indices / fps).astype(np.float32),
                  frame_index=indices, index=indices + index_offset,
                  episode_index=np.full(end - start, episode_index, dtype=np.int64),
                  task_index=np.zeros(end - start, dtype=np.int64))
    rgb_values = {}
    for camera, (width, height) in camera_sizes.items():
        rgb = source[f"observations/images/{camera}"][start:end]
        if rgb.shape[1:3] != (height, width):
            rgb = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                            for frame in rgb])
        rgb_values[camera] = rgb
        if f"observations/images_depth/{camera}" in source:
            depth = source[f"observations/images_depth/{camera}"][start:end]
            if depth.shape[1:3] != (height, width):
                depth = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
                                  for frame in depth])
            values[f"observation.depth.{camera}"] = depth
        path = f"observations/camera_timestamps/{camera}"
        if path in source:
            values[f"observation.camera_timestamps.{camera}"] = source[path][start:end]
    return values, rgb_values


def convert(source, output, args, count, camera_sizes, features, hf_features, *,
            episode_index=0, index_offset=0):
    statistics = {key: RunningStats() for key in features}
    episode_chunk = episode_index // CHUNKS_SIZE
    parquet_path = output / DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        writer = stack.enter_context(pq.ParquetWriter(
            parquet_path, hf_features.arrow_schema, compression="zstd"))
        videos = {
            camera: stack.enter_context(VideoWriter(
                output / VIDEO_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index,
                                           video_key=f"observation.images.{camera}"),
                width, height, args.fps, args.crf))
            for camera, (width, height) in camera_sizes.items()
        }
        for start in range(0, count, args.batch_size):
            end = min(start + args.batch_size, count)
            values, images = batch_values(source, start, end, camera_sizes, args.fps,
                                          episode_index=episode_index, index_offset=index_offset)
            for camera, rgb in images.items():
                for frame in rgb:
                    videos[camera].write(frame)
                statistics[f"observation.images.{camera}"].update(rgb, image=True)
            for key, value in values.items():
                statistics[key].update(value)
            table = pa.Table.from_arrays(
                [arrow_column(values[field.name], field.type)
                 for field in hf_features.arrow_schema], schema=hf_features.arrow_schema)
            writer.write_table(table)
            print(f"[convert] episode {episode_index}: {end}/{count} frames", flush=True)
    return {key: stat.export(image=features[key]["dtype"] == "video")
            for key, stat in statistics.items()}


def validate_episode(source_path, output, info, episode, index_offset, batch_size):
    episode_index, count, fps = episode["episode_index"], episode["length"], info["fps"]
    episode_chunk = episode_index // info["chunks_size"]
    cameras = [key.removeprefix("observation.images.") for key, value
               in info["features"].items() if value["dtype"] == "video"]
    camera_sizes = {camera: (info["features"][f"observation.images.{camera}"]["shape"][1],
                             info["features"][f"observation.images.{camera}"]["shape"][0])
                    for camera in cameras}
    parquet = pq.ParquetFile(output / DATA_PATH.format(episode_chunk=episode_chunk,
                                                    episode_index=episode_index))
    if parquet.metadata.num_rows != count:
        raise ValueError("Parquet frame count mismatch")
    expected_keys = {key for key, value in info["features"].items() if value["dtype"] != "video"}
    if set(parquet.schema_arrow.names) != expected_keys:
        raise ValueError("Parquet feature keys do not match info.json")
    with h5py.File(source_path, "r") as source:
        source_count, source_cameras, _, _ = inspect_source(source)
        if source_cameras != sorted(cameras):
            raise ValueError(f"Source/output camera mismatch: {source_path}")
        source_features, _ = build_features(source, camera_sizes)
        if source_count != count:
            raise ValueError("Source frame count mismatch")
        # The video encoding info is added after conversion; compare structural fields.
        if source_features != {key: {k: v for k, v in value.items() if k != "info"}
                               for key, value in info["features"].items()}:
            raise ValueError(f"Source/output feature mismatch: {source_path}")
        start = 0
        for batch in parquet.iter_batches(batch_size=batch_size):
            end = start + len(batch)
            expected, _ = batch_values(source, start, end, camera_sizes, fps,
                                       episode_index=episode_index, index_offset=index_offset)
            for key, target in expected.items():
                column = batch.column(batch.schema.get_field_index(key))
                if isinstance(column.type, pa.ExtensionType):
                    actual = column.storage.flatten().flatten().to_numpy().reshape(target.shape)
                elif pa.types.is_fixed_size_list(column.type):
                    actual = column.values.to_numpy().reshape(target.shape)
                else:
                    actual = column.to_numpy()
                if actual.dtype != target.dtype or not np.array_equal(actual, target):
                    raise ValueError(f"Source/output mismatch: {key}, frames {start}:{end}")
            start = end
        video_report = {}
        for camera, (width, height) in camera_sizes.items():
            key = f"observation.images.{camera}"
            path = output / VIDEO_PATH.format(episode_chunk=episode_chunk,
                                             episode_index=episode_index, video_key=key)
            decoded, errors = 0, []
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if (stream.width, stream.height) != (width, height) or float(stream.average_rate) != fps:
                    raise ValueError(f"Video dimensions/FPS mismatch: {camera}")
                for i, frame in enumerate(container.decode(video=0)):
                    if abs(float(frame.pts * frame.time_base) - i / fps) > 1e-4:
                        raise ValueError(f"Video timestamp mismatch: {camera}, frame {i}")
                    if i in (0, count // 2, count - 1):
                        target = cv2.resize(source[f"observations/images/{camera}"][i],
                                            (width, height), interpolation=cv2.INTER_AREA)
                        error = float(np.abs(frame.to_ndarray(format="rgb24").astype(np.float32)
                                             - target.astype(np.float32)).mean())
                        if error > 15:
                            raise ValueError(f"RGB alignment/color check failed: {camera}, MAE={error}")
                        errors.append(error)
                    decoded += 1
            if decoded != count:
                raise ValueError(f"Decoded video frame count mismatch: {camera}: {decoded}")
            video_report[camera] = {"decoded_frames": decoded, "sample_rgb_mae_0_255": errors}
    return {"episode_index": episode_index, "source": str(source_path),
            "frames": count, "videos": video_report}


def validate_stats(stats, features, count):
    if set(stats) != set(features):
        raise ValueError("Statistics keys mismatch")
    for key, value in stats.items():
        if value["count"] != [count]:
            raise ValueError(f"Statistics count mismatch: {key}")
        expected_shape = ((3, 1, 1) if features[key]["dtype"] == "video"
                          else tuple(features[key]["shape"]))
        for name in ("mean", "std", "min", "max"):
            arr = np.asarray(value[name])
            if arr.shape != expected_shape or not np.isfinite(arr).all():
                raise ValueError(f"Invalid statistics: {key}.{name}")


def validate(source_paths, output, batch_size):
    info = json.loads((output / "meta/info.json").read_text())
    episodes = read_jsonl(output / "meta/episodes.jsonl")
    tasks = read_jsonl(output / "meta/tasks.jsonl")
    statistics = iter_jsonl(output / "meta/episodes_stats.jsonl")
    if not (len(source_paths) == len(episodes) == info["total_episodes"]):
        raise ValueError("Source/output episode count mismatch")
    if len(tasks) != 1 or tasks[0]["task_index"] != 0 or info["total_tasks"] != 1:
        raise ValueError("Task metadata mismatch")
    if (info["total_frames"] != sum(episode["length"] for episode in episodes)
            or info["splits"] != {"train": f"0:{len(episodes)}"}
            or info["total_chunks"] != (len(episodes) + info["chunks_size"] - 1) // info["chunks_size"]):
        raise ValueError("Dataset totals/splits mismatch")
    video_count = sum(feature["dtype"] == "video" for feature in info["features"].values())
    if info["total_videos"] != len(episodes) * video_count:
        raise ValueError("Dataset video count mismatch")
    reports, index_offset = [], 0
    for i, (path, episode) in enumerate(zip(source_paths, episodes)):
        stat = next(statistics, None)
        if stat is None:
            raise ValueError("Statistics episode count mismatch")
        if (episode != {"episode_index": i, "tasks": [tasks[0]["task"]], "length": episode["length"]}
                or stat["episode_index"] != i):
            raise ValueError(f"Episode/task metadata mismatch: episode {i}")
        reports.append(validate_episode(path, output, info, episode, index_offset, batch_size))
        validate_stats(stat["stats"], info["features"], episode["length"])
        index_offset += episode["length"]
        print(f"[validate] episode {i}: {episode['length']} frames passed", flush=True)
    if next(statistics, None) is not None:
        raise ValueError("Statistics episode count mismatch")
    camera_sizes = {key.removeprefix("observation.images."): [feature["shape"][1], feature["shape"][0]]
                    for key, feature in info["features"].items() if feature["dtype"] == "video"}
    unique_sizes = {tuple(size) for size in camera_sizes.values()}
    has_depth = any(key.startswith("observation.depth.") for key in info["features"])
    report = {"passed": True, "frames": info["total_frames"], "total_episodes": len(episodes),
            "resolution_wh": list(next(iter(unique_sizes))) if len(unique_sizes) == 1 else None,
            "camera_resolutions_wh": camera_sizes,
            "numeric_and_timestamp_values": "all frames exactly match source",
            "depth_values": ("all frames/pixels exactly match nearest-neighbor resized source uint16 mm"
                             if has_depth else "not present in source; omitted"),
            "episodes": reports, "statistics": "feature shapes/counts/finite values verified"}
    if len(reports) == 1:
        report["videos"] = reports[0]["videos"]
    return report


def validate_official_reader(source_paths, output):
    """Check real v2.1 loading, RGB/depth tensors, action chunks and padding."""
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

    if CODEBASE_VERSION != "v2.1":
        raise RuntimeError(f"Official-reader check requires v2.1, found {CODEBASE_VERSION}; "
                           "use lerobot==0.3.3 in a separate environment")
    info = json.loads((output / "meta/info.json").read_text())
    fps = info["fps"]
    episodes = read_jsonl(output / "meta/episodes.jsonl")
    samples = []
    cache_before = datasets.config.HF_DATASETS_CACHE
    try:
        with tempfile.TemporaryDirectory(prefix="4rc-v21-reader-") as cache:
            datasets.config.HF_DATASETS_CACHE = Path(cache)
            dataset = LeRobotDataset(
                "local/4_cams", root=output, video_backend="pyav",
                delta_timestamps={"action": [0.0, 1.0 / fps]},
            )
            if len(dataset) != info["total_frames"] or dataset.fps != fps:
                raise ValueError("Official reader returned incorrect length or FPS")
            index_offset = 0
            for source_path, episode in zip(source_paths, episodes):
                count = episode["length"]
                with h5py.File(source_path, "r") as source:
                    for index in sorted({0, count // 2, count - 1}):
                        global_index = index_offset + index
                        samples.append(global_index)
                        item = dataset[global_index]
                        check_reader_item(item, source, info, index)
                    if dataset[index_offset + count - 1]["action_is_pad"].tolist() != [False, True]:
                        raise ValueError("Official reader episode boundary padding mismatch")
                index_offset += count
    finally:
        datasets.config.HF_DATASETS_CACHE = cache_before
    return {"passed": True, "codebase_version": CODEBASE_VERSION,
            "sample_frames": samples,
            "checks": "RGB/optional depth tensors, exact numeric values, action chunks and boundary padding",
            "datasets_version": datasets.__version__, "pyarrow_version": pa.__version__}


def check_reader_item(item, source, info, index):
    for key, path in NUMERIC_SOURCES.items():
        if path not in source:
            continue
        actual = item[key][0] if key == "action" else item[key]
        np.testing.assert_array_equal(actual.numpy(), source[path][index])
    for key, feature in info["features"].items():
        if feature["dtype"] == "video":
            height, width, channels = feature["shape"]
            image = item[key]
            if tuple(image.shape) != (channels, height, width):
                raise ValueError(f"Official reader RGB shape mismatch: {key}")
            if float(image.min()) < 0 or float(image.max()) > 1:
                raise ValueError(f"Official reader RGB range mismatch: {key}")
        elif key.startswith("observation.depth."):
            camera = key.removeprefix("observation.depth.")
            height, width = feature["shape"]
            expected = cv2.resize(source[f"observations/images_depth/{camera}"][index],
                                  (width, height), interpolation=cv2.INTER_NEAREST)
            np.testing.assert_array_equal(item[key].numpy(), expected)
    if tuple(item["action"].shape) != (2, source["action"].shape[1]):
        raise ValueError("Official reader action chunk shape mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "datasets/4_cams/episode_0.hdf5",
                        help="HDF5 file or directory; recursively finds .hdf5/.h5 files.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets/4_cams_lerobot_v21")
    parser.add_argument("--num-episodes", type=int,
                        help="Use at most the first N naturally sorted episodes; defaults to all.")
    parser.add_argument("--task", default="unspecified", help="Task instruction; source has no task label.")
    parser.add_argument("--robot-type", default=None, help="Optional robot type; no model is inferred.")
    parser.add_argument("--fps", type=int, help="Defaults to HDF5 capture_fps_target; no frame resampling.")
    parser.add_argument("--width", type=int, help="Output width; defaults to each camera's source width.")
    parser.add_argument("--height", type=int, help="Output height; defaults to each camera's source height.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality, 0..51; depth is always lossless.")
    parser.add_argument("--validate-only", action="store_true", help="Verify an existing output against --input.")
    parser.add_argument("--check-lerobot-reader", action="store_true",
                        help="Also check loading with an installed official v2.1 reader (lerobot 0.3.3).")
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.crf <= 51 or not args.task.strip():
        parser.error("Require positive batch size, CRF in 0..51 and a non-empty task")
    if args.num_episodes is not None and args.num_episodes < 1:
        parser.error("--num-episodes must be a positive integer")
    args.input, args.output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    try:
        source_paths = discover_sources(args.input)
    except ValueError as error:
        parser.error(str(error))
    if args.num_episodes is not None:
        source_paths = source_paths[:args.num_episodes]
    if args.validate_only:
        report = validate(source_paths, args.output, args.batch_size)
        if args.check_lerobot_reader:
            report["official_lerobot_reader"] = validate_official_reader(source_paths, args.output)
        write_json(args.output / "meta/validation.json", report)
        print(json.dumps(report, indent=2))
        return
    if args.output.exists():
        parser.error(f"Output already exists (will not overwrite): {args.output}")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required on PATH")
    started = time.monotonic()
    # Inspect every input before encoding so incompatible episodes fail early.
    records, calibrations, features, dataset_fps = [], [], None, None
    for episode_index, path in enumerate(source_paths):
        with h5py.File(path, "r") as source:
            try:
                count, cameras, entries, attrs = inspect_source(source)
                camera_sizes = output_camera_sizes(source, cameras, args.width, args.height)
                current_features, current_hf_features = build_features(source, camera_sizes)
            except ValueError as error:
                parser.error(f"{path}: {error}")
            fps = args.fps if args.fps is not None else attrs.get("capture_fps_target")
            if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
                parser.error(f"{path}: provide --fps or a positive integer capture_fps_target attribute")
            if features is None:
                features, hf_features, dataset_fps = current_features, current_hf_features, fps
            elif current_features != features or fps != dataset_fps:
                parser.error(f"{path}: all episodes must have matching cameras, numeric dimensions, "
                             "output resolutions, optional depth/timestamp features and FPS; "
                             "convert differing layouts separately")
            timing = {}
            for camera in cameras:
                timestamp_path = f"observations/camera_timestamps/{camera}"
                if timestamp_path in source and count > 1:
                    times = source[timestamp_path][:]
                    elapsed = float(times[-1] - times[0])
                    timing[camera] = {"elapsed_seconds": elapsed,
                                      "effective_fps": (count - 1) / elapsed if elapsed > 0 else None,
                                      "min_interval_seconds": float(np.diff(times).min()),
                                      "max_interval_seconds": float(np.diff(times).max())}
            records.append({
                "episode_index": episode_index, "frames": count, "fps": fps,
                "source": str(path), "source_bytes": path.stat().st_size,
                "source_datasets": entries, "source_attributes": attrs,
                "source_camera_timing": timing,
                "numeric_source_mapping": {k: v for k, v in NUMERIC_SOURCES.items() if v in source},
                "rgb_semantics": attrs.get("rgb_semantics", "assumed rgb8; source has no label"),
            })
            calibrations.append({"episode_index": episode_index, "cameras": calibration_metadata(
                source, camera_sizes)})
    args.fps = dataset_fps
    episode_count = len(records)
    total_frames = sum(record["frames"] for record in records)
    has_depth = any(key.startswith("observation.depth.") for key in features)
    print(f"[input] {episode_count} episodes, {total_frames} frames, "
          f"{len(cameras)} cameras, {'RGB-D' if has_depth else 'RGB only'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Only publish the final directory once all episodes and validation pass.
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}.", dir=args.output.parent) as temp:
        output = Path(temp) / "dataset"
        (output / "meta").mkdir(parents=True)
        with (output / "meta/episodes_stats.jsonl").open("w", encoding="utf-8") as stream:
            index_offset = 0
            for record in records:
                episode_index = record["episode_index"]
                print(f"[source] episode {episode_index}: {record['source']}", flush=True)
                with h5py.File(record["source"], "r") as source:
                    stats = convert(source, output, args, record["frames"], camera_sizes, features,
                                    hf_features, episode_index=episode_index, index_offset=index_offset)
                stream.write(json.dumps({"episode_index": episode_index, "stats": stats},
                                        allow_nan=False, separators=(",", ":")) + "\n")
                index_offset += record["frames"]
        for camera, (width, height) in camera_sizes.items():
            features[f"observation.images.{camera}"]["info"] = {
                "video.height": height, "video.width": width,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": args.fps,
                "video.channels": 3, "has_audio": False,
            }
        info = {"codebase_version": "v2.1", "robot_type": args.robot_type,
                "total_episodes": episode_count, "total_frames": total_frames, "total_tasks": 1,
                "total_videos": len(cameras) * episode_count,
                "total_chunks": (episode_count + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
                "chunks_size": CHUNKS_SIZE, "fps": args.fps,
                "splits": {"train": f"0:{episode_count}"},
                "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": features}
        write_json(output / "meta/info.json", info)
        write_jsonl(output / "meta/tasks.jsonl", [{"task_index": 0, "task": args.task}])
        write_jsonl(output / "meta/episodes.jsonl", [
            {"episode_index": record["episode_index"], "tasks": [args.task], "length": record["frames"]}
            for record in records])
        write_json(output / "meta/calibration.json", calibrations[0]["cameras"]
                   if episode_count == 1 else {"episodes": calibrations})
        unique_sizes = set(camera_sizes.values())
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(args.input), "source_bytes": sum(record["source_bytes"] for record in records),
            "episodes": records,
            "converter": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "format_reference": FORMAT_REFERENCE,
            "resolution_config": "source camera dimensions unless overridden via --width/--height",
            "output_size_wh": list(next(iter(unique_sizes))) if len(unique_sizes) == 1 else None,
            "output_camera_sizes_wh": camera_sizes, "padding_applied": False,
            "rgb_resize": "Original size preserved; OpenCV INTER_AREA only when dimensions differ; "
                          "input/output RGB, no BGR swap",
            "rgb_encoding": {"codec": "h264", "pixel_format": "yuv420p", "crf": args.crf},
            "depth_resize": ("Original size preserved; OpenCV INTER_NEAREST only when dimensions differ; "
                             "uint16 millimeters; zero invalid; no clipping"
                             if has_depth else None),
            "depth_storage": ("Hugging Face Array2D in Parquet, Zstandard lossless compression"
                              if has_depth else None),
            "timeline": "timestamp=frame_index/fps within each episode; index is global across episodes; "
                        "original frame order/count and any camera timestamps are preserved",
            "action_alignment": "source action[t] copied directly; no shift or inferred actions",
            "component_names": "source column order retained; component names are positional placeholders",
            "task_label": "unspecified is a placeholder" if args.task == "unspecified" else "provided via --task",
            "stats": "per-episode population moments; RGB normalized to [0,1]; "
                     "available depth uses per-pixel stats in mm, including invalid zeros",
            "command_options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "dependencies": {"numpy": np.__version__, "h5py": h5py.__version__,
                             "opencv": cv2.__version__, "datasets": datasets.__version__,
                             "pyarrow": pa.__version__, "av": av.__version__},
        }
        if episode_count == 1:
            for key in ("source_datasets", "source_attributes", "source_camera_timing", "numeric_source_mapping"):
                provenance[key] = records[0][key]
        write_json(output / "meta/conversion.json", provenance)
        print("[validate] Checking every numeric/available depth value and decoding every video", flush=True)
        report = validate(source_paths, output, args.batch_size)
        if args.check_lerobot_reader:
            report["official_lerobot_reader"] = validate_official_reader(source_paths, output)
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(output / "meta/validation.json", report)
        output.rename(args.output)
    resolution = ", ".join(f"{camera}={width}x{height}"
                           for camera, (width, height) in camera_sizes.items())
    print(f"[done] {args.output}: {episode_count} episodes, {total_frames} frames, {len(cameras)} cameras, "
          f"{resolution}, {args.fps} FPS; validation passed", flush=True)


if __name__ == "__main__":
    main()
