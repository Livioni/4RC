#!/usr/bin/env python3
"""Evaluate RoboTwin TCP and metric depth with propagated sliding-window queries.

    python eval_tcp.py
    torchrun --standalone --nproc_per_node=4 eval_tcp.py
    python eval_tcp.py --tasks click_bell --episode-ids 2 8 --output-dir outputs/tcp_smoke

See README_TCP_EVAL_CN.md for metric definitions, outputs, and resume behavior.
"""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from zipfile import BadZipFile

import numpy as np
from PIL import Image
import torch

from eval.tcp_depth_metrics import (
    ARMS, DEPTH_RANGES, DEPTH_STAT_NAMES, depth_frame_statistics,
    macro_metrics, summarize_depth, summarize_tcp, tcp_error_arrays,
)
from geometry_inference import (
    PAD_LEFT, PAD_TOP, PADDED_HEIGHT, PADDED_WIDTH, SOURCE_HEIGHT, SOURCE_WIDTH,
    resolve_device, resolve_dtype,
)
from tcp_inference import TCP_GROUND_TRUTH_DIRS, load_ground_truth_query_points, load_tcp_model
from tcp_sliding_window_inference import (
    BOUNDARY_POLICIES, CAMERA_SUFFIX, RPY_CONVENTION, TRAIN_MAX_FRAMES,
    infer_episode_sliding_windows, load_episode_inputs, write_json_atomic,
)


TRAIN_OUTPUT = Path("outputs/4rc-robotwin-mixed-tcp-point-query")
SCHEMA_VERSION = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets/eval_sets"))
    parser.add_argument("--model", type=Path, default=TRAIN_OUTPUT / "final_checkpoint/model.safetensors")
    parser.add_argument("--output-dir", type=Path, default=TRAIN_OUTPUT / "eval_tcp")
    parser.add_argument("--tasks", nargs="+", help="Exact task directory names")
    parser.add_argument("--episode-ids", nargs="+", type=int, help="Directory IDs (0-14), never source metadata IDs")
    parser.add_argument("--split", choices=("all", "clean", "random"), default="all")
    parser.add_argument("--view", choices=tuple(TCP_GROUND_TRUTH_DIRS), default="third_views")
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--boundary-merge", choices=BOUNDARY_POLICIES, default="previous")
    parser.add_argument("--device", default="auto", help="Single process: auto/cuda:0/cpu; torchrun: auto/cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--resume", action="store_true", help="Reuse matching complete episodes; retry failures")
    args = parser.parse_args(argv)
    if not 2 <= args.window_size <= TRAIN_MAX_FRAMES:
        parser.error(f"--window-size must be in [2,{TRAIN_MAX_FRAMES}]")
    if args.episode_ids and any(i not in range(15) for i in args.episode_ids):
        parser.error("--episode-ids must be in [0,14]")
    return args


def episode_split(episode_id: int) -> str:
    if episode_id not in range(15):
        raise ValueError(f"Expected episode directory ID 0-14, got {episode_id}")
    return "clean" if episode_id < 5 else "random"


def discover_episodes(args: argparse.Namespace) -> list[dict]:
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    tasks = {p.name: p for p in root.iterdir() if p.is_dir()}
    selected_tasks = sorted(set(args.tasks) if args.tasks else tasks)
    missing = set(selected_tasks) - tasks.keys()
    if missing:
        raise ValueError(f"Unknown tasks: {sorted(missing)}")
    ids = sorted(set(args.episode_ids) if args.episode_ids else range(15))
    result = []
    for task in selected_tasks:
        directories = {}
        for path in tasks[task].iterdir():
            match = re.fullmatch(r"episode_(\d+)", path.name)
            if path.is_dir() and match:
                index = int(match[1])
                episode_split(index)
                if index in directories:
                    raise ValueError(f"Duplicate episode ID {index} in {tasks[task]}")
                directories[index] = path
        for index in ids:
            split = episode_split(index)
            if args.split != "all" and split != args.split:
                continue
            # Missing expected episodes remain in the denominator and fail explicitly.
            path = directories.get(index, tasks[task] / f"episode_{index:07d}")
            result.append({"task": task, "episode": path.name, "episode_id": index,
                           "split": split, "path": str(path)})
    if not result:
        raise ValueError("No episodes selected")
    return result


def file_identity(path: Path) -> dict:
    try:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except FileNotFoundError:
        return {"path": str(path.resolve()), "missing": True}


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def episode_fingerprint(item: dict, view: str, config: dict) -> str:
    episode = Path(item["path"])
    tcp_dir = episode / TCP_GROUND_TRUTH_DIRS[view]
    paths = [episode / "metadata.json", episode / "intrinsics" / f"{view}.npy",
             tcp_dir / "metadata.json", tcp_dir / "left_state.npy", tcp_dir / "right_state.npy"]
    for kind in ("images", "depths"):
        directory = episode / kind / view
        paths.extend(sorted(p for p in directory.iterdir() if p.is_file()) if directory.is_dir() else [directory])
    return digest({"config": config, "files": [file_identity(p) for p in paths]})


def make_config(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict:
    source_root = Path(__file__).resolve().parent
    sources = ("eval_tcp.py", "eval/tcp_depth_metrics.py", "tcp_inference.py",
               "tcp_sliding_window_inference.py", "geometry_inference.py", "arc/rotation.py")
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_sha256": digest({p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in sources}),
        "data_root": str(args.data_root.expanduser().resolve()),
        "model": file_identity(args.model.expanduser()),
        "view": args.view, "window_size": args.window_size,
        "boundary_merge": args.boundary_merge, "dtype": str(dtype), "device_type": device.type,
        "torch_version": torch.__version__, "depth_ranges_m": DEPTH_RANGES,
        "include_initial_frame": True, "alignment": "none", "seed": 42,
    }


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_bytes_atomic(path, buffer.getvalue())


def result_path(output_dir: Path, item: dict) -> Path:
    return output_dir / "episodes" / item["task"] / f"{item['episode']}.json"


def load_cached_result(path: Path, fingerprint: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    if result.get("fingerprint") != fingerprint:
        raise ValueError(f"Episode inputs or evaluation configuration changed: {path}; use a new --output-dir")
    if result.get("status") != "complete":
        return None
    try:
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            required = [*result["array_keys"], "fingerprint"]
            if any(key not in arrays for key in required):
                return None
            if str(arrays["fingerprint"].item()) != fingerprint:
                return None
            # Force decompression/CRC validation before declaring the cache complete.
            for key in required:
                arrays[key]
    except (OSError, ValueError, EOFError, KeyError, BadZipFile):
        return None
    return result


def load_tcp_truth(episode: Any, view: str) -> np.ndarray:
    path = episode.episode_path
    metadata = json.loads((path / "metadata.json").read_text())
    tcp_path = path / TCP_GROUND_TRUTH_DIRS[view]
    tcp_meta = json.loads((tcp_path / "metadata.json").read_text())
    n = metadata["num_frames"]
    expected = {
        "shape": [n, 7], "columns": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"],
        "camera": view, "coordinate_frame": f"{view} {CAMERA_SUFFIX}",
        "position_unit": "meter", "rotation_unit": "radian", "rpy_convention": RPY_CONVENTION,
        "gripper": {"type": "binary", "open": 1, "closed": 0, "source_threshold": 0.5},
    }
    for key, value in expected.items():
        if tcp_meta.get(key) != value:
            raise ValueError(f"Incompatible TCP metadata {tcp_path}: {key}")
    if episode.frame_indices != list(range(n)):
        raise ValueError(f"Expected all {n} episode frames, numbered 0 to {n - 1}")
    states = []
    for arm in ("left", "right"):
        state = np.load(tcp_path / f"{arm}_state.npy", allow_pickle=False)
        if state.shape != (n, 7) or not np.isfinite(state).all():
            raise ValueError(f"Invalid {arm} TCP truth in {tcp_path}")
        if not np.isin(state[:, 6], (0, 1)).all():
            raise ValueError(f"Nonbinary {arm} gripper truth in {tcp_path}")
        states.append(state[episode.frame_indices])
    return np.stack(states, axis=1)


class DepthCollector:
    """Consume finalized depth frames without retaining full-resolution maps."""

    def __init__(self, episode_path: Path, view: str):
        self.path, self.view = episode_path, view
        self.frame_indices: list[int] = []
        self.statistics: dict[str, list[np.ndarray]] = {name: [] for name in DEPTH_RANGES}
        self.errors: list[dict] = []
        self.metadata_error = None
        try:
            meta = json.loads((episode_path / "metadata.json").read_text())
            depth_meta = meta.get("depth", {})
            if depth_meta.get("unit") != "millimeter" or depth_meta.get("invalid_value") != 0:
                raise ValueError("Depth metadata must specify millimeter units and invalid_value=0")
        except (OSError, ValueError) as error:
            self.metadata_error = str(error)

    def __call__(self, frame_index: int, padded_depth: np.ndarray) -> None:
        self.frame_indices.append(frame_index)
        stats = {name: np.zeros(len(DEPTH_STAT_NAMES)) for name in DEPTH_RANGES}
        try:
            if self.metadata_error:
                raise ValueError(self.metadata_error)
            if padded_depth.shape != (PADDED_HEIGHT, PADDED_WIDTH):
                raise ValueError(f"Expected padded depth [252,322], got {padded_depth.shape}")
            prediction = padded_depth[PAD_TOP:PAD_TOP + SOURCE_HEIGHT, PAD_LEFT:PAD_LEFT + SOURCE_WIDTH]
            depth_file = self.path / "depths" / self.view / f"{frame_index:06d}.png"
            with Image.open(depth_file) as image:
                target = np.asarray(image, dtype=np.float64) / 1000.0
            if target.shape != (SOURCE_HEIGHT, SOURCE_WIDTH):
                raise ValueError(f"Invalid depth shape {target.shape}: {depth_file}")
            for name, limit in DEPTH_RANGES.items():
                stats[name] = depth_frame_statistics(prediction, target, limit)
                if stats[name][-1]:
                    self.errors.append({"frame_index": frame_index, "range": name,
                                        "error": f"{int(stats[name][-1])} nonfinite depth predictions"})
        except (OSError, ValueError) as error:
            self.errors.append({"frame_index": frame_index, "range": "both", "error": str(error)})
        for name in DEPTH_RANGES:
            self.statistics[name].append(stats[name])


def evaluate_episode(
    model: Any, item: dict, args: argparse.Namespace, device: torch.device,
    dtype: torch.dtype, fingerprint: str, rank: int,
) -> dict:
    started = time.perf_counter()
    result = {**item, "schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
              "rank": rank, "status": "failed", "tcp_status": "incomplete",
              "depth_status": dict.fromkeys(DEPTH_RANGES, "incomplete"), "errors": []}
    arrays = {"fingerprint": np.asarray(fingerprint)}
    collector = None
    last_progress = [0.0]

    def progress(fraction: float, message: str) -> None:
        now = time.monotonic()
        if now - last_progress[0] >= 30 or fraction == 1:
            print(f"[rank {rank}] {item['task']}/{item['episode']} {message}", flush=True)
            last_progress[0] = now

    try:
        episode = load_episode_inputs(Path(item["path"]), args.view)
        states = load_tcp_truth(episode, args.view)
        query = load_ground_truth_query_points(episode.episode_path, args.view, episode.frame_indices[0])
        collector = DepthCollector(episode.episode_path, args.view)
        prediction = infer_episode_sliding_windows(
            model, episode, query, "projected first-frame ground-truth TCP",
            device=device, dtype=dtype, window_size=args.window_size,
            boundary_merge=args.boundary_merge, keep_geometry=False,
            max_points_per_frame=0, progress_callback=progress, depth_frame_callback=collector,
        )
        errors = tcp_error_arrays(prediction.tcp, states, np.asarray(episode.frame_indices), episode.frame_rate)
        arrays.update({f"error_{key}": value for key, value in errors.items()})
        arrays.update({f"pred_{key}": value for key, value in prediction.tcp.items()})
        arrays.update({"gt_tcp_state": states, "frame_indices": np.asarray(episode.frame_indices)})
        result.update({
            "num_frames": len(episode.frame_indices), "frame_rate_hz": episode.frame_rate,
            "windows": prediction.window_records, "source_windows": prediction.source_windows,
            "initial_query_points_px": query.tolist(), "tcp_status": "success",
            "tcp_metrics": {arm: summarize_tcp(errors, arm) for arm in ARMS},
        })
        if collector.frame_indices != episode.frame_indices:
            raise RuntimeError("Depth callbacks did not cover the episode exactly once in order")
        result["depth_metrics"] = {}
        result["errors"].extend({"stage": "depth", **error} for error in collector.errors)
        for name in DEPTH_RANGES:
            statistics = np.asarray(collector.statistics[name])
            arrays[f"depth_stats_{name}"] = statistics
            metrics = summarize_depth(statistics)
            failed = any(e["range"] in (name, "both") for e in collector.errors)
            status = "failed" if failed else "success" if metrics["valid_pixel_count"] else "empty"
            result["depth_status"][name] = status
            result["depth_metrics"][name] = metrics if not failed else {
                key: value if key.endswith("_count") else None for key, value in metrics.items()
            }
        result["status"] = "complete" if all(s != "failed" for s in result["depth_status"].values()) else "failed"
    except Exception as error:
        result["errors"].append({"stage": "episode", "error_type": type(error).__name__, "error": str(error)})
        if collector is not None:
            result["completed_depth_frames"] = len(collector.frame_indices)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["array_keys"] = sorted(arrays)
    path = result_path(args.output_dir, item)
    save_arrays(path.with_suffix(".npz"), arrays)
    write_json_atomic(result, path)
    return result


def coverage(records: list[dict], family: str, depth_range: str | None = None) -> dict:
    statuses = [r["tcp_status"] if family == "tcp" else r["depth_status"][depth_range] for r in records]
    success = statuses.count("success")
    empty = statuses.count("empty")
    return {"planned_episode_count": len(records), "successful_episode_count": success,
            "failed_episode_count": len(records) - success - empty, "empty_episode_count": empty,
            "completion_rate": success / len(records) if records else None}


def group_rows(records: list[dict], task: str, split: str) -> tuple[list[dict], list[dict]]:
    tcp_rows, depth_rows = [], []
    tcp_records = [r for r in records if r["tcp_status"] == "success"]
    for arm in ARMS:
        base = {"task": task, "split": split, "arm": arm, **coverage(records, "tcp")}
        macros = macro_metrics([r["tcp_metrics"][arm] for r in tcp_records])
        tcp_rows.append({**base, "aggregation": "episode_macro", **macros})
        pooled = {}
        if tcp_records:
            keys = tcp_records[0]["_tcp_arrays"].keys()
            errors = {key: np.concatenate([r["_tcp_arrays"][key] for r in tcp_records]) for key in keys}
            pooled = summarize_tcp(errors, arm)
        tcp_rows.append({**base, "aggregation": "frame_micro", **pooled})
    for name in DEPTH_RANGES:
        depth_records = [r for r in records if r["depth_status"][name] == "success"]
        base = {"task": task, "split": split, "depth_range": name, **coverage(records, "depth", name)}
        metrics = macro_metrics([r["depth_metrics"][name] for r in depth_records])
        depth_rows.append({**base, "aggregation": "episode_macro", **metrics})
        pooled = summarize_depth(np.concatenate([r["_depth_arrays"][name] for r in depth_records])) if depth_records else {}
        depth_rows.append({**base, "aggregation": "pixel_micro", **pooled})
    return tcp_rows, depth_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    write_bytes_atomic(path, buffer.getvalue().encode("utf-8"))


def build_reports(output_dir: Path, items: list[dict], config: dict, world_size: int) -> dict:
    records = []
    for item in items:
        path = result_path(output_dir, item)
        record = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            record["_tcp_arrays"] = {key.removeprefix("error_"): arrays[key] for key in arrays.files if key.startswith("error_")}
            record["_depth_arrays"] = {name: arrays[f"depth_stats_{name}"] for name in DEPTH_RANGES if f"depth_stats_{name}" in arrays}
        records.append(record)
    task_tcp, task_depth, global_tcp, global_depth = [], [], [], []
    tasks = sorted({r["task"] for r in records})
    for task in tasks:
        for split in ("clean", "random", "all"):
            group = [r for r in records if r["task"] == task and (split == "all" or r["split"] == split)]
            tcp, depth = group_rows(group, task, split)
            task_tcp.extend(tcp)
            task_depth.extend(depth)
    # Global task_macro is a mean of task episode_macro metrics, excluding empty tasks.
    for split in ("clean", "random", "all"):
        group = [r for r in records if split == "all" or r["split"] == split]
        tcp, depth = group_rows(group, "__all__", split)
        global_tcp.extend(tcp)
        global_depth.extend(depth)
        for arm in ARMS:
            rows = [row for row in task_tcp if row["split"] == split and row["arm"] == arm
                    and row["aggregation"] == "episode_macro" and row["successful_episode_count"]]
            metric_keys = group[0]["tcp_metrics"][arm].keys() if group and group[0].get("tcp_metrics") else None
            if metric_keys is None:
                good = next((r for r in group if r.get("tcp_metrics")), None)
                metric_keys = good["tcp_metrics"][arm].keys() if good else []
            means = macro_metrics([{key: row[key] for key in metric_keys} for row in rows])
            global_tcp.append({"task": "__all__", "split": split, "arm": arm,
                               "aggregation": "task_macro", **coverage(group, "tcp"), **means})
        for name in DEPTH_RANGES:
            good = next((r for r in group if r["depth_status"][name] == "success"), None)
            keys = good["depth_metrics"][name].keys() if good else []
            rows = [row for row in task_depth if row["split"] == split and row["depth_range"] == name
                    and row["aggregation"] == "episode_macro" and row["successful_episode_count"]]
            means = macro_metrics([{key: row[key] for key in keys} for row in rows])
            global_depth.append({"task": "__all__", "split": split, "depth_range": name,
                                 "aggregation": "task_macro", **coverage(group, "depth", name), **means})
    tcp_episodes, depth_episodes, failures = [], [], []
    for record in records:
        base = {key: record[key] for key in ("task", "episode", "episode_id", "split")}
        for arm in ARMS:
            tcp_episodes.append({**base, "arm": arm, "status": record["tcp_status"],
                                 **record.get("tcp_metrics", {}).get(arm, {})})
        for name in DEPTH_RANGES:
            depth_episodes.append({**base, "depth_range": name, "status": record["depth_status"][name],
                                   **record.get("depth_metrics", {}).get(name, {})})
        if record["status"] != "complete":
            failures.append({**base, "errors": record["errors"], "tcp_status": record["tcp_status"],
                             "depth_status": record["depth_status"]})
    for name, rows in (("task_metrics.csv", task_tcp), ("episode_metrics.csv", tcp_episodes),
                       ("depth_task_metrics.csv", task_depth), ("depth_episode_metrics.csv", depth_episodes),
                       ("global_tcp_metrics.csv", global_tcp), ("global_depth_metrics.csv", global_depth)):
        write_csv(output_dir / name, rows)
    write_bytes_atomic(output_dir / "failures.jsonl", "".join(json.dumps(f, ensure_ascii=False, allow_nan=False) + "\n" for f in failures).encode())
    summary = {
        "schema_version": SCHEMA_VERSION, "config": config, "world_size": world_size,
        "task_count": len(tasks), "planned_episode_count": len(items),
        "complete_episode_count": len(items) - len(failures), "failed_episode_count": len(failures),
        "selected_episodes": [{key: item[key] for key in ("task", "episode", "split")} for item in items],
        "tcp": global_tcp, "depth": global_depth,
        "metric_definitions": {
            "coordinate_frame": f"per-frame {config['view']} {CAMERA_SUFFIX}",
            "tcp_position_unit": "mm", "tcp_rotation_unit": "degree",
            "depth_unit": "meter", "depth_alignment": "none",
            "depth_ranges": "within_3m: finite 0<GT<=3m; all_valid: finite GT>0",
            "depth_delta": "max(pred/GT,GT/pred) < 1.25**k; nonpositive predictions always fail",
            "depth_log_epsilon_m": 1e-5, "depth_stat_columns": list(DEPTH_STAT_NAMES),
            "episode_macro": "equal mean of defined episode metrics; counts sum",
            "task_macro": "equal mean of defined task episode_macro metrics; counts sum",
            "frame_micro": "pooled TCP frame errors; motion over pairs; endpoints over episodes",
            "pixel_micro": "pooled valid depth pixels; RMSE from total squared error / pixel count",
            "failure_policy": "only complete trajectories per metric family enter metrics; coverage includes failures",
            "undefined_metrics": "null; excluded from macro means",
        },
    }
    write_json_atomic(summary, output_dir / "summary.json")
    return summary


def prepare_output(state: Any, output_dir: Path, config: dict, resume: bool) -> None:
    error = [None]
    if state.is_main_process:
        try:
            manifest = output_dir / "run_config.json"
            if manifest.exists():
                if not resume:
                    raise ValueError(f"Output already contains a run: {output_dir}; use --resume or a new --output-dir")
                if json.loads(manifest.read_text()) != config:
                    raise ValueError("Run configuration/checkpoint/code changed; use a new --output-dir")
            elif output_dir.exists() and any(output_dir.iterdir()):
                raise ValueError(f"Output directory is nonempty without a run manifest: {output_dir}")
            write_json_atomic(config, manifest)
        except Exception as exc:
            error[0] = str(exc)
    if state.num_processes > 1:
        torch.distributed.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError(error[0])


def main() -> int:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and args.device not in ("auto", "cuda"):
        raise ValueError("With torchrun use --device auto/cuda; LOCAL_RANK selects the GPU")
    if not args.model.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    from accelerate import PartialState

    state = PartialState(cpu=args.device == "cpu", timeout=timedelta(hours=24))
    device = state.device if state.num_processes > 1 else resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(42)
    np.random.seed(42)
    args.output_dir = args.output_dir.expanduser().resolve()
    items = discover_episodes(args)
    config = make_config(args, device, dtype)
    prepare_output(state, args.output_dir, config, args.resume)
    jobs = items[state.process_index::state.num_processes]
    print(f"[rank {state.process_index}] {len(jobs)}/{len(items)} episodes on {device}, {dtype}; TCP + depth (<=3m/all valid)", flush=True)
    model = None
    try:
        for number, item in enumerate(jobs, 1):
            fingerprint = episode_fingerprint(item, args.view, config)
            path = result_path(args.output_dir, item)
            cached = load_cached_result(path, fingerprint) if args.resume else None
            if cached is not None:
                print(f"[rank {state.process_index}] [{number}/{len(jobs)}] cached {item['task']}/{item['episode']}", flush=True)
                continue
            if model is None:
                print(f"[rank {state.process_index}] Loading {args.model}", flush=True)
                model = load_tcp_model(args.model, device)
            result = evaluate_episode(model, item, args, device, dtype, fingerprint, state.process_index)
            print(f"[rank {state.process_index}] [{number}/{len(jobs)}] {item['task']}/{item['episode']}: "
                  f"{result['status']} ({result['elapsed_seconds']:.1f}s)", flush=True)
            for error in result["errors"][:3]:
                print(f"  {error}", flush=True)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    state.wait_for_everyone()
    exit_code = [0]
    if state.is_main_process:
        try:
            summary = build_reports(args.output_dir, items, config, state.num_processes)
            print(f"Saved reports to {args.output_dir}: {summary['complete_episode_count']}/{len(items)} complete episodes", flush=True)
            exit_code[0] = int(summary["failed_episode_count"] > 0)
        except Exception as error:
            print(f"Report generation failed: {type(error).__name__}: {error}", flush=True)
            exit_code[0] = 1
    if state.num_processes > 1:
        torch.distributed.broadcast_object_list(exit_code, src=0)
        torch.distributed.destroy_process_group()
    return exit_code[0]


if __name__ == "__main__":
    raise SystemExit(main())
