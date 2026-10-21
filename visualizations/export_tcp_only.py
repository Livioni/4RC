#!/usr/bin/env python3
"""Extract TCP overlays or clean scene vertices from existing Stage 1 PLY files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from visualizations.visualize_droid_stage1 import PLY_DTYPE, write_ply
from tcp_sliding_window_inference import write_json_atomic

DEFAULT_INPUT = ROOT / "outputs/visualizations/droid_stage1/AUTOLab__Mon_Aug_21_14:20:05_2023/22008760"


def read_vertices(path: Path, expected_count: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Missing PLY header terminator: {path}")
            header.append(line.decode("ascii").rstrip())
            if header[-1] == "end_header":
                break
        properties = [line for line in header if line.startswith("property ")]
        expected_properties = [f"property float {a}" for a in ("x", "y", "z")]
        expected_properties += [f"property uchar {c}" for c in ("red", "green", "blue")]
        if ("format binary_little_endian 1.0" not in header
                or f"element vertex {expected_count}" not in header
                or properties != expected_properties):
            raise ValueError(f"Unexpected PLY layout or vertex count: {path}")
        vertices = np.fromfile(stream, dtype=PLY_DTYPE)
    if len(vertices) != expected_count:
        raise ValueError(f"Unexpected PLY payload size: {path}")
    return vertices


def export(input_root: Path, output: Path, *, scene_only: bool = False) -> Path:
    input_root, output = input_root.expanduser().resolve(), output.expanduser().resolve()
    source = json.loads((input_root / "manifest.json").read_text())
    if source["status"] != "complete" or source["coordinate_frame"] != "robot_base":
        raise ValueError("Expected complete Stage 1 outputs in robot base coordinates")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty; choose a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", source_manifest=str(input_root / "manifest.json"),
                  coordinate_frame="robot_base", unit="meter",
                  content="scene only" if scene_only else "TCP overlays",
                  tcp=source["tcp"], groups=[])
    write_json_atomic(result, output / "manifest.json")
    for group in source["groups"]:
        directory = output / Path(group["directory"]).name
        directory.mkdir()
        record = dict(frame_indices=group["frame_indices"], directory=directory.name, files={})
        for kind in ("gt", "pred"):
            trajectory = None
            for stride in source["point_cloud_strides"]:
                filename = f"{kind}_stride{stride}.ply"
                info = group["ply"][filename]
                vertices = read_vertices(input_root / group["directory"] / filename, info["vertex_count"])
                start, count = info["scene_vertex_count"], info["tcp_vertex_count"]
                if start + count + info["pose_axis_vertex_count"] != len(vertices):
                    raise ValueError(f"Inconsistent scene/TCP/pose counts: {filename}")
                if scene_only:
                    save_vertices(directory / filename, vertices[:start], "Scene RGB only; no TCP trajectory, markers or pose axes")
                    record["files"][filename] = dict(
                        vertex_count=start, scene_frame_indices=info["frame_indices"],
                        source_ply=str((input_root / group["directory"] / filename).resolve()),
                    )
                    continue
                current = vertices[start:start + count]
                if trajectory is None:
                    trajectory = current.copy()
                    name = f"{kind}_trajectory.ply"
                    save_tcp(directory / name, trajectory, with_pose=False)
                    record["files"][name] = dict(vertex_count=count, trajectory_frame_indices=info["tcp_frame_indices"])
                elif not np.array_equal(trajectory, current):
                    raise ValueError(f"Trajectory differs across scene strides: {filename}")
                name = f"{kind}_tcp_stride{stride}.ply"
                save_tcp(directory / name, vertices[start:], with_pose=True)
                record["files"][name] = dict(
                    vertex_count=len(vertices) - start, trajectory_vertex_count=count,
                    pose_axis_vertex_count=info["pose_axis_vertex_count"],
                    trajectory_frame_indices=info["tcp_frame_indices"], pose_frame_indices=info["pose_frame_indices"],
                    source_ply=str((input_root / group["directory"] / filename).resolve()),
                )
        result["groups"].append(record)
        write_json_atomic(result, output / "manifest.json")
        print(f"Exported: {directory}", flush=True)
    result["status"] = "complete"
    write_json_atomic(result, output / "manifest.json")
    return output


def save_tcp(path: Path, vertices: np.ndarray, *, with_pose: bool) -> None:
    content = "TCP trajectory and markers (GT purple, prediction orange)"
    if with_pose:
        content += ", followed by stride-selected pose axes"
    save_vertices(path, vertices, content + "; no scene points")


def save_vertices(path: Path, vertices: np.ndarray, content: str) -> None:
    xyz = np.stack([vertices[k] for k in ("x", "y", "z")], axis=-1)
    rgb = np.stack([vertices[k] for k in ("red", "green", "blue")], axis=-1)
    write_ply(path, [(xyz, rgb)], content=content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Completed visualization output directory")
    parser.add_argument("--output", type=Path, help="Default: <input>/tcp_only (or scene_only with --scene-only)")
    parser.add_argument("--scene-only", action="store_true", help="Export clean scene point clouds instead of TCP overlays")
    args = parser.parse_args()
    output = export(args.input, args.output or args.input / ("scene_only" if args.scene_only else "tcp_only"),
                    scene_only=args.scene_only)
    print(f"Done: {output}")


if __name__ == "__main__":
    main()
