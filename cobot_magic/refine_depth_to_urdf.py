#!/usr/bin/env python3
"""Fit a shared optical-axis depth correction against a fixed, measured URDF."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.optimize import least_squares

from cobot_magic.convert_to_4d_dataset import validate_episode, write_json
from cobot_magic.geometry import (
    CAMERA, DEFAULT_URDF, RobotModel, file_digest, origin_matrix, resolve_mesh,
)


CORRECTED_DIRECTORY = "depths_corrected"
DIAGNOSTIC_DIRECTORY = "depth_correction_diagnostics"
CORRECTION_FORMAT = "cobot_magic_affine_depth_v1"
FILTERS = {
    "min_depth_m": 0.1, "max_depth_m": 3.0,
    "silhouette_erosion_pixels": 1, "max_local_depth_range_m": 0.03,
    "min_abs_normal_ray_cosine": 0.35, "max_raw_residual_m": 0.08,
}
FIT_SETTINGS = {
    "loss": "huber", "huber_scale_m": 0.008,
    "scale_bounds": [0.9, 1.1], "bias_bounds_m": [-0.05, 0.05],
}
ACCEPTANCE = {
    "folds": 4, "min_samples_per_arm": 100,
    "min_median_improvement_fraction": 0.20, "max_p90_increase_mm": 1.0,
}


def source_manifest(episode):
    """Hash original episode contents, excluding derived outputs and staging files."""
    episode = Path(episode)
    paths = [episode / "metadata.json", episode / "timestamps.npy"]
    for directory in ("images", "depths", "intrinsic", "extrinsic", "observations", "action", "TCP"):
        paths.extend(p for p in (episode / directory).rglob("*") if p.is_file())
    return {p.relative_to(episode).as_posix(): file_digest(p) for p in sorted(paths)}


def mesh_manifest(model):
    filenames = {node.get("filename") for node in model.xml.findall("link/visual/geometry/mesh")}
    return {name: file_digest(resolve_mesh(model.path, name)) for name in sorted(filenames)}


def correct_depth_mm(depth, scale, bias_m):
    """Preserve invalid zeroes; round and saturate valid uint16 millimetre samples."""
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Expected a uint16 depth image [H,W]")
    if not np.isfinite([scale, bias_m]).all() or scale <= 0:
        raise ValueError("Expected finite positive depth scale and finite bias")
    valid = depth != 0
    values = np.rint(depth[valid].astype(np.float64) * scale + bias_m * 1000)
    clipped = int(np.count_nonzero((values < 1) | (values > 65535)))
    output = np.zeros_like(depth)
    output[valid] = np.clip(values, 1, 65535).astype(np.uint16)
    return output, clipped


def corrected_samples(observed_m, parameters):
    # Score the actual rounded PNG values, including saturation, rather than an
    # unquantized approximation that could make the validation look better.
    return np.clip(np.rint((parameters[0] * observed_m + parameters[1]) * 1000),
                   1, 65535) / 1000


@dataclass
class Samples:
    observed: np.ndarray
    reference: np.ndarray
    link: np.ndarray
    frame: np.ndarray

    @classmethod
    def concatenate(cls, chunks):
        if not chunks:
            raise ValueError("No usable robot/depth correspondences")
        return cls(*(np.concatenate([getattr(c, key) for c in chunks])
                     for key in ("observed", "reference", "link", "frame")))


def fit_model(samples, select=None):
    select = np.ones(len(samples.observed), dtype=bool) if select is None else select
    observed, reference = samples.observed[select], samples.reference[select]
    if len(observed) < 200 or np.ptp(observed) < 0.1:
        raise ValueError("Depth fit requires >=200 samples spanning >=0.1 m")
    result = least_squares(
        lambda p: p[0] * observed + p[1] - reference, [1.0, 0.0],
        loss=FIT_SETTINGS["loss"], f_scale=FIT_SETTINGS["huber_scale_m"],
        bounds=([FIT_SETTINGS["scale_bounds"][0], FIT_SETTINGS["bias_bounds_m"][0]],
                [FIT_SETTINGS["scale_bounds"][1], FIT_SETTINGS["bias_bounds_m"][1]]))
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Depth fit failed: {result.message}")
    return result.x


def error_metrics(errors_m):
    if not len(errors_m):
        return {"count": 0, "median_signed_mm": None, "median_abs_mm": None,
                "p90_abs_mm": None, "rmse_mm": None}
    errors = np.asarray(errors_m, dtype=np.float64) * 1000
    return {"count": len(errors), "median_signed_mm": float(np.median(errors)),
            "median_abs_mm": float(np.median(np.abs(errors))),
            "p90_abs_mm": float(np.percentile(np.abs(errors), 90)),
            "rmse_mm": float(np.sqrt(np.mean(errors ** 2)))}


def compare_samples(samples, parameters, select=None):
    select = np.ones(len(samples.observed), dtype=bool) if select is None else select
    observed, reference = samples.observed[select], samples.reference[select]
    return {"raw": error_metrics(observed - reference),
            "corrected": error_metrics(corrected_samples(observed, parameters) - reference)}


def passes_acceptance(metrics):
    raw, corrected = metrics["raw"], metrics["corrected"]
    return bool(raw["count"] >= ACCEPTANCE["min_samples_per_arm"]
                and raw["median_abs_mm"] > 0
                and corrected["median_abs_mm"] <= raw["median_abs_mm"] * (
                    1 - ACCEPTANCE["min_median_improvement_fraction"])
                and corrected["p90_abs_mm"] <= raw["p90_abs_mm"] + (
                    ACCEPTANCE["max_p90_increase_mm"]))


def cross_validate(samples, frame_indices, link_names):
    folds = []
    left_ids = [i for i, name in enumerate(link_names) if name.startswith("fl_")]
    arm_masks = {"left": np.isin(samples.link, left_ids),
                 "right": ~np.isin(samples.link, left_ids)}
    held_out_errors, held_out_corrected = [], []
    for fold, frames in enumerate(np.array_split(frame_indices, ACCEPTANCE["folds"])):
        validation = np.isin(samples.frame, frames)
        parameters = fit_model(samples, ~validation)
        metrics = {"all": compare_samples(samples, parameters, validation)}
        metrics.update({name: compare_samples(samples, parameters, validation & mask)
                        for name, mask in arm_masks.items()})
        passed = all(passes_acceptance(m) for m in metrics.values())
        folds.append({"fold": fold, "validation_frames": frames.tolist(),
                      "scale": float(parameters[0]), "bias_m": float(parameters[1]),
                      "metrics": metrics, "passed": passed})
        held_out_errors.append(samples.observed[validation] - samples.reference[validation])
        held_out_corrected.append(corrected_samples(samples.observed[validation], parameters)
                                  - samples.reference[validation])
        before, after = metrics["all"]["raw"], metrics["all"]["corrected"]
        print(f"Fold {fold + 1}/4: median {before['median_abs_mm']:.2f} -> "
              f"{after['median_abs_mm']:.2f} mm; P90 {before['p90_abs_mm']:.2f} -> "
              f"{after['p90_abs_mm']:.2f} mm; {'PASS' if passed else 'FAIL'}", flush=True)
    return {"passed": all(f["passed"] for f in folds), "criteria": ACCEPTANCE,
            "folds": folds, "out_of_fold": {
                "raw": error_metrics(np.concatenate(held_out_errors)),
                "corrected": error_metrics(np.concatenate(held_out_corrected))}}


def flatten_visual_mesh(scene, visual):
    """Apply every scene node, URDF visual origin and mesh scale, in that order."""
    import trimesh

    node = visual.find("geometry/mesh")
    scale = np.eye(4)
    scale[:3, :3] = np.diag(np.fromstring(node.get("scale", "1 1 1"), sep=" "))
    visual_transform = origin_matrix(visual.find("origin")) @ scale
    vertices, faces, offset = [], [], 0
    for name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[name]
        geometry = scene.geometry[geometry_name]
        vertices.append(trimesh.transform_points(geometry.vertices, visual_transform @ transform))
        faces.append(geometry.faces + offset)
        offset += len(geometry.vertices)
    if not vertices:
        raise ValueError("URDF visual contains no triangle geometry")
    return np.concatenate(vertices), np.concatenate(faces).astype(np.uint32)


class RobotDepthRenderer:
    """CPU raycasting; unnormalised camera rays make t_hit optical-axis Z."""

    def __init__(self, model, intrinsic, extrinsic, *, width=320, height=240, threads=4):
        import open3d as o3d
        import trimesh

        self.o3d, self.model, self.extrinsic = o3d, model, extrinsic
        self.threads = threads
        y, x = np.indices((height, width))
        self.directions = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsic).T
        self.rays = o3d.core.Tensor(np.concatenate(
            [np.zeros_like(self.directions), self.directions], axis=-1).astype(np.float32))
        self.link_names = []
        self.parts, scenes = [], {}
        # The observed front arms are the reference. Other URDF chains have
        # unmeasured/default states and cannot supply calibration targets.
        for name, link in model.links.items():
            if not name.startswith(("fl_", "fr_")):
                continue
            for visual in link.findall("visual"):
                mesh_node = visual.find("geometry/mesh")
                if mesh_node is None:
                    raise ValueError(f"Expected mesh visual for measured link {name}")
                path = resolve_mesh(model.path, mesh_node.get("filename"))
                if path not in scenes:
                    scenes[path] = trimesh.load(path, force="scene", process=False)
                vertices, faces = flatten_visual_mesh(scenes[path], visual)
                if name not in self.link_names:
                    self.link_names.append(name)
                self.parts.append((name, vertices, faces, self.link_names.index(name)))
        if not self.parts:
            raise ValueError("No measured front-arm visual meshes")

    def render(self, qpos):
        o3d = self.o3d
        fk = self.model.forward(np.asarray(qpos)[None])
        scene = o3d.t.geometry.RaycastingScene(nthreads=self.threads)
        geometry_to_link = []
        for name, vertices, faces, link_id in self.parts:
            transform = self.extrinsic @ fk[name][0]
            camera_vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            geometry_id = scene.add_triangles(
                o3d.core.Tensor(camera_vertices.astype(np.float32)), o3d.core.Tensor(faces))
            if geometry_id != len(geometry_to_link):
                raise ValueError("Unexpected non-contiguous raycast geometry IDs")
            geometry_to_link.append(link_id)
        hits = scene.cast_rays(self.rays, nthreads=self.threads)
        reference = hits["t_hit"].numpy()
        geometry_ids = hits["geometry_ids"].numpy()
        normals = hits["primitive_normals"].numpy()
        links = np.full(reference.shape, -1, dtype=np.int16)
        finite = np.isfinite(reference)
        links[finite] = np.asarray(geometry_to_link, dtype=np.int16)[geometry_ids[finite]]
        cosine = np.abs(np.sum(normals * self.directions, axis=-1)) / np.linalg.norm(
            self.directions, axis=-1)
        return reference, links, cosine


def select_correspondences(depth, reference, links, cosine, frame):
    observed = depth.astype(np.float64) / 1000
    rendered = np.isfinite(reference) & (reference > 0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    core = cv2.erode(rendered.astype(np.uint8), kernel).astype(bool)
    observed32 = observed.astype(np.float32)
    local_range = cv2.dilate(observed32, kernel) - cv2.erode(observed32, kernel)
    valid = (core & (observed > FILTERS["min_depth_m"])
             & (observed < FILTERS["max_depth_m"])
             & (cosine > FILTERS["min_abs_normal_ray_cosine"])
             & (local_range < FILTERS["max_local_depth_range_m"]))
    selected = valid & (np.abs(observed - reference) < FILTERS["max_raw_residual_m"])
    samples = Samples(observed[selected], reference[selected].astype(np.float64),
                      links[selected], np.full(np.count_nonzero(selected), frame, dtype=np.int32))
    counts = {"frame": int(frame), "rendered_pixels": int(rendered.sum()),
              "rendered_with_raw_depth": int(np.count_nonzero(rendered & (depth > 0))),
              "smooth_interior_pixels": int(valid.sum()), "selected_pixels": int(selected.sum()),
              "residual_gate_rejected_pixels": int(np.count_nonzero(valid & ~selected))}
    return samples, counts


def load_depth(episode, frame, directory="depths"):
    path = Path(episode) / directory / CAMERA / f"{frame:06d}.png"
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.shape != (240, 320):
        raise ValueError(f"Invalid depth frame: {path}")
    return depth


def comparison_panel(episode, frame, raw, corrected, reference):
    rgb = cv2.imread(str(Path(episode) / "images" / CAMERA / f"{frame:06d}.png"))
    mask = np.isfinite(reference) & (reference > 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 255, 0), 1)
    panels = [rgb]
    for depth in (raw, corrected):
        valid = mask & (raw > 0)
        error = np.zeros(raw.shape, dtype=np.float64)
        error[valid] = (depth[valid].astype(float) - reference[valid] * 1000) / 50
        magnitude = np.clip(np.abs(error), 0, 1)
        image = np.full((*raw.shape, 3), 32, dtype=np.uint8)
        color = np.full((*raw.shape, 3), 255.0)
        color[:, :, 1] *= 1 - magnitude
        color[:, :, 0] *= 1 - np.clip(error, 0, 1)
        color[:, :, 2] *= 1 - np.clip(-error, 0, 1)
        image[valid] = color[valid].astype(np.uint8)
        panels.append(image)
    titles = [f"frame {frame}: RGB + URDF outline", "Raw residual (-50 blue / +50 red mm)",
              "Corrected residual (same pixels)"]
    labelled = []
    for panel, title in zip(panels, titles):
        header = np.full((28, 320, 3), 245, dtype=np.uint8)
        cv2.putText(header, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)
        labelled.append(np.concatenate([header, panel]))
    return np.concatenate(labelled, axis=1)


def validate_corrected_depths(episode, directory=None, *, urdf=None):
    """Check provenance and all corrected pixels without importing Open3D."""
    episode = Path(episode)
    directory = Path(directory) if directory is not None else episode / CORRECTED_DIRECTORY
    meta = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("format") != CORRECTION_FORMAT or calibration.get("accepted") is not True:
        raise ValueError("Not an accepted depth correction")
    if source_manifest(episode) != calibration["source_files_sha256"]:
        raise ValueError("Corrected depths are stale: original episode contents changed")
    model_path = Path(urdf or meta["urdf"]).expanduser()
    if urdf is None and not model_path.is_file():
        model_path = DEFAULT_URDF
    model = RobotModel(model_path, joint_defaults=meta["unobserved_joint_values"])
    if (file_digest(model.path) != calibration["urdf_sha256"]
            or mesh_manifest(model) != calibration["visual_meshes_sha256"]):
        raise ValueError("Corrected depths are stale: URDF or visual meshes changed")
    depth_meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    original_meta = json.loads((episode / "depths/metadata.json").read_text(encoding="utf-8"))
    if any(depth_meta.get(k) != v for k, v in original_meta.items()):
        raise ValueError("Corrected depth metadata disagrees with source depth format")
    count = meta["frame_count"]
    if {p.name for p in (directory / CAMERA).iterdir()} != {f"{i:06d}.png" for i in range(count)}:
        raise ValueError("Corrected depths must contain all matching source frames")
    scale, bias = calibration["scale"], calibration["bias_m"]
    clipped, valid = 0, 0
    for i in range(count):
        raw = load_depth(episode, i)
        corrected = cv2.imread(str(directory / CAMERA / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        expected, clipped_count = correct_depth_mm(raw, scale, bias)
        if (corrected is None or corrected.dtype != np.uint16 or corrected.shape != raw.shape
                or not np.array_equal(corrected, expected)):
            raise ValueError(f"Corrected depth disagrees with recorded model: frame {i}")
        valid += int(np.count_nonzero(corrected))
        clipped += clipped_count
    return {"frames": count, "valid_depth_pixels": valid, "clipped_pixels": clipped,
            "source_files_unchanged": True}


def run(args):
    episode = args.input.expanduser().resolve()
    output, diagnostic = episode / CORRECTED_DIRECTORY, episode / DIAGNOSTIC_DIRECTORY
    for path in (output, diagnostic):
        if path.exists():
            raise FileExistsError(f"Result already exists: {path}")
    validate_episode(episode)
    meta = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    count = meta["frame_count"]
    if count < 8:
        raise ValueError("Need at least 8 frames for four temporal validation folds")
    original_sources = source_manifest(episode)
    urdf = args.urdf or Path(meta["urdf"])
    if args.urdf is None and not urdf.expanduser().is_file():
        urdf = DEFAULT_URDF
    model = RobotModel(urdf, joint_defaults=meta["unobserved_joint_values"])
    if file_digest(model.path) != meta["urdf_sha256"]:
        raise ValueError("URDF content differs from the episode")
    visual_sources = mesh_manifest(model)
    intrinsic = np.load(episode / "intrinsic" / f"{CAMERA}.npy", allow_pickle=False)
    extrinsic = np.load(episode / "extrinsic" / f"{CAMERA}.npy", allow_pickle=False)
    qpos = np.load(episode / "observations/joint_position.npy", allow_pickle=False)
    renderer = RobotDepthRenderer(model, intrinsic, extrinsic, threads=args.threads)
    frame_indices = np.linspace(0, count - 1, min(args.sample_frames, count), dtype=int)
    print(f"Collecting {len(frame_indices)} calibration frames; camera/URDF/joints fixed", flush=True)
    chunks, filter_counts = [], []
    for slot, i in enumerate(frame_indices):
        reference, links, cosine = renderer.render(qpos[i])
        samples, counts = select_correspondences(load_depth(episode, i), reference, links, cosine, i)
        chunks.append(samples)
        filter_counts.append(counts)
        if (slot + 1) % 12 == 0 or slot + 1 == len(frame_indices):
            print(f"Calibration frames {slot + 1}/{len(frame_indices)}", flush=True)
    training = Samples.concatenate(chunks)
    cv = cross_validate(training, frame_indices, renderer.link_names)
    parameters = fit_model(training)
    support = np.percentile(training.observed, [0, 1, 50, 99, 100]).tolist()
    report = {
        "format": CORRECTION_FORMAT, "accepted": cv["passed"],
        "metric_definition": "optical-axis Z residual in mm, using identical raw-selected correspondences",
        "cross_validation": cv, "fit_frame_indices": frame_indices.tolist(),
        "fit_metrics": compare_samples(training, parameters), "fit_filter_counts": filter_counts,
        "limitations": [
            "Correspondences exclude missing depth, boundaries, discontinuities, grazing angles and raw residuals >=80 mm.",
            "A global depth correction cannot resolve silhouette, model, timing or local surface discrepancies.",
            "Correction applies to every nonzero pixel; depths outside the fitted range are extrapolated.",
            "Parameters describe this episode and are not an independent sensor calibration.",
        ],
    }
    calibration = {
        "format": CORRECTION_FORMAT, "accepted": cv["passed"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": "global_affine_optical_axis_z",
        "equation": "z_corrected_m = scale * z_raw_m + bias_m",
        "scale": float(parameters[0]), "bias_m": float(parameters[1]),
        "quantization": "round to nearest mm (numpy.rint); valid pixels saturated to [1,65535]; zero preserved",
        "fit_settings": FIT_SETTINGS, "correspondence_filters": FILTERS,
        "fit_frame_indices": frame_indices.tolist(),
        "fit_depth_percentiles_m": dict(zip(("min", "p01", "median", "p99", "max"), support)),
        "source_episode": str(episode), "source_files_sha256": original_sources,
        "urdf_sha256": file_digest(model.path), "visual_meshes_sha256": visual_sources,
        "unobserved_joint_values": meta["unobserved_joint_values"],
    }
    with tempfile.TemporaryDirectory(prefix=".depths_corrected-", dir=episode) as temporary:
        staging = Path(temporary)
        if cv["passed"]:
            (staging / CAMERA).mkdir()
            audit_chunks, frames, panels = [], [], []
            preview_indices = set(np.linspace(0, count - 1, min(6, count), dtype=int).tolist())
            clipped_total, outside_support = 0, 0
            print(f"Accepted: scale={parameters[0]:.8f}, bias={parameters[1] * 1000:.3f} mm; "
                  f"exporting and auditing all {count} frames", flush=True)
            for i in range(count):
                raw = load_depth(episode, i)
                corrected, clipped = correct_depth_mm(raw, *parameters)
                clipped_total += clipped
                outside_support += int(np.count_nonzero(
                    (raw > 0) & ((raw / 1000 < support[0]) | (raw / 1000 > support[-1]))))
                if not cv2.imwrite(str(staging / CAMERA / f"{i:06d}.png"), corrected):
                    raise OSError(f"Cannot write corrected depth frame {i}")
                reference, links, cosine = renderer.render(qpos[i])
                samples, counts = select_correspondences(raw, reference, links, cosine, i)
                audit_chunks.append(samples)
                frames.append({**counts, "metrics": compare_samples(samples, parameters)})
                if i in preview_indices:
                    panels.append(comparison_panel(episode, i, raw, corrected, reference))
                if (i + 1) % 50 == 0 or i + 1 == count:
                    print(f"Exported/audited {i + 1}/{count} frames", flush=True)
            audit = Samples.concatenate(audit_chunks)
            per_link = {name: compare_samples(audit, parameters, audit.link == j)
                        for j, name in enumerate(renderer.link_names)}
            per_arm = {arm: compare_samples(audit, parameters, np.isin(audit.link, [
                j for j, name in enumerate(renderer.link_names) if name.startswith(prefix)]))
                       for arm, prefix in (("left", "fl_"), ("right", "fr_"))}
            regressions = [name for name, metrics in per_link.items() if metrics["raw"]["count"]
                           and any(metrics["corrected"][key] > metrics["raw"][key]
                                   for key in ("median_abs_mm", "p90_abs_mm"))]
            report["all_frame_audit"] = {
                "includes_fit_frames": True, "metrics": compare_samples(audit, parameters),
                "per_arm": per_arm, "per_link": per_link, "links_with_regressions": regressions,
                "per_frame": frames, "clipped_pixels": clipped_total,
                "pixels_outside_fit_depth_range": outside_support,
            }
            if not cv2.imwrite(str(staging / "comparison.jpg"), np.concatenate(panels)):
                raise OSError("Cannot write comparison image")
            depth_meta = json.loads((episode / "depths/metadata.json").read_text(encoding="utf-8"))
            depth_meta["correction"] = {"source": "../depths", "parameters": "calibration.json",
                                        "report": "report.json"}
            write_json(staging / "metadata.json", depth_meta)
        write_json(staging / "calibration.json", calibration)
        if cv["passed"]:
            report["output_validation"] = validate_corrected_depths(episode, staging, urdf=model.path)
            if (mesh_manifest(model) != visual_sources or source_manifest(episode) != original_sources):
                raise ValueError("Original source data changed during depth correction")
        write_json(staging / "report.json", report)
        destination = output if cv["passed"] else diagnostic
        if destination.exists():
            raise FileExistsError(f"Result already exists: {destination}")
        staging.rename(destination)
    print(f"Saved: {destination}", flush=True)
    if not cv["passed"]:
        raise ValueError("Temporal validation failed; only diagnostics were saved, no corrected depth PNGs")
    metrics = report["all_frame_audit"]["metrics"]
    print(f"All-frame selected-surface median: {metrics['raw']['median_abs_mm']:.2f} -> "
          f"{metrics['corrected']['median_abs_mm']:.2f} mm; P90: "
          f"{metrics['raw']['p90_abs_mm']:.2f} -> {metrics['corrected']['p90_abs_mm']:.2f} mm", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Converted episode directory")
    parser.add_argument("--urdf", type=Path, help="Relocated URDF with identical content")
    parser.add_argument("--sample-frames", type=int, default=48, help="Uniform frames used in four temporal folds")
    parser.add_argument("--threads", type=int, default=4, help="CPU raycast threads")
    args = parser.parse_args(argv)
    if args.sample_frames < 8 or args.threads < 1:
        parser.error("Require sample-frames >=8 and threads >=1")
    return args


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        raise SystemExit("Depth correction interrupted; unpublished staging files removed")
    except ImportError as error:
        raise SystemExit(f"Depth refinement dependency missing: {error}. Install open3d>=0.19.") from error
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
