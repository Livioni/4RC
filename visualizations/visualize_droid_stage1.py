#!/usr/bin/env python3
"""Export Stage 1 depth, per-window PCA, and per-frame and merged RGB point clouds."""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from droid_script.infer_4rc_stage1 import (
    ground_truth_queries, load_episode, load_model, load_views, validate_query,
)
from droid_script.rerun_visualizations.data import poses_from_states
from tcp_sliding_window_inference import resolve_device, resolve_dtype, write_json_atomic

DEFAULT_INPUT = ROOT / "datasets/droid_eval_sets/GuptaLab__Sun_Apr_30_12:45:39_2023"
WINDOW = 32
STRIDES = (4, 8)
TCP_COLORS = {"gt": (180, 60, 255), "pred": (255, 140, 0)}
AXIS_COLORS = ((255, 0, 0), (0, 0, 255), (0, 255, 0))  # X red, Y blue, Z green.
HEIGHT, WIDTH = 180, 320
PAD_H, PAD_W = 182, 322
PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])


def complete_groups(count: int, start: int, maximum: int) -> tuple[list[list[int]], list[int]]:
    if not 0 <= start < count or maximum < 0:
        raise ValueError("Start frame must be within the episode; max-frames must be nonnegative")
    end = min(count, start + maximum) if maximum else count
    groups = [list(range(i, min(i + WINDOW, end))) for i in range(start, end, WINDOW)]
    return groups, []


def validate_camera(k: np.ndarray, c2w: np.ndarray) -> None:
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError("Invalid pinhole intrinsic matrix")
    if (c2w.shape != (4, 4) or not np.isfinite(c2w).all()
            or not np.allclose(c2w[3], [0, 0, 0, 1])
            or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=2e-4)
            or not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=2e-4)):
        raise ValueError("Invalid camera-to-base rigid transform")


def native_ray_intrinsics(focal: np.ndarray, principal: np.ndarray) -> np.ndarray:
    """Ray solver uses a [0,2] image plane; compensate for RGB padding."""
    k = np.eye(3, dtype=np.float32)
    k[0, 0], k[1, 1] = focal * np.array([PAD_W, PAD_H]) / 2
    k[:2, 2] = principal * np.array([PAD_W, PAD_H]) / 2 - 1
    return k


def read_depth(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.array(image)
    if depth.shape != (HEIGHT, WIDTH) or depth.dtype != np.uint16:
        raise ValueError(f"Expected 180x320 uint16 millimeter depth: {path}")
    return depth.astype(np.float32) * 0.001


def backproject(depth: np.ndarray, rgb: np.ndarray, k: np.ndarray,
                c2w: np.ndarray, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    pixels = np.stack([xx[valid], yy[valid], np.ones(valid.sum())], axis=-1)
    camera = (pixels @ np.linalg.inv(k).T) * depth[valid, None]
    points = camera @ c2w[:3, :3].T + c2w[:3, 3]
    finite = np.isfinite(points).all(axis=-1)
    return points[finite].astype(np.float32), rgb[valid][finite]


def positions_to_base(positions: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Apply either one calibrated camera or a per-frame camera-to-base pose."""
    return (np.einsum("...ij,...j->...i", c2w[..., :3, :3], positions)
            + c2w[..., :3, 3]).astype(np.float32)


def validate_rotations(rotations: np.ndarray) -> None:
    if (rotations.shape[-2:] != (3, 3) or not np.isfinite(rotations).all()
            or not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-3)):
        raise ValueError("Expected finite proper TCP rotation matrices")


def rotations_to_base(rotations: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """R_base_tcp = R_base_camera @ R_camera_tcp."""
    result = (c2w[..., :3, :3] @ rotations).astype(np.float32)
    validate_rotations(result)
    return result


def pose_axes_cloud(positions: np.ndarray, rotations: np.ndarray,
                    length: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Point-sampled shafts and cone arrowheads in the TCP's positive X/Y/Z axes."""
    if (positions.shape != (len(rotations), 3) or rotations.shape != (len(positions), 3, 3)
            or not np.isfinite(positions).all()):
        raise ValueError("Expected finite TCP positions [N,3] and rotations [N,3,3]")
    validate_rotations(rotations)
    if not (np.isfinite(length) and np.isfinite(radius) and 0 < radius < length / 6):
        raise ValueError("Require 0 < TCP axis radius < axis length / 6")
    # At most 1 mm axial spacing. Three-radius cone bases distinguish arrow direction.
    shaft_end = 0.75 * length
    shaft_t = np.linspace(0, shaft_end, max(2, int(np.ceil(shaft_end / .001)) + 1))
    cone_t = np.linspace(shaft_end, length, max(2, int(np.ceil((length - shaft_end) / .001)) + 1))
    theta = np.arange(12) * (2 * np.pi / 12)
    local_clouds, local_colors = [], []
    for axis in range(3):
        direction, normal, binormal = np.eye(3)[[axis, (axis + 1) % 3, (axis + 2) % 3]]
        radial = np.cos(theta)[:, None] * normal + np.sin(theta)[:, None] * binormal
        shaft = shaft_t[:, None, None] * direction + radius * radial
        cone_radius = 3 * radius * (length - cone_t) / (length - shaft_end)
        cone = cone_t[:, None, None] * direction + cone_radius[:, None, None] * radial
        centerline = np.r_[shaft_t, cone_t][:, None] * direction
        local = np.concatenate([centerline, shaft.reshape(-1, 3), cone.reshape(-1, 3)])
        local_clouds.append(local)
        local_colors.append(np.tile(np.array(AXIS_COLORS[axis], np.uint8), (len(local), 1)))
    local = np.concatenate(local_clouds)
    world = np.einsum("nij,pj->npi", rotations, local) + positions[:, None]
    return world.reshape(-1, 3).astype(np.float32), np.tile(np.concatenate(local_colors), (len(positions), 1))


def interpolate_trajectory(positions: np.ndarray, spacing: float) -> list[np.ndarray]:
    """Piecewise linear interpolation, retaining samples and never bridging NaNs."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("TCP spacing must be finite and positive")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Expected TCP positions [frames,3]")
    valid = np.isfinite(positions).all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    runs = []
    for start, end in edges.reshape(-1, 2):
        original = positions[start:end]
        samples = [original[:1]]
        for a, b in zip(original[:-1], original[1:]):
            steps = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
            samples.append(a + np.linspace(0, 1, steps + 1)[1:, None] * (b - a))
        runs.append(np.concatenate(samples).astype(np.float32))
    return runs


def trajectory_cloud(positions: np.ndarray, color, spacing: float,
                     radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Sample a colored tube as vertices; plain point-cloud PLY readers need no edges."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("TCP radius must be finite and positive")
    runs = interpolate_trajectory(positions, spacing)
    clouds = []
    theta = np.arange(12) * (2 * np.pi / 12)
    for centers in runs:
        tangent = np.zeros_like(centers)
        if len(centers) > 1:
            tangent[0], tangent[-1] = centers[1] - centers[0], centers[-1] - centers[-2]
            tangent[1:-1] = centers[2:] - centers[:-2]
        lengths = np.linalg.norm(tangent, axis=1)
        tangent[lengths < 1e-8] = [0, 0, 1]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        axis = np.eye(3)[np.argmin(np.abs(tangent), axis=1)]
        normal = np.cross(tangent, axis)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        binormal = np.cross(tangent, normal)
        rings = centers[:, None] + radius * (
            np.cos(theta)[None, :, None] * normal[:, None]
            + np.sin(theta)[None, :, None] * binormal[:, None]
        )
        clouds.extend([centers, rings.reshape(-1, 3)])
    # Visible spherical samples mark the original per-frame TCP locations.
    n = 48
    z = 1 - 2 * (np.arange(n) + .5) / n
    angle = np.arange(n) * np.pi * (3 - np.sqrt(5))
    unit = np.stack([np.sqrt(1 - z*z) * np.cos(angle),
                     np.sqrt(1 - z*z) * np.sin(angle), z], axis=-1)
    originals = positions[np.isfinite(positions).all(axis=1)]
    if len(originals):
        clouds.append((originals[:, None] + 1.5 * radius * unit).reshape(-1, 3))
    points = np.concatenate(clouds).astype(np.float32) if clouds else np.empty((0, 3), np.float32)
    colors = np.tile(np.asarray(color, np.uint8), (len(points), 1))
    return points, colors


def write_ply(path: Path, clouds: list[tuple[np.ndarray, np.ndarray]], *,
              content: str = "Scene RGB, TCP trajectory (GT purple, prediction orange), then stride TCP pose axes") -> int:
    count = sum(len(points) for points, _ in clouds)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment XYZ in meters in robot base coordinates\n"
        f"comment {content}\n"
        "comment TCP pose axes: X red, Y blue, Z green; arrows point along positive local axes\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for points, colors in clouds:
            vertices = np.empty(len(points), dtype=PLY_DTYPE)
            for i, name in enumerate(("x", "y", "z")):
                vertices[name] = points[:, i]
            for i, name in enumerate(("red", "green", "blue")):
                vertices[name] = colors[:, i]
            vertices.tofile(stream)
    return count


def fit_pca(features: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Fit and color one window jointly; never share statistics across windows."""
    from sklearn.decomposition import PCA

    if not np.isfinite(features).all():
        raise ValueError("Nonfinite backbone features")
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    pca.fit(flat)
    scores = pca.transform(flat)
    low, high = np.percentile(scores, [1, 99], axis=0)
    span = high - low
    colors = np.clip((scores - low) / np.where(span > 1e-8, span, 1), 0, 1)
    colors[:, span <= 1e-8] = 0.5
    params = dict(mean=pca.mean_, components=pca.components_,
                  explained_variance_ratio=pca.explained_variance_ratio_,
                  color_low=low, color_high=high)
    return colors.reshape(*features.shape[:-1], 3).astype(np.float32), params


def pca_image(colors: np.ndarray) -> Image.Image:
    patches = torch.from_numpy(colors.reshape(13, 23, 3)).permute(2, 0, 1)[None]
    expanded = F.interpolate(patches, (PAD_H, PAD_W), mode="bilinear", align_corners=False)
    native = expanded[0, :, 1:-1, 1:-1].permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.clip(native * 255, 0, 255)))


def depth_image(depth: np.ndarray, max_depth: float) -> Image.Image:
    from matplotlib import colormaps

    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.where(valid, depth / max_depth, 0).clip(0, 1)
    pixels = (colormaps["turbo"](normalized)[..., :3] * 255).astype(np.uint8)
    pixels[~valid] = 0
    return Image.fromarray(pixels)


def save_images(output: Path, frame: int, rgb: np.ndarray, gt: np.ndarray,
                pred: np.ndarray, colors: np.ndarray, max_depth: float) -> None:
    images = [Image.fromarray(rgb), depth_image(gt, max_depth),
              depth_image(pred, max_depth), pca_image(colors)]
    names = ["rgb", "gt_depth", "pred_depth", "pca"]
    for name, image in zip(names, images):
        image.save(output / name / f"{frame:06d}.png")
    canvas = Image.new("RGB", (WIDTH * 4, HEIGHT + 72), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["RGB", "GT depth (m)", "Predicted depth (m)", "Patchify PCA (pre-pos)"]
    for i, (image, title) in enumerate(zip(images, titles)):
        canvas.paste(image, (WIDTH * i, 26))
        draw.text((WIDTH * i + 6, 7), f"{title} | frame {frame:06d}", fill="black")
    ramp = np.linspace(0, max_depth, WIDTH - 40, dtype=np.float32)
    bar = depth_image(np.tile(ramp, (10, 1)), max_depth)
    for offset in (WIDTH, WIDTH * 2):
        canvas.paste(bar, (offset + 20, HEIGHT + 34))
        draw.text((offset + 20, HEIGHT + 48), "0 m", fill="black")
        draw.text((offset + WIDTH - 65, HEIGHT + 48), f"{max_depth:g} m", fill="black")
    draw.text((6, HEIGHT + 36), "Black depth: invalid; above range: saturated", fill="black")
    canvas.save(output / "comparison" / f"{frame:06d}.png")


def infer_group(model, paths, indices, query, device, dtype, seed):
    from arc.models.arc.utils.transform import get_extrinsic_from_camray

    views, rgb = load_views(paths)
    views = [{**v, "img": v["img"].to(device)} for v in views]
    query_tensor = torch.from_numpy(validate_query(query) + np.float32(1))[None].to(device)
    autocast = (contextlib.nullcontext() if dtype == torch.float32 else
                torch.autocast(device_type=device.type, dtype=dtype))
    def cpu(value):
        return value.detach().float().cpu().numpy().copy()
    patch_features = []

    def capture_patch_embed(_module, _inputs, output):
        # Capture before positional embeddings, special tokens or attention.
        # PatchEmbed flattens batch and frames; inference here has batch size 1.
        patch_features.append(cpu(output))

    hook = model.backbone.pretrained.patch_embed.register_forward_hook(capture_patch_embed)
    try:
        with torch.inference_mode(), autocast:
            pred = model(
                views, force_no_output_conversion=True, inference_track=False,
                decode_camera=False, decode_motion=False, decode_tcp=True,
                tcp_query_points=query_tensor,
                return_aux_pyramid=False, return_backbone_features=False, ref_view_strategy="first",
            )
    finally:
        hook.remove()
    if len(patch_features) != 1:
        raise ValueError(f"Expected one patch embedding call, got {len(patch_features)}")
    features = patch_features.pop()
    depth = cpu(pred["depth"][0, :, 1:-1, 1:-1])
    confidence = cpu(pred["depth_conf"][0, :, 1:-1, 1:-1])
    tcp_camera = cpu(pred["tcp_position"][0, :, 0])
    tcp_rotation = cpu(pred["tcp_rotation"][0, :, 0])
    tcp_confidence = cpu(pred["tcp_confidence"][0, :, 0])
    count = len(indices)
    if tcp_camera.shape != (count, 3) or not np.isfinite(tcp_camera).all():
        raise ValueError(f"Invalid predicted TCP positions in group {indices[0]}-{indices[-1]}")
    if tcp_confidence.shape != (count,) or not np.isfinite(tcp_confidence).all():
        raise ValueError(f"Invalid predicted TCP confidence in group {indices[0]}-{indices[-1]}")
    if tcp_rotation.shape != (count, 3, 3):
        raise ValueError(f"Invalid predicted TCP rotation shape: {tcp_rotation.shape}")
    validate_rotations(tcp_rotation)
    if features.shape != (count, 299, 1536):
        raise ValueError(f"Unexpected patch embedding shape: {features.shape}")
    ray, ray_conf = pred["ray"].float(), pred["ray_conf"].float()
    del pred
    intrinsics, cameras = [], []
    for slot, frame in enumerate(indices):
        try:
            # The existing solver modifies confidence weights in place.
            torch.manual_seed(seed + frame)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
                c2w, focal, principal = get_extrinsic_from_camray(
                    ray[:, slot:slot + 1], ray_conf[:, slot:slot + 1].clone(),
                    ray.shape[-3], ray.shape[-2],
                )
            k = native_ray_intrinsics(cpu(focal[0, 0]), cpu(principal[0, 0]))
            camera = cpu(c2w[0, 0])
            validate_camera(k, camera)
            intrinsics.append(k)
            cameras.append(camera)
        except (ValueError, RuntimeError) as error:
            raise RuntimeError(f"Ray calibration failed: group {indices[0]}-{indices[-1]}, frame {frame}") from error
    return (rgb, depth, confidence, features, np.stack(intrinsics), np.stack(cameras),
            tcp_camera, tcp_confidence, tcp_rotation)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--camera", default="22008760")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/Droid-Stage1/checkpoint-500000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0: all remaining; includes incomplete final group")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-depth", type=float, default=3.0, help="Depth color range and PLY cutoff, meters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tcp-spacing", type=float, default=0.002, help="Maximum TCP interpolation spacing, meters")
    parser.add_argument("--tcp-radius", type=float, default=0.003, help="Radius of TCP trajectory point tube, meters")
    parser.add_argument("--tcp-axis-length", type=float, default=0.03, help="TCP pose arrow length, meters")
    parser.add_argument("--tcp-axis-radius", type=float, default=0.001, help="TCP pose arrow shaft radius, meters")
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_depth) or args.max_depth <= 0:
        parser.error("--max-depth must be finite and positive")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be in [0, 2**32)")
    for name in ("tcp_spacing", "tcp_radius", "tcp_axis_length", "tcp_axis_radius"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.tcp_axis_radius >= args.tcp_axis_length / 6:
        parser.error("--tcp-axis-radius must be smaller than --tcp-axis-length / 6")
    return args


def run(args):
    episode = load_episode(args.input, args.camera)
    groups, skipped = complete_groups(len(episode.image_paths), args.start_frame, args.max_frames)
    gt_c2w = np.linalg.inv(np.load(episode.path / "extrinsic" / f"{episode.camera}.npy", allow_pickle=False))
    validate_camera(episode.intrinsics, gt_c2w)
    queries, query_valid = ground_truth_queries(episode)
    gt_state = np.load(episode.path / "TCP" / episode.camera / "state.npy", allow_pickle=False)
    gt_tcp_camera = gt_state[:, :3]
    gt_rotation_camera = poses_from_states(gt_state)[:, :3, :3]
    for indices in groups:
        if not query_valid[indices[0]]:
            raise ValueError(f"TCP must project inside the first image of each window; invalid frame {indices[0]}")
    # Check all selected depths before allocating the model.
    for frame in (i for group in groups for i in group):
        read_depth(episode.path / "depths" / episode.camera / episode.image_paths[frame].name)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output = (args.output or ROOT / "outputs/visualizations/droid_stage1" /
              episode.path.name / episode.camera).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; use a new --output path: {output}")
    for name in ("rgb", "gt_depth", "pred_depth", "pca", "comparison", "groups"):
        (output / name).mkdir(parents=True, exist_ok=True)
    manifest = dict(
        status="running", input=str(episode.path), camera=episode.camera,
        checkpoint=str(args.checkpoint.expanduser().resolve()), output=str(output),
        device=str(device), dtype=str(dtype), seed=args.seed, window_size=WINDOW,
        processed_frame_indices=[i for group in groups for i in group], skipped_frame_indices=skipped,
        depth_unit="meter", depth_color_range=[0, args.max_depth],
        ply_depth_range=f"0 < depth <= {args.max_depth}",
        coordinate_frame="robot_base", gt_calibration="dataset",
        pred_calibration="predicted ray + ray_conf; no camera decoder",
        pca_feature="patch_embed output: 14x14 projected patches, before positional embeddings and attention",
        pca_scope=f"per {WINDOW}-frame window",
        point_cloud_strides=list(STRIDES),
        per_frame_ply=True,
        tcp=dict(query_source="GT TCP projected with GT intrinsics at each window's first frame",
                 coordinate_frame="robot_base", interpolation="piecewise linear",
                 spacing_m=args.tcp_spacing, tube_radius_m=args.tcp_radius,
                 marker_radius_m=1.5 * args.tcp_radius,
                 colors=TCP_COLORS, trajectory_frame_stride=1,
                 pose_axes=dict(colors=dict(zip(("x", "y", "z"), AXIS_COLORS)),
                                length_m=args.tcp_axis_length, shaft_radius_m=args.tcp_axis_radius,
                                frames="current frame in per-frame PLY; stride-selected frames in merged PLY",
                                rotation_convention="R_base_tcp = R_base_camera @ R_camera_tcp",
                                gt_rpy_convention="Rz(yaw) @ Ry(pitch) @ Rx(roll)"),
                 prediction_transform="per-frame ray-derived camera-to-base"),
        groups=[],
    )
    write_json_atomic(manifest, output / "manifest.json")
    print(f"Processing {len(groups)} groups / {sum(map(len, groups))} frames; skipped {skipped}", flush=True)
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, device)
    for number, indices in enumerate(groups):
        print(f"[{number+1}/{len(groups)}] frames {indices[0]:06d}-{indices[-1]:06d}", flush=True)
        paths = [episode.image_paths[i] for i in indices]
        rgb, depth, confidence, features, pred_k, pred_c2w, tcp_camera, tcp_confidence, tcp_rotation = infer_group(
            model, paths, indices, queries[indices[0]], device, dtype, args.seed)
        gt_depth = np.stack([read_depth(episode.path / "depths" / episode.camera / p.name) for p in paths])
        gt_tcp = positions_to_base(gt_tcp_camera[indices], gt_c2w)
        pred_tcp = positions_to_base(tcp_camera, pred_c2w)
        gt_rotation = rotations_to_base(gt_rotation_camera[indices], gt_c2w)
        pred_rotation = rotations_to_base(tcp_rotation, pred_c2w)
        group_dir = output / "groups" / f"{indices[0]:06d}_{indices[-1]:06d}"
        group_dir.mkdir()
        np.savez_compressed(
            group_dir / "arrays.npz", frame_indices=indices, gt_depth=gt_depth,
            pred_depth=depth, pred_depth_confidence=confidence, gt_intrinsics=episode.intrinsics,
            gt_camera_to_base=gt_c2w, pred_intrinsics=pred_k, pred_camera_to_base=pred_c2w,
            gt_tcp_camera=gt_tcp_camera[indices], gt_tcp_base=gt_tcp,
            pred_tcp_camera=tcp_camera, pred_tcp_base=pred_tcp, pred_tcp_confidence=tcp_confidence,
            gt_tcp_rotation_camera=gt_rotation_camera[indices], gt_tcp_rotation_base=gt_rotation,
            pred_tcp_rotation_camera=tcp_rotation, pred_tcp_rotation_base=pred_rotation,
            tcp_query_pixel=queries[indices[0]],
        )
        clouds = {
            "gt": [backproject(d, c, episode.intrinsics, gt_c2w, args.max_depth) for d, c in zip(gt_depth, rgb)],
            "pred": [backproject(d, c, k, e, args.max_depth) for d, c, k, e in zip(depth, rgb, pred_k, pred_c2w)],
        }
        trajectories = {
            source: trajectory_cloud(positions, TCP_COLORS[source], args.tcp_spacing, args.tcp_radius)
            for source, positions in (("gt", gt_tcp), ("pred", pred_tcp))
        }
        record = dict(frame_indices=indices, directory=str(group_dir.relative_to(output)),
                      tcp_query_frame=indices[0], tcp_query_pixel=queries[indices[0]].tolist(), ply={})
        for source, frames in clouds.items():
            positions, rotations = (gt_tcp, gt_rotation) if source == "gt" else (pred_tcp, pred_rotation)
            for slot, frame in enumerate(indices):
                filename = f"{source}_{frame:06d}.ply"
                marker = trajectory_cloud(positions[slot:slot + 1], TCP_COLORS[source],
                                          args.tcp_spacing, args.tcp_radius)
                axes = pose_axes_cloud(positions[slot:slot + 1], rotations[slot:slot + 1],
                                       args.tcp_axis_length, args.tcp_axis_radius)
                count = write_ply(
                    group_dir / filename, [frames[slot], marker, axes],
                    content=f"Frame {frame}: {source} scene RGB, TCP marker, then TCP pose axes")
                record["ply"][filename] = dict(
                    frame_indices=[frame], vertex_count=count,
                    scene_vertex_count=len(frames[slot][0]),
                    tcp_vertex_count=len(marker[0]), tcp_frame_indices=[frame],
                    pose_axis_vertex_count=len(axes[0]), pose_frame_indices=[frame],
                    vertex_order="scene, TCP marker, then TCP pose axes",
                )
            for stride in STRIDES:
                filename = f"{source}_stride{stride}.ply"
                selected = frames[::stride]
                axes = pose_axes_cloud(positions[::stride], rotations[::stride],
                                       args.tcp_axis_length, args.tcp_axis_radius)
                count = write_ply(group_dir / filename, selected + [trajectories[source], axes])
                record["ply"][filename] = dict(
                    frame_indices=indices[::stride], vertex_count=count,
                    scene_vertex_count=sum(len(points) for points, _ in selected),
                    tcp_vertex_count=len(trajectories[source][0]), tcp_frame_indices=indices,
                    pose_axis_vertex_count=len(axes[0]), pose_frame_indices=indices[::stride],
                    vertex_order="scene, TCP trajectory and markers, then TCP pose axes",
                )
        colors, pca_params = fit_pca(features, args.seed)
        np.savez_compressed(group_dir / "pca_parameters.npz", **pca_params)
        record["pca_parameters"] = str((group_dir / "pca_parameters.npz").relative_to(output))
        record["pca_explained_variance_ratio"] = pca_params["explained_variance_ratio"].tolist()
        for slot, frame in enumerate(indices):
            save_images(output, frame, rgb[slot], gt_depth[slot], depth[slot],
                        colors[slot], args.max_depth)
        del features, colors
        print(f"Independent window PCA and images saved: group {number+1}/{len(groups)}", flush=True)
        manifest["groups"].append(record)
        write_json_atomic(manifest, output / "manifest.json")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    manifest["status"] = "complete"
    write_json_atomic(manifest, output / "manifest.json")
    print(f"Done: {output}", flush=True)
    return output


if __name__ == "__main__":
    run(parse_args())
