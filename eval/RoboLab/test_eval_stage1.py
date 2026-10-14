"""CPU-only integration tests for RoboLab adapters and DROID metric parity."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.Droid import eval_stage1 as droid
from eval.RoboLab import eval_stage1 as robolab


CAMERA = "third_person"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_episode(path, *, dataset="robolab", count=6, status="failure", static=False, hidden=False):
    for directory in ("images", "depths", "TCP"):
        (path / directory / CAMERA).mkdir(parents=True)
    for directory in ("intrinsic", "extrinsic"):
        (path / directory).mkdir()
    meta = {"frame_count": count, "fps": 15, "status": status}
    if dataset == "robolab":
        meta.update(format="robolab_droid_like_v2", extrinsics="base_to_camera")
    write_json(path / "metadata.json", meta)
    tcp_meta = {
        "columns": ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open"],
        "position_unit": "meter", "rotation_unit": "radian",
        "rpy_convention": "fixed-axis XYZ (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))",
        "coordinate_frame": "OpenCV (+x right, +y down, +z forward)",
        "tcp_definition_id": "droid_generate_tcp_v2",
    }
    if dataset == "droid":
        tcp_meta.update(camera_id=CAMERA, coordinate_frame="OpenCV camera (+x right, +y down, +z forward)",
                        extrinsics="world_to_camera, applied directly to world TCP pose")
    write_json(path / "TCP" / CAMERA / "metadata.json", tcp_meta)
    write_json(path / "depths/metadata.json", {
        "units": "millimeters", "depth_type": "distance_to_image_plane",
        "cameras": {CAMERA: {"frames": count, "width": 320, "height": 180}},
    })
    k = np.array([[200., 0, 160], [0, 180, 90], [0, 0, 1]])
    np.save(path / "intrinsic" / f"{CAMERA}.npy", k)
    np.save(path / "extrinsic" / f"{CAMERA}.npy", np.eye(4))
    t = np.arange(count)
    states = np.zeros((count, 7))
    states[:, :3] = np.stack([.03 * np.sin(t), .04 * np.cos(t), 1 + .01 * t], axis=-1)
    states[:, 6] = np.linspace(.1, .9, count)
    if static:
        states[:, :3] = [0, 0, 1]
    if hidden:
        states[:, 2] = -1
    np.save(path / "TCP" / CAMERA / "state.npy", states)
    np.save(path / "timestamps.npy", t / 15.)
    for i in range(count):
        Image.fromarray(np.zeros((180, 320, 3), dtype=np.uint8)).save(path / "images" / CAMERA / f"{i:06d}.png")
        Image.fromarray(np.full((180, 320), 1000, dtype=np.uint16)).save(path / "depths" / CAMERA / f"{i:06d}.png")
    return path


@pytest.fixture(autouse=True)
def single_process(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")


@pytest.fixture
def fake_inference(monkeypatch):
    calls = []

    def infer(model, paths, query, device, dtype):
        path = paths[0].parents[2]
        ids = np.array([int(p.stem) for p in paths])
        calls.append((path.name, ids.tolist(), query.copy()))
        states = np.load(path / "TCP" / CAMERA / "state.npy")[ids]
        k = np.load(path / "intrinsic" / f"{CAMERA}.npy")
        n = len(paths)
        offset = {"moving": .01, "short": .03, "static": .02}.get(path.name, .01)
        positions = states[:, :3].copy()
        positions[:, 0] += offset
        if path.name == "moving":
            # Frame 2 is the shared window boundary: the previous window wins.
            if ids[0] == 0:
                positions[ids == 2, 0] = 5
            positions[ids == 3, 2] = -1
        enc = np.zeros((n, 9))
        enc[:, 7:] = [2 * np.arctan(182 / (2 * k[1, 1])), 2 * np.arctan(322 / (2 * k[0, 0]))]
        rays, _ = droid.ray_target(k, np.eye(4), 4, 6)
        return {
            "tcp_position": positions[:, None].astype(np.float32),
            "tcp_rotation": np.broadcast_to(np.eye(3), (n, 1, 3, 3)).copy(),
            "tcp_gripper": states[:, 6, None].copy(), "tcp_confidence": np.ones((n, 1)),
            "camera_to_world": np.broadcast_to(np.eye(4), (n, 4, 4)).copy(),
            "intrinsics": np.broadcast_to(k, (n, 3, 3)).copy(), "pose_enc": enc,
            "depth": np.full((n, 182, 322), 1 + offset),
            "ray": np.broadcast_to(rays, (n, *rays.shape)).copy(),
        }

    monkeypatch.setattr(droid, "infer_window", infer)
    monkeypatch.setattr(droid.stage1, "load_model", lambda *a: object())
    return calls


def args_for(tmp_path, module, root, output="results", extra=()):
    weights = tmp_path / "model.safetensors"
    if not weights.exists():
        weights.write_bytes(b"mock weights; model loading is replaced in tests")
    args = ["--data-root", str(root), "--model", str(weights), "--output-dir", str(tmp_path / output),
            "--device", "cpu", "--dtype", "float32", "--window-size", "3"]
    return module.parse_args([*args, *extra])


def test_discovery_and_cli(tmp_path):
    root = tmp_path / "data"
    make_episode(root / "BTask" / "failure", status="failure")
    make_episode(root / "ATask" / "success", status="success")
    args = args_for(tmp_path, robolab, root)
    assert [j["episode"] for j in robolab.discover_jobs(args)] == ["success", "failure"]
    assert args.val_set is None
    args.max_episodes = 1
    assert robolab.discover_jobs(args)[0]["task"] == "ATask"
    args.episodes = ["failure"]
    assert robolab.discover_jobs(args)[0]["episode"] == "failure"
    args.episodes = ["missing"]
    with pytest.raises(ValueError, match="not found"):
        robolab.discover_jobs(args)
    args.episodes = None
    args.cameras = ["missing"]
    with pytest.raises(ValueError, match="No RoboLab cameras"):
        robolab.discover_jobs(args)
    (root / "CTask" / "failure").mkdir(parents=True)
    with pytest.raises(ValueError, match="Duplicate episode"):
        robolab.discover_jobs(args)
    for options in (["--window-size", "1"], ["--max-frames", "1"], ["--rpe-delta-frames", "0"],
                    ["--val-set", "unused"], ["--recompute-tcp-metrics-from", "cache", "--resume"]):
        with pytest.raises(SystemExit):
            robolab.parse_args(options)
    original = droid.parse_args([])
    assert original.val_set.name == "eval_set.txt"
    assert original.window_size == 9 and original.rpe_delta_frames == 1


def test_dry_run_and_discovery_failure(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    make_episode(root / "Task" / "ok")
    (root / "Task" / "broken").mkdir()
    args = args_for(tmp_path, robolab, root, extra=["--dry-run"])
    args.model.unlink()  # Dry-run must not require weights or create results.
    monkeypatch.setattr(droid.stage1, "load_model", lambda *a: pytest.fail("unexpected model load"))
    robolab.run(args)
    assert "discovery failures: 1" in capsys.readouterr().out
    assert not args.output_dir.exists()
    jobs = robolab.discover_jobs(args)
    assert jobs[0]["camera"] == "__discovery__"


def test_robolab_labels_and_units(tmp_path):
    path = make_episode(tmp_path / "Task" / "episode")
    ep = droid.stage1.load_episode(path, CAMERA)
    states, extrinsic = robolab.read_labels(ep)
    assert states[0, 6] == .1  # Already openness, never invert the label.
    np.testing.assert_equal(extrinsic, np.eye(4))
    tcp_meta_path = path / "TCP" / CAMERA / "metadata.json"
    original = json.loads(tcp_meta_path.read_text())
    write_json(tcp_meta_path, {**original, "coordinate_frame": "OpenCV camera (+x right, +y down, +z forward)"})
    robolab.read_labels(ep)
    for key, value in (("position_unit", "millimeter"), ("rotation_unit", "degree"),
                       ("coordinate_frame", "world"), ("tcp_definition_id", "other"),
                       ("columns", ["closedness"])):
        write_json(tcp_meta_path, {**original, key: value})
        with pytest.raises(ValueError, match=key):
            robolab.read_labels(ep)
    write_json(tcp_meta_path, original)
    with pytest.raises(ValueError, match="expects DROID"):
        droid.read_labels(ep)
    depth_path = path / "depths/metadata.json"
    depth = json.loads(depth_path.read_text())
    write_json(depth_path, {**depth, "depth_type": "distance_to_camera"})
    with pytest.raises(ValueError, match="optical-axis"):
        robolab.read_labels(ep)
    write_json(depth_path, depth)
    np.save(path / "extrinsic" / f"{CAMERA}.npy", np.zeros((4, 4)))
    with pytest.raises(ValueError, match="rigid"):
        robolab.read_labels(ep)


def test_droid_robolab_metrics_match_end_to_end(tmp_path, fake_inference):
    scenes = [("moving", 6, {}), ("short", 3, {"status": "success"}),
              ("static", 4, {"static": True}), ("skip", 5, {"hidden": True}), ("missing", 3, {})]
    summaries, outputs = [], []
    for dataset, module in (("droid", droid), ("robolab", robolab)):
        root = tmp_path / dataset
        for name, count, options in scenes:
            path = root / name if dataset == "droid" else root / "Task" / name
            make_episode(path, dataset=dataset, count=count, **options)
            if name == "missing":
                (path / "depths" / CAMERA / "000000.png").unlink()
        extra = []
        if dataset == "droid":
            split = tmp_path / "val.txt"
            split.write_text("\n".join(sorted(name for name, _, _ in scenes)))
            extra = ["--val-set", str(split)]
        args = args_for(tmp_path, module, root, output=f"{dataset}_results", extra=extra)
        summaries.append(module.run(args))
        outputs.append(args.output_dir)
    for key in ("frame_metrics", "depth_metrics", "ray_metrics", "tracking_metrics", "coverage", "metric_definitions"):
        assert summaries[0][key] == summaries[1][key], key
    summary = summaries[1]
    assert summary["coverage"]["complete_camera_count"] == 3
    assert summary["coverage"]["skipped_camera_count"] == 1
    assert summary["coverage"]["failed_camera_count"] == 1
    rows = {row["aggregation"]: row for row in summary["frame_metrics"]}
    assert rows["camera_macro"]["tcp_position_m_rmse"] == pytest.approx(.02, abs=1e-7)
    assert rows["frame_micro"]["tcp_position_m_rmse"] == pytest.approx(np.sqrt((4*.01**2 + 3*.03**2 + 4*.02**2)/11), abs=1e-7)
    assert rows["frame_micro"]["tcp_sim3_alignment_undefined_count"] == 1
    folder = outputs[1] / "cameras/moving" / CAMERA
    record = json.loads((folder / "metrics.json").read_text())
    tracking = record["tracking_metrics"]
    assert tracking["tcp_tracking_frame_count"] == 5
    assert tracking["tcp_tracking_unprojectable_frame_count"] == 1
    assert tracking["tcp_tracking_prediction_out_of_view_frame_count"] == 2
    assert tracking["tcp_tracking_pts_within_16px"] == pytest.approx(3/5)
    assert record["windows"][1]["query_source"]["frame_index"] == 1
    with np.load(folder / "frames.npz") as arrays:
        np.testing.assert_array_equal(arrays["metric_mask"], [True, True, False, False, True, True])
        np.testing.assert_array_equal(arrays["source_window"], [0, 0, 0, 1, 1, 2])
        assert arrays["tcp_position"][2, 0] == 5  # Shared frame retains previous prediction.
    for csv_path in outputs[0].glob("*.csv"):
        if csv_path.name in ("camera_metrics.csv", "tracking_camera_metrics.csv"):
            continue  # Failure reason contains the source filesystem path.
        assert csv_path.read_text() == (outputs[1] / csv_path.name).read_text()


def test_resume_fingerprints_and_recompute(tmp_path, fake_inference, monkeypatch):
    root = tmp_path / "data"
    path = make_episode(root / "Task" / "moving")
    args = args_for(tmp_path, robolab, root)
    original = robolab.run(args)
    record_path = args.output_dir / "cameras/moving" / CAMERA / "metrics.json"
    record_bytes = record_path.read_bytes()
    calls = len(fake_inference)
    args.resume = True
    robolab.run(args)
    assert len(fake_inference) == calls
    assert record_path.read_bytes() == record_bytes
    timestamps = path / "timestamps.npy"
    stat = timestamps.stat()
    os.utime(timestamps, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    robolab.run(args)
    assert len(fake_inference) == 2*calls
    (record_path.parent / "frames.npz").write_bytes(b"corrupt zip")
    robolab.run(args)
    assert len(fake_inference) == 3*calls
    record_path.write_text("{")
    robolab.run(args)
    assert len(fake_inference) == 4*calls
    args.window_size = 4
    with pytest.raises(ValueError, match="configuration changed"):
        robolab.run(args)
    monkeypatch.setattr(droid.stage1, "load_model", lambda *a: pytest.fail("offline model load"))
    monkeypatch.setattr(robolab, "discover_jobs", lambda *a: pytest.fail("offline data discovery"))
    for i in range(2):
        source = args.output_dir if i == 0 else tmp_path / "recomputed0"
        target = tmp_path / f"recomputed{i}"
        offline = robolab.parse_args(["--recompute-tcp-metrics-from", str(source), "--output-dir", str(target)])
        summary = robolab.run(offline)
        for key in ("frame_metrics", "depth_metrics", "ray_metrics", "tracking_metrics", "coverage"):
            assert summary[key] == original[key], key
        directory = target / "cameras/moving" / CAMERA
        assert (directory / "trajectory_errors.npz").exists()
        assert (directory / "tracking_errors.npz").exists()
        assert not (directory / "frames.npz").exists()
