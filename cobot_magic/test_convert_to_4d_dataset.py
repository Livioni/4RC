"""Check RGB export, existing-data selection and in-place TCP recalculation."""
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from cobot_magic import convert_to_4d_dataset as converter
from cobot_magic.geometry import CAMERA, DEFAULT_URDF, ROOT, RobotModel, read_calibration


@pytest.mark.parametrize("mode", ["--recompute-tcp", "--validate-only", "--append-wrist-rgb"])
@pytest.mark.parametrize("flag", ["--input", "--output"])
@pytest.mark.parametrize("level", ["dataset", "task", "episode"])
def test_existing_data_selection_handles_supported_directory_levels(tmp_path, monkeypatch, mode, flag, level):
    dataset = tmp_path / "dataset"
    task = dataset / "place_pen_in_cups"
    episodes = [task / f"place_pen_in_cups__episode_{i}" for i in range(2)]
    for episode in episodes:
        episode.mkdir(parents=True)
        (episode / "metadata.json").write_text("{}")
        # These files describe internal arrays and must not be treated as episodes.
        (episode / "depths").mkdir()
        (episode / "depths/metadata.json").write_text("{}")
        (episode / f"TCP/{CAMERA}").mkdir(parents=True)
        (episode / f"TCP/{CAMERA}/metadata.json").write_text("{}")
    target = {"dataset": dataset, "task": task, "episode": episodes[0]}[level]
    calibration = tmp_path / "calibration.yaml"
    calls = []

    def recompute(path, calibration_path, **kwargs):
        assert calibration_path == calibration
        calls.append(path)
        return {"frames": 2}

    def validate(path):
        calls.append(path)
        return {"frames": 2}

    def append(path):
        calls.append(path)
        return {"written": {"left_view": 0, "right_view": 0},
                "skipped": {"left_view": 2, "right_view": 2}}

    monkeypatch.setattr(converter, "recompute_episode", recompute)
    monkeypatch.setattr(converter, "validate_episode", validate)
    monkeypatch.setattr(converter, "inspect_wrist_append", lambda path: {})
    monkeypatch.setattr(converter, "append_wrist_rgb", append)
    converter.main([mode, flag, str(target), "--calibration", str(calibration)])
    assert calls == (episodes[:1] if level == "episode" else episodes)


@pytest.mark.parametrize("mode", ["--recompute-tcp", "--validate-only", "--append-wrist-rgb"])
def test_existing_data_modes_reject_different_input_and_output(tmp_path, mode):
    with pytest.raises(SystemExit) as error:
        converter.parse_args([mode, "--input", str(tmp_path / "first"),
                              "--output", str(tmp_path / "second")])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["--recompute-tcp", "--validate-only", "--append-wrist-rgb"])
def test_existing_data_modes_accept_matching_input_and_output(tmp_path, mode):
    args = converter.parse_args([mode, "--input", str(tmp_path), "--output", str(tmp_path)])
    assert args.output == tmp_path


def test_conversion_defaults_and_separate_source_destination_are_preserved(tmp_path):
    args = converter.parse_args([])
    assert args.input == ROOT / "datasets/4_cams"
    assert args.output == ROOT / "datasets/4_cams_robolab"
    args = converter.parse_args(["--input", str(tmp_path / "raw"), "--output", str(tmp_path / "converted")])
    assert args.input == tmp_path / "raw"
    assert args.output == tmp_path / "converted"


def test_empty_converted_directory_has_clear_error(tmp_path):
    with pytest.raises(ValueError, match="No converted episodes found under"):
        converter.converted_episodes(tmp_path)


@pytest.fixture
def source_episode(tmp_path):
    source = tmp_path / "episode.hdf5"
    with h5py.File(source, "w") as recording:
        qpos = np.zeros((2, 14), dtype=np.float32)
        qpos[:, [6, 13]] = 0.05
        recording["observations/qpos"] = qpos
        recording["action"] = qpos + 0.1
        recording["observations/images/cam_third_person"] = np.full((2, 480, 640, 3), 80, dtype=np.uint8)
        recording["observations/images_depth/cam_third_person"] = np.full((2, 480, 640), 1000, dtype=np.uint16)
        recording.attrs["capture_fps_target"] = 30
    return source


@pytest.mark.parametrize("camera", ["cam_left_wrist", "cam_right_wrist"])
@pytest.mark.parametrize("shape,dtype", [
    ((1, 480, 640, 3), np.uint8),
    ((2, 240, 320, 3), np.uint8),
    ((2, 480, 640), np.uint8),
    ((2, 480, 640, 3), np.uint16),
])
def test_inspect_source_rejects_invalid_wrist_rgb(source_episode, camera, shape, dtype):
    with h5py.File(source_episode, "a") as recording:
        recording.create_dataset(f"observations/images/{camera}", shape=shape, dtype=dtype)
    with pytest.raises(ValueError, match=f"expected observations/images/{camera} uint8 RGB"):
        converter.inspect_source(source_episode)


@pytest.mark.parametrize("with_wrists", [False, True])
def test_recompute_command_updates_calibration_and_preserves_rgbd(tmp_path, source_episode, with_wrists):
    source = source_episode
    cameras = {CAMERA: "cam_third_person"}
    if with_wrists:
        cameras.update(left_view="cam_left_wrist", right_view="cam_right_wrist")
        with h5py.File(source, "a") as recording:
            for camera, colors in (
                ("cam_left_wrist", [[17, 43, 211], [202, 101, 31]]),
                ("cam_right_wrist", [[53, 151, 29], [9, 71, 231]]),
            ):
                rgb = np.broadcast_to(np.array(colors, dtype=np.uint8)[:, None, None, :],
                                      (2, 480, 640, 3)).copy()
                rgb[:, 240:, :, 0] //= 2
                recording[f"observations/images/{camera}"] = rgb
    assert converter.inspect_source(source)["rgb_cameras"] == cameras
    calibration = ROOT / "cobot_magic/G_106.yaml"
    intrinsic, extrinsic = read_calibration(calibration)
    original_extrinsic = extrinsic.copy()
    original_extrinsic[0, 3] += 0.1
    task = tmp_path / "converted/place_pen_in_cups"
    episode = task / "place_pen_in_cups__episode_0"
    converter.convert_episode(source, episode,
                              {"task": task.name, "instruction": "Place the pen in the cups.", "status": "success"},
                              RobotModel(DEFAULT_URDF), np.array([[0, 0, 0.13], [0, 0, 0.13]]),
                              intrinsic, original_extrinsic, calibration)
    with h5py.File(source, "r") as recording:
        for camera, source_camera in cameras.items():
            directory = episode / "images" / camera
            assert sorted(p.name for p in directory.iterdir()) == ["000000.png", "000001.png"]
            for i in range(2):
                actual = cv2.cvtColor(cv2.imread(str(directory / f"{i:06d}.png")), cv2.COLOR_BGR2RGB)
                assert actual.shape == (240, 320, 3) and actual.dtype == np.uint8
                expected = cv2.resize(recording[f"observations/images/{source_camera}"][i],
                                      (320, 240), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(actual, expected)
    if not with_wrists:
        # Episodes written before wrist export did not have a camera mapping.
        metadata = json.loads((episode / "metadata.json").read_text())
        metadata.pop("rgb_cameras")
        converter.write_json(episode / "metadata.json", metadata)
    preserved = [*episode.glob("images/*/*.png"), *episode.glob("depths/*/*.png"),
                 episode / "action/joint_position.npy", episode / "timestamps.npy"]
    before = {path.relative_to(episode): (path.read_bytes(), path.stat().st_ino) for path in preserved}
    original_poses = np.load(episode / f"TCP/{CAMERA}/pose.npy")
    converter.main(["--input", str(task), "--recompute-tcp", "--calibration", str(calibration)])
    converter.validate_episode(episode)
    np.testing.assert_array_equal(np.load(episode / f"extrinsic/{CAMERA}.npy"), extrinsic)
    poses = np.load(episode / f"TCP/{CAMERA}/pose.npy")
    np.testing.assert_allclose(poses[..., 0, 3], original_poses[..., 0, 3] - 0.1)
    for relative, (content, inode) in before.items():
        path = episode / relative
        assert path.read_bytes() == content
        assert path.stat().st_ino == inode
    metadata = json.loads((episode / "metadata.json").read_text())
    assert metadata["language_instruction"] == ["Place the pen in the cups."]
    assert "recalculated_utc" in metadata
    if with_wrists:
        assert metadata["rgb_cameras"] == cameras
        assert metadata["validation"]["rgb_cameras"] == list(cameras)
        damaged = episode / "images/left_view/000001.png"
        assert cv2.imwrite(str(damaged), np.zeros((10, 20, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="Invalid images/left_view frame 1"):
            converter.validate_episode(episode)
        damaged.unlink()
        with pytest.raises(ValueError, match="mismatched images/left_view frames"):
            converter.validate_episode(episode)


@pytest.fixture
def wrist_append_episode(tmp_path, source_episode):
    with h5py.File(source_episode, "a") as recording:
        for camera, color in (("cam_left_wrist", [17, 43, 211]), ("cam_right_wrist", [202, 101, 31])):
            recording[f"observations/images/{camera}"] = np.broadcast_to(
                np.array(color, dtype=np.uint8), (2, 480, 640, 3))
        qpos = np.asarray(recording["observations/qpos"])
    episode = tmp_path / "converted/task/episode_0"
    converter.write_json(episode / "metadata.json", {
        "format": converter.FORMAT, "complete": True, "frame_count": 2,
        "width": 320, "height": 240, "source": {"path": str(source_episode)},
        "language_instruction": ["Preserve this instruction."],
    })
    converter.save_array(episode, "observations/joint_position.npy", qpos)
    for relative in ("images/third_person/000000.png", "depths_completed/third_person/000000.png",
                     "TCP/third_person/state.npy", "action/joint_position.npy"):
        target = episode / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"existing file must remain exactly unchanged")
    return episode


def file_snapshot(path):
    return {p.relative_to(path): (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()}


def test_wrist_append_preserves_existing_files_and_resumes(wrist_append_episode):
    episode = wrist_append_episode
    existing = episode / "images/left_view/000000.png"
    existing.parent.mkdir()
    assert cv2.imwrite(str(existing), np.full((240, 320, 3), 71, dtype=np.uint8))
    before = file_snapshot(episode)
    report = converter.append_wrist_rgb(episode)
    assert report["written"] == {"left_view": 1, "right_view": 2}
    assert report["skipped"] == {"left_view": 1, "right_view": 0}
    after = file_snapshot(episode)
    assert {name: after[name] for name in before} == before
    with h5py.File(episode.parents[2] / "episode.hdf5", "r") as source:
        for camera in ("left_view", "right_view"):
            rgb = cv2.cvtColor(cv2.imread(str(episode / f"images/{camera}/000001.png")), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(rgb, cv2.resize(
                source[f"observations/images/{converter.RGB_CAMERAS[camera]}"][1], (320, 240)))
    assert converter.append_wrist_rgb(episode)["written"] == {"left_view": 0, "right_view": 0}
    assert file_snapshot(episode) == after
    (episode / "images/right_view/000001.png").unlink()
    remaining = file_snapshot(episode)
    assert converter.append_wrist_rgb(episode)["written"] == {"left_view": 0, "right_view": 1}
    resumed = file_snapshot(episode)
    assert {name: resumed[name] for name in remaining} == remaining


def test_wrist_append_dry_run_does_not_create_files(wrist_append_episode):
    before = file_snapshot(wrist_append_episode)
    converter.main(["--append-wrist-rgb", "--dry-run", "--input", str(wrist_append_episode)])
    assert file_snapshot(wrist_append_episode) == before
    assert not (wrist_append_episode / "images/left_view").exists()


def test_wrist_append_rejects_misaligned_source_before_writing(wrist_append_episode):
    episode = wrist_append_episode
    before = file_snapshot(episode)
    source_path = json.loads((episode / "metadata.json").read_text())["source"]["path"]
    with h5py.File(source_path, "a") as source:
        source["observations/qpos"][0, 0] = 1.0
    with pytest.raises(ValueError, match="Source joints/frame alignment mismatch"):
        converter.append_wrist_rgb(episode)
    assert file_snapshot(episode) == before


def test_exclusive_png_publication_never_replaces_existing_file(tmp_path):
    target = tmp_path / "000000.png"
    target.write_bytes(b"another writer published this first")
    before = file_snapshot(tmp_path)
    assert not converter.write_png_exclusive(target, np.zeros((240, 320, 3), dtype=np.uint8))
    assert file_snapshot(tmp_path) == before
