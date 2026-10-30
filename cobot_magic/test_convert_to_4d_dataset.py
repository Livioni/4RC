"""Check existing-data selection and in-place TCP recalculation."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cobot_magic import convert_to_4d_dataset as converter
from cobot_magic.geometry import CAMERA, DEFAULT_URDF, ROOT, RobotModel, read_calibration


@pytest.mark.parametrize("mode", ["--recompute-tcp", "--validate-only"])
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

    monkeypatch.setattr(converter, "recompute_episode", recompute)
    monkeypatch.setattr(converter, "validate_episode", validate)
    converter.main([mode, flag, str(target), "--calibration", str(calibration)])
    assert calls == (episodes[:1] if level == "episode" else episodes)


@pytest.mark.parametrize("mode", ["--recompute-tcp", "--validate-only"])
def test_existing_data_modes_reject_different_input_and_output(tmp_path, mode):
    with pytest.raises(SystemExit) as error:
        converter.parse_args([mode, "--input", str(tmp_path / "first"),
                              "--output", str(tmp_path / "second")])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["--recompute-tcp", "--validate-only"])
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


def test_recompute_command_updates_calibration_and_preserves_rgbd(tmp_path):
    source = tmp_path / "episode.hdf5"
    with h5py.File(source, "w") as recording:
        qpos = np.zeros((2, 14), dtype=np.float32)
        qpos[:, [6, 13]] = 0.05
        recording["observations/qpos"] = qpos
        recording["action"] = qpos + 0.1
        recording["observations/images/cam_third_person"] = np.full((2, 480, 640, 3), 80, dtype=np.uint8)
        recording["observations/images_depth/cam_third_person"] = np.full((2, 480, 640), 1000, dtype=np.uint16)
        recording.attrs["capture_fps_target"] = 30
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
