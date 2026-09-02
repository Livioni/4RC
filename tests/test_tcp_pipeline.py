from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arc.datasets.robotwin import RoboTwin4RC
from arc.loss.tcp_tracking import TCPTrackingLoss
from arc.models.arc.heads.head_act import inverse_log_transform, log_transform
from arc.models.arc.heads.tcp_head import TCPTrackHead
from arc.rotation import rpy_to_matrix
from tcp_inference import collect_rgb_paths, warn_if_out_of_training_distribution


def _write_episode(root: Path, *, position_unit: str = "meter") -> Path:
    episode = root / "task" / "episode_0000000"
    for relative in (
        "images/head_view",
        "depths/head_view",
        "intrinsics",
        "extrinsics",
        "TCP_head",
    ):
        (episode / relative).mkdir(parents=True, exist_ok=True)

    frame_count = 8
    color = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 1000, dtype=np.uint16)
    for index in range(frame_count):
        Image.fromarray(color).save(episode / "images/head_view" / f"{index:06d}.png")
        Image.fromarray(depth).save(episode / "depths/head_view" / f"{index:06d}.png")

    intrinsics = np.array(
        [[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsics = np.repeat(
        np.eye(4, dtype=np.float32)[None, :3], frame_count, axis=0
    )
    np.save(episode / "intrinsics/head_view.npy", intrinsics)
    np.save(episode / "extrinsics/head_view.npy", extrinsics)

    left = np.zeros((frame_count, 7), dtype=np.float32)
    right = np.zeros((frame_count, 7), dtype=np.float32)
    left[:, 2] = right[:, 2] = 0.5
    left[4:, 0] = 1.0
    left[:, 6] = right[:, 6] = 1.0
    np.save(episode / "TCP_head/left_state.npy", left)
    np.save(episode / "TCP_head/right_state.npy", right)

    (episode / "metadata.json").write_text(json.dumps({"frequency_hz": 15}))
    tcp_metadata = {
        "shape": [frame_count, 7],
        "columns": [
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "gripper_open",
        ],
        "position_unit": position_unit,
        "rotation_unit": "radian",
        "rpy_convention": RoboTwin4RC.RPY_CONVENTION,
        "coordinate_frame": f"head_view {RoboTwin4RC.CAMERA_SUFFIX}",
        "camera": "head_view",
        "gripper": {
            "type": "binary",
            "open": 1,
            "closed": 0,
            "source_threshold": 0.5,
        },
    }
    (episode / "TCP_head/metadata.json").write_text(json.dumps(tcp_metadata))
    return episode


@pytest.mark.parametrize(
    ("function", "expected"),
    (
        (inverse_log_transform, lambda x: torch.sign(x) * torch.expm1(x.abs())),
        (log_transform, lambda x: torch.sign(x) * torch.log1p(x.abs())),
    ),
)
def test_signed_transforms_preserve_values_and_unit_zero_gradient(function, expected):
    values = torch.tensor([-3.0, -0.1, 0.0, 0.1, 3.0], requires_grad=True)
    output = function(values)
    torch.testing.assert_close(output.detach(), expected(values.detach()))
    output.sum().backward()
    assert values.grad[2].item() == pytest.approx(1.0)
    assert torch.isfinite(output).all()


def test_tcp_head_decodes_query_in_float32_under_bfloat16_autocast():
    head = TCPTrackHead(embed_dim=8, hidden_dim=4)
    levels = [torch.randn(1, 4, 3, 8) for _ in range(4)]
    query = torch.tensor(
        [
            [
                [0.1234567, -0.2345678, 0.5678912, 1.234567, -0.456789, 2.345678, 1.0],
                [0.3456789, 0.1234567, 0.6789123, -2.123456, 0.345678, -1.234567, 0.0],
            ]
        ]
    )

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = head(levels, query)

    assert all(value.dtype == torch.float32 for value in output.values())
    torch.testing.assert_close(output["tcp_position"][:, 0], query[..., :3])
    torch.testing.assert_close(
        output["tcp_rotation"][:, 0], rpy_to_matrix(query[..., 3:6])
    )


def test_tcp_loss_reaches_every_output_branch_and_reports_motion_metrics():
    torch.manual_seed(0)
    head = TCPTrackHead(embed_dim=8, hidden_dim=4)
    levels = [torch.randn(1, 4, 3, 8, requires_grad=True) for _ in range(4)]
    state = torch.zeros(1, 4, 2, 7)
    state[0, :, 0, 0] = torch.tensor([0.0, 0.02, 0.05, 0.09])
    state[0, :, 1, 1] = torch.tensor([0.0, -0.01, -0.03, -0.06])
    state[0, :, 0, 5] = torch.tensor([0.0, 0.1, 0.2, 0.3])
    state[..., 6] = 1.0
    predictions = head(levels, state[:, 0])
    batch = {
        "tcp_state": state,
        "frame_times": torch.arange(4, dtype=torch.float32)[None] / 15.0,
    }

    losses = TCPTrackingLoss()(predictions, batch)
    losses["objective"].backward()
    gradient = head.mlp[-1].weight.grad

    assert torch.linalg.vector_norm(gradient[:3]) > 0
    assert torch.linalg.vector_norm(gradient[3:9]) > 0
    assert torch.linalg.vector_norm(gradient[9]) > 0
    assert torch.linalg.vector_norm(gradient[10]) > 0
    assert losses["metric_tcp_position_nonquery_m"] > 0
    assert losses["metric_tcp_predicted_displacement_m"] == pytest.approx(0.0)
    assert losses["metric_tcp_static_baseline_m"] > 0
    assert all(torch.isfinite(value) for value in losses.values())


def test_robotwin_uses_metadata_rate_and_never_crosses_invalid_transition(tmp_path):
    episode_path = _write_episode(tmp_path)
    dataset = RoboTwin4RC(
        tmp_path,
        min_views=2,
        max_views=3,
        min_interval=1,
        max_interval=1,
        reverse_probability=0.5,
        augment=False,
    )
    episode = dataset.episodes[0]

    assert dataset.tcp_directory == "TCP_head"
    assert episode.frame_rate == pytest.approx(15.0)
    assert episode.valid_segments == ((0, 4), (4, 8))
    assert dataset.invalid_transition_count == 1
    assert dataset.segmented_episode_count == 1

    for seed in range(100):
        indices, interval = dataset.sample_frame_indices(
            episode, np.random.default_rng(seed), num_views=3
        )
        ordered = np.sort(indices)
        assert interval == 1
        assert any(
            ordered[0] >= start and ordered[-1] < end
            for start, end in episode.valid_segments
        )

    sample = dataset[(0, 3, 7)]
    frame_indices = sample["frame_indices"].numpy()
    expected_left = np.load(episode_path / "TCP_head/left_state.npy")[frame_indices]
    expected_right = np.load(episode_path / "TCP_head/right_state.npy")[frame_indices]
    expected_state = np.stack((expected_left, expected_right), axis=1)
    np.testing.assert_allclose(sample["tcp_state"].numpy(), expected_state)
    np.testing.assert_allclose(
        np.abs(np.diff(sample["frame_times"].numpy())), np.full(2, 1.0 / 15.0)
    )


def test_robotwin_rejects_incompatible_tcp_metadata(tmp_path):
    _write_episode(tmp_path, position_unit="millimeter")
    with pytest.raises(ValueError, match="position_unit"):
        RoboTwin4RC(tmp_path, augment=False)


def test_inference_selects_fixed_interval_clips_and_warns_for_ood(tmp_path, capsys):
    for index in range(30):
        (tmp_path / f"{index:06d}.png").touch()

    automatic = collect_rgb_paths(
        tmp_path, 4, start_frame=10, frame_interval=5
    )
    assert [int(path.stem) for path in automatic] == [10, 15, 20, 25]
    default = collect_rgb_paths(tmp_path, 4)
    assert [int(path.stem) for path in default] == [0, 1, 2, 3]
    explicit = collect_rgb_paths(tmp_path, 18, [25, 20, 15, 10])
    assert [int(path.stem) for path in explicit] == [25, 20, 15, 10]

    warn_if_out_of_training_distribution(automatic)
    assert capsys.readouterr().out == ""
    warn_if_out_of_training_distribution(
        [tmp_path / "000000.png", tmp_path / "000006.png", tmp_path / "000012.png"]
    )
    assert "outside the TCP training distribution" in capsys.readouterr().out
