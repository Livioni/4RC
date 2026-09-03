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

from arc.datasets.robotwin import FixedImageBatchSampler, RoboTwin4RC
from arc.loss.tcp_tracking import TCPTrackingLoss
from arc.models.arc.heads.head_act import inverse_log_transform, log_transform
from arc.models.arc.heads.motiondecoder import MotionDecoder
from arc.models.arc.heads.tcp_head import TCPTrackHead, TCPVisualQueryEncoder
from arc.rotation import rpy_to_matrix
from tcp_inference import (
    _validate_query_points,
    add_interactive_query_point,
    collect_rgb_paths,
    compute_default_third_camera_view,
    load_ground_truth_query_points,
    render_query_overlay,
    warn_if_out_of_training_distribution,
)
from train_4rc import distributed_scheduler_steps, prepare_tcp_query_points


def _write_episode(root: Path, *, position_unit: str = "meter") -> Path:
    episode = root / "task" / "episode_0000000"
    for relative in (
        "images/third_views",
        "depths/third_views",
        "intrinsics",
        "extrinsics",
        "TCP_third",
    ):
        (episode / relative).mkdir(parents=True, exist_ok=True)

    frame_count = 8
    color = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.full((240, 320), 1000, dtype=np.uint16)
    for index in range(frame_count):
        Image.fromarray(color).save(episode / "images/third_views" / f"{index:06d}.png")
        Image.fromarray(depth).save(episode / "depths/third_views" / f"{index:06d}.png")

    intrinsics = np.array(
        [[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsics = np.repeat(
        np.eye(4, dtype=np.float32)[None, :3], frame_count, axis=0
    )
    np.save(episode / "intrinsics/third_views.npy", intrinsics)
    np.save(episode / "extrinsics/third_views.npy", extrinsics)

    left = np.zeros((frame_count, 7), dtype=np.float32)
    right = np.zeros((frame_count, 7), dtype=np.float32)
    left[:, 2] = right[:, 2] = 0.5
    left[4:, 0] = 1.0
    left[:, 6] = right[:, 6] = 1.0
    np.save(episode / "TCP_third/left_state.npy", left)
    np.save(episode / "TCP_third/right_state.npy", right)

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
        "coordinate_frame": f"third_views {RoboTwin4RC.CAMERA_SUFFIX}",
        "camera": "third_views",
        "gripper": {
            "type": "binary",
            "open": 1,
            "closed": 0,
            "source_threshold": 0.5,
        },
    }
    (episode / "TCP_third/metadata.json").write_text(json.dumps(tcp_metadata))
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


def test_tcp_head_decodes_absolute_pose_in_float32_under_bfloat16_autocast():
    head = TCPTrackHead(embed_dim=8, hidden_dim=4, window_size=1)
    levels = [torch.randn(1, 4, 3, 8) for _ in range(4)]
    mean = torch.tensor([[0.1, -0.2, 0.5], [0.3, 0.2, 0.7]])
    head.set_position_stats(mean, torch.full((2, 3), 0.25))

    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = head(levels)

    assert all(value.dtype == torch.float32 for value in output.values())
    torch.testing.assert_close(output["tcp_position"][:, 0], mean.unsqueeze(0))
    identity = torch.eye(3).expand(1, 2, 3, 3)
    torch.testing.assert_close(output["tcp_rotation"][:, 0], identity)


def test_tcp_loss_reaches_every_output_branch_and_reports_core_metrics():
    torch.manual_seed(0)
    head = TCPTrackHead(embed_dim=8, hidden_dim=4, window_size=1)
    levels = [torch.randn(1, 4, 3, 8, requires_grad=True) for _ in range(4)]
    state = torch.zeros(1, 4, 2, 7)
    state[0, :, 0, 0] = torch.tensor([0.0, 0.02, 0.05, 0.09])
    state[0, :, 1, 1] = torch.tensor([0.0, -0.01, -0.03, -0.06])
    state[0, :, 0, 5] = torch.tensor([0.0, 0.1, 0.2, 0.3])
    state[..., 6] = 1.0
    predictions = head(levels)
    batch = {
        "tcp_state": state,
        "frame_times": torch.arange(4, dtype=torch.float32)[None] / 15.0,
        "tcp_query_valid": torch.ones(1, 2, dtype=torch.bool),
    }

    losses = TCPTrackingLoss()(predictions, batch)
    losses["objective"].backward()
    gradient = head.mlp[-1].weight.grad

    assert torch.linalg.vector_norm(gradient[:3]) > 0
    assert torch.linalg.vector_norm(gradient[3:9]) > 0
    assert torch.linalg.vector_norm(gradient[9]) > 0
    assert torch.linalg.vector_norm(gradient[10]) > 0
    assert set(losses) == {
        "objective",
        "loss_tcp_pose",
        "loss_tcp_temporal",
        "loss_tcp_gripper",
        "metric_tcp_position_m",
        "metric_tcp_rotation_deg",
        "metric_tcp_gripper_accuracy",
    }
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

    assert dataset.tcp_directory == "TCP_third"
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
    expected_left = np.load(episode_path / "TCP_third/left_state.npy")[frame_indices]
    expected_right = np.load(episode_path / "TCP_third/right_state.npy")[frame_indices]
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


def test_robotwin_masks_depth_beyond_three_metres_and_projects_tcp(tmp_path):
    episode = _write_episode(tmp_path)
    depth_path = episode / "depths/third_views/000000.png"
    depth = np.full((240, 320), 1000, dtype=np.uint16)
    depth[0, 0] = 3000
    depth[0, 1] = 3001
    Image.fromarray(depth).save(depth_path)

    dataset = RoboTwin4RC(
        tmp_path,
        min_views=2,
        max_views=2,
        min_interval=1,
        max_interval=1,
        reverse_probability=0.0,
        augment=False,
    )
    sample = dataset[(0, 2, 0)]
    y = dataset.PAD_TOP
    x = dataset.PAD_LEFT
    assert sample["valid_mask"][0, y, x]
    assert not sample["valid_mask"][0, y, x + 1]
    assert sample["depth"][0, y, x].item() == pytest.approx(3.0)
    assert sample["depth"][0, y, x + 1].item() == 0.0
    torch.testing.assert_close(
        sample["tcp_query_points"],
        torch.tensor([[161.0, 126.0], [161.0, 126.0]]),
    )
    assert sample["tcp_query_valid"].all()
    assert dataset.tcp_position_mean.shape == (2, 3)
    assert dataset.tcp_position_std.min() >= 1e-3


def test_visual_query_encoder_samples_local_tokens_and_patch_positions():
    encoder = TCPVisualQueryEncoder(
        embed_dim=4, patch_size=14, window_size=1, adapter_dim=2
    )
    with torch.no_grad():
        encoder.arm_embedding.zero_()
        encoder.offset_embedding.zero_()
    patches = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    points = torch.tensor([[[7.0, 7.0], [21.0, 21.0]]])
    tokens, positions = encoder(
        patches, points, image_height=28, image_width=28
    )
    torch.testing.assert_close(tokens, patches[:, [0, 3]])
    torch.testing.assert_close(positions, torch.tensor([[[0, 0], [1, 1]]]))


def test_motion_decoder_accepts_spatial_positions_for_sparse_queries():
    torch.manual_seed(0)
    decoder = MotionDecoder(
        patch_size=14, embed_dim=16, depth=1, num_heads=4, use_adaln=False
    ).eval()
    tokens = torch.randn(1, 2, 6, 16)
    images = torch.randn(1, 2, 3, 28, 28)
    queries = torch.randn(1, 2, 16)
    first = decoder(
        tokens,
        images,
        patch_start_idx=2,
        query_tokens=queries,
        query_positions=torch.tensor([[[0, 0], [1, 1]]]),
    )
    second = decoder(
        tokens,
        images,
        patch_start_idx=2,
        query_tokens=queries,
        query_positions=torch.tensor([[[1, 1], [0, 0]]]),
    )
    assert first.shape == (1, 2, 3, 16)
    assert not torch.allclose(first, second)


def test_query_jitter_stays_in_image_and_preserves_invalid_arms():
    torch.manual_seed(0)
    batch = {
        "tcp_query_points": torch.tensor([[[1.0, 6.0], [100.0, 100.0]]]),
        "tcp_query_valid": torch.tensor([[True, False]]),
        "padding": torch.tensor([[1, 1, 6, 6]]),
        "source_size": torch.tensor([[240, 320]]),
    }
    points = prepare_tcp_query_points(
        batch,
        global_step=99,
        total_steps=100,
        config={
            "tcp_query_exact_ratio": 0.0,
            "tcp_query_curriculum_warmup_ratio": 0.0,
            "tcp_query_curriculum_transition_ratio": 0.0,
            "tcp_query_max_jitter_patches": 1.0,
        },
    )
    assert not torch.allclose(points[0, 0], batch["tcp_query_points"][0, 0])
    assert 1.0 <= points[0, 0, 0] <= 320.0
    assert 6.0 <= points[0, 0, 1] <= 245.0
    torch.testing.assert_close(points[0, 1], batch["tcp_query_points"][0, 1])


def test_tcp_loss_ignores_an_invalid_arm():
    head = TCPTrackHead(embed_dim=8, hidden_dim=4, window_size=1)
    levels = [torch.randn(1, 3, 3, 8) for _ in range(4)]
    predictions = head(levels)
    state = torch.zeros(1, 3, 2, 7)
    state[..., 6] = 1.0
    batch = {
        "tcp_state": state.clone(),
        "frame_times": torch.arange(3, dtype=torch.float32)[None],
        "tcp_query_valid": torch.tensor([[True, False]]),
    }
    first = TCPTrackingLoss()(predictions, batch)["objective"]
    batch["tcp_state"][:, :, 1, :6] = 1000.0
    second = TCPTrackingLoss()(predictions, batch)["objective"]
    torch.testing.assert_close(first, second)


def test_inference_query_points_are_original_image_coordinates():
    points = _validate_query_points([[0, 0], [319, 239]], "test")
    assert points.shape == (2, 2)
    with pytest.raises(ValueError, match="must lie inside"):
        _validate_query_points([[0, 0], [320, 239]], "test")


def test_ground_truth_query_points_project_into_original_first_frame(tmp_path):
    episode = _write_episode(tmp_path)
    expected = np.array([[160.0, 120.0], [160.0, 120.0]], dtype=np.float32)

    from_episode = load_ground_truth_query_points(episode, "third_views", 0)
    from_rgb_dir = load_ground_truth_query_points(
        episode / "images" / "third_views", "third_views", 0
    )

    np.testing.assert_allclose(from_episode, expected)
    np.testing.assert_allclose(from_rgb_dir, expected)


def test_ground_truth_query_points_require_episode_metadata(tmp_path):
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="requires a RoboTwin episode"):
        load_ground_truth_query_points(rgb_dir, "third_views", 0)


def test_interactive_query_selection_is_left_then_right_and_stops_at_two():
    points = add_interactive_query_point([], (12, 34))
    points = add_interactive_query_point(points, (210, 123))
    assert points == [[12.0, 34.0], [210.0, 123.0]]
    assert add_interactive_query_point(points, (1, 2)) == points


def test_default_viser_view_sits_behind_third_camera():
    extrinsic_w2c = np.eye(4, dtype=np.float32)[:3]
    position, look_at, up = compute_default_third_camera_view(
        extrinsic_w2c,
        scene_center=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        scene_extent=1.0,
    )

    np.testing.assert_allclose(position, [0.0, 0.0, -0.15])
    np.testing.assert_allclose(look_at, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(up, [0.0, -1.0, 0.0])


def test_interactive_query_overlay_preserves_native_image():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    overlay = render_query_overlay(image, [[12.0, 34.0], [210.0, 123.0]])
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert np.any(overlay != image)
    assert not np.any(image)


def test_distributed_scheduler_steps_match_accelerate_convention():
    assert distributed_scheduler_steps(
        1_000, 50_000, num_processes=8, split_batches=False
    ) == (8_000, 400_000)
    assert distributed_scheduler_steps(
        1_000, 50_000, num_processes=8, split_batches=True
    ) == (1_000, 50_000)


def test_batch_sampler_changes_worker_sample_seeds_between_epochs(tmp_path):
    _write_episode(tmp_path)
    dataset = RoboTwin4RC(
        tmp_path,
        min_views=2,
        max_views=4,
        min_interval=1,
        max_interval=1,
        reverse_probability=0.0,
        augment=False,
    )
    sampler = FixedImageBatchSampler(
        dataset,
        images_per_batch=4,
        scene_counts=(1,),
        batches_per_epoch=3,
        seed=17,
    )
    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert [batch[0][2] for batch in first_epoch] != [
        batch[0][2] for batch in second_epoch
    ]
