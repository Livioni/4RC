"""Ordered action steps and migration from physical-time Stage2 checkpoints."""
import json
import runpy
import sys

import numpy as np
import pytest
import torch

from arc.datasets import DatasetSource, WeightedDatasetMixture, collate_training_samples
from arc.datasets.robotwin_action import RoboTwinActionDataset, build_action_dataset
from arc.loss.tcp_tracking import TCPTrackingLoss
from stage2_helpers import TinyArc, TinyLanguage, tiny_batch
from test_action_dataset import write_episode
from test_action_policy import TinyDataset, policy


def test_index_conditions_preserve_order_without_timestamps():
    indexed = policy().eval()
    batch = tiny_batch()
    batch.pop("frame_times")
    reconstruction, features = indexed.reconstruct(batch["images"], batch["tcp_query_points"])
    condition = indexed.make_condition(batch["images"], batch["intrinsics"], batch["instruction"],
                                       reconstruction, features)
    assert not torch.allclose(condition.future_step_embedding[:, 0], condition.future_step_embedding[:, -1])
    noise, flow_time = torch.randn(1, 16, 20), torch.tensor([.4])
    expected = indexed(batch, noise=noise, flow_time=flow_time)
    for times in (torch.zeros(1, 8), torch.tensor([[-4., -3., -2., -1., -.7, -.4, -.1, 0.]])):
        changed = indexed(batch | {"frame_times": times}, noise=noise, flow_time=flow_time)
        torch.testing.assert_close(changed["action_velocity"], expected["action_velocity"], atol=0, rtol=0)
    result = indexed.sample_actions(batch["images"], batch["instruction"], batch["tcp_query_points"],
                                   intrinsics=batch["intrinsics"], steps=2,
                                   generator=torch.Generator().manual_seed(42))
    assert result["action_position"].shape == (1, 16, 2, 3)
    assert result["future_step_indices"].tolist() == [list(range(1, 17))]
    assert "future_frame_times" not in result



def test_disabled_velocity_loss_never_reads_times_and_keeps_pose_gradients():
    model = policy()
    batch = tiny_batch()
    prediction, _ = model.reconstruct(batch["images"], batch["tcp_query_points"])
    criterion = TCPTrackingLoss(temporal_weight=0)
    del batch["frame_times"]
    expected = criterion(prediction, batch)
    for times in (torch.zeros(1, 8), torch.full((1, 8), float("nan"))):
        actual = criterion(prediction, batch | {"frame_times": times})
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert expected["loss_tcp_temporal"] == 0
    expected["objective"].backward()
    grad = model.arc.tcp_track_head.linear.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    with pytest.raises(KeyError, match="frame_times"):
        TCPTrackingLoss(temporal_weight=0.2)(prediction, batch)


@pytest.mark.parametrize("anchor", [0, 3, 7, 12])
@pytest.mark.parametrize("rate", [10, 30])
def test_index_dataset_uses_next_sixteen_records_and_real_history_times(tmp_path, anchor, rate):
    episode = write_episode(tmp_path, frames=20)
    metadata = json.loads((episode / "metadata.json").read_text())
    (episode / "metadata.json").write_text(json.dumps(metadata | {"frequency_hz": rate}))
    ds = RoboTwinActionDataset(tmp_path, validation_fraction=0, augment=False,
                               prediction_horizon=16)
    ds.starts[0] = np.array([anchor - 7])
    sample = ds.get_sample(0, 8, 42)
    indices = torch.arange(anchor - 7, anchor + 1).clamp_min(0)
    torch.testing.assert_close(sample["frame_indices"], indices)
    torch.testing.assert_close(sample["frame_times"], indices.float() / rate)
    future = torch.arange(anchor + 1, anchor + 17)
    torch.testing.assert_close(sample["future_actions"][:, 0, 0], -.2 + future.clamp_max(19) * .002)
    torch.testing.assert_close(sample["future_action_valid"], future < 20)
    assert "future_frame_times" not in sample
    assert sample["future_step_indices"].tolist() == list(range(1, 17))
    batch = collate_training_samples([sample, sample])
    assert batch["future_actions"].shape == (2, 16, 2, 10)
    # Verify the runner-to-adapter mode, not only direct dataset construction.
    config = dict(data_sources=[dict(name="test", options=dict(root=str(tmp_path)))],
                  history_frames=8, prediction_horizon=16, validation_fraction=0)
    sample = build_action_dataset(config).sources[0].dataset.get_sample(0, 8, 42)
    assert "future_frame_times" not in sample
    assert sample["future_step_indices"].tolist() == list(range(1, 17))


def test_index_history_and_future_stay_in_valid_segment(tmp_path):
    episode = write_episode(tmp_path, frames=6)
    path = episode / "TCP_third/left_state.npy"
    state = np.load(path)
    state[3:, 0] += .6
    np.save(path, state)
    ds = RoboTwinActionDataset(tmp_path, validation_fraction=0, augment=False,
                               prediction_horizon=16)
    ds.starts[0] = np.array([4 - 7])
    sample = ds.get_sample(0, 8, 42)
    assert sample["frame_indices"].tolist() == [3] * 7 + [4]
    assert sample["future_action_valid"].tolist() == [True] + [False] * 15
    torch.testing.assert_close(sample["future_actions"][0, 0, 0], torch.tensor(state[5, 0]))


@pytest.fixture
def small_runner(monkeypatch):
    import train_4rc_stage2 as runner
    import arc.models.arc.arc_action as action_module
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(runner, "Arc", TinyArc)
    monkeypatch.setattr(action_module, "FrozenT5Encoder", lambda *a, **k: TinyLanguage())
    def dataset(config, split):
        ds = TinyDataset(config["prediction_horizon"])
        ds.tcp_position_mean = torch.full((2, 3), .25)
        ds.tcp_position_std = torch.full((2, 3), .5)
        return WeightedDatasetMixture([DatasetSource("tiny", "robotwin_action", ds, 1.)])
    monkeypatch.setattr(runner, "build_action_dataset", dataset)
    return runner


def small_config(root, **updates):
    config = {k: v for k, v in runpy.run_path("configs/train/4rc-stage2-action.py").items() if not k.startswith("_")}
    config.update(output_dir=str(root), batch_size=1, action_dim=32, action_depth=2, action_heads=4,
                  mixed_precision="no", max_train_steps=2, num_train_epochs=None,
                  gradient_accumulation_steps=1, batches_per_epoch=4, warmup_steps=0,
                  num_workers=0, report_to=[], checkpointing_steps=1, visualize_every_steps=0,
                  validate_every_steps=1, validation_batches=1, sampling_steps=1)
    return config | updates


@pytest.fixture
def legacy_checkpoint(tmp_path):
    from safetensors.torch import save_model
    checkpoint = tmp_path / "previous_stage2"
    checkpoint.mkdir()
    model = policy()
    model.set_action_position_stats(torch.full((2, 3), .25), torch.full((2, 3), .5))
    # Parameter names/shapes are unchanged; obsolete training metadata must not
    # override the new run's sequence conditioning, loss settings or step count.
    save_model(model, str(checkpoint / "model.safetensors"))
    config = small_config(checkpoint, time_encoding="physical", time_unit_seconds=1/15,
                          tcp_temporal_weight=.2, max_train_steps=60000)
    (checkpoint / "config.json").write_text(json.dumps(config))
    return checkpoint


@pytest.mark.parametrize("file_only", [False, True])
def test_stage2_weight_initialization_preserves_scale_and_restarts_training(tmp_path, small_runner, legacy_checkpoint, monkeypatch, file_only):
    runner = small_runner
    source = legacy_checkpoint / "model.safetensors" if file_only else legacy_checkpoint
    output = tmp_path / "initialized"
    config = small_config(output, stage1_checkpoint="missing-stage1", stage2_checkpoint=str(source), max_train_steps=1)
    path = tmp_path / "initialized.json"
    path.write_text(json.dumps(config))
    expected = runner.read_weights(legacy_checkpoint)
    original_build = runner.build_optimizer
    initialized = []
    def checked_build(model, cfg):
        torch.testing.assert_close(model.state_dict(), expected)
        optimizer = original_build(model, cfg)
        assert not optimizer.state
        initialized.append(True)
        return optimizer
    monkeypatch.setattr(runner, "build_optimizer", checked_build)
    # Distinct new-data statistics must not overwrite the loaded action scale.
    def different_dataset(cfg, split):
        ds = TinyDataset(cfg["prediction_horizon"])
        ds.tcp_position_mean = torch.full((2, 3), 9.)
        ds.tcp_position_std = torch.full((2, 3), 8.)
        return WeightedDatasetMixture([DatasetSource("new", "robotwin_action", ds, 1.)])
    monkeypatch.setattr(runner, "build_action_dataset", different_dataset)
    original_load = runner.Accelerator.load_state
    def forbidden_load(*args, **kwargs):
        raise AssertionError("Weight initialization must not load optimizer/RNG/training state")
    monkeypatch.setattr(runner.Accelerator, "load_state", forbidden_load)
    monkeypatch.setattr(sys, "argv", ["train_4rc_stage2.py", "--config", str(path)])
    runner.main()
    assert initialized == [True]
    state = json.loads((output / "final_checkpoint/trainer_state.json").read_text())
    assert state == dict(epoch=0, batch_in_epoch=1, global_step=1)
    saved = json.loads((output / "final_checkpoint/config.json").read_text())
    assert not {"time_encoding", "time_unit_seconds", "tcp_velocity_scale"} & saved.keys()
    assert saved["tcp_temporal_weight"] == 0 and saved["prediction_horizon"] == 16
    trained = runner.read_weights(output / "final_checkpoint")
    for name in ("action_position_mean", "action_position_std",
                 "arc.tcp_track_head.position_mean", "arc.tcp_track_head.position_std"):
        torch.testing.assert_close(trained[name], expected[name])
    optimizer = torch.load(output / "final_checkpoint/optimizer.bin", map_location="cpu", weights_only=True)
    assert max(float(value["step"]) for value in optimizer["state"].values()) == 1
    monkeypatch.setattr(runner, "build_optimizer", original_build)
    monkeypatch.setattr(runner.Accelerator, "load_state", original_load)
    monkeypatch.setattr(sys, "argv", ["train_4rc_stage2.py", "--resume", str(output / "final_checkpoint"),
                                    "--output-dir", str(tmp_path / "continued"), "--max-train-steps", "2"])
    runner.main()
    continued = json.loads((tmp_path / "continued/final_checkpoint/trainer_state.json").read_text())
    assert continued["global_step"] == 2 and continued["batch_in_epoch"] == 2


def test_index_mode_requires_velocity_loss_disabled(small_runner, tmp_path):
    with pytest.raises(ValueError, match="tcp_temporal_weight=0"):
        small_runner.validate_config(small_config(tmp_path, tcp_temporal_weight=.2))


def test_physical_time_config_cannot_reactivate_old_mode(small_runner, tmp_path):
    with pytest.raises(ValueError, match="--stage2-checkpoint"):
        small_runner.validate_config(small_config(tmp_path, time_encoding="physical"))


def test_evaluation_accepts_stage2_weights_without_training_state(tmp_path, small_runner, legacy_checkpoint, monkeypatch):
    output = tmp_path / "evaluation"
    config = small_config(output, stage1_checkpoint="missing-stage1", stage2_checkpoint=str(legacy_checkpoint))
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(config))
    def forbidden(*args, **kwargs):
        raise AssertionError("Weight-only evaluation must not build an optimizer or restore training state")
    monkeypatch.setattr(small_runner, "build_optimizer", forbidden)
    monkeypatch.setattr(small_runner.Accelerator, "load_state", forbidden)
    monkeypatch.setattr(sys, "argv", ["train_4rc_stage2.py", "--config", str(path), "--eval-only"])
    small_runner.main()
    metrics = json.loads((output / "validation/step-00000000.json").read_text())
    assert metrics["recovered/valid_trajectories"] == 1
