import json
import runpy
import sys

import pytest
import torch

from arc.action import future_actions_in_current_camera, project_tcp, safe_rotation_6d_to_matrix
from arc.models.arc.heads.action_head import TCPActionPolicy
from arc.loss.action import flow_matching_loss
from stage2_helpers import TinyArc, TinyLanguage, tiny_batch

torch.set_num_threads(2)


def policy():
    model = TCPActionPolicy(TinyArc(), language_encoder=TinyLanguage(), dim=32, depth=2, heads=4)
    # Exercise nonzero gates, as after the zero-initialized head starts learning.
    torch.nn.init.normal_(model.dit.output_projection.weight, std=0.03)
    for block in model.dit.blocks:
        torch.nn.init.constant_(block.modulation[-1].bias, 0.1)
    return model


@pytest.mark.parametrize("batch_size", [1, 2])
def test_joint_gradient_shapes_and_teacher_forcing(batch_size):
    import train_4rc_stage2 as runner
    config = runpy.run_path("configs/train/4rc-stage2-action.py")
    model = policy()
    batch = tiny_batch(batch_size)
    prediction = model(batch)
    assert prediction["action_velocity"].shape == (batch_size, 16, 20)
    flow_matching_loss(prediction)["objective"].backward()
    assert model.arc.backbone.weight.grad.abs().sum() > 0
    assert model.history_pool.pool.in_proj_weight.grad.abs().sum() > 0
    assert model.arc.tcp_visual_query_encoder.adapter[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.arc.tcp_track_head.parameters())
    assert all(p.grad is None for p in model.language_encoder.parameters())
    model.zero_grad()
    geometry, tcp = runner.build_criteria(config)
    prediction = model(batch)
    objective, _ = runner.compute_losses(prediction, batch, geometry, tcp, config)
    assert torch.isfinite(objective)
    objective.backward()
    assert model.arc.head.weight.grad.abs().sum() > 0
    assert model.arc.tcp_track_head.linear.weight.grad.abs().sum() > 0
    assert model.arc.motion_decoder.weight.grad.abs().sum() > 0


def test_history_mask_time_and_centre_encoding():
    model = policy().eval()
    batch = tiny_batch()
    reconstruction, features = model.reconstruct(batch["images"], batch["tcp_query_points"])
    kwargs = dict(
        centres=batch["history_tcp_query_points"], valid=batch["history_tcp_valid"],
    )
    def condition(frame_times=batch["frame_times"], **overrides):
        return model.make_condition(
            batch["images"], batch["intrinsics"], frame_times, batch["future_frame_times"],
            batch["instruction"], reconstruction, features, **(kwargs | overrides),
        )
    c = condition()
    assert c.history.shape == (1, 16, 32)
    common = (c.history, c.text, c.text_valid, c.future_time_embedding, model.token_type.weight[1])
    v1, h1 = model.dit(torch.randn(1, 16, 20), torch.tensor([0.1]), *common, return_history=True)
    v2, h2 = model.dit(torch.randn(1, 16, 20) * 10, torch.tensor([0.9]), *common, return_history=True)
    torch.testing.assert_close(h1, h2)
    assert not torch.allclose(v1, v2)
    moved = condition(centres=batch["history_tcp_query_points"] + 2)
    assert not torch.allclose(c.history, moved.history)
    times = batch["frame_times"].clone()
    times[:, :-1] -= 1
    assert not torch.allclose(c.history, condition(frame_times=times).history)


def test_sampling_reuses_encoding_and_reports_missing_history():
    model = policy().eval()
    batch = tiny_batch()
    result = model.sample_actions(
        batch["images"], batch["instruction"], batch["tcp_query_points"],
        batch["frame_times"], batch["intrinsics"], steps=3,
        generator=torch.Generator().manual_seed(123),
    )
    assert model.arc.calls == 1
    assert result["action_position"].shape == (1, 16, 2, 3)
    assert result["action_rotation"].shape == (1, 16, 2, 3, 3)
    torch.testing.assert_close(torch.linalg.det(result["action_rotation"]), torch.ones(1, 16, 2))
    assert result["success"].all()
    reconstruction, features = model.reconstruct(batch["images"], batch["tcp_query_points"])
    reconstruction["tcp_position"] = -torch.ones_like(reconstruction["tcp_position"])
    condition = model.make_condition(
        batch["images"], batch["intrinsics"], batch["frame_times"], batch["future_frame_times"],
        batch["instruction"], reconstruction, features,
    )
    result = model.sample_condition(condition, steps=1)
    assert not result["success"].any()
    assert torch.isnan(result["action_position"]).all()
    assert (result["action_gripper"] == -1).all()
    model.train()
    assert not model.language_encoder.training


def test_projection_and_camera_rotation():
    positions = torch.tensor([[[0., 0, 1], [10., 0, 1]], [[0., 0, -1], [float("nan"), 0, 1]]])
    intrinsics = torch.tensor([[100., 0, 161], [0, 100, 126], [0, 0, 1]]).expand(2, -1, -1)
    uv, valid = project_tcp(positions, intrinsics, image_height=252, image_width=322)
    assert valid.tolist() == [[True, False], [False, False]]
    torch.testing.assert_close(uv[0, 0], torch.tensor([161., 126.]))
    assert torch.isfinite(uv).all()
    # A fixed world TCP observed from translated and rotated future cameras.
    angle = torch.tensor(0.3)
    camera = torch.tensor([[angle.cos(), -angle.sin(), 0, 0.2],
                           [angle.sin(), angle.cos(), 0, 0.0], [0., 0, 1, 0.]])
    state = torch.zeros(1, 2, 7)
    world_position = torch.tensor([[0.2, 0.1, 1.], [-0.2, 0.1, 1.]])
    state[..., :3] = world_position @ camera[:3, :3].T + camera[:3, 3]
    state[..., 5] = angle
    target = future_actions_in_current_camera(state, camera[None], torch.eye(4)[:3])
    torch.testing.assert_close(target[0, :, :3], world_position)
    torch.testing.assert_close(
        safe_rotation_6d_to_matrix(target[..., 3:9]), torch.eye(3).expand(1, 2, 3, 3),
        atol=1e-6, rtol=1e-6,
    )
    degenerate = safe_rotation_6d_to_matrix(torch.zeros(2, 6))
    torch.testing.assert_close(torch.linalg.det(degenerate), torch.ones(2))


class TinyDataset:
    min_views = max_views = 8
    tcp_position_mean = torch.zeros(2, 3)
    tcp_position_std = torch.ones(2, 3)
    def __len__(self):
        return 2
    def set_epoch(self, epoch):
        pass
    def eligible_indices(self, count):
        import numpy as np
        return np.arange(2) if count == 8 else np.empty(0, dtype=int)
    def manifest(self):
        return [{"episode": "tiny"}]
    def get_sample(self, index, num_views, sample_seed):
        with torch.random.fork_rng():
            torch.manual_seed(sample_seed % (2**31))
            batch = tiny_batch()
        sample = {key: value[0] for key, value in batch.items()}
        sample.update(task=f"task{index}", episode=f"episode{index}")
        return sample


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("num_train_epochs", [2, None])
def test_stage2_checkpoint_resume_and_evaluation(tmp_path, monkeypatch, accumulation, num_train_epochs):
    import train_4rc_stage2 as runner
    import arc.models.arc.heads.action_head as action_module
    from arc.datasets import DatasetSource, WeightedDatasetMixture
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(runner, "Arc", TinyArc)
    monkeypatch.setattr(action_module, "FrozenT5Encoder", lambda *a, **k: TinyLanguage())
    mixture = WeightedDatasetMixture([DatasetSource("tiny", "robotwin_action", TinyDataset(), 1.)])
    monkeypatch.setattr(runner, "build_action_dataset", lambda config, split: mixture)
    stage1 = tmp_path / "stage1.pt"
    arc = TinyArc()
    torch.save(arc.state_dict(), stage1)
    config = {k: v for k, v in runpy.run_path("configs/train/4rc-stage2-action.py").items() if not k.startswith("_")}
    config.update(
        stage1_checkpoint=str(stage1), output_dir=str(tmp_path / "full"),
        action_dim=32, action_depth=2, action_heads=4, mixed_precision="no",
        max_train_steps=2, num_train_epochs=num_train_epochs, gradient_accumulation_steps=accumulation,
        batches_per_epoch=2 * accumulation, warmup_steps=0, num_workers=0, report_to=[],
        checkpointing_steps=1, visualize_every_steps=0, validate_every_steps=1,
        validation_batches=1, sampling_steps=1,
    )
    checked = runner.validate_config(config)
    assert checked["train_batch_images"] == 8
    assert runner.validate_config(config | {"batch_size": 2})["train_batch_images"] == 16
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["train_4rc_stage2.py", "--config", str(path)])
    runner.main()
    full = runner.read_weights(tmp_path / "full/final_checkpoint")
    torch.testing.assert_close(full["arc.tcp_track_head.position_mean"], arc.tcp_track_head.position_mean)
    torch.testing.assert_close(full["action_position_mean"], mixture.tcp_position_mean)
    monkeypatch.setattr(sys, "argv", [
        "train_4rc_stage2.py", "--resume", str(tmp_path / "full/checkpoint-1"),
        "--output-dir", str(tmp_path / "resumed"),
    ])
    runner.main()
    resumed = runner.read_weights(tmp_path / "resumed/final_checkpoint")
    for key in full:
        torch.testing.assert_close(full[key], resumed[key], msg=lambda msg: f"{key}: {msg}")
    state = json.loads((tmp_path / "resumed/final_checkpoint/trainer_state.json").read_text())
    assert state["global_step"] == 2
    metrics = json.loads((tmp_path / "full/validation/step-00000002.json").read_text())
    assert "recovered/position_ade_m" in metrics
    assert "teacher_forced/position_ade_m" in metrics
    assert "shuffled_instruction/position_ade_m" in metrics
    monkeypatch.setattr(sys, "argv", [
        "train_4rc_stage2.py", "--resume", str(tmp_path / "full/final_checkpoint"),
        "--output-dir", str(tmp_path / "evaluated"), "--eval-only",
    ])
    runner.main()
    assert (tmp_path / "evaluated/validation/step-00000000.json").exists()



def test_bfloat16_joint_loss_keeps_camera_inverses_float32():
    import train_4rc_stage2 as runner
    config = runpy.run_path("configs/train/4rc-stage2-action.py")
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "cpu")
    model = policy().to(device)
    batch = runner.move_batch(tiny_batch(), device)
    geometry, tcp = (criterion.to(device) for criterion in runner.build_criteria(config))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        prediction = model(batch)
        objective, _ = runner.compute_losses(prediction, batch, geometry, tcp, config)
    assert objective.dtype == torch.float32 and torch.isfinite(objective)
    # CPU BF16 backward depends on the host's oneDNN instruction set.
    if device.type == "cuda":
        objective.backward()


def test_safetensors_shared_parameters_remain_strict(tmp_path):
    from safetensors.torch import save_model, save_file
    from train_4rc_stage2 import load_model_weights, read_weights
    norm = torch.nn.LayerNorm(4)
    original = torch.nn.ModuleList([norm, norm])
    checkpoint = tmp_path / "model.safetensors"
    save_model(original, str(checkpoint))
    restored_norm = torch.nn.LayerNorm(4)
    restored = torch.nn.ModuleList([restored_norm, restored_norm])
    load_model_weights(restored, checkpoint)
    torch.testing.assert_close(restored[1].weight, original[0].weight)
    state = read_weights(checkpoint)
    state.pop(next(key for key in state if key.endswith("weight")))
    invalid = tmp_path / "invalid.safetensors"
    save_file(state, str(invalid))
    with pytest.raises(RuntimeError):
        load_model_weights(restored, invalid)

