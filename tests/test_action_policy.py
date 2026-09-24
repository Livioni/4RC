import json
import runpy
import sys

import numpy as np
import pytest
import torch

from arc.action import future_actions_in_current_camera, project_tcp, safe_rotation_6d_to_matrix
from arc.models.arc.arc_action import GlobalVisualEncoder, TCPActionPolicy
from arc.loss.action import flow_matching_loss
from stage2_helpers import TinyArc, TinyLanguage, tiny_batch

torch.set_num_threads(2)


def policy(arc=None, **options):
    model = TCPActionPolicy(TinyArc() if arc is None else arc, language_encoder=TinyLanguage(), dim=32, depth=2, heads=4, **options)
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
    assert model.global_encoder.projection[-1].weight.grad.abs().sum() > 0
    assert model.global_encoder.xy_projection[-1].weight.grad.abs().sum() > 0
    assert model.token_type.weight.grad.abs().sum(-1).gt(0).all()
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


@pytest.mark.parametrize("anchor", [0, 3, 7])
@pytest.mark.parametrize("history_gt_ratio", [0., 1.])
def test_padded_dataset_history_has_finite_joint_gradients(tmp_path, anchor, history_gt_ratio):
    import train_4rc_stage2 as runner
    from arc.datasets import collate_training_samples
    from arc.datasets.robotwin_action import RoboTwinActionDataset
    from test_action_dataset import write_episode
    write_episode(tmp_path, frames=10)
    horizon = 16
    dataset = RoboTwinActionDataset(tmp_path, validation_fraction=0, augment=False,
                                   prediction_horizon=horizon)
    dataset.starts[0] = np.array([anchor - 7])
    batch = collate_training_samples([dataset.get_sample(0, 8, 42)])
    config = runpy.run_path("configs/train/4rc-stage2-action.py")
    assert "future_frame_times" not in batch
    batch.pop("frame_times")  # No physical-time condition or velocity supervision.
    geometry, tcp = runner.build_criteria(config)
    model = policy(prediction_horizon=horizon)
    prediction = model(batch, history_gt_ratio=history_gt_ratio)
    objective, _ = runner.compute_losses(prediction, batch, geometry, tcp, config)
    assert torch.isfinite(objective)
    objective.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.arc.backbone.weight.grad.abs().sum() > 0
    assert model.dit.output_projection.weight.grad.abs().sum() > 0


def test_history_mask_step_and_centre_encoding():
    model = policy().eval()
    batch = tiny_batch()
    reconstruction, features = model.reconstruct(batch["images"], batch["tcp_query_points"])
    kwargs = dict(
        centres=batch["history_tcp_query_points"], valid=batch["history_tcp_valid"],
    )
    def condition(**overrides):
        return model.make_condition(
            batch["images"], batch["intrinsics"],
            batch["instruction"], reconstruction, features, **(kwargs | overrides),
        )
    c = condition()
    assert c.history.shape == (1, 9 + 16, 32)
    assert c.history_valid.shape == (1, 8, 2)
    common = (c.history, c.text, c.text_valid, c.future_step_embedding, model.token_type.weight[1])
    v1, h1 = model.dit(torch.randn(1, 16, 20), torch.tensor([0.1]), *common, return_history=True)
    v2, h2 = model.dit(torch.randn(1, 16, 20) * 10, torch.tensor([0.9]), *common, return_history=True)
    torch.testing.assert_close(h1, h2)
    assert not torch.allclose(v1, v2)
    moved = condition(centres=batch["history_tcp_query_points"] + 2)
    torch.testing.assert_close(c.history[:, :9], moved.history[:, :9])
    assert not torch.allclose(c.history[:, 9:], moved.history[:, 9:])
    # Padded future keys remain invisible to valid actions and the full prefix.
    noise = torch.randn(1, 16, 20)
    future_valid = torch.arange(16)[None] < 7
    first, prefix = model.dit(noise, torch.tensor([0.4]), *common,
                              return_history=True, future_valid=future_valid)
    noise[:, 7:] = torch.randn_like(noise[:, 7:]) * 100
    second, changed_prefix = model.dit(noise, torch.tensor([0.4]), *common,
                                       return_history=True, future_valid=future_valid)
    torch.testing.assert_close(first[:, :7], second[:, :7])
    torch.testing.assert_close(prefix, changed_prefix)


@pytest.mark.parametrize("image_size", [(42, 42), (28, 70), (14, 14)])
def test_global_prefix_uses_only_last_frame_global_patch_channels(image_size):
    model = policy().eval()
    batch = tiny_batch(batch_size=2)
    height, width = image_size
    batch["images"] = torch.zeros(2, 8, 3, height, width)
    count = (height // 14) * (width // 14)
    patches = torch.randn(2, 8, count, 32)
    special = torch.randn(2, 8, 32)

    def condition(value, special_tokens=special):
        return model.make_condition(
            batch["images"], batch["intrinsics"], batch["instruction"], {},
            [(value, special_tokens, special_tokens)] * 4,
            centres=batch["history_tcp_query_points"], valid=batch["history_tcp_valid"],
        )

    original = condition(patches)
    assert original.history.shape == (2, count + 16, 32)
    assert original.history_valid.shape == (2, 8, 2)
    distractors = patches.clone()
    distractors[:, :-1] = torch.randn_like(distractors[:, :-1])
    distractors[:, -1, :, :16] = torch.randn_like(distractors[:, -1, :, :16])
    changed = condition(distractors, special + 100)
    torch.testing.assert_close(original.history[:, :count], changed.history[:, :count])
    # A per-patch encoder keeps the row-major order and every patch independently.
    modified = patches.clone()
    index = count // 2
    modified[0, -1, index, -16:] += torch.linspace(-2, 2, 16)
    changed = condition(modified)
    unaffected = torch.ones(2, count, dtype=torch.bool)
    unaffected[0, index] = False
    torch.testing.assert_close(original.history[:, :count][unaffected], changed.history[:, :count][unaffected])
    assert not torch.allclose(original.history[0, index], changed.history[0, index])


def test_global_patch_outside_tcp_neighbourhood_changes_actions_and_gets_gradients():
    model = policy().eval()
    batch = tiny_batch()
    batch["images"] = torch.zeros(1, 8, 3, 84, 84)
    patches = torch.randn(1, 8, 36, 32, requires_grad=True)

    def condition(value):
        return model.make_condition(
            batch["images"], batch["intrinsics"], batch["instruction"], {}, [(value, None, None)] * 4,
            centres=batch["history_tcp_query_points"], valid=batch["history_tcp_valid"],
        )

    original = condition(patches)
    modified = patches.detach().clone()
    # Bottom-right patch lies outside both TCP sampling neighbourhoods.
    modified[:, -1, -1, -16:] += torch.linspace(-3, 3, 16)
    changed = condition(modified)
    torch.testing.assert_close(original.history[:, 36:], changed.history[:, 36:])
    noise = torch.randn(1, 16, 20)
    def predict(c):
        return model.dit(noise, torch.tensor([0.5]), c.history, c.text, c.text_valid,
                         c.future_step_embedding, model.token_type.weight[1])
    velocity = predict(original)
    assert not torch.allclose(velocity, predict(changed))
    velocity.square().mean().backward()
    assert patches.grad[:, -1, -1, -16:].abs().sum() > 0
    assert patches.grad[:, :-1, -1].eq(0).all()
    assert patches.grad[..., :16].eq(0).all()


def test_global_encoder_preserves_spatial_identity_and_checks_patch_grid():
    encoder = GlobalVisualEncoder(16, 32)
    patches = torch.ones(2, 6, 16)
    time = torch.randn(2, 32)
    token_type = torch.randn(32)
    output = encoder(patches, time, token_type, image_height=28, image_width=42)
    assert output.shape == (2, 6, 32)
    # Identical visual features still identify different columns and rows.
    assert not torch.allclose(output[:, 0], output[:, 1])
    assert not torch.allclose(output[:, 0], output[:, 3])
    with pytest.raises(ValueError, match="patch count"):
        encoder(patches[:, :-1], time, token_type, image_height=28, image_width=42)


@pytest.mark.parametrize("droid", [False, True])
@pytest.mark.parametrize("enabled,rate", [(True, 1e-4), (False, 1e-4), (True, 0.)])
def test_global_optimizer_membership_and_freezing(droid, enabled, rate):
    if droid:
        from droid_script import train_4rc_stage2 as runner
        from droid_script.tests.helpers import SmallArc
        model = policy(SmallArc())
        config = runpy.run_path("configs/train/4rc-stage2-droid.py")
    else:
        import train_4rc_stage2 as runner
        model = policy()
        config = runpy.run_path("configs/train/4rc-stage2-action.py")
    config.update(train_history_pool=enabled, lr_history_pool=rate, train_backbone=False)
    optimizer = runner.build_optimizer(model, config)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    trainable = enabled and rate > 0
    for module in (model.global_encoder, model.history_pool, model.physical_time, model.token_type):
        assert all(p.requires_grad == trainable for p in module.parameters())
    assert not any(p.requires_grad for p in model.arc.backbone.parameters())
    if trainable:
        group = next(group for group in optimizer.param_groups if group["name"] == "history_pool")
        assert group["lr"] == rate
        assert {id(p) for p in model.global_encoder.parameters()} <= {id(p) for p in group["params"]}
        if not droid:
            flow_matching_loss(model(tiny_batch()))["objective"].backward()
            assert model.global_encoder.projection[-1].weight.grad.abs().sum() > 0
            assert all(p.grad is None for p in model.arc.backbone.parameters())


def test_legacy_stage2_weights_remain_incompatible(tmp_path):
    from train_4rc_stage2 import load_model_weights
    model = policy()
    old_state = {key: value for key, value in model.state_dict().items()
                 if not key.startswith("global_encoder.")}
    old_state["token_type.weight"] = old_state["token_type.weight"][:2]
    checkpoint = tmp_path / "legacy_stage2.pt"
    torch.save(old_state, checkpoint)
    with pytest.raises(RuntimeError, match="token_type.weight"):
        load_model_weights(model, checkpoint)


def test_sampling_reuses_encoding_and_reports_missing_history():
    model = policy().eval()
    batch = tiny_batch()
    result = model.sample_actions(
        batch["images"], batch["instruction"], batch["tcp_query_points"],
        batch["intrinsics"], steps=3,
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
        batch["images"], batch["intrinsics"],
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
    def __init__(self, horizon=16):
        self.horizon = horizon
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
            batch = tiny_batch(horizon=self.horizon)
        sample = {key: value[0] for key, value in batch.items()}
        sample.update(task=f"task{index}", episode=f"episode{index}")
        return sample


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("num_train_epochs", [2, None])
def test_stage2_checkpoint_resume_and_evaluation(tmp_path, monkeypatch, accumulation, num_train_epochs):
    import train_4rc_stage2 as runner
    import arc.models.arc.arc_action as action_module
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
        stage1_checkpoint=str(stage1), output_dir=str(tmp_path / "full"), batch_size=1,
        prediction_horizon=16,
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



def test_partial_future_loss_and_attention_ignore_padding():
    model = policy()
    batch = tiny_batch(batch_size=2)
    mask = torch.arange(16)[None] < torch.tensor([1, 7])[:, None]
    batch["future_action_valid"] = mask
    noise = torch.randn(2, 16, 20)
    time = torch.tensor([0.3, 0.6])
    first = model(batch, noise=noise, flow_time=time)
    first["action_velocity"].retain_grad()
    loss = flow_matching_loss(first)["objective"]
    loss.backward()
    assert first["action_velocity"].grad[~mask].eq(0).all()
    assert first["action_velocity"].grad[mask].abs().sum() > 0
    # Changing padded target/noise must not affect any supervised output.
    batch["future_actions"][~mask] = 12345
    noise[~mask] = -999
    second = model(batch, noise=noise, flow_time=time)
    torch.testing.assert_close(first["action_velocity"][mask], second["action_velocity"][mask])
    torch.testing.assert_close(loss, flow_matching_loss(second)["objective"])
    # Each trajectory contributes equally after averaging its valid offsets.
    velocity = torch.zeros(2, 16, 20, requires_grad=True)
    target = torch.ones_like(velocity)
    target[1] = 2
    result = flow_matching_loss({"action_velocity": velocity, "action_target_velocity": target,
                                 "future_action_valid": mask})
    torch.testing.assert_close(result["objective"], torch.tensor(7.5))


def test_partial_future_metrics_use_last_valid_step():
    from arc.loss.action import action_metric_sums, finalize_action_metrics
    target = tiny_batch(batch_size=2)["future_actions"]
    mask = torch.arange(16)[None] < torch.tensor([1, 3])[:, None]
    position = target[..., :3].clone()
    position[..., 0] += torch.arange(1, 17)[None, :, None]
    prediction = {
        "success": torch.ones(2, dtype=torch.bool),
        "action_position": position,
        "action_rotation": safe_rotation_6d_to_matrix(target[..., 3:9]),
        "action_gripper": torch.zeros(2, 16, 2, dtype=torch.long),
    }
    sums = action_metric_sums(prediction, target, mask)
    metrics = finalize_action_metrics(sums)
    assert metrics["position_ade_m"] == pytest.approx(1.5)
    assert metrics["position_fde_m"] == pytest.approx(2.0)
    assert metrics["gripper_accuracy"] == 1
    prediction["action_position"][~mask] = float('nan')
    prediction["action_gripper"][~mask] = 1
    changed = action_metric_sums(prediction, target, mask)
    for key in sums:
        torch.testing.assert_close(sums[key], changed[key])
    prediction["success"][:] = False
    empty = finalize_action_metrics(action_metric_sums(prediction, target, mask))
    assert empty["valid_trajectories"] == 0


def test_history_gt_schedule_endpoints_and_resume():
    from train_4rc_stage2 import history_tcp_gt_ratio
    config = {"history_tcp_gt_initial_ratio": 1.0, "history_tcp_gt_final_ratio": 0.5}
    assert history_tcp_gt_ratio(config, 0, 101) == 1.0
    assert history_tcp_gt_ratio(config, 50, 101) == 0.75
    assert history_tcp_gt_ratio(config, 100, 101) == 0.5
    assert history_tcp_gt_ratio(config, 150, 101) == 0.5
    assert history_tcp_gt_ratio(config, 0, 1) == 1.0
    # Resuming computes the same point on the curve, independent of epoch/batch.
    resumed = [history_tcp_gt_ratio(config, step, 101) for step in range(60, 101)]
    uninterrupted = [history_tcp_gt_ratio(config, step, 101) for step in range(101)]
    assert resumed == uninterrupted[60:]


def test_history_query_mixture_is_per_clip_and_detached():
    model = policy()
    batch = tiny_batch(batch_size=32)
    positions = torch.tensor([[-0.1, 0., 1.], [0.1, 0., 1.]]).expand(32, 8, -1, -1).clone().requires_grad_()
    reconstruction = {"tcp_position": positions}
    predicted, predicted_valid = project_tcp(positions.detach(), batch["intrinsics"], image_height=42, image_width=42)
    gt, valid, selected = model.training_history_queries(batch, reconstruction, 1.)
    torch.testing.assert_close(gt, batch["history_tcp_query_points"])
    assert selected.all()
    recovered, valid, selected = model.training_history_queries(batch, reconstruction, 0.)
    torch.testing.assert_close(recovered, predicted)
    torch.testing.assert_close(valid, predicted_valid)
    assert not selected.any() and not recovered.requires_grad
    torch.manual_seed(42)
    mixed, valid, selected = model.training_history_queries(batch, reconstruction, 0.5)
    assert selected.any() and (~selected).any()
    torch.testing.assert_close(mixed[selected], batch["history_tcp_query_points"][selected])
    torch.testing.assert_close(mixed[~selected], predicted[~selected])
    # Invalid predicted history stays missing instead of silently reverting to GT.
    missing, valid, selected = model.training_history_queries(batch, {"tcp_position": -torch.ones_like(positions)}, 0.)
    assert not valid.any() and not selected.any() and missing.eq(0).all()


def test_predicted_history_forward_supports_missing_queries(monkeypatch):
    model = policy()
    batch = tiny_batch()
    original = model.reconstruct
    def missing_reconstruction(*args):
        reconstruction, features = original(*args)
        reconstruction["tcp_position"] = -torch.ones_like(reconstruction["tcp_position"])
        return reconstruction, features
    monkeypatch.setattr(model, "reconstruct", missing_reconstruction)
    prediction = model(batch, history_gt_ratio=0.)
    assert prediction["history_gt_fraction"] == 0
    assert prediction["history_valid_fraction"] == 0
    objective = flow_matching_loss(prediction)["objective"]
    assert torch.isfinite(objective)
    objective.backward()
    assert model.history_pool.missing_token.grad is not None
    assert model.history_pool.missing_token.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.arc.tcp_track_head.parameters())
