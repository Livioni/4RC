import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

from arc.datasets import collate_training_samples
from arc.loss.action import flow_matching_loss, action_metric_sums, finalize_action_metrics
from arc.models.arc.arc_action import TCPActionPolicy
from arc.models.arc.utils.transform import extri_intri_to_pose_encoding
from arc.datasets.droid import DroidDataset
from arc.loss.droid_geometry import prepare_geometry_batch, absolute_camera_loss
from droid_script.checkpoints import migrate_stage1_weights
from droid_script.tests.helpers import SmallArc, SmallLanguage


def config_for(stage, files, output):
    config = {k: v for k, v in runpy.run_path(f"configs/train/4rc-stage{stage}-droid.py").items()
              if not k.startswith("_") and not callable(v)}
    config.update({key: value for key, value in files.items() if key != "root"})
    config["data_sources"] = [{"name": "droid", "type": "droid", "weight": 1,
                               "options": {"root": files["root"]}}]
    config.update(output_dir=str(output), report_to=[], mixed_precision="no", num_workers=0,
                  max_train_steps=2, num_train_epochs=None, batches_per_epoch=4, warmup_steps=0,
                  checkpointing_steps=1, visualize_every_steps=0, log_every_steps=1,
                  gradient_accumulation_steps=1, augment=False)
    if stage == 1:
        config.update(min_views=2, max_views=4, train_batch_images=4, scene_counts=[1, 2], pretrained_model=None)
    else:
        config.update(action_dim=32, action_depth=2, action_heads=4, batch_size=1,
                      validation_batches=1, validation_batch_size=1, sampling_steps=1, validate_every_steps=1)
    return config


def test_absolute_camera_and_ray_targets(droid_files):
    sample = DroidDataset(**droid_files, augment=False).get_sample(0, 3, 42)
    batch = collate_training_samples([sample])
    prediction = {"ray": torch.zeros(1, 3, 13, 23, 6)}
    prepared = prepare_geometry_batch(batch, prediction)
    torch.testing.assert_close(prepared["extrinsics"], batch["extrinsics"])
    assert not torch.allclose(prepared["extrinsics"][0, 0], torch.eye(4))
    c2w = torch.linalg.inv(batch["extrinsics"])
    torch.testing.assert_close(prepared["ray_map"][..., 3:], c2w[..., None, None, :3, 3].expand(1, 3, 13, 23, 3))
    encoding = extri_intri_to_pose_encoding(c2w, batch["intrinsics"], (182, 322))
    perfect = absolute_camera_loss({"pose_enc": encoding}, batch)
    assert perfect["objective"].abs() < 1e-6
    sign = encoding.clone()
    sign[..., 3:7] *= -1
    torch.testing.assert_close(perfect["objective"], absolute_camera_loss({"pose_enc": sign}, batch)["objective"])
    changed = encoding.clone().requires_grad_()
    delta = torch.zeros_like(changed)
    delta[..., 0] = .2
    loss = absolute_camera_loss({"pose_enc": changed + delta}, batch)
    loss["objective"].backward()
    assert changed.grad[..., :3].abs().sum() > 0


def test_single_arm_policy_joint_gradients_and_continuous_gripper(droid_files):
    from droid_script.train_4rc_stage2 import build_criteria, compute_losses, build_optimizer
    dataset = DroidDataset(**droid_files, stage=2, augment=False)
    batch = collate_training_samples([dataset.get_sample(0, 8, 42)])
    model = TCPActionPolicy(SmallArc(), language_encoder=SmallLanguage(), dim=32, depth=2, heads=4,
                            padding=(1, 1, 1, 1), decode_camera=True)
    model.arc.set_tcp_position_stats(dataset.tcp_position_mean, dataset.tcp_position_std)
    model.set_action_position_stats(dataset.tcp_position_mean, dataset.tcp_position_std)
    config = config_for(2, droid_files, "unused")
    optimizer = build_optimizer(model, config)
    assert "camera_decoder" in {group["name"] for group in optimizer.param_groups}
    criteria = build_criteria(config)
    before = model.arc.cam_dec.fc_t.weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        assert prediction["action_velocity"].shape == (1, 16, 10)
        objective, _ = compute_losses(prediction, batch, *criteria, config)
        objective.backward()
        assert torch.isfinite(objective)
        # TCP/DiT output layers start at zero. Their upstream modules receive
        # gradients after the first optimizer update.
        if step == 1:
            for module in (model.arc.cam_dec, model.arc.head, model.arc.motion_decoder,
                           model.arc.tcp_track_head, model.dit):
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        optimizer.step()
    assert not torch.equal(before, model.arc.cam_dec.fc_t.weight)
    model.eval()
    result = model.sample_actions(batch["images"], batch["instruction"], batch["tcp_query_points"],
                                  batch["frame_times"], batch["intrinsics"], steps=1)
    assert result["action_position"].shape == (1, 16, 1, 3)
    assert result["action_gripper_open"].shape == (1, 16, 1)
    assert ((result["action_gripper_open"] >= 0) & (result["action_gripper_open"] <= 1)).all()
    metrics = finalize_action_metrics(action_metric_sums(result, batch["future_actions"], batch["future_action_valid"]))
    assert "gripper_mae" in metrics


def test_native_weight_migration_rejects_unexpected_shapes():
    original = SmallArc(num_arms=2)
    migrated = SmallArc(num_arms=1)
    migrate_stage1_weights(migrated, original.state_dict())
    torch.testing.assert_close(migrated.tcp_visual_query_encoder.arm_embedding,
                               original.tcp_visual_query_encoder.arm_embedding.mean(0, keepdim=True))
    broken = original.state_dict()
    broken["backbone.weight"] = broken["backbone.weight"][:1]
    with pytest.raises(ValueError, match="Unexpected pretrained shape"):
        migrate_stage1_weights(migrated, broken)


def run_pipeline(files, tmp_path, monkeypatch, accumulation=1):
    import droid_script.train_4rc_stage1 as stage1
    import droid_script.train_4rc_stage2 as stage2
    import arc.models.arc.arc_action as action_module
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(stage1, "load_model", lambda *a, **kw: SmallArc())
    monkeypatch.setattr(stage2, "Arc", SmallArc)
    monkeypatch.setattr(action_module, "FrozenT5Encoder", SmallLanguage)
    configs = {}
    for stage, runner in ((1, stage1), (2, stage2)):
        output = tmp_path / f"stage{stage}"
        config = config_for(stage, files, output)
        config.update(gradient_accumulation_steps=accumulation, batches_per_epoch=4 * accumulation)
        if stage == 2:
            config["stage1_checkpoint"] = str(tmp_path / "stage1/final_checkpoint")
        config_path = tmp_path / f"stage{stage}.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
        runner.main()
        initial = stage2.read_weights(output / "final_checkpoint")
        resumed_root = tmp_path / f"resumed{stage}"
        monkeypatch.setattr(sys, "argv", ["train", "--resume", str(output / "checkpoint-1"),
                                         "--output-dir", str(resumed_root)])
        runner.main()
        resumed = stage2.read_weights(resumed_root / "final_checkpoint")
        for key in initial:
            torch.testing.assert_close(initial[key], resumed[key], msg=lambda msg: f"{stage}/{key}: {msg}")
        state = json.loads((resumed_root / "final_checkpoint/trainer_state.json").read_text())
        assert state["global_step"] == 2
        for name in ("train_set.txt", "val_set.txt"):
            assert (resumed_root / "final_checkpoint/splits" / name).is_file()
        configs[stage] = json.loads((output / "final_checkpoint/config.json").read_text())
    assert configs[1]["split_hashes"] == configs[2]["split_hashes"]
    metrics = json.loads((tmp_path / "stage2/validation/step-00000002.json").read_text())
    assert "geometry/metric_camera_translation_m" in metrics
    assert "recovered/gripper_mae" in metrics
    monkeypatch.setattr(sys, "argv", ["train", "--resume", str(tmp_path / "stage2/final_checkpoint"),
                                     "--output-dir", str(tmp_path / "evaluation"), "--eval-only"])
    stage2.main()
    assert (tmp_path / "evaluation/validation/step-00000000.json").is_file()
    return configs


@pytest.mark.parametrize("accumulation", [1, 2])
def test_two_stage_training_resume_and_eval(droid_files, tmp_path, monkeypatch, accumulation):
    run_pipeline(droid_files, tmp_path, monkeypatch, accumulation)


def test_changed_splits_rejected_on_resume(droid_files, tmp_path):
    from argparse import Namespace
    from droid_script.checkpoints import load_run_config
    config = config_for(1, droid_files, tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = Namespace(config=str(config_path), resume=None)
    loaded = load_run_config(args, 1)
    config_path.write_text(json.dumps(loaded))
    Path(droid_files["train_set"]).write_text("train_b\n")
    with pytest.raises(ValueError, match="split_hashes"):
        load_run_config(Namespace(config=None, resume=str(tmp_path)), 1)


@pytest.mark.parametrize("resume_in_config", [False, True])
def test_rebuilt_index_rejected_on_resume(droid_files, tmp_path, resume_in_config):
    from argparse import Namespace
    from arc.datasets.droid_index import ensure_index
    from droid_script.checkpoints import load_run_config
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_for(1, droid_files, tmp_path)))
    loaded = load_run_config(Namespace(config=str(config_path), resume=None), 1)
    config_path.write_text(json.dumps(loaded))
    ensure_index(droid_files["root"], droid_files["index_path"], rebuild=True)
    args = Namespace(config=None, resume=str(tmp_path))
    if resume_in_config:
        loaded["resume"] = str(tmp_path)
        next_config = tmp_path / "resume.json"
        next_config.write_text(json.dumps(loaded))
        args = Namespace(config=str(next_config), resume=None)
    with pytest.raises(ValueError, match="data_index_uuid"):
        load_run_config(args, 1)
