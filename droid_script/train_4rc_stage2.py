#!/usr/bin/env python3
"""DROID stage two: monocular reconstruction, absolute cameras and single-arm action generation.

accelerate launch droid_script/train_4rc_stage2.py --stage1-checkpoint /path/to/stage1
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import count
import json
import logging
import math
from pathlib import Path
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arc.datasets import WeightedMultiSourceBatchSampler, collate_training_samples
from arc.datasets.droid import build_action_dataset
from arc.loss import TCPTrackingLoss
from arc.loss.droid_geometry import DroidGeometryLoss as GeometryLoss
from droid_script.checkpoints import load_run_config, snapshot_splits, ARCHITECTURE_KEYS
from arc.loss.action import flow_matching_loss, action_metric_sums, finalize_action_metrics
from arc.models.arc.arc import Arc
from arc.models.arc.arc_action import TCPActionPolicy
from droid_script.train_4rc_stage1 import (
    cosine_warmup_scheduler, distributed_scheduler_steps, prepare_geometry_batch,
    save_checkpoint, save_depth_preview, training_total_steps, align_resumed_scheduler,
)

LOGGER = logging.getLogger("4rc.stage2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--data-root")
    parser.add_argument("--train-set")
    parser.add_argument("--val-set")
    parser.add_argument("--index-path")
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"))
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--history-frames", type=int)
    parser.add_argument("--prediction-horizon", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def load_config(args):
    return validate_config(load_run_config(args, stage=2), eval_only=args.eval_only)


def validate_config(config, *, eval_only=False):
    config = dict(config)
    for key in ("batch_size", "history_frames", "prediction_horizon", "sampling_steps", "action_dim", "action_depth", "action_heads", "text_max_length", "validation_batches", "validation_batch_size"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key, default in (("history_tcp_gt_initial_ratio", 1.0), ("history_tcp_gt_final_ratio", 0.5)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
        config[key] = value
    if config["history_tcp_gt_final_ratio"] > config["history_tcp_gt_initial_ratio"]:
        raise ValueError("History GT ratio must stay constant or decrease")
    if config["action_dim"] % config["action_heads"] or config["action_dim"] % 4:
        raise ValueError("action_dim must be divisible by action_heads and 4")
    if config.get("normalize_geometry", False):
        raise ValueError("Stage two uses metric geometry; normalize_geometry must be False")
    if eval_only and not config.get("resume"):
        raise ValueError("--eval-only requires --resume pointing to a stage-two checkpoint")
    if not config.get("resume") and not config.get("stage1_checkpoint"):
        raise ValueError("A new stage-two run requires --stage1-checkpoint")
    config["train_batch_images"] = config["batch_size"] * config["history_frames"]
    config["scene_counts"] = (config["batch_size"],)
    config["min_views"] = config["max_views"] = config["history_frames"]
    config["min_interval"] = config["max_interval"] = 1
    config["reverse_probability"] = 0.0
    if not eval_only:
        training_total_steps(1, config.get("num_train_epochs"), config.get("max_train_steps"))
    if config.get("tcp_temporal_weight", 0) != 0:
        raise ValueError("Stage2 uses sequence indices with tcp_temporal_weight=0")
    config.pop("time_unit_seconds", None)
    config.pop("tcp_velocity_scale", None)
    config["tcp_temporal_weight"] = 0.0
    for key in ("output_dir", "stage1_checkpoint", "resume"):
        if config.get(key):
            config[key] = str(Path(config[key]).expanduser())
    config["training_stage"] = 2
    return config


def history_tcp_gt_ratio(config, global_step, total_steps):
    """Linear probability over cumulative optimizer updates, including resume.

    global_step counts completed updates: the first update uses the initial
    ratio and the last planned update uses the final ratio. A one-step run
    uses the initial ratio.
    """
    progress = min(1.0, max(0.0, global_step / max(1, total_steps - 1)))
    initial = float(config.get("history_tcp_gt_initial_ratio", 1.0))
    final = float(config.get("history_tcp_gt_final_ratio", 0.5))
    return initial + (final - initial) * progress


def weight_file(checkpoint):
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        candidates = [path / name for name in ("model.safetensors", "pytorch_model.bin", "model.pt")]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"No model weights found at {checkpoint}")
    return path


def read_weights(checkpoint):
    path = weight_file(checkpoint)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    return {key.removeprefix("model.").removeprefix("module."): value for key, value in state.items()}


def load_model_weights(model, checkpoint):
    path = weight_file(checkpoint)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_model
        # Safetensors removes aliases of shared LayerNorm parameters in DualDPT.
        # Its model loader restores aliases while still checking real omissions.
        load_model(model, str(path), strict=True, device="cpu")
    else:
        model.load_state_dict(read_weights(path), strict=True)


def build_policy(config):
    arc = Arc(tcp_query_window_size=config.get("tcp_query_window_size", 3), num_arms=1)
    if not config.get("resume"):
        checkpoint = Path(config["stage1_checkpoint"]).expanduser()
        metadata = (checkpoint if checkpoint.is_dir() else checkpoint.parent) / "config.json"
        saved = json.loads(metadata.read_text())
        for key in (*ARCHITECTURE_KEYS, "split_hashes"):
            old, new = saved.get(key), config[key]
            if key == "padding":
                old, new = list(old or []), list(new)
            if old != new:
                raise ValueError(f"Stage-one checkpoint has incompatible {key}")
        # Strict loading prevents silently starting with random recovery heads.
        LOGGER.info("Load checkpoint from stage1: %s", config["stage1_checkpoint"])
        load_model_weights(arc, config["stage1_checkpoint"])
    policy = TCPActionPolicy(
        arc, t5_model=config["t5_model"], text_max_length=config["text_max_length"],
        dim=config["action_dim"], depth=config["action_depth"], heads=config["action_heads"],
        prediction_horizon=config["prediction_horizon"],
        padding=config["padding"], decode_camera=True,
    )
    if config.get("resume"):
        checkpoint_config = Path(config["resume"]) / "config.json"
        if checkpoint_config.is_file():
            saved = json.loads(checkpoint_config.read_text())
            for key in ("action_dim", "action_depth", "action_heads", "prediction_horizon"):
                if config[key] != saved[key]:
                    raise ValueError(f"Cannot resume with changed architecture: {key}")
        load_model_weights(policy, config["resume"])
    return policy


def build_optimizer(policy, config):
    policy.requires_grad_(False)
    groups = []
    specifications = [
        ("backbone", [policy.arc.backbone], "train_backbone", "lr_backbone"),
        ("geometry_head", [policy.arc.head], "train_geometry_head", "lr_head"),
        ("camera_decoder", [policy.arc.cam_dec], "train_camera_decoder", "lr_camera"),
        ("motion_decoder", [policy.arc.motion_decoder], "train_motion_decoder", "lr_motion_decoder"),
        ("query_encoder", [policy.arc.tcp_visual_query_encoder], "train_query_encoder", "lr_query_encoder"),
        ("tcp_head", [policy.arc.tcp_track_head], "train_tcp_head", "lr_tcp_head"),
        ("history_pool", [policy.history_pool, policy.global_encoder, policy.physical_time, policy.token_type], "train_history_pool", "lr_history_pool"),
        ("action_head", [policy.dit], "train_action_head", "lr_action_head"),
        ("language_projection", [policy.language_projection], "train_language_projection", "lr_language_projection"),
    ]
    seen = set()
    for name, modules, flag, rate in specifications:
        lr = float(config[rate])
        if lr < 0:
            raise ValueError(f"{rate} must be non-negative")
        enabled = bool(config[flag]) and lr > 0
        parameters = []
        for module in modules:
            module.requires_grad_(enabled)
            if enabled:
                parameters.extend(module.parameters())
        if parameters:
            ids = {id(p) for p in parameters}
            if len(ids) != len(parameters) or seen & ids:
                raise RuntimeError(f"Duplicate optimizer parameters in {name}")
            seen.update(ids)
            groups.append({"name": name, "params": parameters, "lr": lr})
    if not groups:
        raise ValueError("No trainable stage-two parameters")
    return torch.optim.AdamW(
        groups, betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"], weight_decay=config["weight_decay"],
    )


def make_loader(dataset, config, *, validation=False):
    batch_size = config.get("validation_batch_size", 1) if validation else config["batch_size"]
    sampler = WeightedMultiSourceBatchSampler(
        dataset, images_per_batch=batch_size * config["history_frames"],
        scene_counts=(batch_size,),
        batches_per_epoch=config["validation_batches"] if validation else config.get("batches_per_epoch"),
        recent_buffer_size=config["recent_buffer_size"], seed=config["seed"] + int(validation),
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=config["num_workers"],
        pin_memory=True, persistent_workers=config["num_workers"] > 0,
        collate_fn=collate_training_samples,
        generator=torch.Generator().manual_seed(config["seed"] + int(validation)),
        **({"prefetch_factor": config.get("prefetch_factor", 2), "multiprocessing_context": "spawn"} if config["num_workers"] else {}),
    )
    return loader, sampler


def build_criteria(config):
    geometry = GeometryLoss(
        **{key: config[key] for key in ("camera_loss_weight", "camera_translation_weight", "camera_rotation_weight", "camera_fov_weight")},
        depth_weight=config["depth_loss_weight"], ray_weight=config["ray_loss_weight"],
        gamma=config["loss_gamma"], alpha=config["loss_alpha"],
        depth_valid_range=config["depth_valid_range"], gradient_scales=config["gradient_scales"],
    )
    tcp = TCPTrackingLoss(
        gripper_encoding="continuous",
        point_scale=config["tcp_point_scale"], virtual_point_radius=config["tcp_virtual_point_radius"],
        rotation_weight=config["tcp_rotation_weight"], temporal_weight=config["tcp_temporal_weight"],
        gripper_weight=config["tcp_gripper_weight"],
        gamma=config["loss_gamma"], alpha=config["loss_alpha"],
    )
    return geometry, tcp


def compute_losses(prediction, batch, geometry, tcp, config):
    reconstruction = prediction["reconstruction"]
    # Camera inverses and geometry supervision must not inherit BF16 autocast.
    with torch.autocast(device_type=batch["images"].device.type, enabled=False):
        geometry_batch = prepare_geometry_batch(dict(batch), reconstruction, normalize=False)
        geometry_loss = geometry(reconstruction, geometry_batch)
        tcp_loss = tcp(reconstruction, batch)
        action_loss = flow_matching_loss(
            prediction, position_weight=config["action_position_weight"],
            rotation_weight=config["action_rotation_weight"], gripper_weight=config["action_gripper_weight"],
        )
    objective = (
        config["geometry_loss_weight"] * geometry_loss["objective"]
        + config["tcp_loss_weight"] * tcp_loss["objective"]
        + config["action_loss_weight"] * action_loss["objective"]
    )
    logs = {"objective": objective.detach()}
    for name, values in (("geometry", geometry_loss), ("tcp", tcp_loss), ("action", action_loss)):
        logs.update({f"{name}/{key}": value.detach() for key, value in values.items()})
    for key in ("history_gt_fraction", "history_valid_fraction"):
        if key in prediction:
            logs[f"condition/{key}"] = prediction[key].detach()
    return objective, logs


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate(policy, loader, geometry, tcp, config, accelerator):
    policy.eval()
    totals = defaultdict(lambda: defaultdict(float))
    reconstruction_totals = defaultdict(float)
    elapsed = defaultdict(float)
    batches = 0
    for index, cpu_batch in enumerate(loader):
        batch = move_batch(cpu_batch, accelerator.device)
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with accelerator.autocast():
            reconstruction, features = policy.reconstruct(batch["images"], batch["tcp_query_points"])
            recovered = policy.make_condition(
                batch["images"], batch["intrinsics"], batch["instruction"], reconstruction, features,
            )
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize()
        encode_elapsed = time.perf_counter() - started
        for mode in ("recovered", "teacher_forced", "shuffled_instruction"):
            if mode == "shuffled_instruction" and not all(batch["shuffled_instruction"]):
                continue
            with accelerator.autocast():
                if mode == "teacher_forced":
                    condition = policy.make_condition(
                        batch["images"], batch["intrinsics"], batch["instruction"], reconstruction, features,
                        centres=batch["history_tcp_query_points"], valid=batch["history_tcp_valid"],
                    )
                elif mode == "shuffled_instruction":
                    from dataclasses import replace
                    text, text_valid = policy.language_encoder(batch["shuffled_instruction"])
                    condition = replace(recovered, text=policy.language_projection(text), text_valid=text_valid)
                else:
                    condition = recovered
                generator = torch.Generator(device=accelerator.device).manual_seed(config["seed"] + index)
                if accelerator.device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = policy.sample_condition(condition, steps=config["sampling_steps"], generator=generator)
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed[mode] += time.perf_counter() - started
            if mode == "recovered":
                elapsed["recovered_end_to_end"] += encode_elapsed + time.perf_counter() - started
            for name, value in action_metric_sums(prediction, batch["future_actions"], batch.get("future_action_valid")).items():
                totals[mode][name] += float(value)
        with torch.autocast(device_type=accelerator.device.type, enabled=False):
            geometry_loss = geometry(reconstruction, prepare_geometry_batch(dict(batch), reconstruction))
            tcp_loss = tcp(reconstruction, batch)
        for prefix, losses in (("geometry", geometry_loss), ("tcp", tcp_loss)):
            for name, value in losses.items():
                reconstruction_totals[f"{prefix}/{name}"] += float(value)
        batches += 1
    if not batches:
        raise RuntimeError("Validation produced no batches")
    metrics = {f"{mode}/{key}": value for mode, sums in totals.items() for key, value in finalize_action_metrics(sums).items()}
    metrics.update({key: value / batches for key, value in reconstruction_totals.items()})
    metrics.update({f"seconds_per_batch/{key}": value / batches for key, value in elapsed.items()})
    return metrics


def save_stage2(accelerator, config, name, *, epoch, batch_in_epoch, global_step):
    root = Path(config["output_dir"])
    save_checkpoint(
        accelerator, root, name, epoch=epoch, batch_in_epoch=batch_in_epoch, global_step=global_step,
    )
    if accelerator.is_main_process:
        (root / name / "config.json").write_text(json.dumps(config, indent=2))


def run_validation(model, validation_loader, geometry, tcp, config, accelerator, step):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and validation_loader is not None:
        # Independent generator in evaluation preserves training's RNG stream.
        devices = [accelerator.device.index] if accelerator.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            metrics = evaluate(accelerator.unwrap_model(model), validation_loader, geometry, tcp, config, accelerator)
        accelerator.log({f"validation/{key}": value for key, value in metrics.items()}, step=step)
        path = Path(config["output_dir"]) / "validation"
        path.mkdir(exist_ok=True)
        (path / f"step-{step:08d}.json").write_text(json.dumps(metrics, indent=2))
        LOGGER.info("Validation: %s", metrics)
    accelerator.wait_for_everyone()
    model.train()


def main():
    args = parse_args()
    config = load_config(args)
    # Evaluation only needs weights; training resume needs the complete state.
    if config.get("resume") and not args.eval_only:
        resume_dir = Path(config["resume"])
        for filename in ("trainer_state.json", "model.safetensors", "optimizer.bin", "scheduler.bin"):
            if not (resume_dir / filename).is_file():
                raise FileNotFoundError(f"Resume checkpoint is missing {filename}: {resume_dir}")
    set_seed(config["seed"])
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with=config["report_to"] or None,
        project_config=ProjectConfiguration(project_dir=config["output_dir"], logging_dir=str(Path(config["output_dir"]) / config.get("logging_dir", "logs"))),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=config.get("find_unused_parameters", True))],
    )
    logging.basicConfig(level=logging.INFO if accelerator.is_main_process else logging.WARNING)
    root = Path(config["output_dir"])
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(config, indent=2))
        snapshot_splits(config, root)
    tracker_config = {key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value for key, value in config.items()}
    init_kwargs = {}
    if config.get("wandb_run_name"):
        init_kwargs["wandb"] = {"name": config["wandb_run_name"]}
    accelerator.init_trackers("4RC-DROID-Stage2", config=tracker_config, init_kwargs=init_kwargs)
    policy = build_policy(config)
    geometry, tcp = build_criteria(config)
    geometry, tcp = geometry.to(accelerator.device), tcp.to(accelerator.device)
    # Validation is performed on the main rank with the unwrapped DDP model.
    validation_dataset = build_action_dataset(config, "validation") if accelerator.is_main_process else None
    validation_loader = make_loader(validation_dataset, config, validation=True)[0] if validation_dataset else None
    if args.eval_only:
        policy.to(accelerator.device)
        if accelerator.is_main_process and validation_loader is None:
            raise ValueError("No validation episodes; check val_set and max_episodes")
        run_validation(policy, validation_loader, geometry, tcp, config, accelerator, 0)
        accelerator.end_training()
        return

    with accelerator.main_process_first():
        dataset = build_action_dataset(config, "train")
    if not config.get("resume"):
        policy.set_action_position_stats(dataset.tcp_position_mean, dataset.tcp_position_std)
    # Never overwrite policy.arc.tcp_track_head.position_mean/std here.
    if accelerator.is_main_process:
        manifest = {"stage1_validation_overlap": config["stage1_validation_overlap"]}
        for split, mixture in (("train", dataset), ("validation", validation_dataset)):
            manifest[split] = {source.name: source.dataset.manifest() for source in mixture.sources} if mixture else {}
        (root / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    loader, sampler = make_loader(dataset, config)
    optimizer = build_optimizer(policy, config)
    policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
    steps_per_epoch = math.ceil(len(loader) / config["gradient_accumulation_steps"])
    num_train_epochs = config.get("num_train_epochs")
    total_steps = training_total_steps(
        steps_per_epoch, num_train_epochs, config.get("max_train_steps"),
    )
    warmup, scheduler_steps = distributed_scheduler_steps(
        config["warmup_steps"], total_steps, num_processes=accelerator.num_processes, split_batches=accelerator.split_batches,
    )
    raw_scheduler = cosine_warmup_scheduler(
        optimizer, warmup, scheduler_steps, eta_min_factor=config["eta_min_factor"],
    )
    scheduler = accelerator.prepare_scheduler(raw_scheduler)
    epoch_start = batch_start = global_step = 0
    if config.get("resume"):
        accelerator.load_state(config["resume"])
        state = json.loads((Path(config["resume"]) / "trainer_state.json").read_text())
        epoch_start, batch_start, global_step = state["epoch"], state["batch_in_epoch"], state["global_step"]
        LOGGER.info("Resumed epoch=%d batch=%d step=%d", epoch_start, batch_start, global_step)
        saved_scheduler_step = raw_scheduler.last_epoch
        learning_rates = align_resumed_scheduler(
            raw_scheduler, global_step,
            num_processes=accelerator.num_processes, split_batches=accelerator.split_batches,
        )
        LOGGER.info(
            "Resume LR aligned: global_step=%d scheduler_step=%d (saved=%d); "
            "phase=%s; warmup remaining=%d optimizer steps",
            global_step, raw_scheduler.last_epoch, saved_scheduler_step,
            "cosine" if global_step >= config["warmup_steps"] else "initial warmup",
            max(0, config["warmup_steps"] - global_step),
        )
        for group, lr in zip(optimizer.param_groups, learning_rates):
            LOGGER.info("Resume lr/%s = %.10g", group.get("name", "group"), lr)
    LOGGER.info(
        "Training target: %d cumulative optimizer steps; %d remaining; epoch limit=%s",
        total_steps, max(0, total_steps - global_step), num_train_epochs,
    )
    LOGGER.info(
        "Per GPU: %d clips x %d history frames = %d images; effective batch = %d clips; DiT tokens = %d",
        config["batch_size"], config["history_frames"], config["train_batch_images"],
        config["batch_size"] * accelerator.num_processes * config["gradient_accumulation_steps"],
        config["num_arms"] * config["history_frames"] + config["prediction_horizon"],
    )
    LOGGER.info("Optimizer groups: %s", [(group["name"], group["lr"]) for group in optimizer.param_groups])
    next_epoch, next_batch = epoch_start, batch_start
    epochs = count(epoch_start) if num_train_epochs is None else range(epoch_start, num_train_epochs)
    for epoch in epochs:
        if global_step >= total_steps:
            break
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        offset = batch_start if epoch == epoch_start else 0
        epoch_loader = accelerator.skip_first_batches(loader, offset) if offset else loader
        policy.train()
        progress = tqdm(epoch_loader, total=len(loader), initial=offset, disable=not accelerator.is_local_main_process)
        for index, batch in enumerate(progress, start=offset):
            with accelerator.accumulate(policy):
                with accelerator.autocast():
                    gt_ratio = history_tcp_gt_ratio(config, global_step, total_steps)
                    prediction = policy(batch, history_gt_ratio=gt_ratio)
                    objective, logs = compute_losses(prediction, batch, geometry, tcp, config)
                accelerator.backward(objective)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            next_epoch, next_batch = epoch, index + 1
            if next_batch >= len(loader):
                next_epoch, next_batch = epoch + 1, 0
            if accelerator.sync_gradients:
                global_step += 1
                metrics = {key: float(accelerator.reduce(value, reduction="mean")) for key, value in logs.items()}
                metrics["condition/history_gt_probability"] = gt_ratio
                metrics.update({f"lr/{group['name']}": group["lr"] for group in optimizer.param_groups})
                accelerator.log(metrics, step=global_step)
                if global_step % config["log_every_steps"] == 0:
                    progress.set_postfix(loss=f"{metrics['objective']:.4f}")
                if accelerator.is_main_process and config["visualize_every_steps"] > 0 and global_step % config["visualize_every_steps"] == 0:
                    save_depth_preview(root / "visuals" / f"step-{global_step:08d}.png", batch, prediction["reconstruction"])
                if config["validate_every_steps"] > 0 and global_step % config["validate_every_steps"] == 0:
                    run_validation(policy, validation_loader, geometry, tcp, config, accelerator, global_step)
                if config["checkpointing_steps"] > 0 and global_step % config["checkpointing_steps"] == 0:
                    save_stage2(accelerator, config, f"checkpoint-{global_step}", epoch=next_epoch, batch_in_epoch=next_batch, global_step=global_step)
                if global_step >= total_steps:
                    break
        batch_start = 0
        if config.get("save_each_epoch") and global_step < total_steps:
            save_stage2(accelerator, config, f"epoch-{epoch + 1}", epoch=next_epoch, batch_in_epoch=next_batch, global_step=global_step)
    run_validation(policy, validation_loader, geometry, tcp, config, accelerator, global_step)
    save_stage2(accelerator, config, "final_checkpoint", epoch=next_epoch, batch_in_epoch=next_batch, global_step=global_step)
    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
