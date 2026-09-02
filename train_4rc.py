#!/usr/bin/env python3
"""Single-stage geometry and sparse TCP tracking training on RoboTwin.

Example:
    accelerate launch train_4rc.py --config configs/train/4rc-giant-train.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arc.models.arc.arc import Arc
from arc.datasets import (
    FixedImageBatchSampler,
    RoboTwin4RC,
    collate_clips,
    views_from_batch,
)
from arc.datasets.utils import (
    compute_gt_ray_map,
    geometry_to_first_camera,
    resize_ray_valid_mask,
)
from arc.loss import GeometryLoss, TCPTrackingLoss


LOGGER = logging.getLogger("4rc.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/4rc-giant-train.py")
    parser.add_argument("--data-root")
    parser.add_argument("--pretrained-model")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    namespace = runpy.run_path(args.config)
    if isinstance(namespace.get("config"), dict):
        config = dict(namespace["config"])
    else:
        config = {
            key: value
            for key, value in namespace.items()
            if not key.startswith("_") and key != "config" and not callable(value)
        }
    if not config:
        raise ValueError(f"No public configuration values found in {args.config}")
    overrides = {
        "data_root": args.data_root,
        "pretrained_model": args.pretrained_model,
        "resume": args.resume,
        "output_dir": args.output_dir,
        "max_episodes": args.max_episodes,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def load_model(pretrained_model: str | None) -> Arc:
    if not pretrained_model:
        return Arc()
    path = Path(pretrained_model).expanduser()
    if path.is_file():
        model = Arc()
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(path))
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            key.removeprefix("model."): value for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        LOGGER.info("Loaded local weights (%d missing, %d unexpected)", len(missing), len(unexpected))
        return model
    return Arc.from_pretrained(pretrained_model)


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min_factor: float,
) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return eta_min_factor + (1.0 - eta_min_factor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, scale)


def build_optimizer(model: Arc, config: dict[str, Any]) -> torch.optim.AdamW:
    groups = []
    for name, module, learning_rate in (
        ("backbone", model.backbone, config["lr_backbone"]),
        ("geometry_head", model.head, config["lr_head"]),
        ("motion_decoder", model.motion_decoder, config["lr_motion_decoder"]),
        ("tcp_query_encoder", model.tcp_query_encoder, config["lr_tcp"]),
        ("tcp_track_head", model.tcp_track_head, config["lr_tcp"]),
    ):
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": learning_rate, "name": name})
    if not groups:
        raise RuntimeError("No trainable model parameters")
    return torch.optim.AdamW(
        groups,
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_epsilon"],
        weight_decay=config["weight_decay"],
    )


def prepare_geometry_batch(
    batch: dict[str, Any],
    predictions: dict,
    *,
    normalize: bool = False,
) -> dict[str, Any]:
    depth, extrinsics, scale = geometry_to_first_camera(
        batch["depth"],
        batch["valid_mask"],
        batch["intrinsics"],
        batch["extrinsics"],
        normalize=normalize,
    )
    batch["depth"] = depth
    batch["extrinsics"] = extrinsics
    batch["scale"] = scale
    ray_height, ray_width = predictions["ray"].shape[-3:-1]
    image_height, image_width = batch["images"].shape[-2:]
    batch["ray_map"] = compute_gt_ray_map(
        extrinsics,
        batch["intrinsics"],
        ray_height,
        ray_width,
        image_height,
        image_width,
    )
    batch["ray_valid_mask"] = resize_ray_valid_mask(
        batch["original_mask"], ray_height, ray_width
    )
    return batch


def save_depth_preview(path: Path, batch: dict[str, Any], predictions: dict) -> None:
    padding = batch["padding"][0].detach().cpu().tolist()
    left, right, top, bottom = padding
    image = batch["images"][0, 0, :, top:-bottom, left:-right]
    image = ((image.detach().float().cpu() + 1.0) * 127.5).clamp(0, 255).byte()
    image = image.permute(1, 2, 0).numpy()
    pred = predictions["depth"][0, 0, top:-bottom, left:-right].detach().float().cpu()
    target = batch["depth"][0, 0, top:-bottom, left:-right].detach().float().cpu()
    valid = batch["valid_mask"][0, 0, top:-bottom, left:-right].detach().cpu()
    if valid.any():
        maximum = torch.quantile(target[valid], 0.98).clamp_min(1e-6)
    else:
        maximum = target.new_tensor(1.0)

    def depth_image(depth: torch.Tensor) -> np.ndarray:
        gray = (depth / maximum).clamp(0, 1).mul(255).byte().numpy()
        return np.repeat(gray[..., None], 3, axis=-1)

    preview = np.concatenate((image, depth_image(target), depth_image(pred)), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(path)


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    name: str,
    *,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
) -> None:
    checkpoint_dir = output_dir / name
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        state = {
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "global_step": global_step,
        }
        with (checkpoint_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    config = load_config(args)
    output_dir = Path(config["output_dir"]).expanduser()
    project_config = ProjectConfiguration(
        project_dir=str(output_dir), logging_dir=str(output_dir / config["logging_dir"])
    )
    # DualDPT keeps prediction layers for every ray-pyramid level for checkpoint
    # compatibility, while the geometry objective supervises only the final level.
    # Those intermediate prediction layers therefore intentionally have no grad.
    ddp = DistributedDataParallelKwargs(
        find_unused_parameters=config.get("find_unused_parameters", True)
    )
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with=config.get("report_to"),
        project_config=project_config,
        kwargs_handlers=[ddp],
    )
    logging.basicConfig(
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(config["seed"], device_specific=True)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    tracker_config = {
        key: json.dumps(value) if isinstance(value, (tuple, list, dict)) else value
        for key, value in config.items()
    }
    accelerator.init_trackers("4RC-RoboTwin-TCP", config=tracker_config)

    dataset = RoboTwin4RC(
        config["data_root"],
        view=config["view"],
        min_views=config["min_views"],
        max_views=config["max_views"],
        min_interval=config["min_interval"],
        max_interval=config["max_interval"],
        reverse_probability=config.get("reverse_probability", 0.5),
        frame_rate=config.get("frame_rate"),
        max_tcp_linear_speed=config.get("max_tcp_linear_speed", 3.0),
        max_tcp_angular_speed=config.get("max_tcp_angular_speed", 4.0 * math.pi),
        seed=config["seed"],
        augment=config["augment"],
        max_episodes=config.get("max_episodes"),
    )
    batch_sampler = FixedImageBatchSampler(
        dataset,
        images_per_batch=config["train_batch_images"],
        scene_counts=config["scene_counts"],
        batches_per_epoch=config.get("batches_per_epoch"),
        recent_buffer_size=config["recent_buffer_size"],
        seed=config["seed"],
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_clips,
    )

    model = load_model(config.get("pretrained_model"))
    trainable = model.configure_trainable_modules(
        backbone=config["train_backbone"],
        geometry_head=config["train_geometry_head"],
        camera_decoder=config["train_camera_decoder"],
        motion_decoder=config["train_motion_decoder"],
        tcp_tracker=config.get("train_tcp_tracker", True),
    )
    optimizer = build_optimizer(model, config)
    geometry_criterion = GeometryLoss(
        depth_weight=config["depth_loss_weight"],
        ray_weight=config["ray_loss_weight"],
        gamma=config["loss_gamma"],
        alpha=config["loss_alpha"],
        depth_valid_range=config["depth_valid_range"],
        gradient_scales=config["gradient_scales"],
    )
    tcp_criterion = TCPTrackingLoss(
        point_scale=config.get("tcp_point_scale", 0.1),
        virtual_point_radius=config.get("tcp_virtual_point_radius", 0.03),
        rotation_weight=config.get("tcp_rotation_weight", 0.5),
        temporal_weight=config.get("tcp_temporal_weight", 0.2),
        gripper_weight=config.get("tcp_gripper_weight", 0.2),
        velocity_scale=config.get("tcp_velocity_scale", 1.0),
        gamma=config["loss_gamma"],
        alpha=config["loss_alpha"],
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    geometry_criterion = geometry_criterion.to(accelerator.device)
    tcp_criterion = tcp_criterion.to(accelerator.device)

    steps_per_epoch = math.ceil(len(dataloader) / config["gradient_accumulation_steps"])
    planned_steps = steps_per_epoch * config["num_train_epochs"]
    total_steps = config.get("max_train_steps") or planned_steps
    scheduler = cosine_warmup_scheduler(
        optimizer,
        config["warmup_steps"],
        total_steps,
        config["eta_min_factor"],
    )
    scheduler = accelerator.prepare_scheduler(scheduler)

    initial_epoch = 0
    initial_batch = 0
    global_step = 0
    if config.get("resume"):
        resume_dir = Path(config["resume"]).expanduser()
        accelerator.load_state(str(resume_dir))
        with (resume_dir / "trainer_state.json").open(encoding="utf-8") as handle:
            trainer_state = json.load(handle)
        initial_epoch = int(trainer_state["epoch"])
        initial_batch = int(trainer_state["batch_in_epoch"])
        global_step = int(trainer_state["global_step"])
        LOGGER.info("Resumed epoch=%d batch=%d step=%d", initial_epoch, initial_batch, global_step)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    LOGGER.info("Dataset: %d episodes; trainable modules: %s", len(dataset), trainable)
    LOGGER.info(
        "TCP trajectory filtering: %d invalid transitions across %d episodes",
        dataset.invalid_transition_count,
        dataset.segmented_episode_count,
    )
    LOGGER.info(
        "Per-device dynamic batches: %s (%d images)",
        batch_sampler.active_combinations,
        batch_sampler.images_per_batch,
    )
    LOGGER.info("Trainable parameters: %s / %s", f"{trainable_parameters:,}", f"{total_parameters:,}")

    stop_training = global_step >= total_steps
    next_epoch_state = initial_epoch
    next_batch_state = initial_batch
    for epoch in range(initial_epoch, config["num_train_epochs"]):
        if stop_training:
            break
        dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        model.train()
        batch_offset = initial_batch if epoch == initial_epoch else 0
        epoch_dataloader = (
            accelerator.skip_first_batches(dataloader, batch_offset)
            if batch_offset
            else dataloader
        )
        progress = tqdm(
            total=len(dataloader),
            initial=batch_offset,
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch + 1}",
        )
        for local_batch_index, batch in enumerate(epoch_dataloader):
            batch_index = batch_offset + local_batch_index
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    predictions = model(
                        views_from_batch(batch),
                        inference_track=False,
                        decode_camera=False,
                        decode_motion=False,
                        tcp_query_state=batch["tcp_state"][:, 0],
                        decode_tcp=True,
                        return_aux_pyramid=False,
                        ref_view_strategy="first",
                    )
                batch = prepare_geometry_batch(
                    batch,
                    predictions,
                    normalize=config.get("normalize_geometry", False),
                )
                geometry_losses = geometry_criterion(predictions, batch)
                tcp_losses = tcp_criterion(predictions, batch)
                total_objective = (
                    geometry_losses["objective"]
                    + config.get("tcp_loss_weight", 1.0) * tcp_losses["objective"]
                )
                losses = {f"geometry/{key}": value for key, value in geometry_losses.items()}
                losses.update({f"tcp/{key}": value for key, value in tcp_losses.items()})
                losses["objective"] = total_objective
                accelerator.backward(total_objective)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.update(1)
            next_epoch_state = epoch
            next_batch_state = batch_index + 1
            if next_batch_state >= len(dataloader):
                next_epoch_state = epoch + 1
                next_batch_state = 0
            if accelerator.sync_gradients:
                global_step += 1
                metrics = {
                    key: value.detach().float().item() for key, value in losses.items()
                }
                for group in optimizer.param_groups:
                    metrics[f"lr/{group.get('name', 'group')}"] = group["lr"]
                accelerator.log(metrics, step=global_step)
                if global_step % config["log_every_steps"] == 0:
                    progress.set_postfix(objective=f"{metrics['objective']:.4f}")
                if (
                    accelerator.is_main_process
                    and config["visualize_every_steps"] > 0
                    and global_step % config["visualize_every_steps"] == 0
                ):
                    save_depth_preview(
                        output_dir / "visuals" / f"step-{global_step:08d}.png",
                        batch,
                        predictions,
                    )
                if (
                    config["checkpointing_steps"] > 0
                    and global_step % config["checkpointing_steps"] == 0
                ):
                    save_checkpoint(
                        accelerator,
                        output_dir,
                        f"checkpoint-{global_step}",
                        epoch=next_epoch_state,
                        batch_in_epoch=next_batch_state,
                        global_step=global_step,
                    )
                if global_step >= total_steps:
                    stop_training = True
                    break
        progress.close()
        initial_batch = 0
        if config.get("save_each_epoch") and not stop_training:
            save_checkpoint(
                accelerator,
                output_dir,
                f"epoch-{epoch + 1}",
                epoch=epoch + 1,
                batch_in_epoch=0,
                global_step=global_step,
            )
        if stop_training:
            break

    save_checkpoint(
        accelerator,
        output_dir,
        "final_checkpoint",
        epoch=next_epoch_state,
        batch_in_epoch=next_batch_state,
        global_step=global_step,
    )
    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    finally:
        # ``Accelerator.end_training`` handles the normal path. Also clean up
        # after an exception so NCCL does not report a leaked process group.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
