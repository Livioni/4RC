"""Recover TCP position learning from the existing RoboTwin-TCP checkpoint."""

from pathlib import Path
import runpy


base_path = Path(__file__).with_name("4rc-giant-train.py")
base = runpy.run_path(str(base_path))
config = {
    key: value
    for key, value in base.items()
    if not key.startswith("_") and key != "config" and not callable(value)
}
config.update(
    output_dir="outputs/4rc-robotwin-tcp-recovery",
    pretrained_model="checkpoints/RoboTwin-TCP/model.safetensors",
    train_backbone=False,
    train_geometry_head=False,
    train_camera_decoder=False,
    # tcp_tracker still trains the shared motion decoder, without enabling the
    # unused dense tracking head.
    train_motion_decoder=False,
    train_tcp_tracker=True,
    depth_loss_weight=0.0,
    ray_loss_weight=0.0,
    warmup_steps=500,
    max_train_steps=10_000,
)
