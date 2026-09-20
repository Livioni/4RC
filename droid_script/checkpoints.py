"""DROID configuration, portable split snapshots and explicit weight migration."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import runpy
import shutil

import torch

LOGGER = logging.getLogger(__name__)
ARCHITECTURE_KEYS = ("num_arms", "geometry_frame", "tcp_frame", "frame_rate", "gripper_encoding", "padding")


def load_run_config(args, stage):
    resume = Path(args.resume).expanduser() if args.resume else None
    path = Path(args.config) if args.config else (
        resume / "config.json" if resume else Path(__file__).resolve().parents[1] / "configs" / "train" / f"4rc-stage{stage}-droid.py")
    if path.suffix == ".json":
        config = json.loads(path.read_text())
    else:
        values = runpy.run_path(str(path))
        config = values.get("config", {key: value for key, value in values.items()
                                      if not key.startswith("_") and not callable(value)})
    for key, value in vars(args).items():
        if key not in ("config", "eval_only", "data_root") and value is not None:
            config[key] = value
    if getattr(args, "data_root", None):
        config["data_sources"][0]["options"]["root"] = args.data_root
    resume = Path(config["resume"]).expanduser() if config.get("resume") else None
    expected = {"num_arms": 1, "geometry_frame": "robot_base", "tcp_frame": "camera",
                "frame_rate": 15, "gripper_encoding": "continuous", "padding": [1, 1, 1, 1]}
    for key, value in expected.items():
        actual = config.get(key)
        if key == "padding" and actual is not None:
            actual = list(actual)
        if actual != value:
            raise ValueError(f"DROID requires {key}={value!r}, got {actual!r}")
    if config.get("normalize_geometry", False) or not config.get("train_camera_decoder"):
        raise ValueError("DROID requires metric geometry and an enabled camera decoder")
    if config["camera_loss_weight"] <= 0 or config["lr_camera"] <= 0:
        raise ValueError("Camera loss and learning rate must be positive")
    config["training_stage"] = stage
    hashes = {}
    for key in ("train_set", "val_set"):
        path = Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run python -m droid_script.prepare_droid_dataset")
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    config["split_hashes"] = hashes
    from arc.datasets.droid_index import index_metadata
    config["data_index_uuid"] = index_metadata(config["index_path"])["uuid"]
    if resume:
        saved = json.loads((resume / "config.json").read_text())
        for key in (*ARCHITECTURE_KEYS, "training_stage", "split_hashes", "data_index_uuid"):
            old, new = saved.get(key), config[key]
            if key == "padding":
                old, new = list(old or []), list(new)
            if old != new:
                raise ValueError(f"Cannot resume with changed {key}")
    return config


def snapshot_splits(config, directory):
    directory = Path(directory)
    destination = directory / "splits"
    destination.mkdir(parents=True, exist_ok=True)
    for key, name in (("train_set", "train_set.txt"), ("val_set", "val_set.txt")):
        source = Path(config[key])
        if hashlib.sha256(source.read_bytes()).hexdigest() != config["split_hashes"][key]:
            raise ValueError(f"Split file changed during training: {source}")
        if source.resolve() != (destination / name).resolve():
            shutil.copyfile(source, destination / name)


def weight_file(checkpoint):
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        path = next((path / name for name in ("model.safetensors", "pytorch_model.bin", "model.pt")
                     if (path / name).is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"No model weights found: {checkpoint}")
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


def migrate_stage1_weights(model, state):
    state = dict(state)
    key = "tcp_visual_query_encoder.arm_embedding"
    if key in state and state[key].shape[0] == 2 and model.num_arms == 1:
        state[key] = state[key].mean(0, keepdim=True)
        LOGGER.info("Migrated two arm identity embeddings by averaging")
    # Statistics are installed from the DROID training split after loading.
    for key in ("tcp_track_head.position_mean", "tcp_track_head.position_std"):
        state.pop(key, None)
    expected = model.state_dict()
    # Safetensors stores a single copy of tied DualDPT parameters.
    aliases = {}
    for key, value in expected.items():
        aliases.setdefault((value.untyped_storage().data_ptr(), value.storage_offset(), tuple(value.shape)), []).append(key)
    for names in aliases.values():
        source = next((key for key in names if key in state), None)
        if source:
            for key in names:
                state.setdefault(key, state[source])
    for key, value in state.items():
        if key in expected and value.shape != expected[key].shape:
            raise ValueError(f"Unexpected pretrained shape: {key}: {tuple(value.shape)} != {tuple(expected[key].shape)}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_prefixes = ("tcp_visual_query_encoder.", "tcp_track_head.")
    unexpected = [key for key in unexpected if not key.startswith("tcp_query_encoder.")]
    disallowed = [key for key in missing if not key.startswith(allowed_prefixes)]
    if disallowed or unexpected:
        raise ValueError(f"Unexpected pretrained omissions: {disallowed}; extra keys: {unexpected}")
    LOGGER.info("Loaded stage-one initialization: missing/new TCP parameters=%s", missing)


def load_stage1_model(pretrained_model=None, *, tcp_query_window_size=3):
    from arc.models.arc.arc import Arc
    model = Arc(num_arms=1, tcp_query_window_size=tcp_query_window_size)
    if pretrained_model:
        path = Path(pretrained_model).expanduser()
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=pretrained_model, filename="model.safetensors"))
        migrate_stage1_weights(model, read_weights(path))
    return model
