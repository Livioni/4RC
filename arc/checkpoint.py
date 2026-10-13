"""Save complete training checkpoints, retaining only the latest group."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile

from accelerate.utils import broadcast_object_list, gather_object

_CHECKPOINT_NAME = re.compile(r"(?:checkpoint-\d+|epoch-\d+|final_checkpoint)\Z")


def _on_main(accelerator, operation):
    """Share filesystem failures so workers do not wait at a trailing barrier."""
    result = [None, None]
    if accelerator.is_main_process:
        try:
            result[0] = operation()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    broadcast_object_list(result)
    if result[1] is not None:
        raise RuntimeError(f"Checkpoint save failed: {result[1]}")
    return result[0]


def save_checkpoint(accelerator, output_dir, name, *, epoch, batch_in_epoch, global_step,
                    metadata_writer=None):
    """Publish weights, optimizer/RNG state and metadata before removing older saves.

    All ranks participate in save_state. metadata_writer runs on the main rank
    inside the staging directory, before it becomes the latest checkpoint.
    """
    if not _CHECKPOINT_NAME.fullmatch(name):
        raise ValueError(f"Unsupported training checkpoint name: {name}")
    output_dir = Path(output_dir)
    accelerator.wait_for_everyone()

    def prepare():
        output_dir.mkdir(parents=True, exist_ok=True)
        return tempfile.mkdtemp(prefix=".checkpoint-pending-", dir=output_dir)

    staging = Path(_on_main(accelerator, prepare))
    error = None
    try:
        accelerator.save_state(str(staging))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [message for message in gather_object([error]) if message is not None]
    if errors:
        _on_main(accelerator, lambda: shutil.rmtree(staging))
        raise RuntimeError(f"Checkpoint state save failed: {errors}")

    def publish():
        config = output_dir / "config.json"
        if config.is_file():
            shutil.copyfile(config, staging / "config.json")
        if metadata_writer is not None:
            metadata_writer(staging)
        state = dict(epoch=epoch, batch_in_epoch=batch_in_epoch, global_step=global_step)
        (staging / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

        destination = output_dir / name
        # A resumed run can save final_checkpoint (or the same step) again.
        # Move the previous complete directory aside only after staging succeeds.
        backup = output_dir / f".checkpoint-previous-{name}"
        if backup.exists():
            if not destination.exists():
                backup.rename(destination)
            else:
                shutil.rmtree(backup)
        if destination.is_symlink():
            raise ValueError(f"Checkpoint destination must not be a symlink: {destination}")
        if destination.exists():
            destination.rename(backup)
        try:
            staging.rename(destination)
        except Exception:
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)

        for previous in output_dir.iterdir():
            if (previous != destination and not previous.is_symlink() and previous.is_dir()
                    and _CHECKPOINT_NAME.fullmatch(previous.name)
                    and (previous / "trainer_state.json").is_file()):
                shutil.rmtree(previous)

    try:
        _on_main(accelerator, publish)
    except Exception:
        _on_main(accelerator, lambda: shutil.rmtree(staging) if staging.exists() else None)
        raise
    accelerator.wait_for_everyone()
