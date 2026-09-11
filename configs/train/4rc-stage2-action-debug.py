"""Single-GPU stage-two debugging on a 24 GB GPU."""
import runpy as _runpy
from pathlib import Path as _Path

_base = _runpy.run_path(str(_Path(__file__).with_name("4rc-stage2-action.py")))
globals().update({k: v for k, v in _base.items() if not k.startswith("_")})

# Keep reconstruction heads and the action modules trainable within GPU memory.
train_backbone = False
gradient_accumulation_steps = 1
warmup_steps = 0
log_every_steps = 1
# Debugging should not require an external experiment-tracking login.
report_to = []
