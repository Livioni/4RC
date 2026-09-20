"""Opt-in real-data smoke training with small CPU backbones, not the giant model."""
import os
from pathlib import Path

import pytest

from droid_script.tests.test_training import run_pipeline


@pytest.mark.skipif(os.environ.get("DROID_RUN_REAL_TESTS") != "1", reason="Set DROID_RUN_REAL_TESTS=1 for real data")
def test_real_droid_two_stage_pipeline(tmp_path, monkeypatch):
    root = Path(os.environ.get("DROID_DATA_ROOT", "datasets/droid_episodes")).resolve()
    index = Path(os.environ.get("DROID_INDEX_PATH", "droid_script/cache/droid.sqlite")).resolve()
    train = tmp_path / "real_train.txt"
    val = tmp_path / "real_val.txt"
    for source, destination in (("train_set.txt", train), ("val_set.txt", val)):
        names = (Path("droid_script/splits") / source).read_text().splitlines()[:2]
        destination.write_text("\n".join(names) + "\n")
    run_pipeline(dict(root=str(root), index_path=str(index), train_set=str(train), val_set=str(val)),
                 tmp_path, monkeypatch)
