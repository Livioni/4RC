import pytest
import torch

from arc.datasets.droid_index import ensure_index
from droid_script.tests.helpers import write_episode

torch.set_num_threads(2)


@pytest.fixture
def droid_files(tmp_path):
    root = tmp_path / "data"
    for name in ("train_a", "train_b", "val_a"):
        write_episode(root, name)
    index = ensure_index(root, tmp_path / "index.sqlite")
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_text("train_a\ntrain_b\n")
    val.write_text("val_a\n")
    return dict(root=str(root), index_path=str(index), train_set=str(train), val_set=str(val))
