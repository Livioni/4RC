"""Task selection before RoboTwin discovery, statistics, and Stage2 splitting."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch

from arc.datasets import build_training_dataset, WeightedMultiSourceBatchSampler
from arc.datasets.robotwin import RoboTwin4RC
from arc.datasets.robotwin_action import (
    NoActionEpisodes, RoboTwinActionDataset, build_action_dataset,
)
from test_action_dataset import write_episode


def action_config(root, **updates):
    return dict(
        data_sources=[dict(name="robotwin", options=dict(root=str(root)))],
        history_frames=8, prediction_horizon=16, validation_fraction=0,
        augment=False,
    ) | updates


@pytest.fixture(params=[RoboTwin4RC, RoboTwinActionDataset])
def dataset_factory(request):
    def build(root, **options):
        if request.param is RoboTwinActionDataset:
            options.setdefault("validation_fraction", 0)
        return request.param(root, augment=False, **options)
    return build


@pytest.mark.parametrize("options,expected", [
    ({}, ["lift", "place", "push"]),
    ({"tasks": None}, ["lift", "place", "push"]),
    ({"tasks": ["place"]}, ["place"]),
    ({"tasks": ["push", "lift"]}, ["lift", "push"]),
    ({"tasks": ("place", "lift", "place")}, ["lift", "place"]),
])
def test_task_selection_and_default_order(tmp_path, dataset_factory, options, expected):
    for task in ("push", "place", "lift"):
        write_episode(tmp_path, task=task, frames=3)
    dataset = dataset_factory(tmp_path, **options)
    assert [episode.task for episode in dataset.episodes] == expected


def test_task_filter_precedes_reads_limits_and_statistics(tmp_path, dataset_factory):
    for name in ("episode_0", "episode_1"):
        write_episode(tmp_path, task="selected", name=name, frames=3)
    reference = dataset_factory(tmp_path, max_episodes=1)
    # Both excluded tasks sort before the selected task. One is corrupt and
    # the other would contaminate position statistics if it were included.
    broken = write_episode(tmp_path, task="a_broken", frames=3)
    (broken / "extrinsics/third_views.npy").write_bytes(b"invalid numpy file")
    other = write_episode(tmp_path, task="b_other", frames=3)
    for arm in ("left", "right"):
        path = other / "TCP_third" / f"{arm}_state.npy"
        states = np.load(path)
        states[:, 0] += 0.5
        np.save(path, states)
    filtered = dataset_factory(tmp_path, tasks=["selected"], max_episodes=1)
    assert filtered.episodes == reference.episodes
    torch.testing.assert_close(filtered.tcp_position_mean, reference.tcp_position_mean)
    torch.testing.assert_close(filtered.tcp_position_std, reference.tcp_position_std)
    if isinstance(filtered, RoboTwinActionDataset):
        assert filtered.manifest() == reference.manifest()
        np.testing.assert_array_equal(filtered.starts[0], reference.starts[0])


@pytest.mark.parametrize("tasks", [[], (), "lift", {"lift"}, [""], [" "], [1], [None], [["lift"]]])
def test_invalid_task_selection_is_rejected(tmp_path, tasks):
    with pytest.raises(ValueError, match="tasks must be"):
        build_action_dataset(action_config(tmp_path, tasks=tasks))


@pytest.mark.parametrize("tasks", [["unknown"], ["lift", "unknown"], ["li*"], ["../lift"]])
def test_missing_tasks_report_root_and_name_before_episode_limit(tmp_path, tasks):
    write_episode(tmp_path, frames=3)
    with pytest.raises(ValueError, match="Unknown RoboTwin task") as error:
        build_action_dataset(action_config(tmp_path, tasks=tasks, max_episodes=1))
    assert str(tmp_path) in str(error.value)
    assert tasks[-1] in str(error.value)


def split_episode_names(source, task):
    names = {}
    for index in range(100):
        name = f"episode_{index}"
        fraction = int(hashlib.sha256(f"{source}/{task}/{name}".encode()).hexdigest()[:16], 16) / 2**64
        names.setdefault(fraction < 0.5, name)
        if len(names) == 2:
            return names
    raise AssertionError("Could not find train and validation episodes")


def test_mixed_sources_share_tasks_and_preserve_split_weights_and_sampling(tmp_path):
    sources = []
    for source_name, weight in (("robotwin", 1.0), ("robotwin_random", 2.0)):
        root = tmp_path / source_name
        for task in ("lift", "place"):
            for name in split_episode_names(source_name, task).values():
                write_episode(root, task=task, name=name, frames=3)
        sources.append(dict(name=source_name, weight=weight, options=dict(root=str(root))))
    sources.append(dict(name="disabled", weight=0, options=dict(root=str(tmp_path / "missing"))))
    config = action_config(tmp_path, data_sources=sources, validation_fraction=0.5)
    paths = {}
    for split in ("train", "validation"):
        full = build_action_dataset(config, split)
        selected = build_action_dataset(config | {"tasks": ["lift"]}, split)
        assert selected.source_names == ("robotwin", "robotwin_random")
        np.testing.assert_allclose(selected.weights, [1/3, 2/3])
        paths[split] = set()
        for original, filtered in zip(full.sources, selected.sources):
            expected = [e for e in original.dataset.episodes if e.task == "lift"]
            assert filtered.dataset.episodes == expected
            assert {row["task"] for row in filtered.dataset.manifest()} == {"lift"}
            assert filtered.dataset.shuffled_instructions == [""]
            paths[split].update(e.path for e in expected)
        sampler = WeightedMultiSourceBatchSampler(
            selected, images_per_batch=8, scene_counts=(1,), batches_per_epoch=4,
        )
        for batch in sampler:
            assert selected[batch[0]]["task"] == "lift"
    assert paths["train"] and paths["validation"]
    assert paths["train"].isdisjoint(paths["validation"])


@pytest.mark.parametrize("split", ["train", "validation"])
def test_every_enabled_source_must_contain_selected_tasks(tmp_path, split):
    write_episode(tmp_path / "standard", task="lift", frames=3)
    write_episode(tmp_path / "random", task="place", frames=3)
    config = action_config(tmp_path, tasks=["lift"], data_sources=[
        dict(name="standard", options=dict(root=str(tmp_path / "standard"))),
        dict(name="random", options=dict(root=str(tmp_path / "random"))),
    ])
    with pytest.raises(ValueError, match="Unknown RoboTwin task") as error:
        build_action_dataset(config, split)
    assert str(tmp_path / "random") in str(error.value)
    assert "lift" in str(error.value)


def test_global_tasks_cannot_be_overridden_by_source_options(tmp_path):
    for task in ("lift", "place"):
        write_episode(tmp_path, task=task, frames=3)
    config = action_config(tmp_path, tasks=["lift"])
    config["data_sources"][0]["options"]["tasks"] = ["place"]
    mixture = build_action_dataset(config)
    assert [e.task for e in mixture.sources[0].dataset.episodes] == ["lift"]


def test_empty_selected_data_and_splits_keep_existing_failure_behavior(tmp_path):
    write_episode(tmp_path, task="lift", frames=3)
    (tmp_path / "empty").mkdir()
    with pytest.raises(RuntimeError, match="No valid RoboTwin episodes.*empty"):
        build_action_dataset(action_config(tmp_path, tasks=["empty"]))
    write_episode(tmp_path, task="no_future", frames=1)
    config = action_config(tmp_path, tasks=["no_future"])
    with pytest.raises(NoActionEpisodes, match="No train action episodes"):
        build_action_dataset(config)
    assert build_action_dataset(config, "validation") is None
    assert build_action_dataset(action_config(tmp_path, tasks=["lift"]), "validation") is None


def test_stage1_does_not_apply_stage2_top_level_tasks(tmp_path):
    for task in ("lift", "place"):
        write_episode(tmp_path, task=task, frames=3)
    mixture = build_training_dataset(dict(data_root=str(tmp_path), tasks=["lift"], augment=False))
    assert [e.task for e in mixture.sources[0].dataset.episodes] == ["lift", "place"]


@pytest.mark.parametrize("saved_tasks", [None, ["place"]])
def test_saved_and_explicit_configs_preserve_task_selection(tmp_path, monkeypatch, saved_tasks):
    import train_4rc_stage2 as runner
    for task in ("lift", "place"):
        write_episode(tmp_path / "data", task=task, frames=3)
    config_path = Path(__file__).resolve().parents[1] / "configs/train/4rc-stage2-action.py"
    config = {k: v for k, v in runpy.run_path(str(config_path)).items() if not k.startswith("_")}
    config.update(action_config(tmp_path / "data"))
    if saved_tasks is None:
        config.pop("tasks")  # Old checkpoint configurations do not have this key.
    else:
        config["tasks"] = saved_tasks
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["train_4rc_stage2.py", "--resume", str(checkpoint)])
    restored = runner.load_config(runner.parse_args())
    assert restored.get("tasks") == saved_tasks
    dataset = build_action_dataset(restored).sources[0].dataset
    assert [e.task for e in dataset.episodes] == (saved_tasks or ["lift", "place"])

    override = tmp_path / "train.py"
    config["tasks"] = ["lift"]
    override.write_text("\n".join(f"{key} = {value!r}" for key, value in config.items()))
    monkeypatch.setattr(sys, "argv", [
        "train_4rc_stage2.py", "--resume", str(checkpoint), "--config", str(override),
    ])
    restored = runner.load_config(runner.parse_args())
    assert restored["resume"] == str(checkpoint)
    assert restored["tasks"] == ["lift"]
    assert [e.task for e in build_action_dataset(restored).sources[0].dataset.episodes] == ["lift"]
