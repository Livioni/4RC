"""Regression checks for cumulative training limits and resumed LR curves."""
import importlib
import math

import pytest
import torch


@pytest.fixture(params=["train_4rc_stage1", "train_4rc_stage2"])
def runner(request):
    return importlib.import_module(request.param)


def test_cumulative_limits(runner):
    assert runner.training_total_steps(5, None, 12) == 12
    assert runner.training_total_steps(5, 2, None) == 10
    assert runner.training_total_steps(5, 2, 12) == 12


@pytest.mark.parametrize("epochs,steps", [(None, None), (0, 12), (True, 12), (2, 0), (2, 1.5)])
def test_invalid_limits(runner, epochs, steps):
    with pytest.raises(ValueError):
        runner.training_total_steps(5, epochs, steps)


@pytest.mark.parametrize("processes,split_batches", [(1, False), (4, False), (4, True)])
@pytest.mark.parametrize("global_step", [2, 8, 24])
def test_resume_recomputes_extended_curve(runner, processes, split_batches, global_step):
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    old = runner.cosine_warmup_scheduler(optimizer, 4, 10, 0.1)
    for _ in range(global_step):
        optimizer.step()
        old.step()
    saved_optimizer, saved_scheduler = optimizer.state_dict(), old.state_dict()
    restored = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    warmup, total = runner.distributed_scheduler_steps(
        4, 20, num_processes=processes, split_batches=split_batches,
    )
    scheduler = runner.cosine_warmup_scheduler(restored, warmup, total, 0.1)
    restored.load_state_dict(saved_optimizer)
    scheduler.load_state_dict(saved_scheduler)
    rates = runner.align_resumed_scheduler(
        scheduler, global_step, num_processes=processes, split_batches=split_batches,
    )
    expected = 0.01 * (
        global_step / 4 if global_step < 4
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, (global_step - 4) / 16)))
    )
    assert rates == pytest.approx([expected])
    assert restored.param_groups[0]["lr"] == pytest.approx(expected)
    assert scheduler.get_last_lr() == pytest.approx([expected])
    assert scheduler.last_epoch == global_step * (1 if split_batches else processes)


@pytest.mark.parametrize("module_name", ["train_4rc_stage1", "train_4rc_stage2"])
def test_incomplete_resume_fails_before_loading_data(module_name, tmp_path, monkeypatch):
    from argparse import Namespace
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "parse_args", lambda: Namespace(eval_only=False))
    monkeypatch.setattr(module, "load_config", lambda args: {
        "resume": str(tmp_path), "num_train_epochs": None, "max_train_steps": 10,
    })
    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        module.main()
