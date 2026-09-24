import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import stage2_sliding_window_inference as inference
from arc.models.arc.arc_action import TCPActionPolicy
from stage2_helpers import TinyArc, TinyLanguage, tiny_batch


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "episode_0"
    for directory in ("images/third_views", "intrinsics", "extrinsics", "TCP_third"):
        (path / directory).mkdir(parents=True)
    frames = 17
    for index in range(frames):
        Image.fromarray(np.full((240, 320, 3), index, dtype=np.uint8)).save(path / f"images/third_views/{index}.png")
    k = np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1]], dtype=np.float32)
    np.save(path / "intrinsics/third_views.npy", k)
    extrinsics = np.broadcast_to(np.eye(4, dtype=np.float32), (frames, 4, 4)).copy()
    np.save(path / "extrinsics/third_views.npy", extrinsics)
    for arm, x in zip(inference.ARMS, (-0.2, 0.2)):
        state = np.zeros((frames, 7), dtype=np.float32)
        state[:, 0], state[:, 2] = x, 1
        np.save(path / f"TCP_third/{arm}_state.npy", state)
    (path / "metadata.json").write_text(json.dumps({"frequency_hz": 15, "instructions": ["", "lift shoes"]}))
    return inference.load_episode(path, "third_views")


def test_preprocessing_and_input_validation(episode):
    queries = inference.initial_truth_queries(episode)
    images, k, q = inference.prepare_images(episode.image_paths[:8], episode.intrinsics, queries)
    assert images.shape == (1, 8, 3, 252, 322)
    assert images[0, 0].eq(-1).all()
    np.testing.assert_allclose(q[0], queries + [1, 6])
    np.testing.assert_allclose(k[0, 0, :2, 2], [161, 126])
    assert episode.instruction == "lift shoes"
    episode.image_paths[2].unlink()
    with pytest.raises(ValueError, match="contiguous"):
        inference.load_episode(episode.path, episode.view)


@pytest.mark.parametrize("frames,stride,expected", [(8, 7, [(0, 8)]), (17, 7, [(0, 8), (7, 15), (9, 17)]),
    (10, 1, [(0, 8), (1, 9), (2, 10)]), (15, 7, [(0, 8), (7, 15)])])
def test_windows(frames, stride, expected):
    assert inference.build_windows(frames, 8, stride) == expected


@pytest.mark.parametrize("frames,stride", [(7, 7), (8, 0), (16, 8)])
def test_invalid_windows(frames, stride):
    with pytest.raises(ValueError):
        inference.build_windows(frames, 8, stride)


def test_camera_alignment_and_cloud():
    extrinsics = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
    extrinsics[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    extrinsics[1, :3, 3] = [0.5, 0, 0]
    world = np.array([[0.1, 0.2, 1], [-0.1, -0.2, 1]], dtype=np.float32)
    observed = np.einsum("kij,aj->kai", extrinsics[:, :3, :3], world) + extrinsics[:, None, :3, 3]
    rotations = np.broadcast_to(extrinsics[:, None, :3, :3], (2, 2, 3, 3)).copy()
    positions, rotation = inference.transform_history(observed, rotations, extrinsics)
    np.testing.assert_allclose(positions, np.broadcast_to(observed[-1:], positions.shape), atol=1e-6)
    np.testing.assert_allclose(rotation[0], rotation[1], atol=1e-6)
    points, colors = inference.make_point_cloud(np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1, 3), dtype=np.uint8),
                                               np.eye(3), extrinsics[0], extrinsics[1])
    np.testing.assert_allclose(points[0], [0.5, 0, 1])
    empty, _ = inference.make_point_cloud(np.zeros((1, 1)), np.ones((1, 1)), colors.reshape(1, 1, 3),
                                          np.eye(3), extrinsics[0], extrinsics[1])
    assert empty.shape == (0, 3)


def test_single_forward_and_reproducibility():
    torch.set_num_threads(2)
    policy = TCPActionPolicy(TinyArc(), language_encoder=TinyLanguage(), dim=32, depth=2, heads=4).eval()
    batch = tiny_batch()
    args = (policy, batch["images"], batch["intrinsics"], batch["tcp_query_points"], "lift shoes")
    first = inference.infer_stage2_window(*args, steps=2)
    assert policy.arc.calls == 1
    assert first["depth"].shape == (8, 42, 42)
    assert first["action_position"].shape == (16, 2, 3)
    second = inference.infer_stage2_window(*args, steps=2)
    np.testing.assert_array_equal(first["action_position"], second["action_position"])


def test_truth_partial_missing_and_camera_transform(episode):
    truth = inference.load_tcp_truth(episode)
    episode.extrinsics[:, 0, 3] = np.arange(17) * 0.01
    truth[:, :, 0] += np.arange(17)[:, None] * 0.01
    gt = inference.future_truth(episode, truth, 14, 16)
    assert gt["valid"].sum() == 2
    np.testing.assert_allclose(gt["position"][:2, :, 0], [[-0.06, 0.34]] * 2, atol=1e-6)
    assert not inference.future_truth(episode, truth, 16, 16)["valid"].any()
    assert not inference.future_truth(episode, None, 7, 16)["valid"].any()
    (episode.path / "TCP_third/left_state.npy").unlink()
    assert inference.load_tcp_truth(episode) is None


def fake_window(policy, images, intrinsics, query_points, instruction, **kwargs):
    frames = images.shape[1]
    # Fixture RGB encodes the absolute source frame; timestamps are not inputs.
    frame_index = np.rint((images[0, :, 0, 10, 10].numpy() + 1) * 255 / 2).astype(int)
    positions = np.zeros((frames, 2, 3), dtype=np.float32)
    positions[..., 0] = frame_index[:, None] * 0.001 + [-0.2, 0.2]
    positions[..., 2] = 1
    return {"depth": np.ones((frames, 252, 322), dtype=np.float32),
        "depth_confidence": np.ones((frames, 252, 322), dtype=np.float32),
        "history_position": positions, "history_rotation": np.broadcast_to(np.eye(3), (frames, 2, 3, 3)).copy(),
        "history_gripper_probability": np.zeros((frames, 2)), "history_confidence": np.ones((frames, 2)),
        "history_valid": np.ones((frames, 2), dtype=bool), "success": True,
        "action_position": np.broadcast_to(positions[-1], (16, 2, 3)).copy(),
        "action_rotation": np.broadcast_to(np.eye(3), (16, 2, 3, 3)).copy(), "action_gripper": np.zeros((16, 2), dtype=int),
        "action_gripper_score": np.zeros((16, 2)), "future_step_indices": np.arange(1, 17)}


def test_episode_propagation_and_memory_geometry(episode, tmp_path, monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append(args[3].numpy().copy())
        return fake_window(*args, **kwargs)
    monkeypatch.setattr(inference, "infer_stage2_window", run)
    policy = type("Policy", (), {"prediction_horizon": 16})()
    output = tmp_path / "result"
    result = inference.infer_episode_sliding_windows(policy, episode, inference.initial_truth_queries(episode), output,
        config={"history_frames": 8}, query_source="test")
    assert result["complete"] and len(result["windows"]) == 3
    # Final tail-aligned window starts at frame 9, not at previous window's last frame 14.
    np.testing.assert_allclose(calls[2][0, :, 0], 161 + 100 * (np.array([-0.2, 0.2]) + 0.009), atol=1e-5)
    stored = json.loads((output / "predictions.json").read_text())
    assert stored["windows"][-1]["ground_truth"]["position"][0][0] == [None] * 3
    assert stored["windows"][-1]["metrics"]["position_ade_m"] is None
    assert len(result["_geometry"]) == 3
    assert result["_geometry"][0]["depth"].shape == (8, 240, 320)
    assert "_geometry" not in stored and "geometry_file" not in stored["windows"][0]
    assert not list(output.rglob("*.npz")) and not list(output.rglob("*.npy"))
    slots = inference.build_playback_frames(result)
    assert len(slots) == 17
    assert slots[7] == (0, 7)  # Shared boundary is displayed only once.
    assert slots[8] == (1, 1)
    assert slots[-1] == (2, 7)  # Tail-aligned window still reaches the final frame.


def test_invalid_propagation_saves_partial_output(episode, tmp_path, monkeypatch):
    def invalid(*args, **kwargs):
        result = fake_window(*args, **kwargs)
        result["history_position"][-1, 0, 2] = -1
        result["success"] = False
        result["action_position"][:] = np.nan
        return result
    monkeypatch.setattr(inference, "infer_stage2_window", invalid)
    policy = type("Policy", (), {"prediction_horizon": 16})()
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="behind-camera"):
        inference.infer_episode_sliding_windows(policy, episode, inference.initial_truth_queries(episode), output,
            config={"history_frames": 8}, query_source="test")
    saved = json.loads((output / "predictions.json").read_text())
    assert not saved["complete"] and len(saved["windows"]) == 1 and "error" in saved
    assert saved["windows"][0]["action_position"][0][0] == [None] * 3


def test_no_dependency_on_reference_or_train_scripts():
    import ast
    source = Path(inference.__file__).read_text()
    imports = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert not any(name and (name.startswith("train_") or name == "tcp_sliding_window_inference") for name in imports)


def test_interactive_selection_without_truth(episode, monkeypatch):
    """Exercise actual Gradio callbacks without starting a browser/server/model."""
    from argparse import Namespace
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    for name in ("left_state.npy", "right_state.npy"):
        (episode.path / "TCP_third" / name).unlink()
    captured = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: captured.append(self))
    inference.start_interactive_page(Namespace(ui_host="127.0.0.1", ui_port=7860), episode,
                                     torch.device("cpu"), torch.float32)
    demo = captured[0]
    select = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "select")
    first = select([], gr.SelectData(None, {"index": [140, 120], "value": None}))
    assert first[1] == [[140, 120]] and not first[3]["interactive"]
    second = select(first[1], gr.SelectData(None, {"index": [180, 120], "value": None}))
    assert second[1] == [[140, 120], [180, 120]] and second[3]["interactive"]
    third = select(second[1], gr.SelectData(None, {"index": [150, 121], "value": None}))
    assert third[1] == [[150, 121]]
    truth_button = next(block for block in demo.blocks.values()
                        if isinstance(getattr(block, "value", None), str) and block.value == "使用首帧真值投影")
    assert not truth_button.interactive
    demo.close()


def test_headless_discards_geometry(episode, tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "infer_stage2_window", fake_window)
    policy = type("Policy", (), {"prediction_horizon": 16})()
    output = tmp_path / "headless"
    result = inference.infer_episode_sliding_windows(policy, episode, inference.initial_truth_queries(episode), output,
        config={"history_frames": 8}, query_source="test", keep_geometry=False)
    assert result["complete"] and not result["_geometry"]
    assert sorted(p.name for p in output.iterdir()) == ["predictions.json"]
    with pytest.raises(ValueError, match="in-memory geometry"):
        inference.build_playback_frames(result)


def test_outside_queries_continue_sliding_windows(episode, tmp_path, monkeypatch, capsys):
    calls = []

    def outside(*args, **kwargs):
        calls.append(args[3].numpy().copy())
        result = fake_window(*args, **kwargs)
        result["history_position"][:, 0, :2] = [-2, 2]
        return result

    monkeypatch.setattr(inference, "infer_stage2_window", outside)
    policy = type("Policy", (), {"prediction_horizon": 16})()
    output = tmp_path / "outside"
    result = inference.infer_episode_sliding_windows(
        policy, episode, inference.initial_truth_queries(episode), output,
        config={"history_frames": 8}, query_source="test",
    )
    assert result["complete"] and len(calls) == 3
    for query in calls[1:]:
        np.testing.assert_array_equal(query[0, 0], [1, 245])  # Including image padding.
    stored = json.loads((output / "predictions.json").read_text())
    for window in stored["windows"][1:]:
        assert window["query_points_projected_px"][0] == [-40, 320]
        assert window["query_points_px"][0] == [0, 239]
        assert window["query_points_clipped"] == [True, False]
        assert window["history_position"][0][0] == [-2, 2, 1]
    assert "clipped to image bounds" in capsys.readouterr().out
    # User input and first-frame truth retain strict validation.
    with pytest.raises(ValueError, match="outside"):
        inference.project_queries(np.array([[-2, 2, 1], [0, 0, 1]]), episode.intrinsics)


def test_index_window_needs_no_times_or_frequency():
    from test_action_policy import policy
    model = policy(prediction_horizon=16).eval()
    batch = tiny_batch(horizon=16)
    result = inference.infer_stage2_window(
        model, batch["images"], batch["intrinsics"], batch["tcp_query_points"],
        instruction="lift shoes", steps=2,
    )
    assert result["action_position"].shape == (16, 2, 3)
    assert result["future_step_indices"].tolist() == list(range(1, 17))
    assert "future_frame_times" not in result


def test_index_episode_outputs_steps_and_record_alignment(episode, tmp_path):
    from test_action_policy import policy
    model = policy(prediction_horizon=16).eval()
    result = inference.infer_episode_sliding_windows(
        model, episode, inference.initial_truth_queries(episode), tmp_path / "index",
        config={"history_frames": 8}, query_source="test",
        steps=1, keep_geometry=False,
    )
    assert result["complete"] and result["prediction_horizon"] == 16
    assert result["format_version"] == 2 and result["source_frequency_hz"] == 15
    assert "frequency_hz" not in result  # The source rate is not an execution rate.
    for record in result["windows"]:
        assert record["future_step_indices"].tolist() == list(range(1, 17))
        assert "future_frame_times" not in record
        np.testing.assert_array_equal(record["future_frame_indices"], record["anchor_frame"] + np.arange(1, 17))
        assert record["ground_truth"]["valid"].sum() == min(16, 16 - record["anchor_frame"])
    saved = json.loads((tmp_path / "index/predictions.json").read_text())
    assert saved["time_encoding"] == "index"
    assert all("future_frame_times" not in w for w in saved["windows"])


@pytest.mark.parametrize("old_metadata", [False, True])
def test_checkpoint_loader_always_uses_sequence_indices(tmp_path, monkeypatch, old_metadata):
    from safetensors.torch import save_model
    import arc.models.arc.arc as arc_module
    import arc.models.arc.arc_action as action_module
    from test_action_policy import policy
    monkeypatch.setattr(arc_module, "Arc", TinyArc)
    monkeypatch.setattr(action_module, "FrozenT5Encoder", lambda *a, **k: TinyLanguage())
    horizon = 16
    model = policy(prediction_horizon=horizon).eval()
    save_model(model, str(tmp_path / "model.safetensors"))
    config = dict(training_stage=2, normalize_geometry=False, action_dim=32, action_depth=2,
                  action_heads=4, prediction_horizon=horizon, text_max_length=128,
                  t5_model="unused")
    if old_metadata:
        config.update(time_encoding="physical", time_unit_seconds=1/15)
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded, _ = inference.load_stage2_policy(tmp_path, torch.device("cpu"))
    assert loaded.prediction_horizon == horizon
    batch = tiny_batch()
    result = loaded.sample_actions(batch["images"], batch["instruction"], batch["tcp_query_points"],
                                   batch["intrinsics"], steps=1)
    assert result["future_step_indices"].tolist() == [list(range(1, 17))]
    assert "future_frame_times" not in result
    torch.testing.assert_close(loaded.state_dict(), model.state_dict())
