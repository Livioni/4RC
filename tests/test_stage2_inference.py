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


@pytest.mark.parametrize("frames,stride,expected", [
    (1, 1, [(0, 1)]), (3, 1, [(0, 1), (0, 2), (0, 3)]),
    (8, 7, [(0, 1), (0, 8)]), (17, 7, [(0, 1), (0, 8), (7, 15), (9, 17)]),
    (10, 1, [(max(0, t - 7), t + 1) for t in range(10)]),
    (15, 7, [(0, 1), (0, 8), (7, 15)]),
])
def test_windows(frames, stride, expected):
    assert inference.build_windows(frames, 8, stride) == expected


@pytest.mark.parametrize("frames,stride", [(0, 1), (8, 0), (16, 8)])
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


def test_historical_truth_moving_camera_and_nonzero_source_indices(episode):
    from dataclasses import replace

    truth = inference.load_tcp_truth(episode)
    world = truth[0, :, :3].copy()
    angle = np.arange(len(truth), dtype=np.float32) * 0.03
    extrinsics = episode.extrinsics.copy()
    extrinsics[:, 0, 0] = extrinsics[:, 1, 1] = np.cos(angle)
    extrinsics[:, 0, 1], extrinsics[:, 1, 0] = -np.sin(angle), np.sin(angle)
    extrinsics[:, 0, 3] = np.arange(len(truth)) * 0.01
    truth[..., :3] = np.einsum("tij,aj->tai", extrinsics[:, :3, :3], world) + extrinsics[:, None, :3, 3]
    truth[..., 5] = angle[:, None]
    # Episode-local frame 0 is source frame 3, and the window starts later still.
    episode = replace(episode, image_paths=episode.image_paths[3:],
                      frame_indices=episode.frame_indices[3:], extrinsics=extrinsics[3:])
    gt = inference.tcp_truth_in_anchor_camera(episode, truth, np.arange(2, 10), 9)
    expected = world @ extrinsics[12, :3, :3].T + extrinsics[12, :3, 3]
    np.testing.assert_allclose(gt["position"], np.broadcast_to(expected, (8, 2, 3)), atol=1e-6)
    np.testing.assert_allclose(gt["rotation"], np.broadcast_to(extrinsics[12, :3, :3], (8, 2, 3, 3)), atol=1e-6)
    assert gt["valid"].all()

    truth[7, 0, 0] = np.nan
    gt = inference.tcp_truth_in_anchor_camera(episode, truth[:11], np.arange(2, 10), 9)
    assert gt["valid"].tolist() == [True, True, False, True, True, True, False, False]
    assert np.isnan(gt["position"][~gt["valid"]]).all()
    missing = inference.tcp_truth_in_anchor_camera(episode, None, np.arange(2, 10), 9)
    assert not missing["valid"].any() and np.isnan(missing["position"]).all()


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
        config={"history_frames": 8}, query_source="test", stride=7)
    assert result["complete"] and len(result["windows"]) == 4
    # Startup windows retain the initial query; the final window begins at frame 9.
    np.testing.assert_array_equal(calls[0], calls[1])
    np.testing.assert_allclose(calls[3][0, :, 0], 161 + 100 * (np.array([-0.2, 0.2]) + 0.009), atol=1e-5)
    stored = json.loads((output / "predictions.json").read_text())
    assert stored["windows"][-1]["ground_truth"]["position"][0][0] == [None] * 3
    assert stored["windows"][-1]["metrics"]["position_ade_m"] is None
    assert len(result["_geometry"]) == 4
    assert result["_geometry"][0]["depth"].shape == (1, 240, 320)
    assert result["_geometry"][1]["depth"].shape == (8, 240, 320)
    assert "_geometry" not in stored and "geometry_file" not in stored["windows"][0]
    assert not list(output.rglob("*.npz")) and not list(output.rglob("*.npy"))
    slots = inference.build_playback_frames(result)
    assert slots == [(0, 0), (1, 7), (2, 7), (3, 7)]
    assert [result["windows"][i]["start"] + local for i, local in slots] == [0, 7, 14, 16]


@pytest.mark.parametrize("num_frames", [1, 3, 10])
def test_causal_startup_never_reads_future_rgb(episode, tmp_path, monkeypatch, num_frames):
    from dataclasses import replace

    episode = replace(episode, image_paths=episode.image_paths[:num_frames],
                      frame_indices=episode.frame_indices[:num_frames], extrinsics=episode.extrinsics[:num_frames])
    reads, calls = [], []
    original_load = inference.load_rgb
    def load(path):
        reads.append(int(path.stem))
        return original_load(path)
    def run(*args, **kwargs):
        anchor = len(calls)
        expected = np.maximum(np.arange(anchor - 7, anchor + 1), 0)
        assert reads == expected.tolist()
        reads.clear()
        rgb_indices = np.rint((args[1][0, :, 0, 10, 10].numpy() + 1) * 255 / 2).astype(int)
        np.testing.assert_array_equal(rgb_indices, expected)
        calls.append(args[3].numpy().copy())
        prediction = fake_window(*args, **kwargs)
        # Distinct reconstruction outputs for repeated slots must never turn into
        # a visible motion history or be used to replace the original query.
        padding = max(0, 7 - anchor)
        prediction["history_position"][:padding, :, 0] += 1
        return prediction
    monkeypatch.setattr(inference, "load_rgb", load)
    monkeypatch.setattr(inference, "infer_stage2_window", run)
    result = inference.infer_episode_sliding_windows(
        type("Policy", (), {"prediction_horizon": 16})(), episode,
        inference.initial_truth_queries(episode), tmp_path / "causal",
        config={"history_frames": 8}, query_source="test",
    )
    assert len(calls) == num_frames
    slots = inference.build_playback_frames(result)
    for t, record in enumerate(result["windows"]):
        assert record["anchor_frame"] == t
        np.testing.assert_array_equal(record["input_frame_indices"], np.maximum(np.arange(t - 7, t + 1), 0))
        np.testing.assert_array_equal(record["frame_indices"], np.arange(max(0, t - 7), t + 1))
        np.testing.assert_array_equal(record["future_frame_indices"], np.arange(t + 1, t + 17))
        assert record["history_padding"] == max(0, 7 - t)
        assert len(record["history_position"]) == min(t + 1, 8)
        assert record["history_position"][:, :, 0].max() < 0.3
        assert slots[t] == (t, min(t, 7))
    for query in calls[:8]:
        np.testing.assert_array_equal(query, calls[0])


@pytest.mark.parametrize("truth_available", [True, False])
def test_viewer_history_truth_and_future_anchor(episode, tmp_path, monkeypatch, truth_available):
    import contextlib
    import threading
    from types import SimpleNamespace

    queries = inference.initial_truth_queries(episode)
    if not truth_available:
        for arm in inference.ARMS:
            (episode.path / "TCP_third" / f"{arm}_state.npy").unlink()
    monkeypatch.setattr(inference, "infer_stage2_window", fake_window)
    policy = SimpleNamespace(prediction_horizon=16)
    output = tmp_path / "viewer"
    result = inference.infer_episode_sliding_windows(
        policy, episode, queries, output, config={"history_frames": 8}, query_source="test",
    )
    stored = json.loads((output / "predictions.json").read_text())
    for record in stored["windows"]:
        length = record["end"] - record["start"]
        assert len(record["history_ground_truth"]["position"]) == length
        assert record["history_ground_truth"]["valid"] == [truth_available] * length

    # Exercise frame selection and overlays without a browser or server thread.
    viewer = inference.Stage2Viewer.__new__(inference.Stage2Viewer)
    viewer.result, viewer.episode = result, episode
    viewer.frame_slots = inference.build_playback_frames(result)
    viewer._lock, viewer._closed = threading.RLock(), threading.Event()
    viewer._nodes = []
    viewer.server = SimpleNamespace(atomic=contextlib.nullcontext)
    viewer.rgb_handle, viewer.info = SimpleNamespace(), SimpleNamespace()
    for name, value in dict(frame=0, future=1, show_cloud=False, show_history=True,
                            show_history_gt=True, show_prediction=True, show_gt=True,
                            show_axes=True).items():
        setattr(viewer, name, SimpleNamespace(value=value))
    trajectories, axes = {}, {}
    viewer._trajectory = lambda name, points, *args, **kwargs: trajectories.update({name: points.copy()})
    viewer._axes = lambda name, points, rotations: axes.update({name: points.copy()})
    viewer.refresh(frame_slot=0)
    assert trajectories["history"].shape == (1, 2, 3)
    assert trajectories["history_truth"].shape == (1, 2, 3)
    np.testing.assert_array_equal(viewer.rgb_handle.image, inference.load_rgb(episode.image_paths[0]))
    assert "future frames 1–1" in viewer.info.content
    viewer.refresh(frame_slot=8)  # Frame 8 uses a prediction anchored at frame 8.
    gt = result["windows"][8]["history_ground_truth"]
    np.testing.assert_allclose(trajectories["history_truth"], gt["position"], equal_nan=True)
    assert trajectories["history_truth"].shape == (8, 2, 3)  # Independent of Future steps.
    np.testing.assert_allclose(axes["history_truth_axes"], gt["position"][-1], equal_nan=True)
    assert trajectories["truth"].shape == (2, 2, 3)
    np.testing.assert_allclose(trajectories["truth"][0], gt["position"][-1], equal_nan=True)
    if truth_available:
        # The GT anchor must not be replaced by the reconstructed anchor.
        assert not np.allclose(trajectories["truth"][0], trajectories["prediction"][0])
    else:
        assert np.isnan(trajectories["history_truth"]).all()
        assert np.isnan(trajectories["truth"]).all()

    viewer.show_history_gt.value = False
    trajectories.clear()
    axes.clear()
    viewer.refresh(frame_slot=8)
    assert "history_truth" not in trajectories and "history_truth_axes" not in axes
    assert "truth" in trajectories  # The future GT toggle remains independent.


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
            config={"history_frames": 8}, query_source="test", stride=7)
    saved = json.loads((output / "predictions.json").read_text())
    assert not saved["complete"] and len(saved["windows"]) == 2 and "error" in saved
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
        config={"history_frames": 8}, query_source="test", stride=7,
    )
    assert result["complete"] and len(calls) == 4
    np.testing.assert_array_equal(calls[0], calls[1])
    for query in calls[2:]:
        np.testing.assert_array_equal(query[0, 0], [1, 245])  # Including image padding.
    stored = json.loads((output / "predictions.json").read_text())
    for window in stored["windows"][2:]:
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
    assert result["format_version"] == 3 and result["source_frequency_hz"] == 15
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
