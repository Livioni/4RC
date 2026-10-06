"""Inference contracts: absolute geometry, query propagation and optional labels."""
import json
import numpy as np
from PIL import Image
import pytest
import torch

from droid_script import infer_4rc_stage1 as inference
from droid_script.tests.helpers import SmallArc, write_episode


@pytest.fixture
def episode(tmp_path):
    return inference.load_episode(write_episode(tmp_path, "episode", frames=12))


def test_preprocessing_preserves_native_rgb_and_padding(episode):
    rgb = np.zeros((180, 320, 3), dtype=np.uint8)
    rgb[:, 1] = 255
    rgb[1] = 64
    Image.fromarray(rgb).save(episode.image_paths[0])
    views, colors = inference.load_views(episode.image_paths[:1])
    image = views[0]["img"][0]
    assert image.shape == (3, 182, 322)
    np.testing.assert_array_equal(colors[0], rgb)
    np.testing.assert_allclose(image[:, 1:-1, 1:-1].permute(1, 2, 0), rgb / 127.5 - 1, atol=1e-7)
    torch.testing.assert_close(image[:, 1:-1, 0], image[:, 1:-1, 2])
    torch.testing.assert_close(image[:, 0], image[:, 2])
    torch.testing.assert_close(views[0]["true_shape"], torch.tensor([[182, 322]]))


def test_auto_start_explicit_start_and_missing_gt(episode):
    path = episode.path / "TCP" / episode.camera / "state.npy"
    state = np.load(path)
    state[:3, 1] = -100
    np.save(path, state)
    start, query, source = inference.resolve_initial_query(episode, None)
    assert start == 3 and source == "projected GT TCP"
    assert query.shape == (1, 2)
    with pytest.raises(ValueError, match="frame 0 GT.*outside"):
        inference.resolve_initial_query(episode, 0)
    path.unlink()
    start, query, source = inference.resolve_initial_query(episode, None, [20, 30])
    assert start == 0 and source == "manual pixel"
    with pytest.raises(FileNotFoundError, match="--interactive"):
        inference.resolve_initial_query(episode, None)


def test_invalid_episode_and_queries(episode):
    with pytest.raises(ValueError, match="available cameras"):
        inference.load_episode(episode.path, "unknown")
    for point in ([320, 1], [1, 180], [np.nan, 3], [[2, 3], [4, 5]]):
        with pytest.raises(ValueError):
            inference.validate_query(point)
    with pytest.raises(ValueError, match="z > 0"):
        inference.project_query([[0, 0, 0]], episode.intrinsics, "frame 2")
    with pytest.raises(ValueError, match="fewer than 2"):
        episode.clip(11)
    episode.image_paths[4].unlink()
    with pytest.raises(FileNotFoundError, match="missing frames"):
        inference.load_episode(episode.path)


def test_checkpoint_restores_stats_and_tied_weights(tmp_path, monkeypatch):
    import arc.models.arc.arc as arc_module
    from safetensors.torch import save_model

    def model_factory(**kwargs):
        model = SmallArc(**kwargs)
        model.tied_head = model.head
        return model
    monkeypatch.setattr(arc_module, "Arc", model_factory)
    model = model_factory(num_arms=1)
    model.set_tcp_position_stats(torch.tensor([[.2, -.3, 1.5]]), torch.tensor([[.12, .31, .5]]))
    save_model(model, str(tmp_path / "model.safetensors"))
    loaded = inference.load_model(tmp_path, torch.device("cpu"))
    torch.testing.assert_close(loaded.tcp_track_head.position_mean, model.tcp_track_head.position_mean)
    torch.testing.assert_close(loaded.tcp_track_head.position_std, model.tcp_track_head.position_std)
    assert not loaded.training
    state = dict(model.state_dict())
    state.pop("tcp_track_head.position_std")
    with pytest.raises(ValueError, match="position_std"):
        inference.restore_inference_weights(loaded, state)
    state = dict(model.state_dict())
    state.pop("cam_dec.fc_t.weight")
    with pytest.raises(ValueError, match="cam_dec.fc_t.weight"):
        inference.restore_inference_weights(loaded, state)
    save_model(model_factory(num_arms=2), str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="single-arm"):
        inference.load_model(tmp_path, torch.device("cpu"))


class FakeModel:
    def __init__(self):
        self.queries = []

    def __call__(self, views, **kwargs):
        self.queries.append(kwargs["tcp_query_points"].cpu().numpy())
        assert kwargs["force_no_output_conversion"] and kwargs["decode_tcp"]
        assert not kwargs["inference_track"] and not kwargs["decode_motion"]
        frames = len(views)
        # Predict different opening per window to verify the boundary policy.
        opening_logit = len(self.queries) * 0.25
        pose = torch.zeros(1, frames, 9)
        pose[..., 0] = 2  # camera origin in robot-base coordinates
        pose[..., 6] = 1
        pose[..., 7:] = 1
        return {"tcp_position": torch.tensor([0., 0., 1.]).expand(1, frames, 1, 3),
                "tcp_rotation": torch.eye(3).expand(1, frames, 1, 3, 3),
                "tcp_gripper_logit": torch.full((1, frames, 1), opening_logit),
                "tcp_confidence": torch.ones(1, frames, 1) * 2,
                "depth": torch.ones(1, frames, 182, 322),
                "depth_conf": torch.ones(1, frames, 182, 322), "pose_enc": pose}


def test_sliding_windows_feed_predictions_and_keep_previous_boundary(episode, tmp_path):
    model = FakeModel()
    paths = episode.clip(2)
    query = np.array([[17., 29.]], dtype=np.float32)
    pred = inference.infer_episode(model, episode, paths, query, "manual", device=torch.device("cpu"),
                                   dtype=torch.float32, window_size=4, keep_geometry=True, max_points=20)
    np.testing.assert_array_equal(model.queries[0], [[[18, 30]]])
    np.testing.assert_allclose(model.queries[1], [[[161, 91]]])  # previous prediction, not GT
    assert [w["frame_indices"] for w in pred.windows] == [list(range(2, 6)), list(range(5, 9)), list(range(8, 12))]
    assert pred.tcp["position"].shape == (10, 1, 3)
    assert pred.tcp["gripper"][3, 0] == pytest.approx(torch.sigmoid(torch.tensor(.25)).item())
    assert pred.source_windows[3] == [0, 1]
    assert len(pred.clouds) == 10 and len(pred.camera_to_base) == 10
    assert all(len(cloud["points"]) == 20 for cloud in pred.clouds)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    result = inference.build_result(episode, paths, pred, query, "manual", checkpoint, 4)
    encoded = json.loads(json.dumps(result, allow_nan=False))
    assert encoded["skipped_prefix_frames"] == [0, 1]
    assert encoded["frames"][0]["time_seconds"] == 2 / 15
    assert encoded["frames"][-1]["frame_index"] == 11
    assert isinstance(encoded["frames"][0]["tcp"]["gripper_open"], float)
    assert "left" not in encoded["frames"][0] and "right" not in encoded["frames"][0]
    np.testing.assert_array_equal(pred.tcp["position"][0, 0], [0, 0, 1])
    np.testing.assert_array_equal(inference.tcp_to_base(pred.tcp, pred.camera_to_base)["position"][0, 0], [2, 0, 1])



def test_invalid_boundary_does_not_overwrite_previous_result(episode, tmp_path):
    class InvalidBoundaryModel(FakeModel):
        def __call__(self, views, **kwargs):
            pred = super().__call__(views, **kwargs)
            pred["tcp_position"] = pred["tcp_position"].clone()
            pred["tcp_position"][0, -1, 0, 2] = -1
            return pred

    args = inference.parse_args(["--input", str(episode.path), "--window-size", "4"])
    args.output = tmp_path / "previous.json"
    args.output.write_text('{"previous": true}')
    with pytest.raises(ValueError, match="Camera cam_a, window 0 prediction at frame 3"):
        inference.run_and_save(args, episode, InvalidBoundaryModel(), 0, np.array([[30., 40.]]),
                               "manual", torch.device("cpu"), torch.float32)
    assert json.loads(args.output.read_text()) == {"previous": True}


def test_window_error_names_camera_and_frames(episode):
    Image.new("RGB", (320, 240)).save(episode.image_paths[1])
    with pytest.raises(RuntimeError, match="Camera cam_a, window 1/1, frames 0-2.*320x180"):
        inference.infer_episode(FakeModel(), episode, episode.image_paths[:3], np.array([[30., 40.]]),
                                 "manual", device=torch.device("cpu"), dtype=torch.float32)

def test_point_cloud_and_tcp_share_absolute_frame():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    c2w = np.concatenate([rotation, np.array([[1], [2], [3]])], axis=1)[None]
    tcp = {"position": np.array([[[1., 0, 2]]]), "rotation": np.eye(3)[None, None]}
    transformed = inference.tcp_to_base(tcp, c2w)
    np.testing.assert_allclose(transformed["position"], [[[1, 3, 5]]])
    np.testing.assert_allclose(transformed["rotation"][0, 0], rotation)
    k = np.array([[[100., 0, 161], [0, 100, 91], [0, 0, 1]]])
    depth = np.ones((1, 182, 322), np.float32) * 2
    color = np.zeros((180, 320, 3), np.uint8)
    clouds = inference.frame_clouds(depth, np.ones_like(depth), [color], c2w, k, 0, 0)
    # Original pixel (160,90) is padded pixel (161,91), along the optical axis.
    np.testing.assert_allclose(clouds[0]["points"][90 * 320 + 160], [1, 2, 5])


def test_gt_viewer_trajectory_uses_dataset_extrinsic(episode):
    from droid_script.stage1_viewer import load_gt_trajectory

    paths = episode.image_paths[2:6]
    observed = load_gt_trajectory(episode, paths)
    state = np.load(episode.path / "TCP" / episode.camera / "state.npy")
    world_to_camera = np.load(episode.path / "extrinsic" / f"{episode.camera}.npy")
    expected = (state[2:6, :3] - world_to_camera[:3, 3]) @ world_to_camera[:3, :3]
    np.testing.assert_allclose(observed, expected)


@pytest.mark.parametrize("kwargs", [{}, {"no_gt": True}])
def test_interactive_build_and_frame_change_reset(episode, tmp_path, kwargs):
    pytest.importorskip("gradio")
    pytest.importorskip("viser")
    from droid_script.stage1_viewer import build_interactive, selection_for_frame
    if kwargs.get("no_gt"):
        (episode.path / "TCP" / episode.camera / "state.npy").unlink()
        uv = valid = None
    else:
        uv, valid = inference.ground_truth_queries(episode)
    image, selection, available = selection_for_frame(episode, 2, uv, valid)
    assert image.shape == (180, 320, 3) and selection == []
    assert available == (not kwargs.get("no_gt", False))
    args = inference.parse_args(["--input", str(episode.path), "--interactive"])
    args.output = tmp_path / "result.json"
    demo, close = build_interactive(args, episode, torch.device("cpu"), torch.float32)
    try:
        config = demo.get_config_file()
        assert any(c.get("props", {}).get("label") == "点击一个 TCP" for c in config["components"])
        from types import SimpleNamespace
        callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values()}
        selected = callbacks["select"](episode.camera, 2, SimpleNamespace(index=(50, 60)))
        assert selected[1] == [[50, 60]]
        assert selected[3] == [episode.camera, 2]
        assert selected[5]["interactive"]
        cleared = callbacks["clear"](episode.camera, 3)
        assert cleared[1] == [] and cleared[3] is None and not cleared[5]["interactive"]
    finally:
        demo.close()
        close()


def test_cli_rejects_invalid_options():
    for options in (["--window-size", "19"], ["--max-frames", "1"],
                    ["--start-frame", "-1"], ["--interactive", "--port", "7860"],
                    ["--tcp-query-point", "2", "3", "--interactive"]):
        with pytest.raises(SystemExit):
            inference.parse_args(["--input", "unused", *options])


def test_camera_switch_updates_query_inference_and_output(episode, tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    pytest.importorskip("viser")
    from types import SimpleNamespace
    from droid_script import stage1_viewer

    state_path = episode.path / "TCP" / "cam_b" / "state.npy"
    state = np.load(state_path)
    state[:3, 1] = -100
    np.save(state_path, state)
    args = inference.parse_args(["--input", str(episode.path), "--interactive", "--max-frames", "4"])
    args.model = tmp_path / "model.pt"
    args.model.touch()
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(inference, "DEFAULT_OUTPUT_ROOT", output_root)
    loads, viewers = [], []

    def load(*args):
        loads.append(True)
        return FakeModel()

    def viewer(args, selected, prediction, paths):
        item = SimpleNamespace(camera=selected.camera, stopped=False,
                               server=SimpleNamespace(get_port=lambda: 8020))
        item.stop = lambda: setattr(item, "stopped", True)
        viewers.append(item)
        return item

    monkeypatch.setattr(inference, "load_model", load)
    monkeypatch.setattr(stage1_viewer, "start_viewer", viewer)
    demo, close = stage1_viewer.build_interactive(args, episode, torch.device("cpu"), torch.float32)
    try:
        callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values()}
        dropdown = next(c["props"] for c in demo.get_config_file()["components"] if c.get("props", {}).get("label") == "相机")
        assert dropdown["value"] == "cam_a"
        assert [item[0] for item in dropdown["choices"]] == ["cam_a", "cam_b"]
        changed = callbacks["switch_camera"]("cam_b")
        assert changed[0]["value"] == 3
        assert changed[2] == [] and changed[4] is None and not changed[6]["interactive"]
        assert changed[7]["interactive"]
        assert changed[9] is None and changed[10] is None
        np.testing.assert_array_equal(changed[1], np.asarray(Image.open(episode.path / "images/cam_b/000003.png")))
        gt = callbacks["use_gt"]("cam_b", 3)
        selected_episode = inference.load_episode(episode.path, "cam_b")
        expected_uv, _ = inference.ground_truth_queries(selected_episode)
        np.testing.assert_allclose(gt[1], expected_uv[3:4])
        with pytest.raises(gr.Error, match="重新选择 TCP"):
            callbacks["run"]("cam_b", 3, gt[1], gt[2], ["cam_a", 3], None)
        assert not loads
        result = callbacks["run"]("cam_b", 3, gt[1], gt[2], gt[3], None, progress=lambda *a, **k: None)
        assert result[0]["camera_id"] == "cam_b" and result[0]["start_frame"] == 3
        assert result[0]["frames"][0]["image"].endswith("images/cam_b/000003.png")
        path_b = output_root / episode.path.name / "cam_b/tcp_episode.json"
        assert json.loads(path_b.read_text())["camera_id"] == "cam_b"
        assert viewers[-1].camera == "cam_b"
        callbacks["switch_camera"]("cam_a")
        assert viewers[0].stopped
        gt = callbacks["use_gt"]("cam_a", 0)
        callbacks["run"]("cam_a", 0, gt[1], gt[2], gt[3], None, progress=lambda *a, **k: None)
        assert (output_root / episode.path.name / "cam_a/tcp_episode.json").is_file()
        assert json.loads(path_b.read_text())["camera_id"] == "cam_b"
        assert len(loads) == 1 and args.output is None
        # An explicit output path remains an intentional override for either camera.
        args.output = tmp_path / "explicit.json"
        callbacks["run"]("cam_a", 0, gt[1], gt[2], gt[3], None, progress=lambda *a, **k: None)
        assert json.loads(args.output.read_text())["camera_id"] == "cam_a"
    finally:
        demo.close()
        close()
    assert all(item.stopped for item in viewers)


def test_switch_to_camera_without_gt_keeps_manual_selection(episode):
    pytest.importorskip("gradio")
    pytest.importorskip("viser")
    from droid_script.stage1_viewer import build_interactive
    (episode.path / "TCP/cam_b/state.npy").unlink()
    args = inference.parse_args(["--input", str(episode.path), "--interactive"])
    demo, close = build_interactive(args, episode, torch.device("cpu"), torch.float32)
    try:
        callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values()}
        changed = callbacks["switch_camera"]("cam_b")
        assert changed[0]["value"] == 0
        assert not changed[7]["interactive"]
        assert "GT 选点不可用" in changed[8]
        assert changed[2] == []
    finally:
        demo.close()
        close()
