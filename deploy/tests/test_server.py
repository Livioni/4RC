import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from deploy.protocol import MAX_MESSAGE_BYTES, pack_message, unpack_message
from deploy.server import (
    InvalidRequest, PredictionFailed, Stage2Predictor, Stage2Server, prepare_inputs, resolve_checkpoint,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(frames=4):
    rng = np.random.default_rng(17)
    return {
        "type": "predict", "request_id": "observation-1",
        "images": rng.integers(0, 256, (frames, 240, 320, 3), dtype=np.uint8),
        "wrist_images": rng.integers(0, 256, (2, 240, 320, 3), dtype=np.uint8),
        "tcp_query_points": np.array([[10, 20], [300, 200]], dtype=np.float32),
        "intrinsics": np.array([[100, 0, 160], [0, 100, 120], [0, 0, 1]], dtype=np.float32),
        "current_robot_state": np.linspace(0, 1, 14, dtype=np.float32),
        "instruction": "Place the pen in the cup.",
    }


class FakePolicy(torch.nn.Module):
    """Implements the real reconstruction/condition/sampling interface without weights."""

    def __init__(self):
        super().__init__()
        self.failure = None
        self.invalid_history = False

    def reconstruct(self, images, initial_query_points):
        assert not self.training and not torch.is_grad_enabled()
        assert not self.decode_geometry and not self.decode_camera
        self.observed_images = images.clone()
        self.observed_queries = initial_query_points.clone()
        batch, frames = images.shape[:2]
        position = torch.zeros(batch, frames, 2, 3)
        position[..., 0] = torch.arange(frames)[None, :, None] * 0.01
        position[..., 2] = 1
        if self.invalid_history:
            position[:, -1, 0, 0] = 10  # Projects outside the image.
            position[:, -1, 1, 2] = float("nan")
        return {
            "tcp_position": position,
            "tcp_rotation": torch.eye(3).expand(batch, frames, 2, 3, 3).clone(),
            "tcp_confidence": torch.ones(batch, frames, 2),
        }, object()

    def make_condition(self, images, intrinsics, instructions, reconstruction, features, *,
                       current_robot_state, wrist_images):
        self.observed_intrinsics = intrinsics.clone()
        self.observed_state = current_robot_state.clone()
        self.observed_wrists = wrist_images.clone()
        self.observed_instruction = instructions
        return SimpleNamespace(history_valid=torch.ones(1, images.shape[1], 2, dtype=torch.bool))

    def sample_condition(self, condition, *, steps, generator):
        if self.failure == "exception":
            raise RuntimeError("test model error")
        position = torch.rand(1, 32, 2, 3, generator=generator)
        if self.failure == "nan":
            position[0, 0, 0, 0] = float("nan")
        return {
            "success": torch.tensor([self.failure != "unsuccessful"]),
            "action_position": position,
            "action_rotation": torch.eye(3).expand(1, 32, 2, 3, 3),
            "action_gripper": torch.tensor([0, 1]).expand(1, 32, 2),
            "action_gripper_score": torch.tensor([-0.5, 0.5]).expand(1, 32, 2),
            "future_step_indices": torch.arange(1, 33)[None],
        }


def make_predictor(policy=None):
    return Stage2Predictor(
        policy if policy is not None else FakePolicy(),
        {"history_frames": 4, "prediction_horizon": 32, "sampling_steps": 4},
        device=torch.device("cpu"), dtype=torch.float32,
    )


@pytest.mark.parametrize("frames,indices", [
    (1, [0, 0, 0, 0]), (2, [0, 0, 0, 1]), (3, [0, 0, 1, 2]), (4, [0, 1, 2, 3]),
])
def test_preprocessing_matches_training_and_offline_inference(tmp_path, frames, indices):
    from stage2_sliding_window_inference import prepare_images

    request = observation(frames)
    original_k = request["intrinsics"].copy()
    original_query = request["tcp_query_points"].copy()
    paths = []
    for index, pixels in enumerate(request["images"]):
        path = tmp_path / f"{index}.png"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    reference = prepare_images([paths[i] for i in indices], original_k, original_query)
    result = prepare_inputs(request, 4)
    assert result.images.shape == (1, 4, 3, 252, 322)
    for actual, expected in zip((result.images, result.intrinsics, result.query_points), reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert result.history_padding == 4 - frames
    np.testing.assert_array_equal(request["intrinsics"], original_k)
    np.testing.assert_array_equal(request["tcp_query_points"], original_query)
    expected_wrists = torch.from_numpy(request["wrist_images"]).permute(0, 3, 1, 2).float() / 255
    torch.testing.assert_close(result.wrist_images[0], expected_wrists, rtol=0, atol=0)
    torch.testing.assert_close(result.current_robot_state[0], torch.from_numpy(request["current_robot_state"]))


@pytest.mark.parametrize("key,value", [
    ("images", np.zeros((4, 480, 640, 3), dtype=np.uint8)),
    ("images", np.zeros((4, 240, 320, 3), dtype=np.float32)),
    ("images", np.zeros((0, 240, 320, 3), dtype=np.uint8)),
    ("images", np.zeros((5, 240, 320, 3), dtype=np.uint8)),
    ("wrist_images", np.zeros((1, 240, 320, 3), dtype=np.uint8)),
    ("wrist_images", np.zeros((2, 192, 256, 3), dtype=np.uint8)),
    ("tcp_query_points", [[320, 10], [10, 10]]),
    ("tcp_query_points", [[10, 240], [10, 10]]),
    ("tcp_query_points", [[-0.1, 10], [10, 10]]),
    ("tcp_query_points", [[float("nan"), 10], [10, 10]]),
    ("intrinsics", np.zeros((3, 3))),
    ("current_robot_state", np.zeros(13)),
    ("current_robot_state", [0] * 13 + [1.1]),
    ("current_robot_state", [0] * 6 + [-0.001] + [0] * 7),
    ("current_robot_state", [float("inf")] * 14),
    ("instruction", "   "),
])
def test_invalid_observations_are_rejected(key, value):
    request = observation()
    request[key] = value
    with pytest.raises(InvalidRequest):
        prepare_inputs(request, 4)


@pytest.mark.parametrize("key", ["images", "wrist_images", "current_robot_state", "instruction", "intrinsics", "tcp_query_points"])
def test_missing_inputs_are_rejected(key):
    request = observation()
    del request[key]
    with pytest.raises(InvalidRequest):
        prepare_inputs(request, 4)


def test_predict_returns_native_actions_and_unpadded_history():
    predictor = make_predictor()
    request = observation(frames=2)
    result = predictor.predict(request)
    assert result["success"] is True and result["request_id"] == request["request_id"]
    assert result["action_position"].shape == (32, 2, 3)
    assert result["action_rotation"].shape == (32, 2, 3, 3)
    assert result["action_gripper"].dtype == np.int64
    np.testing.assert_array_equal(result["action_gripper"], np.tile([0, 1], (32, 1)))
    np.testing.assert_array_equal(result["future_step_indices"], np.arange(1, 33))
    assert result["history_position"].shape == (2, 2, 3)
    assert result["history_padding"] == 2
    np.testing.assert_allclose(result["history_query_points_px"][:, 0], [[162, 120], [163, 120]])
    assert result["history_valid"].all()
    assert result["inference_ms"] > 0
    np.testing.assert_array_equal(result["action_position"], predictor.predict(request)["action_position"])


def test_invalid_history_has_a_mask_and_no_clipped_queries():
    policy = FakePolicy()
    policy.invalid_history = True
    result = make_predictor(policy).predict(observation())
    assert not result["history_valid"][-1].any()
    np.testing.assert_array_equal(result["history_query_points_px"][-1], 0)
    np.testing.assert_array_equal(result["history_position"][-1], 0)
    assert np.isfinite(result["history_position"]).all()


@pytest.mark.parametrize("failure", ["nan", "unsuccessful"])
def test_model_failures_do_not_return_action_chunks(failure):
    policy = FakePolicy()
    policy.failure = failure
    with pytest.raises(PredictionFailed):
        make_predictor(policy).predict(observation())


def test_native_wrist_encoder_performs_training_resize_and_padding():
    from arc.models.arc.wrist_encoder import FrozenDINOv2Encoder

    class Encoder(torch.nn.Module):
        config = SimpleNamespace(patch_size=14, hidden_size=8)

        def forward(self, *, pixel_values):
            self.pixels = pixel_values
            return SimpleNamespace(last_hidden_state=torch.zeros(len(pixel_values), 267, 8))

    encoder = Encoder()
    wrist_encoder = FrozenDINOv2Encoder(encoder=encoder)
    prepared = prepare_inputs(observation(), 4)
    features = wrist_encoder(prepared.wrist_images)
    expected = F.interpolate(prepared.wrist_images.flatten(0, 1), size=(192, 256),
                             mode="bicubic", align_corners=False, antialias=True)
    expected = F.pad(expected, (5, 5, 2, 2), mode="reflect")
    expected = (expected - wrist_encoder.image_mean) / wrist_encoder.image_std
    assert features.shape == (1, 2, 266, 8)
    torch.testing.assert_close(encoder.pixels, expected, rtol=0, atol=0)


def test_checkpoint_parent_and_direct_paths(tmp_path):
    direct = tmp_path / "final_checkpoint"
    direct.mkdir()
    for name in ("config.json", "model.safetensors", "action_norm.json", "state_norm.json"):
        (direct / name).touch()
    assert resolve_checkpoint(tmp_path) == direct
    assert resolve_checkpoint(direct) == direct
    (direct / "state_norm.json").unlink()
    with pytest.raises(ValueError, match="state_norm.json"):
        resolve_checkpoint(tmp_path)


async def _connect(running):
    port = running.sockets[0].getsockname()[1]
    return await connect(f"ws://127.0.0.1:{port}", compression=None, max_size=MAX_MESSAGE_BYTES, proxy=None)


def test_websocket_roundtrip_errors_and_recovery():
    async def scenario():
        policy = FakePolicy()
        service = Stage2Server(make_predictor(policy))
        async with serve(service.handle, "127.0.0.1", 0, compression=None, max_size=MAX_MESSAGE_BYTES) as running:
            async with await _connect(running) as client:
                metadata = unpack_message(await client.recv())
                assert metadata["type"] == "metadata" and metadata["protocol_version"] == 1
                assert metadata["history_frames"] == 4 and metadata["prediction_horizon"] == 32
                assert metadata["image_size_wh"] == [320, 240]
                for payload in ["text", b"\xc1", pack_message({"type": "unknown"}),
                                pack_message({"type": np.array([1, 2])}),
                                pack_message({"type": "predict", "request_id": True})]:
                    await client.send(payload)
                    error = unpack_message(await client.recv())
                    assert error["type"] == "error" and not error["success"]
                    assert error["error"]["code"] == "invalid_request"
                    assert "action_position" not in error
                request = observation()
                payload = pack_message(request)
                assert len(payload) > 1024 * 1024  # Exceeds websockets' default 1 MiB limit.
                for failure, code in [("unsuccessful", "prediction_failed"), ("nan", "prediction_failed"),
                                      ("exception", "inference_error"), (None, None)]:
                    policy.failure = failure
                    await client.send(payload)
                    response = unpack_message(await client.recv())
                    assert response["request_id"] == request["request_id"]
                    if code:
                        assert response["error"]["code"] == code
                        assert not response["success"] and "action_position" not in response
                    else:
                        assert response["success"] and response["action_position"].shape == (32, 2, 3)
    asyncio.run(scenario())


def test_multiple_connections_serialize_inference_and_keep_health_responsive():
    class BlockingPredictor:
        def __init__(self):
            self.inner = make_predictor()
            self.started, self.release = threading.Event(), threading.Event()
            self.active = self.maximum_active = 0

        def metadata(self):
            return self.inner.metadata()

        def predict(self, request):
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            try:
                self.started.set()
                if not self.release.wait(timeout=10):
                    raise RuntimeError("test worker timed out")
                time.sleep(0.02)
                return self.inner.predict(request)
            finally:
                self.active -= 1

    async def scenario():
        predictor = BlockingPredictor()
        service = Stage2Server(predictor)
        async with serve(service.handle, "127.0.0.1", 0, compression=None, max_size=MAX_MESSAGE_BYTES) as running:
            async with await _connect(running) as first, await _connect(running) as second:
                await first.recv()
                await second.recv()
                try:
                    await first.send(pack_message(observation()))
                    assert await asyncio.to_thread(predictor.started.wait, 5)
                    await second.send(pack_message({"type": "health", "request_id": "ping"}))
                    health = unpack_message(await asyncio.wait_for(second.recv(), timeout=2))
                    assert health["ready"] and health["request_id"] == "ping"
                    await second.send(pack_message(observation(frames=1)))
                finally:
                    predictor.release.set()
                assert unpack_message(await first.recv())["success"]
                assert unpack_message(await second.recv())["success"]
                assert predictor.maximum_active == 1
    asyncio.run(scenario())
