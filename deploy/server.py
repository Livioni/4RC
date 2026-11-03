#!/usr/bin/env python3
"""Stateless Stage2 WebSocket inference. See deploy/README_CN.md for the contract."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from arc.action import project_tcp
from arc.state_normalization import validate_robot_states
from deploy.protocol import (
    MAX_MESSAGE_BYTES, PROTOCOL_VERSION, ProtocolError, pack_message, unpack_message,
)

LOGGER = logging.getLogger("stage2.server")
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/TCP4D-task1-stage2"
WIDTH, HEIGHT = 320, 240
PADDING = (1, 1, 6, 6)  # left, right, top, bottom, as in PenCups4RC.


class InvalidRequest(ValueError):
    """An observation does not match the training/deployment contract."""


class PredictionFailed(RuntimeError):
    """The policy did not produce a usable finite action chunk."""


@dataclass
class PreparedInputs:
    images: torch.Tensor
    intrinsics: torch.Tensor
    query_points: torch.Tensor
    wrist_images: torch.Tensor
    current_robot_state: torch.Tensor
    instruction: str
    history_padding: int


def resolve_checkpoint(path: Path) -> Path:
    """Accept either a complete checkpoint or its parent experiment directory."""
    path = path.expanduser().resolve()
    if not (path / "model.safetensors").is_file():
        path = path / "final_checkpoint"
    required = ("config.json", "model.safetensors", "action_norm.json", "state_norm.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete checkpoint {path}: missing {', '.join(missing)}")
    return path


def _float_array(request: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        value = np.asarray(request[key])
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidRequest(f"{key} must be a finite numeric array with shape {list(shape)}") from exc
    return value


def _rgb(request: dict, key: str) -> np.ndarray:
    value = request.get(key)
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise InvalidRequest(f"{key} must be an RGB uint8 ndarray, not floats, BGR or encoded image bytes")
    if value.ndim != 4 or value.shape[1:] != (HEIGHT, WIDTH, 3):
        raise InvalidRequest(f"{key} must have shape [N,240,320,3]; automatic resizing is disabled")
    return value


def prepare_inputs(request: dict, history_frames: int) -> PreparedInputs:
    """Convert unpadded RGB/pixels to exactly the tensors used during training."""
    rgb = _rgb(request, "images")
    frames = len(rgb)
    if not 1 <= frames <= history_frames:
        raise InvalidRequest(f"images must contain 1 to {history_frames} frames, oldest first")
    wrist = _rgb(request, "wrist_images")
    if len(wrist) != 2:
        raise InvalidRequest("wrist_images must contain the current left and right wrist images")
    query = _float_array(request, "tcp_query_points", (2, 2))
    if np.any(query < 0) or np.any(query[:, 0] >= WIDTH) or np.any(query[:, 1] >= HEIGHT):
        raise InvalidRequest("tcp_query_points must be inside the first unpadded 320x240 RGB frame")
    intrinsics = _float_array(request, "intrinsics", (3, 3))
    if (
        intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1], rtol=0, atol=1e-6)
        or abs(np.linalg.det(intrinsics.astype(np.float64))) < 1e-8
    ):
        raise InvalidRequest("intrinsics must be nonsingular OpenCV K with fx,fy > 0 and last row [0,0,1]")
    state = _float_array(request, "current_robot_state", (14,))
    try:
        validate_robot_states(state[None])
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    instruction = request.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise InvalidRequest("instruction must be a non-empty task instruction")

    padding = history_frames - frames
    indices = np.maximum(np.arange(history_frames) - padding, 0)
    pixels = torch.from_numpy(np.ascontiguousarray(rgb[indices])).permute(0, 3, 1, 2).float().div_(255)
    images = F.pad(pixels, PADDING, mode="reflect").mul_(2).sub_(1).unsqueeze(0)
    intrinsics[0, 2] += PADDING[0]
    intrinsics[1, 2] += PADDING[2]
    query += np.array([PADDING[0], PADDING[2]], dtype=np.float32)
    wrists = torch.from_numpy(np.ascontiguousarray(wrist)).permute(0, 3, 1, 2).float().div_(255)
    return PreparedInputs(
        images=images,
        intrinsics=torch.from_numpy(intrinsics).expand(1, history_frames, 3, 3).clone(),
        query_points=torch.from_numpy(query).unsqueeze(0),
        wrist_images=wrists.unsqueeze(0), current_robot_state=torch.from_numpy(state).unsqueeze(0),
        instruction=instruction.strip(), history_padding=padding,
    )


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


class Stage2Predictor:
    """One policy, with no per-client history or cross-request random state."""

    def __init__(self, policy, config: dict, *, device: torch.device, dtype: torch.dtype,
                 steps: int | None = None, seed: int = 42):
        self.history_frames = config["history_frames"]
        self.prediction_horizon = config["prediction_horizon"]
        self.steps = config.get("sampling_steps", 4) if steps is None else steps
        if any(type(value) is not int or value < 1 for value in
               (self.history_frames, self.prediction_horizon, self.steps)):
            raise ValueError("History, horizon and sampling steps must be positive integers")
        if not 0 <= seed < 2**63:
            raise ValueError("seed must be in [0,2**63)")
        self.policy = policy.eval().requires_grad_(False)
        self.policy.decode_camera = False
        self.policy.decode_geometry = False
        self.config, self.device, self.dtype, self.seed = config, device, dtype, seed

    def metadata(self) -> dict:
        return {
            "type": "metadata", "protocol_version": PROTOCOL_VERSION,
            "model": "4RC-stage2", "history_frames": self.history_frames,
            "prediction_horizon": self.prediction_horizon,
            "image_size_wh": [WIDTH, HEIGHT], "image_format": "RGB uint8 HWC",
            "padding_lrtb": list(PADDING), "padded_image_size_wh": [322, 252],
            "wrist_image_size_wh": [WIDTH, HEIGHT], "wrist_encoder_image_size_wh": [266, 196],
            "arm_order": ["left", "right"], "query_frame": "first_input_frame",
            "query_coordinates": "unpadded pixel (u,v)",
            "action_coordinate_frame": "last_observation_camera",
            "history_coordinate_frame": "each_observation_camera",
            "camera_axes": "OpenCV: x right, y down, z forward",
            "position_unit": "m", "rotation_format": "3x3 matrix",
            "gripper_convention": "1=open, 0=closed; score>=0 means open; score is not a probability",
            "state_layout": [f"{side}.{name}" for side in ("left", "right")
                             for name in (*[f"joint_{i}" for i in range(6)], "gripper")],
            "state_joint_unit": "rad", "state_gripper_range": [0, 1],
            "training_frame_rate_hz": self.config.get("frame_rate", 30.0),
            "training_frame_stride": self.config.get("frame_stride", 1),
            "sampling_steps": self.steps, "seed": self.seed,
            "device": str(self.device), "dtype": str(self.dtype).removeprefix("torch."),
            "max_message_bytes": MAX_MESSAGE_BYTES,
        }

    @torch.inference_mode()
    def predict(self, request: dict) -> dict:
        started = time.perf_counter()
        prepared = prepare_inputs(request, self.history_frames)
        images, intrinsics, query, state, wrists = [value.to(self.device) for value in (
            prepared.images, prepared.intrinsics, prepared.query_points,
            prepared.current_robot_state, prepared.wrist_images,
        )]
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        context = (torch.autocast(device_type=self.device.type, dtype=self.dtype)
                   if self.dtype != torch.float32 else contextlib.nullcontext())
        with context:
            reconstruction, features = self.policy.reconstruct(images, query)
            condition = self.policy.make_condition(
                images, intrinsics, [prepared.instruction], reconstruction, features,
                current_robot_state=state, wrist_images=wrists,
            )
            actions = self.policy.sample_condition(condition, steps=self.steps, generator=generator)
        if not bool(actions["success"][0]):
            raise PredictionFailed("Policy reported an unsuccessful action prediction")
        horizon = self.prediction_horizon
        output = {}
        for key, shape in (
            ("action_position", (horizon, 2, 3)), ("action_rotation", (horizon, 2, 3, 3)),
            ("action_gripper", (horizon, 2)), ("action_gripper_score", (horizon, 2)),
            ("future_step_indices", (horizon,)),
        ):
            tensor = actions[key][0].detach()
            value = tensor.cpu().numpy() if key in ("action_gripper", "future_step_indices") else _numpy(tensor)
            if value.shape != shape or not np.isfinite(value).all():
                raise PredictionFailed(f"Policy returned invalid {key}; no action chunk is available")
            output[key] = value
        if not np.isin(output["action_gripper"], [0, 1]).all():
            raise PredictionFailed("Policy returned an invalid gripper state")

        projected, visible = project_tcp(
            reconstruction["tcp_position"], intrinsics,
            image_height=images.shape[-2], image_width=images.shape[-1], padding=PADDING,
        )
        position = _numpy(reconstruction["tcp_position"][0])
        rotation = _numpy(reconstruction["tcp_rotation"][0])
        confidence = _numpy(reconstruction["tcp_confidence"][0])
        valid = (visible[0] & condition.history_valid[0]).cpu().numpy()
        valid &= (np.isfinite(position).all(-1) & np.isfinite(rotation).all(axis=(-1, -2))
                  & np.isfinite(confidence))
        uv = _numpy(projected[0]) - np.array([PADDING[0], PADDING[2]], dtype=np.float32)
        real_frames = slice(prepared.history_padding, None)
        # Invalid history is zero-filled and explicitly masked, never clipped to a border.
        output.update({
            "history_position": np.where(valid[..., None], position, 0)[real_frames],
            "history_rotation": np.where(valid[..., None, None], rotation, 0)[real_frames],
            "history_confidence": np.where(valid, confidence, 0)[real_frames],
            "history_query_points_px": np.where(valid[..., None], uv, 0)[real_frames],
            "history_valid": valid[real_frames], "history_padding": prepared.history_padding,
        })
        return {
            "type": "prediction", "request_id": request["request_id"], "success": True,
            "inference_ms": (time.perf_counter() - started) * 1000, **output,
        }


class Stage2Server:
    def __init__(self, predictor: Stage2Predictor):
        self.predictor = predictor
        self._inference_lock = asyncio.Lock()

    async def _predict(self, request: dict) -> dict:
        async with self._inference_lock:
            work = asyncio.create_task(asyncio.to_thread(self.predictor.predict, request))
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                # Cancelling an await cannot stop a CUDA/worker thread. Keep ownership
                # until it finishes, so a disconnect/shutdown cannot overlap policies.
                with contextlib.suppress(Exception):
                    await work
                raise

    async def handle(self, connection) -> None:
        try:
            await connection.send(pack_message(self.predictor.metadata()))
            async for payload in connection:
                request_id = None
                try:
                    request = unpack_message(payload)
                    candidate = request.get("request_id")
                    if type(candidate) in (str, int):
                        request_id = candidate
                    request_type = request.get("type")
                    if not isinstance(request_type, str):
                        raise InvalidRequest("type must be 'predict' or 'health'")
                    if request_type == "health":
                        response = {"type": "health", "request_id": request_id, "success": True, "ready": True}
                    elif request_type == "predict":
                        if request_id is None:
                            raise InvalidRequest("predict requires a string or integer request_id")
                        response = await self._predict(request)
                        LOGGER.info("request_id=%s inference_ms=%.1f", request_id, response["inference_ms"])
                    else:
                        raise InvalidRequest("Unknown request type; expected 'predict' or 'health'")
                except (ProtocolError, InvalidRequest) as exc:
                    response = self._error(request_id, "invalid_request", str(exc))
                except PredictionFailed as exc:
                    LOGGER.warning("request_id=%s prediction failed: %s", request_id, exc)
                    response = self._error(request_id, "prediction_failed", str(exc))
                except Exception:
                    LOGGER.exception("request_id=%s inference failed", request_id)
                    response = self._error(request_id, "inference_error", "Inference failed; see server logs")
                await connection.send(pack_message(response))
        except ConnectionClosed:
            LOGGER.info("Client disconnected")

    @staticmethod
    def _error(request_id, code: str, message: str) -> dict:
        return {"type": "error", "request_id": request_id, "success": False,
                "error": {"code": code, "message": message}}


async def run_server(predictor: Stage2Predictor, host: str, port: int) -> None:
    service = Stage2Server(predictor)
    async with serve(service.handle, host, port, max_size=MAX_MESSAGE_BYTES, max_queue=1, compression=None):
        LOGGER.info("Stage2 ready at ws://%s:%s (history=%s, horizon=%s)",
                    host, port, predictor.history_frames, predictor.prediction_horizon)
        await asyncio.Future()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Experiment directory or complete final_checkpoint directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="auto/cuda/cuda:0/cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Default: checkpoint sampling_steps")
    parser.add_argument("--seed", type=int, default=42, help="Reset the sampling generator to this seed for every request")
    parser.add_argument("--t5-model", help="Optional local T5-base directory; default: checkpoint configuration")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.sampling_steps is not None and args.sampling_steps < 1:
        parser.error("--sampling-steps must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0,2**63)")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Lazy import: --help, the codec and preprocessing tests never construct a model.
    from stage2_sliding_window_inference import load_stage2_policy, resolve_device_dtype

    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        device, dtype = resolve_device_dtype(args.device, args.dtype)
        LOGGER.info("Loading %s on %s with %s autocast", checkpoint, device, dtype)
        policy, config = load_stage2_policy(checkpoint, device, t5_model=args.t5_model)
        predictor = Stage2Predictor(policy, config, device=device, dtype=dtype, steps=args.sampling_steps, seed=args.seed)
        asyncio.run(run_server(predictor, args.host, args.port))
    except KeyboardInterrupt:
        LOGGER.info("Stage2 stopped")
    except Exception:
        LOGGER.exception("Stage2 server failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
