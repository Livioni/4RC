"""V2: real joint state and unpooled TCP prefix with full image/text context."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from arc.action import project_tcp, safe_rotation_6d_to_matrix, sinusoidal
from arc.action_normalization import componentwise_normalize, componentwise_unnormalize
from arc.models.arc.wrist_encoder import FrozenDINOv2Encoder

ACTION_HEAD_TYPE = "groot_tcp_v1"
ARCHITECTURE_KEYS = (
    "action_head_type", "action_dim", "action_depth", "action_heads", "condition_dim",
    "action_output_dim", "action_dropout", "global_token_grid", "prediction_horizon",
    "text_max_length", "tcp_query_window_size", "state_normalization_version",
)


def require_action_head_config(config):
    if config.get("action_head_type") != ACTION_HEAD_TYPE:
        raise ValueError(f"Only {ACTION_HEAD_TYPE} Stage2 checkpoints are supported; initialize from Stage1")
    missing = [key for key in ARCHITECTURE_KEYS if key not in config]
    if missing:
        raise ValueError(f"Incomplete V2 architecture metadata: {missing}")
    if config["state_normalization_version"] != 1:
        raise ValueError("V2 requires state_normalization_version=1")


def masked_values(values, valid):
    """Clear padding before any linear projection, including NaN/Inf padding."""
    values = values.masked_fill(~valid[..., None], 0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Non-finite values at valid conditioning/action positions")
    return values


class FrozenT5Encoder(nn.Module):
    output_dim = 768

    def __init__(self, model_name: str = "google-t5/t5-base", max_length: int = 128):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Keep only the encoder stack: do not register tied embeddings twice.
        self.encoder = T5EncoderModel.from_pretrained(model_name).encoder
        self.output_dim = self.encoder.config.d_model
        self.max_length = max_length
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not instructions or any(not text.strip() for text in instructions):
            raise ValueError("Every action sample requires a non-empty instruction")
        encoded = self.tokenizer(
            instructions, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        ).to(next(self.encoder.parameters()).device)
        output = self.encoder(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
        return output.last_hidden_state, encoded["attention_mask"].bool()


class ScalarTimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.frequency_dim = 256
        self.mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal(value, self.frequency_dim))


class TCPHistoryEncoder(nn.Module):
    """Keep frame/arm/row-major local tokens; sampler supplies arm/offset identity."""

    def __init__(self, input_dim, dim):
        super().__init__()
        self.dim = dim
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.xy_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.frame_encoder = ScalarTimeEncoder(dim)
        self.valid_embedding = nn.Embedding(2, dim)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, patches, centres, valid, *, image_height, image_width):
        batch, frames, arms, local, _ = patches.shape
        token_valid = valid[..., None].expand(-1, -1, -1, local)
        patches = masked_values(patches, token_valid)
        xy = masked_values(centres.float(), valid)
        spatial = self.xy_projection(torch.cat((
            sinusoidal(xy[..., 0] / image_width * (2 * math.pi), self.dim // 2),
            sinusoidal(xy[..., 1] / image_height * (2 * math.pi), self.dim // 2),
        ), -1))
        relative = torch.arange(1 - frames, 1, device=patches.device, dtype=torch.float32)
        encoded = self.projection(patches) + spatial[..., None, :]
        encoded = encoded + self.frame_encoder(relative)[None, :, None, None]
        encoded = encoded + self.valid_embedding(valid.long())[..., None, :]
        encoded = self.output_norm(encoded).masked_fill(~token_valid[..., None], 0)
        return encoded.reshape(batch, frames * arms * local, self.dim), token_valid.flatten(1)


class GlobalVisualEncoder(nn.Module):
    """Spatial resize of the action-only grid, then project and encode grid centres."""

    def __init__(self, input_dim, dim, patch_size=14, target_grid=None):
        super().__init__()
        self.dim, self.patch_size = dim, patch_size
        self.target_grid = tuple(target_grid) if target_grid is not None else None
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.xy_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, patches, *, image_height, image_width):
        gh, gw = image_height // self.patch_size, image_width // self.patch_size
        if patches.shape[1] != gh * gw:
            raise ValueError("Patch count must match the original image grid")
        if self.target_grid is not None:
            values = patches.transpose(1, 2).reshape(patches.shape[0], -1, gh, gw)
            # Antialiased interpolation also works under bf16 autocast via FP32.
            patches = F.interpolate(values.float(), size=self.target_grid, mode="bilinear",
                                    align_corners=False, antialias=True).flatten(2).transpose(1, 2).to(patches)
            gh, gw = self.target_grid
        y, x = torch.meshgrid(
            torch.arange(gh, device=patches.device, dtype=torch.float32),
            torch.arange(gw, device=patches.device, dtype=torch.float32), indexing="ij",
        )
        # New-grid cell centres expressed in normalized original padded-image coordinates.
        spatial = self.xy_projection(torch.cat((
            sinusoidal((x.flatten() + 0.5) / gw * (2 * math.pi), self.dim // 2),
            sinusoidal((y.flatten() + 0.5) / gh * (2 * math.pi), self.dim // 2),
        ), -1))
        return self.projection(patches) + spatial[None]


class ActionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.projection = nn.Linear(20, dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.position = nn.Embedding(1024, dim)

    def forward(self, actions, time):
        time = sinusoidal(time, self.dim)[:, None].expand(-1, actions.shape[1], -1)
        tokens = self.mlp(torch.cat((self.projection(actions), time), -1))
        return tokens + self.position.weight[None, :actions.shape[1]]


class ActionDiTBlock(nn.Module):
    """One attention per block: cross/self alternate, with ordinary flow-time AdaLN."""

    def __init__(self, dim, heads, condition_dim, dropout, *, cross):
        super().__init__()
        self.cross, self.heads, self.head_dim = cross, heads, dim // heads
        self.attention_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(condition_dim if cross else dim, dim)
        self.v = nn.Linear(condition_dim if cross else dim, dim)
        self.attention_output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.dropout = dropout

    def split_heads(self, value):
        return value.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def context_kv(self, context):
        return self.split_heads(self.k(context)), self.split_heads(self.v(context))

    def forward(self, x, time, valid, context, context_valid, cached_kv=None):
        shift, scale = self.modulation(time).chunk(2, -1)
        normalized = self.attention_norm(x) * (1 + scale[:, None]) + shift[:, None]
        source, source_valid = (context, context_valid) if self.cross else (normalized, valid)
        k, v = self.context_kv(source) if cached_kv is None else cached_kv
        update = F.scaled_dot_product_attention(
            self.split_heads(self.q(normalized)), k, v,
            attn_mask=source_valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).flatten(2)
        x = x + self.attention_output(update)
        x = x + self.ffn(self.ffn_norm(x))
        return x.masked_fill(~valid[..., None], 0)


class TCPActionDiT(nn.Module):
    def __init__(self, dim=1280, depth=12, heads=32, condition_dim=2048,
                 output_dim=1024, dropout=0.2, gradient_checkpointing=True):
        super().__init__()
        if dim % heads or dim % 4 or depth < 2 or depth % 2:
            raise ValueError("DiT requires an even depth and width divisible by heads and 4")
        self.flow_time = ScalarTimeEncoder(dim)
        self.blocks = nn.ModuleList([
            ActionDiTBlock(dim, heads, condition_dim, dropout, cross=i % 2 == 0) for i in range(depth)
        ])
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.output_projection = nn.Linear(dim, output_dim)
        self.gradient_checkpointing = gradient_checkpointing

    def prepare_context(self, context, valid):
        context = masked_values(context, valid)
        return tuple(block.context_kv(context) if block.cross else None for block in self.blocks)

    def forward(self, action, flow_time, prefix_tokens, prefix_valid, context_tokens, context_valid,
                *, future_valid=None, context_kv=None):
        if future_valid is None:
            future_valid = torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
        action = masked_values(action, future_valid)
        prefix_tokens = masked_values(prefix_tokens, prefix_valid)
        context_tokens = masked_values(context_tokens, context_valid)
        valid = torch.cat((prefix_valid, future_valid), 1).bool()
        x = torch.cat((prefix_tokens, action), 1)
        time = self.flow_time(flow_time)
        for i, block in enumerate(self.blocks):
            kv = context_kv[i] if context_kv is not None else None
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, time, valid, context_tokens, context_valid, use_reentrant=False)
            else:
                x = block(x, time, valid, context_tokens, context_valid, kv)
        shift, scale = self.output_modulation(time).chunk(2, -1)
        output = self.output_norm(x[:, prefix_tokens.shape[1]:]) * (1 + scale[:, None]) + shift[:, None]
        return self.output_projection(output).masked_fill(~future_valid[..., None], 0)


@dataclass
class ActionCondition:
    prefix_tokens: torch.Tensor
    prefix_valid: torch.Tensor
    context_tokens: torch.Tensor
    context_valid: torch.Tensor
    history_valid: torch.Tensor


class TCPActionPolicy(nn.Module):
    def __init__(
        self, arc, *, language_encoder=None, t5_model="google-t5/t5-base", text_max_length=128,
        dim=1280, depth=12, heads=32, condition_dim=2048, output_dim=1024, dropout=0.2,
        prediction_horizon=32, global_token_grid=(14, 19), gradient_checkpointing=True,
        padding=(1, 1, 6, 6), wrist_model="facebook/dinov2-giant", wrist_pretrained=True,
        wrist_encoder_config=None, wrist_cache_dir=None, wrist_encoder=None,
    ):
        super().__init__()
        if not 1 <= prediction_horizon <= 1024:
            raise ValueError("Prediction horizon must be in [1,1024]")
        if arc.tcp_visual_query_encoder.window_size != 3 or arc.tcp_visual_query_encoder.num_arms != 2:
            raise ValueError("V2 requires two arms and a 3x3 TCP query window")
        if dim % 4 or condition_dim % 4:
            raise ValueError("Token widths must be divisible by 4")
        if len(global_token_grid) != 2 or any(int(n) < 1 for n in global_token_grid):
            raise ValueError("global_token_grid must contain two positive dimensions")
        self.arc, self.num_arms = arc, 2
        self.decode_camera = False
        self.decode_geometry = True
        self.prediction_horizon, self.padding = prediction_horizon, tuple(padding)
        self.language_encoder = language_encoder if language_encoder is not None else FrozenT5Encoder(t5_model, text_max_length)
        self.language_encoder.requires_grad_(False).eval()
        self.wrist_encoder = wrist_encoder if wrist_encoder is not None else FrozenDINOv2Encoder(
            wrist_model, pretrained=wrist_pretrained, model_config=wrist_encoder_config, cache_dir=wrist_cache_dir,
        )
        self.wrist_encoder.requires_grad_(False).eval()
        self.state_encoder = nn.Sequential(nn.Linear(14, 1024), nn.ReLU(), nn.Linear(1024, dim))
        self.history_encoder = TCPHistoryEncoder(arc.tcp_visual_query_encoder.embed_dim, dim)
        self.prefix_type = nn.Embedding(3, dim)  # state, TCP, action
        self.global_encoder = GlobalVisualEncoder(2 * arc.tcp_visual_query_encoder.embed_dim, condition_dim,
                                                  arc.tcp_visual_query_encoder.patch_size, global_token_grid)
        self.wrist_visual_encoder = GlobalVisualEncoder(self.wrist_encoder.output_dim, condition_dim,
                                                        self.wrist_encoder.patch_size)
        self.language_projection = nn.Sequential(nn.LayerNorm(self.language_encoder.output_dim),
                                                  nn.Linear(self.language_encoder.output_dim, condition_dim))
        self.context_type = nn.Embedding(4, condition_dim)  # third-person, left, right, text
        self.context_norm = nn.LayerNorm(condition_dim)
        self.action_encoder = ActionEncoder(dim)
        self.dit = TCPActionDiT(dim, depth, heads, condition_dim, output_dim, dropout, gradient_checkpointing)
        self.action_decoder = nn.Sequential(nn.Linear(output_dim, 1024), nn.ReLU(), nn.Linear(1024, 20))
        for kind, width in (("action", 20), ("state", 14)):
            for name in ("offset", "scale"):
                self.register_buffer(f"{kind}_{name}", torch.full((1, width), float("nan")))

    def train(self, mode=True):
        super().train(mode)
        self.language_encoder.eval()
        self.wrist_encoder.eval()
        return self

    def encode_wrist_images(self, wrist_images):
        if wrist_images is None:
            raise ValueError("V2 requires current left/right wrist_images")
        with torch.no_grad():
            return self.wrist_encoder(wrist_images)

    def set_state_stats(self, offset, scale):
        offset = torch.as_tensor(offset, dtype=torch.float32, device=self.state_offset.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.state_scale.device)
        componentwise_normalize(torch.zeros_like(self.state_offset), offset, scale)
        if not bool((offset[:, [6, 13]] == 0.5).all() & (scale[:, [6, 13]] == 0.5).all()):
            raise ValueError("State grippers require [0,1] to [-1,1] normalization")
        self.state_offset.copy_(offset)
        self.state_scale.copy_(scale)

    def normalize_state(self, state, batch):
        if state is None or state.shape != (batch, 14) or not bool(torch.isfinite(state).all()):
            raise ValueError("current_robot_state must be finite real joint/gripper state [B,14]")
        if not bool(((state[:, [6, 13]] >= 0) & (state[:, [6, 13]] <= 1)).all()):
            raise ValueError("Real state grippers must be in [0,1]")
        return componentwise_normalize(state.float(), self.state_offset, self.state_scale)

    def set_action_component_stats(self, offset, scale):
        offset = torch.as_tensor(offset, dtype=torch.float32, device=self.action_offset.device)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=self.action_scale.device)
        componentwise_normalize(torch.zeros_like(self.action_offset), offset, scale)
        identity = torch.arange(self.num_arms * 10, device=offset.device) % 10 >= 3
        if not bool((offset[:, identity] == 0).all() & (scale[:, identity] == 1).all()):
            raise ValueError("Rotation6D and gripper require identity normalization")
        self.action_offset.copy_(offset)
        self.action_scale.copy_(scale)

    def normalize_actions(self, actions):
        if actions.shape[-2:] != (self.num_arms, 10):
            raise ValueError(f"Expected actions [...,H,{self.num_arms},10]")
        return componentwise_normalize(actions.float().flatten(-2), self.action_offset, self.action_scale)

    def denormalize_actions(self, actions):
        value = componentwise_unnormalize(actions.float(), self.action_offset, self.action_scale)
        return value.unflatten(-1, (self.num_arms, 10))

    def reconstruct(self, images, initial_query_points):
        if images.ndim != 5 or initial_query_points.shape != (images.shape[0], self.num_arms, 2):
            raise ValueError(f"Expected images [B,K,3,H,W] and initial TCP points [B,{self.num_arms},2]")
        prediction = self.arc(
            [{"img": frame} for frame in images.unbind(1)],
            inference_track=False, decode_camera=self.decode_camera, decode_motion=False,
            decode_geometry=self.decode_geometry,
            decode_tcp=True, tcp_query_points=initial_query_points,
            return_aux_pyramid=False, return_backbone_features=True,
            force_no_output_conversion=True, ref_view_strategy="first",
        )
        return prediction, prediction.pop("backbone_features")

    def encode_text(self, instructions):
        text, valid = self.language_encoder(list(instructions))
        text = self.language_projection(masked_values(text, valid)) + self.context_type.weight[3]
        return text, valid.bool()

    def replace_instruction(self, condition, instructions):
        text, valid = self.encode_text(instructions)
        gh, gw = self.global_encoder.target_grid
        image_count = gh * gw + 2 * (self.wrist_encoder.image_height // self.wrist_encoder.patch_size) * (self.wrist_encoder.image_width // self.wrist_encoder.patch_size)
        context = torch.cat((condition.context_tokens[:, :image_count], self.context_norm(text)), 1)
        context_valid = torch.cat((condition.context_valid[:, :image_count], valid), 1)
        return ActionCondition(condition.prefix_tokens, condition.prefix_valid,
                               context.masked_fill(~context_valid[..., None], 0), context_valid,
                               condition.history_valid)

    def make_condition(
        self, images, intrinsics, instructions, reconstruction, features, *, current_robot_state,
        centres=None, valid=None, wrist_images=None, wrist_features=None,
    ):
        batch, frames, _, height, width = images.shape
        state = self.state_encoder(self.normalize_state(current_robot_state, batch))[:, None]
        if centres is None:
            if valid is not None:
                raise ValueError("A validity override requires explicit history centres")
            centres, valid = project_tcp(reconstruction["tcp_position"], intrinsics,
                                         image_height=height, image_width=width, padding=self.padding)
        elif valid is None:
            raise ValueError("Teacher-forced centres require a validity mask")
        valid = valid.bool() & torch.isfinite(centres).all(-1)
        centres = masked_values(centres.float(), valid)
        channels = self.arc.tcp_visual_query_encoder.embed_dim
        last_layer_patches = features[-1][0]
        patches = last_layer_patches[..., -channels:]
        sampled, _ = self.arc.tcp_visual_query_encoder(
            patches.flatten(0, 1), centres.flatten(0, 1), image_height=height, image_width=width,
        )
        sampled = sampled.reshape(batch, frames, 2, 9, channels)
        valid = valid & torch.isfinite(sampled).all(dim=(-1, -2))
        history, history_valid = self.history_encoder(sampled, centres, valid,
                                                     image_height=height, image_width=width)
        prefix = torch.cat((state + self.prefix_type.weight[0], history + self.prefix_type.weight[1]), 1)
        prefix_valid = torch.cat((torch.ones(batch, 1, device=images.device, dtype=torch.bool), history_valid), 1)
        global_tokens = self.global_encoder(last_layer_patches[:, -1], image_height=height, image_width=width)
        context = [global_tokens + self.context_type.weight[0]]
        if wrist_images is not None and wrist_features is not None:
            raise ValueError("Pass either wrist_images or cached wrist_features")
        if wrist_features is None:
            wrist_features = self.encode_wrist_images(wrist_images)
        count = (self.wrist_encoder.image_height // self.wrist_encoder.patch_size) * (self.wrist_encoder.image_width // self.wrist_encoder.patch_size)
        if wrist_features.shape != (batch, 2, count, self.wrist_encoder.output_dim):
            raise ValueError("Cached wrist features must match the left/right patch grid")
        for side in range(2):
            context.append(self.wrist_visual_encoder(wrist_features[:, side],
                image_height=self.wrist_encoder.image_height, image_width=self.wrist_encoder.image_width)
                + self.context_type.weight[1 + side])
        text, text_valid = self.encode_text(instructions)
        image_tokens = torch.cat(context, 1)
        context_valid = torch.cat((torch.ones(image_tokens.shape[:2], device=images.device, dtype=torch.bool), text_valid), 1)
        context_tokens = self.context_norm(torch.cat((image_tokens, text.to(image_tokens)), 1))
        return ActionCondition(prefix.masked_fill(~prefix_valid[..., None], 0), prefix_valid,
                               masked_values(context_tokens, context_valid), context_valid, valid)

    @torch.no_grad()
    def training_history_queries(self, batch, reconstruction, gt_ratio):
        """Choose one coherent GT or recovered trajectory per sample.

        Recovered coordinates are conditioning inputs, not an action-loss path
        into the TCP recovery head. Invalid predictions retain missing tokens.
        """
        gt_centres = batch["history_tcp_query_points"]
        gt_valid = batch["history_tcp_valid"].bool()
        batch_size = gt_centres.shape[0]
        if gt_ratio == 1:
            return gt_centres, gt_valid, torch.ones(batch_size, dtype=torch.bool, device=gt_centres.device)
        pred_centres, pred_valid = project_tcp(
            reconstruction["tcp_position"].detach(), batch["intrinsics"],
            image_height=batch["images"].shape[-2], image_width=batch["images"].shape[-1],
            padding=self.padding,
        )
        use_gt = (torch.rand(batch_size, device=gt_centres.device) < gt_ratio
                  if gt_ratio > 0 else torch.zeros(batch_size, dtype=torch.bool, device=gt_centres.device))
        centres = torch.where(use_gt[:, None, None, None], gt_centres, pred_centres)
        valid = torch.where(use_gt[:, None, None], gt_valid, pred_valid)
        return centres, valid, use_gt

    @staticmethod
    def flow_buckets(time):
        return (time * 1000).long().clamp(0, 999).float()

    def predict_velocity(self, noisy, time, condition, *, future_valid=None, context_kv=None):
        if future_valid is None:
            future_valid = torch.ones(noisy.shape[:2], device=noisy.device, dtype=torch.bool)
        noisy = masked_values(noisy, future_valid)
        buckets = self.flow_buckets(time)
        action = self.action_encoder(noisy, buckets) + self.prefix_type.weight[2]
        hidden = self.dit(action, buckets, condition.prefix_tokens, condition.prefix_valid,
                          condition.context_tokens, condition.context_valid,
                          future_valid=future_valid, context_kv=context_kv)
        return self.action_decoder(hidden).masked_fill(~future_valid[..., None], 0)

    def forward(self, batch, *, noise=None, flow_time=None, history_gt_ratio=1.0):
        # Fail before the expensive reconstruction if deployment/training state is missing.
        self.normalize_state(batch.get("current_robot_state"), batch["images"].shape[0])
        reconstruction, features = self.reconstruct(batch["images"], batch["tcp_query_points"])
        centres, valid, use_gt = self.training_history_queries(batch, reconstruction, history_gt_ratio)
        condition = self.make_condition(
            batch["images"], batch["intrinsics"], batch["instruction"], reconstruction, features,
            current_robot_state=batch["current_robot_state"], centres=centres, valid=valid,
            wrist_images=batch.get("wrist_images"),
        )
        target = self.normalize_actions(batch["future_actions"])
        future_valid = batch.get("future_action_valid")
        if future_valid is None:
            future_valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        future_valid = future_valid.bool()
        target = masked_values(target, future_valid)
        noise = torch.randn_like(target) if noise is None else noise
        noise = masked_values(noise, future_valid)
        if flow_time is None:
            u = torch.distributions.Beta(1.5, 1.0).sample((target.shape[0],)).to(target.device)
            flow_time = (1 - u) * 0.999
        if flow_time.shape != (target.shape[0],) or not bool((torch.isfinite(flow_time) & (flow_time >= 0) & (flow_time < 1)).all()):
            raise ValueError("flow_time must be finite [B] in [0,1)")
        tau = flow_time[:, None, None]
        noisy = (1 - tau) * noise + tau * target
        velocity = self.predict_velocity(noisy, flow_time, condition, future_valid=future_valid)
        return {"reconstruction": reconstruction, "action_velocity": velocity,
                "action_target_velocity": target - noise, "future_action_valid": future_valid,
                "history_gt_fraction": use_gt.float().mean(),
                "history_valid_fraction": condition.history_valid.float().mean()}

    @torch.no_grad()
    def sample_condition(self, condition, *, steps=4, generator=None, cache_context=True):
        if self.training:
            raise RuntimeError("Call policy.eval() before sampling")
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        batch = condition.prefix_tokens.shape[0]
        actions = torch.randn(batch, self.prediction_horizon, 20, device=condition.prefix_tokens.device,
                              dtype=torch.float32, generator=generator)
        context_kv = self.dit.prepare_context(condition.context_tokens, condition.context_valid) if cache_context else None
        for index in range(steps):
            tau = torch.full((batch,), index / steps, device=actions.device)
            actions = actions + self.predict_velocity(actions, tau, condition, context_kv=context_kv).float() / steps
        value = self.denormalize_actions(actions)
        success = torch.isfinite(value).flatten(1).all(1)
        position = value[..., :3]
        rotation = safe_rotation_6d_to_matrix(value[..., 3:9])
        gripper = (value[..., 9] >= 0).long()
        result = {
            "success": success,
            "action_position": position.masked_fill(~success[:, None, None, None], float("nan")),
            "action_rotation": rotation.masked_fill(~success[:, None, None, None, None], float("nan")),
            "action_gripper": gripper.masked_fill(~success[:, None, None], -1),
            "action_gripper_open": ((value[..., 9] + 1) / 2).clamp(0, 1).masked_fill(~success[:, None, None], float("nan")),
            "action_gripper_score": value[..., 9].masked_fill(~success[:, None, None], float("nan")),
            "future_step_indices": torch.arange(
                1, self.prediction_horizon + 1, device=actions.device,
            ).expand(batch, -1),
        }
        return result

    @torch.no_grad()
    def sample_actions(self, images, instructions, initial_query_points, intrinsics, *,
                       current_robot_state, wrist_images, steps=4, generator=None):
        """Current real state [B,14], padded third-person history and current wrist RGBs."""
        if self.training:
            raise RuntimeError("Call policy.eval() before sampling")
        self.normalize_state(current_robot_state, images.shape[0])
        reconstruction, features = self.reconstruct(images, initial_query_points)
        condition = self.make_condition(images, intrinsics, instructions, reconstruction, features,
                                         current_robot_state=current_robot_state, wrist_images=wrist_images)
        return self.sample_condition(condition, steps=steps, generator=generator)
