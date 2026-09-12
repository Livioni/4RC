"""Pooled TCP history and a text-conditioned flow-matching action policy."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from arc.action import project_tcp, safe_rotation_6d_to_matrix, sinusoidal


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


class TCPHistoryPool(nn.Module):
    """One centre-query attention pool per frame and arm; no temporal pooling."""

    def __init__(self, input_dim: int = 1536, dim: int = 512, heads: int = 8):
        super().__init__()
        if dim % 4 or dim % heads:
            raise ValueError("History dimension must be divisible by 4 and heads")
        self.dim = dim
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.pool_norm = nn.LayerNorm(dim)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.pooled_norm = nn.LayerNorm(dim)
        self.xy_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.arm_embedding = nn.Embedding(2, dim)
        self.valid_embedding = nn.Embedding(2, dim)
        self.missing_token = nn.Parameter(torch.zeros(dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self, patches: torch.Tensor, centres: torch.Tensor, valid: torch.Tensor,
        time_embedding: torch.Tensor, history_type: torch.Tensor,
        *, image_height: int, image_width: int,
    ) -> torch.Tensor:
        batch, frames = patches.shape[:2]
        values = self.projection(patches).reshape(batch * frames * 2, 9, self.dim)
        centre = values[:, 4:5]
        normalized = self.pool_norm(values)
        pooled, _ = self.pool(normalized[:, 4:5], normalized, normalized, need_weights=False)
        pooled = self.pooled_norm(centre + pooled).reshape(batch, frames, 2, self.dim)
        pooled = torch.where(valid[..., None], pooled, self.missing_token.to(pooled))
        xy = torch.nan_to_num(centres.float())
        xy = torch.where(valid[..., None], xy, torch.zeros_like(xy))
        # Continuous centre coordinates, not the index of the flattened token.
        x = xy[..., 0] / image_width * (2 * math.pi)
        y = xy[..., 1] / image_height * (2 * math.pi)
        spatial = self.xy_projection(torch.cat((
            sinusoidal(x, self.dim // 2), sinusoidal(y, self.dim // 2),
        ), dim=-1))
        spatial = spatial * valid[..., None]
        identity = self.arm_embedding.weight[None, None]
        output = (
            pooled + spatial + time_embedding[:, :, None]
            + identity + history_type + self.valid_embedding(valid.long())
        )
        return self.output_norm(output).flatten(1, 2)


class ActionDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.text_attention = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    @staticmethod
    def _modulate(x, count, shift, scale):
        return torch.cat((x[:, :count], x[:, count:] * (1 + scale[:, None]) + shift[:, None]), 1)

    @staticmethod
    def _residual(x, update, count, gate):
        return x + torch.cat((update[:, :count], gate[:, None] * update[:, count:]), 1)

    def forward(self, x, time_condition, history_count, mask, text, text_valid, padding_mask=None):
        modulation = self.modulation(time_condition).chunk(9, dim=-1)
        for i in range(3):
            shift, scale, gate = modulation[3 * i:3 * i + 3]
            q = self._modulate(self.norms[i](x), history_count, shift, scale)
            if i == 0:
                update, _ = self.self_attention(q, q, q, attn_mask=mask, key_padding_mask=padding_mask, need_weights=False)
            elif i == 1:
                update, _ = self.text_attention(
                    q, text, text, key_padding_mask=~text_valid, need_weights=False,
                )
            else:
                update = self.mlp(q)
            x = self._residual(x, update, history_count, gate)
        return x


class TCPActionDiT(nn.Module):
    def __init__(self, dim: int = 512, depth: int = 8, heads: int = 8):
        super().__init__()
        self.action_projection = nn.Linear(20, dim)
        self.flow_time = ScalarTimeEncoder(dim)
        self.blocks = nn.ModuleList([ActionDiTBlock(dim, heads) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.output_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.output_projection = nn.Linear(dim, 20)
        nn.init.zeros_(self.output_modulation[-1].weight)
        nn.init.zeros_(self.output_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, noisy_actions, flow_time, history, text, text_valid,
        future_time_embedding, future_type, *, return_history=False, future_valid=None,
    ):
        action = self.action_projection(noisy_actions) + future_time_embedding + future_type
        x = torch.cat((history, action), dim=1)
        count = history.shape[1]
        # Historical hidden states cannot read future actions or their noise.
        mask = torch.zeros(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        mask[:count, count:] = True
        padding_mask = None
        if future_valid is not None:
            padding_mask = torch.cat((
                torch.zeros(x.shape[0], count, dtype=torch.bool, device=x.device),
                ~future_valid.bool(),
            ), dim=1)
        condition = self.flow_time(flow_time)
        for block in self.blocks:
            x = block(x, condition, count, mask, text, text_valid, padding_mask)
        shift, scale = self.output_modulation(condition).chunk(2, dim=-1)
        future = self.output_norm(x[:, count:]) * (1 + scale[:, None]) + shift[:, None]
        velocity = self.output_projection(future)
        return (velocity, x[:, :count]) if return_history else velocity


@dataclass
class ActionCondition:
    history: torch.Tensor
    text: torch.Tensor
    text_valid: torch.Tensor
    future_time_embedding: torch.Tensor
    future_frame_times: torch.Tensor
    history_valid: torch.Tensor


class TCPActionPolicy(nn.Module):
    def __init__(
        self, arc: nn.Module, *, language_encoder: nn.Module | None = None,
        t5_model: str = "google-t5/t5-base", text_max_length: int = 128,
        dim: int = 512, depth: int = 8, heads: int = 8, prediction_horizon: int = 16,
        time_unit_seconds: float = 1 / 15, padding=(1, 1, 6, 6),
    ):
        super().__init__()
        if prediction_horizon < 1 or time_unit_seconds <= 0:
            raise ValueError("Prediction horizon and time unit must be positive")
        if arc.tcp_visual_query_encoder.window_size != 3:
            raise ValueError("Stage two requires a 3x3 TCP query window")
        self.arc = arc
        self.language_encoder = language_encoder if language_encoder is not None else FrozenT5Encoder(
            t5_model, text_max_length,
        )
        self.language_encoder.requires_grad_(False)
        self.language_encoder.eval()
        self.language_projection = nn.Sequential(
            nn.LayerNorm(self.language_encoder.output_dim),
            nn.Linear(self.language_encoder.output_dim, dim),
        )
        self.history_pool = TCPHistoryPool(arc.tcp_visual_query_encoder.embed_dim, dim, heads)
        self.physical_time = ScalarTimeEncoder(dim)
        self.token_type = nn.Embedding(2, dim)
        self.dit = TCPActionDiT(dim, depth, heads)
        self.prediction_horizon = prediction_horizon
        self.time_unit_seconds = time_unit_seconds
        self.padding = tuple(padding)
        self.register_buffer("action_position_mean", torch.zeros(2, 3))
        self.register_buffer("action_position_std", torch.ones(2, 3))

    def train(self, mode=True):
        super().train(mode)
        self.language_encoder.eval()
        return self

    def set_action_position_stats(self, mean, std):
        if mean.shape != (2, 3) or std.shape != (2, 3):
            raise ValueError("Action position statistics must be [2,3]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Action statistics must be finite with positive standard deviations")
        self.action_position_mean.copy_(mean.to(self.action_position_mean))
        self.action_position_std.copy_(std.to(self.action_position_std))

    def normalize_actions(self, actions):
        value = actions.float().clone()
        value[..., :3] = (value[..., :3] - self.action_position_mean) / self.action_position_std
        return value.flatten(-2)

    def denormalize_actions(self, actions):
        value = actions.float().reshape(*actions.shape[:2], 2, 10).clone()
        value[..., :3] = value[..., :3] * self.action_position_std + self.action_position_mean
        return value

    def reconstruct(self, images, initial_query_points):
        if images.ndim != 5 or initial_query_points.shape != (images.shape[0], 2, 2):
            raise ValueError("Expected images [B,K,3,H,W] and initial TCP points [B,2,2]")
        prediction = self.arc(
            [{"img": frame} for frame in images.unbind(1)],
            inference_track=False, decode_camera=False, decode_motion=False,
            decode_tcp=True, tcp_query_points=initial_query_points,
            return_aux_pyramid=False, return_backbone_features=True,
            force_no_output_conversion=True, ref_view_strategy="first",
        )
        return prediction, prediction.pop("backbone_features")

    def make_condition(
        self, images, intrinsics, frame_times, future_frame_times, instructions,
        reconstruction, features, *, centres=None, valid=None,
    ):
        batch, frames, _, height, width = images.shape
        if frame_times.shape != (batch, frames) or future_frame_times.shape != (batch, self.prediction_horizon):
            raise ValueError("Observation/future times do not match the configured horizons")
        if (frame_times[:, 1:] <= frame_times[:, :-1]).any():
            raise ValueError("Historical observations must be in strictly increasing time order")
        if (future_frame_times[:, 0] <= frame_times[:, -1]).any() or (
            future_frame_times[:, 1:] <= future_frame_times[:, :-1]
        ).any():
            raise ValueError("Future action times must strictly follow observations")
        if centres is None:
            if valid is not None:
                raise ValueError("A validity override requires explicit history centres")
            centres, valid = project_tcp(
                reconstruction["tcp_position"], intrinsics,
                image_height=height, image_width=width, padding=self.padding,
            )
        elif valid is None:
            raise ValueError("Teacher-forced centres require a validity mask")
        valid = valid.bool() & torch.isfinite(centres).all(-1)
        centres = torch.where(valid[..., None], centres.float(), torch.zeros_like(centres.float()))
        channels = self.arc.tcp_visual_query_encoder.embed_dim
        patches = features[-1][0][..., -channels:]
        # Reuse this module without registering a duplicate child/optimizer group.
        sampled, _ = self.arc.tcp_visual_query_encoder(
            patches.flatten(0, 1), centres.flatten(0, 1),
            image_height=height, image_width=width,
        )
        sampled = sampled.reshape(batch, frames, 2, 9, channels)
        relative = (frame_times - frame_times[:, -1:]) / self.time_unit_seconds
        history = self.history_pool(
            sampled, centres, valid, self.physical_time(relative), self.token_type.weight[0],
            image_height=height, image_width=width,
        )
        text, text_valid = self.language_encoder(list(instructions))
        text = self.language_projection(text)
        future_time = self.physical_time(
            (future_frame_times - frame_times[:, -1:]) / self.time_unit_seconds,
        )
        return ActionCondition(history, text, text_valid, future_time, future_frame_times, valid)

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

    def forward(self, batch, *, noise=None, flow_time=None, history_gt_ratio=1.0):
        """Joint training with a scheduled mixture of GT/recovered history queries."""
        reconstruction, features = self.reconstruct(batch["images"], batch["tcp_query_points"])
        centres, valid, use_gt = self.training_history_queries(batch, reconstruction, history_gt_ratio)
        condition = self.make_condition(
            batch["images"], batch["intrinsics"], batch["frame_times"],
            batch["future_frame_times"], batch["instruction"], reconstruction, features,
            centres=centres, valid=valid,
        )
        target = self.normalize_actions(batch["future_actions"])
        future_valid = batch.get("future_action_valid")
        if future_valid is None:
            future_valid = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        target = target.masked_fill(~future_valid.bool()[..., None], 0)
        noise = torch.randn_like(target) if noise is None else noise
        flow_time = torch.rand(target.shape[0], device=target.device) if flow_time is None else flow_time
        tau = flow_time[:, None, None]
        noisy = ((1 - tau) * noise + tau * target).masked_fill(~future_valid.bool()[..., None], 0)
        velocity = self.dit(
            noisy, flow_time, condition.history, condition.text, condition.text_valid,
            condition.future_time_embedding, self.token_type.weight[1], future_valid=future_valid,
        )
        return {
            "reconstruction": reconstruction,
            "action_velocity": velocity,
            "action_target_velocity": target - noise,
            "future_action_valid": future_valid,
            "history_gt_fraction": use_gt.float().mean(),
            "history_valid_fraction": valid.float().mean(),
        }

    @torch.no_grad()
    def sample_condition(self, condition, *, steps=8, generator=None):
        if self.training:
            raise RuntimeError("Call policy.eval() before sampling")
        if steps < 1:
            raise ValueError("Sampling steps must be positive")
        batch = condition.history.shape[0]
        actions = torch.randn(
            batch, self.prediction_horizon, 20, device=condition.history.device,
            dtype=torch.float32, generator=generator,
        )
        for index in range(steps):
            tau = torch.full((batch,), index / steps, device=actions.device)
            actions = actions + self.dit(
                actions, tau, condition.history, condition.text, condition.text_valid,
                condition.future_time_embedding, self.token_type.weight[1],
            ).float() / steps
        value = self.denormalize_actions(actions)
        success = condition.history_valid.flatten(1).any(1) & torch.isfinite(value).flatten(1).all(1)
        position = value[..., :3]
        rotation = safe_rotation_6d_to_matrix(value[..., 3:9])
        gripper = (value[..., 9] >= 0).long()
        return {
            "success": success,
            "action_position": position.masked_fill(~success[:, None, None, None], float("nan")),
            "action_rotation": rotation.masked_fill(~success[:, None, None, None, None], float("nan")),
            "action_gripper": gripper.masked_fill(~success[:, None, None], -1),
            "action_gripper_score": value[..., 9].masked_fill(~success[:, None, None], float("nan")),
            "future_frame_times": condition.future_frame_times,
        }

    @torch.no_grad()
    def sample_actions(
        self, images, instructions, initial_query_points, frame_times, intrinsics,
        *, action_frequency_hz=15.0, steps=8, generator=None,
    ):
        """Padded RGB in [-1,1]; query points and intrinsics use padded pixels.

        No historical ground-truth poses or projections are accepted here.
        """
        if self.training:
            raise RuntimeError("Call policy.eval() before sampling")
        if action_frequency_hz <= 0:
            raise ValueError("Action frequency must be positive")
        future_times = frame_times[:, -1:] + torch.arange(
            1, self.prediction_horizon + 1, device=images.device,
        )[None] / action_frequency_hz
        reconstruction, features = self.reconstruct(images, initial_query_points)
        condition = self.make_condition(
            images, intrinsics, frame_times, future_times, instructions, reconstruction, features,
        )
        return self.sample_condition(condition, steps=steps, generator=generator)

