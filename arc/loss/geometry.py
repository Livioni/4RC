"""4RC geometry-only objective: uncertainty depth, gradients, and rays."""

from __future__ import annotations

import torch
from torch import nn


def _zero_connected(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.sum() * 0.0


def _quantile_filter(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.numel() == 0 or not 0.0 < fraction < 1.0:
        return values
    threshold = torch.quantile(values.detach(), fraction)
    kept = values[values.detach() <= threshold]
    return kept if kept.numel() else values


def uncertainty_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    *,
    gamma: float,
    alpha: float,
    valid_range: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DA3-style confidence-weighted L1 plus raw L1 for monitoring."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} vs {target.shape}")
    if prediction.ndim == valid_mask.ndim:
        error = (prediction - target).abs()
    else:
        error = (prediction - target).abs().mean(dim=-1)
    valid = valid_mask & torch.isfinite(error) & torch.isfinite(confidence)
    if valid.sum() == 0:
        zero = _zero_connected(prediction)
        return zero, zero.detach()
    error = error[valid]
    confidence = confidence[valid].clamp_min(1e-6)
    weighted = gamma * error * confidence - alpha * torch.log(confidence)
    weighted = _quantile_filter(weighted, valid_range)
    raw = _quantile_filter(error, valid_range)
    return weighted.mean(), raw.mean().detach()


def multiscale_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    confidence: torch.Tensor,
    *,
    gamma: float,
    alpha: float,
    scales: int = 4,
) -> torch.Tensor:
    """Uncertainty-aware image-space depth-gradient loss."""
    batch, sequence, height, width = prediction.shape
    prediction = prediction.reshape(batch * sequence, height, width)
    target = target.reshape(batch * sequence, height, width)
    valid_mask = valid_mask.reshape(batch * sequence, height, width)
    confidence = confidence.reshape(batch * sequence, height, width)
    total = _zero_connected(prediction)
    used_scales = 0
    for scale in range(scales):
        step = 2**scale
        pred = prediction[:, ::step, ::step]
        gt = target[:, ::step, ::step]
        mask = valid_mask[:, ::step, ::step]
        conf = confidence[:, ::step, ::step].clamp_min(1e-6)
        residual = pred - gt

        grad_x = (residual[:, :, 1:] - residual[:, :, :-1]).abs().clamp(max=100)
        mask_x = mask[:, :, 1:] & mask[:, :, :-1]
        grad_y = (residual[:, 1:, :] - residual[:, :-1, :]).abs().clamp(max=100)
        mask_y = mask[:, 1:, :] & mask[:, :-1, :]

        parts = []
        if mask_x.any():
            conf_x = conf[:, :, 1:][mask_x]
            parts.append(gamma * grad_x[mask_x] * conf_x - alpha * torch.log(conf_x))
        if mask_y.any():
            conf_y = conf[:, 1:, :][mask_y]
            parts.append(gamma * grad_y[mask_y] * conf_y - alpha * torch.log(conf_y))
        if parts:
            total = total + torch.cat(parts).mean()
            used_scales += 1
    return total / max(used_scales, 1)


class GeometryLoss(nn.Module):
    """Loss from 4RC Eq. (6-7), excluding camera and motion terms."""

    def __init__(
        self,
        *,
        depth_weight: float = 1.0,
        ray_weight: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 0.2,
        depth_valid_range: float = 0.98,
        gradient_scales: int = 4,
        min_valid_pixels: int = 100,
    ) -> None:
        super().__init__()
        self.depth_weight = depth_weight
        self.ray_weight = ray_weight
        self.gamma = gamma
        self.alpha = alpha
        self.depth_valid_range = depth_valid_range
        self.gradient_scales = gradient_scales
        self.min_valid_pixels = min_valid_pixels

    def forward(self, predictions: dict, batch: dict) -> dict[str, torch.Tensor]:
        pred_depth = predictions["depth"].float()
        pred_depth_conf = predictions["depth_conf"].float()
        gt_depth = batch["depth"].float()
        depth_mask = batch["valid_mask"].bool()

        if depth_mask.sum() < self.min_valid_pixels:
            depth_uncertainty = _zero_connected(pred_depth)
            depth_raw = depth_uncertainty.detach()
            depth_gradient = _zero_connected(pred_depth)
        else:
            depth_uncertainty, depth_raw = uncertainty_l1(
                pred_depth,
                gt_depth,
                depth_mask,
                pred_depth_conf,
                gamma=self.gamma,
                alpha=self.alpha,
                valid_range=self.depth_valid_range,
            )
            depth_gradient = multiscale_gradient_loss(
                pred_depth,
                gt_depth,
                depth_mask,
                pred_depth_conf,
                gamma=self.gamma,
                alpha=self.alpha,
                scales=self.gradient_scales,
            )

        pred_ray = predictions["ray"].float()
        pred_ray_conf = predictions["ray_conf"].float()
        ray_uncertainty, ray_raw = uncertainty_l1(
            pred_ray,
            batch["ray_map"].float(),
            batch["ray_valid_mask"].bool(),
            pred_ray_conf,
            gamma=self.gamma,
            alpha=self.alpha,
        )

        depth_loss = depth_uncertainty + depth_gradient
        objective = self.depth_weight * depth_loss + self.ray_weight * ray_uncertainty
        ray_temporal_std = (
            pred_ray.std(dim=1, unbiased=False).mean().detach()
            if pred_ray.shape[1] > 1
            else pred_ray.new_zeros(())
        )
        return {
            "objective": objective,
            "loss_depth": depth_loss.detach(),
            "loss_depth_uncertainty": depth_uncertainty.detach(),
            "loss_depth_gradient": depth_gradient.detach(),
            "metric_depth_l1": depth_raw,
            "loss_ray": ray_uncertainty.detach(),
            "metric_ray_l1": ray_raw,
            "metric_ray_temporal_std": ray_temporal_std,
        }
