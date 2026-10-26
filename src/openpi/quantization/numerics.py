"""Quantizers, the structural prior, and the literal covariance objective."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.quantization.config import PTQConfig


@torch.no_grad()
def quantize_weight(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed, zero-point-zero INT codebook with per-output-channel MSE scales.

    Values remain floating point (QDQ), not packed integers. INT2's signed range
    is [-2, 1]. The five-point clipping search also avoids reducing INT2 to the
    three codes obtained with a naive absmax scale.
    """
    if bits == 16:
        return weight.detach().clone()
    if not 2 <= bits <= 8 or weight.ndim != 2:
        raise ValueError("Expected a linear weight and 2..8 bits (or 16 for bypass).")
    qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    result = torch.empty_like(weight)
    for start in range(0, weight.shape[0], 256):
        values = weight[start : start + 256].float()
        base = values.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
        best_error = torch.full_like(base, float("inf"))
        best = torch.zeros_like(values)
        for factor in (0.5, 0.75, 1.0, 1.25, 1.5):
            scale = base * factor
            candidate = (values / scale).round().clamp(qmin, qmax) * scale
            error = (candidate - values).square().mean(-1, keepdim=True)
            better = error < best_error
            best = torch.where(better, candidate, best)
            best_error = torch.minimum(best_error, error)
        result[start : start + 256] = best.to(weight.dtype)
    return result


class FakeQuantLinear(nn.Linear):
    """One floating GEMM with offline W-QDQ and static per-input-channel A-QDQ.

    A retained BF16 weight still receives A8 inputs. Protected/excluded original
    linears are left untouched. No master weight or compensation branch is kept.
    """

    def __init__(self, source: nn.Linear, bits: int, act_bits: int, absmax: torch.Tensor, *, quantize=True):
        super().__init__(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.weight_bits = bits
        self.activation_bits = act_bits
        self.activation_quant_enabled = True
        self.observe = False
        self.register_buffer("activation_absmax", absmax.to(source.weight.device, torch.float32).clone())
        self.register_buffer("observed_absmax", torch.zeros_like(self.activation_absmax), persistent=False)
        with torch.no_grad():
            self.weight.copy_(quantize_weight(source.weight, bits) if quantize else source.weight)
            if source.bias is not None:
                self.bias.copy_(source.bias)
        self.requires_grad_(requires_grad=False)

    def forward(self, inputs):
        if self.observe:
            observed = inputs.detach().float().reshape(-1, self.in_features).abs().amax(0)
            self.observed_absmax.copy_(torch.maximum(self.observed_absmax, observed))
        if self.activation_quant_enabled:
            limit = 2 ** (self.activation_bits - 1) - 1
            scale = self.activation_absmax.clamp_min(1e-8) / limit
            values = inputs.float()
            inputs = ((values / scale).round().clamp(-limit - 1, limit) * scale).to(inputs.dtype)
        return F.linear(inputs, self.weight, self.bias)

    @torch.no_grad()
    def fold_affine(self, correction: torch.Tensor, shift: torch.Tensor):
        """Fold in float32, then re-quantize to the SAME assigned weight precision."""
        correction = correction.to(self.weight.device, torch.float32)
        shift = shift.to(self.weight.device, torch.float32)
        weight = correction @ self.weight.float()
        bias = shift if self.bias is None else correction @ self.bias.float() + shift
        self.weight.copy_(quantize_weight(weight, self.weight_bits).to(self.weight.dtype))
        if self.bias is None:
            # A biasless projection needs a fused linear bias to represent a
            # nonzero affine shift. This is disclosed and serialized explicitly.
            self.bias = nn.Parameter(bias.to(self.weight.dtype), requires_grad=False)
        else:
            self.bias.copy_(bias.to(self.bias.dtype))


def structural_weights(actions: torch.Tensor, config: PTQConfig) -> torch.Tensor:
    """Eqs. 3--8: six virtual planar segments; gripper only in the generic mode.

    Input is normalized policy actions, NOT robot joints. Returns the normalized
    prior before the supplementary-table sensitivity gain is applied.
    """
    if actions.ndim == 3:
        if config.action_timestep >= actions.shape[1]:
            raise ValueError("action_timestep exceeds the action chunk")
        actions = actions[:, config.action_timestep]
    if actions.ndim != 2 or actions.shape[-1] < 7:
        raise ValueError("Expected [B, H, D] or [B, D] with D >= 7")
    q = actions[..., :7].detach().double()
    theta = q.cumsum(-1)[..., :6]
    j = torch.zeros(q.shape[0], 3, 7, device=q.device, dtype=q.dtype)
    j[:, 0, :6] = -theta.sin().flip(-1).cumsum(-1).flip(-1)
    j[:, 1, :6] = theta.cos().flip(-1).cumsum(-1).flip(-1)
    j[:, 2, :] = 1
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    pinv = j.transpose(-1, -2) @ torch.linalg.solve(
        j @ j.transpose(-1, -2) + config.damping * eye, eye.expand(q.shape[0], -1, -1)
    )
    modes = q.new_tensor([config.planar_weight, config.planar_weight, config.control_weight])
    return (pinv.abs() * modes).sum(-1).mean(0)


def normalize_prior(scores: torch.Tensor, config: PTQConfig) -> tuple[torch.Tensor, torch.Tensor]:
    prior = scores.float() / scores.float().mean().clamp_min(config.eps)
    effective = (1 + config.sensitivity_gain * (prior - 1)).clamp(0.05, 20.0)
    # Common mean keeps weighted/equal gradient scales comparable.
    return prior, effective / effective.mean()


def weighted_flow_loss(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weight only the seven real LIBERO dimensions; padded channels are masked."""
    if losses.ndim != 3 or losses.shape[-1] < len(weights):
        raise ValueError("Expected unreduced [B, horizon, action_dim] flow losses")
    return (losses[..., : len(weights)] * weights.to(losses.device, losses.dtype)).mean()


@dataclass
class Moments:
    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        values = values.detach().reshape(-1, values.shape[-1]).to(dtype=torch.float64)
        if not len(values):
            return
        mean = values.mean(0)
        centered = values - mean
        m2 = (centered.T @ centered).cpu()
        mean = mean.cpu()
        if self.count == 0:
            self.mean, self.m2, self.count = mean, m2, len(values)
            return
        delta = mean - self.mean
        total = self.count + len(values)
        self.m2 += m2 + torch.outer(delta, delta) * (self.count * len(values) / total)
        self.mean += delta * (len(values) / total)
        self.count = total

    def covariance(self):
        if self.count < 2:
            raise ValueError("At least two activation tokens are needed for covariance fitting")
        return self.m2 / self.count


def fit_csrc(teacher: Moments, quantized: Moments, config: PTQConfig):
    """Solve Eq.14 as a dense quartic problem, then truncate M-I as in Eq.15.

    The paper leaves Omega and the solver unspecified. We make these adaptation
    choices explicit: Omega_ij=sqrt(r_i*r_j), r=clipped variance ratios; CPU
    float64 LBFGS initialized at identity, strong-Wolfe line search.
    """
    fp, q = teacher.covariance(), quantized.covariance()
    dimension = len(fp)
    if dimension > config.csrc_max_dim:
        raise ValueError(f"CSRC width {dimension} exceeds csrc_max_dim={config.csrc_max_dim}")
    gain = (fp.diag().clamp_min(0).sqrt() / (q.diag().clamp_min(0).sqrt() + config.eps)).clamp(
        config.csrc_scale_min, config.csrc_scale_max
    )
    aligned = gain[:, None] * q * gain[None, :]
    ratios = (fp.diag() / aligned.diag().clamp_min(config.eps)).clamp(0.25, 4)
    omega = torch.outer(ratios, ratios).sqrt()
    identity = torch.eye(dimension, dtype=torch.float64)
    matrix = nn.Parameter(identity.clone())

    def objective(value):
        mismatch = omega * (fp - value @ aligned @ value.T)
        return (
            config.csrc_feature_weight * mismatch.square().sum()
            + config.csrc_identity_weight * (value - identity).square().sum()
        ) / dimension**2

    before = float(objective(matrix).detach())
    solver = torch.optim.LBFGS(
        [matrix],
        lr=1,
        max_iter=config.csrc_max_iter,
        tolerance_grad=config.csrc_tolerance,
        tolerance_change=config.csrc_tolerance,
        line_search_fn="strong_wolfe",
    )

    def closure():
        solver.zero_grad()
        loss = objective(matrix)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite CSRC covariance objective")
        loss.backward()
        return loss

    with torch.enable_grad():
        solver.step(closure)
    with torch.no_grad():
        dense_objective = float(objective(matrix))
        if not torch.isfinite(matrix).all() or dense_objective > before + max(1e-9, abs(before) * 1e-5):
            raise RuntimeError("CSRC fitting failed to improve the dense objective")
        u, singular, vh = torch.linalg.svd(matrix - identity, full_matrices=False)
        rank = min(config.csrc_rank, dimension)
        truncated = identity + (u[:, :rank] * singular[:rank]) @ vh[:rank]
        correction = truncated * gain[None, :]
        shift = teacher.mean - correction @ quantized.mean
        diagnostics = {
            "objective_before": before,
            "objective_dense": dense_objective,
            "objective_truncated": float(objective(truncated)),
            "rank": rank,
            "solver": "LBFGS/strong_wolfe/identity_init",
            "max_iter": config.csrc_max_iter,
            "iterations": int(solver.state[matrix].get("n_iter", 0)),
            "omega": "sqrt(outer(clip(var_fp/var_aligned_q,0.25,4)))",
            "teacher_tokens": teacher.count,
            "quantized_tokens": quantized.count,
        }
    return correction.float(), shift.float(), diagnostics
