"""QuantVLA numerical base adapted to openpi's joint-attention pi0.5.

References: QuantVLA Appendix D and AIoT-MLSys-Lab/QuantVLA's released
duquant_preprocess.py / dit_atm.py. The release uses weight-energy smoothing,
signed per-channel activation QDQ, and multiplicative teacher/student matching.
No GR00T coefficients or pretrained rotations are reused for pi0.5.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.quantization.config import PTQConfig
from openpi.quantization.numerics import FakeQuantLinear
from openpi.quantization.numerics import quantize_weight


def block_rotate(values, matrices, block_size, *, transpose=False):
    """Apply independent orthogonal blocks, including an unpadded tail block."""
    dimension = values.shape[-1]
    padded = F.pad(values.float(), (0, (-dimension) % block_size))
    blocks = padded.reshape(-1, len(matrices), block_size)
    matrices = matrices.to(device=values.device, dtype=torch.float32)
    if transpose:
        matrices = matrices.transpose(-1, -2)
    result = torch.einsum("rnb,nbc->rnc", blocks, matrices)
    return result.reshape(*values.shape[:-1], -1)[..., :dimension]


def rotation_basis(weight, block_size, *, rows=False, device="auto"):
    """Batched covariance eigensystems give the small left SVD bases.

    This avoids computing the very large, unused right singular vectors of
    rectangular MLP matrices. Signs/degenerate bases need not match GR00T SVD.
    """
    work_device = weight.device if device == "auto" else torch.device(device)
    work = weight.detach().to(device=work_device, dtype=torch.float32)
    if rows:
        work = work.T
    dimension = work.shape[1]
    count = math.ceil(dimension / block_size)
    result = torch.eye(block_size, device=work_device).repeat(count, 1, 1)
    full = dimension // block_size
    if full:
        blocks = work[:, : full * block_size].reshape(-1, full, block_size).permute(1, 0, 2)
        covariance = torch.bmm(blocks.transpose(1, 2), blocks) / max(1, work.shape[0])
        result[:full] = torch.linalg.eigh(covariance).eigenvectors.flip(-1)
    tail = dimension % block_size
    if tail:
        block = work[:, full * block_size :]
        covariance = block.T @ block / max(1, work.shape[0])
        result[-1, :tail, :tail] = torch.linalg.eigh(covariance).eigenvectors.flip(-1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite QuantVLA rotation")
    return result.to(weight.device)


class QuantVLALinear(FakeQuantLinear):
    """Floating QDQ in a block-rotated basis, with original output restoration.

    W' = R_out W[:,perm] R_in; x' = x[...,perm] R_in.
    y = linear(QDQ(x'), QDQ(W'), R_out b) R_out.
    Rotation buffers are serialized; there is no floating master weight.
    """

    transform_buffer_names = ("input_permutation", "input_rotations", "output_rotations")

    def __init__(self, source, bits, act_bits, absmax, config, *, initialize=True):
        super().__init__(source, bits, act_bits, absmax, quantize=False)
        self.block_size = config.rotation_block_size
        self.output_block_size = config.rotation_output_block_size
        self.scale_mode = config.weight_scale_mode
        self.transform_enabled = bits < 16 or act_bits < 16
        self.row_rotation = config.row_rotation == "restore" and self.transform_enabled
        self.activation_percentile = config.activation_percentile
        self.activation_observation_limit = config.activation_calibration_batches
        self.observation_batches = 0
        self.register_buffer("input_permutation", torch.arange(self.in_features, device=self.weight.device))
        self.register_buffer(
            "input_rotations",
            torch.eye(self.block_size, device=self.weight.device).repeat(
                math.ceil(self.in_features / self.block_size), 1, 1
            ),
        )
        self.register_buffer(
            "output_rotations",
            torch.eye(self.output_block_size, device=self.weight.device).repeat(
                math.ceil(self.out_features / self.output_block_size), 1, 1
            ),
        )
        if initialize:
            with torch.no_grad():
                weight = source.weight.detach().float()
                if self.transform_enabled:
                    if config.rotation_permute:
                        energy = weight.square().mean(0)
                        energy = (1 - config.quantvla_smoothing) * energy + config.quantvla_smoothing * energy.mean()
                        ordered = torch.argsort(energy, descending=True, stable=True)
                        permutation = torch.empty_like(ordered)
                        permutation[::2] = ordered[: math.ceil(len(ordered) / 2)]
                        permutation[1::2] = ordered.flip(0)[: len(ordered) // 2]
                        self.input_permutation.copy_(permutation)
                    weight = weight[:, self.input_permutation]
                    self.input_rotations.copy_(rotation_basis(weight, self.block_size, device=config.rotation_device))
                    if self.row_rotation:
                        self.output_rotations.copy_(
                            rotation_basis(weight, self.output_block_size, rows=True, device=config.rotation_device)
                        )
                    weight = block_rotate(weight, self.input_rotations, self.block_size)
                    if self.row_rotation:
                        weight = block_rotate(weight.T, self.output_rotations, self.output_block_size, transpose=True).T
                        if self.bias is not None:
                            self.bias.copy_(self._rotate_output(source.bias[None], transpose=True)[0])
                self.weight.copy_(self._qdq(weight).to(self.weight.dtype))

    def _qdq(self, weight):
        return quantize_weight(weight, self.weight_bits, scale_mode=self.scale_mode, reciprocal_grid=True)

    def _rotate_output(self, values, *, transpose=False):
        if not self.row_rotation:
            return values.float()
        return block_rotate(values, self.output_rotations, self.output_block_size, transpose=transpose)

    def reset_observer(self):
        self.observed_absmax.zero_()
        self.observation_batches = 0
        self.observe = True

    def forward(self, inputs):
        if self.transform_enabled:
            inputs = block_rotate(inputs[..., self.input_permutation], self.input_rotations, self.block_size).to(
                inputs.dtype
            )
        if self.observe and self.observation_batches < self.activation_observation_limit:
            values = inputs.detach().float().reshape(-1, self.in_features).abs()
            threshold = torch.quantile(values, self.activation_percentile / 100, dim=0)
            self.observed_absmax.copy_(torch.maximum(self.observed_absmax, threshold))
            self.observation_batches += 1
        if self.activation_quant_enabled and self.activation_bits < 16:
            limit = 2 ** (self.activation_bits - 1) - 1
            scale = self.activation_absmax.clamp_min(1e-8) / limit
            inputs = ((inputs.float() / scale).round().clamp(-limit - 1, limit) * scale).to(inputs.dtype)
        result = F.linear(inputs, self.weight, self.bias)
        return self._rotate_output(result).to(result.dtype) if self.row_rotation else result

    @torch.no_grad()
    def fold_affine(self, correction, shift):
        # CSRC is expressed in restored output coordinates. Conjugate it into
        # the stored row-rotated basis; do not fold it as if W' were original W.
        correction = correction.to(self.weight.device, torch.float32)
        shift = shift.to(self.weight.device, torch.float32)
        if self.row_rotation:
            correction = self._rotate_output(correction, transpose=True)
            correction = self._rotate_output(correction.T, transpose=True).T
            shift = self._rotate_output(shift[None], transpose=True)[0]
        weight = correction @ self.weight.float()
        bias = shift if self.bias is None else correction @ self.bias.float() + shift
        self.weight.copy_(self._qdq(weight).to(self.weight.dtype))
        if self.bias is None:
            self.bias = nn.Parameter(bias.to(self.weight.dtype), requires_grad=False)
        else:
            self.bias.copy_(bias.to(self.bias.dtype))


def matching_gain(teacher, student, log_clamp, neutral_band, eps=1e-6):
    """Multiplicative teacher/student correction used by the released code."""
    teacher = torch.as_tensor(teacher, dtype=torch.float64)
    student = torch.as_tensor(student, dtype=torch.float64)
    raw = torch.where(student > eps, teacher / (student + eps), torch.ones_like(student))
    log = raw.clamp_min(eps).log().clamp(-log_clamp, log_clamp)
    return torch.where(log.abs() < neutral_band, torch.ones_like(log), log.exp()).float()


class AttentionStatistics:
    """Capture real post-RoPE suffix-query logits and expert output energies.

    Joint prefill uses both prefix and suffix keys, just like cached inference.
    Hooks and callbacks exist only during offline calibration.
    """

    def __init__(self, model):
        self.modules = {
            name: module
            for name, module in model.named_modules()
            if "gemma_expert.model.layers." in name and name.endswith(".self_attn")
        }
        self.logits = {}
        self.outputs = {}
        self.handles = []
        self.active = True

    def __enter__(self):
        if not self.modules:
            raise ValueError("QuantVLA ATM/OHB found no action-expert attention modules")
        for name, module in self.modules.items():
            if not hasattr(module, "head_dim"):
                raise ValueError(f"Missing attention head geometry: {name}")
            if hasattr(module, "_quantvla_logits_observer"):
                raise RuntimeError("Nested QuantVLA attention observers are unsupported")
        for name, module in self.modules.items():
            module._quantvla_logits_observer = self._capture_logits(name)  # noqa: SLF001 -- paired offline adapter hook
            self.handles.append(module.o_proj.register_forward_hook(self._capture_output(name)))
        return self

    def _capture_logits(self, name):
        @torch.no_grad()
        def capture(query, key, attention_mask, scaling):
            if not self.active:
                return
            if query.shape[1] % key.shape[1]:
                raise ValueError("Invalid grouped-query attention geometry")
            key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
            logits = query.float() @ key.float().transpose(-1, -2) * scaling
            valid = torch.ones_like(logits, dtype=torch.bool)
            if attention_mask is not None:
                mask = attention_mask[..., : key.shape[-2]]
                valid &= mask if mask.dtype == torch.bool else mask >= 0
            values = torch.where(valid, logits, 0).double()
            packed = torch.stack([values.sum((0, 2, 3)), values.square().sum((0, 2, 3)), valid.sum((0, 2, 3))]).cpu()
            self.logits[name] = self.logits.get(name, torch.zeros_like(packed)) + packed

        return capture

    def _capture_output(self, name):
        @torch.no_grad()
        def capture(module, args, output):
            if self.active:
                packed = torch.tensor(
                    [output.detach().double().square().sum().item(), output.numel()], dtype=torch.float64
                )
                self.outputs[name] = self.outputs.get(name, torch.zeros_like(packed)) + packed

        return capture

    def finalize(self):
        missing = set(self.modules) - self.logits.keys()
        if missing:
            raise RuntimeError(f"No joint-attention logits captured; update gemma_pytorch.py: {sorted(missing)}")
        result = {}
        for name, (total, squares, count) in self.logits.items():
            if (count == 0).any():
                raise ValueError(f"No valid attention entries: {name}")
            std = (squares / count - (total / count).square()).clamp_min(0).sqrt()
            energy = self.outputs[name]
            result[name] = {"logit_std": std.tolist(), "output_rms": float((energy[0] / energy[1]).sqrt())}
        return result

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for module in self.modules.values():
            if hasattr(module, "_quantvla_logits_observer"):
                del module._quantvla_logits_observer  # noqa: SLF001


@torch.no_grad()
def fold_attention_matching(model, teacher, student, config: PTQConfig, *, component):
    result = {}
    for name, reference in teacher.items():
        if name not in student:
            raise ValueError(f"Missing student attention statistics: {name}")
        module = model.get_submodule(name)
        if component == "atm":
            gain = matching_gain(
                reference["logit_std"],
                student[name]["logit_std"],
                config.atm_log_clamp,
                config.matching_neutral_band,
                config.eps,
            )
            # Scale Q alone, not K: pi0.5 has shared prefix K/V and GQA. This
            # corrects every suffix-query head against prefix AND suffix keys.
            rows = gain.repeat_interleave(module.head_dim).to(module.q_proj.weight.device)
            if rows.numel() != module.q_proj.out_features or isinstance(module.q_proj, FakeQuantLinear):
                raise ValueError("ATM requires protected floating expert Q projections")
            module.q_proj.weight.mul_(rows[:, None].to(module.q_proj.weight.dtype))
            if module.q_proj.bias is not None:
                module.q_proj.bias.mul_(rows.to(module.q_proj.bias.dtype))
        elif component == "ohb":
            gain = matching_gain(
                reference["output_rms"],
                student[name]["output_rms"],
                config.ohb_log_clamp,
                config.matching_neutral_band,
                config.eps,
            )
            if isinstance(module.o_proj, FakeQuantLinear):
                raise ValueError("OHB requires protected floating expert output projections")
            module.o_proj.weight.mul_(gain.to(module.o_proj.weight.device, module.o_proj.weight.dtype))
            if module.o_proj.bias is not None:
                module.o_proj.bias.mul_(gain.to(module.o_proj.bias.device, module.o_proj.bias.dtype))
        else:
            raise ValueError(component)
        result[name] = gain.tolist()
    return result
