"""Replayable, deterministic pi0.5 calibration, without policy fine-tuning."""

from collections.abc import Callable, Iterable
import contextlib
import dataclasses
import logging
import math
import re
import time

import torch
from torch import nn

from openpi.quantization.config import PTQConfig
from openpi.quantization.numerics import FakeQuantLinear
from openpi.quantization.numerics import Moments
from openpi.quantization.numerics import fit_csrc
from openpi.quantization.numerics import normalize_prior
from openpi.quantization.numerics import structural_weights
from openpi.quantization.numerics import weighted_flow_loss

logger = logging.getLogger(__name__)
BatchFactory = Callable[[], Iterable[tuple[object, torch.Tensor]]]


def move_to(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to(item, device) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(
            value, **{field.name: move_to(getattr(value, field.name), device) for field in dataclasses.fields(value)}
        )
    if isinstance(value, list | tuple):
        return type(value)(move_to(item, device) for item in value)
    return value


def replace_module(model, name, replacement):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, replacement)


def select_modules(model: nn.Module, config: PTQConfig):
    linears = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    expert_blocks = sorted(
        {int(match.group(1)) for name in linears if (match := re.search(r"gemma_expert\.model\.layers\.(\d+)\.", name))}
    )
    protected_blocks = (
        set(expert_blocks[-config.skip_last_expert_blocks :]) if config.skip_last_expert_blocks else set()
    )
    protected = [
        name
        for name in linears
        if (match := re.search(r"gemma_expert\.model\.layers\.(\d+)\.mlp\.", name))
        and int(match.group(1)) in protected_blocks
    ]
    quant = {
        name: module
        for name, module in linears.items()
        if re.fullmatch(config.quant_regex, name) and name not in protected
    }
    rank = {name: module for name, module in quant.items() if re.fullmatch(config.rank_regex, name)}
    csrc = {name: module for name, module in quant.items() if re.fullmatch(config.csrc_regex, name)}
    if not quant:
        raise ValueError("quant_regex matched no eligible Linear modules. Inspect the model module names.")
    if config.mode != "uniform" and not rank:
        raise ValueError("rank_regex matched no eligible layers after protecting the last expert blocks.")
    if config.mode != "uniform" and len({module.weight.numel() for module in rank.values()}) != 1:
        raise ValueError(
            "Ranking candidates must have equal weight cost for exact same-budget controls. "
            "Restrict rank_regex to equal-cost projections."
        )
    if config.use_csrc and not csrc:
        raise ValueError("csrc_regex matched no eligible projection")
    if any(module.out_features > config.csrc_max_dim for module in csrc.values()) and config.use_csrc:
        raise ValueError("CSRC projection exceeds configured dense-fit dimension limit")
    return quant, rank, csrc if config.use_csrc else {}, protected


def allocate_precision(candidates: dict[str, nn.Linear], scores: dict[str, float], config: PTQConfig):
    """Global top-k layers, with exact equal-cost BF16 budgets across controls."""
    if config.mode == "uniform":
        return []
    names = sorted(candidates)
    count = math.ceil(len(names) * config.bf16_ratio)
    if config.mode == "random":
        generator = torch.Generator().manual_seed(config.allocation_seed)
        return sorted(names[index] for index in torch.randperm(len(names), generator=generator)[:count].tolist())
    if set(names) != set(scores) or not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Missing or nonfinite gradient sensitivity scores")
    return sorted(sorted(names, key=lambda name: (-scores[name], name))[:count])


def forward_batch(model, observation, actions, seed):
    device = next(model.parameters()).device
    observation, actions = move_to(observation, device), actions.to(device=device, dtype=torch.float32)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    # Reset only the calibration RNG stream, not the caller's inference RNG.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed(seed)
        noise = model.sample_noise(actions.shape, device)
        flow_time = model.sample_time(len(actions), device)
        return model(observation, actions, noise=noise, time=flow_time, train=False)


@contextlib.contextmanager
def activation_collection(model, quant, targets, config, *, collect_inputs):
    """Gather streaming covariance; omit invalid prefix tokens for CSRC."""
    moments = {name: Moments() for name in targets}
    inputs = {name: torch.zeros(module.in_features) for name, module in quant.items()} if collect_inputs else {}
    current_mask = [None]
    original_embed_prefix = getattr(model, "embed_prefix", None)
    had_instance_method = "embed_prefix" in model.__dict__
    if original_embed_prefix is not None:

        def embed_prefix(*args, **kwargs):
            result = original_embed_prefix(*args, **kwargs)
            current_mask[0] = result[1]
            return result

        model.embed_prefix = embed_prefix
    handles = []

    def make_input_hook(name):
        def hook(module, args):
            values = args[0].detach().float().reshape(-1, module.in_features)
            inputs[name] = torch.maximum(inputs[name], values.abs().amax(0).cpu())

        return hook

    def make_output_hook(name):
        def hook(module, args, output):
            values = output.detach()
            mask = current_mask[0]
            if mask is not None and values.ndim == 3 and tuple(mask.shape) == tuple(values.shape[:2]):
                values = values[mask.bool()]
            values = values.reshape(-1, module.out_features)
            if len(values) > config.max_tokens_per_call:
                # Deterministic uniform token subsampling, shared by all passes.
                indices = torch.linspace(0, len(values) - 1, config.max_tokens_per_call, device=values.device).long()
                values = values[indices]
            moments[name].update(values)

        return hook

    try:
        for name, module in quant.items():
            if collect_inputs:
                handles.append(module.register_forward_pre_hook(make_input_hook(name)))
            if name in targets:
                handles.append(module.register_forward_hook(make_output_hook(name)))
        yield inputs, moments
    finally:
        for handle in handles:
            handle.remove()
        if original_embed_prefix is not None:
            if had_instance_method:
                model.embed_prefix = original_embed_prefix
            else:
                del model.embed_prefix


def _sync(model):
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize(next(model.parameters()).device)


def calibrate(model: nn.Module, batches: BatchFactory, config: PTQConfig) -> dict:
    """Calibrate an in-memory model. Only the CSRC matrix is optimized offline.

    All policies, including controls, profile the same full-precision teacher.
    No optimizer updates the pretrained policy; gradients only determine ranking.
    The returned report is also the deployment reconstruction manifest.
    """
    quant, candidates, csrc, protected = select_modules(model, config)
    original_state_keys = list(model.state_dict())
    initial_training = model.training
    model.eval()
    parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    for parameter, _ in parameter_flags:
        parameter.requires_grad_(requires_grad=False)
    device = next(model.parameters()).device
    timings = {}
    report = {
        "format_version": 1,
        "implementation": "floating_QDQ_not_packed_integer",
        "config": config.to_dict(),
        "protected_layers": protected,
        "original_state_keys": original_state_keys,
        "adaptation": {
            "allocation_unit": "linear_layer",
            "scope": "expert_MLP_except_last_two_blocks",
            "csrc_interface": "prefix_key_value_projections_before_RoPE",
            "weight_scale": "per_output_channel_MSE_5_point_signed_zero_point_0",
            "activation_scale": "static_per_input_channel_absmax",
            "padding_loss": "only_first_seven_real_action_channels",
            "reference": "Supplementary A.3/Table 1 defaults ported to pi05",
            "calibration_unit": "sampled_frame_with_action_chunk_not_whole_trajectory",
        },
    }

    def start():
        _sync(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        return time.perf_counter()

    def finish(name, stamp):
        _sync(model)
        timings[name] = {"seconds": time.perf_counter() - stamp}
        if device.type == "cuda":
            timings[name].update(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        logger.info("%s completed in %.2fs", name, timings[name]["seconds"])

    try:
        stamp = start()
        prior_sum, sample_count = torch.zeros(7, dtype=torch.float64), 0
        for _, actions in batches():
            batch_count = len(actions)
            prior_sum += structural_weights(actions, config).cpu() * batch_count
            sample_count += batch_count
        if sample_count == 0:
            raise ValueError("Calibration set is empty")
        prior, effective = normalize_prior(prior_sum / sample_count, config)
        weights = torch.ones(7) if config.mode in {"equal_weight", "uniform", "random"} else effective
        report.update(
            calibration_samples=sample_count,
            structural_prior=prior.tolist(),
            gain_adjusted_prior=effective.tolist(),
            loss_weights=weights.tolist(),
        )
        finish("structural_prior", stamp)

        stamp = start()
        scores = dict.fromkeys(candidates, 0.0)
        grad_counts = dict.fromkeys(candidates, 0)
        need_grad = config.mode in {"da_ptq", "equal_weight"}
        if need_grad:
            for module in candidates.values():
                module.weight.requires_grad_(requires_grad=True)
        probe_count = 0
        teacher_loss = 0.0
        with activation_collection(model, quant, csrc, config, collect_inputs=True) as (input_ranges, teacher_moments):
            for step, (observation, actions) in enumerate(batches()):
                probe = need_grad and step < config.probe_steps
                with torch.set_grad_enabled(probe):
                    losses = forward_batch(model, observation, actions, config.seed + step)
                    loss = weighted_flow_loss(losses, weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite teacher flow loss at batch {step}")
                if probe:
                    loss.backward()
                    for name, module in candidates.items():
                        gradient = module.weight.grad
                        if gradient is not None:
                            scores[name] += float(gradient.detach().float().abs().mean())
                            grad_counts[name] += 1
                    probe_count += 1
                    model.zero_grad(set_to_none=True)
                teacher_loss += float(weighted_flow_loss(losses.detach(), torch.ones(7))) * len(actions)
                del losses, loss
                if step % 16 == 0:
                    logger.info("Teacher profiling batch %d", step)
        for module in candidates.values():
            module.weight.requires_grad_(requires_grad=False)
        if need_grad:
            if any(count == 0 for count in grad_counts.values()):
                raise RuntimeError("Some ranking candidates received no gradient; inspect the module selection")
            scores = {name: value / grad_counts[name] for name, value in scores.items()}
        else:
            scores = {}
        report.update(gradient_scores=scores, probe_batches=probe_count, teacher_flow_mse=teacher_loss / sample_count)
        high_layers = allocate_precision(candidates, scores, config)
        report["high_precision_layers"] = high_layers
        report["candidate_layers"] = sorted(candidates)
        report["candidate_high_fraction"] = len(high_layers) / max(1, len(candidates))
        finish("teacher_statistics_and_gradients", stamp)

        stamp = start()
        added_biases = []
        layer_metadata = {}
        for name, source in quant.items():
            bits = 16 if name in high_layers else config.weight_bits
            if not torch.isfinite(input_ranges[name]).all() or input_ranges[name].max() == 0:
                raise ValueError(f"No finite nonzero input range for {name}")
            replacement = FakeQuantLinear(source, bits, config.activation_bits, input_ranges[name])
            replace_module(model, name, replacement)
            layer_metadata[name] = {
                "weight_bits": bits,
                "activation_bits": config.activation_bits,
                "original_bias": source.bias is not None,
            }
        # Do not retain the old teacher Linear modules or their GPU parameters.
        del quant, candidates, csrc
        parameter_flags = None
        quantized_modules = {name: model.get_submodule(name) for name in layer_metadata}
        targets = {
            name: module
            for name, module in quantized_modules.items()
            if config.use_csrc and re.fullmatch(config.csrc_regex, name)
        }
        finish("weight_quantization", stamp)

        stamp = start()
        quant_loss = 0.0
        with (
            activation_collection(model, quantized_modules, targets, config, collect_inputs=False) as (_, q_moments),
            torch.no_grad(),
        ):
            for step, (observation, actions) in enumerate(batches()):
                losses = forward_batch(model, observation, actions, config.seed + step)
                quant_loss += float(weighted_flow_loss(losses, torch.ones(7))) * len(actions)
        report["quantized_flow_mse_before_csrc"] = quant_loss / sample_count
        finish("mixed_precision_statistics", stamp)

        stamp = start()
        report["csrc"] = {}
        for name, module in targets.items():
            logger.info("Fitting and folding CSRC: %s", name)
            correction, shift, diagnostics = fit_csrc(teacher_moments[name], q_moments[name], config)
            had_bias = module.bias is not None
            module.fold_affine(correction, shift)
            if not had_bias:
                added_biases.append(name)
            diagnostics["requantized_weight_bits"] = module.weight_bits
            report["csrc"][name] = diagnostics
        report["added_fused_bias_layers"] = added_biases
        finish("csrc_fit_svd_fold_requantize", stamp)
        del teacher_moments, q_moments

        stamp = start()
        # Observe downstream ranges of the FINAL weight configuration. No CSRC
        # branches remain, and this pass does not modify any model weights.
        for module in quantized_modules.values():
            module.observe = True
            module.observed_absmax.zero_()
        with torch.no_grad():
            for step, (observation, actions) in enumerate(batches()):
                forward_batch(model, observation, actions, config.seed + step)
        for name, module in quantized_modules.items():
            module.observe = False
            if not torch.isfinite(module.observed_absmax).all():
                raise FloatingPointError(f"Nonfinite final activation range: {name}")
            module.activation_absmax.copy_(module.observed_absmax.clamp_min(1e-8))
        finish("final_activation_calibration", stamp)
        stamp = start()
        final_loss, final_count = 0.0, 0
        with torch.no_grad():
            for step, (observation, actions) in enumerate(batches()):
                if step >= config.probe_steps:
                    break
                losses = forward_batch(model, observation, actions, config.seed + step)
                loss = weighted_flow_loss(losses, torch.ones(7))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Final calibrated model has nonfinite flow loss")
                final_loss += float(loss) * len(actions)
                final_count += len(actions)
        report["final_flow_mse_probe_subset"] = final_loss / final_count
        finish("final_flow_loss_check", stamp)
        report["layers"] = layer_metadata
        report["timings"] = timings
        total = sum(parameter.numel() for parameter in model.parameters())
        low = sum(module.weight.numel() for module in quantized_modules.values() if module.weight_bits < 16)
        report["parameter_counts"] = {
            "total": total,
            "low_bit_assigned_weights": low,
            "high_or_excluded": total - low,
            "high_or_excluded_fraction": (total - low) / total,
            "note": "QDQ keeps floating storage; excluded weights retain source dtype",
        }
        return report
    finally:
        model.zero_grad(set_to_none=True)
        # The final calibrated model is inference-only. On a profiling failure,
        # restore flags of all parameters still present in the original model.
        if parameter_flags is not None:
            for parameter, flag in parameter_flags:
                parameter.requires_grad_(flag)
        model.train(initial_training)
