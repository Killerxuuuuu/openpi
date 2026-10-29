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
from openpi.quantization.config import expert_block_index
from openpi.quantization.config import protected_expert_tail_blocks
from openpi.quantization.numerics import FakeQuantLinear
from openpi.quantization.numerics import Moments
from openpi.quantization.numerics import fit_csrc
from openpi.quantization.numerics import normalize_prior
from openpi.quantization.numerics import structural_weights
from openpi.quantization.numerics import weighted_flow_loss
from openpi.quantization.quantvla import AttentionStatistics
from openpi.quantization.quantvla import QuantVLALinear
from openpi.quantization.quantvla import fold_attention_matching

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
    tail_blocks = set(protected_expert_tail_blocks(linears, config.skip_last_expert_blocks))
    first_blocks = set(expert_blocks[: config.skip_first_expert_blocks]) if config.profile != "legacy" else set()
    protected = [
        name
        for name in linears
        if expert_block_index(name) in tail_blocks or (expert_block_index(name) in first_blocks and ".mlp." in name)
    ]
    if config.profile != "legacy":
        protected.extend(
            name
            for name in linears
            if re.fullmatch(r".*gemma_expert\.model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)", name)
        )
    if config.profile == "cogact":
        protected.extend(
            name
            for name in linears
            if ".language_model.layers." in name and re.search(r"\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$", name)
        )
    protected = sorted(set(protected))
    quant = {
        name: module
        for name, module in linears.items()
        if re.fullmatch(config.quant_regex, name) and name not in protected
    }
    rank = {name: module for name, module in quant.items() if re.fullmatch(config.rank_regex, name)}
    csrc_pool = linears if config.profile == "cogact" else quant
    csrc = {
        name: module
        for name, module in csrc_pool.items()
        if re.fullmatch(config.csrc_regex, name) and expert_block_index(name) not in tail_blocks
    }
    if not quant:
        raise ValueError("quant_regex matched no eligible Linear modules. Inspect the model module names.")
    if config.mode not in {"uniform", "quantvla"} and not rank:
        raise ValueError("rank_regex matched no eligible layers after protecting the last expert blocks.")
    if (
        config.profile != "cogact"
        and config.mode not in {"uniform", "quantvla"}
        and len({module.weight.numel() for module in rank.values()}) != 1
    ):
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
    if config.mode in {"uniform", "quantvla"}:
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
    input_counts = dict.fromkeys(quant, 0)
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
            if config.profile != "legacy" and input_counts[name] >= config.activation_calibration_batches:
                return
            values = args[0].detach().float().reshape(-1, module.in_features)
            if config.profile == "cogact":
                values = values[:4096]
            threshold = (
                torch.quantile(values.abs(), config.activation_percentile / 100, dim=0)
                if config.profile != "legacy"
                else values.abs().amax(0)
            )
            inputs[name] = torch.maximum(inputs[name], threshold.cpu())
            input_counts[name] += 1

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
        for name, module in {**quant, **targets}.items():
            if collect_inputs and name in quant:
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

    Historical profiles probe the full-precision teacher; the CogACT recipe
    probes the W-QDQ student through its own stage orchestrator.
    No optimizer fine-tunes policy weights; gradients only determine ranking.
    The returned report is also the deployment reconstruction manifest.
    """
    if config.profile == "cogact":
        from openpi.quantization.cogact import calibrate_cogact

        return calibrate_cogact(model, batches, config)
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
        "format_version": 2 if config.profile != "legacy" else 1,
        "implementation": "floating_QDQ_not_packed_integer",
        "config": config.to_dict(),
        "protected_layers": protected,
        "mandatory_expert_tail_blocks": protected_expert_tail_blocks(
            dict(model.named_modules()), config.skip_last_expert_blocks
        ),
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
    if config.profile != "legacy":
        report["adaptation"].update(
            scope="QuantVLA_LLM_linears_and_expert_MLP_with_fixed_boundary_protection",
            fixed_attention_protection="all_expert_Q_K_V_O_weights_and_activations",
            boundary_protection_source="existing_CogACT_adapter_first1_last2_not_a_claim_from_QuantVLA_paper",
            quantvla_reference="QuantVLA_2602.20309v3_Appendix_D_and_released_GR00T_code",
            quantvla_port="new_pi05_adapter_not_bit_exact_official_pi05_reproduction",
            activation_scale="signed_per_input_channel_percentile_as_released_code",
            rotation="weight_covariance_eigh_SVD_equivalent_blocks_with_output_restore",
            smoothing="released_code_weight_energy_mix_not_unimplemented_diagonal_Lambda",
            matching="multiply_teacher_student_ratios_as_released_code; log_clamp_Appendix_D",
        )
    if config.profile == "cogact":
        report["adaptation"].update(
            reference="local_CogACT_code_and_README; partial_alignment_not_exact_reproduction",
            fixed_attention_protection="all_language_and_expert_Q_K_V_O",
            retained_differences="whole_layer_FP_teacher_ranking; paper_covariance_CSRC; pi05_flow_and_KV_interface; OHB_teacher_match_direction",
            physical_prior="local_CogACT_seven_link_per_batch_surrogate_not_physical_robot_kinematics",
            matching="ATM_linear_clip_0.5_2.0; OHB_log_clip_multiply_not_local_inverse_beta",
            csrc_interface="floating_prefix_KV; compensate_changed_MLP_inputs_without_quantizing_attention",
            boundary_protection_source="local_README_first0_last2_plus_mandatory_full_tail_guard",
        )
    report["adaptation"]["mandatory_tail_protection"] = (
        "at_least_last_two_full_expert_blocks; no_weight_or_activation_QDQ; source_dtype_retained"
    )

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
        weights = torch.ones(7) if config.mode in {"equal_weight", "uniform", "quantvla", "random"} else effective
        report.update(
            calibration_samples=sample_count,
            structural_prior=prior.tolist(),
            gain_adjusted_prior=effective.tolist(),
            loss_weights=weights.tolist(),
            loss_weight_scope="per_batch"
            if config.profile == "cogact" and config.mode == "da_ptq"
            else "dataset_fixed",
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
        matching_enabled = config.use_quantvla and (config.atm_enabled or config.ohb_enabled)
        attention_context = AttentionStatistics(model) if matching_enabled else contextlib.nullcontext()
        with (
            activation_collection(model, quant, csrc, config, collect_inputs=True) as (input_ranges, teacher_moments),
            attention_context as attention_statistics,
        ):
            for step, (observation, actions) in enumerate(batches()):
                if matching_enabled:
                    attention_statistics.active = step < config.quantvla_calibration_steps
                probe = need_grad and step < config.probe_steps
                with torch.set_grad_enabled(probe):
                    losses = forward_batch(model, observation, actions, config.seed + step)
                    batch_weights = weights
                    if config.profile == "cogact" and config.mode == "da_ptq":
                        _, batch_weights = normalize_prior(structural_weights(actions, config), config)
                    loss = weighted_flow_loss(losses, batch_weights)
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
            teacher_attention = attention_statistics.finalize() if matching_enabled else {}
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
        # Release old module references one layer at a time. Holding quant,
        # candidate and parameter_flags dictionaries would double GPU weights.
        floating_targets = {name: module for name, module in csrc.items() if name not in quant}
        floating_metadata = {
            name: {"original_bias": module.bias is not None} for name, module in floating_targets.items()
        }
        del candidates, csrc
        parameter_flags = None
        for name in list(quant):
            source = quant.pop(name)
            bits = 16 if name in high_layers else config.weight_bits
            act_bits = (
                config.fallback_activation_bits
                if name in high_layers and config.profile != "legacy"
                else config.activation_bits
            )
            if not torch.isfinite(input_ranges[name]).all() or input_ranges[name].max() == 0:
                raise ValueError(f"No finite nonzero input range for {name}")
            replacement = (
                QuantVLALinear(source, bits, act_bits, input_ranges[name], config)
                if config.use_quantvla
                else FakeQuantLinear(source, bits, act_bits, input_ranges[name])
            )
            replace_module(model, name, replacement)
            layer_metadata[name] = {
                "weight_bits": bits,
                "activation_bits": act_bits,
                "original_bias": source.bias is not None,
                "wrapper": "quantvla" if config.use_quantvla else "qdq",
            }
        # Do not retain the old teacher Linear modules or their GPU parameters.
        del quant, source
        quantized_modules = {name: model.get_submodule(name) for name in layer_metadata}
        targets = {
            name: module
            for name, module in quantized_modules.items()
            if config.use_csrc and re.fullmatch(config.csrc_regex, name)
        }
        targets.update(floating_targets)
        report["floating_csrc_layers"] = floating_metadata
        finish("weight_quantization", stamp)

        def calibrate_ranges(stage, *, bootstrap=False):
            stamp = start()
            for module in quantized_modules.values():
                if isinstance(module, QuantVLALinear):
                    module.reset_observer()
                else:
                    module.observe = True
                    module.observed_absmax.zero_()
                module.activation_quant_enabled = not bootstrap
            with torch.no_grad():
                for step, (observation, actions) in enumerate(batches()):
                    if config.profile != "legacy" and step >= config.activation_calibration_batches:
                        break
                    forward_batch(model, observation, actions, config.seed + step)
            for name, module in quantized_modules.items():
                module.observe = False
                module.activation_quant_enabled = True
                if not torch.isfinite(module.observed_absmax).all():
                    raise FloatingPointError(f"Nonfinite final activation range: {name}")
                module.activation_absmax.copy_(module.observed_absmax.clamp_min(1e-8))
            finish(stage, stamp)

        if config.use_quantvla:
            calibrate_ranges("rotated_activation_calibration", bootstrap=True)

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
            if isinstance(module, FakeQuantLinear):
                module.fold_affine(correction, shift)
            else:
                with torch.no_grad():
                    correction = correction.to(module.weight.device, torch.float32)
                    shift = shift.to(module.weight.device, torch.float32)
                    bias = shift if module.bias is None else correction @ module.bias.float() + shift
                    module.weight.copy_((correction @ module.weight.float()).to(module.weight.dtype))
                    if module.bias is None:
                        module.bias = nn.Parameter(bias.to(module.weight.dtype), requires_grad=False)
                    else:
                        module.bias.copy_(bias.to(module.bias.dtype))
            if not had_bias:
                added_biases.append(name)
            diagnostics["requantized_weight_bits"] = module.weight_bits if isinstance(module, FakeQuantLinear) else None
            diagnostics["floating_projection"] = not isinstance(module, FakeQuantLinear)
            report["csrc"][name] = diagnostics
        report["added_fused_bias_layers"] = added_biases
        finish("csrc_fit_svd_fold_requantize", stamp)
        del teacher_moments, q_moments

        if matching_enabled:
            calibrate_ranges("post_csrc_activation_calibration")

            def measure_attention(stage):
                stamp = start()
                with AttentionStatistics(model) as statistics, torch.no_grad():
                    for step, (observation, actions) in enumerate(batches()):
                        if step >= config.quantvla_calibration_steps:
                            break
                        forward_batch(model, observation, actions, config.seed + step)
                    result = statistics.finalize()
                finish(stage, stamp)
                return result

            report["quantvla_matching"] = {"teacher": teacher_attention, "atm": {}, "ohb": {}}
            student = measure_attention("quantvla_attention_statistics")
            if config.atm_enabled:
                report["quantvla_matching"]["atm"] = fold_attention_matching(
                    model, teacher_attention, student, config, component="atm"
                )
            if config.ohb_enabled:
                if config.atm_enabled:
                    student = measure_attention("post_atm_output_statistics")
                report["quantvla_matching"]["ohb"] = fold_attention_matching(
                    model, teacher_attention, student, config, component="ohb"
                )
            report["quantvla_matching"]["folded_into_weights"] = True

        calibrate_ranges("final_activation_calibration")
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
