"""CogACT numerical recipe with a pi0.5 model/data interface.

The quantizer, rotations, allocation, compensation solvers and folding routines
come from the vendored, unmodified local reference. Only model names, the flow
loss, and the shared prefix-K/V attention boundary are adapted here.
"""
# ruff: noqa: SLF001 -- intentional calls into the vendored numerical reference

import contextlib
import logging
import time

import torch
from torch import nn

from openpi.quantization import cogact_primitives as reference
from openpi.quantization.calibration import forward_batch
from openpi.quantization.calibration import replace_module
from openpi.quantization.calibration import select_modules
from openpi.quantization.cogact_sampling import SOURCE_SHA256 as SAMPLING_SHA256
from openpi.quantization.config import PTQConfig
from openpi.quantization.config import expert_block_index
from openpi.quantization.config import protected_expert_tail_blocks
from openpi.quantization.numerics import weighted_flow_loss
from openpi.quantization.quantvla import AttentionStatistics

logger = logging.getLogger(__name__)
RECIPE = "cogact_reference_v3"


def reference_config(config: PTQConfig):
    """Map explicit experiment parameters to the actual CogACT configuration."""
    return reference.QuantVLAConfig(
        enable=True,
        weight_scale_mode=config.weight_scale_mode,
        act_percentile=config.activation_percentile,
        act_calib_batches=config.activation_calibration_batches,
        calibration_seed=config.seed,
        atm_enable=config.atm_enabled,
        ohb_enable=config.ohb_enabled,
        atm_neutral_threshold=config.matching_neutral_band,
        ohb_neutral_threshold=config.matching_neutral_band,
        ohb_log_clamp=config.ohb_log_clamp,
        lvc_affine_enable=config.use_csrc,
        lvc_affine_include_regex=config.csrc_regex,
        lvc_affine_dense_enable=config.csrc_method == "dense",
        lvc_affine_scale_min=config.csrc_scale_min,
        lvc_affine_scale_max=config.csrc_scale_max,
        lvc_affine_shift_max=config.csrc_shift_max,
        lvc_affine_neutral_scale_threshold=config.matching_neutral_band,
        lvc_affine_group_size=config.csrc_group_size,
        lvc_affine_group_shrink=config.csrc_group_shrink,
        lvc_affine_dense_ridge=config.csrc_dense_ridge,
        lvc_affine_dense_min_eig=config.csrc_dense_min_eig,
        lvc_affine_dense_max_dim=config.csrc_max_dim,
        lvc_affine_dense_weight_gain=config.csrc_dense_weight_gain,
        lvc_affine_dense_feature_lambda=config.csrc_feature_weight,
        lvc_affine_dense_identity_lambda=config.csrc_identity_weight,
        lvc_affine_dense_solver_rcond=config.csrc_dense_solver_rcond,
        lvc_affine_dense_structure=config.csrc_dense_structure,
        lvc_affine_dense_block_size=config.csrc_dense_block_size,
        lvc_affine_dense_rank=config.csrc_rank,
        lvc_affine_dense_delta_clip=config.csrc_dense_delta_clip,
        lvc_affine_dense_fp16_bypass_enable=config.csrc_bypass_enabled,
        lvc_affine_dense_fp16_bypass_rank=config.csrc_rank,
        physical_sensitivity_enable=config.mode in {"da_ptq", "equal_weight", "random"},
        physical_sensitivity_scope="dit_mlp",
        physical_sensitivity_high_bit_ratio=config.bf16_ratio,
        physical_sensitivity_high_weight_bits=16,
        physical_sensitivity_low_weight_bits=config.weight_bits,
        physical_sensitivity_action_timestep=config.action_timestep,
        physical_sensitivity_fk_link_lengths=config.physical_link_lengths,
        physical_sensitivity_damping=config.damping,
        physical_sensitivity_translation_weight=config.planar_weight,
        physical_sensitivity_rotation_weight=config.control_weight,
        physical_sensitivity_gain=config.sensitivity_gain,
        eps=config.eps,
    )


class CogACTLinear(reference.PTQLinear):
    """Unmodified reference execution, with pi0.5 deployment metadata aliases."""

    @property
    def activation_bits(self):
        return self.act_bits

    @property
    def activation_absmax(self):
        return self.act_channel_absmax


def make_linear(source, name, config, *, initialize=True):
    options = {
        "weight_bits": config.weight_bits,
        "act_bits": config.activation_bits,
        "weight_scale_mode": config.weight_scale_mode,
        "act_percentile": config.activation_percentile,
        "act_calib_batches": config.activation_calibration_batches,
        "pre_rotate_enable": config.use_quantvla,
        "pre_rotate_permute": config.rotation_permute,
        "pre_rotate_rin": True,
        "pre_rotate_rout": config.row_rotation == "restore",
        "pre_rotate_rout_mode": config.row_rotation,
        "pre_rotate_block_size": config.rotation_block_size,
        "pre_rotate_block_out_size": config.rotation_output_block_size,
        "pre_rotate_lambda_smooth": config.quantvla_smoothing,
        "pre_rotate_svd_device": config.rotation_device,
        # The original signed-weight/config-validated cache avoids repeating
        # expensive full SVDs for every allocation variant. Deployment uses the
        # artifact's buffers and never reads this calibration cache.
        "pre_rotate_cache_enable": config.rotation_cache_enabled,
        "pre_rotate_cache_dir": config.rotation_cache_dir,
        "module_name": "action_model.net." + name if expert_block_index(name) is not None else "llm." + name,
    }
    if initialize:
        return CogACTLinear.from_linear(source, **options)
    # Do not initialize rotations or re-QDQ saved weights at inference.
    result = CogACTLinear(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
        **options,
    )
    with torch.no_grad():
        result.weight.copy_(source.weight)
        if result.bias is not None:
            result.bias.copy_(source.bias)
    # No inference operation uses the CPU masters. Release them before moving
    # the policy to CUDA; inference never requantizes deployment weights.
    result._buffers["weight_master"] = torch.empty(0)
    if result.bias_master is not None:
        result._buffers["bias_master"] = torch.empty(0)
    return result.requires_grad_(requires_grad=False)


def deployment_buffers(module):
    return {
        name: value.detach().cpu().contiguous()
        for name, value in module.named_buffers(recurse=False)
        if name not in {"weight_master", "bias_master"}
    }


def restore_buffers(module, buffers, *, strides=None):
    """Restore variable-size SVD/bypass buffers without running calibration."""
    for name, saved_value in buffers.items():
        # Variable-shaped rotation/bypass tensors cannot use load_state_dict's
        # shape checks against the empty constructor buffers.
        value = saved_value.to(module.weight.device)
        if strides and tuple(strides[name]) != value.stride():
            restored = torch.empty_strided(value.shape, strides[name], device=value.device, dtype=value.dtype)
            restored.copy_(value)
            value = restored
        module._buffers[name] = value
    module._rin_block_indices = sorted(
        int(name.removeprefix("pre_rotate_rin_")) for name in buffers if name.startswith("pre_rotate_rin_")
    )
    module._rout_block_indices = sorted(
        int(name.removeprefix("pre_rotate_rout_")) for name in buffers if name.startswith("pre_rotate_rout_")
    )
    module.observe_acts = False
    module.runtime_act_quant = True


class ConditionStatistics:
    """Reference float32 per-call moments, at the pi0.5 condition boundary."""

    def __init__(self, targets, *, dense):
        self.targets, self.dense = targets, dense
        self.payload = {}
        self.handles = []

    def __enter__(self):
        for name, module in self.targets.items():

            def capture(_module, _args, output, key=name):
                values = output.detach().float().reshape(-1, output.shape[-1]).cpu()
                count = len(values)
                if not count or not torch.isfinite(values).all():
                    raise FloatingPointError(f"Empty/nonfinite condition statistics at {key}")
                sums, squares = values.sum(0), values.square().sum(0)
                if key not in self.payload:
                    self.payload[key] = {
                        "sum": torch.zeros_like(sums),
                        "squares": torch.zeros_like(squares),
                        "count": 0,
                    }
                    if self.dense:
                        self.payload[key]["xtx"] = torch.zeros(values.shape[-1], values.shape[-1])
                entry = self.payload[key]
                entry["sum"] += sums
                entry["squares"] += squares
                entry["count"] += count
                if self.dense:
                    entry["xtx"] += values.T @ values

            self.handles.append(module.register_forward_hook(capture))
        return self

    def finalize(self):
        if set(self.payload) != set(self.targets):
            raise RuntimeError("Some compensation projections received no observations")
        moments, second = {}, {}
        for name, entry in self.payload.items():
            mean = entry["sum"] / entry["count"]
            variance = (entry["squares"] / entry["count"] - mean.square()).clamp_min(0)
            moments[name] = {"mean": mean.tolist(), "std": variance.sqrt().tolist(), "count": entry["count"]}
            if self.dense:
                second[name] = {key: entry[key] for key in ("sum", "xtx", "count")}
        return {"lvc_proj_moments": moments, "lvc_proj_second_order": second}

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def match_parameters(teacher, student, config):
    source_config = reference_config(config)
    alpha = reference.apply_atm_calibration(None, teacher, student, source_config)
    beta = reference.apply_ohb_calibration(None, teacher, student, source_config)
    return alpha, beta


@torch.no_grad()
def fold_matching(model, alpha, beta):
    """Reference logit multiplier and inverse-beta OHB, no runtime observer.

    For pi0.5 GQA, Q*=alpha equals Q*=sqrt(alpha), K*=sqrt(alpha)
    per query head. Scaling shared K weights would also change language queries
    and cannot represent different multipliers within one KV group.
    """
    for name, values in alpha.items():
        module = model.get_submodule(name)
        gain = torch.as_tensor(values, dtype=torch.float32).repeat_interleave(module.head_dim)
        if gain.numel() != module.q_proj.out_features:
            raise ValueError(f"Attention head mismatch at {name}")
        module.q_proj.weight.mul_(gain.to(module.q_proj.weight)[:, None])
        if module.q_proj.bias is not None:
            module.q_proj.bias.mul_(gain.to(module.q_proj.bias))
    for name, value in beta.items():
        module = model.get_submodule(name).o_proj
        inverse = 1.0 / value
        module.weight.mul_(inverse)
        if module.bias is not None:
            module.bias.mul_(inverse)


def allocate_channels(model, candidates, scores, config):
    """Use the exact global threshold/tie and per-row RTN reference path."""
    if config.mode in {"uniform", "quantvla"}:
        return {}
    if config.mode == "random":
        generator = torch.Generator().manual_seed(config.allocation_seed)
        # Distinct random priorities give exactly ceil(ratio*N) high rows.
        counts = [module.out_features for module in candidates.values()]
        priorities = torch.randperm(sum(counts), generator=generator).float()
        scores = dict(zip(candidates, priorities.split(counts), strict=True))
    if set(scores) != set(candidates) or any(not torch.isfinite(value).all() for value in scores.values()):
        raise ValueError("Missing/nonfinite output-channel sensitivity scores")
    # Scope comes from the selected candidate set, including user regexes.
    # This also prevents a broad scope from touching fixed boundary guards.
    original_names = {
        name: module.module_name for name, module in model.named_modules() if isinstance(module, CogACTLinear)
    }
    try:
        for name in original_names:
            model.get_submodule(name).module_name = "action_model.net." + name if name in candidates else "llm." + name
        result = reference.apply_physical_sensitivity_mixed_precision(model, scores, reference_config(config))
    finally:
        for name, value in original_names.items():
            model.get_submodule(name).module_name = value
    high_parameters = sum(
        int((module.adaptive_weight_bits == 16).sum()) * module.in_features for module in candidates.values()
    )
    result["high_parameter_fraction"] = high_parameters / max(
        1, sum(module.weight.numel() for module in candidates.values())
    )
    result["allocation_unit"] = "output_channel"
    result["tie_policy"] = "score_greater_equal_kth_threshold; may_exceed_requested_count"
    return result


def calibrate_cogact(model, batches, config):
    """Teacher -> W-QDQ probe -> channel assignment -> student -> folds -> A8."""
    quant, candidates, targets, protected = select_modules(model, config)
    original_keys = list(model.state_dict())
    original_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    initial_training = model.training
    model.eval().requires_grad_(requires_grad=False)
    source_config = reference_config(config)
    report = {
        "format_version": 3,
        "implementation": "floating_QDQ_not_packed_integer",
        "recipe": RECIPE,
        "reference_source_sha256": reference.SOURCE_SHA256,
        "reference_sampling_sha256": SAMPLING_SHA256,
        "config": config.to_dict(),
        "original_state_keys": original_keys,
        "protected_layers": protected,
        "mandatory_expert_tail_blocks": protected_expert_tail_blocks(
            dict(model.named_modules()), config.skip_last_expert_blocks
        ),
        "candidate_layers": sorted(candidates),
        "adaptation": {
            "allocation_unit": "output_channel",
            "sensitivity_probe": "weight_QDQ_student; A_quant_off; per_batch_prior; seed_plus_200000",
            "adaptive_weight_scale": "reference_per_row_RTN_even_when_general_scale_is_MSE",
            "rotation": "reference_SVD_source_dtype; CPU_masters; source_restore_bias_convention",
            "attention_statistics": "mean_of_per_forward_std_and_RMS; valid_pi05_attention_entries",
            "matching": "shared_pre_fold_student_statistics; reference_alpha_and_inverse_beta",
            "atm_adapter": "Q_times_alpha_equivalent_to_sqrt_alpha_on_Q_and_repeated_K; shared_prefix_GQA",
            "csrc_interface": "prefix_KV_before_RoPE; pi05_has_no_separate_z_embedder",
            "csrc_solver": "reference_" + config.csrc_method,
            "csrc_enablement": "explicit_experiment_setting; local_CogACT_entrypoint_defaults_off",
            "bias_adapter": "zero_bias_added_before_statistics_when_required_by_biasless_pi05_KV",
            "loss": "pi05_flow_matching_first_seven_channels; not_CogACT_diffusion_epsilon_loss",
            "sampling": "action_XY_bin_quota; seeded_LeRobot_frame_stream; not_Bridge_RLDS_stream",
            "calibration_unit": "sampled_frame_with_action_chunk",
        },
        "floating_csrc_layers": {
            name: {"original_bias": module.bias is not None} for name, module in targets.items() if name not in quant
        },
        "layers": {},
        "csrc": {},
        "timings": {},
        "stage_order": [],
    }
    device = next(model.parameters()).device
    float_dtype = next(parameter.dtype for parameter in model.parameters() if parameter.is_floating_point())
    use_autocast = device.type == "cuda" and float_dtype in (torch.float16, torch.bfloat16)

    def forward(observation, actions, seed):
        with torch.autocast(device_type="cuda", dtype=float_dtype, enabled=use_autocast):
            return forward_batch(model, observation, actions, seed)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @contextlib.contextmanager
    def stage(name):
        sync()
        start = time.perf_counter()
        logger.info("START %s", name)
        report["stage_order"].append(name)
        yield
        sync()
        report["timings"][name] = {"seconds": time.perf_counter() - start}
        logger.info("DONE %s (%.2fs)", name, report["timings"][name]["seconds"])

    # The source z_embedder has a bias. Give biasless pi0.5 projections an
    # exactly-zero bias before fitting, so the unchanged source folding routine
    # can represent its shift. Exclude these added parameters from base weights.
    added_biases = []
    for name, module in targets.items():
        if module.bias is None:
            module.bias = nn.Parameter(module.weight.new_zeros(module.out_features), requires_grad=False)
            added_biases.append(name)
    report["added_fused_bias_layers"] = added_biases

    def stat_pass(label, *, observe, collect=True):
        for module in wrapped.values():
            module.observe_acts = observe
            module.runtime_act_quant = observe
            if observe:
                module.reset_activation_observer_()
        attention_enabled = config.use_quantvla and (config.atm_enabled or config.ohb_enabled)
        attention_context = (
            AttentionStatistics(model, aggregation="cogact")
            if collect and attention_enabled
            else contextlib.nullcontext()
        )
        condition_context = (
            ConditionStatistics(targets, dense=config.csrc_method == "dense") if collect else contextlib.nullcontext()
        )
        loss_sum, count = 0.0, 0
        with stage(label), torch.no_grad(), attention_context as attn, condition_context as condition:
            for step, (observation, actions) in enumerate(batches()):
                if step >= config.quantvla_calibration_steps:
                    break
                losses = forward(observation, actions, config.seed + step)
                loss = weighted_flow_loss(losses, torch.ones(7))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {label} loss at batch {step}")
                loss_sum += float(loss) * len(actions)
                count += len(actions)
                if step % 16 == 0:
                    logger.info("%s batch %d", label, step)
            if count == 0:
                raise ValueError("Calibration set is empty")
            stats = condition.finalize() if collect else {}
            summary = attn.finalize() if collect and attention_enabled else {}
            stats["attn_logit_std"] = {name: value["logit_std"] for name, value in summary.items()}
            stats["ohb_rms"] = {name: value["output_rms"] for name, value in summary.items()}
        for module in wrapped.values():
            module.observe_acts = module.runtime_act_quant = False
        return stats, loss_sum / count, count

    wrapped = {}
    try:
        teacher_stats, teacher_loss, count = stat_pass("teacher_statistics", observe=False)
        report.update(teacher_flow_mse=teacher_loss, calibration_samples=count)
        with stage("weight_QDQ_wrapping"):
            for index, name in enumerate(list(quant), start=1):
                logger.info("W-QDQ/SVD layer %d/%d: %s", index, len(report["layers"]) + len(quant), name)
                source = quant.pop(name)
                original_bias = name + ".bias" in original_keys
                replacement = make_linear(source, name, config)
                logger.info("W-QDQ ready: %s; rotation_cache_hit=%s", name, replacement.pre_rotate_cache_hit)
                replace_module(model, name, replacement)
                wrapped[name] = replacement
                if name in targets:
                    targets[name] = replacement
                if name in candidates:
                    candidates[name] = replacement
                report["layers"][name] = {
                    "wrapper": "cogact",
                    "weight_bits": config.weight_bits,
                    "activation_bits": config.activation_bits,
                    "original_bias": original_bias,
                    "rotation_cache_hit": replacement.pre_rotate_cache_hit,
                }
                del source
        scores = {name: torch.zeros(module.out_features) for name, module in candidates.items()}
        counts = dict.fromkeys(candidates, 0)
        probe_count = 0
        weights_sum = torch.zeros(7)
        if config.mode in {"da_ptq", "equal_weight"}:
            with stage("weight_QDQ_student_channel_probe"):
                for module in candidates.values():
                    module.weight.requires_grad_(requires_grad=True)
                    if module.bias is not None:
                        module.bias.requires_grad_(requires_grad=True)
                try:
                    for step, (observation, actions) in enumerate(batches()):
                        if step >= config.probe_steps:
                            break
                        weights = (
                            reference._build_physical_channel_weights({"actions": actions[..., :7]}, source_config)
                            if config.mode == "da_ptq"
                            else torch.ones(7)
                        )
                        model.zero_grad(set_to_none=True)
                        losses = forward(observation, actions, config.seed + 200000 + step)
                        loss = weighted_flow_loss(losses, weights)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite sensitivity probe loss")
                        loss.backward()
                        for name, module in candidates.items():
                            if module.weight.grad is not None:
                                scores[name] += module.weight.grad.detach().abs().float().mean(1).cpu()
                                counts[name] += 1
                        weights_sum += weights.cpu()
                        probe_count += 1
                        logger.info("Channel sensitivity probe %d/%d", step + 1, config.probe_steps)
                finally:
                    model.zero_grad(set_to_none=True)
                    for module in candidates.values():
                        module.requires_grad_(requires_grad=False)
                if any(count == 0 for count in counts.values()):
                    raise RuntimeError("Some candidate layers received no sensitivity gradient")
                scores = {name: value / counts[name] for name, value in scores.items()}
        report["probe_batches"] = probe_count
        report["loss_weights"] = (weights_sum / max(1, probe_count)).tolist() if probe_count else [1.0] * 7
        report["loss_weight_scope"] = "per_batch"
        report["gradient_scores"] = {name: value.tolist() for name, value in scores.items()} if probe_count else {}
        with stage("global_channel_allocation"):
            allocation = allocate_channels(model, candidates, scores, config)
        report["channel_allocation"] = allocation
        report["high_precision_channels"] = {
            name: torch.where(module.adaptive_weight_bits.cpu() == 16)[0].tolist()
            for name, module in candidates.items()
            if bool(module.adaptive_weight_bits_enabled)
        }
        report["high_precision_layers"] = [
            name
            for name, rows in report["high_precision_channels"].items()
            if len(rows) == candidates[name].out_features
        ]
        report["candidate_high_fraction"] = allocation.get("high_ratio_actual", 0.0)
        for name, module in candidates.items():
            if bool(module.adaptive_weight_bits_enabled):
                report["layers"][name]["adaptive_weight_bits"] = True
                report["layers"][name]["adaptive_low_bits"] = config.weight_bits
                report["layers"][name]["adaptive_high_bits"] = 16

        student_stats, student_loss, _ = stat_pass("student_statistics_with_A8_observation", observe=True)
        report["quantized_flow_mse_before_csrc"] = student_loss
        with stage("ATM_OHB_from_same_student_statistics"):
            alpha, beta = match_parameters(teacher_stats, student_stats, config) if config.use_quantvla else ({}, {})
            fold_matching(model, alpha, beta)
        report["quantvla_matching"] = {
            "teacher": {key: teacher_stats[key] for key in ("attn_logit_std", "ohb_rms")},
            "student": {key: student_stats[key] for key in ("attn_logit_std", "ohb_rms")},
            "atm_alpha": alpha,
            "ohb_beta": beta,
            "ohb_fold_direction": "inverse_beta",
            "folded_into_weights": True,
        }
        with stage("reference_CSRC_" + config.csrc_method):
            if config.use_csrc:
                apply = (
                    reference.apply_lvc_dense_calibration
                    if config.csrc_method == "dense"
                    else reference.apply_lvc_affine_calibration
                )
                compensation = apply(model, teacher_stats, student_stats, source_config)
                if compensation["modules_applied"] != len(targets):
                    raise RuntimeError(f"Incomplete CSRC folding: {compensation}; expected {len(targets)} targets")
                report["csrc"] = compensation
        # Match the source's final activation-only pass after all folds.
        _, final_loss, _ = stat_pass("final_activation_calibration", observe=True, collect=False)
        report["final_flow_mse_probe_subset"] = final_loss
        report["activation_calibration"] = {
            "batches_per_layer": {name: int(module.act_obs_batches) for name, module in wrapped.items()},
            "runtime_fallback": "reference_dynamic_absmax_until_first_observed_channel_scale",
        }
        for name, module in wrapped.items():
            if not bool(module.act_scale_ready) or not torch.isfinite(module.act_channel_absmax).all():
                raise RuntimeError(f"Invalid/missing calibrated activation scales: {name}")
            module.observe_acts = False
            module.runtime_act_quant = True
            report["layers"][name]["deployment_buffers"] = sorted(deployment_buffers(module))
            # SVD returns column-major bases. Preserve their strides as well as
            # values so a reload selects the same floating matmul path.
            report["layers"][name]["buffer_strides"] = {
                key: list(value.stride())
                for key, value in module.named_buffers(recurse=False)
                if key not in {"weight_master", "bias_master"}
            }
        return report
    finally:
        model.zero_grad(set_to_none=True)
        model.train(initial_training)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_flags.get(name, False))
