"""Explicit supplementary-A.3 defaults and pi0.5 adaptation choices."""

from dataclasses import asdict
from dataclasses import dataclass
import math
import re
from typing import Literal

TEXT = r"paligemma_with_expert\.paligemma\.(?:model\.)?language_model\.layers\.\d+"
EXPERT = r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+"


def expert_block_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)gemma_expert\.model\.layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def protected_expert_tail_blocks(names, requested: int = 2) -> list[int]:
    """The final two action-expert blocks are mandatory in every profile/mode."""
    blocks = sorted({index for name in names if (index := expert_block_index(name)) is not None})
    return blocks[-max(2, requested) :]


@dataclass(frozen=True)
class PTQConfig:
    # Legacy artifacts remain loadable; new experiments share a QuantVLA base.
    profile: Literal["legacy", "quantvla", "cogact"] = "quantvla"
    mode: Literal["uniform", "quantvla", "da_ptq", "equal_weight", "random"] = "da_ptq"
    weight_bits: int = 2
    activation_bits: int = 8
    bf16_ratio: float = 0.30
    calibration_samples: int = 512
    batch_size: int = 1
    spatial_bins: int = 6
    spatial_warmup_samples: int = 128
    probe_steps: int = 16
    skip_first_expert_blocks: int | None = None
    skip_last_expert_blocks: int = 2  # Effective protection is always at least two full blocks.
    fallback_activation_bits: int | None = None
    activation_percentile: float = 99.9
    activation_calibration_batches: int | None = None
    quantvla_calibration_steps: int | None = None
    max_calibration_episodes_per_task: int = 5
    rotation_block_size: int | None = None
    rotation_output_block_size: int | None = None
    rotation_permute: bool = True
    row_rotation: Literal["restore", "off"] = "restore"
    rotation_device: Literal["auto", "cpu", "cuda"] = "auto"
    quantvla_smoothing: float = 0.15
    weight_scale_mode: Literal["mse", "rtn"] = "mse"
    atm_enabled: bool = True
    ohb_enabled: bool = True
    atm_log_clamp: float = 0.30
    ohb_log_clamp: float = 0.30
    matching_neutral_band: float | None = None
    damping: float = 3e-4
    planar_weight: float = 1.8
    control_weight: float = 0.15
    sensitivity_gain: float = 1.6
    physical_link_lengths: str = "0.35,0.30,0.25,0.20,0.15,0.10,0.08"
    action_dims: int = 7
    action_timestep: int = 0
    seed: int = 0
    allocation_seed: int = 0
    quant_regex: str = (
        rf"(?:{TEXT}|{EXPERT})\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
        r"|mlp\.(?:gate_proj|up_proj|down_proj))$"
    )
    rank_regex: str = rf"{EXPERT}\.mlp\.(?:gate_proj|up_proj|down_proj)$"
    # Correct conditioning K/V projections before RoPE/cache insertion. Output
    # widths are small; compensation never crosses RoPE, norms, or residuals.
    csrc_regex: str = rf"{TEXT}\.self_attn\.(?:k_proj|v_proj)$"
    csrc_enabled: bool = True
    csrc_rank: int = 8
    csrc_feature_weight: float = 1.0
    csrc_identity_weight: float = 8.0
    csrc_scale_min: float = 0.5
    csrc_scale_max: float = 2.0
    csrc_max_iter: int = 64
    csrc_max_dim: int = 1024
    csrc_tolerance: float = 1e-7
    max_tokens_per_call: int = 1024
    eps: float = 1e-6

    def __post_init__(self):
        for key, value in {
            "skip_first_expert_blocks": 0 if self.profile == "cogact" else 1,
            "fallback_activation_bits": 8 if self.profile == "cogact" else 16,
            "activation_calibration_batches": 64 if self.profile == "cogact" else 32,
            "rotation_block_size": 16 if self.profile == "cogact" else 64,
            "rotation_output_block_size": 16 if self.profile == "cogact" else 64,
            "quantvla_calibration_steps": (
                math.ceil(self.calibration_samples / max(1, self.batch_size)) if self.profile == "cogact" else 128
            ),
            "matching_neutral_band": 0.02 if self.profile == "cogact" else 0.03,
        }.items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)
        if self.profile not in {"legacy", "quantvla", "cogact"}:
            raise ValueError(f"Unknown quantization profile: {self.profile}")
        if self.mode not in {"uniform", "quantvla", "da_ptq", "equal_weight", "random"}:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.mode == "quantvla" and self.profile == "legacy":
            raise ValueError("The quantvla baseline requires --profile quantvla or cogact")
        if not 2 <= self.weight_bits <= 8 or not 2 <= self.activation_bits <= 8:
            raise ValueError("Fake quantization supports weight/activation bits in [2, 8].")
        if not 0 <= self.bf16_ratio <= 1:
            raise ValueError("bf16_ratio must be in [0, 1].")
        for key in (
            "calibration_samples",
            "batch_size",
            "spatial_bins",
            "spatial_warmup_samples",
            "probe_steps",
            "csrc_rank",
            "csrc_max_iter",
            "csrc_max_dim",
            "max_tokens_per_call",
            "activation_calibration_batches",
            "quantvla_calibration_steps",
            "rotation_block_size",
            "rotation_output_block_size",
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if min(self.skip_first_expert_blocks, self.skip_last_expert_blocks, self.action_timestep) < 0:
            raise ValueError("Block count and action timestep cannot be negative.")
        if self.max_calibration_episodes_per_task < 0:
            raise ValueError("Episode limit must be nonnegative (0 means unlimited)")
        if self.fallback_activation_bits not in {8, 16}:
            raise ValueError("Fallback activation precision must be 8 or 16")
        if not 0 < self.activation_percentile <= 100:
            raise ValueError("Activation percentile must be in (0, 100]")
        if not 0 <= self.quantvla_smoothing <= 1:
            raise ValueError("QuantVLA energy smoothing must be in [0, 1]")
        if self.row_rotation not in {"restore", "off"} or self.rotation_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid rotation mode/device")
        if self.weight_scale_mode not in {"mse", "rtn"}:
            raise ValueError("Unknown weight scale mode")
        if self.action_dims != 7:
            raise ValueError("The paper's structural prior is defined for seven action dimensions.")
        if not 0 < self.csrc_scale_min <= self.csrc_scale_max:
            raise ValueError("Invalid CSRC gain clipping bounds.")
        for key in (
            "damping",
            "planar_weight",
            "control_weight",
            "sensitivity_gain",
            "eps",
            "csrc_feature_weight",
            "csrc_identity_weight",
            "csrc_tolerance",
            "atm_log_clamp",
            "ohb_log_clamp",
            "matching_neutral_band",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if self.damping <= 0 or self.eps <= 0:
            raise ValueError("damping and eps must be positive")
        links = [float(value) for value in self.physical_link_lengths.split(",")]
        if len(links) != 7 or not all(math.isfinite(value) and value > 0 for value in links):
            raise ValueError("physical_link_lengths must contain seven finite positive values")
        for pattern in (self.quant_regex, self.rank_regex, self.csrc_regex):
            re.compile(pattern)

    @property
    def use_csrc(self) -> bool:
        return self.csrc_enabled and self.mode not in {"uniform", "quantvla"}

    @property
    def use_quantvla(self) -> bool:
        return self.profile in {"quantvla", "cogact"} and self.mode != "uniform"

    def to_dict(self) -> dict:
        return asdict(self)
