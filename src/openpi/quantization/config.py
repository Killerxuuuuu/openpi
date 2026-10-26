"""Explicit supplementary-A.3 defaults and pi0.5 adaptation choices."""

from dataclasses import asdict
from dataclasses import dataclass
import math
import re
from typing import Literal

TEXT = r"paligemma_with_expert\.paligemma\.(?:model\.)?language_model\.layers\.\d+"
EXPERT = r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+"


@dataclass(frozen=True)
class PTQConfig:
    # uniform = W2A8 only; the other modes all retain CSRC and the same BF16 budget.
    mode: Literal["uniform", "da_ptq", "equal_weight", "random"] = "da_ptq"
    weight_bits: int = 2
    activation_bits: int = 8
    bf16_ratio: float = 0.30
    calibration_samples: int = 512
    batch_size: int = 1
    spatial_bins: int = 6
    spatial_warmup_samples: int = 128
    probe_steps: int = 16
    skip_last_expert_blocks: int = 2
    damping: float = 3e-4
    planar_weight: float = 1.8
    control_weight: float = 0.15
    sensitivity_gain: float = 1.6
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
        if self.mode not in {"uniform", "da_ptq", "equal_weight", "random"}:
            raise ValueError(f"Unknown mode: {self.mode}")
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
        ):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.skip_last_expert_blocks < 0 or self.action_timestep < 0:
            raise ValueError("Block count and action timestep cannot be negative.")
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
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key}")
        if self.damping <= 0 or self.eps <= 0:
            raise ValueError("damping and eps must be positive")
        for pattern in (self.quant_regex, self.rank_regex, self.csrc_regex):
            re.compile(pattern)

    @property
    def use_csrc(self) -> bool:
        return self.csrc_enabled and self.mode != "uniform"

    def to_dict(self) -> dict:
        return asdict(self)
