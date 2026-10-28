"""Compare shared numerical primitives against the actual local CogACT file."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from openpi.quantization.config import PTQConfig
from openpi.quantization.numerics import normalize_prior
from openpi.quantization.numerics import quantize_weight
from openpi.quantization.numerics import structural_weights
from openpi.quantization.quantvla import QuantVLALinear
from openpi.quantization.quantvla import fold_attention_matching


@pytest.fixture(scope="module")
def reference():
    path = Path(__file__).resolve().parents[3] / "DA-PTQ/vla/quantvla_ptq.py"
    if not path.is_file():
        pytest.skip("Local CogACT reference checkout is not installed")
    spec = importlib.util.spec_from_file_location("cogact_reference_for_parity_tests", path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    yield result
    del sys.modules[spec.name]


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("mode", ["mse", "rtn"])
def test_shared_weight_qdq_matches_local_cogact(reference, bits, mode):
    torch.manual_seed(56)
    weight = torch.randn(15, 33)
    weight[0].zero_()
    if mode == "mse":
        scales = reference._compute_mse_scales_per_outchannel(weight, bits)  # noqa: SLF001
        limit = 2 ** (bits - 1) - 1
        expected = (weight / scales[:, None]).clamp(-limit - 1, limit).round() * scales[:, None]
    else:
        expected = reference._symmetric_quant_dequant(weight, bits, per_channel=True)  # noqa: SLF001
    torch.testing.assert_close(quantize_weight(weight, bits, scale_mode=mode, reciprocal_grid=True), expected)


def test_cogact_observer_matches_reference_including_sample_limit(reference):
    torch.manual_seed(71)
    source = nn.Linear(4, 3, bias=False)
    old = reference.PTQLinear.from_linear(source, pre_rotate_enable=False, pre_rotate_cache_enable=False)
    old.observe_acts = True
    new = QuantVLALinear(source, 16, 8, torch.ones(4), PTQConfig(profile="cogact"), initialize=False)
    new.reset_observer()
    new.activation_quant_enabled = False
    for _ in range(2):
        values = torch.randn(4200, 4)
        values[4096:] = 1e5  # Both observers intentionally use only the first 4096 rows.
        old(values)
        new(values)
    torch.testing.assert_close(new.observed_absmax, old.act_channel_absmax, rtol=0, atol=0)


def test_cogact_profile_resolves_defaults_without_changing_old_profile():
    aligned, old = PTQConfig(profile="cogact"), PTQConfig()
    assert aligned.rotation_block_size == aligned.rotation_output_block_size == 16
    assert aligned.activation_calibration_batches == 64
    assert aligned.skip_first_expert_blocks == 0
    assert aligned.fallback_activation_bits == 8
    assert aligned.quantvla_calibration_steps == 512
    assert aligned.matching_neutral_band == 0.02
    assert old.rotation_block_size == 64
    assert old.activation_calibration_batches == 32
    assert old.fallback_activation_bits == 16


def test_cogact_structural_weights_match_local_per_batch_formula(reference):
    torch.manual_seed(19)
    actions = torch.randn(3, 2, 7)
    config = PTQConfig(profile="cogact")
    source_config = reference.QuantVLAConfig(
        physical_sensitivity_damping=config.damping,
        physical_sensitivity_translation_weight=config.planar_weight,
        physical_sensitivity_rotation_weight=config.control_weight,
        physical_sensitivity_gain=config.sensitivity_gain,
        physical_sensitivity_fk_link_lengths=config.physical_link_lengths,
    )
    expected = reference._build_physical_channel_weights({"actions": actions}, source_config)  # noqa: SLF001
    _, actual = normalize_prior(structural_weights(actions, config), config)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_cogact_atm_clipping_matches_local_formula(reference):
    model = nn.Module()
    model.attention = nn.Module()
    model.attention.head_dim = 2
    model.attention.q_proj = nn.Linear(4, 16)
    teacher_std = [0, 0.2, 0.99, 1.01, 1.03, 2, 3, 10]
    student_std = [1.0] * 8
    expected = reference.apply_atm_calibration(
        None,
        {"attn_logit_std": {"attention": teacher_std}},
        {"attn_logit_std": {"attention": student_std}},
        reference.QuantVLAConfig(),
    )["attention"]
    actual = fold_attention_matching(
        model,
        {"attention": {"logit_std": teacher_std}},
        {"attention": {"logit_std": student_std}},
        PTQConfig(profile="cogact"),
        component="atm",
    )["attention"]
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=0, atol=0)
