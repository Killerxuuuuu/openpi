"""Compare shared numerical primitives against the actual local CogACT file."""

import ast
from collections import defaultdict
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from openpi.quantization import cogact_primitives as vendored
from openpi.quantization import cogact_sampling
from openpi.quantization.cogact import ConditionStatistics
from openpi.quantization.cogact import allocate_channels
from openpi.quantization.cogact import fold_matching
from openpi.quantization.cogact import make_linear
from openpi.quantization.cogact import match_parameters
from openpi.quantization.cogact import reference_config
from openpi.quantization.config import PTQConfig
from openpi.quantization.libero_data import choose_cogact_indices
from openpi.quantization.numerics import normalize_prior
from openpi.quantization.numerics import quantize_weight
from openpi.quantization.numerics import structural_weights
from openpi.quantization.quantvla import AttentionStatistics
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


def test_vendored_definitions_are_the_actual_reference_not_reimplementations(reference):
    path = Path(reference.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == vendored.SOURCE_SHA256
    source = ast.parse(path.read_text(encoding="utf-8-sig"))
    copied = ast.parse(Path(vendored.__file__).read_text(encoding="utf-8"))
    definitions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef | ast.ClassDef)}
    for node in copied.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            assert ast.dump(node) == ast.dump(definitions[node.name]), node.name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale_mode", ["rtn", "mse"])
def test_complete_linear_rotation_adaptive_qdq_and_forward_match_source(reference, dtype, scale_mode):
    torch.manual_seed(110)
    source = nn.Linear(17, 11, dtype=dtype)
    config = PTQConfig(
        profile="cogact",
        weight_scale_mode=scale_mode,
        rotation_block_size=4,
        rotation_output_block_size=4,
        rotation_cache_enabled=False,
    )
    actual = make_linear(source, "paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj", config)
    expected = reference.PTQLinear.from_linear(
        source,
        weight_bits=2,
        act_bits=8,
        weight_scale_mode=scale_mode,
        pre_rotate_enable=True,
        pre_rotate_permute=True,
        pre_rotate_rin=True,
        pre_rotate_rout=True,
        pre_rotate_rout_mode="restore",
        pre_rotate_block_size=4,
        pre_rotate_block_out_size=4,
        pre_rotate_lambda_smooth=0.15,
        pre_rotate_svd_device="auto",
        pre_rotate_cache_enable=False,
    )
    torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
    bits = torch.tensor([16, 2, 2, 16, 2, 2, 2, 16, 2, 2, 2], dtype=torch.int16)
    for module in (actual, expected):
        module.set_adaptive_weight_bits_(bits)
        module.observe_acts = module.runtime_act_quant = True
    for _ in range(3):
        inputs = torch.randn(2, 5, 17).to(dtype)
        torch.testing.assert_close(actual(inputs), expected(inputs), rtol=0, atol=0)
    torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
    assert actual.weight_master.device.type == "cpu"


def test_channel_allocation_matches_source_threshold_ties_with_unequal_row_cost(reference):
    config = PTQConfig(profile="cogact", bf16_ratio=0.3, rotation_cache_enabled=False)
    model = nn.Module()
    model.first = make_linear(nn.Linear(5, 6), "expert", config)
    model.second = make_linear(nn.Linear(9, 4), "expert", config)
    old = nn.Module()
    for name, source in (("first", nn.Linear(5, 6)), ("second", nn.Linear(9, 4))):
        setattr(old, name, reference.PTQLinear.from_linear(source, module_name="action_model.net." + name))
    scores = {"first": torch.tensor([9, 8, 8, 1, 0, 0.0]), "second": torch.tensor([8, 2, 1, 0.0])}
    actual = allocate_channels(model, dict(model.named_children()), scores, config)
    expected = reference.apply_physical_sensitivity_mixed_precision(old, scores, reference_config(config))
    assert actual["high_bit_channels"] == expected["high_bit_channels"] == 4  # third-place tie retains four
    for name, module in model.named_children():
        torch.testing.assert_close(module.adaptive_weight_bits, old.get_submodule(name).adaptive_weight_bits)


@pytest.mark.parametrize("method", ["affine", "dense"])
@pytest.mark.parametrize("quantized", [False, True])
def test_csrc_fit_fold_and_master_requantization_match_actual_source(reference, method, quantized):
    torch.manual_seed(217)
    config = PTQConfig(
        profile="cogact", csrc_method=method, csrc_rank=3, csrc_group_size=3, rotation_cache_enabled=False
    )
    source = nn.Linear(7, 6)
    actual, expected = nn.Module(), nn.Module()
    if quantized:
        actual.projection = make_linear(source, "projection", config)
        expected.projection = reference.PTQLinear.from_linear(
            source,
            weight_bits=2,
            act_bits=8,
            weight_scale_mode="mse",
            pre_rotate_enable=True,
            pre_rotate_permute=True,
            pre_rotate_rin=True,
            pre_rotate_rout=True,
            pre_rotate_rout_mode="restore",
            pre_rotate_block_size=16,
            pre_rotate_block_out_size=16,
            pre_rotate_lambda_smooth=0.15,
            pre_rotate_svd_device="auto",
            pre_rotate_cache_enable=False,
        )
    else:
        actual.projection = copy.deepcopy(source)
        expected.projection = copy.deepcopy(source)
    teacher_values = torch.randn(40, 6) * torch.arange(1, 7) + 0.2
    student_values = teacher_values @ (torch.eye(6) + 0.15 * torch.randn(6, 6)) + 0.4

    def statistics(values):
        with ConditionStatistics({"projection": nn.Identity()}, dense=method == "dense") as collector:
            collector.targets["projection"](values[:17])
            collector.targets["projection"](values[17:])
            return collector.finalize()

    teacher, student = statistics(teacher_values), statistics(student_values)
    original_apply = (
        reference.apply_lvc_dense_calibration if method == "dense" else reference.apply_lvc_affine_calibration
    )
    copied_apply = vendored.apply_lvc_dense_calibration if method == "dense" else vendored.apply_lvc_affine_calibration
    expected_report = original_apply(expected, teacher, student, reference_config(config))
    actual_report = copied_apply(actual, teacher, student, reference_config(config))
    assert actual_report == expected_report
    assert actual_report["modules_applied"] == 1
    torch.testing.assert_close(actual.projection.weight, expected.projection.weight, rtol=0, atol=0)
    torch.testing.assert_close(actual.projection.bias, expected.projection.bias, rtol=0, atol=0)
    if quantized:
        torch.testing.assert_close(actual.projection.weight_master, expected.projection.weight_master, rtol=0, atol=0)


def test_ohb_inverse_and_atm_effective_logits_match_source(reference):
    model = nn.Module()
    model.attention = nn.Module()
    attention = model.attention
    attention.head_dim = 2
    attention.q_proj, attention.o_proj = nn.Linear(4, 16), nn.Linear(16, 4)
    teacher = {"attn_logit_std": {"attention": [1.4] * 8}, "ohb_rms": {"attention": 1.3}}
    student = {"attn_logit_std": {"attention": [1.0] * 8}, "ohb_rms": {"attention": 1.0}}
    config = PTQConfig(profile="cogact")
    alpha, beta = match_parameters(teacher, student, config)
    assert alpha == reference.apply_atm_calibration(None, teacher, student, reference_config(config))
    assert beta == reference.apply_ohb_calibration(None, teacher, student, reference_config(config))
    queries, keys, inputs = torch.randn(1, 3, 4), torch.randn(1, 2, 5, 2), torch.randn(1, 3, 16)
    old_q, old_o = attention.q_proj(queries), attention.o_proj(inputs)
    fold_matching(model, alpha, beta)
    torch.testing.assert_close(attention.o_proj(inputs), old_o / beta["attention"])
    scale = torch.tensor(alpha["attention"])[None, :, None, None].sqrt()
    q = old_q.reshape(1, 3, 8, 2).transpose(1, 2)
    k = keys.repeat_interleave(4, 1)
    expected_logits = (q * scale) @ (k * scale).transpose(-1, -2)
    actual_logits = attention.q_proj(queries).reshape(1, 3, 8, 2).transpose(1, 2) @ k.transpose(-1, -2)
    torch.testing.assert_close(actual_logits, expected_logits)


def test_attention_statistics_averages_each_call_like_source(reference):
    model = nn.Module()
    model.gemma_expert = nn.Module()
    model.gemma_expert.model = nn.Module()
    model.gemma_expert.model.layers = nn.ModuleList([nn.Module()])
    module = nn.Module()
    module.head_dim = 2
    module.o_proj = nn.Linear(4, 4)
    model.gemma_expert.model.layers[0].self_attn = module
    expected_std, expected_rms = [], []
    with AttentionStatistics(model, aggregation="cogact") as collector:
        for length, shift in ((3, 0), (5, 17)):
            q, k = torch.randn(2, 2, length, 2) + shift, torch.randn(2, 2, length, 2)
            module._quantvla_logits_observer(q, k, None, 0.5)  # noqa: SLF001
            output = module.o_proj(torch.randn(1, length, 4) + shift)
            expected_std.append(reference._compute_logits_std_per_head(q, k, 0.5))  # noqa: SLF001
            expected_rms.append(reference._compute_rms_scalar(output))  # noqa: SLF001
        stats = collector.finalize()["gemma_expert.model.layers.0.self_attn"]
    torch.testing.assert_close(torch.tensor(stats["logit_std"]), torch.stack(expected_std).mean(0))
    assert stats["output_rms"] == float(torch.stack(expected_rms).mean())


def test_action_xy_sampler_is_the_actual_entrypoint_algorithm(reference):
    path = Path(reference.__file__).parent.parent / "scripts/quantize_quantvla.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cogact_sampling.SOURCE_SHA256
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    copied = ast.parse(Path(cogact_sampling.__file__).read_text(encoding="utf-8"))
    names = {node.name for node in copied.body if isinstance(node, ast.FunctionDef)}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "torch": torch,
        "math": math,
        "defaultdict": defaultdict,
        "clone_batch_to_cpu": reference.clone_batch_to_cpu,
    }
    exec(
        compile(
            "from __future__ import annotations\n" + "\n\n".join(ast.unparse(node) for node in nodes),
            "original_sampler",
            "exec",
        ),
        namespace,
    )
    actions = np.random.default_rng(71).normal(size=(75, 7)).astype(np.float32)
    eligible = np.arange(len(actions))
    order = np.random.default_rng(11).permutation(eligible)
    payload = [{"actions": torch.tensor(actions[index : index + 1]), "frame_indices": [int(index)]} for index in order]
    expected, _ = namespace["_collect_spatial_balanced_calibration_batches"](payload, 19, 3, 12, 0, None)
    actual, _, _ = choose_cogact_indices(
        actions, eligible, samples=19, batch_size=1, seed=11, spatial_bins=3, warmup_samples=12
    )
    assert actual == [entry["frame_indices"][0] for entry in expected]


def test_original_rotation_cache_reuses_identical_weights_and_validates_changes(tmp_path):
    torch.manual_seed(99)
    source = nn.Linear(7, 5)
    config = PTQConfig(profile="cogact", rotation_cache_dir=str(tmp_path))
    first = make_linear(source, "cached_projection", config)
    second = make_linear(source, "cached_projection", config)
    assert not first.pre_rotate_cache_hit
    assert second.pre_rotate_cache_hit
    inputs = torch.randn(2, 7)
    torch.testing.assert_close(first(inputs), second(inputs), rtol=0, atol=0)
    with torch.no_grad():
        source.weight.add_(0.1)
    changed = make_linear(source, "cached_projection", config)
    assert not changed.pre_rotate_cache_hit
