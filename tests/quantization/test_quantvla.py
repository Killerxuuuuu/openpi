"""Numerical and actual joint-Gemma adapter checks; no full checkpoint needed."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from safetensors.torch import load_model
from test_da_ptq import MLP
import torch
from torch import nn

from openpi.quantization.calibration import calibrate
from openpi.quantization.calibration import select_modules
from openpi.quantization.checkpoint import restore_quantization
from openpi.quantization.checkpoint import save_checkpoint
from openpi.quantization.config import PTQConfig
from openpi.quantization.libero_data import eligible_frames
from openpi.quantization.quantvla import AttentionStatistics
from openpi.quantization.quantvla import QuantVLALinear
from openpi.quantization.quantvla import fold_attention_matching
from openpi.quantization.quantvla import matching_gain


@pytest.mark.parametrize("row_rotation", ["restore", "off"])
def test_rotation_and_csrc_preserve_float_function_including_tail(row_rotation):
    torch.manual_seed(55)
    config = PTQConfig(rotation_block_size=4, rotation_output_block_size=4, row_rotation=row_rotation)
    source = nn.Linear(7, 5)
    module = QuantVLALinear(source, 16, 8, torch.ones(7), config)
    module.activation_quant_enabled = False
    inputs = torch.randn(2, 3, 7)
    expected = source(inputs)
    torch.testing.assert_close(module(inputs), expected, atol=2e-6, rtol=2e-5)
    correction = torch.eye(5) + torch.randn(5, 5) * 0.1
    shift = torch.randn(5)
    module.fold_affine(correction, shift)
    torch.testing.assert_close(module(inputs), expected @ correction.T + shift, atol=3e-6, rtol=3e-5)


def test_fallback_bypasses_weight_activation_and_rotations():
    source = nn.Linear(7, 5)
    module = QuantVLALinear(source, 16, 16, torch.ones(7) * 1e-6, PTQConfig())
    values = torch.randn(2, 7)
    assert not module.transform_enabled
    torch.testing.assert_close(module(values), source(values), rtol=0, atol=0)


@pytest.fixture
def flow_model(monkeypatch):
    # Import the real openpi joint attention implementation. Only construction
    # and the HF math primitives are replaced with small Torch equivalents.
    hf = types.ModuleType("transformers")
    hf.GemmaForCausalLM = hf.PaliGemmaForConditionalGeneration = object
    auto = types.ModuleType("transformers.models.auto")
    auto.CONFIG_MAPPING = {}
    gemma = types.ModuleType("transformers.models.gemma")
    primitives = types.SimpleNamespace()

    def rotary(query, key, cos, sin, **kwargs):
        def apply(value):
            rotated = torch.stack([-value[..., 1], value[..., 0]], dim=-1)
            return value * cos[:, None] + rotated * sin[:, None]

        return apply(query), apply(key)

    def attention(module, query, key, value, mask, scaling):
        repeats = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(repeats, 1), value.repeat_interleave(repeats, 1)
        logits = query @ key.transpose(-1, -2) * scaling
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9) if mask.dtype == torch.bool else logits + mask
        output = (logits.softmax(-1) @ value).transpose(1, 2).contiguous()
        return output, None

    primitives.apply_rotary_pos_emb = rotary
    primitives.eager_attention_forward = attention
    primitives._gated_residual = lambda residual, value, gate: residual + value  # noqa: SLF001
    gemma.modeling_gemma = primitives
    for name, module in (
        ("transformers", hf),
        ("transformers.models.auto", auto),
        ("transformers.models.gemma", gemma),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[2] / "src/openpi/models_pytorch/gemma_pytorch.py"
    spec = importlib.util.spec_from_file_location("actual_joint_gemma_quantvla_test", source)
    actual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)

    class Norm(nn.Module):
        def forward(self, values, cond=None):
            return values, None

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.head_dim = 2
            self.self_attn.scaling = 2**-0.5
            for name, width in (("q_proj", 16), ("k_proj", 4), ("v_proj", 4), ("o_proj", 16)):
                setattr(self.self_attn, name, nn.Linear(16 if name != "o_proj" else width, width))
            self.mlp = MLP()
            self.mlp.gate_proj = nn.Linear(16, 32, bias=False)
            self.mlp.up_proj = nn.Linear(16, 32, bias=False)
            self.mlp.down_proj = nn.Linear(32, 16, bias=False)
            self.input_layernorm, self.post_attention_layernorm = Norm(), Norm()

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(4)])
            self.norm = Norm()

        def rotary_emb(self, dummy, positions):
            angle = positions.float()[..., None] * 0.2
            return angle.cos(), angle.sin()

    class Flow(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = actual.PaliGemmaWithExpertModel.__new__(actual.PaliGemmaWithExpertModel)
            backend = self.paligemma_with_expert
            nn.Module.__init__(backend)
            backend.paligemma = nn.Module()
            backend.paligemma.language_model = Language()
            backend.paligemma.model = types.SimpleNamespace(language_model=backend.paligemma.language_model)
            backend.paligemma.config = types.SimpleNamespace(text_config=types.SimpleNamespace(num_hidden_layers=4))
            backend.gemma_expert = nn.Module()
            backend.gemma_expert.model = Language()
            self.action_in_proj = nn.Linear(9, 16)
            self.action_out_proj = nn.Linear(16, 9)

        def sample_noise(self, shape, device):
            return torch.randn(shape, device=device)

        def sample_time(self, size, device):
            return torch.rand(size, device=device)

        def embed_prefix(self, observation):
            mask = torch.ones(observation.shape[:2], dtype=torch.bool)
            mask[:, -1] = False
            return observation, mask, None

        def forward(self, observation, actions, noise=None, time=None, *, train=False):
            prefix, valid, _ = self.embed_prefix(observation)
            suffix = self.action_in_proj(time[:, None, None] * noise + (1 - time[:, None, None]) * actions)
            length = prefix.shape[1] + suffix.shape[1]
            mask = torch.ones(len(actions), 1, length, length, dtype=torch.bool)
            mask[..., : prefix.shape[1], prefix.shape[1] :] = False
            mask[..., :, prefix.shape[1] - 1] = False
            outputs, _ = self.paligemma_with_expert(
                inputs_embeds=[prefix, suffix],
                attention_mask=mask,
                position_ids=torch.arange(length)[None].expand(len(actions), -1),
            )
            return (self.action_out_proj(outputs[1]) - (noise - actions)).square()

    return Flow


def replay():
    generator = torch.Generator().manual_seed(41)
    for _ in range(3):
        yield torch.randn(1, 4, 16, generator=generator), torch.randn(1, 2, 9, generator=generator)


@pytest.mark.parametrize("mode", ["quantvla", "da_ptq", "equal_weight", "random"])
@pytest.mark.parametrize("profile", ["quantvla", "cogact"])
@pytest.mark.parametrize("csrc_method", ["affine", "dense"])
def test_actual_joint_adapter_calibration_export_reload(flow_model, tmp_path, mode, profile, csrc_method):
    torch.manual_seed(5)

    def build():
        result = flow_model().eval()
        if profile == "cogact":
            for layer in result.paligemma_with_expert.paligemma.language_model.layers:
                layer.self_attn.k_proj.bias = None
                layer.self_attn.v_proj.bias = None
        return result

    model = build()
    config = PTQConfig(
        profile=profile,
        csrc_method=csrc_method,
        mode=mode,
        calibration_samples=3,
        probe_steps=2,
        csrc_rank=2,
        csrc_max_iter=3,
        rotation_block_size=4,
        rotation_output_block_size=4,
        activation_calibration_batches=2,
        quantvla_calibration_steps=2,
        rotation_cache_enabled=False,
    )
    _, candidates, _, protected = select_modules(model, config)
    assert len(candidates) == (6 if profile == "cogact" else 3)
    assert len(protected) == (38 if profile == "cogact" else 25)
    before = {name: model.get_submodule(name).weight.detach().clone() for name in protected}
    report = calibrate(model, replay, config)
    assert report["format_version"] == (3 if profile == "cogact" else 2)
    if profile == "cogact":
        assert report["adaptation"]["allocation_unit"] == "output_channel"
        if mode != "quantvla":
            total = sum(module.out_features for module in candidates.values())
            assert sum(map(len, report["high_precision_channels"].values())) >= math.ceil(total * 0.3)
    else:
        assert len(report["high_precision_layers"]) == (0 if mode == "quantvla" else math.ceil(len(candidates) * 0.3))
    assert bool(report["csrc"]) == (mode != "quantvla")
    assert report["quantvla_matching"]["folded_into_weights"]
    if profile == "cogact":
        stages = report["stage_order"]
        assert stages.index("teacher_statistics") < stages.index("weight_QDQ_wrapping")
        if mode in {"da_ptq", "equal_weight"}:
            assert stages.index("weight_QDQ_wrapping") < stages.index("weight_QDQ_student_channel_probe")
            assert report["probe_batches"] == 2
        assert stages.index("student_statistics_with_A8_observation") < stages.index(
            "ATM_OHB_from_same_student_statistics"
        )
        assert stages.index("ATM_OHB_from_same_student_statistics") < stages.index("reference_CSRC_" + csrc_method)
        assert stages[-1] == "final_activation_calibration"
        assert report["quantvla_matching"]["ohb_fold_direction"] == "inverse_beta"
    for name in protected:
        # ATM/OHB intentionally folds into Q/O; other fixed guards are unchanged.
        if not name.endswith(("q_proj", "o_proj")) and name not in report["floating_csrc_layers"]:
            torch.testing.assert_close(model.get_submodule(name).weight, before[name], rtol=0, atol=0)
    for name in report["high_precision_layers"]:
        module = model.get_submodule(name)
        if profile == "cogact":
            assert (module.adaptive_weight_bits == 16).all()
        else:
            assert module.weight_bits == 16
        assert module.activation_bits == config.fallback_activation_bits
    if profile == "cogact":
        assert all(".self_attn." not in name for name in report["layers"])
        if config.use_csrc:
            assert len(report["floating_csrc_layers"]) == 8
            for name in report["floating_csrc_layers"]:
                assert not isinstance(model.get_submodule(name), QuantVLALinear)
                assert model.get_submodule(name).bias is not None
    assert math.isfinite(report["final_flow_mse_probe_subset"])
    observation, actions = next(replay())
    noise, time = torch.randn_like(actions), torch.tensor([0.5])
    expected = model(observation, actions, noise=noise, time=time)
    save_checkpoint(model, report, tmp_path / "artifact")
    restored = build()
    load_model(restored, str(tmp_path / "artifact/model.safetensors"))
    restore_quantization(restored, tmp_path / "artifact")
    torch.testing.assert_close(restored(observation, actions, noise=noise, time=time), expected, rtol=0, atol=0)
    assert not any(hasattr(module, "_quantvla_logits_observer") for module in model.modules())


def test_attention_statistics_masks_gqa_and_fold_direction(flow_model):
    model = flow_model()
    name = "paligemma_with_expert.gemma_expert.model.layers.0.self_attn"
    module = model.get_submodule(name)
    query, key = torch.randn(1, 8, 2, 2), torch.randn(1, 2, 5, 2)
    mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)
    mask[..., -1] = False
    with AttentionStatistics(model) as stats:
        for other in stats.modules.values():
            other._quantvla_logits_observer(query, key, mask, 0.5)  # noqa: SLF001
            other.o_proj(torch.randn(1, 2, 16))
        result = stats.finalize()
    expected = (query @ key.repeat_interleave(4, 1).transpose(-1, -2) * 0.5)[..., :-1]
    torch.testing.assert_close(
        torch.tensor(result[name]["logit_std"]), expected.permute(1, 0, 2, 3).reshape(8, -1).std(-1, correction=0)
    )
    reference = {name: {"logit_std": [1.1] * 8, "output_rms": 1.1}}
    student = {name: {"logit_std": [1.0] * 8, "output_rms": 1.0}}
    inputs = torch.randn(1, 2, 16)
    old_q, old_k, old_o = module.q_proj(inputs), module.k_proj(inputs), module.o_proj(inputs)
    config = PTQConfig()
    alpha = fold_attention_matching(model, reference, student, config, component="atm")[name]
    beta = fold_attention_matching(model, reference, student, config, component="ohb")[name]
    torch.testing.assert_close(module.q_proj(inputs), old_q * alpha[0])
    torch.testing.assert_close(module.k_proj(inputs), old_k, rtol=0, atol=0)
    torch.testing.assert_close(module.o_proj(inputs), old_o * beta)
    torch.testing.assert_close(matching_gain([100, 1.01], [1, 1], 0.3, 0.03), torch.tensor([math.exp(0.3), 1]))


def test_missing_logits_fails_instead_of_silent_atm_noop(flow_model):
    with AttentionStatistics(flow_model()) as stats, pytest.raises(RuntimeError, match="No joint-attention"):
        stats.finalize()


def test_episode_cap_applies_independently_per_task():
    tasks = np.repeat([0, 1], 12)
    episodes = np.repeat([10, 11, 12, 20, 21, 22], 4)
    selected = eligible_frames(tasks, episodes, 2)
    assert selected.tolist() == list(range(8)) + list(range(12, 20))
    assert eligible_frames(tasks, episodes, 0).tolist() == list(range(24))


@pytest.mark.parametrize(
    ("profile", "mode"),
    [(profile, mode) for profile in ("legacy", "quantvla") for mode in ("uniform", "da_ptq", "equal_weight", "random")]
    + [("quantvla", "quantvla")],
)
@pytest.mark.parametrize("requested_tail_count", [0, 1, 2])
def test_final_two_full_expert_blocks_are_always_protected(flow_model, profile, mode, requested_tail_count):
    model = flow_model()
    for layer in model.paligemma_with_expert.gemma_expert.model.layers[-2:]:
        layer.extra_projection = nn.Linear(16, 16)
        layer.to(torch.bfloat16)
    config = PTQConfig(
        profile=profile,
        mode=mode,
        skip_last_expert_blocks=requested_tail_count,
        quant_regex=r".*",  # A broad user override must not bypass the fixed guard.
    )
    quantized, candidates, csrc, protected = select_modules(model, config)
    tail_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(f"gemma_expert.model.layers.{index}." in name for index in (2, 3))
    ]
    assert len(tail_names) == 16  # Q/K/V/O, gate/up/down and custom projection per block
    for name in tail_names:
        assert name in protected
        assert name not in quantized
        assert name not in candidates
        assert name not in csrc
        assert model.get_submodule(name).weight.dtype == torch.bfloat16


def test_loading_rejects_artifact_that_quantizes_mandatory_tail(flow_model, tmp_path):
    model = flow_model().eval()
    config = PTQConfig(profile="legacy", mode="uniform", calibration_samples=3, probe_steps=1)
    report = calibrate(model, replay, config)
    assert report["mandatory_expert_tail_blocks"] == [2, 3]
    destination = tmp_path / "artifact"
    save_checkpoint(model, report, destination)
    manifest = destination / "da_ptq.json"
    payload = json.loads(manifest.read_text())
    payload["layers"]["paligemma_with_expert.gemma_expert.model.layers.3.self_attn.q_proj"] = {
        "weight_bits": 2,
        "activation_bits": 8,
        "original_bias": True,
    }
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="mandatory final action-expert"):
        restore_quantization(flow_model(), destination)
