import dataclasses
import json

import pytest
from safetensors.torch import load_model
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.quantization.calibration import allocate_precision
from openpi.quantization.calibration import calibrate
from openpi.quantization.calibration import select_modules
from openpi.quantization.checkpoint import restore_quantization
from openpi.quantization.checkpoint import save_checkpoint
from openpi.quantization.config import PTQConfig
from openpi.quantization.libero_data import choose_indices
from openpi.quantization.numerics import FakeQuantLinear
from openpi.quantization.numerics import Moments
from openpi.quantization.numerics import fit_csrc
from openpi.quantization.numerics import normalize_prior
from openpi.quantization.numerics import quantize_weight
from openpi.quantization.numerics import structural_weights
from openpi.quantization.numerics import weighted_flow_loss


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, values):
        return self.down_proj(F.silu(self.gate_proj(values)) * self.up_proj(values))


class TinyFlowPolicy(nn.Module):
    """Exercise real module matching, gradients, masks, folding and restoration."""

    def __init__(self):
        super().__init__()
        self.action_in_proj = nn.Linear(9, 4)
        self.action_out_proj = nn.Linear(4, 9)
        self.paligemma_with_expert = nn.Module()
        vlm = nn.Module()
        vlm.language_model = nn.Module()
        vlm.language_model.layers = nn.ModuleList([nn.Module()])
        vlm.language_model.layers[0].self_attn = nn.Module()
        vlm.language_model.layers[0].self_attn.k_proj = nn.Linear(4, 4, bias=False)
        vlm.language_model.layers[0].self_attn.v_proj = nn.Linear(4, 4, bias=False)
        self.paligemma_with_expert.paligemma = vlm
        expert = nn.Module()
        expert.model = nn.Module()
        expert.model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
        for layer in expert.model.layers:
            layer.mlp = MLP()
        self.paligemma_with_expert.gemma_expert = expert

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def sample_time(self, size, device):
        return torch.rand(size, device=device)

    def embed_prefix(self, observation):
        return observation, torch.ones(observation.shape[:2], dtype=torch.bool), None

    def forward(self, observation, actions, noise=None, time=None, *, train=True):
        assert not train
        values, _, _ = self.embed_prefix(observation)
        attn = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn
        conditioning = (attn.k_proj(values) + attn.v_proj(values)).mean(1, keepdim=True)
        noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        hidden = self.action_in_proj(noisy) + conditioning
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            hidden = hidden + layer.mlp(hidden)
        velocity = self.action_out_proj(hidden)
        return (velocity - (noise - actions)).square()


def batches():
    generator = torch.Generator().manual_seed(991)
    for _ in range(3):
        yield torch.randn(1, 5, 4, generator=generator), torch.randn(1, 2, 9, generator=generator)


def toy_config(mode="da_ptq"):
    return PTQConfig(mode=mode, calibration_samples=3, probe_steps=2, csrc_rank=2, csrc_max_iter=12)


def test_int2_codebook_and_zero_rows():
    values = torch.linspace(-1, 1, 31).repeat(2, 1)
    quantized = quantize_weight(values, 2)
    assert len(quantized[0].unique()) == 4
    assert torch.isfinite(quantized).all()
    assert torch.equal(quantize_weight(torch.zeros(3, 8), 2), torch.zeros(3, 8))
    assert torch.equal(quantize_weight(values, 16), values)


def test_activation_qdq_and_fused_affine():
    torch.manual_seed(3)
    source = nn.Linear(4, 3, bias=False)
    module = FakeQuantLinear(source, 16, 8, torch.ones(4))
    values = torch.randn(5, 4) * 0.5
    before = module(values)
    correction = torch.diag(torch.tensor([0.9, 1.1, 1.2]))
    shift = torch.tensor([0.1, -0.2, 0.05])
    module.fold_affine(correction, shift)
    torch.testing.assert_close(module(values), before @ correction.T + shift)
    assert module.bias is not None
    assert not any("bypass" in name or "master" in name for name, _ in module.named_buffers())


def test_w2_is_requantized_after_folding():
    source = nn.Linear(5, 4)
    module = FakeQuantLinear(source, 2, 8, torch.ones(5))
    folded = torch.eye(4) + torch.ones(4, 4) * 0.03
    expected = quantize_weight(folded @ module.weight.detach().clone(), 2)
    module.fold_affine(folded, torch.zeros(4))
    torch.testing.assert_close(module.weight, expected)


def test_padding_has_zero_loss_and_zero_gradient():
    prediction = torch.randn(2, 3, 32, requires_grad=True)
    losses = prediction.square()
    first = weighted_flow_loss(losses, torch.ones(7))
    with torch.no_grad():
        prediction[..., 7:] += 10000
    second = weighted_flow_loss(prediction.square(), torch.ones(7))
    torch.testing.assert_close(first, second)
    second.backward()
    assert torch.count_nonzero(prediction.grad[..., 7:]) == 0


def test_paper_surrogate_has_no_gripper_planar_segment():
    config = PTQConfig()
    actions = torch.zeros(4, 2, 32)
    scores = structural_weights(actions, config)
    actions[..., 6] = 42
    actions[..., 7:] = 999
    torch.testing.assert_close(scores, structural_weights(actions, config))
    assert scores.shape == (7,)
    prior, effective = normalize_prior(scores, config)
    torch.testing.assert_close(prior.mean(), torch.tensor(1.0))
    torch.testing.assert_close(effective.mean(), torch.tensor(1.0))
    assert torch.isfinite(effective).all()
    assert (effective > 0).all()


def test_streaming_covariance_matches_whole_data():
    values = torch.randn(37, 5, dtype=torch.float64) + 100
    stats = Moments()
    for chunk in values.split(7):
        stats.update(chunk)
    torch.testing.assert_close(stats.mean, values.mean(0))
    torch.testing.assert_close(stats.covariance(), torch.cov(values.T, correction=0))


def test_csrc_solver_moment_alignment_and_objective():
    torch.manual_seed(1)
    values = torch.randn(100, 4, dtype=torch.float64)
    matrix = torch.eye(4, dtype=torch.float64) + 0.08 * torch.ones(4, 4, dtype=torch.float64)
    teacher, quant = Moments(), Moments()
    teacher.update(values @ matrix + 0.5)
    quant.update(values)
    correction, shift, diagnostics = fit_csrc(teacher, quant, PTQConfig(csrc_rank=4))
    torch.testing.assert_close(correction.double() @ quant.mean + shift.double(), teacher.mean, atol=1e-6, rtol=1e-6)
    assert diagnostics["objective_dense"] <= diagnostics["objective_before"] + 1e-9
    assert diagnostics["rank"] == 4


def test_selection_budget_and_protected_layers():
    model, config = TinyFlowPolicy(), toy_config()
    quant, candidates, _, protected = select_modules(model, config)
    assert len(candidates) == 6
    assert len(protected) == 6
    assert all(name not in quant for name in protected)
    scores = {name: float(index) for index, name in enumerate(candidates)}
    ranked = allocate_precision(candidates, scores, config)
    random_config = dataclasses.replace(config, mode="random", allocation_seed=8)
    random = allocate_precision(candidates, {}, random_config)
    assert random == allocate_precision(candidates, {}, random_config)
    assert len(ranked) == len(random) == 2
    assert sum(candidates[name].weight.numel() for name in ranked) == sum(
        candidates[name].weight.numel() for name in random
    )


@pytest.mark.parametrize("mode", ["uniform", "da_ptq", "equal_weight", "random"])
def test_calibration_export_reload_roundtrip(mode, tmp_path):
    torch.manual_seed(812)
    model = TinyFlowPolicy().eval()
    initial_protected = model.paligemma_with_expert.gemma_expert.model.layers[3].mlp.up_proj.weight.detach().clone()
    report = calibrate(model, batches, toy_config(mode))
    torch.testing.assert_close(
        model.paligemma_with_expert.gemma_expert.model.layers[3].mlp.up_proj.weight, initial_protected
    )
    assert report["calibration_samples"] == 3
    assert report["loss_weights"] == [1.0] * 7 if mode != "da_ptq" else len(report["loss_weights"]) == 7
    if mode == "uniform":
        assert report["high_precision_layers"] == []
        assert report["csrc"] == {}
    else:
        assert len(report["high_precision_layers"]) == 2
        assert len(report["csrc"]) == 2
    destination = tmp_path / mode
    save_checkpoint(model, report, destination)
    assert json.loads((destination / "da_ptq.json").read_text())["implementation"] == "floating_QDQ_not_packed_integer"
    restored = TinyFlowPolicy().eval()
    load_model(restored, str(destination / "model.safetensors"))
    restore_quantization(restored, destination)
    observation, actions = next(batches())
    noise, time = torch.randn_like(actions), torch.tensor([0.4])
    torch.testing.assert_close(
        model(observation, actions, noise, time, train=False),
        restored(observation, actions, noise, time, train=False),
        rtol=0,
        atol=0,
    )
    with pytest.raises(FileExistsError):
        save_checkpoint(model, report, destination)


def test_equal_weights_do_not_disable_mixed_precision():
    torch.manual_seed(123)
    model = TinyFlowPolicy()
    report = calibrate(model, batches, toy_config("equal_weight"))
    assert report["loss_weights"] == [1.0] * 7
    assert report["gradient_scores"]
    assert report["high_precision_layers"]
    assert report["csrc"]


def test_spatial_task_sampling_is_deterministic_and_balanced():
    import numpy as np

    tasks = np.repeat(np.arange(4), 50)
    positions = np.random.default_rng(8).normal(size=(200, 3))
    kwargs = {"samples": 40, "seed": 17, "spatial_bins": 6, "warmup_samples": 128}
    first, warmup = choose_indices(tasks, positions, **kwargs)
    assert (first, warmup) == choose_indices(tasks, positions, **kwargs)
    assert len(set(first)) == 40
    assert len(set(warmup)) == 128
    assert np.bincount(tasks[first]).tolist() == [10, 10, 10, 10]


def test_checkpoint_preserves_tied_parameter_loading(tmp_path):
    model = TinyFlowPolicy().eval()
    model.tied_output = model.action_out_proj
    report = calibrate(model, batches, toy_config("uniform"))
    save_checkpoint(model, report, tmp_path / "tied")
    restored = TinyFlowPolicy().eval()
    restored.tied_output = restored.action_out_proj
    load_model(restored, str(tmp_path / "tied/model.safetensors"))
    restore_quantization(restored, tmp_path / "tied")
    assert restored.tied_output.weight.data_ptr() == restored.action_out_proj.weight.data_ptr()


def test_bfloat16_qdq_output_stays_bfloat16():
    source = nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
    module = FakeQuantLinear(source, 2, 8, torch.ones(8))
    values = torch.randn(2, 8, dtype=torch.bfloat16)
    assert module(values).dtype == torch.bfloat16
    assert module.weight.dtype == torch.bfloat16
