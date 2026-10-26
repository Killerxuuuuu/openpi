"""Exercise the actual PI0Pytorch flow/inference code with a tiny Torch backend.

The substitutes only remove heavyweight JAX/config/vision dependencies; the
forward, flow target, masking interfaces and sampler are from the real source.
This is not a substitute for a full-checkpoint LIBERO evaluation.
"""

import importlib
import importlib.util
from pathlib import Path
import sys
import types

from safetensors.torch import load_model
from test_da_ptq import TinyFlowPolicy
from test_da_ptq import toy_config
import torch
from torch import nn

from openpi.quantization.calibration import calibrate
from openpi.quantization.checkpoint import restore_quantization
from openpi.quantization.checkpoint import save_checkpoint


def test_actual_pi05_forward_calibration_and_sampler(monkeypatch, tmp_path):
    models_package = importlib.import_module("openpi.models")
    pytorch_package = importlib.import_module("openpi.models_pytorch")
    preprocess_modes = []
    preprocessing = types.ModuleType("openpi.models_pytorch.preprocessing_pytorch")

    def preprocess(observation, *, train):
        preprocess_modes.append(train)
        return observation

    preprocessing.preprocess_observation_pytorch = preprocess
    gemma_backend = types.ModuleType("openpi.models_pytorch.gemma_pytorch")
    gemma_backend.PaliGemmaWithExpertModel = object
    gemma_config = types.ModuleType("openpi.models.gemma")
    monkeypatch.setitem(sys.modules, "openpi.models.gemma", gemma_config)
    monkeypatch.setitem(sys.modules, "openpi.models_pytorch.gemma_pytorch", gemma_backend)
    monkeypatch.setitem(sys.modules, "openpi.models_pytorch.preprocessing_pytorch", preprocessing)
    monkeypatch.setattr(models_package, "gemma", gemma_config, raising=False)
    monkeypatch.setattr(pytorch_package, "gemma_pytorch", gemma_backend, raising=False)
    monkeypatch.setattr(pytorch_package, "preprocessing_pytorch", preprocessing, raising=False)
    source = Path(__file__).resolve().parents[2] / "src/openpi/models_pytorch/pi0_pytorch.py"
    spec = importlib.util.spec_from_file_location("actual_pi05_forward_for_test", source)
    actual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)

    class Backend(nn.Module):
        def __init__(self):
            super().__init__()
            tiny = TinyFlowPolicy().paligemma_with_expert
            self.paligemma, self.gemma_expert = tiny.paligemma, tiny.gemma_expert
            self.paligemma.language_model.config = types.SimpleNamespace()
            self.gemma_expert.model.config = types.SimpleNamespace()
            self.paligemma.language_model.layers[0].self_attn.q_proj = nn.Linear(4, 4, bias=False)
            self.image_embedding = nn.Linear(3, 4)
            self.token_embedding = nn.Embedding(32, 4)

        def embed_image(self, image):
            return self.image_embedding(image.mean((-1, -2)))[:, None]

        def embed_language_tokens(self, tokens):
            return self.token_embedding(tokens)

        def forward(self, *, inputs_embeds, past_key_values=None, **kwargs):
            prefix, suffix = inputs_embeds
            if prefix is not None:
                attention = self.paligemma.language_model.layers[0].self_attn
                context = (attention.q_proj(prefix) + attention.k_proj(prefix) + attention.v_proj(prefix)).mean(
                    1, keepdim=True
                )
            else:
                context = past_key_values["context"]
            if suffix is not None:
                suffix = suffix + context
                for layer in self.gemma_expert.model.layers:
                    suffix = suffix + layer.mlp(suffix)
            return [prefix, suffix], {"context": context}

    def build_model():
        model = actual.PI0Pytorch.__new__(actual.PI0Pytorch)
        nn.Module.__init__(model)
        model.config = types.SimpleNamespace(action_dim=9, action_horizon=2, pi05=True)
        model.pi05 = True
        model.gradient_checkpointing_enabled = False
        model.paligemma_with_expert = Backend()
        model.action_in_proj = nn.Linear(9, 4)
        model.action_out_proj = nn.Linear(4, 9)
        model.time_mlp_in = nn.Linear(4, 4)
        model.time_mlp_out = nn.Linear(4, 4)
        return model.eval()

    generator = torch.Generator().manual_seed(1)
    observation = types.SimpleNamespace(
        images={"base_0_rgb": torch.randn(1, 3, 4, 4, generator=generator)},
        image_masks={"base_0_rgb": torch.ones(1, dtype=torch.bool)},
        state=torch.zeros(1, 9),
        tokenized_prompt=torch.ones(1, 3, dtype=torch.long),
        tokenized_prompt_mask=torch.tensor([[True, True, False]]),
    )
    actions = torch.randn(1, 2, 9, generator=generator)
    noise = torch.randn(1, 2, 9, generator=generator)
    model = build_model()
    assert model(observation, actions, noise, torch.tensor([0.5]), train=False).shape == (1, 2, 9)
    # SimpleNamespace is immutable in this replay; real observations use the
    # dataclass path in move_to. Everything here stays on CPU.
    report = calibrate(model, lambda: iter([(observation, actions)] * 3), toy_config())
    assert report["csrc"]
    assert preprocess_modes
    assert not any(preprocess_modes)
    sampled = model.sample_actions("cpu", observation, noise=noise, num_steps=2)
    assert sampled.shape == (1, 2, 9)
    assert torch.isfinite(sampled).all()
    save_checkpoint(model, report, tmp_path / "calibrated")
    restored = build_model()
    load_model(restored, str(tmp_path / "calibrated/model.safetensors"))
    restore_quantization(restored, tmp_path / "calibrated")
    torch.testing.assert_close(
        sampled, restored.sample_actions("cpu", observation, noise=noise, num_steps=2), rtol=0, atol=0
    )
