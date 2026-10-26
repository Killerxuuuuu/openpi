"""Conversion regressions without requiring a multi-GB JAX checkpoint."""

import ast
from pathlib import Path
import types

import numpy as np
import torch

from openpi.shared.checkpoint_assets import find_checkpoint_assets


def test_downloaded_checkpoint_assets_take_precedence(tmp_path):
    checkpoint = tmp_path / "pi05_libero"
    (checkpoint / "assets").mkdir(parents=True)
    (tmp_path / "assets").mkdir()
    assert find_checkpoint_assets(checkpoint) == checkpoint / "assets"


def test_legacy_assets_layout(tmp_path):
    (tmp_path / "assets").mkdir()
    assert find_checkpoint_assets(tmp_path / "params") == tmp_path / "assets"
    assert find_checkpoint_assets(tmp_path / "missing" / "params") is None


def test_gemma_conversion_uses_config_not_directory_name():
    source = Path(__file__).resolve().parents[2] / "examples/convert_jax_model_to_pytorch.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "slice_gemma_state_dict"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    state = {
        "llm/layers/attn/attn_vec_einsum_1/w": np.zeros((1, 1, 2, 2)),
        "llm/layers/attn/kv_einsum_1/w": np.zeros((1, 2, 1, 2, 2)),
        "llm/layers/attn/q_einsum_1/w": np.zeros((1, 1, 2, 2)),
        "llm/layers/mlp_1/gating_einsum": np.zeros((1, 2, 2, 3)),
        "llm/layers/mlp_1/linear": np.zeros((1, 3, 2)),
        "llm/layers/pre_attention_norm_1/Dense_0/bias": np.ones((1, 6)),
        "llm/layers/pre_ffw_norm_1/Dense_0/bias": np.ones((1, 6)),
        "llm/layers/pre_attention_norm_1/Dense_0/kernel": np.ones((1, 2, 6)),
        "llm/layers/pre_ffw_norm_1/Dense_0/kernel": np.ones((1, 2, 6)),
        "llm/final_norm_1/Dense_0/bias": np.ones(6),
        "llm/final_norm_1/Dense_0/kernel": np.ones((2, 6)),
    }
    config = types.SimpleNamespace(width=2, depth=1, num_heads=1, head_dim=2)
    converted = namespace["slice_gemma_state_dict"](
        state, config, num_expert=1, checkpoint_dir="/data/custom_name", pi05=True
    )
    assert "paligemma_with_expert.gemma_expert.model.norm.dense.weight" in converted
    assert converted["paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj.weight"].shape == (3, 2)
