"""Self-contained floating-QDQ checkpoints loadable by the standard server."""

import json
from pathlib import Path
import shutil

from safetensors.torch import load_file
from safetensors.torch import save_file
from safetensors.torch import save_model
from torch import nn

from openpi.quantization.calibration import replace_module
from openpi.quantization.numerics import FakeQuantLinear

MANIFEST = "da_ptq.json"
SIDECAR = "da_ptq.safetensors"


class _BaseStateView:
    def __init__(self, model, original_keys):
        self.model, self.original_keys = model, original_keys

    def state_dict(self):
        state = self.model.state_dict()
        # Keep shared storage relationships for safetensors.save_model, which
        # handles tied embeddings. Additional bias/scales go into the sidecar.
        # Detect sharing BEFORE a CPU copy: independently copying tied GPU
        # tensors would destroy their storage identity and break strict reload.
        return {key: state[key].detach() for key in self.original_keys}


def save_checkpoint(model, report: dict, destination: str | Path, *, assets_source: str | Path | None = None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a new experiment directory")
    destination.mkdir(parents=True)
    save_model(_BaseStateView(model, report["original_state_keys"]), str(destination / "model.safetensors"))
    state = {}
    for name, info in report["layers"].items():
        module = model.get_submodule(name)
        state[f"{name}.activation_absmax"] = module.activation_absmax.detach().cpu().contiguous()
        if not info["original_bias"] and module.bias is not None:
            state[f"{name}.added_bias"] = module.bias.detach().cpu().contiguous()
    save_file(state, str(destination / SIDECAR))
    if assets_source is not None:
        source = Path(assets_source)
        if not source.is_dir():
            raise FileNotFoundError(f"Checkpoint normalization assets missing: {source}")
        shutil.copytree(source, destination / "assets")
    # Write manifest last: its presence is the deployment-ready marker.
    manifest = {key: value for key, value in report.items() if key != "original_state_keys"}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


def restore_quantization(model, checkpoint_dir: str | Path) -> dict:
    """Call AFTER loading standard weights and applying the normal dtype policy.

    Stored QDQ weights must not be rounded again; doing so with a newly fitted
    scale changes them. Only runtime A8 and fused affine biases are reconstructed.
    """
    directory = Path(checkpoint_dir)
    report = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if report.get("format_version") != 1 or report.get("implementation") != "floating_QDQ_not_packed_integer":
        raise ValueError("Unsupported DA-PTQ checkpoint format")
    sidecar = load_file(str(directory / SIDECAR), device="cpu")
    if hasattr(model, "config") and "model_config" in report:
        for key in ("action_dim", "action_horizon", "pi05"):
            if getattr(model.config, key) != report["model_config"][key]:
                raise ValueError(f"Model configuration differs from calibrated checkpoint: {key}")
    for name, info in report["layers"].items():
        source = model.get_submodule(name)
        if not isinstance(source, nn.Linear) or isinstance(source, FakeQuantLinear):
            raise ValueError(f"Expected an unwrapped Linear at {name}")
        replacement = FakeQuantLinear(
            source, info["weight_bits"], info["activation_bits"], sidecar[f"{name}.activation_absmax"], quantize=False
        )
        bias_key = f"{name}.added_bias"
        if bias_key in sidecar:
            replacement.bias = nn.Parameter(
                sidecar[bias_key].to(source.weight.device, source.weight.dtype), requires_grad=False
            )
        replace_module(model, name, replacement)
    return report
