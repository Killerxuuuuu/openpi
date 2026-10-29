"""Calibrate W2A8/pi05-LIBERO and the two same-budget DA-MPA controls.

Run --help without loading openpi's GPU/JAX/LeRobot dependencies. Use --dry-run
to inspect the supplementary-A.3 settings before scheduling GPU experiments.
"""

import argparse
import dataclasses
import gc
import json
import logging
import os
from pathlib import Path
import sys


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint-dir", required=True, type=Path, help="Converted pi05_libero PyTorch checkpoint")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--mode", choices=["all", "uniform", "quantvla", "da_ptq", "equal_weight", "random"], default="all"
    )
    result.add_argument("--profile", choices=["quantvla", "legacy", "cogact"], default="quantvla")
    result.add_argument("--config-name", default="pi05_libero")
    result.add_argument(
        "--dataset-root", type=Path, help="Optional local LeRobot dataset root, containing meta/info.json"
    )
    result.add_argument("--selection-file", type=Path, help="Reuse a saved calibration_selection.json")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int)
    result.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    result.add_argument("--bf16-ratio", type=float, default=0.30)
    result.add_argument("--calibration-samples", type=int, default=512)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--spatial-bins", type=int, default=6)
    result.add_argument("--spatial-warmup-samples", type=int, default=128)
    result.add_argument("--probe-steps", type=int, default=16)
    result.add_argument("--skip-first-expert-blocks", type=int)
    result.add_argument(
        "--skip-last-expert-blocks",
        type=int,
        default=2,
        help="Protect full final expert blocks; at least two are always protected, including when set to 0 or 1",
    )
    result.add_argument("--fallback-activation-bits", type=int, choices=[8, 16])
    result.add_argument("--activation-percentile", type=float, default=99.9)
    result.add_argument("--activation-calibration-batches", type=int)
    result.add_argument("--quantvla-calibration-steps", type=int)
    result.add_argument("--max-calibration-episodes-per-task", type=int)
    result.add_argument("--rotation-block-size", type=int)
    result.add_argument("--rotation-output-block-size", type=int)
    result.add_argument("--rotation-permute", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--row-rotation", choices=["restore", "off"], default="restore")
    result.add_argument("--rotation-device", choices=["auto", "cpu", "cuda"], default="auto")
    result.add_argument("--rotation-cache-enabled", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--rotation-cache-dir", default=".cache/cogact_reference_v3")
    result.add_argument("--quantvla-smoothing", type=float, default=0.15)
    result.add_argument("--weight-scale-mode", choices=["mse", "rtn"], default="mse")
    result.add_argument("--atm-enabled", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--ohb-enabled", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--atm-log-clamp", type=float, default=0.30)
    result.add_argument("--ohb-log-clamp", type=float, default=0.30)
    result.add_argument("--matching-neutral-band", type=float)
    result.add_argument("--weight-bits", type=int, default=2)
    result.add_argument("--activation-bits", type=int, default=8)
    result.add_argument("--damping", type=float, default=3e-4)
    result.add_argument("--planar-weight", type=float, default=1.8)
    result.add_argument("--control-weight", type=float, default=0.15)
    result.add_argument("--sensitivity-gain", type=float, default=1.6)
    result.add_argument("--physical-link-lengths", default="0.35,0.30,0.25,0.20,0.15,0.10,0.08")
    result.add_argument("--action-timestep", type=int, default=0)
    result.add_argument("--csrc-rank", type=int, default=8)
    result.add_argument("--csrc-method", choices=["affine", "dense"], default="affine")
    result.add_argument("--csrc-group-size", type=int, default=32)
    result.add_argument("--csrc-group-shrink", type=float, default=0.55)
    result.add_argument("--csrc-shift-max", type=float, default=3.0)
    result.add_argument("--csrc-dense-ridge", type=float, default=1e-4)
    result.add_argument("--csrc-dense-min-eig", type=float, default=1e-6)
    result.add_argument("--csrc-dense-weight-gain", type=float, default=1.0)
    result.add_argument("--csrc-dense-solver-rcond", type=float, default=1e-3)
    result.add_argument(
        "--csrc-dense-structure", choices=["full", "blockdiag", "lowrank", "blockdiag_lowrank"], default="lowrank"
    )
    result.add_argument("--csrc-dense-block-size", type=int, default=64)
    result.add_argument("--csrc-dense-delta-clip", type=float, default=0.25)
    result.add_argument("--csrc-bypass-enabled", action=argparse.BooleanOptionalAction, default=False)
    result.add_argument("--csrc-feature-weight", type=float, default=1.0)
    result.add_argument("--csrc-identity-weight", type=float, default=8.0)
    result.add_argument("--csrc-max-iter", type=int, default=64)
    result.add_argument("--csrc-max-dim", type=int)
    result.add_argument("--csrc-scale-min", type=float, default=0.5)
    result.add_argument("--csrc-scale-max", type=float, default=2.0)
    result.add_argument("--no-csrc", action="store_true", help="Optional DA-MPA-only ablation")
    result.add_argument("--quant-regex", help="Override explicit module selection; full match")
    result.add_argument("--rank-regex", help="Override MLP ranking candidates; full match")
    result.add_argument("--csrc-regex", help="Override legal compensation projections; full match")
    result.add_argument("--unbalanced-sampling", action="store_true")
    result.add_argument(
        "--dry-run", action="store_true", help="Validate settings and print jobs, without loading a model"
    )
    return result


def configurations(args):
    from openpi.quantization.config import PTQConfig

    overrides = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(PTQConfig)
        if hasattr(args, field.name) and getattr(args, field.name) is not None and field.name != "mode"
    }
    overrides["csrc_enabled"] = not args.no_csrc
    jobs = []
    modes = ["uniform", "quantvla", "da_ptq", "equal_weight", "random"]
    if args.profile == "legacy":
        modes.remove("quantvla")
    for mode in modes if args.mode == "all" else [args.mode]:
        seeds = args.random_seeds if mode == "random" else [0]
        for seed in seeds:
            config = PTQConfig(**overrides, mode=mode, allocation_seed=seed)
            name = f"random_seed_{seed}" if mode == "random" else mode
            jobs.append((name, config))
    if len({name for name, _ in jobs}) != len(jobs):
        raise ValueError("random-seeds must be distinct")
    return jobs


def main(argv=None):
    args = parser().parse_args(argv)
    jobs = configurations(args)
    # Resolve profile defaults before these values are used by data sampling.
    args.seed = jobs[0][1].seed
    args.max_calibration_episodes_per_task = jobs[0][1].max_calibration_episodes_per_task
    if args.dry_run:
        print(json.dumps({name: config.to_dict() for name, config in jobs}, indent=2))
        return
    # openpi's data processing uses JAX, but the policy and gradients use Torch.
    # Keep JAX off the accelerator to avoid its allocator competing for VRAM.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import torch

    from openpi.quantization.calibration import calibrate
    from openpi.quantization.checkpoint import save_checkpoint
    from openpi.quantization.libero_data import LiberoCalibrationData
    from openpi.training import config as config_api

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    checkpoint = args.checkpoint_dir.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(
            "A converted PyTorch checkpoint with model.safetensors is required. "
            "Use examples/convert_jax_model_to_pytorch.py first."
        )
    if (checkpoint / "da_ptq.json").exists():
        raise ValueError("Use the original full-precision checkpoint, not an already calibrated artifact")
    if not (checkpoint / "assets").is_dir():
        raise FileNotFoundError("The original policy normalization assets must accompany the checkpoint")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. --dry-run and CPU unit tests do not require a GPU.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in jobs:
        if (args.output_dir / name).exists():
            raise FileExistsError(f"Experiment already exists: {args.output_dir / name}")
    training = config_api.get_config(args.config_name)
    if not training.model.pi05:
        raise ValueError("This driver expects a pi0.5 model configuration")
    # Calibration needs eager hooks and gradients. Only disable compilation, not
    # the checkpoint's model architecture, normalization, or action horizon.
    training = dataclasses.replace(training, model=dataclasses.replace(training.model, pytorch_compile_mode=None))
    data = LiberoCalibrationData(
        training,
        checkpoint,
        samples=args.calibration_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        root=args.dataset_root,
        selection_file=args.selection_file,
        task_balanced=not args.unbalanced_sampling,
        spatial_bins=args.spatial_bins,
        spatial_warmup_samples=args.spatial_warmup_samples,
        max_episodes_per_task=args.max_calibration_episodes_per_task if args.profile != "legacy" else 0,
        sampling_recipe="cogact" if args.profile == "cogact" else "workspace_xyz",
        action_timestep=jobs[0][1].action_timestep,
    )
    selection_path = args.output_dir / "calibration_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != data.manifest:
            raise ValueError("Existing output root uses a different calibration selection")
    else:
        selection_path.write_text(json.dumps(data.manifest, indent=2), encoding="utf-8")
    for name, config in jobs:
        logging.info("Starting %s; mode=%s; BF16 ratio=%.3f", name, config.mode, config.bf16_ratio)
        model = training.model.load_pytorch(training, str(checkpoint / "model.safetensors"))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        model = model.to(args.device).eval()
        report = calibrate(model, data, config)
        report["model_config"] = {
            "config_name": args.config_name,
            "pi05": True,
            "action_dim": training.model.action_dim,
            "action_horizon": training.model.action_horizon,
        }
        report["calibration_data"] = data.manifest
        report["source_checkpoint"] = str(checkpoint)
        report["environment"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith("cuda")
            else args.device,
        }
        save_checkpoint(model, report, args.output_dir / name, assets_source=checkpoint / "assets")
        logging.info("Saved %s", args.output_dir / name)
        del model, report
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
