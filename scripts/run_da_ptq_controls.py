"""Run QuantVLA-based DA-PTQ controls on independent GPUs with shared frames.

Calibration artifacts are immutable and reused after validating configuration,
source and selection. Every evaluation gets a new directory. This does not
resume individual episodes inside a partially completed evaluation.
"""

import concurrent.futures
import dataclasses
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from quantize_da_ptq import configurations
from quantize_da_ptq import parser as quantization_parser


def parser():
    result = quantization_parser()
    result.description = __doc__
    result.add_argument("--stage", choices=["calibrate", "evaluate", "all"], default="all")
    result.add_argument("--gpus", nargs="+", type=int, default=[6, 7], help="Physical CUDA indices, one worker each")
    result.add_argument(
        "--variants", nargs="+", default=["da_ptq", "equal_weight", "random_seed_0", "random_seed_1", "random_seed_2"]
    )
    result.add_argument("--eval-python", type=Path, help="LIBERO CPython executable (required for evaluation)")
    result.add_argument("--num-trials", type=int, default=2)
    result.add_argument("--seeds", nargs="+", type=int, default=[7])
    result.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    result.add_argument("--base-port", type=int, default=2414)
    result.add_argument(
        "--include-bf16", action="store_true", help="Also evaluate original checkpoint as a sanity check"
    )
    result.add_argument("--save-video", action="store_true")
    result.add_argument("--video-max-trials", type=int, default=2, help="Videos per task/seed; 0 means all")
    return result


def plans(args):
    if args.mode != "all":
        raise ValueError("Use --variants to select jobs; leave --mode at all")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        raise ValueError("GPU indices must be nonnegative and distinct")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("Variants must be distinct")
    available = dict(configurations(args))
    if unknown := set(args.variants) - available.keys():
        raise ValueError(f"Unknown/unconfigured variants: {sorted(unknown)}")
    if not args.variants and not args.include_bf16:
        raise ValueError("No jobs requested")
    if args.num_trials < 1 or args.video_max_trials < 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Invalid evaluation counts/seeds")
    if args.base_port < 1024 or args.base_port + len(args.gpus) - 1 > 65535:
        raise ValueError("Invalid policy ports")
    return {name: available[name] for name in args.variants}


def validate_artifact(directory, config, checkpoint, dataset, selection):
    """A manifest written last marks a complete, reusable calibration."""
    required = ["da_ptq.json", "model.safetensors", "da_ptq.safetensors", "assets"]
    if not all((directory / name).exists() for name in required):
        raise ValueError(
            f"Incomplete calibration at {directory}; choose a new --output-dir or move the partial variant"
        )
    report = json.loads((directory / "da_ptq.json").read_text(encoding="utf-8"))
    if report["config"] != config.to_dict() or Path(report["source_checkpoint"]).resolve() != checkpoint:
        raise ValueError(f"Configuration/source changed at {directory}; choose a new --output-dir")
    if report["calibration_data"] != selection:
        raise ValueError(f"Calibration frames changed at {directory}")
    if dataset is not None and Path(selection["dataset_root"]).resolve() != dataset:
        raise ValueError(f"Dataset path changed at {directory}")


def calibration_command(args, name, config, root, selection):
    command = [
        sys.executable,
        "-u",
        str(root / "scripts/quantize_da_ptq.py"),
        "--checkpoint-dir",
        str(args.checkpoint_dir.resolve()),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--output-dir",
        str((args.output_dir / "quantized").resolve()),
        "--selection-file",
        str(selection),
        "--mode",
        config.mode,
        "--device",
        "cuda:0",
    ]
    options = quantization_parser()._option_string_actions  # noqa: SLF001 -- argparse exposes no public option lookup
    for field in dataclasses.fields(config):
        if field.name in {"mode", "allocation_seed", "csrc_enabled"}:
            continue
        option = "--" + field.name.replace("_", "-")
        if option not in options:
            continue
        value = getattr(config, field.name)
        if isinstance(value, bool):
            command.append(option if value else "--no-" + option[2:])
        else:
            command.extend([option, str(value)])
    if not config.csrc_enabled:
        command.append("--no-csrc")
    if config.mode == "random":
        command.extend(["--random-seeds", str(config.allocation_seed)])
    if args.unbalanced_sampling:
        command.append("--unbalanced-sampling")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    jobs = plans(args)
    groups = [list(jobs)[offset :: len(args.gpus)] for offset in range(len(args.gpus))]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "render_backend": os.environ.get("MUJOCO_GL"),
                    "groups": dict(zip(map(str, args.gpus), groups, strict=True)),
                    "jobs": {name: config.to_dict() for name, config in jobs.items()},
                    "include_bf16": args.include_bf16,
                },
                indent=2,
            )
        )
        return
    root = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint_dir.resolve()
    dataset = args.dataset_root.resolve() if args.dataset_root is not None else None
    if dataset is None or not (dataset / "meta/info.json").is_file():
        raise ValueError("Provide --dataset-root containing meta/info.json")
    if not (checkpoint / "model.safetensors").is_file() or not (checkpoint / "assets").is_dir():
        raise ValueError("Provide original converted checkpoint with model.safetensors and assets")
    if (checkpoint / "da_ptq.json").exists():
        raise ValueError("Start from the original BF16 checkpoint")
    if args.stage != "calibrate" and (args.eval_python is None or not args.eval_python.is_file()):
        raise ValueError("Provide --eval-python pointing to the LIBERO CPython environment")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    quantized = output / "quantized"
    quantized.mkdir(exist_ok=True)
    selection_path = quantized / "calibration_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
    pending = []
    for name, config in jobs.items():
        directory = quantized / name
        if directory.exists():
            if selection is None:
                raise ValueError("Existing artifacts have no shared calibration_selection.json")
            validate_artifact(directory, config, checkpoint, dataset, selection)
            print(f"Reuse completed calibration: {name}", flush=True)
        elif args.stage == "evaluate":
            raise FileNotFoundError(f"Not calibrated: {directory}; first use --stage calibrate/all")
        else:
            pending.append(name)
    if pending:
        # Create/validate one selection before workers start. GPU processes
        # receive exactly this manifest, not competing writes or distinct draws.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        from openpi.quantization.libero_data import LiberoCalibrationData
        from openpi.training import config as config_api

        training = config_api.get_config(args.config_name)
        data = LiberoCalibrationData(
            training,
            checkpoint,
            samples=args.calibration_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            root=dataset,
            selection_file=selection_path if selection is not None else args.selection_file,
            task_balanced=not args.unbalanced_sampling,
            spatial_bins=args.spatial_bins,
            spatial_warmup_samples=args.spatial_warmup_samples,
            max_episodes_per_task=args.max_calibration_episodes_per_task if args.profile == "quantvla" else 0,
        )
        if selection is not None and selection != data.manifest:
            raise ValueError("Calibration selection settings changed; use a new --output-dir")
        selection = data.manifest
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        del data
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S_UTC") + f"_{os.getpid()}"
    invocation = output / ("calibration_" + stamp if args.stage == "calibrate" else "evaluation_" + stamp)
    invocation.mkdir()
    print(f"Invocation: {invocation}", flush=True)
    (invocation / "request.json").write_text(
        json.dumps(
            {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "configurations": {name: config.to_dict() for name, config in jobs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def worker(offset):
        gpu, names = args.gpus[offset], groups[offset]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
        }
        with (invocation / f"gpu{gpu}.log").open("w", encoding="utf-8") as log:

            def run(command):
                print(f"GPU {gpu}: {' '.join(command)}", flush=True)
                subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

            for name in names:
                if name in pending:
                    run(calibration_command(args, name, jobs[name], root, selection_path))
                    validate_artifact(quantized / name, jobs[name], checkpoint, dataset, selection)
            if args.stage == "calibrate" or (not names and not (args.include_bf16 and offset == 0)):
                return {}
            destination = invocation / f"gpu{gpu}"
            command = [
                sys.executable,
                "-u",
                str(root / "scripts/evaluate_da_ptq_matrix.py"),
                "--runs-dir",
                str(quantized),
                "--output-dir",
                str(destination),
                "--variants",
                *names,
                "--eval-python",
                str(args.eval_python.resolve()),
                "--num-trials",
                str(args.num_trials),
                "--seeds",
                *map(str, args.seeds),
                "--suites",
                *args.suites,
                "--port",
                str(args.base_port + offset),
                "--device",
                "cuda:0",
            ]
            if args.include_bf16 and offset == 0:
                command.extend(["--bf16-checkpoint", str(checkpoint)])
            if args.save_video:
                command.extend(["--save-video", "--video-max-trials", str(args.video_max_trials)])
            run(command)
            return json.loads((destination / "results.json").read_text(encoding="utf-8"))

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        for result in executor.map(worker, range(len(args.gpus))):
            results.update(result)
    if args.stage != "calibrate":
        (invocation / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Completed. Artifacts: {quantized}; logs/results: {invocation}", flush=True)


if __name__ == "__main__":
    main()
