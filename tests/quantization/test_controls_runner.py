import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("controls_runner_test", scripts / "run_da_ptq_controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(runner, tmp_path, *extra):
    return runner.parser().parse_args(
        [
            "--checkpoint-dir",
            str(tmp_path / "bf16"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "runs"),
            *extra,
        ]
    )


def test_worker_commands_preserve_nondefault_quantvla_parameters(runner, tmp_path):
    args = arguments(
        runner,
        tmp_path,
        "--activation-percentile",
        "99.5",
        "--quantvla-calibration-steps",
        "16",
        "--skip-first-expert-blocks",
        "2",
        "--no-ohb-enabled",
        "--fallback-activation-bits",
        "8",
    )
    jobs = runner.plans(args)
    assert list(jobs) == ["da_ptq", "equal_weight", "random_seed_0", "random_seed_1", "random_seed_2"]
    for name, config in jobs.items():
        command = runner.calibration_command(args, name, config, tmp_path, tmp_path / "selection.json")
        # The real calibration parser must recover precisely the same config.
        calibration_args = runner.quantization_parser().parse_args(command[3:])
        recovered = dict(runner.configurations(calibration_args))
        assert recovered == {name: config}
        assert calibration_args.device == "cuda:0"


def test_calibration_cache_rejects_changed_budget_or_selection(runner, tmp_path):
    args = arguments(runner, tmp_path)
    config = runner.plans(args)["equal_weight"]
    directory = tmp_path / "artifact"
    directory.mkdir()
    for filename in ("model.safetensors", "da_ptq.safetensors"):
        (directory / filename).touch()
    (directory / "assets").mkdir()
    selection = {"dataset_root": str(args.dataset_root.resolve()), "indices": [1, 2]}
    report = {
        "config": config.to_dict(),
        "source_checkpoint": str(args.checkpoint_dir.resolve()),
        "calibration_data": selection,
    }
    (directory / "da_ptq.json").write_text(json.dumps(report))
    runner.validate_artifact(directory, config, args.checkpoint_dir.resolve(), args.dataset_root.resolve(), selection)
    with pytest.raises(ValueError, match="Configuration/source"):
        runner.validate_artifact(
            directory,
            dataclasses.replace(config, bf16_ratio=0.2),
            args.checkpoint_dir.resolve(),
            args.dataset_root.resolve(),
            selection,
        )
    with pytest.raises(ValueError, match="frames changed"):
        runner.validate_artifact(
            directory,
            config,
            args.checkpoint_dir.resolve(),
            args.dataset_root.resolve(),
            {**selection, "indices": [3, 4]},
        )


def test_runner_dry_run_loads_no_model_or_data(runner, tmp_path, capsys):
    runner.main(
        [
            "--checkpoint-dir",
            str(tmp_path / "missing"),
            "--dataset-root",
            str(tmp_path / "missing_data"),
            "--output-dir",
            str(tmp_path / "not_created"),
            "--gpus",
            "6",
            "7",
            "--dry-run",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["groups"] == {
        "6": ["da_ptq", "random_seed_0", "random_seed_2"],
        "7": ["equal_weight", "random_seed_1"],
    }
    assert not (tmp_path / "not_created").exists()
    assert "openpi.training.config" not in sys.modules


def test_evaluate_stage_uses_cached_models_two_devices_and_video(runner, tmp_path, monkeypatch):
    args = arguments(runner, tmp_path)
    args.checkpoint_dir.mkdir()
    (args.checkpoint_dir / "model.safetensors").touch()
    (args.checkpoint_dir / "assets").mkdir()
    (args.dataset_root / "meta").mkdir(parents=True)
    (args.dataset_root / "meta/info.json").write_text("{}")
    quantized = args.output_dir / "quantized"
    quantized.mkdir(parents=True)
    selection = {"dataset_root": str(args.dataset_root.resolve()), "indices": [7, 9]}
    (quantized / "calibration_selection.json").write_text(json.dumps(selection))
    for name, config in runner.plans(args).items():
        directory = quantized / name
        directory.mkdir()
        for filename in ("model.safetensors", "da_ptq.safetensors"):
            (directory / filename).touch()
        (directory / "assets").mkdir()
        (directory / "da_ptq.json").write_text(
            json.dumps(
                {
                    "config": config.to_dict(),
                    "source_checkpoint": str(args.checkpoint_dir.resolve()),
                    "calibration_data": selection,
                }
            )
        )
    calls = []

    def evaluate(command, *, cwd, env, stdout, stderr, check):
        assert command[2].endswith("evaluate_da_ptq_matrix.py")  # no calibration subprocess
        calls.append((command, env))
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir()
        names = command[command.index("--variants") + 1 : command.index("--eval-python")]
        (destination / "results.json").write_text(json.dumps({name: {"average": 0.5} for name in names}))

    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    runner.main(
        [
            "--checkpoint-dir",
            str(args.checkpoint_dir),
            "--dataset-root",
            str(args.dataset_root),
            "--output-dir",
            str(args.output_dir),
            "--stage",
            "evaluate",
            "--eval-python",
            sys.executable,
            "--save-video",
            "--video-max-trials",
            "1",
        ]
    )
    assert {env["CUDA_VISIBLE_DEVICES"] for _, env in calls} == {"6", "7"}
    assert {command[command.index("--port") + 1] for command, _ in calls} == {"2414", "2415"}
    for command, _env in calls:
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--save-video" in command
        assert command[command.index("--video-max-trials") + 1] == "1"
    summary = next(args.output_dir.glob("evaluation_*/results.json"))
    assert set(json.loads(summary.read_text())) == set(runner.plans(args))
