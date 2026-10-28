import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import threading

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


@pytest.mark.parametrize(("workers_per_gpu", "include_bf16"), [(1, False), (3, True)])
def test_evaluate_stage_uses_cached_models_two_devices_and_video(
    runner, tmp_path, monkeypatch, workers_per_gpu, include_bf16
):
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
        if "--bf16-checkpoint" in command:
            names.insert(0, "bf16")
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
            "--workers-per-gpu",
            str(workers_per_gpu),
            *(["--include-bf16"] if include_bf16 else []),
        ]
    )
    assert {env["CUDA_VISIBLE_DEVICES"] for _, env in calls} == {"6", "7"}
    assert {command[command.index("--port") + 1] for command, _ in calls} == {
        str(port) for port in range(2414, 2420 if workers_per_gpu == 3 else 2416)
    }
    if workers_per_gpu == 3:
        assert len(calls) == 6
        assert all(sum(env["CUDA_VISIBLE_DEVICES"] == str(gpu) for _, env in calls) == 3 for gpu in (6, 7))
    for command, _env in calls:
        assert command[command.index("--device") + 1] == "cuda:0"
        assert "--save-video" in command
        assert command[command.index("--video-max-trials") + 1] == "1"
    summary = next(args.output_dir.glob("evaluation_*/results.json"))
    assert set(json.loads(summary.read_text())) == set(runner.plans(args)) | ({"bf16"} if include_bf16 else set())


def test_multiworker_layout_has_unique_ports_and_no_duplicate_models(runner, tmp_path):
    args = arguments(runner, tmp_path, "--workers-per-gpu", "3", "--include-bf16")
    jobs = runner.plans(args)
    workers = runner.worker_layout(args, jobs)
    assert [worker["variants"] for worker in workers] == [
        ["da_ptq"],
        ["equal_weight"],
        ["random_seed_0"],
        ["random_seed_1"],
        ["random_seed_2"],
        [],
    ]
    assert workers[-1]["include_bf16"]
    assert len({worker["label"] for worker in workers}) == len({worker["port"] for worker in workers}) == 6
    for invalid in ("0", "-1"):
        with pytest.raises(ValueError, match="workers-per-gpu"):
            runner.plans(arguments(runner, tmp_path, "--workers-per-gpu", invalid))
    with pytest.raises(ValueError, match="ports"):
        runner.plans(arguments(runner, tmp_path, "--workers-per-gpu", "3", "--base-port", "65531"))


@pytest.mark.parametrize("include_bf16", [False, True])
def test_split_suites_uses_twelve_slots_and_merges_every_cell_once(runner, tmp_path, monkeypatch, include_bf16):
    args = arguments(
        runner,
        tmp_path,
        "--stage",
        "evaluate",
        "--split-suites",
        "--gpus",
        "4",
        "5",
        "6",
        "7",
        "--workers-per-gpu",
        "3",
        "--eval-python",
        sys.executable,
        "--num-trials",
        "50",
        "--save-video",
        *(["--include-bf16"] if include_bf16 else []),
    )
    jobs = runner.plans(args)
    workers = runner.worker_layout(args, jobs)
    assert len(workers) == len({worker["port"] for worker in workers}) == 12
    assert all(sum(worker["gpu"] == gpu for worker in workers) == 3 for gpu in (4, 5, 6, 7))
    invocation = tmp_path / "evaluation"
    invocation.mkdir()
    rates = {"libero_spatial": 0.1, "libero_object": 0.3, "libero_goal": 0.5, "libero_10": 0.7}
    barrier = threading.Barrier(12)
    lock = threading.Lock()
    calls, used_ports = [], set()

    def evaluate(command, *, cwd, env, stdout, stderr, check):
        assert command[2].endswith("evaluate_da_ptq_matrix.py")
        suite = command[command.index("--suites") + 1]
        names = command[command.index("--variants") + 1 : command.index("--eval-python")]
        if "--bf16-checkpoint" in command:
            assert names == []
            names = ["bf16"]
        assert len(names) == 1
        port = command[command.index("--port") + 1]
        with lock:
            first = port not in used_ports
            used_ports.add(port)
            calls.append((names[0], suite, env["CUDA_VISIBLE_DEVICES"], port))
        if first:
            barrier.wait(timeout=15)  # exercise all 12 simultaneous slot owners
        assert command[command.index("--device") + 1] == "cuda:0"
        assert command[command.index("--num-trials") + 1] == "50"
        assert "--save-video" in command
        destination = Path(command[command.index("--output-dir") + 1])
        destination.mkdir(parents=True)
        (destination / "results.json").write_text(
            json.dumps(
                {
                    names[0]: {"suite_success_rates": {suite: rates[suite]}, "average": rates[suite]},
                }
            )
        )

    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    results = runner.run_split_evaluation(
        args,
        jobs,
        workers,
        tmp_path,
        invocation,
        tmp_path / "quantized",
        args.checkpoint_dir,
    )
    expected_names = set(jobs) | ({"bf16"} if include_bf16 else set())
    assert {(name, suite) for name, suite, _, _ in calls} == {
        (name, suite) for name in expected_names for suite in rates
    }
    assert len(calls) == len(expected_names) * 4
    assert set(results) == expected_names
    for result in results.values():
        assert result["suite_success_rates"] == rates
        assert result["average"] == pytest.approx(0.4)
    assert (invocation / "gpu4_worker0.log").is_file()


def test_split_suites_rejects_calibration_and_incomplete_aggregation(runner, tmp_path):
    with pytest.raises(ValueError, match="requires --stage evaluate"):
        runner.plans(arguments(runner, tmp_path, "--split-suites"))
    with pytest.raises(RuntimeError, match="Incomplete/extra"):
        runner.merge_suite_results(
            [("da_ptq", "libero_spatial"), ("da_ptq", "libero_goal")],
            {
                ("da_ptq", "libero_spatial"): 0.5,
            },
        )
