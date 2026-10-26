import json
import os
from pathlib import Path
import subprocess
import sys


def test_matrix_dry_run_without_model_or_dataset(tmp_path):
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    destination = tmp_path / "not_created"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/quantize_da_ptq.py"),
            "--checkpoint-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(destination),
            "--mode",
            "all",
            "--dry-run",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    jobs = json.loads(result.stdout)
    assert set(jobs) == {"uniform", "da_ptq", "equal_weight", "random_seed_0", "random_seed_1", "random_seed_2"}
    assert all(job["weight_bits"] == 2 for job in jobs.values())
    assert all(job["activation_bits"] == 8 for job in jobs.values())
    assert all(job["bf16_ratio"] == 0.3 for job in jobs.values())
    assert not destination.exists()


def test_control_cli_accepts_only_override_being_requested(tmp_path):
    root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/quantize_da_ptq.py"),
            "--checkpoint-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "unused"),
            "--mode",
            "equal_weight",
            "--bf16-ratio",
            "0.2",
            "--dry-run",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    jobs = json.loads(result.stdout)
    assert list(jobs) == ["equal_weight"]
    assert jobs["equal_weight"]["bf16_ratio"] == 0.2
    assert jobs["equal_weight"]["csrc_enabled"] is True
