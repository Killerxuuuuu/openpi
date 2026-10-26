"""Evaluate calibrated variants with the same four LIBERO suites and seeds."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bf16-checkpoint", type=Path, help="Optional original converted checkpoint")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["uniform", "da_ptq", "equal_weight", "random_seed_0", "random_seed_1", "random_seed_2"],
    )
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--eval-python", default=sys.executable, help="Python executable in the LIBERO environment")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    jobs = [(name, (args.runs_dir / name).resolve()) for name in args.variants]
    if args.bf16_checkpoint is not None:
        jobs.insert(0, ("bf16", args.bf16_checkpoint.resolve()))
    for name, checkpoint in jobs:
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing converted/calibrated checkpoint for {name}: {checkpoint}")
        if name != "bf16" and not (checkpoint / "da_ptq.json").is_file():
            raise ValueError(f"Not a completed DA-PTQ artifact: {checkpoint}")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to avoid mixing episode sets")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=1)
    except (urllib.error.URLError, TimeoutError):
        pass
    else:
        raise RuntimeError(f"Port {args.port} already has a policy server")
    args.output_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "packages/openpi-client/src"), environment.get("PYTHONPATH", "")]
    )
    for name, checkpoint in jobs:
        job_dir = args.output_dir / name
        job_dir.mkdir()
        with (job_dir / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "scripts/serve_da_ptq.py"),
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--device",
                    args.device,
                    "--port",
                    str(args.port),
                ],
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server for {name} exited; see {job_dir / 'server.log'}")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=1):
                            break
                    except (urllib.error.URLError, TimeoutError) as exc:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"Policy startup timed out for {name}") from exc
                        time.sleep(0.5)
                for suite in args.suites:
                    subprocess.run(
                        [
                            args.eval_python,
                            str(root / "examples/libero/eval_da_ptq.py"),
                            "--suite",
                            suite,
                            "--port",
                            str(args.port),
                            "--num-trials",
                            str(args.num_trials),
                            "--output-dir",
                            str(job_dir.resolve()),
                            "--seeds",
                            *map(str, args.seeds),
                        ],
                        cwd=root,
                        env=environment,
                        check=True,
                    )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    summaries = {}
    for name, _ in jobs:
        suites = {}
        for suite in args.suites:
            summary = json.loads((args.output_dir / name / suite / "summary.json").read_text())
            if summary["invalid_episodes"] or summary["completed_episodes"] != summary["expected_episodes"]:
                raise RuntimeError(f"Incomplete/invalid evaluation: {name}/{suite}")
            suites[suite] = summary["task_average_success_rate"]
        summaries[name] = {"suite_success_rates": suites, "average": sum(suites.values()) / len(suites)}
    (args.output_dir / "results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
