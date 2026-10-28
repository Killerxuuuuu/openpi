"""Paired LIBERO evaluation with unique traces and machine-readable outcomes.

This client can run in the separate LIBERO Python environment used by openpi.
Calibration and policy serving stay in the main PyTorch environment.
"""

import argparse
import collections
import json
import logging
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], default="libero_spatial"
    )
    parser.add_argument("--num-trials", type=int, default=50, help="Trials per task per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-max-trials", type=int, default=0, help="Videos per task/seed; 0 saves every trial")
    args = parser.parse_args()
    if (
        args.num_trials < 1
        or args.replan_steps < 1
        or args.wait_steps < 0
        or args.video_max_trials < 0
        or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("Invalid trial/replan/wait counts or duplicate seeds")

    import imageio
    from libero.libero import benchmark

    # Reuse the upstream environment construction and quaternion convention.
    from main import LIBERO_DUMMY_ACTION
    from main import LIBERO_ENV_RESOLUTION
    from main import _get_libero_env
    from main import _quat2axisangle
    import numpy as np
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}[args.suite]
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    run_dir = args.output_dir / args.suite
    # Refuse accidental overwrites/mixing partial results from another policy.
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        **vars(args),
        "output_dir": str(args.output_dir),
        "server": metadata,
        "rendering": {key: os.environ.get(key) for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "LP_NUM_THREADS")},
    }
    (run_dir / "evaluation_config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    records = []

    def write_summary():
        valid = [record for record in records if record["termination"] != "error"]
        task_rates = []
        for task_id in range(suite.n_tasks):
            subset = [record for record in valid if record["task_id"] == task_id]
            if subset:
                task_rates.append(sum(record["success"] for record in subset) / len(subset))
        summary = {
            "suite": args.suite,
            "completed_episodes": len(records),
            "valid_episodes": len(valid),
            "invalid_episodes": len(records) - len(valid),
            "expected_episodes": suite.n_tasks * len(args.seeds) * args.num_trials,
            "successes": sum(record["success"] for record in valid),
            "success_rate": sum(record["success"] for record in valid) / len(valid) if valid else None,
            "task_average_success_rate": float(np.mean(task_rates)) if task_rates else None,
            "episodes": records,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for seed in args.seeds:
        np.random.seed(seed)
        for task_id in range(suite.n_tasks):
            initial_states = suite.get_task_init_states(task_id)
            if args.num_trials > len(initial_states):
                raise ValueError("Requested more trials than available initial states; use multiple --seeds instead")
            env, description = _get_libero_env(suite.get_task(task_id), LIBERO_ENV_RESOLUTION, seed)
            try:
                for trial in range(args.num_trials):
                    save_video = args.save_video and (args.video_max_trials == 0 or trial < args.video_max_trials)
                    prefix = f"seed_{seed}_task_{task_id:02d}_trial_{trial:03d}"
                    env.reset()
                    obs = env.set_init_state(initial_states[trial])
                    action_plan = collections.deque()
                    done, error = False, None
                    actions, positions, quaternions, grippers = [], [], [], []
                    call_ms, noise_seeds, frames, reset_seconds = [], [], [], 0.0
                    start = time.perf_counter()
                    try:
                        wait_start = time.perf_counter()
                        for _ in range(args.wait_steps):
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        reset_seconds = time.perf_counter() - wait_start
                        for _ in range(max_steps):
                            if not action_plan:
                                img = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(
                                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224
                                    )
                                )
                                wrist = image_tools.convert_to_uint8(
                                    image_tools.resize_with_pad(
                                        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224
                                    )
                                )
                                noise_seed = int(
                                    np.random.SeedSequence([seed, task_id, trial, len(call_ms)]).generate_state(1)[0]
                                )
                                element = {
                                    "observation/image": img,
                                    "observation/wrist_image": wrist,
                                    "observation/state": np.concatenate(
                                        (
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"].copy()),
                                            obs["robot0_gripper_qpos"],
                                        )
                                    ),
                                    "prompt": str(description),
                                    "_da_ptq_noise_seed": noise_seed,
                                }
                                request_start = time.perf_counter()
                                chunk = np.asarray(client.infer(element)["actions"])
                                call_ms.append((time.perf_counter() - request_start) * 1000)
                                noise_seeds.append(noise_seed)
                                if (
                                    chunk.ndim != 2
                                    or chunk.shape[1] != 7
                                    or len(chunk) < args.replan_steps
                                    or not np.isfinite(chunk).all()
                                ):
                                    raise ValueError(f"Invalid action chunk shape/content: {chunk.shape}")
                                action_plan.extend(chunk[: args.replan_steps])
                            action = action_plan.popleft()
                            obs, _, done, _ = env.step(action.tolist())
                            actions.append(action.copy())
                            positions.append(obs["robot0_eef_pos"].copy())
                            quaternions.append(obs["robot0_eef_quat"].copy())
                            grippers.append(obs["robot0_gripper_qpos"].copy())
                            if save_video:
                                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                            if done:
                                break
                    except Exception as exc:
                        error = repr(exc)
                    elapsed = time.perf_counter() - start
                    record = {
                        "seed": seed,
                        "task_id": task_id,
                        "trial": trial,
                        "instruction": str(description),
                        "success": bool(done) and error is None,
                        "termination": "error" if error else "success" if done else "timeout",
                        "error": error,
                        "control_steps": len(actions),
                        "episode_seconds": elapsed,
                        "stabilization_seconds": reset_seconds,
                        "policy_roundtrip_ms": call_ms,
                        "trace": prefix + ".npz",
                    }
                    np.savez_compressed(
                        run_dir / (prefix + ".npz"),
                        action=np.asarray(actions, dtype=np.float32).reshape(-1, 7),
                        eef_position=np.asarray(positions, dtype=np.float32).reshape(-1, 3),
                        eef_quaternion=np.asarray(quaternions, dtype=np.float32).reshape(-1, 4),
                        gripper_qpos=np.asarray(grippers, dtype=np.float32),
                        initial_state=np.asarray(initial_states[trial]),
                        policy_noise_seeds=np.asarray(noise_seeds, dtype=np.uint32),
                    )
                    records.append(record)
                    write_summary()
                    logging.info("%s %s success=%s steps=%d", args.suite, prefix, record["success"], len(actions))
                    if save_video and frames:
                        imageio.mimwrite(run_dir / (prefix + ".mp4"), frames, fps=10)
                    if error:
                        # Abort instead of disguising server/config errors as
                        # genuine zero-success policy failures.
                        raise RuntimeError(f"Invalid episode {prefix}: {error}")
            finally:
                env.close()
    write_summary()


if __name__ == "__main__":
    main()
