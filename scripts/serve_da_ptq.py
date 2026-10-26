"""Serve a BF16 or DA-PTQ pi05 checkpoint with per-request paired noise seeds."""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import numpy as np

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as config_api

    logging.basicConfig(level=logging.INFO, force=True)
    config = config_api.get_config(args.config_name)
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir, pytorch_device=args.device)
    manifest_path = args.checkpoint_dir / "da_ptq.json"
    experiment = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"config": {"mode": "bf16"}}
    metadata = {
        **policy.metadata,
        "da_ptq": {
            "mode": experiment["config"]["mode"],
            "allocation_seed": experiment["config"].get("allocation_seed"),
            "action_dim": config.model.action_dim,
            "action_horizon": config.model.action_horizon,
            "checkpoint": str(args.checkpoint_dir.resolve()),
            "compile": False,
            "fake_quantization": manifest_path.exists(),
        },
    }

    class PairedNoisePolicy:
        def infer(self, observation):
            observation = dict(observation)
            seed = observation.pop("_da_ptq_noise_seed", None)
            kwargs = {}
            if seed is not None:
                shape = (config.model.action_horizon, config.model.action_dim)
                kwargs["noise"] = np.random.default_rng(int(seed)).standard_normal(shape).astype(np.float32)
            return policy.infer(observation, **kwargs)

    websocket_policy_server.WebsocketPolicyServer(
        PairedNoisePolicy(), host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
