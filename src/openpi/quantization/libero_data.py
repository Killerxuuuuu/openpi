"""Fixed, replayable LIBERO demonstration frames with official pi05 transforms."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def choose_indices(task_ids, positions, *, samples, seed, spatial_bins, warmup_samples, balanced=True):
    """Round-robin tasks and spatial bins using workspace positions, not actions."""
    rng = np.random.default_rng(seed)
    if not balanced:
        return rng.choice(len(task_ids), size=samples, replace=False).tolist(), []
    warmup = rng.choice(len(task_ids), min(warmup_samples, len(task_ids)), replace=False)
    lower, upper = positions[warmup].min(0), positions[warmup].max(0)
    coordinates = np.floor((positions - lower) / np.maximum(upper - lower, 1e-8) * spatial_bins)
    coordinates = np.clip(coordinates, 0, spatial_bins - 1).astype(int)
    bin_ids = (coordinates * np.array([spatial_bins**2, spatial_bins, 1])).sum(1)
    task_sequences = []
    for task in np.unique(task_ids):
        task_indices = np.flatnonzero(task_ids == task)
        bins = [
            rng.permutation(task_indices[bin_ids[task_indices] == bin_id])
            for bin_id in rng.permutation(np.unique(bin_ids[task_indices]))
        ]
        sequence = [
            int(bucket[offset]) for offset in range(max(map(len, bins))) for bucket in bins if offset < len(bucket)
        ]
        task_sequences.append(sequence)
    indices = []
    for offset in range(max(map(len, task_sequences))):
        for sequence in task_sequences:
            if offset < len(sequence):
                indices.append(sequence[offset])
            if len(indices) == samples:
                rng.shuffle(indices)
                return indices, warmup.tolist()
    raise ValueError("Not enough distinct calibration frames")


class LiberoCalibrationData:
    def __init__(
        self,
        train_config,
        checkpoint_dir,
        *,
        samples,
        batch_size,
        seed,
        root=None,
        selection_file=None,
        task_balanced=True,
        spatial_bins=6,
        spatial_warmup_samples=128,
    ):
        # Only this adapter needs openpi's full JAX/LeRobot environment.
        import jax
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        from openpi import transforms
        from openpi.models import model as model_api
        from openpi.training import checkpoints
        from openpi.training import data_loader

        self._jax, self._model_api = jax, model_api
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.repo_id is None or data_config.asset_id is None:
            raise ValueError("LIBERO repo_id and checkpoint asset_id are required")
        norm_stats = checkpoints.load_norm_stats(Path(checkpoint_dir) / "assets", data_config.asset_id)
        data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
        dataset_root = None if root is None else Path(root)
        metadata = LeRobotDatasetMetadata(data_config.repo_id, root=dataset_root)
        self.raw = LeRobotDataset(
            data_config.repo_id,
            root=dataset_root,
            delta_timestamps={
                key: [t / metadata.fps for t in range(train_config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        dataset = self.raw
        if data_config.prompt_from_task:
            dataset = data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(metadata.tasks)])
        self.dataset = data_loader.transform_dataset(dataset, data_config)
        self.batch_size = batch_size
        if not 0 < samples <= len(dataset):
            raise ValueError(f"Requested {samples} distinct calibration frames from {len(dataset)} available")
        task_ids = np.asarray(self.raw.hf_dataset["task_index"]).reshape(-1).astype(int)
        positions = np.asarray(self.raw.hf_dataset["state"], dtype=np.float64)[:, :3]
        if positions.shape != (len(dataset), 3) or not np.isfinite(positions).all():
            raise ValueError("LIBERO state must contain finite Cartesian end-effector positions")
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "repo": data_config.repo_id,
                    "frames": len(dataset),
                    "tasks": task_ids.tolist(),
                    "hf_fingerprint": getattr(self.raw.hf_dataset, "_fingerprint", None),
                    "position_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if selection_file is not None:
            payload = json.loads(Path(selection_file).read_text(encoding="utf-8"))
            if payload["dataset_fingerprint"] != fingerprint:
                raise ValueError("Saved calibration selection belongs to a different dataset")
            indices = payload["indices"]
            warmup = payload.get("spatial_warmup_indices", [])
            if (
                len(indices) != samples
                or len(set(indices)) != samples
                or min(indices) < 0
                or max(indices) >= len(dataset)
            ):
                raise ValueError("Saved calibration selection has an invalid size or indices")
        else:
            indices, warmup = choose_indices(
                task_ids,
                positions,
                samples=samples,
                seed=seed,
                spatial_bins=spatial_bins,
                warmup_samples=spatial_warmup_samples,
                balanced=task_balanced,
            )
        self.indices = indices
        self.manifest = {
            "repo_id": data_config.repo_id,
            "dataset_root": str(dataset_root),
            "dataset_fingerprint": fingerprint,
            "sampling_unit": "frame_with_action_chunk",
            "task_balanced": task_balanced,
            "seed": seed,
            "indices": indices,
            "task_ids": task_ids[indices].tolist(),
            "batch_size": batch_size,
            "spatial_bins": spatial_bins,
            "spatial_warmup_indices": warmup,
            "normalization": "original_checkpoint_assets",
            "action_horizon": train_config.model.action_horizon,
        }

    def __call__(self):
        # No random augmentation is applied here or in calibration forward.
        for start in range(0, len(self.indices), self.batch_size):
            samples = [self.dataset[index] for index in self.indices[start : start + self.batch_size]]
            batch = self._jax.tree.map(
                lambda *items: torch.as_tensor(np.stack([np.asarray(x) for x in items])), *samples
            )
            yield self._model_api.Observation.from_dict(batch), batch["actions"]
