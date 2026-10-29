"""Unmodified action-XY quota sampler from the local CogACT entrypoint."""
# ruff: noqa -- preserve reference definitions
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from openpi.quantization.cogact_primitives import clone_batch_to_cpu
SOURCE_SHA256 = "147b916997b1a20774cdf431c2df8e61e4c76b26ece77b919305c5571daba1a0"

def _extract_action_xy_from_batch(batch_cpu: Dict[str, Any], action_timestep: int) -> Optional[Tuple[float, float]]:
    actions = batch_cpu.get("actions", None)
    if not torch.is_tensor(actions):
        return None

    data = actions.detach().to(torch.float32)
    if data.ndim == 3:
        timestep = max(0, min(int(action_timestep), int(data.shape[1]) - 1))
        if data.shape[2] < 2:
            return None
        xy = data[:, timestep, :2]
    elif data.ndim == 2:
        if data.shape[1] < 2:
            return None
        xy = data[:, :2]
    else:
        return None

    if xy.numel() == 0:
        return None
    return float(xy[:, 0].mean().item()), float(xy[:, 1].mean().item())


def _build_spatial_edges(coords: Sequence[Tuple[float, float]], bins: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if len(coords) == 0:
        return None

    xs = torch.tensor([c[0] for c in coords], dtype=torch.float32)
    ys = torch.tensor([c[1] for c in coords], dtype=torch.float32)

    x_min, x_max = float(xs.min().item()), float(xs.max().item())
    y_min, y_max = float(ys.min().item()), float(ys.max().item())

    if abs(x_max - x_min) < 1e-8:
        x_min -= 1e-3
        x_max += 1e-3
    if abs(y_max - y_min) < 1e-8:
        y_min -= 1e-3
        y_max += 1e-3

    x_edges = torch.linspace(x_min, x_max, int(bins) + 1, dtype=torch.float32)
    y_edges = torch.linspace(y_min, y_max, int(bins) + 1, dtype=torch.float32)
    return x_edges, y_edges


def _spatial_bin_index(
    coord: Optional[Tuple[float, float]],
    edges: Optional[Tuple[torch.Tensor, torch.Tensor]],
    bins: int,
) -> Optional[int]:
    if coord is None or edges is None:
        return None

    x_edges, y_edges = edges
    x, y = coord
    x_bin = int(torch.bucketize(torch.tensor(x, dtype=torch.float32), x_edges[1:-1], right=False).item())
    y_bin = int(torch.bucketize(torch.tensor(y, dtype=torch.float32), y_edges[1:-1], right=False).item())

    x_bin = max(0, min(int(bins) - 1, x_bin))
    y_bin = max(0, min(int(bins) - 1, y_bin))
    return x_bin * int(bins) + y_bin


def _collect_spatial_balanced_calibration_batches(
    dataloader: DataLoader,
    calibrate_steps: int,
    bins: int,
    warmup_steps: int,
    action_timestep: int,
    calib_bar,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target = max(1, int(calibrate_steps))
    bins = max(1, int(bins))
    warmup_steps = max(1, int(warmup_steps))
    per_bin_quota = max(1, int(math.ceil(float(target) / float(bins * bins))))

    iterator = iter(dataloader)
    warmup_payload: List[Tuple[Dict[str, Any], Optional[Tuple[float, float]]]] = []
    for _ in range(warmup_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_cpu = clone_batch_to_cpu(batch)
        coord = _extract_action_xy_from_batch(batch_cpu, action_timestep=action_timestep)
        warmup_payload.append((batch_cpu, coord))

    if len(warmup_payload) == 0:
        return [], {
            "strategy": "spatial-balanced",
            "bins": bins,
            "quota": per_bin_quota,
            "warmup_steps": 0,
            "coords_found": 0,
            "selected_by_bin": {},
            "fallback_selected": 0,
        }

    warmup_coords = [coord for _, coord in warmup_payload if coord is not None]
    edges = _build_spatial_edges(warmup_coords, bins=bins)

    calibration_batches: List[Dict[str, Any]] = []
    fallback_batches: List[Dict[str, Any]] = []
    bin_counts: Dict[int, int] = defaultdict(int)

    def _try_select(batch_cpu: Dict[str, Any], coord: Optional[Tuple[float, float]]) -> None:
        if len(calibration_batches) >= target:
            return

        selected = False
        bin_idx = _spatial_bin_index(coord, edges=edges, bins=bins)
        if bin_idx is not None and bin_counts[bin_idx] < per_bin_quota:
            calibration_batches.append(batch_cpu)
            bin_counts[bin_idx] += 1
            selected = True
            if calib_bar is not None:
                calib_bar.update(1)

        if (not selected) and len(fallback_batches) < target:
            fallback_batches.append(batch_cpu)

    for batch_cpu, coord in warmup_payload:
        _try_select(batch_cpu, coord)

    while len(calibration_batches) < target:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_cpu = clone_batch_to_cpu(batch)
        coord = _extract_action_xy_from_batch(batch_cpu, action_timestep=action_timestep)
        _try_select(batch_cpu, coord)

    if len(calibration_batches) < target:
        needed = target - len(calibration_batches)
        for batch_cpu in fallback_batches[:needed]:
            calibration_batches.append(batch_cpu)
            if calib_bar is not None:
                calib_bar.update(1)

    selected_by_bin = {str(k): int(v) for k, v in sorted(bin_counts.items())}
    stats = {
        "strategy": "spatial-balanced",
        "bins": bins,
        "quota": per_bin_quota,
        "warmup_steps": len(warmup_payload),
        "coords_found": len(warmup_coords),
        "selected_by_bin": selected_by_bin,
        "fallback_selected": max(0, len(calibration_batches) - int(sum(bin_counts.values()))),
    }
    return calibration_batches, stats
