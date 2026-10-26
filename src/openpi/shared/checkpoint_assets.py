"""Locate normalization assets for root-based and legacy checkpoint layouts."""

from pathlib import Path


def find_checkpoint_assets(checkpoint_dir: str | Path) -> Path | None:
    checkpoint = Path(checkpoint_dir)
    # Official pi05_libero download layout: checkpoint/{params,assets}.
    # Keep the older parent-assets convention as a fallback only.
    for candidate in (checkpoint / "assets", checkpoint.parent / "assets"):
        if candidate.is_dir():
            return candidate
    return None
