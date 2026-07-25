"""Shared constants, schema helpers, and I/O for the VoxelScan pipeline.

Conventions used everywhere in this project:

* Volumes are 3D NumPy arrays indexed as ``volume[z, y, x]`` (axis order
  ``zyx``). ``z`` is treated as depth into the pipe wall, ``y``/``x`` are the
  in-plane scan coordinates. Every coordinate we store or report is in this
  same ``zyx`` order so ground truth and detections line up without juggling.
* Intensities are stored as ``float32`` on a nominal 0-255 acoustic-amplitude
  scale. These are *simulated* backscatter amplitudes, not calibrated physics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# --- Defect taxonomy --------------------------------------------------------
# The four defect classes we plant and later try to recover. Ordered so that
# indices are stable across the project (used for colour maps / class ids).
DEFECT_TYPES = ["pitting", "crack", "inclusion", "wall-thinning"]

# Colours used for 3D overlays (Phase 3) and report tables (Phase 4). Kept
# here so the whole pipeline agrees on which colour means which defect type.
TYPE_COLORS = {
    "pitting": "#e6194b",       # red
    "crack": "#f58231",         # orange
    "inclusion": "#4363d8",     # blue
    "wall-thinning": "#3cb44b", # green
    "unknown": "#808080",       # grey (unmatched / unclassified)
}

AXIS_ORDER = "zyx"

# --- Repo-relative paths ----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
VOLUMES_DIR = DATA_DIR / "volumes"       # float32 intensity volumes (.npy)
LABELS_DIR = DATA_DIR / "gt_labels"      # ground-truth label volumes (.npz)
MANIFESTS_DIR = DATA_DIR / "manifests"   # ground-truth manifests (.json)
PREVIEWS_DIR = DATA_DIR / "previews"     # slice preview PNGs from generation


def ensure_dirs(*dirs: Path) -> None:
    """Create the given directories (and parents) if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# --- JSON helpers -----------------------------------------------------------
def _json_default(obj):
    """Make NumPy scalars/arrays JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(obj, path: Path) -> None:
    ensure_dirs(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# --- Volume I/O -------------------------------------------------------------
def save_volume(volume: np.ndarray, path: Path) -> None:
    ensure_dirs(Path(path).parent)
    np.save(path, volume.astype(np.float32))


def load_volume(path: Path) -> np.ndarray:
    return np.load(path)


# --- Severity index ---------------------------------------------------------
# A simple, transparent index combining how big a defect is with how strongly
# it deviates from the baseline. This is an ANALYSIS ranking aid, NOT a coded
# engineering severity grade - reports label it as such to avoid overclaiming.
# Thresholds are calibrated to ~tertiles of the confirmed-defect population in
# the generated set (p33~300, p66~530), so labels split roughly into thirds.
SEVERITY_HIGH = 530.0
SEVERITY_MEDIUM = 300.0


def severity_score(intensity_delta: float, equivalent_diameter: float) -> float:
    """Severity index = |intensity delta| x equivalent diameter."""
    return abs(float(intensity_delta)) * float(equivalent_diameter)


def severity_label(score: float) -> str:
    if score >= SEVERITY_HIGH:
        return "high"
    if score >= SEVERITY_MEDIUM:
        return "medium"
    return "low"


def save_labels(labels: np.ndarray, path: Path) -> None:
    """Save an integer label volume. It is mostly zeros, so compress it."""
    ensure_dirs(Path(path).parent)
    np.savez_compressed(path, labels=labels.astype(np.int32))


def load_labels(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["labels"]
