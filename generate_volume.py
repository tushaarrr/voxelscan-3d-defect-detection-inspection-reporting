"""Phase 1 - Synthetic volume generator (the controlled ground truth).

Creates 3D volumes that simulate an acoustic/ultrasound scan of a pipe-wall
section, plants defects of known type/location/size, and writes a ground-truth
manifest so every downstream number (recall, precision, sizing error) is
measurable against something we planted on purpose.

IMPORTANT: this is *simulated* volumetric scan data. It models the analysis
workflow (find, measure, visualise, report millimetre-scale defects in dense
volumetric data), not the underlying acoustic physics.

Run ``python generate_volume.py`` to build the default set of 20 volumes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

import common
from common import (
    DEFECT_TYPES,
    VOLUMES_DIR,
    LABELS_DIR,
    MANIFESTS_DIR,
    PREVIEWS_DIR,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DefectSpec:
    """Base geometry/contrast for one defect class, before difficulty scaling.

    ``radius_zyx`` gives (min, max) radius ranges per axis in voxels. ``delta``
    is the (min, max) intensity offset added to the baseline inside the defect
    (sign encodes bright vs. dark anomalies). ``thin_axis`` optionally forces
    one axis to be thin (used for planar cracks / broad wall-thinning).
    """
    radius_zyx: tuple  # ((zmin, zmax), (ymin, ymax), (xmin, xmax))
    delta: tuple       # (min, max) signed intensity delta
    thin_axis_choices: tuple = ()  # axes eligible to be the "thin" one


# Per-type base parameters. Signs are chosen so the set contains both bright
# (inclusion, crack) and dark (pitting, wall-thinning) anomalies, which forces
# the detector to look at deviations in *both* directions.
DEFECT_SPECS = {
    "pitting": DefectSpec(
        radius_zyx=((2.5, 4.5), (2.5, 4.5), (2.5, 4.5)),
        delta=(-40.0, -28.0),
    ),
    "crack": DefectSpec(
        # Planar: two long radii, one thin radius (the thin axis varies).
        radius_zyx=((6.0, 13.0), (6.0, 13.0), (6.0, 13.0)),
        delta=(32.0, 46.0),
        thin_axis_choices=(0, 1, 2),
    ),
    "inclusion": DefectSpec(
        radius_zyx=((3.5, 6.0), (3.5, 6.0), (3.5, 6.0)),
        delta=(38.0, 50.0),
    ),
    "wall-thinning": DefectSpec(
        # Broad, shallow slab: large in-plane, thin in depth -> subtle & wide.
        radius_zyx=((2.5, 4.5), (10.0, 20.0), (10.0, 20.0)),
        delta=(-26.0, -16.0),
    ),
}


@dataclass
class VolumeConfig:
    """Everything needed to reproduce one volume."""
    volume_id: str
    seed: int
    difficulty: float  # 0 (easy) .. 1 (hard); recorded for reference
    shape: tuple = (128, 128, 128)  # (z, y, x)
    baseline: float = 120.0         # nominal material amplitude at the surface
    attenuation: float = 0.30       # fractional amplitude loss across full depth
    noise_std: float = 6.0          # gaussian backscatter/speckle std
    contrast_scale: float = 1.0     # multiplies each defect's intensity delta
    size_scale: float = 1.0         # multiplies each defect's radii
    n_defects: int = 5
    margin: int = 8                 # keep defect centres this far from edges
    edge_taper: float = 0.15        # soft-edge fraction on defect boundaries


def make_config(volume_id: str, difficulty: float, rng: np.random.Generator) -> VolumeConfig:
    """Build a config for a given difficulty in [0, 1].

    Higher difficulty -> more noise, lower contrast, smaller defects. The
    number of defects is randomised so the test set is not uniform.
    """
    t = float(np.clip(difficulty, 0.0, 1.0))
    return VolumeConfig(
        volume_id=volume_id,
        seed=int(rng.integers(0, 2**31 - 1)),
        difficulty=t,
        # Wide gradient: the hard end is deliberately low-SNR / small / low-
        # contrast so detection recall degrades (a flat 100% would not be a
        # meaningful measured result).
        noise_std=float(np.interp(t, [0, 1], [4.0, 22.0])),
        contrast_scale=float(np.interp(t, [0, 1], [1.30, 0.40])),
        size_scale=float(np.interp(t, [0, 1], [1.35, 0.55])),
        n_defects=int(rng.integers(3, 9)),  # 3..8 inclusive
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _sample_radii(spec: DefectSpec, size_scale: float, rng: np.random.Generator):
    """Sample per-axis radii (voxels) for one defect, applying size scaling."""
    radii = np.array([rng.uniform(lo, hi) for (lo, hi) in spec.radius_zyx])
    radii *= size_scale
    if spec.thin_axis_choices:
        thin_axis = int(rng.choice(spec.thin_axis_choices))
        radii[thin_axis] = max(1.0, radii[thin_axis] * 0.18)  # squash to a plane
    return np.maximum(radii, 1.0)


def _ellipsoid_masks(shape, center, radii, taper):
    """Return (core_mask, soft_mask, region_slices) for an axis-aligned ellipsoid.

    Work only inside a local bounding box for speed. ``core_mask`` is the crisp
    boolean ground-truth shape; ``soft_mask`` (0..1) has a linear edge taper so
    the applied intensity change is not perfectly sharp (more realistic).
    """
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    edge = 1.0 + taper  # normalised radius where the soft mask reaches 0

    # Local bounding box padded by the taper.
    lo = np.floor(center - radii * edge - 1).astype(int)
    hi = np.ceil(center + radii * edge + 1).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array(shape))
    slices = tuple(slice(lo[a], hi[a]) for a in range(3))

    coords = np.ogrid[slices[0], slices[1], slices[2]]
    q2 = np.zeros([hi[a] - lo[a] for a in range(3)], dtype=float)
    for a in range(3):
        q2 = q2 + ((coords[a] - center[a]) / radii[a]) ** 2
    q = np.sqrt(q2)

    core_mask = q <= 1.0
    soft_mask = np.clip((edge - q) / (edge - 1.0), 0.0, 1.0)
    return core_mask, soft_mask, slices


# ---------------------------------------------------------------------------
# Volume assembly
# ---------------------------------------------------------------------------
def _base_material(cfg: VolumeConfig, rng: np.random.Generator) -> np.ndarray:
    """Baseline material amplitude with depth attenuation + gaussian speckle.

    The depth attenuation gives a spatially varying baseline, which is exactly
    why a *local* baseline (rather than one global threshold) is needed to
    detect defects later.
    """
    z, y, x = cfg.shape
    depth = np.linspace(0.0, 1.0, z, dtype=np.float32)
    baseline = cfg.baseline * (1.0 - cfg.attenuation * depth)  # (z,)
    volume = np.repeat(baseline[:, None, None], y, axis=1)
    volume = np.repeat(volume, x, axis=2)
    volume = volume + rng.normal(0.0, cfg.noise_std, size=cfg.shape).astype(np.float32)
    return volume.astype(np.float32)


def _place_defects(cfg: VolumeConfig, rng: np.random.Generator):
    """Pick non-overlapping defect placements. Returns a list of placement dicts."""
    placements = []
    occupied_boxes = []  # (lo_zyx, hi_zyx) with padding, to reject overlaps
    attempts = 0
    max_attempts = cfg.n_defects * 40

    while len(placements) < cfg.n_defects and attempts < max_attempts:
        attempts += 1
        dtype = str(rng.choice(DEFECT_TYPES))
        spec = DEFECT_SPECS[dtype]
        radii = _sample_radii(spec, cfg.size_scale, rng)

        # Centre must keep the whole ellipsoid inside the margin.
        lo_bound = np.maximum(radii, cfg.margin)
        hi_bound = np.array(cfg.shape) - lo_bound
        if np.any(hi_bound <= lo_bound):
            continue  # defect too big for this volume; resample
        center = np.array([rng.uniform(lo_bound[a], hi_bound[a]) for a in range(3)])

        # Reject if its padded bbox overlaps an existing defect.
        pad = 3.0
        box_lo = center - radii - pad
        box_hi = center + radii + pad
        if any(_boxes_overlap(box_lo, box_hi, o_lo, o_hi) for o_lo, o_hi in occupied_boxes):
            continue
        occupied_boxes.append((box_lo, box_hi))

        delta = float(rng.uniform(*spec.delta)) * cfg.contrast_scale
        placements.append({"type": dtype, "center": center, "radii": radii, "delta": delta})

    return placements


def _boxes_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    return bool(np.all(a_lo < b_hi) and np.all(b_lo < a_hi))


def generate_volume(cfg: VolumeConfig):
    """Generate one volume + ground-truth label volume + manifest.

    Returns ``(volume, gt_labels, manifest)``.
    """
    rng = np.random.default_rng(cfg.seed)
    volume = _base_material(cfg, rng)
    gt_labels = np.zeros(cfg.shape, dtype=np.int32)
    placements = _place_defects(cfg, rng)

    defects = []
    for i, p in enumerate(placements):
        core, soft, slc = _ellipsoid_masks(cfg.shape, p["center"], p["radii"], cfg.edge_taper)
        if not core.any():
            continue

        # Apply the (soft-edged) intensity change to the volume.
        volume[slc] += (p["delta"] * soft).astype(np.float32)

        # Record ground truth from the crisp core mask.
        label_id = i + 1
        sub_labels = gt_labels[slc]
        sub_labels[core] = label_id  # non-overlapping by construction

        idx = np.argwhere(core)
        idx_global = idx + np.array([slc[a].start for a in range(3)])
        centroid = idx_global.mean(axis=0)
        bbox_lo = idx_global.min(axis=0)
        bbox_hi = idx_global.max(axis=0) + 1  # exclusive upper bound
        size_voxels = int(core.sum())
        eq_diameter = float((6.0 * size_voxels / np.pi) ** (1.0 / 3.0))

        defects.append({
            "defect_id": f"{cfg.volume_id}_d{i}",
            "label_id": label_id,
            "type": p["type"],
            "center_zyx": [round(float(c), 2) for c in centroid],
            "bbox_zyx": [int(v) for v in np.concatenate([bbox_lo, bbox_hi])],
            "size_voxels": size_voxels,
            "equivalent_diameter": round(eq_diameter, 2),
            "intensity_delta": round(float(p["delta"]), 2),
        })

    volume = np.clip(volume, 0.0, 255.0).astype(np.float32)

    manifest = {
        "volume_id": cfg.volume_id,
        "shape": list(cfg.shape),
        "axis_order": common.AXIS_ORDER,
        "data_note": "SIMULATED volumetric scan data - models the analysis "
                     "workflow, not acoustic physics.",
        "config": _config_summary(cfg),
        "n_defects": len(defects),
        "defects": defects,
    }
    return volume, gt_labels, manifest


def _config_summary(cfg: VolumeConfig) -> dict:
    """A JSON-friendly, rounded view of the config for the manifest."""
    d = asdict(cfg)
    for k in ("baseline", "attenuation", "noise_std", "contrast_scale",
              "size_scale", "difficulty", "edge_taper"):
        d[k] = round(float(d[k]), 3)
    return d


# ---------------------------------------------------------------------------
# Preview rendering (Phase 1 sanity check)
# ---------------------------------------------------------------------------
def save_preview(volume, gt_labels, manifest, path):
    """Save a 2-panel PNG: the geometric mid-slice and the most-defective slice.

    Ground-truth bounding boxes that intersect the shown slice are overlaid.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    common.ensure_dirs(path.parent)
    z_mid = volume.shape[0] // 2

    # Slice (along z) that contains the most defect voxels, for guaranteed
    # visibility. Falls back to the mid-slice if nothing was planted.
    per_slice = (gt_labels > 0).reshape(volume.shape[0], -1).sum(axis=1)
    z_defect = int(per_slice.argmax()) if per_slice.max() > 0 else z_mid

    vmin, vmax = np.percentile(volume, [1, 99])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, z, title in ((axes[0], z_mid, f"Mid-slice z={z_mid}"),
                         (axes[1], z_defect, f"Most-defective slice z={z_defect}")):
        ax.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        for d in manifest["defects"]:
            zmin, ymin, xmin, zmax, ymax, xmax = d["bbox_zyx"]
            if zmin <= z < zmax:  # bbox intersects this slice
                color = common.TYPE_COLORS.get(d["type"], "#ffffff")
                ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, edgecolor=color, linewidth=1.5))
                ax.text(xmin, ymax + 1, d["type"], color=color, fontsize=7)

    diff = manifest["config"]["difficulty"]
    fig.suptitle(f"{manifest['volume_id']}  |  difficulty={diff:.2f}  |  "
                 f"{manifest['n_defects']} planted defects  (SIMULATED data)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / batch generation
# ---------------------------------------------------------------------------
def generate_set(n_volumes: int, master_seed: int, make_previews: bool = True):
    """Generate ``n_volumes`` spanning an easy->hard difficulty gradient."""
    common.ensure_dirs(VOLUMES_DIR, LABELS_DIR, MANIFESTS_DIR, PREVIEWS_DIR)
    rng = np.random.default_rng(master_seed)
    difficulties = np.linspace(0.0, 1.0, n_volumes)

    summaries = []
    for i, diff in enumerate(difficulties):
        vol_id = f"vol_{i:03d}"
        cfg = make_config(vol_id, float(diff), rng)
        volume, gt_labels, manifest = generate_volume(cfg)

        common.save_volume(volume, VOLUMES_DIR / f"{vol_id}.npy")
        common.save_labels(gt_labels, LABELS_DIR / f"{vol_id}.npz")
        common.save_json(manifest, MANIFESTS_DIR / f"{vol_id}.json")
        if make_previews:
            save_preview(volume, gt_labels, manifest, PREVIEWS_DIR / f"{vol_id}.png")

        counts = {t: 0 for t in DEFECT_TYPES}
        for d in manifest["defects"]:
            counts[d["type"]] += 1
        summaries.append((vol_id, cfg.difficulty, cfg.noise_std, manifest["n_defects"], counts))

    _print_summary(summaries)
    return summaries


def _print_summary(summaries):
    print(f"\nGenerated {len(summaries)} volumes -> {VOLUMES_DIR}\n")
    header = f"{'volume':<9}{'diff':>6}{'noise':>7}{'defects':>9}   by type"
    print(header)
    print("-" * len(header))
    total = 0
    type_totals = {t: 0 for t in DEFECT_TYPES}
    for vol_id, diff, noise, n, counts in summaries:
        total += n
        for t in DEFECT_TYPES:
            type_totals[t] += counts[t]
        by_type = ", ".join(f"{t}:{counts[t]}" for t in DEFECT_TYPES if counts[t])
        print(f"{vol_id:<9}{diff:>6.2f}{noise:>7.1f}{n:>9}   {by_type}")
    print("-" * len(header))
    print(f"{'TOTAL':<9}{'':>6}{'':>7}{total:>9}   " +
          ", ".join(f"{t}:{type_totals[t]}" for t in DEFECT_TYPES))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 3D scan volumes with planted defects.")
    parser.add_argument("-n", "--n-volumes", type=int, default=20,
                        help="number of volumes to generate (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="master RNG seed")
    parser.add_argument("--no-previews", action="store_true", help="skip PNG previews")
    args = parser.parse_args()
    generate_set(args.n_volumes, args.seed, make_previews=not args.no_previews)


if __name__ == "__main__":
    main()
