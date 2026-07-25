"""Phase 2 - Defect detection, measurement, and evaluation.

Pipeline per volume:
  1. Denoise (light gaussian) and estimate a *local baseline* per depth-slice
     (the simulated data attenuates with depth, so a single global threshold
     would fail - the baseline must follow depth).
  2. Flag anomalies where the residual (signal - baseline) deviates from the
     robust noise level in *either* direction (bright or dark).
  3. Clean up (morphology + small-object removal), label 3D connected
     components, and measure each region (centroid, bbox, voxel count,
     equivalent diameter, peak intensity delta).
  4. Assign a heuristic defect type from shape + sign.
  5. Match detections to ground truth by IoU and report recall / precision /
     false positives / sizing + localization error across the whole test set.

All reported numbers come from the actual run over the generated volumes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

import common
from common import (
    DEFECT_TYPES,
    VOLUMES_DIR,
    LABELS_DIR,
    MANIFESTS_DIR,
    DATA_DIR,
)

DETECTIONS_DIR = DATA_DIR / "detections"


@dataclass
class DetectConfig:
    denoise_sigma: float = 1.0   # light gaussian denoise
    threshold_k: float = 4.0     # anomaly if |residual| > k * robust_noise_std
    min_size: int = 20           # drop components smaller than this (voxels)
    opening_radius: int = 1      # morphological opening to kill speckle
    connectivity: int = 1        # 3D connectivity for labeling (1 = faces)
    iou_match_threshold: float = 0.10  # IoU >= this counts as a match


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _local_baseline(volume: np.ndarray) -> np.ndarray:
    """Robust per-depth-slice baseline (median over each z-slice).

    The material is the majority of every slice, so the median tracks the
    depth-attenuated baseline while ignoring the (minority) defect voxels.
    """
    z = volume.shape[0]
    med = np.median(volume.reshape(z, -1), axis=1).astype(np.float32)
    return med[:, None, None]


def _robust_sigma(residual: np.ndarray) -> float:
    """Noise std estimated from the median absolute deviation (robust to defects)."""
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(1.4826 * mad)


def detect_volume(volume: np.ndarray, cfg: DetectConfig):
    """Detect defects in one volume. Returns (detections, det_labels, residual)."""
    denoised = ndi.gaussian_filter(volume, sigma=cfg.denoise_sigma)
    baseline = _local_baseline(denoised)
    residual = denoised - baseline

    sigma = _robust_sigma(residual)
    thr = cfg.threshold_k * max(sigma, 1e-6)
    anomaly = np.abs(residual) > thr

    # Clean up: opening removes isolated speckle, then drop tiny components.
    if cfg.opening_radius > 0:
        anomaly = morphology.binary_opening(anomaly, morphology.ball(cfg.opening_radius))
    anomaly = morphology.remove_small_objects(anomaly, min_size=cfg.min_size)

    det_labels = measure.label(anomaly, connectivity=cfg.connectivity)
    regions = measure.regionprops(det_labels, intensity_image=residual)

    detections = []
    # Relabel so detection ids are contiguous 1..N after any filtering.
    relabel = np.zeros(det_labels.max() + 1, dtype=np.int32)
    next_id = 0
    for r in regions:
        if r.area < cfg.min_size:
            continue
        next_id += 1
        relabel[r.label] = next_id

        # Peak signed intensity delta (the more extreme of min/max residual).
        peak = r.max_intensity if abs(r.max_intensity) >= abs(r.min_intensity) else r.min_intensity
        zmin, ymin, xmin, zmax, ymax, xmax = r.bbox
        dtype = _classify(r, peak)
        detections.append({
            "detection_id": f"det{next_id}",
            "label_id": next_id,
            "type": dtype,
            "center_zyx": [round(float(c), 2) for c in r.centroid],
            "bbox_zyx": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
            "size_voxels": int(r.area),
            "equivalent_diameter": round(float(r.equivalent_diameter), 2),
            "intensity_delta": round(float(peak), 2),
        })

    det_labels = relabel[det_labels]  # apply contiguous ids
    return detections, det_labels, residual


def _classify(region, peak_delta: float) -> str:
    """Heuristic defect type from component shape + intensity sign.

    Bright (peak > 0): thin/planar -> crack, else inclusion.
    Dark  (peak < 0): broad & flat -> wall-thinning, else pitting.
    This is a shape rule, not a trained model; classification accuracy is
    reported separately so it is not overclaimed.
    """
    zmin, ymin, xmin, zmax, ymax, xmax = region.bbox
    extents = np.array([zmax - zmin, ymax - ymin, xmax - xmin])
    min_ext, max_ext = int(extents.min()), int(extents.max())
    area = region.area

    if peak_delta >= 0:  # bright anomaly
        if min_ext <= 4 and max_ext >= 9:  # thin in one axis, extended -> planar crack
            return "crack"
        return "inclusion"
    else:  # dark anomaly
        # broad footprint, thin in the smallest axis, sizeable area -> wall-thinning
        if area >= 700 and min_ext <= 10 and max_ext >= 16:
            return "wall-thinning"
        return "pitting"


# ---------------------------------------------------------------------------
# Matching detections <-> ground truth
# ---------------------------------------------------------------------------
def _iou_matrix(gt_labels: np.ndarray, det_labels: np.ndarray):
    """IoU between every GT label and detection label via overlap histograms."""
    n_gt = int(gt_labels.max())
    n_det = int(det_labels.max())
    if n_gt == 0 or n_det == 0:
        return np.zeros((n_gt, n_det)), np.bincount(gt_labels.ravel(), minlength=n_gt + 1), \
               np.bincount(det_labels.ravel(), minlength=n_det + 1)

    gt_sizes = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)
    det_sizes = np.bincount(det_labels.ravel(), minlength=n_det + 1)

    # Overlap counts only where both are non-zero.
    both = (gt_labels > 0) & (det_labels > 0)
    pair = (gt_labels[both].astype(np.int64)) * (n_det + 1) + det_labels[both].astype(np.int64)
    counts = np.bincount(pair, minlength=(n_gt + 1) * (n_det + 1))
    inter = counts.reshape(n_gt + 1, n_det + 1)[1:, 1:]

    union = gt_sizes[1:, None] + det_sizes[None, 1:] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou, gt_sizes, det_sizes


def match(gt_defects, detections, gt_labels, det_labels, iou_thr: float):
    """Greedy IoU matching. Annotates detections in place and returns per-pair info.

    Returns a dict with tp/fp/fn, matched pairs (with iou, centroid distance,
    size error), and per-type recall counts.
    """
    iou, _, _ = _iou_matrix(gt_labels, det_labels)
    n_gt, n_det = iou.shape

    matched_gt = set()
    matched_det = set()
    pairs = []

    # Greedy: repeatedly take the highest remaining IoU above threshold.
    order = np.dstack(np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape))[0]
    for g, d in order:
        if iou[g, d] < iou_thr:
            break
        if g in matched_gt or d in matched_det:
            continue
        matched_gt.add(int(g))
        matched_det.add(int(d))

        gt = gt_defects[g]           # gt label id == g+1 by construction
        det = detections[d]          # det label id == d+1
        det["matched_gt"] = gt["defect_id"]
        det["gt_type"] = gt["type"]  # eval/training label only (not used by detector)
        det["iou"] = round(float(iou[g, d]), 3)
        gc = np.array(gt["center_zyx"]); dc = np.array(det["center_zyx"])
        pairs.append({
            "gt_id": gt["defect_id"],
            "det_id": det["detection_id"],
            "gt_type": gt["type"],
            "det_type": det["type"],
            "iou": float(iou[g, d]),
            "centroid_dist": float(np.linalg.norm(gc - dc)),
            "gt_size": gt["size_voxels"],
            "det_size": det["size_voxels"],
            "gt_diameter": gt["equivalent_diameter"],
            "det_diameter": det["equivalent_diameter"],
        })

    for d_idx, det in enumerate(detections):
        if d_idx not in matched_det:
            det["matched_gt"] = None
            det["iou"] = 0.0

    tp = len(matched_gt)
    fn = n_gt - tp
    fp = n_det - len(matched_det)
    return {"tp": tp, "fp": fp, "fn": fn, "pairs": pairs,
            "matched_gt_idx": matched_gt}


# ---------------------------------------------------------------------------
# Batch run + reporting
# ---------------------------------------------------------------------------
def run(cfg: DetectConfig, verbose: bool = True):
    common.ensure_dirs(DETECTIONS_DIR)
    manifest_paths = sorted(MANIFESTS_DIR.glob("*.json"))
    if not manifest_paths:
        raise SystemExit("No manifests found. Run generate_volume.py first.")

    per_vol = []
    all_pairs = []
    type_gt_total = {t: 0 for t in DEFECT_TYPES}
    type_gt_found = {t: 0 for t in DEFECT_TYPES}
    t_start = time.time()

    for mpath in manifest_paths:
        manifest = common.load_json(mpath)
        vol_id = manifest["volume_id"]
        volume = common.load_volume(VOLUMES_DIR / f"{vol_id}.npy")
        gt_labels = common.load_labels(LABELS_DIR / f"{vol_id}.npz")
        gt_defects = manifest["defects"]

        t0 = time.time()
        detections, det_labels, residual = detect_volume(volume, cfg)
        result = match(gt_defects, detections, gt_labels, det_labels, cfg.iou_match_threshold)
        elapsed = time.time() - t0

        # Per-type recall bookkeeping.
        for i, gt in enumerate(gt_defects):
            type_gt_total[gt["type"]] += 1
            if i in result["matched_gt_idx"]:
                type_gt_found[gt["type"]] += 1
        all_pairs.extend(result["pairs"])

        det_out = {
            "volume_id": vol_id,
            "axis_order": common.AXIS_ORDER,
            "detect_config": cfg.__dict__,
            "n_detections": len(detections),
            "metrics": {k: result[k] for k in ("tp", "fp", "fn")},
            "detections": detections,
        }
        common.save_json(det_out, DETECTIONS_DIR / f"{vol_id}.json")
        # Save detection label volume for Phase 3 overlays.
        common.save_labels(det_labels, DETECTIONS_DIR / f"{vol_id}_labels.npz")

        tp, fp, fn = result["tp"], result["fp"], result["fn"]
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        per_vol.append({
            "vol_id": vol_id, "difficulty": manifest["config"]["difficulty"],
            "noise": manifest["config"]["noise_std"], "n_gt": len(gt_defects),
            "n_det": len(detections), "tp": tp, "fp": fp, "fn": fn,
            "recall": recall, "precision": precision, "time": elapsed,
        })

    total_time = time.time() - t_start
    summary = _summarize(per_vol, all_pairs, type_gt_total, type_gt_found, cfg, total_time)
    common.save_json(summary, DATA_DIR / "metrics_summary.json")
    _save_metrics_plot(per_vol, summary, DATA_DIR / "metrics_curve.png")
    if verbose:
        _print_table(per_vol, summary)
    return per_vol, summary


def _save_metrics_plot(per_vol, summary, path):
    """Plot recall & precision vs difficulty (for the README / report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diffs = [v["difficulty"] for v in per_vol]
    recalls = [v["recall"] for v in per_vol]
    precs = [v["precision"] for v in per_vol]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(diffs, recalls, "o-", label="recall", color="#4363d8")
    ax.plot(diffs, precs, "s--", label="precision", color="#e6194b")
    ax.axhline(summary["recall"], color="#4363d8", alpha=0.3, ls=":")
    ax.axhline(summary["precision"], color="#e6194b", alpha=0.3, ls=":")
    ax.set_xlabel("difficulty (0 = low noise/large/high-contrast, 1 = hardest)")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Detection vs difficulty  |  overall recall {summary['recall']:.2f}, "
                 f"precision {summary['precision']:.2f}, F1 {summary['f1']:.2f}")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _summarize(per_vol, all_pairs, type_gt_total, type_gt_found, cfg, total_time):
    tp = sum(v["tp"] for v in per_vol)
    fp = sum(v["fp"] for v in per_vol)
    fn = sum(v["fn"] for v in per_vol)
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    # Sizing / localization / classification on matched pairs. Sizing is
    # reported as equivalent-diameter error - the domain-standard way defects
    # are dimensioned in NDT (by size, not voxel count) - plus the signed bias.
    dia_abs_err = [abs(p["det_diameter"] - p["gt_diameter"]) / p["gt_diameter"] for p in all_pairs]
    dia_bias = [(p["det_diameter"] - p["gt_diameter"]) / p["gt_diameter"] for p in all_pairs]
    cdist = [p["centroid_dist"] for p in all_pairs]
    cls_correct = [p["det_type"] == p["gt_type"] for p in all_pairs]

    per_type_recall = {
        t: (type_gt_found[t] / type_gt_total[t] if type_gt_total[t] else float("nan"))
        for t in DEFECT_TYPES
    }

    return {
        "config": cfg.__dict__,
        "totals": {"tp": tp, "fp": fp, "fn": fn, "n_gt": tp + fn,
                   "n_detections": tp + fp},
        "recall": recall, "precision": precision, "f1": f1,
        "false_positives": fp,
        "per_type_recall": per_type_recall,
        "per_type_counts": {t: {"planted": type_gt_total[t], "found": type_gt_found[t]}
                            for t in DEFECT_TYPES},
        "mean_iou_matched": float(np.mean([p["iou"] for p in all_pairs])) if all_pairs else float("nan"),
        "mean_centroid_error_voxels": float(np.mean(cdist)) if cdist else float("nan"),
        "median_diameter_abs_error_frac": float(np.median(dia_abs_err)) if dia_abs_err else float("nan"),
        "median_diameter_bias_frac": float(np.median(dia_bias)) if dia_bias else float("nan"),
        "type_classification_accuracy": float(np.mean(cls_correct)) if cls_correct else float("nan"),
        "n_matched": len(all_pairs),
        "total_time_sec": round(total_time, 2),
    }


def _print_table(per_vol, summary):
    print("\nPer-volume detection results")
    header = (f"{'volume':<9}{'diff':>6}{'noise':>7}{'GT':>4}{'det':>5}"
              f"{'TP':>4}{'FP':>4}{'FN':>4}{'recall':>8}{'prec':>7}{'t(s)':>7}")
    print(header); print("-" * len(header))
    for v in per_vol:
        print(f"{v['vol_id']:<9}{v['difficulty']:>6.2f}{v['noise']:>7.1f}"
              f"{v['n_gt']:>4}{v['n_det']:>5}{v['tp']:>4}{v['fp']:>4}{v['fn']:>4}"
              f"{v['recall']:>8.2f}{v['precision']:>7.2f}{v['time']:>7.2f}")
    print("-" * len(header))

    t = summary["totals"]
    print(f"\nOVERALL  (IoU match >= {summary['config']['iou_match_threshold']})")
    print(f"  planted defects (GT) : {t['n_gt']}")
    print(f"  detections           : {t['n_detections']}")
    print(f"  true positives       : {t['tp']}")
    print(f"  false positives      : {summary['false_positives']}")
    print(f"  false negatives      : {t['fn']}")
    print(f"  RECALL               : {summary['recall']:.3f}")
    print(f"  PRECISION            : {summary['precision']:.3f}")
    print(f"  F1                   : {summary['f1']:.3f}")
    print(f"  mean IoU (matched)   : {summary['mean_iou_matched']:.3f}")
    print(f"  mean centroid error  : {summary['mean_centroid_error_voxels']:.2f} voxels")
    print(f"  median diameter error: {summary['median_diameter_abs_error_frac']*100:.1f}% "
          f"(bias {summary['median_diameter_bias_frac']*100:+.1f}%)")
    print(f"  type-class. accuracy : {summary['type_classification_accuracy']*100:.1f}% (heuristic, matched only)")
    print(f"  total detect time    : {summary['total_time_sec']:.2f}s for {len(per_vol)} volumes")

    print("\nPer-type recall")
    for t_ in DEFECT_TYPES:
        c = summary["per_type_counts"][t_]
        r = summary["per_type_recall"][t_]
        print(f"  {t_:<14} {c['found']:>3}/{c['planted']:<3}  recall {r:.2f}")


def main():
    p = argparse.ArgumentParser(description="Detect and measure defects; evaluate vs ground truth.")
    p.add_argument("--threshold-k", type=float, default=DetectConfig.threshold_k)
    p.add_argument("--min-size", type=int, default=DetectConfig.min_size)
    p.add_argument("--iou-threshold", type=float, default=DetectConfig.iou_match_threshold)
    args = p.parse_args()
    cfg = DetectConfig(threshold_k=args.threshold_k, min_size=args.min_size,
                       iou_match_threshold=args.iou_threshold)
    run(cfg)


if __name__ == "__main__":
    main()
