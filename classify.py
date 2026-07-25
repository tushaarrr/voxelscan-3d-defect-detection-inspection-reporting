"""Phase 2 (optional add-on) - scikit-learn defect-type classifier.

The rule-based classifier in ``detect.py`` fills the type field cheaply but is
weak. This module trains a small RandomForest on shape + intensity features of
*matched* detections (label = the ground-truth type of the defect they hit) and
reports a *cross-validated* accuracy, so the number is honest (no train/test
leakage). It then writes the model's predicted type back into each detection as
``ml_type`` for the render/report to use.

Run after ``detect.py``:  ``python classify.py``
"""

from __future__ import annotations

import numpy as np

import common
from common import DEFECT_TYPES, MANIFESTS_DIR, DATA_DIR
from detect import DETECTIONS_DIR

MODEL_PATH = DATA_DIR / "type_classifier.joblib"

FEATURE_NAMES = [
    "log_size", "eq_diameter", "delta_sign", "abs_delta",
    "ext_min", "ext_mid", "ext_max",
    "aspect_max_min", "aspect_mid_min", "flatness", "fill_ratio",
]


def features(det) -> list[float]:
    """Shape + intensity feature vector for one detection (schema-only, blind to GT)."""
    zmin, ymin, xmin, zmax, ymax, xmax = det["bbox_zyx"]
    ext = np.sort(np.array([zmax - zmin, ymax - ymin, xmax - xmin], dtype=float))
    ext = np.maximum(ext, 1.0)
    size = float(det["size_voxels"])
    delta = float(det["intensity_delta"])
    bbox_vol = float(ext[0] * ext[1] * ext[2])
    return [
        float(np.log(size)),
        float(det["equivalent_diameter"]),
        1.0 if delta >= 0 else -1.0,
        abs(delta),
        ext[0], ext[1], ext[2],
        ext[2] / ext[0],
        ext[1] / ext[0],
        ext[0] / ext[2],
        size / bbox_vol,
    ]


def _load_training_set():
    """Collect (X, y, heuristic_pred) from all matched detections."""
    X, y, heuristic = [], [], []
    for mpath in sorted(MANIFESTS_DIR.glob("*.json")):
        vol_id = common.load_json(mpath)["volume_id"]
        det = common.load_json(DETECTIONS_DIR / f"{vol_id}.json")
        for d in det["detections"]:
            if d.get("gt_type"):  # matched detection -> has a training label
                X.append(features(d))
                y.append(d["gt_type"])
                heuristic.append(d["type"])
    return np.array(X), np.array(y), np.array(heuristic)


def train_and_evaluate():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import joblib

    X, y, heuristic = _load_training_set()
    if len(X) < len(DEFECT_TYPES) * 3:
        raise SystemExit("Not enough matched detections to train. Run detect.py first.")

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0,
                                 class_weight="balanced")
    # Honest, leakage-free accuracy via stratified 5-fold cross-validation.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_pred = cross_val_predict(clf, X, y, cv=skf)
    cv_acc = float(np.mean(cv_pred == y))
    heur_acc = float(np.mean(heuristic == y))

    # Per-class cross-validated recall + confusion counts.
    per_class = {}
    for t in DEFECT_TYPES:
        mask = y == t
        per_class[t] = {
            "support": int(mask.sum()),
            "cv_recall": float(np.mean(cv_pred[mask] == t)) if mask.any() else float("nan"),
        }

    # Fit the final model on all data and persist it (used for display labels).
    clf.fit(X, y)
    joblib.dump({"model": clf, "features": FEATURE_NAMES}, MODEL_PATH)

    _print(cv_acc, heur_acc, per_class, len(X))
    common.save_json({
        "n_matched": int(len(X)),
        "cv_accuracy": cv_acc,
        "heuristic_accuracy": heur_acc,
        "per_class": per_class,
        "features": FEATURE_NAMES,
    }, DATA_DIR / "classifier_metrics.json")
    return cv_acc, heur_acc


def _print(cv_acc, heur_acc, per_class, n):
    print(f"\nDefect-type classifier (RandomForest, 5-fold CV on {n} matched detections)")
    print(f"  CV accuracy        : {cv_acc*100:.1f}%")
    print(f"  heuristic baseline : {heur_acc*100:.1f}%")
    print("  per-class CV recall:")
    for t in DEFECT_TYPES:
        c = per_class[t]
        print(f"    {t:<14} recall {c['cv_recall']:.2f}  (support {c['support']})")


def predict_types(detections):
    """Predict ml_type for a list of detections using the saved model."""
    import joblib
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    if not detections:
        return []
    X = np.array([features(d) for d in detections])
    return list(clf.predict(X))


def annotate_all():
    """Write ml_type into every detection file (using the final model)."""
    for mpath in sorted(MANIFESTS_DIR.glob("*.json")):
        vol_id = common.load_json(mpath)["volume_id"]
        path = DETECTIONS_DIR / f"{vol_id}.json"
        det = common.load_json(path)
        preds = predict_types(det["detections"])
        for d, pt in zip(det["detections"], preds):
            d["ml_type"] = pt
        common.save_json(det, path)


def main():
    train_and_evaluate()
    annotate_all()
    print("\nWrote ml_type into detection files and saved model to",
          MODEL_PATH.relative_to(common.REPO_ROOT))


if __name__ == "__main__":
    main()
