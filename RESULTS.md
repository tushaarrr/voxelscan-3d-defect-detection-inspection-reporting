# VoxelScan — Analysis Results

_Generated 2026-07-25 13:04 from the actual pipeline run (`python run_all.py`). Every number here is measured, not hand-entered._

> **Data is 100% synthetic.** These results characterise the analysis workflow on simulated volumetric scan data with planted defects — not calibrated acoustics and not any real asset or client data.

## 1. Headline results

| Metric | Value |
|---|---|
| Volumes analysed | 20 |
| Planted defects (ground truth) | 117 |
| Detections made | 104 |
| **Detection recall** | **0.863** (101/117) |
| **Detection precision** | **0.971** (101/104) |
| F1 | 0.914 |
| False positives (total) | 3 |
| False negatives (total) | 16 |
| Mean IoU (matched) | 0.532 |
| Centroid localization error | 0.31 voxels |
| Diameter sizing error (median) | 22.2% (bias +20.5%) |
| **Defect-type classifier (5-fold CV)** | **97.0%** (heuristic baseline 69.3%) |
| Detection compute | 7.0s for 20 volumes (~0.35s each) |

## 2. Dataset composition

- 20 volumes of shape 128×128×128, difficulty 0.00→1.00 (simulated noise std 4.0→22.0).
- 117 planted defects by type: pitting 42, crack 25, inclusion 22, wall-thinning 28.

## 3. Per-volume detection results

| Volume | Difficulty | Noise | GT | Detected | TP | FP | FN | Recall | Precision |
|---|---|---|---|---|---|---|---|---|---|
| vol_000 | 0.00 | 4.0 | 7 | 7 | 7 | 0 | 0 | 1.00 | 1.00 |
| vol_001 | 0.05 | 4.9 | 5 | 5 | 5 | 0 | 0 | 1.00 | 1.00 |
| vol_002 | 0.10 | 5.9 | 8 | 8 | 8 | 0 | 0 | 1.00 | 1.00 |
| vol_003 | 0.16 | 6.8 | 7 | 7 | 7 | 0 | 0 | 1.00 | 1.00 |
| vol_004 | 0.21 | 7.8 | 3 | 3 | 3 | 0 | 0 | 1.00 | 1.00 |
| vol_005 | 0.26 | 8.7 | 8 | 8 | 8 | 0 | 0 | 1.00 | 1.00 |
| vol_006 | 0.32 | 9.7 | 7 | 7 | 7 | 0 | 0 | 1.00 | 1.00 |
| vol_007 | 0.37 | 10.6 | 7 | 7 | 7 | 0 | 0 | 1.00 | 1.00 |
| vol_008 | 0.42 | 11.6 | 3 | 3 | 3 | 0 | 0 | 1.00 | 1.00 |
| vol_009 | 0.47 | 12.5 | 5 | 5 | 5 | 0 | 0 | 1.00 | 1.00 |
| vol_010 | 0.53 | 13.5 | 5 | 5 | 5 | 0 | 0 | 1.00 | 1.00 |
| vol_011 | 0.58 | 14.4 | 8 | 8 | 8 | 0 | 0 | 1.00 | 1.00 |
| vol_012 | 0.63 | 15.4 | 6 | 6 | 6 | 0 | 0 | 1.00 | 1.00 |
| vol_013 | 0.68 | 16.3 | 7 | 8 | 7 | 1 | 0 | 1.00 | 0.88 |
| vol_014 | 0.74 | 17.3 | 5 | 7 | 5 | 2 | 0 | 1.00 | 0.71 |
| vol_015 | 0.79 | 18.2 | 4 | 2 | 2 | 0 | 2 | 0.50 | 1.00 |
| vol_016 | 0.84 | 19.2 | 6 | 3 | 3 | 0 | 3 | 0.50 | 1.00 |
| vol_017 | 0.90 | 20.1 | 3 | 2 | 2 | 0 | 1 | 0.67 | 1.00 |
| vol_018 | 0.95 | 21.1 | 7 | 2 | 2 | 0 | 5 | 0.29 | 1.00 |
| vol_019 | 1.00 | 22.0 | 6 | 1 | 1 | 0 | 5 | 0.17 | 1.00 |
| **overall** | | | 117 | 104 | 101 | 3 | 16 | **0.863** | **0.971** |

Recall holds at 1.00 across the easy/mid volumes and degrades on the hardest (low-SNR, small, low-contrast) volumes; precision stays high throughout. This graceful degradation is the point of the difficulty gradient — a flat 100% would not be a meaningful measured result.

## 4. Per-type detection recall

| Defect type | Recall | Found / planted |
|---|---|---|
| pitting | 0.86 | 36/42 |
| crack | 0.88 | 22/25 |
| inclusion | 1.00 | 22/22 |
| wall-thinning | 0.75 | 21/28 |

Compact, high-contrast inclusions are easiest; broad, low-contrast wall-thinning is hardest — as expected.

## 5. Defect-type classifier

RandomForest on shape + intensity features of matched detections, evaluated with stratified 5-fold cross-validation on 101 matched detections.

- **CV accuracy: 97.0%** (rule-based baseline 69.3%).

| Defect type | CV recall | Support |
|---|---|---|
| pitting | 1.00 | 36 |
| crack | 0.91 | 22 |
| inclusion | 0.95 | 22 |
| wall-thinning | 1.00 | 21 |

## 6. Confirmed-defect severity distribution

Severity index = |intensity deviation| × equivalent diameter (a ranking aid, **not** an engineering grade), thresholded at population tertiles.

| Severity | Count | Share |
|---|---|---|
| high | 34 | 34% |
| medium | 33 | 33% |
| low | 34 | 34% |

## 7. Detection vs difficulty

![Detection vs difficulty](docs/metrics_curve.png)

## 8. Where the outputs live

- Per-scan PDF reports: `data/reports/` (a sample is at `docs/sample_report.pdf`).
- 3D renders + orbit GIF: `data/renders/` (demo at `docs/demo_3d.gif`).
- Orthogonal slice views: `data/slices/`.
- Raw metrics: `data/metrics_summary.json`, `data/classifier_metrics.json`.
- Regenerate everything (and this doc): `python run_all.py`.

## Methodology notes

- **Detection:** light gaussian denoise → local baseline per depth-slice (robust median, so it follows the simulated depth attenuation) → flag residuals beyond k·σ (robust noise estimate) in *both* directions → morphological cleanup → 3D connected components → per-region measurement.
- **Matching:** greedy IoU matching between detected and ground-truth label volumes (threshold 0.1).
- **Review:** human-in-the-loop confirm/reject/reclassify; the reported run used the documented automatic reviewer policy for reproducibility.
