"""Phase 3 - 3D visualization + human-in-the-loop review.

Three pieces:

  * ``render_volume_3d`` - PyVista render of the scanned block (a translucent
    "ghost" of the volume) with each detected defect drawn as a solid,
    colour-coded isosurface. Optional orbit GIF. This is the centerpiece.
  * ``render_slices`` - Matplotlib axial/coronal/sagittal slices through a
    defect (for the PDF report).
  * ``review_detections`` - the human-in-the-loop step: each detection is
    presented with its measurements and marked confirmed / rejected /
    reclassified. Decisions are stored. Runs interactively at a TTY, or with
    ``--auto`` applies a documented reviewer policy so the pipeline is
    reproducible end-to-end.

Off-screen rendering is enabled so this works headless (see README).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import common
from common import DEFECT_TYPES, TYPE_COLORS, VOLUMES_DIR, DATA_DIR
from detect import DETECTIONS_DIR

RENDERS_DIR = DATA_DIR / "renders"
SLICES_DIR = DATA_DIR / "slices"
REVIEWS_DIR = DATA_DIR / "reviews"


# ---------------------------------------------------------------------------
# 3D rendering (PyVista)
# ---------------------------------------------------------------------------
def _defect_surface(det_labels, label_id):
    """Marching-cubes surface of one detection label, lightly smoothed."""
    import pyvista as pv
    from scipy import ndimage as ndi

    mask = (det_labels == label_id).astype(np.float32)
    mask = ndi.gaussian_filter(mask, sigma=0.8)  # smoother surface
    grid = pv.ImageData(dimensions=np.array(mask.shape) + 1)  # cell data
    grid.cell_data["m"] = mask.flatten(order="F")
    grid = grid.cells_to_points("m")
    try:
        surf = grid.contour([0.5], scalars="m")
    except Exception:
        return None
    return surf if surf.n_points > 0 else None


def render_volume_3d(vol_id, detections=None, det_labels=None, gif=False,
                     color_by="type", window_size=(1100, 850)):
    """Render the volume + colour-coded defect surfaces to a PNG (and maybe GIF).

    ``color_by`` is "type" or "severity". Returns the PNG path.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    common.ensure_dirs(RENDERS_DIR)
    volume = common.load_volume(VOLUMES_DIR / f"{vol_id}.npy")
    if detections is None:
        det = common.load_json(DETECTIONS_DIR / f"{vol_id}.json")
        detections = det["detections"]
    if det_labels is None:
        det_labels = common.load_labels(DETECTIONS_DIR / f"{vol_id}_labels.npz")

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("#0e1116", top="#243040")  # dark industrial gradient

    # Translucent "ghost" of the scanned block for spatial context.
    vgrid = pv.ImageData(dimensions=np.array(volume.shape) + 1)
    vgrid.cell_data["intensity"] = volume.flatten(order="F")
    opacity = np.linspace(0, 0.22, 256)  # faint fog, brighter material = denser
    try:
        plotter.add_volume(vgrid, scalars="intensity", cmap="bone",
                           opacity=opacity, show_scalar_bar=False)
    except Exception:
        pass  # volume rendering unavailable -> defects + outline still render
    plotter.add_mesh(vgrid.outline(), color="#8fa3b8", line_width=1.5)

    # Defect surfaces, colour-coded.
    legend_entries = {}
    for d in detections:
        surf = _defect_surface(det_labels, d["label_id"])
        if surf is None:
            continue
        if color_by == "severity":
            sev = d.get("severity_label") or common.severity_label(
                common.severity_score(d["intensity_delta"], d["equivalent_diameter"]))
            color = {"high": "#d7191c", "medium": "#fdae61", "low": "#1a9641"}[sev]
            key = sev
        else:
            color = TYPE_COLORS.get(d.get("final_type", d["type"]), TYPE_COLORS["unknown"])
            key = d.get("final_type", d["type"])
        plotter.add_mesh(surf, color=color, opacity=0.9, smooth_shading=True)
        legend_entries[key] = color

    if legend_entries:
        plotter.add_legend([[k, c] for k, c in sorted(legend_entries.items())],
                           bcolor="#1b222b", size=(0.22, 0.22))
    plotter.add_text(f"{vol_id}  -  {len(detections)} detected defects\n(SIMULATED scan data)",
                     font_size=11, color="white")
    plotter.add_axes(color="white")
    plotter.camera_position = "iso"
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 20

    png_path = RENDERS_DIR / f"{vol_id}_3d.png"
    plotter.screenshot(str(png_path))

    if gif:
        gif_path = RENDERS_DIR / f"{vol_id}_3d.gif"
        path = plotter.generate_orbital_path(n_points=36, shift=volume.shape[0] * 0.6)
        plotter.open_gif(str(gif_path))
        plotter.orbit_on_path(path, write_frames=True, step=0.05)
        plotter.close()
        return png_path, gif_path

    plotter.close()
    return png_path


# ---------------------------------------------------------------------------
# 2D orthogonal slices (Matplotlib)
# ---------------------------------------------------------------------------
def render_slices(vol_id, detection, path, volume=None):
    """Axial/coronal/sagittal slices through a detection's centroid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    common.ensure_dirs(path.parent)
    if volume is None:
        volume = common.load_volume(VOLUMES_DIR / f"{vol_id}.npy")

    cz, cy, cx = [int(round(c)) for c in detection["center_zyx"]]
    zmin, ymin, xmin, zmax, ymax, xmax = detection["bbox_zyx"]
    vmin, vmax = np.percentile(volume, [1, 99])
    color = TYPE_COLORS.get(detection.get("final_type", detection["type"]), "#ffffff")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    panels = [
        ("Axial (z=%d)" % cz, volume[cz], (xmin, ymin, xmax - xmin, ymax - ymin), "x", "y"),
        ("Coronal (y=%d)" % cy, volume[:, cy, :], (xmin, zmin, xmax - xmin, zmax - zmin), "x", "z"),
        ("Sagittal (x=%d)" % cx, volume[:, :, cx], (ymin, zmin, ymax - ymin, zmax - zmin), "y", "z"),
    ]
    for ax, (title, img, rect, xl, yl) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.add_patch(Rectangle((rect[0], rect[1]), rect[2], rect[3],
                               fill=False, edgecolor=color, linewidth=1.8))
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)

    fig.suptitle(f"{vol_id}  -  {detection['detection_id']}  "
                 f"({detection.get('final_type', detection['type'])})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Human-in-the-loop review
# ---------------------------------------------------------------------------
def _auto_policy(det) -> tuple[str, str]:
    """Documented automatic reviewer stand-in used for reproducible runs.

    Confirms detections that are both large enough and high-contrast enough to
    be trustworthy; flags the rest as rejected (low confidence). Returns
    (status, note). This is NOT a human - the interactive path is the real
    human-in-the-loop; --auto just makes the pipeline runnable unattended.
    """
    size = det["size_voxels"]
    delta = abs(det["intensity_delta"])
    if size >= 30 and delta >= 12:
        return "confirmed", "auto: size & contrast above confidence thresholds"
    return "rejected", "auto: low confidence (small and/or low contrast)"


def review_detections(vol_id, auto=False):
    """Present detections for confirm/reject/reclassify; store the decisions."""
    common.ensure_dirs(REVIEWS_DIR)
    det = common.load_json(DETECTIONS_DIR / f"{vol_id}.json")
    detections = det["detections"]

    interactive = (not auto) and sys.stdin.isatty()
    reviewer = "human" if interactive else "auto"

    reviewed = []
    for d in detections:
        score = common.severity_score(d["intensity_delta"], d["equivalent_diameter"])
        d = dict(d)
        d["severity_score"] = round(score, 1)
        d["severity_label"] = common.severity_label(score)
        # Prefer the trained classifier's type; fall back to the heuristic.
        d["final_type"] = d.get("ml_type", d["type"])

        if interactive:
            status = _prompt_one(d)
        else:
            status, note = _auto_policy(d)
            d["review_note"] = note
        d["review_status"] = status
        d["reviewer"] = reviewer
        reviewed.append(d)

    out = {
        "volume_id": vol_id,
        "reviewer": reviewer,
        "n_confirmed": sum(r["review_status"] == "confirmed" for r in reviewed),
        "n_rejected": sum(r["review_status"] == "rejected" for r in reviewed),
        "detections": reviewed,
    }
    common.save_json(out, REVIEWS_DIR / f"{vol_id}.json")
    return out


def _prompt_one(d) -> str:
    """Interactive CLI prompt for a single detection."""
    print(f"\n  {d['detection_id']}  type={d['type']}  "
          f"size={d['size_voxels']} vox  eq.dia={d['equivalent_diameter']}  "
          f"delta={d['intensity_delta']}  severity={d['severity_label']}")
    print(f"    location (z,y,x)={d['center_zyx']}")
    while True:
        ans = input("    [c]onfirm / [r]eject / [t]ype reclassify / [s]kip? ").strip().lower()
        if ans in ("c", ""):
            return "confirmed"
        if ans == "r":
            return "rejected"
        if ans == "s":
            return "skipped"
        if ans == "t":
            print("     types: " + ", ".join(f"{i}={t}" for i, t in enumerate(DEFECT_TYPES)))
            sel = input("     new type index: ").strip()
            if sel.isdigit() and int(sel) < len(DEFECT_TYPES):
                d["final_type"] = DEFECT_TYPES[int(sel)]
                print(f"     reclassified -> {d['final_type']}")
            return "confirmed"
        print("     (enter c, r, t, or s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _process_one(vol_id, gif, auto_review):
    review = review_detections(vol_id, auto=auto_review)
    detections = review["detections"]
    det_labels = common.load_labels(DETECTIONS_DIR / f"{vol_id}_labels.npz")
    result = render_volume_3d(vol_id, detections=detections, det_labels=det_labels, gif=gif)
    png = result[0] if isinstance(result, tuple) else result

    # Slice views for the most severe confirmed defects (up to 3).
    volume = common.load_volume(VOLUMES_DIR / f"{vol_id}.npy")
    confirmed = [d for d in detections if d["review_status"] == "confirmed"]
    confirmed.sort(key=lambda d: d["severity_score"], reverse=True)
    for d in confirmed[:3]:
        render_slices(vol_id, d, SLICES_DIR / f"{vol_id}_{d['detection_id']}.png", volume=volume)

    print(f"{vol_id}: render -> {png.name}, {review['n_confirmed']} confirmed / "
          f"{review['n_rejected']} rejected, slices for top {min(3, len(confirmed))}")


def main():
    p = argparse.ArgumentParser(description="Render 3D defects + slices and run HITL review.")
    p.add_argument("--vol", default="vol_000", help="volume id (default vol_000)")
    p.add_argument("--all", action="store_true", help="process every volume")
    p.add_argument("--gif", action="store_true", help="also export an orbit GIF")
    p.add_argument("--auto-review", action="store_true",
                   help="apply the automatic reviewer policy (non-interactive)")
    args = p.parse_args()

    if args.all:
        for mpath in sorted((DATA_DIR / "manifests").glob("*.json")):
            _process_one(common.load_json(mpath)["volume_id"], args.gif, True)
    else:
        _process_one(args.vol, args.gif, args.auto_review)


if __name__ == "__main__":
    main()
