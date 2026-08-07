"""
Population-statistics QC for landmark labels.

Landmark labels are not self-verifying. The nodule build could render a nodule
alone and require its shadow to land on the projected pixel -- that tested the
label. Projecting a landmark into two views and triangulating back does not
substitute: it round-trips to 1e-13 mm whether or not the point is anatomically
correct, because it re-tests the geometry kernel, not the anatomy. A landmark
placed in the wrong lung passes that check perfectly.

Anatomy is stereotyped, so use the population instead. Build the distribution of
each anatomical relation over all patients, flag the tails with a robust
(median/MAD) score, and look at them. A bad extraction lands in a tail almost
every time.

    python check_landmark_qc.py --data /data/lidc/drr_lm
    python check_landmark_qc.py --data /data/lidc/drr_lm --montage qc.png
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from typing import Dict, List

import numpy as np

# relation -> (function of the per-patient landmark dict, human description)
RELATIONS = {
    "apex_separation_mm":
        (lambda p: p["lung_apex_left"][0] - p["lung_apex_right"][0],
         "left minus right apex, in x (+x = patient left)"),
    "recess_separation_mm":
        (lambda p: p["costophrenic_recess_left"][0] - p["costophrenic_recess_right"][0],
         "left minus right costophrenic recess, in x"),
    "centroid_separation_mm":
        (lambda p: p["lung_centroid_left"][0] - p["lung_centroid_right"][0],
         "left minus right lung centroid, in x"),
    "lung_height_left_mm":
        (lambda p: p["lung_apex_left"][2] - p["costophrenic_recess_left"][2],
         "apex above recess, left lung"),
    "lung_height_right_mm":
        (lambda p: p["lung_apex_right"][2] - p["costophrenic_recess_right"][2],
         "apex above recess, right lung"),
    "apex_z_asymmetry_mm":
        (lambda p: p["lung_apex_left"][2] - p["lung_apex_right"][2],
         "apex height difference L-R"),
    "recess_z_asymmetry_mm":
        (lambda p: p["costophrenic_recess_left"][2] - p["costophrenic_recess_right"][2],
         "recess depth difference L-R (right is usually higher: liver)"),
    "centroid_midline_mm":
        (lambda p: (p["lung_centroid_left"][0] + p["lung_centroid_right"][0]) / 2,
         "midpoint of the two centroids in x (0 = scanner midline)"),
}


# Relations whose plausible range is known from anatomy, not from the data.
# A tail test cannot judge these: it only knows what is unusual, not what is
# impossible.
PLAUSIBLE = {"lung_height_left_mm": (140.0, 400.0),
             "lung_height_right_mm": (140.0, 400.0)}

SCALE_FLOOR_MM = 4.0


def robust_flags(v: np.ndarray, k: float = 4.0, floor: float = SCALE_FLOOR_MM):
    """
    Median/MAD outlier score, with a floor on the scale.

    MAD alone fails on a spiky distribution. Both lung apices are percentiles of
    the same z grid, so most patients get *exactly* the same value and the MAD
    collapses towards zero -- after which a benign 1mm asymmetry scores as a
    25-sigma outlier and 40% of the cohort gets flagged. The floor says: a
    difference smaller than a few millimetres is not evidence of anything,
    however tight the distribution around it is.
    """
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25)) / 1.349
    scale = max(mad, iqr, floor)
    return (v - med) / scale, med, scale


def load(data_dir: str):
    """Per-patient landmark positions, taken once per series."""
    per: Dict[str, Dict[str, List[float]]] = {}
    meta: Dict[str, str] = {}
    for line in open(os.path.join(data_dir, "manifest.jsonl")):
        r = json.loads(line)
        key = r["series_uid"]
        meta[key] = r["patient_id"]
        d = per.setdefault(key, {})
        for n in r["nodules"]:
            d.setdefault(n.get("name", "nodule"), n["center_mm"])
    return per, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=float, default=4.0, help="MAD threshold")
    ap.add_argument("--montage", default=None, help="crops of the worst cases")
    ap.add_argument("--montage-n", type=int, default=12)
    args = ap.parse_args(argv)

    per, meta = load(args.data)
    print(f"{len(per)} series, {len(set(meta.values()))} patients\n")

    counts = collections.Counter()
    for d in per.values():
        for name in d:
            counts[name] += 1
    print("landmark coverage:")
    for name, c in sorted(counts.items()):
        print(f"  {name:28} {c:5d}  {c / max(len(per), 1):.0%}")

    print("\npopulation relations (mm):")
    flagged: Dict[str, List[str]] = collections.defaultdict(list)
    report = {}
    for rel, (fn, desc) in RELATIONS.items():
        keys, vals = [], []
        for k, d in per.items():
            try:
                vals.append(float(fn(d)))
                keys.append(k)
            except (KeyError, TypeError):
                continue
        if len(vals) < 20:
            print(f"  {rel:26} too few ({len(vals)})")
            continue
        v = np.array(vals)
        score, med, mad = robust_flags(v, args.k)
        bad = np.abs(score) > args.k
        if rel in PLAUSIBLE:
            lo, hi = PLAUSIBLE[rel]
            bad = bad | (v < lo) | (v > hi)
        print(f"  {rel:26} median {med:8.1f}  scale {mad:6.1f}  "
              f"p1..p99 [{np.percentile(v, 1):7.1f},{np.percentile(v, 99):7.1f}]  "
              f"flagged {int(bad.sum()):3d}  -- {desc}")
        report[rel] = dict(median=med, mad=mad, n=len(v), flagged=int(bad.sum()))
        for i in np.nonzero(bad)[0]:
            flagged[keys[i]].append(f"{rel}={v[i]:.0f}")

    # A sign error is not an outlier -- it is a systematic error, and the median
    # catches it where a tail test never would.
    print("\nsanity on the medians (these are anatomy, not thresholds):")
    checks = [
        ("apex separation positive (left apex is at +x)",
         report.get("apex_separation_mm", {}).get("median", 0) > 20),
        ("lung height 150-350mm",
         150 < report.get("lung_height_left_mm", {}).get("median", 0) < 350),
        ("centroid midline within 30mm of 0",
         abs(report.get("centroid_midline_mm", {}).get("median", 999)) < 30),
    ]
    for desc, okv in checks:
        print(f"  [{'OK ' if okv else 'BAD'}] {desc}")

    print(f"\n{len(flagged)} series flagged on at least one relation "
          f"({len(flagged) / max(len(per), 1):.1%}):")
    worst = sorted(flagged.items(), key=lambda kv: -len(kv[1]))[:20]
    for k, rs in worst:
        print(f"  {meta[k]:18} {', '.join(rs[:3])}")

    impossible = sorted({k for k, rs in flagged.items()
                         if any(r.split("=")[0] in PLAUSIBLE for r in rs)})
    print(f"\n{len(impossible)} series are anatomically impossible, not merely "
          f"unusual ({len(impossible) / max(len(per), 1):.1%}) -- drop these:")
    for k in impossible[:20]:
        print(f"  {meta[k]:18} {k}")
    with open(os.path.join(args.data, "drop_series.txt"), "w") as f:
        f.write("\n".join(impossible))

    out = os.path.join(args.data, "landmark_qc.json")
    with open(out, "w") as f:
        json.dump(dict(n_series=len(per), coverage=dict(counts), relations=report,
                       flagged={meta[k]: v for k, v in flagged.items()}), f, indent=2)
    print(f"\n-> {out}")
    if flagged and args.montage:
        montage(args, per, meta, [k for k, _ in worst])
    print("\nLook at the flagged cases before training. The contamination rate is "
          f"{len(flagged) / max(len(per), 1):.1%}; if that is above a few percent the "
          "extraction needs work, not the model.")
    return 0


def montage(args, per, meta, keys):
    """Crop every view around each landmark of the worst-scoring series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    recs = collections.defaultdict(list)
    for line in open(os.path.join(args.data, "manifest.jsonl")):
        r = json.loads(line)
        recs[r["series_uid"]].append(r)

    keys = keys[:args.montage_n]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6 * len(keys), 7.6), squeeze=False)
    colors = {"lung_apex_left": "c", "lung_apex_right": "c",
              "costophrenic_recess_left": "y", "costophrenic_recess_right": "y",
              "lung_centroid_left": "r", "lung_centroid_right": "r"}
    for col, k in enumerate(keys):
        for row, view in enumerate(("PA", "LAT")):
            ax = axes[row][col]
            rr = [r for r in recs[k] if r["view"] == view]
            if not rr:
                ax.axis("off"); continue
            r = rr[0]
            ax.imshow(Image.open(os.path.join(args.data, r["image"])), cmap="gray")
            for n in r["nodules"]:
                x, y = n["pixel"]
                ax.plot(x - 0.5, y - 0.5, "+",
                        color=colors.get(n.get("name"), "m"), ms=11, mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{meta[k]} {view}", fontsize=8)
    fig.suptitle("worst-scoring series -- cyan=apex yellow=recess red=centroid", fontsize=12)
    fig.savefig(args.montage, dpi=110, bbox_inches="tight")
    print(f"-> {args.montage}")


if __name__ == "__main__":
    raise SystemExit(main())
