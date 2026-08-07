"""
Is the nodule actually visible in the DRR?

Before blaming the model, measure whether the target is in the image at all.
For every labelled nodule, compare the pixels inside its disk against an
annulus around it, in the exact 8-bit PNG the model is shown:

    CNR    = (mean_in - mean_out) / std_out      how many noise units it stands out by
    Weber  = (mean_in - mean_out) / mean_out     relative brightness change

Rules of thumb from detection psychophysics: CNR < 1 is invisible, 1-3 is
marginal, > 3-5 is comfortably visible. If most nodules sit below 1, the Phase
0 gate is not failing -- it is unanswerable as posed, and the fix is rendering
or windowing, not more training.

    python check_visibility.py --data /data/lidc/drr
    python check_visibility.py --data /data/lidc/drr --montage out.png
    python check_visibility.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np


def cnr(img: np.ndarray, cx: float, cy: float, r_px: float,
        inner: float = 1.0, gap: float = 2.0, outer: float = 4.0) -> Optional[Dict]:
    """
    Contrast of a disk against the annulus around it.

    `gap` leaves a dead zone so the nodule's own blur does not contaminate the
    background estimate, which would flatter the contrast.
    """
    h, w = img.shape
    r_px = max(r_px, 1.5)
    R = outer * r_px
    x0, x1 = int(max(0, cx - R)), int(min(w, cx + R + 1))
    y0, y1 = int(max(0, cy - R)), int(min(h, cy + R + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = img[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - cx, yy - cy)

    m_in = d <= inner * r_px
    m_out = (d >= gap * r_px) & (d <= outer * r_px)
    if m_in.sum() < 3 or m_out.sum() < 12:
        return None
    a, b = patch[m_in].mean(), patch[m_out].mean()
    sd = patch[m_out].std()
    return dict(cnr=float((a - b) / sd) if sd > 1e-9 else 0.0,
                weber=float((a - b) / b) if abs(b) > 1e-9 else 0.0,
                mean_in=float(a), mean_out=float(b), std_out=float(sd),
                r_px=float(r_px))


def run(args) -> int:
    from PIL import Image

    recs = [json.loads(l) for l in open(os.path.join(args.data, "manifest.jsonl"))]
    rows: List[Dict] = []
    for r in recs[:args.limit or len(recs)]:
        path = os.path.join(args.data, r["image"])
        if not os.path.exists(path):
            continue
        img = np.asarray(Image.open(path).convert("L"))
        mm_per_px = float(r.get("geometry", {}).get("det_spacing", [1.0])[0])
        for n in r["nodules"]:
            rad = float(n.get("radius_mm", 0))
            if rad < args.min_radius_mm:
                continue
            x, y = n["pixel"]
            m = cnr(img, x, y, rad / mm_per_px)
            if m:
                rows.append(m | dict(image=r["image"], view=r["view"],
                                     radius_mm=rad, x=x, y=y,
                                     patient=r["patient_id"]))
    if not rows:
        print("no measurable nodules"); return 1

    c = np.array([r["cnr"] for r in rows])
    wv = np.array([abs(r["weber"]) for r in rows])
    print(f"{len(rows)} nodules measured in {len({r['image'] for r in rows})} images "
          f"(radius >= {args.min_radius_mm}mm)\n")
    print(f"  CNR      median {np.median(c):6.2f} | mean {c.mean():6.2f} | "
          f"p10 {np.percentile(c, 10):5.2f} | p90 {np.percentile(c, 90):5.2f}")
    print(f"  |Weber|  median {np.median(wv):6.3f} ({np.median(wv) * 100:.1f}% "
          f"brightness change)\n")
    below1 = float((np.abs(c) < 1).mean())
    below3 = float((np.abs(c) < 3).mean())
    print(f"  |CNR| < 1 (invisible):  {below1:.1%}")
    print(f"  |CNR| < 3 (marginal):   {below3:.1%}")
    print(f"  |CNR| >= 3 (visible):   {1 - below3:.1%}\n")

    if below1 > 0.5:
        print("  VERDICT: most nodules are below the visibility floor. The gate is")
        print("  not failing -- the target is not in the picture. Fix the rendering")
        print("  (window the lung field, raise detector resolution, thinner slab)")
        print("  before training anything else.")
    elif below3 > 0.5:
        print("  VERDICT: marginal. A model may find the large ones only; expect the")
        print("  error distribution to be bimodal rather than merely wide.")
    else:
        print("  VERDICT: nodules are visible. A failed gate is a model/data-size")
        print("  problem, not a rendering one.")

    by_view: Dict[str, List[float]] = {}
    for r in rows:
        by_view.setdefault(r["view"], []).append(r["cnr"])
    print("\n  by view: " + "  ".join(
        f"{v} median {np.median(x):.2f} (n={len(x)})" for v, x in sorted(by_view.items())))

    big = [r for r in rows if r["radius_mm"] >= 5]
    if big:
        print(f"  radius >= 5mm: median CNR {np.median([r['cnr'] for r in big]):.2f} "
              f"(n={len(big)})")

    with open(os.path.join(args.data, "visibility.json"), "w") as f:
        json.dump(dict(n=len(rows), cnr_median=float(np.median(c)),
                       frac_below_1=below1, frac_below_3=below3,
                       rows=rows[:2000]), f, indent=2)
    print(f"\n-> {os.path.join(args.data, 'visibility.json')}")

    if args.montage:
        montage(args, rows)
    return 0


def montage(args, rows):
    """Crops around the best- and worst-contrast nodules. Look at them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = sorted(rows, key=lambda r: -abs(r["cnr"]))
    pick = rows[:8] + rows[len(rows) // 2 - 4:len(rows) // 2 + 4] + rows[-8:]
    labels = ["best"] * 8 + ["median"] * 8 + ["worst"] * 8
    fig, axes = plt.subplots(3, 8, figsize=(20, 8))
    for ax, r, lab in zip(axes.ravel(), pick, labels):
        img = np.asarray(Image.open(os.path.join(args.data, r["image"])).convert("L"))
        h = max(24, int(r["r_px"] * 6))
        x, y = int(r["x"]), int(r["y"])
        crop = img[max(0, y - h):y + h, max(0, x - h):x + h]
        ax.imshow(crop, cmap="gray")
        ax.add_patch(plt.Circle((x - max(0, x - h), y - max(0, y - h)),
                                r["r_px"], ec="r", fc="none", lw=1.2))
        ax.set_title(f"{lab} CNR {r['cnr']:.1f}  r={r['radius_mm']:.1f}mm", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("nodule crops from the DRRs the model is shown "
                 "(red circle = the label)", fontsize=12)
    fig.savefig(args.montage, dpi=110, bbox_inches="tight")
    print(f"-> {args.montage}")


def selftest() -> int:
    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    rng = np.random.default_rng(0)
    flat = rng.normal(128, 10, (200, 200))
    m = cnr(flat, 100, 100, 5)
    t("pure noise -> CNR ~ 0", abs(m["cnr"]) < 0.5, f"{m['cnr']:.3f}")

    yy, xx = np.mgrid[0:200, 0:200]
    for amp, want in ((5.0, "invisible"), (50.0, "visible")):
        img = flat + amp * (np.hypot(xx - 100, yy - 100) <= 5)
        m = cnr(img, 100, 100, 5)
        if want == "invisible":
            t(f"blob {amp:g} over noise 10 -> marginal", abs(m["cnr"]) < 1.5, f"{m['cnr']:.2f}")
        else:
            t(f"blob {amp:g} over noise 10 -> clearly visible", m["cnr"] > 3, f"{m['cnr']:.2f}")

    img = flat + 30 * (np.hypot(xx - 100, yy - 100) <= 5)
    a = cnr(img, 100, 100, 5)["cnr"]
    b = cnr(img, 100, 100, 5, gap=1.0)["cnr"]
    t("gap keeps blur out of the background", a >= b - 1e-9, f"gap=2 {a:.2f} vs gap=1 {b:.2f}")
    t("off-image returns None", cnr(flat, -50, -50, 5) is None)

    print("\n" + ("all self-tests passed" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--min-radius-mm", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--montage", default=None, help="write a crop montage here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.data:
        ap.error("--data is required (or --selftest)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
