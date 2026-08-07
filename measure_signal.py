"""
How strong is a nodule's contribution to the DRR, exactly?

The annulus-based CNR in check_visibility.py is confounded: for a 10mm nodule
the background ring lands 20-40mm away, which is chest wall and diaphragm, and
no plane fit removes a step edge. It reports the anatomy around the nodule, not
the nodule -- which is why bigger nodules came out looking darker.

This measures the thing itself. Attenuation integrates linearly along a ray, so
the nodule's contribution to the DRR *is* the render of a nodule-only volume:

    signal_grey = peak of DRR(nodule only), converted to 8-bit via the same
                  window the PNG was saved with
    clutter     = local detrended std of the actual PNG, away from the nodule
    true CNR    = signal_grey / clutter

No annulus, no assumption about what surrounds the nodule, and the geometry
comes from the manifest, so it is the same ray cast that made the image.

    python measure_signal.py --data /data/lidc/drr --root /data/lidc --limit 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def view_from_manifest(vol, g: Dict):
    """Rebuild the exact DRRView the image was rendered with."""
    from geometry_kernel import DRRView

    return DRRView(vol, g["source_mm"], g["det_center_mm"], g["det_u"], g["det_v"],
                   det_spacing=tuple(g["det_spacing"]), det_size=tuple(g["det_size"]))


def local_clutter(img: np.ndarray, cx: float, cy: float, half: int = 40) -> float:
    """Std of the PNG in a box around the nodule, after removing a plane."""
    h, w = img.shape
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return float("nan")
    p = img[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    A = np.stack([xx.ravel() - cx, yy.ravel() - cy, np.ones(p.size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
    resid = p.ravel() - A @ coef
    return float(resid.std())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="build_drr_dataset.py output dir")
    ap.add_argument("--root", required=True, help="LIDC root, to find the CT series")
    ap.add_argument("--limit", type=int, default=200, help="nodules to measure")
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from PIL import Image
    from build_drr_dataset import scan_tree
    from geometry_kernel import load_ct, nodule_difference_volume

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    print(f"scanning {args.root} for the CT series ...", flush=True)
    series, _ = scan_tree(args.root)
    by_uid = {s.series_uid: s.ct_dir for s in series}
    print(f"  {len(by_uid)} series on disk")

    recs = [json.loads(l) for l in open(os.path.join(args.data, "manifest.jsonl"))]
    tasks = []
    for r in recs:
        for n in r["nodules"]:
            tasks.append((r, n))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(tasks)
    tasks = tasks[:args.limit]
    # group by series so each volume is loaded once
    tasks.sort(key=lambda t: t[0]["series_uid"])
    print(f"  measuring {len(tasks)} nodules across "
          f"{len({t[0]['series_uid'] for t in tasks})} series\n", flush=True)

    rows: List[Dict] = []
    vol = cur = None
    t0 = time.time()
    for i, (r, n) in enumerate(tasks, 1):
        uid = r["series_uid"]
        if uid not in by_uid:
            continue
        if uid != cur:
            try:
                vol, cur = load_ct(by_uid[uid]), uid
            except Exception as e:
                print(f"  skip {uid[-12:]}: {type(e).__name__}")
                cur = None
                continue
        try:
            view = view_from_manifest(vol, r["geometry"])
            dv, fill = nodule_difference_volume(vol, n["center_mm"],
                                                float(n["radius_mm"]))
            shadow = view.with_volume(dv).render(n_samples=args.samples)
        except Exception as e:
            print(f"  skip nodule: {type(e).__name__}: {e}")
            continue

        lo, hi = r["window"]
        peak_L = float(shadow.max())                    # line-integral units
        peak_grey = peak_L / max(hi - lo, 1e-9) * 255.0  # what the PNG could show
        img = np.asarray(Image.open(os.path.join(args.data, r["image"])).convert("L"))
        x, y = n["pixel"]
        clut = local_clutter(img, x, y)
        rows.append(dict(radius_mm=float(n["radius_mm"]), view=r["view"],
                         peak_L=peak_L, peak_grey=peak_grey, clutter=clut,
                         true_cnr=peak_grey / clut if clut and clut > 1e-9 else float("nan"),
                         patient=r["patient_id"]))
        if i % 25 == 0:
            print(f"  [{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f}m", flush=True)

    if not rows:
        print("nothing measured"); return 1

    g = np.array([r["peak_grey"] for r in rows])
    cl = np.array([r["clutter"] for r in rows])
    tc = np.array([r["true_cnr"] for r in rows])
    rad = np.array([r["radius_mm"] for r in rows])
    ok = np.isfinite(tc)

    print("\n" + "=" * 68)
    print(f"{len(rows)} nodules measured exactly\n")
    print(f"  nodule signal      median {np.median(g):6.2f} grey levels of 255 "
          f"(p10 {np.percentile(g, 10):.2f}, p90 {np.percentile(g, 90):.2f})")
    print(f"  local clutter      median {np.median(cl):6.2f} grey levels")
    print(f"  TRUE CNR           median {np.median(tc[ok]):6.2f}  "
          f"(p10 {np.percentile(tc[ok], 10):.2f}, p90 {np.percentile(tc[ok], 90):.2f})\n")
    print(f"  signal < 1 grey level (below 8-bit quantisation): "
          f"{float((g < 1).mean()):.1%}")
    print(f"  true CNR < 1: {float((tc[ok] < 1).mean()):.1%}   "
          f"< 0.5: {float((tc[ok] < 0.5).mean()):.1%}\n")
    print(f"  corr(radius, signal) = {np.corrcoef(rad, g)[0, 1]:+.3f}   "
          "(MUST be positive -- more tissue attenuates more)")
    for lo_, hi_ in [(3, 5), (5, 7), (7, 10), (10, 99)]:
        m = (rad >= lo_) & (rad < hi_) & ok
        if m.sum() > 3:
            print(f"    r {lo_:2d}-{hi_:2d}mm  n={int(m.sum()):4d}  "
                  f"signal {np.median(g[m]):6.2f} grey  CNR {np.median(tc[m]):5.2f}")

    med = float(np.median(tc[ok]))
    print()
    if np.corrcoef(rad, g)[0, 1] < 0:
        print("  Signal falls with nodule size -- that is impossible physically.")
        print("  Something upstream is wrong; do not interpret the CNR yet.")
    elif float((g < 1).mean()) > 0.5:
        print("  VERDICT: most nodules move the image by less than one 8-bit level.")
        print("  They are quantised away before the model sees them. Save the DRR")
        print("  with a tighter window or in 16-bit -- this one IS a windowing fix.")
    elif med < 0.5:
        print("  VERDICT: the nodule is far below local anatomical clutter. A single")
        print("  projection genuinely does not carry it -- which is the null-space")
        print("  claim the paper rests on, measured rather than asserted.")
    elif med < 2:
        print("  VERDICT: marginal, the regime real radiographs live in. Detection")
        print("  needs context and shape, not local contrast; expect a 2D gate to")
        print("  need far more than ~700 training images.")
    else:
        print("  VERDICT: the nodule is well above local clutter. A failed gate is a")
        print("  training/data-size problem, not a visibility one.")

    out = os.path.join(args.data, "signal.json")
    with open(out, "w") as f:
        json.dump(dict(n=len(rows), median_peak_grey=float(np.median(g)),
                       median_clutter=float(np.median(cl)), median_true_cnr=med,
                       frac_below_1_grey=float((g < 1).mean()), rows=rows), f, indent=2)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
