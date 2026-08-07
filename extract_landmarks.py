"""
Anatomical landmarks from a CT volume, in millimetres.

Nodules are the hardest target in the chest: ~860 HU over a ~10mm chord, which
measures out at ~1 grey level of 255 against ~17 of anatomical clutter (see
docs/STEP2.md). Landmarks win on chord length rather than density -- a diaphragm
dome is 8x a nodule, a heart border 12x -- so they are found comfortably above
the clutter, which is what E1 and E2 need in order to measure the *shape* of the
localisation error rather than the failure to localise at all.

No segmentation network: everything here is HU thresholds plus connected
components, which is enough for structures this large and keeps the dependency
list at SimpleITK + numpy.

    python extract_landmarks.py --ct <series dir>            # one volume, printed
    python extract_landmarks.py --selftest                   # phantom, no data

Every landmark carries a `valid` flag and a reason. A landmark that cannot be
found reliably must be dropped, not guessed: a silently wrong target is worse
than a missing one, and at 700 patients nobody inspects them by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# HU bands. Lung parenchyma is roughly -950..-500; airway lumen is below -950,
# which is also where the air *outside* the patient lives -- hence the
# border-connectivity test rather than a threshold alone.
LUNG_LO, LUNG_HI = -950.0, -500.0
AIR_HI = -950.0
BONE_LO = 250.0


def _cc(mask: np.ndarray, keep: int = 0):
    """Connected components, relabelled largest-first. Returns (labels, sizes)."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(img), sortByObjectSize=True)
    lab = sitk.GetArrayFromImage(cc)
    n = int(lab.max())
    sizes = [int((lab == i).sum()) for i in range(1, min(n, 8) + 1)]
    return lab, sizes


def lung_mask(vol) -> Optional[np.ndarray]:
    """The two lungs, as one boolean volume."""
    a = vol.array
    raw = (a > LUNG_LO) & (a < LUNG_HI)
    lab, sizes = _cc(raw)
    if not sizes:
        return None
    total = a.size
    # Lungs are large but not the whole volume; take components down to 15% of
    # the biggest, which keeps both lungs whether or not they are joined.
    keep = [i + 1 for i, s in enumerate(sizes) if s > 0.15 * sizes[0]
            and s > 0.005 * total]
    if not keep:
        return None
    return np.isin(lab, keep)


def airway_mask(vol) -> Optional[np.ndarray]:
    """Trachea and main bronchi: interior air, excluding the air around the body."""
    a = vol.array
    raw = a < AIR_HI
    lab, sizes = _cc(raw)
    if not sizes:
        return None
    nz, ny, nx = a.shape
    best = None
    for i in range(1, min(len(sizes), 8) + 1):
        m = lab == i
        # exterior air touches the in-plane border on most slices
        border = m[:, 0, :].sum() + m[:, -1, :].sum() + m[:, :, 0].sum() + m[:, :, -1].sum()
        if border > 0.02 * m.sum():
            continue
        if best is None or m.sum() > best.sum():
            best = m
    return best


def _centroid_mm(vol, mask: np.ndarray):
    """Mask centroid -> mm. Voxel index order is (i, j, k) = (x, y, z)."""
    import torch

    k, j, i = np.nonzero(mask)
    if not len(i):
        return None
    v = torch.tensor([i.mean(), j.mean(), k.mean()], dtype=torch.float64)
    return vol.voxel_to_mm(v)


def find_carina(vol, airway: np.ndarray = None) -> Dict:
    """
    The tracheal bifurcation.

    Selecting "the largest interior air component" does NOT give the airway:
    well-aerated lung is also below -950 HU, so that component is the lungs,
    and walking down it finds the two lung apices splitting -- which is how an
    earlier version placed the carina *above* the apices in 12 of 19 scans.

    The trachea is separated instead by cross-sectional area: ~150-300 mm2 per
    slice, against thousands for a lung. Track that blob down, and the carina is
    the last slice where it is still one lumen.
    """
    import SimpleITK as sitk
    import torch

    a = vol.array
    sx, sy, _ = vol.spacing
    px_mm2 = float(sx * sy)
    air = a < -975.0

    # midline from the body mask, since patients are not centred in the scanner
    body = a > -500.0
    if body.sum() < 1000:
        return dict(valid=False, reason="no body mask")
    mid_i = float(np.nonzero(body)[2].mean())

    def blobs(z):
        """Trachea-sized, near-midline air blobs on one slice: (area, i, j)."""
        m = air[z]
        if not m.any():
            return []
        lab = sitk.GetArrayFromImage(sitk.RelabelComponent(
            sitk.ConnectedComponent(sitk.GetImageFromArray(m.astype(np.uint8)))))
        out = []
        for v in range(1, min(int(lab.max()), 12) + 1):
            sel = lab == v
            area = float(sel.sum()) * px_mm2
            if not (30.0 <= area <= 900.0):        # excludes lung outright
                continue
            jj, ii = np.nonzero(sel)
            if abs(ii.mean() - mid_i) * sx > 50.0:
                continue
            out.append((area, float(ii.mean()), float(jj.mean())))
        return out

    nz = a.shape[0]
    # +z is superior and numpy axis 0 is k, so descend from the top of the scan
    zs = [z for z in range(nz) if air[z].any()]
    if len(zs) < 10:
        return dict(valid=False, reason="no air column")

    seen_single = None
    run = 0
    for z in range(max(zs), min(zs) - 1, -1):
        b = blobs(z)
        if len(b) == 1:
            seen_single = (z, b[0])
            run = 0
        elif len(b) >= 2 and seen_single is not None:
            run += 1
            if run >= 3:                        # split, and it stayed split
                zc, (_, ii, jj) = seen_single
                v = torch.tensor([ii, jj, float(zc)], dtype=torch.float64)
                return dict(valid=True, mm=vol.voxel_to_mm(v), z_index=zc)
        else:
            run = 0
    if seen_single is None:
        return dict(valid=False, reason="no trachea-sized midline air column")
    return dict(valid=False, reason="no bifurcation below the trachea")


def extract(vol, want: Optional[List[str]] = None) -> Dict[str, Dict]:
    """All landmarks for one volume. Each entry: {valid, mm, ...}."""
    import torch

    out: Dict[str, Dict] = {}
    lungs = lung_mask(vol)
    if lungs is None or lungs.sum() < 1000:
        return {"_error": dict(valid=False, reason="no lung mask")}

    k, j, i = np.nonzero(lungs)
    mm = vol.voxel_to_mm(torch.tensor(
        np.stack([i, j, k], axis=1).astype(np.float64), dtype=torch.float64))
    x, y, z = mm[:, 0].numpy(), mm[:, 1].numpy(), mm[:, 2].numpy()

    # +x is patient left in LPS. Split at the lung-mask midline rather than at
    # x=0, because the patient is not always centred in the scanner.
    mid = float(np.median(x))
    for side, sel in (("left", x > mid), ("right", x <= mid)):
        if sel.sum() < 500:
            out[f"lung_centroid_{side}"] = dict(valid=False, reason="side too small")
            continue
        out[f"lung_centroid_{side}"] = dict(
            valid=True, mm=[float(x[sel].mean()), float(y[sel].mean()),
                            float(z[sel].mean())], n_voxels=int(sel.sum()))
        # apex: most superior lung voxels of that side, averaged for stability
        zt = np.percentile(z[sel], 99.5)
        top = sel & (z >= zt)
        out[f"lung_apex_{side}"] = dict(
            valid=bool(top.sum() >= 20),
            mm=[float(x[top].mean()), float(y[top].mean()), float(z[top].mean())],
            n_voxels=int(top.sum()))
        # diaphragm dome: most inferior lung voxels -- the lung/liver interface
        zb = np.percentile(z[sel], 0.5)
        bot = sel & (z <= zb)
        out[f"diaphragm_dome_{side}"] = dict(
            valid=bool(bot.sum() >= 20),
            mm=[float(x[bot].mean()), float(y[bot].mean()), float(z[bot].mean())],
            n_voxels=int(bot.sum()))

    c = find_carina(vol)
    out["carina"] = (dict(valid=True, mm=[float(t) for t in c["mm"]])
                     if c["valid"] else c)

    # spine: bone centroid, taken at the carina's height when we have one so the
    # landmark is a defined point rather than "somewhere along the column"
    bone = vol.array > BONE_LO
    if bone.sum() > 500:
        zc = out.get("carina", {}).get("mm")
        if zc:
            kc = int(round(float(vol.mm_to_voxel(torch.tensor(zc, dtype=torch.float64))[2])))
            lo, hi = max(0, kc - 3), min(bone.shape[0], kc + 4)
            sl = np.zeros_like(bone); sl[lo:hi] = bone[lo:hi]
            bone = sl
        # the vertebral column is the most posterior large bone mass (+y = posterior)
        c = _centroid_mm(vol, bone)
        out["spine_centroid"] = (dict(valid=True, mm=[float(t) for t in c])
                                 if c is not None else dict(valid=False, reason="no bone"))
    else:
        out["spine_centroid"] = dict(valid=False, reason="no bone")

    if want:
        out = {k: v for k, v in out.items() if k in want}
    return out


# --------------------------------------------------------------------------


E1_SET = ["carina", "lung_apex_left", "lung_apex_right",
          "diaphragm_dome_left", "diaphragm_dome_right"]


def anatomy_checks(lm: Dict[str, Dict]) -> List[str]:
    """
    Relations that must hold in LPS for any chest. These catch a landmark that
    was found confidently and placed wrongly, which a validity flag cannot.
    """
    bad = []
    g = lambda k: lm.get(k, {}).get("mm") if lm.get(k, {}).get("valid") else None
    car, al, ar = g("carina"), g("lung_apex_left"), g("lung_apex_right")
    dl, dr, sp = g("diaphragm_dome_left"), g("diaphragm_dome_right"), g("spine_centroid")

    if car and abs(car[0]) > 45:
        bad.append(f"carina off-midline (x={car[0]:.0f}mm)")
    if car and sp and car[1] >= sp[1]:
        bad.append("carina not anterior to spine")
    if car and dl and dr and car[2] <= max(dl[2], dr[2]):
        bad.append("carina not superior to the diaphragm domes")
    if car and al and ar and car[2] >= min(al[2], ar[2]):
        bad.append("carina not inferior to the lung apices")
    if al and ar and abs(al[2] - ar[2]) > 50:
        bad.append(f"apices asymmetric in z ({abs(al[2] - ar[2]):.0f}mm)")
    if dl and dr and abs(dl[2] - dr[2]) > 60:
        bad.append(f"domes asymmetric in z ({abs(dl[2] - dr[2]):.0f}mm)")
    if al and ar and dl and dr and min(al[2], ar[2]) <= max(dl[2], dr[2]):
        bad.append("apices not above domes")
    return bad


def scan(args) -> int:
    """Run the extractor over many series; report rates and draw the overlays."""
    import collections
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from build_drr_dataset import scan_tree
    from geometry_kernel import DRRView, load_ct

    series, _ = scan_tree(args.scan)
    series = series[:args.limit]
    print(f"{len(series)} series\n")

    found = collections.Counter()
    failures = collections.Counter()
    problems = collections.Counter()
    panels = []
    for n, s in enumerate(series, 1):
        try:
            vol = load_ct(s.ct_dir)
        except Exception as e:
            failures[f"load: {type(e).__name__}"] += 1
            continue
        lm = extract(vol)
        for k in E1_SET:
            if lm.get(k, {}).get("valid"):
                found[k] += 1
            else:
                failures[f"{k}: {lm.get(k, {}).get('reason', 'missing')}"] += 1
        for p in anatomy_checks(lm):
            problems[p.split(" (")[0]] += 1

        if len(panels) < args.montage_n and lm.get("carina", {}).get("valid"):
            panels.append((s.patient_id, vol, lm))
        print(f"  [{n}/{len(series)}] {s.patient_id}  " +
              " ".join(k.split('_')[0][:4] + ("+" if lm.get(k, {}).get("valid") else "-")
                       for k in E1_SET), flush=True)

    print(f"\n{'=' * 64}\nfound (of {len(series)} series):")
    for k in E1_SET:
        print(f"  {k:22} {found[k]:4d}  {found[k] / max(len(series), 1):.0%}")
    if failures:
        print("\nfailures:")
        for r, c in failures.most_common(8):
            print(f"  {c:4d}  {r}")
    print("\nanatomy check violations (found but placed wrongly):")
    if problems:
        for p, c in problems.most_common():
            print(f"  {c:4d}  {p}")
    else:
        print("  none")

    if panels and args.montage:
        fig, axes = plt.subplots(2, len(panels), figsize=(5.2 * len(panels), 10),
                                 squeeze=False)
        colors = dict(carina="r", lung_apex_left="c", lung_apex_right="c",
                      diaphragm_dome_left="y", diaphragm_dome_right="y")
        for col, (pid, vol, lm) in enumerate(panels):
            for row, orient in enumerate(("PA", "LAT")):
                view = DRRView.standard(vol, orient)
                img = view.render(n_samples=args.samples)
                ax = axes[row][col]
                lo, hi = np.percentile(img[img > 0], (0.5, 99.5))
                ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
                for k in E1_SET:
                    d = lm.get(k, {})
                    if not d.get("valid"):
                        continue
                    u, v, ok = view.project(torch.tensor(d["mm"], dtype=torch.float64))
                    if not bool(ok[0]):
                        continue
                    ax.plot(float(u[0]) - 0.5, float(v[0]) - 0.5, "+",
                            color=colors[k], ms=13, mew=2)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{pid} {orient}", fontsize=8)
        fig.suptitle("landmarks projected onto the DRRs   "
                     "red=carina  cyan=apices  yellow=domes", fontsize=12)
        fig.savefig(args.montage, dpi=110, bbox_inches="tight")
        print(f"\n-> {args.montage}   LOOK AT THIS before building anything")
    return 0


def selftest() -> int:
    from geometry_kernel import synthetic_thorax

    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    vol = synthetic_thorax()
    lm = extract(vol)
    print(f"  phantom landmarks: {sorted(lm)}\n")

    lc = lm.get("lung_centroid_left", {})
    rc = lm.get("lung_centroid_right", {})
    t("both lung centroids found", lc.get("valid") and rc.get("valid"))
    if lc.get("valid") and rc.get("valid"):
        # phantom lungs are at x = +-62mm; +x is patient left
        t("left lung is at +x, right at -x", lc["mm"][0] > 20 and rc["mm"][0] < -20,
          f"left {lc['mm'][0]:.1f}, right {rc['mm'][0]:.1f}")
        t("lung centroids are symmetric about the midline",
          abs(lc["mm"][0] + rc["mm"][0]) < 25,
          f"sum {lc['mm'][0] + rc['mm'][0]:.1f} mm")
        t("centroids sit inside the phantom's lungs",
          abs(abs(lc["mm"][0]) - 62) < 25 and abs(abs(rc["mm"][0]) - 62) < 25)

    al, ar = lm.get("lung_apex_left", {}), lm.get("lung_apex_right", {})
    if al.get("valid") and lc.get("valid"):
        t("apex is superior to the centroid", al["mm"][2] > lc["mm"][2],
          f"apex z {al['mm'][2]:.1f} > centroid z {lc['mm'][2]:.1f}")
    dl = lm.get("diaphragm_dome_left", {})
    if dl.get("valid") and lc.get("valid"):
        t("diaphragm dome is inferior to the centroid", dl["mm"][2] < lc["mm"][2],
          f"dome z {dl['mm'][2]:.1f} < centroid z {lc['mm'][2]:.1f}")

    sp = lm.get("spine_centroid", {})
    if sp.get("valid"):
        t("spine is posterior to the lung centroids (+y = posterior)",
          sp["mm"][1] > max(lc["mm"][1], rc["mm"][1]),
          f"spine y {sp['mm'][1]:.1f} vs lungs {lc['mm'][1]:.1f}/{rc['mm'][1]:.1f}")
        t("spine is near the midline", abs(sp["mm"][0]) < 30, f"x {sp['mm'][0]:.1f}")

    # the phantom has no airway, so the carina must fail rather than invent one
    t("carina reports failure on a phantom with no airway",
      not lm.get("carina", {}).get("valid"),
      lm.get("carina", {}).get("reason", ""))

    print("\n" + ("all self-tests passed" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", help="a CT series directory")
    ap.add_argument("--scan", help="LIDC root: run over many series and report")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--montage", default="landmarks.png")
    ap.add_argument("--montage-n", type=int, default=3)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.scan:
        return scan(args)
    if not args.ct:
        ap.error("--ct, --scan or --selftest is required")

    from geometry_kernel import load_ct

    vol = load_ct(args.ct)
    lm = extract(vol)
    for name, d in sorted(lm.items()):
        if d.get("valid"):
            print(f"  {name:24} {[round(v, 1) for v in d['mm']]} mm")
        else:
            print(f"  {name:24} INVALID: {d.get('reason')}")
    print(json.dumps({k: v for k, v in lm.items()}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
