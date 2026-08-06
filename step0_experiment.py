"""
Step 0, the full experiment.

Sweeps every CT volume it can find x every annotated nodule x every view type,
and records one row per (case, nodule, view).  Writes:

    results/step0/results.csv     one row per (case, nodule, view)
    results/step0/results.json    the same, plus per-case geometry checks
    results/step0/REPORT.md       the write-up, with the failure list up top
    figures/step0/case_*.png      a contact sheet per nodule

    python step0_experiment.py                       # everything under the project
    python step0_experiment.py --roots /data/lidc     # a real LIDC tree
    python step0_experiment.py --max-nodules 5 --samples 256

Two gates decide whether Step 0 is done:

  0.2  the marker lands on the nodule.  On real data the honest version is the
       SOP UID check -- the XML names the slice the radiologist annotated, so
       if our affine sends imageZposition anywhere else, we are wrong.
  0.4  the circle lands on the nodule's shadow.  Attenuation is linear along a
       ray, so DRR(with) - DRR(without) is exactly the render of a nodule-only
       volume; its argmax must be the pixel we projected to.  That works for a
       3mm nodule that no human could pick out of a DRR by eye.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry_kernel import (
    AIR_HU, CTVolume, DRRView, MIPSlabView, SliceView, _read_tag, contour_to_mm,
    find_ct_series, load_ct, nodule_difference_volume, parse_lidc_ct_xml,
    ray_consistency, slices_seeing, synthetic_thorax, triangulate,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures", "step0")
OUT = os.path.join(HERE, "results", "step0")
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------------------
# case discovery
# --------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    ct_dir: Optional[str] = None
    xml: Optional[str] = None
    synthetic: bool = False


def discover(roots: List[str]) -> List[Case]:
    """Find CT series, and pair each with the XML that names its SeriesInstanceUid."""
    cases: List[Case] = []
    xmls: List[Tuple[str, str]] = []          # (path, series uid it refers to)
    for root in roots:
        for dirpath, _d, files in os.walk(root):
            if ".venv" in dirpath:
                continue
            for f in files:
                if f.endswith(".xml"):
                    path = os.path.join(dirpath, f)
                    try:
                        m = re.search(r"SeriesInstanceUid>\s*([\d.]+)",
                                      open(path, errors="ignore").read())
                        if m:
                            xmls.append((path, m.group(1)))
                    except Exception:
                        pass

    seen = set()
    for root in roots:
        for d in find_ct_series(root):
            if d in seen:
                continue
            seen.add(d)
            try:
                uid = _series_uid(d)
            except Exception:
                uid = None
            xml = next((p for p, u in xmls if u == uid), None)
            cases.append(Case(name=os.path.basename(d.rstrip("/"))[:28] or d,
                              ct_dir=d, xml=xml))
    cases.append(Case(name="phantom", synthetic=True))
    return cases


def _series_uid(ct_dir: str) -> str:
    f = sorted(x for x in os.listdir(ct_dir) if x.lower().endswith(".dcm"))[0]
    return _read_tag(os.path.join(ct_dir, f), "0020|000e")


# --------------------------------------------------------------------------
# per-nodule measurement
# --------------------------------------------------------------------------


def nodule_points(vol: CTVolume, xml: str) -> List[Dict]:
    """Every annotated nodule in the XML, as a centroid + an extent."""
    out = []
    for n in parse_lidc_ct_xml(xml):
        mm = contour_to_mm(vol, n["points"])
        c = mm.mean(dim=0)
        if mm.shape[0] > 1:
            radius = max(1.5, float((mm - c).norm(dim=1).max()))
        else:
            radius = 1.5          # a <3mm single-locus mark
        out.append(dict(nodule_id=n["nodule_id"], session=n["session"],
                        n_points=n["n_points"], center_mm=c, radius_mm=radius,
                        sop_uid=n["points"][0]["sop_uid"]))
    return out


def measure_views(vol: CTVolume, p: torch.Tensor, radius_mm: float,
                  n_samples: int) -> Tuple[List[Dict], Dict]:
    """One row per view type, plus the rendered images for the contact sheet."""
    rows, art = [], {}

    planar = [
        ("axial", SliceView.through_point(vol, 2, p, thickness_mm=2.5)),
        ("coronal", SliceView.through_point(vol, 1, p, thickness_mm=2.5)),
        ("sagittal", SliceView.through_point(vol, 0, p, thickness_mm=2.5)),
        ("mip-axial", MIPSlabView.through_point(vol, 2, p, thickness_mm=20.0)),
    ]
    for kind, view in planar:
        sw = view.w(p)
        uc, ur = view.in_plane_pixels(p)
        want = int(view.grid.containing_patch(uc, ur)[0])
        pairs = dict(sw.as_list())
        best = max(pairs.values()) if pairs else 0.0
        rows.append(dict(
            view=kind, n_patches=len(sw), total_weight=float(sw.total.sum()),
            grid=f"{view.grid.n_rows}x{view.grid.n_cols}",
            pixel_col=float(uc), pixel_row=float(ur), top_patch=want,
            top_is_containing=bool(pairs and pairs.get(want, 0.0) >= best - 1e-9),
            shadow_err_px="", shadow_err_mm="", peak_signal=""))
        art[kind] = view

    for orient in ("PA", "LAT"):
        view = DRRView.standard(vol, orient)
        uc, ur, ok = view.project(p)
        uc, ur = float(uc[0]), float(ur[0])
        sw = view.w(p)
        want = int(view.grid.containing_patch(uc, ur)[0])
        pairs = dict(sw.as_list())
        best = max(pairs.values()) if pairs else 0.0

        # the gate: render the difference volume and find its argmax
        err_px = err_mm = peak = float("nan")
        try:
            dv, _fill = nodule_difference_volume(vol, p, radius_mm)
            shadow = view.with_volume(dv).render(n_samples=n_samples)
            r, c = np.unravel_index(int(np.argmax(shadow)), shadow.shape)
            err_px = math.hypot(c + 0.5 - uc, r + 0.5 - ur)
            err_mm = err_px * view.det_spacing[0]
            peak = float(shadow.max())
            art[f"shadow-{orient}"] = shadow
        except Exception as e:
            art[f"shadow-{orient}-error"] = f"{type(e).__name__}: {e}"

        rows.append(dict(
            view=f"drr-{orient.lower()}", n_patches=len(sw),
            total_weight=float(sw.total.sum()),
            grid=f"{view.grid.n_rows}x{view.grid.n_cols}",
            pixel_col=uc, pixel_row=ur, top_patch=want,
            top_is_containing=bool(pairs and pairs.get(want, 0.0) >= best - 1e-9),
            shadow_err_px=err_px, shadow_err_mm=err_mm, peak_signal=peak))
        art[f"drr-{orient}"] = view
    return rows, art


def measure_sanity(vol: CTVolume, p: torch.Tensor, views) -> Dict:
    pa, lat = views["drr-PA"], views["drr-LAT"]
    spread, _ = ray_consistency(pa, p, fracs=(0.4, 0.7, 1.0, 1.3))
    ucp, urp, _ = pa.project(p)
    ucl, url, _ = lat.project(p)
    hat = triangulate([pa, lat], [(float(ucp[0]), float(urp[0])),
                                  (float(ucl[0]), float(url[0]))])
    tri = float(torch.linalg.norm(hat - p))

    step = float(vol.spacing[2])
    k = float(vol.mm_to_voxel(p)[2])
    nz = vol.size_ijk[2]
    counts, ok = [], True
    for mult in (1.0, 2.0, 3.0, 4.5):
        h = mult * step / 2.0 / step
        want = sum(1 for i in range(nz) if abs(i - k) < h)
        got = len(slices_seeing(vol, p, axis=2, thickness_mm=mult * step))
        counts.append(f"{mult:g}x:{got}/{want}")
        ok &= got == want
    return dict(ray_consistency_px=spread, triangulation_mm=tri,
                slice_counts=" ".join(counts), slice_counts_ok=ok)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def contact_sheet(case: str, nid: str, vol: CTVolume, p: torch.Tensor,
                  art: Dict, n_samples: int) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2))

    ax = axes[0, 0]
    view = art["axial"]
    img = view.render()
    uc, ur = (float(x) for x in view.in_plane_pixels(p))
    ax.imshow(img, cmap="gray", vmin=-1000, vmax=400)
    ax.plot(uc - 0.5, ur - 0.5, "r+", ms=16, mew=1.8)
    ax.set_title(f"{view.name}  marker must be ON the nodule", fontsize=9)

    ax = axes[0, 1]
    r = 30
    c0, r0 = int(uc), int(ur)
    y0, x0 = max(0, r0 - r), max(0, c0 - r)
    ax.imshow(img[y0:r0 + r, x0:c0 + r], cmap="gray", vmin=-1000, vmax=400)
    ax.plot(uc - x0 - 0.5, ur - y0 - 0.5, "r+", ms=20, mew=1.8)
    ax.set_title("zoom (60px)", fontsize=9)

    ax = axes[0, 2]
    d = view.w(p).dense()[0].numpy()
    ax.imshow(img, cmap="gray", vmin=-1000, vmax=400)
    ax.imshow(d, cmap="autumn", alpha=(d > 0) * 0.6, interpolation="nearest",
              extent=(0, view.source_hw[1], view.source_hw[0], 0))
    ax.set_xlim(x0, c0 + r); ax.set_ylim(r0 + r, y0)
    ax.set_title(f"w over patches ({view.grid.n_rows}x{view.grid.n_cols})", fontsize=9)

    for col, orient in enumerate(("PA", "LAT")):
        ax = axes[1, col]
        view = art[f"drr-{orient}"]
        drr = view.render(n_samples=n_samples)
        uc, ur, _ = view.project(p)
        uc, ur = float(uc[0]), float(ur[0])
        lo, hi = np.percentile(drr[drr > 0], (0.5, 99.5)) if np.any(drr > 0) else (0, 1)
        ax.imshow(drr, cmap="gray", vmin=lo, vmax=hi)
        ax.add_patch(plt.Circle((uc - 0.5, ur - 0.5), 20, ec="r", fc="none", lw=1.5))
        ax.set_title(f"{view.name}  circle must sit on the shadow", fontsize=9)

        if col == 0 and f"shadow-{orient}" in art:
            axz = axes[1, 2]
            shadow = art[f"shadow-{orient}"]
            h = 26
            sy0, sx0 = max(0, int(ur) - h), max(0, int(uc) - h)
            axz.imshow(shadow[sy0:int(ur) + h, sx0:int(uc) + h], cmap="magma")
            axz.add_patch(plt.Circle((uc - sx0 - 0.5, ur - sy0 - 0.5), 8,
                                     ec="cyan", fc="none", lw=1.5))
            axz.set_title("nodule shadow, isolated\nDRR(with) - DRR(without), PA",
                          fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{case}  --  {nid}", fontsize=12)
    path = os.path.join(FIG, f"case_{_slug(case)}_{_slug(nid)}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower()[:40]


# --------------------------------------------------------------------------


def geometry_checks(vol: CTVolume) -> Dict:
    out = {}
    mm0 = vol.voxel_to_mm(torch.zeros(3, dtype=torch.float64))
    out["origin_err_mm"] = float((mm0 - torch.tensor(vol.origin,
                                                     dtype=torch.float64)).abs().max())
    v = torch.tensor([[3.0, 17.0, 5.0], [200.0, 100.0, 40.0]], dtype=torch.float64)
    out["inverse_err_voxel"] = float((v - vol.mm_to_voxel(vol.voxel_to_mm(v))).abs().max())
    if not vol.meta.get("synthetic"):
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol.meta["files"])
        img = reader.Execute()
        idx = tuple(int(s // 2) for s in vol.size_ijk)
        native = torch.tensor(img.TransformIndexToPhysicalPoint(idx), dtype=torch.float64)
        out["sitk_err_mm"] = float((native - vol.voxel_to_mm(
            torch.tensor(idx, dtype=torch.float64))).abs().max())
        out["slice_step_mm"] = vol.meta["slice_step_mm"]
        out["slice_thickness_tag"] = vol.meta.get("slice_thickness_tag")
    return out


def run(args) -> int:
    cases = discover(args.roots)
    print(f"discovered {len(cases)} case(s): " +
          ", ".join(f"{c.name}{'*' if c.xml else ''}" for c in cases))
    print("  (* = has a matching annotation XML)\n")

    rows, case_records, failures = [], [], []
    t0 = time.time()

    for case in cases:
        print(f"=== {case.name} " + "=" * max(0, 60 - len(case.name)))
        if case.synthetic:
            vol = synthetic_thorax()
            nodules = [dict(nodule_id="synthetic", session=-1, n_points=0,
                            center_mm=torch.tensor(vol.meta["nodule_mm"],
                                                   dtype=torch.float64),
                            radius_mm=vol.meta["nodule_radius_mm"], sop_uid="")]
        else:
            try:
                vol = load_ct(case.ct_dir)
            except Exception as e:
                print(f"  SKIP: {type(e).__name__}: {e}")
                case_records.append(dict(case=case.name, error=str(e)))
                continue
            nodules = nodule_points(vol, case.xml) if case.xml else []

        g = geometry_checks(vol)
        print(f"  {vol.size_ijk} @ {tuple(round(s, 4) for s in vol.spacing)} mm   "
              f"origin err {g['origin_err_mm']:.1e} mm, inverse err "
              f"{g['inverse_err_voxel']:.1e} vox" +
              (f", vs SimpleITK {g['sitk_err_mm']:.1e} mm" if "sitk_err_mm" in g else ""))
        for k in ("origin_err_mm", "inverse_err_voxel", "sitk_err_mm"):
            if g.get(k, 0.0) > 1e-6:
                failures.append(f"{case.name}: {k} = {g[k]:.2e}")
        if not nodules:
            print("  no annotated nodule (geometry checked, gates not exercised)")
        case_records.append(dict(case=case.name, ct_dir=case.ct_dir, xml=case.xml,
                                 size=list(vol.size_ijk), spacing=list(vol.spacing),
                                 n_nodules=len(nodules), **g))

        for nod in nodules[:args.max_nodules]:
            p, nid = nod["center_mm"], nod["nodule_id"]
            vx = vol.mm_to_voxel(p)
            hu = float(vol.sample_hu(vx))

            sop_ok = None
            if nod["sop_uid"] and not case.synthetic:
                k = int(round(float(vx[2])))
                got = (_read_tag(vol.meta["files"][k], "0008|0018")
                       if 0 <= k < len(vol.meta["files"]) else "")
                sop_ok = got == nod["sop_uid"]
                if not sop_ok:
                    failures.append(f"{case.name}/{nid}: SOP UID mismatch at slice {k}")

            print(f"  {nid}: {[round(float(x), 2) for x in p]} mm, voxel "
                  f"{[round(float(x), 1) for x in vx]}, {hu:.0f} HU, r={nod['radius_mm']:.1f}mm"
                  + ("" if sop_ok is None else f", SOP {'OK' if sop_ok else 'MISMATCH'}"))

            vrows, art = measure_views(vol, p, nod["radius_mm"], args.samples)
            views = {k: v for k, v in art.items() if k.startswith("drr-")}
            san = measure_sanity(vol, p, views)

            for r in vrows:
                if not r["top_is_containing"]:
                    failures.append(f"{case.name}/{nid}/{r['view']}: containing patch "
                                    "is not top-weighted")
                if r["view"].startswith("drr") and isinstance(r["shadow_err_px"], float):
                    if not (r["shadow_err_px"] < 4.0):
                        failures.append(f"{case.name}/{nid}/{r['view']}: shadow off by "
                                        f"{r['shadow_err_px']:.1f}px")
                rows.append(dict(case=case.name, nodule=nid, n_points=nod["n_points"],
                                 hu=round(hu, 1), radius_mm=round(nod["radius_mm"], 2),
                                 sop_ok="" if sop_ok is None else sop_ok, **r, **san))
            if san["ray_consistency_px"] > 1e-6:
                failures.append(f"{case.name}/{nid}: ray consistency "
                                f"{san['ray_consistency_px']:.2e} px")
            if san["triangulation_mm"] > 1e-6:
                failures.append(f"{case.name}/{nid}: triangulation "
                                f"{san['triangulation_mm']:.2e} mm")
            if not san["slice_counts_ok"]:
                failures.append(f"{case.name}/{nid}: slice counts {san['slice_counts']}")

            shadows = [r["shadow_err_px"] for r in vrows
                       if isinstance(r.get("shadow_err_px"), float)]
            print(f"     shadow err {['%.2f px' % s for s in shadows]}, "
                  f"ray {san['ray_consistency_px']:.1e} px, "
                  f"tri {san['triangulation_mm']:.1e} mm, slices {san['slice_counts']}")
            if not args.no_figures:
                print(f"     -> {contact_sheet(case.name, nid, vol, p, art, args.samples)}")
        print()

    write_outputs(rows, case_records, failures, time.time() - t0, args)
    return 1 if failures else 0


def write_outputs(rows, case_records, failures, secs, args):
    if rows:
        with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dict(cases=case_records, rows=rows, failures=failures,
                       seconds=round(secs, 1), samples=args.samples), f, indent=2)

    nod = sorted({(r["case"], r["nodule"]) for r in rows})
    shadow = [r["shadow_err_px"] for r in rows if isinstance(r.get("shadow_err_px"), float)]
    lines = [
        "# Step 0 -- full experiment", "",
        f"{len(case_records)} volume(s), {len(nod)} nodule(s), {len(rows)} "
        f"(nodule x view) measurements, {secs:.0f}s, {args.samples} samples/ray.", "",
        f"**{'FAILURES: ' + str(len(failures)) if failures else 'All checks passed.'}**", "",
    ]
    for f in failures:
        lines.append(f"- {f}")
    if failures:
        lines.append("")

    lines += ["## Volumes", "",
              "| case | size | spacing (mm) | origin err | A vs SimpleITK | nodules |",
              "|---|---|---|---|---|---|"]
    for c in case_records:
        if "error" in c:
            lines.append(f"| {c['case']} | load failed: {c['error'][:60]} | | | | |")
            continue
        lines.append(
            f"| {c['case']} | {'x'.join(map(str, c['size']))} | "
            f"{', '.join(f'{s:.3f}' for s in c['spacing'])} | "
            f"{c['origin_err_mm']:.1e} | "
            f"{c.get('sitk_err_mm', float('nan')):.1e} | {c['n_nodules']} |")

    lines += ["", "## Per-nodule, per-view", "",
              "| case | nodule | HU | view | patches | total w | top=containing | "
              "shadow err (px) | shadow err (mm) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        se = (f"{r['shadow_err_px']:.2f}" if isinstance(r.get("shadow_err_px"), float) else "")
        sm = (f"{r['shadow_err_mm']:.2f}" if isinstance(r.get("shadow_err_mm"), float) else "")
        lines.append(f"| {r['case']} | {r['nodule']} | {r['hu']:.0f} | {r['view']} | "
                     f"{r['n_patches']} | {r['total_weight']:.3f} | "
                     f"{'yes' if r['top_is_containing'] else 'NO'} | {se} | {sm} |")

    lines += ["", "## Sanity suite", "",
              "| case | nodule | ray consistency (px) | triangulation (mm) | "
              "slice counts got/want |", "|---|---|---|---|---|"]
    for cn in nod:
        r = next(x for x in rows if (x["case"], x["nodule"]) == cn)
        lines.append(f"| {cn[0]} | {cn[1]} | {r['ray_consistency_px']:.2e} | "
                     f"{r['triangulation_mm']:.2e} | {r['slice_counts']} |")

    if shadow:
        lines += ["", "## The day-2 gate", "",
                  f"Shadow localisation error over {len(shadow)} (nodule x DRR view) "
                  f"pairs: max **{max(shadow):.2f} px**, mean {np.mean(shadow):.2f} px.",
                  "", "Measured as the argmax of DRR(with nodule) - DRR(without), which "
                  "equals the render of a nodule-only volume because attenuation "
                  "integrates linearly along a ray. This does not depend on the nodule "
                  "being visible to a human in the DRR."]

    faint = [r for r in rows if r["total_weight"] < 0.1]
    if faint:
        lines += ["", "## Views that cannot see their nodule", "",
                  "These are not failures. `w` returning ~0 means the view genuinely "
                  "does not show that point, which is the fact the architecture is "
                  "built on -- supervising attention here would be supervising a lie.", "",
                  "| case | nodule | view | total w | why |", "|---|---|---|---|---|"]
        for r in faint:
            why = ("something denser occupies the same column, so the MIP displays "
                   "that instead" if r["view"].startswith("mip")
                   else "projects outside the detector, or is occluded")
            lines.append(f"| {r['case']} | {r['nodule']} | {r['view']} | "
                         f"{r['total_weight']:.3f} | {why} |")

    lines += ["", "## Coverage", "",
              "The spec asks for five nodules across five patients. That is bounded by "
              "the data on disk, not by the code: only annotated series can exercise the "
              "gates. Point `--roots` at a larger LIDC tree and this sweeps it unchanged.", ""]

    path = os.path.join(OUT, "REPORT.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("=" * 68)
    print(f"{len(rows)} measurements over {len(nod)} nodule(s) in {secs:.0f}s")
    if shadow:
        print(f"day-2 gate: max shadow error {max(shadow):.2f} px "
              f"({max(shadow) * 1.0:.2f} px), mean {np.mean(shadow):.2f} px")
    print(f"{'FAILURES: ' + str(len(failures)) if failures else 'all checks passed'}")
    for f in failures:
        print(f"  - {f}")
    print(f"report  -> {path}")
    print(f"csv     -> {os.path.join(OUT, 'results.csv')}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[HERE],
                    help="directories to search for CT series and XMLs")
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--max-nodules", type=int, default=10)
    ap.add_argument("--no-figures", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
