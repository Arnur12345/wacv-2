"""
Step 0, the tests -- 0.1 through 0.5.

    python step0_tests.py                      # phantom (works with no data)
    python step0_tests.py --ct /path/to/series # a real LIDC CT series
    python step0_tests.py --ct ... --xml nodule.xml

Every check prints PASS/FAIL and writes a picture into figures/step0/.
Geometry bugs are silent, so each visual test also has a numeric twin: on the
phantom the nodule's mm coordinate is known exactly, so "the marker is on the
nodule" becomes "the HU at that voxel is the nodule's HU", and "the circle is
on the shadow" becomes "the argmax of (DRR with nodule - DRR without) is the
pixel we projected to".
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry_kernel import (
    PATCH, CTVolume, DRRView, MIPSlabView, SliceView, contour_to_mm, find_ct_series,
    load_ct, parse_lidc_ct_xml, parse_lidc_cxr_xml, ray_consistency, slices_seeing,
    synthetic_thorax, triangulate,
)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "step0")
os.makedirs(FIG, exist_ok=True)

_results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def skip(name: str, why: str):
    print(f"  [SKIP] {name}  --  {why}")


def top_patch_ok(view, p) -> Tuple[bool, str]:
    """
    The patch containing the point must carry the maximum weight -- allowing
    ties, because a point exactly on a patch boundary genuinely splits its
    weight between two patches and picking a winner there is arbitrary.
    """
    uc, ur = (view.in_plane_pixels(p) if hasattr(view, "in_plane_pixels")
              else view.project(p)[:2])
    want = int(view.grid.containing_patch(uc, ur)[0])
    pairs = dict(view.w(p).as_list())
    if not pairs:
        return False, "w is empty"
    best = max(pairs.values())
    got = pairs.get(want, 0.0)
    return (got >= best - 1e-9,
            f"patch {want} has w={got:.4f}, max is {best:.4f}"
            + ("  (exact boundary tie)" if abs(got - best) < 1e-9 and
               sum(abs(v - best) < 1e-9 for v in pairs.values()) > 1 else ""))


def head(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def save(fig, name: str) -> str:
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"       -> {path}")
    return path


def show_ct(ax, img, vmin=-1000, vmax=400):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------
# 0.1  get the affine out of the DICOM
# --------------------------------------------------------------------------


def step_0_1(vol: CTVolume):
    head("0.1  the affine")
    print(f"  size (i,j,k) = {vol.size_ijk}   spacing = "
          f"{tuple(round(s, 4) for s in vol.spacing)}")
    print(f"  origin       = {tuple(round(o, 3) for o in vol.origin)}")
    print(f"  direction    = {np.round(vol.direction, 4).tolist()}")

    mm0 = vol.voxel_to_mm(torch.zeros(3, dtype=torch.float64))
    exp = torch.tensor(vol.origin, dtype=torch.float64)
    check("voxel (0,0,0) -> ImagePositionPatient",
          torch.allclose(mm0, exp, atol=1e-6),
          f"{[round(float(x), 3) for x in mm0]}")

    c_mm = vol.center_mm
    plausible = (abs(float(c_mm[0])) < 250 and abs(float(c_mm[1])) < 250)
    extent = [round(float(s * (n - 1)), 1) for s, n in zip(vol.spacing, vol.size_ijk)]
    check("centre voxel lands in a plausible chest",
          plausible, f"centre {[round(float(x), 1) for x in c_mm]} mm, extent {extent} mm")

    v = torch.tensor([[3.0, 17.0, 5.0], [200.0, 100.0, 40.0]], dtype=torch.float64)
    back = vol.mm_to_voxel(vol.voxel_to_mm(v))
    check("A_inv is the exact inverse of A", torch.allclose(v, back, atol=1e-6),
          f"max err {float((v - back).abs().max()):.2e} voxel")

    if not vol.meta.get("synthetic"):
        import SimpleITK as sitk

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(vol.meta["files"])
        img = reader.Execute()
        idx = tuple(int(s // 2) for s in vol.size_ijk)
        native = torch.tensor(img.TransformIndexToPhysicalPoint(idx), dtype=torch.float64)
        ours = vol.voxel_to_mm(torch.tensor(idx, dtype=torch.float64))
        check("A agrees with SimpleITK's own index->physical transform",
              torch.allclose(native, ours, atol=1e-6),
              f"max err {float((native - ours).abs().max()):.2e} mm")

        step = vol.meta["slice_step_mm"]
        check("slice step derived from positions, not SliceThickness",
              math.isfinite(step) and abs(step - vol.spacing[2]) < 1e-3,
              f"step {step:.4f} mm, affine z-spacing {vol.spacing[2]:.4f} mm, "
              f"SliceThickness tag {vol.meta.get('slice_thickness_tag')!r}")

    # picture: three orthogonal slices through the centre
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, axis in zip(axes, (2, 1, 0)):
        sv = SliceView.through_point(vol, axis, c_mm, thickness_mm=3.0)
        show_ct(ax, sv.render())
        ax.set_title(sv.name, fontsize=9)
    fig.suptitle("0.1  the volume through its centre", fontsize=11)
    save(fig, "0_1_affine.png")


# --------------------------------------------------------------------------
# 0.2  land a nodule
# --------------------------------------------------------------------------


def step_0_2(vol: CTVolume, nodule_mm, label: str = "nodule", is_nodule: bool = True):
    head("0.2  land a nodule")
    if not is_nodule:
        print("  NOTE: no nodule available (no --xml), so this is the volume centre.\n"
              "        The affine round-trips below are still real tests; the\n"
              "        'is it on the nodule' gate is NOT being tested.")
    p = torch.tensor([float(c) for c in nodule_mm], dtype=torch.float64)
    v = vol.mm_to_voxel(p)
    print(f"  {label} at {[round(float(x), 2) for x in p]} mm "
          f"-> voxel {[round(float(x), 2) for x in v]}")

    check("nodule is inside the volume", bool(vol.contains_voxel(v)),
          f"size {vol.size_ijk}")

    back = vol.voxel_to_mm(v)
    check("mm -> voxel -> mm round-trips",
          torch.allclose(p, back, atol=1e-6),
          f"max err {float((p - back).abs().max()):.2e} mm")

    # The strongest real-data check available: the XML names the SOP Instance
    # UID of the slice the radiologist was looking at.  If our affine sends the
    # XML's imageZposition to a different slice, the z mapping is wrong -- and
    # this catches it without anyone having to squint at a 3mm nodule.
    sop = (vol.meta.get("nodule_sop_uid") or "")
    if sop and not vol.meta.get("synthetic"):
        from geometry_kernel import _read_tag

        k = int(round(float(v[2])))
        got = _read_tag(vol.meta["files"][k], "0008|0018") if 0 <= k < len(vol.meta["files"]) else ""
        check("z maps to the exact slice the reader annotated (SOP UID match)",
              got == sop, f"slice {k}: {got[-24:]} vs XML {sop[-24:]}")

    hu = float(vol.sample_hu(v))
    if vol.meta.get("synthetic"):
        want = vol.meta["nodule_hu"]
        check("HU at the marked voxel is the nodule, not lung",
              abs(hu - want) < 25.0, f"HU {hu:.1f} (nodule {want}, lung ~-820)")
    elif is_nodule:
        check("HU at the marked voxel is nodule-like, not air or lung",
              hu > -400.0, f"HU {hu:.1f}")
    else:
        skip("HU at the marked voxel is the nodule", f"not a nodule; HU here is {hu:.1f}")

    sv = SliceView.through_point(vol, 2, p, thickness_mm=2.0)
    img = sv.render()
    u_col, u_row = sv.in_plane_pixels(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    show_ct(axes[0], img)
    axes[0].plot(float(u_col) - 0.5, float(u_row) - 0.5, "r+", ms=18, mew=2)
    axes[0].set_title(f"{sv.name}  --  marker must be ON the nodule", fontsize=9)

    r = 26
    c0, r0 = int(u_col), int(u_row)
    crop = img[max(0, r0 - r):r0 + r, max(0, c0 - r):c0 + r]
    show_ct(axes[1], crop)
    axes[1].plot(float(u_col) - max(0, c0 - r) - 0.5, float(u_row) - max(0, r0 - r) - 0.5,
                 "r+", ms=22, mew=2)
    axes[1].set_title("zoom", fontsize=9)
    fig.suptitle("0.2  the single most important test in the project", fontsize=11)
    save(fig, "0_2_nodule_axial.png")
    return p


# --------------------------------------------------------------------------
# 0.3  w for the easy views
# --------------------------------------------------------------------------


def step_0_3(vol: CTVolume, p: torch.Tensor):
    head("0.3  w for slices, reformats and MIP slabs")

    sv = SliceView.through_point(vol, 2, p, thickness_mm=2.0)
    sw = sv.w(p)
    print(f"  patch grid: source {sv.source_hw} -> model {sv.grid.model_hw} "
          f"= {sv.grid.n_rows}x{sv.grid.n_cols} patches of {PATCH}px")
    print(f"  w returned {len(sw)} (patch, weight) pairs, total "
          f"{float(sw.total.sum()):.3f}")
    check("w is nonzero for a point in the slab", len(sw) > 0 and float(sw.total.sum()) > 0.9)
    check("weights sum to <= 1", float(sw.total.sum()) <= 1.0 + 1e-9,
          f"{float(sw.total.sum()):.4f}")

    u_col, u_row = sv.in_plane_pixels(p)
    top_idx = int(sv.grid.containing_patch(u_col, u_row)[0])
    top_w = dict(sw.as_list()).get(top_idx, 0.0)
    x0, y0, x1, y1 = sv.grid.patch_bbox_source(top_idx)
    check("the patch containing the point carries the top weight", *top_patch_ok(sv, p))

    img = sv.render()
    crop = img[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))]
    if vol.meta.get("synthetic"):
        want = vol.meta["nodule_hu"]
        check("the top patch, cropped, actually contains nodule tissue",
              bool(np.any(np.abs(crop - want) < 25.0)),
              f"crop HU range [{crop.min():.0f}, {crop.max():.0f}]")

    far = p.clone(); far[2] += 40.0
    check("w is empty for a point outside the slab",
          len(sv.w(far)) == 0)

    # reformats: same function, axes permuted
    for axis, nm in ((1, "coronal"), (0, "sagittal")):
        rv = SliceView.through_point(vol, axis, p, thickness_mm=3.0)
        check(f"reformat ({nm}) is the same w with axes permuted", *top_patch_ok(rv, p))

    # MIP: the rect becomes a soft argmax over the segment.  Test it against
    # the column's own extremes, which is meaningful on any data: whatever is
    # densest along the segment is what the MIP shows and must score ~1, and
    # the least dense point on the same column must score far lower.
    thick = 30.0
    mv = MIPSlabView.through_point(vol, 2, p, thickness_mm=thick)
    v = vol.mm_to_voxel(p)
    m = 25
    probe = v.repeat(m, 1)
    probe[:, 2] = v[2] + torch.linspace(-thick / 2, thick / 2, m,
                                        dtype=torch.float64) / vol.spacing[2]
    hu = vol.sample_hu(probe)
    keep = vol.contains_voxel(probe)
    hu = torch.where(keep, hu, torch.full_like(hu, -1e4))
    p_hi = vol.voxel_to_mm(probe[int(hu.argmax())])
    p_lo = vol.voxel_to_mm(probe[int(hu.argmin())])
    w_hi = float(mv.w(p_hi).total.sum())
    w_lo = float(mv.w(p_lo).total.sum())
    if float(hu.max() - hu[keep].min()) < 200.0:
        skip("MIP slab: soft argmax over the segment", "column is too uniform to tell")
    else:
        check("MIP slab replaces the rect with a soft-max over the segment",
              w_hi > 0.99 and w_lo < 0.5,
              f"densest point on the column {w_hi:.3f} "
              f"({float(hu.max()):.0f} HU) vs least dense {w_lo:.2e} "
              f"({float(hu[keep].min()):.0f} HU)")

    coarse = MIPSlabView.through_point(vol, 2, p, thickness_mm=thick, n_depth=9)
    fine = MIPSlabView.through_point(vol, 2, p, thickness_mm=thick, n_depth=61)
    check("MIP weight does not depend on the depth sampling density",
          abs(float(coarse.w(p_hi).total.sum()) - float(fine.w(p_hi).total.sum())) < 0.02,
          f"9 samples {float(coarse.w(p_hi).total.sum()):.3f} vs "
          f"61 samples {float(fine.w(p_hi).total.sum()):.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, view in zip(axes[:3],
                        [sv,
                         SliceView.through_point(vol, 1, p, thickness_mm=3.0),
                         SliceView.through_point(vol, 0, p, thickness_mm=3.0)]):
        show_ct(ax, view.render())
        uc, ur = view.in_plane_pixels(p)
        ax.plot(float(uc) - 0.5, float(ur) - 0.5, "r+", ms=15, mew=1.6)
        d = view.w(p).dense()[0].numpy()
        ax.imshow(d, cmap="autumn", alpha=(d > 0) * 0.55,
                  extent=(0, view.source_hw[1], view.source_hw[0], 0),
                  interpolation="nearest")
        ax.set_title(f"{view.name}\nw over patches", fontsize=8)
    show_ct(axes[3], crop)
    axes[3].set_title(f"top patch #{top_idx} (w={top_w:.3f}) cropped", fontsize=8)
    fig.suptitle("0.3  w lives in patch space, in every view", fontsize=11)
    save(fig, "0_3_patch_weights.png")


# --------------------------------------------------------------------------
# 0.4  the DRR ray -- the gate
# --------------------------------------------------------------------------


def step_0_4(vol: CTVolume, p: torch.Tensor, n_samples: int = 256):
    head("0.4  the DRR ray")
    views = {o: DRRView.standard(vol, o) for o in ("PA", "LAT")}

    figs = {}
    for o, view in views.items():
        print(f"  {view.name}: source {[round(float(x), 1) for x in view.source_mm]} mm, "
              f"detector centre {[round(float(x), 1) for x in view.det_center_mm]} mm, "
              f"{view.det_size[0]}x{view.det_size[1]} px @ "
              f"{view.det_spacing[0]:.4f} mm")
        drr = view.render(n_samples=n_samples)
        u_col, u_row, ok = view.project(p)
        uc, ur = float(u_col[0]), float(u_row[0])
        check(f"{view.name}: nodule projects inside the detector", bool(ok[0]),
              f"pixel ({uc:.1f}, {ur:.1f})")

        sw = view.w(p)
        check(f"{view.name}: w returns patches for it",
              len(sw) > 0 and float(sw.total.sum()) > 0.9,
              f"{len(sw)} patches, total {float(sw.total.sum()):.3f}")
        if len(sw):
            i, _ = sw.top(1)[0]
            x0, y0, x1, y1 = view.grid.patch_bbox_source(i)
            check(f"{view.name}: top patch contains the projected pixel",
                  x0 <= uc < x1 and y0 <= ur < y1)
        figs[o] = (view, drr, uc, ur)

    # The definitive test, made numeric.  A realistic +60 HU nodule contributes
    # ~0.2% of a full-body path integral, so its shadow is invisible by eye on
    # the DRR itself -- exactly as on a real radiograph.  Render the same view
    # with the nodule removed and subtract: that isolates the shadow, and the
    # question "is the circle on it?" becomes "is the argmax of the difference
    # the pixel we projected to?".  Both the number and the picture below.
    diffs = {}
    if vol.meta.get("synthetic"):
        bare = synthetic_thorax(size_ijk=vol.size_ijk, spacing=vol.spacing,
                                nodule_mm=vol.meta["nodule_mm"], with_nodule=False)
        for o, (view, drr, uc, ur) in figs.items():
            bare_view = DRRView.standard(bare, o, det_size=view.det_size,
                                         det_spacing=view.det_spacing)
            diff = drr - bare_view.render(n_samples=n_samples)
            diffs[o] = diff
            r, c = np.unravel_index(int(np.argmax(diff)), diff.shape)
            err = math.hypot(c + 0.5 - uc, r + 0.5 - ur)
            tol = max(4.0, 3.0 / view.det_spacing[0])
            check(f"{view.name}: the circle sits on the nodule's shadow",
                  err < tol,
                  f"shadow peak at ({c + 0.5:.1f}, {r + 0.5:.1f}), projected "
                  f"({uc:.1f}, {ur:.1f}), off by {err:.1f}px "
                  f"= {err * view.det_spacing[0]:.2f}mm (tol {tol:.0f}px)")

    ncol = 2 + len(diffs)
    fig, axes = plt.subplots(1, ncol, figsize=(5.3 * ncol, 5.8))
    axes = np.atleast_1d(axes)
    for ax, (o, (view, drr, uc, ur)) in zip(axes[:2], figs.items()):
        lo, hi = np.percentile(drr[drr > 0], (0.5, 99.5)) if np.any(drr > 0) else (0, 1)
        ax.imshow(drr, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        _circle(ax, uc, ur)
        ax.set_title(f"{view.name}  --  circle must sit on the shadow", fontsize=9)
    for ax, (o, diff) in zip(axes[2:], diffs.items()):
        view, _drr, uc, ur = figs[o]
        ax.imshow(diff, cmap="magma", interpolation="nearest")
        _circle(ax, uc, ur, color="cyan")
        ax.set_title(f"{view.name}: nodule shadow, isolated\n"
                     f"(with nodule) - (without)", fontsize=9)
    fig.suptitle("0.4  the dot landing: the gate for the whole project", fontsize=11)
    save(fig, "0_4_drr_overlay.png")
    return views


def _circle(ax, uc, ur, color="r", r=22):
    ax.add_patch(plt.Circle((uc - 0.5, ur - 0.5), r, ec=color, fc="none", lw=1.6))
    ax.plot(uc - 0.5, ur - 0.5, "+", color=color, ms=10, mew=1.2)
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------
# 0.5  the sanity suite
# --------------------------------------------------------------------------


def step_0_5(vol: CTVolume, p: torch.Tensor, views):
    head("0.5  the sanity suite")

    pa = views["PA"]
    spread, uv = ray_consistency(pa, p, fracs=(0.4, 0.7, 1.0, 1.3))
    check("ray consistency: same ray, different depths -> same pixel",
          spread < 1e-6,
          f"max disagreement {spread:.2e} px over 4 depths "
          f"(the null space of this view, measured)")

    ucp, urp, _ = pa.project(p)
    ucl, url, _ = views["LAT"].project(p)
    hat = triangulate([pa, views["LAT"]],
                      [(float(ucp[0]), float(urp[0])), (float(ucl[0]), float(url[0]))])
    err = float(torch.linalg.norm(hat - p))
    check("two-view intersection recovers the original point",
          err < 1e-6, f"error {err:.2e} mm at {[round(float(x), 3) for x in hat]}")

    step = float(vol.spacing[2])
    k = float(vol.mm_to_voxel(p)[2])
    nz = vol.size_ijk[2]

    def expected(t: float) -> int:
        """Slices whose rect contains the point: |i - k| * step < t/2."""
        h = t / 2.0 / step
        return sum(1 for i in range(nz) if abs(i - k) < h)

    hits = slices_seeing(vol, p, axis=2, thickness_mm=step)
    check("slice count: exactly one contiguous slice sees the point",
          len(hits) in (1, 2), f"{len(hits)} slice(s): {hits[:6]}")

    # An exact prediction, not a range.  If thickness were read as voxels the
    # counts would be off by a factor of the slice spacing and this explodes.
    ok, detail = True, []
    for mult in (1.0, 2.0, 3.0, 4.5):
        t = mult * step
        got, want = len(slices_seeing(vol, p, axis=2, thickness_mm=t)), expected(t)
        ok &= (got == want)
        detail.append(f"{mult:g}x:{got}(want {want})")
    check("slice count matches the rect prediction exactly, in mm",
          ok, ", ".join(detail))

    wrong = slices_seeing(vol, p, axis=2, thickness_mm=step * 10)
    print(f"       (units check: thickness in voxels instead of mm would give "
          f"~{len(wrong)} slices -- that is the mm/voxel bug)")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ts = np.linspace(0.5, 6.0, 24) * step
    ax.plot(ts, [len(slices_seeing(vol, p, axis=2, thickness_mm=float(t))) for t in ts],
            "o-", ms=3)
    ax.set_xlabel("slab thickness (mm)"); ax.set_ylabel("slices with nonzero w")
    ax.axvline(step, color="r", ls="--", lw=1, label=f"one slice = {step:.2f} mm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("0.5  slice count vs thickness (linear, through the origin)", fontsize=10)
    save(fig, "0_5_slice_count.png")


# --------------------------------------------------------------------------
# bonus: the real LIDC radiographs that ship in dicom/
# --------------------------------------------------------------------------


def bonus_real_cxr(dicom_dir: str, xml_path: str):
    head("bonus: the real LIDC CXR + its reader marks")
    try:
        import SimpleITK as sitk

        marks = parse_lidc_cxr_xml(xml_path)
        if not marks:
            print("  no CXR marks in the XML; skipping")
            return
        by_sop = {}
        for m in marks:
            by_sop.setdefault(m["sop_uid"], []).append(m)

        files = sorted(f for f in os.listdir(dicom_dir) if f.lower().endswith(".dcm"))
        panels = []
        for f in files:
            path = os.path.join(dicom_dir, f)
            r = sitk.ImageFileReader(); r.SetFileName(path); r.ReadImageInformation()
            sop = r.GetMetaData("0008|0018").strip() if r.HasMetaDataKey("0008|0018") else ""
            orient = r.GetMetaData("0020|0020").strip() if r.HasMetaDataKey("0020|0020") else "?"
            img = sitk.GetArrayFromImage(sitk.ReadImage(path))[0]
            panels.append((f, orient, img, by_sop.get(sop, [])))

        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.8))
        axes = np.atleast_1d(axes)
        for ax, (f, orient, img, ms) in zip(axes, panels):
            lo, hi = np.percentile(img, (1, 99))
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            for m in ms:
                ax.plot(m["x"], m["y"], "r+", ms=14, mew=1.4)
                ax.add_patch(plt.Circle((m["x"], m["y"]), 60, ec="r", fc="none", lw=1.2))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{f}  PatientOrientation={orient}  ({len(ms)} marks)", fontsize=9)
        fig.suptitle("real LIDC radiographs with the CXR reader marks", fontsize=11)
        save(fig, "bonus_real_cxr.png")
        print(f"  {len(marks)} marks across {len(by_sop)} image(s)")
    except Exception as e:                                    # never fatal
        print(f"  skipped: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------


def pick_nodule_from_xml(vol: CTVolume, xml_path: str):
    """Centroid of the first nodule that has more than a couple of contour points."""
    nods = parse_lidc_ct_xml(xml_path)
    if not nods:
        return None
    nods.sort(key=lambda n: -len(n["points"]))
    pts = nods[0]["points"]
    mm = contour_to_mm(vol, pts)
    kind = "contour" if len(pts) > 1 else "single locus (a <3mm nodule)"
    print(f"  XML nodule {nods[0]['nodule_id']!r}: {len(pts)} point(s), {kind}")
    vol.meta["nodule_sop_uid"] = pts[0]["sop_uid"]
    vol.meta["nodule_n_points"] = len(pts)

    import re
    xml_series = re.search(r"SeriesInstanceUid>\s*([\d.]+)", open(xml_path).read())
    check("the XML belongs to the loaded series (SeriesInstanceUid match)",
          bool(xml_series) and xml_series.group(1) == vol.meta["series_uid"],
          f"...{vol.meta['series_uid'][-20:]}")
    return mm.mean(dim=0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", help="directory holding a CT DICOM series")
    ap.add_argument("--xml", help="LIDC CT reading-session XML")
    ap.add_argument("--search", help="walk this root for CT series and use the first")
    ap.add_argument("--samples", type=int, default=256, help="DRR samples per ray")
    ap.add_argument("--no-bonus", action="store_true")
    args = ap.parse_args(argv)

    ct_dir = args.ct
    if ct_dir is None and args.search:
        found = find_ct_series(args.search)
        print(f"found {len(found)} CT series under {args.search}")
        ct_dir = found[0] if found else None

    if ct_dir:
        vol = load_ct(ct_dir)
        print(f"loaded CT series {vol.meta['series_uid']} "
              f"({vol.meta['n_slices']} slices) from {ct_dir}")
    else:
        vol = synthetic_thorax()
        print("no CT series given -- using the synthetic thorax phantom.\n"
              "  (pass --ct <series dir> to run the same suite on real LIDC data)")

    nodule, is_nodule = None, True
    if args.xml and ct_dir:
        nodule = pick_nodule_from_xml(vol, args.xml)
    if nodule is None:
        if vol.meta.get("synthetic"):
            nodule = torch.tensor(vol.meta["nodule_mm"], dtype=torch.float64)
        else:
            nodule, is_nodule = vol.center_mm, False

    step_0_1(vol)
    p = step_0_2(vol, nodule, is_nodule=is_nodule)
    step_0_3(vol, p)
    views = step_0_4(vol, p, n_samples=args.samples)
    step_0_5(vol, p, views)

    here = os.path.dirname(os.path.abspath(__file__))
    if not args.no_bonus and os.path.isfile(os.path.join(here, "dicom", "068.xml")):
        bonus_real_cxr(os.path.join(here, "dicom"), os.path.join(here, "dicom", "068.xml"))

    head("summary")
    failed = [n for n, ok in _results if not ok]
    print(f"  {len(_results) - len(failed)}/{len(_results)} checks passed")
    for n in failed:
        print(f"  FAILED: {n}")
    print(f"  figures in {FIG}")
    if failed:
        print("\n  It is always a sign. Flip one at a time and re-render.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
