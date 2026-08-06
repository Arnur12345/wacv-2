"""
Step 2 dataset: LIDC-IDRI -> DRR images + nodule pixel coordinates.

    python build_drr_dataset.py --root /data/lidc/raw --out /data/lidc/drr
    python build_drr_dataset.py --root /data/lidc/raw --out /tmp/smoke --limit 3

Produces, per patient:

    images/<patient>_<series8>_PA.png     8-bit, what the VLM sees
    images/<patient>_<series8>_LAT.png
    manifest.jsonl                        one record per image
    skipped.jsonl                         every rejection, with its reason
    summary.json                          counts, QC stats, split sizes

Each manifest record carries the projection geometry as numbers -- source
position, detector centre, detector axes, spacing, size -- not just the image.
Later steps recompute `w` from those fields; nothing has to be re-derived.

Ground truth policy (both configurable):
  * reader marks are clustered by proximity across reading sessions;
  * a cluster becomes a target only if >= MIN_READERS distinct readers marked
    it (default 3 of 4), and the target is the mean of the per-reader centroids
    so that a reader who drew 40 contour points does not outvote one who drew 4.

Every label is self-verified before it is written. Attenuation integrates
linearly along a ray, so DRR(with nodule) - DRR(without) is exactly the render
of a nodule-only volume; its brightest pixel must be the pixel we projected to.
A sample whose shadow lands more than --shadow-tol px away is written to
skipped.jsonl instead of the manifest. A wrong label is worse than no label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# --------------------------------------------------------------------------
# discovery -- pair each CT series with the XML that names its SeriesInstanceUid
# --------------------------------------------------------------------------


@dataclass
class Series:
    ct_dir: str
    series_uid: str
    patient_id: str
    xml: Optional[str] = None
    n_files: int = 0


def scan_tree(root: str) -> Tuple[List[Series], Dict[str, str]]:
    """One walk of the tree: candidate CT dirs, and every XML by series UID."""
    from geometry_kernel import _read_tag

    dcm_dirs, xml_by_uid = [], {}
    for dirpath, _dirs, files in os.walk(root):
        dcm = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm) >= 3:
            dcm_dirs.append((dirpath, sorted(dcm)[0], len(dcm)))
        for f in files:
            if f.lower().endswith(".xml"):
                path = os.path.join(dirpath, f)
                try:
                    txt = open(path, errors="ignore").read(8192)
                except OSError:
                    continue
                m = re.search(r"SeriesInstanceUid>\s*([\d.]+)", txt)
                if m:
                    xml_by_uid.setdefault(m.group(1), path)

    out = []
    for dirpath, first, n in dcm_dirs:
        p = os.path.join(dirpath, first)
        try:
            if _read_tag(p, "0008|0060") != "CT":
                continue
            out.append(Series(ct_dir=dirpath, series_uid=_read_tag(p, "0020|000e"),
                              patient_id=_read_tag(p, "0010|0020") or "UNKNOWN",
                              n_files=n))
        except Exception:
            continue
    for s in out:
        s.xml = xml_by_uid.get(s.series_uid)
    return out, xml_by_uid


# --------------------------------------------------------------------------
# reader consensus
# --------------------------------------------------------------------------


def cluster_nodules(vol, xml: str, cluster_mm: float, min_readers: int) -> List[Dict]:
    """
    Group per-reader marks into nodules, keep the ones enough readers agreed on.

    Single-link agglomeration on centroids: two marks join if they are within
    `cluster_mm`, which is how the same physical nodule marked by four readers
    becomes one target instead of four.
    """
    import torch
    from geometry_kernel import contour_to_mm, parse_lidc_ct_xml

    marks = []
    for n in parse_lidc_ct_xml(xml):
        mm = contour_to_mm(vol, n["points"])
        c = mm.mean(dim=0)
        radius = (max(1.5, float((mm - c).norm(dim=1).max()))
                  if mm.shape[0] > 1 else 1.5)
        marks.append(dict(session=n["session"], nodule_id=n["nodule_id"],
                          center=c, radius_mm=radius, n_points=n["n_points"]))
    if not marks:
        return []

    parent = list(range(len(marks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(marks)):
        for j in range(i + 1, len(marks)):
            d = float(torch.linalg.norm(marks[i]["center"] - marks[j]["center"]))
            if d <= cluster_mm:
                parent[find(i)] = find(j)

    groups: Dict[int, List[Dict]] = {}
    for i, m in enumerate(marks):
        groups.setdefault(find(i), []).append(m)

    out = []
    for gid, g in groups.items():
        readers = sorted({m["session"] for m in g})
        # one centroid per reader first, then average -- so a reader with 40
        # contour points does not outweigh a reader with 4
        per_reader = []
        for r in readers:
            cs = [m["center"] for m in g if m["session"] == r]
            per_reader.append(sum(cs) / len(cs))
        center = sum(per_reader) / len(per_reader)
        spread = (max(float((c - center).norm()) for c in per_reader)
                  if len(per_reader) > 1 else 0.0)
        out.append(dict(
            center_mm=[float(x) for x in center],
            radius_mm=float(np.mean([m["radius_mm"] for m in g])),
            n_readers=len(readers), readers=readers,
            reader_spread_mm=spread,
            nodule_ids=sorted({m["nodule_id"] for m in g}),
            kept=len(readers) >= min_readers))
    out.sort(key=lambda d: (-d["n_readers"], d["center_mm"]))
    return out


# --------------------------------------------------------------------------
# one patient
# --------------------------------------------------------------------------


def window_to_png(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5):
    """Line integral -> 8-bit. Denser is brighter, as on a radiograph."""
    v = img[img > 0]
    lo, hi = (np.percentile(v, (lo_pct, hi_pct)) if v.size else (0.0, 1.0))
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8), float(lo), float(hi)


def process_series(task: Dict) -> Dict:
    """Render one series and label every consensus nodule in it."""
    import torch
    from PIL import Image

    torch.set_num_threads(1)
    from geometry_kernel import (DRRView, load_ct, nodule_difference_volume)

    cfg = task["cfg"]
    rec = dict(patient_id=task["patient_id"], series_uid=task["series_uid"],
               ct_dir=task["ct_dir"], images=[], skipped=[])
    try:
        vol = load_ct(task["ct_dir"])
    except Exception as e:
        rec["skipped"].append(dict(reason="load_failed", detail=f"{type(e).__name__}: {e}"))
        return rec

    rec["size"] = list(vol.size_ijk)
    rec["spacing"] = [float(s) for s in vol.spacing]

    nodules = cluster_nodules(vol, task["xml"], cfg["cluster_mm"], cfg["min_readers"])
    rec["n_nodule_clusters"] = len(nodules)
    kept = [n for n in nodules if n["kept"]]
    for n in nodules:
        if not n["kept"]:
            rec["skipped"].append(dict(reason="too_few_readers",
                                       detail=f"{n['n_readers']} reader(s)",
                                       center_mm=n["center_mm"]))
    if not kept:
        return rec

    centers = torch.tensor([n["center_mm"] for n in kept], dtype=torch.float64)
    short = task["series_uid"].split(".")[-1][:8]

    for orient in cfg["views"]:
        view = DRRView.standard(vol, orient, det_size=tuple(cfg["det_size"]))
        try:
            drr = view.render(n_samples=cfg["samples"])
        except Exception as e:
            rec["skipped"].append(dict(reason="render_failed", view=orient,
                                       detail=f"{type(e).__name__}: {e}"))
            continue

        uc, ur, ok = view.project(centers)
        W, H = view.det_size
        nods, dropped = [], 0
        for i, n in enumerate(kept):
            if not bool(ok[i]):
                rec["skipped"].append(dict(reason="outside_detector", view=orient,
                                           center_mm=n["center_mm"]))
                dropped += 1
                continue
            err = float("nan")
            if cfg["verify"]:
                try:
                    dv, _ = nodule_difference_volume(vol, centers[i], n["radius_mm"])
                    shadow = view.with_volume(dv).render(n_samples=cfg["samples"])
                    r, c = np.unravel_index(int(np.argmax(shadow)), shadow.shape)
                    err = math.hypot(c + 0.5 - float(uc[i]), r + 0.5 - float(ur[i]))
                except Exception as e:
                    rec["skipped"].append(dict(reason="verify_failed", view=orient,
                                               detail=f"{type(e).__name__}: {e}"))
                    dropped += 1
                    continue
                if not (err <= cfg["shadow_tol"]):
                    rec["skipped"].append(dict(reason="shadow_mismatch", view=orient,
                                               detail=f"{err:.2f}px",
                                               center_mm=n["center_mm"]))
                    dropped += 1
                    continue
            x, y = float(uc[i]), float(ur[i])
            nods.append(dict(
                center_mm=n["center_mm"], radius_mm=n["radius_mm"],
                n_readers=n["n_readers"], reader_spread_mm=n["reader_spread_mm"],
                pixel=[round(x, 2), round(y, 2)],
                pixel_norm1000=[round(x / W * 1000, 1), round(y / H * 1000, 1)],
                shadow_err_px=(None if math.isnan(err) else round(err, 3))))

        if not nods:
            continue

        img8, lo, hi = window_to_png(drr)
        name = f"{task['patient_id']}_{short}_{orient}.png"
        path = os.path.join(cfg["out"], "images", name)
        Image.fromarray(img8).save(path, optimize=True)

        rec["images"].append(dict(
            image=os.path.join("images", name), view=orient,
            width=W, height=H, patient_id=task["patient_id"],
            series_uid=task["series_uid"], nodules=nods, n_dropped=dropped,
            window=[lo, hi],
            geometry=dict(
                source_mm=[float(x) for x in view.source_mm],
                det_center_mm=[float(x) for x in view.det_center_mm],
                det_u=[float(x) for x in view.det_u],
                det_v=[float(x) for x in view.det_v],
                det_spacing=[float(s) for s in view.det_spacing],
                det_size=[int(s) for s in view.det_size],
                volume_size=list(vol.size_ijk),
                volume_spacing=[float(s) for s in vol.spacing],
                affine=[[float(x) for x in row] for row in vol.A.tolist()]),
        ))
    return rec


def _run_one(task):
    try:
        return process_series(task)
    except Exception:
        return dict(patient_id=task["patient_id"], series_uid=task["series_uid"],
                    ct_dir=task["ct_dir"], images=[],
                    skipped=[dict(reason="crashed", detail=traceback.format_exc()[-800:])])


# --------------------------------------------------------------------------


def split_of(patient_id: str, fracs=(0.7, 0.15)) -> str:
    """Deterministic patient-level split. The same patient never straddles."""
    h = int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < fracs[0] else ("val" if h < fracs[0] + fracs[1] else "test")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="e.g. /data/lidc/raw")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--views", nargs="+", default=["PA", "LAT"])
    ap.add_argument("--det-size", nargs=2, type=int, default=[504, 504],
                    help="detector pixels; keep a multiple of 14")
    ap.add_argument("--samples", type=int, default=192, help="DRR samples per ray")
    ap.add_argument("--min-readers", type=int, default=3)
    ap.add_argument("--cluster-mm", type=float, default=8.0)
    ap.add_argument("--shadow-tol", type=float, default=4.0,
                    help="max px between projected point and its rendered shadow")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the shadow check (faster, unverified labels)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--limit", type=int, default=0, help="first N series only")
    ap.add_argument("--force", action="store_true", help="ignore existing output")
    ap.add_argument("--dry-run", action="store_true",
                    help="discovery only: what pairs, what does not, and why")
    args = ap.parse_args(argv)

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    cfg = dict(out=args.out, views=args.views, det_size=args.det_size,
               samples=args.samples, min_readers=args.min_readers,
               cluster_mm=args.cluster_mm, shadow_tol=args.shadow_tol,
               verify=not args.no_verify)

    print(f"scanning {args.root} ...", flush=True)
    t0 = time.time()
    series, xmls = scan_tree(args.root)
    annotated = [s for s in series if s.xml]
    print(f"  {len(series)} CT series, {len(xmls)} annotation XMLs, "
          f"{len(annotated)} paired ({time.time() - t0:.0f}s)")

    if args.dry_run or not annotated:
        patients = {s.patient_id for s in series}
        pat_ann = {s.patient_id for s in annotated}
        print(f"\n  patients: {len(patients)} total, {len(pat_ann)} with annotations")
        unpaired = [s for s in series if not s.xml]
        if unpaired:
            print(f"\n  {len(unpaired)} CT series with no matching XML, e.g.:")
            for s in unpaired[:3]:
                print(f"    {s.patient_id}  {s.n_files} slices  {s.ct_dir}")
        orphan = set(xmls) - {s.series_uid for s in series}
        if orphan:
            print(f"\n  {len(orphan)} XML(s) whose series is not on disk")
        if annotated:
            secs = len(annotated) * (len(args.views) * 3.0 + 2.0) / max(1, args.workers)
            print(f"\n  would process {len(annotated)} series on {args.workers} worker(s)"
                  f"  ~{secs / 60:.0f} min")
            for s in annotated[:3]:
                print(f"    {s.patient_id}  {s.n_files} slices  <- {os.path.basename(s.xml)}")
        else:
            print("\n  NOTHING WILL BE PRODUCED: no CT series has a matching "
                  "SeriesInstanceUid in any XML.")
            print("  The TCIA image download does not include the radiologist "
                  "annotations; they are a separate\n  'LIDC-XML-only.zip' on the "
                  "collection page. Unzip it anywhere under --root (or pass its\n"
                  "  parent as --root) and re-run: pairing is by SeriesInstanceUid, "
                  "not by path.")
            return 1
        if args.dry_run:
            return 0

    man_path = os.path.join(args.out, "manifest.jsonl")
    skip_path = os.path.join(args.out, "skipped.jsonl")
    done = set()
    if not args.force and os.path.exists(man_path):
        for line in open(man_path):
            try:
                done.add(json.loads(line)["series_uid"])
            except Exception:
                pass
        print(f"  resuming: {len(done)} series already in the manifest")

    todo = [s for s in annotated if s.series_uid not in done]
    if args.limit:
        todo = todo[:args.limit]
    tasks = [dict(ct_dir=s.ct_dir, series_uid=s.series_uid, patient_id=s.patient_id,
                  xml=s.xml, cfg=cfg) for s in todo]
    print(f"  {len(tasks)} series to process on {args.workers} worker(s)\n", flush=True)

    n_img = n_nod = n_skip = 0
    shadow_errs, spreads = [], []
    t0 = time.time()
    ctx = mp.get_context("spawn")
    mode = "a" if (done and not args.force) else "w"
    with open(man_path, mode) as man, open(skip_path, mode) as skf:
        if args.workers > 1:
            pool = ctx.Pool(args.workers)
            it = pool.imap_unordered(_run_one, tasks, chunksize=1)
        else:
            pool, it = None, map(_run_one, tasks)

        for i, rec in enumerate(it, 1):
            for img in rec["images"]:
                img["split"] = split_of(img["patient_id"])
                man.write(json.dumps(img) + "\n")
                n_img += 1
                n_nod += len(img["nodules"])
                shadow_errs += [n["shadow_err_px"] for n in img["nodules"]
                                if n["shadow_err_px"] is not None]
                spreads += [n["reader_spread_mm"] for n in img["nodules"]]
            for s in rec["skipped"]:
                skf.write(json.dumps(dict(patient_id=rec["patient_id"],
                                          series_uid=rec["series_uid"], **s)) + "\n")
                n_skip += 1
            man.flush(); skf.flush()
            if i % 10 == 0 or i == len(tasks):
                el = time.time() - t0
                eta = el / i * (len(tasks) - i)
                print(f"  [{i}/{len(tasks)}] {n_img} images, {n_nod} nodules, "
                      f"{n_skip} skipped, {el / 60:.1f}m elapsed, {eta / 60:.1f}m left",
                      flush=True)
        if pool:
            pool.close(); pool.join()

    splits: Dict[str, int] = {}
    patients: Dict[str, set] = {}
    for line in open(man_path):
        r = json.loads(line)
        splits[r["split"]] = splits.get(r["split"], 0) + 1
        patients.setdefault(r["split"], set()).add(r["patient_id"])

    summary = dict(
        root=args.root, out=args.out, config=cfg,
        series_found=len(series), series_annotated=len(annotated),
        images=n_img, nodule_labels=n_nod, skipped=n_skip,
        images_per_split=splits,
        patients_per_split={k: len(v) for k, v in patients.items()},
        shadow_err_px=dict(
            n=len(shadow_errs),
            mean=float(np.mean(shadow_errs)) if shadow_errs else None,
            p95=float(np.percentile(shadow_errs, 95)) if shadow_errs else None,
            max=float(np.max(shadow_errs)) if shadow_errs else None),
        reader_spread_mm=dict(
            mean=float(np.mean(spreads)) if spreads else None,
            max=float(np.max(spreads)) if spreads else None),
        minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 66)
    print(f"{n_img} images, {n_nod} nodule labels, {n_skip} skipped, "
          f"{summary['minutes']:.0f} min")
    print(f"splits (images): {splits}")
    print(f"splits (patients): {summary['patients_per_split']}")
    if shadow_errs:
        print(f"label self-check: mean {summary['shadow_err_px']['mean']:.2f} px, "
              f"p95 {summary['shadow_err_px']['p95']:.2f} px, "
              f"max {summary['shadow_err_px']['max']:.2f} px")
    if spreads:
        print(f"inter-reader spread: mean {summary['reader_spread_mm']['mean']:.2f} mm, "
              f"max {summary['reader_spread_mm']['max']:.2f} mm")
    print(f"-> {man_path}")
    if n_skip:
        print(f"-> {skip_path}   (read this; it is where the surprises are)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
