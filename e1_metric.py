"""
The E1 metric, defined before any result exists.

E1 asks whether localisation error is shaped by the projection geometry: with
one view, depth along the ray is unrecoverable, so error should be elongated
along it; with two orthogonal views it should not be.

    R = SD(along-ray error) / SD(cross-ray error, per axis)

Cross-ray error lives in the 2D plane orthogonal to the ray, so its per-axis
scale is sqrt(mean|e_perp|^2 / 2). Without that /2 the ratio is biased by a
factor of sqrt(2) and a null result would read as a positive one.

The reference ray is always the FIRST view's ray, in every condition, so R is
comparable across conditions rather than being redefined per condition.

## Registered predictions

  single view      R > 1      error elongated along the ray it cannot resolve
  two orthogonal   R -> 1     both directions constrained
  full CT          R -> 1     no projection null space at all
  prior-only       R flat     a model ignoring the image cannot know the ray
  silhouette-only  R flat     ditto; it sees an outline, not the geometry

## What falsifies it

  * single-view R <= 1
  * two-view R not closer to 1 than single-view R
  * prior-only R varying with condition (would mean R tracks something else)

Report R with a bootstrap CI. A difference inside overlapping CIs is not a
result. No variant of this metric is to be substituted after seeing outputs.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def decompose(errors: np.ndarray, rays: np.ndarray):
    """
    Split each error into along-ray and cross-ray parts.

    errors [N,3] mm, rays [N,3] (need not be unit). Returns (along [N], perp [N,3]).
    """
    e = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    d = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-12, None)
    along = np.einsum("ij,ij->i", e, d)
    perp = e - along[:, None] * d
    return along, perp


def anisotropy(errors: np.ndarray, rays: np.ndarray,
               n_boot: int = 2000, seed: int = 0) -> Dict:
    """
    R = SD(along) / SD(cross per axis), with a bootstrap CI.

    Also returns `alignment`: |principal eigenvector of cov(e) . mean ray|,
    which is 1 when the error ellipsoid's long axis points along the ray. R
    says how elongated; alignment says whether it is elongated in the direction
    the geometry predicts. Both are needed -- a large R with low alignment
    would mean the elongation is real but not ray-driven.
    """
    along, perp = decompose(errors, rays)
    n = along.shape[0]
    if n < 3:
        return dict(n=n, R=float("nan"))

    def ratio(idx):
        a, p = along[idx], perp[idx]
        sd_a = float(a.std(ddof=1))
        sd_c = float(np.sqrt((p ** 2).sum(axis=1).mean() / 2.0))
        return sd_a / sd_c if sd_c > 1e-12 else float("inf")

    idx0 = np.arange(n)
    R = ratio(idx0)

    rng = np.random.default_rng(seed)
    boots = np.array([ratio(rng.integers(0, n, n)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]

    e = np.asarray(errors, dtype=np.float64).reshape(-1, 3)
    cov = np.cov(e.T)
    w, v = np.linalg.eigh(cov)
    principal = v[:, int(np.argmax(w))]
    d = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    d = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-12, None)
    mean_ray = d.mean(axis=0)
    mean_ray = mean_ray / max(np.linalg.norm(mean_ray), 1e-12)

    return dict(
        n=n, R=R,
        ci95=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if boots.size else (float("nan"), float("nan")),
        sd_along_mm=float(along.std(ddof=1)),
        sd_cross_mm=float(np.sqrt((perp ** 2).sum(axis=1).mean() / 2.0)),
        alignment=float(abs(principal @ mean_ray)),
        eigenvalues_mm2=[float(x) for x in np.sort(w)[::-1]],
        median_error_mm=float(np.median(np.linalg.norm(e, axis=1))))


def ray_directions(sources_mm: np.ndarray, targets_mm: np.ndarray) -> np.ndarray:
    """Unit vectors source -> target: the direction each view cannot resolve."""
    s = np.asarray(sources_mm, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(targets_mm, dtype=np.float64).reshape(-1, 3)
    d = t - s
    return d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-12, None)


def compare(conditions: Dict[str, Dict]) -> str:
    """Format several conditions into the table E1 reports."""
    lines = [f"  {'condition':<18} {'n':>5} {'R':>7} {'95% CI':>16} "
             f"{'along':>7} {'cross':>7} {'align':>6} {'med err':>8}",
             "  " + "-" * 78]
    for name, m in conditions.items():
        if not np.isfinite(m.get("R", np.nan)):
            lines.append(f"  {name:<18} {m.get('n', 0):>5}   (insufficient data)")
            continue
        lines.append(
            f"  {name:<18} {m['n']:>5} {m['R']:>7.3f} "
            f"[{m['ci95'][0]:>6.3f},{m['ci95'][1]:>6.3f}] "
            f"{m['sd_along_mm']:>7.2f} {m['sd_cross_mm']:>7.2f} "
            f"{m['alignment']:>6.3f} {m['median_error_mm']:>8.2f}")
    return "\n".join(lines)


def selftest() -> int:
    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    rng = np.random.default_rng(0)
    n = 4000
    # a fixed ray, as a single fixed view would give
    d = np.tile(np.array([0.0, 1.0, 0.0]), (n, 1))

    iso = rng.normal(0, 5.0, (n, 3))
    m = anisotropy(iso, d, n_boot=400)
    t("isotropic error -> R = 1", abs(m["R"] - 1) < 0.06, f"R={m['R']:.3f}")
    t("  and the CI covers 1", m["ci95"][0] < 1 < m["ci95"][1],
      f"[{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]")

    # elongated 4x along the ray: what one view should produce
    el = iso.copy(); el[:, 1] *= 4.0
    m = anisotropy(el, d, n_boot=400)
    t("4x elongation along the ray -> R = 4", abs(m["R"] - 4) < 0.3, f"R={m['R']:.3f}")
    t("  and alignment = 1", m["alignment"] > 0.98, f"{m['alignment']:.3f}")

    # elongated across the ray: the opposite sign of effect
    cr = iso.copy(); cr[:, 0] *= 4.0
    m = anisotropy(cr, d, n_boot=400)
    t("elongation across the ray -> R < 1", m["R"] < 0.6, f"R={m['R']:.3f}")

    # the /2 matters: without it isotropic error would not read as 1
    along, perp = decompose(iso, d)
    naive = along.std(ddof=1) / np.sqrt((perp ** 2).sum(axis=1).mean())
    t("per-axis normalisation is what makes isotropic read as 1",
      abs(naive - 1 / np.sqrt(2)) < 0.06,
      f"naive {naive:.3f} vs correct {anisotropy(iso, d, n_boot=1)['R']:.3f}")

    # varying rays, elongation applied along each: still detected
    d2 = rng.normal(size=(n, 3)); d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    e2 = rng.normal(0, 5.0, (n, 3))
    a2 = np.einsum("ij,ij->i", e2, d2)
    e2 = e2 + 3.0 * a2[:, None] * d2                 # stretch along each own ray
    m = anisotropy(e2, d2, n_boot=400)
    t("elongation along per-sample rays is detected", m["R"] > 3.0, f"R={m['R']:.3f}")

    # a model that ignores geometry: error independent of ray -> R = 1
    e3 = rng.normal(0, 5.0, (n, 3))
    m = anisotropy(e3, d2, n_boot=400)
    t("prior-only style error (ray-independent) -> R = 1", abs(m["R"] - 1) < 0.06,
      f"R={m['R']:.3f}")

    print("\n" + ("all self-tests passed" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
