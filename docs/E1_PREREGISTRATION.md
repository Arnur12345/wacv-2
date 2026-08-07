# E1 — pre-registration

Written before the model exists and before any localisation output has been
seen. The metric is implemented and tested in `e1_metric.py`; its self-tests
pass on synthetic errors with known anisotropy. Nothing below is to be changed
after results are seen. If it is changed, the change and its date go in §7.

Date registered: 2026-08-08.

---

## 1. The claim

Localisation error is shaped by the projection geometry. A single view cannot
resolve depth along its ray, so error should be elongated along that ray. Two
orthogonal views constrain both directions, so it should not be.

## 2. The metric

For each test sample: predicted point `p̂`, ground truth `p`, error `e = p̂ − p`
in millimetres. The reference ray `d` is the unit vector from the **first
view's** source through `p`.

```
along_i = e_i · d_i
perp_i  = e_i − along_i · d_i
R = SD(along) / sqrt( mean(|perp|²) / 2 )
```

The `/2` is not cosmetic: `perp` occupies the 2D plane orthogonal to the ray, so
without per-axis normalisation an isotropic error yields R = 0.69 rather than
1.0, and a null result would present as a positive one.

The reference ray is the first view's ray **in every condition**, so R is
comparable across conditions rather than redefined per condition.

Reported alongside R:

- **95% bootstrap CI** (2000 resamples). Differences within overlapping CIs are
  not results.
- **alignment** = |principal eigenvector of cov(e) · mean ray|. R says how
  elongated the error is; alignment says whether it is elongated in the
  direction the geometry predicts. A high R with low alignment is not support.
- median error, SD along, SD cross, covariance eigenvalues.

## 3. Conditions

| condition | views |
|---|---|
| `1view-pa` | PA only |
| `1view-lat` | LAT only |
| `2view` | PA + LAT |
| `full-ct` | axial slices |

Plus two baselines that must appear in every table:

- **prior-only** — predicts the training-set mean position for the requested
  landmark class, ignoring the image entirely. Anatomy is stereotyped, so this
  scores well on absolute error; it is the control that shows absolute error is
  the wrong headline number.
- **silhouette-only** — image reduced to a binary body/lung outline. Required
  because lung centroids carry no local contrast: they are recovered by
  inferring the silhouette and computing its centre, so a model could score
  well without using the forward operator for anything.

## 4. Registered predictions

| | R |
|---|---|
| `1view-pa`, `1view-lat` | **> 1** |
| `2view` | **→ 1**, and closer to 1 than either single view |
| `full-ct` | **→ 1** |
| prior-only | **flat across conditions** |
| silhouette-only | **flat across conditions** |

Additionally: alignment > 0.8 for single-view conditions. A model that ignores
the image cannot know where the ray points, so its error cannot align with it.

## 5. What falsifies this

- single-view R ≤ 1
- two-view R no closer to 1 than single-view R
- prior-only R varying with condition — that would mean R tracks something
  other than the geometry, and invalidates the whole measurement
- single-view alignment < 0.5 — elongation not ray-driven

Any of these is reported as a negative result, not explained away.

## 6. Targets

Primary, in order of how directly they are visible:

1. `lung_apex_left/right` — boundary extremes
2. `costophrenic_recess_left/right` — sharp high-contrast corners

Secondary:

3. `lung_centroid_left/right` — **no local contrast at the point**; recovered
   from the global silhouette. Reported separately, never pooled with the
   primaries, and always beside the silhouette-only baseline.

R is reported per landmark class as well as pooled. Pooling alone could hide a
class where the effect is absent.

Dropped: `carina` (50% recall after four extraction iterations). Noted for v2:
vertebral centroids via TotalSegmentator — point-like, off-diagonal, a ladder
per patient — deliberately deferred as a new dependency and failure surface.

## 7. Label QC — what it is and what it is not

Landmark labels are **not** self-verifying the way nodule labels were. The
nodule check rendered the nodule alone and required its shadow to land on the
projected pixel, which tested the label. Projecting a landmark into two views
and triangulating back does **not** substitute: it round-trips to 1e-13 mm
whether or not the point is anatomically correct, because it re-tests the
geometry kernel rather than the anatomy. A point in the wrong lung passes.

The QC is population statistics instead. Per landmark class over all patients:
left–right centroid separation, apex height above the recess, recess z relative
to lung extent, distance from the lung midline. Anatomy is stereotyped enough
that a bad extraction lands in a distribution tail; flag the tails, inspect 30
by eye, and quantify the contamination rate.

## 8. Changes after registration

None yet.
