# Step 0 — the geometry kernel: findings

One function, `w(point_mm, view) -> [(patch_index, weight)]`, mapping a point in
patient millimetres to the patches of a view that see it. No network, no
training. Every view type is a different `w` behind the same interface, which is
the architectural claim rather than preparation for it.

Status: **validated on 3 volumes / 2 nodules / 12 (nodule × view) measurements,
all checks passing.** Reproduce with `python step0_experiment.py`.

---

## 1. What is in the repo

| file | what it is |
|---|---|
| `geometry_kernel.py` | the module: `CTVolume`, `PatchGrid`, `SparseWeights`, `SliceView`, `MIPSlabView`, `DRRView`, the DRR renderer, LIDC XML parsers |
| `step0_tests.py` | the 0.1–0.5 checks with figures; `--ct <dir> --xml <file>` for real data |
| `step0_experiment.py` | sweeps every volume × nodule × view, writes `results/step0/` |
| `results/step0/REPORT.md` | the generated results table |

## 2. Conventions, fixed once

Most of the two days the spec warns about are spent re-deriving these. They are
stated in the module docstring and never re-derived anywhere else.

| | convention |
|---|---|
| world | LPS mm, straight from `ImagePositionPatient`. +x patient left, +y posterior, +z superior |
| voxel | SimpleITK index order `(i,j,k)` = (column, row, slice); numpy array is `[z,y,x]` |
| pixel | edge-based continuous: pixel *n* spans `[n, n+1)`, centre `n+0.5`; voxel *v* sits at `u = v + 0.5` |
| patch | `q = u/P - 0.5`, so integer *q* is a patch **centre**; containing patch is `floor(q + 0.5)`; flat index `row * n_cols + col` |
| detector | pixel (0,0) at a **corner**, detector centre is the origin of the (u,v) mm frame |

`PatchGrid` carries both the native image size and the size the vision tower
sees, so `w` stays correct when a 512² slice is resized to 518² (37×37 patches).

The DRR renderer and the DRR projection read the *same* descriptor fields. A
convention error cannot desynchronise them — it can only move both together,
which is what the overlay test catches.

## 3. Results

### The day-1 gate — the marker lands on the nodule

On real data the honest version is not visual. The LIDC XML names the SOP
Instance UID of the slice the radiologist was annotating, so if the affine sends
`imageZposition` anywhere else, we are wrong:

- `true_dicom_ct` / `Nodule 001`: z = −228.585 mm → voxel k = **109.0** (exactly
  integer), whose SOP UID matches the XML **exactly**. Series UID matches too.
- Affine round-trips: origin error **0.0e+00 mm**, `A⁻¹A` error **5.7e−14 voxel**,
  and `A` agrees with SimpleITK's own `TransformIndexToPhysicalPoint` to
  **0.0e+00 mm**.

### The day-2 gate — the circle lands on the shadow

Attenuation integrates linearly along a ray, so

```
DRR(with nodule) − DRR(without) == DRR(nodule-only volume)
```

The nodule-only volume is a small crop with the nodule sphere replaced by the
median HU of its surrounding shell. One 0.5 s render of a ~19³ box replaces two
full renders, **and it does not depend on the nodule being visible to a human**:

| case | view | shadow error |
|---|---|---|
| true_dicom_ct / Nodule 001 | PA | **0.32 px** (0.34 mm) |
| true_dicom_ct / Nodule 001 | LAT | **0.48 px** (0.51 mm) |
| phantom | PA / LAT | 0.53 / 1.33 px |

The phantom numbers agree with an independent implementation (re-render with the
nodule deleted: 0.54 / 1.36 mm), so the method is cross-validated.

### Sanity suite

| check | result |
|---|---|
| ray consistency (4 depths on one ray → one pixel) | **5.7e−14 px** |
| two-view triangulation back to the original point | **1.8e−13 mm** |
| slice count vs the rect prediction, at 1×/2×/3×/4.5× spacing | exact at every thickness |

Ray consistency at 1e−14 is the null space measured rather than assumed: those
points are genuinely indistinguishable from that view, which is the physical
fact the paper rests on.

## 4. Findings that matter beyond Step 0

**A view can be blind to its own nodule, and `w` says so.** For
`true_dicom_ct / Nodule 001`, `mip-axial` returns total weight **0.017** while
every other view returns 1.000. Checked directly: along the 20 mm column through
that nodule, a structure 7.5 mm away in z is **246 HU denser** (168 vs −78 HU).
A MIP slab there displays the diaphragm, not the nodule. This is the soft argmax
working, not a bug — and supervising attention on that patch would be
supervising a lie. It is a measured instance of the phenomenon the paper is
about, on real data, from day 2.

**The MIP weight must be sampling-invariant.** A first version scored a point by
its *share* of the softmax over the segment. That gives 1/m for a uniform
column — a weight that changes when you change the number of depth samples.
Replaced with a soft argmax, `exp((I(p) − I_max)/τ)`: 1.000 for the point that
owns the maximum, 4e−07 for one occluded on the same column, and identical at 9
or 61 samples.

**`SliceThickness` really does lie.** In `true_dicom_ct` the tag says 1.25 mm
while consecutive `ImagePositionPatient` give a true step of **0.625 mm** —
overlapping reconstruction. Trusting the tag would have doubled every z extent
in the volume. Slice spacing is always derived from positions, and the loader
refuses volumes whose steps are inconsistent.

## 5. Traps hit, so they are not hit twice

- **LIDC ships two XML schemas with two namespaces.** CT reads are under
  `http://www.nih.gov`, CXR reads under `http://www.nih.gov/idri`. A parser
  bound to one returns **zero nodules, silently**, for the other. Both parsers
  now match on the local tag name.
- **A folder of DICOMs is not a volume.** `dicom/` holds two DX radiographs, not
  a CT series; an affine built from it is the identity, so every round-trip test
  passes vacuously. `load_ct` now refuses it by name.
- **Degenerate test points hide behind "it passed".** The volume centre of a
  512×512×117 scan lands *exactly* on a patch boundary (two patches tie) and
  *exactly* on a slice centre. Tests must allow the tie and predict the slice
  count exactly, or they fail for reasons that have nothing to do with the code.
- **Near-diagonal annotations cannot detect an x/y swap.** `Nodule 001` sits at
  pixel (347, 349). What rules a swap out is anatomy in the rendered slice: the
  liver is image-left = patient right = −x, the vertebral body is image-bottom =
  posterior = +y.

## 6. What this does not cover

- The spec asks for **five nodules across five patients**; the data on disk
  supports two nodules, one of them synthetic. The sweep is data-bound, not
  code-bound — point `--roots` at a larger LIDC tree and it runs unchanged.
- The one real nodule is a **single-locus <3 mm mark**. A ≥3 mm contoured nodule
  would give the first DRR shadow a human could confirm by eye.
- DRR geometry is our own renderer's. Swapping in an external renderer means
  filling `DRRView`'s fields from *that* renderer, not from imagination.

## 7. Reproduce

```bash
python step0_experiment.py                      # everything on disk
python step0_experiment.py --roots /data/lidc   # a real LIDC tree
python step0_tests.py --ct true_dicom_ct --xml true_dicom_ct/143.xml
```
