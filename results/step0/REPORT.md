# Step 0 -- full experiment

1 volume(s), 1 nodule(s), 6 (nodule x view) measurements, 2s, 192 samples/ray.

**All checks passed.**

## Volumes

| case | size | spacing (mm) | origin err | A vs SimpleITK | nodules |
|---|---|---|---|---|---|
| phantom | 384x384x160 | 0.900, 0.900, 1.250 | 0.0e+00 | nan | 1 |

## Per-nodule, per-view

| case | nodule | HU | view | patches | total w | top=containing | shadow err (px) | shadow err (mm) |
|---|---|---|---|---|---|---|---|---|
| phantom | synthetic | 60 | axial | 9 | 1.000 | yes |  |  |
| phantom | synthetic | 60 | coronal | 9 | 1.000 | yes |  |  |
| phantom | synthetic | 60 | sagittal | 9 | 1.000 | yes |  |  |
| phantom | synthetic | 60 | mip-axial | 9 | 1.000 | yes |  |  |
| phantom | synthetic | 60 | drr-pa | 25 | 1.000 | yes | 0.53 | 0.54 |
| phantom | synthetic | 60 | drr-lat | 25 | 1.000 | yes | 1.33 | 1.36 |

## Sanity suite

| case | nodule | ray consistency (px) | triangulation (mm) | slice counts got/want |
|---|---|---|---|---|
| phantom | synthetic | 2.84e-14 | 1.46e-13 | 1x:1/1 2x:2/2 3x:3/3 4.5x:4/4 |

## The day-2 gate

Shadow localisation error over 2 (nodule x DRR view) pairs: max **1.33 px**, mean 0.93 px.

Measured as the argmax of DRR(with nodule) - DRR(without), which equals the render of a nodule-only volume because attenuation integrates linearly along a ray. This does not depend on the nodule being visible to a human in the DRR.

## Coverage

The spec asks for five nodules across five patients. That is bounded by the data on disk, not by the code: only annotated series can exercise the gates. Point `--roots` at a larger LIDC tree and this sweeps it unchanged.
