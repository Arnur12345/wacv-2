# LIDC-IDRI CT dataset: standalone technical reference

This document describes only the source CT dataset installed at
`/data/lidc/raw`. It is intended as a portable reference for using LIDC-IDRI in
another project. It does not describe any existing experiment, stimulus set,
model, prompt, or result.

## 1. Dataset identity

The Lung Image Database Consortium and Image Database Resource Initiative
(LIDC-IDRI) collection is a public dataset of diagnostic and lung-cancer
screening thoracic computed-tomography examinations. It was created as a
reference resource for research in pulmonary-nodule detection, segmentation,
characterization, and computer-aided diagnosis.

Seven academic centers and eight medical-imaging companies participated in its
creation. The collection's detailed description reports 1,018 cases. The
current TCIA image-download table reports 1,010 subjects because the historical
collection description, downloadable imaging inventory, and treatment of a
small number of repeated studies are not expressed using exactly the same unit.
For a new project, report both the official collection description and the
counts actually observed in the frozen local manifest instead of silently
assuming either number.

The collection contains:

- thoracic CT images in DICOM format;
- radiologist annotations in XML;
- chest radiographs for a subset of subjects;
- limited diagnosis information for a subset;
- supporting nodule-count and nodule-size tables.

The CT images and XML annotations are the components normally meant by “using
LIDC-IDRI.” The diagnosis spreadsheet is incomplete and should not be treated
as a comprehensive patient-level malignancy ground truth.

Authoritative references:

- [TCIA LIDC-IDRI collection page](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- Dataset DOI: [10.7937/K9/TCIA.2015.LO9QL9SX](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX)
- Armato et al., *The Lung Image Database Consortium (LIDC) and Image Database
  Resource Initiative (IDRI): A completed reference database of lung nodules on
  CT scans*, *Medical Physics* 38(2), 2011,
  [DOI 10.1118/1.3528204](https://doi.org/10.1118/1.3528204)
- [`pylidc` documentation](https://pylidc.github.io/)
- Hancock and Magnan, *Lung nodule malignancy classification using only
  radiologist-quantified image features as inputs to statistical learning
  algorithms*, [DOI 10.1117/1.JMI.3.4.044504](https://doi.org/10.1117/1.JMI.3.4.044504)

## 2. License and required attribution

TCIA lists LIDC-IDRI under the Creative Commons Attribution 3.0 license:

- [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/)
- [TCIA citation and acknowledgement instructions](https://www.cancerimagingarchive.net/collection/lidc-idri/)

Derived images, masks, tables, and trained models should retain clear dataset
provenance. A publication should cite the TCIA dataset DOI and the principal
LIDC-IDRI paper, and should include the acknowledgement requested on the TCIA
collection page.

Although the dataset is publicly available and de-identified, retain its
original identifiers and DICOM metadata only in controlled research storage.
Do not claim that public availability makes every possible redistribution or
clinical use appropriate. Check institutional requirements before publishing
bulk DICOM copies or derived data containing DICOM metadata.

## 3. Local installation

The dataset is installed on the server at:

```text
/data/lidc/raw
```

Patient identifiers follow this pattern:

```text
LIDC-IDRI-0001
LIDC-IDRI-0002
...
```

Depending on the TCIA download tool and manifest version, patient directories
may be directly below the root or nested under an additional `LIDC-IDRI`
directory. Study- and series-specific UID directories usually appear beneath
each patient. A representative hierarchy is:

```text
/data/lidc/raw/
└── LIDC-IDRI/
    └── LIDC-IDRI-0001/
        └── <StudyInstanceUID or download directory>/
            └── <SeriesInstanceUID or download directory>/
                ├── 000001.dcm
                ├── 000002.dcm
                └── ...
```

The exact intermediate directory names are not a reliable data key. Match a
series using DICOM `PatientID`, `StudyInstanceUID`, and `SeriesInstanceUID`, not
by assuming a fixed path depth or lexical directory order.

`pylidc` includes a local relational database created from the annotation XML,
but it does not include the DICOM pixel files. Both components are required to
load image volumes and align annotations with CT slices.

## 4. CT image representation

### 4.1 DICOM series

A CT scan is stored as a DICOM series containing one file per reconstructed
axial image in most cases. Important identifiers and geometric tags include:

| Concept | DICOM attribute | Purpose |
|---|---|---|
| Patient identifier | `PatientID` | Links images to an LIDC subject |
| Study identifier | `StudyInstanceUID` | Identifies an imaging study |
| Series identifier | `SeriesInstanceUID` | Identifies the CT series |
| Slice position | `ImagePositionPatient` | Physical location of each image |
| Image orientation | `ImageOrientationPatient` | Direction of rows and columns |
| In-plane spacing | `PixelSpacing` | Millimetres between neighboring pixels |
| Slice thickness | `SliceThickness` | Nominal reconstructed section thickness |
| Pixel rescale | `RescaleSlope`, `RescaleIntercept` | Converts stored pixels to CT values |
| Reconstruction kernel | `ConvolutionKernel` | Reconstruction/sharpness information when present |
| Tube settings | `KVP`, `XRayTubeCurrent` | Acquisition information when present |

DICOM metadata completeness varies. Never assume that every scanner model or
series provides every optional field in the same representation.

### 4.2 Hounsfield units

Raw DICOM `pixel_array` values are stored integers and are not necessarily
Hounsfield units. Convert them using:

```text
HU = stored_pixel × RescaleSlope + RescaleIntercept
```

For most CT series the slope is 1 and the intercept is near −1024, but code must
read the actual tags. `pylidc.Scan.to_volume()` loads the series in slice order
and applies the rescale values. If using `pydicom` directly, perform and test
this conversion explicitly.

Windowing should be considered visualization, not a change to the underlying
CT measurement. Preserve the original HU volume or the DICOM source whenever a
future task may need a different window.

### 4.3 Pixel spacing, slice spacing, and slice thickness

These quantities are distinct:

- **Pixel spacing** is the physical distance between adjacent pixel centers in
  the axial plane. `pylidc.Scan.pixel_spacing` uses one value because LIDC
  transverse pixels have equal row and column resolution.
- **Slice spacing** is the distance between reconstructed slice positions. In
  `pylidc`, it is calculated as the median difference between sorted z
  coordinates.
- **Slice thickness** is the DICOM reconstruction thickness and can differ from
  slice spacing because slices can overlap or have gaps.

Do not substitute slice thickness for slice spacing when measuring z-axis
distance. Do not assume uniform z positions without checking
`scan.slice_zvals`. The [`pylidc Scan` documentation](https://pylidc.github.io/scan.html)
explicitly notes that slice spacing and thickness are not always equal and that
one median spacing does not prove perfect uniformity.

### 4.4 Scanner and protocol heterogeneity

LIDC-IDRI is a multi-center dataset, so scans vary in acquisition and
reconstruction. Relevant sources of heterogeneity include:

- scanner manufacturer and model;
- in-plane resolution;
- slice thickness and inter-slice spacing;
- reconstruction kernel;
- radiation exposure parameters;
- contrast usage;
- patient positioning and inspiratory level;
- image matrix, field of view, and number of slices;
- nodule type, conspicuity, location, and surrounding anatomy.

This diversity is useful for generalization research but can become confounding
if acquisition properties correlate with labels or train/test membership. Save
the original acquisition metadata and summarize its distribution for every
project-specific split.

## 5. Radiologist annotation protocol

Four experienced thoracic radiologists reviewed each CT examination using a
two-phase process.

### 5.1 Blinded phase

Each reader independently reviewed the scan without seeing the marks made by
the other readers. Findings were placed into three broad categories:

1. nodules with greatest in-plane dimension at least 3 mm;
2. nodules smaller than 3 mm;
3. non-nodule findings at least 3 mm.

For nodules at least 3 mm, readers supplied slice-by-slice outlines and
subjective characteristics. Smaller nodules and non-nodules were generally
represented by point locations rather than full volumetric contours.

### 5.2 Unblinded phase

Each radiologist reviewed their own marks together with anonymized marks from
the other three readers. A reader could retain, modify, add, or remove a mark.
The final XML records these unblinded results.

The protocol deliberately did not force consensus. Therefore disagreement is a
feature of the reference data, not necessarily an annotation error. One
physical nodule may have one to four reader annotations, and readers may differ
in boundary, slice extent, diameter, and subjective characteristics.

### 5.3 What an annotation means

In `pylidc`, a `Scan` owns many `Annotation` objects. An `Annotation` represents
one reader's final outline and characterization of one nodule at least 3 mm. It
contains one or more `Contour` objects, normally one per axial slice on which
the reader outlined the lesion.

Separate annotations do not automatically mean separate physical nodules. The
four readers' marks must be clustered spatially to determine which annotations
refer to the same lesion.

## 6. Subjective nodule characteristics

For outlined nodules, LIDC radiologists recorded ordinal characteristics such
as:

- subtlety;
- internal structure;
- calcification pattern;
- sphericity;
- margin;
- lobulation;
- spiculation;
- texture or solidity;
- subjective malignancy likelihood.

Most characteristics use ordinal scales, commonly 1–5, but not every attribute
uses the same value set. The integer values should be interpreted using the
official mapping in [`pylidc.Annotation`](https://pylidc.github.io/annotation.html),
not as continuous measurements.

The malignancy score is a radiologist's subjective visual assessment under the
protocol's assumed patient scenario. It is not a pathology-confirmed outcome.
Do not label every score of 4 or 5 “cancer” or every score of 1 or 2 “benign.”
If a project requires confirmed diagnosis, it must use the limited diagnosis
table carefully or link to another appropriately governed source.

TCIA reports that approximately 100 cases from the initial release used
inconsistent rating conventions for spiculation and lobulation across sites.
This issue affects those semantic features but not the existence of the CT
series or all contour geometry. A project using these ratings should document a
cleaning or sensitivity-analysis policy.

Reader positions inside XML files are not persistent reader identities across
patients. For example, “reader 1” in one scan is not guaranteed to be the same
radiologist as “reader 1” in another. Reader-specific calibration or random
effects cannot be estimated across cases from these indices.

## 7. Geometric measurements available through `pylidc`

The [`Annotation` API](https://pylidc.github.io/annotation.html) exposes several
derived measurements.

### 7.1 Greatest axial diameter

`ann.diameter` estimates the largest axial-plane diameter from a reader's
contours and reports millimetres. It accounts for in-plane pixel spacing. The
documentation warns that the estimate does not fully handle cases where the
diameter line passes outside an irregular nodule boundary or through an
internal cavity.

This is a per-reader measurement. If four annotations belong to one lesion,
retain all four values. Mean, median, majority threshold, adjudicated choice,
and consensus-mask diameter are different ground-truth policies and must be
named explicitly.

### 7.2 Bounding box and centroid

- `ann.bbox()` returns source-volume slices enclosing the reader's contour.
- `ann.bbox_dims()` reports the physical dimensions of that box in mm.
- `ann.centroid` returns the center of mass in image-index coordinates.

The centroid is not guaranteed to fall within a highly irregular or cavitary
lesion. A bounding box is not a segmentation and includes surrounding tissue.

### 7.3 Binary mask

`ann.boolean_mask()` rasterizes one reader's contours inside the annotation
bounding box. The corresponding bounding box is required to place this local
mask into the full CT volume correctly.

Contour coordinates, NumPy arrays, and displayed images can use different row,
column, x, and y conventions. Establish the convention once and verify it by
overlaying contours on several axial images before processing the full dataset.

### 7.4 Surface area and volume

`ann.surface_area` and `ann.volume` are contour-derived physical estimates.
Volume integrates outlined areas across slice positions. These values depend on
reader boundaries, slice sampling, and implementation details. They should not
be presented as pathology measurements.

### 7.5 Consensus masks

`pylidc.utils.consensus()` combines a cluster of reader annotations at a chosen
agreement level. For four readers, a 50% consensus includes voxels supported by
at least half of the masks. Changing the consensus level changes the mask and
must be recorded as part of the dataset definition.

A consensus mask is a derived research target, not an original fifth-reader
annotation. Preserve the contributing annotation IDs and consensus parameters.

## 8. Working with `pylidc`

### 8.1 Installation

Use an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pylidc pydicom numpy scipy matplotlib
```

`pylidc 0.2.3` contains code written for older NumPy versions and may fail on
modern NumPy because aliases such as `np.int` were removed. A minimal
process-local compatibility shim, executed before importing code paths that use
the aliases, is:

```python
import numpy as np

for name, scalar in (("int", int), ("float", float), ("bool", bool)):
    if name not in np.__dict__:
        setattr(np, name, scalar)

import pylidc as pl
```

This preserves the old aliases only in the current process. Alternatively,
maintain a small tested fork of `pylidc` that replaces deprecated aliases with
explicit Python or NumPy types. Pin the exact environment in any released
project.

### 8.2 Pointing `pylidc` to the DICOM files

On Linux, `pylidc` normally reads its DICOM root from `~/.pylidcrc`. A typical
configuration is:

```ini
[dicom]
path = /data/lidc/raw/LIDC-IDRI
warn = True
```

Use the directory that directly contains the `LIDC-IDRI-dddd` patient folders.
If they are directly under `/data/lidc/raw`, use that path instead. Recent TCIA
downloads may contain varying subdirectory structures below each patient;
`pylidc` recursively searches for the series matching the stored DICOM UIDs.

### 8.3 Basic inventory

```python
import pylidc as pl

print("scans:", pl.query(pl.Scan).count())
print("reader annotations:", pl.query(pl.Annotation).count())

scan = pl.query(pl.Scan).order_by(pl.Scan.patient_id).first()
print(scan.patient_id)
print(scan.study_instance_uid)
print(scan.series_instance_uid)
print(scan.pixel_spacing)
print(scan.slice_thickness)
print(scan.slice_spacing)
print(len(scan.annotations))
```

The annotation-database scan count does not prove that every corresponding
DICOM series is present locally. Verify pixel availability by loading every
series or by resolving every scan path.

### 8.4 Loading a CT volume

```python
scan = pl.query(pl.Scan).filter(
    pl.Scan.patient_id == "LIDC-IDRI-0001"
).first()

volume_hu = scan.to_volume(verbose=False)
print(volume_hu.shape)
print(volume_hu.dtype, volume_hu.min(), volume_hu.max())
print(scan.slice_zvals)
```

The `pylidc` volume convention is `(row, column, slice)`, not the frequently
used machine-learning convention `(slice, row, column)`. Record any transpose
when exporting to NIfTI, PyTorch, MONAI, or another framework.

### 8.5 Clustering reader annotations

```python
clusters = scan.cluster_annotations(verbose=False)

for cluster_index, annotations in enumerate(clusters):
    print(
        cluster_index,
        len(annotations),
        [float(annotation.diameter) for annotation in annotations],
    )
```

`cluster_annotations()` estimates which marks refer to the same physical
nodule using distances between annotation contours. Its default metric is
`"min"`; its default tolerance begins at the scan pixel spacing and is reduced
if a cluster would contain more than four annotations. Save the metric,
tolerance settings, software version, scan ID, and annotation IDs. Cluster
indices alone are not durable identifiers across different algorithms.

### 8.6 Building a consensus mask

```python
from pylidc.utils import consensus

annotations = clusters[0]
mask, bbox = consensus(
    annotations,
    clevel=0.5,
    pad=None,
    ret_masks=False,
)

local_ct = volume_hu[bbox]
assert local_ct.shape == mask.shape
```

In `pylidc 0.2.3`, `consensus()` does not accept a `verbose` keyword. The
returned mask is local to `bbox`; retain both objects when aligning it with the
full scan.

## 9. Recommended dataset inventory

Before defining any task, create an immutable series-level inventory. One row
per CT series should include at least:

```text
patient_id
scan_database_id
study_instance_uid
series_instance_uid
dicom_directory
n_slices
rows
columns
pixel_spacing_row_mm
pixel_spacing_col_mm
slice_thickness_mm
median_slice_spacing_mm
minimum_slice_spacing_mm
maximum_slice_spacing_mm
orientation
manufacturer
manufacturer_model
convolution_kernel
kvp
tube_current
contrast_used
volume_min_hu
volume_max_hu
dicom_load_status
```

Create a separate lesion table with one row per clustered physical nodule:

```text
lesion_id
patient_id
scan_database_id
cluster_parameters
annotation_ids
n_readers
reader_diameters_mm
reader_volumes_mm3
reader_characteristics
reader_agreement_summary
```

Create a third table with one row per reader annotation. Avoid packing all
reader information into a single lossy average. This three-level design—series,
lesion, annotation—supports both consensus targets and disagreement analysis.

For each generated file, save a SHA-256 hash. Store the environment lock file,
source-code commit, query logic, and timestamp alongside the manifests.

## 10. Quality-control checklist

Run these checks before building a downstream cohort:

1. Every database scan selected for use resolves to exactly one local DICOM
   series.
2. Slices are sorted by physical position, not filename or instance number
   alone.
3. Rows and columns are consistent within a series.
4. Pixel spacing is present, positive, and consistent within a series.
5. Slice positions are monotonic after sorting; gaps and duplicate positions
   are recorded.
6. Rescale slope and intercept are applied before HU-based processing.
7. Image orientation is checked before any left/right or spatial claim.
8. Several reader contours are overlaid on the corresponding axial images.
9. Local masks are placed back into the full-volume coordinates correctly.
10. Cluster membership is inspected for a sample of crowded or adjacent
    nodules.
11. Diameter and volume distributions are reviewed for implausible outliers.
12. Acquisition metadata distributions are compared across planned data
    splits.
13. Missing semantic ratings and limited diagnoses are handled explicitly.
14. No patient appears in more than one machine-learning split.

A visual audit should include small and large nodules, central and peripheral
lesions, solid and ground-glass appearances, different slice thicknesses, and
cases with reader disagreement.

## 11. Split construction and leakage prevention

Split by `patient_id`, never by individual image, slice, reader annotation, or
nodule. Multiple slices, annotations, nodules, and sometimes studies can belong
to the same subject. Slice-level random splitting creates severe leakage because
neighboring CT slices are highly correlated.

If a project selects multiple nodules from one scan, all of them must remain in
the same split. If acquisition settings or institution proxies differ strongly
between splits, report that imbalance and consider stratification or grouped
sampling.

Do not tune preprocessing, thresholds, or consensus definitions on the test
set. Freeze lesion clustering and ground-truth aggregation before evaluating a
model. When comparing methods, every method should consume the same saved
patient split and lesion manifest.

## 12. Choosing a ground-truth policy

LIDC-IDRI intentionally provides multiple expert opinions rather than one
adjudicated answer. The correct aggregation depends on the task.

### Detection

Define how many readers must mark a lesion before it is treated as positive.
A one-reader target maximizes sensitivity to subtle findings but includes more
disputed marks. A three- or four-reader target represents stronger agreement
but excludes real difficult cases.

### Segmentation

Options include union, intersection, majority consensus, probabilistic
agreement maps, one randomly selected reader, or evaluation against each reader
separately. Union and intersection have different boundary biases. State the
consensus level and retain the individual masks.

### Diameter or threshold classification

Mean or median reader diameter gives a simple scalar, but reader disagreement
is especially consequential near a threshold. Report reader spread and consider
an indeterminate zone for lesions whose individual measurements cross the
decision boundary.

### Malignancy modeling

Radiologist malignancy scores are subjective ordinal assessments. They can be
used as perception labels if described accurately. They should not be called
histopathology labels. Pathology-linked modeling requires a carefully audited
subset of the limited diagnosis data or a different dataset.

## 13. Important limitations

- The dataset is retrospective and was assembled for research, not as a
  prospective screening trial.
- Acquisition protocols and scanners are heterogeneous.
- The collection is not a prevalence-representative clinical population.
- Many findings lack pathology or longitudinal outcome.
- Reader annotations are not forced consensus.
- Smaller nodules and non-nodules do not have the same full-contour annotation
  structure as nodules at least 3 mm.
- Semantic characteristics are ordinal and subjective.
- Reader identities cannot be tracked consistently across cases.
- Some early cases have known inconsistencies in spiculation and lobulation
  rating conventions.
- Contour-derived diameter, area, and volume depend on image resolution and
  reader boundary choices.
- `pylidc` is convenient but old; its version and compatibility patches must be
  frozen for reproducibility.
- A `pylidc` database row does not guarantee the corresponding DICOM files are
  installed or readable.

These limitations do not make the collection unsuitable. They define what its
labels mean and which claims require additional evidence.

## 14. Moving the dataset context to another project

For a project that will read the existing source data on the same server,
preserve this information:

```text
dataset: LIDC-IDRI
TCIA DOI: 10.7937/K9/TCIA.2015.LO9QL9SX
local DICOM root: /data/lidc/raw
annotation interface: pylidc
coordinate convention: volume[row, column, slice]
physical units: millimetres and Hounsfield units
```

Copy the new project's source code, manifests, environment lock, and derived
artifacts. The 133 GB-scale original image collection can remain in
`/data/lidc/raw` and be referenced read-only by multiple projects. Avoid making
untracked duplicate DICOM trees.

If the raw data must be copied to another server, transfer the complete patient
directories and verify series-level file counts and hashes. A partial copy may
still allow annotation queries while failing only when a particular scan is
loaded, so perform an explicit full inventory after transfer.

The minimum reproducibility package for any new LIDC-IDRI project should
contain:

- source collection DOI and license;
- exact local/source snapshot or download-manifest identifier;
- immutable patient/series/lesion manifests;
- patient-level split files;
- annotation-clustering and consensus parameters;
- DICOM-to-HU and coordinate-handling code;
- software versions and compatibility patches;
- audit images with contours or masks overlaid;
- hashes for all derived files;
- a clear statement of what “ground truth” means in that project.

