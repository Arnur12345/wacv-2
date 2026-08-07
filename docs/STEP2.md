# A measured detectability floor for LIDC nodules in DRR projections

A pulmonary nodule changes a chest DRR by about **one grey level in 255**, against
roughly **17 grey levels** of local anatomical structure. That ratio — not model
capacity, not training budget — is why a vision-language model cannot be taught to
point at one from a single projection.

This is a measurement, and it decides the target the rest of the project uses.

---

## 1. What was tested

LoRA fine-tune of Qwen3-VL-4B on 1,006 DRRs (713 patients, PA + LAT, ≥3-reader
consensus nodules, labels self-verified to 0.75 px). Task: emit the nodule's
pixel coordinates. Baseline: always predict the training-set mean coordinate.

| | median error | vs constant baseline |
|---|---|---|
| constant (predicts the training mean) | 97.2 mm | 1.00x |
| Qwen3-VL-4B zero-shot | 103.9 mm | 0.94x |
| Qwen3-VL-4B + LoRA | 93.6 mm | **1.04x** |
| + LoRA, largest nodules only (r ≥ 7 mm) | 99.3 mm | **0.94x** |

Nothing was learned. On the largest nodules the tuned model is *worse* than a
constant, and its predictions collapse into one region of the image — the
signature of a target it cannot see.

**The pipeline is not at fault.** The same code memorises 16 images to
**0.24 mm** (100% within 10 mm), which exercises label alignment, answer-token
masking, gradient flow, generation and parsing end to end. Whatever is missing
is missing from the images.

---

## 2. Measuring the signal exactly

Attenuation integrates linearly along a ray, so a nodule's contribution to a DRR
is exactly the render of a **nodule-only volume**: a small crop with the nodule
sphere replaced by the median HU of its surrounding shell. One cheap render of a
~19³ box gives the nodule's shadow with no reference to what surrounds it.

    signal = peak of DRR(nodule only), in the same 8-bit window as the saved PNG
    clutter = local detrended std of the actual PNG
    CNR = signal / clutter

Measured over 200 nodules:

| | |
|---|---|
| nodule signal | **1.15** grey levels of 255 (median); 3.6 at r 3–5 mm, 7.1 at r 7–10 mm |
| local clutter | **16.8** grey levels |
| true CNR | **0.08** median; 100% below 1.0, 94% below 0.5 |
| below 1 grey level (quantised away) | 45.5% |
| `corr(radius, signal)` | **+0.876** |

Controls with the same instrument, same 5 mm sphere: **lung reads 0.00** (the
null control is valid), a **bone sphere reads 2.3–3.5** — *less* than a nodule,
because rib-vs-soft-tissue is ~410 HU while nodule-vs-lung is ~860 HU. Ribs are
visible in radiographs because they are long high-contrast *edges*, not because a
ball of bone is bright. The renderer is evidenced sound by the clutter itself:
16.8 grey levels of local structure is rich anatomy, not a blurred image.

---

## 3. Two instruments that were wrong first

Worth recording, because both failure modes look like results.

**Windowing is not the explanation.** CNR is a ratio of contrast to local
variation, and any linear re-window scales both equally — it is invariant. The
visible quantisation banding in the crops is real but is not what hides nodules.

**An annulus-based CNR is confounded here.** Comparing a disk against a ring at
2–4 radii reaches 20–40 mm outward for a large nodule, into chest wall and
diaphragm. It reported *nodules getting darker as they get bigger*
(`corr(radius, signal) = −0.273`), which is physically impossible — more tissue
attenuates more. Fitting and removing a local plane halved the clutter but left
the correlation untouched, because a chest wall is a step, not a slope. The sign
of that correlation is the cheapest available check that a contrast instrument is
trustworthy: it must be positive.

---

## 4. What this does and does not show

**It is a range-space result, not a null-space one.** The projection's null space
is depth along the ray. In-plane position is in the *range* space — it is exactly
what a single view preserves. The failure here is at in-plane localisation, so it
is a **detectability** failure: the nodule cannot be found at all, in any
direction. It is not evidence that one view is geometrically insufficient.

This distinction matters because a detectability floor **blocks both headline
experiments**. E1 measures the *shape* of the localisation error covariance and
asks whether its long axis aligns with the ray. If error is dominated by "could
not find the target", the covariance is isotropic in every condition and adding a
second view changes nothing measurable. E2 fails for the same reason.

**It is also not a claim of impossibility.** The 16.8 grey levels of clutter are
*structured* — ribs, vessels — not noise. Humans and CNNs exploit structure, which
is how radiologists detect nodules at this contrast at all (with 60–80%
sensitivity). The honest statement is: **not recoverable from local contrast, at
this detector resolution, at this data scale.**

---

## 5. Consequence: the target changes, the thesis does not

The thesis is that attention gathered through the known forward operator yields
geometry-correct 3D localisation and generalises to unseen view geometries.
Nothing in that says *nodule*; it says *target*. Nodules are the hardest
structure in the chest — which is why CT screening replaced chest radiography.

Retargeting to **anatomical landmarks** (carina, aortic arch, cardiac apex,
diaphragm domes, vertebral body centroids, organ centroids) keeps the claim and
unblocks the experiments: high contrast, exact 3D ground truth, no annotation
cost, on the volumes already processed. The claim also gets broader — "3D
localisation from arbitrary projection geometries" rather than "nodule
localisation".

**A prior-only baseline is mandatory.** Anatomy is stereotyped, so a model that
ignores the image entirely scores well on landmarks. The discriminating result is
not absolute error; it is that **the error covariance changes shape as views are
added, while the prior-only model's does not.**

Nodules return as a stress test: graceful degradation as CNR falls, with the
distribution in §2 as the explanation. That separates detection limits from
geometry limits, which is a cleaner story than either alone.

---

## 6. Reproduce

```bash
python qwen_pointer.py train --data /data/lidc/drr --out runs/gate1
python qwen_pointer.py eval  --data /data/lidc/drr --adapter runs/gate1
python qwen_pointer.py eval  --data /data/lidc/drr --adapter runs/gate1 --min-radius-mm 7

python measure_signal.py --data /data/lidc/drr --root /data/lidc --limit 200
python check_visibility.py --data /data/lidc/drr   # the confounded instrument, kept for the record
```

The overfit control — the one that proves the pipeline rather than the data:

```bash
python qwen_pointer.py train --data /data/lidc/drr --out /tmp/overfit --limit 16 --epochs 40 --lr 2e-4
python qwen_pointer.py eval  --data /data/lidc/drr --adapter /tmp/overfit --split train --limit 16 --skip-zeroshot
```
