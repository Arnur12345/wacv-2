Fair. Here's the build, in order, no code.

---

## Step 0 — The geometry layer (do this first, alone)

Write one function that answers a single question:

> **Given a physical point in mm and a view, which pixels of that view see it, and how strongly?**

That's `w`. Nothing else. No model, no torch training, no VLM.

Four versions, easiest first:
1. **Axial slice** — trivial. Apply the inverse affine, get voxel indices, check z falls within slice thickness.
2. **Reformat** — same thing, axes permuted.
3. **MIP slab** — same as axial but soft-max weighted over the slab.
4. **DRR ray** — the only real one. Project the point through the source-detector geometry to a detector pixel.

**Debug it visually, not numerically.** Put a bright dot at a known mm coordinate, render, confirm it lands where you said. Then put a dot at a nodule's true position and check it lands on the nodule. If that works, your geometry is right and everything downstream is ordinary deep learning.

Budget: 2–3 days. Do not move on until the dot lands.

---

## Step 1 — The DRR renderer

Don't write one. Use an existing differentiable DRR library, or skip rendering entirely and take DRR-RATE's precomputed pairs.

You need **the projection geometry as numbers**, not just the images. If a library gives you pictures without the source position and detector spacing, it's useless to you.

---

## Step 2 — Phase 0 gate

Before building anything: LoRA-tune Qwen3-VL to point at nodules in DRR images. 2D only. "Where is the nodule?" → pixel coordinates.

If it can't do this, stop. The 3D version cannot work if the 2D version doesn't.

---

## Step 3 — Slots

A learnable embedding table of M vectors, plus a fixed array of M mm-coordinates on a coarse grid over the volume. That's the whole thing — an embedding table and a coordinate array.

---

## Step 4 — The gather

One cross-attention layer. Queries = slots. Keys/values = frozen vision-tower patches from all views, concatenated.

The only unusual part: **add log(w) to the attention logits** before softmax, where `w` comes from Step 0.

That's it. That's the mechanism. Everything else in this project is scaffolding around one added term in one attention layer.

Practical warnings:
- `w` is mostly zeros → use `-inf` masking on far-below-threshold entries, not literal `log(0)`
- Precompute `w` per (volume, view-set) and cache it. It doesn't depend on weights, so it never needs recomputing during training.
- Softening temperature on `w` — start soft, anneal sharp. Sharp from step one gives you dead gradients.

---

## Step 5 — The null-space tensor

Per slot, sum outer products of ray directions weighted by `w`. Eigendecompose the 3×3. Concatenate the three eigenvalues and three eigenvectors (12 numbers) to the slot embedding through a small MLP.

Pure numpy-level math on tiny matrices. An afternoon.

---

## Step 6 — Wiring in

Slots → linear projection → prepend to Qwen3-VL's token sequence in place of its normal image tokens. LoRA the LM. Freeze the vision tower.

**Zero-init the projection** so the model starts unchanged and can't diverge on step one.

---

## Step 7 — Train

Task: predict nodule position in mm. Loss: coordinate regression, or text output parsed to numbers.

Vary the view set across training samples — sometimes one DRR, sometimes two, sometimes CT slices. This is what teaches the model to *use* the conditioning information rather than memorize a fixed input format.

---

## Step 8 — The two experiments that are the paper

**E1:** localize under 1 view vs. 2 vs. full CT. Compute the error covariance. Check its long axis aligns with the ray.

**E2:** train on 0°/90°, test at 45°. Baselines can't. You should be able to.

---

## The honest order of difficulty

| | |
|---|---|
| Hard | Step 0 (geometry), Step 2 (does it work at all) |
| Medium | Step 1 (DRR tooling), Step 7 (training loop) |
| Easy | Steps 3–6 — genuinely a few hundred lines total |

Almost all your risk is in the first three days. Once the dot lands where it should, this is a normal LoRA project with one extra term in one attention layer.

Start with the dot.