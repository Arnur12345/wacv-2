"""
Step 2 -- the Phase 0 gate.

LoRA-tune Qwen3-VL to point at a nodule in a DRR. 2D only: image in, pixel
coordinates out. No `w`, no slots, no 3D. If this does not work, nothing later
can, so the job of this script is to answer one question honestly.

    python qwen_pointer.py selftest                       # no GPU, no model
    python qwen_pointer.py train --data /data/lidc/drr --out runs/gate1
    python qwen_pointer.py eval  --data /data/lidc/drr --adapter runs/gate1

Coordinates are normalised to 0-1000 (Qwen's grounding convention) so nothing
depends on the resolution the processor happens to pick. Metrics are reported
back in millimetres using the detector spacing the manifest carries.

## Reading the result

`eval` always reports three numbers, and the middle one is the point:

  centre    always predict the training-set mean coordinate. Knows nothing.
  zero-shot the base model, no tuning.
  tuned     the LoRA.

A tuned model that does not clearly beat `centre` has not learned to find
nodules -- it has learned where nodules usually are. Chest nodules cluster in
the mid-lung, so a constant prediction scores far better than chance, and that
is exactly how a failed gate disguises itself as a working one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

_ASK = ("Reply with only its centre as JSON: {\"x\": <0-1000>, \"y\": <0-1000>}, "
        "where x and y are the position across and down the image, "
        "normalised to 0-1000.")
_IMG = "This is a chest radiograph (a DRR rendered from CT). "

# The prompt has to be true of the image it is attached to. LIDC images carry
# 1-8 nodules, so "there is exactly one" is a lie for most of them, and a lie
# in the prompt is indistinguishable from a model that cannot see.
PROMPTS = {
    "largest": _IMG + "It contains one or more pulmonary nodules. "
                      "Find the LARGEST one. " + _ASK,
    "single":  _IMG + "There is exactly one pulmonary nodule in it. " + _ASK,
    "any":     _IMG + "It contains one or more pulmonary nodules. "
                      "Find any one of them. " + _ASK,
}
PROMPT = PROMPTS["largest"]

LANDMARK_PHRASE = {
    "lung_apex_left": "the apex (topmost point) of the patient's LEFT lung",
    "lung_apex_right": "the apex (topmost point) of the patient's RIGHT lung",
    "costophrenic_recess_left": "the left costophrenic recess "
                               "(the sharp bottom corner of the left lung)",
    "costophrenic_recess_right": "the right costophrenic recess "
                                "(the sharp bottom corner of the right lung)",
    "lung_centroid_left": "the centre of the patient's LEFT lung",
    "lung_centroid_right": "the centre of the patient's RIGHT lung",
}


def set_landmark(name: Optional[str]):
    """Ask for the landmark by name; a vague prompt is an unanswerable task."""
    global PROMPT
    if not name:
        return
    what = LANDMARK_PHRASE.get(name, name.replace("_", " "))
    PROMPT = (_IMG + f"Find {what}. " + _ASK)


def set_target_mode(mode: str):
    global PROMPT
    PROMPT = PROMPTS[mode]

# tensors with a batch dim to strip / pad; everything else the processor
# returns (pixel_values, image_grid_thw, ...) is already flat across images
TEXT_KEYS = ("input_ids", "attention_mask", "labels", "token_type_ids")


# --------------------------------------------------------------------------
# answer format -- shared by training targets and eval parsing
# --------------------------------------------------------------------------


def format_answer(x: float, y: float) -> str:
    return json.dumps({"x": int(round(x)), "y": int(round(y))})


def parse_answer(text: str) -> Optional[Tuple[float, float]]:
    """
    Pull (x, y) out of whatever the model said.

    Strict JSON first, then any x/y-ish key pair, then the first two numbers.
    Returns None if nothing usable is there -- a parse failure is a real
    failure and gets counted, never silently treated as a bad guess.
    """
    if not text:
        return None
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if "x" in d and "y" in d:
                return float(d["x"]), float(d["y"])
        except Exception:
            pass
    m = re.search(r'["\']?x["\']?\s*[:=]\s*(-?[\d.]+).{0,12}?["\']?y["\']?\s*[:=]\s*(-?[\d.]+)',
                  text, re.S | re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class Sample:
    image: str
    x1000: float
    y1000: float
    width: int
    height: int
    mm_per_px: float
    patient_id: str
    view: str
    radius_mm: float
    n_readers: int
    # every nodule in the same image, for the lenient "did it find *a* nodule"
    # metric -- an image with 8 nodules should not be graded as if it had one
    all_xy1000: Tuple[Tuple[float, float], ...] = ()


def load_manifest(data_dir: str, split: str, multi: str = "largest",
                  views: Optional[List[str]] = None,
                  min_radius_mm: float = 3.0,
                  landmark: Optional[str] = None) -> Tuple[List[Sample], Dict]:
    """
    Read manifest.jsonl into samples.

    `min_radius_mm` drops LIDC's <3mm single-point marks and keeps the nodules
    a radiologist actually outlined. `multi` decides what an image with several
    nodules means:

      largest  target is the biggest one; every image with >=1 nodule is used
      any      same target, but graded against the nearest nodule (lenient)
      single   only images left with exactly one nodule -- clean but small
    """
    path = os.path.join(data_dir, "manifest.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} -- run build_drr_dataset.py first")

    out, stats = [], dict(records=0, wrong_split=0, too_small=0, multi_nodule=0,
                          no_nodule=0, missing_image=0, wrong_view=0,
                          wrong_landmark=0)
    for line in open(path):
        r = json.loads(line)
        stats["records"] += 1
        if split not in ("all", r.get("split")):
            stats["wrong_split"] += 1
            continue
        if views and r["view"] not in views:
            stats["wrong_view"] += 1
            continue
        nods = [n for n in r.get("nodules", [])
                if float(n.get("radius_mm", 0)) >= min_radius_mm]
        if landmark:
            # Landmarks all share a nominal radius, so "largest" cannot pick
            # among them -- it would train on an arbitrary one per image. Select
            # by name and the target becomes well defined again.
            before = len(nods)
            nods = [n for n in nods if n.get("name") == landmark]
            if before and not nods:
                stats["wrong_landmark"] += 1
        if not nods:
            stats["too_small" if r.get("nodules") else "no_nodule"] += 1
            continue
        if len(nods) > 1 and multi == "single":
            stats["multi_nodule"] += 1
            continue
        img = os.path.join(data_dir, r["image"])
        if not os.path.exists(img):
            stats["missing_image"] += 1
            continue
        mm_per_px = float(r.get("geometry", {}).get("det_spacing", [1.0])[0])
        target = max(nods, key=lambda n: float(n.get("radius_mm", 0)))
        x, y = target["pixel_norm1000"]
        out.append(Sample(image=img, x1000=float(x), y1000=float(y),
                          width=r["width"], height=r["height"],
                          mm_per_px=mm_per_px, patient_id=r["patient_id"],
                          view=r["view"], radius_mm=float(target.get("radius_mm", 0)),
                          n_readers=int(target.get("n_readers", 0)),
                          all_xy1000=tuple((float(n["pixel_norm1000"][0]),
                                            float(n["pixel_norm1000"][1]))
                                           for n in nods)))
    stats["kept"] = len(out)
    stats["patients"] = len({s.patient_id for s in out})
    stats["nodules_per_image"] = (round(float(np.mean([len(s.all_xy1000) for s in out])), 2)
                                  if out else 0)
    return out, stats


def to_pixels(x1000: float, y1000: float, w: int, h: int) -> Tuple[float, float]:
    return x1000 / 1000.0 * w, y1000 / 1000.0 * h


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def score(preds: List[Optional[Tuple[float, float]]], samples: List[Sample],
          thresholds_mm=(10.0, 20.0, 30.0)) -> Dict:
    """
    Distance from prediction to truth, in mm at the detector.

    A None prediction counts as a failure, not as a missing value: it is
    included in the hit rates and in `parse_fail`, so nothing improves the
    score by refusing to answer.
    """
    errs, near, fails = [], [], 0
    for p, s in zip(preds, samples):
        if p is None:
            fails += 1
            continue
        px, py = to_pixels(p[0], p[1], s.width, s.height)
        tx, ty = to_pixels(s.x1000, s.y1000, s.width, s.height)
        errs.append(math.hypot(px - tx, py - ty) * s.mm_per_px)
        # lenient: distance to whichever nodule in this image is closest
        cands = s.all_xy1000 or ((s.x1000, s.y1000),)
        near.append(min(math.hypot(*(np.subtract(to_pixels(cx, cy, s.width, s.height),
                                                 (px, py)))) * s.mm_per_px
                        for cx, cy in cands))
    n = len(samples)
    e = np.array(errs) if errs else np.zeros(0)
    a = np.array(near) if near else np.zeros(0)
    out = dict(n=n, parse_fail=fails,
               median_mm=float(np.median(e)) if e.size else None,
               mean_mm=float(e.mean()) if e.size else None,
               p90_mm=float(np.percentile(e, 90)) if e.size else None,
               nearest_median_mm=float(np.median(a)) if a.size else None)
    for t in thresholds_mm:
        out[f"hit@{t:g}mm"] = float((e <= t).sum()) / n if n else 0.0
        out[f"nearhit@{t:g}mm"] = float((a <= t).sum()) / n if n else 0.0
    return out


def centre_baseline(train: List[Sample], test: List[Sample]) -> Dict:
    """Always predict the training-set mean. The bar the LoRA must clear."""
    mx = float(np.mean([s.x1000 for s in train])) if train else 500.0
    my = float(np.mean([s.y1000 for s in train])) if train else 500.0
    return score([(mx, my)] * len(test), test) | {"predicts": [round(mx, 1), round(my, 1)]}


def fmt(name: str, m: Dict) -> str:
    if m.get("median_mm") is None:
        return f"  {name:<10} no parseable predictions ({m['parse_fail']}/{m['n']} failed)"
    return (f"  {name:<10} median {m['median_mm']:6.1f} mm | p90 {m['p90_mm']:6.1f} | "
            f"hit@10mm {m['hit@10mm']:.1%} | hit@20mm {m['hit@20mm']:.1%} "
            f"|| nearest: median {m['nearest_median_mm']:6.1f} "
            f"hit@10mm {m['nearhit@10mm']:.1%}" +
            (f" | {m['parse_fail']} unparseable" if m["parse_fail"] else ""))


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------


def load_model(model_id: str, dtype="bfloat16", adapter: Optional[str] = None,
               train: bool = False, max_pixels: int = 512 * 512):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    td = getattr(torch, dtype)
    # Bound the token count here, not by pre-resizing the PIL image: left to its
    # defaults the processor will happily upscale a 504x504 DRR to ~896x896 and
    # hand the vision tower 4096 patches per image.
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=64 * 28 * 28, max_pixels=max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=td, device_map=None, attn_implementation="sdpa")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    return model, processor


def lora_targets(model, verbose=True) -> List[str]:
    """
    Discover the LM's linear projections by shape rather than by hard-coded
    names, and exclude the vision tower -- it stays frozen, per the design, and
    module naming shifts between model releases.
    """
    import torch.nn as nn

    names = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(k in name for k in ("visual", "vision_tower", "vision_model",
                                   "merger", "patch_embed")):
            continue
        leaf = name.split(".")[-1]
        if leaf in ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"):
            names.add(leaf)
    if verbose:
        print(f"  LoRA target modules: {sorted(names)}")
    if not names:
        raise RuntimeError("found no LM projection layers to adapt -- check the model id")
    return sorted(names)


def build_messages(image_path: str) -> List[Dict]:
    return [{"role": "user", "content": [{"type": "image", "image": image_path},
                                         {"type": "text", "text": PROMPT}]}]


class PointerDataset:
    def __init__(self, samples: List[Sample], processor, max_pixels: int):
        self.samples, self.processor, self.max_pixels = samples, processor, max_pixels

    def __len__(self):
        return len(self.samples)

    def _image(self, path):
        from PIL import Image
        # sizing is the processor's job (min_pixels/max_pixels were set when it
        # was loaded); resizing here as well would fight it
        return Image.open(path).convert("RGB")

    def __getitem__(self, i):
        import torch
        s = self.samples[i]
        msgs = build_messages(s.image)
        prompt = self.processor.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
        answer = format_answer(s.x1000, s.y1000) + (self.processor.tokenizer.eos_token or "")
        img = self._image(s.image)

        full = self.processor(text=[prompt + answer], images=[img], return_tensors="pt")
        only = self.processor(text=[prompt], images=[img], return_tensors="pt")
        n_prompt = only["input_ids"].shape[1]

        # Only the text tensors carry a batch dim to strip. `pixel_values` is
        # already [n_patches, dim] and `image_grid_thw` is [n_images, 3] --
        # indexing [0] there throws the image away and the vision tower then
        # sees one patch where the grid claims thousands.
        item = {k: (v[0] if k in TEXT_KEYS else v) for k, v in full.items()}
        labels = item["input_ids"].clone()
        labels[:n_prompt] = -100                       # loss on the answer only
        item["labels"] = labels
        return item


def collate(batch, pad_id: int):
    """
    Text tensors are padded to a rectangle; everything else -- pixel_values,
    image_grid_thw -- is concatenated along dim 0, because Qwen packs all
    images of a batch into one flat patch sequence and describes them with one
    grid row each. Stacking those instead of concatenating is the same bug in
    a different place.
    """
    import torch

    out = {}
    n = max(b["input_ids"].shape[0] for b in batch)
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if k in TEXT_KEYS:
            pad = {"input_ids": pad_id, "attention_mask": 0,
                   "labels": -100, "token_type_ids": 0}[k]
            out[k] = torch.stack([
                torch.cat([v, torch.full((n - v.shape[0],), pad, dtype=v.dtype)])
                for v in vals])
        else:
            out[k] = torch.cat(vals, dim=0)
    return out


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


def cmd_train(args) -> int:
    set_target_mode(args.multi)
    set_landmark(args.landmark)
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_s, st = load_manifest(args.data, "train", args.multi, args.views, args.min_radius_mm,
                                   args.landmark)
    val_s, sv = load_manifest(args.data, "val", args.multi, args.views, args.min_radius_mm,
                                   args.landmark)
    if args.limit:
        train_s, val_s = train_s[:args.limit], val_s[:max(4, args.limit // 8)]
    print(f"train {len(train_s)} samples / {st['patients']} patients, "
          f"val {len(val_s)} / {sv['patients']}")
    print(f"  target={args.multi!r} min_radius={args.min_radius_mm}mm | "
          f"{st['nodules_per_image']} nodules/image | dropped: "
          f"{st['too_small']} all-nodules-too-small, {st['multi_nodule']} multi-nodule")
    print(f"  prompt: {PROMPT[:96]}...")
    if not train_s:
        print("no training data"); return 1

    model, processor = load_model(args.model, args.dtype, train=True,
                                 max_pixels=args.max_pixels)
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                     bias="none", task_type="CAUSAL_LM",
                     target_modules=lora_targets(model))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    pad_id = processor.tokenizer.pad_token_id or 0
    processor.tokenizer.padding_side = "right"
    dl = DataLoader(PointerDataset(train_s, processor, args.max_pixels),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers,
                    collate_fn=lambda b: collate(b, pad_id))

    steps = max(1, len(dl) // args.grad_accum) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.03, anneal_strategy="cos")
    dev = next(model.parameters()).device
    print(f"{steps} optimiser steps over {args.epochs} epoch(s) on {dev}\n")

    os.makedirs(args.out, exist_ok=True)
    model.train()
    t0, step, run = time.time(), 0, None
    for ep in range(args.epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            step_loss = loss.detach().item() * args.grad_accum
            run = step_loss if run is None else 0.98 * run + 0.02 * step_loss
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    print(f"  ep{ep} step {step}/{steps} loss {run:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {el / 60:.1f}m "
                          f"eta {(steps - step) * el / max(step, 1) / 60:.0f}m", flush=True)
        model.save_pretrained(args.out)
        print(f"  epoch {ep} done -> {args.out}")

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump(vars(args) | {"train_samples": len(train_s),
                                "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(f"\nsaved adapter -> {args.out}")
    print(f"now: python qwen_pointer.py eval --data {args.data} --adapter {args.out}")
    return 0


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


def _generate(model, processor, samples, args) -> List[Optional[Tuple[float, float]]]:
    import torch
    from PIL import Image

    processor.tokenizer.padding_side = "left"
    dev = next(model.parameters()).device
    preds, raw = [], []
    ds = PointerDataset(samples, processor, args.max_pixels)
    for i in range(0, len(samples), args.batch_size):
        chunk = samples[i:i + args.batch_size]
        texts = [processor.apply_chat_template(build_messages(s.image), tokenize=False,
                                               add_generation_prompt=True) for s in chunk]
        imgs = [ds._image(s.image) for s in chunk]
        inp = processor(text=texts, images=imgs, return_tensors="pt",
                        padding=True).to(dev)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=processor.tokenizer.pad_token_id)
        for j in range(len(chunk)):
            txt = processor.tokenizer.decode(gen[j][inp["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
            raw.append(txt)
            preds.append(parse_answer(txt))
        if args.verbose and i == 0:
            print(f"  sample output: {raw[0]!r}")
        print(f"  {min(i + args.batch_size, len(samples))}/{len(samples)}", end="\r",
              flush=True)
    print(" " * 40, end="\r")
    return preds, raw


def cmd_eval(args) -> int:
    set_target_mode(args.multi)
    set_landmark(args.landmark)
    train_s, _ = load_manifest(args.data, "train", args.multi, args.views, args.min_radius_mm,
                                   args.landmark)
    test_s, st = load_manifest(args.data, args.split, args.multi, args.views, args.min_radius_mm,
                                   args.landmark)
    if args.limit:
        test_s = test_s[:args.limit]
    if not test_s:
        print(f"no samples in split {args.split!r}"); return 1
    print(f"eval on {len(test_s)} samples / {st['patients']} patients "
          f"(split={args.split})\n")

    results = {"centre": centre_baseline(train_s, test_s)}
    print(fmt("centre", results["centre"]) +
          f"   [constant {results['centre']['predicts']}]")

    rows = []
    if not args.skip_zeroshot:
        model, processor = load_model(args.model, args.dtype,
                                      max_pixels=args.max_pixels)
        model.eval()
        preds, raw = _generate(model, processor, test_s, args)
        results["zero-shot"] = score(preds, test_s)
        print(fmt("zero-shot", results["zero-shot"]))
        del model
        import torch, gc
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.adapter:
        model, processor = load_model(args.model, args.dtype, adapter=args.adapter,
                                      max_pixels=args.max_pixels)
        model.eval()
        preds, raw = _generate(model, processor, test_s, args)
        results["tuned"] = score(preds, test_s)
        print(fmt("tuned", results["tuned"]))
        for p, r, s in zip(preds, raw, test_s):
            px, py = (to_pixels(*p, s.width, s.height) if p else (None, None))
            tx, ty = to_pixels(s.x1000, s.y1000, s.width, s.height)
            rows.append(dict(image=os.path.basename(s.image), patient=s.patient_id,
                             view=s.view, truth_px=[round(tx, 1), round(ty, 1)],
                             pred_px=None if p is None else [round(px, 1), round(py, 1)],
                             err_mm=None if p is None else
                             round(math.hypot(px - tx, py - ty) * s.mm_per_px, 2),
                             raw=r[:120]))

    verdict = None
    if "tuned" in results and results["tuned"]["median_mm"] is not None:
        c, t = results["centre"]["median_mm"], results["tuned"]["median_mm"]
        ratio = c / t if t > 0 else float("inf")
        passed = t < args.gate_mm and ratio >= 1.5
        verdict = dict(gate_mm=args.gate_mm, tuned_median_mm=t, centre_median_mm=c,
                       improvement_over_centre=round(ratio, 2), passed=bool(passed))
        print(f"\n{'=' * 66}")
        print(f"tuned median {t:.1f} mm vs centre-baseline {c:.1f} mm "
              f"({ratio:.2f}x better)")
        if passed:
            print(f"GATE PASSED (median < {args.gate_mm:g} mm and clearly beats the "
                  "constant baseline).")
        elif t >= args.gate_mm:
            print(f"GATE FAILED: median {t:.1f} mm is above the {args.gate_mm:g} mm bar.")
        else:
            print("GATE NOT PASSED: the model beats the bar but not the constant "
                  "baseline\n  -- it has learned where nodules usually are, "
                  "not how to find this one.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(dict(results=results, verdict=verdict, split=args.split,
                       n=len(test_s), model=args.model, adapter=args.adapter),
                  f, indent=2)
    if rows:
        with open(os.path.join(args.out, "predictions.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        worst = sorted([r for r in rows if r["err_mm"] is not None],
                       key=lambda r: -r["err_mm"])[:5]
        if worst:
            print("\nworst predictions (look at these before believing any number):")
            for r in worst:
                print(f"  {r['err_mm']:7.1f} mm  {r['image']}  pred {r['pred_px']} "
                      f"truth {r['truth_px']}")
    print(f"\n-> {os.path.join(args.out, 'eval.json')}")
    return 0 if (verdict or {}).get("passed") else 0


# --------------------------------------------------------------------------
# stats -- decide the task shape from the data, before training on it
# --------------------------------------------------------------------------


def cmd_stats(args) -> int:
    import collections

    recs = [json.loads(l) for l in open(os.path.join(args.data, "manifest.jsonl"))]
    print(f"{len(recs)} images, {sum(len(r['nodules']) for r in recs)} nodule labels, "
          f"{len({r['patient_id'] for r in recs})} patients\n")
    print("LIDC marks both >=3mm nodules (contoured) and <3mm nodules (a single")
    print("point). The builder gives a single-point mark radius exactly 1.5mm, so")
    print("radius > 1.5 is the contoured subset -- the ones a radiologist outlined.\n")

    for thr in (0.0, 1.51, 3.0):
        hist, n_nod = collections.Counter(), 0
        for r in recs:
            k = sum(1 for n in r["nodules"] if n.get("radius_mm", 0) > thr)
            hist[k] += 1
            n_nod += k
        label = "all marks" if thr == 0 else f"radius > {thr}mm"
        per = " ".join(f"{k}:{hist[k]}" for k in sorted(hist)[:8])
        print(f"  {label:<16} {n_nod:5d} nodules | per-image  {per}")
        print(f"  {'':<16} exactly-one: {hist[1]:4d} images | "
              f">=1 (usable for 'largest'): {sum(v for k, v in hist.items() if k):4d}")
    print()

    skip = os.path.join(args.data, "skipped.jsonl")
    if os.path.exists(skip):
        c = collections.Counter(json.loads(l).get("reason") for l in open(skip))
        print("skipped.jsonl:")
        for reason, n in c.most_common():
            print(f"  {n:6d}  {reason}")
    return 0


# --------------------------------------------------------------------------
# selftest -- runs with no GPU, no model, no dataset
# --------------------------------------------------------------------------


def cmd_selftest(args) -> int:
    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    print("answer parsing")
    for txt, want in [('{"x": 650, "y": 684}', (650, 684)),
                      ('Sure! {"x":12,"y":34} hope that helps', (12, 34)),
                      ('```json\n{"x": 1, "y": 2}\n```', (1, 2)),
                      ('x=650, y=684', (650, 684)),
                      ('The nodule is at 650 684.', (650, 684)),
                      ('{"x": 650}', None), ('no idea', None), ('', None)]:
        got = parse_answer(txt)
        t(f"parse {txt[:34]!r}", (got is None and want is None) or
          (got is not None and want is not None and
           abs(got[0] - want[0]) < 1e-6 and abs(got[1] - want[1]) < 1e-6),
          f"got {got}, want {want}")
    t("round-trip", parse_answer(format_answer(327.5, 344.8)) == (328.0, 345.0))

    print("\nmetrics")
    s = [Sample("i.png", 500, 500, 504, 504, 1.0, "p", "PA", 3.0, 3) for _ in range(4)]
    perfect = score([(500, 500)] * 4, s)
    t("perfect predictions -> 0 mm", perfect["median_mm"] == 0 and perfect["hit@10mm"] == 1.0)
    # 1000-normalised units: 10 units = 10/1000*504 px = 5.04 px = 5.04 mm here
    off = score([(500 + 10, 500)] * 4, s)
    t("known offset in mm", abs(off["median_mm"] - 10 / 1000 * 504) < 1e-6,
      f"{off['median_mm']:.3f} mm")
    half = score([(500, 500), None, (500, 500), None], s)
    t("unparseable counts as a miss, not a skip",
      half["parse_fail"] == 2 and half["hit@10mm"] == 0.5,
      f"hit@10mm {half['hit@10mm']:.2f} over n={half['n']}")

    print("\ncentre baseline")
    tr = [Sample("i", 400, 600, 504, 504, 1.0, "a", "PA", 3, 3),
          Sample("i", 600, 400, 504, 504, 1.0, "b", "PA", 3, 3)]
    cb = centre_baseline(tr, s)
    t("predicts the training mean", cb["predicts"] == [500.0, 500.0], str(cb["predicts"]))
    t("scores 0 mm when truth is the mean", cb["median_mm"] == 0)

    print("\n" + ("all self-tests passed" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
        p.add_argument("--data", required=True, help="output dir of build_drr_dataset.py")
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--max-pixels", type=int, default=512 * 512)
        p.add_argument("--multi", choices=["largest", "any", "single"],
                       default="largest", help="what a multi-nodule image means")
        p.add_argument("--landmark", default=None,
                       help="target one landmark by name, e.g. lung_apex_left")
        p.add_argument("--min-radius-mm", type=float, default=3.0,
                       help="drop LIDC <3mm single-point marks")
        p.add_argument("--views", nargs="*", default=None, help="e.g. PA")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--workers", type=int, default=0)

    pt = sub.add_parser("train"); common(pt)
    pt.add_argument("--out", default="runs/gate1")
    pt.add_argument("--epochs", type=int, default=2)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--rank", type=int, default=16)
    pt.add_argument("--grad-accum", type=int, default=4)
    pt.add_argument("--grad-checkpointing", action="store_true", default=True)
    pt.add_argument("--log-every", type=int, default=10)
    pt.add_argument("--seed", type=int, default=0)

    pe = sub.add_parser("eval"); common(pe)
    pe.add_argument("--adapter", default=None, help="omit to score the base model only")
    pe.add_argument("--out", default="runs/gate1")
    pe.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    pe.add_argument("--max-new-tokens", type=int, default=32)
    pe.add_argument("--gate-mm", type=float, default=20.0)
    pe.add_argument("--skip-zeroshot", action="store_true")
    pe.add_argument("--verbose", action="store_true", default=True)

    ps = sub.add_parser("stats")
    ps.add_argument("--data", required=True)

    sub.add_parser("selftest")

    args = ap.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval, "stats": cmd_stats,
            "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
