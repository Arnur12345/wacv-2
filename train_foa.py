"""
Train FOA: predict a target's position in MILLIMETRES from projections.

This is the step where the task changes. The Step 2 gate asked for a pixel in
one image; this asks for a 3D point, which is the only question whose answer can
be shaped by the projection geometry -- and therefore the only one E1 can
measure.

    python train_foa.py selftest                       # no GPU, no transformers
    python train_foa.py cache --data /data/lidc/drr_lm  # precompute w
    python train_foa.py train --data /data/lidc/drr_lm --out runs/foa1
    python train_foa.py train --data /data/lidc/drr_lm --out runs/uniform --uniform-w
    python train_foa.py eval  --data /data/lidc/drr_lm --adapter runs/foa1

## The gate

Run twice, identically, once with --uniform-w. If geometric `w` does not beat
uniform `w`, the mechanism is not doing the work and nothing downstream is worth
measuring.

## Views come from the manifest

Each record carries source, detector centre and axes, spacing and size, plus the
volume affine -- so the views are rebuilt exactly as rendered, and no CT is
loaded at training time. Views are built at patch=28 because Qwen3-VL merges 2x2
patches; `check_alignment` enforces it against the tower's actual token count.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from foa_model import (PATCH_EFFECTIVE, FOA, FOAConfig, WCache, check_alignment,
                       effective_width, geometry_for_sample, temperature_at,
                       w_cache_key)
from slots import SlotGrid, build_log_w

PROMPT_PREFIX = ("Chest projections of one patient are provided as spatial "
                 "tokens. ")
PROMPT_SUFFIX = ('Give the position of {target} in patient millimetres as JSON: '
                 '{{"x": <mm>, "y": <mm>, "z": <mm>}}.')


def format_mm(p: Sequence[float]) -> str:
    return json.dumps({k: round(float(v), 1) for k, v in zip("xyz", p)})


def parse_mm(text: str) -> Optional[Tuple[float, float, float]]:
    """Three numbers out of whatever the model said, or None."""
    if not text:
        return None
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if all(k in d for k in "xyz"):
                return tuple(float(d[k]) for k in "xyz")
        except Exception:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 3:
        return tuple(float(x) for x in nums[:3])
    return None


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------


@dataclass
class FOASample:
    series_uid: str
    patient_id: str
    split: str
    target_name: str
    target_mm: Tuple[float, float, float]
    images: List[str]                 # one per view, in view order
    view_specs: List[Dict]           # geometry blocks, same order
    affine: List[List[float]]
    volume_size: List[int]


def load_samples(data_dir: str, landmark: str, views: Sequence[str] = ("PA", "LAT"),
                 drop_file: str = "drop_series.txt") -> List[FOASample]:
    """
    Group the per-image manifest into per-patient multi-view samples.

    A sample needs every requested view, so a series missing one is dropped
    rather than silently trained with a view absent -- which would confound the
    view-count comparison E1 depends on.
    """
    drop = set()
    p = os.path.join(data_dir, drop_file)
    if os.path.exists(p):
        drop = {l.strip() for l in open(p) if l.strip()}

    by_series: Dict[str, Dict] = {}
    for line in open(os.path.join(data_dir, "manifest.jsonl")):
        r = json.loads(line)
        if r["series_uid"] in drop:
            continue
        d = by_series.setdefault(r["series_uid"], {})
        d[r["view"]] = r

    out = []
    for uid, per_view in by_series.items():
        if any(v not in per_view for v in views):
            continue
        ref = per_view[views[0]]
        tgt = next((n for n in ref["nodules"] if n.get("name") == landmark), None)
        if tgt is None:
            continue
        imgs = [os.path.join(data_dir, per_view[v]["image"]) for v in views]
        if not all(os.path.exists(i) for i in imgs):
            continue
        out.append(FOASample(
            series_uid=uid, patient_id=ref["patient_id"], split=ref["split"],
            target_name=landmark, target_mm=tuple(float(x) for x in tgt["center_mm"]),
            images=imgs, view_specs=[per_view[v]["geometry"] for v in views],
            affine=ref["geometry"]["affine"],
            volume_size=ref["geometry"]["volume_size"]))
    return out


def volume_box_mm(affine, size) -> Tuple[torch.Tensor, torch.Tensor]:
    """The volume's mm bounding box, from the affine alone -- no CT loaded."""
    A = torch.tensor(affine, dtype=torch.float64)
    nx, ny, nz = size
    corners = torch.tensor([[i, j, k, 1.0] for i in (0, nx - 1)
                            for j in (0, ny - 1) for k in (0, nz - 1)],
                           dtype=torch.float64)
    pts = (A @ corners.T).T[:, :3]
    return pts.min(dim=0).values, pts.max(dim=0).values


def rebuild_views(sample: FOASample, n_views: Optional[int] = None,
                  token_grid: Optional[Tuple[int, int]] = None,
                  patch_px: int = PATCH_EFFECTIVE):
    """The exact DRRViews the images were rendered with, at patch=28."""
    from geometry_kernel import DRRView, CTVolume

    A = torch.tensor(sample.affine, dtype=torch.float64)
    nx, ny, nz = sample.volume_size
    dummy = CTVolume(array=np.zeros((1, 1, 1), dtype=np.float32), A=A,
                     A_inv=torch.linalg.inv(A),
                     spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                     direction=np.eye(3))
    out = []
    for g in sample.view_specs[:n_views or len(sample.view_specs)]:
        model_hw = (None if token_grid is None
                    else (token_grid[0] * patch_px, token_grid[1] * patch_px))
        v = DRRView(dummy, g["source_mm"], g["det_center_mm"], g["det_u"], g["det_v"],
                    det_spacing=tuple(g["det_spacing"]), det_size=tuple(g["det_size"]),
                    patch=patch_px, model_hw=model_hw)
        out.append(v)
    return out


def sample_geometry(sample: FOASample, slots: SlotGrid, cache: Optional[WCache],
                    n_views: Optional[int] = None, uniform: bool = False):
    """(slot_mm [M,3], w [M,P], f12 [M,12]) for one sample."""
    views = rebuild_views(sample, n_views)
    lo, hi = volume_box_mm(sample.affine, sample.volume_size)
    slot_mm = slots.coords_mm(lo, hi)
    key = w_cache_key(sample.series_uid, [f"v{i}" for i in range(len(views))],
                      slots.grid, PATCH_EFFECTIVE)
    w, f12 = geometry_for_sample(slot_mm, views, cache, key)
    if uniform:
        # The ablation: same shape, same cost, no geometry. Every slot may look
        # everywhere, which is what the mechanism has to beat.
        w = torch.ones_like(w.double())
    return slot_mm, w.double(), f12.double(), views


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------


def _fake_dataset(root: str) -> str:
    """A manifest written from the phantom, with real geometry blocks."""
    from geometry_kernel import DRRView, synthetic_thorax
    from PIL import Image

    d = os.path.join(root, "foa_ds")
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    vol = synthetic_thorax()
    recs = []
    for s in range(3):
        for orient in ("PA", "LAT"):
            v = DRRView.standard(vol, orient, det_size=(504, 504))
            name = f"images/p{s}_{orient}.png"
            Image.fromarray(np.zeros((504, 504), np.uint8)).save(os.path.join(d, name))
            tgt_mm = torch.tensor([60.0, -20.0, -60.0], dtype=torch.float64)
            uu, vv, _ = v.project(tgt_mm)
            px = [round(float(uu[0]), 2), round(float(vv[0]), 2)]
            recs.append(dict(
                image=name, view=orient, width=504, height=504,
                patient_id=f"P{s}", series_uid=f"uid{s}",
                split="train" if s < 2 else "test",
                window=[0.0, 1.0],
                nodules=[dict(name="lung_apex_left", center_mm=[60.0, -20.0, -60.0],
                              radius_mm=5.0, pixel=px,
                              pixel_norm1000=[px[0] / 504 * 1000,
                                              px[1] / 504 * 1000])],
                geometry=dict(
                    source_mm=[float(x) for x in v.source_mm],
                    det_center_mm=[float(x) for x in v.det_center_mm],
                    det_u=[float(x) for x in v.det_u],
                    det_v=[float(x) for x in v.det_v],
                    det_spacing=[float(x) for x in v.det_spacing],
                    det_size=[int(x) for x in v.det_size],
                    volume_size=list(vol.size_ijk),
                    volume_spacing=[float(x) for x in vol.spacing],
                    affine=[[float(x) for x in row] for row in vol.A.tolist()])))
    with open(os.path.join(d, "manifest.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in recs))
    return d


def selftest() -> int:
    import tempfile
    from foa_model import _StubLM

    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    torch.manual_seed(0)
    print("answer format")
    p = [1.24, -20.0, -60.55]
    once = format_mm(p)
    t("mm round-trips to 0.1mm",
      all(abs(a - b) <= 0.05 for a, b in zip(parse_mm(once), p)) and
      format_mm(parse_mm(once)) == once, once)
    t("prose is tolerated", parse_mm('sure: {"x": 1, "y": 2, "z": 3}') == (1.0, 2.0, 3.0))
    t("two numbers is a failure, not a guess", parse_mm("x=1 y=2") is None)

    with tempfile.TemporaryDirectory() as td:
        d = _fake_dataset(td)
        print("\nsamples")
        S = load_samples(d, "lung_apex_left")
        t("grouped per series across views", len(S) == 3, f"{len(S)} samples")
        t("each sample has both views", all(len(s.images) == 2 for s in S))
        t("target is 3D mm", S[0].target_mm == (60.0, -20.0, -60.0))
        t("splits survive grouping", {s.split for s in S} == {"train", "test"})

        print("\ngeometry from the manifest alone (no CT loaded)")
        slots = SlotGrid((4, 4, 4), 32)
        cache = WCache(os.path.join(td, "wcache"))
        slot_mm, w, f12, views = sample_geometry(S[0], slots, cache)
        lo, hi = volume_box_mm(S[0].affine, S[0].volume_size)
        t("bounding box is a plausible chest",
          bool(((hi - lo) > 100).all() and ((hi - lo) < 600).all()),
          f"extent {[round(float(x)) for x in (hi - lo)]} mm")
        t("views rebuilt at patch 28 -> 324 tokens each",
          views[0].grid.n_patches == 324)
        t("w covers both views", w.shape == (slots.n_slots, 648), str(tuple(w.shape)))
        # a 5x5 Gaussian splat per view over 2 views is ~50 of 648 patches
        t("w is sparse and nonzero", 0 < float((w > 0).float().mean()) < 0.12,
          f"{float((w > 0).float().mean()):.2%} nonzero, "
          f"{float((w > 0).sum(1).double().mean()):.0f} patches/slot")

        # the projected target must land where the manifest said it did
        from geometry_kernel import DRRView
        tgt = torch.tensor(S[0].target_mm, dtype=torch.float64)
        u, v_, okp = views[0].project(tgt)
        rec = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))][0]
        want = rec["nodules"][0]["pixel"]
        t("rebuilt view reproduces the manifest's pixel",
          abs(float(u[0]) - want[0]) < 1.0 and abs(float(v_[0]) - want[1]) < 1.0,
          f"({float(u[0]):.1f},{float(v_[0]):.1f}) vs {want}")

        print("\ncache and ablation")
        _, w2, _, _ = sample_geometry(S[0], slots, cache)
        t("second call hits the cache", torch.allclose(w, w2))
        _, wu, _, _ = sample_geometry(S[0], slots, cache, uniform=True)
        t("uniform-w ablation is all-ones and same shape",
          wu.shape == w.shape and float(wu.min()) == 1.0)
        t("  and it is much wider",
          effective_width(build_log_w(wu)[None]) >
          20 * effective_width(build_log_w(w)[None]),
          f"{effective_width(build_log_w(wu)[None]):.0f} vs "
          f"{effective_width(build_log_w(w)[None]):.1f} patches")

        print("\none training step with stub modules")
        lm = _StubLM(dim=32)
        foa = FOA(lm, patch_dim=16, lm_dim=32,
                  cfg=FOAConfig(slot_grid=(4, 4, 4), slot_dim=32, n_heads=4))
        pre = torch.randint(0, 64, (1, 6))
        suf = torch.randint(0, 64, (1, 5))
        patches = torch.randn(1, 648, 16)
        log_w = build_log_w(w, temperature=temperature_at(0, 10))[None].float()
        labels = torch.randint(0, 64, (1, 5))
        opt = torch.optim.AdamW(list(foa.trainable()), lr=1e-3)
        l0 = foa(pre, suf, patches, log_w, f12.float(), labels=labels).loss
        l0.backward(); opt.step(); opt.zero_grad()
        l1 = foa(pre, suf, patches, log_w, f12.float(), labels=labels).loss
        t("loss is finite and a step changes it",
          bool(torch.isfinite(l0) and torch.isfinite(l1)) and float(l0) != float(l1),
          f"{float(l0):.4f} -> {float(l1):.4f}")

        widths = [effective_width(build_log_w(w, temperature=temperature_at(s, 10))[None])
                  for s in (0, 5, 9)]
        t("annealing narrows the kernel over training",
          widths[0] > widths[-1],
          " -> ".join(f"{x:.1f}" for x in widths))

    print("\n" + ("all self-tests passed" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1



# --------------------------------------------------------------------------
# transformers glue
# --------------------------------------------------------------------------


def lora_targets(model):
    """LM projections only, found by shape. The vision tower stays frozen."""
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
    if not names:
        raise RuntimeError("no LM projections found -- check the model id")
    return sorted(names)


def build_model(model_id: str, cfg: FOAConfig, rank: int = 16, dtype="bfloat16"):
    """Frozen tower + LoRA'd LM + the FOA head, on one device."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    td = getattr(torch, dtype)
    processor = AutoProcessor.from_pretrained(model_id)
    # from_pretrained does not reliably forward these to the image processor, so
    # set them on it directly -- otherwise a 504px DRR is silently upscaled to
    # 896px and the tower emits 32x32 tokens instead of 18x18.
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        ip.min_pixels = 64 * 28 * 28
        ip.max_pixels = 324 * 28 * 28          # 18x18 merged tokens at 504px
        if isinstance(getattr(ip, "size", None), dict):
            ip.size = dict(ip.size, shortest_edge=ip.min_pixels,
                           longest_edge=ip.max_pixels)
    base = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=td, attn_implementation="sdpa")

    for n, p in base.named_parameters():          # freeze the tower explicitly
        if any(k in n for k in ("visual", "vision_tower", "vision_model")):
            p.requires_grad_(False)

    base = get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=lora_targets(base)))

    lm_dim = base.get_input_embeddings().weight.shape[1]
    patch_dim, token_grid, patch_px = _probe_tower(base, processor, td)
    foa = FOA(base, patch_dim=patch_dim, lm_dim=lm_dim, cfg=cfg)
    foa.token_grid, foa.patch_px = token_grid, patch_px
    foa.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  lm_dim={lm_dim} patch_dim={patch_dim} "
          f"token grid {token_grid[0]}x{token_grid[1]} = "
          f"{token_grid[0]*token_grid[1]} per view at {patch_px}px | "
          f"trainable={sum(p.numel() for p in foa.parameters() if p.requires_grad):,}")
    return foa, processor


def _probe_tower(model, processor, td):
    """
    One forward through the tower to learn its output width and TOKEN GRID.

    The grid is read from the processor rather than assumed: `image_grid_thw`
    is in pre-merge 14px patch units, so the merged grid is (h//2, w//2). Views
    are then built to match whatever the tower actually did, which is the only
    way this cannot silently disagree.
    """
    from PIL import Image

    img = Image.fromarray(np.zeros((504, 504, 3), np.uint8))
    enc = processor(text=["x"], images=[img], return_tensors="pt")
    dev = next(model.parameters()).device
    with torch.no_grad():
        f = vision_features(model, enc["pixel_values"].to(dev, td),
                            enc["image_grid_thw"].to(dev))
    t_, h, w = [int(x) for x in enc["image_grid_thw"][0]]
    n_tok = int(f.shape[0])
    # `image_grid_thw` counts 14px patches. Whether the tower hands back merged
    # (h/2 x w/2, 28px) or unmerged (h x w, 14px) features differs by version,
    # so infer it from the count instead of assuming either.
    if n_tok == h * w:
        grid, patch_px = (h, w), 14
    elif n_tok == (h // 2) * (w // 2):
        grid, patch_px = (h // 2, w // 2), 28
    else:
        raise ValueError(
            f"tower emitted {n_tok} tokens for a {h}x{w} patch grid; that is "
            f"neither unmerged ({h*w}) nor 2x2-merged ({(h//2)*(w//2)}). "
            f"Inspect get_image_features before going further.")
    return int(f.shape[-1]), grid, patch_px


def vision_features(model, pixel_values, image_grid_thw) -> torch.Tensor:
    """
    [P, D] patch features from the frozen tower.

    transformers wraps this return value differently across versions, so unwrap
    rather than assume: a bare tensor, a tuple, or an output object.
    """
    inner = model.base_model.model if hasattr(model, "base_model") else model
    with torch.no_grad():
        out = inner.get_image_features(pixel_values=pixel_values,
                                       image_grid_thw=image_grid_thw)
    if isinstance(out, torch.Tensor):
        f = out
    elif isinstance(out, (list, tuple)):
        f = out[0]
    else:
        f = getattr(out, "last_hidden_state", None)
        if f is None:
            f = getattr(out, "image_embeds", None)
    if f is None:
        raise RuntimeError(f"could not unwrap vision features from {type(out)}")
    if isinstance(f, (list, tuple)):
        f = torch.cat([x.reshape(-1, x.shape[-1]) for x in f], dim=0)
    return f.reshape(-1, f.shape[-1])


MARKER = "\n<<SLOTS>>\n"


def prompt_ids(processor, target_name: str, device):
    """(prefix_ids, suffix_ids): the text around the slot block."""
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text",
          "text": PROMPT_PREFIX + MARKER + PROMPT_SUFFIX.format(
              target=target_name.replace("_", " "))}]}],
        tokenize=False, add_generation_prompt=True)
    pre_txt, suf_txt = text.split(MARKER)
    tok = processor.tokenizer
    pre = tok(pre_txt, add_special_tokens=False, return_tensors="pt").input_ids
    suf = tok(suf_txt, add_special_tokens=False, return_tensors="pt").input_ids
    return pre.to(device), suf.to(device)


def encode_sample(foa, processor, s: FOASample, slots: SlotGrid, cache, args,
                  view_idx: Optional[Sequence[int]] = None, train: bool = True):
    """Everything one forward needs: patches, log_w, f12, prefix/suffix, labels."""
    from PIL import Image

    dev = next(foa.parameters()).device
    td = next(foa.parameters()).dtype
    if view_idx is None:
        view_idx = list(range(len(s.images)))

    imgs = [Image.open(s.images[i]).convert("RGB") for i in view_idx]
    enc = processor(text=["x"] * len(imgs), images=imgs, return_tensors="pt")
    feats = vision_features(foa.lm, enc["pixel_values"].to(dev, td),
                            enc["image_grid_thw"].to(dev))

    sub = FOASample(**{**s.__dict__,
                       "view_specs": [s.view_specs[i] for i in view_idx],
                       "images": [s.images[i] for i in view_idx]})
    key_views = [f"{args.views[i]}" for i in view_idx]
    views = rebuild_views(sub, token_grid=getattr(foa, "token_grid", None),
                          patch_px=getattr(foa, "patch_px", PATCH_EFFECTIVE))
    lo, hi = volume_box_mm(s.affine, s.volume_size)
    slot_mm = slots.coords_mm(lo, hi)
    key = w_cache_key(s.series_uid, key_views + [f"g{views[0].grid.n_patches}"],
                      slots.grid, views[0].patch)
    w, f12 = geometry_for_sample(slot_mm, views, cache, key)
    w = w.double()
    if args.uniform_w:
        w = torch.ones_like(w)
    check_alignment(w.shape[1], feats.shape[0], len(views))

    pre, suf = prompt_ids(processor, s.target_name, dev)
    labels = None
    if train:
        ans = format_mm(s.target_mm) + (processor.tokenizer.eos_token or "")
        ans_ids = processor.tokenizer(ans, add_special_tokens=False,
                                      return_tensors="pt").input_ids.to(dev)
        suf = torch.cat([suf, ans_ids], dim=1)
        labels = torch.full_like(suf, -100)
        labels[:, -ans_ids.shape[1]:] = ans_ids
    return dict(pre=pre, suf=suf, feats=feats[None], w=w, f12=f12.double(),
                labels=labels, views=views, slot_mm=slot_mm)


def cmd_cache(args) -> int:
    S = load_samples(args.data, args.landmark, args.views)
    slots = SlotGrid(tuple(args.slot_grid), 8)
    cache = WCache(os.path.join(args.data, "wcache"))
    t0 = time.time()
    for i, s in enumerate(S, 1):
        for k in range(1, len(args.views) + 1):
            for idx in ([[j] for j in range(len(args.views))] if k == 1
                        else [list(range(len(args.views)))]):
                sub = FOASample(**{**s.__dict__,
                                   "view_specs": [s.view_specs[j] for j in idx],
                                   "images": [s.images[j] for j in idx]})
                n_p = args.token_grid[0] * args.token_grid[1] * len(idx)
                key = w_cache_key(s.series_uid,
                                  [args.views[j] for j in idx] + [f"g{n_p}"],
                                  slots.grid, args.patch_px)
                lo, hi = volume_box_mm(s.affine, s.volume_size)
                geometry_for_sample(
                    slots.coords_mm(lo, hi),
                    rebuild_views(sub, token_grid=tuple(args.token_grid),
                                  patch_px=args.patch_px), cache, key)
        if i % 50 == 0:
            print(f"  [{i}/{len(S)}] {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"cached {len(S)} series -> {cache.root}")
    return 0


def cmd_train(args) -> int:
    random.seed(args.seed); torch.manual_seed(args.seed)
    S = load_samples(args.data, args.landmark, args.views)
    tr = [s for s in S if s.split == "train"]
    va = [s for s in S if s.split == "val"]
    if args.limit:
        tr = tr[:args.limit]
    print(f"train {len(tr)} / val {len(va)} series, target={args.landmark}, "
          f"uniform_w={args.uniform_w}")
    if not tr:
        print("no training data"); return 1

    cfg = FOAConfig(slot_grid=tuple(args.slot_grid), slot_dim=args.slot_dim,
                    n_heads=args.heads)
    foa, processor = build_model(args.model, cfg, rank=args.rank)
    slots = foa.slots
    cache = WCache(os.path.join(args.data, "wcache"))

    steps = max(1, len(tr) // args.grad_accum) * args.epochs
    params = [p for p in foa.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-6)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps, pct_start=0.05)
    print(f"{steps} optimiser steps\n")

    os.makedirs(args.out, exist_ok=True)
    step, run, n_skip, n_skip_run, n_ok = 0, None, 0, 0, 0
    t0 = time.time()
    for ep in range(args.epochs):
        random.shuffle(tr)
        for i, s in enumerate(tr):
            # Vary the view set: this is what teaches the model to *use* the
            # conditioning rather than memorise a fixed input format, and it is
            # what makes a 1-view evaluation in-distribution later.
            n_v = len(args.views)
            idx = (list(range(n_v)) if random.random() > args.view_dropout
                   else [random.randrange(n_v)])
            try:
                b = encode_sample(foa, processor, s, slots, cache, args, idx, True)
                n_skip_run = 0
            except Exception as e:
                n_skip += 1
                n_skip_run += 1
                print(f"  skip {s.patient_id}: {type(e).__name__}: {e}")
                # A misconfiguration skips *every* sample, and a loop that keeps
                # going then reports four epochs of training on nothing and
                # saves a checkpoint. Fail loudly instead.
                if n_skip_run >= 10:
                    raise RuntimeError(
                        f"{n_skip_run} consecutive samples failed to encode -- "
                        f"this is a configuration error, not bad data. Last: {e}")
                continue
            tau = temperature_at(step, steps, args.tau_start, args.tau_end)
            dev = next(foa.parameters()).device
            dt = next(foa.parameters()).dtype
            log_w = build_log_w(b["w"], temperature=tau)[None].to(dev, dt)
            loss = foa(b["pre"], b["suf"], b["feats"], log_w,
                       b["f12"].to(dev, dt), labels=b["labels"]).loss
            (loss / args.grad_accum).backward()
            run = float(loss.detach()) if run is None else 0.98 * run + 0.02 * float(loss.detach())
            n_ok += 1

            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    el = time.time() - t0
                    print(f"  ep{ep} step {step}/{steps} loss {run:.4f} "
                          f"tau {tau:.2f} width {effective_width(log_w.float()):.1f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {el/60:.1f}m "
                          f"eta {(steps-step)*el/max(step,1)/60:.0f}m", flush=True)
        if n_ok == 0:
            raise RuntimeError("epoch finished with zero usable samples; refusing "
                               "to save a checkpoint trained on nothing")
        print(f"  epoch {ep}: {n_ok} samples used, {n_skip} skipped")
        foa.lm.save_pretrained(os.path.join(args.out, "lora"))
        torch.save({k: v for k, v in foa.state_dict().items()
                    if not k.startswith("lm.")}, os.path.join(args.out, "foa.pt"))
        print(f"  epoch {ep} saved -> {args.out}")

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    return 0


@torch.no_grad()
def cmd_eval(args) -> int:
    from e1_metric import anisotropy, compare, ray_directions

    S = load_samples(args.data, args.landmark, args.views)
    te = [s for s in S if s.split == args.split]
    if args.limit:
        te = te[:args.limit]
    print(f"eval {len(te)} series on split={args.split}\n")

    cfg = FOAConfig(slot_grid=tuple(args.slot_grid), slot_dim=args.slot_dim,
                    n_heads=args.heads)
    foa, processor = build_model(args.model, cfg, rank=args.rank)
    if args.adapter:
        from peft import PeftModel
        sd = torch.load(os.path.join(args.adapter, "foa.pt"), map_location="cpu")
        foa.load_state_dict(sd, strict=False)
        print(f"  loaded FOA head from {args.adapter}")
    foa.eval()
    cache = WCache(os.path.join(args.data, "wcache"))

    conditions = {"1view-pa": [0], "1view-lat": [1],
                  "2view": list(range(len(args.views)))}
    records, results = [], {}
    for cname, idx in conditions.items():
        if max(idx) >= len(args.views):
            continue
        errs, rays, n_fail = [], [], 0
        for s in te:
            try:
                b = encode_sample(foa, processor, s, foa.slots, cache, args, idx, False)
            except Exception:
                n_fail += 1
                continue
            dev = next(foa.parameters()).device
            dt = next(foa.parameters()).dtype
            log_w = build_log_w(b["w"], temperature=args.tau_end)[None].to(dev, dt)
            tok = foa.slot_tokens(b["feats"], log_w, b["f12"].to(dev, dt))
            emb = foa.lm.get_input_embeddings()
            seq = torch.cat([emb(b["pre"]), tok.to(emb.weight.dtype),
                             emb(b["suf"])], dim=1)
            out = foa.lm.generate(inputs_embeds=seq,
                                  attention_mask=torch.ones(seq.shape[:2],
                                                            dtype=torch.long,
                                                            device=seq.device),
                                  max_new_tokens=48, do_sample=False,
                                  pad_token_id=processor.tokenizer.pad_token_id)
            txt = processor.tokenizer.decode(out[0], skip_special_tokens=True)
            p = parse_mm(txt)
            if p is None:
                n_fail += 1
                continue
            e = np.array(p) - np.array(s.target_mm)
            src = np.array([float(x) for x in b["views"][0].source_mm])
            errs.append(e); rays.append(np.array(s.target_mm) - src)
            records.append(dict(condition=cname, patient=s.patient_id,
                                pred=list(map(float, p)),
                                truth=list(map(float, s.target_mm)),
                                error_mm=float(np.linalg.norm(e)), raw=txt[:120]))
        if errs:
            m = anisotropy(np.stack(errs), np.stack(rays))
            m["parse_fail"] = n_fail
            results[cname] = m
        print(f"  {cname}: {len(errs)} ok, {n_fail} failed", flush=True)

    print("\n" + compare(results))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e1.json"), "w") as f:
        json.dump(dict(results={k: {kk: vv for kk, vv in v.items()}
                                for k, v in results.items()},
                       records=records, uniform_w=args.uniform_w), f, indent=2)
    print(f"\n-> {os.path.join(args.out, 'e1.json')}")
    print("\nRegistered predictions (docs/E1_PREREGISTRATION.md): single-view R > 1, "
          "two-view R -> 1, alignment > 0.8 for single views.")
    return 0



@torch.no_grad()
def cmd_probe(args) -> int:
    """
    Does the tower flatten patches row-major, the way PatchGrid assumes?

    `w` indexes patches as row * n_cols + col. If the tower emits column-major
    or windowed order, every weight lands on the wrong patch -- and nothing
    errors, nothing NaNs, training just proceeds while attending to the wrong
    places. Test it with content: put a bright square in one known patch and see
    which token reacts.
    """
    from PIL import Image

    cfg = FOAConfig(slot_grid=tuple(args.slot_grid), slot_dim=args.slot_dim,
                    n_heads=args.heads)
    foa, processor = build_model(args.model, cfg, rank=args.rank)
    dev = next(foa.parameters()).device
    dt = next(foa.parameters()).dtype
    rows, cols = foa.token_grid
    px = foa.patch_px

    def feats_of(img_arr):
        enc = processor(text=["x"], images=[Image.fromarray(img_arr)],
                        return_tensors="pt")
        return vision_features(foa.lm, enc["pixel_values"].to(dev, dt),
                               enc["image_grid_thw"].to(dev)).float()

    # Differential probe. ViT features are contextualised and some tokens carry
    # huge norms regardless of content ("registers"), so an absolute deviation
    # finds those every time. Subtract the blank-image features and the only
    # thing left is the response to the square.
    blank = np.zeros((rows * px, cols * px, 3), np.uint8)
    f0 = feats_of(blank)

    probes = [(0, 1), (1, 0), (rows // 2, cols // 3), (rows - 1, cols - 1)]
    got_idx = []
    ok_all = True
    for (r, c) in probes:
        a = blank.copy()
        a[r * px:(r + 1) * px, c * px:(c + 1) * px] = 255
        d = (feats_of(a) - f0).norm(dim=-1)
        got = int(d.argmax())
        got_idx.append(got)
        want_row_major = r * cols + c
        want_col_major = c * rows + r
        tag = ("row-major" if got == want_row_major else
               "COLUMN-MAJOR" if got == want_col_major else "other")
        share = float(d[got] / d.sum())
        ok_all &= (got == want_row_major)
        print(f"  bright patch at (row={r:2d}, col={c:2d}): token {got:4d} "
              f"(row-major expects {want_row_major:4d}) -> {tag}"
              f"   [response share {share:.1%}]")

    if all(g == got_idx[0] for g in got_idx):
        print("\n  Every probe picked the same token: the differential response is "
              "not\n  spatially localised. Patch identity may not survive this "
              "tower's\n  output; inspect get_image_features before trusting any "
              "index map.")
        return 1

    print()
    if ok_all:
        print("  PASS: the tower is row-major; w indexes patches correctly.")
    else:
        print("  FAIL: w is indexed in the wrong order. Do not run the gate --\n"
              "  fix the flattening in build_w_matrix first, or every weight\n"
              "  lands on the wrong patch and FOA loses for the wrong reason.")
    return 0 if ok_all else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for name in ("cache", "train", "eval", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--landmark", default="lung_apex_left")
        p.add_argument("--views", nargs="+", default=["PA", "LAT"])
        p.add_argument("--slot-grid", nargs=3, type=int, default=[6, 6, 6])
        p.add_argument("--uniform-w", action="store_true",
                       help="the ablation: same cost, no geometry")
        if name == "cache":
            p.add_argument("--token-grid", nargs=2, type=int, required=True,
                           help="the tower's grid, printed by `train` at startup")
            p.add_argument("--patch-px", type=int, required=True,
                           help="14 or 28, also printed by `train` at startup")
            continue
        p.add_argument("--out", default="runs/foa1")
        p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
        p.add_argument("--slot-dim", type=int, default=1024)
        p.add_argument("--heads", type=int, default=8)
        p.add_argument("--rank", type=int, default=16)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--tau-start", type=float, default=4.0)
        p.add_argument("--tau-end", type=float, default=1.0)
        if name == "probe":
            continue
        if name == "train":
            p.add_argument("--epochs", type=int, default=4)
            p.add_argument("--lr", type=float, default=2e-4)
            p.add_argument("--grad-accum", type=int, default=8)
            p.add_argument("--log-every", type=int, default=10)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--view-dropout", type=float, default=0.4,
                           help="probability of training on a single view")
        else:
            p.add_argument("--adapter", default=None)
            p.add_argument("--split", default="test")
    args = ap.parse_args(argv)
    return {"selftest": lambda a: selftest(), "cache": cmd_cache,
            "train": cmd_train, "eval": cmd_eval, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
