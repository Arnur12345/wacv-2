"""
Step 6: wire the slots into Qwen3-VL.

    slots -> gather(log w) -> null-space features -> linear projection
          -> prepended to the LM's token sequence in place of image tokens

The vision tower is frozen and called by us, so the LM never sees pixels: it
sees M slot tokens that each stand at a known millimetre location, carrying
only evidence the geometry says is visible from there.

## The patch-grid trap

Qwen3-VL merges 2x2 vision patches before the LM sees them, so a 504x504 image
yields an 18x18 = 324 token grid, not 36x36. `w` must therefore be built with
patch=28 (14 x the 2x2 merge), or every weight lands on the wrong token and the
whole thing still runs, still trains, and quietly attends to the wrong places.
`PATCH_EFFECTIVE` below is that number; `check_alignment` refuses to proceed if
the tower disagrees with it.

## Zero-init

The projection is zero-init so the image cannot influence the LM at step 0.
"Reproduces the base model exactly" cannot mean bitwise-identical logits -- the
M slot tokens are present in the sequence either way -- so the testable
statement is: at step 0 the output does not depend on the image, and after any
weight change it does. Both are asserted in the self-test.

## w is cached

`w` depends only on geometry, never on weights, so it is computed once per
(volume, view-set, slot grid) and cached to disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from slots import (NEG_INF, Gather, NullSpaceEncoder, SlotGrid, build_log_w,
                   build_w_matrix, conditioning_features, null_space_features,
                   null_space_tensor)

PATCH_EFFECTIVE = 28        # 14px vision patch x 2x2 merge


# --------------------------------------------------------------------------
# w cache
# --------------------------------------------------------------------------


def w_cache_key(series_uid: str, view_names: Sequence[str],
                grid: Tuple[int, int, int], patch: int) -> str:
    raw = f"{series_uid}|{','.join(view_names)}|{grid}|{patch}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


class WCache:
    """Disk cache for w and the null-space features. Geometry, so never stale."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def get(self, key: str):
        p = os.path.join(self.root, key + ".npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        return torch.from_numpy(z["w"]), torch.from_numpy(z["f12"])

    def put(self, key: str, w: torch.Tensor, f12: torch.Tensor):
        np.savez_compressed(os.path.join(self.root, key + ".npz"),
                            w=w.cpu().numpy().astype(np.float32),
                            f12=f12.cpu().numpy().astype(np.float32))


def geometry_for_sample(slot_mm: torch.Tensor, views: Sequence,
                        cache: Optional[WCache] = None,
                        key: Optional[str] = None,
                        center_mm: Optional[torch.Tensor] = None):
    """
    (w [M,P], conditioning features [M,15]), from cache when available.

    Slot coordinates are expressed RELATIVE to the volume centre. In absolute
    LPS the z of a slot swings by hundreds of millimetres between patients --
    that is the scanner table offset, not anatomy -- so an absolute coordinate
    feature is mostly a nuisance variable the model cannot predict from pixels.
    """
    if cache is not None and key is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    w = build_w_matrix(slot_mm, views)
    rel = slot_mm if center_mm is None else (slot_mm - center_mm[None, :])
    f12 = conditioning_features(null_space_tensor(slot_mm, views, w), rel)
    if cache is not None and key is not None:
        cache.put(key, w, f12)
    return w, f12


# --------------------------------------------------------------------------
# temperature schedule
# --------------------------------------------------------------------------


def temperature_at(step: int, total: int, t_start: float = 4.0,
                   t_end: float = 1.0) -> float:
    """
    Soft to sharp, cosine. Sharp from step one leaves each slot attending to a
    handful of patches with almost no gradient anywhere else, and a slot whose
    w is slightly wrong can never recover the patches it needs.
    """
    if total <= 1:
        return t_end
    f = min(max(step / (total - 1), 0.0), 1.0)
    return t_end + (t_start - t_end) * 0.5 * (1 + math.cos(math.pi * f))


def effective_width(log_w: torch.Tensor) -> float:
    """
    Mean effective number of patches per slot: exp(entropy of softmax(log_w)).

    This is the number to log alongside temperature -- it says what the kernel
    is actually doing, where the temperature only says what we asked for.
    """
    valid = (log_w > NEG_INF / 2).any(dim=-1)
    if not bool(valid.any()):
        return 0.0
    p = log_w[valid].softmax(dim=-1)
    h = -(p * (p.clamp_min(1e-12)).log()).sum(dim=-1)
    return float(h.exp().mean())


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


@dataclass
class FOAConfig:
    slot_grid: Tuple[int, int, int] = (6, 6, 6)
    slot_dim: int = 1024
    n_heads: int = 8
    temperature: float = 1.0
    use_nullspace: bool = True


class FOA(nn.Module):
    """
    Slots -> gather -> null-space conditioning -> zero-init projection -> LM.

    `lm` must expose `get_input_embeddings()` and accept `inputs_embeds` and
    `attention_mask`. `vision` maps a batch of images to patch features
    [P, patch_dim]; it is called under no_grad and never trained.
    """

    def __init__(self, lm: nn.Module, patch_dim: int, lm_dim: int,
                 cfg: FOAConfig = FOAConfig()):
        super().__init__()
        self.cfg = cfg
        self.lm = lm
        self.slots = SlotGrid(cfg.slot_grid, cfg.slot_dim)
        self.gather = Gather(cfg.slot_dim, patch_dim, cfg.n_heads)
        self.nullspace = NullSpaceEncoder(cfg.slot_dim) if cfg.use_nullspace else None
        self.project = nn.Linear(cfg.slot_dim, lm_dim)
        # Zero-init: at step 0 no image information can reach the LM at all.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    # -- slot tokens ------------------------------------------------------

    def slot_tokens(self, patches: torch.Tensor, log_w: torch.Tensor,
                    f12: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, M, lm_dim] -- the tokens the LM will see instead of pixels."""
        B = patches.shape[0]
        s = self.slots().unsqueeze(0).expand(B, -1, -1)
        if self.nullspace is not None and f12 is not None:
            s = self.nullspace(s, f12)
        s = self.gather(s, patches, log_w)
        return self.project(s)

    def forward(self, prefix_ids: torch.Tensor, suffix_ids: torch.Tensor,
                patches: torch.Tensor, log_w: torch.Tensor,
                f12: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        Sequence = [prefix text] [M slot tokens] [suffix text].

        Built as `inputs_embeds` rather than by substituting a placeholder token
        id: the LM then never needs pixel_values, and there is no risk of the
        image-token path firing on a placeholder we borrowed.
        """
        emb = self.lm.get_input_embeddings()
        pre, suf = emb(prefix_ids), emb(suffix_ids)
        tok = self.slot_tokens(patches, log_w, f12).to(pre.dtype)
        seq = torch.cat([pre, tok, suf], dim=1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=seq.device)

        full_labels = None
        if labels is not None:
            pad = torch.full((labels.shape[0], pre.shape[1] + tok.shape[1]),
                             -100, dtype=labels.dtype, device=labels.device)
            full_labels = torch.cat([pad, labels], dim=1)
        return self.lm(inputs_embeds=seq, attention_mask=mask, labels=full_labels)

    # -- parameter groups -------------------------------------------------

    def trainable(self):
        """Everything except the LM and the (externally frozen) vision tower."""
        for m in (self.slots, self.gather, self.project):
            yield from m.parameters()
        if self.nullspace is not None:
            yield from self.nullspace.parameters()


def check_alignment(n_patches_w: int, n_tokens_vision: int, view_count: int):
    """
    Refuse to run if `w`'s patch count differs from the tower's token count.

    This is the one mismatch that produces no error and no NaN: both are square
    grids, so a factor-of-two disagreement just attends to the wrong patch and
    trains happily.
    """
    if n_patches_w != n_tokens_vision:
        raise ValueError(
            f"w covers {n_patches_w} patches but the vision tower emitted "
            f"{n_tokens_vision} tokens for {view_count} view(s). Build the views "
            f"with patch={PATCH_EFFECTIVE} (14px patch x 2x2 merge), and check "
            f"model_hw is what the processor actually produced.")


# --------------------------------------------------------------------------
# self-test -- a stub LM, so this runs with no GPU and no transformers
# --------------------------------------------------------------------------


class _StubLM(nn.Module):
    """
    Minimal stand-in with *causal self-attention*.

    The attention is not decoration: without it the stub is position-wise, slot
    tokens cannot influence the suffix, and the gradient test silently passes
    nothing back -- which is exactly how a broken wiring would look.
    """

    def __init__(self, vocab=64, dim=32, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.head = nn.Linear(dim, vocab)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        h = self.norm(inputs_embeds)
        L = h.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1)
        a, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        h = inputs_embeds + a
        h = h + self.ff(h)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=-100)
        return type("Out", (), dict(logits=logits, loss=loss))()


def selftest() -> int:
    from geometry_kernel import DRRView, synthetic_thorax

    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    torch.manual_seed(0)
    vol = synthetic_thorax()
    # views built at the EFFECTIVE patch size, which is the whole point
    views = [DRRView.standard(vol, o, det_size=(504, 504)) for o in ("PA", "LAT")]
    for v in views:
        v.patch = PATCH_EFFECTIVE
        v.__post_init__()

    print("patch-grid alignment")
    n_per = views[0].grid.n_patches
    t("504px at patch 28 -> 18x18 tokens per view",
      (views[0].grid.n_rows, views[0].grid.n_cols) == (18, 18) and n_per == 324,
      f"{views[0].grid.n_rows}x{views[0].grid.n_cols}={n_per}")
    try:
        check_alignment(2 * n_per, 2 * n_per, 2)
        check_alignment(2 * n_per, 2 * 1296, 2)
        t("mismatch is refused", False)
    except ValueError:
        t("mismatch is refused", True)

    print("\ngeometry and cache")
    cfg = FOAConfig(slot_grid=(4, 4, 4), slot_dim=32, n_heads=4)
    sg = SlotGrid(cfg.slot_grid, cfg.slot_dim)
    lo = vol.voxel_to_mm(torch.zeros(3, dtype=torch.float64))
    hi = vol.voxel_to_mm(torch.tensor([s - 1 for s in vol.size_ijk], dtype=torch.float64))
    coords = sg.coords_mm(torch.minimum(lo, hi), torch.maximum(lo, hi))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cache = WCache(td)
        key = w_cache_key("uid", ["PA", "LAT"], cfg.slot_grid, PATCH_EFFECTIVE)
        w1, f1 = geometry_for_sample(coords, views, cache, key)
        w2, f2 = geometry_for_sample(coords, views, cache, key)
        t("w cache round-trips",
          torch.allclose(w1.double(), w2.double(), atol=1e-6) and
          torch.allclose(f1.double(), f2.double(), atol=1e-5))
        t("cached w has one column per token", w1.shape[1] == 2 * n_per,
          f"{tuple(w1.shape)}")

    print("\ntemperature schedule")
    ts = [temperature_at(s, 100) for s in (0, 50, 99)]
    t("anneals from soft to sharp", ts[0] > ts[1] > ts[2] and abs(ts[2] - 1.0) < 1e-6,
      " -> ".join(f"{x:.2f}" for x in ts))
    widths = [effective_width(build_log_w(w1.double(), temperature=T)) for T in (4.0, 1.0, 0.25)]
    t("sharper temperature narrows the effective kernel",
      widths[0] > widths[1] > widths[2],
      "  ".join(f"T={T}: {x:.1f} patches" for T, x in zip((4.0, 1.0, 0.25), widths)))

    print("\nzero-init")
    lm = _StubLM(dim=32)
    foa = FOA(lm, patch_dim=16, lm_dim=32, cfg=cfg)
    B, P = 1, 2 * n_per
    pre = torch.randint(0, 64, (B, 5))
    suf = torch.randint(0, 64, (B, 4))
    log_w = build_log_w(w1.double(), temperature=1.0)[None].float()
    f12 = f1.float()

    img_a = torch.randn(B, P, 16)
    img_b = torch.randn(B, P, 16) * 100
    out_a = foa(pre, suf, img_a, log_w, f12).logits
    out_b = foa(pre, suf, img_b, log_w, f12).logits
    t("at step 0 the output is independent of the image",
      torch.allclose(out_a, out_b, atol=1e-6),
      f"max diff {float((out_a - out_b).detach().abs().max()):.2e}")
    t("  and the slot tokens are exactly zero",
      float(foa.slot_tokens(img_a, log_w, f12).abs().max()) == 0.0)

    with torch.no_grad():
        foa.project.weight.add_(torch.randn_like(foa.project.weight) * 0.1)
    out_c = foa(pre, suf, img_a, log_w, f12).logits
    out_d = foa(pre, suf, img_b, log_w, f12).logits
    t("after a weight change it does depend on the image",
      float((out_c - out_d).abs().max()) > 1e-3,
      f"max diff {float((out_c - out_d).abs().max()):.2e}")

    print("\ngradients")
    nn.init.zeros_(foa.project.weight); nn.init.zeros_(foa.project.bias)
    labels = torch.randint(0, 64, (B, 4))
    loss = foa(pre, suf, img_a, log_w, f12, labels=labels).loss
    loss.backward()
    gp = foa.project.weight.grad
    t("the zero-init projection still receives gradient",
      gp is not None and float(gp.abs().max()) > 0,
      f"max |grad| {float(gp.abs().max()):.2e}")
    gs = foa.slots.embedding.weight.grad
    # With a zero projection matrix the gradient *to its input* is W^T = 0, so
    # the slot pathway is necessarily dormant on step 0. This is a consequence
    # of zero-init, not a fault -- but it means freezing the projection at zero
    # would leave the slots untrainable forever, so both halves are asserted.
    t("at step 0 the slot pathway is dormant (W^T = 0)",
      gs is None or float(gs.abs().max()) == 0.0)
    opt = torch.optim.SGD(list(foa.trainable()), lr=1e-2)
    opt.step(); opt.zero_grad(set_to_none=True)
    foa(pre, suf, img_a, log_w, f12, labels=labels).loss.backward()
    gs2 = foa.slots.embedding.weight.grad
    t("after one optimiser step the slots do receive gradient",
      gs2 is not None and float(gs2.abs().max()) > 0,
      f"max |grad| {float(gs2.abs().max()):.2e}")
    gg = foa.gather.q.weight.grad
    t("  and so does the gather",
      gg is not None and float(gg.abs().max()) > 0,
      f"max |grad| {float(gg.abs().max()):.2e}")
    t("the LM is not in the trainable set",
      all(not any(p is q for q in foa.trainable()) for p in lm.parameters()))

    print("\nuniform-w ablation (the gate)")
    uni = build_log_w(torch.ones_like(w1.double()))
    t("uniform w attends everywhere, geometric w does not",
      effective_width(uni[None]) > 10 * effective_width(log_w),
      f"uniform {effective_width(uni[None]):.0f} vs geometric "
      f"{effective_width(log_w):.1f} patches")

    print("\n" + ("all self-tests passed" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
