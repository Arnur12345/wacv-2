"""
Steps 3-5: slots, the gather, and the null-space tensor.

This is the mechanism. Everything else in the project is scaffolding around one
added term in one attention layer:

    logits = QK^T / sqrt(d) + log(w)

`w` comes from the geometry kernel and says, for a point in millimetres and a
view, which patches of that view see it. Adding its log to the attention logits
means a slot at a physical location can only attend to patches that actually
image that location -- the forward operator is imposed, not learned.

  Step 3  SlotGrid       M learnable embeddings + M fixed mm coordinates
  Step 4  Gather         one cross-attention layer, biased by log(w)
  Step 5  NullSpace      per slot, eigendecompose sum_v w_v d_v d_v^T

Run `python slots.py` for the self-tests; they need no data and no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9          # a maskable stand-in for log(0); see build_log_w


# --------------------------------------------------------------------------
# Step 3 -- slots
# --------------------------------------------------------------------------


class SlotGrid(nn.Module):
    """
    An embedding table of M vectors, plus a fixed array of M mm coordinates.

    That is the whole thing. The coordinates are not learned: they are where in
    the patient each slot stands, and they are what `w` is evaluated at.
    """

    def __init__(self, grid: Tuple[int, int, int] = (6, 6, 6), dim: int = 768):
        super().__init__()
        self.grid = tuple(grid)
        self.dim = dim
        self.embedding = nn.Embedding(self.n_slots, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    @property
    def n_slots(self) -> int:
        gx, gy, gz = self.grid
        return gx * gy * gz

    def coords_mm(self, lo_mm, hi_mm, device=None) -> torch.Tensor:
        """
        [M, 3] slot centres spanning a volume's bounding box.

        Cell centres rather than corners, so no slot sits on the boundary where
        it would see half a patient.
        """
        lo = torch.as_tensor(lo_mm, dtype=torch.float64, device=device).reshape(3)
        hi = torch.as_tensor(hi_mm, dtype=torch.float64, device=device).reshape(3)
        axes = []
        for a in range(3):
            n = self.grid[a]
            t = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n
            axes.append(lo[a] + t * (hi[a] - lo[a]))
        gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    def forward(self) -> torch.Tensor:
        return self.embedding.weight


# --------------------------------------------------------------------------
# Step 4 -- w over a concatenated patch sequence, and the gather
# --------------------------------------------------------------------------


def build_w_matrix(slot_mm: torch.Tensor, views: Sequence) -> torch.Tensor:
    """
    [M, P_total] weights: how strongly each slot is seen by each patch.

    Views are concatenated in order, so patch p of view v lands at
    offset(v) + p. `w` does not depend on any network weight, so this is
    computed once per (volume, view-set) and cached -- never during training.
    """
    mats = []
    for view in views:
        sw = view.w(slot_mm)
        m = torch.zeros(slot_mm.shape[0], view.grid.n_patches, dtype=torch.float64)
        if len(sw):
            m.index_put_((sw.batch_idx, sw.patch_idx), sw.weight, accumulate=True)
        mats.append(m)
    return torch.cat(mats, dim=1)


def build_log_w(w: torch.Tensor, threshold: float = 1e-6,
                temperature: float = 1.0) -> torch.Tensor:
    """
    log(w) as an additive attention bias.

    `w` is mostly zeros, so use a large negative constant rather than a literal
    log(0): -inf produces NaN the moment a slot has no visible patch at all,
    and at least one slot always will (the corners of the grid sit outside the
    patient). A finite floor keeps softmax well-defined -- such a row simply
    comes out near-uniform over nothing, and the gate below zeroes it.

    `temperature` softens the constraint: start soft, anneal sharp. Sharp from
    step one gives dead gradients, because a slot whose w is slightly wrong can
    never recover the patches it needs.
    """
    safe = w.clamp_min(threshold)
    bias = torch.log(safe) / max(temperature, 1e-6)
    return torch.where(w > threshold, bias, torch.full_like(bias, NEG_INF))


class Gather(nn.Module):
    """
    Step 4: one cross-attention layer. Queries = slots, keys/values = frozen
    vision-tower patches from every view, concatenated.

    The only unusual part is `+ log_w` on the logits.
    """

    def __init__(self, dim: int, patch_dim: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(patch_dim, dim)
        self.v = nn.Linear(patch_dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, slots: torch.Tensor, patches: torch.Tensor,
                log_w: torch.Tensor, return_attn: bool = False):
        """
        slots   [B, M, D]
        patches [B, P, Dv]   frozen vision-tower features, views concatenated
        log_w   [B, M, P]    additive bias from the geometry
        """
        # The head is deliberately kept in fp32 while the frozen tower runs in
        # bf16: it is small, and the log-w bias plus softmax are precisely where
        # bf16 rounding would blunt the geometric constraint. Cast inputs up
        # rather than casting the head down.
        wd = self.q.weight.dtype
        slots = slots.to(wd)
        patches = patches.to(wd)
        B, M, _ = slots.shape
        P = patches.shape[1]
        q = self.q(slots).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(patches).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(patches).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        lw = log_w.to(device=logits.device, dtype=logits.dtype)
        logits = logits + lw[:, None, :, :]        # broadcast over heads
        attn = logits.softmax(dim=-1)

        ctx = (attn @ v).transpose(1, 2).reshape(B, M, -1)
        # A slot that no view can see must contribute nothing. Without this the
        # softmax over an all-masked row is uniform, and the slot would inject
        # an average of the whole image -- exactly the leak the mask exists to
        # prevent.
        visible = (lw > NEG_INF / 2).any(dim=-1, keepdim=True).to(ctx.dtype)
        ctx = ctx * visible
        out = self.norm(slots + self.out(ctx))
        return (out, attn) if return_attn else out


# --------------------------------------------------------------------------
# Step 5 -- the null-space tensor
# --------------------------------------------------------------------------


def null_space_tensor(slot_mm: torch.Tensor, views: Sequence,
                      w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Per slot, N = sum_v w_v d_v d_v^T over the ray directions that image it.

    N is rank 1 for a single view: its dominant eigenvector is the ray, and the
    two small eigenvalues span the plane the view does constrain. Two
    orthogonal views make it rank 2. The eigenvalues therefore say how many
    independent directions of evidence this slot has, and the eigenvectors say
    which -- which is precisely the geometric fact E1 measures.

    Returns [M, 3, 3].
    """
    M = slot_mm.shape[0]
    N = torch.zeros(M, 3, 3, dtype=torch.float64)
    off = 0
    for view in views:
        n_p = view.grid.n_patches
        if w is None:
            wv = torch.ones(M, dtype=torch.float64)
        else:
            wv = w[:, off:off + n_p].sum(dim=1)
        off += n_p
        d = slot_mm - view.source_mm[None, :]
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        N += wv[:, None, None] * (d[:, :, None] @ d[:, None, :])
    return N


def null_space_features(N: torch.Tensor) -> torch.Tensor:
    """
    [M, 12]: three eigenvalues, then the three eigenvectors (3 x 3 = 9).

    Sorted by eigenvalue descending so the ordering is meaningful, and each
    eigenvector sign-canonicalised (largest-magnitude component made positive).
    An eigenvector's sign is arbitrary; without fixing it the same geometry
    would present to the MLP as two different feature vectors.
    """
    vals, vecs = torch.linalg.eigh(N)                       # ascending
    order = torch.argsort(vals, dim=-1, descending=True)
    vals = torch.gather(vals, 1, order)
    vecs = torch.gather(vecs, 2, order[:, None, :].expand(-1, 3, -1))

    idx = vecs.abs().argmax(dim=1, keepdim=True)            # [M,1,3]
    sign = torch.sign(torch.gather(vecs, 1, idx))
    vecs = vecs * torch.where(sign == 0, torch.ones_like(sign), sign)

    # columns are the eigenvectors; flatten as v1, v2, v3
    flat = vecs.transpose(1, 2).reshape(N.shape[0], 9)
    return torch.cat([vals, flat], dim=1)


class NullSpaceEncoder(nn.Module):
    """
    Step 5: the conditioning tensor -> a small MLP -> added to the slot.

    The input is 12 null-space numbers PLUS the slot's own mm coordinate.

    The coordinate is not decoration. Without it a slot knows what its evidence
    cannot say (the eigen-structure) but not where it stands, so a model that
    correctly finds the target at slot m still cannot name a position in
    millimetres -- and the only strategy left is to emit the dataset mean, which
    is exactly what the first run did.
    """

    N_FEATS = 15          # 3 eigenvalues + 9 eigenvector components + 3 mm

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(self.N_FEATS, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, slots: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # geometry is computed on CPU in float64; follow the slots
        f = feats.to(device=slots.device, dtype=slots.dtype)
        if f.shape[-1] != self.N_FEATS:
            raise ValueError(f"expected {self.N_FEATS} conditioning features "
                             f"(12 null-space + 3 mm), got {f.shape[-1]}")
        return slots + self.mlp(f)[None]


def conditioning_features(N: torch.Tensor, slot_mm: torch.Tensor,
                          scale_mm: float = 100.0) -> torch.Tensor:
    """[M, 15]: the null-space 12, then the slot's position in units of 10cm."""
    return torch.cat([null_space_features(N), slot_mm / scale_mm], dim=1)


# --------------------------------------------------------------------------


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


def selftest() -> int:
    from geometry_kernel import DRRView, synthetic_thorax

    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))

    torch.manual_seed(0)
    vol = synthetic_thorax()
    pa = DRRView.standard(vol, "PA")
    lat = DRRView.standard(vol, "LAT")

    print("Step 3 -- slots")
    sg = SlotGrid(grid=(4, 4, 4), dim=64)
    lo = vol.voxel_to_mm(torch.zeros(3, dtype=torch.float64))
    hi = vol.voxel_to_mm(torch.tensor([s - 1 for s in vol.size_ijk], dtype=torch.float64))
    coords = sg.coords_mm(torch.minimum(lo, hi), torch.maximum(lo, hi))
    t("M coordinates for M slots", coords.shape == (sg.n_slots, 3), str(tuple(coords.shape)))
    t("slots lie inside the volume box",
      bool((coords >= torch.minimum(lo, hi)).all() and (coords <= torch.maximum(lo, hi)).all()))

    print("\nStep 4 -- w and the gather")
    w = build_w_matrix(coords, [pa, lat])
    t("w spans both views' patches",
      w.shape == (sg.n_slots, pa.grid.n_patches + lat.grid.n_patches),
      str(tuple(w.shape)))
    t("w is sparse", float((w > 0).float().mean()) < 0.02,
      f"{float((w > 0).float().mean()):.4%} nonzero")
    t("each slot's weight per view is at most 1",
      float(w[:, :pa.grid.n_patches].sum(1).max()) <= 1.0 + 1e-9)

    log_w = build_log_w(w)
    t("masked entries are finite (no log(0) NaN)", bool(torch.isfinite(log_w).all()))
    t("log_w matches log(w) where visible",
      torch.allclose(log_w[w > 1e-6], torch.log(w[w > 1e-6])))

    B, P, Dv = 1, w.shape[1], 32
    g = Gather(dim=64, patch_dim=Dv, n_heads=4)
    patches = torch.randn(B, P, Dv)
    slots = sg().unsqueeze(0)
    out, attn = g(slots, patches, log_w[None].float(), return_attn=True)
    t("gather returns slot-shaped output", out.shape == slots.shape, str(tuple(out.shape)))
    t("attention is zero on patches that cannot see the slot",
      float(attn.detach()[0, :, w > 1e-6].sum()) > 0 and
      float(attn.detach()[0][:, w <= 1e-6].max()) < 1e-6,
      f"max off-support attention {float(attn.detach()[0][:, w <= 1e-6].max()):.2e}")

    # a slot no view sees must contribute nothing at all
    far = torch.tensor([[1e4, 1e4, 1e4]], dtype=torch.float64)
    w_far = build_w_matrix(far, [pa, lat])
    t("an unseen slot has all-zero w", float(w_far.sum()) == 0.0)
    # The invariant is not "output is zero" -- the residual and LayerNorm give a
    # nonzero output for any input. It is that an unseen slot's output cannot
    # depend on the image, which is what a uniform-attention leak would violate.
    s_far = torch.zeros(1, 1, 64)
    lw_far = build_log_w(w_far)[None].float()
    o_a = g(s_far, torch.randn(B, P, Dv), lw_far)
    o_b = g(s_far, torch.randn(B, P, Dv) * 50, lw_far)
    t("an unseen slot's output is independent of the image (no leak)",
      torch.allclose(o_a, o_b, atol=1e-6),
      f"max diff {float((o_a - o_b).abs().max()):.2e}")
    o_seen_a = g(slots, patches, log_w[None].float())
    o_seen_b = g(slots, patches * 50, log_w[None].float())
    t("  while a visible slot's output does depend on it",
      float((o_seen_a - o_seen_b).abs().max()) > 1e-3)

    print("\nStep 5 -- the null-space tensor")
    N1 = null_space_tensor(coords, [pa])
    e1 = torch.linalg.eigvalsh(N1)
    t("one view -> rank 1 (two eigenvalues ~ 0)",
      float(e1[:, :2].abs().max()) < 1e-9 and float(e1[:, 2].min()) > 0.9,
      f"eigenvalues ~ {[round(float(x), 3) for x in e1[0]]}")

    d = (coords - pa.source_mm[None, :])
    d = d / d.norm(dim=1, keepdim=True)
    _, v1 = torch.linalg.eigh(N1)
    dom = v1[:, :, 2]
    t("  its dominant eigenvector is the ray",
      float((dom * d).sum(1).abs().min()) > 0.999,
      f"min |cos| {float((dom * d).sum(1).abs().min()):.6f}")

    N2 = null_space_tensor(coords, [pa, lat])
    e2 = torch.linalg.eigvalsh(N2)
    t("two orthogonal views -> rank 2",
      float(e2[:, 0].abs().max()) < 1e-6 and float(e2[:, 1].min()) > 0.5,
      f"eigenvalues ~ {[round(float(x), 3) for x in e2[0]]}")
    t("  the unconstrained direction shrinks when a view is added",
      float(e2[:, 0].mean()) <= float(e1[:, 0].mean()) + 1e-9)

    f12 = null_space_features(N2)
    t("features are 12 numbers per slot", f12.shape == (sg.n_slots, 12),
      str(tuple(f12.shape)))
    t("eigenvalues come out descending",
      bool((f12[:, 0] >= f12[:, 1]).all() and (f12[:, 1] >= f12[:, 2]).all()))
    t("eigenvector signs are canonical (deterministic features)",
      torch.allclose(null_space_features(N2.clone()), f12))

    f15 = conditioning_features(N2, coords)
    t("conditioning = 12 null-space + 3 mm position", f15.shape == (sg.n_slots, 15),
      str(tuple(f15.shape)))
    t("  the mm part is the slot's own coordinate",
      torch.allclose(f15[:, 12:] * 100.0, coords))
    t("  and two slots at different places differ there",
      not torch.allclose(f15[0, 12:], f15[-1, 12:]))
    enc = NullSpaceEncoder(dim=64)
    t("encoder is zero-init: starts as a no-op",
      torch.allclose(enc(slots, f15), slots))
    t("  a 12-vector is refused rather than silently broadcast",
      _raises(lambda: enc(slots, f12)))

    print("\n" + ("all self-tests passed" if ok else "SELF-TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
