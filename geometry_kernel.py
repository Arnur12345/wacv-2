"""
Step 0 -- the geometry kernel.

One module. Input: an mm coordinate and a view descriptor.
Output: a sparse list of (patch index, weight).

    from geometry_kernel import load_ct, SliceView, MIPSlabView, DRRView

    vol  = load_ct("/data/lidc/raw/LIDC-IDRI-0001/.../")
    view = DRRView.standard(vol, "PA")
    sw   = view.w(nodule_mm)          # -> SparseWeights(patch_idx, weight)

Every view type is a different `w` behind the same interface.  No neural
network, no training; the whole thing is testable by looking at pictures
(see step0_tests.py, which also asserts the picture-level facts numerically).

Conventions -- fixed here once, used everywhere, never re-derived:

  world       LPS millimetres, exactly what DICOM ImagePositionPatient gives.
              +x = patient left, +y = posterior, +z = superior.
  voxel       SimpleITK *index* order (i, j, k) = (column, row, slice).
              A @ [i,j,k,1] = mm.  numpy array is [k, j, i] = [z, y, x].
  pixel       continuous, edge-based: pixel n spans [n, n+1), centre n+0.5.
              Voxel index v therefore sits at continuous pixel u = v + 0.5.
  patch       continuous patch coordinate q = u/P - 0.5, so integer q is a
              patch *centre* and the containing patch is floor(q + 0.5).
              Flat patch index = row * n_cols + col.
  detector    pixel (0,0) at a *corner*; the detector centre is the origin of
              the (u, v) mm frame.  Pixel (c, r) centre sits at
              det_centre + (c - (W-1)/2)*su*u_hat + (r - (H-1)/2)*sv*v_hat.

The DRR renderer and the DRR projection share those descriptor fields, so a
convention error cannot desynchronise them -- it can only move both together,
which is what the overlay test catches.
"""

from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DTYPE = torch.float64          # geometry is cheap; do it in double
PATCH = 14                     # ViT patch size -- w lives in patch space
MU_WATER = 0.02                # mm^-1, ~60 keV effective
AIR_HU = -1000.0

# voxel axis (i,j,k) -> numpy array axis of an [z, y, x] volume
_VOX_TO_ARRAY = {0: 2, 1: 1, 2: 0}


# --------------------------------------------------------------------------
# 0.1  the affine
# --------------------------------------------------------------------------


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=DTYPE)


def _as_points(p) -> Tuple[torch.Tensor, bool]:
    """Accept [3] or [N,3]; return ([N,3] float64, was_single)."""
    q = _t(p)
    if q.ndim == 1:
        return q.reshape(1, 3), True
    if q.ndim != 2 or q.shape[-1] != 3:
        raise ValueError(f"points must be [3] or [N,3], got {tuple(q.shape)}")
    return q, False


@dataclass
class CTVolume:
    """A CT volume plus the one matrix that matters."""

    array: np.ndarray          # [z, y, x] Hounsfield units, float32
    A: torch.Tensor            # 4x4, voxel (i,j,k) -> mm (LPS)
    A_inv: torch.Tensor        # 4x4, mm -> voxel
    spacing: Tuple[float, float, float]   # mm per index step along i, j, k
    origin: Tuple[float, float, float]
    direction: np.ndarray      # 3x3, columns are the i/j/k world directions
    meta: Dict = field(default_factory=dict)

    _mu: Optional[torch.Tensor] = field(default=None, repr=False, compare=False)

    # -- basic geometry -----------------------------------------------------

    @property
    def size_ijk(self) -> Tuple[int, int, int]:
        z, y, x = self.array.shape
        return (x, y, z)

    @property
    def center_voxel(self) -> torch.Tensor:
        return _t([(s - 1) / 2.0 for s in self.size_ijk])

    @property
    def center_mm(self) -> torch.Tensor:
        return self.voxel_to_mm(self.center_voxel)

    def voxel_to_mm(self, v) -> torch.Tensor:
        v, single = _as_points(v)
        h = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
        out = (self.A @ h.T).T[:, :3]
        return out[0] if single else out

    def mm_to_voxel(self, p) -> torch.Tensor:
        p, single = _as_points(p)
        h = torch.cat([p, torch.ones_like(p[:, :1])], dim=-1)
        out = (self.A_inv @ h.T).T[:, :3]
        return out[0] if single else out

    def axis_unit(self, axis: int) -> torch.Tensor:
        """World-space unit vector along voxel axis `axis`."""
        col = self.A[:3, axis]
        return col / torch.linalg.norm(col)

    def contains_voxel(self, v) -> torch.Tensor:
        v, single = _as_points(v)
        hi = _t([s - 1 for s in self.size_ijk])
        ok = ((v >= 0) & (v <= hi)).all(dim=-1)
        return ok[0] if single else ok

    # -- sampling -----------------------------------------------------------

    def crop(self, lo_ijk, hi_ijk) -> "CTVolume":
        """Sub-volume in voxel index units.  The affine follows the crop."""
        lo = [max(0, int(math.floor(float(v)))) for v in lo_ijk]
        hi = [min(int(s), int(math.ceil(float(v))) + 1)
              for v, s in zip(hi_ijk, self.size_ijk)]
        if any(h <= l for l, h in zip(lo, hi)):
            raise ValueError(f"empty crop {lo} .. {hi}")
        arr = self.array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]].copy()
        A = self.A.clone()
        A[:3, 3] = self.voxel_to_mm(_t(lo))
        return CTVolume(array=arr, A=A, A_inv=_t(np.linalg.inv(A.numpy())),
                        spacing=self.spacing, origin=tuple(A[:3, 3].tolist()),
                        direction=self.direction, meta=dict(self.meta, cropped_from=lo))

    def voxel_grid_mm(self) -> torch.Tensor:
        """[Z, Y, X, 3] world mm of every voxel centre.  For small crops only."""
        nx, ny, nz = self.size_ijk
        k, j, i = torch.meshgrid(torch.arange(nz, dtype=DTYPE),
                                 torch.arange(ny, dtype=DTYPE),
                                 torch.arange(nx, dtype=DTYPE), indexing="ij")
        v = torch.stack([i, j, k], dim=-1).reshape(-1, 3)
        return self.voxel_to_mm(v).reshape(nz, ny, nx, 3)

    def mu_volume(self) -> torch.Tensor:
        """Attenuation volume [1,1,Z,Y,X] float32.  Zero padding == air."""
        if self._mu is None:
            hu = torch.as_tensor(self.array, dtype=torch.float32)
            mu = torch.clamp(hu - AIR_HU, min=0.0) / 1000.0 * MU_WATER
            self._mu = mu[None, None]
        return self._mu

    def sample_hu(self, vox_pts) -> torch.Tensor:
        """Trilinear HU at continuous voxel coords [N,3] (i,j,k). Air outside."""
        v, single = _as_points(vox_pts)
        hu = torch.as_tensor(self.array, dtype=torch.float32)[None, None]
        g = _normalised_grid(v.to(torch.float32), self.size_ijk)
        # shifted so that zero padding reads as air
        out = F.grid_sample(
            hu - AIR_HU, g.reshape(1, 1, 1, -1, 3),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).reshape(-1).to(DTYPE) + AIR_HU
        return out[0] if single else out


def _normalised_grid(v: torch.Tensor, size_ijk) -> torch.Tensor:
    """Voxel coords (i,j,k) -> grid_sample coords (x,y,z) in [-1,1]."""
    size = torch.tensor(size_ijk, dtype=v.dtype, device=v.device)
    denom = torch.clamp(size - 1, min=1)
    return 2.0 * v / denom - 1.0


def from_sitk(image, meta: Optional[Dict] = None) -> CTVolume:
    """Build a CTVolume from a SimpleITK image.  Do not assemble A by hand."""
    import SimpleITK as sitk

    origin = np.array(image.GetOrigin(), dtype=np.float64)
    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)

    A = np.eye(4, dtype=np.float64)
    A[:3, :3] = direction @ np.diag(spacing)
    A[:3, 3] = origin

    arr = sitk.GetArrayFromImage(image).astype(np.float32)  # [z, y, x]
    return CTVolume(
        array=arr,
        A=_t(A),
        A_inv=_t(np.linalg.inv(A)),
        spacing=tuple(spacing.tolist()),
        origin=tuple(origin.tolist()),
        direction=direction,
        meta=meta or {},
    )


def load_ct(dicom_dir: str, series_uid: Optional[str] = None,
            check_spacing: bool = True, tol: float = 1e-2) -> CTVolume:
    """
    Load a CT series and return the volume + affine.

    Traps handled here:
      * a folder may hold several series (and, in LIDC, stray DX radiographs)
        -- pick the CT series, or the largest, or the one you name;
      * slices are not always sorted -- SimpleITK sorts by position, and we
        re-check the sorted positions ourselves;
      * z-spacing is sometimes irregular -- derived from consecutive
        ImagePositionPatient, never from SliceThickness, and verified.
    """
    import SimpleITK as sitk

    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(dicom_dir)

    reader = sitk.ImageSeriesReader()
    uids = list(reader.GetGDCMSeriesIDs(dicom_dir))
    if not uids:
        raise RuntimeError(f"no DICOM series found under {dicom_dir}")

    candidates = []
    for uid in uids:
        files = list(reader.GetGDCMSeriesFileNames(dicom_dir, uid))
        modality = _read_tag(files[0], "0008|0060")
        candidates.append((uid, files, modality))

    if series_uid is not None:
        chosen = [c for c in candidates if c[0] == series_uid]
        if not chosen:
            raise KeyError(f"series {series_uid} not in {dicom_dir}")
    else:
        chosen = [c for c in candidates if c[2] == "CT" and len(c[1]) >= 3]
        if not chosen:
            found = {c[2]: len(c[1]) for c in candidates}
            raise RuntimeError(
                f"{dicom_dir} holds no multi-slice CT series (modalities: {found}). "
                "A folder of DX radiographs is not a volume -- an affine built "
                "from it is the identity and every round-trip test passes vacuously."
            )
    uid, files, modality = max(chosen, key=lambda c: len(c[1]))

    positions = np.array([_position(f) for f in files], dtype=np.float64)
    normal = _slice_normal(files[0])
    proj = positions @ normal
    order = np.argsort(proj)
    files = [files[i] for i in order]
    proj = proj[order]

    steps = np.diff(proj)
    if check_spacing and len(steps):
        spread = float(steps.max() - steps.min())
        if spread > tol:
            raise ValueError(
                f"irregular slice spacing in {uid}: steps range "
                f"[{steps.min():.4f}, {steps.max():.4f}] mm (spread {spread:.4f}). "
                "Resample before using a single affine."
            )

    reader.SetFileNames(files)
    image = reader.Execute()

    meta = dict(
        series_uid=uid,
        modality=modality,
        n_slices=len(files),
        slice_step_mm=float(np.median(steps)) if len(steps) else float("nan"),
        slice_thickness_tag=_read_tag(files[0], "0018|0050"),
        files=files,
        source=dicom_dir,
    )
    return from_sitk(image, meta)


def _read_tag(path: str, tag: str) -> str:
    import SimpleITK as sitk

    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return r.GetMetaData(tag).strip() if r.HasMetaDataKey(tag) else ""


def _position(path: str) -> np.ndarray:
    v = _read_tag(path, "0020|0032")
    return np.array([float(x) for x in v.split("\\")], dtype=np.float64)


def _slice_normal(path: str) -> np.ndarray:
    v = _read_tag(path, "0020|0037")
    if not v:
        return np.array([0.0, 0.0, 1.0])
    o = np.array([float(x) for x in v.split("\\")], dtype=np.float64)
    n = np.cross(o[:3], o[3:])
    return n / np.linalg.norm(n)


# --------------------------------------------------------------------------
# patch space -- get this right once and reuse it
# --------------------------------------------------------------------------


@dataclass
class PatchGrid:
    """
    The pixel -> patch mapping, including the resize the tower applies.

    source_hw is the native image (CT slice, detector).  model_hw is what the
    vision tower actually sees; it must be a multiple of `patch`.  A point is
    mapped native pixel -> model pixel -> continuous patch coordinate.
    """

    source_hw: Tuple[int, int]
    model_hw: Tuple[int, int]
    patch: int = PATCH

    def __post_init__(self):
        for s in self.model_hw:
            if s % self.patch:
                raise ValueError(f"model_hw {self.model_hw} not a multiple of {self.patch}")

    @classmethod
    def from_source(cls, source_hw, patch: int = PATCH, model_hw=None) -> "PatchGrid":
        if model_hw is None:
            model_hw = tuple(max(patch, int(round(s / patch)) * patch) for s in source_hw)
        return cls(tuple(int(s) for s in source_hw), tuple(int(s) for s in model_hw), patch)

    @property
    def n_rows(self) -> int:
        return self.model_hw[0] // self.patch

    @property
    def n_cols(self) -> int:
        return self.model_hw[1] // self.patch

    @property
    def n_patches(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def scale(self) -> Tuple[float, float]:
        return (self.model_hw[0] / self.source_hw[0], self.model_hw[1] / self.source_hw[1])

    def pixel_to_patch_coord(self, u_col: torch.Tensor, u_row: torch.Tensor):
        """Continuous source pixels -> continuous patch coords (integer = centre)."""
        sr, sc = self.scale
        qc = (u_col * sc) / self.patch - 0.5
        qr = (u_row * sr) / self.patch - 0.5
        return qc, qr

    def containing_patch(self, u_col, u_row):
        """Flat index of the patch a point falls in.  A point exactly on a
        patch boundary belongs to the higher-index patch, and its weight ties
        with the neighbour it sits between -- that tie is correct, so never
        assert 'the top patch is the containing one' without allowing it."""
        qc, qr = self.pixel_to_patch_coord(_t(u_col).reshape(-1), _t(u_row).reshape(-1))
        c = torch.floor(qc + 0.5).long()
        r = torch.floor(qr + 0.5).long()
        return (r * self.n_cols + c)

    def patch_bbox_source(self, flat_idx: int) -> Tuple[float, float, float, float]:
        """Flat patch index -> (col0, row0, col1, row1) in *source* pixels."""
        r, c = divmod(int(flat_idx), self.n_cols)
        sr, sc = self.scale
        return (c * self.patch / sc, r * self.patch / sr,
                (c + 1) * self.patch / sc, (r + 1) * self.patch / sr)

    def splat(self, qc: torch.Tensor, qr: torch.Tensor,
              sigma: float = 0.5, radius: Optional[int] = None):
        """
        Spread each point over the patch footprint.

        Returns (patch_idx [N,K], weight [N,K], valid [N,K]).  Weight is
        normalised by the *full* Gaussian mass, so a point that falls partly
        outside the image keeps a total weight < 1 -- partial visibility is
        information, not something to renormalise away.
        """
        n = qc.shape[0]
        if sigma <= 0:
            c = torch.floor(qc + 0.5).long().reshape(n, 1)
            r = torch.floor(qr + 0.5).long().reshape(n, 1)
            wgt = torch.ones(n, 1, dtype=DTYPE)
        else:
            if radius is None:
                radius = max(1, int(math.ceil(2 * sigma)))
            off = torch.arange(-radius, radius + 1)
            k = off.numel()
            c0 = torch.floor(qc + 0.5).long()[:, None]
            r0 = torch.floor(qr + 0.5).long()[:, None]
            c = (c0 + off[None, :])[:, None, :].expand(n, k, k).reshape(n, k * k)
            r = (r0 + off[None, :])[:, :, None].expand(n, k, k).reshape(n, k * k)
            d2 = (c.to(DTYPE) - qc[:, None]) ** 2 + (r.to(DTYPE) - qr[:, None]) ** 2
            wgt = torch.exp(-d2 / (2 * sigma * sigma))
            wgt = wgt / torch.clamp(wgt.sum(dim=1, keepdim=True), min=1e-12)

        valid = (c >= 0) & (c < self.n_cols) & (r >= 0) & (r < self.n_rows)
        idx = torch.where(valid, r * self.n_cols + c, torch.zeros_like(c))
        return idx, wgt * valid.to(DTYPE), valid


@dataclass
class SparseWeights:
    """The output of w: a sparse list of (patch index, weight)."""

    patch_idx: torch.Tensor    # [K] long
    weight: torch.Tensor       # [K] float64
    batch_idx: torch.Tensor    # [K] long -- which input point
    grid: PatchGrid
    view_name: str = ""
    n_points: int = 1

    def __len__(self) -> int:
        return int(self.patch_idx.numel())

    @property
    def total(self) -> torch.Tensor:
        """Total weight per input point -- 0 means 'this view cannot see it'."""
        out = torch.zeros(self.n_points, dtype=DTYPE)
        return out.index_add_(0, self.batch_idx, self.weight)

    def for_point(self, i: int) -> "SparseWeights":
        m = self.batch_idx == i
        return SparseWeights(self.patch_idx[m], self.weight[m],
                             torch.zeros(int(m.sum()), dtype=torch.long),
                             self.grid, self.view_name, 1)

    def dense(self) -> torch.Tensor:
        """[n_points, n_rows, n_cols] weight map."""
        d = torch.zeros(self.n_points * self.grid.n_patches, dtype=DTYPE)
        d.index_add_(0, self.batch_idx * self.grid.n_patches + self.patch_idx, self.weight)
        return d.reshape(self.n_points, self.grid.n_rows, self.grid.n_cols)

    def top(self, k: int = 1, point: int = 0):
        """[(flat_patch_idx, weight), ...] sorted by weight for one point."""
        sw = self.for_point(point) if self.n_points > 1 else self
        if len(sw) == 0:
            return []
        order = torch.argsort(sw.weight, descending=True)[:k]
        return [(int(sw.patch_idx[i]), float(sw.weight[i])) for i in order]

    def as_list(self):
        return list(zip(self.patch_idx.tolist(), self.weight.tolist()))

    @staticmethod
    def empty(grid: PatchGrid, name: str = "", n_points: int = 1) -> "SparseWeights":
        z = torch.zeros(0, dtype=torch.long)
        return SparseWeights(z, torch.zeros(0, dtype=DTYPE), z, grid, name, n_points)


def _pack(idx, wgt, valid, grid, name, n_points, drop_below=1e-8) -> SparseWeights:
    n, k = idx.shape
    batch = torch.arange(n)[:, None].expand(n, k)
    keep = valid & (wgt > drop_below)
    return SparseWeights(idx[keep], wgt[keep], batch[keep], grid, name, n_points)


# --------------------------------------------------------------------------
# the common interface
# --------------------------------------------------------------------------


class View:
    """A view descriptor.  The only thing every view must do is answer `w`."""

    name: str = "view"
    grid: PatchGrid

    def w(self, points_mm) -> SparseWeights:      # pragma: no cover - interface
        raise NotImplementedError

    def render(self) -> np.ndarray:               # pragma: no cover - interface
        raise NotImplementedError


def w(points_mm, view: View) -> SparseWeights:
    """The function this whole step exists to produce."""
    return view.w(points_mm)


# --------------------------------------------------------------------------
# 0.3  w for the easy views -- slices, reformats, MIP slabs
# --------------------------------------------------------------------------

# axis -> (row_axis, col_axis, flip_row, flip_col) in voxel index space
_PLANE = {
    2: (1, 0, False, False),   # axial:    rows = y(j), cols = x(i)
    1: (2, 0, True, False),    # coronal:  rows = z(k) flipped so head is up
    0: (2, 1, True, False),    # sagittal: rows = z(k) flipped, cols = y(j)
}
_AXIS_NAME = {0: "sagittal", 1: "coronal", 2: "axial"}


@dataclass
class SliceView(View):
    """
    An axis-aligned slice (or slab) of the CT.  Reformats are the same
    function with the axes permuted -- that is literally all `axis` does.
    """

    volume: CTVolume
    axis: int = 2
    index: float = 0.0             # slab centre, in voxel index units
    thickness_mm: float = 2.5
    sigma_patch: float = 0.5
    model_hw: Optional[Tuple[int, int]] = None
    patch: int = PATCH

    def __post_init__(self):
        if self.axis not in _PLANE:
            raise ValueError("axis must be 0, 1 or 2")
        self.row_axis, self.col_axis, self.flip_row, self.flip_col = _PLANE[self.axis]
        size = self.volume.size_ijk
        self.source_hw = (size[self.row_axis], size[self.col_axis])
        self.grid = PatchGrid.from_source(self.source_hw, self.patch, self.model_hw)
        self.name = f"{_AXIS_NAME[self.axis]}@{self.position_mm:.1f}mm"

    # -- construction -------------------------------------------------------

    @classmethod
    def at_mm(cls, volume: CTVolume, axis: int, position_mm: float,
              thickness_mm: float = 2.5, **kw) -> "SliceView":
        """`position_mm` is measured along the world axis that `axis` maps to."""
        c = volume.center_voxel.clone()
        p0 = volume.voxel_to_mm(c)
        n = volume.axis_unit(axis)
        step = float(volume.spacing[axis])
        # how far, in index units, to move so that (p . n_world_axis) == position
        world_axis = int(torch.argmax(torch.abs(n)))
        delta_mm = position_mm - float(p0[world_axis])
        idx = float(c[axis]) + delta_mm / (step * float(n[world_axis]))
        return cls(volume, axis=axis, index=idx, thickness_mm=thickness_mm, **kw)

    @classmethod
    def through_point(cls, volume: CTVolume, axis: int, point_mm,
                      thickness_mm: float = 2.5, **kw) -> "SliceView":
        v = volume.mm_to_voxel(point_mm)
        return cls(volume, axis=axis, index=float(v[axis]),
                   thickness_mm=thickness_mm, **kw)

    # -- geometry -----------------------------------------------------------

    @property
    def normal(self) -> torch.Tensor:
        return self.volume.axis_unit(self.axis)

    @property
    def center_mm(self) -> torch.Tensor:
        c = self.volume.center_voxel.clone()
        c[self.axis] = self.index
        return self.volume.voxel_to_mm(c)

    @property
    def position_mm(self) -> float:
        n = self.normal
        return float(torch.dot(self.center_mm, n / torch.linalg.norm(n)))

    def signed_depth(self, points_mm) -> torch.Tensor:
        """Signed mm distance from the slab centre plane."""
        p, single = _as_points(points_mm)
        d = (p - self.center_mm[None, :]) @ self.normal
        return d[0] if single else d

    def in_plane_pixels(self, points_mm):
        """-> (u_col, u_row) continuous source pixels, matching render()."""
        p, _ = _as_points(points_mm)
        v = self.volume.mm_to_voxel(p)
        u_col = v[:, self.col_axis] + 0.5
        u_row = v[:, self.row_axis] + 0.5
        if self.flip_col:
            u_col = self.source_hw[1] - u_col
        if self.flip_row:
            u_row = self.source_hw[0] - u_row
        return u_col, u_row

    # -- w ------------------------------------------------------------------

    def depth_weight(self, points_mm) -> torch.Tensor:
        """The rect: 1 inside the slab, 0 outside."""
        d = self.signed_depth(points_mm)
        d = d.reshape(-1)
        return (d.abs() < self.thickness_mm / 2.0).to(DTYPE)

    def w(self, points_mm) -> SparseWeights:
        p, _ = _as_points(points_mm)
        n = p.shape[0]
        wz = self.depth_weight(p)
        u_col, u_row = self.in_plane_pixels(p)
        qc, qr = self.grid.pixel_to_patch_coord(u_col, u_row)
        idx, wgt, valid = self.grid.splat(qc, qr, self.sigma_patch)
        return _pack(idx, wgt * wz[:, None], valid, self.grid, self.name, n)

    # -- picture ------------------------------------------------------------

    def _slab_bounds(self) -> Tuple[int, int]:
        half = self.thickness_mm / 2.0 / float(self.volume.spacing[self.axis])
        lo = int(math.ceil(self.index - half - 1e-9))
        hi = int(math.floor(self.index + half + 1e-9))
        n = self.volume.size_ijk[self.axis]
        lo = max(0, min(n - 1, lo))
        hi = max(lo, min(n - 1, hi))
        if hi < lo:
            lo = hi = int(round(self.index))
        return lo, hi

    def _reduce(self, slab: np.ndarray, ax: int) -> np.ndarray:
        return slab.mean(axis=ax)

    def render(self) -> np.ndarray:
        lo, hi = self._slab_bounds()
        ax = _VOX_TO_ARRAY[self.axis]
        slab = np.take(self.volume.array, range(lo, hi + 1), axis=ax)
        img = self._reduce(slab, ax)
        # remaining numpy axes are already (row_axis, col_axis) by construction
        if self.flip_row:
            img = img[::-1]
        if self.flip_col:
            img = img[:, ::-1]
        return np.ascontiguousarray(img)


@dataclass
class MIPSlabView(SliceView):
    """
    Same interface, softer rect: the depth weight is a soft-max over the
    segment, so a point only counts to the degree it is what the MIP shows.
    """

    tau_hu: float = 60.0           # softmax temperature, in HU
    n_depth: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        self.name = f"mip-{_AXIS_NAME[self.axis]}@{self.position_mm:.1f}mm"

    def _depth_samples(self) -> int:
        if self.n_depth:
            return int(self.n_depth)
        step = float(self.volume.spacing[self.axis])
        return max(2, int(round(self.thickness_mm / step)) + 1)

    def depth_weight(self, points_mm) -> torch.Tensor:
        p, _ = _as_points(points_mm)
        n_pts = p.shape[0]
        v = self.volume.mm_to_voxel(p)
        m = self._depth_samples()
        half = self.thickness_mm / 2.0 / float(self.volume.spacing[self.axis])
        offs = torch.linspace(-half, half, m, dtype=DTYPE)

        probe = v[:, None, :].repeat(1, m, 1)
        probe[:, :, self.axis] = self.index + offs[None, :]
        hu = self.volume.sample_hu(probe.reshape(-1, 3)).reshape(n_pts, m)
        # A soft argmax, not a soft share: wz = exp((I(p) - max_I) / tau).
        # It answers "is this point what the MIP displays?", it is 1 when the
        # point owns the maximum and ~0 when something denser is in front of
        # it, and -- unlike a normalised softmax share -- it does not depend on
        # how many depth samples we happen to take.
        soft = torch.exp((hu - hu.max(dim=1, keepdim=True).values) / self.tau_hu)

        # where does the point itself sit along the segment?
        d = (v[:, self.axis] - self.index) / max(half, 1e-9)      # in [-1, 1]
        inside = d.abs() <= 1.0
        pos = (d.clamp(-1, 1) + 1) / 2 * (m - 1)
        lo = pos.floor().long().clamp(0, m - 1)
        hi = (lo + 1).clamp(0, m - 1)
        f = (pos - lo.to(DTYPE))
        gather = lambda t: soft.gather(1, t[:, None]).squeeze(1)
        wz = gather(lo) * (1 - f) + gather(hi) * f
        return wz * inside.to(DTYPE)

    def _reduce(self, slab: np.ndarray, ax: int) -> np.ndarray:
        return slab.max(axis=ax)


# --------------------------------------------------------------------------
# 0.4  the DRR ray
# --------------------------------------------------------------------------


@dataclass
class DRRView(View):
    """
    A cone-beam DRR.  Source, detector plane and detector spacing are fields
    of this object, and `render` and `project` both read them -- so the
    projection cannot disagree with the renderer about conventions.
    """

    volume: CTVolume
    source_mm: torch.Tensor        # [3]
    det_center_mm: torch.Tensor    # [3]
    det_u: torch.Tensor            # [3] unit, detector +column direction
    det_v: torch.Tensor            # [3] unit, detector +row direction
    det_spacing: Tuple[float, float] = (0.5, 0.5)   # (mm/col, mm/row)
    det_size: Tuple[int, int] = (504, 504)          # (W cols, H rows)
    sigma_patch: float = 0.6
    model_hw: Optional[Tuple[int, int]] = None
    patch: int = PATCH
    name: str = "drr"

    def __post_init__(self):
        self.source_mm = _t(self.source_mm).reshape(3)
        self.det_center_mm = _t(self.det_center_mm).reshape(3)
        self.det_u = _unit(_t(self.det_u).reshape(3))
        self.det_v = _unit(_t(self.det_v).reshape(3))
        W, H = self.det_size
        self.source_hw = (int(H), int(W))
        self.grid = PatchGrid.from_source(self.source_hw, self.patch, self.model_hw)

    # -- construction -------------------------------------------------------

    @classmethod
    def standard(cls, volume: CTVolume, orientation: str = "PA",
                 sdd: float = 1800.0, sod: float = 1500.0,
                 det_size: Tuple[int, int] = (504, 504),
                 det_spacing: Optional[Tuple[float, float]] = None,
                 isocenter_mm=None, margin: float = 1.10, **kw) -> "DRRView":
        """
        A textbook chest projection in LPS.

          PA   beam runs posterior -> anterior; image +col = patient left,
               image +row = inferior.
          LAT  beam runs right -> left; image +col = posterior, +row = inferior.

        sdd / sod default to the LIDC DX headers (1800 mm source-to-detector),
        with the isocentre at 1500 mm so the detector clears the patient.
        """
        iso = volume.center_mm if isocenter_mm is None else _t(isocenter_mm)
        o = orientation.upper()
        if o in ("PA", "AP", "FRONTAL"):
            beam = _t([0.0, -1.0, 0.0]) if o != "AP" else _t([0.0, 1.0, 0.0])
            du, dv = _t([1.0, 0.0, 0.0]), _t([0.0, 0.0, -1.0])
        elif o in ("LAT", "LATERAL", "LL"):
            beam = _t([1.0, 0.0, 0.0])
            du, dv = _t([0.0, 1.0, 0.0]), _t([0.0, 0.0, -1.0])
        else:
            raise ValueError(f"unknown orientation {orientation!r}")

        source = iso - beam * sod
        det_center = iso + beam * (sdd - sod)

        view = cls(volume, source, det_center, du, dv,
                   det_spacing=det_spacing or (0.5, 0.5),
                   det_size=det_size, name=f"drr-{o.lower()}", **kw)
        if det_spacing is None:
            view.det_spacing = view._fit_spacing(margin)
        return view

    def _fit_spacing(self, margin: float) -> Tuple[float, float]:
        """Pick mm/pixel so the whole volume projects inside the detector."""
        nx, ny, nz = self.volume.size_ijk
        corners = torch.tensor(
            [[i, j, k] for i in (0, nx - 1) for j in (0, ny - 1) for k in (0, nz - 1)],
            dtype=DTYPE)
        pts = self.volume.voxel_to_mm(corners)
        a, b, _, valid = self._detector_mm(pts)
        a, b = a[valid], b[valid]
        W, H = self.det_size
        su = float(2 * a.abs().max()) * margin / W
        sv = float(2 * b.abs().max()) * margin / H
        s = max(su, sv, 1e-3)
        return (s, s)

    # -- geometry -----------------------------------------------------------

    @property
    def det_normal(self) -> torch.Tensor:
        return _unit(torch.linalg.cross(self.det_u, self.det_v))

    def _detector_mm(self, points_mm):
        """Ray from source through p, hit the detector plane, return (a, b, lam, valid)."""
        p, _ = _as_points(points_mm)
        n = self.det_normal
        d = p - self.source_mm[None, :]
        denom = d @ n
        num = torch.dot(self.det_center_mm - self.source_mm, n)
        safe = denom.abs() > 1e-12
        lam = torch.where(safe, num / torch.where(safe, denom, torch.ones_like(denom)),
                          torch.zeros_like(denom))
        hit = self.source_mm[None, :] + lam[:, None] * d
        rel = hit - self.det_center_mm[None, :]
        a = rel @ self.det_u
        b = rel @ self.det_v
        # lam > 1 means the point is between source and detector: the normal case.
        valid = safe & (lam > 0)
        return a, b, lam, valid

    def project(self, points_mm):
        """
        mm -> continuous detector pixels.

        Returns (u_col, u_row, valid).  u_col/u_row are edge-based continuous
        pixels, so pixel (c, r) covers [c, c+1) x [r, r+1).
        """
        a, b, _, valid = self._detector_mm(points_mm)
        su, sv = self.det_spacing
        W, H = self.det_size
        u_col = a / su + W / 2.0
        u_row = b / sv + H / 2.0
        inside = (u_col >= 0) & (u_col < W) & (u_row >= 0) & (u_row < H)
        return u_col, u_row, valid & inside

    def ray_through(self, points_mm):
        """(origin, unit direction) of the ray that images each point."""
        p, single = _as_points(points_mm)
        d = _unit_rows(p - self.source_mm[None, :])
        o = self.source_mm[None, :].expand_as(d)
        return (o[0], d[0]) if single else (o, d)

    # -- w ------------------------------------------------------------------

    def w(self, points_mm) -> SparseWeights:
        """
        A point maps to essentially one detector pixel; the Gaussian over
        neighbouring patches keeps the gradient from being a delta function.
        """
        p, _ = _as_points(points_mm)
        n = p.shape[0]
        u_col, u_row, valid = self.project(p)
        qc, qr = self.grid.pixel_to_patch_coord(u_col, u_row)
        idx, wgt, ok = self.grid.splat(qc, qr, self.sigma_patch)
        return _pack(idx, wgt * valid[:, None].to(DTYPE), ok, self.grid, self.name, n)

    # -- picture ------------------------------------------------------------

    def detector_points_mm(self) -> torch.Tensor:
        """[H, W, 3] mm position of every detector pixel centre."""
        W, H = self.det_size
        su, sv = self.det_spacing
        a = (torch.arange(W, dtype=DTYPE) - (W - 1) / 2.0) * su
        b = (torch.arange(H, dtype=DTYPE) - (H - 1) / 2.0) * sv
        return (self.det_center_mm[None, None, :]
                + a[None, :, None] * self.det_u[None, None, :]
                + b[:, None, None] * self.det_v[None, None, :])

    def render(self, n_samples: int = 256, chunk_rows: int = 16) -> np.ndarray:
        """
        Line integral of attenuation, [H, W] float32.  Higher = denser, so a
        plain grey colormap gives bone-white the way a radiograph looks.

        Sampling happens in *voxel* space, which makes the volume bounding-box
        test exact for any direction cosine matrix, not just axis-aligned ones.
        """
        W, H = self.det_size
        size = self.volume.size_ijk
        mu = self.volume.mu_volume()

        R_inv = self.volume.A_inv[:3, :3]
        s_vox = self.volume.mm_to_voxel(self.source_mm)
        det = self.detector_points_mm()                       # [H, W, 3]
        lo = torch.zeros(3, dtype=DTYPE)
        hi = torch.tensor([s - 1 for s in size], dtype=DTYPE)
        tt = torch.linspace(0.0, 1.0, n_samples + 1, dtype=DTYPE)
        tmid = ((tt[:-1] + tt[1:]) / 2)[None, :]

        out = torch.zeros(H, W, dtype=torch.float32)
        for r0 in range(0, H, chunk_rows):
            r1 = min(H, r0 + chunk_rows)
            dm = (det[r0:r1] - self.source_mm).reshape(-1, 3)          # [M,3] mm
            length = torch.linalg.norm(dm, dim=1)
            dv = (R_inv @ dm.T).T                                      # [M,3] voxel

            inv = torch.where(dv.abs() > 1e-12, 1.0 / torch.where(dv.abs() > 1e-12, dv,
                              torch.ones_like(dv)), torch.full_like(dv, float("inf")))
            t0 = (lo[None, :] - s_vox[None, :]) * inv
            t1 = (hi[None, :] - s_vox[None, :]) * inv
            near = torch.maximum(torch.minimum(t0, t1), torch.zeros_like(t0)).max(dim=1).values
            far = torch.minimum(torch.maximum(t0, t1), torch.ones_like(t1)).min(dim=1).values
            hitmask = far > near
            near = torch.where(hitmask, near, torch.zeros_like(near))
            far = torch.where(hitmask, far, torch.zeros_like(far))

            lam = near[:, None] + (far - near)[:, None] * tmid          # [M,N]
            pts = s_vox[None, None, :] + lam[:, :, None] * dv[:, None, :]
            g = _normalised_grid(pts.reshape(1, 1, -1, n_samples, 3).to(torch.float32), size)
            vals = F.grid_sample(mu, g, mode="bilinear",
                                 padding_mode="zeros", align_corners=True)
            vals = vals.reshape(-1, n_samples).to(DTYPE)
            step = (far - near) * length / n_samples                   # mm per sample
            out[r0:r1] = (vals.sum(dim=1) * step).reshape(r1 - r0, W).to(torch.float32)
        return out.numpy()


    def with_volume(self, volume: CTVolume) -> "DRRView":
        """Same imaging geometry, different volume.  Nothing about the source,
        detector or spacing is recomputed -- that is the point."""
        return DRRView(volume, self.source_mm, self.det_center_mm,
                       self.det_u, self.det_v, self.det_spacing, self.det_size,
                       self.sigma_patch, self.model_hw, self.patch, self.name)


def nodule_difference_volume(volume: CTVolume, center_mm, radius_mm: float,
                             pad: float = 4.0) -> Tuple[CTVolume, float]:
    """
    The volume you would have to subtract to delete a nodule.

    Attenuation integrates linearly along a ray, so

        DRR(with nodule) - DRR(without) == DRR(this volume)

    which turns the day-2 gate -- "the circle sits on the nodule's shadow" --
    into a single cheap render of a small crop, on real data, for a nodule of
    any size.  The sphere is filled with the median HU of the shell around it,
    so "without" means "replaced by whatever surrounds it", not "replaced by air".

    Returns (difference volume, fill HU).
    """
    c = _t(center_mm).reshape(3)
    v = volume.mm_to_voxel(c)
    half = [radius_mm * pad / s for s in volume.spacing]
    sub = volume.crop([float(v[i]) - half[i] for i in range(3)],
                      [float(v[i]) + half[i] for i in range(3)])

    d = torch.linalg.norm(sub.voxel_grid_mm() - c, dim=-1).numpy()
    inside = d <= radius_mm
    shell = (d > radius_mm) & (d <= 2.0 * radius_mm)
    if not inside.any():
        raise ValueError("nodule sphere fell outside the crop")
    fill = float(np.median(sub.array[shell])) if shell.any() else AIR_HU

    diff = np.full_like(sub.array, AIR_HU)
    diff[inside] = AIR_HU + np.clip(sub.array[inside] - fill, 0.0, None)
    return CTVolume(array=diff, A=sub.A, A_inv=sub.A_inv, spacing=sub.spacing,
                    origin=sub.origin, direction=sub.direction,
                    meta=dict(difference_volume=True, fill_hu=fill,
                              radius_mm=radius_mm)), fill


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v)


def _unit_rows(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-12)


# --------------------------------------------------------------------------
# 0.5  the sanity suite, as reusable functions
# --------------------------------------------------------------------------


def ray_consistency(view: DRRView, point_mm, fracs=(0.35, 0.6, 0.9)):
    """
    Points on the same ray at different depths must land on the same pixel.

    Returns (max_pixel_disagreement, list of (u_col, u_row)).  Zero here is
    the null space of this view, measured rather than assumed: those points
    are genuinely indistinguishable from here.
    """
    p = _t(point_mm).reshape(3)
    s = view.source_mm
    d = p - s
    pts = torch.stack([s + f * d for f in fracs])
    u, v, ok = view.project(pts)
    uv = torch.stack([u, v], dim=1)
    spread = float((uv - uv.mean(dim=0, keepdim=True)).abs().max())
    return spread, uv


def triangulate(views: Sequence[DRRView], pixels: Sequence[Tuple[float, float]]) -> torch.Tensor:
    """
    Least-squares intersection of the back-projected rays.

    Each pixel is un-projected to a ray; the returned point minimises the sum
    of squared distances to all of them.  This is the geometry E1 depends on.
    """
    A = torch.zeros(3, 3, dtype=DTYPE)
    b = torch.zeros(3, dtype=DTYPE)
    I = torch.eye(3, dtype=DTYPE)
    for view, (u_col, u_row) in zip(views, pixels):
        su, sv = view.det_spacing
        W, H = view.det_size
        a_mm = (float(u_col) - W / 2.0) * su
        b_mm = (float(u_row) - H / 2.0) * sv
        hit = view.det_center_mm + a_mm * view.det_u + b_mm * view.det_v
        d = _unit(hit - view.source_mm)
        M = I - torch.outer(d, d)
        A += M
        b += M @ view.source_mm
    return torch.linalg.solve(A, b)


def slices_seeing(volume: CTVolume, point_mm, axis: int = 2,
                  thickness_mm: float = 2.5, **kw) -> List[int]:
    """
    Which slices along `axis` give this point nonzero weight?

    Exactly one for contiguous slices, two if they overlap.  Ten means the
    thickness is in voxels where it should be in mm.
    """
    hits = []
    for i in range(volume.size_ijk[axis]):
        v = SliceView(volume, axis=axis, index=float(i),
                      thickness_mm=thickness_mm, **kw)
        if float(v.w(point_mm).total.sum()) > 0:
            hits.append(i)
    return hits


# --------------------------------------------------------------------------
# a phantom, so step 0 is testable before the CT download finishes
# --------------------------------------------------------------------------


def synthetic_thorax(size_ijk: Tuple[int, int, int] = (384, 384, 160),
                     spacing: Tuple[float, float, float] = (0.9, 0.9, 1.25),
                     nodule_mm: Optional[Sequence[float]] = None,
                     nodule_radius_mm: float = 5.0,
                     nodule_hu: float = 60.0,
                     with_nodule: bool = True) -> CTVolume:
    """
    A crude LPS thorax with one solid nodule in the left lung.

    Not a substitute for LIDC -- it exists so that 0.2 through 0.5 have a
    ground-truth mm coordinate that is known exactly, which turns every
    "look at the picture" test into an assertion as well.
    """
    nx, ny, nz = size_ijk
    sx, sy, sz = spacing
    origin = np.array([-(nx - 1) * sx / 2, -(ny - 1) * sy / 2, -(nz - 1) * sz / 2 - 150.0])

    A = np.eye(4)
    A[:3, :3] = np.diag(spacing)
    A[:3, 3] = origin

    x = origin[0] + np.arange(nx) * sx          # patient left +
    y = origin[1] + np.arange(ny) * sy          # posterior +
    z = origin[2] + np.arange(nz) * sz          # superior +
    X, Y = np.meshgrid(x, y, indexing="xy")     # -> [ny, nx] = [row=y, col=x]

    if nodule_mm is None:
        nodule_mm = (58.0, -18.0, float(z[nz // 2] + 8.0))
    nodule_mm = tuple(float(c) for c in nodule_mm)

    body = ((X / 170.0) ** 2 + (Y / 112.0) ** 2) <= 1.0
    lung_l = (((X - 62.0) / 52.0) ** 2 + ((Y + 8.0) / 78.0) ** 2) <= 1.0
    lung_r = (((X + 62.0) / 52.0) ** 2 + ((Y + 8.0) / 78.0) ** 2) <= 1.0
    spine = ((X / 19.0) ** 2 + ((Y - 86.0) / 21.0) ** 2) <= 1.0
    shell = np.abs(np.sqrt((X / 150.0) ** 2 + ((Y - 4.0) / 100.0) ** 2) - 1.0) < 0.035

    vol = np.empty((nz, ny, nx), dtype=np.float32)
    for k, zk in enumerate(z):
        s = np.full((ny, nx), AIR_HU, dtype=np.float32)
        s[body] = 40.0                                        # soft tissue
        lung_z = abs(zk - (z[nz // 2] + 10.0)) < 95.0
        if lung_z:
            taper = 1.0 - (abs(zk - (z[nz // 2] + 10.0)) / 95.0) ** 3
            core = (lung_l | lung_r) & body
            if taper > 0.15:
                s[core] = -820.0
        s[spine & body] = 480.0                               # vertebral column
        if math.sin(zk / 11.0) > 0.55:                        # a few ribs
            s[shell & body] = 420.0
        vol[k] = s

    if with_nodule:
        cx, cy, cz = nodule_mm
        kz = np.abs(z - cz) <= nodule_radius_mm
        R2 = (X - cx) ** 2 + (Y - cy) ** 2
        for k in np.nonzero(kz)[0]:
            rr = nodule_radius_mm ** 2 - (z[k] - cz) ** 2
            if rr > 0:
                vol[k][R2 <= rr] = nodule_hu

    meta = dict(synthetic=True, nodule_mm=nodule_mm,
                nodule_radius_mm=nodule_radius_mm, nodule_hu=nodule_hu,
                with_nodule=with_nodule)
    return CTVolume(array=vol, A=_t(A), A_inv=_t(np.linalg.inv(A)),
                    spacing=tuple(spacing), origin=tuple(origin.tolist()),
                    direction=np.eye(3), meta=meta)


# --------------------------------------------------------------------------
# LIDC XML -- CT contours (0.2) and the CXR reads that ship in this repo
# --------------------------------------------------------------------------

# LIDC ships two schemas with two different namespaces: the CT reads use
# http://www.nih.gov and the CXR reads http://www.nih.gov/idri.  Matching on
# the namespaced tag silently returns zero nodules for the other one, so match
# on the local name and be done with it.


def _local(elem) -> str:
    return elem.tag.rsplit("}", 1)[-1]


def _iter_local(root, name):
    for e in root.iter():
        if _local(e) == name:
            yield e


def _child_text(elem, name, default=None):
    for e in elem:
        if _local(e) == name:
            return e.text
    return default


def parse_lidc_ct_xml(path: str) -> List[Dict]:
    """
    Nodules from a LIDC CT reading session.

    Contour points come out as (x_pixel, y_pixel, z_mm): the XML gives the
    in-plane coordinates in *pixels* and the slice position in mm, so the
    z coordinate is already world and only x/y need the affine.  Use
    `contour_to_mm` to finish the conversion.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    out: List[Dict] = []
    for si, sess in enumerate(_iter_local(root, "readingSession")):
        for read in _iter_local(sess, "unblindedReadNodule"):
            nid = (_child_text(read, "noduleID", "") or "").strip()
            pts = []
            for roi in _iter_local(read, "roi"):
                zpos = _child_text(roi, "imageZposition")
                sop = (_child_text(roi, "imageSOP_UID", "") or "").strip()
                incl = (_child_text(roi, "inclusion", "TRUE") or "TRUE").strip().upper() == "TRUE"
                for em in _iter_local(roi, "edgeMap"):
                    pts.append(dict(
                        x=float(_child_text(em, "xCoord")),
                        y=float(_child_text(em, "yCoord")),
                        z_mm=float(zpos) if zpos else float("nan"),
                        sop_uid=sop, inclusion=incl))
            if pts:
                out.append(dict(session=si, nodule_id=nid, points=pts,
                                n_points=len(pts)))
    return out


def contour_to_mm(volume: CTVolume, points: Sequence[Dict]) -> torch.Tensor:
    """
    LIDC contour points -> mm.  x/y are column/row pixel indices; z_mm is
    already a world coordinate, so we only trust the affine for x and y and
    then overwrite z with the value the XML states.
    """
    xy = torch.tensor([[p["x"], p["y"]] for p in points], dtype=DTYPE)
    zmm = torch.tensor([p["z_mm"] for p in points], dtype=DTYPE)
    k = torch.zeros(len(points), dtype=DTYPE)
    for i, zv in enumerate(zmm):
        probe = _t([0.0, 0.0, float(zv)])
        k[i] = volume.mm_to_voxel(probe)[2]
    vox = torch.stack([xy[:, 0], xy[:, 1], k], dim=1)
    mm = volume.voxel_to_mm(vox)
    mm[:, 2] = zmm
    return mm


def parse_lidc_cxr_xml(path: str) -> List[Dict]:
    """The CXR read format: marks in radiograph pixels, keyed by SOP UID."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    out: List[Dict] = []
    for si, sess in enumerate(_iter_local(root, "CXRreadingSession")):
        for read in _iter_local(sess, "unblindedRead"):
            nid = (_child_text(read, "noduleID", "") or "").strip()
            for roi in _iter_local(read, "roi"):
                sop = (_child_text(roi, "imageSOP_UID", "") or "").strip()
                for em in _iter_local(roi, "edgeMap"):
                    out.append(dict(session=si, nodule_id=nid, sop_uid=sop,
                                    x=float(_child_text(em, "xCoord")),
                                    y=float(_child_text(em, "yCoord"))))
    return out


def find_ct_series(root: str, max_depth: int = 6) -> List[str]:
    """Directories under `root` that hold a multi-slice CT series."""
    hits = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if dirpath[len(root):].count(os.sep) > max_depth:
            continue
        dcm = [f for f in filenames if f.lower().endswith(".dcm")]
        if len(dcm) >= 3:
            try:
                if _read_tag(os.path.join(dirpath, dcm[0]), "0008|0060") == "CT":
                    hits.append(dirpath)
            except Exception:
                pass
    return sorted(hits)


__all__ = [
    "CTVolume", "PatchGrid", "SparseWeights", "View",
    "SliceView", "MIPSlabView", "DRRView",
    "load_ct", "from_sitk", "synthetic_thorax", "w", "nodule_difference_volume",
    "ray_consistency", "triangulate", "slices_seeing",
    "parse_lidc_ct_xml", "parse_lidc_cxr_xml", "contour_to_mm", "find_ct_series",
    "PATCH", "MU_WATER",
]
