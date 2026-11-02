"""Gaussian cube files: read, write, sample.

CONTRACT (implemented by the surfaces builder):

    Cube(origin_bohr (3,), axes_bohr (3,3) rows = step vectors, data (nx,ny,nz) float64,
         numbers (N,), positions_bohr (N,3), comment)
    read_cube(path) -> Cube            # handles negative atom count (MO cubes) and >1 value per point
    write_cube(path, cube) -> Path
    Cube.points_A() / Cube.sample(points_A) -> trilinear interpolation (outside the grid -> nan)
    Cube.to_pyvista() -> pyvista.ImageData (or StructuredGrid for non-orthogonal axes), point data "values",
                         coordinates in Å

Format (Gaussian ``cubegen``; also written by Psi4, ORCA, Multiwfn, CP2K, VASP tools)::

    comment line 1
    comment line 2
    NATOMS  X0 Y0 Z0  [NVAL]        origin; NATOMS < 0: MO cube, an MO-list line follows the atoms
    N1  a1x a1y a1z                 voxel count and step vector of axis 1 (slowest)
    N2  a2x a2y a2z
    N3  a3x a3y a3z                 axis 3 varies fastest in the data
    Z  q  x y z                     one line per atom (q: nuclear charge, often 0.0)
    [NMO  mo1 mo2 ...]              only when NATOMS < 0 (may wrap over several lines)
    data: N1*N2*N3*NVAL values, axis 3 fastest, the NVAL values of one point consecutive,
          6 per line, a line break after each axis-3 row

A positive voxel count means Bohr, a negative one Å (then the origin, step vectors and atom coordinates are
all read as Å). The data array keeps the file's order, ``data[i, j, k]`` at ``origin + i a1 + j a2 + k a3``;
VTK wants x fastest, so :meth:`Cube.to_pyvista` flattens in Fortran order. Several values per point
(multi-MO cubes) are kept in ``Cube.values`` (nx, ny, nz, nval); ``data`` is the first of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOHR_A = 0.529177210903          # Å per bohr (CODATA 2018)
_ORTHO_TOL = 1e-8                # |off-diagonal| below this (bohr) counts as an axis-aligned grid


@dataclass
class Cube:
    origin_bohr: np.ndarray
    axes_bohr: np.ndarray
    data: np.ndarray
    numbers: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    positions_bohr: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    comment: str = ""                                  # the two header lines, joined by "\n"
    nuclear_charges: np.ndarray | None = None          # 2nd column of the atom lines (None: = numbers)
    values: np.ndarray | None = None                   # (nx, ny, nz, nval) when the file has nval > 1
    mo_ids: list[int] = field(default_factory=list)    # MO cubes (negative atom count): the MO numbers

    def __post_init__(self) -> None:
        self.origin_bohr = np.asarray(self.origin_bohr, float).reshape(3)
        self.axes_bohr = np.asarray(self.axes_bohr, float).reshape(3, 3)
        self.data = np.asarray(self.data, float)
        if self.data.ndim != 3:
            raise ValueError(f"cube data must be 3D (nx, ny, nz), got shape {self.data.shape}")
        self.numbers = np.asarray(self.numbers, int).reshape(-1)
        self.positions_bohr = np.asarray(self.positions_bohr, float).reshape(-1, 3)
        if len(self.numbers) != len(self.positions_bohr):
            raise ValueError("numbers and positions_bohr have different lengths")

    # ------------------------------------------------------------------ geometry
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.data.shape)

    @property
    def origin_A(self) -> np.ndarray:
        return self.origin_bohr * BOHR_A

    @property
    def axes_A(self) -> np.ndarray:
        return self.axes_bohr * BOHR_A

    @property
    def positions_A(self) -> np.ndarray:
        return self.positions_bohr * BOHR_A

    @property
    def voxel_volume_A3(self) -> float:
        return float(abs(np.linalg.det(self.axes_A)))

    @property
    def is_orthogonal(self) -> bool:
        """True for an axis-aligned grid (step vectors along x, y, z): representable as ``ImageData``."""
        a = self.axes_bohr
        return bool(np.all(np.abs(a - np.diag(np.diag(a))) < _ORTHO_TOL) and np.all(np.diag(a) > 0))

    def points_A(self) -> np.ndarray:
        """(nx*ny*nz, 3) grid point coordinates (Å), in the order of ``data.ravel()`` (z fastest)."""
        nx, ny, nz = self.shape
        idx = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), axis=-1)
        return (self.origin_bohr + idx.reshape(-1, 3) @ self.axes_bohr) * BOHR_A

    def fractional_index(self, points_A) -> np.ndarray:
        """Continuous grid indices (i, j, k) of points given in Å (works for non-orthogonal axes)."""
        p = np.atleast_2d(np.asarray(points_A, float)) / BOHR_A - self.origin_bohr
        return np.linalg.solve(self.axes_bohr.T, p.T).T

    def sample(self, points_A, values: np.ndarray | None = None) -> np.ndarray:
        """Trilinear interpolation of ``values`` (default ``data``) at points in Å; outside the grid -> nan."""
        grid = self.data if values is None else np.asarray(values, float)
        f = self.fractional_index(points_A)
        n = np.array(grid.shape[:3])
        tol = 1e-9
        inside = np.all((f >= -tol) & (f <= n - 1 + tol), axis=1)
        f = np.clip(f, 0, n - 1)
        i0 = np.minimum(np.floor(f).astype(int), np.maximum(n - 2, 0))
        t = f - i0
        out = np.zeros(len(f))
        for dx in (0, 1):
            wx = t[:, 0] if dx else 1 - t[:, 0]
            ix = np.minimum(i0[:, 0] + dx, n[0] - 1)
            for dy in (0, 1):
                wy = t[:, 1] if dy else 1 - t[:, 1]
                iy = np.minimum(i0[:, 1] + dy, n[1] - 1)
                for dz in (0, 1):
                    wz = t[:, 2] if dz else 1 - t[:, 2]
                    iz = np.minimum(i0[:, 2] + dz, n[2] - 1)
                    out += wx * wy * wz * grid[ix, iy, iz]
        out[~inside] = np.nan
        return out

    def boundary_max(self) -> float:
        """Largest value on the six faces of the grid (an isosurface below it is cut open by the box)."""
        d = self.data
        return float(max(d[0].max(), d[-1].max(), d[:, 0].max(), d[:, -1].max(), d[:, :, 0].max(),
                         d[:, :, -1].max()))

    def to_pyvista(self):
        """``ImageData`` (axis-aligned grid) or ``StructuredGrid`` (skewed axes) in Å, point data "values"
        (plus "values_<mo>" for each value of a multi-value cube)."""
        import pyvista as pv

        nx, ny, nz = self.shape
        if self.is_orthogonal:
            grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=tuple(np.diag(self.axes_A)),
                                origin=tuple(self.origin_A))
        else:
            pts = self.points_A().reshape(nx, ny, nz, 3)
            grid = pv.StructuredGrid(pts[..., 0], pts[..., 1], pts[..., 2])
        grid.point_data["values"] = self.data.ravel(order="F")
        if self.values is not None and self.values.shape[-1] > 1:
            labels = self.mo_ids if len(self.mo_ids) == self.values.shape[-1] else range(self.values.shape[-1])
            for k, lab in enumerate(labels):
                grid.point_data[f"values_{lab}"] = self.values[..., k].ravel(order="F")
        grid.set_active_scalars("values")
        return grid

    def with_data(self, data: np.ndarray, comment: str | None = None) -> Cube:
        """Same grid and atoms, other values (e.g. an ESP computed on a density cube's grid)."""
        return Cube(self.origin_bohr.copy(), self.axes_bohr.copy(), np.asarray(data, float).reshape(self.shape),
                    self.numbers.copy(), self.positions_bohr.copy(), self.comment if comment is None else comment,
                    None if self.nuclear_charges is None else self.nuclear_charges.copy())


# ====================================================================== reading
_FORTRAN_EXP = re.compile(r"(?<=[0-9.])([+-]\d{3})")    # "1.234-100" (Fortran drops the E) -> "1.234E-100"


def _floats(text: str) -> np.ndarray:
    text = text.replace("D", "E").replace("d", "e")
    try:
        return np.array(text.split(), dtype=float)
    except ValueError:
        return np.array(_FORTRAN_EXP.sub(r"E\1", text).split(), dtype=float)


def read_cube(path) -> Cube:
    """Read a Gaussian cube file (Bohr or Å, orbital cubes, several values per point, skewed grids)."""
    with open(Path(path), encoding="utf-8", errors="replace") as fh:
        head = [fh.readline() for _ in range(6)]
        if not head[5]:
            raise ValueError(f"{path}: not a cube file (header too short)")
        comment = head[0].rstrip("\r\n") + "\n" + head[1].rstrip("\r\n")
        tok = head[2].split()
        natoms_signed = int(tok[0])
        natoms = abs(natoms_signed)
        origin = np.array(tok[1:4], float)
        nval_header = int(tok[4]) if len(tok) > 4 else 1
        counts, axes = [], np.zeros((3, 3))
        for i in range(3):
            t = head[3 + i].split()
            counts.append(int(t[0]))
            axes[i] = np.array(t[1:4], float)
        angstrom = any(c < 0 for c in counts)
        shape = tuple(abs(c) for c in counts)
        numbers = np.zeros(natoms, int)
        nuc = np.zeros(natoms)
        pos = np.zeros((natoms, 3))
        for a in range(natoms):
            t = fh.readline().split()
            if len(t) < 5:
                raise ValueError(f"{path}: atom line {a + 1} is incomplete")
            numbers[a] = int(float(t[0]))
            nuc[a] = float(t[1])
            pos[a] = np.array(t[2:5], float)
        mo_ids: list[int] = []
        if natoms_signed < 0:
            t = fh.readline().split()
            nmo = int(t[0])
            mo_ids = [int(x) for x in t[1:]]
            while len(mo_ids) < nmo:
                more = fh.readline().split()
                if not more:
                    raise ValueError(f"{path}: MO list is truncated")
                mo_ids += [int(x) for x in more]
            mo_ids = mo_ids[:nmo]
        raw = _floats(fh.read())
    npts = int(np.prod(shape))
    nval = nval_header
    if natoms_signed < 0 and len(mo_ids) > 1 and raw.size == npts * len(mo_ids):
        nval = len(mo_ids)
    if raw.size != npts * nval:
        if npts and raw.size % npts == 0:
            nval = raw.size // npts
        else:
            raise ValueError(f"{path}: expected {npts * nval} values for grid {shape} x {nval}, found {raw.size}")
    vals = raw.reshape(*shape, nval)
    scale = 1.0 / BOHR_A if angstrom else 1.0
    return Cube(origin * scale, axes * scale, vals[..., 0].copy(), numbers, pos * scale, comment,
                nuclear_charges=nuc, values=vals if nval > 1 else None, mo_ids=mo_ids)


# ====================================================================== writing
def write_cube(path, cube: Cube, fmt: str = "%13.5E") -> Path:
    """Write ``cube`` in Bohr (positive voxel counts), 6 values per line, a line break per z row."""
    path = Path(path)
    nx, ny, nz = cube.shape
    vals = cube.values if cube.values is not None else cube.data[..., None]
    nval = vals.shape[-1]
    comment = (cube.comment or "QuantaBench cube\n").split("\n")
    lines = [comment[0][:200], (comment[1] if len(comment) > 1 else "")[:200]]
    nat = len(cube.numbers)
    mo = cube.mo_ids if cube.mo_ids else []
    o = cube.origin_bohr
    tail = f"{nval:5d}" if nval > 1 and not mo else ""
    lines.append(f"{-nat if mo else nat:5d}{o[0]:12.6f}{o[1]:12.6f}{o[2]:12.6f}{tail}")
    for n, a in zip((nx, ny, nz), cube.axes_bohr):
        lines.append(f"{n:5d}{a[0]:12.6f}{a[1]:12.6f}{a[2]:12.6f}")
    nuc = cube.numbers.astype(float) if cube.nuclear_charges is None else cube.nuclear_charges
    for z, q, p in zip(cube.numbers, nuc, cube.positions_bohr):
        lines.append(f"{int(z):5d}{q:12.6f}{p[0]:12.6f}{p[1]:12.6f}{p[2]:12.6f}")
    if mo:
        lines.append(" ".join(str(x) for x in [len(mo), *mo]))
    m = nz * nval                       # values per (x, y) row
    row_fmt = (fmt * 6 + "\n") * (m // 6) + (fmt * (m % 6) + "\n" if m % 6 else "")
    flat = vals.reshape(nx * ny, m)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
        for r in flat:
            fh.write(row_fmt % tuple(r))
    return path


def read_any(obj) -> Cube:
    """A :class:`Cube` as is, or read from a path."""
    return obj if isinstance(obj, Cube) else read_cube(obj)
