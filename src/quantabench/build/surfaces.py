"""Surfaces and adsorbate placement, e.g. a PFAS molecule above a graphite slab for steered MD.

The graphite builder uses an orthorhombic 4-atom honeycomb cell (a = 2.46 Å, b = 4.26 Å, C-C = 1.42 Å)
with AB (Bernal) stacking at 3.35 Å, so slabs tile exactly under periodic boundaries and come out
as a rectangular box that LAMMPS and OVITO handle without a triclinic cell.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk, surface
from ase.constraints import FixAtoms
from ase.neighborlist import neighbor_list

CC_BOND = 1.42
INTERLAYER = 3.35
A_GRAPHENE = CC_BOND * np.sqrt(3)       # 2.4596 Å
B_RECT = 3 * CC_BOND                     # 4.26 Å

# 4-atom rectangular graphene cell (fractional y in units of B_RECT)
_RECT_BASIS = np.array([[0.0, 0.0], [0.0, CC_BOND], [A_GRAPHENE / 2, 1.5 * CC_BOND], [A_GRAPHENE / 2, 2.5 * CC_BOND]])

TAG_SURFACE = 0
TAG_ADSORBATE = 1


def graphene_sheet(nx: int, ny: int, z: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> Atoms:
    pts = []
    for i in range(nx):
        for j in range(ny):
            for bx, by in _RECT_BASIS:
                pts.append(((i * A_GRAPHENE + bx + shift[0]) % (nx * A_GRAPHENE),
                            (j * B_RECT + by + shift[1]) % (ny * B_RECT), z))
    return Atoms("C" * len(pts), positions=pts, cell=[nx * A_GRAPHENE, ny * B_RECT, 0.0], pbc=[True, True, False])


def graphite_slab(nx: int = 16, ny: int = 9, layers: int = 3, vacuum: float = 30.0,
                  fix_bottom_layers: int = 1) -> Atoms:
    """Graphite (0001) slab, AB stacked. ``nx, ny`` repeat the 2.46 x 4.26 Å cell
    (16 x 9 gives about 39 x 38 Å). ``vacuum`` is the empty space above the top layer."""
    slab = Atoms(pbc=[True, True, False])
    for layer in range(layers):
        shift = (0.0, CC_BOND) if layer % 2 else (0.0, 0.0)
        slab += graphene_sheet(nx, ny, z=layer * INTERLAYER, shift=shift)
    top = (layers - 1) * INTERLAYER
    slab.set_cell([nx * A_GRAPHENE, ny * B_RECT, top + vacuum])
    slab.pbc = [True, True, False]
    slab.set_tags([TAG_SURFACE] * len(slab))
    layer_index = np.rint(slab.positions[:, 2] / INTERLAYER).astype(int)
    slab.arrays["layer"] = layer_index
    if fix_bottom_layers:
        slab.set_constraint(FixAtoms(indices=np.where(layer_index < fix_bottom_layers)[0]))
    return slab


def metal_slab(symbol: str = "Au", miller=(1, 1, 1), size=(6, 6, 4), vacuum: float = 30.0,
               a: float | None = None) -> Atoms:
    """FCC/BCC metal slab via ASE (for comparison sorbents and validation)."""
    s = surface(bulk(symbol, a=a, cubic=True), miller, size[2], vacuum=0.0)
    s = s.repeat((size[0], size[1], 1))
    s.center(vacuum=vacuum / 2, axis=2)
    s.set_tags([TAG_SURFACE] * len(s))
    return s


def orient(molecule: Atoms, mode: str = "flat") -> Atoms:
    """Rotate a molecule about its center of mass. ``flat``: longest axis in x, flattest axis along z
    (lies on the surface). ``upright``: longest axis along z (head group down if it is heavier)."""
    m = molecule.copy()
    m.positions -= m.get_center_of_mass()
    if mode == "as-is":
        return m
    _, vecs = np.linalg.eigh(np.cov(m.positions.T))   # columns ascending variance
    if mode == "flat":
        target = np.column_stack([vecs[:, 2], vecs[:, 1], vecs[:, 0]])  # long->x, mid->y, short->z
    elif mode == "upright":
        target = np.column_stack([vecs[:, 0], vecs[:, 1], vecs[:, 2]])  # long->z
    else:
        raise ValueError(f"unknown orientation {mode!r}")
    if np.linalg.det(target) < 0:
        target[:, 0] *= -1
    m.positions = m.positions @ target
    return m


def place_on_surface(slab: Atoms, molecule: Atoms, height: float = 3.5, orientation: str = "flat",
                     xy: tuple[float, float] | None = None) -> Atoms:
    """Put ``molecule`` so its lowest atom sits ``height`` Å above the topmost surface atom,
    centered in x/y (or at ``xy``). Tags: 0 = surface, 1 = adsorbate."""
    mol = orient(molecule, orientation)
    top = slab.positions[:, 2].max()
    cx, cy = xy if xy is not None else (slab.cell[0, 0] / 2, slab.cell[1, 1] / 2)
    mol.positions += np.array([cx, cy, top + height - mol.positions[:, 2].min()])
    mol.set_tags([TAG_ADSORBATE] * len(mol))
    combined = slab.copy()
    combined += mol
    if mol.positions[:, 2].max() > combined.cell[2, 2] - 5.0:
        combined.cell[2, 2] = mol.positions[:, 2].max() + 15.0
    return combined


def min_distance_report(atoms: Atoms, cutoff: float = 1.0) -> list[tuple[int, int, float]]:
    """Atom pairs closer than ``cutoff`` Å (overlaps that make MD 'explode')."""
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    return sorted((int(a), int(b), float(x)) for a, b, x in zip(i, j, d) if a < b)


def surface_adsorbate_gap(atoms: Atoms) -> float:
    tags = atoms.get_tags()
    s, a = atoms.positions[tags == TAG_SURFACE], atoms.positions[tags == TAG_ADSORBATE]
    if len(s) == 0 or len(a) == 0:
        return float("nan")
    from scipy.spatial import cKDTree

    d, _ = cKDTree(s).query(a)
    return float(d.min())
