"""Descriptors for QSAR/QSPR work, from the 2D graph and the 3D conformer ensemble.

3D values are reported for the lowest-energy conformer and as Boltzmann-weighted averages,
because flexible PFAS chains change volume and surface area noticeably between conformers.

Areas and volumes of the union of atom spheres (``chem.radii``):

    sasa_A2         solvent-accessible area: radii of ``SurfaceOptions.radii_set`` (Bondi by default) +
                    ``SurfaceOptions.probe_A`` (1.4 Å, water); FreeSASA (Lee-Richards)
    vdw_area_A2     van der Waals area (no probe), same radii
    vdw_volume_A3   van der Waals volume, same radii (grid, ``VOLUME_SPACING_A``)
    cpk_area_A2     Spartan's CPK area: its own radii table (``radii.SPARTAN``), no probe
    cpk_volume_A3   Spartan's CPK volume, the same; equal to Spartan's CPKVOLUME to 0.1 % at its geometry
    volume_A3       RDKit's grid volume with RDKit's radii (kept for older tables)

Before 2026-11-06 ``sasa_A2`` passed FreeSASA's ProtOr classifier radii, which are 0 for every atom outside
a protein residue: the value was the union of 1.4 Å probe spheres on the atom centres, not the
solvent-accessible area. Tables written before then have that value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdFreeSASA, rdMolDescriptors

from . import radii as radii_mod

#: Grid spacing for sphere-union volumes (Å): about 0.2 % from the converged volume for PFAS-sized molecules.
VOLUME_SPACING_A = 0.15


@dataclass(frozen=True)
class SurfaceOptions:
    """Radii and probe for ``sasa_A2``/``vdw_*`` (``BuildConfig.radii_set``/``radii_overrides``/``sasa_probe_A``).
    ``cpk_*`` always use Spartan's table, so they stay comparable with Spartan documents."""
    radii_set: str = "bondi"
    overrides: tuple[tuple[int, float], ...] = field(default_factory=tuple)
    probe_A: float = 1.4

    @classmethod
    def make(cls, radii_set: str = "bondi", overrides=None, probe_A: float = 1.4) -> SurfaceOptions:
        over = tuple(sorted(radii_mod.normalize_overrides(overrides).items()))
        return cls((radii_set or "bondi").lower(), over, float(probe_A))


DEFAULT_SURFACE = SurfaceOptions()

DEBYE_PER_E_ANG = 4.80320471


def graph_descriptors(mol: Chem.Mol) -> dict:
    heavy = Chem.RemoveHs(mol)
    n_c = sum(1 for a in heavy.GetAtoms() if a.GetSymbol() == "C")
    n_f = sum(1 for a in heavy.GetAtoms() if a.GetSymbol() == "F")
    f_count = [sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "F")
               for a in heavy.GetAtoms() if a.GetSymbol() == "C"]
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": Descriptors.MolWt(mol),
        "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "heavy_atoms": heavy.GetNumAtoms(),
        "n_carbon": n_c,
        "n_fluorine": n_f,
        "f_to_c_ratio": n_f / n_c if n_c else 0.0,
        "n_cf2": sum(1 for n in f_count if n == 2),
        "n_cf3": sum(1 for n in f_count if n == 3),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp_crippen": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }


def sphere_area(mol: Chem.Mol, conf_id: int, radii, probe_A: float = 0.0) -> float:
    """Area (Å²) of the union of spheres of ``radii`` + ``probe_A`` around the atoms of conformer ``conf_id``
    (FreeSASA, Lee-Richards): the solvent-accessible area with a probe, the van der Waals area without."""
    opts = rdFreeSASA.SASAOpts()
    opts.probeRadius = float(probe_A)
    return float(rdFreeSASA.CalcSASA(mol, [float(r) for r in radii], confIdx=conf_id, opts=opts))


def sphere_union_volume(positions, radii, spacing: float = VOLUME_SPACING_A) -> float:
    """Volume (Å³) of the union of spheres: grid points within any sphere, counted on a grid of ``spacing``
    (each atom only marks the box around its sphere)."""
    pos = np.asarray(positions, float).reshape(-1, 3)
    r = np.asarray(radii, float).reshape(-1)
    if not len(r):
        return 0.0
    h = float(spacing)
    lo = (pos - r[:, None]).min(0) - h
    shape = np.ceil(((pos + r[:, None]).max(0) + h - lo) / h).astype(int) + 1
    grid = np.zeros(shape, bool)
    for p, ri in zip(pos, r):
        i0 = np.maximum(np.floor((p - ri - lo) / h).astype(int), 0)
        i1 = np.minimum(np.ceil((p + ri - lo) / h).astype(int) + 1, shape)
        axes = [lo[k] + h * np.arange(i0[k], i1[k]) - p[k] for k in range(3)]
        d2 = axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2
        grid[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= d2 <= ri * ri
    return float(grid.sum() * h ** 3)


def surface_descriptors_3d(mol: Chem.Mol, conf_id: int, opts: SurfaceOptions = DEFAULT_SURFACE) -> dict:
    """``sasa_A2``, ``vdw_area_A2``, ``vdw_volume_A3`` (radii of ``opts``) and ``cpk_area_A2``,
    ``cpk_volume_A3`` (Spartan's radii) of one conformer (module docstring)."""
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = mol.GetConformer(conf_id).GetPositions()
    r = radii_mod.radii(z, opts.radii_set, dict(opts.overrides))
    r_cpk = radii_mod.radii(z, "spartan")
    return {
        "sasa_A2": sphere_area(mol, conf_id, r, opts.probe_A),
        "vdw_area_A2": sphere_area(mol, conf_id, r, 0.0),
        "vdw_volume_A3": sphere_union_volume(pos, r),
        "cpk_area_A2": sphere_area(mol, conf_id, r_cpk, 0.0),
        "cpk_volume_A3": sphere_union_volume(pos, r_cpk),
    }


def dipole_debye(positions: np.ndarray, charges: np.ndarray, masses: np.ndarray) -> float:
    """Dipole from point charges about the center of mass (the convention for ions)."""
    com = (positions * masses[:, None]).sum(0) / masses.sum()
    mu = ((positions - com) * charges[:, None]).sum(0)
    return float(np.linalg.norm(mu) * DEBYE_PER_E_ANG)


def conformer_descriptors(mol: Chem.Mol, conf_id: int, surface: SurfaceOptions = DEFAULT_SURFACE) -> dict:
    return {
        "volume_A3": float(AllChem.ComputeMolVolume(mol, confId=conf_id, gridSpacing=0.2)),
        **surface_descriptors_3d(mol, conf_id, surface),
        "radius_of_gyration_A": rdMolDescriptors.CalcRadiusOfGyration(mol, confId=conf_id),
        "asphericity": rdMolDescriptors.CalcAsphericity(mol, confId=conf_id),
        "npr1": rdMolDescriptors.CalcNPR1(mol, confId=conf_id),
        "npr2": rdMolDescriptors.CalcNPR2(mol, confId=conf_id),
    }


def ensemble_descriptors(mol: Chem.Mol, conf_ids: list[int], weights: np.ndarray,
                         surface: SurfaceOptions = DEFAULT_SURFACE) -> dict:
    if not conf_ids:
        return {}
    per = [conformer_descriptors(mol, cid, surface) for cid in conf_ids]
    out = {f"{k}_best": v for k, v in per[0].items()}
    for k in per[0]:
        out[f"{k}_boltz"] = float(sum(w * p[k] for w, p in zip(weights, per)))
    return out
