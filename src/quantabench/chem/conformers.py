"""Conformer generation (RDKit ETKDGv3), force-field pre-optimization (MMFF94s, UFF fallback),
energy-window filtering and symmetry-aware RMSD de-duplication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors

KCAL_PER_EV = 23.060548


@dataclass
class ConformerSet:
    mol: Chem.Mol                 # with explicit H and all kept conformers
    conf_ids: list[int]           # ordered by energy (lowest first)
    energies_kcal: list[float]    # force-field energies, same order
    forcefield: str               # "MMFF94s" | "UFF" | "none"
    n_embedded: int


def suggested_count(mol: Chem.Mol, cap: int) -> int:
    """Hawkins-style rule of thumb: more rotors, more conformers; capped by the user."""
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n = 50 if rot <= 7 else 200 if rot <= 12 else 300
    return max(1, min(n, cap))


def embed(mol: Chem.Mol, n: int, seed: int = 42, threads: int = 1) -> list[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = int(threads)
    params.pruneRmsThresh = 0.1
    params.useSmallRingTorsions = True
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params))
    if not ids:  # hard cases (cages, crowded stereo): random starting coordinates
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(1, n // 2), params=params))
    if not ids:
        raise RuntimeError("RDKit could not embed 3D coordinates for this structure")
    return ids


def forcefield_optimize(mol: Chem.Mol, threads: int = 1, max_iters: int = 2000) -> tuple[list[float], str]:
    """Optimize every conformer in place. Returns (energies in kcal/mol by conformer order, force field)."""
    if AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=threads, maxIters=max_iters, mmffVariant="MMFF94s")
        return [e for _, e in res], "MMFF94s"
    if AllChem.UFFHasAllMoleculeParams(mol):
        res = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=threads, maxIters=max_iters)
        return [e for _, e in res], "UFF"
    return [0.0] * mol.GetNumConformers(), "none"


def prune(mol: Chem.Mol, conf_ids: list[int], energies: list[float], window_kcal: float,
          rmsd_threshold: float, max_keep: int | None = None) -> tuple[list[int], list[float]]:
    """Keep conformers within ``window_kcal`` of the minimum and at least ``rmsd_threshold`` Å
    (heavy atoms, symmetry-aware) apart from every lower-energy kept conformer."""
    order = np.argsort(energies)
    e_min = energies[order[0]]
    heavy = Chem.RemoveHs(mol)
    kept_ids: list[int] = []
    kept_e: list[float] = []
    for i in order:
        cid, e = conf_ids[i], energies[i]
        if window_kcal is not None and e - e_min > window_kcal:
            break
        duplicate = False
        for kid, ke in zip(kept_ids, kept_e):
            if abs(e - ke) > 1.5:  # very different energies are different conformers
                continue
            rms = rdMolAlign.GetBestRMS(heavy, heavy, prbId=cid, refId=kid, maxMatches=2000)
            if rms < rmsd_threshold:
                duplicate = True
                break
        if not duplicate:
            kept_ids.append(cid)
            kept_e.append(e)
        if max_keep and len(kept_ids) >= max_keep:
            break
    return kept_ids, kept_e


def generate(mol: Chem.Mol, max_conformers: int = 50, window_kcal: float = 6.0, rmsd_threshold: float = 0.5,
             seed: int = 42, threads: int = 1) -> ConformerSet:
    mol = Chem.Mol(mol)
    if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() and any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol)
    n = suggested_count(mol, max_conformers)
    mol.RemoveAllConformers()
    ids = embed(mol, n, seed=seed, threads=threads)
    energies, ff = forcefield_optimize(mol, threads=threads)
    # forcefield_optimize returns energies in conformer order, which matches ``ids``
    kept, kept_e = prune(mol, ids, energies, window_kcal, rmsd_threshold)
    return ConformerSet(mol=mol, conf_ids=kept, energies_kcal=kept_e, forcefield=ff, n_embedded=len(ids))


def boltzmann_weights(energies_kcal: list[float], temperature_k: float = 298.15) -> np.ndarray:
    e = np.asarray(energies_kcal, dtype=float)
    if e.size == 0:
        return e
    kt = 0.0019872043 * temperature_k  # kcal/mol
    w = np.exp(-(e - e.min()) / kt)
    return w / w.sum()
