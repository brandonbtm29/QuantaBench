"""Conformer generation (RDKit ETKDGv3), force-field pre-optimization (MMFF94s, UFF fallback),
energy-window filtering and symmetry-aware RMSD de-duplication."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors

KCAL_PER_EV = 23.060548


@dataclass
class ConformerSet:
    mol: Chem.Mol                 # with explicit H and all kept conformers
    conf_ids: list[int]           # ordered by energy (lowest first)
    energies_kcal: list[float]    # force-field energies, same order
    forcefield: str               # "MMFF94s" | "UFF" | "none"
    n_embedded: int
    notes: list[str] = field(default_factory=list)   # what the conformer search had to work around


def suggested_count(mol: Chem.Mol, cap: int) -> int:
    """Hawkins-style rule of thumb: more rotors, more conformers; capped by the user."""
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n = 50 if rot <= 7 else 200 if rot <= 12 else 300
    return max(1, min(n, cap))


def _invariant(exc: Exception) -> str:
    """The one informative line of an RDKit invariant violation (its message is a C++ stack trace)."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip() and "*" not in ln]
    return next((ln for ln in lines if ln != "Invariant Violation"), lines[0] if lines else type(exc).__name__)


#: Share of the per-molecule time limit (``BuildConfig.molecule_timeout_s``) the embedding may use by default
EMBED_TIME_FRACTION = 0.25


def embed_time_limit(molecule_timeout_s: float | None = None) -> float:
    """Seconds the ETKDG embedding of one molecule may take: ``EMBED_TIME_FRACTION`` of the per-molecule time
    limit (None: the one this process's run recorded, ``resources.molecule_limits``); 0 = no limit."""
    if molecule_timeout_s is None:
        from .. import resources

        molecule_timeout_s = resources.molecule_limits().molecule_timeout_s
    limit = float(molecule_timeout_s or 0.0)
    return EMBED_TIME_FRACTION * limit if limit > 0 else 0.0


class _OutOfTime(RuntimeError):
    pass


def embed(mol: Chem.Mol, n: int, seed: int = 42, threads: int = 1, notes: list[str] | None = None,
          timeout_s: float = 0.0) -> list[int]:
    """ETKDGv3 conformers (with small-ring torsions); random starting coordinates if that finds none.

    RDKit 2026.09's small-ring torsion terms abort the embedding of plain cyclopentanes (cyclopentane,
    cyclopentanol, cis-1,2-cyclopentanediol: "Invariant Violation: bad direction in linearSearch", for every
    seed). On such an error the embedding is repeated without small-ring torsions, then with the next seed
    and random coordinates (still deterministic); ``notes`` (optional list) gets a line saying so.

    ``timeout_s`` > 0 bounds the whole embedding (RDKit has no checkpoint inside it). RDKit's
    ``EmbedParameters.timeout`` (whole seconds, since 2024.03) stops one ``EmbedMultipleConfs`` call, but a
    call stopped that way returns a single conformer however many it had finished. So one conformer is
    embedded first and timed: when ``n`` of them would not fit the time left, fewer are asked for (a note
    says how many), and every call carries the time left as RDKit's timeout as a hard stop (a note says when
    it fired, and with how many conformers). Molecules that embed within the limit get the same conformers
    as without it."""
    t0 = time.monotonic()

    def left() -> float:
        return timeout_s - (time.monotonic() - t0) if timeout_s > 0 else 0.0

    def run(params, count: int) -> list[int]:
        if timeout_s > 0:
            remaining = left()
            if remaining <= 0:
                raise _OutOfTime()
            params.timeout = max(1, math.ceil(remaining))
        return list(AllChem.EmbedMultipleConfs(mol, numConfs=count, params=params))

    def attempt(seed_: int, small_rings: bool, random_coords: bool) -> list[int]:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed_)
        params.numThreads = int(threads)
        params.pruneRmsThresh = 0.1
        params.useSmallRingTorsions = small_rings
        params.useRandomCoords = random_coords
        with rdBase.BlockLogs():      # an invariant's C++ stack trace goes to the log; the exception is enough
            count = n
            if timeout_s > 0 and n > 1:     # time one conformer; ask for no more than fit the time left
                t1 = time.monotonic()
                one = run(params, 1)
                per = time.monotonic() - t1
                if one and per * n > left():
                    count = max(1, int(0.8 * left() / max(per, 1e-6)))
                    if notes is not None:
                        notes.append(f"conformer embedding limited by the time limit: {count} of {n} conformers "
                                     f"(one took {per:.1f} s, {timeout_s:.0f} s for the embedding)")
                    if count == 1:
                        return one        # no time for more: the timed conformer is the result
            ids = run(params, count)
            if not ids and not random_coords:  # hard cases (cages, crowded stereo): random starting coordinates
                params.useRandomCoords = True
                ids = run(params, max(1, count // 2))
        return ids

    first: RuntimeError | None = None
    try:
        for seed_, small_rings, random_coords in ((seed, True, False), (seed, False, False), (seed + 1, False, True)):
            try:
                ids = attempt(seed_, small_rings, random_coords)
            except _OutOfTime:
                raise
            except RuntimeError as exc:
                first = first or exc
                continue
            if first is not None and notes is not None:
                notes.append(f"RDKit ETKDG aborted ({_invariant(first)}); embedded without small-ring torsions"
                             + (f", seed {seed_}, random start" if random_coords else ""))
            break
        else:
            raise RuntimeError(f"RDKit could not embed 3D coordinates for this structure ({_invariant(first)})")
    except _OutOfTime:
        raise RuntimeError(f"conformer embedding ran out of time ({timeout_s:.0f} s) without a conformer") from None
    elapsed = time.monotonic() - t0
    out_of_time = timeout_s > 0 and elapsed >= 0.98 * timeout_s
    if out_of_time and ids and notes is not None:
        notes.append(f"conformer embedding stopped by the time limit ({timeout_s:.0f} s for the embedding) after "
                     f"{elapsed:.0f} s: {len(ids)} of {n} conformers")
    if not ids:
        if out_of_time:
            raise RuntimeError(f"conformer embedding ran out of time ({timeout_s:.0f} s) without a conformer")
        raise RuntimeError("RDKit could not embed 3D coordinates for this structure")
    return ids


def forcefield_optimize(mol: Chem.Mol, threads: int = 1, max_iters: int = 2000) -> tuple[list[float], str]:
    """Optimize every conformer in place. Returns (energies in kcal/mol by conformer order, force field)."""
    with rdBase.BlockLogs():   # "UFFTYPER: Unrecognized atom type" per atom and call; the result says it
        has_mmff, has_uff = AllChem.MMFFHasAllMoleculeParams(mol), AllChem.UFFHasAllMoleculeParams(mol)
    if has_mmff:
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=threads, maxIters=max_iters, mmffVariant="MMFF94s")
        return [e for _, e in res], "MMFF94s"
    if has_uff:
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
             seed: int = 42, threads: int = 1, timeout_s: float | None = None) -> ConformerSet:
    """Conformers of ``mol``: ETKDGv3 embedding (``embed``), force-field optimization, pruning.
    ``timeout_s``: time limit of the embedding (``embed``); None = ``embed_time_limit()`` (a share of the
    per-molecule limit this process's run recorded), 0 = none."""
    mol = Chem.Mol(mol)
    if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() and any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol)
    n = suggested_count(mol, max_conformers)
    mol.RemoveAllConformers()
    notes: list[str] = []
    limit = embed_time_limit() if timeout_s is None else max(0.0, float(timeout_s))
    ids = embed(mol, n, seed=seed, threads=threads, notes=notes, timeout_s=limit)
    energies, ff = forcefield_optimize(mol, threads=threads)
    # forcefield_optimize returns energies in conformer order, which matches ``ids``
    kept, kept_e = prune(mol, ids, energies, window_kcal, rmsd_threshold)
    return ConformerSet(mol=mol, conf_ids=kept, energies_kcal=kept_e, forcefield=ff, n_embedded=len(ids), notes=notes)


def boltzmann_weights(energies_kcal: list[float], temperature_k: float = 298.15) -> np.ndarray:
    e = np.asarray(energies_kcal, dtype=float)
    if e.size == 0:
        return e
    kt = 0.0019872043 * temperature_k  # kcal/mol
    w = np.exp(-(e - e.min()) / kt)
    return w / w.sum()
