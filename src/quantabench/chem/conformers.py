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
    # how many of ``conf_ids`` the refinement takes; 0 = ``BuildConfig.refine_top``. A merged CREST + RDKit pool
    # (``search_rule.merge_pools``) sets it to all of its conformers: its two energy scales cannot pick a top N
    refine_count: int = 0

    def refine_ids(self, refine_top: int) -> list[int]:
        """The conformers the refinement takes (lowest first within each source)."""
        n = self.refine_count if self.refine_count > 0 else max(1, int(refine_top))
        return self.conf_ids[:n]


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
#: Seconds the first conformer of an attempt may take before the embedding moves on to the next way of embedding
#: (``embed``). With RDKit 2026.09's small-ring torsions, ETKDG finds no conformer for sugar-rich glycosides
#: (digitoxin, digoxin, glycyrrhizin, maltopentaose: b05-b12, 2026-11-12) in 7.5 minutes; without them it takes
#: about a second.
FIRST_CONFORMER_S = 60.0
#: Share of the time left that the batch after the timed first conformer may use (``embed``). The rest tops up a batch
#: that RDKit's timeout stopped: RDKit then returns at most one conformer, or none (gitoxin, 2026-11-17: its first
#: conformer took 9 s with small-ring torsions, the batch of 39 none in the 441 s left, and the molecule failed with
#: "ran out of time"). The top-up embeds without small-ring torsions, about a second per conformer for glycosides.
EMBED_BATCH_SHARE = 0.8


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
    as without it.

    An ETKDG start that finds no first conformer within ``FIRST_CONFORMER_S`` gives way to the next one (without
    small-ring torsions, then random starting coordinates), with a note: sugar-rich glycosides found none with
    small-ring torsions in 7.5 minutes and took about a second without them. A call stopped by RDKit's timeout
    before its first conformer returns the id -1, which is not a conformer (it failed those molecules with
    "Bad Conformer Id")."""
    t0 = time.monotonic()

    def left() -> float:
        return timeout_s - (time.monotonic() - t0) if timeout_s > 0 else 0.0

    def run(params, count: int, cap: float = 0.0) -> list[int]:
        """``count`` conformers; RDKit's timeout is the time left (``timeout_s``) and at most ``cap`` s. A call that
        RDKit's timeout stopped before its first conformer returns [-1] (RDKit 2026.09): not a conformer, dropped."""
        limit = 0.0
        if timeout_s > 0:
            limit = left()
            if limit <= 0:
                raise _OutOfTime()
        if cap > 0:
            limit = min(limit, cap) if limit > 0 else cap
        params.timeout = max(1, math.ceil(limit)) if limit > 0 else 0
        return [int(i) for i in AllChem.EmbedMultipleConfs(mol, numConfs=count, params=params) if int(i) >= 0]

    def make_params(seed_: int, small_rings: bool, random_coords: bool):
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed_)
        params.numThreads = int(threads)
        params.pruneRmsThresh = 0.1
        params.useSmallRingTorsions = small_rings
        params.useRandomCoords = random_coords
        return params

    def top_up(saved, got: list[int], count: int, seed_: int, small_rings: bool, random_coords: bool) -> list[int]:
        """After a batch that RDKit's timeout stopped (``got``: what it returned, at most one conformer): keep the
        timed first conformer when the batch returned none, and embed the rest without small-ring torsions in the
        time left (1 - EMBED_BATCH_SHARE of it), added to what is there."""
        if not got:
            mol.RemoveAllConformers()
            got = [int(mol.AddConformer(saved, assignId=True))]
        more: list[int] = []
        if count > len(got) and left() > 0:
            params = make_params(seed_ + 1, False, random_coords)
            params.clearConfs = False
            try:
                more = [i for i in run(params, count - len(got)) if i not in got]
            except RuntimeError:          # out of time, or an RDKit invariant: what is there is the result
                more = []
        if notes is not None:
            notes.append(f"conformer embedding: the time limit stopped RDKit's batch of {count}"
                         f"{' with small-ring torsions' if small_rings else ''}; kept {len(got)} conformer(s) and "
                         f"added {len(more)} without small-ring torsions")
        return got + more

    def attempt(seed_: int, small_rings: bool, random_coords: bool) -> list[int]:
        params = make_params(seed_, small_rings, random_coords)
        with rdBase.BlockLogs():      # an invariant's C++ stack trace goes to the log; the exception is enough
            count = n
            saved = None
            # one conformer first: it is timed (to fit the time limit), and an ETKDG start that finds none within
            # FIRST_CONFORMER_S gives way to the next way of embedding instead of using up the time limit
            if not random_coords or (timeout_s > 0 and n > 1):
                t1 = time.monotonic()
                one = run(params, 1, cap=0.0 if random_coords else FIRST_CONFORMER_S)
                per = time.monotonic() - t1
                if not one:
                    if not random_coords:
                        if notes is not None:
                            notes.append(f"RDKit ETKDG{' with small-ring torsions' if small_rings else ''} found no "
                                         f"conformer in {per:.0f} s")
                        return []
                elif n == 1:
                    return one            # the conformer asked for
                elif timeout_s > 0 and per * n > left():
                    count = max(1, int(0.8 * left() / max(per, 1e-6)))
                    if notes is not None:
                        notes.append(f"conformer embedding limited by the time limit: {count} of {n} conformers "
                                     f"(one took {per:.1f} s, {timeout_s:.0f} s for the embedding)")
                    if count == 1:
                        return one        # no time for more: the timed conformer is the result
                if timeout_s > 0 and one:
                    saved = Chem.Conformer(mol.GetConformer(one[0]))
            if saved is not None:         # the batch may use EMBED_BATCH_SHARE of the time left; a stopped one is topped up
                t2 = time.monotonic()
                cap = EMBED_BATCH_SHARE * left()
                ids = run(params, count, cap=cap)
                if len(ids) < min(count, 2) and time.monotonic() - t2 >= 0.95 * cap:
                    return top_up(saved, ids, count, seed_, small_rings, random_coords)
            else:
                ids = run(params, count)
            if not ids and not random_coords:  # hard cases (cages, crowded stereo): random starting coordinates
                params.useRandomCoords = True
                ids = run(params, max(1, count // 2))
        return ids

    first: RuntimeError | None = None
    ids: list[int] = []
    try:
        for seed_, small_rings, random_coords in ((seed, True, False), (seed, False, False), (seed + 1, False, True)):
            try:
                ids = attempt(seed_, small_rings, random_coords)
            except _OutOfTime:
                raise
            except RuntimeError as exc:
                first = first or exc
                continue
            if not ids:                   # none found this way (the note says how long it tried): the next way
                continue
            if (seed_, small_rings, random_coords) != (seed, True, False) and notes is not None:
                why = f"RDKit ETKDG aborted ({_invariant(first)})" if first is not None else "RDKit ETKDG found none"
                notes.append(f"{why}; embedded without small-ring torsions"
                             + (f", seed {seed_}, random start" if random_coords else ""))
            break
        else:
            if first is not None:
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


#: Ring conformers of the RDKit search (``BuildConfig.ring_conformers``; Spartan's Preferences > Settings > Conformer
#: Rules, whose "Thorough" "considers twist-boat conformers of six-membered rings (in addition to chair conformers)"):
#: "normal" = the rings ETKDG and the force field give (chairs, with ETKDG's small-ring torsions); "thorough" = also
#: twist-boats of every saturated six-membered ring (``twist_boats``)
RING_CONFORMERS = ("normal", "thorough")
#: "thorough": twist-boats are made from this many of the lowest force-field conformers
TWIST_BOAT_SEEDS = 5
#: ring torsions (degrees, around the ring) of a twist-boat; ``twist_boats`` holds a ring at each of its three
#: rotations and their mirror images, then lets the force field relax it (cyclohexane: MMFF94s +5.9 kcal/mol)
TWIST_BOAT_TORSIONS = (30.0, -65.0, 30.0, 30.0, -65.0, 30.0)


def check_ring_conformers(value: str) -> str:
    if value not in RING_CONFORMERS:
        raise ValueError(f"ring_conformers must be one of {', '.join(RING_CONFORMERS)}, not {value!r}")
    return value


def flexible_six_rings(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """The saturated six-membered rings of ``mol`` (sp3 atoms, single bonds, not aromatic): the rings that have
    twist-boat conformers."""
    out = []
    info = mol.GetRingInfo()
    for ring, bonds in zip(info.AtomRings(), info.BondRings()):
        if len(ring) != 6:
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if any(a.GetIsAromatic() or a.GetHybridization() != Chem.HybridizationType.SP3 for a in atoms):
            continue
        if any(mol.GetBondWithIdx(b).GetBondType() != Chem.BondType.SINGLE for b in bonds):
            continue
        out.append(tuple(ring))
    return out


def twist_boats(mol: Chem.Mol, conf_ids: list[int], energies: list[float], forcefield: str,
                notes: list[str] | None = None) -> tuple[list[int], list[float]]:
    """Twist-boat conformers ("thorough" ring conformers): for each saturated six-membered ring of each of the
    ``TWIST_BOAT_SEEDS`` lowest conformers, a copy with the ring held at a twist-boat (``TWIST_BOAT_TORSIONS``, its
    three rotations and their mirror images) is optimized with the force field, then optimized freely. Returns the
    new conformer ids (added to ``mol``) and their energies (kcal/mol); one that falls back to a chair is a duplicate
    for ``prune`` to drop. Nothing without a force field or a flexible ring."""
    rings = flexible_six_rings(mol)
    if not rings or forcefield not in ("MMFF94s", "UFF") or not conf_ids:
        return [], []
    with rdBase.BlockLogs():
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s") if forcefield == "MMFF94s" else None

    def field(cid: int):
        if props is not None:
            return AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        return AllChem.UFFGetMoleculeForceField(mol, confId=cid)

    patterns = []
    for shift in range(3):
        rot = TWIST_BOAT_TORSIONS[shift:] + TWIST_BOAT_TORSIONS[:shift]
        patterns += [rot, tuple(-t for t in rot)]
    seeds = [conf_ids[i] for i in np.argsort(energies)[:TWIST_BOAT_SEEDS]]
    new_ids: list[int] = []
    new_e: list[float] = []
    with rdBase.BlockLogs():
        for cid in seeds:
            for ring in rings:
                for pattern in patterns:
                    nid = int(mol.AddConformer(Chem.Conformer(mol.GetConformer(cid)), assignId=True))
                    ff = field(nid)
                    for k, target in enumerate(pattern):
                        a, b, c, d = (ring[(k + j) % 6] for j in range(4))
                        if props is not None:
                            ff.MMFFAddTorsionConstraint(a, b, c, d, False, target - 2.0, target + 2.0, 500.0)
                        else:
                            ff.UFFAddTorsionConstraint(a, b, c, d, False, target - 2.0, target + 2.0, 500.0)
                    ff.Minimize(maxIts=2000)
                    ff = field(nid)
                    ff.Minimize(maxIts=2000)
                    new_ids.append(nid)
                    new_e.append(float(ff.CalcEnergy()))
    if notes is not None:
        notes.append(f"thorough ring conformers: {len(new_ids)} twist-boat start(s) of {len(rings)} six-membered "
                     f"ring(s) from the {len(seeds)} lowest conformer(s)")
    return new_ids, new_e


def generate(mol: Chem.Mol, max_conformers: int = 50, window_kcal: float = 6.0, rmsd_threshold: float = 0.5,
             seed: int = 42, threads: int = 1, timeout_s: float | None = None, seeds=(),
             ring_conformers: str = "normal") -> ConformerSet:
    """Conformers of ``mol``: ETKDGv3 embedding (``embed``), force-field optimization, pruning.
    ``timeout_s``: time limit of the embedding (``embed``); None = ``embed_time_limit()`` (a share of the
    per-molecule limit this process's run recorded), 0 = none. ``ring_conformers`` (``RING_CONFORMERS``):
    "thorough" adds twist-boats of saturated six-membered rings (``twist_boats``) before the pruning, which then keeps
    at most ``max_conformers``; "normal" changes nothing.

    ``seeds``: coordinate arrays (one per conformer, on ``mol``'s atoms with explicit H; ``seed_coordinates``) from
    the input file. They are the first conformers: optimized with the force field like the others, never pruned,
    and first in ``conf_ids`` (so the refinement's top N includes them); they count towards ``max_conformers``
    (``max_conformers=1`` with one seed embeds nothing)."""
    check_ring_conformers(ring_conformers)
    mol = Chem.Mol(mol)
    if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() and any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol)
    seeds = [np.asarray(s, dtype=float) for s in seeds if np.shape(s) == (mol.GetNumAtoms(), 3)]
    n = suggested_count(mol, max_conformers)
    mol.RemoveAllConformers()
    notes: list[str] = []
    limit = embed_time_limit() if timeout_s is None else max(0.0, float(timeout_s))
    ids: list[int] = []
    if not seeds:
        ids = embed(mol, n, seed=seed, threads=threads, notes=notes, timeout_s=limit)
    elif n > len(seeds):
        try:
            ids = embed(mol, n - len(seeds), seed=seed, threads=threads, notes=notes, timeout_s=limit)
        except RuntimeError as exc:   # the file's coordinates are conformers enough
            notes.append(f"no new conformers: {exc}; the file's coordinates only")
    seed_ids: list[int] = []
    if seeds:                         # the file's coordinates first, then the embedded conformers
        embedded = [Chem.Conformer(mol.GetConformer(i)) for i in ids]
        mol.RemoveAllConformers()
        for pos in seeds:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, xyz in enumerate(pos):
                conf.SetAtomPosition(i, [float(v) for v in xyz])
            conf.Set3D(True)
            seed_ids.append(mol.AddConformer(conf, assignId=True))
        ids = [mol.AddConformer(conf, assignId=True) for conf in embedded]
    energies, ff = forcefield_optimize(mol, threads=threads)
    # forcefield_optimize returns energies in conformer order, which matches ``seed_ids + ids``
    all_ids = seed_ids + ids
    max_keep = None
    more_ids: list[int] = []
    if ring_conformers == "thorough":
        more_ids, more_e = twist_boats(mol, all_ids, list(energies), ff, notes)
        all_ids, energies = all_ids + more_ids, list(energies) + more_e
        max_keep = max_conformers
    kept, kept_e = prune(mol, all_ids, energies, window_kcal, rmsd_threshold, max_keep=max_keep)
    if more_ids:
        notes[-1] += f"; {len(set(kept) & set(more_ids))} kept after pruning"
    if seed_ids:
        by_id = dict(zip(all_ids, energies))
        rest = [(k, e) for k, e in zip(kept, kept_e) if k not in seed_ids]
        kept = seed_ids + [k for k, _e in rest]
        kept_e = [by_id[k] for k in seed_ids] + [e for _k, e in rest]
    return ConformerSet(mol=mol, conf_ids=kept, energies_kcal=kept_e, forcefield=ff, n_embedded=len(ids), notes=notes)


def _graph_only(mol: Chem.Mol) -> Chem.Mol:
    """The heavy-atom graph of ``mol``: elements and connections only (charges, bond orders, hydrogens and stereo
    removed), so a structure as read matches its standardized and protonated form."""
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)
        a.SetNumRadicalElectrons(0)
        a.SetIsotope(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
        b.SetStereo(Chem.BondStereo.STEREONONE)
    out = m.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def seed_coordinates(source: Chem.Mol, target: Chem.Mol) -> tuple[list[np.ndarray], str]:
    """The 3D conformers of ``source`` (a structure as read from a file) as coordinates of ``target`` (its prepared
    form with explicit H: standardized, protonated, without counter-ions): the heavy atoms where the file has them
    (matched by element and connection, so charges and bond orders may differ), the hydrogens placed by RDKit.
    Returns (coordinate arrays, "" or why there are none)."""
    if source is None or source.GetNumConformers() == 0:
        return [], "no coordinates"
    if not any(c.Is3D() for c in source.GetConformers()):
        return [], "2D coordinates only"
    with rdBase.BlockLogs():
        src = Chem.RemoveHs(source, sanitize=False)
        heavy = Chem.RemoveHs(target, sanitize=False)
        if Chem.AddHs(heavy).GetNumAtoms() != target.GetNumAtoms():
            return [], "the prepared structure's hydrogens could not be placed"
        match = _graph_only(src).GetSubstructMatch(_graph_only(heavy))
    if len(match) != heavy.GetNumAtoms():
        return [], "the prepared structure's atoms do not match the file's"
    out = []
    for conf in src.GetConformers():
        if not conf.Is3D():
            continue
        m = Chem.Mol(heavy)
        m.RemoveAllConformers()
        c = Chem.Conformer(heavy.GetNumAtoms())
        for i, j in enumerate(match):
            c.SetAtomPosition(i, conf.GetAtomPosition(j))
        c.Set3D(True)
        m.AddConformer(c, assignId=True)
        with rdBase.BlockLogs():
            out.append(Chem.AddHs(m, addCoords=True).GetConformer().GetPositions())
    return out, ""


def boltzmann_weights(energies_kcal: list[float], temperature_k: float = 298.15) -> np.ndarray:
    e = np.asarray(energies_kcal, dtype=float)
    if e.size == 0:
        return e
    kt = 0.0019872043 * temperature_k  # kcal/mol
    w = np.exp(-(e - e.min()) / kt)
    return w / w.sum()
