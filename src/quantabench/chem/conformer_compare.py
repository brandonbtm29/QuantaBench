"""Compare the conformers two programs found for the same molecules: Spartan against QuantaBench, or two
QuantaBench runs (RDKit against CREST, gas against water). VAL-3 in docs/dev/BACKLOG.md, docs/VALIDATION.md.

CONTRACT:

    read_sources(paths, program=None, include=(), log=None) -> list[ConformerSet]   # auto-detects the kind
    read_run(folder) -> list[ConformerSet]                         # a QuantaBench run folder (any machine's share)
    read_spartan_documents(paths, include=()) -> list[ConformerSet]   # .spartan documents (chem.io_spartan)
    read_spartan_import(table, sdf, include=()) -> list[ConformerSet]  # `import --conformers all --table --sdf`
    merge_sets(sets) -> list[ConformerSet]                         # one set per constitution, duplicates merged
    SymmetryModel(numbers, bonds)                                  # heavy atoms, sibling groups, automorphisms
    symmetric_rmsd(model, ref, probe, mirror=True) -> Alignment    # exact symmetric heavy-atom RMSD
    torsion_deltas(model, ref, probe, alignment) -> {kind: [deg]}  # backbone / rotor / polar_h
    rescore(sets, phases=("gas", "water"), relax="", ...) -> int   # GFN2-xTB (tblite) single points, optimization
    compare(a_sets, b_sets, ...) -> Comparison                     # per molecule, per conformer, summary
    write_outputs(comparison, out_dir) -> list[Path]               # molecules.csv, conformers.csv, levels.csv,
                                                                   # summary.csv, levels_summary.csv, report.md

Conformer sets
--------------
A :class:`ConformerSet` is one molecule's conformers in one atom order: atomic numbers, bonds (orders ignored),
charge, and per conformer the geometry, its energies at every level it has (kJ/mol, ``energies[level]``; only
differences within a level are used) and dipoles (debye, ``dipoles[level]``). Levels are named by program:

- Spartan: ``Spartan B3LYP/6-311+G** water`` (``SPENERGY`` of a single point, hartree; its ``DIPOLE_VEC`` moved to
  the centre of mass for ions, ``io_spartan.molecule_row``), ``Spartan WB97X-D/6-31G(d)`` and ``Spartan DLXD2XV``
  (an Est. Density Functional job: ``SPENERGY`` and ``POSTENERGY``; the dipole is the SCF's ``QM_DIPOLE``),
  ``Spartan PM6`` (``E_KCAL`` of a semi-empirical single point), ``Spartan MMFF94 (search)`` (``DC_LIST_ENERGY_KCAL``,
  the conformer search's energy that extracted conformers carry) and ``Spartan <method> (search)`` for the saved
  conformers of a conformer-search document. Energies of an optimization (``OPT``) belong to the optimized geometry,
  which the import does not read (it keeps the starting one): they are left out. Failed jobs give no energies.
- QuantaBench: ``QuantaBench <method> <phase>`` (``tblite-gfn2 water``; CREST without refinement: its level), the
  energy of each kept conformer (``<id>_conformers.sdf``, ``QB_Energy_eV``) and its dipole (``<id>_summary.json``).
- The common level of :func:`rescore`: ``common GFN2-xTB gas SP``, ``common GFN2-xTB ALPB(water) SP`` and
  ``common GFN2-xTB ALPB(water) opt`` (after a short optimization at that level). An optimization that changes
  the bonds (``chem.crest``'s distance rule: a proton moved, an O-S contact formed) is set aside: that conformer
  gets no optimized energy, and with optimizations only optimized energies decide.

Sets are joined by constitution (``io_spartan.constitution_key``: elements, connectivity and total charge; bond
orders, charge placement, stereo and atom order do not matter), so an anion that Spartan drew with the charge on
another oxygen still joins. Within one side, sets of one constitution are merged into the first one's atom order
(:func:`merge_sets`), and conformers with the same geometry (RMSD < 0.01 Å without re-mapping atoms: the 5 conformers
of a ``DFT_Top5`` document are copies of 5 of the ``All_Conformers_Top30`` ones) become one conformer with both
levels' energies.

Symmetry: why not plain best-RMS enumeration
--------------------------------------------
A symmetric RMSD minimizes over the molecule's automorphisms (atom permutations that keep the graph). For a
perfluoroalkyl chain they multiply: CF3 (3! = 6) x each CF2 (2) x SO3- (6): 4,608 heavy-atom permutations for PFOS,
98,304 for a C12 sulfonate, and RDKit's ``GetBestRMS`` aligns every one (with a ``maxMatches`` cap the true best can
simply be missed). The permutations factor, though: the explosion comes only from **sibling groups**, terminal
heavy atoms of one element and H count on the same parent (the F of CF3/CF2, the O of SO3-/COO-/NO2, the methyl
carbons of N(CH3)3+). Swapping siblings is always an automorphism, and these are exactly the resonance-equivalent
atoms (both carboxylate O, all sulfonate O; the =O and -OH of an acid differ by their H, so they are not siblings).

:class:`SymmetryModel` therefore builds a **reduced graph**: the heavy atoms that are not siblings (the skeleton)
plus one pseudo-atom per sibling group at the group's centroid. Its automorphisms are few (chain reversal of a
symmetric molecule, the two CF3 of a C(CF3)2 branch). :func:`symmetric_rmsd` then

- enumerates the full heavy-atom automorphism group exactly (reduced automorphisms x every permutation inside every
  group, all Kabsch fits in one batched SVD) when it has at most ``EXACT_MAPS`` (2,000) members: TFA, PFBA, PFOA
  (768) and most of the validation set;
- otherwise aligns on the reduced points for each reduced automorphism, assigns each group's atoms optimally
  (all k! orders for k <= 4, else the Hungarian method), re-fits on all heavy atoms and repeats to convergence
  (at most 8 rounds), from three starts (the reduced points, the skeleton alone, all heavy atoms). For alike
  conformers this is the exact minimum (the tests check it against the full enumeration with noisy, rotated
  copies whose group atoms are renumbered: PFHxA, PFHxS, a quaternary ammonium); for very different ones (RMSD
  > 1 Å) it can come out a few hundredths of an Å high, which never turns a 0.5 Å decision. PFOS: 1.6 ms per
  pair instead of 30 ms; a C10 sulfonate (18,432 automorphisms): 2 ms instead of 120 ms. A skeleton whose reduced
  points are collinear (TFA: C, C and two centroids on the C-C axis) has an undetermined rotation and always takes
  the exact path. The exact path equals RDKit's ``GetBestRMS`` (tests), except that an acid's =O and -OH are not
  interchangeable here (RDKit's heavy-atom graph has no H to tell them apart).

The cost is then that of the skeleton's few automorphisms, not 6 x 2^n. Hydrogens are left out (heavy-atom RMSD, the
usual measure; QuantaBench prunes its own conformers with it, at 0.5 Å).

Mirror images: a conformer and its mirror image have the same energy and the same |dipole|; Spartan's conformer
lists keep both (PFAS chains are helical: P and M helices are separate entries with one MMFF energy). With
``mirror=True`` (the default) the probe is also compared reflected, but a match must keep every stereocentre's
configuration (signed volume of its first three heavy neighbors under the mapping): a reflected chiral conformer is
a conformer of the other enantiomer, which is accepted only when the two sides drew opposite enantiomers (a SMILES
without R/S gives random configurations).

Torsions (:func:`torsion_deltas`, under the RMSD's atom mapping): ``backbone`` torsions about non-ring bonds between
skeleton atoms (the end atoms chosen skeleton atom first, then a unique terminal atom, then a sibling); ``rotor``
torsions where one end is a symmetric top (CF3, COO-, SO3-, N(CH3)3+, all of its substituents siblings); ``polar_h``
torsions of a terminal OH/SH hydrogen (syn/anti carboxylic acids, which heavy atoms do not see). Two conformers are
**the same** when their RMSD is at most ``rmsd_cut`` (0.5 Å), every backbone and polar-H torsion agrees within
``torsion_cut`` (60°; gauche/anti differ by 120°, the P/M twist of a perfluoro chain by ~30°) and the
configurations agree. Rotor torsions are reported, not used.

The comparison
--------------
Side A is the reference (Spartan), side B the probe (QuantaBench). For each joined molecule, all A x B pairs are
compared. Each side ranks its own conformers at its own best level (``level_a``/``level_b``: Spartan DLXD2XV, then
DFT, then semi-empirical/xTB, then force fields; ``--a-level`` picks another). Reported per molecule
(molecules.csv) and per side and level (levels.csv):

- how many of A's conformers within 5 and 10 kJ/mol of A's minimum B found (and the reverse);
- whether the two lowest conformers are the same (``same_best``), the RMSD, the mirror flag and the largest and
  mean backbone torsion difference between them, the rotor and polar-H differences;
- where A's best is in B's list (``a_best_match_rel_b_kj``) and B's best in A's.

Search or ranking (``cause``), with the common level (:func:`rescore`; the optimized energies when there are some,
else the single points, in ``phase``):

- ``same best``;
- ``ranking: found, ranked higher``: B has A's best conformer but its own energies put another one lower;
- ``ranking: missed, higher at common level``: B never produced A's best, but the common level ranks it above B's
  best, so B's ranking would not have picked it anyway (energies, not search);
- ``search: missed, lower at common level``: B never produced A's best and the common level puts it below B's best:
  B's search missed a conformer its own level prefers.

``union_best``: the lowest conformer of both sets together at the common level, and on which side(s) it is.

Dipoles (``phase``): A's own dipole of its best conformer (``dip_a_ref``, e.g. B3LYP/C-PCM), B's own dipole of its
best (``dip_b_best``, the validation's QuantaBench value), the common level's dipole at A's best geometry
(``dip_common_a_best``: the method alone, same conformer) and at B's conformer that matches A's best. The report
splits the A-B dipole gap into method (same geometry) and conformer.

Every number here is per conformer as each program wrote it: Spartan's All_Conformers documents hold MMFF
geometries (the DFT_Top5 single points are at those), QuantaBench's are optimized at its level, so single points at
the common level put Spartan's conformers slightly higher (strain); the optimization removes that.
"""

from __future__ import annotations

import contextlib
import csv
import fnmatch
import hashlib
import itertools
import json
import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KJ_PER_HARTREE = 2625.4996394799
KJ_PER_KCAL = 4.184
KJ_PER_EV = 96.48533212331
BOHR_ANG = 0.529177210903
DEBYE_PER_E_ANG = 4.80320471
DEFAULT_RMSD_A = 0.5
DEFAULT_TORSION_DEG = 60.0
DEFAULT_WINDOWS_KJ = (5.0, 10.0)
#: the full heavy-atom automorphism group is enumerated when it has at most this many members
EXACT_MAPS = 2000
#: at most this many automorphisms of the reduced graph (a cap no molecule of the validation sets reaches)
MAX_REDUCED_MAPS = 1000
#: conformers of one side closer than this (Å, same atom order, no re-mapping) are one conformer
DUPLICATE_RMSD_A = 0.01
COMMON = "common GFN2-xTB"
#: ``cause`` values
CAUSES = ("same best", "ranking: found, ranked higher", "ranking: missed, higher at common level",
          "search: missed, lower at common level", "missed (no common energies)", "no energies")


# ------------------------------------------------------------------------------------------------ data
@dataclass
class Conformer:
    positions: np.ndarray                                   # (N, 3) Å, in the set's atom order
    energies: dict[str, float] = field(default_factory=dict)   # level -> kJ/mol
    dipoles: dict[str, float] = field(default_factory=dict)    # level -> debye
    source: str = ""                                        # where it came from (document + storage, run/id)
    tag: str = ""                                           # Spartan's TORLABEL, QuantaBench's rank
    extra: dict = field(default_factory=dict)               # "relaxed": positions after the common optimization


@dataclass
class ConformerSet:
    key: str                                  # io_spartan.constitution_key
    numbers: np.ndarray
    bonds: list[tuple[int, int]]
    charge: int
    multiplicity: int = 1
    program: str = ""                         # "Spartan", "QuantaBench", "QuantaBench (CREST)"
    label: str = ""                           # CAS / id
    labels: list[str] = field(default_factory=list)
    name: str = ""
    inchikey: str = ""
    formula: str = ""
    group: str = ""                           # Spartan cohort (C3A_...), "" for QuantaBench
    phase: str = ""                           # a QuantaBench run's phase ("gas", "water"); "" = not known
    sources: list[str] = field(default_factory=list)
    conformers: list[Conformer] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_heavy(self) -> int:
        return int((np.asarray(self.numbers) > 1).sum())

    def levels(self) -> dict[str, int]:
        """Level -> number of conformers with an energy at it."""
        out: dict[str, int] = {}
        for c in self.conformers:
            for lv in c.energies:
                out[lv] = out.get(lv, 0) + 1
        return out

    def energies(self, level: str) -> np.ndarray:
        return np.array([c.energies.get(level, np.nan) for c in self.conformers], float)

    def rel(self, level: str) -> np.ndarray:
        e = self.energies(level)
        return e - np.nanmin(e) if np.isfinite(e).any() else e

    def best(self, level: str) -> int | None:
        e = self.energies(level)
        return int(np.nanargmin(e)) if np.isfinite(e).any() else None


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def charge_class(q: int) -> str:
    return "neutral" if q == 0 else "anion" if q < 0 else "cation"


def level_quality(level: str) -> int:
    """How much a level is trusted to rank conformers (higher first): Spartan's DLXD2XV (its own ranking of an Est.
    Density Functional search) 50, method/basis (HF, DFT) 40, semi-empirical and xTB and ML potentials 30, other 20,
    force fields and conformer-search energies 10. Common levels are never a side's own level (0)."""
    u = level.upper()
    if level.startswith(COMMON):
        return 0
    if "(SEARCH)" in u:
        return 10
    if "DLXD2XV" in u:
        return 50
    if "/" in u:
        return 40
    if any(k in u for k in ("GFN", "XTB", "PM6", "PM7", "AM1", "AIMNET", "MOPAC")):
        return 30
    if any(k in u for k in ("MMFF", "UFF", "DLFF", "GFNFF", "MM ")):
        return 10
    return 20


def pick_level(cs: ConformerSet, prefer: str = "") -> str:
    """The level a side ranks its conformers by: the first level containing ``prefer`` (case-insensitive) when given,
    else the best by :func:`level_quality`, ties to the level with more conformers. "" without energies."""
    levels = cs.levels()
    if prefer:
        hit = [lv for lv in levels if prefer.lower() in lv.lower()]
        if hit:
            return max(hit, key=lambda lv: (levels[lv], lv))
    own = [lv for lv in levels if not lv.startswith(COMMON)]
    return max(own, key=lambda lv: (level_quality(lv), levels[lv], lv)) if own else ""


def common_level(phase: str, relaxed: bool = False) -> str:
    gas = phase in ("", "gas", "none")
    return f"{COMMON} {'gas' if gas else f'ALPB({phase})'} {'opt' if relaxed else 'SP'}"


# ------------------------------------------------------------------------------------------------ graphs
def _surrogate(labels: Sequence[int], edges: Iterable[tuple[int, int]]):
    """An RDKit molecule whose atomic numbers are ``labels`` (any positive ints < 119) and whose bonds are single:
    the graph that substructure matching enumerates automorphisms and isomorphisms on."""
    from rdkit import Chem

    rw = Chem.RWMol()
    for z in labels:
        a = Chem.Atom(int(z))
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    for i, j in edges:
        if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
            rw.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return m


def atom_map(a: ConformerSet, b: ConformerSet) -> np.ndarray | None:
    """``m`` with ``m[i]`` = the atom of ``b`` that is atom ``i`` of ``a`` (one graph isomorphism of the
    connectivity; any one, symmetry is handled per conformer pair), None when the graphs differ."""
    if len(a.numbers) != len(b.numbers) or len(a.bonds) != len(b.bonds):
        return None
    from rdkit import rdBase

    sa, sb = _surrogate(a.numbers, a.bonds), _surrogate(b.numbers, b.bonds)
    with rdBase.BlockLogs():
        m = sb.GetSubstructMatch(sa)
    return np.asarray(m, int) if len(m) == len(a.numbers) else None


# ------------------------------------------------------------------------------------------------ symmetry
def _kabsch(ref: np.ndarray, prb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with prb @ R + t the best (proper) fit of ``prb`` onto ``ref``."""
    pc, rc = prb.mean(0), ref.mean(0)
    h = (prb - pc).T @ (ref - rc)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return r, rc - pc @ r


def _rmsd_batch(ref: np.ndarray, prb: np.ndarray) -> np.ndarray:
    """Kabsch RMSD of ``ref`` (n, 3) to each of ``prb`` (K, n, 3) after the best proper rotation."""
    rc = ref - ref.mean(0)
    pc = prb - prb.mean(1, keepdims=True)
    h = np.einsum("kni,nj->kij", pc, rc)
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    s[:, -1] *= np.where(d == 0, 1.0, d)
    e = (pc ** 2).sum((1, 2)) + (rc ** 2).sum() - 2.0 * s.sum(1)
    return np.sqrt(np.maximum(e, 0.0) / len(ref))


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n = np.linalg.norm(b1)
    if n < 1e-8:
        return 0.0
    b1 = b1 / n
    v, w = b0 - (b0 @ b1) * b1, b2 - (b2 @ b1) * b1
    return float(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))


_PERMS = {k: np.array(list(itertools.permutations(range(k))), int) for k in range(1, 5)}


@dataclass
class Alignment:
    rmsd: float
    perm: np.ndarray            # heavy-atom map: reference heavy atom k <-> probe heavy atom perm[k]
    mirrored: bool
    config_ok: bool             # every stereocentre keeps its configuration under the map
    method: str                 # "exact" | "reduced" | "trivial"
    n_maps: int                 # mappings tried


class SymmetryModel:
    """One molecule's heavy atoms in skeleton and sibling groups, the automorphisms of the reduced graph, its
    torsions and stereocentres (module docstring). Indices of ``skeleton``, ``groups``, ``perm`` and the torsions
    are heavy-atom indices (positions in ``heavy``)."""

    def __init__(self, numbers: Sequence[int], bonds: Iterable[tuple[int, int]], max_maps: int = MAX_REDUCED_MAPS):
        z = np.asarray(numbers, int)
        n = len(z)
        nbrs: list[set[int]] = [set() for _ in range(n)]
        for i, j in bonds:
            i, j = int(i), int(j)
            if i != j and 0 <= i < n and 0 <= j < n:
                nbrs[i].add(j)
                nbrs[j].add(i)
        self.numbers = z
        self.heavy = np.flatnonzero(z > 1)
        hidx = {int(a): k for k, a in enumerate(self.heavy)}
        nh = len(self.heavy)
        self.n_h_on = [sum(1 for j in nbrs[a] if z[j] == 1) for a in self.heavy]
        self.hnbrs: list[list[int]] = [sorted(hidx[j] for j in nbrs[a] if z[j] > 1) for a in self.heavy]
        self.label = [(int(z[a]), self.n_h_on[k]) for k, a in enumerate(self.heavy)]
        # sibling groups: terminal heavy atoms with one label on one parent
        groups: list[list[int]] = []
        parents: list[int] = []
        for p in range(nh):
            buckets: dict[tuple, list[int]] = {}
            for t in self.hnbrs[p]:
                if len(self.hnbrs[t]) == 1:
                    buckets.setdefault(self.label[t], []).append(t)
            for members in buckets.values():
                if len(members) >= 2:
                    groups.append(sorted(members))
                    parents.append(p)
        in_group = {t for g in groups for t in g}
        self.groups, self.group_parent = groups, parents
        self.skeleton = [k for k in range(nh) if k not in in_group]
        self.polar_h = {hidx[int(a)]: next(j for j in nbrs[a] if z[j] == 1) for a in self.heavy
                        if z[a] in (8, 16) and len(self.hnbrs[hidx[int(a)]]) == 1 and self.n_h_on[hidx[int(a)]] == 1}
        # the reduced graph: skeleton atoms, then one pseudo-atom per group
        s_index = {k: r for r, k in enumerate(self.skeleton)}
        n_s = len(self.skeleton)
        labels = [("s", *self.label[k]) for k in self.skeleton] + \
                 [("g", *self.label[g[0]], len(g)) for g in groups]
        edges = [(s_index[k], s_index[m]) for k in self.skeleton for m in self.hnbrs[k] if m in s_index and k < m]
        edges += [(n_s + gi, s_index[parents[gi]]) for gi in range(len(groups))]
        codes = {lab: 1 + i for i, lab in enumerate(sorted(set(labels)))}
        self.reduced_auts, self.truncated = self._automorphisms([codes[lab] for lab in labels], edges, max_maps)
        self.n_full = len(self.reduced_auts) * math.prod(math.factorial(len(g)) for g in groups)
        self._full: np.ndarray | None = None
        self._ring_bonds = self._rings(nh, edges=[(k, m) for k in range(nh) for m in self.hnbrs[k] if k < m])
        self.torsions = self._torsions()
        self.stereocentres = self._stereocentres(nbrs)

    # -- graph helpers
    @staticmethod
    def _automorphisms(labels: list[int], edges: list[tuple[int, int]], cap: int) -> tuple[np.ndarray, bool]:
        if not labels:
            return np.zeros((1, 0), int), False
        from rdkit import rdBase

        if max(labels) > 118:      # more distinct labels than elements: no symmetry search (never seen)
            return np.arange(len(labels))[None, :], True
        m = _surrogate(labels, edges)
        with rdBase.BlockLogs():
            matches = m.GetSubstructMatches(m, uniquify=False, maxMatches=cap, useChirality=False)
        if not matches:
            return np.arange(len(labels))[None, :], False
        # a match t maps query atom r to t[r]: the reference's reduced point r <-> the probe's t[r]
        return np.asarray(matches, int), len(matches) >= cap

    @staticmethod
    def _rings(nh: int, edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
        """Heavy-atom bonds in a ring (a bond is in a ring when its ends stay connected without it)."""
        adj: dict[int, set[int]] = {k: set() for k in range(nh)}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        out = set()
        for i, j in edges:
            seen, todo = {i}, [i]
            while todo:
                x = todo.pop()
                for y in adj[x]:
                    if (x, y) in ((i, j), (j, i)) or y in seen:
                        continue
                    seen.add(y)
                    todo.append(y)
            if j in seen:
                out.add((min(i, j), max(i, j)))
        return out

    def _symmetric_top(self, b: int, c: int) -> bool:
        """All of b's substituents other than c are siblings (CF3, COO-, SO3-, N(CH3)3+) or hydrogens only."""
        others = [m for m in self.hnbrs[b] if m != c]
        if not others:
            return True
        g = next((g for g in self.groups if others[0] in g), None)
        return g is not None and sorted(others) == g and (self.n_h_on[b] == 0)

    def _choose_end(self, b: int, c: int) -> int | None:
        """The neighbor of b (not c) that defines a torsion about b-c: a skeleton atom with neighbors of its own
        first, then a unique terminal atom, then a sibling; the lowest index among equals."""
        others = [m for m in self.hnbrs[b] if m != c]
        if not others:
            return None
        in_group = {t for g in self.groups for t in g}

        def rank(m):
            if m not in in_group and len(self.hnbrs[m]) > 1:
                return (0, -self.label[m][0], m)
            if m not in in_group:
                return (1, -self.label[m][0], m)
            return (2, -self.label[m][0], m)

        return min(others, key=rank)

    def _torsions(self) -> list[tuple[int, int, int, int, str]]:
        """(a, b, c, d, kind): heavy indices; for ``polar_h`` d is the O/S whose hydrogen ends the torsion."""
        out = []
        nh = len(self.heavy)
        for b in range(nh):
            for c in self.hnbrs[b]:
                if c <= b or (b, c) in self._ring_bonds:
                    continue
                if len(self.hnbrs[b]) < 2 or len(self.hnbrs[c]) < 2:
                    continue
                a, d = self._choose_end(b, c), self._choose_end(c, b)
                if a is None or d is None:
                    continue
                kind = "rotor" if self._symmetric_top(b, c) or self._symmetric_top(c, b) else "backbone"
                out.append((a, b, c, d, kind))
        for o in self.polar_h:
            b = self.hnbrs[o][0]
            a = self._choose_end(b, o)
            if a is not None:
                out.append((a, b, o, o, "polar_h"))
        return out

    def _stereocentres(self, nbrs: list[set[int]]) -> list[tuple[int, int, int, int]]:
        """(centre, n1, n2, n3) heavy indices: atoms with four neighbors of four different symmetry classes, at
        least three of them heavy."""
        from rdkit import Chem

        z = self.numbers
        edges = [(i, j) for i in range(len(z)) for j in nbrs[i] if i < j]
        try:
            ranks = list(Chem.CanonicalRankAtoms(_surrogate(z, edges), breakTies=False))
        except Exception:  # noqa: BLE001 - no stereo check is an answer
            return []
        hidx = {int(a): k for k, a in enumerate(self.heavy)}
        out = []
        for k, a in enumerate(self.heavy):
            nb = sorted(nbrs[int(a)])
            heavy_nb = [j for j in nb if z[j] > 1]
            if len(nb) != 4 or len(heavy_nb) < 3 or len({ranks[j] for j in nb}) != 4:
                continue
            heavy_nb.sort(key=lambda j: ranks[j])
            out.append((k, hidx[heavy_nb[0]], hidx[heavy_nb[1]], hidx[heavy_nb[2]]))
        return out

    # -- mappings
    def _base_perm(self, sigma: np.ndarray) -> np.ndarray:
        """The heavy-atom map of one reduced automorphism, each group's members in their listed order."""
        nh = len(self.heavy)
        n_s = len(self.skeleton)
        p = np.arange(nh)
        for r, k in enumerate(self.skeleton):
            p[k] = self.skeleton[int(sigma[r])]
        for gi, g in enumerate(self.groups):
            p[g] = self.groups[int(sigma[n_s + gi]) - n_s]
        return p

    def full_maps(self, cap: int | None = None) -> tuple[np.ndarray, bool]:
        """Every heavy-atom automorphism (reduced ones x the permutations inside each group), at most ``cap``."""
        if self._full is not None and cap is None:
            return self._full, False
        perms = np.array([self._base_perm(s) for s in self.reduced_auts], int)
        truncated = False
        for g in self.groups:
            k = len(g)
            ps = _PERMS[k] if k in _PERMS else np.array(list(itertools.permutations(range(k))), int)
            if cap is not None and len(perms) * len(ps) > cap:
                truncated = True
                break
            cur = perms[:, g]
            new = np.repeat(perms, len(ps), axis=0)
            new[:, g] = cur[:, ps].reshape(-1, k)
            perms = new
        if cap is None:
            self._full = perms
        return perms, truncated

    def reduced_points(self, posh: np.ndarray) -> np.ndarray:
        pts = [posh[self.skeleton]] if self.skeleton else []
        if self.groups:
            pts.append(np.array([posh[g].mean(0) for g in self.groups]))
        return np.vstack(pts) if pts else np.zeros((0, 3))

    def degenerate(self, posh: np.ndarray) -> bool:
        """Reduced points (nearly) on one line: the alignment on them leaves a rotation undetermined."""
        red = self.reduced_points(posh)
        if len(red) < 3:
            return True
        s = np.linalg.svd(red - red.mean(0), compute_uv=False)
        return bool(s[1] < 0.3)

    def config_agrees(self, ref_h: np.ndarray, prb_h: np.ndarray, perm: np.ndarray) -> bool:
        for c, a, b, d in self.stereocentres:
            vr = np.linalg.det(np.array([ref_h[a] - ref_h[c], ref_h[b] - ref_h[c], ref_h[d] - ref_h[c]]))
            q = perm
            vp = np.linalg.det(np.array([prb_h[q[a]] - prb_h[q[c]], prb_h[q[b]] - prb_h[q[c]],
                                         prb_h[q[d]] - prb_h[q[c]]]))
            if abs(vr) > 0.1 and abs(vp) > 0.1 and np.sign(vr) != np.sign(vp):
                return False
        return True


def _assign_groups(model: SymmetryModel, ref_h: np.ndarray, prb_t: np.ndarray, perm: np.ndarray,
                   sigma: np.ndarray) -> np.ndarray:
    """Re-assign the atoms inside every group to their nearest partners (``prb_t`` already fitted)."""
    from scipy.optimize import linear_sum_assignment

    out = perm.copy()
    n_s = len(model.skeleton)
    for gi, g in enumerate(model.groups):
        img = np.asarray(model.groups[int(sigma[n_s + gi]) - n_s])
        d2 = ((ref_h[g][:, None, :] - prb_t[img][None, :, :]) ** 2).sum(-1)
        k = len(g)
        if k in _PERMS:
            ps = _PERMS[k]
            best = ps[int(np.argmin(d2[np.arange(k)[None, :], ps].sum(1)))]
            out[g] = img[best]
        else:
            r, c = linear_sum_assignment(d2)
            out[np.asarray(g)[r]] = img[c]
    return out


def _reduced_fit(model: SymmetryModel, ref_h: np.ndarray, prb_h: np.ndarray) -> tuple[float, np.ndarray, int]:
    red_r = model.reduced_points(ref_h)
    red_p = model.reduced_points(prb_h)
    skel = model.skeleton
    best = (math.inf, None)
    n_try = 0
    for sigma in model.reduced_auts:
        base = model._base_perm(sigma)
        # three starts: the reduced points, the skeleton alone, every heavy atom in the base order; the first is
        # exact for similar conformers, the others keep very different ones out of a local minimum
        starts = [_kabsch(red_r, red_p[sigma]), _kabsch(ref_h, prb_h[base])]
        if len(skel) >= 3:
            starts.append(_kabsch(ref_h[skel], prb_h[base[skel]]))
        for rot, t in starts:
            n_try += 1
            perm = base
            for _it in range(8):
                new = _assign_groups(model, ref_h, prb_h @ rot + t, perm, sigma)
                rot, t = _kabsch(ref_h, prb_h[new])
                if np.array_equal(new, perm):
                    break
                perm = new
            rms = float(np.sqrt(((prb_h[perm] @ rot + t - ref_h) ** 2).sum(1).mean()))
            if rms < best[0]:
                best = (rms, perm)
    return best[0], best[1], n_try


def symmetric_rmsd(model: SymmetryModel, ref: np.ndarray, probe: np.ndarray, mirror: bool = True) -> Alignment:
    """Symmetric heavy-atom RMSD of two conformers in one atom order (``ref``, ``probe``: (N, 3) Å), over the
    molecule's automorphisms and, with ``mirror``, the probe's mirror image (module docstring). A mapping that
    keeps every stereocentre's configuration beats one that does not, whatever their RMSDs."""
    ref_h = np.asarray(ref, float)[model.heavy]
    prb0 = np.asarray(probe, float)[model.heavy]
    nh = len(ref_h)
    if nh <= 1:
        return Alignment(0.0, np.arange(nh), False, True, "trivial", 1)
    found: list[Alignment] = []
    for mirrored in (False, True) if mirror else (False,):
        prb_h = prb0 * np.array([-1.0, 1.0, 1.0]) if mirrored else prb0
        if model.n_full <= EXACT_MAPS or model.degenerate(ref_h):
            perms, _trunc = model.full_maps() if model.n_full <= EXACT_MAPS else model.full_maps(cap=EXACT_MAPS)
            rms = _rmsd_batch(ref_h, prb_h[perms])
            order = np.argsort(rms)
            k = int(order[0])
            if model.stereocentres:            # the best map that keeps the configuration, if any is close
                for kk in order[:64]:
                    if model.config_agrees(ref_h, prb_h, perms[int(kk)]):
                        k = int(kk)
                        break
            found.append(Alignment(float(rms[k]), perms[k].copy(), mirrored,
                                   model.config_agrees(ref_h, prb_h, perms[k]), "exact", len(perms)))
        else:
            rms, perm, n = _reduced_fit(model, ref_h, prb_h)
            found.append(Alignment(rms, perm, mirrored, model.config_agrees(ref_h, prb_h, perm), "reduced", n))
    return min(found, key=lambda a: (not a.config_ok, a.rmsd))


def torsion_deltas(model: SymmetryModel, ref: np.ndarray, probe: np.ndarray, al: Alignment) -> dict[str, list[float]]:
    """|Δφ| (degrees) of every torsion of the model between ``ref`` and ``probe`` under the alignment's map (the
    probe reflected when the alignment is mirrored), by kind: backbone, rotor, polar_h."""
    ref = np.asarray(ref, float)
    prb = np.asarray(probe, float) * (np.array([-1.0, 1.0, 1.0]) if al.mirrored else 1.0)
    hv, perm = model.heavy, al.perm
    out: dict[str, list[float]] = {"backbone": [], "rotor": [], "polar_h": []}
    for a, b, c, d, kind in model.torsions:
        if kind == "polar_h":
            hr, hp = model.polar_h[c], model.polar_h.get(int(perm[c]))
            if hp is None:
                continue
            fr = _dihedral(ref[hv[a]], ref[hv[b]], ref[hv[c]], ref[hr])
            fp = _dihedral(prb[hv[perm[a]]], prb[hv[perm[b]]], prb[hv[perm[c]]], prb[hp])
        else:
            fr = _dihedral(*(ref[hv[x]] for x in (a, b, c, d)))
            fp = _dihedral(*(prb[hv[perm[x]]] for x in (a, b, c, d)))
        out[kind].append(abs((fr - fp + 180.0) % 360.0 - 180.0))
    return out


@dataclass
class PairResult:
    rmsd: float
    mirrored: bool
    config_ok: bool
    torsion_max: float          # backbone and polar-H torsions (the criterion)
    torsion_mean: float         # backbone
    rotor_max: float
    polar_h_max: float
    same: bool


def compare_pair(model: SymmetryModel, ref: np.ndarray, probe: np.ndarray, rmsd_cut: float = DEFAULT_RMSD_A,
                 torsion_cut: float = DEFAULT_TORSION_DEG, mirror: bool = True) -> PairResult:
    al = symmetric_rmsd(model, ref, probe, mirror)
    td = torsion_deltas(model, ref, probe, al)
    crit = td["backbone"] + td["polar_h"]
    tmax = max(crit) if crit else 0.0
    same = al.config_ok and al.rmsd <= rmsd_cut and (torsion_cut <= 0 or tmax <= torsion_cut)
    return PairResult(al.rmsd, al.mirrored, al.config_ok, tmax,
                      float(np.mean(td["backbone"])) if td["backbone"] else 0.0,
                      max(td["rotor"]) if td["rotor"] else 0.0, max(td["polar_h"]) if td["polar_h"] else 0.0, same)


# ------------------------------------------------------------------------------------------------ readers
def _merge_into(base: ConformerSet, other: ConformerSet) -> bool:
    """Add ``other``'s conformers to ``base`` (same constitution) in ``base``'s atom order; False when the graphs
    cannot be mapped."""
    if base.numbers.tolist() == other.numbers.tolist() and sorted(map(tuple, map(sorted, base.bonds))) == \
            sorted(map(tuple, map(sorted, other.bonds))):
        m = np.arange(len(base.numbers))
    else:
        m = atom_map(base, other)
        if m is None:
            return False
    for c in other.conformers:
        c.positions = np.asarray(c.positions, float)[m]
        if "relaxed" in c.extra:
            c.extra["relaxed"] = np.asarray(c.extra["relaxed"], float)[m]
        base.conformers.append(c)
    for lab in other.labels:
        if lab and lab not in base.labels:
            base.labels.append(lab)
    for s in other.sources:
        if s not in base.sources:
            base.sources.append(s)
    base.label = base.label or other.label
    base.name = base.name or other.name
    base.inchikey = base.inchikey or other.inchikey
    base.group = base.group or other.group
    base.phase = base.phase or other.phase
    base.notes += [n for n in other.notes if n not in base.notes]
    return True


def _fingerprint(pos: np.ndarray) -> np.ndarray:
    """Sorted heavy-atom distances: the same for any atom numbering, rotation and mirror image."""
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    return np.sort(d[np.triu_indices(len(pos), 1)])


def _dedupe(cs: ConformerSet) -> None:
    """Conformers with the same geometry (symmetric RMSD < DUPLICATE_RMSD_A, not mirrored: the copies a DFT_Top5
    document makes of All_Conformers ones) become one, with both energies. Mirror images stay two conformers, as
    the program listed them."""
    kept: list[Conformer] = []
    prints: list[np.ndarray] = []
    heavy = np.asarray(cs.numbers) > 1
    model: SymmetryModel | None = None
    for c in cs.conformers:
        dup = None
        fp = _fingerprint(c.positions[heavy])
        for k, kfp in zip(kept, prints):
            if len(fp) != len(kfp) or (len(fp) and np.abs(fp - kfp).max() > 4 * DUPLICATE_RMSD_A):
                continue
            if model is None:
                model = SymmetryModel(cs.numbers, cs.bonds)
            if symmetric_rmsd(model, k.positions, c.positions, mirror=False).rmsd < DUPLICATE_RMSD_A:
                dup = k
                break
        if dup is None:
            kept.append(c)
            prints.append(fp)
            continue
        for lv, v in c.energies.items():
            dup.energies.setdefault(lv, v)
        for lv, v in c.dipoles.items():
            dup.dipoles.setdefault(lv, v)
        if c.source and c.source not in dup.source.split(" + "):
            dup.source += f" + {c.source}"
        dup.tag = dup.tag or c.tag
    cs.conformers = kept


def merge_sets(sets: Iterable[ConformerSet]) -> list[ConformerSet]:
    """One set per constitution key (the first set's atom order), same-geometry conformers merged."""
    by_key: dict[str, ConformerSet] = {}
    out: list[ConformerSet] = []
    for s in sets:
        if not s.key or not s.conformers:
            continue
        base = by_key.get(s.key)
        if base is None:
            by_key[s.key] = s
            out.append(s)
        elif not _merge_into(base, s):
            out.append(s)          # same key, unmappable graph (cannot happen for a real key): kept apart
    for s in out:
        _dedupe(s)
    return out


def _mol_graph(mol) -> tuple[np.ndarray, list[tuple[int, int]]]:
    return (np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int),
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])


def read_run(folder: str | Path) -> list[ConformerSet]:
    """A QuantaBench run folder (or one machine's share of a grid run): every molecule folder with
    ``<id>_summary.json`` and ``<id>_conformers.sdf`` whose status is ok."""
    from . import export
    from .io_spartan import constitution_key

    folder = Path(folder)
    out = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if not (sub / f"{sub.name}_summary.json").is_file():
            continue
        try:
            res = export.read_molecule(sub, sub.name)
        except Exception:  # noqa: BLE001, S112 - an unreadable molecule is skipped, not fatal
            continue
        if res.status != "ok" or res.mol is None or not res.conformers:
            continue
        numbers, bonds = _mol_graph(res.mol)
        phase = res.solvent or "gas"
        method = (res.descriptors or {}).get("method") or res.method
        level = f"QuantaBench {method} {phase}"
        crest = res.conformer_search == "crest"
        cs = ConformerSet(key=constitution_key(numbers, bonds, res.charge), numbers=numbers, bonds=bonds,
                          charge=int(res.charge), multiplicity=int(res.multiplicity or 1),
                          program="QuantaBench (CREST)" if crest else "QuantaBench", label=res.input.id,
                          labels=[res.input.id], name=res.input.name or "", inchikey=res.inchikey or "",
                          formula=str((res.descriptors or {}).get("formula", "")), phase=phase,
                          sources=[str(folder)])
        for c in res.conformers:
            pos = np.array(res.mol.GetConformer(c.conf_id).GetPositions(), float)
            conf = Conformer(pos, {level: float(c.energy_ev) * KJ_PER_EV}, source=f"{folder.name}/{sub.name}",
                             tag=f"rank {c.rank}")
            if c.dipole_debye is not None:
                conf.dipoles[level] = float(c.dipole_debye)
            cs.conformers.append(conf)
        out.append(cs)
    return out


_CONF_DOC = re.compile(r"_Conf\.(.+)$")


def _doc_label(source_file: str) -> str:
    """The molecule label in a Spartan per-molecule conformer document's name (``..._Conf.<label>.spartan``)."""
    stem = Path(re.split(r"[/\\]", source_file)[-1]).stem
    m = _CONF_DOC.search(stem)
    return m.group(1).strip() if m else ""


def _spartan_values(row: dict) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """(energies kJ/mol, dipoles D, notes) of one Spartan table row (``io_spartan.molecule_row``, or a long
    table's row) by level (module docstring)."""
    e: dict[str, float] = {}
    d: dict[str, float] = {}
    notes: list[str] = []
    status = str(row.get("spartan_status") or "")
    task = str(row.get("spartan_task") or "").upper()
    solvent = str(row.get("spartan_solvent") or "")
    suffix = f" {solvent}" if solvent else ""
    method = str(row.get("spartan_method") or "")
    basis = str(row.get("spartan_basis") or "")
    sp_level = ""
    if status.startswith("failed"):
        notes.append(f"Spartan job {status}: no energies at its level")
    elif task.startswith("OPT") and status == "completed":
        notes.append(f"{method} OPT: its energy belongs to the optimized geometry, which is not read")
    else:
        post = _f(row.get("spartan_postenergy"))
        if post is not None:
            e[f"Spartan {row.get('spartan_postenergyat') or 'POSTENERGY'}{suffix}"] = post * KJ_PER_HARTREE
        sp = _f(row.get("spartan_spenergy"))
        if sp is not None:
            sp_level = f"Spartan {row.get('spartan_spenergyat') or (method + ('/' + basis if basis else ''))}{suffix}"
            e[sp_level] = sp * KJ_PER_HARTREE
        ek = _f(row.get("spartan_e_kcal"))
        if sp is None and post is None and ek is not None and status == "completed":
            sp_level = f"Spartan {method}{'/' + basis if basis else ''}{suffix}"
            e[sp_level] = ek * KJ_PER_KCAL
        dip = _f(row.get("dipole_debye"))
        if dip is not None and status == "completed":
            lv = str(row.get("qm_dipole_level") or "")
            d[f"Spartan {lv}{suffix}" if lv else (sp_level or f"Spartan {method}{suffix}")] = dip
    dc = _f(row.get("spartan_dc_list_energy_kcal"))
    if dc is not None:
        e[f"Spartan {row.get('spartan_optat') or 'MM'} (search)"] = dc * KJ_PER_KCAL
    ce = _f(row.get("spartan_conf_energy_kcal"))
    if ce is not None:
        e[f"Spartan {row.get('spartan_conf_method') or method} (search)"] = ce * KJ_PER_KCAL
    return e, d, notes


def _spartan_set(row: dict, numbers, bonds, positions, label: str) -> ConformerSet | None:
    from .io_spartan import constitution_key, perceived_bonds
    from .validate import cohort_of

    charge = int(_f(row.get("charge")) or 0)
    if not bonds:
        bonds = perceived_bonds(numbers, positions) or []
    key = constitution_key(numbers, bonds, charge)
    if not key:
        return None
    e, d, notes = _spartan_values(row)
    doc = re.split(r"[/\\]", str(row.get("source_file", "")))[-1]
    src = f"{doc} {row.get('storage', '')}"
    if row.get("conformer") not in (None, ""):
        src += f"[{row['conformer']}]"
    conf = Conformer(np.asarray(positions, float), e, d, source=src.strip(), tag=str(row.get("spartan_torlabel") or "").strip())
    return ConformerSet(key=key, numbers=np.asarray(numbers, int), bonds=[(int(i), int(j)) for i, j in bonds],
                        charge=charge, multiplicity=int(_f(row.get("multiplicity")) or 1), program="Spartan",
                        label=label, labels=[label] if label else [], inchikey=str(row.get("inchikey") or ""),
                        formula=str(row.get("formula") or ""), group=cohort_of(str(row.get("source_file", ""))),
                        sources=[str(row.get("source_file", ""))], conformers=[conf], notes=notes)


def _included(source_file: str, include: Sequence[str]) -> bool:
    if not include:
        return True
    name = re.split(r"[/\\]", source_file)[-1]
    return any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(source_file, pat) for pat in include)


def _row_label(row: dict) -> str:
    return str(row.get("label_resolved") or "") or _doc_label(str(row.get("source_file", ""))) or \
        (str(row.get("cas_candidates") or "").split(";")[0])


def read_spartan_documents(paths: Iterable[str | Path], include: Sequence[str] = (),
                           log: Callable[[str], None] | None = None) -> list[ConformerSet]:
    """Spartan documents (``.spartan``): one conformer per saved conformer of a conformer search, else per
    molecule (its Input geometry). Labels: the molecule's own, the document name's ``_Conf.<label>``, else resolved
    by structure from a base document among ``paths`` (``io_spartan.LabelIndex``). Only documents whose name
    matches one of ``include`` (fnmatch patterns; all when empty) give conformers; all of them give labels."""
    from . import io_spartan as sp

    paths = [str(p) for p in paths]
    mols = []
    index = sp.LabelIndex()
    for path in paths:
        errors: list[str] = []
        n = 0
        for mol in sp.iter_molecules(path, path, errors):
            n += 1
            key = sp._cached_structure_key(mol)
            if sp.has_own_label(mol.label):
                index.add(path, key, mol.label)
            if _included(path, include):
                mols.append(mol)
        if log:
            log(f"{Path(path).name}: {n} molecules" + (f", {len(errors)} problems" if errors else ""))
    out = []
    for mol in mols:
        own = mol.label if sp.has_own_label(mol.label) else ""
        label = own or _doc_label(mol.source_file)
        if not label:
            lab, _src, _note = index.resolve(mol.source_file, sp._cached_structure_key(mol))
            label = lab
        ks = range(mol.n_conformers) if mol.conformer_positions_A is not None else [None]
        for k in ks:
            row = sp.molecule_row(mol, k)
            row["label_resolved"] = label
            bonds = [(i, j) for i, j, _o in mol.input.bonds]
            cs = _spartan_set(row, mol.numbers, bonds, mol.positions_A(k), label)
            if cs is not None:
                out.append(cs)
    return out


def read_spartan_import(table: str | Path, sdf: str | Path, include: Sequence[str] = ()) -> list[ConformerSet]:
    """The long table and SDF of ``quantabench-cli import ... --conformers all --table T --sdf S``: each SDF record
    (geometry, bonds) joined to its table row by source file, molecule index and conformer."""
    from rdkit import Chem, rdBase

    csv.field_size_limit(1 << 30)
    with Path(table).open(newline="", encoding="utf-8-sig") as fh:
        rows = {(r.get("source_file", ""), str(r.get("molecule_index", "")), str(r.get("conformer", "") or "")): r
                for r in csv.DictReader(fh)}
    out = []
    with rdBase.BlockLogs():
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
        for m in supplier:
            if m is None:
                continue
            key = tuple(m.GetProp(p) if m.HasProp(p) else "" for p in ("source_file", "molecule_index", "conformer"))
            row = rows.get(key)
            if row is None or not _included(row.get("source_file", ""), include):
                continue
            numbers, bonds = _mol_graph(m)
            cs = _spartan_set(row, numbers, bonds, m.GetConformer().GetPositions(), _row_label(row))
            if cs is not None:
                out.append(cs)
    return out


def _is_run_folder(p: Path) -> bool:
    if (p / "summary.csv").is_file() or (p / "jobs.sqlite").is_file():
        return True
    return any((s / f"{s.name}_summary.json").is_file() for s in p.iterdir() if s.is_dir())


def read_sources(paths: Iterable[str | Path], include: Sequence[str] = (),
                 log: Callable[[str], None] | None = None) -> list[ConformerSet]:
    """Conformer sets from any mix of: QuantaBench run folders (and folders of them, one level down: a grid run's
    phase folder), ``.spartan`` documents (and folders of them), and a Spartan import's long table ``.csv`` with its
    ``.sdf`` (given together, paired in order). Merged by constitution (:func:`merge_sets`)."""
    sets: list[ConformerSet] = []
    docs: list[Path] = []
    tables: list[Path] = []
    sdfs: list[Path] = []
    for p in map(lambda x: Path(x).expanduser(), paths):
        if p.is_dir():
            if _is_run_folder(p):
                sets += read_run(p)
                if log:
                    log(f"{p}: QuantaBench run")
                continue
            subs = [s for s in sorted(p.iterdir()) if s.is_dir() and _is_run_folder(s)]
            for s in subs:
                sets += read_run(s)
                if log:
                    log(f"{s}: QuantaBench run")
            docs += sorted(p.glob("*.spartan"))
            if not subs and not list(p.glob("*.spartan")):
                raise ValueError(f"{p}: neither a QuantaBench run folder nor a folder of run folders or .spartan files")
        elif p.suffix.lower() == ".spartan":
            docs.append(p)
        elif p.suffix.lower() == ".csv":
            tables.append(p)
        elif p.suffix.lower() == ".sdf":
            sdfs.append(p)
        else:
            raise ValueError(f"{p}: not a run folder, .spartan document, or import table (.csv + .sdf)")
    if len(tables) != len(sdfs):
        raise ValueError("a Spartan import is a long table (.csv) and its SDF (.sdf), given together")
    for t, s in zip(tables, sdfs):
        got = read_spartan_import(t, s, include)
        if log:
            log(f"{t.name} + {s.name}: {len(got)} conformers")
        sets += got
    if docs:
        sets += read_spartan_documents(docs, include, log)
    return merge_sets(sets)


# ------------------------------------------------------------------------------------------------ common level
def _rescore_key(numbers, positions, charge, mult, phases, relax, fmax, steps) -> str:
    h = hashlib.sha1(usedforsecurity=False)      # a cache key, not security
    h.update(np.asarray(numbers, np.int64).tobytes())
    h.update(np.round(np.asarray(positions, float), 4).tobytes())
    h.update(json.dumps([int(charge), int(mult), list(phases), relax, fmax, steps]).encode())
    return h.hexdigest()


def _dipole_com(numbers, positions, mu_e_ang, charge) -> float:
    from ase.data import atomic_masses

    m = atomic_masses[np.asarray(numbers, int)]
    com = (np.asarray(positions, float) * m[:, None]).sum(0) / m.sum()
    return float(np.linalg.norm(np.asarray(mu_e_ang, float) - float(charge) * com) * DEBYE_PER_E_ANG)


def _init_worker() -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def rescore_one(task: dict) -> dict:
    """GFN2-xTB (tblite) single points of one geometry in each phase, and an optional optimization in ``relax``:
    energies (kJ/mol), dipoles about the centre of mass (D), the relaxed positions. Errors become ``error_*`` keys."""
    from .eng_tblite import TBLiteCalculator

    numbers, pos = np.asarray(task["numbers"], int), np.asarray(task["positions"], float)
    q, mult = int(task["charge"]), int(task["multiplicity"])
    out: dict = {"key": task["key"]}
    for phase in task["phases"]:
        lv = common_level(phase)
        try:
            calc = TBLiteCalculator(numbers, "GFN2-xTB", q, mult, solvent="" if phase == "gas" else phase)
            res = calc.singlepoint(pos)
            out[f"E|{lv}"] = float(res.get("energy")) * KJ_PER_HARTREE
            out[f"D|{lv}"] = _dipole_com(numbers, pos, np.asarray(res.get("dipole"), float) * BOHR_ANG, q)
        except Exception as exc:  # noqa: BLE001 - one failed SCF is a missing value
            out[f"error|{lv}"] = f"{type(exc).__name__}: {exc}"
    relax = task.get("relax")
    if relax:
        from ase import Atoms

        from .optimize import optimize

        lv = common_level(relax, relaxed=True)
        try:
            atoms = Atoms(numbers=numbers, positions=pos)
            atoms.calc = TBLiteCalculator(numbers, "GFN2-xTB", q, mult, solvent="" if relax == "gas" else relax)
            r = optimize(atoms, fmax=float(task["fmax"]), steps=int(task["steps"]))
            out[f"E|{lv}"] = r.energy_ev * KJ_PER_EV
            out[f"D|{lv}"] = _dipole_com(numbers, atoms.positions, atoms.calc.results["dipole"], q)
            out["relaxed"] = np.round(atoms.positions, 5).tolist()
            out["relax_converged"], out["relax_steps"] = bool(r.converged), int(r.steps)
        except Exception as exc:  # noqa: BLE001
            out[f"error|{lv}"] = f"{type(exc).__name__}: {exc}"
    return out


def rescore(sets: Sequence[ConformerSet], phases: Sequence[str] = ("gas", "water"), relax: str = "",
            fmax: float = 0.03, steps: int = 300, workers: int = 1, cache: str | Path | None = None,
            only: Callable[[ConformerSet, int], bool] | None = None,
            log: Callable[[str], None] | None = None) -> int:
    """Put the common level on every conformer of ``sets`` (``energies``/``dipoles`` under :func:`common_level`;
    ``extra["relaxed"]``): GFN2-xTB single points in each of ``phases`` ("gas" or a solvent name, ALPB) and, with
    ``relax`` (a phase), an LBFGS optimization to ``fmax`` (eV/Å) in at most ``steps`` steps (``only(set, index)``
    limits the optimizations). ``cache`` (JSON lines) keeps the results by geometry, so a rerun only computes new
    conformers. ``workers`` > 1 runs processes (spawned, one OpenMP thread each). Returns the number computed."""
    cache_path = Path(cache) if cache else None
    done: dict[str, dict] = {}
    if cache_path and cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(ValueError, KeyError):
                d = json.loads(line)
                done[d["key"]] = d
    todo: dict[str, dict] = {}
    where: dict[str, list[Conformer]] = {}
    for cs in sets:
        for k, c in enumerate(cs.conformers):
            rl = relax if relax and (only is None or only(cs, k)) else ""
            key = _rescore_key(cs.numbers, c.positions, cs.charge, cs.multiplicity, phases, rl, fmax, steps)
            where.setdefault(key, []).append(c)
            if key not in done and key not in todo:
                todo[key] = {"key": key, "numbers": cs.numbers.tolist(), "positions": c.positions.tolist(),
                             "charge": cs.charge, "multiplicity": cs.multiplicity, "phases": list(phases),
                             "relax": rl, "fmax": fmax, "steps": steps}
    tasks = list(todo.values())
    if log:
        log(f"common level: {len(where)} geometries, {len(where) - len(tasks)} cached, {len(tasks)} to compute"
            + (f" (optimization in {relax})" if relax else ""))
    fh = cache_path.open("a", encoding="utf-8") if cache_path else None
    try:
        def keep(res: dict) -> None:
            done[res["key"]] = res
            if fh:
                fh.write(json.dumps(res) + "\n")
                fh.flush()

        if workers > 1 and len(tasks) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     initializer=_init_worker) as ex:
                futs = [ex.submit(rescore_one, t) for t in tasks]
                for n, f in enumerate(as_completed(futs), 1):
                    keep(f.result())
                    if log and (n % 100 == 0 or n == len(futs)):
                        log(f"  common level: {n}/{len(futs)}")
        else:
            for n, t in enumerate(tasks, 1):
                keep(rescore_one(t))
                if log and (n % 100 == 0 or n == len(tasks)):
                    log(f"  common level: {n}/{len(tasks)}")
    finally:
        if fh:
            fh.close()
    radii = {id(cs): _covalent_radii(cs.numbers) for cs in sets}
    owner = {id(c): cs for cs in sets for c in cs.conformers}
    for key, confs in where.items():
        res = done.get(key)
        if not res:
            continue
        for c in confs:
            relaxed = res.get("relaxed")
            broken = False
            if relaxed is not None:
                r = radii[id(owner[id(c)])]
                broken = _distance_bonds(c.positions, r) != _distance_bonds(np.asarray(relaxed, float), r)
            for k, v in res.items():
                if broken and k[2:] == common_level(relax, True):
                    continue          # the optimization changed the bonds (a proton moved): not this conformer
                if k.startswith("E|"):
                    c.energies[k[2:]] = float(v)
                elif k.startswith("D|"):
                    c.dipoles[k[2:]] = float(v)
            if relaxed is not None and not broken:
                c.extra["relaxed"] = np.asarray(relaxed, float)
                c.extra["relax_converged"] = res.get("relax_converged")
            if broken:
                c.extra["relax_topology_changed"] = True
    return len(tasks)


def _covalent_radii(numbers) -> np.ndarray:
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    return np.array([pt.GetRcovalent(int(z)) for z in numbers], float)


def _distance_bonds(pos: np.ndarray, radii: np.ndarray, scale: float = 1.25) -> set[tuple[int, int]]:
    """Bonded pairs by distance (``chem.crest``'s rule: d < 1.25 x the sum of the covalent radii)."""
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    i, j = np.nonzero(np.triu(d < scale * (radii[:, None] + radii[None, :]), k=1))
    return set(zip(i.tolist(), j.tolist()))


# ------------------------------------------------------------------------------------------------ comparison
@dataclass
class MoleculeComparison:
    a: ConformerSet
    b: ConformerSet | None
    row: dict
    pairs: dict[tuple[int, int], PairResult] = field(default_factory=dict)
    level_rows: list[dict] = field(default_factory=list)
    conformer_rows: list[dict] = field(default_factory=list)


@dataclass
class Comparison:
    molecules: list[MoleculeComparison]
    only_a: list[ConformerSet]
    only_b: list[ConformerSet]
    settings: dict
    summary: list[dict] = field(default_factory=list)
    a_name: str = "A"
    b_name: str = "B"


def _window_cols(prefix: str, rel: np.ndarray, found: np.ndarray, windows: Sequence[float]) -> dict:
    out = {}
    for w in windows:
        inside = np.isfinite(rel) & (rel <= w + 1e-9)
        tag = f"{w:g}"
        out[f"{prefix}_n_win{tag}"] = int(inside.sum())
        out[f"{prefix}_found_win{tag}"] = int((inside & found).sum())
    return out


def _closest(pairs: dict, i: int, n: int, side: str) -> tuple[int | None, PairResult | None]:
    """The partner of conformer i (side "a": over b's j; "b": over a's) with the lowest RMSD among those that keep
    the configuration."""
    best = (None, None)
    for j in range(n):
        p = pairs[(i, j)] if side == "a" else pairs[(j, i)]
        if not p.config_ok:
            continue
        if best[1] is None or p.rmsd < best[1].rmsd:
            best = (j, p)
    return best


def _rel_of(cs: ConformerSet, level: str, k: int | None) -> float | None:
    if k is None or not level:
        return None
    r = cs.rel(level)
    return float(r[k]) if np.isfinite(r[k]) else None


def compare_molecule(a: ConformerSet, b: ConformerSet, *, level_a: str = "", level_b: str = "",
                     phase: str = "", rmsd_cut: float = DEFAULT_RMSD_A, torsion_cut: float = DEFAULT_TORSION_DEG,
                     mirror: bool = True, windows: Sequence[float] = DEFAULT_WINDOWS_KJ) -> MoleculeComparison:
    """All A x B conformer pairs of one molecule (``b`` already in ``a``'s atom order), the coverage, the best
    conformers, the cause and the dipoles (module docstring)."""
    model = SymmetryModel(a.numbers, a.bonds)
    la, lb = level_a or pick_level(a), level_b or pick_level(b)
    na, nb = len(a.conformers), len(b.conformers)
    pairs = {(i, j): compare_pair(model, a.conformers[i].positions, b.conformers[j].positions, rmsd_cut,
                                  torsion_cut, mirror) for i in range(na) for j in range(nb)}
    same = np.array([[pairs[(i, j)].same for j in range(nb)] for i in range(na)], bool).reshape(na, nb)
    found_a, found_b = same.any(1), same.any(0)
    ra, rb = a.rel(la) if la else np.full(na, np.nan), b.rel(lb) if lb else np.full(nb, np.nan)
    ia, ib = a.best(la) if la else None, b.best(lb) if lb else None
    row: dict = {"key": a.key, "label": a.label or b.label, "b_label": b.label,
                 "labels": ";".join(sorted(set(a.labels + b.labels))), "name": b.name or a.name,
                 "inchikey": a.inchikey or b.inchikey, "formula": a.formula or b.formula, "charge": a.charge,
                 "charge_class": charge_class(a.charge), "group": a.group or b.group, "n_heavy": a.n_heavy,
                 "n_atoms": len(a.numbers), "symmetry_maps": model.n_full, "symmetry_path":
                     "exact" if model.n_full <= EXACT_MAPS else "reduced", "program_a": a.program,
                 "program_b": b.program, "n_a": na, "n_b": nb,
                 "level_a": la, "level_b": lb, "a_best": "" if ia is None else ia, "b_best": "" if ib is None else ib}
    row.update(_window_cols("a", ra, found_a, windows))
    row.update(_window_cols("b", rb, found_b, windows))
    row["a_found_all"], row["b_found_all"] = int(found_a.sum()), int(found_b.sum())
    if ia is not None:
        ca = a.conformers[ia]
        row["a_best_source"], row["a_best_tag"] = ca.source, ca.tag
        row["a_best_found"] = bool(found_a[ia])
        j, p = _closest(pairs, ia, nb, "a")
        row["a_best_closest_rmsd"] = p.rmsd if p else ""
        matches = [jj for jj in range(nb) if same[ia, jj]]
        if matches:
            jm = min(matches, key=lambda jj: (rb[jj] if np.isfinite(rb[jj]) else math.inf))
            row["a_best_match_b"] = jm
            row["a_best_match_rel_b_kj"] = _rel_of(b, lb, jm)
    if ib is not None:
        cb = b.conformers[ib]
        row["b_best_source"], row["b_best_tag"] = cb.source, cb.tag
        row["b_best_found"] = bool(found_b[ib])
        matches = [ii for ii in range(na) if same[ii, ib]]
        if matches:
            im = min(matches, key=lambda ii: (ra[ii] if np.isfinite(ra[ii]) else math.inf))
            row["b_best_match_a"] = im
            row["b_best_match_rel_a_kj"] = _rel_of(a, la, im)
            search_a = next((lv for lv in a.levels() if lv.endswith("(search)")), "")
            if search_a and search_a != la:
                row["b_best_match_rel_a_search_kj"] = _rel_of(a, search_a, im)
    if ia is not None and ib is not None:
        p = pairs[(ia, ib)]
        row.update(same_best=bool(p.same), best_pair_rmsd=p.rmsd, best_pair_mirrored=p.mirrored,
                   best_pair_config_ok=p.config_ok, best_pair_torsion_max=p.torsion_max,
                   best_pair_torsion_mean=p.torsion_mean, best_pair_rotor_max=p.rotor_max,
                   best_pair_polar_h_max=p.polar_h_max)
    # the common level: optimized energies when both bests have them, else single points, in the phase
    ph = phase or b.phase or "gas"
    row["phase"] = ph
    row["relax_bonds_changed_a"] = sum(1 for c in a.conformers if c.extra.get("relax_topology_changed"))
    row["relax_bonds_changed_b"] = sum(1 for c in b.conformers if c.extra.get("relax_topology_changed"))
    dec = ""
    opt = common_level(ph, True)
    relaxed_run = any(opt in c.energies for c in a.conformers + b.conformers)
    # with optimizations, only optimized energies decide (a single point at an MMFF geometry carries its strain)
    for lv in (opt,) if relaxed_run else (common_level(ph, False),):
        if ia is not None and ib is not None and lv in a.conformers[ia].energies and lv in b.conformers[ib].energies:
            dec = lv
            break
    if ia is not None:
        row["a_best_relax_bonds_changed"] = bool(a.conformers[ia].extra.get("relax_topology_changed"))
    row["common_level"] = dec
    cause = "no energies" if ia is None or ib is None else ""
    if not cause:
        if row.get("same_best"):
            cause = CAUSES[0]
        elif row.get("a_best_found"):
            cause = CAUSES[1]
        elif not dec:
            cause = CAUSES[4]
        else:
            de = a.conformers[ia].energies[dec] - b.conformers[ib].energies[dec]
            row["de_common_a_best_minus_b_best_kj"] = de
            cause = CAUSES[3] if de < 0 else CAUSES[2]
    row["cause"] = cause
    if dec and ia is not None and ib is not None:
        row.setdefault("de_common_a_best_minus_b_best_kj",
                       a.conformers[ia].energies[dec] - b.conformers[ib].energies[dec])
        ea, eb = a.energies(dec), b.energies(dec)
        if np.isfinite(ea).any() or np.isfinite(eb).any():
            emin = np.nanmin(np.concatenate([ea, eb]))
            side_a = bool(np.isfinite(ea).any() and np.nanmin(ea) <= emin + 1e-9)
            k = int(np.nanargmin(ea)) if side_a else int(np.nanargmin(eb))
            in_other = bool(found_a[k]) if side_a else bool(found_b[k])
            cu = (a if side_a else b).conformers[k]
            if not in_other and "relaxed" in cu.extra:     # it may relax into a conformer the other side has
                others = b.conformers if side_a else a.conformers
                in_other = any(compare_pair(model, cu.extra["relaxed"], o.extra.get("relaxed", o.positions),
                                            rmsd_cut, torsion_cut, mirror).same for o in others)
            row["union_best"] = "both" if in_other else ("A only" if side_a else "B only")
            row["union_best_de_b_kj"] = float(eb[ib] - emin)
            if dec in cu.dipoles:
                row["dip_common_union_best"] = cu.dipoles[dec]
            row["a_best_common_rank"] = int((ea < ea[ia] - 1e-9).sum()) if np.isfinite(ea[ia]) else ""
        ca = a.conformers[ia]
        if "relaxed" in ca.extra:
            rel_pos = ca.extra["relaxed"]
            stay = compare_pair(model, ca.positions, rel_pos, rmsd_cut, torsion_cut, mirror)
            row["a_best_relaxed_rmsd"] = stay.rmsd
            row["a_best_relaxed_same"] = stay.same
            hits = [jj for jj in range(nb) if compare_pair(model, rel_pos, b.conformers[jj].positions, rmsd_cut,
                                                           torsion_cut, mirror).same]
            row["a_best_relaxed_found"] = bool(hits)
    # dipoles
    if ia is not None:
        ca = a.conformers[ia]
        ref_lv = next((lv for lv in ca.dipoles if lv == la), None) or next(
            (lv for lv in ca.dipoles if not lv.startswith(COMMON)), None)
        if ref_lv:
            row["dip_a_ref"], row["dip_a_ref_level"] = ca.dipoles[ref_lv], ref_lv
        for tag, lv in (("sp", common_level(ph)), ("opt", common_level(ph, True))):
            if lv in ca.dipoles:
                row[f"dip_common_a_best_{tag}"] = ca.dipoles[lv]
    if ib is not None:
        cb = b.conformers[ib]
        own = next((lv for lv in cb.dipoles if lv == lb), None) or next(
            (lv for lv in cb.dipoles if not lv.startswith(COMMON)), None)
        if own:
            row["dip_b_best"] = cb.dipoles[own]
        if common_level(ph) in cb.dipoles:
            row["dip_common_b_best_sp"] = cb.dipoles[common_level(ph)]
        if row.get("a_best_match_b") is not None:
            cm = b.conformers[int(row["a_best_match_b"])]
            own_m = next((lv for lv in cm.dipoles if lv == lb), None)
            if own_m:
                row["dip_b_match_a_best"] = cm.dipoles[own_m]
    mc = MoleculeComparison(a, b, row, pairs)
    for side, cs, lv_own, found, n_other in (("A", a, la, found_a, nb), ("B", b, lb, found_b, na)):
        for lv, cnt in cs.levels().items():
            r = cs.rel(lv)
            k = cs.best(lv)
            lr = {"key": a.key, "label": row["label"], "side": side, "level": lv, "own_level": lv == lv_own,
                  "n": cnt, "best": "" if k is None else k, "best_found": bool(found[k]) if k is not None else ""}
            lr.update(_window_cols("set", r, found, windows))
            mc.level_rows.append(lr)
        for k, c in enumerate(cs.conformers):
            j, p = _closest(pairs, k, n_other, "a" if side == "A" else "b")
            cr = {"key": a.key, "label": row["label"], "side": side, "index": k, "source": c.source, "tag": c.tag,
                  "own_level": lv_own, "rel_own_kj": _rel_of(cs, lv_own, k), "found": bool(found[k]),
                  "closest": "" if j is None else j, "closest_rmsd": p.rmsd if p else "",
                  "closest_torsion_max": p.torsion_max if p else "", "closest_mirrored": p.mirrored if p else ""}
            for lv in (common_level(ph, False), common_level(ph, True), common_level("gas", False)):
                if lv in c.energies:
                    cr[f"rel {lv}"] = _rel_of(cs, lv, k)
            own_d = c.dipoles.get(lv_own)
            cr["dipole_own"] = own_d if own_d is not None else next(
                (v for lv, v in c.dipoles.items() if not lv.startswith(COMMON)), "")
            for lv in (common_level(ph, False), common_level(ph, True)):
                if lv in c.dipoles:
                    cr[f"dipole {lv}"] = c.dipoles[lv]
            cr["energies"] = "; ".join(f"{lv}={v:.3f}" for lv, v in c.energies.items())
            mc.conformer_rows.append(cr)
    return mc


def compare(a_sets: Sequence[ConformerSet], b_sets: Sequence[ConformerSet], *, level_a: str = "",
            level_b: str = "", phase: str = "", rmsd_cut: float = DEFAULT_RMSD_A,
            torsion_cut: float = DEFAULT_TORSION_DEG, mirror: bool = True,
            windows: Sequence[float] = DEFAULT_WINDOWS_KJ, a_name: str = "A", b_name: str = "B",
            log: Callable[[str], None] | None = None) -> Comparison:
    """Join the two sides by constitution and compare every joined molecule (:func:`compare_molecule`)."""
    b_by_key = {s.key: s for s in b_sets}
    mols, only_a = [], []
    seen = set()
    for n, a in enumerate(a_sets, 1):
        b = b_by_key.get(a.key)
        if b is None:
            only_a.append(a)
            continue
        seen.add(a.key)
        m = atom_map(a, b)
        if m is None:
            only_a.append(a)
            continue
        b2 = ConformerSet(**{**b.__dict__, "numbers": a.numbers, "bonds": a.bonds, "conformers": [
            Conformer(np.asarray(c.positions)[m], c.energies, c.dipoles, c.source, c.tag,
                      {k: (np.asarray(v)[m] if k == "relaxed" else v) for k, v in c.extra.items()})
            for c in b.conformers]})
        mols.append(compare_molecule(a, b2, level_a=level_a, level_b=level_b, phase=phase, rmsd_cut=rmsd_cut,
                                     torsion_cut=torsion_cut, mirror=mirror, windows=windows))
        if log and len(mols) % 20 == 0:
            log(f"  compared {len(mols)} molecules ({n} of {len(a_sets)})")
    only_b = [s for s in b_sets if s.key not in seen]
    settings = {"rmsd_cut_A": rmsd_cut, "torsion_cut_deg": torsion_cut, "mirror": mirror, "windows_kj": list(windows),
                "level_a": level_a or "auto", "level_b": level_b or "auto", "phase": phase or "auto"}
    comp = Comparison(mols, only_a, only_b, settings, a_name=a_name, b_name=b_name)
    comp.summary = summarize(comp)
    return comp


# ------------------------------------------------------------------------------------------------ summary
def _mae(pairs: list[tuple[float, float]]) -> float | None:
    return float(np.mean([abs(x - y) for x, y in pairs])) if pairs else None


def _group_summary(rows: list[dict], windows: Sequence[float]) -> dict:
    out: dict = {"n_molecules": len(rows)}
    has_best = [r for r in rows if r.get("same_best") is not None and r.get("cause") != "no energies"]
    out["n_with_bests"] = len(has_best)
    out["same_best"] = sum(1 for r in has_best if r.get("same_best"))
    out["same_best_rate"] = out["same_best"] / len(has_best) if has_best else None
    out["a_best_found"] = sum(1 for r in has_best if r.get("a_best_found"))
    out["b_best_found"] = sum(1 for r in has_best if r.get("b_best_found"))
    for side in ("a", "b"):
        for w in windows:
            tag = f"{w:g}"
            n = sum(r.get(f"{side}_n_win{tag}", 0) for r in rows)
            f = sum(r.get(f"{side}_found_win{tag}", 0) for r in rows)
            full = sum(1 for r in rows if r.get(f"{side}_n_win{tag}", 0) and
                       r.get(f"{side}_found_win{tag}") == r.get(f"{side}_n_win{tag}"))
            out[f"{side}_win{tag}_conformers"], out[f"{side}_win{tag}_found"] = n, f
            out[f"{side}_win{tag}_coverage"] = f / n if n else None
            out[f"{side}_win{tag}_molecules_complete"] = full
    for c in CAUSES:
        out[f"cause: {c}"] = sum(1 for r in rows if r.get("cause") == c)
    out["search"] = sum(1 for r in rows if str(r.get("cause", "")).startswith("search"))
    out["ranking"] = sum(1 for r in rows if str(r.get("cause", "")).startswith("ranking"))
    for u in ("both", "A only", "B only"):
        out[f"union_best: {u}"] = sum(1 for r in rows if r.get("union_best") == u)
    de = [r["union_best_de_b_kj"] for r in rows if isinstance(r.get("union_best_de_b_kj"), float)]
    out["union_best_de_b_kj_mean"] = float(np.mean(de)) if de else None
    rm = [r["best_pair_rmsd"] for r in has_best if isinstance(r.get("best_pair_rmsd"), float)]
    out["best_pair_rmsd_median"] = float(np.median(rm)) if rm else None
    tm = [r["best_pair_torsion_max"] for r in has_best if isinstance(r.get("best_pair_torsion_max"), float)]
    out["best_pair_torsion_max_median"] = float(np.median(tm)) if tm else None
    dip = [r for r in rows if isinstance(r.get("dip_a_ref"), float) and isinstance(r.get("dip_b_best"), float)]
    out["n_dipole"] = len(dip)
    out["dipole_mae_b_best_vs_a_ref"] = _mae([(r["dip_b_best"], r["dip_a_ref"]) for r in dip])
    with_sp = [r for r in dip if isinstance(r.get("dip_common_a_best_sp"), float)]
    out["n_dipole_common"] = len(with_sp)
    out["dipole_mae_b_best_vs_a_ref_same_set"] = _mae([(r["dip_b_best"], r["dip_a_ref"]) for r in with_sp])
    out["dipole_mae_common_at_a_best_vs_a_ref"] = _mae([(r["dip_common_a_best_sp"], r["dip_a_ref"]) for r in with_sp])
    out["dipole_mae_b_best_vs_common_at_a_best"] = _mae([(r["dip_b_best"], r["dip_common_a_best_sp"]) for r in with_sp])
    opt = [r for r in dip if isinstance(r.get("dip_common_a_best_opt"), float)]
    out["dipole_mae_common_opt_at_a_best_vs_a_ref"] = _mae([(r["dip_common_a_best_opt"], r["dip_a_ref"]) for r in opt])
    chk = [r for r in rows if isinstance(r.get("dip_common_b_best_sp"), float) and isinstance(r.get("dip_b_best"), float)]
    out["dipole_check_common_vs_own_at_b_best"] = _mae([(r["dip_common_b_best_sp"], r["dip_b_best"]) for r in chk])
    u = [r for r in dip if isinstance(r.get("dip_common_union_best"), float)]
    out["dipole_mae_common_union_best_vs_a_ref"] = _mae([(r["dip_common_union_best"], r["dip_a_ref"]) for r in u])
    m = [r for r in dip if isinstance(r.get("dip_b_match_a_best"), float)]
    out["n_dipole_b_match"] = len(m)
    out["dipole_mae_b_match_vs_a_ref"] = _mae([(r["dip_b_match_a_best"], r["dip_a_ref"]) for r in m])
    out["dipole_mae_b_best_vs_a_ref_matched"] = _mae([(r["dip_b_best"], r["dip_a_ref"]) for r in m])
    a = out["dipole_mae_b_best_vs_a_ref_same_set"]
    b = out["dipole_mae_common_at_a_best_vs_a_ref"]
    out["dipole_gap_explained_by_conformer"] = (1.0 - b / a) if a and b is not None else None
    return out


def level_summary(comp: Comparison) -> list[dict]:
    """Per side and level (levels.csv summed over molecules): conformers within each window of that level's
    minimum and how many the other side found; molecules whose lowest conformer at that level the other side has."""
    agg: dict[tuple[str, str], dict] = {}
    windows = comp.settings["windows_kj"]
    for m in comp.molecules:
        for r in m.level_rows:
            k = (r["side"], r["level"])
            g = agg.setdefault(k, {"side": comp.a_name if r["side"] == "A" else comp.b_name, "level": r["level"],
                                   "molecules": 0, "conformers": 0, "best_found": 0, "own_level": 0,
                                   **{f"{x}_win{w:g}": 0 for w in windows for x in ("n", "found")}})
            g["molecules"] += 1
            g["conformers"] += int(r["n"])
            g["best_found"] += int(r["best_found"] is True)
            g["own_level"] += int(bool(r["own_level"]))
            for w in windows:
                g[f"n_win{w:g}"] += int(r[f"set_n_win{w:g}"])
                g[f"found_win{w:g}"] += int(r[f"set_found_win{w:g}"])
    return sorted(agg.values(), key=lambda g: (g["side"] != comp.a_name, g["level"].startswith(COMMON), g["level"]))


def summarize(comp: Comparison) -> list[dict]:
    rows = [m.row for m in comp.molecules]
    windows = comp.settings["windows_kj"]
    out = [{"group_type": "all", "group": "all", **_group_summary(rows, windows)}]
    for gt, col in (("charge_class", "charge_class"), ("group", "group"), ("cause", "cause")):
        for g in sorted({str(r.get(col, "")) for r in rows}):
            if not g:
                continue
            out.append({"group_type": gt, "group": g, **_group_summary([r for r in rows if str(r.get(col)) == g],
                                                                       windows)})
    return out


# ------------------------------------------------------------------------------------------------ outputs
def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}" if math.isfinite(v) else ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return "" if v is None else v


def _write_csv(path: Path, rows: list[dict]) -> Path:
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    return path


def _g(v, fmt: str = ".3g") -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return format(v, fmt)
    return str(v)


def _pct(n, d) -> str:
    return f"{n}/{d} ({100.0 * n / d:.0f} %)" if d else "-"


def write_outputs(comp: Comparison, out_dir: str | Path, inputs: dict | None = None) -> list[Path]:
    """molecules.csv, levels.csv, conformers.csv, summary.csv, levels_summary.csv and report.md in ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mol_rows = [m.row for m in comp.molecules]
    for s, side in ((comp.only_a, comp.a_name), (comp.only_b, comp.b_name)):
        mol_rows += [{"key": x.key, "label": x.label, "name": x.name, "charge": x.charge, "cause": f"only in {side}",
                      "n_a" if side == comp.a_name else "n_b": len(x.conformers)} for x in s]
    paths = [_write_csv(out / "molecules.csv", mol_rows),
             _write_csv(out / "levels.csv", [r for m in comp.molecules for r in m.level_rows]),
             _write_csv(out / "conformers.csv", [r for m in comp.molecules for r in m.conformer_rows]),
             _write_csv(out / "summary.csv", comp.summary),
             _write_csv(out / "levels_summary.csv", level_summary(comp))]
    paths.append(_report(out / "report.md", comp, inputs or {}))
    return paths


def _report(path: Path, comp: Comparison, inputs: dict) -> Path:
    A, B = comp.a_name, comp.b_name
    s = comp.summary[0] if comp.summary else {}
    rows = [m.row for m in comp.molecules]
    st = comp.settings
    w = st["windows_kj"]
    lines = [f"# Conformers: {A} vs {B}", ""]
    for k, v in inputs.items():
        lines.append(f"- {k}: {v}")
    levels_a = sorted({r.get("level_a", "") for r in rows} - {""})
    levels_b = sorted({r.get("level_b", "") for r in rows} - {""})
    lines += [f"- molecules joined (by constitution): {len(rows)}; only in {A}: {len(comp.only_a)}; only in {B}: "
              f"{len(comp.only_b)}",
              f"- {A} ranks by: {', '.join(levels_a) or '-'}; {B} ranks by: {', '.join(levels_b) or '-'}",
              f"- same conformer: symmetric heavy-atom RMSD <= {st['rmsd_cut_A']} Å, backbone and polar-H torsions "
              f"within {st['torsion_cut_deg']}°, same configuration; mirror images {'allowed' if st['mirror'] else 'not allowed'}",
              f"- common level: {', '.join(sorted({r.get('common_level', '') for r in rows} - {''})) or 'none'}",
              "", "## Headline", "",
              "| measure | value |", "|---|---|",
              f"| same lowest conformer | {_pct(s.get('same_best', 0), s.get('n_with_bests', 0))} |",
              f"| {A}'s lowest found by {B} | {_pct(s.get('a_best_found', 0), s.get('n_with_bests', 0))} |",
              f"| {B}'s lowest found by {A} | {_pct(s.get('b_best_found', 0), s.get('n_with_bests', 0))} |"]
    for side, name, other in (("a", A, B), ("b", B, A)):
        for x in w:
            t = f"{x:g}"
            lines.append(f"| {name}'s conformers within {t} kJ/mol of its minimum found by {other} | "
                         f"{_pct(s.get(f'{side}_win{t}_found', 0), s.get(f'{side}_win{t}_conformers', 0))}; "
                         f"all of them in {s.get(f'{side}_win{t}_molecules_complete', 0)} molecules |")
    lines += [f"| median RMSD / largest torsion difference of the two lowest | {_g(s.get('best_pair_rmsd_median'))} Å / "
              f"{_g(s.get('best_pair_torsion_max_median'))}° |", ""]
    ls = level_summary(comp)
    if ls:
        lines += ["## Coverage by level", "",
                  "Each side's conformers ranked at each level they have (a common level ranks both sides alike):", "",
                  "| side | level | molecules | conformers | " + " | ".join(f"within {x:g} kJ/mol: found" for x in w)
                  + " | lowest found |", "|---|---|---|---|" + "---|" * len(w) + "---|"]
        for g in ls:
            lines.append(f"| {g['side']} | {g['level']} | {g['molecules']} | {g['conformers']} | "
                         + " | ".join(_pct(g[f"found_win{x:g}"], g[f"n_win{x:g}"]) for x in w)
                         + f" | {_pct(g['best_found'], g['molecules'])} |")
        lines.append("")
    lines += ["## Search or ranking", "",
              f"Why {B}'s lowest conformer is not {A}'s (the common level decides whether {B} missed a conformer "
              "its own level prefers):", "", "| cause | molecules |", "|---|---|"]
    for c in CAUSES:
        n = s.get(f"cause: {c}", 0)
        if n:
            lines.append(f"| {c} | {n} |")
    lines += ["", f"Search: {s.get('search', 0)}, ranking (energies): {s.get('ranking', 0)}.", ""]
    ub = [(u, s.get(f"union_best: {u}", 0)) for u in ("both", "A only", "B only")]
    if any(n for _u, n in ub):
        lines += [f"Lowest conformer of both sets together at the common level: in both sets {ub[0][1]}, only in {A}'s "
                  f"{ub[1][1]}, only in {B}'s {ub[2][1]}; {B}'s lowest lies {_g(s.get('union_best_de_b_kj_mean'))} "
                  "kJ/mol above it on average.", ""]
    if s.get("n_dipole"):
        lines += ["## Dipoles", "", "| comparison | N | MAE (D) |", "|---|---|---|",
                  f"| {B}'s lowest vs {A}'s lowest ({A}'s level) | {s['n_dipole']} | {_g(s.get('dipole_mae_b_best_vs_a_ref'))} |"]
        if s.get("n_dipole_common"):
            n = s["n_dipole_common"]
            lines += [f"| the same, molecules with common-level values | {n} | {_g(s.get('dipole_mae_b_best_vs_a_ref_same_set'))} |",
                      f"| common level at {A}'s lowest geometry vs {A}'s level (method only) | {n} | "
                      f"{_g(s.get('dipole_mae_common_at_a_best_vs_a_ref'))} |",
                      f"| {B}'s lowest vs common level at {A}'s lowest (conformer only) | {n} | "
                      f"{_g(s.get('dipole_mae_b_best_vs_common_at_a_best'))} |"]
            if s.get("dipole_mae_common_opt_at_a_best_vs_a_ref") is not None:
                lines.append(f"| common level, optimized from {A}'s lowest, vs {A}'s level | | "
                             f"{_g(s.get('dipole_mae_common_opt_at_a_best_vs_a_ref'))} |")
            if s.get("dipole_mae_common_union_best_vs_a_ref") is not None:
                lines.append(f"| common level's lowest of both sets together vs {A}'s level | | "
                             f"{_g(s.get('dipole_mae_common_union_best_vs_a_ref'))} |")
            if s.get("dipole_check_common_vs_own_at_b_best") is not None:
                lines.append(f"| check: common level vs {B}'s own at {B}'s lowest | | "
                             f"{_g(s.get('dipole_check_common_vs_own_at_b_best'))} |")
        if s.get("n_dipole_b_match"):
            lines.append(f"| {B}'s conformer matching {A}'s lowest vs {A}'s level (where {B} has it) | "
                         f"{s['n_dipole_b_match']} | {_g(s.get('dipole_mae_b_match_vs_a_ref'))} (its lowest: "
                         f"{_g(s.get('dipole_mae_b_best_vs_a_ref_matched'))}) |")
        ex = s.get("dipole_gap_explained_by_conformer")
        if ex is not None:
            lines += ["", f"The conformer explains {100 * ex:.0f} % of the dipole gap: at {A}'s own lowest geometry the "
                          f"common level is {_g(s.get('dipole_mae_common_at_a_best_vs_a_ref'))} D from {A}, against "
                          f"{_g(s.get('dipole_mae_b_best_vs_a_ref_same_set'))} D for {B}'s own lowest conformer."]
        lines.append("")
    lines += ["## By group", "", "| group | N | same lowest | A's lowest found | A within "
              f"{w[0]:g} kJ/mol found | search | ranking | dipole MAE B vs A | at A's geometry |", "|---|---|---|---|---|---|---|---|---|"]
    for g in comp.summary[1:]:
        if g["group_type"] == "cause":
            continue
        t = f"{w[0]:g}"
        lines.append(f"| {g['group_type']}: {g['group']} | {g['n_molecules']} | {_pct(g['same_best'], g['n_with_bests'])} | "
                     f"{_pct(g['a_best_found'], g['n_with_bests'])} | {_pct(g[f'a_win{t}_found'], g[f'a_win{t}_conformers'])} | "
                     f"{g['search']} | {g['ranking']} | {_g(g.get('dipole_mae_b_best_vs_a_ref'))} | "
                     f"{_g(g.get('dipole_mae_common_at_a_best_vs_a_ref'))} |")
    lines += ["", "## Molecules with the largest dipole gap", "",
              "| molecule | charge | cause | best-pair RMSD (Å) | torsion max (°) | A dipole | B dipole | common at A's lowest |",
              "|---|---|---|---|---|---|---|---|"]
    gap = sorted((r for r in rows if isinstance(r.get("dip_a_ref"), float) and isinstance(r.get("dip_b_best"), float)),
                 key=lambda r: -abs(r["dip_b_best"] - r["dip_a_ref"]))
    for r in gap[:15]:
        lines.append(f"| {r['label']} {r.get('name', '')} | {r['charge']} | {r.get('cause', '')} | {_g(r.get('best_pair_rmsd'))} | "
                     f"{_g(r.get('best_pair_torsion_max'))} | {_g(r['dip_a_ref'])} | {_g(r['dip_b_best'])} | "
                     f"{_g(r.get('dip_common_a_best_sp'))} |")
    lines += ["", "## Files", "",
              "- `molecules.csv`: one row per molecule (coverage, the two lowest conformers, cause, common-level energies "
              "and dipoles); molecules found on one side only at the end",
              "- `levels.csv`: per molecule, side and level: conformers, those within the windows and how many the other "
              "side found",
              "- `conformers.csv`: every conformer with its relative energies, dipoles and closest partner",
              "- `summary.csv`: the statistics, overall and per charge class, group and cause",
              "- `levels_summary.csv`: the coverage table above",
              "", "## Caveats", "",
              "- Each side ranks at its own level. Spartan's DFT_Top5 documents re-rank only 5 conformers (those of "
              "its Top5 MMFF search), with single points at the MMFF geometries.",
              "- Spartan's All_Conformers geometries are MMFF; single points at the common level put them higher (strain), "
              "the optimization removes that. A conformer that relaxes into another basin is flagged "
              "(`a_best_relaxed_same`). An optimization that changes the bonds (a proton moves) is set aside "
              "(`relax_bonds_changed_a`/`_b`).",
              "- Mirror images count as the same conformer (same energy, same |dipole|), never across a stereocentre.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
