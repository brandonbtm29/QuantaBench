"""Editable 3D molecule for the model kit (UI-7): an RDKit ``RWMol`` with one conformer, changed only through
undoable commands. GUI-free: the 3D view's build mode and the 2D sketcher drive the same model.

Building works as in Spartan's model kit. Pick an atom type (element and hybridization, ``PALETTE``) or a
fragment (``chem.fragments``) and attach it at a *free valence* of an atom: the new atom sits at its ideal
bond length along the valence direction, and its own open valences are filled with hydrogens (``fill_h``, the
default; Spartan fills them when you leave the builder). A free valence is either a hydrogen (attaching
replaces it, and the new atom takes over the hydrogen's index) or an open direction with no atom on it, which
is what ``fill_h=False`` leaves. Clicking an atom with a type selected replaces it (``replace_atom``).

Chemistry, VSEPR style:

- each atom stores its hybridization (``sp3``, ``sp2``, ``sp``: 4, 3 or 2 electron domains) and its valence
  V, the bond-order sum it has when complete (C 4, N 3, O 2; S 2, 4 or 6; P 3 or 5);
- lone pairs LP = (valence electrons - formal charge - V) / 2 take domains, so an atom makes S = domains - LP
  sigma bonds, and the rest of its valence (V - S) goes into pi bonds;
- a free valence can take a bond of order 1 + the atom's pi bonds still to make (Spartan's double free
  valence on an sp2 carbon). Attaching an sp2 atom to an sp2 atom that still has a pi bond to make gives a
  double bond (two sp atoms: a triple bond); ``=O``, ``=S`` and ``≡N`` always attach with a double or
  triple bond. Attaching ``=O`` to S or P raises its valence (S 2 -> 4 -> 6, P 3 -> 5).

Every change is a ``Command`` with ``apply(model)`` and ``undo(model)``. ``History`` keeps the undo and redo
stacks here and in tests; the GUI wraps the same commands in a ``QUndoStack`` (``model.history`` only needs a
``push(command)`` that applies it). Atom indices are 0-based, as in the 3D view's labels.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

from . import fragments as frag_lib
from .fragments import Fragment

SEED = 42
DOMAINS = {"s": 1, "sp": 2, "sp2": 3, "sp3": 4, "sp3d": 5, "sp3d2": 6}   # 5 and 6: read, not built
_HYB_OF = {1: "s", 2: "sp", 3: "sp2", 4: "sp3"}
#: Valences an atom of these elements may take; attaching ``=O`` moves it up the list.
HYPERVALENT = {"P": (3, 5), "S": (2, 4, 6), "Se": (2, 4, 6)}
_PT = Chem.GetPeriodicTable()
_BOND_TYPE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
_ORDER_NAME = {1: "single", 2: "double", 3: "triple"}
_TET = math.acos(-1.0 / 3.0)            # 109.47°


class EditError(ValueError):
    """An edit the model kit cannot make. The message says why and is meant for the user."""


# ====================================================================== atom types (the palette)
def default_valence(symbol: str, charge: int = 0) -> int:
    """Usual valence of ``symbol`` at ``charge``: 8 - (outer electrons - charge) above 4 electrons, else the
    electron count (C 4, N 3, O 2, F 1; N+ 4, O- 1, C+ and C- 3)."""
    if symbol == "H":
        return 1 if charge == 0 else 0
    e = _PT.GetNOuterElecs(symbol) - charge
    return max(0, e if e <= 4 else 8 - e)


def _capacity(symbol: str, charge: int, hyb: str, valence: int) -> tuple[int, int, int]:
    """(lone pairs, sigma bonds, pi bonds) an atom of this kind makes when complete."""
    if symbol == "H":
        return 0, 1, 0
    lp = max(0, (_PT.GetNOuterElecs(symbol) - charge - valence) // 2)
    sigma = max(0, DOMAINS.get(hyb, 4) - lp)
    return lp, sigma, valence - sigma


@dataclass(frozen=True)
class AtomType:
    """One button of the atom palette: element, hybridization and formal charge. ``bond_order`` fixes the
    order of the bond to the atom it is attached to (2 for ``=O`` and ``=S``, 3 for ``≡N``); None lets the
    kit choose. ``valence`` None means the element's usual valence at this charge."""

    key: str
    symbol: str
    hyb: str = "sp3"
    charge: int = 0
    bond_order: int | None = None
    valence: int | None = None

    @property
    def total_valence(self) -> int:
        return self.valence if self.valence is not None else default_valence(self.symbol, self.charge)

    @property
    def sigma(self) -> int:
        """Sigma bonds an atom of this type makes (its free valences when it stands alone)."""
        return _capacity(self.symbol, self.charge, self.hyb, self.total_valence)[1]

    @property
    def pi(self) -> int:
        return _capacity(self.symbol, self.charge, self.hyb, self.total_valence)[2]


_TYPES = (
    # Spartan's Organic tab, in its order
    AtomType("C sp3", "C", "sp3"), AtomType("C sp2", "C", "sp2"), AtomType("C sp", "C", "sp"),
    AtomType("N sp3", "N", "sp3"), AtomType("N sp2", "N", "sp2"), AtomType("N sp", "N", "sp"),
    AtomType("P", "P", "sp3"), AtomType("O sp3", "O", "sp3"), AtomType("O sp2", "O", "sp2"),
    AtomType("S", "S", "sp3"), AtomType("Si", "Si", "sp3"), AtomType("H", "H", "s"),
    AtomType("F", "F", "sp3"), AtomType("Cl", "Cl", "sp3"), AtomType("Br", "Br", "sp3"), AtomType("I", "I", "sp3"),
    AtomType("=O", "O", "sp2", bond_order=2), AtomType("=S", "S", "sp2", bond_order=2),
    AtomType("≡N", "N", "sp", bond_order=3),
    # charged and hypervalent types (Spartan's "More" list)
    AtomType("N+ sp3", "N", "sp3", charge=1), AtomType("N+ sp2", "N", "sp2", charge=1),
    AtomType("O-", "O", "sp3", charge=-1), AtomType("S(IV)", "S", "sp3", valence=4),
    AtomType("S(VI)", "S", "sp3", valence=6), AtomType("P(V)", "P", "sp3", valence=5), AtomType("B", "B", "sp2"),
)
PALETTE: dict[str, AtomType] = {t.key: t for t in _TYPES}
#: The Organic tab of the palette, in Spartan's order.
ORGANIC = tuple(t.key for t in _TYPES[:19])
_TYPE_ALIASES = {"c": "C sp3", "n": "N sp3", "o": "O sp3", "#n": "≡N", "n#": "≡N", "o=": "=O", "s=": "=S",
                 "s(ii)": "S", "p(iii)": "P", "o sp3-": "O-", "o-sp3": "O-"}
_TYPE_INDEX = {k.lower().replace(" ", ""): k for k in PALETTE} | {k.replace(" ", ""): v for k, v in _TYPE_ALIASES.items()}


def atom_type(kind: str | AtomType) -> AtomType:
    """A palette entry by key (``"C sp2"``, ``"=O"``, ``"≡N"`` or ``"#N"``; a bare element means its sp3 type)."""
    if isinstance(kind, AtomType):
        return kind
    key = _TYPE_INDEX.get(str(kind).strip().lower().replace(" ", ""))
    if key is None:
        if str(kind).strip() in PALETTE:
            return PALETTE[str(kind).strip()]
        raise EditError(f"Unknown atom type {kind!r} (palette: {', '.join(PALETTE)})")
    return PALETTE[key]


if any(t.sigma < 1 or t.pi < 0 for t in _TYPES):   # every palette type must be able to bond
    raise RuntimeError("chem.edit: a palette atom type cannot bond")


# ====================================================================== bond lengths
#: Pyykkö and Atsumi's covalent radii (single, double, triple; Å), the fallback for pairs not in _LENGTHS.
_RADII = {"H": (0.32, 0.32, 0.32), "B": (0.85, 0.78, 0.73), "C": (0.75, 0.67, 0.60), "N": (0.71, 0.60, 0.54),
          "O": (0.63, 0.57, 0.53), "F": (0.64, 0.59, 0.53), "Si": (1.16, 1.07, 1.02), "P": (1.11, 1.02, 0.94),
          "S": (1.03, 0.94, 0.95), "Cl": (0.99, 0.95, 0.93), "Se": (1.16, 1.07, 1.07), "Br": (1.14, 1.09, 1.10),
          "I": (1.33, 1.29, 1.25)}
#: Typical bond lengths (Å) of common pairs, close to MMFF94's minima, so a new atom starts near where
#: Minimize will put it.
_LENGTHS = {
    ("C", "H", 1): 1.09, ("H", "N", 1): 1.01, ("H", "O", 1): 0.97, ("H", "S", 1): 1.34, ("H", "Si", 1): 1.48,
    ("H", "P", 1): 1.42, ("B", "H", 1): 1.19,
    ("C", "C", 1): 1.53, ("C", "C", 2): 1.34, ("C", "C", 3): 1.20,
    ("C", "N", 1): 1.46, ("C", "N", 2): 1.28, ("C", "N", 3): 1.16,
    ("C", "O", 1): 1.42, ("C", "O", 2): 1.22, ("C", "F", 1): 1.35, ("C", "Cl", 1): 1.77,
    ("Br", "C", 1): 1.94, ("C", "I", 1): 2.14, ("C", "S", 1): 1.81, ("C", "S", 2): 1.61,
    ("C", "Si", 1): 1.87, ("C", "P", 1): 1.84,
    ("N", "O", 1): 1.40, ("N", "O", 2): 1.22, ("N", "N", 1): 1.45, ("N", "N", 2): 1.25, ("N", "N", 3): 1.10,
    ("O", "S", 1): 1.58, ("O", "S", 2): 1.45, ("O", "P", 1): 1.61, ("O", "P", 2): 1.48, ("N", "S", 1): 1.65,
    ("O", "Si", 1): 1.64, ("O", "O", 1): 1.47, ("S", "S", 1): 2.05,
}


def bond_length(a: str, b: str, order: int = 1) -> float:
    """Ideal length (Å) of an ``a``-``b`` bond of ``order`` 1, 2 or 3."""
    x, y = sorted((a, b))
    if (x, y, order) in _LENGTHS:
        return _LENGTHS[(x, y, order)]
    k = min(max(order, 1), 3) - 1
    ra = _RADII[a][k] if a in _RADII else _PT.GetRcovalent(a)
    rb = _RADII[b][k] if b in _RADII else _PT.GetRcovalent(b)
    return ra + rb


# ====================================================================== geometry
def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _perpendicular(u: np.ndarray) -> np.ndarray:
    axis = np.eye(3)[int(np.argmin(np.abs(u)))]
    return _unit(np.cross(u, axis))


def _rotation(axis, angle: float) -> np.ndarray:
    """Right-handed rotation by ``angle`` (radians) about ``axis`` (Rodrigues)."""
    k = _unit(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1 - math.cos(angle)) * (kx @ kx)


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = _unit(a), _unit(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return _rotation(_perpendicular(a), math.pi)
    return _rotation(np.cross(a, b), math.acos(c))


def domain_directions(existing: Sequence[np.ndarray], n_domains: int,
                      reference: np.ndarray | None = None) -> list[np.ndarray]:
    """Unit vectors of an atom's electron domains not taken by ``existing`` (unit vectors to its neighbours),
    for 4 (tetrahedral), 3 (trigonal planar), 2 (linear) or 1 domains. With one existing bond, ``reference``
    (a vector off that bond's axis) sets the torsion: the first new direction leans toward it; the kit passes
    the opposite of a neighbour's neighbour, so new substituents come out anti and staggered."""
    dirs = [_unit(d) for d in existing]
    n = len(dirs)
    if n >= n_domains or n_domains not in (1, 2, 3, 4):
        return []
    if n == 0:
        if n_domains == 4:
            return [_unit(v) for v in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
        if n_domains == 3:
            return [np.array([math.cos(t), math.sin(t), 0.0]) for t in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)]
        return [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])][:n_domains]
    if n == 1:
        u = dirs[0]
        if n_domains == 2:
            return [-u]
        w = None
        if reference is not None:
            r = np.asarray(reference, float)
            w = _unit(r - np.dot(r, u) * u)
        if w is None or not w.any():
            w = _perpendicular(u)
        if n_domains == 3:
            c, s = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
            return [_unit(u * c + w * s), _unit(u * c - w * s)]
        v = np.cross(u, w)
        c, s = math.cos(_TET), math.sin(_TET)
        return [_unit(u * c + s * (w * math.cos(p) + v * math.sin(p))) for p in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)]
    if n == 2:
        u1, u2 = dirs
        total = u1 + u2
        b = _unit(-total) if np.linalg.norm(total) > 1e-3 else _perpendicular(u1)
        if n_domains == 3:
            return [b]
        normal = np.cross(u1, u2)
        normal = _unit(normal) if np.linalg.norm(normal) > 1e-3 else _unit(np.cross(u1, b))
        half = _TET / 2
        return [_unit(b * math.cos(half) + normal * math.sin(half)), _unit(b * math.cos(half) - normal * math.sin(half))]
    total = dirs[0] + dirs[1] + dirs[2]       # n == 3, tetrahedral
    if np.linalg.norm(total) > 0.1:
        return [_unit(-total)]
    return [_unit(np.cross(dirs[1] - dirs[0], dirs[2] - dirs[0]))]


# ====================================================================== records
@dataclass(frozen=True)
class FreeValence:
    """An open valence of ``atom``: a direction (unit vector) where a bond can go. ``h`` is the hydrogen that
    fills it (attaching replaces that hydrogen), None for an open direction with no atom on it.
    ``max_order`` is the highest bond order it takes without changing the atom (2: a double free valence)."""

    atom: int
    direction: tuple[float, float, float]
    h: int | None = None
    max_order: int = 1

    @property
    def vector(self) -> np.ndarray:
        return np.array(self.direction, float)


@dataclass(frozen=True)
class AtomState:
    """What the kit knows about one atom (see the module docstring for the rules)."""

    symbol: str
    charge: int
    hyb: str
    valence: int          # bond-order sum when complete
    lone_pairs: int
    sigma_slots: int      # sigma bonds it makes when complete
    degree: int           # neighbours now, hydrogens included
    bond_sum: int         # bond-order sum now

    @property
    def open(self) -> int:
        """Valence not used by a bond (an atom with all its hydrogens has 0)."""
        return self.valence - self.bond_sum

    @property
    def sigma_free(self) -> int:
        return self.sigma_slots - self.degree

    @property
    def pi_free(self) -> int:
        """Pi bonds still to make (a double free valence when > 0)."""
        return self.open - self.sigma_free


@dataclass(frozen=True)
class Placement:
    """Where a fragment went: its attachment atom, all its atoms (in fragment order), the hydrogen at its
    ``[*:2]`` chain end (where the next unit attaches) and, for a fragment added on its own, the hydrogen
    at its ``[*:1]`` attachment point."""

    anchor: int
    atoms: tuple[int, ...]
    exit: int | None = None
    entry: int | None = None


@dataclass(frozen=True)
class MinimizeResult:
    energy: float          # kcal/mol (0 when nothing was minimized)
    forcefield: str        # "MMFF94", "MMFF94s", "UFF" or "none"
    converged: bool
    moved: int = 0         # atoms free to move


# ====================================================================== fragment templates
@dataclass(frozen=True)
class _Template:
    symbols: tuple[str, ...]
    charges: tuple[int, ...]
    hybs: tuple[str, ...]
    valences: tuple[int, ...]
    positions: np.ndarray
    bonds: tuple[tuple[int, int, int], ...]
    anchor: int
    attach_dir: np.ndarray      # unit vector from the anchor toward the atom it bonds to
    order: int                  # order of that bond
    exit: int | None            # the hydrogen at [*:2]


def _bond_orders(atom: Chem.Atom) -> int:
    return sum(int(round(b.GetBondTypeAsDouble())) for b in atom.GetBonds())


def _perceive(atom: Chem.Atom) -> tuple[str, int]:
    """(hybridization, valence) of an atom in a complete, kekulized molecule with explicit hydrogens, by
    VSEPR: domains = sigma bonds (radicals count) + lone pairs."""
    sym = atom.GetSymbol()
    if sym == "H":
        return "s", 1
    try:
        implicit_h = atom.GetTotalNumHs()     # hydrogens that are not atoms of their own
    except RuntimeError:                      # property cache not computed: all hydrogens are atoms here
        implicit_h = atom.GetNumExplicitHs()
    sigma = atom.GetDegree() + implicit_h + atom.GetNumRadicalElectrons()
    valence = _bond_orders(atom) + implicit_h + atom.GetNumRadicalElectrons()
    lp = max(0, (_PT.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetFormalCharge() - valence) // 2)
    domains = sigma + lp
    return _HYB_OF.get(domains, "sp3d" if domains == 5 else "sp3d2" if domains >= 6 else "s"), valence


def _optimize_conformers(mol: Chem.Mol) -> list[float]:
    with rdBase.BlockLogs():
        if AllChem.MMFFHasAllMoleculeParams(mol):
            return [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, mmffVariant="MMFF94")]
        if AllChem.UFFHasAllMoleculeParams(mol):
            return [e for _, e in AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)]
    return [0.0] * mol.GetNumConformers()


def _embed(mol: Chem.Mol, seed: int = SEED, n_confs: int = 4) -> None:
    """3D coordinates in place: ETKDG, force-field optimized, keeping the lowest of ``n_confs`` conformers."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    with rdBase.BlockLogs():
        ids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
        if not ids:
            params.useRandomCoords = True
            ids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not ids:
        raise EditError(f"RDKit could not make 3D coordinates for {Chem.MolToSmiles(Chem.RemoveHs(mol))}")
    energies = _optimize_conformers(mol)
    best = Chem.Conformer(mol.GetConformer(ids[int(np.argmin(energies))]))
    mol.RemoveAllConformers()
    best.SetId(0)
    mol.AddConformer(best, assignId=False)


def _extend_chains(mol: Chem.Mol) -> None:
    """Set every acyclic single bond between two backbone atoms to anti (180°) and re-optimize, so chains
    start extended (zig-zag) as Spartan builds them, not curled the way a random conformer may be. The
    backbone neighbour on each side is the one with the most heavy neighbours (a CF2's next carbon, not its
    fluorines)."""
    def heavy_degree(a: Chem.Atom) -> int:
        return sum(n.GetAtomicNum() > 1 for n in a.GetNeighbors())

    Chem.FastFindRings(mol)
    conf = mol.GetConformer()
    changed = False
    for b in mol.GetBonds():
        j, k = b.GetBeginAtom(), b.GetEndAtom()
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing() or j.GetAtomicNum() == 1 or k.GetAtomicNum() == 1:
            continue
        side = []
        for a, other in ((j, k), (k, j)):
            nbrs = [n for n in a.GetNeighbors() if n.GetIdx() != other.GetIdx() and n.GetAtomicNum() > 1]
            side.append(max(nbrs, key=lambda n: (heavy_degree(n), -n.GetIdx())) if nbrs else None)
        if side[0] is None or side[1] is None:
            continue
        rdMolTransforms.SetDihedralDeg(conf, side[0].GetIdx(), j.GetIdx(), k.GetIdx(), side[1].GetIdx(), 180.0)
        changed = True
    if changed:
        _optimize_conformers(mol)


@lru_cache(maxsize=256)
def _template(smiles: str) -> _Template:
    """3D template of a fragment: embedded with a methyl carbon standing in for the molecule it bonds to, so
    the attachment atom has a real bond in that direction."""
    mol, d1, d2 = frag_lib.attachment_points(Fragment("fragment", smiles, "group"))
    rw = Chem.RWMol(mol)
    anchor = rw.GetAtomWithIdx(d1).GetNeighbors()[0].GetIdx()
    bt = rw.GetBondBetweenAtoms(d1, anchor).GetBondType()
    order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bt)
    if order is None:
        raise EditError(f"Fragment {smiles!r}: the attachment bond must be single, double or triple")
    stand_in = rw.GetAtomWithIdx(d1)
    stand_in.SetAtomicNum(6)
    stand_in.SetAtomMapNum(0)
    stand_in.SetNoImplicit(False)
    if d2 is not None:
        rw.GetAtomWithIdx(d2).SetAtomicNum(1)
        rw.GetAtomWithIdx(d2).SetAtomMapNum(0)
    Chem.SanitizeMol(rw)
    mh = Chem.AddHs(rw)              # new hydrogens go at the end: the indices above stay valid
    _embed(mh, SEED, n_confs=6)      # the lowest of 6 (a chair cyclohexane), then chains straightened
    _extend_chains(mh)
    Chem.Kekulize(mh, clearAromaticFlags=True)
    drop = {d1} | {n.GetIdx() for n in mh.GetAtomWithIdx(d1).GetNeighbors() if n.GetAtomicNum() == 1}
    keep = [i for i in range(mh.GetNumAtoms()) if i not in drop]
    new = {old: k for k, old in enumerate(keep)}
    pos = mh.GetConformer().GetPositions()
    atoms = [mh.GetAtomWithIdx(i) for i in keep]
    perceived = [_perceive(a) for a in atoms]
    bonds = tuple((new[b.GetBeginAtomIdx()], new[b.GetEndAtomIdx()], int(round(b.GetBondTypeAsDouble())))
                  for b in mh.GetBonds() if b.GetBeginAtomIdx() in new and b.GetEndAtomIdx() in new)
    positions = pos[keep].copy()
    positions.setflags(write=False)
    return _Template(symbols=tuple(a.GetSymbol() for a in atoms), charges=tuple(a.GetFormalCharge() for a in atoms),
                     hybs=tuple(h for h, _ in perceived), valences=tuple(v for _, v in perceived),
                     positions=positions, bonds=bonds, anchor=new[anchor],
                     attach_dir=_unit(pos[d1] - pos[anchor]), order=order,
                     exit=new[d2] if d2 is not None else None)


def _fragment(fragment: str | Fragment) -> Fragment:
    if isinstance(fragment, Fragment):
        return fragment
    try:
        return frag_lib.get(fragment)
    except KeyError:
        pass
    try:
        return frag_lib.from_smiles(fragment)
    except ValueError as exc:
        raise EditError(f"No fragment named {fragment!r}, and it is not a SMILES with an attachment point "
                        f"({exc})") from None


# ====================================================================== the model
def _empty() -> Chem.RWMol:
    rw = Chem.RWMol()
    rw.AddConformer(Chem.Conformer(0), assignId=False)
    return rw


class EditableMolecule:
    """An RDKit ``RWMol`` with one 3D conformer and the model-kit operations on it.

    Queries (``free_valences``, ``distance``, ``to_smiles``...) read the model; edits (``attach_atom``,
    ``minimize``...) run a ``Command`` through ``history``, so each can be undone. ``fill_h`` is the default
    for edits that create open valences; ``auto_relax`` runs a quick local relax after each structural edit
    (the GUI turns it on)."""

    def __init__(self, *, fill_h: bool = True, auto_relax: bool = False, variant: str = "MMFF94"):
        self._mol = _empty()
        self.fill_h = fill_h
        self.auto_relax = auto_relax
        self.variant = variant               # MMFF94 or MMFF94s
        self.last_edit: tuple[int, ...] = ()  # atoms the last structural edit touched
        self.history = History(self)
        self.listeners: list[Callable[[EditableMolecule], None]] = []

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mol(cls, mol: Chem.Mol, *, seed: int = SEED, **kw) -> EditableMolecule:
        """From an RDKit molecule: its first 3D conformer is kept (missing hydrogens are added with
        coordinates); a molecule without 3D coordinates is embedded (ETKDG + MMFF94)."""
        m = Chem.Mol(mol)
        with rdBase.BlockLogs():
            m.UpdatePropertyCache(strict=False)
        conf = m.GetConformer() if m.GetNumConformers() else None
        if conf is not None and conf.Is3D() and np.abs(conf.GetPositions()[:, 2]).max(initial=0.0) > 1e-3:
            keep = Chem.Conformer(conf)
            m.RemoveAllConformers()
            keep.SetId(0)
            m.AddConformer(keep, assignId=False)
            m = Chem.AddHs(m, addCoords=True)
        else:
            m.RemoveAllConformers()
            m = Chem.AddHs(m)
            if m.GetNumAtoms():
                _embed(m, seed)
            else:
                m.AddConformer(Chem.Conformer(0), assignId=False)
        Chem.Kekulize(m, clearAromaticFlags=True)
        rw = Chem.RWMol(m)
        for a in rw.GetAtoms():
            hyb, valence = _perceive(a)
            a.SetProp("qb_hyb", hyb)
            a.SetIntProp("qb_valence", valence)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetNumRadicalElectrons(0)
        model = cls(**kw)
        model._mol = rw
        return model

    @classmethod
    def from_smiles(cls, smiles: str, **kw) -> EditableMolecule:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise EditError(f"Not a valid SMILES string: {smiles!r}")
        return cls.from_mol(mol, **kw)

    @classmethod
    def from_molblock(cls, block: str, **kw) -> EditableMolecule:
        with rdBase.BlockLogs():
            mol = Chem.MolFromMolBlock(block, removeHs=False)
        if mol is None:
            raise EditError("Could not read the MOL block")
        return cls.from_mol(mol, **kw)

    @classmethod
    def from_input(cls, item, **kw) -> EditableMolecule:
        """From a ``chem.inputs.MoleculeInput`` (its structure file's coordinates when it has them)."""
        return cls.from_mol(item.to_mol(), **kw)

    @classmethod
    def from_atom(cls, kind: str | AtomType = "C sp3", **kw) -> EditableMolecule:
        """A new model holding one atom of ``kind`` (with its hydrogens unless ``fill_h=False``)."""
        model = cls(**kw)
        model.add_atom(kind)
        return model

    @classmethod
    def from_fragment(cls, fragment: str | Fragment, **kw) -> EditableMolecule:
        """A new model holding one fragment; its attachment point becomes a hydrogen (``Placement.entry``)."""
        model = cls(**kw)
        model.add_fragment(fragment)
        return model

    # ------------------------------------------------------------------ reading the model
    @property
    def mol(self) -> Chem.Mol:
        """A copy of the molecule as it is (kekulized, explicit hydrogens, open valences left open)."""
        return Chem.Mol(self._mol)

    @property
    def n_atoms(self) -> int:
        return self._mol.GetNumAtoms()

    @property
    def positions(self) -> np.ndarray:
        if not self.n_atoms:
            return np.zeros((0, 3))
        return self._mol.GetConformer().GetPositions()

    def symbol(self, i: int) -> str:
        return self._atom(i).GetSymbol()

    def label(self, i: int) -> str:
        return f"{self.symbol(i)}{i}"

    def neighbors(self, i: int) -> list[int]:
        return [n.GetIdx() for n in self._atom(i).GetNeighbors()]

    def bond_order(self, i: int, j: int) -> int:
        """Order of the ``i``-``j`` bond, 0 when they are not bonded."""
        self._atom(i), self._atom(j)
        b = self._mol.GetBondBetweenAtoms(int(i), int(j))
        return 0 if b is None else int(round(b.GetBondTypeAsDouble()))

    def hydrogens(self, atoms: Iterable[int] | None = None) -> list[int]:
        """Hydrogen atoms among ``atoms`` (all when None)."""
        idx = range(self.n_atoms) if atoms is None else atoms
        return [i for i in idx if self._is_h(i)]

    def atom_state(self, i: int) -> AtomState:
        a = self._atom(i)
        sym, q = a.GetSymbol(), a.GetFormalCharge()
        if a.HasProp("qb_hyb"):
            hyb, valence = a.GetProp("qb_hyb"), a.GetIntProp("qb_valence")
        else:
            hyb, valence = _perceive(a)
        lp, sigma, _ = _capacity(sym, q, hyb, valence)
        return AtomState(sym, q, hyb, valence, lp, sigma, a.GetDegree(), _bond_orders(a))

    def free_valences(self, atom: int, include_h: bool = True) -> list[FreeValence]:
        """Free valences of ``atom``: its hydrogens (``include_h``), then open directions with no atom on them.
        A hydrogen has none (use ``valence_of`` for the valence it fills)."""
        if self._is_h(atom):
            return []
        st = self.atom_state(atom)
        top = 1 + max(0, st.pi_free)
        out = []
        if include_h:
            p = self._pos(atom)
            out = [FreeValence(atom, tuple(_unit(self._pos(h) - p)), h, top)
                   for h in self.neighbors(atom) if self._is_h(h)]
        n_open = max(0, min(st.sigma_free, st.open))
        out += [FreeValence(atom, tuple(d), None, top) for d in self._open_directions(atom)[:n_open]]
        return out

    def all_free_valences(self, include_h: bool = True) -> list[FreeValence]:
        """Every free valence of the model (what build mode draws as stubs)."""
        return [v for i in range(self.n_atoms) for v in self.free_valences(i, include_h)]

    def valence_of(self, h: int) -> FreeValence:
        """The free valence hydrogen ``h`` fills (clicking a hydrogen stub)."""
        if not self._is_h(h):
            raise EditError(f"{self.label(h)} is not a hydrogen: pick a hydrogen or a free valence")
        nbrs = self.neighbors(h)
        if len(nbrs) != 1:
            raise EditError(f"{self.label(h)} is not bonded to exactly one atom")
        p = nbrs[0]
        st = self.atom_state(p)
        return FreeValence(p, tuple(_unit(self._pos(h) - self._pos(p))), h, 1 + max(0, st.pi_free))

    def problems(self) -> list[str]:
        """Atoms with unfilled valences or too many bonds (what to fix before a calculation)."""
        out = []
        for i in range(self.n_atoms):
            st = self.atom_state(i)
            if st.open > 0:
                what = "double-bond valence" if st.sigma_free <= 0 else "free valence"
                out.append(f"{self.label(i)} ({st.hyb}) has {st.open} unfilled {what}{'s' if st.open > 1 else ''}")
            elif st.open < 0:
                out.append(f"{self.label(i)} has more bonds than its valence ({st.bond_sum} > {st.valence})")
        return out

    # ------------------------------------------------------------------ measuring
    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self._pos(i) - self._pos(j)))

    def angle(self, i: int, j: int, k: int) -> float:
        a, b = _unit(self._pos(i) - self._pos(j)), _unit(self._pos(k) - self._pos(j))
        return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))

    def dihedral(self, i: int, j: int, k: int, m: int) -> float:
        """Dihedral angle in degrees (-180, 180], RDKit's sign convention."""
        return float(rdMolTransforms.GetDihedralDeg(self._mol.GetConformer(), int(i), int(j), int(k), int(m)))

    # ------------------------------------------------------------------ export
    def to_mol(self, fill_open: bool = True) -> Chem.Mol:
        """A sanitized copy with stereochemistry from the 3D coordinates; ``fill_open`` puts hydrogens on open
        valences (a pi bond still to make stays a radical). Raises EditError for an invalid structure."""
        m = self._completed() if fill_open else Chem.RWMol(self._mol)
        for a in m.GetAtoms():
            a.ClearProp("qb_hyb")
            a.ClearProp("qb_valence")
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(m)
        except Exception as exc:  # noqa: BLE001 - RDKit's message is the reason
            raise EditError(f"The structure is not valid: {str(exc).strip() or type(exc).__name__}") from None
        if m.GetNumAtoms():
            Chem.AssignStereochemistryFrom3D(m)
        return m.GetMol()

    def to_smiles(self, isomeric: bool = True) -> str:
        with rdBase.BlockLogs():
            return Chem.MolToSmiles(Chem.RemoveHs(self.to_mol()), isomericSmiles=isomeric)

    def to_molblock(self) -> str:
        return Chem.MolToMolBlock(self.to_mol())

    def to_input(self, id: str | None = None, name: str = ""):
        """A ``chem.inputs.MoleculeInput`` for the pipeline (its ``mol`` carries the 3D coordinates)."""
        from .inputs import MoleculeInput, safe_id

        smiles = self.to_smiles()
        return MoleculeInput(id=safe_id(id or name or smiles), smiles=smiles, name=name or id or "",
                             source="model kit", mol=self.to_mol())

    def energy(self) -> tuple[float, str]:
        """Force-field energy (kcal/mol) of the current coordinates and the force field used."""
        if self.n_atoms < 2:
            return 0.0, "none"
        m = self._completed()
        with rdBase.BlockLogs():
            Chem.SanitizeMol(m)
        ff, name = _forcefield(m, self.variant)
        return float(ff.CalcEnergy()), name

    # ------------------------------------------------------------------ edits (each one undoable)
    def do(self, command: Command):
        """Run ``command`` through ``history`` and return its result."""
        return self.history.push(command)

    def add_atom(self, kind: str | AtomType = "C sp3", position=(0.0, 0.0, 0.0), *, fill_h: bool | None = None) -> int:
        """Place a new atom on its own (Spartan's first click in empty space). Returns its index."""
        return self.do(AddAtom(kind, tuple(position), fill_h))

    def add_fragment(self, fragment: str | Fragment, position=(0.0, 0.0, 0.0)) -> Placement:
        return self.do(AddFragment(fragment, tuple(position)))

    def attach_atom(self, valence: FreeValence | int, kind: str | AtomType, *, order: int | None = None,
                    fill_h: bool | None = None) -> int:
        """Attach an atom of ``kind`` at ``valence`` (a FreeValence, or a hydrogen's index). Returns the new
        atom's index: the replaced hydrogen's index, or a new one for an open valence."""
        return self.do(AttachAtom(valence, kind, order, fill_h))

    def attach_fragment(self, valence: FreeValence | int, fragment: str | Fragment, *,
                        fill_h: bool | None = None) -> Placement:
        return self.do(AttachFragment(valence, fragment, fill_h))

    def replace_atom(self, atom: int, kind: str | AtomType, *, fill_h: bool | None = None) -> int:
        """Change ``atom``'s element, hybridization or charge in place and rebuild its hydrogens. Returns its
        index afterwards (removing hydrogens can move it down). On a hydrogen this attaches instead."""
        return self.do(ReplaceAtom(atom, kind, fill_h))

    def delete_atoms(self, atoms: Iterable[int], *, fill_h: bool | None = None) -> dict[int, int]:
        """Delete atoms with their hydrogens; their neighbours get hydrogens where the bonds were (``fill_h``).
        Returns old index -> new index of the atoms kept."""
        return self.do(DeleteAtoms(tuple(atoms), fill_h))

    def delete_fragment(self, atom: int) -> dict[int, int]:
        """Delete the whole molecule (connected fragment) ``atom`` belongs to."""
        return self.do(DeleteAtoms(tuple(self._component(atom)), False))

    def set_bond_order(self, i: int, j: int, order: int, *, fill_h: bool | None = None) -> dict[int, int]:
        """Make the ``i``-``j`` bond single, double or triple (0 breaks it); both atoms are re-hybridized and
        get or lose hydrogens. Returns old index -> new index."""
        return self.do(SetBondOrder(i, j, order, fill_h))

    def set_charge(self, atom: int, charge: int, *, fill_h: bool | None = None) -> dict[int, int]:
        """Set a formal charge; the atom gets or loses hydrogens (an amine N+1 becomes ammonium, a hydroxyl O-1
        an oxide). Returns old index -> new index."""
        return self.do(SetCharge(atom, charge, fill_h))

    def make_bond(self, i: int, j: int, order: int = 1) -> dict[int, int]:
        """Bond two atoms, removing a hydrogen from each when they have no open valence (the one pointing
        most toward the other atom). Returns old index -> new index."""
        return self.do(MakeBond(i, j, order))

    def break_bond(self, i: int, j: int, *, fill_h: bool | None = None) -> None:
        return self.do(BreakBond(i, j, fill_h))

    def set_distance(self, i: int, j: int, value: float) -> None:
        """Set a bond length (or the distance between two separate molecules), moving the smaller side."""
        return self.do(SetDistance(i, j, value))

    def set_angle(self, i: int, j: int, k: int, value: float) -> None:
        """Set the ``i``-``j``-``k`` angle (degrees), rotating the smaller side about ``j``."""
        return self.do(SetAngle(i, j, k, value))

    def set_dihedral(self, i: int, j: int, k: int, m: int, value: float) -> None:
        """Set the ``i``-``j``-``k``-``m`` dihedral (degrees), rotating the smaller side about the ``j``-``k``
        bond. A bond in a ring is refused."""
        return self.do(SetDihedral(i, j, k, m, value))

    def fill_hydrogens(self) -> list[int]:
        """Put hydrogens on every open valence (Spartan does this when you leave the builder)."""
        return self.do(FillHydrogens())

    def minimize(self, frozen: Iterable[int] = (), *, max_iters: int = 2000) -> MinimizeResult:
        """MMFF94 (UFF when MMFF has no parameters) with ``frozen`` atoms fixed. Open valences count as
        hydrogens while minimizing."""
        return self.do(Minimize(tuple(frozen), max_iters))

    def relax(self, atoms: Iterable[int] | None = None, *, radius: int = 2, max_iters: int = 200) -> MinimizeResult:
        """Quick local relax: only atoms within ``radius`` bonds of ``atoms`` (default: the last edit) move."""
        return self.do(Relax(None if atoms is None else tuple(atoms), radius, max_iters))

    # ================================================================== internals
    def _atom(self, i: int) -> Chem.Atom:
        if not 0 <= int(i) < self.n_atoms:
            raise EditError(f"No atom {i} (the model has {self.n_atoms})")
        return self._mol.GetAtomWithIdx(int(i))

    def _is_h(self, i: int) -> bool:
        return self._atom(i).GetAtomicNum() == 1

    def _pos(self, i: int) -> np.ndarray:
        p = self._mol.GetConformer().GetAtomPosition(int(i))
        return np.array([p.x, p.y, p.z])

    def _set_pos(self, i: int, xyz) -> None:
        self._mol.GetConformer().SetAtomPosition(int(i), Point3D(*map(float, xyz)))

    def _set_positions(self, pos: np.ndarray) -> None:
        conf = self._mol.GetConformer()
        for i, p in enumerate(np.asarray(pos, float)):
            conf.SetAtomPosition(i, Point3D(*p))

    def _snapshot(self):
        return Chem.RWMol(self._mol), self.last_edit

    def _restore(self, snapshot) -> None:
        self._mol = Chem.RWMol(snapshot[0])
        self.last_edit = snapshot[1]

    def _notify(self) -> None:
        for fn in list(self.listeners):
            fn(self)

    def _fill_flag(self, fill_h: bool | None) -> bool:
        return self.fill_h if fill_h is None else bool(fill_h)

    def _type_atom(self, a: Chem.Atom, symbol: str, charge: int, hyb: str, valence: int) -> None:
        a.SetAtomicNum(_PT.GetAtomicNumber(symbol))
        a.SetFormalCharge(int(charge))
        a.SetIsotope(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetNumRadicalElectrons(0)
        a.SetIsAromatic(False)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        a.SetProp("qb_hyb", hyb)
        a.SetIntProp("qb_valence", int(valence))

    def _new_atom(self, symbol: str, charge: int, hyb: str, valence: int, xyz) -> int:
        i = self._mol.AddAtom(Chem.Atom(_PT.GetAtomicNumber(symbol)))
        self._type_atom(self._mol.GetAtomWithIdx(i), symbol, charge, hyb, valence)
        self._set_pos(i, xyz)
        return i

    def _set_type(self, i: int, symbol: str, charge: int, hyb: str, valence: int) -> None:
        self._type_atom(self._atom(i), symbol, charge, hyb, valence)

    def _set_hyb(self, i: int, hyb: str, valence: int) -> None:
        a = self._atom(i)
        a.SetProp("qb_hyb", hyb)
        a.SetIntProp("qb_valence", int(valence))

    def _add_bond(self, i: int, j: int, order: int) -> None:
        self._mol.AddBond(int(i), int(j), _BOND_TYPE[order])

    def _set_order(self, i: int, j: int, order: int) -> None:
        self._mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(_BOND_TYPE[order])

    def _remove_atoms(self, atoms: Iterable[int]) -> dict[int, int]:
        """Remove atoms; returns old index -> new index for every atom kept."""
        gone = set(int(a) for a in atoms)
        n = self.n_atoms
        for i in sorted(gone, reverse=True):
            self._mol.RemoveAtom(i)
        kept = [i for i in range(n) if i not in gone]
        return {old: new for new, old in enumerate(kept)}

    def _identity(self) -> dict[int, int]:
        return {i: i for i in range(self.n_atoms)}

    def _edited(self, touched: Iterable[int]) -> None:
        self.last_edit = tuple(sorted({int(i) for i in touched if 0 <= int(i) < self.n_atoms}))
        if self.auto_relax and self.last_edit:
            try:                       # best effort: a half-built structure may have no force field yet
                self._op_relax(self.last_edit, 2, 200)
            except EditError:
                pass

    # ------------------------------------------------------------------ valence geometry
    def _open_directions(self, i: int, among: Sequence[int] | None = None) -> list[np.ndarray]:
        """Directions of ``i``'s sigma slots not taken by the neighbours ``among`` (default: all of them)."""
        st = self.atom_state(i)
        nbrs = list(self.neighbors(i) if among is None else among)
        p = self._pos(i)
        dirs = [_unit(self._pos(n) - p) for n in nbrs]
        reference = None
        if len(nbrs) == 1:
            b = nbrs[0]
            others = sorted((c for c in self.neighbors(b) if c != i), key=lambda c: (self._is_h(c), c))
            if others:
                r = self._pos(others[0]) - self._pos(b)
                reference = -(r - np.dot(r, dirs[0]) * dirs[0])
        remaining = domain_directions(dirs, DOMAINS.get(st.hyb, 0), reference)
        return remaining[:max(0, st.sigma_slots - len(nbrs))]

    def _resolve(self, valence: FreeValence | int) -> FreeValence:
        if isinstance(valence, FreeValence):
            self._atom(valence.atom)
            if valence.h is not None:
                v = self.valence_of(valence.h)
                if v.atom != valence.atom:
                    raise EditError(f"{self.label(valence.h)} is not bonded to {self.label(valence.atom)}")
                return v
            st = self.atom_state(valence.atom)
            if st.sigma_free <= 0 or st.open <= 0:
                raise EditError(f"{self.label(valence.atom)} has no open valence there")
            return FreeValence(valence.atom, tuple(_unit(valence.vector)), None, 1 + max(0, st.pi_free))
        if isinstance(valence, (int, np.integer)):
            return self.valence_of(int(valence))
        raise TypeError("valence must be a FreeValence or a hydrogen's index")

    def _fill(self, i: int) -> list[int]:
        """Hydrogens on ``i``'s open sigma slots; returns the new atoms."""
        if self._is_h(i):
            return []
        st = self.atom_state(i)
        n = min(st.sigma_free, st.open)
        if n <= 0:
            return []
        p = self._pos(i)
        added = []
        for d in self._open_directions(i)[:n]:
            h = self._new_atom("H", 0, "s", 1, p + d * bond_length(st.symbol, "H"))
            self._add_bond(i, h, 1)
            added.append(h)
        return added

    def _rebuild_h(self, i: int, fill: bool) -> dict[int, int]:
        """Put ``i``'s hydrogens where its hybridization wants them (adding or removing some)."""
        st = self.atom_state(i)
        nbrs = self.neighbors(i)
        heavy = [n for n in nbrs if not self._is_h(n)]
        hs = [n for n in nbrs if self._is_h(n)]
        heavy_sum = sum(self.bond_order(i, n) for n in heavy)
        want = max(0, min(st.sigma_slots - len(heavy), st.valence - heavy_sum))
        if not fill:
            want = min(want, len(hs))
        dirs = self._open_directions(i, heavy)[:want]
        p, length = self._pos(i), bond_length(st.symbol, "H")
        for h, d in zip(hs, dirs):
            self._set_pos(h, p + d * length)
        for d in dirs[len(hs):]:
            h = self._new_atom("H", 0, "s", 1, p + d * length)
            self._add_bond(i, h, 1)
        extra = hs[len(dirs):]
        return self._remove_atoms(extra) if extra else self._identity()

    def _adjust_h(self, i: int, fill: bool) -> dict[int, int]:
        """Add hydrogens to open slots, or remove some when ``i`` has more bonds than it can make."""
        st = self.atom_state(i)
        excess = max(-st.open, -st.sigma_free, 0)
        if excess:
            hs = [n for n in self.neighbors(i) if self._is_h(n)]
            if len(hs) < excess:
                raise EditError(f"{self.label(i)} would have too many bonds")
            return self._remove_atoms(hs[-excess:])
        if fill:
            self._fill(i)
        return self._identity()

    def _raise_valence(self, i: int, extra_pi: int, what: str) -> bool:
        """Give ``i`` ``extra_pi`` more pi bonds by raising its valence (S, P); False when none are needed."""
        if extra_pi <= 0:
            return False
        st = self.atom_state(i)
        target = st.valence + 2 * extra_pi
        if target not in HYPERVALENT.get(st.symbol, ()) or _PT.GetNOuterElecs(st.symbol) - st.charge - target < 0:
            raise EditError(f"{self.label(i)} ({st.symbol} {st.hyb}) has no free {_ORDER_NAME[1 + extra_pi]}"
                            f" valence for {what}: use an sp2 or sp atom, or change the bond order afterwards")
        self._set_hyb(i, st.hyb, target)
        return True

    # ------------------------------------------------------------------ graph helpers
    def _bfs(self, start: int, blocked_atom: int | None = None, cut: tuple[int, int] | None = None) -> set[int]:
        seen = {start}
        queue = deque([start])
        cut_set = set(cut) if cut else set()
        while queue:
            a = queue.popleft()
            for n in self.neighbors(a):
                if n == blocked_atom or n in seen or (cut and {a, n} == cut_set):
                    continue
                seen.add(n)
                queue.append(n)
        return seen

    def _component(self, atom: int) -> set[int]:
        self._atom(atom)
        return self._bfs(atom)

    def _rotate(self, atoms: Iterable[int], axis, angle_rad: float, center) -> None:
        rot = _rotation(axis, angle_rad)
        c = np.asarray(center, float)
        for a in atoms:
            self._set_pos(a, (self._pos(a) - c) @ rot.T + c)

    # ------------------------------------------------------------------ operations (run by commands)
    def _op_add_atom(self, kind, position, fill_h) -> int:
        t = atom_type(kind)
        i = self._new_atom(t.symbol, t.charge, t.hyb, t.total_valence, position)
        added = self._fill(i) if self._fill_flag(fill_h) else []
        self._edited([i, *added])
        return i

    def _op_add_fragment(self, fragment, position) -> Placement:
        tpl = _template(_fragment(fragment).smiles)
        pos = np.array(tpl.positions)
        a_sym = tpl.symbols[tpl.anchor]
        stub = pos[tpl.anchor] + tpl.attach_dir * bond_length(a_sym, "H")
        shift = np.asarray(position, float) - np.vstack([pos, stub]).mean(axis=0)
        idx = [self._new_atom(s, q, h, v, xyz + shift)
               for s, q, h, v, xyz in zip(tpl.symbols, tpl.charges, tpl.hybs, tpl.valences, pos)]
        for a, b, o in tpl.bonds:
            self._add_bond(idx[a], idx[b], o)
        entry = None
        if tpl.order == 1:
            entry = self._new_atom("H", 0, "s", 1, stub + shift)
            self._add_bond(idx[tpl.anchor], entry, 1)
        self._edited(idx + ([entry] if entry is not None else []))
        return Placement(anchor=idx[tpl.anchor], atoms=tuple(idx),
                         exit=idx[tpl.exit] if tpl.exit is not None else None, entry=entry)

    def _op_attach_atom(self, valence, kind, order, fill_h) -> int:
        t = atom_type(kind)
        fill = self._fill_flag(fill_h)
        v = self._resolve(valence)
        p = v.atom
        st = self.atom_state(p)
        if order is None:
            order = t.bond_order or 1 + min(max(0, st.pi_free), t.pi)
        if order not in (1, 2, 3):
            raise EditError(f"Bond order must be 1, 2 or 3, not {order}")
        if order - 1 > t.pi:
            raise EditError(f"{t.key} cannot make a {_ORDER_NAME[order]} bond")
        raised = self._raise_valence(p, order - 1 - max(0, st.pi_free), t.key)
        xyz = self._pos(p) + v.vector * bond_length(st.symbol, t.symbol, order)
        if v.h is not None:
            x = v.h
            self._set_type(x, t.symbol, t.charge, t.hyb, t.total_valence)
            self._set_pos(x, xyz)
            self._set_order(p, x, order)
        else:
            x = self._new_atom(t.symbol, t.charge, t.hyb, t.total_valence, xyz)
            self._add_bond(p, x, order)
        touched = [p, x]
        if fill:
            touched += self._fill(x)
            if raised:
                touched += self._fill(p)
        self._edited(touched)
        return x

    def _op_attach_fragment(self, valence, fragment, fill_h) -> Placement:
        frag = _fragment(fragment)
        tpl = _template(frag.smiles)
        v = self._resolve(valence)
        p = v.atom
        st = self.atom_state(p)
        raised = self._raise_valence(p, tpl.order - 1 - max(0, st.pi_free), frag.caption)
        d = v.vector
        a_sym = tpl.symbols[tpl.anchor]
        origin = self._pos(p) + d * bond_length(st.symbol, a_sym, tpl.order)
        local = (np.array(tpl.positions) - tpl.positions[tpl.anchor]) @ _align(tpl.attach_dir, -d).T
        skip = {p} | ({v.h} if v.h is not None else set())
        others = np.array([self._pos(i) for i in range(self.n_atoms) if i not in skip]).reshape(-1, 3)
        coords = origin + local @ _best_torsion(local, tpl.anchor, d, origin, others).T
        idx = []
        for k, (s, q, h, val) in enumerate(zip(tpl.symbols, tpl.charges, tpl.hybs, tpl.valences)):
            if k == tpl.anchor and v.h is not None:
                self._set_type(v.h, s, q, h, val)
                self._set_pos(v.h, coords[k])
                idx.append(v.h)
            else:
                idx.append(self._new_atom(s, q, h, val, coords[k]))
        for a, b, o in tpl.bonds:
            self._add_bond(idx[a], idx[b], o)
        if v.h is not None:
            self._set_order(p, v.h, tpl.order)
        else:
            self._add_bond(p, idx[tpl.anchor], tpl.order)
        touched = [p, *idx]
        if raised and self._fill_flag(fill_h):
            touched += self._fill(p)
        self._edited(touched)
        return Placement(anchor=idx[tpl.anchor], atoms=tuple(idx), exit=idx[tpl.exit] if tpl.exit is not None else None)

    def _fit_valence(self, i: int, symbol: str, charge: int, heavy_sum: int, n_heavy: int, pi: int,
                     valences: Sequence[int], hyb: str | None = None) -> tuple[str, int]:
        """(hybridization, valence) for atom ``i`` with ``n_heavy`` heavy neighbours, bond-order sum
        ``heavy_sum`` and ``pi`` pi bonds, trying ``valences`` in order. With ``hyb`` given the
        hybridization is fixed; otherwise it follows from the domains. Raises EditError when none fits."""
        e = _PT.GetNOuterElecs(symbol) - charge if symbol != "H" else 1
        for valence in valences:
            if valence < heavy_sum or (symbol != "H" and e - valence < 0):
                continue
            if hyb is not None:
                lp, sigma, pi_cap = _capacity(symbol, charge, hyb, valence)
                if n_heavy <= sigma and pi <= pi_cap:
                    return hyb, valence
                continue
            lp = max(0, (e - valence) // 2) if symbol != "H" else 0
            domains = (valence - pi) + lp
            if domains in (2, 3, 4) and n_heavy <= valence - pi:
                return _HYB_OF[domains], valence
            if symbol == "H" and valence == 1 and n_heavy <= 1 and pi == 0:
                return "s", 1
        raise EditError(f"{self.label(i)} cannot be {symbol}{'+' if charge > 0 else '-' if charge < 0 else ''} "
                        f"with {n_heavy} bond(s) to heavy atoms (bond orders {heavy_sum})")

    def _op_replace(self, atom, kind, fill_h) -> int:
        t = atom_type(kind)
        fill = self._fill_flag(fill_h)
        if self._is_h(atom):
            if t.symbol == "H":
                return atom
            return self._op_attach_atom(self.valence_of(atom), t, None, fill_h)
        heavy = [n for n in self.neighbors(atom) if not self._is_h(n)]
        orders = {n: self.bond_order(atom, n) for n in heavy}
        raised = None
        if t.bond_order:
            if len(heavy) != 1:
                raise EditError(f"{t.key} replaces an atom with exactly one bond to a heavy atom")
            n = heavy[0]
            extra = t.bond_order - orders[n]
            if extra > 0 and self._raise_valence(n, extra - max(0, self.atom_state(n).pi_free), t.key):
                raised = n
            orders[n] = t.bond_order
        heavy_sum = sum(orders.values())
        pi = sum(o - 1 for o in orders.values())
        valences = [t.total_valence, *(v for v in HYPERVALENT.get(t.symbol, ()) if v > t.total_valence)]
        hyb, valence = self._fit_valence(atom, t.symbol, t.charge, heavy_sum, len(heavy), pi, valences, t.hyb)
        self._set_type(atom, t.symbol, t.charge, hyb, valence)
        for n, o in orders.items():
            if self.bond_order(atom, n) != o:
                self._set_order(atom, n, o)
        mapping = self._rebuild_h(atom, fill)
        new = mapping[atom]
        touched = [new, *(mapping[n] for n in heavy), *self.neighbors(new)]
        if raised is not None and fill:
            touched += self._fill(mapping[raised])
        self._edited(touched)
        return new

    def _op_delete(self, atoms, fill_h) -> dict[int, int]:
        asked = {int(a) for a in atoms}
        for a in asked:
            self._atom(a)
        targets = set(asked)
        for a in asked:
            if not self._is_h(a):
                targets |= {n for n in self.neighbors(a) if self._is_h(n)}
        refill = [(n, _unit(self._pos(a) - self._pos(n)))
                  for a in sorted(targets) if not self._is_h(a) for n in self.neighbors(a) if n not in targets]
        mapping = self._remove_atoms(targets)
        touched = []
        fill = self._fill_flag(fill_h)
        for n, d in refill:
            i = mapping[n]
            touched.append(i)
            st = self.atom_state(i)
            if fill and st.sigma_free > 0 and st.open > 0:
                h = self._new_atom("H", 0, "s", 1, self._pos(i) + d * bond_length(st.symbol, "H"))
                self._add_bond(i, h, 1)
                touched.append(h)
        self._edited(touched)
        return mapping

    def _op_set_bond_order(self, i, j, order, fill_h) -> dict[int, int]:
        if order == 0:
            self._op_break_bond(i, j, fill_h)
            return self._identity()
        if order not in (1, 2, 3):
            raise EditError(f"Bond order must be 0, 1, 2 or 3, not {order}")
        old = self.bond_order(i, j)
        if not old:
            raise EditError(f"{self.label(i)} and {self.label(j)} are not bonded")
        if old == order:
            return self._identity()
        if order > 1 and (self._is_h(i) or self._is_h(j)):
            raise EditError("A bond to hydrogen can only be single")
        plans = {}
        for x, y in ((i, j), (j, i)):
            st = self.atom_state(x)
            heavy = [n for n in self.neighbors(x) if not self._is_h(n)]
            orders = [order if n == y else self.bond_order(x, n) for n in heavy]
            valences = [st.valence, *(v for v in HYPERVALENT.get(st.symbol, ()) if v != st.valence)]
            plans[x] = self._fit_valence(x, st.symbol, st.charge, sum(orders), len(heavy),
                                         sum(o - 1 for o in orders), valences)
        self._set_order(i, j, order)
        mapping = self._identity()
        for x in (i, j):
            cur = mapping[x]
            changed = plans[x][0] != self.atom_state(cur).hyb
            self._set_hyb(cur, *plans[x])
            step = self._rebuild_h(cur, self._fill_flag(fill_h)) if changed else self._adjust_h(cur, self._fill_flag(fill_h))
            mapping = {old_i: step[new_i] for old_i, new_i in mapping.items() if new_i in step}
        a, b = mapping[i], mapping[j]
        self._edited([a, b, *self.neighbors(a), *self.neighbors(b)])
        return mapping

    def _op_set_charge(self, atom, charge, fill_h) -> dict[int, int]:
        st = self.atom_state(atom)
        if st.symbol == "H":
            raise EditError("Charges on hydrogen are not supported by the model kit")
        if charge == st.charge:
            return self._identity()
        valence = st.valence + default_valence(st.symbol, charge) - default_valence(st.symbol, st.charge)
        heavy = [n for n in self.neighbors(atom) if not self._is_h(n)]
        orders = [self.bond_order(atom, n) for n in heavy]
        pi_cap = st.valence - st.sigma_slots
        hyb, valence = self._fit_valence(atom, st.symbol, charge, sum(orders), len(heavy),
                                         max(pi_cap, sum(o - 1 for o in orders)), [valence])
        a = self._atom(atom)
        a.SetFormalCharge(int(charge))
        changed = hyb != st.hyb
        self._set_hyb(atom, hyb, valence)
        mapping = self._rebuild_h(atom, self._fill_flag(fill_h)) if changed else self._adjust_h(atom, self._fill_flag(fill_h))
        new = mapping[atom]
        self._edited([new, *self.neighbors(new)])
        return mapping

    def _op_make_bond(self, i, j, order) -> dict[int, int]:
        if i == j:
            raise EditError("Pick two different atoms")
        if self.bond_order(i, j):
            raise EditError(f"{self.label(i)} and {self.label(j)} are already bonded (change the bond order instead)")
        if self._is_h(i) or self._is_h(j):
            raise EditError("Bond heavy atoms: a hydrogen already has its one bond")
        if order not in (1, 2, 3):
            raise EditError(f"Bond order must be 1, 2 or 3, not {order}")
        remove = []
        for x, y in ((i, j), (j, i)):
            st = self.atom_state(x)
            self._raise_valence(x, order - 1 - max(0, st.pi_free), f"a {_ORDER_NAME[order]} bond")
            st = self.atom_state(x)
            k = max(1 - st.sigma_free, order - st.open, 0)
            toward = _unit(self._pos(y) - self._pos(x))
            hs = sorted((n for n in self.neighbors(x) if self._is_h(n)),
                        key=lambda h, x=x, toward=toward: -float(np.dot(_unit(self._pos(h) - self._pos(x)), toward)))
            if len(hs) < k:
                raise EditError(f"{self.label(x)} has no free valence for another bond")
            remove += hs[:k]
        mapping = self._remove_atoms(remove) if remove else self._identity()
        a, b = mapping[i], mapping[j]
        self._add_bond(a, b, order)
        self._edited([a, b])
        return mapping

    def _op_break_bond(self, i, j, fill_h) -> None:
        if not self.bond_order(i, j):
            raise EditError(f"{self.label(i)} and {self.label(j)} are not bonded")
        if self._is_h(i) or self._is_h(j):
            raise EditError("To remove a hydrogen, delete it")
        di = _unit(self._pos(j) - self._pos(i))
        self._mol.RemoveBond(int(i), int(j))
        touched = [i, j]
        if self._fill_flag(fill_h):
            for x, d in ((i, di), (j, -di)):
                st = self.atom_state(x)
                if st.sigma_free > 0 and st.open > 0:
                    h = self._new_atom("H", 0, "s", 1, self._pos(x) + d * bond_length(st.symbol, "H"))
                    self._add_bond(x, h, 1)
                    touched.append(h)
        self._edited(touched)

    def _op_set_distance(self, i, j, value) -> None:
        if i == j or value <= 0:
            raise EditError("Pick two different atoms and a positive distance")
        if self.bond_order(i, j):
            side_j = self._bfs(j, cut=(i, j))
            if i in side_j:
                raise EditError(f"The {self.label(i)}-{self.label(j)} bond is in a ring: its length cannot be set "
                                "by moving one side (minimize instead)")
            side_i = self._bfs(i, cut=(i, j))
        else:
            side_i = self._component(i)
            if j in side_i:
                raise EditError(f"{self.label(i)} and {self.label(j)} are not bonded: set the length of a bond, or "
                                "the distance between two separate molecules")
            side_j = self._component(j)
        u = _unit(self._pos(j) - self._pos(i))
        delta = value - self.distance(i, j)
        move, vec = (side_j, u * delta) if len(side_j) <= len(side_i) else (side_i, -u * delta)
        for a in move:
            self._set_pos(a, self._pos(a) + vec)

    def _op_set_angle(self, i, j, k, value) -> None:
        if len({i, j, k}) < 3 or not 0 < value < 180:
            raise EditError("Pick three different atoms and an angle between 0 and 180 degrees")
        if not (self.bond_order(i, j) and self.bond_order(j, k)):
            raise EditError(f"{self.label(j)} must be bonded to both {self.label(i)} and {self.label(k)}")
        side_k = self._bfs(k, blocked_atom=j)
        if i in side_k:
            raise EditError(f"{self.label(i)}-{self.label(j)}-{self.label(k)} is in a ring: the angle cannot be "
                            "set by rotating one side (minimize instead)")
        side_i = self._bfs(i, blocked_atom=j)
        c = self._pos(j)
        u_i, u_k = self._pos(i) - c, self._pos(k) - c
        normal = np.cross(u_i, u_k)
        if np.linalg.norm(normal) < 1e-6:
            normal = _perpendicular(_unit(u_i))
        delta = math.radians(value - self.angle(i, j, k))
        if len(side_k) <= len(side_i):
            self._rotate(side_k, normal, delta, c)
        else:
            self._rotate(side_i, normal, -delta, c)

    def _op_set_dihedral(self, i, j, k, m, value) -> None:
        if len({i, j, k, m}) < 4:
            raise EditError("Pick four different atoms")
        if not self.bond_order(j, k):
            raise EditError(f"{self.label(j)} and {self.label(k)} must be bonded")
        side_k = self._bfs(k, cut=(j, k))
        if j in side_k:
            raise EditError(f"The {self.label(j)}-{self.label(k)} bond is in a ring: its dihedral cannot be set "
                            "by rotating one side")
        side_j = self._bfs(j, cut=(j, k))
        if i not in side_j or m not in side_k:
            raise EditError(f"{self.label(i)} must be on {self.label(j)}'s side of the bond and {self.label(m)} on "
                            f"{self.label(k)}'s")
        delta = math.radians(((value - self.dihedral(i, j, k, m)) + 180.0) % 360.0 - 180.0)
        axis = self._pos(k) - self._pos(j)
        if len(side_k) <= len(side_j):     # turning the far side about j->k raises the dihedral
            self._rotate(side_k, axis, delta, self._pos(k))
        else:
            self._rotate(side_j, axis, -delta, self._pos(j))

    def _op_fill(self) -> list[int]:
        added = [h for i in range(self.n_atoms) for h in self._fill(i)]
        if added:
            self._edited(added)
        return added

    def _completed(self) -> Chem.RWMol:
        """A copy with hydrogens on every open sigma slot, appended after the model's atoms."""
        rw = Chem.RWMol(self._mol)
        for i in range(self.n_atoms):
            if self._is_h(i):
                continue
            st = self.atom_state(i)
            n = min(st.sigma_free, st.open)
            if n <= 0:
                continue
            p = self._pos(i)
            for d in self._open_directions(i)[:n]:
                h = rw.AddAtom(Chem.Atom(1))
                rw.GetAtomWithIdx(h).SetNoImplicit(True)
                rw.GetConformer().SetAtomPosition(h, Point3D(*(p + d * bond_length(st.symbol, "H"))))
                rw.AddBond(i, h, Chem.BondType.SINGLE)
        return rw

    def _op_minimize(self, frozen, max_iters, movable: set[int] | None = None) -> MinimizeResult:
        n = self.n_atoms
        if n < 2:
            return MinimizeResult(0.0, "none", True, 0)
        m = self._completed()
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(m)
        except Exception as exc:  # noqa: BLE001
            raise EditError(f"Cannot minimize an invalid structure: {str(exc).strip()}") from None
        ff, name = _forcefield(m, self.variant)
        fixed = {int(f) for f in frozen}
        if movable is not None:
            free = set(movable)
            free |= {h for h in range(n, m.GetNumAtoms())
                     if m.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx() in free}
            fixed |= set(range(m.GetNumAtoms())) - free
        for f in sorted(fixed):
            ff.AddFixedPoint(f)
        ff.Initialize()
        converged = ff.Minimize(maxIts=max_iters) == 0
        energy = float(ff.CalcEnergy())
        self._set_positions(np.array(ff.Positions()).reshape(-1, 3)[:n])
        return MinimizeResult(energy, name, converged, m.GetNumAtoms() - len(fixed))

    def _op_relax(self, atoms, radius, max_iters) -> MinimizeResult:
        centre = [a for a in (atoms if atoms is not None else self.last_edit) if 0 <= a < self.n_atoms]
        if not centre or self.n_atoms < 2:
            return MinimizeResult(0.0, "none", True, 0)
        region = set(centre)
        frontier = set(centre)
        for _ in range(max(0, radius)):
            frontier = {n for a in frontier for n in self.neighbors(a)} - region
            region |= frontier
        region |= {n for a in list(region) for n in self.neighbors(a) if self._is_h(n)}
        return self._op_minimize((), max_iters, movable=region)


def _forcefield(m: Chem.Mol, variant: str):
    with rdBase.BlockLogs():
        if AllChem.MMFFHasAllMoleculeParams(m):
            props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant=variant)
            ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=0, ignoreInterfragInteractions=False)
            if ff is not None:
                return ff, variant
        ff = AllChem.UFFGetMoleculeForceField(m, confId=0, ignoreInterfragInteractions=False)
    if ff is None:
        raise EditError("Neither MMFF94 nor UFF has parameters for this structure")
    return ff, "UFF"


def _best_torsion(local: np.ndarray, anchor: int, axis: np.ndarray, origin: np.ndarray,
                  others: np.ndarray, steps: int = 24) -> np.ndarray:
    """Rotation about ``axis`` (through the anchor) that keeps the fragment farthest from ``others``: the
    largest minimum distance, ties (within 0.05 Å) broken by the smallest sum of r^-6."""
    if len(others) == 0 or len(local) < 2:
        return np.eye(3)
    mask = np.arange(len(local)) != anchor
    scores = []
    for k in range(steps):
        rot = _rotation(axis, 2 * math.pi * k / steps)
        pts = origin + local[mask] @ rot.T
        r = np.linalg.norm(pts[:, None, :] - others[None, :, :], axis=2)
        scores.append((float(r.min()), float((np.maximum(r, 0.3) ** -6).sum()), rot))
    best_min = max(s[0] for s in scores)
    return min((s for s in scores if s[0] >= best_min - 0.05), key=lambda s: s[1])[2]


# ====================================================================== commands and history
class Command:
    """One undoable edit. ``apply`` runs it the first time and replays its stored result afterwards (so a
    redo gives exactly the same coordinates); ``undo`` puts the model back as it was. A command that fails
    leaves the model unchanged and raises EditError."""

    result = None
    _before = None
    _after = None

    @property
    def text(self) -> str:
        return type(self).__name__

    def apply(self, model: EditableMolecule):
        if self._after is not None:
            model._restore(self._after)
        else:
            before = model._snapshot()
            try:
                self.result = self._run(model)
            except Exception:
                model._restore(before)
                raise
            self._before, self._after = before, model._snapshot()
        model._notify()
        return self.result

    def undo(self, model: EditableMolecule) -> None:
        if self._before is None:
            raise RuntimeError("undo() before apply()")
        model._restore(self._before)
        model._notify()

    def _run(self, model: EditableMolecule):
        raise NotImplementedError


@dataclass(eq=False)
class AddAtom(Command):
    kind: str | AtomType = "C sp3"
    position: tuple = (0.0, 0.0, 0.0)
    fill_h: bool | None = None

    @property
    def text(self) -> str:
        return f"Add {atom_type(self.kind).key}"

    def _run(self, model):
        return model._op_add_atom(self.kind, self.position, self.fill_h)


@dataclass(eq=False)
class AddFragment(Command):
    fragment: str | Fragment
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def text(self) -> str:
        return f"Add {_fragment(self.fragment).caption}"

    def _run(self, model):
        return model._op_add_fragment(self.fragment, self.position)


@dataclass(eq=False)
class AttachAtom(Command):
    valence: FreeValence | int
    kind: str | AtomType
    order: int | None = None
    fill_h: bool | None = None

    @property
    def text(self) -> str:
        return f"Attach {atom_type(self.kind).key}"

    def _run(self, model):
        return model._op_attach_atom(self.valence, self.kind, self.order, self.fill_h)


@dataclass(eq=False)
class AttachFragment(Command):
    valence: FreeValence | int
    fragment: str | Fragment
    fill_h: bool | None = None

    @property
    def text(self) -> str:
        return f"Attach {_fragment(self.fragment).caption}"

    def _run(self, model):
        return model._op_attach_fragment(self.valence, self.fragment, self.fill_h)


@dataclass(eq=False)
class ReplaceAtom(Command):
    atom: int
    kind: str | AtomType
    fill_h: bool | None = None

    @property
    def text(self) -> str:
        return f"Replace with {atom_type(self.kind).key}"

    def _run(self, model):
        return model._op_replace(self.atom, self.kind, self.fill_h)


@dataclass(eq=False)
class DeleteAtoms(Command):
    atoms: tuple[int, ...]
    fill_h: bool | None = None
    text = "Delete"

    def _run(self, model):
        return model._op_delete(self.atoms, self.fill_h)


@dataclass(eq=False)
class SetBondOrder(Command):
    i: int
    j: int
    order: int
    fill_h: bool | None = None

    @property
    def text(self) -> str:
        return "Break bond" if self.order == 0 else f"Make {_ORDER_NAME.get(self.order, '?')} bond"

    def _run(self, model):
        return model._op_set_bond_order(self.i, self.j, self.order, self.fill_h)


@dataclass(eq=False)
class SetCharge(Command):
    atom: int
    charge: int
    fill_h: bool | None = None
    text = "Set charge"

    def _run(self, model):
        return model._op_set_charge(self.atom, self.charge, self.fill_h)


@dataclass(eq=False)
class MakeBond(Command):
    i: int
    j: int
    order: int = 1
    text = "Make bond"

    def _run(self, model):
        return model._op_make_bond(self.i, self.j, self.order)


@dataclass(eq=False)
class BreakBond(Command):
    i: int
    j: int
    fill_h: bool | None = None
    text = "Break bond"

    def _run(self, model):
        return model._op_break_bond(self.i, self.j, self.fill_h)


@dataclass(eq=False)
class SetDistance(Command):
    i: int
    j: int
    value: float
    text = "Set distance"

    def _run(self, model):
        return model._op_set_distance(self.i, self.j, self.value)


@dataclass(eq=False)
class SetAngle(Command):
    i: int
    j: int
    k: int
    value: float
    text = "Set angle"

    def _run(self, model):
        return model._op_set_angle(self.i, self.j, self.k, self.value)


@dataclass(eq=False)
class SetDihedral(Command):
    i: int
    j: int
    k: int
    m: int
    value: float
    text = "Set dihedral"

    def _run(self, model):
        return model._op_set_dihedral(self.i, self.j, self.k, self.m, self.value)


@dataclass(eq=False)
class FillHydrogens(Command):
    text = "Add hydrogens"

    def _run(self, model):
        return model._op_fill()


@dataclass(eq=False)
class Minimize(Command):
    frozen: tuple[int, ...] = ()
    max_iters: int = 2000
    text = "Minimize"

    def _run(self, model):
        return model._op_minimize(self.frozen, self.max_iters)


@dataclass(eq=False)
class Relax(Command):
    atoms: tuple[int, ...] | None = None
    radius: int = 2
    max_iters: int = 200
    text = "Relax"

    def _run(self, model):
        return model._op_relax(self.atoms, self.radius, self.max_iters)


class History:
    """Undo and redo stacks of commands applied to one model (the GUI uses a QUndoStack instead)."""

    def __init__(self, model: EditableMolecule, limit: int = 500):
        self.model = model
        self.limit = max(1, limit)
        self._undo: list[Command] = []
        self._redo: list[Command] = []

    def push(self, command: Command):
        result = command.apply(self.model)
        self._undo.append(command)
        del self._undo[:-self.limit]
        self._redo.clear()
        return result

    def undo(self) -> bool:
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo(self.model)
        self._redo.append(cmd)
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.apply(self.model)
        self._undo.append(cmd)
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_text(self) -> str:
        return self._undo[-1].text if self._undo else ""

    @property
    def redo_text(self) -> str:
        return self._redo[-1].text if self._redo else ""

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def __len__(self) -> int:
        return len(self._undo)
