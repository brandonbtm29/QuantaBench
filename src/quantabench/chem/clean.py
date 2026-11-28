"""Clean up a structure (BUILD-CLEAN): a fresh 2D layout of a drawing, and the checks a 3D clean-up must pass.

    layout_2d(mol)                  the 2D layout rule, in place: RDKit's CoordGen, or a pinned 120° zig-zag when the
                                    longest acyclic heavy-atom path has ``ZIGZAG_MIN_ATOMS`` or more (the 2D view
                                    lays out every molecule with it: ``gui.viewer2d.compute_2d_coords``)
    layout_problems(mol)            bonds drawn far from the standard length, atoms drawn on top of each other
    clean_2d(mol) -> Clean2D        ``mol`` laid out again, checked: the rule first, then CoordGen alone, then RDKit's
                                    own depictor, the first whose drawing keeps the connectivity and the stereo (and
                                    is drawn cleanly, when one is)
    clean_molblock(block)           the same for a MOL block (the Sketch dock): wedges recomputed for the new
                                    coordinates; the block's own stereo (wedges, double bonds as drawn) is kept
    drawn(mol)                      what a drawing of ``mol`` says: written as a MOL block and read back
    connectivity_changes(ref, new)  atoms (element, charge, isotope, hydrogens, radicals) and bonds that differ
    stereo_changes(ref, new)        the stereo ``ref`` defines that ``new`` does not keep (same atoms, same order)
    new_stereo(ref, new)            the stereo ``new`` defines where ``ref`` left it open (a centre that was flat)
    geometry_bond_problems(mol)     3D: bonded atoms pulled apart, or atoms pushed into bonding distance

A drawing is checked by what it says, not by the flags the molecule carries: the new layout is written as a MOL block
(wedges from the chiral tags and the new coordinates, double bonds as drawn) and read back with RDKit, and that is
compared with the original. Stereo the original leaves open (an unmarked centre, a double bond drawn as either) is not
compared. A layout that changes anything is not used; when none of the three keeps everything, ``CleanError`` says what
would change and the drawing stays as it was.

The 3D clean-up is the model kit's undoable command ``chem.edit.CleanUp`` (``EditableMolecule.clean_up``): MMFF94
(UFF where MMFF94 has no parameters) with the held atoms fixed, refused with these checks.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point2D

#: Longest acyclic heavy-atom path (atoms) from which the chain is laid out as an explicit zig-zag.
ZIGZAG_MIN_ATOMS = 6
#: RDKit's standard 2D bond length (Å-like depiction units).
DEPICTION_BOND = 1.5
#: 2D: bonds drawn outside this range (times ``DEPICTION_BOND``) make a layout unclean.
BOND_RANGE = (0.7, 1.4)
#: 2D: atoms drawn closer than this (times ``DEPICTION_BOND``) are on top of each other.
OVERLAP = 0.4
#: 3D: a bond longer than this times the sum of the two covalent radii reads as broken.
BOND_STRETCH = 1.6
#: 3D: two atoms that are not bonded and closer than this times the sum of their covalent radii read as bonded.
CONTACT = 1.1
_PT = Chem.GetPeriodicTable()
_DEFINED_BOND = (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS,
                 Chem.BondStereo.STEREOTRANS)
_UNSET_ATOM = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_OTHER)
_BOND_PROPS = ("_MolFileBondStereo", "_MolFileBondCfg", "_UnknownStereo")

Label = Callable[[Chem.Mol, int], str]


class CleanError(ValueError):
    """The clean-up would change the structure (or cannot be checked); the message says what, for the user."""


def atom_label(mol: Chem.Mol, i: int) -> str:
    """``C3``: the element and the 0-based index, as the 3D view's labels and the model kit write atoms."""
    return f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}"


def drawing_label(mol: Chem.Mol, i: int) -> str:
    """``atom 4 (C)``: counted from 1 in drawing order, as the Sketch dock's structure problems are."""
    return f"atom {int(i) + 1} ({mol.GetAtomWithIdx(int(i)).GetSymbol()})"


# ====================================================================== the 2D layout
def longest_acyclic_path(mol: Chem.Mol) -> list[int]:
    """Longest simple path through acyclic heavy atoms (each acyclic component is a tree, so two BFS
    sweeps find its diameter exactly)."""
    ok = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and not a.IsInRing()}
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ok] for i in ok}

    def farthest(start: int) -> tuple[int, dict[int, int | None]]:
        prev: dict[int, int | None] = {start: None}
        queue = [start]
        for u in queue:
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    queue.append(v)
        return queue[-1], prev

    best: list[int] = []
    seen: set[int] = set()
    for s in sorted(ok):
        if s in seen:
            continue
        a, prev = farthest(s)
        seen |= set(prev)
        b, prev = farthest(a)
        path = [b]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        if len(path) > len(best):
            best = path
    return best


def _straighten(mol: Chem.Mol) -> None:
    if mol.GetNumBonds():
        rdDepictor.StraightenDepiction(mol, minimizeRotation=True)


def _with_coordgen(on: bool, fn) -> None:
    prefer = rdDepictor.GetPreferCoordGen()
    try:
        rdDepictor.SetPreferCoordGen(on)
        fn()
    finally:
        rdDepictor.SetPreferCoordGen(prefer)


def _coordgen(mol: Chem.Mol) -> str:
    """CoordGen alone (no pinned chain), bond lengths normalized and straightened."""
    mol.RemoveAllConformers()

    def run():
        rdDepictor.Compute2DCoords(mol)
        if mol.GetNumBonds():
            rdDepictor.NormalizeDepiction(mol, canonicalize=1)
    _with_coordgen(True, run)
    _straighten(mol)
    return "CoordGen"


def _rdkit_depictor(mol: Chem.Mol) -> str:
    """RDKit's own (legacy) depictor: the fallback when CoordGen's layout would change a double bond."""
    mol.RemoveAllConformers()
    _with_coordgen(False, lambda: rdDepictor.Compute2DCoords(mol))
    _straighten(mol)
    return "RDKit depictor"


def layout_problems(mol: Chem.Mol) -> list[str]:
    """What is wrong with ``mol``'s 2D layout: bonds far from the standard length (``BOND_RANGE`` times it) and atoms
    drawn on top of each other (closer than ``OVERLAP`` times it). Empty for a clean drawing."""
    if not mol.GetNumConformers() or mol.GetNumAtoms() < 2:
        return []
    pos = np.asarray(mol.GetConformer().GetPositions(), float)[:, :2]
    lo, hi = BOND_RANGE
    out = []
    bonded = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded.add((min(i, j), max(i, j)))
        d = float(np.linalg.norm(pos[i] - pos[j])) / DEPICTION_BOND
        if not lo <= d <= hi:
            out.append(f"a bond drawn {d:.2f} times the standard length")
    for i in range(len(pos) - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1) / DEPICTION_BOND
        for k in np.nonzero(d < OVERLAP)[0]:
            if (i, i + 1 + int(k)) not in bonded:
                out.append("two atoms drawn on top of each other")
    return out


def layout_2d(mol: Chem.Mol) -> str:
    """Give ``mol`` a single 2D conformer: CoordGen, or a pinned zig-zag for long acyclic chains. Returns which.

    CoordGen gives the best general layouts (rings, macrocycles), but it draws long perfluoroalkyl chains as crankshafts
    or curls, and RDKit's legacy depictor draws CF2 chains as straight lines. When the longest acyclic heavy-atom path
    has ``ZIGZAG_MIN_ATOMS`` or more, it is pinned to an ideal 120° zig-zag and RDKit's depictor places everything else
    around it (the two F of each CF2 then splay outward, the way PFAS are drawn in the literature). Where the depictor
    cannot place the rest around the pinned chain (GenX: its C=O and a C-F came out at half length), CoordGen's layout
    is used instead (``layout_problems``). Bond lengths are RDKit's 1.5 units and the result is straightened (most
    bonds at 30°/90° to the x axis) with minimal rotation, so long molecules come out horizontal.
    """
    path = longest_acyclic_path(mol)
    if len(path) < ZIGZAG_MIN_ATOMS:
        return _coordgen(mol)
    mol.RemoveAllConformers()
    dx = DEPICTION_BOND * math.cos(math.radians(30.0))
    dy = DEPICTION_BOND * 0.5
    cmap = {atom: Point2D(k * dx, (k % 2) * dy) for k, atom in enumerate(path)}
    # CoordGen does not honour a coordinate map here
    _with_coordgen(False, lambda: rdDepictor.Compute2DCoords(mol, coordMap=cmap))
    _straighten(mol)
    bad = len(layout_problems(mol))
    if bad:
        other = Chem.Mol(mol)
        _coordgen(other)
        if len(layout_problems(other)) < bad:
            mol.RemoveAllConformers()
            mol.AddConformer(Chem.Conformer(other.GetConformer()), assignId=True)
            return "CoordGen"
    return "zig-zag"


#: The layouts ``clean_2d`` tries, in order.
LAYOUTS: tuple[Callable[[Chem.Mol], str], ...] = (layout_2d, _coordgen, _rdkit_depictor)


# ====================================================================== drawing and reading back
def molblock_2d(mol: Chem.Mol) -> str:
    """MOL V3000 block of ``mol``'s 2D conformer, with the wedges worked out from its chiral tags and these coordinates
    (the wedges a MOL file it was read from had are dropped: they belonged to the old coordinates)."""
    m = Chem.Mol(mol)
    for b in m.GetBonds():
        if b.GetBondDir() in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH, Chem.BondDir.UNKNOWN):
            b.SetBondDir(Chem.BondDir.NONE)
        for prop in _BOND_PROPS:
            if b.HasProp(prop):
                b.ClearProp(prop)
    with rdBase.BlockLogs():
        return Chem.MolToV3KMolBlock(m)


def drawn(mol: Chem.Mol) -> Chem.Mol:
    """What a drawing of ``mol`` (its 2D conformer) says: ``molblock_2d`` read back by RDKit, every atom kept in
    order. Raises CleanError when RDKit cannot read it back."""
    with rdBase.BlockLogs():
        back = Chem.MolFromMolBlock(molblock_2d(mol), sanitize=True, removeHs=False)
    if back is None:
        raise CleanError("RDKit could not read the new layout back, so its stereo cannot be checked.")
    return back


# ====================================================================== the checks
def _atom_key(a: Chem.Atom) -> tuple:
    return (a.GetSymbol(), a.GetFormalCharge(), a.GetIsotope(), a.GetTotalNumHs(), a.GetNumRadicalElectrons())


def _bond_keys(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondTypeAsDouble() for b in mol.GetBonds()}


_ORDER_WORD = {1.0: "single", 1.5: "aromatic", 2.0: "double", 3.0: "triple"}


def connectivity_changes(ref: Chem.Mol, new: Chem.Mol, label: Label = atom_label) -> list[str]:
    """What differs between ``ref`` and ``new`` (the same atoms in the same order): elements, charges, isotopes,
    hydrogen counts and radicals of the atoms, and the bonds with their orders. Empty when nothing does."""
    if ref.GetNumAtoms() != new.GetNumAtoms():
        return [f"{ref.GetNumAtoms()} atoms became {new.GetNumAtoms()}"]
    out = []
    names = ("element", "charge", "isotope", "hydrogens", "unpaired electrons")
    for a, b in zip(ref.GetAtoms(), new.GetAtoms(), strict=True):
        ka, kb = _atom_key(a), _atom_key(b)
        for name, x, y in zip(names, ka, kb, strict=True):
            if x != y:
                out.append(f"{label(ref, a.GetIdx())}: {name} {x} → {y}")
    ba, bb = _bond_keys(ref), _bond_keys(new)
    for pair in sorted(set(ba) | set(bb)):
        i, j = pair
        what = f"{label(ref, i)}-{label(ref, j)}"
        if pair not in bb:
            out.append(f"the {what} bond is gone")
        elif pair not in ba:
            out.append(f"a new {what} bond")
        elif ba[pair] != bb[pair]:
            out.append(f"the {what} bond: {_ORDER_WORD.get(ba[pair], ba[pair])} → "
                       f"{_ORDER_WORD.get(bb[pair], bb[pair])}")
    return out


def _stereo_copy(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    with rdBase.BlockLogs():
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return m


def _defined(mol: Chem.Mol) -> tuple[set[int], set[int]]:
    """Atoms and double bonds (bond indices) with a defined configuration."""
    atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() not in _UNSET_ATOM}
    bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetStereo() in _DEFINED_BOND}
    return atoms, bonds


def _only(mol: Chem.Mol, atoms: set[int], bonds: set[int]) -> Chem.Mol:
    """A copy with only the stereo of ``atoms`` and ``bonds`` (the rest left open)."""
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        if a.GetIdx() not in atoms:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in m.GetBonds():
        if b.GetIdx() not in bonds and b.GetBondType() == Chem.BondType.DOUBLE:
            b.SetStereo(Chem.BondStereo.STEREONONE)
            b.SetBondDir(Chem.BondDir.NONE)
    return m


def _smiles(mol: Chem.Mol) -> str:
    with rdBase.BlockLogs():
        return Chem.MolToSmiles(mol, isomericSmiles=True)


def _cip(mol: Chem.Mol) -> tuple[dict[int, str], dict[int, str]]:
    m = Chem.Mol(mol)
    try:
        from rdkit.Chem import rdCIPLabeler

        with rdBase.BlockLogs():
            rdCIPLabeler.AssignCIPLabels(m)
    except Exception:  # noqa: BLE001 - a graph beyond the labeler's limits: the messages go without R/S
        return {}, {}
    atoms = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    bonds = {b.GetIdx(): b.GetProp("_CIPCode") for b in m.GetBonds() if b.HasProp("_CIPCode")}
    return atoms, bonds


def _bond_name(mol: Chem.Mol, k: int, label: Label) -> str:
    b = mol.GetBondWithIdx(k)
    return f"the {label(mol, b.GetBeginAtomIdx())}={label(mol, b.GetEndAtomIdx())} double bond"


_GROUP_WORD = {Chem.StereoGroupType.STEREO_ABSOLUTE: "abs", Chem.StereoGroupType.STEREO_AND: "and",
               Chem.StereoGroupType.STEREO_OR: "or"}


def _groups(mol: Chem.Mol) -> list[tuple[str, tuple[int, ...]]]:
    """Enhanced stereo groups (abs, and = racemic, or = one of the two) with their atoms."""
    return sorted((_GROUP_WORD.get(g.GetGroupType(), str(g.GetGroupType())), tuple(sorted(a.GetIdx() for a in
                   g.GetAtoms()))) for g in mol.GetStereoGroups())


def stereo_changes(ref: Chem.Mol, new: Chem.Mol, label: Label = atom_label) -> list[str]:
    """The configurations ``ref`` defines that ``new`` (the same atoms in the same order) does not keep: a centre
    inverted or made undefined, a double bond turned from E to Z (or back) or left undefined, an enhanced stereo group
    (abs, and, or) changed. Stereo ``ref`` leaves open is not compared. Empty when every defined configuration is
    kept."""
    a, b = _stereo_copy(ref), _stereo_copy(new)
    if a.GetNumAtoms() != b.GetNumAtoms():
        return ["the atoms changed"]
    out = []
    if _groups(ref) != _groups(new):
        out.append("the enhanced stereo groups (abs, and, or) would change")
    atoms, bonds = _defined(a)
    if not atoms and not bonds:
        return out
    b_atoms, b_bonds = _defined(b)
    ca, cb = _cip(a), _cip(b)
    for i in sorted(atoms):
        if i not in b_atoms:
            out.append(f"{label(ref, i)} ({ca[0].get(i, 'defined')}) would lose its configuration")
        elif i in ca[0] and i in cb[0] and ca[0][i] != cb[0][i]:
            out.append(f"{label(ref, i)} would turn from {ca[0][i]} to {cb[0][i]}")
    for k in sorted(bonds):
        if k not in b_bonds:
            out.append(f"{_bond_name(ref, k, label)} ({ca[1].get(k, 'defined')}) would lose its configuration")
        elif k in ca[1] and k in cb[1] and ca[1][k] != cb[1][k]:
            out.append(f"{_bond_name(ref, k, label)} would turn from {ca[1][k]} to {cb[1][k]}")
    if not out and _smiles(_only(a, atoms, bonds)) != _smiles(_only(b, atoms, bonds)):
        out.append("the stereo would change (a configuration without an R/S or E/Z label: ring cis/trans or similar)")
    return out


def new_stereo(ref: Chem.Mol, new: Chem.Mol, label: Label = atom_label) -> list[str]:
    """The configurations ``new`` defines where ``ref`` (the same atoms) left them open, e.g. a centre that was flat."""
    a, b = _stereo_copy(ref), _stereo_copy(new)
    if a.GetNumAtoms() != b.GetNumAtoms():
        return []
    atoms, bonds = _defined(a)
    b_atoms, b_bonds = _defined(b)
    cb = _cip(b)
    out = [f"{label(ref, i)} is now {cb[0].get(i, 'defined')}" for i in sorted(b_atoms - atoms)]
    out += [f"{_bond_name(ref, k, label)} is now {cb[1].get(k, 'defined')}" for k in sorted(b_bonds - bonds)]
    return out


def geometry_bond_problems(mol: Chem.Mol, label: Label = atom_label, conf_id: int = -1) -> list[str]:
    """3D: bonds of ``mol`` stretched past ``BOND_STRETCH`` times the covalent radii (they read as broken), and pairs
    of atoms that are not bonded but closer than ``CONTACT`` times their radii (they read as bonded: programs that
    take bonds from the coordinates, such as xTB and CREST, would see another molecule). Empty when there are none."""
    n = mol.GetNumAtoms()
    if n < 2 or not mol.GetNumConformers():
        return []
    pos = np.asarray(mol.GetConformer(conf_id).GetPositions(), float)
    rad = np.array([_PT.GetRcovalent(a.GetAtomicNum()) or 0.7 for a in mol.GetAtoms()])
    out = []
    bonded = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonded.add((min(i, j), max(i, j)))
        d = float(np.linalg.norm(pos[i] - pos[j]))
        if d > BOND_STRETCH * (rad[i] + rad[j]):
            out.append(f"the {label(mol, i)}-{label(mol, j)} bond would be {d:.2f} Å long (it reads as broken)")
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        close = np.nonzero(d < CONTACT * (rad[i] + rad[i + 1:]))[0]
        for k in close:
            j = i + 1 + int(k)
            if (i, j) not in bonded:
                out.append(f"{label(mol, i)} and {label(mol, j)} would be {float(d[k]):.2f} Å apart, within bonding "
                           "distance")
    return out


# ====================================================================== 2D clean-up
@dataclass
class Clean2D:
    mol: Chem.Mol                 # ``mol`` with its new 2D conformer (atoms, properties and stereo flags kept)
    layout: str                   # "zig-zag", "CoordGen" or "RDKit depictor"
    notes: list[str] = field(default_factory=list)   # layouts passed over, and why


def clean_2d(mol: Chem.Mol, label: Label = atom_label, ref: Chem.Mol | None = None) -> Clean2D:
    """A copy of ``mol`` laid out again: the layout rule (``layout_2d``) first, then CoordGen alone, then RDKit's own
    depictor, the first whose drawing (``drawn``) has ``ref``'s connectivity and keeps every configuration ``ref``
    defines. ``ref`` defaults to ``mol`` itself: its chiral tags and double-bond stereo. Raises CleanError, saying
    what would change, when none does (``mol`` is not changed)."""
    ref = mol if ref is None else ref
    notes: list[str] = []
    tried: set[str] = set()
    untidy: Clean2D | None = None          # the first layout that keeps the structure but is not drawn cleanly
    for fn in LAYOUTS:
        new = Chem.Mol(mol)
        name = fn(new)
        if name in tried:
            continue
        tried.add(name)
        try:
            back = drawn(new)
        except CleanError as exc:
            notes.append(f"{name}: {exc}")
            continue
        problems = connectivity_changes(ref, back, label) + stereo_changes(ref, back, label)
        if problems:
            notes.append(f"{name}: " + "; ".join(problems[:4]))
            continue
        untidy_why = layout_problems(new)
        if not untidy_why:
            return Clean2D(new, name, notes)
        notes.append(f"{name}: {untidy_why[0]}")
        untidy = untidy or Clean2D(new, name)
    if untidy is not None:
        untidy.notes = notes
        return untidy
    raise CleanError("Clean up was refused: no layout keeps the structure as drawn ("
                     + " | ".join(notes) + "). The drawing was not changed.")


def read_drawing(block: str) -> Chem.Mol:
    """A MOL block as RDKit reads it, every atom kept in drawing order (explicit hydrogens too): stereo from its
    wedges, double bonds as drawn. Raises CleanError with the structure problems ``chem.sketch`` reports."""
    from .sketch import SketchError, check_molblock

    try:
        check_molblock(block)
    except SketchError as exc:
        raise CleanError(str(exc)) from None
    with rdBase.BlockLogs():
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    if mol is None:
        raise CleanError("RDKit could not read the drawing.")
    return mol


def clean_molblock(block: str) -> tuple[str, Clean2D]:
    """The drawing in ``block`` (MOL V2000/V3000) laid out again (``clean_2d``) as a MOL V3000 block, with what was
    done. Atom numbers in messages count from 1 in drawing order. Raises CleanError when the drawing cannot be read
    or no layout keeps its structure."""
    mol = read_drawing(block)
    if mol.GetNumAtoms() == 0:
        raise CleanError("The sketch is empty: draw a molecule first.")
    res = clean_2d(mol, label=drawing_label)
    return molblock_2d(res.mol), res
