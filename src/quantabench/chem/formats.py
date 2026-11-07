"""Structure file formats beyond SDF/MOL, PDB, XYZ and SMILES lists (UI-4): ``inputs.read_file`` calls these.

=========================  =======================================================================================
MOL2 (``.mol2``)           RDKit, one record per ``@<TRIPOS>MOLECULE`` block
Marvin (``.mrv``)          RDKit
ChemDraw (``.cdxml``)      RDKit's ChemDraw reader
ChemDraw (``.cdx``)        binary CDX turned into CDXML (``interchange.cdx_to_cdxml``), then RDKit's reader: RDKit 2026.09's
                           Python reader takes every file as CDXML, so it cannot read binary CDX itself
CML (``.cml``)             a small reader for atoms, bonds, charges and hydrogen counts (CML 2 and the array form)
InChI lists (``.inchi``)   one InChI per line, optionally followed by an id (``inputs.parse_text``)
CIF (``.cif``)             ASE; molecules are rebuilt whole across the cell edges, one entry per unique molecule
Gaussian (``.gjf .com``)   ASE for the geometry (Cartesian or Z-matrix), charge and multiplicity from the file
ORCA (``.inp``)            ``* xyz charge mult`` and ``* xyzfile charge mult file.xyz`` blocks
anything else              Open Babel, when the user has the ``obabel`` program (Settings > Engines): converted to
                           SDF in a temporary folder, then read as SDF. Open Babel is GPL-2.0: it is only ever run
                           as the user's own program, never imported or bundled
=========================  =======================================================================================

Formats without bonds (CIF, Gaussian, ORCA) get them from ``rdkit.Chem.rdDetermineBonds``, with the charge the file
states; a geometry with several molecules is split by connectivity. Each reader returns a ``FileRead``: the records,
which reader was used (``reader``, kept in ``MoleculeInput.meta[READER]`` and logged), and records it skipped.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds

#: ``MoleculeInput.meta`` keys written by the readers (``chem.import_options`` adds its own)
READER = "qb_reader"           # which reader read the record, e.g. "RDKit (MOL2)", "Open Babel 3.1.1 (via SDF)"
COORDS = "qb_coords"           # "2d" | "3d" | "" (no coordinates): what the file's coordinates are

#: extensions read here, with the reader's name
READERS = {".mol2": "RDKit (MOL2)", ".mrv": "RDKit (Marvin MRV)", ".cdxml": "RDKit (ChemDraw CDXML)",
           ".cdx": "QuantaBench CDX + RDKit (ChemDraw CDX)", ".cml": "QuantaBench (CML)",
           ".cif": "ASE + RDKit rdDetermineBonds (CIF)", ".gjf": "ASE + RDKit rdDetermineBonds (Gaussian input)",
           ".com": "ASE + RDKit rdDetermineBonds (Gaussian input)",
           ".inp": "QuantaBench + RDKit rdDetermineBonds (ORCA input)"}
#: formats whose coordinates are always 3D (all z = 0 there is a planar molecule, not a drawing)
THREE_D = {".xyz", ".pdb", ".mol2", ".cif", ".gjf", ".com", ".inp"}
OPENBABEL_HINT = ("install Open Babel (conda install -c conda-forge openbabel) and QuantaBench converts other formats "
                  "with it (Settings > Engines > Open Babel)")


@dataclass
class Record:
    """One structure as read, before ``ImportOptions``."""

    mol: Chem.Mol
    title: str = ""
    props: dict[str, str] = field(default_factory=dict)
    multiplicity: int | None = None      # stated in the file (Gaussian, ORCA)


@dataclass
class FileRead:
    records: list[Record] = field(default_factory=list)
    reader: str = ""
    skipped: list[str] = field(default_factory=list)      # "record 3 (title): reason"
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ helpers
def coord_kind(mol: Chem.Mol, ext: str = "") -> str:
    """"2d", "3d" or "" (no coordinates). A MOL/SDF-like structure is 2D when its header says so or every z is 0;
    in a 3D-only format (XYZ, PDB, MOL2, CIF, Gaussian, ORCA) a flat molecule is planar, not a drawing."""
    if mol is None or mol.GetNumConformers() == 0:
        return ""
    if ext in THREE_D:
        return "3d"
    conf = mol.GetConformer()
    if not conf.Is3D():
        return "2d"
    if mol.GetNumAtoms() > 1 and all(abs(conf.GetAtomPosition(i).z) < 1e-4 for i in range(mol.GetNumAtoms())):
        return "2d"
    return "3d"


def _reason(exc: Exception) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _sanitize_reason(mol: Chem.Mol | None) -> str:
    if mol is None:
        return "unreadable record"
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(Chem.Mol(mol))
    except Exception as exc:  # noqa: BLE001 - RDKit's message is the reason
        return _reason(exc)
    return "RDKit could not read the record"


# ------------------------------------------------------------------------------------------------ MOL2
def read_mol2(path: Path) -> FileRead:
    """Multi-molecule MOL2: split on ``@<TRIPOS>MOLECULE``; the line after it is the title."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?m)^(?=@<TRIPOS>MOLECULE)", text)
    out = FileRead(reader=READERS[".mol2"])
    k = 0
    for block in blocks:
        if not block.strip().startswith("@<TRIPOS>MOLECULE"):
            continue
        k += 1
        lines = block.splitlines()
        title = lines[1].strip() if len(lines) > 1 else ""
        with rdBase.BlockLogs():
            mol = Chem.MolFromMol2Block(block, removeHs=False)
            if mol is None:
                raw = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
        if mol is None:
            out.skipped.append(f"record {k}{' (' + title + ')' if title else ''}: {_sanitize_reason(raw)}")
            continue
        out.records.append(Record(mol, title))
    if not k:
        raise ValueError(f"No @<TRIPOS>MOLECULE record in {path.name}")
    return out


# ------------------------------------------------------------------------------------------------ Marvin
def read_mrv(path: Path) -> FileRead:
    with rdBase.BlockLogs():
        try:
            mol = Chem.MolFromMrvFile(str(path), removeHs=False)
        except Exception as exc:  # noqa: BLE001 - RDKit raises for malformed XML
            raise ValueError(f"RDKit could not read {path.name}: {_reason(exc)}") from None
    if mol is None:
        raise ValueError(f"RDKit could not read a molecule from {path.name}")
    title = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    return FileRead([Record(mol, title, _props(mol))], READERS[".mrv"])


def _props(mol: Chem.Mol) -> dict[str, str]:
    """The molecule's properties as written (not GetPropsAsDict, which turns "007" into 7)."""
    return {k: mol.GetProp(k) for k in mol.GetPropNames()}


# ------------------------------------------------------------------------------------------------ ChemDraw
def _chemdraw_records(mols, reader: str, name: str) -> FileRead:
    out = FileRead(reader=reader)
    for mol in mols:
        if mol is None:
            continue
        title = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        out.records.append(Record(mol, title, {k: v for k, v in _props(mol).items() if not k.startswith("CDX")}))
    if not out.records:
        raise ValueError(f"No structure in {name} (RDKit found no fragment it could read)")
    return out


def read_cdxml(path: Path) -> FileRead:
    from rdkit.Chem import rdChemDraw

    with rdBase.BlockLogs():
        try:
            mols = rdChemDraw.MolsFromChemDrawFile(str(path), removeHs=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"RDKit could not read {path.name}: {_reason(exc)}") from None
    return _chemdraw_records(mols, READERS[".cdxml"], path.name)


def read_cdx(path: Path) -> FileRead:
    return read_cdx_bytes(path.read_bytes(), path.name)


def read_cdx_bytes(data: bytes, name: str = "CDX data") -> FileRead:
    """Binary ChemDraw CDX (a file, or ChemDraw's clipboard format) -> records, through ``cdx_to_cdxml``."""
    from rdkit.Chem import rdChemDraw

    data = bytes(data)
    xml = data.decode("utf-8", "replace") if data.lstrip()[:1] == b"<" else cdx_to_cdxml(data)
    with rdBase.BlockLogs():
        try:
            mols = rdChemDraw.MolsFromChemDrawBlock(xml, removeHs=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"RDKit could not read the structures of {name}: {_reason(exc)}") from None
    return _chemdraw_records(mols, READERS[".cdx"], name)


def cdx_to_cdxml(data: bytes) -> str:
    """Binary CDX -> CDXML text for RDKit's reader. One converter serves files and the clipboard:
    ``chem.interchange.cdx_to_cdxml`` (checked against ChemDraw-written files; the header is optional)."""
    from .interchange import cdx_to_cdxml as convert

    return convert(data)


# ------------------------------------------------------------------------------------------------ CML
_CML_ORDERS = {"1": Chem.BondType.SINGLE, "S": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE,
               "D": Chem.BondType.DOUBLE, "3": Chem.BondType.TRIPLE, "T": Chem.BondType.TRIPLE,
               "A": Chem.BondType.AROMATIC, "1.5": Chem.BondType.AROMATIC}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _cml_atoms(array: ET.Element) -> list[dict[str, str]]:
    """Atoms of an ``atomArray``: child ``atom`` elements, or the CML 1 array form (``atomID="a1 a2"``...)."""
    atoms = [dict(a.attrib) for a in array if _local(a.tag) == "atom"]
    if atoms:
        return atoms
    columns = {k: v.split() for k, v in array.attrib.items()}
    ids = columns.get("atomID") or []
    return [{("id" if k == "atomID" else k): v[i] for k, v in columns.items() if i < len(v)} for i in range(len(ids))]


def _cml_bonds(array: ET.Element | None) -> list[dict[str, str]]:
    if array is None:
        return []
    bonds = [dict(b.attrib) for b in array if _local(b.tag) == "bond"]
    if bonds:
        return bonds
    refs1, refs2 = array.attrib.get("atomRef1", "").split(), array.attrib.get("atomRef2", "").split()
    orders = array.attrib.get("order", "").split()
    return [{"atomRefs2": f"{a} {b}", "order": orders[i] if i < len(orders) else "1"}
            for i, (a, b) in enumerate(zip(refs1, refs2))]


def _cml_molecule(node: ET.Element) -> Record:
    children = {_local(c.tag): c for c in node}
    if "atomArray" not in children:
        raise ValueError("no atomArray")
    atoms = _cml_atoms(children["atomArray"])
    bonds = _cml_bonds(children.get("bondArray"))
    rw = Chem.RWMol()
    index: dict[str, int] = {}
    coords: list[tuple[float, float, float]] = []
    three_d = all("x3" in a for a in atoms)
    two_d = not three_d and all("x2" in a for a in atoms)
    hcounts: dict[int, int] = {}
    for a in atoms:
        atom = Chem.Atom(a.get("elementType", "C").strip() or "C")
        if a.get("formalCharge"):
            atom.SetFormalCharge(int(float(a["formalCharge"])))
        if a.get("isotopeNumber") or a.get("isotope"):
            atom.SetIsotope(int(float(a.get("isotopeNumber") or a.get("isotope"))))
        if a.get("spinMultiplicity"):
            atom.SetNumRadicalElectrons(max(0, int(float(a["spinMultiplicity"])) - 1))
        idx = rw.AddAtom(atom)
        index[a.get("id", f"a{idx + 1}")] = idx
        if a.get("hydrogenCount"):
            hcounts[idx] = int(float(a["hydrogenCount"]))
        if three_d:
            coords.append((float(a["x3"]), float(a["y3"]), float(a["z3"])))
        elif two_d:
            coords.append((float(a["x2"]), float(a["y2"]), 0.0))
    for b in bonds:
        refs = b.get("atomRefs2", "").split()
        if len(refs) != 2 or refs[0] not in index or refs[1] not in index:
            raise ValueError(f"bond with unknown atoms {b.get('atomRefs2')!r}")
        kind = _CML_ORDERS.get(b.get("order", "1").strip(), Chem.BondType.SINGLE)
        i, j = index[refs[0]], index[refs[1]]
        rw.AddBond(i, j, kind)
        if kind == Chem.BondType.AROMATIC:
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
            rw.GetAtomWithIdx(j).SetIsAromatic(True)
            rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
    for idx, count in hcounts.items():   # hydrogenCount counts the explicit H atoms too
        atom = rw.GetAtomWithIdx(idx)
        explicit = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1)
        atom.SetNumExplicitHs(max(0, count - explicit))
        atom.SetNoImplicit(True)
    mol = rw.GetMol()
    if coords:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(coords):
            conf.SetAtomPosition(i, xyz)
        conf.Set3D(three_d)
        mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    if three_d:
        Chem.AssignStereochemistryFrom3D(mol)
    title = node.attrib.get("title", "") or next((c.text or "" for c in node if _local(c.tag) == "name"), "")
    props = {}
    for c in node.iter():                  # <property title="..."><scalar>value</scalar></property>
        key = c.attrib.get("title") or c.attrib.get("dictRef") if _local(c.tag) == "property" else None
        scalar = next((s.text for s in c if _local(s.tag) == "scalar"), None) if key else None
        if key and scalar is not None:
            props[key] = scalar.strip()
    return Record(mol, title.strip(), props)


def read_cml(path: Path) -> FileRead:
    text = path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"<!(?:ENTITY|DOCTYPE)", text):   # CML needs neither; refusing them rules out entity expansion
        raise ValueError(f"{path.name}: XML with a DOCTYPE or ENTITY declaration is not read")
    try:
        root = ET.fromstring(text)  # noqa: S314 - no DOCTYPE/ENTITY (checked above); expat resolves no external ones
    except ET.ParseError as exc:
        raise ValueError(f"{path.name} is not valid XML: {exc}") from None
    molecules = [m for m in root.iter() if _local(m.tag) == "molecule"
                 and any(_local(c.tag) == "atomArray" for c in m)]
    out = FileRead(reader=READERS[".cml"])
    for k, node in enumerate(molecules, 1):
        try:
            out.records.append(_cml_molecule(node))
        except Exception as exc:  # noqa: BLE001 - reported per molecule
            title = node.attrib.get("title", "") or node.attrib.get("id", "")
            out.skipped.append(f"molecule {k}{' (' + title + ')' if title else ''}: {_reason(exc)}")
    if not molecules:
        raise ValueError(f"No molecule with an atomArray in {path.name}")
    return out


# ------------------------------------------------------------------------------------------------ bonds from geometry
def _xyz_mol(numbers, positions) -> Chem.Mol:
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(numbers))
    for i, (z, xyz) in enumerate(zip(numbers, positions)):
        rw.AddAtom(Chem.Atom(int(z)))
        conf.SetAtomPosition(i, tuple(float(v) for v in xyz))
    conf.Set3D(True)
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    return mol


#: charges tried, in order, for a fragment whose charge the file does not give
_TRY_CHARGES = (0, -1, 1, -2, 2)


def _bond_orders(frag: Chem.Mol, charge: int) -> Chem.Mol | None:
    m = Chem.Mol(frag)
    try:
        with rdBase.BlockLogs():
            rdDetermineBonds.DetermineBondOrders(m, charge=charge)
            Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001 - this charge does not fit; the caller tries the next
        return None
    return m


def _open_shell(frag: Chem.Mol) -> Chem.Mol | None:
    """Single bonds and radicals (RDKit's bond-order assignment is closed-shell only): for a radical stated by the
    file's multiplicity."""
    m = Chem.Mol(frag)
    for a in m.GetAtoms():
        a.SetNoImplicit(True)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(m)
    except Exception:  # noqa: BLE001
        return None
    return m


def molecules_from_geometry(numbers, positions, charge: int | None = 0,
                            multiplicity: int | None = None) -> tuple[list[Chem.Mol], list[str]]:
    """Molecules (with bond orders and formal charges) of a bare geometry, split by connectivity.

    ``charge`` is the total charge from the file (None: unknown). One molecule takes it; several get the charges
    (0 first, then -1, +1, -2, +2) that make each of them a closed structure and add up to it. An open-shell
    molecule (``multiplicity`` > 1) whose bond orders RDKit cannot assign keeps single bonds, with radicals where
    hydrogens are missing. Returns (molecules, notes)."""
    mol = _xyz_mol(numbers, positions)
    with rdBase.BlockLogs():
        rdDetermineBonds.DetermineConnectivity(mol)
    frags = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    notes: list[str] = []
    if len(frags) == 1:
        tries = (charge,) if charge is not None else _TRY_CHARGES
        for q in tries:
            m = _bond_orders(frags[0], q)
            if m is not None:
                if charge is None and q:
                    notes.append(f"charge {q:+d} assigned from the geometry (no charge in the file)")
                return [m], notes
        if (multiplicity or 1) > 1 and not charge and (m := _open_shell(frags[0])) is not None:
            notes.append(f"multiplicity {multiplicity}: single bonds and radicals (bond orders are only assigned to "
                         "closed-shell structures)")
            return [m], notes
        raise ValueError(f"RDKit could not assign bonds to the geometry with charge {charge if charge is not None else 0:+d}")
    options = []
    for frag in frags:
        fits = {q: m for q in _TRY_CHARGES if (m := _bond_orders(frag, q)) is not None}
        if not fits:
            raise ValueError(f"RDKit could not assign bonds to a fragment of {frag.GetNumAtoms()} atoms")
        options.append(fits)
    target = charge if charge is not None else None
    best: dict[int, list[int]] = {0: []}                # total charge -> charges per fragment so far
    for fits in options:
        nxt: dict[int, list[int]] = {}
        for total, chosen in best.items():
            for q in fits:
                key = total + q
                cand = [*chosen, q]
                if key not in nxt or sum(map(abs, cand)) < sum(map(abs, nxt[key])):
                    nxt[key] = cand
        best = nxt
    if target is not None and target in best:
        chosen = best[target]
    else:
        chosen = min(best.values(), key=lambda c: sum(map(abs, c)))
        if target is not None:
            notes.append(f"the fragments' charges ({', '.join(f'{q:+d}' for q in chosen)}) do not add up to the "
                         f"file's charge {target:+d}")
    if any(chosen):
        notes.append(f"{len(frags)} molecules in the geometry, charges {', '.join(f'{q:+d}' for q in chosen)}")
    return [fits[q] for fits, q in zip(options, chosen)], notes


# ------------------------------------------------------------------------------------------------ CIF
def _whole_molecules(atoms) -> list[list[int]]:
    """Atom indices of each molecule of a periodic cell, with the positions unwrapped in place (molecules that cross
    a cell edge are made whole by following their bonds through the minimum image)."""
    import numpy as np
    from ase.neighborlist import NeighborList, natural_cutoffs

    nl = NeighborList(natural_cutoffs(atoms, mult=1.1), self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    orig = atoms.get_positions()
    shift = np.zeros_like(orig)          # lattice translation that brings each atom next to its bonded neighbours
    cell = np.asarray(atoms.get_cell())
    seen = [False] * len(atoms)
    groups = []
    for start in range(len(atoms)):
        if seen[start]:
            continue
        seen[start] = True
        group, todo = [start], [start]
        while todo:
            i = todo.pop()
            idx, offsets = nl.get_neighbors(i)
            for j, off in zip(idx, offsets):
                if not seen[j]:          # j's image bonded to i (as i is placed now)
                    seen[j] = True
                    shift[j] = shift[i] + off @ cell
                    group.append(j)
                    todo.append(j)
        groups.append(sorted(group))
    atoms.set_positions(orig + shift)
    return groups


def read_cif(path: Path) -> FileRead:
    """Molecular crystal: the unit cell (ASE applies the symmetry), molecules rebuilt whole, one entry per unique
    molecule (``n_in_cell`` in its properties: how often it occurs in the cell)."""
    import ase.io

    try:
        atoms = ase.io.read(str(path), format="cif")
    except Exception as exc:  # noqa: BLE001 - ASE's message says what is wrong in the file
        raise ValueError(f"ASE could not read {path.name}: {_reason(exc)}") from None
    groups = _whole_molecules(atoms)
    out = FileRead(reader=READERS[".cif"])
    seen: dict[str, Record] = {}
    numbers, positions = atoms.get_atomic_numbers(), atoms.get_positions()
    for k, group in enumerate(groups, 1):
        try:
            mols, notes = molecules_from_geometry(numbers[group], positions[group], charge=None)
        except ValueError as exc:
            out.skipped.append(f"molecule {k} ({len(group)} atoms): {exc}")
            continue
        out.warnings += [f"{path.name}: molecule {k}: {n}" for n in notes]
        for mol in mols:
            with rdBase.BlockLogs():
                key = Chem.MolToSmiles(Chem.RemoveHs(mol))
            if key in seen:
                seen[key].props["n_in_cell"] = str(int(seen[key].props["n_in_cell"]) + 1)
                continue
            seen[key] = Record(mol, "", {"n_in_cell": "1"})
            out.records.append(seen[key])
    if not groups:
        raise ValueError(f"No atoms in {path.name}")
    return out


# ------------------------------------------------------------------------------------------------ Gaussian, ORCA
def _gaussian_charge(text: str) -> tuple[int, int]:
    """(charge, multiplicity): the first line after the route section and the title (Link 0 lines start with %)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(("%", "!"))):
        i += 1
    if i >= len(lines) or not lines[i].lstrip().startswith("#"):
        raise ValueError("no route section (a line starting with #)")
    while i < len(lines) and lines[i].strip():           # the route may span lines
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines) and lines[i].strip():           # title
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    parts = lines[i].replace(",", " ").split() if i < len(lines) else []
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        raise ValueError("no charge and multiplicity line after the title") from None


def _geometry_records(numbers, positions, charge: int, multiplicity: int, title: str,
                      reader: str) -> FileRead:
    mols, notes = molecules_from_geometry(numbers, positions, charge, multiplicity)
    out = FileRead(reader=reader, warnings=list(notes))
    for mol in mols:
        out.records.append(Record(mol, title, multiplicity=multiplicity if len(mols) == 1 else None))
    return out


def read_gaussian(path: Path) -> FileRead:
    import ase.io

    text = path.read_text(encoding="utf-8", errors="replace")
    charge, mult = _gaussian_charge(text)
    try:
        atoms = ase.io.read(str(path), format="gaussian-in")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"ASE could not read the geometry of {path.name}: {_reason(exc)}") from None
    return _geometry_records(atoms.get_atomic_numbers(), atoms.get_positions(), charge, mult, "",
                             READERS[path.suffix.lower()])


def read_orca(path: Path) -> FileRead:
    """ORCA input: ``* xyz charge mult`` ... ``*`` or ``* xyzfile charge mult name.xyz`` (next to the input)."""
    import ase.io
    from ase.data import atomic_numbers

    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?im)^\s*\*\s*(xyzfile|xyz)\s+(-?\d+)\s+(\d+)\s*(\S*)\s*$", text)
    if not m:
        raise ValueError(f"No '* xyz charge mult' or '* xyzfile' block in {path.name} (other ORCA coordinate "
                         "formats are not read)")
    kind, charge, mult, name = m.group(1).lower(), int(m.group(2)), int(m.group(3)), m.group(4)
    if kind == "xyzfile":
        atoms = ase.io.read(str(path.parent / name), format="xyz")
        numbers, positions = atoms.get_atomic_numbers(), atoms.get_positions()
    else:
        numbers, positions = [], []
        for line in text[m.end():].splitlines():
            if line.strip().startswith("*"):
                break
            parts = line.split()
            if len(parts) < 4 or line.lstrip().startswith("#"):
                continue
            symbol = re.sub(r"[^A-Za-z]", "", parts[0]).capitalize()
            if symbol not in atomic_numbers:
                raise ValueError(f"unknown element {parts[0]!r} in {path.name}")
            numbers.append(atomic_numbers[symbol])
            positions.append([float(v) for v in parts[1:4]])
        if not numbers:
            raise ValueError(f"No atoms in the * xyz block of {path.name}")
    return _geometry_records(numbers, positions, charge, mult, "", READERS[".inp"])


FILE_READERS: dict[str, Callable[[Path], FileRead]] = {
    ".mol2": read_mol2, ".mrv": read_mrv, ".cdxml": read_cdxml, ".cdx": read_cdx, ".cml": read_cml,
    ".cif": read_cif, ".gjf": read_gaussian, ".com": read_gaussian, ".inp": read_orca}


def read(path: Path) -> FileRead:
    return FILE_READERS[path.suffix.lower()](path)


# ------------------------------------------------------------------------------------------------ Open Babel
def openbabel():
    """The user's ``obabel`` program (``external.find_program``: Settings > Engines, QUANTABENCH_OBABEL, PATH,
    conda environments), or None. A ``wsl:`` path is not used here (file conversion runs on this side)."""
    from . import external

    info = external.find_program("obabel")
    return None if info is None or info.via_wsl else info


def openbabel_to_sdf(path: Path, out_dir: Path, timeout: float = 300.0) -> tuple[Path, str]:
    """Convert ``path`` to SDF with Open Babel in ``out_dir``: (the SDF, the reader's name). Raises ValueError when
    Open Babel is missing or writes nothing."""
    info = openbabel()
    if info is None:
        raise ValueError(f"Unsupported file type {path.suffix} ({OPENBABEL_HINT})")
    out = out_dir / (path.stem + ".sdf")
    cmd = [*info.command(), str(path), "-osdf", "-O", str(out)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,  # noqa: S603
                            errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Open Babel ({info.path}) could not run: {exc}") from None
    if not out.is_file() or out.stat().st_size == 0:
        tail = " ".join((cp.stderr or cp.stdout or "").strip().splitlines()[-3:])
        raise ValueError(f"Open Babel could not read {path.name}{': ' + tail if tail else ''}")
    version = f" {info.version}" if info.version else ""
    return out, f"Open Babel{version} (via SDF)"


def with_openbabel(path: Path, read_sdf: Callable[[Path], list]) -> tuple[list, str]:
    """(inputs read from Open Babel's SDF of ``path``, reader name); the temporary folder is removed after."""
    with tempfile.TemporaryDirectory(prefix="qb-obabel-") as d:
        sdf, reader = openbabel_to_sdf(path, Path(d))
        return read_sdf(sdf), reader
