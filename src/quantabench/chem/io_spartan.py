"""Read Spartan '24 documents (``.spartan``): one table row per molecule, coordinates for QuantaBench runs.

Spartan cannot hold very large sets in one document, so big projects end up as many documents of a
hundred or so molecules each. This module reads any number of them, molecule by molecule (never a whole
document, let alone all documents, in memory), into one CSV table keyed by the molecule label and,
optionally, one SDF of all geometries that a QuantaBench run can start from.

CONTRACT:

    available() -> tuple[bool, str]
    is_spartan(path) -> bool                                  # by suffix
    open_container(path) -> OleContainer                      # olefile, imported lazily
    DictContainer({"Molecules/M0001/Input.gz": bytes, ...})   # the same interface without a file (tests)
    parse_input(text) -> SpartanInput                         # Input.gz text
    parse_properties(text) -> dict[str, SpartanProperty]      # PropertyArchive.gz text
    read_molecule(container, storage, source_file, index) -> SpartanMolecule
    iter_molecules(path_or_container, source_file=None) -> Iterator[SpartanMolecule]
    molecule_row(mol) -> dict                                 # one table row (best conformer)
    conformer_rows(mol) -> list[dict]                         # one row per saved conformer
    to_mol(mol, conformer=None) -> RDKit Mol (Spartan's bonds; rdDetermineBonds as a fallback)
    to_result(mol) -> a built MoleculeResult (saved conformers ranked by energy) for MainWindow.add_result
    write_table(paths, table, sdf=None, conformers="best", log=None) -> ImportSummary

File format
-----------
A ``.spartan`` file is an OLE compound document (the container of old ``.doc`` files) with one storage per
molecule, ``Molecules/M0001``, ``Molecules/M0002``, ... (four digits in the samples; any digits accepted),
plus document-level streams (``Document``, ``Sheet/SheetData``, ``JobLog``, ``Embedded/Blob00``) that are not
read. In each molecule storage:

- ``Input.gz``: gzip, CRLF text. Keyword lines first. A keyword line that begins with ``C`` and a blank
  (``C MS_CONFP2_BW HF ...``) is continued on the next line; Spartan's own ``INITIAL_KEYWORDS`` property
  lists that ``"C"`` as a keyword, and the line after the last ``C`` line is the last keyword line. Then the
  label (the CAS number in the PFAS QSAR documents; ``M0018`` or ``M0010[1]`` in a document of extracted
  conformers), a ``charge multiplicity`` line, ``Z x y z`` lines in Å and ``ENDCART``. Optional blocks
  follow: ``ATOMLABELS`` (quoted atom names), ``HESSIAN`` ... ``ENDHESS`` (N per-atom codes, then one
  ``i j order`` line per bond, 1-based, order 1/2/3), ``BEGINPROPIN``, ``BEGINPREFERENCES``. When the label
  line and the ``charge multiplicity`` line are not where the continuation rule puts them, the parser falls
  back to the first two-integer line that is followed by a ``Z x y z`` line (label = the line before it).
- ``PropertyArchive.gz``: gzip, CRLF text of ``PROP`` records. Scalars: ``PROP VALUE NAME = 1.23`` and
  ``PROP STRING NAME = "text"``. Blocks: ``PROP VALUE NAME d1 [d2] BEGIN`` ... ``END`` (floats, row-major,
  shape (d1, d2)) and ``PROP STRING NAME n BEGIN`` (one quoted string per line). The kind may carry
  qualifiers after colons: ``VALUE:BYATOM`` (one row per atom), ``VALUE:ATOMPAIR``, ``:STATIC``.
- ``Geometry``, ``Molecule``, ``Calculations``, ``Archive.carc``: binary, not read. ``Output.gz`` /
  ``Output2.gz``: the program's text output (used here only to work out the units below).

In the documents checked, the Cartesian block of ``Input.gz`` is the molecule's *input* geometry (the
``Geometry`` stream holds the same coordinates). After a conformer search it is still the input structure;
the search's result is the conformer library in ``PropertyArchive.gz``:

- ``CONF_COORDS`` (n_saved, 3M): the saved conformers, Å, for the M atoms listed in ``CONF_ATOM_INDICES``
  (1-based ``Input.gz`` indices). In most molecules of the batch checked M < N: Spartan leaves out atoms it
  marks in ``CONF_ATOMS_PRUNED`` (terminal F/H of rotor groups, sometimes dozens). Spartan's own "extract
  conformers" restores them by a rule that is not in the file; here each missing atom is carried over from
  the input geometry with the rigid motion of its bonded neighbor's frame (:func:`complete_conformers`).
  The stored atoms match Spartan's extracted conformers to 1e-9 Å; the placed ones typically differ from
  Spartan's placement by about half an Å (up to ~2 Å), which matters for an SDF geometry but not for the
  table's values (Spartan's own) or for a QuantaBench run (which builds its conformers from the bonds).
  Such rows say so in ``spartan_notes``.
- ``CONF_ENERGY_LIST`` (n_saved): energies of the final stage in kcal/mol, relative to ``CONF_ENERGY_OFFSET``
  (kcal/mol; the "Base Energy" that Output.gz prints in kJ/mol). Absolute = offset + relative.
- ``CONF_SOLVENERGY_LIST``, ``CONF_MMFF_E`` (n_saved, kcal/mol), ``CONF_DEGENCOUNT`` (n_saved), ``CONF_DIPOLE``
  (n_saved, 4: vector and magnitude, debye; zeros after a DLFF03 search), ``CONFMULTI_METHODS`` (one per stage,
  e.g. MMFF94 then DLFF03), ``CONF_NUM_SAVED``. ``CONF_SELECTED`` / ``CONFMULTI_LOWENTRY`` index the
  stage-1 list, not the saved list, so the best saved conformer is taken as the minimum of ``CONF_ENERGY_LIST``.

Units (checked against the program's own Output.gz, since no manual was available): energies in kcal/mol
(``CONF_ENERGY_LIST``, ``CONF_ENERGY_OFFSET``, ``CONFMULTI_ENERS``, ``PARTITION_ENERGY_INCREMENT``,
``SOLV_*``, ``MM_SOLVATION``, ``DLFF_CORRECTION``), ``CONFORMATIONAL_ENTROPY`` in cal/(mol K) (Output.gz
prints it in J/(mol K), 4.184 times larger), coordinates in Å, dipoles in debye. ``ENERGY_COMMENT`` labels the base energy
"(au)", but Output.gz prints the same number in kJ/mol; it is passed through unchanged as text. The DLFF03
and MMFF energies are on force-field scales, not total electronic energies.

Conformers
----------
A molecule with a conformer library reports ``n_conformers`` (rows of ``CONF_COORDS``) and the properties
and geometry of its lowest-energy saved conformer; ``conformers="all"`` writes one row (and one SDF record)
per saved conformer instead. A molecule without a library has ``n_conformers`` = 1 and the ``Input.gz``
geometry. A document of extracted conformers (each conformer its own molecule storage, labelled ``M0018``,
``M0010[1]``, ...) reads as that many molecules: Spartan does not keep the parent's label there, so join such a
document to its parents by geometry, not by label.

Columns
-------
Every row has ``source_file`` (the path as given), ``molecule_index`` (1-based position in its document),
``storage`` (``M0001``), ``label``, ``duplicate_label`` (1 when the same non-empty label occurs in more than
one row of the whole table, in the same file or another; all rows are kept), ``conformer`` (1-based, only
with ``conformers="all"``), then the QuantaBench columns below, then ``spartan_<name>`` (lower case) for every
other scalar ``PROP`` (numbers as floats, strings as text) and the ``spartan_conf_*`` values of the reported
conformer. Arrays (gradients, coordinates, per-atom blocks) are not written to the table; per-atom blocks of
length N go into the SDF as atom property lists (``spartan_mm_charge``, ...).

Spartan to QuantaBench (a Spartan value takes a QuantaBench column only when it is the same quantity in the
same units; everything else stays ``spartan_*``):

    =====================================  ==================  =============================================
    Spartan                                QuantaBench         note
    =====================================  ==================  =============================================
    Input.gz label line                    label (id: safe)    the key of the table
    Input.gz charge                        charge              e
    Input.gz multiplicity                  multiplicity
    rows of CONF_COORDS (else 1)           n_conformers        saved conformers
    atomic numbers + Spartan's bonds       formula, smiles     RDKit (CalcMolFormula, canonical SMILES)
    atomic numbers                         n_atoms
    =====================================  ==================  =============================================

Not mapped on purpose: the DLFF03/MMFF conformer energies (force-field scale, kcal/mol; QuantaBench's
``best_energy_eV`` is the refinement method's total energy), ``DIPOLE_MAG``/``DIPOLE_VEC`` (MMFF point-charge
dipole of the input geometry, not the conformer's), ``CONFORMATIONAL_ENTROPY`` (QuantaBench has only the
vibrational ``vib_entropy_cal_molK``). The QSAR properties of Spartan's quantum-chemical jobs (HOMO/LUMO,
CPK area/volume, ovality, ESP areas, polarizability, ...) were not in the documents available when this was
written (they hold MMFF/DLFF03 conformer searches), so their Spartan names are not in :data:`COLUMN_MAP` yet;
they come out as ``spartan_<name>`` until a document with them confirms names and units. Candidates once
checked: HOMO/LUMO energy in eV -> ``homo_ev``/``lumo_ev`` (+ ``gap_ev``); dipole in debye ->
``dipole_debye``. CPK area/volume are van der Waals (CPK) surfaces, not QuantaBench's ``sasa_A2`` (solvent
accessible, probe 1.4 Å) or ``volume_A3``, and Spartan's ESP areas are not on QuantaBench's 0.002 au density
surface, so those stay ``spartan_*`` either way.
"""

from __future__ import annotations

import contextlib
import csv
import gzip
import json
import math
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUFFIX = ".spartan"
#: Spartan ``PROP`` name -> (QuantaBench column, factor to QuantaBench's unit). Only names checked on real
#: documents belong here (see the module docstring for the candidates).
COLUMN_MAP: dict[str, tuple[str, float]] = {}
#: Columns every row starts with, in this order.
KEY_COLUMNS = ("source_file", "molecule_index", "storage", "label", "duplicate_label")
CORE_COLUMNS = ("formula", "smiles", "n_atoms", "charge", "multiplicity", "n_conformers")
_STORAGE = re.compile(r"^M\d+$")
_PROP = re.compile(r"^PROP\s+(\S+)\s+(\S+)\s*(.*)$")
_KCAL_PER_EV = 23.060548           # chem.vibrations.KCAL_PER_EV
_KT_KCAL_298 = 0.0019872043 * 298.15


def available() -> tuple[bool, str]:
    from ..resources import has_package

    return (True, "") if has_package("olefile") else (False, "pip install olefile")


def is_spartan(path: str | Path) -> bool:
    return Path(path).suffix.lower() == SUFFIX


# ----------------------------------------------------------------------------------------- containers
class DictContainer:
    """A Spartan document as a mapping of stream path (``Molecules/M0001/Input.gz``) to bytes."""

    def __init__(self, streams: Mapping[str, bytes]):
        self.streams = dict(streams)

    def molecules(self) -> list[str]:
        names = {p.split("/")[1] for p in self.streams if p.startswith("Molecules/") and p.count("/") >= 2}
        return sorted((n for n in names if _STORAGE.match(n)), key=lambda n: int(n[1:]))

    def read(self, path: str) -> bytes | None:
        return self.streams.get(path)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class OleContainer(DictContainer):
    """A ``.spartan`` file read through olefile; streams are read on demand."""

    def __init__(self, path: str | Path):
        import olefile

        self.path = Path(path)
        if not olefile.isOleFile(str(self.path)):
            raise ValueError(f"{self.path.name}: not a Spartan document (not an OLE compound file)")
        self._ole = olefile.OleFileIO(str(self.path))

    def molecules(self) -> list[str]:
        names = {e[1] for e in self._ole.listdir(streams=True, storages=True)
                 if len(e) >= 2 and e[0] == "Molecules" and _STORAGE.match(e[1])}
        return sorted(names, key=lambda n: int(n[1:]))

    def read(self, path: str) -> bytes | None:
        if not self._ole.exists(path):
            return None
        with self._ole.openstream(path) as fh:
            return fh.read()

    def close(self) -> None:
        self._ole.close()


def open_container(path: str | Path) -> OleContainer:
    ok, why = available()
    if not ok:
        raise RuntimeError(f"olefile is not installed ({why})")
    return OleContainer(path)


def decode_stream(data: bytes) -> str:
    """Text of a stream: gunzipped when gzip, CRLF -> LF."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")


# ----------------------------------------------------------------------------------------- Input.gz
@dataclass
class SpartanInput:
    keywords: str
    label: str
    charge: int
    multiplicity: int
    numbers: list[int]
    positions_A: np.ndarray                         # (N, 3)
    atom_labels: list[str] = field(default_factory=list)
    bonds: list[tuple[int, int, int]] = field(default_factory=list)   # 0-based i, j, order


def _two_ints(line: str) -> tuple[int, int] | None:
    p = line.split()
    if len(p) != 2:
        return None
    try:
        return int(p[0]), int(p[1])
    except ValueError:
        return None


def _atom_line(line: str) -> tuple[int, float, float, float] | None:
    p = line.split()
    if len(p) != 4:
        return None
    try:
        return int(p[0]), float(p[1]), float(p[2]), float(p[3])
    except ValueError:
        return None


def parse_input(text: str) -> SpartanInput:
    """Parse ``Input.gz`` text (see the module docstring). Raises ValueError without a geometry."""
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and re.match(r"^C(\s|$)", lines[i]):     # "C ..." continues on the next line
        i += 1
    kw_end = i                                                    # index of the last keyword line
    lab = kw_end + 1
    if not (lab + 1 < len(lines) and _two_ints(lines[lab + 1]) and
            (lab + 2 >= len(lines) or _atom_line(lines[lab + 2]) or lines[lab + 2].strip() == "ENDCART")):
        lab = None
        for k in range(1, len(lines) - 1):
            if _two_ints(lines[k]) and _atom_line(lines[k + 1]):
                lab, kw_end = k - 1, k - 2
                break
        if lab is None:
            raise ValueError("Input.gz: no 'charge multiplicity' line followed by Cartesian coordinates")
    keywords = " ".join(re.sub(r"^C(\s|$)", "", ln).strip() for ln in lines[:kw_end + 1]).strip()
    charge, mult = _two_ints(lines[lab + 1])
    numbers, xyz = [], []
    k = lab + 2
    while k < len(lines):
        a = _atom_line(lines[k])
        if a is None:
            break
        numbers.append(a[0])
        xyz.append(a[1:])
        k += 1
    if not numbers:
        raise ValueError("Input.gz: no Cartesian coordinates")
    rest = lines[k:]
    atom_labels: list[str] = []
    bonds: list[tuple[int, int, int]] = []
    n = len(numbers)
    for j, ln in enumerate(rest):
        tag = ln.strip()
        if tag == "ATOMLABELS":
            for lab_line in rest[j + 1:]:
                if lab_line.strip() == "ENDATOMLABELS":
                    break
                atom_labels.append(lab_line.strip().strip('"'))
        elif tag == "HESSIAN":
            vals: list[str] = []
            for h in rest[j + 1:]:
                if h.strip() == "ENDHESS":
                    break
                vals += h.split()
            triples = vals[n:]                                  # N per-atom codes, then i j order
            if len(triples) % 3 == 0:
                with contextlib.suppress(ValueError):
                    bonds = [(int(triples[t]) - 1, int(triples[t + 1]) - 1, int(triples[t + 2]))
                             for t in range(0, len(triples), 3)]
    return SpartanInput(keywords=keywords, label=lines[lab].strip(), charge=charge, multiplicity=mult,
                        numbers=numbers, positions_A=np.asarray(xyz, dtype=float),
                        atom_labels=atom_labels if len(atom_labels) == n else [], bonds=bonds)


# ----------------------------------------------------------------------------------------- PropertyArchive.gz
@dataclass
class SpartanProperty:
    name: str
    kind: str                       # "VALUE" | "STRING"
    qualifiers: tuple[str, ...]     # ("BYATOM",), ("STATIC",), ("ATOMPAIR",), ...
    shape: tuple[int, ...]          # () for scalars
    value: object                   # float | str | np.ndarray | list[str]

    @property
    def is_scalar(self) -> bool:
        return self.shape == ()


def _float(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def parse_properties(text: str) -> dict[str, SpartanProperty]:
    """Parse ``PropertyArchive.gz`` text into {NAME: SpartanProperty}. Lines it does not understand are skipped;
    a block whose value count does not match its dimensions is kept flat."""
    props: dict[str, SpartanProperty] = {}
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _PROP.match(lines[i])
        i += 1
        if not m:
            continue
        kind_q, name, rest = m.groups()
        kind, *qual = kind_q.split(":")
        rest = rest.strip()
        if rest.startswith("="):
            raw = rest[1:].strip()
            value: object = raw[1:-1] if kind == "STRING" and len(raw) >= 2 and raw[0] == raw[-1] == '"' else raw
            if kind != "STRING":
                value = _float(raw)
            props[name] = SpartanProperty(name, kind, tuple(qual), (), value)
            continue
        toks = rest.split()
        if not toks or toks[-1] != "BEGIN":
            continue
        try:
            dims = tuple(int(t) for t in toks[:-1])
        except ValueError:
            dims = ()
        body: list[str] = []
        while i < len(lines) and lines[i].strip() != "END":
            body.append(lines[i])
            i += 1
        i += 1                                                  # the END line
        if kind == "STRING":
            value = [b.strip()[1:-1] if b.strip().startswith('"') else b.strip() for b in body]
        else:
            arr = np.array([_float(t) for b in body for t in b.split()], dtype=float)
            value = arr.reshape(dims) if dims and math.prod(dims) == arr.size else arr
        props[name] = SpartanProperty(name, kind, tuple(qual), dims or (len(body),), value)
    return props


# ----------------------------------------------------------------------------------------- molecules
@dataclass
class SpartanMolecule:
    source_file: str
    molecule_index: int                              # 1-based position in the document
    storage: str                                     # "M0001"
    input: SpartanInput
    properties: dict[str, SpartanProperty] = field(default_factory=dict)
    conformer_positions_A: np.ndarray | None = None  # (n_saved, N, 3)
    conformer_energies_kcal: np.ndarray | None = None   # absolute (offset + relative), final stage
    best_conformer: int | None = None                # 0-based index into the saved conformers
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.input.label

    @property
    def numbers(self) -> list[int]:
        return self.input.numbers

    @property
    def n_atoms(self) -> int:
        return len(self.input.numbers)

    @property
    def n_conformers(self) -> int:
        return 1 if self.conformer_positions_A is None else len(self.conformer_positions_A)

    def positions_A(self, conformer: int | None = None) -> np.ndarray:
        """Geometry of saved conformer ``conformer`` (default: the lowest-energy one), else of Input.gz."""
        if self.conformer_positions_A is None:
            return self.input.positions_A
        k = self.best_conformer if conformer is None else conformer
        return self.conformer_positions_A[k if k is not None else 0]

    def per_atom(self) -> dict[str, np.ndarray]:
        """``VALUE:BYATOM`` blocks with one value per atom (charges, radii, ...), lower-case names."""
        return {p.name.lower(): np.asarray(p.value, dtype=float) for p in self.properties.values()
                if "BYATOM" in p.qualifiers and isinstance(p.value, np.ndarray) and p.value.shape == (self.n_atoms,)}

    def conformer_values(self, conformer: int | None = None) -> dict:
        """``spartan_conf_*`` values of one saved conformer (default: the lowest-energy one)."""
        if self.conformer_positions_A is None:
            return {}
        k = self.best_conformer if conformer is None else conformer
        out: dict = {"spartan_conf_index": k + 1}
        methods = self.properties.get("CONFMULTI_METHODS")
        if methods is not None and isinstance(methods.value, list) and methods.value:
            out["spartan_conf_method"] = methods.value[-1]
        e = self.conformer_energies_kcal
        if e is not None:
            out["spartan_conf_energy_kcal"] = float(e[k])
            out["spartan_conf_rel_kcal"] = float(e[k] - e.min())
            w = self.boltzmann_weights()
            if w is not None:
                out["spartan_conf_boltzmann_weight"] = float(w[k])
        n = self.n_conformers
        for prop, col in (("CONF_SOLVENERGY_LIST", "spartan_conf_solvation_kcal"),
                          ("CONF_MMFF_E", "spartan_conf_mmff_kcal"), ("CONF_DEGENCOUNT", "spartan_conf_degeneracy")):
            v = _per_conformer(self.properties.get(prop), n)
            if v is not None:
                out[col] = float(v[k])
        d = self.properties.get("CONF_DIPOLE")
        if d is not None and isinstance(d.value, np.ndarray) and d.value.size == 4 * n:
            out["spartan_conf_dipole_debye"] = float(d.value.reshape(n, 4)[k, 3])
        return out

    def boltzmann_weights(self, kt_kcal: float = _KT_KCAL_298) -> np.ndarray | None:
        """Weights of the saved conformers at 298.15 K from the final-stage energies and Spartan's degeneracy
        counts (when present), as Spartan's own "Boltz%" table does."""
        e = self.conformer_energies_kcal
        if e is None:
            return None
        g = _per_conformer(self.properties.get("CONF_DEGENCOUNT"), len(e))
        x = np.exp(-(e - e.min()) / kt_kcal) * (g if g is not None else 1.0)
        return x / x.sum()


def _per_conformer(prop: SpartanProperty | None, n: int) -> np.ndarray | None:
    if prop is None:
        return None
    v = np.atleast_1d(np.asarray(prop.value, dtype=float)).ravel()
    return v if v.size == n else None


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotation R and centroids so that (x - c_src) @ R + c_dst maps ``src`` onto ``dst`` (Kabsch). ``dst`` may
    carry leading batch dimensions (K, n, 3); R and c_dst then do too."""
    cs, cd = src.mean(-2), dst.mean(-2)
    h = np.swapaxes(src - cs, -1, -2) @ (dst - cd[..., None, :])
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    d = np.where(d == 0, 1.0, d)
    fix = np.ones(u.shape[:-1])
    fix[..., -1] = d
    return (u * fix[..., None, :]) @ vt, cs, cd


def complete_conformers(partial: np.ndarray, indices: np.ndarray, reference: np.ndarray,
                        bonds: list[tuple[int, int, int]]) -> np.ndarray:
    """Full (K, N, 3) geometries from K conformers that store only the atoms ``indices`` (0-based; ``partial``
    is (K, M, 3)): each missing atom is carried over from ``reference`` (the Input.gz geometry) with the rigid
    motion that maps its bonded neighbor and that neighbor's other stored neighbors (at least three atoms,
    nearest stored atoms added when needed) from the reference onto each conformer. Bond lengths and angles at
    the missing atom are the reference's; its torsion follows the neighbors' frame."""
    partial = np.asarray(partial, dtype=float)
    n = len(reference)
    full = np.repeat(np.asarray(reference, dtype=float)[None], len(partial), axis=0)
    full[:, indices] = partial
    present = np.zeros(n, bool)
    present[indices] = True
    nbrs: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j, _o in bonds:
        if 0 <= i < n and 0 <= j < n:
            nbrs[i].add(j)
            nbrs[j].add(i)
    ref = np.asarray(reference, dtype=float)
    for h in np.flatnonzero(~present):
        d_ref = np.linalg.norm(ref - ref[h], axis=1)
        d_ref[~present] = np.inf
        bonded = [a for a in nbrs[h] if present[a]]
        a = bonded[0] if bonded else int(np.argmin(d_ref))
        anchors = [a] + [b for b in sorted(nbrs[a]) if present[b] and b != a]
        if len(anchors) < 3:
            d_a = np.linalg.norm(ref - ref[a], axis=1)
            d_a[~present] = np.inf
            extra = [int(b) for b in np.argsort(d_a) if int(b) not in anchors and np.isfinite(d_a[b])]
            anchors += extra[:3 - len(anchors)]
        if len(anchors) < 3:
            continue                                    # fewer than three stored atoms: keep the reference
        r, cs, cd = _rigid_fit(ref[anchors], full[:, anchors])
        full[:, h] = ((ref[h] - cs)[None, None, :] @ r)[:, 0] + cd
    return full


def _conformers(mol: SpartanMolecule) -> None:
    props, n = mol.properties, mol.n_atoms
    cc = props.get("CONF_COORDS")
    if cc is None:
        return
    flat = np.asarray(cc.value, dtype=float).ravel()
    ai = props.get("CONF_ATOM_INDICES")
    idx = np.arange(n) if ai is None else np.atleast_1d(np.asarray(ai.value, dtype=float)).astype(int).ravel() - 1
    m = len(idx)
    if flat.size == 0 or flat.size % (3 * m) or m > n or (idx.size and (idx.min() < 0 or idx.max() >= n)):
        mol.warnings.append(f"CONF_COORDS has {flat.size} values for {m} atoms of {n}: ignored")
        return
    partial = flat.reshape(-1, m, 3)
    if m == n and np.array_equal(idx, np.arange(n)):
        mol.conformer_positions_A = partial
    else:
        missing = n - m
        mol.conformer_positions_A = complete_conformers(partial, idx, mol.input.positions_A, mol.input.bonds)
        mol.warnings.append(f"the saved conformers store {m} of {n} atoms; the other {missing} were placed "
                            "from the input geometry (rigid fit to their neighbors)")
    k = len(mol.conformer_positions_A)
    saved = props.get("CONF_NUM_SAVED")
    if saved is not None and saved.is_scalar and not math.isnan(saved.value) and int(saved.value) != k:
        mol.warnings.append(f"CONF_NUM_SAVED = {int(saved.value)} but CONF_COORDS holds {k} conformers")
    rel = _per_conformer(props.get("CONF_ENERGY_LIST"), k)
    if rel is None:
        mol.warnings.append("no CONF_ENERGY_LIST for the saved conformers: the first one is reported")
        mol.best_conformer = 0
        return
    off = props.get("CONF_ENERGY_OFFSET")
    offset = float(off.value) if off is not None and off.is_scalar and not math.isnan(off.value) else 0.0
    mol.conformer_energies_kcal = rel + offset
    mol.best_conformer = int(np.argmin(rel))


def read_molecule(container, storage: str, source_file: str = "", index: int = 0) -> SpartanMolecule:
    """One molecule storage (``M0001``) of a document. Raises ValueError when it has no readable Input.gz."""
    raw = container.read(f"Molecules/{storage}/Input.gz")
    if raw is None:
        raise ValueError(f"{storage}: no Input.gz")
    inp = parse_input(decode_stream(raw))
    mol = SpartanMolecule(source_file=source_file, molecule_index=index, storage=storage, input=inp)
    pa = container.read(f"Molecules/{storage}/PropertyArchive.gz")
    if pa is None:
        mol.warnings.append("no PropertyArchive.gz (no calculation results)")
    else:
        mol.properties = parse_properties(decode_stream(pa))
    _conformers(mol)
    return mol


def iter_molecules(source, source_file: str | None = None,
                   errors: list[str] | None = None) -> Iterator[SpartanMolecule]:
    """Molecules of a document one at a time (``source``: a path or a container). A molecule that cannot be
    read is skipped with a message appended to ``errors`` (raised when ``errors`` is None)."""
    own = not hasattr(source, "molecules")
    container = open_container(source) if own else source
    name = source_file if source_file is not None else (str(source) if own else "")
    try:
        for idx, storage in enumerate(container.molecules(), 1):
            try:
                yield read_molecule(container, storage, name, idx)
            except (ValueError, OSError, EOFError) as exc:
                if errors is None:
                    raise
                errors.append(f"{Path(name).name or name}:{storage}: {exc}")
    finally:
        if own:
            container.close()


# ----------------------------------------------------------------------------------------- RDKit / rows
_BOND_TYPES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE"}


def to_mol(mol: SpartanMolecule, conformer: int | None = None):
    """RDKit molecule in Spartan's atom order with the geometry of ``conformer`` (default: the lowest-energy
    saved conformer, else Input.gz). Bonds come from the HESSIAN block of Input.gz when they give a sane
    molecule with the document's total charge; otherwise rdDetermineBonds (then connectivity only), as
    ``io_cclib.to_mol``. The property ``qb_bonds`` says which."""
    from rdkit import Chem, rdBase
    from rdkit.Geometry import Point3D

    pos = mol.positions_A(conformer)
    charge = mol.input.charge
    if mol.input.bonds and all(o in _BOND_TYPES for _, _, o in mol.input.bonds):
        _quiet = rdBase.BlockLogs()          # a failed attempt is expected for ions; it falls through below
        rw = Chem.RWMol()
        for z in mol.numbers:
            a = Chem.Atom(int(z))
            a.SetNoImplicit(True)
            rw.AddAtom(a)
        try:
            for i, j, o in mol.input.bonds:
                rw.AddBond(i, j, getattr(Chem.BondType, _BOND_TYPES[o]))
            m = rw.GetMol()
            _set_conformer(m, pos, Point3D)
            Chem.SanitizeMol(m)
            radicals = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
            # no formal charges are set from Spartan's bonds, so they only describe neutral molecules
            if charge == 0 and radicals == mol.input.multiplicity - 1:
                m.SetProp("qb_bonds", "spartan")
                return m
        except Exception:   # noqa: S110 (Spartan's bonds do not sanitize: perceive them below instead)
            pass
        finally:
            del _quiet
    from rdkit.Chem import rdDetermineBonds

    xyz = f"{mol.n_atoms}\n\n" + "\n".join(
        f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {x:.8f} {y:.8f} {w:.8f}"
        for z, (x, y, w) in zip(mol.numbers, pos))
    raw = Chem.MolFromXYZBlock(xyz)
    if raw is None:
        raise ValueError(f"{mol.label}: RDKit could not read the geometry")
    m = Chem.Mol(raw)
    try:
        rdDetermineBonds.DetermineBonds(m, charge=charge)
        m.SetProp("qb_bonds", "rdDetermineBonds")
    except Exception:
        m = Chem.Mol(raw)
        rdDetermineBonds.DetermineConnectivity(m)
        m.UpdatePropertyCache(strict=False)
        m.SetProp("qb_bonds", "connectivity only")
    return m


def _set_conformer(m, pos, Point3D) -> None:
    from rdkit import Chem

    conf = Chem.Conformer(m.GetNumAtoms())
    for k, (x, y, z) in enumerate(pos):
        conf.SetAtomPosition(k, Point3D(float(x), float(y), float(z)))
    conf.Set3D(True)
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)


def hill_formula(numbers: Iterable[int], charge: int = 0) -> str:
    """Hill formula, with RDKit's charge suffix (``C8HF15O2``, ``C8F15O2-``)."""
    from collections import Counter

    from ase.data import chemical_symbols

    c = Counter(chemical_symbols[int(z)] for z in numbers)
    order = (["C", "H"] if "C" in c else []) + sorted(k for k in c if k not in (("C", "H") if "C" in c else ()))
    s = "".join(f"{el}{c[el] if c[el] > 1 else ''}" for el in order)
    if charge:
        s += ("+" if charge > 0 else "-") + (str(abs(charge)) if abs(charge) > 1 else "")
    return s


def molecule_row(mol: SpartanMolecule, conformer: int | None = None, rdkit_mol=None) -> dict:
    """One table row: key columns, QuantaBench columns, ``spartan_*`` scalars and the conformer's values."""
    row: dict = {"source_file": mol.source_file, "molecule_index": mol.molecule_index, "storage": mol.storage,
                 "label": mol.label, "duplicate_label": 0}
    if conformer is not None:
        row["conformer"] = conformer + 1
    smiles, formula = "", hill_formula(mol.numbers, mol.input.charge)
    if rdkit_mol is not None:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        with contextlib.suppress(Exception):
            smiles = Chem.MolToSmiles(Chem.RemoveHs(rdkit_mol, sanitize=False))
        with contextlib.suppress(Exception):
            formula = rdMolDescriptors.CalcMolFormula(rdkit_mol)
    row.update(formula=formula, smiles=smiles, n_atoms=mol.n_atoms, charge=mol.input.charge,
               multiplicity=mol.input.multiplicity, n_conformers=mol.n_conformers,
               spartan_keywords=mol.input.keywords)
    for name, p in mol.properties.items():
        if not p.is_scalar:
            continue
        if name in COLUMN_MAP and p.kind == "VALUE":
            col, factor = COLUMN_MAP[name]
            row[col] = float(p.value) * factor
        else:
            row[f"spartan_{name.lower()}"] = p.value
    if "homo_ev" in row and "lumo_ev" in row:
        row["gap_ev"] = row["lumo_ev"] - row["homo_ev"]
    row.update(mol.conformer_values(conformer))
    if mol.warnings:
        row["spartan_notes"] = " | ".join(mol.warnings)
    return row


def conformer_rows(mol: SpartanMolecule) -> list[dict]:
    return [molecule_row(mol, k) for k in range(mol.n_conformers)]


def to_result(mol: SpartanMolecule):
    """A built ``MoleculeResult`` for the GUI: the saved conformers ranked by energy (eV, rel. kcal/mol and
    Boltzmann weights from Spartan's energies), or the Input.gz geometry as one conformer."""
    from rdkit import Chem

    from .inputs import MoleculeInput, safe_id
    from .pipeline import ConformerResult, MoleculeResult

    order = (list(np.argsort(mol.conformer_energies_kcal, kind="stable"))
             if mol.conformer_energies_kcal is not None else list(range(mol.n_conformers)))
    base = to_mol(mol, int(order[0]) if mol.conformer_positions_A is not None else None)
    m = Chem.Mol(base)
    m.RemoveAllConformers()
    weights = mol.boltzmann_weights()
    confs = []
    for rank, k in enumerate(order):
        k = int(k)
        c = Chem.Conformer(base.GetConformer())
        for a, (x, y, z) in enumerate(mol.positions_A(k if mol.conformer_positions_A is not None else None)):
            c.SetAtomPosition(a, (float(x), float(y), float(z)))
        cid = m.AddConformer(c, assignId=True)
        e = mol.conformer_energies_kcal
        confs.append(ConformerResult(
            rank=rank, conf_id=cid, energy_ev=float(e[k]) / _KCAL_PER_EV if e is not None else 0.0,
            rel_kcal=float(e[k] - e.min()) if e is not None else 0.0,
            weight=float(weights[k]) if weights is not None else 1.0 / len(order)))
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(m, sanitize=False))
    except Exception:
        smiles = ""
    label = mol.label or f"{Path(mol.source_file).stem}_{mol.storage}"
    inp = MoleculeInput(id=safe_id(label, "spartan"), smiles=smiles, name=label,
                        source=f"imported (Spartan): {mol.source_file} {mol.storage}", mol=Chem.Mol(m),
                        charge=mol.input.charge, multiplicity=mol.input.multiplicity,
                        meta={"program": "Spartan", "keywords": mol.input.keywords})
    row = molecule_row(mol, rdkit_mol=base)
    desc = {k: v for k, v in row.items() if k not in KEY_COLUMNS and isinstance(v, (int, float)) and k != "charge"}
    desc["formula"] = row["formula"]
    method = row.get("spartan_conf_method") or mol.input.keywords.split(" ")[0] or "imported"
    notes = [f"imported from {Path(mol.source_file).name} {mol.storage} (Spartan)"] + mol.warnings
    return MoleculeResult(input=inp, status="ok", smiles=smiles, charge=mol.input.charge,
                          multiplicity=mol.input.multiplicity, method=f"spartan: {method}", device="",
                          conformers=confs, mol=m, notes=notes, descriptors=desc,
                          conformer_search="imported (Spartan)")


# ----------------------------------------------------------------------------------------- the table
@dataclass
class ImportSummary:
    files: int = 0
    files_failed: int = 0
    molecules: int = 0
    rows: int = 0
    duplicate_labels: int = 0                       # labels that occur in more than one row
    errors: list[str] = field(default_factory=list)
    table: str = ""
    sdf: str = ""


def _write_sdf_record(writer, m, row: dict, per_atom: dict[str, np.ndarray]) -> None:
    m.SetProp("_Name", str(row["label"] or row["storage"]))
    for key in ("label", "source_file", "molecule_index", "storage", "conformer", "n_conformers",
                "spartan_conf_energy_kcal", "spartan_conf_rel_kcal", "spartan_conf_boltzmann_weight"):
        if row.get(key) not in (None, ""):
            v = row[key]
            m.SetProp(key, f"{v:.6f}" if isinstance(v, float) else str(v))
    for name, vals in per_atom.items():          # RDKit's SDMolSupplier reads atom.dprop.* back as atom props
        m.SetProp(f"atom.dprop.spartan_{name}", " ".join(f"{x:.6f}" for x in vals))
    writer.write(m)


def write_table(paths: Iterable[str | Path], table: str | Path, sdf: str | Path | None = None,
                conformers: str = "best", log: Callable[[str], None] | None = None,
                need_mol: bool = True) -> ImportSummary:
    """Read ``paths`` (``.spartan`` documents) into one CSV ``table`` (and one ``sdf`` of the reported
    geometries), streaming: rows go to a temporary JSON-lines file next to ``table`` while the documents are
    read, then to the CSV once the column set and the labels seen more than once are known.
    ``conformers``: "best" (one row per molecule, the lowest-energy saved conformer) or "all"."""
    if conformers not in ("best", "all"):
        raise ValueError(f"conformers must be 'best' or 'all', not {conformers!r}")
    table = Path(table)
    table.parent.mkdir(parents=True, exist_ok=True)
    summary = ImportSummary(table=str(table), sdf=str(sdf) if sdf else "")
    columns: dict[str, None] = dict.fromkeys(KEY_COLUMNS + (("conformer",) if conformers == "all" else ())
                                             + CORE_COLUMNS)
    label_count: dict[str, int] = {}
    writer = None
    if sdf:
        from rdkit import Chem

        Path(sdf).parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(sdf))
    need_mol = need_mol or writer is not None
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", dir=table.parent, prefix=f".{table.stem}_",
                                     suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            for path in paths:
                summary.files += 1
                errors: list[str] = []
                n_before = summary.molecules
                try:
                    for mol in iter_molecules(path, str(path), errors):
                        summary.molecules += 1
                        ks = range(mol.n_conformers) if conformers == "all" and mol.conformer_positions_A is not None \
                            else [None]
                        for k in ks:
                            m = None
                            if need_mol:
                                try:
                                    m = to_mol(mol, k)
                                except Exception as exc:
                                    mol.warnings.append(f"no RDKit molecule ({type(exc).__name__}: {exc})")
                            row = molecule_row(mol, k, m)
                            if writer is not None and m is not None:
                                _write_sdf_record(writer, m, row, mol.per_atom())
                            columns.update(dict.fromkeys(row))
                            if row["label"] and k in (None, 0):      # a molecule counts once, not per conformer
                                label_count[row["label"]] = label_count.get(row["label"], 0) + 1
                            tmp.write(json.dumps(row, default=_json_default) + "\n")
                            summary.rows += 1
                except Exception as exc:          # unreadable document: report it, go on with the others
                    summary.files_failed += 1
                    errors.append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
                summary.errors += errors
                if log:
                    log(f"{Path(path).name}: {summary.molecules - n_before} molecules"
                        + (f", {len(errors)} problems" if errors else ""))
            if writer is not None:
                writer.close()
            dupes = {lab for lab, n in label_count.items() if n > 1}
            summary.duplicate_labels = len(dupes)
            tmp.flush()
            tmp.seek(0)
            with table.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
                w.writeheader()
                for line in tmp:
                    row = json.loads(line)
                    row["duplicate_label"] = int(row["label"] in dupes)
                    w.writerow(row)
        finally:
            tmp.close()
            with contextlib.suppress(OSError):
                tmp_path.unlink()
    return summary


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    return str(v)
