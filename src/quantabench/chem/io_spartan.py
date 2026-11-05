"""Read Spartan '24 documents (``.spartan``): one table row per molecule, coordinates for QuantaBench runs.

Spartan cannot hold very large sets in one document, so big projects end up as many documents of a
hundred or so molecules each, plus documents made from them (extracted conformers, semi-empirical and DFT
jobs on those). This module reads any number of them, molecule by molecule (never a whole document, let alone
all documents, in memory), into one CSV table keyed by the molecule label, resolves the labels Spartan made
up in the derived documents back to the base document's (:class:`LabelIndex`), optionally writes one SDF of
all geometries that a QuantaBench run can start from, and one wide QSAR table with one row per label and each
level of theory's best result as prefixed columns.

CONTRACT:

    available() -> tuple[bool, str]
    is_spartan(path) -> bool                                  # by suffix
    open_container(path) -> OleContainer                      # olefile, imported lazily
    DictContainer({"Molecules/M0001/Input.gz": bytes, ...})   # the same interface without a file (tests)
    parse_input(text) -> SpartanInput                         # Input(.gz) text
    parse_properties(text) -> dict[str, SpartanProperty]      # PropertyArchive(.gz) text
    parse_commands(text) -> dict; read_level(...) -> SpartanLevel; job_status(joblog) -> str
    read_molecule(container, storage, source_file, index) -> SpartanMolecule   # UnreadableMolecule(reason)
    iter_molecules(path_or_container, source_file=None, errors=None, skipped=None) -> Iterator[SpartanMolecule]
    structure_key(mol) -> str; stereo_key(rdkit_mol) -> str   # the join keys
    molecule_row(mol) -> dict                                 # one table row (best conformer)
    conformer_rows(mol) -> list[dict]                         # one row per saved conformer
    to_mol(mol, conformer=None) -> RDKit Mol (Spartan's bonds; rdDetermineBonds as a fallback)
    to_result(mol) -> a built MoleculeResult (saved conformers ranked by energy) for MainWindow.add_result
    level_prefixes({level: (kind, method)}) -> {level: prefix}
    write_table(paths, table, sdf=None, conformers="best", log=None, wide=None) -> ImportSummary

File format
-----------
A ``.spartan`` file is an OLE compound document (the container of old ``.doc`` files) with one storage per
molecule, ``Molecules/M0001``, ``Molecules/M0002``, ... (four digits in the samples; any digits accepted),
plus document-level streams that are not read: ``Document`` (binary settings), ``Sheet/SheetData`` and
``Sheet/Titles`` (the spreadsheet's column definitions only, e.g. one "Label" column; no values, no labels),
``JobLog`` or ``Output`` (the queue's log: one line per molecule, "COMPLETED"/"FAILED reason") and
``Embedded/Blob00`` (a thumbnail PNG). Spartan for Windows writes the text streams of a molecule gzipped
(``Input.gz``, ``PropertyArchive.gz``, ``Output.gz``), Spartan for macOS (build ``1.3.0llvmGf Mx/macOS``) the
same text uncompressed (``Input``, ``PropertyArchive``, ``Output``): :func:`read_stream` takes either. In each
molecule storage:

- ``Input`` / ``Input.gz``: CRLF text. Keyword lines first. A keyword line that begins with ``C`` and a blank
  (``C MS_CONFP2_BW HF ...``) is continued on the next line; Spartan's own ``INITIAL_KEYWORDS`` property
  lists that ``"C"`` as a keyword, and the line after the last ``C`` line is the last keyword line. Then the
  label (the CAS number in the PFAS QSAR base documents; a name Spartan made up, ``M0003`` or ``M0018[1]``,
  in a document of extracted conformers or of jobs run on them), a ``charge multiplicity`` line, ``Z x y z``
  lines in Å and ``ENDCART``. Optional blocks follow: ``ATOMLABELS`` (quoted atom names), ``HESSIAN`` ...
  ``ENDHESS`` (N per-atom codes, then one ``i j order`` line per bond, 1-based, order 1/2/3), ``BEGINPROPIN``,
  ``BEGINPREFERENCES``. When the label line and the ``charge multiplicity`` line are not where the
  continuation rule puts them, the parser falls back to the first two-integer line that is followed by a
  ``Z x y z`` line (label = the line before it).
- ``PropertyArchive`` / ``.gz``: CRLF text of ``PROP`` records. Scalars: ``PROP VALUE NAME = 1.23`` and
  ``PROP STRING NAME = "text"``. Blocks: ``PROP VALUE NAME d1 [d2] BEGIN`` ... ``END`` (floats, row-major,
  shape (d1, d2)) and ``PROP STRING NAME n BEGIN`` (one quoted string per line). The kind may carry
  qualifiers after colons: ``VALUE:BYATOM`` (one row per atom), ``VALUE:ATOMPAIR``, ``:STATIC``, ``:VIBS``,
  ``STRING:MSG`` (a message instead of a value, e.g. ``LOGP_GC = " Illegal atom"``).
- ``Commands`` (text in both builds): a ``# key : value`` header (``# version : SPARTAN'24 1.3.0 ...``,
  ``# Task : OPT 'Equilibrium Geometry'``, ``# Method : PM6 'Semi-Empirical'``, ``# Basis : 6-311+G**``) and
  the program steps. ``Calculations`` (binary) holds the calculation's keyword string (``OPT PM6 FREQ DOQSAR``,
  ``SOPT B3LYP 6-311+G** DOQSAR SOLVENT=WASSER``). ``JobLog`` (binary) has one record per program run,
  ``Normal exit`` (Windows: ``Normal exit.``) or ``Abnormal exit``, a message ("Geometry failed to converge")
  and the program's path; a molecule that was not calculated in this document has none.
- ``Geometry``, ``Molecule``, ``Archive.carc``: binary, not read (``Geometry`` holds the label, atom names and
  coordinates as doubles; its first atom equals ``Input``'s in the PM6 document checked). ``Output`` /
  ``Output2``: the programs' text output, used here only to work out units.

Geometry. In the documents checked, the Cartesian block of ``Input`` is the molecule's *input* geometry (the
``Geometry`` stream holds the same coordinates, and so does ``ORIGINAL_COORDS``). After a PM6 ``OPT`` job it is
still the starting geometry (identical, to 1e-4 Å, to the conformer it was made from), so the rows and SDF
records of an optimization carry the starting geometry with a note in ``spartan_notes``; the optimized
geometry (probably in ``Archive.carc``) is not read. After a conformer search the search's result is the
conformer library in ``PropertyArchive``:

- ``CONF_COORDS`` (n_saved, 3M): the saved conformers, Å, for the M atoms listed in ``CONF_ATOM_INDICES``
  (1-based ``Input`` indices). In most molecules of the batch checked M < N: Spartan leaves out atoms it
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

Level of theory
---------------
Every row carries ``spartan_version``, ``spartan_task`` (``OPT``, ``SOPT``, ``MS_CONFP2_BW``), ``spartan_method``
(``PM6``, ``B3LYP``, ``MMFF94``), ``spartan_basis``, ``spartan_solvent`` (the ``SOLVENT=`` keyword as written:
``WASSER`` is C-PCM water, dielectric 78.3, in the B3LYP documents' output) and ``spartan_status``
(``completed`` | ``failed: <message> (<program>)`` | "" when the molecule was not calculated in this document),
from ``Commands`` and ``Calculations`` (the Input keywords when those are missing). For a conformer search the
method is the final stage of ``CONFMULTI_METHODS`` (DLFF03 after MMFF): Spartan's ``# Method`` line shows the
dialog's setting there (HF), which is not what the search ran.

Units
-----
Checked against the programs' own Output text, the Spartan '24 manual (orbital energies "evaluated in atomic
units (hartrees)", the ``@hart2eV`` factor; dipole in debye; CPK area/volume in Å²/Å³; ESP polar areas at
|potential| > 75/100/125 kJ/mol) and identities between the archive's own values:

- conformer searches: energies in kcal/mol (``CONF_ENERGY_LIST``, ``CONF_ENERGY_OFFSET``, ``CONFMULTI_ENERS``,
  ``PARTITION_ENERGY_INCREMENT``, ``SOLV_*``, ``MM_SOLVATION``, ``DLFF_CORRECTION``), ``CONFORMATIONAL_ENTROPY``
  in cal/(mol K) (Output.gz prints it in J/(mol K), 4.184 times larger), coordinates in Å, dipoles in debye.
  ``ENERGY_COMMENT`` labels the base energy "(au)", but Output.gz prints the same number in kJ/mol; it is
  passed through unchanged as text. The DLFF03 and MMFF energies are on force-field scales.
- ``E_HOMO``, ``E_LUMO``, ``TOT_HARDNESS`` (= (LUMO - HOMO)/2) and ``TOT_ELECTRONEG`` (= -(HOMO + LUMO)/2):
  hartree. ``MO_ENERGY``: hartree.
- ``E_KCAL``: kcal/mol, the method's energy: the heat of formation for PM6 (the Output's "Heat of Formation"
  in kJ/mol is ``E_KCAL`` x 4.184), the total energy for DFT (``SPENERGY`` in hartree x 627.51). ``OPTENERGY``/``SPENERGY``:
  the same in hartree (``OPTENERGY`` of a single point is the earlier optimization's, see ``OPTAT``).
- Thermochemistry (298.15 K ``THERMO_TEMP``, 1 atm ``THERMO_PRESS``): ``ZEROPOINT``, ``H_TOT``/``H_TRAN``/
  ``H_ROT``/``H_VIB``, ``G_THERMO`` in kcal/mol; ``S_*`` in kcal/(mol K); ``C_*`` (Cv) in cal/(mol K). Checked:
  ``H_TOT`` = ``ZEROPOINT`` + ``H_TRAN`` + ``H_ROT`` + ``H_VIB`` + RT; ``H_TRAN`` = 3/2 RT; ``C_TRAN`` = 3/2 R;
  ``G_THERMO`` = ``H_TOT`` - T ``S_TOT``; ``S_TRAN`` = Sackur-Tetrode for ``TOTAL_MASS`` at 298.15 K and 1 atm.
  ``H_TOT``/``G_THERMO`` are thermal corrections (H - E, G - E, ZPE included), not absolute values.
- ``CPKAREA`` Å², ``CPKVOLUME`` Å³, ``OVALITY`` and ``LogP_GC`` dimensionless, ``TOTAL_MASS``/``WEIGHT`` amu.
- ESP: ``QSAR_ELPOT_*AREA*`` Å² (polar areas at 75/100/125 kJ/mol), ``QSAR_ELPOT_VOLUME`` Å³; ``QSAR_ELPOT_MAX``/
  ``_MIN`` in kcal/mol, not kJ/mol: anions whose polar area at 125 kJ/mol is their whole surface have
  |``QSAR_ELPOT_MAX``| between 125/4.184 and 125, and neutral molecules with |``MAX``| and |``MIN``| below 125
  have a non-zero 125 kJ/mol polar area. ``QSAR_LOCIONPOT_MIN``: eV by the size of its values (no unit in the
  manual).
- ``EST_POLARIZ``: Spartan's *estimate* from an undocumented formula (no unit in the manual; Å³ by its size).

Columns
-------
Every row has ``source_file`` (the path as given), ``molecule_index`` (1-based position in its document),
``storage`` (``M0001``), ``label``, ``duplicate_label`` (1 when the same own label, see below, occurs in more
than one molecule of the whole table; all rows are kept), ``conformer`` (1-based, only with
``conformers="all"``), the label columns (``label_resolved``, ``label_source``, ``label_note``,
``structure_key``), the level columns above, then the QuantaBench columns below, then ``spartan_<name>``
(lower case) for every other scalar ``PROP`` (numbers as floats, strings as text) and the ``spartan_conf_*``
values of the reported conformer. Arrays (gradients, coordinates, per-atom blocks, frequencies) are not
written to the table; per-atom blocks of length N go into the SDF as atom property lists
(``spartan_mm_charge``, ...). The long table keeps the level columns next to the values: a PM6 and a DFT
HOMO are both ``homo_ev``, told apart by ``spartan_method``; the wide table (below) gives each level its own
prefix instead.

Spartan to QuantaBench (a Spartan value takes a QuantaBench column only when it is the same quantity in the
same units, and only from a quantum-chemical calculation, :attr:`SpartanLevel.is_quantum`; everything else
stays ``spartan_*``):

    =====================================  ==================  =============================================
    Spartan                                QuantaBench         note
    =====================================  ==================  =============================================
    Input label line                       label               the key of the table
    Input charge                           charge              e
    Input multiplicity                     multiplicity
    rows of CONF_COORDS (else 1)           n_conformers        saved conformers
    atomic numbers + Spartan's bonds       formula, smiles     RDKit (CalcMolFormula, canonical SMILES)
    atomic numbers                         n_atoms
    E_HOMO, E_LUMO (hartree)               homo_ev, lumo_ev    x 27.211386 (+ gap_ev = lumo_ev - homo_ev)
    DIPOLE_MAG (debye)                     dipole_debye        the method's dipole (PM6, B3LYP)
    ZEROPOINT (kcal/mol)                   zpe_kcal
    H_TOT (kcal/mol)                       h_corr_kcal         H - E, as chem.vibrations
    G_THERMO (kcal/mol)                    g_corr_kcal         G - E
    S_TOT (kcal/(mol K))                   entropy_cal_molK    x 1000; harmonic (Spartan clamps low modes)
    =====================================  ==================  =============================================

Not mapped on purpose: ``E_KCAL``/``OPTENERGY``/``SPENERGY`` (a heat of formation for semi-empirical methods,
a total energy for DFT: one Spartan name, two quantities; QuantaBench's ``qm_energy_ev`` is a total energy),
the DLFF03/MMFF conformer energies (force-field scale), ``DIPOLE_MAG`` of a conformer search (the MMFF
point-charge dipole of the input geometry, not the conformer's), ``CONFORMATIONAL_ENTROPY`` (QuantaBench has
only the vibrational ``vib_entropy_cal_molK``), ``CPKAREA``/``CPKVOLUME`` (van der Waals (CPK) surfaces, not
QuantaBench's ``sasa_A2``, solvent accessible with a 1.4 Å probe, or ``volume_A3``), the ESP areas and extrema
(on Spartan's density surface, not QuantaBench's 0.002 au one), ``EST_POLARIZ`` (an empirical estimate, not a
computed polarizability like ``polarizability_iso_A3``), ``C_TOT`` (no Cv column in QuantaBench).

Labels and the structure join
-----------------------------
Spartan keeps a molecule's label only in the document it was built in. A document of extracted conformers,
or a new job document made from them, labels its molecules ``M0003``, ``M0001[2]``, ... (and the spreadsheet
streams hold no labels either), so every row says where its label comes from:

- ``label_source`` = ``input``: the Input label is the molecule's own (anything but ``M<digits>[<n>]``,
  :data:`SPARTAN_DEFAULT_LABEL`); ``label_resolved`` = ``label``.
- otherwise the row is joined to its *base document*: the document among the inputs whose stem is the longest
  prefix of this one's followed by ``_`` (``C1B_P5_SemiEmp_PM6`` -> ``C1B_P5``; document names are the user's,
  so this is the only use made of them). ``structure_key`` (elements, connectivity from Spartan's bonds, total
  charge; conformation, bond orders and atom order do not matter) must match exactly one label of the base
  document: ``structure``. Several base labels with one structure (the same anion under the CAS numbers of its
  salts, a molecule entered twice) are told apart by the 3D configuration (R/S, E/Z) or the row is
  ``ambiguous``; no base document or no match is ``unresolved``. ``label_note`` says which base document
  matched or why not (with up to five candidate labels). Nothing is guessed.

The wide table
--------------
``write_table(..., wide=path)`` (``import --wide``) writes one row per resolved label (``label_resolved``,
``structure_key``, ``formula``, ``smiles``, ``charge``, ``multiplicity``, ``n_atoms``) and, for each level of
theory in the run, the best row of that label at that level as ``<prefix>_<column>`` plus ``<prefix>_level``
(the level in words), ``<prefix>_n_rows`` (rows of the label at that level, e.g. conformers) and
``<prefix>_n_failed``. Best = completed before failed, then the lowest energy (the conformer energy in a
conformer search, else ``E_KCAL``, else ``OPTENERGY``). The prefix comes from the level, not from the
document name (:func:`level_prefixes`): ``conf`` (conformer search), ``mm`` (molecular mechanics), the method
for semi-empirical ones (``pm6``), ``hf``, ``dft``; when the run has two levels with the same prefix, each
gets its level appended (``conf_dlff03`` and ``conf_mmff``, ``dft_b3lyp_6_311pgss_wasser``), so one column
never holds two levels. Ambiguous and unresolved rows stay out of the wide table (they are in the long one).
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
_HARTREE_EV = 27.211386245988      # chem.engines.HARTREE_EV
#: Spartan ``PROP`` name -> (QuantaBench column, factor to QuantaBench's unit), applied to quantum-chemical
#: calculations only (:attr:`SpartanLevel.is_quantum`; in a conformer search DIPOLE_MAG is the MMFF dipole of the
#: input geometry). Names and units checked on real PM6 and B3LYP documents (see the module docstring).
COLUMN_MAP: dict[str, tuple[str, float]] = {
    "E_HOMO": ("homo_ev", _HARTREE_EV),            # hartree
    "E_LUMO": ("lumo_ev", _HARTREE_EV),            # hartree
    "DIPOLE_MAG": ("dipole_debye", 1.0),           # debye
    "ZEROPOINT": ("zpe_kcal", 1.0),                # kcal/mol
    "H_TOT": ("h_corr_kcal", 1.0),                 # kcal/mol, H - E (ZPE + thermal + RT)
    "G_THERMO": ("g_corr_kcal", 1.0),              # kcal/mol, H_TOT - T S_TOT
    "S_TOT": ("entropy_cal_molK", 1000.0),         # kcal/(mol K)
}
#: Columns every row starts with, in this order.
KEY_COLUMNS = ("source_file", "molecule_index", "storage", "label", "duplicate_label")
LABEL_COLUMNS = ("label_resolved", "label_source", "label_note", "structure_key")
LEVEL_COLUMNS = ("spartan_version", "spartan_task", "spartan_method", "spartan_basis", "spartan_solvent",
                 "spartan_status")
CORE_COLUMNS = ("formula", "smiles", "n_atoms", "charge", "multiplicity", "n_conformers")
#: ``label_source`` values.
LABEL_SOURCES = ("input", "structure", "ambiguous", "unresolved")
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
        # olefile finds a path by scanning each storage's children (thousands under Molecules) on every
        # exists()/openstream(); one index of the directory tree makes a lookup O(1)
        self._index: dict[str, object] = {}
        todo = [("", self._ole.root)]
        while todo:
            prefix, entry = todo.pop()
            for kid in entry.kids:
                p = f"{prefix}{kid.name}"
                self._index[p.lower()] = kid
                if kid.kids:
                    todo.append((p + "/", kid))

    def molecules(self) -> list[str]:
        names = {p.split("/")[1] for p in self._index if p.startswith("molecules/") and p.count("/") == 1}
        return sorted((n.upper() for n in names if _STORAGE.match(n.upper())), key=lambda n: int(n[1:]))

    def read(self, path: str) -> bytes | None:
        import olefile

        entry = self._index.get(path.lower())
        if entry is None or entry.entry_type != olefile.STGTY_STREAM:
            return None
        try:
            fh = self._ole._open(entry.isectStart, entry.size)       # what openstream() does after its lookup
        except AttributeError:                                          # another olefile: the public way
            fh = self._ole.openstream(path)
        with fh:
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
            toks = " ".join(body).split()
            try:
                arr = np.array(toks, dtype=float)          # one C conversion (PM6 FREQ archives: 1e5 numbers)
            except ValueError:
                arr = np.array([_float(t) for t in toks], dtype=float)
            value = arr.reshape(dims) if dims and math.prod(dims) == arr.size else arr
        props[name] = SpartanProperty(name, kind, tuple(qual), dims or (len(body),), value)
    return props


# ----------------------------------------------------------------------------------------- level of theory
_MM_METHODS = {"MMFF", "MMFF94", "MMFF94S", "SYBYL", "DLFF03", "DLFF", "UFF"}
_SEMIEMP_METHODS = {"PM3", "PM6", "PM7", "AM1", "RM1", "MNDO", "MNDOD", "GFN2", "GFN1"}
#: Spartan's method class (``# Method : PM6 'Semi-Empirical'``) -> the kind used for column prefixes.
_KINDS = {"molecular mechanics": "mm", "semi-empirical": "semiempirical", "hartree-fock": "hf",
          "density-functional": "dft"}


@dataclass
class SpartanLevel:
    """Level of theory of one molecule's calculation, from its ``Commands`` and ``Calculations`` streams (the
    Input keywords when those are missing)."""

    version: str = ""               # "SPARTAN'24 1.3.0 Win/64b"
    task: str = ""                  # "OPT", "SOPT", "MS_CONFP2_BW", ...
    task_name: str = ""             # "Equilibrium Geometry", "Single Point Energy", ...
    method: str = ""                # "PM6", "B3LYP", "DLFF03"; a conformer search: its final stage
    method_class: str = ""          # Spartan's: "Semi-Empirical", "Density-Functional", ...
    basis: str = ""                 # "6-311+G**"; "" for MM and semi-empirical methods
    solvent: str = ""               # the SOLVENT= keyword as written ("WASSER"); "" in the gas phase
    keywords: str = ""              # the calculation's keywords ("OPT PM6 FREQ DOQSAR")

    @property
    def is_conformer_search(self) -> bool:
        return (self.task.upper().startswith(("MS_CONF", "CONF"))
                or "conformer distribution" in self.task_name.lower())

    @property
    def kind(self) -> str:
        """``conf`` (conformer search), ``mm``, ``semiempirical``, ``hf``, ``dft`` or ``other``."""
        if self.is_conformer_search:
            return "conf"
        k = _KINDS.get(self.method_class.lower())
        if k:
            return k
        m = self.method.upper()
        return "mm" if m in _MM_METHODS else "semiempirical" if m in _SEMIEMP_METHODS else "other"

    @property
    def is_quantum(self) -> bool:
        """A quantum-chemical calculation (not molecular mechanics, not a conformer search)."""
        return self.kind not in ("conf", "mm") and bool(self.method)

    def describe(self) -> str:
        """``PM6 OPT``, ``B3LYP/6-311+G** SOPT SOLVENT=WASSER``, ``DLFF03 MS_CONFP2_BW``."""
        s = self.method + (f"/{self.basis}" if self.basis else "")
        return " ".join(p for p in (s, self.task, f"SOLVENT={self.solvent}" if self.solvent else "") if p)

    def row(self) -> dict:
        return {"spartan_version": self.version, "spartan_task": self.task, "spartan_method": self.method,
                "spartan_basis": self.basis, "spartan_solvent": self.solvent}


def parse_commands(text: str) -> dict[str, str]:
    """The ``# key : value`` header of a ``Commands`` stream: version, task (+ task_name), method (+
    method_class), basis, and the calculation keywords (``new Keys``). First occurrence of each key wins."""
    out: dict[str, str] = {}
    for line in text.replace("\r\n", "\n").split("\n"):
        m = re.match(r"^#\s*(version|Task|Method|Basis|new Keys)\s*:\s*(.*?)\s*$", line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2)
        if key in out:
            continue
        if key in ("task", "method"):
            q = re.match(r"^(\S+)\s*(?:'([^']*)')?", val)
            if q:
                out[key] = q.group(1)
                out[f"{key}_name"] = q.group(2) or ""
            continue
        toks = val.strip("'").split()
        if key == "version":            # "SPARTAN'24 1.3.0  1.3.0 Win/64b": the build is printed twice
            toks = [t for k, t in enumerate(toks) if k == 0 or t != toks[k - 1]]
        out[key] = " ".join(toks)
    return out


def _calculations_keywords(data: bytes | None) -> str:
    """The keyword string in the binary ``Calculations`` stream (its longest printable run)."""
    if not data:
        return ""
    runs = re.findall(rb"[\x20-\x7e]{3,}", data)
    return " ".join(max(runs, key=len).decode("latin-1").split()) if runs else ""


def _solvent(keywords: str) -> str:
    m = re.search(r"(?:^|\s)SOLVENT=(\S+)", keywords, re.IGNORECASE)
    return "" if not m or m.group(1).upper() == "NONE" else m.group(1)


def read_level(container, storage: str, input_keywords: str = "",
               properties: Mapping[str, SpartanProperty] | None = None) -> SpartanLevel:
    """:class:`SpartanLevel` of one molecule storage. A conformer search names the method of its final stage
    (``CONFMULTI_METHODS``; Spartan's ``# Method`` line shows the dialog's setting, e.g. HF, there)."""
    cmd_raw = read_stream(container, storage, "Commands")
    try:
        cmd = parse_commands(decode_stream(cmd_raw)) if cmd_raw else {}
    except (OSError, EOFError, ValueError):
        cmd = {}
    calc = _calculations_keywords(container.read(f"Molecules/{storage}/Calculations"))
    keywords = calc or cmd.get("new keys", "") or input_keywords
    toks = (input_keywords or keywords).split()
    lev = SpartanLevel(version=cmd.get("version", ""), task=cmd.get("task", toks[0] if toks else ""),
                       task_name=cmd.get("task_name", ""), method=cmd.get("method", toks[1] if len(toks) > 1 else ""),
                       method_class=cmd.get("method_name", ""), basis=cmd.get("basis", ""),
                       solvent=_solvent(keywords) or _solvent(input_keywords), keywords=keywords)
    if not cmd and len(toks) > 2 and re.match(r"^(STO|\d-\d|CC-|AUG|DEF2|6-|3-)", toks[2], re.IGNORECASE):
        lev.basis = toks[2]
    stages = (properties or {}).get("CONFMULTI_METHODS")
    if lev.is_conformer_search and stages is not None and isinstance(stages.value, list) and stages.value:
        lev.method, lev.method_class, lev.basis = stages.value[-1], "", ""
    return lev


def job_status(joblog: bytes | None) -> str:
    """``completed`` | ``failed: <message> (<program>)`` from a molecule's binary ``JobLog`` (one record per
    program run: "Normal exit" or "Abnormal exit", a message, the program's path); "" without a job log (the
    molecule was not calculated in this document)."""
    if not joblog:
        return ""
    recs = re.findall(rb"([A-Za-z]+ exit)\.?\x00([^\x00]*)\x00([^\x00]*)", joblog)  # Windows: "Normal exit."
    if not recs:
        return ""
    for kind, msg, prog in recs:
        if kind != b"Normal exit":
            name = re.split(r"[/\\]", prog.decode("latin-1"))[-1]
            return f"failed: {msg.decode('latin-1').strip() or kind.decode()}" + (f" ({name})" if name else "")
    return "completed"


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
    level: SpartanLevel = field(default_factory=SpartanLevel)
    status: str = ""                                 # job_status(): "completed", "failed: ...", ""

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


class UnreadableMolecule(ValueError):
    """A molecule storage that cannot be read; ``reason`` is one of :data:`SKIP_REASONS`."""

    def __init__(self, storage: str, reason: str, detail: str = ""):
        super().__init__(f"{storage}: {reason}" + (f" ({detail})" if detail else ""))
        self.storage, self.reason, self.detail = storage, reason, detail


#: Why a molecule storage was not read (the keys of ``ImportSummary.skipped``).
SKIP_REASONS = ("no Input stream", "Input stream not readable", "no geometry in Input", "error reading molecule")


def read_stream(container, storage: str, name: str) -> bytes | None:
    """``name.gz`` (Spartan for Windows) or ``name`` (Spartan for macOS writes the same text uncompressed)."""
    raw = container.read(f"Molecules/{storage}/{name}.gz")
    return raw if raw is not None else container.read(f"Molecules/{storage}/{name}")


def read_molecule(container, storage: str, source_file: str = "", index: int = 0) -> SpartanMolecule:
    """One molecule storage (``M0001``) of a document. Raises :class:`UnreadableMolecule` (a ValueError) when it
    has no readable ``Input``/``Input.gz``."""
    raw = read_stream(container, storage, "Input")
    if raw is None:
        raise UnreadableMolecule(storage, "no Input stream", "neither Input.gz nor Input")
    try:
        text = decode_stream(raw)
    except (OSError, EOFError, ValueError) as exc:          # truncated or corrupt gzip
        raise UnreadableMolecule(storage, "Input stream not readable", f"{type(exc).__name__}: {exc}") from exc
    try:
        inp = parse_input(text)
    except ValueError as exc:
        raise UnreadableMolecule(storage, "no geometry in Input", str(exc)) from exc
    mol = SpartanMolecule(source_file=source_file, molecule_index=index, storage=storage, input=inp)
    pa = read_stream(container, storage, "PropertyArchive")
    if pa is None:
        mol.warnings.append("no PropertyArchive (no calculation results)")
    else:
        try:
            mol.properties = parse_properties(decode_stream(pa))
        except (OSError, EOFError, ValueError) as exc:
            mol.warnings.append(f"PropertyArchive not readable ({type(exc).__name__}: {exc})")
    mol.level = read_level(container, storage, inp.keywords, mol.properties)
    mol.status = job_status(container.read(f"Molecules/{storage}/JobLog"))
    if mol.status and mol.level.task.upper().startswith("OPT"):
        mol.warnings.append("geometry: the starting geometry of this optimization (Input); the optimized one "
                            "is not read")
    _conformers(mol)
    return mol


def iter_molecules(source, source_file: str | None = None, errors: list[str] | None = None,
                   skipped: dict[str, int] | None = None) -> Iterator[SpartanMolecule]:
    """Molecules of a document one at a time (``source``: a path or a container). A molecule that cannot be
    read is skipped with a message appended to ``errors`` (raised when ``errors`` is None) and counted in
    ``skipped`` under its reason (:data:`SKIP_REASONS`)."""
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
                reason = exc.reason if isinstance(exc, UnreadableMolecule) else "error reading molecule"
                if skipped is not None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                errors.append(f"{Path(name).name or name}:{storage}: {exc}")
    finally:
        if own:
            container.close()


# ----------------------------------------------------------------------------------------- structure keys
#: A label Spartan made up (``M0003``, ``M0018[1]``): the molecule's own label was lost (extracted conformers,
#: a new job document), so it is resolved from the base document by structure.
SPARTAN_DEFAULT_LABEL = re.compile(r"^M\d{3,}(\[\d+\])?$")


def has_own_label(label: str) -> bool:
    return bool(label.strip()) and not SPARTAN_DEFAULT_LABEL.match(label.strip())


def _digest(s: str) -> str:
    import hashlib

    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def structure_key(mol: SpartanMolecule) -> str:
    """A 16-hex-digit key of the molecule's constitution: elements, connectivity (Spartan's bonds from the
    Input HESSIAN block, else bonds perceived from the geometry) and total charge. Bond orders, H positions and
    the conformation do not enter, so every conformer and every re-calculation of a molecule has the same key,
    whatever Kekulé structure or charge placement a program drew. Stereoisomers share it (see
    :func:`stereo_key`). "" when RDKit cannot canonicalize it."""
    from rdkit import Chem, rdBase

    n = mol.n_atoms
    bonds = [(i, j) for i, j, _o in mol.input.bonds if 0 <= i < n and 0 <= j < n and i != j]
    _quiet = rdBase.BlockLogs()
    try:
        if not bonds:
            from rdkit.Chem import rdDetermineBonds

            xyz = f"{n}\n\n" + "\n".join(
                f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}"
                for z, (x, y, w) in zip(mol.numbers, mol.input.positions_A))
            raw = Chem.MolFromXYZBlock(xyz)
            rdDetermineBonds.DetermineConnectivity(raw)
            bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in raw.GetBonds()]
        rw = Chem.RWMol()
        for z in mol.numbers:
            a = Chem.Atom(int(z))
            a.SetNoImplicit(True)
            rw.AddAtom(a)
        for i, j in bonds:
            if rw.GetBondBetweenAtoms(i, j) is None:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
        m = rw.GetMol()
        m.UpdatePropertyCache(strict=False)
        smi = Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return ""
    finally:
        del _quiet
    return _digest(f"{smi}|{mol.input.charge}")


def _cached_structure_key(mol: SpartanMolecule) -> str:
    key = getattr(mol, "_structure_key", None)
    if key is None:
        key = structure_key(mol)
        mol._structure_key = key
    return key


def stereo_key(rdkit_mol) -> str:
    """Key of the configuration (R/S, E/Z) as the 3D geometry has it: isomeric canonical SMILES. Tells apart
    stereoisomers that share a :func:`structure_key`. "" without a molecule."""
    if rdkit_mol is None:
        return ""
    from rdkit import Chem, rdBase

    _quiet = rdBase.BlockLogs()
    try:
        m = Chem.Mol(rdkit_mol)
        Chem.AssignStereochemistryFrom3D(m)
        return _digest(Chem.MolToSmiles(Chem.RemoveHs(m, sanitize=False)))
    except Exception:
        return ""
    finally:
        del _quiet


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
    own = has_own_label(mol.label)
    row.update(label_resolved=mol.label if own else "", label_source="input" if own else "unresolved",
               label_note="", structure_key=_cached_structure_key(mol))
    row.update(mol.level.row(), spartan_status=mol.status)
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
    mapped = mol.level.is_quantum
    for name, p in mol.properties.items():
        if not p.is_scalar:
            continue
        if mapped and name in COLUMN_MAP and p.kind == "VALUE" and not math.isnan(p.value):
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
    duplicate_labels: int = 0                       # own labels that occur in more than one molecule
    skipped: dict[str, int] = field(default_factory=dict)          # molecules not read, by reason
    label_sources: dict[str, int] = field(default_factory=dict)    # rows by label_source
    prefixes: dict[str, str] = field(default_factory=dict)         # wide-table prefix -> level
    wide_rows: int = 0
    errors: list[str] = field(default_factory=list)
    table: str = ""
    sdf: str = ""
    wide: str = ""

    @property
    def molecules_skipped(self) -> int:
        return sum(self.skipped.values())


class LabelIndex:
    """The molecules with their own label (``label_source`` = input), by document stem and structure key, for
    resolving Spartan's made-up labels in the documents made from them (:meth:`resolve`)."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, dict[str, set[str]]]] = {}    # stem -> structure key -> label -> stereo

    def add(self, source_file: str, key: str, label: str, stereo: str = "") -> None:
        if key:
            self.docs.setdefault(Path(source_file).stem, {}).setdefault(key, {}).setdefault(label, set()).add(stereo)

    def base_for(self, source_file: str) -> str | None:
        """The base document of ``source_file``: the longest stem B of an indexed document with the stem of
        ``source_file`` = ``B_...`` (``C1B_P5_SemiEmp_PM6`` -> ``C1B_P5``)."""
        stem = Path(source_file).stem
        cands = [s for s in self.docs if stem.startswith(s + "_")]
        return max(cands, key=len) if cands else None

    def resolve(self, source_file: str, key: str, stereo: str = "") -> tuple[str, str, str]:
        """(label_resolved, label_source, label_note) for a molecule without its own label. The structure must
        match exactly one label of the base document; several labels with that structure are told apart by
        the configuration (:func:`stereo_key`) or the molecule is ``ambiguous``. Nothing is guessed."""
        base = self.base_for(source_file)
        if base is None:
            return "", "unresolved", "no base document among the inputs"
        if not key:
            return "", "unresolved", "no structure key"
        labels = self.docs[base].get(key)
        if not labels:
            return "", "unresolved", f"structure not in {base}"
        if len(labels) == 1:
            return next(iter(labels)), "structure", base
        same = [lab for lab, st in labels.items() if stereo and stereo in st]
        if len(same) == 1:
            return same[0], "structure", f"{base} (by configuration)"
        return "", "ambiguous", (f"{len(labels)} molecules of {base} have this structure: "
                                 + ", ".join(sorted(labels)[:5]) + (" ..." if len(labels) > 5 else ""))


_KIND_ORDER = ("conf", "mm", "semiempirical", "hf", "dft", "other")


def _tag(s: str) -> str:
    s = s.lower().replace("+", "p").replace("*", "s")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def level_prefixes(levels: Mapping[str, tuple[str, str]]) -> dict[str, str]:
    """Wide-table prefix of each level (``levels``: :meth:`SpartanLevel.describe` -> (kind, method)): ``conf``
    (conformer search), ``mm``, the method for semi-empirical ones (``pm6``), ``hf``, ``dft``. When several
    levels of the run share one of these, each gets the level appended (``dft_b3lyp_6_311pgss_sopt``), so
    one column never mixes levels of theory."""
    groups: dict[str, list[str]] = {}
    for desc, (kind, method) in levels.items():
        b = (_tag(method) or kind) if kind in ("semiempirical", "other") else kind
        groups.setdefault(b, []).append(desc)
    out: dict[str, str] = {}
    for b, descs in groups.items():
        if len(descs) == 1:
            out[descs[0]] = b
            continue
        # the shorter tag (method/basis and solvent, no task) when it tells the levels apart, else all of it
        short = {d: _tag(" ".join(t for k, t in enumerate(d.split()) if k != 1)) for d in descs}
        tags = short if len(set(short.values())) == len(descs) else {d: _tag(d) for d in descs}
        for d in descs:
            t = tags[d]
            t = t[len(b) + 1:] if t.startswith(b + "_") else t
            out[d] = f"{b}_{t or 'unknown'}"
    return out


def _prefix_order(item: tuple[str, str]) -> tuple:
    prefix, kind = item
    return (_KIND_ORDER.index(kind) if kind in _KIND_ORDER else len(_KIND_ORDER), prefix)


def _rank(row: dict, kind: str) -> tuple:
    """Sort key of the rows of one label and level: completed before failed, then the lowest energy (the
    conformer energy in a conformer search, else Spartan's E_KCAL, else OPTENERGY: all rows of one label and
    level have the same ones), then the first."""
    order = ("spartan_conf_energy_kcal", "spartan_e_kcal") if kind == "conf" else \
        ("spartan_e_kcal", "spartan_optenergy", "spartan_conf_energy_kcal")
    e = math.inf
    for k in order:
        v = row.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            e = float(v)
            break
    return (str(row.get("spartan_status", "")).startswith("failed"), e)


#: Columns of the wide table before the per-level ones.
WIDE_COLUMNS = ("label_resolved", "structure_key", "formula", "smiles", "charge", "multiplicity", "n_atoms")
_WIDE_SKIP = set(WIDE_COLUMNS) | {"duplicate_label"}


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
                need_mol: bool = True, wide: str | Path | None = None) -> ImportSummary:
    """Read ``paths`` (``.spartan`` documents) into one CSV ``table`` (and one ``sdf`` of the reported
    geometries, and one ``wide`` CSV of one row per resolved label), streaming: rows go to a temporary
    JSON-lines file next to ``table`` while the documents are read (pass 1), then to the CSV once the column
    set, the labels seen more than once and the base documents' labels are known (pass 2); the wide table
    reads back only each label's best row per level (pass 3). ``conformers``: "best" (one row per molecule,
    the lowest-energy saved conformer) or "all"."""
    if conformers not in ("best", "all"):
        raise ValueError(f"conformers must be 'best' or 'all', not {conformers!r}")
    table = Path(table)
    table.parent.mkdir(parents=True, exist_ok=True)
    summary = ImportSummary(table=str(table), sdf=str(sdf) if sdf else "", wide=str(wide) if wide else "")
    columns: dict[str, None] = dict.fromkeys(KEY_COLUMNS + (("conformer",) if conformers == "all" else ())
                                             + LABEL_COLUMNS + LEVEL_COLUMNS + CORE_COLUMNS)
    label_count: dict[str, int] = {}
    index = LabelIndex()
    levels: dict[str, tuple[str, str]] = {}            # describe() -> (kind, method)
    writer = None
    if sdf:
        from rdkit import Chem

        Path(sdf).parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(sdf))
    need_mol = need_mol or writer is not None
    with tempfile.NamedTemporaryFile("wb+", dir=table.parent, prefix=f".{table.stem}_", suffix=".jsonl",
                                     delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            for path in paths:                                              # ---- pass 1: read
                summary.files += 1
                errors: list[str] = []
                n_before, skipped_before = summary.molecules, summary.molecules_skipped
                try:
                    for mol in iter_molecules(path, str(path), errors, summary.skipped):
                        summary.molecules += 1
                        desc = mol.level.describe()
                        levels.setdefault(desc, (mol.level.kind, mol.level.method))
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
                            row["_level"] = desc
                            row["_stereo"] = stereo_key(m) if k in (None, 0) else ""
                            if row["label_source"] == "input" and k in (None, 0):   # a molecule counts once
                                label_count[row["label"]] = label_count.get(row["label"], 0) + 1
                                index.add(str(path), row["structure_key"], row["label"], row["_stereo"])
                            tmp.write((json.dumps(row, default=_json_default) + "\n").encode("utf-8"))
                            summary.rows += 1
                except Exception as exc:          # unreadable document: report it, go on with the others
                    summary.files_failed += 1
                    errors.append(f"{Path(path).name}: {type(exc).__name__}: {exc}")
                summary.errors += errors
                if log:
                    n_skip = summary.molecules_skipped - skipped_before
                    log(f"{Path(path).name}: {summary.molecules - n_before} molecules"
                        + (f", {n_skip} not read" if n_skip else "")
                        + (f", {len(errors)} problems" if errors else ""))
            if writer is not None:
                writer.close()
            dupes = {lab for lab, n in label_count.items() if n > 1}
            summary.duplicate_labels = len(dupes)
            prefixes = level_prefixes(levels) if wide else {}
            summary.prefixes = {p: d for d, p in prefixes.items()}
            best: dict[tuple[str, str], list] = {}            # (label, prefix) -> [rank, offset, n_rows, n_failed]
            wide_cols: dict[str, dict[str, None]] = {}        # prefix -> its columns, first seen first
            common: dict[str, dict] = {}                      # label -> WIDE_COLUMNS values
            tmp.flush()
            tmp.seek(0)
            offset = 0
            last_id, resolved = None, ("", "unresolved", "")
            with table.open("w", newline="", encoding="utf-8") as fh:          # ---- pass 2: the table
                w = csv.DictWriter(fh, fieldnames=[c for c in columns if not c.startswith("_")],
                                   extrasaction="ignore")
                w.writeheader()
                for line in tmp:
                    row = json.loads(line)
                    row["duplicate_label"] = int(row["label"] in dupes)
                    if row["label_source"] != "input":
                        mol_id = (row["source_file"], row["molecule_index"])
                        if mol_id != last_id:                  # conformer rows of one molecule: resolve once
                            last_id = mol_id
                            resolved = index.resolve(row["source_file"], row["structure_key"], row["_stereo"])
                        row["label_resolved"], row["label_source"], row["label_note"] = resolved
                    src = row["label_source"]
                    summary.label_sources[src] = summary.label_sources.get(src, 0) + 1
                    w.writerow(row)
                    lab = row["label_resolved"]
                    if wide and lab:
                        p = prefixes[row["_level"]]
                        kind = levels[row["_level"]][0]
                        rank = _rank(row, kind)
                        b = best.get((lab, p))
                        if b is None:
                            best[(lab, p)] = [rank, offset, 1, int(rank[0])]
                        else:
                            if rank < b[0]:
                                b[0], b[1] = rank, offset
                            b[2] += 1
                            b[3] += int(rank[0])
                        wide_cols.setdefault(p, {}).update(dict.fromkeys(
                            c for c in row if c not in _WIDE_SKIP and not c.startswith("_")))
                        if lab not in common or (src == "input" and common[lab].get("_src") != "input"):
                            common[lab] = {c: row.get(c, "") for c in WIDE_COLUMNS} | {"_src": src}
                    offset += len(line)
            if wide:                                                           # ---- pass 3: the wide table
                summary.wide_rows = _write_wide(Path(wide), tmp, prefixes, levels, best, wide_cols, common)
        finally:
            tmp.close()
            with contextlib.suppress(OSError):
                tmp_path.unlink()
    return summary


def _write_wide(path: Path, tmp, prefixes: dict[str, str], levels: dict[str, tuple[str, str]],
                best: dict[tuple[str, str], list], wide_cols: dict[str, dict[str, None]],
                common: dict[str, dict]) -> int:
    """One row per resolved label: :data:`WIDE_COLUMNS`, then per level (prefix ``p``) ``p_level``,
    ``p_n_rows`` (rows of this label at that level, e.g. conformers), ``p_n_failed`` and the best row's
    columns as ``p_<column>``."""
    kinds = {p: levels[d][0] for d, p in prefixes.items()}
    level_of = {p: d for d, p in prefixes.items()}
    order = [p for p, _k in sorted(((p, kinds[p]) for p in wide_cols), key=_prefix_order)]
    header = list(WIDE_COLUMNS)
    for p in order:
        header += [f"{p}_level", f"{p}_n_rows", f"{p}_n_failed"] + [f"{p}_{c}" for c in wide_cols[p]]
    by_label: dict[str, list[tuple[str, list]]] = {}
    for (lab, p), b in best.items():
        by_label.setdefault(lab, []).append((p, b))
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for lab, entries in by_label.items():
            out = {c: v for c, v in common[lab].items() if not c.startswith("_")}
            for p, (_r, off, n_rows, n_failed) in entries:
                tmp.seek(off)
                row = json.loads(tmp.readline())
                out.update({f"{p}_level": level_of[p], f"{p}_n_rows": n_rows, f"{p}_n_failed": n_failed})
                out.update({f"{p}_{c}": v for c, v in row.items() if c not in _WIDE_SKIP and not c.startswith("_")})
            w.writerow(out)
            n += 1
    return n


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    return str(v)
