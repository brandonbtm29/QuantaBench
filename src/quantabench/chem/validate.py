"""Validation against Spartan: structure keys, a stratified validation-set sampler, and the comparison of a
QuantaBench run with Spartan's wide QSAR table (``chem.io_spartan``, ``quantabench-cli import --wide``).

CONTRACT:

    species_keys(rdkit_mol) -> {"inchikey", "canonical_smiles"}       # the one structure key function
    keys_from_smiles(smiles, charge=None, formula="") -> {inchikey, canonical_smiles, structure_charge,
                                                          charge_check, key_source}
    skeleton(inchikey) -> "XXXXXXXXXXXXXX-N"                          # stereo-free, protonation-aware
    convert(value, from_unit, to_unit) -> float                       # D, e*A, au; eV, hartree, kcal/mol, ...
    stats(ref, qb, ids) -> Stats                                      # N, MAE, RMSE, MSE, r, rho, fit, outliers
    read_spartan(path) -> SpartanTable                                # the wide table, only the columns used
    sample(table, n=300, seed=SEED, ...) -> Sample                    # `quantabench-cli sample`
    read_quantabench(paths) -> list[QBRecord]                         # run folders, merged or paired tables
    validate(records, table, out_dir, ...) -> Validation              # `quantabench-cli validate`

Structure keys
--------------
Both sides go through :func:`species_keys`: the standard InChIKey (R/S and E/Z as the molecule's stereo tags
have them; coordinates are not used) and RDKit's canonical isomeric SMILES without explicit H. A Spartan wide
table written by this version of ``io_spartan`` carries them (``inchikey``, ``canonical_smiles``, from the
structure as Spartan drew it, with formal charges from its bond orders). An older table, and a QuantaBench
table without ``inchikey``/``canonical_smiles`` columns, get them from their ``smiles`` (:func:`keys_from_smiles`;
an older ``io_spartan`` wrote connectivity-only SMILES for some ions, whose hydrogens no longer match the
formula: those rows get no key, and re-importing the documents gives them one).

Joins, in this order: the InChIKey; the canonical SMILES; the InChIKey without its stereo block
(:func:`skeleton`: first block + protonation flag, so an acid and its anion never match but a QuantaBench
species without R/S matches Spartan's 3D configuration); for an ion, :func:`charge_parent_key` (the charge
drawn on another atom of the same species); then, for Spartan rows without any structure key
only, the id (``label_resolved``/``cas_candidates`` = QuantaBench's ``id`` or ``name``) when the formulas
(without charge) agree. The CAS id cross-checks every structure join: ``id_check`` is ``same`` (the id is
among the matched rows' labels or CAS candidates), ``differs`` (the id is a Spartan label of another
structure: a different protonation state or a wrong entry, both reported), or ``not in Spartan``.

Spartan rows whose ``charge_check`` is ``mismatch`` (Spartan calculated another charge than the structure it
drew: e.g. a carboxylate calculated as +1) match but give no values; so do levels whose job failed
(``--include-failed`` takes them).

Property pairs and units
------------------------
Like with like: a QuantaBench gas-phase value against each gas-phase quantum level in the Spartan table
(PM6), a water value (ALPB) against each water level (B3LYP/6-311+G** and ωB97X-D/6-31G*, C-PCM). All values
are in QuantaBench's units; Spartan's are converted where they are not (``io_spartan.COLUMN_MAP`` already gives
eV for HOMO/LUMO; a table with ``spartan_e_homo`` in hartree is converted here):

    ==============  =====  =================================================  ==========================================
    property        unit   QuantaBench columns (source = column prefix)       Spartan column (per level)
    ==============  =====  =================================================  ==========================================
    dipole          D      <src>_dipole_debye; dipole_debye_best (refinement  <p>_dipole_debye (debye)
                           method: aimnet2, tblite-gfn2, ...)
    homo, lumo      eV     <src>_homo_ev, <src>_lumo_ev                       <p>_homo_ev (or <p>_spartan_e_homo, hartree)
    gap             eV     <src>_gap_ev                                       <p>_gap_ev (or LUMO - HOMO)
    polarizability  Å³     <src>_polarizability_iso_A3, <src>_d4_polar...     <p>_spartan_est_polariz: correlation only
    area            Å²     sasa_A2_best (1.4 Å probe SASA)                    <p>_spartan_cpkarea (vdW/CPK): correlation only
    volume          Å³     volume_A3_best (RDKit vdW grid)                    <p>_spartan_cpkvolume (CPK): correlation only
    q_min, q_max    e      min/max of the best conformer's charges            <p>_spartan_q_minus_mul / _q_plus_mul (most
                           (<id>/<id>_summary.json)                           negative/positive Mulliken charge): correlation
    ==============  =====  =================================================  ==========================================

"Correlation only" pairs (``comparison = correlation`` in summary.csv) are not the same quantity: Spartan's CPK
surface is the van der Waals surface, QuantaBench's ``sasa_A2`` the solvent-accessible one (1.4 Å probe), and
its volume uses other radii; ``EST_POLARIZ`` is Spartan's empirical estimate, not a computed polarizability;
charge partitions differ between methods. Their errors are listed but only r, ρ and the fit are meaningful.

Solvation shifts (a QuantaBench record with gas and water values): QuantaBench's water - gas difference against
Spartan's water-level minus gas-level difference, which mixes levels of theory (B3LYP/6-311+G** C-PCM at the
MMFF geometry minus PM6 at the PM6 geometry): ``comparison = shift (different levels)``.

Statistics (per pair, overall, per cohort, per charge class): N, MAE, RMSE, mean signed error (QuantaBench -
Spartan), Pearson r, Spearman ρ, the least-squares line QuantaBench = slope x Spartan + intercept (and R²),
and the top-k outliers by |error| with their ids.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: The sampler's default seed (the day the PFAS QSAR wide table was made).
SEED = 20261105
#: Elements AIMNet2 covers (``chem.engines``): the default element filter of the sampler, so every sampled
#: molecule can also be refined with AIMNet2.
AIMNET2_ELEMENTS = ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I")
CHARGE_CLASSES = ("neutral", "anion", "cation")
DEFAULT_SIZE_BINS = (10, 20, 35)        # heavy atoms: <=10, 11-20, 21-35, >35
DEFAULT_MAX_HEAVY = 60
REQUIRE = ("any", "pm6", "dft", "both", "none")
PREFER = ("dft", "pm6", "none")

# ------------------------------------------------------------------------------------------------- units
_BOHR_A = 0.529177210903
#: unit -> (dimension, factor to the dimension's QuantaBench unit: D, eV, Å³, Å², e)
UNITS: dict[str, tuple[str, float]] = {
    "D": ("dipole", 1.0), "e*A": ("dipole", 4.80320471), "au": ("dipole", 2.541746473),
    "eV": ("energy", 1.0), "hartree": ("energy", 27.211386245988), "kcal/mol": ("energy", 1 / 23.060548),
    "kJ/mol": ("energy", 1 / 96.4853321),
    "A3": ("volume", 1.0), "bohr3": ("volume", _BOHR_A ** 3),
    "A2": ("area", 1.0), "bohr2": ("area", _BOHR_A ** 2),
    "e": ("charge", 1.0),
}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """``value`` in ``from_unit`` expressed in ``to_unit`` (:data:`UNITS`); ValueError across dimensions."""
    (d1, f1), (d2, f2) = UNITS[from_unit], UNITS[to_unit]
    if d1 != d2:
        raise ValueError(f"cannot convert {from_unit} ({d1}) to {to_unit} ({d2})")
    return float(value) * f1 / f2


@dataclass(frozen=True)
class Prop:
    name: str
    unit: str                 # QuantaBench's unit (UNITS key)
    label: str
    comparison: str           # "comparison" | "correlation"
    note: str = ""


PROPS: dict[str, Prop] = {p.name: p for p in (
    Prop("dipole", "D", "dipole moment", "comparison",
         "an ion's dipole depends on the origin: both sides are about the centre of mass (the Spartan import moves "
         "Spartan's frame-origin DIPOLE_VEC there); it also grows with the charge's distance from the centre of mass, "
         "so for ions it is conformer-sensitive"),
    Prop("homo", "eV", "HOMO energy", "comparison"),
    Prop("lumo", "eV", "LUMO energy", "comparison"),
    Prop("gap", "eV", "HOMO-LUMO gap", "comparison"),
    Prop("polarizability", "A3", "isotropic polarizability", "correlation",
         "Spartan's EST_POLARIZ is an empirical estimate, not a computed polarizability"),
    Prop("area", "A2", "surface area", "correlation",
         "Spartan's CPK (van der Waals) area vs QuantaBench's solvent-accessible area (1.4 Å probe)"),
    Prop("volume", "A3", "molecular volume", "correlation",
         "Spartan's CPK volume vs RDKit's van der Waals grid volume (other radii)"),
    Prop("q_min", "e", "most negative atomic charge", "correlation",
         "charge partitions differ (Spartan: Mulliken; QuantaBench: the refinement engine's charges)"),
    Prop("q_max", "e", "most positive atomic charge", "correlation",
         "charge partitions differ (Spartan: Mulliken; QuantaBench: the refinement engine's charges)"),
)}
#: Spartan wide-table columns per level prefix ``p`` (``<p>_<column>``), first found wins, with their unit.
SPARTAN_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "dipole": (("dipole_debye", "D"),),
    "homo": (("homo_ev", "eV"), ("spartan_e_homo", "hartree")),
    "lumo": (("lumo_ev", "eV"), ("spartan_e_lumo", "hartree")),
    "gap": (("gap_ev", "eV"),),
    "polarizability": (("spartan_est_polariz", "A3"),),
    "area": (("spartan_cpkarea", "A2"),),
    "volume": (("spartan_cpkvolume", "A3"),),
    "q_min": (("spartan_q_minus_mul", "e"),),
    "q_max": (("spartan_q_plus_mul", "e"),),
}
#: QuantaBench columns: (property, regex with a ``src`` group or a fixed column, fixed source name).
_QB_SUFFIX = {"dipole": "_dipole_debye", "homo": "_homo_ev", "lumo": "_lumo_ev", "gap": "_gap_ev",
              "polarizability": "_polarizability_iso_A3"}
_QB_FIXED = (("dipole", "dipole_debye_best", "refine"), ("polarizability", "gfn2_d4_polarizability_A3", "gfn2_d4"),
             ("area", "sasa_A2_best", "rdkit"), ("volume", "volume_A3_best", "rdkit"),
             ("q_min", "charge_min_e", "refine"), ("q_max", "charge_max_e", "refine"))
#: Column prefixes that are never a QuantaBench property source.
_NOT_SOURCES = ("gas", "water", "delta", "vib", "spartan")
SHIFT_PROPS = ("dipole", "homo", "lumo", "gap")


# ------------------------------------------------------------------------------------------------- keys
def species_keys(mol) -> dict:
    """``{"inchikey", "canonical_smiles"}`` of an RDKit molecule: the standard InChIKey (configuration from the
    molecule's stereo tags, not its coordinates) and RDKit's canonical isomeric SMILES without explicit
    hydrogens; "" for what cannot be computed. Spartan's drawn structures (``io_spartan.species``) and
    QuantaBench's species both go through this function, so their keys agree."""
    out = {"inchikey": "", "canonical_smiles": ""}
    if mol is None:
        return out
    from rdkit import Chem, rdBase

    _quiet = rdBase.BlockLogs()
    try:
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        try:
            m = Chem.RemoveHs(m)
        except Exception:
            m = Chem.RemoveHs(m, sanitize=False)
        with contextlib.suppress(Exception):
            out["canonical_smiles"] = Chem.MolToSmiles(m)
        with contextlib.suppress(Exception):
            out["inchikey"] = Chem.MolToInchiKey(m) or ""
    finally:
        del _quiet
    return out


_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_counts(formula: str) -> dict[str, int]:
    """Element counts of a formula (``C8HF15O2-`` -> {C: 8, H: 1, F: 15, O: 2}; the charge suffix is ignored)."""
    out: dict[str, int] = {}
    for el, n in _ELEMENT.findall(re.sub(r"[+-]\d*$", "", formula or "")):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def bare_formula(formula: str) -> str:
    return re.sub(r"[+-]\d*$", "", (formula or "").strip())


#: A charged carbon in a SMILES that an older io_spartan perceived (rdDetermineBonds forcing the document's charge
#: onto a cation calculated as -1, e.g. ``C[S+](C)[C-]1[CH+][CH-]...``): not a structure to join on.
_CHARGED_CARBON = re.compile(r"\[(?:C|c)H?\d?[+-]")


def keys_from_smiles(smiles: str, charge: int | None = None, formula: str = "", strict: bool = False) -> dict:
    """Structure keys from a SMILES (:func:`species_keys`), with ``structure_charge`` (its formal charge),
    ``charge_check`` (``ok``/``mismatch`` against ``charge``, "" without one) and ``key_source``. When
    ``formula`` is given and the SMILES does not have its hydrogens (a connectivity-only perception), or the
    SMILES does not parse, the keys are "" and ``key_source`` says why; so with ``strict`` (Spartan tables
    written before ``inchikey`` existed) when it has a charged carbon."""
    out = {"inchikey": "", "canonical_smiles": "", "structure_charge": "", "charge_check": "", "key_source": ""}
    if not smiles:
        out["key_source"] = "no smiles"
        return out
    if strict and _CHARGED_CARBON.search(smiles):
        out["key_source"] = "smiles has a charged carbon (re-import for keys)"
        return out
    from rdkit import Chem, rdBase

    _quiet = rdBase.BlockLogs()
    try:
        m = Chem.MolFromSmiles(smiles)
    finally:
        del _quiet
    if m is None or any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        out["key_source"] = "smiles not usable"
        return out
    if formula:
        want = formula_counts(formula).get("H", 0)
        got = sum(a.GetTotalNumHs() for a in m.GetAtoms()) + sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 1)
        if want != got:
            out["key_source"] = "smiles lacks the formula's hydrogens (re-import for keys)"
            return out
    q = Chem.GetFormalCharge(m)
    out.update(species_keys(m), structure_charge=q, key_source="smiles",
               charge_check="" if charge is None else ("ok" if q == charge else "mismatch"))
    return out


def charge_parent_key(smiles: str) -> str:
    """A key that ignores where a charge is drawn: the formula and connection layers of the neutralized
    structure's InChI (RDKit's Uncharger; no H layer, so tautomers agree) plus the total formal charge. Standard
    InChI keeps a sulfonimide anion drawn as N- (``/q-1``) apart from the same anion drawn as O- (``/p-1``),
    and even their neutral parents (NH vs S(=N)OH: fixed vs mobile H), so Spartan's drawing and QuantaBench's
    input of one species can have different InChIKeys. "" for an uncharged structure or one that does not
    parse. A last-resort join (``charge-location``): it also equates tautomers."""
    if not smiles or ("+" not in smiles and "-" not in smiles):
        return ""
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    _quiet = rdBase.BlockLogs()
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""
        q = Chem.GetFormalCharge(m)
        if q == 0:
            return ""
        inchi = Chem.MolToInchi(rdMolStandardize.Uncharger().uncharge(m)) or ""
        main = "/".join(part for part in inchi.split("/")[1:] if part[:1] not in ("h", "q", "p", "b", "t", "m",
                                                                                 "s", "i", "f", "r"))
        return f"{main}|{q:+d}" if main else ""
    except Exception:  # noqa: BLE001 - no key is an answer here
        return ""
    finally:
        del _quiet


def skeleton(inchikey: str) -> str:
    """The InChIKey without its stereo/isotope block: first block (connectivity) + protonation flag. Stereo-free
    but charge-aware: PFOA and its anion differ (…-N vs …-M), its enantiomers do not."""
    k = (inchikey or "").strip()
    return f"{k[:14]}-{k[-1]}" if len(k) == 27 else ""


def _int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def charge_class(charge) -> str:
    q = _int(charge)
    return "" if q is None else "neutral" if q == 0 else "anion" if q < 0 else "cation"


_COHORT = re.compile(r"^([A-Za-z]+\d+[A-Za-z]*)_")


def cohort_of(source_file: str) -> str:
    """Cohort of a Spartan document: the stem's prefix before ``_`` (``C1A_P1_SemiEmp_PM6`` -> ``C1A``;
    ``EPA CompTox Essential 76 PFAS_Conf.1005-73-8`` -> ``EPA CompTox Essential 76 PFAS``)."""
    name = re.split(r"[/\\]", source_file or "")[-1]
    stem = name[:-8] if name.lower().endswith(".spartan") else name
    m = _COHORT.match(stem)
    return m.group(1) if m else stem.split("_")[0].strip()


def norm_id(text: str) -> str:
    from .inputs import safe_id

    return safe_id(text or "", "").lower()


# ------------------------------------------------------------------------------------------------- the Spartan table
@dataclass
class Level:
    prefix: str
    text: str                 # "B3LYP/6-311+G** SOPT SOLVENT=WASSER"
    phase: str                # "gas", "water", another canonical solvent, or "" (not a quantum level)
    quantum: bool

    @property
    def method(self) -> str:
        return self.text.split(" ")[0] if self.text else self.prefix


@dataclass
class SpartanRow:
    index: int
    label: str
    label_source: str
    cas_candidates: list[str]
    inchikey: str
    canonical_smiles: str
    skeleton: str
    formula: str
    charge: int | None
    structure_charge: int | None
    charge_check: str
    key_source: str
    cohort: str
    cohorts: list[str]
    n_heavy: int
    elements: set[str]
    values: dict[str, dict[str, float]]     # prefix -> property -> value (QuantaBench units)
    status: dict[str, str]                  # prefix -> spartan_status

    @property
    def labels(self) -> list[str]:
        return ([self.label] if self.label else []) + self.cas_candidates

    @property
    def charge_class(self) -> str:
        return charge_class(self.charge)

    @property
    def fragments(self) -> int:
        return self.canonical_smiles.count(".") + 1 if self.canonical_smiles else 1

    def has(self, phase: str, levels: dict[str, Level], include_failed: bool = False) -> bool:
        """A completed quantum level of ``phase`` with at least a dipole or a HOMO."""
        for p, vals in self.values.items():
            lev = levels.get(p)
            if lev and lev.quantum and lev.phase == phase and ("dipole" in vals or "homo" in vals):
                if include_failed or self.status.get(p, "") == "completed":
                    return True
        return False


@dataclass
class SpartanTable:
    path: str
    rows: list[SpartanRow]
    levels: dict[str, Level]
    columns: list[str]
    keys_computed: int = 0                  # rows whose keys came from their smiles (an older table)

    def index(self) -> dict[str, dict[str, list[int]]]:
        idx: dict[str, dict[str, list[int]]] = {"inchikey": {}, "smiles": {}, "skeleton": {}, "id": {}}
        for r in self.rows:
            for kind, key in (("inchikey", r.inchikey), ("smiles", r.canonical_smiles), ("skeleton", r.skeleton)):
                if key:
                    idx[kind].setdefault(key, []).append(r.index)
            for lab in r.labels:
                idx["id"].setdefault(norm_id(lab), []).append(r.index)
        return idx


def _level_of(text: str) -> tuple[str, bool]:
    from .io_spartan import solvent_name

    m = re.search(r"SOLVENT=(\S+)", text or "", re.IGNORECASE)
    phase = solvent_name(m.group(1)) if m else "gas"
    method = (text or "").split(" ")[0].split("/")[0].upper()
    mm = method in ("MMFF", "MMFF94", "MMFF94S", "SYBYL", "DLFF03", "DLFF", "UFF", "DLXD2XV")
    conf = bool(re.search(r"\b(MS_)?CONF", text or ""))
    return phase or "gas", not (mm or conf)


def read_spartan(path: str | Path, log: Callable[[str], None] | None = None) -> SpartanTable:
    """The wide table, keeping only the columns validation uses. Rows without ``inchikey`` get their keys from
    ``smiles`` (:func:`keys_from_smiles`)."""
    csv.field_size_limit(1 << 30)
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        prefixes = [h[:-6] for h in header if h.endswith("_level")]
        hset = set(header)
        levels: dict[str, Level] = {}
        first_level: dict[str, str] = {}
        rows: list[SpartanRow] = []
        computed = 0
        for k, raw in enumerate(reader):
            for p in prefixes:
                if p not in first_level and raw.get(f"{p}_level"):
                    first_level[p] = raw[f"{p}_level"]
            charge = _int(raw.get("charge"))
            keys = {c: raw.get(c, "") for c in ("inchikey", "canonical_smiles", "structure_charge", "charge_check")}
            key_source = "table" if keys["inchikey"] or keys["canonical_smiles"] else ""
            if not key_source:
                keys = keys_from_smiles(raw.get("smiles", ""), charge, raw.get("formula", ""), strict=True)
                key_source = keys.pop("key_source")
                computed += 1
            values: dict[str, dict[str, float]] = {}
            status: dict[str, str] = {}
            sources = []
            for p in prefixes:
                if not raw.get(f"{p}_level"):
                    continue
                status[p] = raw.get(f"{p}_spartan_status", "")
                sf = raw.get(f"{p}_source_file", "")
                if sf:
                    sources.append((p, sf))
                vals = {}
                for prop, cols in SPARTAN_COLUMNS.items():
                    for col, unit in cols:
                        v = _float(raw.get(f"{p}_{col}")) if f"{p}_{col}" in hset else None
                        if v is not None:
                            vals[prop] = convert(v, unit, PROPS[prop].unit)
                            break
                if "gap" not in vals and "homo" in vals and "lumo" in vals:
                    vals["gap"] = vals["lumo"] - vals["homo"]
                if vals:
                    values[p] = vals
            counts = formula_counts(raw.get("formula", ""))
            cohorts = _cohorts(sources, prefixes)
            rows.append(SpartanRow(
                index=k, label=raw.get("label_resolved", ""),
                label_source=raw.get("label_source", "") or ("input" if raw.get("label_resolved") else ""),
                cas_candidates=[c for c in (raw.get("cas_candidates") or "").split(";") if c],
                inchikey=keys["inchikey"] or "", canonical_smiles=keys["canonical_smiles"] or "",
                skeleton=skeleton(keys["inchikey"] or ""), formula=raw.get("formula", ""), charge=charge,
                structure_charge=_int(keys.get("structure_charge")), charge_check=keys.get("charge_check") or "",
                key_source=key_source, cohort=cohorts[0] if cohorts else "", cohorts=cohorts,
                n_heavy=sum(n for el, n in counts.items() if el != "H"), elements=set(counts),
                values=values, status=status))
    for p in prefixes:
        text = first_level.get(p, "")
        phase, quantum = _level_of(text)
        quantum = quantum and any(f"{p}_{c}" in hset for c in ("homo_ev", "dipole_debye", "spartan_e_homo"))
        levels[p] = Level(p, text, phase if quantum else "", quantum)
    if log:
        log(f"{path.name}: {len(rows)} rows, levels {', '.join(p for p, lev in levels.items() if lev.quantum)}"
            + (f"; keys from smiles for {computed} rows (an older table without inchikey)" if computed else ""))
    return SpartanTable(str(path), rows, levels, header, computed)


def _cohorts(sources: list[tuple[str, str]], prefixes: Sequence[str]) -> list[str]:
    """Cohorts of a wide row, the one that names it first: the source of a quantum level (PM6, DFT) before
    conformer/MM sources, documents outside an ``Archive`` folder before archived ones."""
    def rank(item):
        p, sf = item
        quantum = not p.startswith(("conf", "mm", "dlxd"))       # io_spartan.level_prefixes: conf_*, mm_*
        archived = bool(re.search(r"(^|[/\\])archive([/\\]|$)", sf, re.IGNORECASE))
        return (archived, not quantum, prefixes.index(p))

    out: list[str] = []
    for _p, sf in sorted(sources, key=rank):
        c = cohort_of(sf)
        if c and c not in out:
            out.append(c)
    return out


# ------------------------------------------------------------------------------------------------- the sampler
def size_bin(n_heavy: int, edges: Sequence[int]) -> str:
    lo = 1
    for e in edges:
        if n_heavy <= e:
            return f"{lo}-{e}"
        lo = e + 1
    return f">{edges[-1]}" if edges else "all"


@dataclass
class Sample:
    rows: list[SpartanRow]
    design: dict
    excluded: dict[str, int] = field(default_factory=dict)


def sample(table: SpartanTable, n: int = 300, seed: int = SEED, require: str = "any",
           elements: Iterable[str] | None = AIMNET2_ELEMENTS, charges: Iterable[str] = CHARGE_CLASSES,
           min_heavy: int = 1, max_heavy: int | None = DEFAULT_MAX_HEAVY,
           size_bins: Sequence[int] = DEFAULT_SIZE_BINS, strata: Sequence[str] = ("cohort", "charge_class"),
           prefer: str = "dft", cohorts: dict[str, str] | None = None, allow_charge_mismatch: bool = False,
           allow_fragments: bool = False, include_failed: bool = False) -> Sample:
    """A reproducible, stratified validation set (``quantabench-cli sample``).

    Eligible rows: a structure key; ``require`` Spartan values (``any`` = a completed gas-phase or water quantum
    level, ``pm6``, ``dft`` = water, ``both``, ``none``); elements within ``elements`` (default: AIMNet2's);
    charge class in ``charges``; ``min_heavy`` <= heavy atoms <= ``max_heavy``; Spartan's charge agreeing
    with its drawing (unless ``allow_charge_mismatch``); one fragment (unless ``allow_fragments``). One row per
    structure (InChIKey): the one with the most Spartan values, then the first. ``cohorts`` (label -> cohort)
    overrides the cohort from the document names.

    Quotas: ``n`` split evenly over the strata (``cohort`` x ``charge_class`` by default); a stratum smaller
    than its share gives the rest to the others (water filling, ties to the larger strata). Within a stratum,
    ``prefer`` = ``dft`` takes molecules with DFT-water and PM6 values first, then DFT only, then PM6 only
    (``pm6``: PM6 first; ``none``: no preference), each tier shuffled with ``seed`` and spread over the size
    bins (round robin), so the same table, options and seed always give the same sample."""
    if require not in REQUIRE:
        raise ValueError(f"require must be one of {REQUIRE}, not {require!r}")
    if prefer not in PREFER:
        raise ValueError(f"prefer must be one of {PREFER}, not {prefer!r}")
    allowed = set(elements) if elements else None
    charges = set(charges)
    excluded: dict[str, int] = {}

    def skip(why: str) -> None:
        excluded[why] = excluded.get(why, 0) + 1

    by_key: dict[str, tuple[int, SpartanRow]] = {}
    for r in table.rows:
        key = r.inchikey or r.canonical_smiles
        has_g = r.has("gas", table.levels, include_failed)
        has_w = r.has("water", table.levels, include_failed)
        if not key:
            skip("no structure key")
        elif r.charge_check == "mismatch" and not allow_charge_mismatch:
            skip("Spartan's charge differs from its structure")
        elif r.fragments > 1 and not allow_fragments:
            skip("more than one fragment")
        elif r.charge_class not in charges:
            skip("charge class")
        elif allowed is not None and not r.elements <= allowed:
            skip("elements")
        elif r.n_heavy < min_heavy or (max_heavy is not None and r.n_heavy > max_heavy):
            skip("size")
        elif not {"any": has_g or has_w, "pm6": has_g, "dft": has_w, "both": has_g and has_w, "none": True}[require]:
            skip(f"no Spartan values ({require})")
        else:
            score = sum(len(v) for v in r.values.values())
            old = by_key.get(key)
            if old is None or score > old[0]:
                if old is not None:
                    skip("same structure as another row")
                by_key[key] = (score, r)
            else:
                skip("same structure as another row")
    pool = [r for _s, r in by_key.values()]
    pool.sort(key=lambda r: r.index)

    def stratum(r: SpartanRow) -> tuple:
        vals = {"cohort": (cohorts or {}).get(r.label, r.cohort) or "?", "charge_class": r.charge_class,
                "size": size_bin(r.n_heavy, size_bins)}
        return tuple(vals[s] for s in strata)

    groups: dict[tuple, list[SpartanRow]] = {}
    for r in pool:
        groups.setdefault(stratum(r), []).append(r)
    quotas = _water_fill(n, {k: len(v) for k, v in groups.items()})
    chosen: list[SpartanRow] = []
    for k in sorted(groups):
        members = groups[k]
        rng_k = random.Random(f"{seed}:{'|'.join(k)}")     # noqa: S311 (a reproducible sample, not a secret)
        tiers: dict[int, list[SpartanRow]] = {}
        for r in members:
            g, w = r.has("gas", table.levels, include_failed), r.has("water", table.levels, include_failed)
            tier = {"dft": 0 if (g and w) else 1 if w else 2 if g else 3,
                    "pm6": 0 if (g and w) else 1 if g else 2 if w else 3, "none": 0}[prefer]
            tiers.setdefault(tier, []).append(r)
        order: list[SpartanRow] = []
        for t in sorted(tiers):
            bins: dict[str, list[SpartanRow]] = {}
            for r in tiers[t]:
                bins.setdefault(size_bin(r.n_heavy, size_bins), []).append(r)
            for b in bins.values():
                rng_k.shuffle(b)
            names = sorted(bins)
            rng_k.shuffle(names)
            while any(bins.values()):
                for b in names:
                    if bins[b]:
                        order.append(bins[b].pop())
        chosen += order[:quotas.get(k, 0)]
    design = {
        "n_requested": n, "n_selected": len(chosen), "seed": seed, "require": require, "prefer": prefer,
        "elements": sorted(allowed) if allowed else "any", "charges": sorted(charges), "min_heavy": min_heavy,
        "max_heavy": max_heavy, "size_bins": list(size_bins), "strata": list(strata),
        "eligible": len(pool), "excluded": excluded, "table": table.path,
        "quotas": {"|".join(k): {"available": len(groups[k]), "quota": quotas.get(k, 0)} for k in sorted(groups)},
        "with_pm6": sum(1 for r in chosen if r.has("gas", table.levels, include_failed)),
        "with_dft_water": sum(1 for r in chosen if r.has("water", table.levels, include_failed)),
    }
    design["with_both"] = sum(1 for r in chosen if r.has("gas", table.levels, include_failed)
                              and r.has("water", table.levels, include_failed))
    return Sample(chosen, design, excluded)


def _water_fill(n: int, sizes: dict[tuple, int]) -> dict[tuple, int]:
    """Split ``n`` evenly over the strata, none above its size; what a small stratum cannot take goes to the
    others. Remainders go to the larger strata (then the first by name), so the split is deterministic."""
    quotas = dict.fromkeys(sizes, 0)
    left = min(n, sum(sizes.values()))
    while left > 0:
        open_ = [k for k in sorted(sizes, key=lambda k: (-sizes[k], k)) if quotas[k] < sizes[k]]
        if not open_:
            break
        share = left // len(open_)
        if share == 0:
            for k in open_[:left]:
                quotas[k] += 1
            break
        for k in open_:
            take = min(share, sizes[k] - quotas[k])
            quotas[k] += take
            left -= take
    return quotas


def sample_rows(s: Sample, table: SpartanTable, size_bins: Sequence[int] = DEFAULT_SIZE_BINS,
                cohorts: dict[str, str] | None = None) -> list[dict]:
    """The sample as QuantaBench input rows: ``id``, ``smiles``, ``name`` and the design columns."""
    out = []
    for r in s.rows:
        ident = r.label or (f"struct_{r.inchikey[:14]}" if r.inchikey else f"row{r.index + 1}")
        out.append({
            "id": ident, "smiles": r.canonical_smiles,
            "name": r.label or ("ambiguous: " + ";".join(r.cas_candidates[:5])),
            "cohort": (cohorts or {}).get(r.label, r.cohort), "charge_class": r.charge_class, "charge": r.charge,
            "inchikey": r.inchikey, "n_heavy": r.n_heavy, "size_bin": size_bin(r.n_heavy, size_bins),
            "has_pm6": int(r.has("gas", table.levels)), "has_dft_water": int(r.has("water", table.levels)),
            "label_source": r.label_source, "cas_candidates": ";".join(r.cas_candidates),
        })
    return out


def read_cohorts(path: str | Path) -> dict[str, str]:
    """A CSV mapping labels to cohorts: a ``cohort`` column and a label column (``label``, ``cas``, ``id``)."""
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        lab = next((cols[c] for c in ("label", "label_resolved", "cas", "casrn", "id") if c in cols), None)
        coh = cols.get("cohort")
        if lab is None or coh is None:
            raise ValueError(f"{path}: needs a cohort column and a label/cas/id column")
        return {row[lab]: row[coh] for row in reader if row.get(lab) and row.get(coh)}


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: dict[str, None] = {}
    for r in rows:
        keys.update(dict.fromkeys(r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keys), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})
    return path


def _fmt(v):
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return "" if v is None else v


# ------------------------------------------------------------------------------------------------- QuantaBench side
@dataclass
class QBRecord:
    id: str
    name: str
    inchikey: str
    canonical_smiles: str
    charge: int | None
    formula: str
    key_source: str
    method: str
    phases: dict[str, dict[str, str]]       # "gas"/"water" -> {column: value} (phase prefixes stripped)
    delta: dict[str, str] = field(default_factory=dict)     # paired table: delta_<column> values
    source: str = ""

    @property
    def skeleton(self) -> str:
        return skeleton(self.inchikey)


def _read_table(path: Path) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _run_folder_rows(folder: Path) -> list[dict]:
    summ = folder / "summary.csv"
    if not summ.is_file():
        raise FileNotFoundError(f"{folder}: no summary.csv (a QuantaBench run folder or merged tables)")
    rows = _read_table(summ)
    desc = {r.get("id", ""): r for r in _read_table(folder / "descriptors.csv")} \
        if (folder / "descriptors.csv").is_file() else {}
    out = []
    for r in rows:
        row = dict(r)
        for k, v in desc.get(r.get("id", ""), {}).items():
            if v != "" and (k not in row or row[k] == ""):
                row[k] = v
        js = folder / row.get("id", "") / f"{row.get('id', '')}_summary.json"
        if js.is_file():
            with contextlib.suppress(OSError, ValueError, KeyError, IndexError, TypeError):
                d = json.loads(js.read_text(encoding="utf-8"))
                q = (d.get("conformers") or [{}])[0].get("charges")
                if q:
                    row["charge_min_e"], row["charge_max_e"] = min(q), max(q)
            if not row.get("smiles"):
                row["_sdf"] = str(folder / row["id"] / f"{row['id']}_conformers.sdf")
        out.append(row)
    return out


def _phase_of(row: dict) -> str:
    from . import solvents

    try:
        return solvents.canonical(row.get("solvent", "")) or "gas"
    except ValueError:
        return (row.get("solvent") or "").lower() or "gas"


def _keys_for(row: dict, prefixes: Sequence[str] = ("",)) -> tuple[str, str, str]:
    """(inchikey, canonical_smiles, key_source) of a QuantaBench row: its own columns when present (another
    branch adds them), else from its SMILES, else from its SDF."""
    for p in prefixes:
        ik, cs = row.get(f"{p}inchikey", ""), row.get(f"{p}canonical_smiles", "")
        if ik or cs:
            return ik, cs, "table"
    for p in prefixes:
        for col in ("smiles", "input_smiles"):
            smi = row.get(f"{p}{col}", "")
            if smi:
                k = keys_from_smiles(smi)
                if k["inchikey"] or k["canonical_smiles"]:
                    return k["inchikey"], k["canonical_smiles"], col
    sdf = row.get("_sdf")
    if sdf and Path(sdf).is_file():
        from rdkit import Chem

        with contextlib.suppress(Exception):
            m = next(iter(Chem.SDMolSupplier(sdf, removeHs=False)))
            k = species_keys(m)
            return k["inchikey"], k["canonical_smiles"], "sdf"
    return "", "", ""


def read_quantabench(paths: Iterable[str | Path]) -> list[QBRecord]:
    """QuantaBench results as records, one per molecule id with its gas and/or water values: run folders
    (``summary.csv`` + ``descriptors.csv`` + per-molecule ``<id>_summary.json`` for charges), merged tables
    (a folder or a CSV), and paired gas/water tables (``gas_*``, ``water_*``, ``delta_*`` columns). Rows of the
    same id from a gas and a water run become one record; the phase of a plain row is its ``solvent``."""
    recs: dict[str, QBRecord] = {}
    for path in paths:
        path = Path(path).expanduser()
        rows = _run_folder_rows(path) if path.is_dir() else _read_table(path)
        if not rows:
            continue
        cols = set(rows[0])
        paired = any(c.startswith(("gas_", "water_")) for c in cols)
        for row in rows:
            if (row.get("status") or "ok") != "ok":
                continue
            ident = row.get("id") or row.get("name") or ""
            if not ident:
                continue
            if paired:
                phases = {ph: {c[len(ph) + 1:]: v for c, v in row.items() if c.startswith(ph + "_")}
                          for ph in ("gas", "water")}
                common = {c: v for c, v in row.items() if not c.startswith(("gas_", "water_", "delta_"))}
                for ph in phases:
                    phases[ph] = {**common, **{k: v for k, v in phases[ph].items() if v != ""}}
                phases = {ph: v for ph, v in phases.items() if any(_float(x) is not None for x in v.values())}
                delta = {c[6:]: v for c, v in row.items() if c.startswith("delta_") and v != ""}
                ik, cs, how = _keys_for(row, ("", "gas_", "water_"))
            else:
                phases = {_phase_of(row): row}
                delta = {}
                ik, cs, how = _keys_for(row)
            any_phase = next(iter(phases.values()), row)
            rec = recs.get(ident)
            if rec is None:
                rec = recs[ident] = QBRecord(
                    id=ident, name=row.get("name", ""), inchikey=ik, canonical_smiles=cs,
                    charge=_int(row.get("charge") or any_phase.get("charge")),
                    formula=row.get("formula") or any_phase.get("formula", ""), key_source=how,
                    method=row.get("method") or any_phase.get("method", ""), phases={}, delta={}, source=str(path))
            elif not rec.inchikey and ik:
                rec.inchikey, rec.canonical_smiles, rec.key_source = ik, cs, how
            if str(path) not in rec.source.split(" + "):
                rec.source += f" + {path}"
            for ph, vals in phases.items():
                rec.phases.setdefault(ph, {}).update(vals)
            rec.delta.update(delta)
    return list(recs.values())


def qb_sources(values: dict[str, str]) -> list[tuple[str, str, str]]:
    """(property, source, column) of the QuantaBench property columns in one phase's values."""
    out = []
    for col in values:
        for prop, suf in _QB_SUFFIX.items():
            if col.endswith(suf) and len(col) > len(suf):
                src = col[:-len(suf)]
                if src.split("_")[0] not in _NOT_SOURCES and not src.endswith("_boltz"):
                    out.append((prop, src, col))
    present = set(values)
    for prop, col, src in _QB_FIXED:
        if col in present or (col == "dipole_debye_best" and "dipole_debye" in present):
            out.append((prop, src, col))         # summary.csv's dipole_debye is descriptors' dipole_debye_best
    return out


def _qb_unit(col: str, prop: str) -> str:
    if col.endswith("_debye"):
        return "D"
    if col.endswith(("_ev", "_eV")):
        return "eV"
    if col.endswith("_A3"):
        return "A3"
    if col.endswith("_A2"):
        return "A2"
    return PROPS[prop].unit


# ------------------------------------------------------------------------------------------------- statistics
@dataclass
class Stats:
    n: int
    mae: float
    rmse: float
    mse: float              # mean signed error, QuantaBench - Spartan
    pearson_r: float
    spearman_rho: float
    slope: float            # QuantaBench = slope * Spartan + intercept
    intercept: float
    r2: float
    outliers: list[tuple[str, float, float, float]]     # (id, Spartan, QuantaBench, error), largest |error| first

    def row(self) -> dict:
        return {"n": self.n, "mae": self.mae, "rmse": self.rmse, "mean_signed_error": self.mse,
                "pearson_r": self.pearson_r, "spearman_rho": self.spearman_rho, "slope": self.slope,
                "intercept": self.intercept, "r2": self.r2,
                "top_outliers": "; ".join(f"{i} ({e:+.3g})" for i, _x, _y, e in self.outliers)}


def stats(ref: Sequence[float], qb: Sequence[float], ids: Sequence[str] | None = None, top: int = 5) -> Stats:
    """Error statistics of ``qb`` against ``ref`` (pairs with a non-finite value are dropped)."""
    from scipy.stats import rankdata

    x = np.asarray(ref, float)
    y = np.asarray(qb, float)
    ids = list(ids) if ids is not None else [str(i) for i in range(len(x))]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ids = [i for i, k in zip(ids, ok) if k]
    n = len(x)
    nan = float("nan")
    if n == 0:
        return Stats(0, nan, nan, nan, nan, nan, nan, nan, nan, [])
    err = y - x
    r = rho = slope = icpt = r2 = nan
    if n >= 2 and np.ptp(x) > 0:
        slope, icpt = (float(v) for v in np.polyfit(x, y, 1))
        resid = y - (slope * x + icpt)
        ss = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else nan
    if n >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    order = np.argsort(-np.abs(err), kind="stable")[:top]
    return Stats(n, float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean())), float(err.mean()), r, rho,
                 slope, icpt, r2, [(ids[i], float(x[i]), float(y[i]), float(err[i])) for i in order])


# ------------------------------------------------------------------------------------------------- the comparison
@dataclass
class Match:
    rec: QBRecord
    rows: list[SpartanRow]
    join: str                   # inchikey | smiles | inchikey-nostereo | charge-location | id+formula | ""
    id_check: str               # same | differs | not in Spartan | ""
    note: str = ""
    cohort: str = ""
    charge_class: str = ""


def match(records: list[QBRecord], table: SpartanTable) -> list[Match]:
    """Join each QuantaBench record to the Spartan rows of its structure (module docstring)."""
    idx = table.index()
    rows = table.rows
    parents: dict[str, list[int]] | None = None     # charge_parent_key -> rows, built when first needed
    out = []
    for rec in records:
        found: list[int] = []
        how = ""
        for kind, key in (("inchikey", rec.inchikey), ("smiles", rec.canonical_smiles),
                          ("inchikey-nostereo", rec.skeleton)):
            hits = idx["inchikey" if kind == "inchikey" else "smiles" if kind == "smiles" else "skeleton"].get(key) \
                if key else None
            if hits:
                found, how = list(hits), kind
                break
        pkey = charge_parent_key(rec.canonical_smiles) if not found else ""
        if pkey:
            if parents is None:
                parents = {}
                for r in rows:
                    k = charge_parent_key(r.canonical_smiles)
                    if k:
                        parents.setdefault(k, []).append(r.index)
            if parents.get(pkey):
                found, how = list(parents[pkey]), "charge-location"
        ids = {norm_id(rec.id), norm_id(rec.name)} - {""}
        by_id = sorted({i for k in ids for i in idx["id"].get(k, [])})
        note = ""
        if not found:
            keyless = [i for i in by_id if not (rows[i].inchikey or rows[i].canonical_smiles)]
            same_formula = [i for i in keyless if rec.formula and bare_formula(rows[i].formula) == bare_formula(rec.formula)]
            if same_formula:
                found, how = same_formula, "id+formula"
                note = "joined by id: the Spartan row has no structure key"
        if found:
            labels = {norm_id(lab) for i in found for lab in rows[i].labels}
            id_check = "same" if ids & labels else "differs" if by_id else "not in Spartan"
            if id_check == "differs":
                other = rows[by_id[0]]
                note = (f"id {rec.id} is Spartan's {other.label or other.cas_candidates[0]} with another structure "
                        f"({other.canonical_smiles or other.formula}, charge {other.charge})"
                        + ("; same skeleton, other protonation state"
                           if other.inchikey[:14] and other.inchikey[:14] == rec.inchikey[:14] else ""))
        else:
            id_check = "differs" if by_id else ""
            if by_id:
                other = rows[by_id[0]]
                note = (f"no structure match; id {rec.id} is Spartan's row with {other.canonical_smiles or other.formula} "
                        f"(charge {other.charge})" + ("; same skeleton, other protonation state"
                                                      if other.inchikey[:14] and other.inchikey[:14] == rec.inchikey[:14]
                                                      else ""))
            elif not (rec.inchikey or rec.canonical_smiles):
                note = "no structure key for the QuantaBench molecule"
        found_rows = [rows[i] for i in found]
        pref = {norm_id(rec.id), norm_id(rec.name)}
        found_rows.sort(key=lambda r: (norm_id(r.label) not in pref, r.charge_check == "mismatch",
                                       r.label_source == "ambiguous", r.index))
        mt = Match(rec, found_rows, how, id_check, note)
        mt.cohort = found_rows[0].cohort if found_rows else ""
        mt.charge_class = charge_class(rec.charge) or (found_rows[0].charge_class if found_rows else "")
        if found_rows and all(r.charge_check == "mismatch" for r in found_rows):
            mt.note = (mt.note + "; " if mt.note else "") + "Spartan calculated another charge than its structure: no values"
        out.append(mt)
    return out


@dataclass
class Pair:
    key: str                    # gas_dipole_tblite_gfn2_vs_pm6
    phase: str                  # gas | water | shift
    prop: str
    qb_source: str
    qb_column: str
    spartan_prefix: str         # "dft_..._wasser-pm6" for a shift
    spartan_level: str
    comparison: str
    unit: str
    note: str = ""


def _spartan_value(m: Match, prefix: str, prop: str, include_failed: bool) -> tuple[float | None, str]:
    for r in m.rows:
        if r.charge_check == "mismatch":
            continue
        v = r.values.get(prefix, {}).get(prop)
        if v is not None and (include_failed or r.status.get(prefix, "") == "completed"):
            return v, r.label or ";".join(r.cas_candidates[:3])
    return None, ""


def _qb_value(rec: QBRecord, phase: str, col: str, prop: str) -> float | None:
    vals = rec.phases.get(phase, {})
    v = _float(vals.get(col))
    if v is None and col == "dipole_debye_best":
        v = _float(vals.get("dipole_debye"))
    return None if v is None else convert(v, _qb_unit(col, prop), PROPS[prop].unit)


def build_pairs(matches: list[Match], table: SpartanTable) -> list[Pair]:
    found: dict[tuple, None] = {}
    for m in matches:
        for ph, vals in m.rec.phases.items():
            for prop, src, col in qb_sources(vals):
                if _qb_value(m.rec, ph, col, prop) is not None:
                    found[(ph, prop, src, col)] = None
    pairs = []
    quantum = {p: lev for p, lev in table.levels.items() if lev.quantum}
    for ph, prop, src, col in found:
        for p, lev in quantum.items():
            if lev.phase != ph:
                continue
            if not any(prop in r.values.get(p, {}) for r in table.rows):
                continue
            pp = PROPS[prop]
            pairs.append(Pair(f"{ph}_{prop}_{src}_vs_{p}", ph, prop, src, col, p, lev.text, pp.comparison, pp.unit,
                              pp.note))
    # solvation shifts: QuantaBench water - gas vs Spartan water level - gas level
    gas_levels = [p for p, lev in quantum.items() if lev.phase == "gas"]
    water_levels = [p for p, lev in quantum.items() if lev.phase == "water"]
    for ph, prop, src, col in found:
        if ph != "water" or prop not in SHIFT_PROPS or ("gas", prop, src, col) not in found:
            continue
        for wp in water_levels:
            for gp in gas_levels:
                pairs.append(Pair(f"shift_{prop}_{src}_vs_{wp}-minus-{gp}", "shift", prop, src, col, f"{wp}-{gp}",
                                  f"{quantum[wp].text} minus {quantum[gp].text}", "shift (different levels)",
                                  PROPS[prop].unit, "Spartan's shift mixes levels of theory and geometries"))
    pairs.sort(key=lambda p: (("gas", "water", "shift").index(p.phase), list(PROPS).index(p.prop), p.qb_source,
                              p.spartan_prefix))
    return pairs


def _pair_values(m: Match, pair: Pair, include_failed: bool) -> tuple[float | None, float | None, str]:
    if pair.phase == "shift":
        wp, gp = pair.spartan_prefix.split("-", 1)
        g, w = _qb_value(m.rec, "gas", pair.qb_column, pair.prop), _qb_value(m.rec, "water", pair.qb_column, pair.prop)
        qb = (w - g) if g is not None and w is not None else None
        if qb is None and m.rec.delta.get(pair.qb_column) not in (None, ""):
            qb = _float(m.rec.delta[pair.qb_column])
        sw, lab = _spartan_value(m, wp, pair.prop, include_failed)
        sg, _l = _spartan_value(m, gp, pair.prop, include_failed)
        return (sw - sg if sw is not None and sg is not None else None), qb, lab
    sv, lab = _spartan_value(m, pair.spartan_prefix, pair.prop, include_failed)
    return sv, _qb_value(m.rec, pair.phase, pair.qb_column, pair.prop), lab


@dataclass
class Validation:
    out_dir: Path
    matches: list[Match]
    pairs: list[Pair]
    summary: list[dict]
    files: list[Path]
    warnings: list[str] = field(default_factory=list)


def validate(records: list[QBRecord], table: SpartanTable, out_dir: str | Path, top: int = 10,
             include_failed: bool = False, plots: bool = True, log: Callable[[str], None] | None = None) -> Validation:
    """Compare ``records`` with ``table`` and write ``validation.csv`` (matched molecules: both values and the
    difference per pair), ``join.csv`` (every QuantaBench molecule's join), ``summary.csv`` (statistics per
    pair, overall and per cohort and charge class), ``report.md`` and parity plots (``parity_<pair>.png``, when
    matplotlib is installed)."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    matches = match(records, table)
    pairs = build_pairs([m for m in matches if m.rows], table)
    warnings: list[str] = []
    data: dict[str, list[tuple[Match, float, float, str]]] = {p.key: [] for p in pairs}
    vrows = []
    for m in matches:
        if not m.rows:
            continue
        row = {"id": m.rec.id, "name": m.rec.name, "cohort": m.cohort, "charge_class": m.charge_class,
               "charge": m.rec.charge, "inchikey": m.rec.inchikey, "join": m.join, "id_check": m.id_check,
               "spartan_labels": ";".join(dict.fromkeys(lab for r in m.rows for lab in r.labels)),
               "spartan_rows": len(m.rows), "qb_method": m.rec.method}
        for p in pairs:
            ref, qb, lab = _pair_values(m, p, include_failed)
            if ref is None or qb is None:
                continue
            row[f"{p.key}__spartan"], row[f"{p.key}__qb"], row[f"{p.key}__diff"] = ref, qb, qb - ref
            data[p.key].append((m, ref, qb, lab))
        vrows.append(row)
        if m.rec.delta and "gas" in m.rec.phases and "water" in m.rec.phases:
            # a paired table's delta_* should be water - gas, as QuantaBench's shifts here are
            for col, dv in m.rec.delta.items():
                g, w = _float(m.rec.phases["gas"].get(col)), _float(m.rec.phases["water"].get(col))
                d = _float(dv)
                if None not in (g, w, d) and abs(w - g) > 1e-6 and abs(d + (w - g)) < 1e-6 * max(1.0, abs(d)):
                    warnings.append(f"{m.rec.id}: delta_{col} is gas - water; QuantaBench's shifts here are water - gas")
    pairs = [p for p in pairs if data[p.key]]               # pairs no molecule has both values of: dropped
    ion_dipoles = sum(1 for p in pairs if p.prop == "dipole"
                      for m, *_ in data[p.key] if m.charge_class in ("anion", "cation"))
    if ion_dipoles and not any(c.endswith("_spartan_dipole_origin_debye") for c in table.columns):
        warnings.append(f"{ion_dipoles} ion dipoles: the Spartan table was imported before QuantaBench moved Spartan's "
                        "ion dipoles to the centre of mass (they are about Spartan's frame origin here, off by up to "
                        "~10 D); re-import the documents (quantabench-cli import ... --wide)")
    summary = []
    for p in pairs:
        pts = data[p.key]
        groups: list[tuple[str, str, list]] = [("all", "all", pts)]
        for gtype, attr in (("cohort", "cohort"), ("charge_class", "charge_class")):
            vals = sorted({getattr(m, attr) for m, *_ in pts})
            groups += [(gtype, v, [t for t in pts if getattr(t[0], attr) == v]) for v in vals]
        for gtype, gname, g in groups:
            if not g:
                continue
            st = stats([t[1] for t in g], [t[2] for t in g], [t[0].rec.id for t in g], top if gtype == "all" else 3)
            summary.append({"pair": p.key, "phase": p.phase, "property": p.prop, "unit": p.unit,
                            "qb_source": p.qb_source, "qb_column": p.qb_column, "spartan_level": p.spartan_level,
                            "spartan_prefix": p.spartan_prefix, "comparison": p.comparison, "group_type": gtype,
                            "group": gname, **st.row()})
    files = [write_csv(out / "validation.csv", vrows), write_csv(out / "join.csv", [_join_row(m) for m in matches]),
             write_csv(out / "summary.csv", summary)]
    if plots:
        files += _plots(out, pairs, data, log)
    files.append(_report(out, records, table, matches, pairs, summary, data, top, warnings))
    return Validation(out, matches, pairs, summary, files, warnings)


def _join_row(m: Match) -> dict:
    return {"id": m.rec.id, "name": m.rec.name, "inchikey": m.rec.inchikey, "canonical_smiles": m.rec.canonical_smiles,
            "key_source": m.rec.key_source, "phases": "+".join(sorted(m.rec.phases)), "join": m.join or "unmatched",
            "id_check": m.id_check, "spartan_rows": len(m.rows),
            "spartan_labels": ";".join(dict.fromkeys(lab for r in m.rows for lab in r.labels)),
            "spartan_charge_check": ";".join(sorted({r.charge_check for r in m.rows if r.charge_check})),
            "cohort": m.cohort, "charge_class": m.charge_class, "note": m.note}


def _plots(out: Path, pairs: list[Pair], data: dict, log) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if log:
            log("matplotlib is not installed: no parity plots (pip install 'quantabench[validate]')")
        return []
    colors = {"neutral": "#4C72B0", "anion": "#C44E52", "cation": "#55A868", "": "#8C8C8C"}
    files = []
    for p in pairs:
        pts = data[p.key]
        if len(pts) < 2:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=120)
        for cc in colors:
            sel = [(x, y) for m, x, y, _l in pts if m.charge_class == cc]
            if sel:
                xs, ys = zip(*sel)
                ax.scatter(xs, ys, s=14, alpha=0.75, color=colors[cc], label=cc or "?", edgecolors="none")
        xs = np.array([t[1] for t in pts])
        ys = np.array([t[2] for t in pts])
        lo, hi = float(min(xs.min(), ys.min())), float(max(xs.max(), ys.max()))
        if p.comparison != "correlation":
            ax.plot([lo, hi], [lo, hi], color="#555555", lw=0.8, ls="--", label="y = x")
        st = stats(xs, ys)
        if math.isfinite(st.slope):
            ax.plot([xs.min(), xs.max()], [st.slope * xs.min() + st.intercept, st.slope * xs.max() + st.intercept],
                    color="#222222", lw=1.0, label=f"fit: {st.slope:.2f}x{st.intercept:+.2f}")
        unit = {"A3": "Å³", "A2": "Å²"}.get(p.unit, p.unit)
        ax.set_xlabel(f"Spartan {p.spartan_level.split(' SOLVENT')[0]} ({unit})", fontsize=8)
        ax.set_ylabel(f"QuantaBench {p.qb_source} [{p.phase}] ({unit})", fontsize=8)
        ax.set_title(f"{PROPS[p.prop].label}: N={st.n}, MAE={st.mae:.3g}, r={st.pearson_r:.3f}", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        path = out / f"parity_{p.key}.png"
        fig.savefig(path)
        plt.close(fig)
        files.append(path)
    return files


def _g(v, fmt=".3g") -> str:
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, fmt)


def _report(out: Path, records, table: SpartanTable, matches: list[Match], pairs: list[Pair], summary: list[dict],
            data: dict, top: int, warnings: list[str]) -> Path:
    joins: dict[str, int] = {}
    for m in matches:
        joins[m.join or "unmatched"] = joins.get(m.join or "unmatched", 0) + 1
    checks: dict[str, int] = {}
    for m in matches:
        if m.id_check:
            checks[m.id_check] = checks.get(m.id_check, 0) + 1
    n = len(matches)
    lines = ["# QuantaBench vs Spartan", "",
             f"- QuantaBench: {n} molecules ("
             + ", ".join(sorted({s for r in records for s in r.source.split(" + ")})) + ")",
             f"- Spartan: `{table.path}` ({len(table.rows)} rows; quantum levels: "
             + ", ".join(f"`{p}` = {lev.text} [{lev.phase}]" for p, lev in table.levels.items() if lev.quantum) + ")",
             ""]
    if table.keys_computed:
        lines += [f"Structure keys for {table.keys_computed} Spartan rows came from their `smiles` (a wide table "
                  "written before `inchikey` existed); rows whose SMILES is a connectivity-only perception have none. "
                  "Re-import the documents for keys on every row.", ""]
    lines += ["## Join coverage", "", "| join | molecules |", "|---|---:|"]
    lines += [f"| {k} | {v} ({100 * v / max(n, 1):.0f}%) |" for k, v in sorted(joins.items(), key=lambda kv: -kv[1])]
    lines += ["", "CAS/id cross-check of the structure joins: " + (", ".join(f"{v} {k}" for k, v in sorted(checks.items()))
                                                                   or "none"), ""]
    odd = [m for m in matches if m.id_check == "differs" or "another charge" in m.note]
    if odd:
        lines += ["Id and structure disagree, or Spartan's charge does not fit its structure:", ""]
        lines += [f"- `{m.rec.id}` ({m.join or 'unmatched'}): {m.note}" for m in odd[:30]]
        if len(odd) > 30:
            lines.append(f"- ... {len(odd) - 30} more in join.csv")
        lines.append("")
    lines += ["## Statistics (all matched molecules)", "",
              "QuantaBench - Spartan, in QuantaBench's units. `correlation`: not the same quantity, read r/ρ and the fit "
              "only; `shift`: Spartan's side mixes levels of theory.", "",
              "| pair | comparison | unit | N | MAE | RMSE | MSE | r | ρ | slope | intercept |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        if s["group_type"] != "all":
            continue
        lines.append(f"| {s['pair']} | {s['comparison']} | {s['unit']} | {s['n']} | {_g(s['mae'])} | {_g(s['rmse'])} | "
                     f"{_g(s['mean_signed_error'])} | {_g(s['pearson_r'])} | {_g(s['spearman_rho'])} | {_g(s['slope'])} | "
                     f"{_g(s['intercept'])} |")
    lines += ["", "Per cohort and charge class: `summary.csv` (`group_type`, `group`).", "", "## Outliers", ""]
    for p in pairs:
        pts = sorted(data[p.key], key=lambda t: -abs(t[2] - t[1]))[:top]
        if not pts:
            continue
        lines.append(f"- **{p.key}**: " + "; ".join(f"`{m.rec.id}` {x:.3g} -> {y:.3g} ({y - x:+.3g})"
                                                   for m, x, y, _l in pts))
    notes = {p.prop: PROPS[p.prop].note for p in pairs if PROPS[p.prop].note}
    lines += ["", "## Units and caveats", "",
              "Dipoles in debye, orbital energies in eV (Spartan's hartree x 27.211386), polarizabilities and volumes in "
              "Å³, areas in Å², charges in e. Spartan's PM6 values are at its PM6-optimized geometry, its B3LYP/C-PCM "
              "values single points at the MMFF geometry; QuantaBench's at its own optimized geometry.", ""]
    lines += [f"- {prop}: {note}" for prop, note in notes.items()]
    if any(p.phase == "shift" for p in pairs):
        lines.append("- shifts: QuantaBench water - gas at one level vs Spartan's water level minus its gas level "
                     "(different methods and geometries)")
    for w in sorted(set(warnings))[:10]:
        lines.append(f"- warning: {w}")
    path = out / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
