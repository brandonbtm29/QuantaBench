"""One input in several phases (gas and implicit solvents), and the per-molecule paired table.

    quantabench-cli build --file x.sdf --method tblite-gfn2 --no-protonate --phases gas,water --run-dir OUT
        OUT/gas/       a normal resumable run folder: jobs.sqlite, summary.csv, descriptors.csv, <id>/...
        OUT/water/     the same input and settings with --solvent water
        OUT/paired.csv one row per molecule (``write_paired``)
    quantabench-cli pair OUT/gas OUT/water --out paired.csv     # rebuild it from finished or partial runs

Each phase folder is an ordinary run (``jobs.run_resumable``), so resume, --retry-failed, lanes and the
chunked / grid splits work per phase. The pipeline computes every step it can in the phase's solvent and
records per molecule where each step ran (summary.csv ``phase_notes``; ``pipeline.MoleculeResult.phase_steps``).

Pairing joins the phases by molecule id **and** structure key, never by row order: the key is the InChIKey of
the input record (``input_inchikey``; the input SMILES for runs from before it existed). Rows whose ids match but
whose input structures differ are not paired (``pair_status``). The computed species (``inchikey``), charge,
multiplicity and method must agree too, or the energy differences are left empty with the reason in
``pair_notes``.

Layout of the paired table: the identity columns once (``IDENTITY_COLUMNS``: id, name, structure keys, charges,
the 2D descriptors), ``pair_status`` and ``pair_notes``, the solvation free energies (below), then for every
other column of summary.csv and descriptors.csv ``gas_<col>``, ``water_<col>`` and, for numeric descriptors where
a difference means something, ``delta_<col>`` = water - gas (counts, timings and flags get no delta,
``NO_DELTA``). With more than two phases, or a reference phase other than the gas phase, each other phase ``p``
gets ``delta_<p>_<col>`` and ``dG_solv_<p>_kcal`` instead (differences to the reference phase).

ΔG_solv. With a named solvent, tblite and the xtb program use the parametrized ALPB model (generalized-Born
polar term + SASA-based cavity/dispersion/solvent-structure term + an empirical shift) in the ``gsolv`` reference
state, and the total energy they report in solution **already contains** that solvation free energy: E(water) is
the electronic energy of the solute polarized by the continuum plus G_solv(ALPB). So
``dG_solv_kcal`` = E_water(R_water) - E_gas(R_gas): the best refined conformer of each phase at its own geometry,
i.e. the vertical value below plus the gas-phase distortion energy E_gas(R_water) - E_gas(R_gas) >= 0 (the
conformer and geometry relaxation between the phases).
``dG_solv_G_kcal`` = G_water - G_gas from the frequency step when both phases ran it (G = E + ZPE + thermal
enthalpy - T S, RRHO/qRRHO at each phase's own minimum; the Hessian in water includes the continuum).
``dG_solv_vertical_kcal`` = the water run's ``<engine>_gsolv_kcal``: E(water) - E(gas) at the water geometry, the
number the xtb program prints as Gsolv (with the polarization cost included).
All three refer to the transfer from 1 mol/L ideal gas to 1 mol/L ideal solution (Ben-Naim's ΔG*, tblite's
and xtb's default ``gsolv`` state; the translational terms of G cancel in the difference). For the 1 atm gas ->
1 mol/L convention add RT ln(24.46) = +1.89 kcal/mol at 298.15 K.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

KCAL_PER_EV = 23.060548              # conformers.KCAL_PER_EV
GAS = "gas"
PAIRED_NAME = "paired.csv"

#: Columns written once per molecule (from the reference phase): who it is and what was computed. The 2D
#: descriptors (``descriptors.graph_descriptors``) are the same in every phase.
IDENTITY_COLUMNS = ("id", "name", "input_inchikey", "inchikey", "canonical_smiles", "input_smiles", "smiles",
                    "input_charge", "charge", "multiplicity", "removed_fragments")
GRAPH_COLUMNS = ("formula", "mol_weight", "formal_charge", "heavy_atoms", "n_carbon", "n_fluorine", "f_to_c_ratio",
                 "n_cf2", "n_cf3", "rotatable_bonds", "fraction_csp3", "tpsa", "logp_crippen", "hbd", "hba")
#: Numeric columns without a meaningful phase difference: counts, timings, flags, settings.
NO_DELTA = frozenset({"n_embedded", "n_conformers", "best_converged", "seconds", "vib_n_imaginary",
                      "vib_symmetry_number", "vib_geometry_max_shift_A", "input_charge", "charge", "multiplicity"})


def phase_label(solvent: str) -> str:
    return solvent or GAS


def parse_phases(text: str | Sequence[str]) -> list[str]:
    """'gas,water' -> ['', 'water'] (canonical solvent names, '' = gas phase), in the order given. Raises
    ``ValueError`` for an unknown solvent, a repeated phase or an empty list."""
    from . import solvents

    items = text.split(",") if isinstance(text, str) else list(text)
    out: list[str] = []
    for item in items:
        if not item.strip():
            continue
        name = solvents.canonical(item)
        if name in out:
            raise ValueError(f"phase {phase_label(name)!r} is listed twice")
        out.append(name)
    if not out:
        raise ValueError("no phase given (e.g. --phases gas,water)")
    return out


def phase_dirs(root: str | Path, phases: Iterable[str]) -> dict[str, Path]:
    """{label: <root>/<label>} for canonical solvent names ('' -> 'gas')."""
    root = Path(root).expanduser()
    return {phase_label(s): root / phase_label(s) for s in phases}


# ---------------------------------------------------------------------------------------------- reading
@dataclass
class PhaseTable:
    """The finished molecules of one phase (one or more run folders): id -> summary row + descriptors."""

    label: str
    rows: dict[str, dict] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def _is_run(d: Path) -> bool:
    from .jobs import DB_NAME

    return (d / DB_NAME).is_file() or (d / "summary.csv").is_file()


def expand_run_dirs(paths: Iterable[str | Path]) -> list[Path]:
    """Run folders as given; a folder that is not one but holds phase folders (``build --phases`` output) stands for
    those, in name order with the gas phase first."""
    out: list[Path] = []
    for p in paths:
        d = Path(p).expanduser()
        if d.is_dir() and not _is_run(d):
            subs = sorted((s for s in d.iterdir() if s.is_dir() and _is_run(s)), key=lambda s: (s.name != GAS, s.name))
            if subs:
                out += subs
                continue
        out.append(d)
    return out


def _rows_from_store(d: Path) -> tuple[list[dict], str | None]:
    """(merged rows of the finished molecules, the run's solvent) from the job table, opened read-only."""
    from . import jobs

    with jobs._read_store(d) as store:
        run = store.get_run()
        out = []
        for r in store.conn.execute("SELECT id, status, summary FROM molecules WHERE status IN ('ok', 'failed') "
                                    "AND summary IS NOT NULL ORDER BY seq"):
            s = json.loads(r["summary"])
            out.append({**(s.get("descriptors") or {}), **(s.get("row") or {}), "id": r["id"]})
    solvent = (run or {}).get("config", {}).get("solvent") if run else None
    return out, solvent


def _rows_from_csv(d: Path) -> list[dict]:
    by_id: dict[str, dict] = {}
    for name in ("descriptors.csv", "summary.csv"):      # summary values win where both have a column
        path = d / name
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_id.setdefault(row["id"], {}).update(row)
    return list(by_id.values())


def read_phase_rows(run_dir: str | Path, log: Callable[[str], None] = lambda _m: None) -> tuple[str, list[dict]]:
    """(phase label, rows) of one run folder: from its job table when it has one (always current, also while a
    run is going on), else from summary.csv + descriptors.csv (a ``merge`` output). The label is the run's stored
    solvent, else the rows' ``phase`` column, else their ``solvent`` column."""
    from . import solvents
    from .jobs import DB_NAME

    d = Path(run_dir).expanduser()
    solvent: str | None = None
    rows: list[dict] = []
    if (d / DB_NAME).is_file():
        try:
            rows, solvent = _rows_from_store(d)
        except sqlite3.Error as exc:
            log(f"{d}: {DB_NAME} could not be read ({exc}); using its CSV tables")
            rows = _rows_from_csv(d)
    elif (d / "summary.csv").is_file():
        rows = _rows_from_csv(d)
    else:
        raise FileNotFoundError(f"{d}: neither {DB_NAME} nor summary.csv: not a run folder")
    if solvent is None:
        seen = Counter(str(r.get("phase") or "") for r in rows if r.get("phase"))
        if seen:
            label = seen.most_common(1)[0][0]
            solvent = "" if label == GAS else label
        else:
            solvent = Counter(str(r.get("solvent") or "") for r in rows).most_common(1)[0][0] if rows else ""
    return phase_label(solvents.canonical(solvent)), rows


def read_phases(run_dirs: Iterable[str | Path], log: Callable[[str], None] = lambda _m: None) -> list[PhaseTable]:
    """One ``PhaseTable`` per phase; several folders of one phase (chunks, machines) are combined, an 'ok' row
    winning over a failed one."""
    tables: dict[str, PhaseTable] = {}
    for d in expand_run_dirs(run_dirs):
        if not d.is_dir():
            log(f"{d}: no such folder; skipped")
            continue
        label, rows = read_phase_rows(d, log)
        t = tables.setdefault(label, PhaseTable(label))
        t.sources.append(str(d))
        for row in rows:
            old = t.rows.get(row["id"])
            if old is None or old.get("status") != "ok":
                t.rows[row["id"]] = row
    return list(tables.values())


# --------------------------------------------------------------------------------------------- pairing
def _num(value) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def structure_key(row: dict) -> str:
    """The join key of a row: the input InChIKey, else (results from before it existed, or InChI failed) the input
    SMILES."""
    return str(row.get("input_inchikey") or "") or str(row.get("input_smiles") or "")


def structure_conflict(a: dict, b: dict) -> str:
    """'' when rows ``a`` and ``b`` can be the same input record, else 'KEY vs KEY'. Keys are compared like with
    like: the input InChIKeys when both rows have one, else the input SMILES when both have them (a lane that was
    killed records a failure without an InChIKey), else nothing contradicts the id."""
    for col in ("input_inchikey", "input_smiles"):
        ka, kb = str(a.get(col) or ""), str(b.get(col) or "")
        if ka and kb:
            return "" if ka == kb else f"{kb} vs {ka}"
    return ""


def _same(a, b) -> bool:
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return na == nb
    return str(a if a is not None else "") == str(b if b is not None else "")


def _gsolv_vertical(row: dict) -> float | None:
    for k, v in row.items():
        if k.endswith("_gsolv_kcal") and not k.endswith("_gsolv_term_kcal"):
            return _num(v)
    return None


def _energy_problems(ref: dict, other: dict, ref_label: str, label: str) -> list[str]:
    """Why E(other) - E(ref) would not be a solvation (or transfer) energy; [] when it is."""
    out = []
    for row, lab in ((ref, ref_label), (other, label)):
        if row.get("status") != "ok":
            out.append(f"{lab} {row.get('status') or 'not computed'}")
    if out:
        return out
    if not _same(ref.get("method"), other.get("method")):
        out.append(f"method differs ({ref.get('method')} vs {other.get('method')})")
    for col in ("charge", "multiplicity"):
        if not _same(ref.get(col), other.get(col)):
            out.append(f"{col} differs ({ref.get(col)} vs {other.get(col)})")
    a, b = ref.get("inchikey") or "", other.get("inchikey") or ""
    if a and b and a != b:
        out.append(f"computed species differ ({a} vs {b}): the phases were prepared differently")
    if label != GAS and not (other.get("solvent") or ""):
        out.append(f"the {label} run computed in the gas phase ({other.get('method')} has no solvent model)")
    return out


def pair_rows(tables: Sequence[PhaseTable], reference: str | None = None) -> tuple[list[dict], list[str]]:
    """(rows, columns) of the paired table (module docstring) for ``tables`` (one per phase). ``reference``: the
    phase the differences are taken to (default: the gas phase if present, else the first)."""
    if not tables:
        return [], []
    labels = [t.label for t in tables]
    ref_label = reference or (GAS if GAS in labels else labels[0])
    if ref_label not in labels:
        raise ValueError(f"reference phase {ref_label!r} is not among the runs ({', '.join(labels)})")
    ref = next(t for t in tables if t.label == ref_label)
    others = [t for t in tables if t.label != ref_label]
    single = len(others) == 1 and ref_label == GAS

    def dname(stem: str, label: str) -> str:          # delta_<col> or delta_<phase>_<col>
        return f"{stem}_{'' if single else label + '_'}"

    ids: list[str] = list(ref.rows)
    for t in others:
        ids += [i for i in t.rows if i not in ref.rows and i not in ids]
    identity = [*IDENTITY_COLUMNS, *GRAPH_COLUMNS]
    per_phase: list[str] = []
    for t in tables:
        for row in t.rows.values():
            per_phase += [k for k in row if k not in identity and k not in per_phase]
    numeric = {k for k in per_phase if k not in NO_DELTA and any(
        _num(r.get(k)) is not None for t in tables for r in t.rows.values())}

    out_rows = []
    for mol_id in ids:
        rows = {t.label: t.rows.get(mol_id) for t in tables}
        base = rows[ref_label] or next(r for r in rows.values() if r)
        row: dict = {k: base.get(k, "") for k in identity if k in base or k in IDENTITY_COLUMNS}
        row["id"] = mol_id
        notes: list[str] = []
        status: list[str] = []
        for label, r in rows.items():
            if r is None:
                status.append(f"{label} not computed")
            elif r.get("status") != "ok":
                status.append(f"{label} {r.get('status')}")
            conflict = structure_conflict(base, r) if r is not None and r is not base else ""
            if conflict:
                notes.append(f"{label}: input structure differs ({conflict}): not paired")
                rows[label] = None
        if any(n.endswith("not paired") for n in notes):
            status.append("structure mismatch")
        r0 = rows[ref_label]
        for t in others:
            r1 = rows[t.label]
            stem = dname("dG_solv", t.label)
            if r0 is None or r1 is None:
                continue
            problems = _energy_problems(r0, r1, ref_label, t.label)
            if problems:
                if not all(p.endswith(("failed", "not computed")) for p in problems):
                    notes.append(f"{t.label}: no dG_solv ({'; '.join(problems)})")
                continue
            e0, e1 = _num(r0.get("best_energy_eV")), _num(r1.get("best_energy_eV"))
            if e0 is not None and e1 is not None:
                row[f"{stem}kcal"] = (e1 - e0) * KCAL_PER_EV
            g0, g1 = _num(r0.get("vib_gibbs_kcal")), _num(r1.get("vib_gibbs_kcal"))
            if g0 is not None and g1 is not None:
                if "frequencies: gas" in str(r1.get("phase_notes") or ""):
                    notes.append(f"{t.label}: no dG_solv_G (its frequencies ran in the gas phase)")
                else:
                    row[f"{stem}G_kcal"] = g1 - g0
            if ref_label == GAS:
                v = _gsolv_vertical(r1)
                if v is not None:
                    row[f"{stem}vertical_kcal"] = v
        row["pair_status"] = "ok" if not status else "; ".join(status)
        row["pair_notes"] = " | ".join(notes)
        for col in per_phase:
            for t in tables:
                r = rows[t.label]
                row[f"{t.label}_{col}"] = "" if r is None else r.get(col, "")
            if col in numeric:
                v0 = _num((r0 or {}).get(col))
                for t in others:
                    v1 = _num((rows[t.label] or {}).get(col))
                    if v0 is not None and v1 is not None:
                        row[f"{dname('delta', t.label)}{col}"] = v1 - v0
        out_rows.append(row)

    columns = [c for c in identity if any(c in r for r in out_rows)] + ["pair_status", "pair_notes"]
    for t in others:
        for d in ("kcal", "G_kcal", "vertical_kcal"):
            columns.append(f"{dname('dG_solv', t.label)}{d}")
    for col in per_phase:
        columns += [f"{t.label}_{col}" for t in tables]
        if col in numeric:
            columns += [f"{dname('delta', t.label)}{col}" for t in others]
    return out_rows, columns


@dataclass
class PairSummary:
    path: Path
    phases: list[str]
    sources: dict[str, list[str]]
    molecules: int
    paired: int                       # every phase ok
    with_dg: int                      # dG_solv_kcal (or each dG_solv_<p>_kcal) computed
    status: Counter

    def text(self) -> str:
        lines = [f"{self.molecules} molecule(s), {self.paired} ok in every phase ({', '.join(self.phases)}), "
                 f"{self.with_dg} with dG_solv -> {self.path}"]
        lines += [f"  {n} x {s}" for s, n in self.status.most_common() if s != "ok"]
        return "\n".join(lines)


def write_paired(run_dirs: Iterable[str | Path], out_path: str | Path, reference: str | None = None,
                 log: Callable[[str], None] = lambda _m: None) -> PairSummary:
    """Read the phases' run folders (``read_phases``), pair them (``pair_rows``) and write the CSV."""
    tables = read_phases(run_dirs, log)
    if len(tables) < 2:
        found = ", ".join(t.label for t in tables) or "none"
        raise ValueError(f"pairing needs runs of at least two phases (found: {found})")
    rows, columns = pair_rows(tables, reference)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    dg_cols = [c for c in columns if c.startswith("dG_solv_") and not c.endswith(("_G_kcal", "_vertical_kcal"))]
    with_dg = sum(all(r.get(c) not in (None, "") for c in dg_cols) for r in rows) if dg_cols else 0
    return PairSummary(out, [t.label for t in tables], {t.label: t.sources for t in tables}, len(rows),
                       sum(r["pair_status"] == "ok" for r in rows), with_dg, Counter(r["pair_status"] for r in rows))
