"""The conformer rank step (RANK-2): which level decides the lowest conformer, the window and the Boltzmann weights.

After the refinement, every refined conformer (pruned only for RMSD duplicates, never by its refinement energy) gets a
single point at the rank level at its refined geometry. Those energies order the conformers, set the rank window
(``BuildConfig.rank_window_kj``, default 30 kJ/mol), the Boltzmann weights and ``kept[0]``, so every descriptor taken
from the lowest conformer or the ensemble (``dipole_debye_best``, the ensemble averages, ``xtb_*``, the frequencies,
the QM stage's ``qm_top``) follows the rank level instead of the refinement method. docs/VALIDATION.md (RANK-1,
RANK-1b) has the measurements behind the levels; "RANK-2 integration check" shows the step reproduces them.

Levels (``BuildConfig.rank_method``):

==============  =========================================================================================================
``refine``      the refinement method's energies, pruned with ``energy_window_kcal`` (QuantaBench before RANK-2)
``gxtb+cpcm``   g-xTB gas-phase energy + ΔG_solv = E(GFN1-xTB CPCM) − E(GFN1-xTB gas) at the same geometry (tblite, ddX,
                the run's solvent); in the gas phase it is ``gxtb``
``gxtb``        g-xTB gas-phase energies (``chem.eng_gxtb``)
``aimnet2``     AIMNet2 gas-phase energies (the refinement's own energies when it refined with AIMNet2)
``aimnet2+cpcm`` AIMNet2 + the same GFN1-xTB CPCM ΔG_solv; in the gas phase ``aimnet2``
``qm``          the QM stage's level (``qm_method``) on every conformer: the QM stage then runs once, here
``auto``        by phase (RANK-1b): in a solvent ``gxtb+cpcm``, else ``aimnet2+cpcm``, else ``refine``; in the gas phase
                ``aimnet2`` when its elements, charge and spin take the molecule, else ``gxtb``, else ``refine``
==============  =========================================================================================================

A level that cannot run for a molecule (a program missing, elements outside its parameters, an open shell for AIMNet2,
no QM level) falls back: ``auto`` down its list, an explicit level to ``refine``; the molecule's notes say why, and
its ``rank`` record has the level used. A conformer whose single point fails is left out (noted); when all fail the
molecule is ranked at the refinement level.

Single points: g-xTB runs as a program with the molecule's threads (``OMP_NUM_THREADS``), tblite in this process
with the OpenMP threads the process started with (the lane's), AIMNet2 on the run's device. The g-xTB and tblite
energies are cached per conformer (``SinglePointCache``: the geometry rounded to 1e-6 Å, the charge, spin, solvent and
the engine's build), so a molecule computed again after an interruption, a time limit or a crashed lane does not
redo them. The cache is a SQLite file in QuantaBench's per-machine cache folder (``QUANTABENCH_CACHE_DIR``), never in
the run folder (which may be synced) or the settings.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .engines import HARTREE_EV

RANK_METHODS = ("auto", "refine", "gxtb+cpcm", "gxtb", "aimnet2", "aimnet2+cpcm", "qm")
LEVELS = RANK_METHODS[1:]
DEFAULT_METHOD = "auto"
DEFAULT_WINDOW_KJ = 30.0
KCAL_PER_EV = 23.060548
KJ_PER_KCAL = 4.184
KJ_PER_EV = KCAL_PER_EV * KJ_PER_KCAL
#: the levels ``auto`` tries, by phase (RANK-1b: one rank level per phase, not per fluorine content)
AUTO_SOLVENT = ("gxtb+cpcm", "aimnet2+cpcm", "refine")
AUTO_GAS = ("aimnet2", "gxtb", "refine")
GFN1_ELEMENTS = frozenset(range(1, 87))      # tblite's GFN1-xTB (the CPCM term)
GXTB_ELEMENTS = frozenset(range(1, 104))     # g-xTB v2: H to Lr

CHOICES = (("auto", "Auto (by phase: g-xTB + CPCM in a solvent, AIMNet2 in the gas phase)"),
           ("refine", "The refinement method (no rank step)"),
           ("gxtb+cpcm", "g-xTB + GFN1-xTB CPCM ΔG_solv"),
           ("gxtb", "g-xTB (gas phase)"),
           ("aimnet2", "AIMNet2 (gas phase)"),
           ("aimnet2+cpcm", "AIMNet2 + GFN1-xTB CPCM ΔG_solv"),
           ("qm", "The QM stage's level on every conformer"))


def check_method(method: str) -> str:
    if method not in RANK_METHODS:
        raise ValueError(f"rank_method must be one of {', '.join(RANK_METHODS)}, not {method!r}")
    return method


def with_cpcm(level: str) -> bool:
    return level.endswith("+cpcm")


def gas_level(level: str) -> str:
    """The gas-phase form of ``level`` ('gxtb+cpcm' -> 'gxtb')."""
    return level[: -len("+cpcm")] if with_cpcm(level) else level


def label(level: str, solvent: str = "", refine_label: str = "", qm_method: str = "") -> str:
    """'g-xTB + CPCM(GFN1, water)', 'AIMNet2 (gas)', 'GFN2-xTB (tblite) (refinement)' for the phase notes."""
    cpcm = f" + CPCM(GFN1, {solvent})" if solvent else ""
    return {"gxtb+cpcm": "g-xTB" + cpcm, "gxtb": "g-xTB (gas)", "aimnet2": "AIMNet2 (gas)",
            "aimnet2+cpcm": "AIMNet2" + cpcm, "qm": f"{qm_method or 'QM'} (QM stage)",
            "refine": f"{refine_label or 'the refinement method'} (refinement)"}.get(level, level)


# ----------------------------------------------------------------------------------- which level
@dataclass
class Plan:
    """The rank level of one molecule: ``requested`` (the setting), ``level`` (what runs), ``notes`` (why not the
    requested one, '' when it is)."""

    requested: str
    level: str
    notes: list[str] = field(default_factory=list)


def _cpcm_ok(numbers, solvent: str) -> tuple[bool, str]:
    from . import eng_tblite, solvents

    if not solvent:
        return False, "the gas phase has no solvation term"
    ok, why = eng_tblite.available()
    if not ok:
        return False, f"tblite (the GFN1-xTB CPCM term) is not available: {why}"
    if solvents.for_engine(solvent, "cpcm") is None:
        return False, f"CPCM has no dielectric constant for {solvent}"
    missing = sorted({int(z) for z in numbers} - GFN1_ELEMENTS)
    if missing:
        return False, f"GFN1-xTB (the CPCM term) has no parameters for Z = {', '.join(map(str, missing))}"
    return True, ""


def _gxtb_ok(numbers) -> tuple[bool, str]:
    from . import eng_gxtb

    ok, why = eng_gxtb.available()
    if not ok:
        return False, why
    missing = sorted({int(z) for z in numbers} - GXTB_ELEMENTS)
    if missing:
        return False, f"g-xTB has no parameters for Z = {', '.join(map(str, missing))}"
    return True, ""


def _aimnet_ok(numbers, charge: int, multiplicity: int) -> tuple[bool, str]:
    from . import engines

    ok, why = engines.available("aimnet2")
    if not ok:
        return False, f"AIMNet2 is not available: {why}"
    ok, why = engines.supports("aimnet2", numbers)
    if not ok:
        return False, why
    if int(multiplicity) > 1:
        return False, f"AIMNet2 is a closed-shell model (multiplicity {multiplicity})"
    return True, ""


def usable(level: str, numbers, charge: int, multiplicity: int, solvent: str, qm_method: str = "") -> tuple[bool, str]:
    """(ok, why not) for ranking this molecule at ``level`` here."""
    if level == "refine":
        return True, ""
    if level == "qm":
        if not qm_method:
            return False, "no QM level is set (qm_method)"
        from . import qm

        return qm.available()
    if with_cpcm(level):
        ok, why = _cpcm_ok(numbers, solvent)
        if not ok:
            return False, why
    if gas_level(level) == "gxtb":
        return _gxtb_ok(numbers)
    if gas_level(level) == "aimnet2":
        return _aimnet_ok(numbers, charge, multiplicity)
    return False, f"unknown rank level {level!r}"


def resolve(requested: str, numbers, charge: int = 0, multiplicity: int = 1, solvent: str = "",
            qm_method: str = "") -> Plan:
    """The rank level for one molecule (module docstring: ``auto`` and the fallbacks)."""
    check_method(requested)
    if requested == "refine":
        return Plan(requested, "refine")
    notes: list[str] = []
    if requested == "auto":
        chain: tuple[str, ...] = AUTO_SOLVENT if solvent else AUTO_GAS
    else:
        first = requested
        if with_cpcm(requested) and not solvent:
            first = gas_level(requested)
            notes.append(f"rank {requested}: the gas phase has no solvation term, ranked at {first}")
        chain = (first, "refine")
    for level in chain:
        ok, why = usable(level, numbers, charge, multiplicity, solvent, qm_method)
        if ok:
            return Plan(requested, level, notes)
        notes.append(f"rank {level}: {why}")
    return Plan(requested, "refine", notes)       # not reached: refine is always usable


def summary(plan: Plan) -> str:
    """One note for a molecule whose rank level is not the requested one ('' when it is)."""
    if not plan.notes:
        return ""
    return f"rank {plan.requested} -> {plan.level}: " + "; ".join(n.split(": ", 1)[-1] for n in plan.notes)


# ----------------------------------------------------------------------------------- the cache
def cache_file() -> Path:
    from platformdirs import user_cache_dir

    root = os.environ.get("QUANTABENCH_CACHE_DIR", "").strip() or user_cache_dir("QuantaBench", appauthor=False)
    return Path(root) / "rank_singlepoints.sqlite"


def cache_key(kind: str, engine_id: str, numbers, positions_A, charge: int, multiplicity: int, extra: str = "") -> str:
    """The key of one single point: what ran (``kind``, ``engine_id``: the build or version), on which molecule and
    geometry (rounded to 1e-6 Å), with which charge, spin and solvent (``extra``)."""
    pos = np.round(np.asarray(positions_A, float).reshape(-1, 3), 6) + 0.0      # + 0.0: no -0.0 in the key
    text = json.dumps([kind, engine_id, [int(z) for z in numbers], int(charge), int(multiplicity), extra,
                       pos.tolist()], separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


class SinglePointCache:
    """Energies (Eh) by ``cache_key``. Never fails a molecule: an unreadable or locked cache is skipped."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else cache_file()
        self.conn: sqlite3.Connection | None = None
        self.hits = self.misses = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), timeout=10.0)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("CREATE TABLE IF NOT EXISTS sp (key TEXT PRIMARY KEY, energy_eh REAL NOT NULL, "
                              "kind TEXT, created TEXT)")
            self.conn.commit()
        except (OSError, ValueError, sqlite3.Error):     # unwritable, a bad path (ValueError), a locked file
            self.close()

    def get(self, key: str) -> float | None:
        if self.conn is None:
            return None
        try:
            row = self.conn.execute("SELECT energy_eh FROM sp WHERE key = ?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return float(row[0])

    def put(self, key: str, energy_eh: float, kind: str = "") -> None:
        if self.conn is None:
            return
        with contextlib.suppress(sqlite3.Error):
            self.conn.execute("INSERT OR REPLACE INTO sp (key, energy_eh, kind, created) VALUES (?, ?, ?, ?)",
                              (key, float(energy_eh), kind, time.strftime("%Y-%m-%dT%H:%M:%S")))
            self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            with contextlib.suppress(sqlite3.Error):
                self.conn.close()
        self.conn = None


# ----------------------------------------------------------------------------------- single points
def _tblite_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("tblite")
    except metadata.PackageNotFoundError:
        return "?"


class CpcmTerm:
    """The GFN1-xTB CPCM solvation term of one molecule's conformers (tblite, ddX): one gas-phase and one CPCM
    calculator, each SCF restarted from the previous conformer's wavefunction. That is about 30 % faster than fresh
    SCFs and changes ΔG_solv by about 1e-6 kJ/mol (the SCF convergence); the ddX CPCM SCF is most of the cost."""

    def __init__(self, numbers, charge: int, multiplicity: int, solvent: str, cache: SinglePointCache | None = None):
        from . import solvents

        self.numbers, self.charge, self.multiplicity, self.solvent = list(numbers), int(charge), int(multiplicity), solvent
        self.eps = solvents.for_engine(solvent, "cpcm")
        self.cache = cache
        self.version = _tblite_version()
        self._calcs: dict[str, object] = {}

    def _calc(self, model: str):
        from . import eng_tblite

        if model not in self._calcs:
            self._calcs[model] = eng_tblite.TBLiteCalculator(self.numbers, "GFN1-xTB", self.charge, self.multiplicity,
                                                             solvent=self.solvent if model else "",
                                                             solvation_model=model or "alpb")
        return self._calcs[model]

    def __call__(self, positions_A) -> tuple[float, float]:
        """(E(GFN1-xTB gas), E(GFN1-xTB CPCM)) in Eh at one geometry; ΔG_solv is the difference."""
        out = []
        for model in ("", "cpcm"):
            key = cache_key("tblite-gfn1" + (f"-cpcm-{self.eps:g}" if model else "-gas"), self.version, self.numbers,
                            positions_A, self.charge, self.multiplicity, self.solvent if model else "")
            e = self.cache.get(key) if self.cache is not None else None
            if e is None:
                e = float(self._calc(model).singlepoint(np.asarray(positions_A, float)).get("energy"))
                if self.cache is not None:
                    self.cache.put(key, e, "tblite-gfn1-" + (model or "gas"))
            out.append(e)
        return out[0], out[1]


def cpcm_term_eh(numbers, positions_A, charge: int, multiplicity: int, solvent: str,
                 cache: SinglePointCache | None = None) -> tuple[float, float]:
    """(E(GFN1-xTB gas), E(GFN1-xTB CPCM)) in Eh at one geometry (``CpcmTerm`` for one conformer)."""
    return CpcmTerm(numbers, charge, multiplicity, solvent, cache)(positions_A)


@dataclass
class Ranked:
    """The rank step's outcome for one molecule: the energies (eV) by conformer id (None: its single point failed),
    their terms, the engines used, and notes."""

    level: str
    energies_ev: dict[int, float | None]
    terms: dict[int, dict] = field(default_factory=dict)
    engines: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reused_refinement: bool = False
    qm_done: bool = False
    cache_hits: int = 0
    cache_misses: int = 0


def _positions(mol, conf_id: int) -> np.ndarray:
    return np.asarray(mol.GetConformer(int(conf_id)).GetPositions(), float)


def compute(level: str, mol, conformers, *, charge: int = 0, multiplicity: int = 1, solvent: str = "",
            refine_method: str = "", device: str = "cpu", threads: int = 1, cache: SinglePointCache | None = None,
            should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = lambda _m: None,
            job_name: str = "rank") -> Ranked:
    """Rank energies of ``conformers`` (objects with ``conf_id`` and ``energy_ev``, the refinement's) of ``mol`` at
    ``level`` (not 'refine' or 'qm': those are the pipeline's). Single points that fail leave None and a note."""
    from . import external
    from .optimize import Cancelled

    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    out = Ranked(level, {})
    gas = gas_level(level)
    if gas == "aimnet2" and refine_method == "aimnet2":
        out.reused_refinement = True        # the refined geometries are AIMNet2 minima: their energies are the ranks
        out.engines["aimnet2"] = _package_version("aimnet")
    if with_cpcm(level):
        out.engines["tblite"] = f"tblite {_tblite_version()} GFN1-xTB CPCM (ddX, {solvent})"
    gxtb_prog = calc = None
    cpcm = CpcmTerm(numbers, charge, multiplicity, solvent, cache) if with_cpcm(level) else None
    if gas == "gxtb":
        from . import eng_gxtb

        gxtb_prog = eng_gxtb.find()
        info = eng_gxtb.build_info(gxtb_prog)
        out.engines["gxtb"] = info.to_dict() if info is not None else {}
        engine_id = (info.sha256 or info.xtb_version) if info is not None else "?"
    elif gas == "aimnet2" and not out.reused_refinement:
        from . import engines

        out.engines["aimnet2"] = _package_version("aimnet")
        dev = device if engines.ENGINES["aimnet2"].gpu_capable else "cpu"
        try:
            calc = engines.make_calculator("aimnet2", numbers, charge, multiplicity, dev)
        except Exception:
            if dev == "cpu":
                raise
            calc = engines.make_calculator("aimnet2", numbers, charge, multiplicity, "cpu")
            out.notes.append(f"rank: AIMNet2 failed on {dev}; single points on the CPU")
    with contextlib.ExitStack() as stack:
        folder = stack.enter_context(external.work_dir(f"gxtb_{job_name}")) if gxtb_prog is not None else None
        for c in conformers:
            if should_stop():
                raise Cancelled()
            cid = int(c.conf_id)
            pos = _positions(mol, cid)
            terms: dict = {}
            try:
                if gas == "gxtb":
                    from . import eng_gxtb

                    key = cache_key("gxtb", engine_id, numbers, pos, charge, multiplicity)
                    e_gas = cache.get(key) if cache is not None else None
                    if e_gas is None:
                        e_gas = eng_gxtb.single_point(folder, numbers, pos, charge, multiplicity, threads=threads,
                                                      program=gxtb_prog, should_stop=should_stop)
                        if cache is not None:
                            cache.put(key, e_gas, "gxtb")
                    terms["gxtb_eh"] = e_gas
                    e_ev = e_gas * HARTREE_EV
                elif out.reused_refinement:
                    e_ev = float(c.energy_ev)
                    terms["aimnet2_ev"] = e_ev
                else:
                    from ase import Atoms

                    atoms = Atoms(numbers=numbers, positions=pos)
                    atoms.calc = calc
                    e_ev = float(atoms.get_potential_energy())
                    terms["aimnet2_ev"] = e_ev
                if with_cpcm(level):
                    e1_gas, e1_cpcm = cpcm(pos)
                    terms.update(gfn1_gas_eh=e1_gas, gfn1_cpcm_eh=e1_cpcm,
                                 dgsolv_kj=(e1_cpcm - e1_gas) * HARTREE_EV * KJ_PER_EV)
                    e_ev += (e1_cpcm - e1_gas) * HARTREE_EV
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - one conformer's single point: noted, the others go on
                first = str(exc).splitlines()[0][:200] if str(exc) else ""
                out.notes.append(f"rank: conformer {cid} has no {level} energy ({type(exc).__name__}: {first})")
                log(f"  {out.notes[-1]}")
                out.energies_ev[cid] = None
                continue
            terms["rank_ev"] = e_ev
            out.energies_ev[cid] = e_ev
            out.terms[cid] = terms
    if cache is not None:
        out.cache_hits, out.cache_misses = cache.hits, cache.misses
    return out


def _package_version(name: str) -> str:
    from importlib import metadata

    try:
        return f"{name} {metadata.version(name)}"
    except metadata.PackageNotFoundError:
        return name


def engine_label(ranked: Ranked) -> str:
    """The rank stage's engine for the telemetry rows: 'g-xTB 2.0.1 (sha256 1b4e30b68ed4) + tblite 0.4.0 GFN1-xTB
    CPCM (ddX, water)'."""
    parts = []
    g = ranked.engines.get("gxtb")
    if isinstance(g, dict):
        if g:
            parts.append((f"g-xTB {g['release']}" if g.get("release") else "g-xTB, unknown build")
                         + (f" (sha256 {g['sha256'][:12]})" if g.get("sha256") else ""))
        else:
            parts.append("g-xTB")
    if "aimnet2" in ranked.engines:
        parts.append(str(ranked.engines["aimnet2"]) + (" (the refinement's energies)" if ranked.reused_refinement
                                                       else ""))
    if "tblite" in ranked.engines:
        parts.append(ranked.engines["tblite"])
    return " + ".join(parts) or ranked.level


# ----------------------------------------------------------------------------------- the pipeline's step
def window_keep(energies_ev: dict[int, float], window_kj: float) -> list[int]:
    """Conformer ids within ``window_kj`` of the lowest, lowest first (ties keep their order)."""
    items = sorted(energies_ev.items(), key=lambda kv: kv[1])
    if not items:
        return []
    e0 = items[0][1]
    return [cid for cid, e in items if (e - e0) * KJ_PER_EV <= float(window_kj) + 1e-9]


def record(plan: Plan, pool, ranked: Ranked | None, kept_ids: list[int], window_kj: float, *, solvent: str = "",
           refine_label: str = "", qm_method: str = "", seconds: float = 0.0, threads: int = 1) -> dict:
    """The molecule's ``rank`` record (``MoleculeResult.rank``, in its summary.json): the level, the pool with the
    refine and rank energies of every conformer (kept or not), the pick and whether it changed."""
    by_refine = sorted(pool, key=lambda c: c.energy_ev)
    refine_pick = int(by_refine[0].conf_id) if by_refine else None
    e_ref0 = by_refine[0].energy_ev if by_refine else 0.0
    energies = ranked.energies_ev if ranked is not None else {}
    ok = {k: v for k, v in energies.items() if v is not None}
    e_rank0 = min(ok.values()) if ok else None
    rows = []
    for i, c in enumerate(by_refine):
        cid = int(c.conf_id)
        e = energies.get(cid)
        rows.append({"conf_id": cid, "refine_order": i, "refine_energy_ev": float(c.energy_ev),
                     "refine_rel_kj": (c.energy_ev - e_ref0) * KJ_PER_EV,
                     "rank_energy_ev": e, "rank_rel_kj": None if e is None or e_rank0 is None
                     else (e - e_rank0) * KJ_PER_EV, "kept": cid in kept_ids,
                     **({"terms": ranked.terms[cid]} if ranked is not None and cid in ranked.terms else {})})
    pick = kept_ids[0] if kept_ids else refine_pick
    pick_row = next((r for r in rows if r["conf_id"] == pick), None)
    return {"method": plan.requested, "level": plan.level,
            "label": label(plan.level, solvent, refine_label, qm_method),
            "window_kj": float(window_kj), "n_pool": len(pool), "n_kept": len(kept_ids),
            "pick_conf_id": pick, "refine_pick_conf_id": refine_pick, "pick_changed": pick != refine_pick,
            "pick_refine_order": pick_row["refine_order"] if pick_row else None,
            "pick_refine_rel_kj": pick_row["refine_rel_kj"] if pick_row else None,
            "engines": dict(ranked.engines) if ranked is not None else {},
            "reused_refinement": bool(ranked.reused_refinement) if ranked is not None else False,
            "cache_hits": ranked.cache_hits if ranked is not None else 0,
            "cache_misses": ranked.cache_misses if ranked is not None else 0,
            "threads": int(threads), "seconds": float(seconds), "notes": list(plan.notes),
            "conformers": rows}


def qm_ranked(result, cs, pool, cfg, *, threads: int, should_stop, log) -> Ranked:
    """Rank level ``qm``: the QM stage (``qm.run_stage``) on every conformer of ``pool``; its energies rank them and
    its extras (properties, cubes, frequencies) belong to its lowest. ``result.qm`` is set, so the pipeline's QM
    stage does not run again."""
    from . import qm

    result.conformers = list(pool)
    qcfg = replace(cfg, qm_top=len(pool))
    qm.run_stage(result, cs, qcfg, threads=threads, should_stop=should_stop, log=log)
    out = Ranked("qm", {}, qm_done=True)
    for rec in (result.qm or {}).get("conformers") or []:
        e = rec.get("energy_ev")
        out.energies_ev[int(rec["conf_id"])] = None if e is None else float(e)
        if e is None:
            out.notes.append(f"rank: conformer {rec['conf_id']} has no QM energy")
    for c in pool:
        out.energies_ev.setdefault(int(c.conf_id), None)
    out.engines["qm"] = f"{(result.qm or {}).get('program', 'psi4')} {(result.qm or {}).get('program_version', '')} " \
                        f"{cfg.qm_method}".replace("  ", " ")
    return out
