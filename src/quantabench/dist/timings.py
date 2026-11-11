"""``quantabench-cli timings``: the stage timings every run records (``chem.telemetry``), from many run folders.

    quantabench-cli timings [PATHS ...] [-o timings.csv]      one CSV + a summary per machine, stage and engine
    quantabench-cli timings --fit [--no-save]                 refit the CREST cost model per machine and level

``find_job_tables`` looks for ``jobs.sqlite`` under the given folders (default: the output folder in the settings),
grid parts included (``<grid>/parts/<machine>/<run folder>``); the folders inside a run folder (its molecules)
are not searched. ``read_rows`` opens each job table read-only, as ``jobs.open_run`` does, and joins every stage
row with its attempt and run (machine, CPU, lane plan, settings) and the load sample nearest to it; a run seen
twice (a worker's synced folder and its downloaded copy) is read once, from the copy with the most rows. Job
tables from before the telemetry tables existed are counted and skipped.

Machine sensors (``sensor_summary``, since job table schema 5): per run the mean/max CPU temperature, effective
clock, CPU power, GPU temperature and utilization of its load samples, and ``throttle_check``: likely thermal
throttling when at least ``THROTTLE_MIN_SAMPLES`` samples under load ran below ``THROTTLE_CLOCK_FRACTION`` of the
run's typical clock under the same load, at ``THROTTLE_TEMP_C`` or hotter.

The fit (``fit_cost_model``): CREST time = A x atoms^p x threads^-q on the ``conformers`` rows of CREST runs that
ended ok, per machine and CREST level (``threads.crest_level``: method and --quick), by least squares on the logs.
It is saved (``save_fit``, ``threads.cost_model_path``) only with at least ``MIN_MOLECULES`` molecules over at
least ``MIN_WIDTHS`` thread widths, and only when p and q are in the ranges ``threads.usable`` accepts; runs use it
from then on (``threads.load_cost_model``) instead of the built-in COST_POWER and SPEEDUP_POWER.
"""

from __future__ import annotations

import bisect
import contextlib
import csv
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

MIN_MOLECULES = 30
MIN_WIDTHS = 2
#: SyncThing's temporary folders, and hidden folders, are not searched
SKIP_PREFIXES = (".", "~syncthing~")

RUN_FIELDS = ("machine", "hostname", "os", "os_release", "wsl", "cpu_model", "physical_cores", "logical_cores",
              "performance_cores", "ram_gb", "qb_version", "qb_commit", "thread_budget", "lanes_text", "n_lanes",
              "n_lanes_used", "dynamic_threads", "dynamic_budget", "speed_aware", "method", "conformer_search",
              "crest_level", "solvent", "max_conformers", "refine_top", "frequencies", "xtb_properties", "qm_method",
              "run_folder", "chunk", "load_start")
ATTEMPT_FIELDS = ("lane_device", "lane_threads", "pid", "dispatch_k", "queue_wait_s", "attempt_wall_s",
                  "attempt_outcome", "error", "mol_method", "device", "size", "n_atoms", "n_heavy", "n_rotatable",
                  "n_rings", "formal_charge", "n_F", "n_O", "n_S", "n_N", "n_conformers", "crest_threads",
                  "attempt_predicted_s")
STAGE_FIELDS = ("stage", "seq", "engine", "threads_planned", "threads_borrowed", "threads_given", "t_start_unix",
                "t_end_unix", "wall_s", "cpu_s", "child_cpu_s", "cpu_total_s", "cpu_efficiency", "programs",
                "program_wall_s", "peak_rss_mb", "child_peak_rss_mb", "predicted_s", "predicted_cost", "outcome",
                "n_in", "n_out")
LOAD_FIELDS = ("load1", "cpu_percent", "own_cpu_percent", "mem_percent", "cpu_clock_mhz", "cpu_temp_c",
               "cpu_power_w", "gpu_percent", "gpu_temp_c", "gpu_power_w")
#: likely thermal throttling (``throttle_check``): a load sample at >= LOADED_PCT machine CPU whose effective clock
#: is below THROTTLE_CLOCK_FRACTION of the run's typical clock under the same load (the median of its samples in
#: the same LOAD_BANDS band) while the CPU is at >= THROTTLE_TEMP_C
LOADED_PCT = 50.0
LOAD_BANDS = (50.0, 75.0, 90.0, 101.0)
THROTTLE_CLOCK_FRACTION = 0.85
THROTTLE_TEMP_C = 85.0
THROTTLE_MIN_SAMPLES = 2
CSV_FIELDS = ("run_id", "run_started", "mol_id", "attempt", "lane", *STAGE_FIELDS, *ATTEMPT_FIELDS, *RUN_FIELDS,
              *LOAD_FIELDS, "run_dir")

_SQL = ("SELECT a.run_id, a.mol_id, a.attempt, a.lane, a.lane_device, a.lane_threads, a.pid, a.dispatch_k, "
        "a.queue_wait_s, a.wall_s AS attempt_wall_s, a.outcome AS attempt_outcome, a.error, a.method AS mol_method, "
        "a.device, a.size, a.n_atoms, a.n_heavy, a.n_rotatable, a.n_rings, a.formal_charge, a.n_F, a.n_O, a.n_S, "
        "a.n_N, a.n_conformers, a.crest_threads, a.predicted_s AS attempt_predicted_s, "
        "s.stage, s.seq, s.engine, s.threads_planned, s.threads_borrowed, s.threads_given, s.t_start_unix, "
        "s.t_end_unix, s.wall_s, s.cpu_s, s.child_cpu_s, s.programs, s.program_wall_s, s.peak_rss_mb, "
        "s.child_peak_rss_mb, s.predicted_s, s.predicted_cost, s.outcome, s.n_in, s.n_out "
        "FROM telemetry_attempts a LEFT JOIN telemetry_stages s "
        "ON s.run_id = a.run_id AND s.mol_id = a.mol_id AND s.attempt = a.attempt "
        "ORDER BY a.dispatched_unix, a.mol_id, s.seq")


def _noop(*_a) -> None:
    pass


# ---------------------------------------------------------------------------------------- reading
def find_job_tables(paths: Iterable[str | Path], log: Callable[[str], None] = _noop) -> list[Path]:
    """Run folders (with a ``jobs.sqlite``) under ``paths``, recursively; a run folder's own subfolders (its
    molecules) are not searched."""
    from ..chem.jobs import DB_NAME

    out: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_file() and p.name == DB_NAME:
            out.append(p.parent)
            continue
        if not p.is_dir():
            log(f"{p}: no such folder; skipped")
            continue
        for root, dirs, files in os.walk(p):
            if DB_NAME in files:
                out.append(Path(root))
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if not d.startswith(SKIP_PREFIXES))
    seen, unique = set(), []
    for d in out:
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


@dataclass
class Collected:
    rows: list[dict] = field(default_factory=list)       # one per molecule attempt and stage
    runs: dict[str, dict] = field(default_factory=dict)  # run_id -> its telemetry_runs row (+ run_dir)
    folders: int = 0
    without: list[Path] = field(default_factory=list)    # job tables from before the telemetry tables
    unreadable: list[tuple[Path, str]] = field(default_factory=list)
    duplicates: int = 0                                  # runs read once although found in several copies
    loads: dict[str, list[dict]] = field(default_factory=dict)   # run_id -> its load samples (telemetry_load)


def _derive(row: dict) -> None:
    wall, given = row.get("wall_s"), row.get("threads_given")
    cpu, kids = row.get("cpu_s"), row.get("child_cpu_s")
    total = None
    if cpu is not None and (kids is not None or not row.get("programs")):
        total = cpu + (kids or 0.0)
    row["cpu_total_s"] = total
    row["cpu_efficiency"] = (total / (wall * given) if total is not None and wall and wall > 0 and given else None)


def _nearest_load(samples: list[dict], times: list[float], t: float | None) -> dict:
    if not samples or t is None:
        return {}
    i = bisect.bisect_left(times, t)
    best = min((j for j in (i - 1, i) if 0 <= j < len(samples)), key=lambda j: abs(times[j] - t))
    return {k: samples[best].get(k) for k in LOAD_FIELDS}


def _read_one(run_dir: Path) -> tuple[dict, list[dict], dict] | None:
    """(runs by id, attempt+stage rows, load samples by run id) of one job table; None without telemetry tables."""
    from ..chem import jobs, telemetry

    with jobs._read_store(run_dir) as store:            # noqa: SLF001 - read-only, immutable for another machine
        conn = store.conn
        if not telemetry.has_tables(conn):
            return None
        runs = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM telemetry_runs")}
        rows = [dict(r) for r in conn.execute(_SQL)]
        loads: dict[str, list[dict]] = {}
        with contextlib.suppress(sqlite3.Error):
            for r in conn.execute("SELECT * FROM telemetry_load ORDER BY t_unix"):
                loads.setdefault(r["run_id"], []).append(dict(r))
    return runs, rows, loads


def read_rows(run_dirs: Iterable[str | Path], log: Callable[[str], None] = _noop) -> Collected:
    """Every stage row of the job tables in ``run_dirs``, joined with its attempt, run and the nearest load
    sample (``CSV_FIELDS``). Attempts without a stage row (a lane that died before its first stage) give one row
    with an empty stage."""
    out = Collected()
    found: dict[str, tuple[int, Path, dict, list[dict], list[dict]]] = {}
    for d in run_dirs:
        d = Path(d)
        out.folders += 1
        try:
            got = _read_one(d)
        except (sqlite3.Error, OSError, ValueError) as exc:
            out.unreadable.append((d, f"{type(exc).__name__}: {exc}"))
            continue
        if got is None:
            out.without.append(d)
            continue
        runs, rows, loads = got
        by_run: dict[str, list[dict]] = {}
        for r in rows:
            by_run.setdefault(r["run_id"], []).append(r)
        for run_id, run in runs.items():
            mine = by_run.get(run_id, [])
            old = found.get(run_id)
            if old is not None:
                out.duplicates += 1
                if old[0] >= len(mine):
                    continue
            found[run_id] = (len(mine), d, run, mine, loads.get(run_id, []))
    for run_id, (_n, d, run, rows, loads) in found.items():
        run = {**run, "run_dir": str(d)}
        out.runs[run_id] = run
        out.loads[run_id] = loads
        times = [s.get("t_unix") or 0.0 for s in loads]
        for r in rows:
            t = r.get("t_start_unix")
            if t is not None and r.get("t_end_unix") is not None:
                t = (t + r["t_end_unix"]) / 2
            row = {**{k: run.get(k) for k in RUN_FIELDS}, **r, "run_started": run.get("started"), "run_dir": str(d),
                   **_nearest_load(loads, times, t)}
            _derive(row)
            out.rows.append(row)
    return out


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r.get(k) for k in CSV_FIELDS})
    return path


# ---------------------------------------------------------------------------------------- summary
def _is_crest(row: dict) -> bool:
    return row.get("stage") == "conformers" and str(row.get("engine") or "").startswith("crest")


def summarize(rows: list[dict], crest_power: float | None = None) -> list[dict]:
    """One line per machine, stage and engine over the stages that ended ok: molecules, median wall s, median
    s / atom^p (p = ``crest_power`` (default the built-in COST_POWER) for CREST searches, 1 for the other stages),
    CPU efficiency (CPU s / (wall s x threads given), summed over the rows whose CPU time is known) and the median
    threads given."""
    from . import threads as threads_mod

    p_crest = threads_mod.COST_POWER if crest_power is None else float(crest_power)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("outcome") != "ok" or not r.get("stage") or r.get("wall_s") is None:
            continue
        groups.setdefault((r.get("machine") or "?", r["stage"], r.get("engine") or ""), []).append(r)
    order = {s: i for i, s in enumerate(("prepare", "conformers", "refine", "rank", "descriptors", "properties",
                                         "frequencies", "qm", "export"))}
    out = []
    for (machine, stage, engine), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99),
                                                                              kv[0][2])):
        p = p_crest if _is_crest(rs[0]) else 1.0
        per = [r["wall_s"] / max(1.0, float(r.get("size") or r.get("n_atoms") or 1.0)) ** p for r in rs]
        known = [r for r in rs if r.get("cpu_total_s") is not None and r.get("threads_given")]
        denom = sum(r["wall_s"] * r["threads_given"] for r in known)
        out.append({"machine": machine, "stage": stage, "engine": engine,
                    "molecules": len({(r["run_id"], r["mol_id"], r["attempt"]) for r in rs}),
                    "median_s": statistics.median(r["wall_s"] for r in rs), "median_s_per_atom_p": statistics.median(per),
                    "p": p, "cpu_efficiency": sum(r["cpu_total_s"] for r in known) / denom if denom > 0 else None,
                    "threads": statistics.median(int(r.get("threads_given") or 1) for r in rs),
                    "total_s": sum(r["wall_s"] for r in rs)})
    return out


def _g(x, fmt: str = ".3g") -> str:
    return "-" if x is None else format(x, fmt)


def summary_lines(summary: list[dict]) -> list[str]:
    if not summary:
        return ["No finished stages."]
    w = max(len(s["engine"]) for s in summary)
    lines = [f"{'machine':<16} {'stage':<12} {'engine':<{w}} {'molecules':>9} {'median s':>9} "
             f"{'s/atom^p':>9} {'p':>4} {'CPU eff':>7} {'threads':>7}"]
    for s in summary:
        lines.append(f"{s['machine'][:16]:<16} {s['stage']:<12} {s['engine']:<{w}} {s['molecules']:>9} "
                     f"{_g(s['median_s']):>9} {_g(s['median_s_per_atom_p']):>9} {s['p']:>4g} "
                     f"{_g(s['cpu_efficiency'], '.2f'):>7} {s['threads']:>7g}")
    return lines


# ---------------------------------------------------------------------------------------- sensors
def _band(cpu_percent: float) -> int:
    return next((i for i, top in enumerate(LOAD_BANDS[1:]) if cpu_percent < top), len(LOAD_BANDS) - 2)


def throttle_check(samples: list[dict]) -> dict:
    """Likely thermal throttling in one run's load samples: {"loaded": samples at >= LOADED_PCT CPU with a clock,
    "flagged": those below THROTTLE_CLOCK_FRACTION of the median clock of their load band (all loaded samples when
    a band has fewer than 3) at >= THROTTLE_TEMP_C, "throttling": at least THROTTLE_MIN_SAMPLES flagged, "typical_mhz":
    the median loaded clock, "low_mhz": the lowest flagged clock}."""
    loaded = [s for s in samples if (s.get("cpu_percent") or 0) >= LOADED_PCT and s.get("cpu_clock_mhz")]
    out = {"loaded": len(loaded), "flagged": 0, "throttling": False, "typical_mhz": None, "low_mhz": None}
    if not loaded:
        return out
    every = statistics.median(s["cpu_clock_mhz"] for s in loaded)
    bands: dict[int, list[float]] = {}
    for s in loaded:
        bands.setdefault(_band(s["cpu_percent"]), []).append(s["cpu_clock_mhz"])
    typical = {b: statistics.median(v) if len(v) >= 3 else every for b, v in bands.items()}
    flagged = [s for s in loaded if s.get("cpu_temp_c") is not None and s["cpu_temp_c"] >= THROTTLE_TEMP_C
               and s["cpu_clock_mhz"] < THROTTLE_CLOCK_FRACTION * typical[_band(s["cpu_percent"])]]
    out.update(flagged=len(flagged), throttling=len(flagged) >= THROTTLE_MIN_SAMPLES, typical_mhz=every,
               low_mhz=min((s["cpu_clock_mhz"] for s in flagged), default=None))
    return out


def sensor_summary(got: Collected) -> list[dict]:
    """One row per run with sensor samples: mean/max CPU temperature and effective clock (and min clock), mean/max
    GPU temperature and utilization, mean CPU power, and ``throttle_check``."""
    out = []
    for run_id, samples in got.loads.items():
        with_sensors = [s for s in samples if any(s.get(k) is not None for k in ("cpu_clock_mhz", "cpu_temp_c",
                                                                                 "gpu_temp_c", "gpu_percent"))]
        if not with_sensors:
            continue
        run = got.runs.get(run_id, {})

        def stats(key: str, vals=with_sensors) -> tuple[float | None, float | None, float | None]:
            xs = [s[key] for s in vals if s.get(key) is not None]
            return (sum(xs) / len(xs), max(xs), min(xs)) if xs else (None, None, None)

        row = {"run_id": run_id, "machine": run.get("machine") or "?", "started": run.get("started") or "",
               "run_folder": run.get("run_folder") or "", "samples": len(with_sensors),
               "source": next((s.get("sensor_source") for s in with_sensors if s.get("sensor_source")), "")}
        for key in ("cpu_temp_c", "cpu_clock_mhz", "cpu_power_w", "gpu_temp_c", "gpu_percent", "gpu_power_w"):
            row[f"{key}_mean"], row[f"{key}_max"], row[f"{key}_min"] = stats(key)
        row.update({f"throttle_{k}": v for k, v in throttle_check(with_sensors).items()})
        out.append(row)
    out.sort(key=lambda r: (r["machine"], r["started"]))
    return out


def sensor_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    lines = ["", "Machine sensors per run (load samples every 60 s; mean / max):",
             f"{'machine':<16} {'started':<20} {'samples':>7} {'CPU °C':>11} {'clock GHz':>15} {'CPU W':>6} "
             f"{'GPU °C':>11} {'GPU %':>9}  throttling"]

    def pair(r, key, fmt=".0f", scale=1.0) -> str:
        m, x = r[f"{key}_mean"], r[f"{key}_max"]
        return "-" if m is None else f"{m / scale:{fmt}} / {x / scale:{fmt}}"

    for r in rows:
        if r["throttle_throttling"]:
            flag = (f"LIKELY: {r['throttle_flagged']} of {r['throttle_loaded']} loaded samples at >= "
                    f"{THROTTLE_TEMP_C:.0f} °C ran below {THROTTLE_CLOCK_FRACTION:.0%} of the typical "
                    f"{r['throttle_typical_mhz'] / 1000:.2f} GHz (lowest {r['throttle_low_mhz'] / 1000:.2f} GHz)")
        elif r["throttle_loaded"]:
            flag = f"no ({r['throttle_flagged']} of {r['throttle_loaded']} loaded samples)"
        else:
            flag = "- (no samples under load with a clock)"
        clock = pair(r, "cpu_clock_mhz", ".2f", 1000.0)
        if r["cpu_clock_mhz_min"] is not None:
            clock += f" (min {r['cpu_clock_mhz_min'] / 1000:.2f})"
        lines.append(f"{r['machine'][:16]:<16} {r['started'][:19]:<20} {r['samples']:>7} {pair(r, 'cpu_temp_c'):>11} "
                     f"{clock:>15} {_g(r['cpu_power_w_mean'], '.0f'):>6} {pair(r, 'gpu_temp_c'):>11} "
                     f"{pair(r, 'gpu_percent'):>9}  {flag}")
    return lines


# -------------------------------------------------------------------------------------------- fit
@dataclass
class Fit:
    machine: str
    level: str
    ok: bool
    why: str = ""                 # why it is not used (not enough data, outside the accepted range)
    n_molecules: int = 0
    n_samples: int = 0
    widths: list[int] = field(default_factory=list)
    sizes: tuple[float, float] = (0.0, 0.0)
    runs: int = 0
    power: float | None = None
    speedup_power: float | None = None
    scale_s: float | None = None
    power_se: float | None = None
    speedup_power_se: float | None = None
    scatter: float | None = None
    builtin_scatter: float | None = None
    power_only: float | None = None    # p fitted with q fixed at the built-in (shown when one width is all there is)

    def entry(self) -> dict:
        return {"power": self.power, "speedup_power": self.speedup_power, "scale_s": self.scale_s,
                "power_se": self.power_se, "speedup_power_se": self.speedup_power_se, "scatter": self.scatter,
                "builtin_scatter": self.builtin_scatter, "n_molecules": self.n_molecules, "n_samples": self.n_samples,
                "widths": self.widths, "size_range": list(self.sizes), "runs": self.runs,
                "updated": datetime.now(UTC).isoformat(timespec="seconds")}


def crest_samples(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """{(machine, CREST level): [{"mol", "run", "size", "threads", "seconds"}]} from the CREST searches that ended
    ok (the conformers stage of a CREST run whose conformers came from CREST)."""
    out: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if not _is_crest(r) or r.get("outcome") != "ok":
            continue
        size, w, t = r.get("size") or r.get("n_atoms"), r.get("threads_given"), r.get("wall_s")
        if not size or not w or not t or t <= 0 or size <= 0:
            continue
        level = r.get("crest_level") or str(r.get("engine") or "")[len("crest "):].split(" (")[0]
        out.setdefault((r.get("machine") or "?", level), []).append(
            {"mol": r.get("mol_id"), "run": r.get("run_id"), "size": float(size), "threads": int(w), "seconds": float(t)})
    return out


def fit_cost_model(samples: list[dict], machine: str = "", level: str = "", min_molecules: int = MIN_MOLECULES,
                   min_widths: int = MIN_WIDTHS) -> Fit:
    """Least squares on log t = log A + p log size - q log threads. ``ok`` only with ``min_molecules`` molecules over
    ``min_widths`` thread widths and constants that ``threads.usable`` accepts; ``why`` says what is missing."""
    import numpy as np

    from . import threads as threads_mod

    fit = Fit(machine, level, False, n_samples=len(samples))
    if not samples:
        fit.why = "no CREST searches recorded"
        return fit
    fit.n_molecules = len({s["mol"] for s in samples})
    fit.widths = sorted({s["threads"] for s in samples})
    fit.runs = len({s["run"] for s in samples})
    sizes = np.array([s["size"] for s in samples])
    fit.sizes = (float(sizes.min()), float(sizes.max()))
    ln_n, ln_w = np.log(sizes), np.log([s["threads"] for s in samples])
    ln_t = np.log([s["seconds"] for s in samples])
    p0, q0 = threads_mod.COST_POWER, threads_mod.SPEEDUP_POWER
    resid0 = ln_t - p0 * ln_n + q0 * ln_w
    fit.builtin_scatter = float(math.exp(np.std(resid0)))          # the built-in exponents, best scale
    if len(set(ln_n.round(9))) >= 2:
        x = np.column_stack([np.ones_like(ln_n), ln_n])
        coef = np.linalg.lstsq(x, ln_t + q0 * ln_w, rcond=None)[0]
        fit.power_only = float(coef[1])
    missing = []
    if fit.n_molecules < min_molecules:
        missing.append(f"{fit.n_molecules} molecule(s), needs {min_molecules}")
    if len(fit.widths) < min_widths:
        missing.append(f"{len(fit.widths)} thread width(s) ({', '.join(map(str, fit.widths))}), needs {min_widths}")
    if len(set(ln_n.round(9))) < 3:
        missing.append("fewer than 3 molecule sizes")
    if missing:
        fit.why = "not enough data: " + "; ".join(missing)
        return fit
    x = np.column_stack([np.ones_like(ln_n), ln_n, -ln_w])
    coef, _res, rank, _sv = np.linalg.lstsq(x, ln_t, rcond=None)
    if rank < 3:
        fit.why = "sizes and thread widths are not independent (every size ran on one width)"
        return fit
    resid = ln_t - x @ coef
    dof = max(1, len(ln_t) - 3)
    sigma2 = float(resid @ resid) / dof
    with contextlib.suppress(np.linalg.LinAlgError):
        cov = sigma2 * np.linalg.inv(x.T @ x)
        fit.power_se, fit.speedup_power_se = float(math.sqrt(cov[1, 1])), float(math.sqrt(cov[2, 2]))
    fit.scale_s, fit.power, fit.speedup_power = float(math.exp(coef[0])), float(coef[1]), float(coef[2])
    fit.scatter = float(math.exp(math.sqrt(float(np.mean(resid ** 2)))))
    ok, why = threads_mod.usable(fit.entry())
    fit.ok, fit.why = ok, ("" if ok else f"not used: {why}")
    return fit


def fit_all(rows: list[dict], min_molecules: int = MIN_MOLECULES, min_widths: int = MIN_WIDTHS) -> list[Fit]:
    return [fit_cost_model(s, machine, level, min_molecules, min_widths)
            for (machine, level), s in sorted(crest_samples(rows).items())]


def save_fit(fit: Fit) -> Path:
    """Add ``fit`` to its machine's cost-model file (``threads.cost_model_path``), keeping the other levels."""
    from . import threads as threads_mod

    path = threads_mod.cost_model_path(fit.machine)
    data = threads_mod.read_cost_models(fit.machine) or {"machine": fit.machine, "models": {}}
    data["models"][fit.level] = fit.entry()
    data["updated"] = datetime.now(UTC).isoformat(timespec="seconds")
    from .. import __version__

    data["qb_version"] = __version__
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):        # on Windows, SyncThing or an editor may hold the file open for a moment
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))
    return path


def fit_lines(fit: Fit) -> list[str]:
    from . import threads as threads_mod

    head = (f"{fit.machine} / CREST {fit.level}: {fit.n_samples} search(es) of {fit.n_molecules} molecule(s) in "
            f"{fit.runs} run(s); threads {', '.join(map(str, fit.widths)) or '-'}; atoms {fit.sizes[0]:g}-{fit.sizes[1]:g}")
    lines = [head]
    if fit.power is not None:
        lines.append(f"  fit:      time = {fit.scale_s:.3g} s x atoms^{fit.power:.2f} (+-{_g(fit.power_se, '.2f')}) x "
                     f"threads^-{fit.speedup_power:.2f} (+-{_g(fit.speedup_power_se, '.2f')}); scatter x{fit.scatter:.2f}")
    elif fit.power_only is not None:
        lines.append(f"  p alone (q fixed at {threads_mod.SPEEDUP_POWER:g}): atoms^{fit.power_only:.2f}")
    lines.append(f"  built-in: atoms^{threads_mod.COST_POWER:g}, (threads/lane)^{threads_mod.SPEEDUP_POWER:g}"
                 + (f"; scatter x{fit.builtin_scatter:.2f} on the same data" if fit.builtin_scatter else ""))
    if not fit.ok:
        lines.append(f"  not saved: {fit.why}")
    return lines


# -------------------------------------------------------------------------------------------- CLI
def register_cli(subparsers) -> None:
    p = subparsers.add_parser(
        "timings", help="collect the stage timings runs record (jobs.sqlite) into one CSV; --fit refits the CREST "
                        "cost model",
        description="Finds jobs.sqlite under the folders (default: the output folder in the settings), grid parts "
                    "included, writes one CSV row per molecule attempt and stage with the run's machine and settings, "
                    "and prints a summary per machine, stage and engine. --fit refits CREST time = A x atoms^p x "
                    "threads^-q per machine and CREST level and saves it to <settings>/machines/cost-models/"
                    "<machine>.json, which the dynamic CREST threads then use (docs/ENGINES.md, CREST).")
    p.add_argument("paths", nargs="*", metavar="PATH", help="run folders, or folders that contain them")
    p.add_argument("-o", "--out", help="the CSV to write (default: timings.csv in the first folder)")
    p.add_argument("--fit", action="store_true", help="refit the CREST cost model per machine and level")
    p.add_argument("--no-save", action="store_true", help="with --fit: print the fit, do not save it")
    p.add_argument("--min-molecules", type=int, default=MIN_MOLECULES,
                   help=f"with --fit: fewest molecules for a fit that is saved (default {MIN_MOLECULES})")
    p.add_argument("--min-widths", type=int, default=MIN_WIDTHS,
                   help=f"with --fit: fewest thread widths for a fit that is saved (default {MIN_WIDTHS})")
    p.add_argument("--power", type=float, help="the p of s/atom^p for CREST searches in the summary (default: the "
                                               "built-in exponent)")
    p.set_defaults(func=_cli)


def _cli(args) -> int:
    from .. import settings as settings_mod

    paths = list(args.paths)
    if not paths:
        try:
            paths = [str(settings_mod.load().output_path())]
        except Exception as exc:  # noqa: BLE001
            print(f"No folder given and the settings cannot be read ({exc})", file=sys.stderr)
            return 2

    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    folders = find_job_tables(paths, log)
    got = read_rows(folders, log)
    print(f"{len(folders)} run folder(s) with a job table, {len(got.runs)} run(s) with timings, "
          f"{len({(r['run_id'], r['mol_id'], r['attempt']) for r in got.rows})} molecule attempt(s), "
          f"{sum(1 for r in got.rows if r.get('stage'))} stage row(s)")
    if got.without:
        print(f"  {len(got.without)} run folder(s) from before the timings were recorded (no telemetry tables)")
    for d, why in got.unreadable[:5]:
        print(f"  {d}: could not be read ({why})", file=sys.stderr)
    if got.duplicates:
        print(f"  {got.duplicates} copy(ies) of a run found twice (synced and downloaded): read once")
    if not got.rows:
        return 1
    out = Path(args.out) if args.out else Path(paths[0]).expanduser() / "timings.csv"
    if not args.out and not Path(paths[0]).expanduser().is_dir():
        out = Path("timings.csv")
    try:
        print(f"Wrote {write_csv(got.rows, out)}")
    except OSError as exc:
        print(f"Could not write {out} ({exc})", file=sys.stderr)
    for line in summary_lines(summarize(got.rows, args.power)):
        print(line)
    for line in sensor_lines(sensor_summary(got)):
        print(line)
    if not args.fit:
        return 0
    fits = fit_all(got.rows, args.min_molecules, args.min_widths)
    if not fits:
        print("\nNo CREST searches that ended ok: nothing to fit.")
        return 0
    for fit in fits:
        print()
        for line in fit_lines(fit):
            print(line)
        if fit.ok and not args.no_save:
            try:
                print(f"  saved to {save_fit(fit)}: the dynamic CREST threads of {fit.machine} use it from now on")
            except OSError as exc:
                print(f"  could not save it ({exc})", file=sys.stderr)
    return 0
