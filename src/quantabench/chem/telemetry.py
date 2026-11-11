"""Run telemetry: every batch records its machine, thread plan and stage timings in its own job table.

The lane plan and the dynamic CREST threads (``dist.lanes``, ``dist.threads``) rest on a cost model fitted to a
few runs. Every run now adds data to it (SCHED-1 step 1): ``quantabench-cli timings`` (``dist.timings``) collects
the rows of many run folders into one CSV, and ``timings --fit`` refits the CREST cost model per machine.

Tables in the run folder's ``jobs.sqlite`` (``create_tables``, called by ``JobStore``; old job tables get them the
first time a run opens them, and columns added later are added the same way):

    telemetry_runs      one row per call of ``jobs.run_resumable`` (a run, or a resume of it; each machine part
                        of a grid runs in its own folder, so it has its own row): QuantaBench version and commit,
                        machine, OS, CPU, cores, RAM, thread budget, lane plan, dynamic threads, the cost model in
                        force, engines and their versions, the settings that change the cost, start and end
    telemetry_load      the machine's load at the start and every ``LOAD_INTERVAL_S``: load average (POSIX), CPU
                        percent of the whole machine and of this run's processes, memory in use, and (schema 5,
                        MON-1) the machine's sensors: effective clock, CPU temperature and power, GPU utilization,
                        temperature and power (``SENSOR_COLUMNS``, ``quantabench.sensors``; NULL where unknown)
    telemetry_attempts  one row per molecule attempt: lane and its threads, CREST threads given and borrowed,
                        queue wait, wall, outcome, the molecule's features, the cost model's prediction
    telemetry_stages    one row per attempt and stage (prepare, conformers, refine, rank, descriptors, properties,
                        frequencies, qm, export; rank's engine names the g-xTB build by its SHA-256): engine,
                        threads planned/borrowed/given, wall start and end, CPU seconds of this process and of the
                        programs it ran, peak memory, conformers in and out

``properties`` (the xTB properties) is the part of ``MoleculeResult.timings["descriptors"]`` after the descriptors;
``export`` (writing the molecule's files) is not in ``timings``. Times are Unix seconds (``time.time``) and
durations ``time.perf_counter`` differences. CPU seconds: ``os.times`` of the process that ran the stage (every
thread; ``time.process_time``), plus the programs ``external.run_program`` ran in it (``external.programs_since``); a program whose CPU
time cannot be known (``wsl:`` programs on Windows) makes the stage's ``child_cpu_s`` NULL.

How it is written: ``StageClock`` runs inside ``pipeline.run_molecule`` (in a lane or the main process) and puts
its numbers on ``MoleculeResult.telemetry``; a lane sends them back with the result. The parent (``RunTelemetry``),
already the only writer of the job table, adds the lane and thread fields and inserts the rows just before the
molecule's own commit, so they share its transaction: no extra fsync and no second writer. A lane that dies or is
killed gets a partial row from what the parent saw (its last stage and when that began).

Nothing here may fail a molecule or a run: every entry point catches its own errors, logs one line at most and
goes on. ``QUANTABENCH_TELEMETRY=0`` turns recording off (to measure what it costs).
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

ENV_SWITCH = "QUANTABENCH_TELEMETRY"
LOAD_INTERVAL_S = 60.0
ERROR_CHARS = 300

#: the CREST stage's row is the one the cost model predicts
CREST_STAGE = "conformers"

RUN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "TEXT NOT NULL PRIMARY KEY"),
    ("started", "TEXT"), ("ended", "TEXT"), ("started_unix", "REAL"), ("ended_unix", "REAL"), ("wall_s", "REAL"),
    ("qb_version", "TEXT"), ("qb_commit", "TEXT"), ("entry", "TEXT"),
    ("machine", "TEXT"), ("hostname", "TEXT"), ("os", "TEXT"), ("os_release", "TEXT"), ("wsl", "INTEGER"),
    ("arch", "TEXT"), ("python", "TEXT"), ("cpu_model", "TEXT"), ("physical_cores", "INTEGER"),
    ("logical_cores", "INTEGER"), ("performance_cores", "INTEGER"), ("ram_gb", "REAL"),
    ("thread_budget", "INTEGER"), ("threads_setting", "INTEGER"), ("lanes_text", "TEXT"), ("lanes_json", "TEXT"),
    ("n_lanes", "INTEGER"), ("n_lanes_used", "INTEGER"), ("dynamic_threads", "INTEGER"),
    ("dynamic_budget", "INTEGER"), ("speed_aware", "INTEGER"), ("cost_model", "TEXT"), ("engines_json", "TEXT"),
    ("method", "TEXT"), ("conformer_search", "TEXT"), ("crest_method", "TEXT"), ("crest_quick", "INTEGER"),
    ("crest_level", "TEXT"), ("solvent", "TEXT"), ("max_conformers", "INTEGER"), ("refine_top", "INTEGER"),
    ("frequencies", "INTEGER"), ("xtb_properties", "INTEGER"), ("qm_method", "TEXT"), ("config_json", "TEXT"),
    ("run_folder", "TEXT"), ("chunk", "TEXT"), ("n_total", "INTEGER"), ("n_todo", "INTEGER"),
    ("n_done_before", "INTEGER"), ("load_start", "REAL"),
    ("n_ok", "INTEGER"), ("n_failed", "INTEGER"), ("n_attempts", "INTEGER"),
)
#: the machine's sensors in each load sample (``quantabench.sensors``; schema 5, all nullable): effective CPU clock
#: (MHz), CPU package temperature (°C) and power (W), GPU utilization (%), temperature (°C) and power (W)
SENSOR_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cpu_clock_mhz", "REAL"), ("cpu_temp_c", "REAL"), ("cpu_power_w", "REAL"), ("gpu_percent", "REAL"),
    ("gpu_temp_c", "REAL"), ("gpu_power_w", "REAL"), ("sensor_source", "TEXT"),
)
LOAD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "TEXT NOT NULL"), ("t_unix", "REAL"), ("t_rel_s", "REAL"), ("load1", "REAL"), ("load5", "REAL"),
    ("load15", "REAL"), ("cpu_percent", "REAL"), ("own_cpu_percent", "REAL"), ("mem_percent", "REAL"),
    *SENSOR_COLUMNS,
)
FEATURES = ("n_atoms", "n_heavy", "n_rotatable", "n_rings", "formal_charge", "n_F", "n_O", "n_S", "n_N")
ATTEMPT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "TEXT NOT NULL"), ("mol_id", "TEXT NOT NULL"), ("attempt", "INTEGER NOT NULL"),
    ("lane", "TEXT"), ("lane_device", "TEXT"), ("lane_threads", "INTEGER"), ("pid", "INTEGER"),
    ("dispatch_k", "INTEGER"), ("ready_unix", "REAL"), ("dispatched_unix", "REAL"), ("ended_unix", "REAL"),
    ("queue_wait_s", "REAL"), ("wall_s", "REAL"), ("outcome", "TEXT"), ("error", "TEXT"), ("last_stage", "TEXT"),
    ("method", "TEXT"), ("device", "TEXT"), ("size", "REAL"),
    *((f, "INTEGER") for f in FEATURES),
    ("n_conformers", "INTEGER"), ("crest_threads", "INTEGER"), ("threads_borrowed", "INTEGER"),
    ("predicted_cost", "REAL"), ("predicted_s", "REAL"), ("cpu_s", "REAL"), ("child_cpu_s", "REAL"),
    ("peak_rss_mb", "REAL"),
)
STAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run_id", "TEXT NOT NULL"), ("mol_id", "TEXT NOT NULL"), ("attempt", "INTEGER NOT NULL"),
    ("stage", "TEXT NOT NULL"), ("seq", "INTEGER"), ("lane", "TEXT"), ("engine", "TEXT"),
    ("threads_planned", "INTEGER"), ("threads_borrowed", "INTEGER"), ("threads_given", "INTEGER"),
    ("t_start_unix", "REAL"), ("t_end_unix", "REAL"), ("wall_s", "REAL"), ("cpu_s", "REAL"), ("child_cpu_s", "REAL"),
    ("programs", "INTEGER"), ("program_wall_s", "REAL"), ("peak_rss_mb", "REAL"), ("child_peak_rss_mb", "REAL"),
    ("predicted_s", "REAL"), ("predicted_cost", "REAL"), ("outcome", "TEXT"), ("n_in", "INTEGER"),
    ("n_out", "INTEGER"),
)
TABLES: dict[str, tuple[tuple[tuple[str, str], ...], tuple[str, ...]]] = {
    "telemetry_runs": (RUN_COLUMNS, ()),
    "telemetry_load": (LOAD_COLUMNS, ()),
    "telemetry_attempts": (ATTEMPT_COLUMNS, ("run_id", "mol_id", "attempt")),
    "telemetry_stages": (STAGE_COLUMNS, ("run_id", "mol_id", "attempt", "stage")),
}
OUTCOMES = ("running", "ok", "failed", "timeout", "crashed", "cancelled", "interrupted")


def enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "").strip().lower() not in ("0", "off", "false", "no")


def _now_iso(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------ schema
def create_tables(conn) -> None:
    """Create the telemetry tables, and add the columns a table made by an older version lacks. Inside the
    caller's transaction (``JobStore._create_schema``)."""
    for table, (columns, key) in TABLES.items():
        body = ", ".join(f"{name} {kind}" for name, kind in columns)
        if key:
            body += f", PRIMARY KEY ({', '.join(key)})"
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({body})")
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns:
            if name not in have:            # a column added after this table was made: no constraints there
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind.replace('NOT NULL', '').strip()}")
    conn.execute("CREATE INDEX IF NOT EXISTS telemetry_load_run ON telemetry_load(run_id)")


def has_tables(conn) -> bool:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return all(t in names for t in ("telemetry_runs", "telemetry_attempts", "telemetry_stages"))


def _insert(conn, table: str, row: dict) -> None:
    known = [c for c, _ in TABLES[table][0] if c in row]
    marks = ",".join("?" * len(known))
    conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(known)}) VALUES ({marks})",  # noqa: S608
                 [_sql_value(row[c]) for c in known])


def _sql_value(v):
    if v is None or isinstance(v, (int, float, str)):
        return v
    if isinstance(v, bool):
        return int(v)
    if hasattr(v, "item"):          # numpy scalars
        return v.item()
    return json.dumps(v, default=str)


# ------------------------------------------------------------------------- measuring (any process)
_ps_self = None


def _process_cpu() -> float:
    """User + system CPU seconds of this process, every thread (``os.times`` counts in 10 ms ticks on Linux)."""
    return time.process_time()


def self_peak_rss_mb() -> float | None:
    """This process's peak resident memory so far (a high-water mark), MB."""
    global _ps_self
    try:
        if sys.platform == "win32":
            if _ps_self is None:
                import psutil

                _ps_self = psutil.Process()
            return _ps_self.memory_info().peak_wset / 2**20
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (2**20 if sys.platform == "darwin" else 1024.0)
    except Exception:  # noqa: BLE001
        return None


def molecule_features(mol) -> dict:
    """Atoms (with hydrogens), heavy atoms, rotatable bonds, rings, formal charge and F/O/S/N counts."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    counts: dict[str, int] = {}
    for a in mol.GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    return {"n_atoms": mol.GetNumAtoms() + sum(a.GetTotalNumHs() for a in mol.GetAtoms()),
            "n_heavy": mol.GetNumHeavyAtoms(), "n_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "n_rings": rdMolDescriptors.CalcNumRings(mol), "formal_charge": Chem.GetFormalCharge(mol),
            **{f"n_{e}": counts.get(e, 0) for e in ("F", "O", "S", "N")}}


class StageClock:
    """The stages of one molecule, in the process that computes it (``pipeline.run_molecule``). ``start`` opens a
    stage, ``stop`` closes it as ok, ``abort`` as failed or cancelled; the stages go to ``result.telemetry``
    {"pid", "stages": [...], "features": {...}}. ``extend=True`` adds to what is there (the export stage).
    Every method swallows its own errors."""

    def __init__(self, result, threads: int, extend: bool = False):
        self.on = enabled()
        self.result = result
        self.threads = int(threads or 1)
        self.open: dict | None = None
        if not self.on:
            return
        try:
            tel = getattr(result, "telemetry", None)
            if not extend or not isinstance(tel, dict) or "stages" not in tel:
                result.telemetry = {"pid": os.getpid(), "stages": []}
        except Exception:  # noqa: BLE001
            self.on = False

    def start(self, stage: str, engine: str = "", n_in: int | None = None, threads: int | None = None) -> None:
        if not self.on:
            return
        try:
            from . import external

            if self.open is not None:
                self._close("ok")
            self.open = {"stage": stage, "engine": engine, "n_in": n_in, "threads": int(threads or self.threads),
                         "t0": time.time(), "p0": time.perf_counter(), "cpu0": _process_cpu(),
                         "prog0": external.program_count()}
        except Exception:  # noqa: BLE001
            self.open = None

    def stop(self, n_out: int | None = None, engine: str | None = None) -> None:
        if not self.on or self.open is None:
            return
        try:
            if engine:
                self.open["engine"] = engine
            self._close("ok", n_out)
        except Exception:  # noqa: BLE001
            self.open = None

    def abort(self, outcome: str = "failed") -> None:
        if not self.on or self.open is None:
            return
        try:
            self._close(outcome)
        except Exception:  # noqa: BLE001
            self.open = None

    def features(self, mol) -> None:
        if not self.on or mol is None:
            return
        with contextlib.suppress(Exception):
            self.result.telemetry["features"] = molecule_features(mol)

    def _close(self, outcome: str, n_out: int | None = None) -> None:
        from . import external

        o, self.open = self.open, None
        progs = external.programs_since(o["prog0"])
        cpus = [p.get("cpu_s") for p in progs]
        peaks = [p["peak_rss_mb"] for p in progs if p.get("peak_rss_mb") is not None]
        self.result.telemetry["stages"].append({
            "stage": o["stage"], "engine": o["engine"], "threads": o["threads"],
            "t_start": o["t0"], "t_end": time.time(), "wall_s": time.perf_counter() - o["p0"],
            "cpu_s": max(0.0, _process_cpu() - o["cpu0"]),
            "child_cpu_s": (sum(cpus) if all(c is not None for c in cpus) else None) if progs else 0.0,
            "programs": len(progs), "program_wall_s": sum(p.get("wall_s") or 0.0 for p in progs),
            "program_threads": max((p.get("threads") or 0 for p in progs), default=0) or None,
            "program_names": sorted({p.get("name", "") for p in progs if p.get("name")}),
            "peak_rss_mb": self_peak_rss_mb(), "child_peak_rss_mb": max(peaks) if peaks else None,
            "n_in": o["n_in"], "n_out": n_out, "outcome": outcome})


@contextlib.contextmanager
def export_clock(result, threads: int):
    """Time the export of ``result``'s files as the stage "export" (in the process that writes them)."""
    clock = StageClock(result, threads, extend=True)
    clock.start("export", engine="files")
    try:
        yield clock
    except BaseException:
        clock.abort("failed")
        raise
    if getattr(result, "status", "ok") == "ok":
        clock.stop(n_out=len(getattr(result, "files", None) or []))
    else:                     # the export failed (the caller caught it and failed the molecule)
        clock.abort("failed")


# ------------------------------------------------------------------------------- the machine and run
@functools.lru_cache(maxsize=1)
def machine_info() -> dict:
    """What does not change while this process runs: versions, machine, OS, CPU, cores, RAM."""
    import platform
    import socket

    import psutil

    from .. import __version__, resources

    info: dict = {"qb_version": __version__, "hostname": socket.gethostname(), "os": platform.system(),
                  "os_release": platform.release(), "arch": platform.machine(), "python": platform.python_version(),
                  "entry": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""}
    for key, fn in (("wsl", resources.is_wsl), ("cpu_model", resources.cpu_name),
                    ("physical_cores", resources.physical_cores), ("logical_cores", resources.logical_cores),
                    ("performance_cores", resources.performance_cores),
                    ("ram_gb", lambda: round(psutil.virtual_memory().total / 1024**3, 2))):
        with contextlib.suppress(Exception):
            info[key] = fn()
    with contextlib.suppress(Exception):
        from ..dist import profile

        info["machine"] = profile.machine_name()
        info["qb_commit"] = profile.git_commit()
    return info


#: packages whose version is recorded when they are installed (no import: importlib.metadata only)
PACKAGES = ("rdkit", "numpy", "psutil", "ase", "tblite", "dxtb", "torch", "aimnet")


@functools.lru_cache(maxsize=1)
def package_versions() -> dict:
    from importlib import metadata

    out = {}
    for name in PACKAGES:
        with contextlib.suppress(Exception):
            out[name] = metadata.version(name)
    return out


def engines_info(cfg) -> dict:
    """The engines a run of ``cfg`` uses, with their versions: the programs (CREST, xtb, MOPAC; path, version,
    where found), Psi4's interpreter for a QM stage, and the Python packages (``PACKAGES``)."""
    from . import engines, external

    out: dict = {"packages": package_versions()}
    wanted = []
    if getattr(cfg, "conformer_search", "") == "crest":
        wanted.append("crest")
        if str(getattr(cfg, "crest_method", "")) == "gfn2//gfnff":
            wanted.append("xtb")
    with contextlib.suppress(Exception):
        info = engines.all_engines().get(cfg.method)
        if info is not None and info.external:
            wanted.append(info.external)
    programs = {}
    for name in dict.fromkeys(wanted):
        with contextlib.suppress(Exception):
            p = external.find_program(name)
            programs[name] = None if p is None else {"path": p.path, "version": p.version, "source": p.source,
                                                     "via_wsl": p.via_wsl}
    out["programs"] = programs
    if getattr(cfg, "rank_method", "refine") in ("auto", "gxtb", "gxtb+cpcm"):    # the rank step's g-xTB build
        with contextlib.suppress(Exception):
            from . import eng_gxtb

            b = eng_gxtb.build_info()
            programs["gxtb"] = None if b is None else {**b.to_dict(), "label": b.label()}
    if getattr(cfg, "qm_method", ""):
        with contextlib.suppress(Exception):
            out["psi4_python"] = external.find_python_with("psi4", "psi4_python")
    return out


#: BuildConfig fields that change what a molecule costs (config_json of telemetry_runs)
COST_FIELDS = ("method", "conformer_search", "crest_method", "crest_quick", "solvent", "max_conformers",
               "refine_top", "energy_window_kcal", "rank_method", "rank_window_kj", "rmsd_threshold", "fmax",
               "max_steps", "optimizer", "batched",
               "xtb_properties", "frequencies", "freq_hessian", "qm_method", "qm_task", "qm_top", "qm_properties",
               "qm_frequencies", "qm_cubes", "threads", "gpu", "dxtb_gpu", "memory_budget_gb", "molecule_timeout_s",
               "protonate", "protonation_method", "priority", "free_threads")


def priority_fields(cfg) -> dict:
    """The run priority and free threads in force, for ``config_json`` next to ``priority`` and ``free_threads``
    (``quantabench.priority``; no columns of their own, so the schema stays as it is): the free threads after the
    background floor, and the level the work runs at (a priority is never raised: a run asked 'low' in a worker at
    'background' runs at 'background')."""
    try:
        from .. import priority

        level, free = priority.of(cfg)
        return {"free_threads_effective": priority.effective_free_threads(free, level),
                "priority_effective": priority.lower_of(level, priority.current_level())}
    except Exception:  # noqa: BLE001 - recording never fails a run
        return {}


def outcome_of(res) -> str:
    status = getattr(res, "status", "") or "failed"
    error = str(getattr(res, "error", "") or "")
    if status == "ok":
        return "ok"
    if status == "cancelled":
        return "cancelled"
    if error.startswith("time limit"):
        return "timeout"
    if error.startswith("crashed the process"):
        return "crashed"
    return "failed"


# ------------------------------------------------------------------------------------- load sampler
def sensor_fields(read: Callable[[], object] | None = None) -> dict:
    """The ``SENSOR_COLUMNS`` of one sensor reading (``quantabench.sensors.read``: the effective clock is averaged
    since the previous reading, the rest is the moment's value); {} when sensors are off or the reading fails."""
    try:
        from .. import sensors

        s = (read or sensors.read)()
        out = {k: getattr(s, k, None) for k, _ in SENSOR_COLUMNS if k != "sensor_source"}
        out = {k: v for k, v in out.items() if v is not None}
        if out:
            out["sensor_source"] = str(getattr(s, "source", "") or "")[:120] or None
        return out
    except Exception:  # noqa: BLE001
        return {}


def _sensors_on() -> bool:
    try:
        from .. import sensors

        return sensors.load_config().enabled
    except Exception:  # noqa: BLE001
        return False


class LoadSampler(threading.Thread):
    """Samples the machine's load at the start (after 1 s) and every ``interval`` s, in a thread of its own; the
    samples wait in memory until the run's writer takes them (``take``). Load average on POSIX, the CPU percent
    of the whole machine (all OSes, psutil) and of this process and its children, memory in use, and the machine's
    sensors (``SENSOR_COLUMNS``: effective clock, CPU and GPU temperature and power, GPU utilization) unless they
    are off (``settings.sensors.enabled``, ``QUANTABENCH_SENSORS=0``)."""

    def __init__(self, interval: float = LOAD_INTERVAL_S, first: float = 1.0, sensors: bool | None = None):
        super().__init__(daemon=True, name="quantabench-load")
        self.interval, self.first = float(interval), float(first)
        self.t0 = time.time()
        self._samples: list[dict] = []
        self._lock = threading.Lock()
        self._halt = threading.Event()   # not _stop: Thread._stop() is a method on Python <= 3.12
        self.sensors = sensors           # None: from the settings, in the sampler's thread

    def take(self) -> list[dict]:
        with self._lock:
            out, self._samples = self._samples, []
        return out

    def stop(self) -> None:
        self._halt.set()
        if self.is_alive():
            self.join(2.0)

    def run(self) -> None:
        try:
            import psutil

            me = psutil.Process()
            ncpu = psutil.cpu_count() or 1
            psutil.cpu_percent(None)
            last = (time.monotonic(), self._own_cpu(me))
            if self.sensors is None:
                self.sensors = _sensors_on()
            wait = self.first
            while not self._halt.wait(wait):
                wait = self.interval
                now = time.monotonic()
                own = self._own_cpu(me)
                sample = {"t_unix": time.time(), "t_rel_s": time.time() - self.t0,
                          "cpu_percent": psutil.cpu_percent(None), "mem_percent": psutil.virtual_memory().percent}
                if own is not None and last[1] is not None and now > last[0]:
                    sample["own_cpu_percent"] = max(0.0, 100.0 * (own - last[1]) / ((now - last[0]) * ncpu))
                last = (now, own)
                if hasattr(os, "getloadavg"):
                    with contextlib.suppress(OSError):
                        sample["load1"], sample["load5"], sample["load15"] = os.getloadavg()
                if self.sensors and not self._halt.is_set():
                    sample.update(sensor_fields())
                with self._lock:
                    self._samples.append(sample)
        except Exception:  # noqa: BLE001, S110 - a sampler that fails just stops sampling
            pass

    @staticmethod
    def _own_cpu(me) -> float | None:
        """CPU seconds of this process and every child (lanes, their programs), with the children already reaped."""
        try:
            def total(t) -> float:
                return t.user + t.system + getattr(t, "children_user", 0.0) + getattr(t, "children_system", 0.0)

            s = total(me.cpu_times())
            for child in me.children(recursive=True):
                with contextlib.suppress(Exception):
                    s += total(child.cpu_times())
            return s
        except Exception:  # noqa: BLE001
            return None


# --------------------------------------------------------------------------------------- the writer
class NullTelemetry:
    """Recording off (or it could not start): every call does nothing."""

    run_id = ""
    cost_model = None
    recorded = 0

    def dispatched(self, *_a, **_k) -> None:
        pass

    def requeued(self, *_a, **_k) -> None:
        pass

    def end_attempt(self, *_a, **_k) -> None:
        pass

    def close(self, *_a, **_k) -> None:
        pass


class RunTelemetry:
    """The rows of one ``run_resumable`` call, written through the job table's own connection by the thread that
    owns it. Each write runs just before one of the store's own commits (``mark_running``, ``record``,
    ``record_crash``) and goes out with it; ``close`` commits what is left."""

    def __init__(self, store, run_id: str, t0: float, log: Callable[[str], None], cost_model=None,
                 sampler: LoadSampler | None = None):
        self.store = store
        self.run_id = run_id
        self.t0 = t0
        self.log = log
        self.cost_model = cost_model
        self.sampler = sampler
        self.open: dict[str, dict] = {}            # mol_id -> its attempt row, until it ends
        self.ready_at: dict[str, float] = {}       # a molecule back in the queue (a crashed lane): since when
        self.recorded = 0
        self.failed = False
        self.closed = False

    # -- the start
    @classmethod
    def start(cls, store, cfg, run_dir, *, lanes_planned, lanes_used, thread_budget: int, dynamic_budget: int = 0,
              speed_aware: bool | None = None, chunk=None, n_total: int = 0, n_todo: int = 0, n_done_before: int = 0,
              cost_model=None, log: Callable[[str], None] = lambda _m: None, sample_load: bool = True):
        """A ``RunTelemetry`` with its run row written (and committed), or a ``NullTelemetry`` when recording is
        off or the row cannot be written."""
        if not enabled():
            return NullTelemetry()
        try:
            from dataclasses import asdict

            from ..dist import lanes as lanes_mod

            t0 = time.time()
            run_id = uuid.uuid4().hex
            row = dict(machine_info())
            lanes_planned = list(lanes_planned or [])
            lanes_used = list(lanes_used or [])
            level = ""
            if getattr(cfg, "conformer_search", "") == "crest":
                from ..dist import threads as threads_mod

                level = threads_mod.crest_level(cfg)
            d = asdict(cfg)
            row.update(run_id=run_id, started=_now_iso(t0), started_unix=t0, thread_budget=int(thread_budget or 0),
                       threads_setting=int(cfg.threads or 0),
                       lanes_text=lanes_mod.describe(lanes_planned) if lanes_planned else "",
                       lanes_json=[x.to_dict() for x in lanes_planned], n_lanes=len(lanes_planned),
                       n_lanes_used=len(lanes_used), dynamic_threads=int(bool(dynamic_budget)),
                       dynamic_budget=int(dynamic_budget or 0),
                       speed_aware=None if speed_aware is None else int(bool(speed_aware)),
                       cost_model=None if cost_model is None else cost_model.to_dict(),
                       engines_json=engines_info(cfg), method=cfg.method, conformer_search=cfg.conformer_search,
                       crest_method=cfg.crest_method if level else None,
                       crest_quick=int(bool(cfg.crest_quick)) if level else None, crest_level=level or None,
                       solvent=cfg.solvent, max_conformers=cfg.max_conformers, refine_top=cfg.refine_top,
                       frequencies=int(bool(cfg.frequencies)), xtb_properties=int(bool(cfg.xtb_properties)),
                       qm_method=cfg.qm_method, config_json={**{k: d.get(k) for k in COST_FIELDS},
                                                             **priority_fields(cfg)},
                       run_folder=os.path.basename(os.path.normpath(str(run_dir))), chunk=chunk,
                       n_total=n_total, n_todo=n_todo, n_done_before=n_done_before)
            if hasattr(os, "getloadavg"):
                with contextlib.suppress(OSError):
                    row["load_start"] = os.getloadavg()[0]
            conn = store.conn
            with conn:
                # 'running' rows of an earlier call: that process ended without closing them (the caller holds
                # the run lock, so no live process is on them)
                conn.execute("UPDATE telemetry_attempts SET outcome = 'interrupted' WHERE outcome = 'running'")
                _insert(conn, "telemetry_runs", row)
            sampler = None
            if sample_load:
                sampler = LoadSampler()
                sampler.start()
            return cls(store, run_id, t0, log, cost_model, sampler)
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                log(f"Timings: not recorded for this run ({type(exc).__name__}: {exc})")
            with contextlib.suppress(Exception):
                store.conn.rollback()
            return NullTelemetry()

    def _fail(self, exc: BaseException, what: str) -> None:
        if not self.failed:
            self.failed = True
            with contextlib.suppress(Exception):
                self.log(f"Timings: could not record {what} ({type(exc).__name__}: {exc}); the run goes on")
        with contextlib.suppress(Exception):
            self.store.conn.rollback()        # only telemetry rows can be pending here (the store commits at once)

    def _flush_load(self) -> None:
        if self.sampler is None:
            return
        for s in self.sampler.take():
            _insert(self.store.conn, "telemetry_load", {"run_id": self.run_id, **s})

    # -- molecules
    def requeued(self, mol_id: str) -> None:
        self.ready_at[mol_id] = time.time()

    def dispatched(self, mol_id: str, inp=None, *, lane=None, lane_name: str = "main", lane_device: str = "",
                   lane_threads: int = 1, width: int | None = None, size: float | None = None, k: int | None = None,
                   predicted_s: float | None = None, pid: int | None = None) -> None:
        """``mol_id`` goes to a lane now (call just before ``store.mark_running``, whose commit takes the row).
        ``width``: the threads its CREST call may use (CREST runs only)."""
        try:
            if lane is not None:
                lane_name, lane_device, lane_threads = lane.name, lane.device, lane.threads
            row = self.store.conn.execute("SELECT attempts FROM molecules WHERE id = ?", (mol_id,)).fetchone()
            attempt = (int(row[0]) if row and row[0] is not None else 0) + 1
            if size is None and inp is not None:
                from ..dist import speed

                size = speed.molecule_size(inp)
            now = time.time()
            ready = max(self.t0, self.ready_at.pop(mol_id, self.t0))
            a = {"run_id": self.run_id, "mol_id": mol_id, "attempt": attempt, "lane": lane_name,
                 "lane_device": lane_device, "lane_threads": int(lane_threads or 1), "pid": pid or os.getpid(),
                 "dispatch_k": k, "ready_unix": ready, "dispatched_unix": now, "queue_wait_s": max(0.0, now - ready),
                 "outcome": "running", "size": size}
            if width:
                base = int(lane_threads or 1)
                a["crest_threads"] = int(width)
                a["threads_borrowed"] = max(0, int(width) - base)
                m = self.cost_model
                if m is not None and size is not None:
                    from ..dist import threads as threads_mod

                    a["predicted_cost"] = m.cost(size) / threads_mod.speedup(width, base, m.speedup_power)
                    a["predicted_s"] = m.seconds(size, width) if m.scale_s else predicted_s
            self.open[mol_id] = a
            _insert(self.store.conn, "telemetry_attempts", a)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc, f"the start of {mol_id}")

    def end_attempt(self, mol_id: str, res=None, *, outcome: str | None = None, error: str = "", stage: str = "",
                    stage_since: float | None = None) -> None:
        """``mol_id``'s attempt ended (call just before the store's commit for it: ``record`` or ``record_crash``).
        ``res``: its result (with ``res.telemetry`` from the process that ran it); without one (a lane that died
        or was killed) the row says ``outcome`` and the last ``stage`` the parent saw, since ``stage_since``.
        A second call for the same attempt does nothing."""
        a = self.open.pop(mol_id, None)
        if a is None:
            return
        try:
            now = time.time()
            tel = getattr(res, "telemetry", None) if res is not None else None
            tel = tel if isinstance(tel, dict) else {}
            outcome = outcome or (outcome_of(res) if res is not None else "failed")
            stages = [dict(s) for s in tel.get("stages") or [] if isinstance(s, dict)]
            if not stages and stage and stage not in ("start", "queued"):
                t_start = stage_since or a["dispatched_unix"]
                stages = [{"stage": stage, "t_start": t_start, "t_end": now, "wall_s": max(0.0, now - t_start),
                           "threads": a["lane_threads"],
                           "program_threads": a.get("crest_threads") if stage == CREST_STAGE else None,
                           "outcome": outcome}]
            elif stages and outcome in ("timeout", "crashed") and stages[-1].get("outcome") == "cancelled":
                stages[-1]["outcome"] = outcome        # the stage the time limit stopped
            err = error or (str(getattr(res, "error", "") or "") if res is not None else "")
            a.update(ended_unix=now, wall_s=max(0.0, now - a["dispatched_unix"]), outcome=outcome,
                     error=err[:ERROR_CHARS], last_stage=stages[-1]["stage"] if stages else (stage or None))
            if tel.get("pid"):
                a["pid"] = tel["pid"]
            if res is not None:
                a.update(method=getattr(res, "method", "") or None, device=getattr(res, "device", "") or None,
                         n_conformers=len(getattr(res, "conformers", None) or []))
            for key, value in (tel.get("features") or {}).items():
                if key in FEATURES:
                    a[key] = value
            cpus = [s.get("cpu_s") for s in stages]
            kids = [s.get("child_cpu_s") for s in stages]
            peaks = [x for s in stages for x in (s.get("peak_rss_mb"), s.get("child_peak_rss_mb")) if x is not None]
            a["cpu_s"] = sum(cpus) if stages and all(c is not None for c in cpus) else None
            a["child_cpu_s"] = sum(kids) if stages and all(c is not None for c in kids) else None
            a["peak_rss_mb"] = max(peaks) if peaks else None
            conn = self.store.conn
            _insert(conn, "telemetry_attempts", a)
            for seq, s in enumerate(stages):
                _insert(conn, "telemetry_stages", self._stage_row(a, s, seq))
            self._flush_load()
            self.recorded += 1
        except Exception as exc:  # noqa: BLE001
            self._fail(exc, f"the timings of {mol_id}")

    def _stage_row(self, a: dict, s: dict, seq: int) -> dict:
        planned = int(a.get("lane_threads") or 1)
        given = int(s.get("program_threads") or s.get("threads") or planned)
        row = {"run_id": self.run_id, "mol_id": a["mol_id"], "attempt": a["attempt"], "stage": s.get("stage"),
               "seq": seq, "lane": a.get("lane"), "engine": s.get("engine") or None, "threads_planned": planned,
               "threads_given": given, "threads_borrowed": max(0, given - planned),
               "t_start_unix": s.get("t_start"), "t_end_unix": s.get("t_end"), "wall_s": s.get("wall_s"),
               "cpu_s": s.get("cpu_s"), "child_cpu_s": s.get("child_cpu_s"), "programs": s.get("programs"),
               "program_wall_s": s.get("program_wall_s"), "peak_rss_mb": s.get("peak_rss_mb"),
               "child_peak_rss_mb": s.get("child_peak_rss_mb"), "outcome": s.get("outcome"), "n_in": s.get("n_in"),
               "n_out": s.get("n_out")}
        if s.get("stage") == CREST_STAGE and a.get("crest_threads"):
            row["predicted_s"], row["predicted_cost"] = a.get("predicted_s"), a.get("predicted_cost")
        return row

    # -- the end
    def close(self, counts=None) -> None:
        """End of the call: open attempts (a stop or an exception left them) become 'interrupted', the run row gets
        its end (``counts``: the store's status counts, or the store to ask), and everything is committed. Stops
        the load sampler."""
        if self.closed:
            return
        self.closed = True
        try:
            if self.sampler is not None:
                self.sampler.stop()
            for mol_id in list(self.open):
                self.end_attempt(mol_id, None, outcome="interrupted")
            now = time.time()
            if hasattr(counts, "counts"):
                counts = counts.counts()
            counts = counts or {}
            conn = self.store.conn
            with conn:
                self._flush_load()
                conn.execute("UPDATE telemetry_runs SET ended = ?, ended_unix = ?, wall_s = ?, n_ok = ?, n_failed = ?, "
                             "n_attempts = ? WHERE run_id = ?",
                             (_now_iso(now), now, now - self.t0, counts.get("ok"), counts.get("failed"),
                              self.recorded, self.run_id))
        except Exception as exc:  # noqa: BLE001
            self._fail(exc, "the end of the run")


def start_run(store, cfg, run_dir, **kw):
    """``RunTelemetry.start`` (see there); never raises."""
    return RunTelemetry.start(store, cfg, run_dir, **kw)
