"""Resumable, crash-safe batch runs backed by a small SQLite job table.

An overnight run of a few thousand compounds must survive a crash, a Windows Update reboot or a
closed laptop lid, and must be splittable across machines. ``pipeline.run`` keeps its results in
memory and writes the batch tables at the end, so an interruption loses the tables (the per-molecule
folders survive but nothing records which molecules are done). ``run_resumable`` wraps the same
per-molecule step (``pipeline.run_molecule`` + ``export.write_molecule``) with a job table:

    <run_dir>/jobs.sqlite      run(config, version, created, chunk) + molecules(id, status, ...)
    <run_dir>/jobs.lock        host + pid of the process working on the run (removed on exit)
    <run_dir>/<id>/...         per-molecule files, exactly as pipeline.run writes them
    <run_dir>/summary.csv      rebuilt from the job table, so a resumed run has complete tables
    <run_dir>/descriptors.csv

Each molecule's result is committed in one transaction right after it finishes (WAL journal,
``synchronous=FULL`` so a power cut cannot lose a committed molecule). Rows left ``running`` by a
crash, and rows ``cancelled`` by a stop request, are simply done again on the next call.

Splitting across machines: ``chunk=(i, n)`` (0-based ``i``) selects the molecules whose id hashes
to ``i`` modulo ``n``. The hash depends only on the id, not on row order, so every machine computes
the same partition even if the list is re-sorted or extended. Give each machine its own run_dir
(SQLite must not be written by two machines through a synced folder) and combine the tables with
``merge_tables``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import sqlite3
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from rdkit import Chem

from .. import __version__, resources
from . import export, pipeline
from .inputs import MoleculeInput, dedupe_ids
from .pipeline import BuildConfig, MoleculeResult, ProgressFn

DB_NAME = "jobs.sqlite"
LOCK_NAME = "jobs.lock"
SCHEMA_VERSION = 2                 # 2: molecules.crashes
STATUSES = ("pending", "running", "ok", "failed", "cancelled")
MAX_CRASHES = 2                    # a molecule that killed the process this often is marked failed

# BuildConfig fields that change the numbers in the results. Resuming with a different value would
# silently mix two protocols in one table, so it is refused unless forced. Threads, GPU, output
# formats and the output folder only change speed or which files are written.
SCIENCE_FIELDS = (
    "method", "ph", "protonate", "protonation_method", "keep_largest_fragment", "max_conformers",
    "refine_top", "energy_window_kcal", "rmsd_threshold", "fmax", "max_steps", "optimizer",
    "temperature_k", "xtb_properties", "seed", "frequencies", "freq_hessian", "symmetry_number", "freq_qrrho",
)

Chunk = tuple[int, int]


def _noop(*_args) -> None:
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_default(obj):
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


class ConfigMismatchError(ValueError):
    """The run folder was started with different science settings."""

    def __init__(self, differences: dict[str, tuple], run_dir: Path):
        self.differences = differences
        lines = ", ".join(f"{k}: stored {a!r} vs now {b!r}" for k, (a, b) in differences.items())
        super().__init__(f"{run_dir} was started with different settings ({lines}). Resuming would mix two "
                         "protocols in one table: use the stored settings, start a new run folder, or pass "
                         "force=True to continue anyway.")


class RunInUseError(RuntimeError):
    """Another live process is working on this run folder."""


# ----------------------------------------------------------------------------------------- chunks
def chunk_index(mol_id: str, n: int) -> int:
    """0-based chunk that ``mol_id`` belongs to when a list is split ``n`` ways.

    SHA-256 of the id rather than Python's ``hash`` (salted per process) or the row position (changes
    when the list is re-sorted), so every machine and every Python version agrees.
    """
    digest = hashlib.sha256(mol_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n


def check_chunk(chunk: Chunk | None) -> Chunk | None:
    if chunk is None:
        return None
    i, n = int(chunk[0]), int(chunk[1])
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"chunk must be (i, n) with 0 <= i < n (0-based), got {chunk!r}")
    return (i, n)


def parse_chunk(text: str) -> Chunk:
    """Parse the human, 1-based form used on the command line: "2/3" -> (1, 3)."""
    try:
        k, n = (int(x) for x in text.replace("of", "/").split("/"))
    except ValueError:
        raise ValueError(f"chunk must look like 'k/n' (1-based, e.g. 2/3), got {text!r}") from None
    if not 1 <= k <= n:
        raise ValueError(f"chunk {text!r}: need 1 <= k <= n")
    return (k - 1, n)


def chunk_label(chunk: Chunk | None) -> str:
    """'chunk2of3' (1-based) for folder names; '' for an unsplit run."""
    return "" if chunk is None else f"chunk{chunk[0] + 1}of{chunk[1]}"


def in_chunk(mol_id: str, chunk: Chunk | None) -> bool:
    return chunk is None or chunk_index(mol_id, chunk[1]) == chunk[0]


# ------------------------------------------------------------------------------ input round trip
def input_to_json(inp: MoleculeInput) -> str:
    d = {k: v for k, v in asdict(inp).items() if k != "mol"}
    if inp.mol is not None:  # structure files: keep coordinates, explicit H and formal charges
        d["molblock"] = Chem.MolToMolBlock(inp.mol)
    return _dumps(d)


def input_from_json(text: str) -> MoleculeInput:
    d = json.loads(text)
    molblock = d.pop("molblock", None)
    known = {f.name for f in fields(MoleculeInput)}
    inp = MoleculeInput(**{k: v for k, v in d.items() if k in known})
    if molblock:
        inp.mol = Chem.MolFromMolBlock(molblock, removeHs=False)
    return inp


def _structure_key(d: dict) -> str:
    return d.get("molblock") or d.get("smiles", "")


# ------------------------------------------------------------------------------------- the store
class JobStore:
    """The job table of one run folder. One connection per thread: open it where it is used."""

    def __init__(self, run_dir: str | Path, create: bool = True, read_only: bool = False,
                 immutable: bool = False):
        """``read_only``: open with ``mode=ro`` and no PRAGMAs, so the database file and its -wal are left
        exactly as they are (a writable connection checkpoints on close and deletes -wal/-shm, which a synced
        copy would send back to the machine still writing it). ``immutable``: also skip SQLite's locking and
        the WAL, for a copy of a database that another machine is writing."""
        self.run_dir = Path(run_dir).expanduser()
        self.path = self.run_dir / DB_NAME
        self.read_only = read_only or immutable
        if (not create or self.read_only) and not self.path.exists():
            raise FileNotFoundError(f"No {DB_NAME} in {self.run_dir}: not a resumable run folder")
        if self.read_only:
            uri = self.path.resolve().as_uri() + ("?immutable=1" if immutable else "?mode=ro")
            self.conn = sqlite3.connect(uri, uri=True, timeout=30.0)
            self.conn.row_factory = sqlite3.Row
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        # FULL: a committed molecule survives a power cut, not just a process crash. One fsync per
        # molecule is nothing next to minutes of conformer search.
        self.conn.execute("PRAGMA synchronous=FULL")
        if create:
            self._create_schema()

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS run (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                config TEXT NOT NULL, version TEXT NOT NULL, created TEXT NOT NULL,
                chunk TEXT, updated TEXT)""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS molecules (
                id TEXT PRIMARY KEY, seq INTEGER NOT NULL, input TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', error TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0, started TEXT, finished TEXT,
                result_dir TEXT NOT NULL DEFAULT '', summary TEXT)""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS molecules_status ON molecules(status)")
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(molecules)")}
            if "crashes" not in cols:     # version 1 tables
                self.conn.execute("ALTER TABLE molecules ADD COLUMN crashes INTEGER NOT NULL DEFAULT 0")
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()  # the last connection checkpoints and removes the -wal file

    def __enter__(self) -> JobStore:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- run record
    def get_run(self) -> dict | None:
        row = self.conn.execute("SELECT * FROM run WHERE id = 1").fetchone()
        if row is None:
            return None
        chunk = json.loads(row["chunk"]) if row["chunk"] else None
        return {"config": json.loads(row["config"]), "version": row["version"], "created": row["created"],
                "updated": row["updated"], "chunk": tuple(chunk) if chunk else None}

    def init_run(self, cfg: BuildConfig, chunk: Chunk | None) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO run (id, config, version, created, chunk, updated) VALUES (1, ?, ?, ?, ?, ?)",
                              (_dumps(asdict(cfg)), __version__, _now(), _dumps(chunk) if chunk else None, _now()))

    def update_run(self, cfg: BuildConfig, chunk: Chunk | None) -> None:
        with self.conn:
            self.conn.execute("UPDATE run SET config = ?, chunk = ?, updated = ? WHERE id = 1",
                              (_dumps(asdict(cfg)), _dumps(chunk) if chunk else None, _now()))

    # -- molecules
    def add_inputs(self, inputs: Iterable[MoleculeInput], log: Callable[[str], None] = _noop) -> int:
        """Register inputs. New ids become pending. A known id keeps its row; if its structure changed,
        an unfinished row takes the new input, while a finished one is kept (and reported)."""
        added = 0
        with self.conn:
            seq = self.conn.execute("SELECT COALESCE(MAX(seq), -1) FROM molecules").fetchone()[0]
            for inp in inputs:
                text = input_to_json(inp)
                row = self.conn.execute("SELECT input, status FROM molecules WHERE id = ?", (inp.id,)).fetchone()
                if row is None:
                    seq += 1
                    self.conn.execute("INSERT INTO molecules (id, seq, input) VALUES (?, ?, ?)", (inp.id, seq, text))
                    added += 1
                elif _structure_key(json.loads(row["input"])) != _structure_key(json.loads(text)):
                    if row["status"] in ("ok", "failed"):
                        log(f"{inp.id}: input structure differs from the finished record; keeping the "
                            "stored result (use a new run folder to redo it)")
                    else:
                        self.conn.execute("UPDATE molecules SET input = ? WHERE id = ?", (text, inp.id))
        return added

    def reset_interrupted(self, max_crashes: int = MAX_CRASHES) -> tuple[int, list[str]]:
        """'running' rows can only be left by a crash (the caller holds the run lock). Count the crash; a
        molecule that has now crashed the process ``max_crashes`` times (a segfault in a native library,
        an OOM kill) is marked failed, so it cannot block every later resume (``retry_failed`` redoes it).
        The others become pending. Returns (rows made pending, ids marked failed)."""
        with self.conn:
            self.conn.execute("UPDATE molecules SET crashes = crashes + 1 WHERE status = 'running'")
            failed = [r["id"] for r in self.conn.execute(
                "SELECT id FROM molecules WHERE status = 'running' AND crashes >= ?", (max_crashes,))]
            self.conn.execute("UPDATE molecules SET status = 'failed', finished = ?, "
                              "error = 'crashed the process ' || crashes || ' times' "
                              "WHERE status = 'running' AND crashes >= ?", (_now(), max_crashes))
            n = self.conn.execute("UPDATE molecules SET status = 'pending' WHERE status = 'running'").rowcount
        return n, failed

    def todo(self, retry_failed: bool = False) -> list[tuple[str, MoleculeInput]]:
        """Molecules still to do, in input order, except that molecules which crashed the process go last."""
        wanted = ("pending", "running", "cancelled") + (("failed",) if retry_failed else ())
        marks = ",".join("?" * len(wanted))
        rows = self.conn.execute(f"SELECT id, input FROM molecules WHERE status IN ({marks}) "  # noqa: S608
                                 "ORDER BY crashes, seq", wanted).fetchall()
        return [(r["id"], input_from_json(r["input"])) for r in rows]

    def mark_running(self, mol_id: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE molecules SET status = 'running', attempts = attempts + 1, started = ?, "
                              "error = '' WHERE id = ?", (_now(), mol_id))

    def record(self, result: MoleculeResult, result_dir: str = "") -> None:
        """Store one finished molecule in a single transaction."""
        summary = {"row": result.summary_row(), "descriptors": result.descriptors}
        with self.conn:
            self.conn.execute("UPDATE molecules SET status = ?, error = ?, finished = ?, result_dir = ?, summary = ? "
                              "WHERE id = ?", (result.status, result.error, _now(), result_dir,
                                               _dumps(summary) if result.status != "cancelled" else None,
                                               result.input.id))

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(STATUSES, 0)
        for r in self.conn.execute("SELECT status, COUNT(*) AS n FROM molecules GROUP BY status"):
            out[r["status"]] = r["n"]
        return out

    def failures(self) -> list[tuple[str, str]]:
        return [(r["id"], r["error"]) for r in
                self.conn.execute("SELECT id, error FROM molecules WHERE status = 'failed' ORDER BY seq")]

    def stored_results(self, statuses: Sequence[str] = ("ok", "failed")) -> list[SimpleNamespace]:
        """Finished molecules as light stand-ins for ``MoleculeResult`` that ``export.write_batch_tables``
        accepts (it only needs ``summary_row()``, ``descriptors`` and ``input.id``)."""
        marks = ",".join("?" * len(statuses))
        rows = self.conn.execute(f"SELECT id, status, summary FROM molecules WHERE status IN ({marks}) "  # noqa: S608
                                 "AND summary IS NOT NULL ORDER BY seq", tuple(statuses)).fetchall()
        out = []
        for r in rows:
            s = json.loads(r["summary"])
            row = s["row"]
            out.append(SimpleNamespace(input=SimpleNamespace(id=r["id"]), status=r["status"],
                                       descriptors=s.get("descriptors") or {},
                                       summary_row=lambda row=row: dict(row)))
        return out


# ------------------------------------------------------------------------------------- run lock
_held_locks: set[str] = set()
_held_guard = threading.Lock()


def _process_start(pid: int) -> float | None:
    import psutil

    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def _read_lock(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / LOCK_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _platform_id() -> str:
    """Which OS instance holds a lock. WSL2 takes the Windows computer name as its hostname but has its own
    process ids, so host + pid alone cannot tell a live Windows run from a dead WSL one (or vice versa)."""
    release = platform.uname().release.lower()
    return f"{platform.system()}{'-wsl' if 'microsoft' in release or 'wsl' in release else ''}"


def _lock_alive(info: dict | None) -> bool | None:
    """True/False when the owner can be checked (same host and OS), None for another machine (or for
    Windows vs WSL2 on one machine), which then needs ``force``."""
    if not info:
        return False
    if info.get("host") != socket.gethostname():
        return None
    if info.get("platform", _platform_id()) != _platform_id():
        return None
    pid = int(info.get("pid", -1))
    if pid == os.getpid():
        return info.get("key") in _held_locks
    start = _process_start(pid)
    # compare start times too: after a reboot the pid may belong to an unrelated process
    return start is not None and abs(start - float(info.get("pid_started", 0))) < 1.0


class _RunLock:
    """Keeps two processes (say, the GUI and a CLI resume) from working on one run folder at once."""

    def __init__(self, run_dir: Path, force: bool, log: Callable[[str], None]):
        self.run_dir, self.force, self.log = run_dir, force, log
        self.key = str(run_dir.resolve())
        self.path = run_dir / LOCK_NAME

    def __enter__(self) -> _RunLock:
        with _held_guard:
            if self.key in _held_locks:
                raise RunInUseError(f"{self.run_dir} is already being run by this application")
            info = _read_lock(self.run_dir)
            alive = _lock_alive(info)
            if alive:
                raise RunInUseError(f"{self.run_dir} is being run by process {info.get('pid')} on this machine "
                                    f"since {info.get('since')}")
            if alive is None and info:
                if not self.force:
                    raise RunInUseError(f"{self.run_dir} is locked by {info.get('host')} "
                                        f"({info.get('platform', '?')}, pid {info.get('pid')}, "
                                        f"since {info.get('since')}). If that machine is no longer running it, "
                                        f"pass force=True or delete {LOCK_NAME}.")
                self.log(f"Taking over the lock held by {info.get('host')} (force)")
            elif info:
                self.log(f"Previous run (pid {info.get('pid')}) ended without cleaning up; resuming")
            pid = os.getpid()
            self.path.write_text(_dumps({"host": socket.gethostname(), "platform": _platform_id(),
                                         "pid": pid, "key": self.key,
                                         "pid_started": _process_start(pid) or 0.0, "since": _now()}),
                                 encoding="utf-8")
            _held_locks.add(self.key)
        return self

    def __exit__(self, *_exc) -> None:
        with _held_guard:
            _held_locks.discard(self.key)
            self.path.unlink(missing_ok=True)


# -------------------------------------------------------------------------------------- running
def config_differences(stored: dict, cfg: BuildConfig) -> dict[str, tuple]:
    """Science fields whose stored value differs from ``cfg`` (missing fields compare to the default)."""
    defaults = BuildConfig()
    diffs = {}
    for name in SCIENCE_FIELDS:
        old = stored.get(name, getattr(defaults, name))
        new = getattr(cfg, name)
        if old != new:
            diffs[name] = (old, new)
    return diffs


def _prepare_run(store: JobStore, cfg: BuildConfig, chunk: Chunk | None, force: bool,
                 log: Callable[[str], None]) -> Chunk | None:
    """Create or check the run record; returns the chunk in force (a resume without a chunk keeps
    the stored one)."""
    run = store.get_run()
    if run is None:
        store.init_run(cfg, chunk)
        return chunk
    if chunk is None:
        chunk = run["chunk"]
    diffs = config_differences(run["config"], cfg)
    if run["chunk"] != chunk:
        diffs["chunk"] = (run["chunk"], chunk)
    if diffs and not force:
        raise ConfigMismatchError(diffs, store.run_dir)
    if diffs:
        log("Continuing with changed settings (force): " + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in diffs.items()))
    if run["version"] != __version__:
        log(f"Run started with QuantaBench {run['version']}, continuing with {__version__}")
    store.update_run(cfg, chunk)
    return chunk


def rebuild_tables(run_dir: str | Path) -> list[Path]:
    """Rewrite summary.csv (every finished molecule, ok and failed, in input order) and
    descriptors.csv (ok molecules) from the job table."""
    with JobStore(run_dir, read_only=True) as store:
        stored = store.stored_results()
    return export.write_batch_tables(stored, Path(run_dir)) if stored else []


def run_resumable(inputs: Sequence[MoleculeInput] | None, cfg: BuildConfig, run_dir: str | Path,
                  progress: ProgressFn = _noop, should_stop: Callable[[], bool] = lambda: False,
                  log: Callable[[str], None] = _noop,
                  on_result: Callable[[MoleculeResult], None] = _noop, retry_failed: bool = False,
                  chunk: Chunk | None = None, force: bool = False) -> tuple[list[MoleculeResult], Path]:
    """Run (or continue) a batch in ``run_dir``; returns the results computed by this call.

    ``inputs`` are added to the job table (``None`` resumes with what is stored). Finished molecules
    are skipped; ``retry_failed`` also redoes failed ones. ``chunk=(i, n)``, 0-based, keeps only the
    molecules with ``chunk_index(id, n) == i``. Callbacks follow ``pipeline.run``; ``progress`` counts
    over all molecules of this run folder, including those finished earlier.

    Raises ``ConfigMismatchError`` if the folder was started with different science settings (or
    another chunk) unless ``force``, and ``RunInUseError`` if another process is working on it.
    """
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    chunk = check_chunk(chunk)
    results: list[MoleculeResult] = []
    with _RunLock(run_dir, force, log), JobStore(run_dir) as store:
        chunk = _prepare_run(store, cfg, chunk, force, log)
        if cfg.method == "auto" and chunk is not None:
            log("Note: method 'auto' picks the engine per machine; when splitting a list across machines, "
                "set the method explicitly so all chunks use the same level of theory.")
        if inputs:
            warnings: list[str] = []
            selected = [x for x in dedupe_ids(inputs, warnings) if in_chunk(x.id, chunk)]
            for w in warnings:
                log(w)
            added = store.add_inputs(selected, log)
            if chunk is not None:
                log(f"{chunk_label(chunk)}: {len(selected)} of {len(inputs)} input molecules belong to this chunk")
            if added:
                log(f"Registered {added} new molecule(s)")
        n_reset, crashed = store.reset_interrupted()
        if n_reset:
            log(f"Resuming {n_reset} molecule(s) interrupted by a crash (they run last)")
        for mol_id in crashed:
            log(f"{mol_id}: crashed the process {MAX_CRASHES} times; marked failed (--retry-failed redoes it)")

        threads = resources.apply_thread_limits(cfg.threads)
        device = resources.select_device(cfg.gpu, cfg.gpu_index, cfg.gpu_memory_fraction)
        log(f"Resources: {threads} CPU threads, device {device}")

        todo = store.todo(retry_failed)
        total = sum(store.counts().values())
        done_before = total - len(todo)
        if done_before:
            log(f"{done_before} of {total} molecule(s) already finished; {len(todo)} to do")
        for k, (mol_id, inp) in enumerate(todo):
            if should_stop():
                break
            i = done_before + k
            progress(i, total, "start", mol_id)
            store.mark_running(mol_id)
            res = pipeline.run_molecule(inp, cfg, device, threads, should_stop, log,
                                        stage=lambda s, i=i, mol_id=mol_id: progress(i, total, s, mol_id))
            result_dir = ""
            if cfg.write_files and res.status == "ok":
                try:
                    files = export.write_molecule(res, run_dir / mol_id, cfg.formats)
                    res.files = [str(p) for p in files]
                    result_dir = mol_id  # relative, so the folder can move between machines
                except Exception as exc:  # a disk or format error must not end an overnight run
                    res.status, res.error = "failed", f"export failed: {type(exc).__name__}: {exc}"
                    log(f"  {mol_id}: {res.error}")
            store.record(res, result_dir)
            results.append(res)
            on_result(res)
            progress(i + 1, total, res.status, mol_id)
        counts = store.counts()
    if cfg.write_files:
        try:
            rebuild_tables(run_dir)
        except OSError as exc:  # e.g. summary.csv open in Excel on Windows
            log(f"Could not rewrite the batch tables ({exc}); close them and call rebuild_tables")
    log(f"{counts['ok']} ok, {counts['failed']} failed, "
        f"{counts['pending'] + counts['running'] + counts['cancelled']} remaining. Results in {run_dir}")
    return results, run_dir


# --------------------------------------------------------------------------------------- status
@dataclass
class RunStatus:
    run_dir: Path
    counts: dict[str, int]
    config: dict
    version: str
    created: str
    chunk: Chunk | None
    active: bool                       # a live process holds the run lock
    interrupted: int = 0               # 'running' rows without a live process, counted as pending
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def finished(self) -> int:
        return self.counts["ok"] + self.counts["failed"]

    @property
    def remaining(self) -> int:
        return self.total - self.finished

    def summary(self) -> str:
        parts = [f"{self.counts['ok']} ok", f"{self.counts['failed']} failed", f"{self.remaining} remaining"]
        if self.interrupted:
            parts.append(f"{self.interrupted} interrupted")
        where = f" ({chunk_label(self.chunk)})" if self.chunk else ""
        state = "running" if self.active else "idle"
        return f"{self.finished}/{self.total} done{where}, {state}: " + ", ".join(parts)


def open_run(run_dir: str | Path) -> RunStatus:
    """Status of a run folder for the GUI and CLI. Read-only: the database and its -wal are opened with
    ``mode=ro`` and never checkpointed, so checking a run that synced over from another machine changes
    nothing that SyncThing would send back. When the lock names another machine, the copy is read as
    immutable (no locking, WAL ignored), so the counts may lag that machine by its last few molecules."""
    run_dir = Path(run_dir).expanduser()
    active = _lock_alive(_read_lock(run_dir))
    with JobStore(run_dir, read_only=True, immutable=active is None and _read_lock(run_dir) is not None) as store:
        run = store.get_run()
        if run is None:
            raise ValueError(f"{run_dir / DB_NAME} has no run record")
        counts = store.counts()
        failures = store.failures()
    interrupted = 0
    if active is False and counts["running"]:  # unknown host (None): trust the table as it is
        interrupted = counts["running"]
        counts["pending"] += counts["running"]
        counts["running"] = 0
    return RunStatus(run_dir=run_dir, counts=counts, config=run["config"], version=run["version"],
                     created=run["created"], chunk=run["chunk"], active=bool(active), interrupted=interrupted,
                     failures=failures)


def merge_tables(run_dirs: Iterable[str | Path], out_dir: str | Path,
                 log: Callable[[str], None] = _noop) -> list[Path]:
    """Combine summary.csv and descriptors.csv of several run folders (e.g. one per machine/chunk).

    Reads the CSV tables, which ``run_resumable`` rewrites when it ends. A folder without them (a chunk
    that crashed or is still running) is read from its job table instead, opened read-only, and reported
    through ``log``, as is a folder that does not exist. If an id appears twice, an 'ok' row wins over a
    failed one.
    """
    run_dirs = [Path(d).expanduser() for d in run_dirs]
    out_dir = Path(out_dir).expanduser()
    tables: dict[str, list[dict]] = {"summary.csv": [], "descriptors.csv": []}
    keys: dict[str, list[str]] = {"summary.csv": [], "descriptors.csv": []}
    for d in run_dirs:
        if not d.is_dir():
            log(f"{d}: no such folder; skipped")
            continue
        if not (d / "summary.csv").exists():
            if not (d / DB_NAME).exists():
                log(f"{d}: no summary.csv and no {DB_NAME}; skipped")
                continue
            try:
                with JobStore(d, read_only=True) as store:
                    stored = store.stored_results()
            except sqlite3.Error as exc:
                log(f"{d}: no summary.csv, and {DB_NAME} could not be read ({exc}); skipped")
                continue
            log(f"{d}: no summary.csv (interrupted or still running?); used the {len(stored)} finished "
                f"molecule(s) in {DB_NAME}. `quantabench-cli build --resume {d}` finishes it.")
            for r in stored:
                row = {k: "" if v is None else v for k, v in r.summary_row().items()}
                tables["summary.csv"].append(row)
                keys["summary.csv"] += [k for k in row if k not in keys["summary.csv"]]
                if r.descriptors:
                    drow = {"id": r.input.id, **r.descriptors}
                    tables["descriptors.csv"].append(drow)
                    keys["descriptors.csv"] += [k for k in drow if k not in keys["descriptors.csv"]]
            continue
        for name in tables:
            path = d / name
            if not path.exists():
                continue
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                keys[name] += [k for k in reader.fieldnames or [] if k not in keys[name]]
                tables[name] += list(reader)
    written = []
    for name, rows in tables.items():
        merged: dict[str, dict] = {}
        for row in rows:
            old = merged.get(row["id"])
            if old is None or old.get("status") != "ok":
                merged[row["id"]] = row
        if not merged:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys[name])
            w.writeheader()
            w.writerows(merged.values())
        written.append(path)
    return written
