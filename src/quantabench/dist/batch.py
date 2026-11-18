"""``quantabench-cli batch``: staged batches with phases, attempts, a deadline and resume (BATCH-1).

    quantabench-cli batch run [MOLECULES ...] --file IN.csv [--phases gas,water] [--run-dir DIR]
        [--deadline 07:00|ISO|none] [--machines auto|local|NAME,...] [--include-paused] [--max-attempts 4]
        [--no-pair] [the science options of `grid build`]
    quantabench-cli batch run --list batches.txt [--root DIR] ...     several batches, one after the other
    quantabench-cli batch status RUN_DIR

These were the QSAR runs' scripts on the workstation (``scratch/batches/run_tranche.py`` and ``qb_grid.py``, Nov 11-18:
15 tranches of 1,000 molecules and the CREST rebuild), now a command with tests.

A **batch** is one molecule list run as phases (``gas``, then each solvent: ``--phases gas,water``) into its own run
folder (``--run-dir``; default ``<output folder>/<date>_<label>``). Each phase runs as one or more **attempts**; an
attempt is a ``grid build --wait`` (``client.grid_build``, in this process) over the molecules that no run folder of
the phase has finished yet (``finished_ids``: ok or failed, from job tables, merged tables and snapshots):

- **Machines.** Before each attempt the workers are asked (``ready_machines``): with ``--machines auto`` each one
  that answers, is not busy with a hold, runs this QuantaBench version and this git commit joins, plus this machine;
  a named machine that is not ready stops the batch (exit 4). ``--include-paused``: a paused machine joins too (the
  grid moves its unstarted molecules to the others at once and work sharing gives it work after its resume: the
  MacBook's 02:00-08:00 window). A machine that pauses or stops answering during an attempt is the grid's business
  (DIST-7: its unstarted molecules move, ``--wait`` does not hang on it).
- **Deadline** (``--deadline``: ``HH:MM`` = the next one, an ISO time, or ``none``). At the deadline the attempt
  stops following, the local share stops at its next molecule (resumable), and the remote shares are cancelled (they
  stop after their current molecule; their run folders keep what is done). The batch ends with exit 5. Run the same
  command again to resume: only unfinished molecules are run; finished ones are never recomputed.
- **Snapshots.** After an attempt that did not merge, and before each new attempt, what the other machines finished
  in their synced run folders (``<output folder>/<run folder>``) is copied into the phase folder as
  ``partial_<run folder>`` (``private_snapshot``: a private copy of the job table, so SyncThing gets no -wal file).
  The remote shares are read from every ``grid.json`` of the phase, so an attempt that was killed still counts: on
  2026-11-18 an attempt killed before the runner recorded its remote shares left 337 finished molecules uncounted
  and the next attempt queued them again.
- **Attempts.** An attempt that merges ends the phase unless molecules are still without a result (a machine that
  went away with them): then the next attempt runs those, up to ``--max-attempts`` (default 4; then exit 1).
- **Batch lists.** ``--list FILE`` (one input file per line, ``#`` comments): each to ``<root>/<file stem>``, in
  order; a deadline or a failure ends the list.
- **After the phases:** the gas phase is paired with the others (``phases.write_paired``: ``paired.csv``) and
  ``report.md`` says what ran where and how long it took.

The run folder holds ``batch.json`` (settings, inputs, started/ended), ``log.txt`` (the batch's own lines, with
times), ``<phase>/phase.json`` (its attempts), ``<phase>/todo_rN.csv``, ``grid_<phase>_rN.log`` (the grid's output)
and, at the end, ``paired.csv`` and ``report.md``.

Exit codes: 0 every phase of every batch done; 5 stopped at the deadline or by Ctrl+C (resumable); 4 a named machine
is not ready; 1 an attempt failed or a phase used up its attempts (resumable); 2 bad input or options.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NOT_READY, EXIT_STOPPED = 0, 1, 2, 4, 5
BATCH_FILE = "batch.json"
PHASE_FILE = "phase.json"
LOG_FILE = "log.txt"
REPORT_FILE = "report.md"
DEFAULT_MAX_ATTEMPTS = 4


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_deadline(text: str | None, now: datetime | None = None) -> float | None:
    """'07:00' -> the next 07:00 (today while it is ahead, else tomorrow); an ISO date and time; '' or 'none' ->
    None. Raises ValueError for anything else."""
    if text is None or str(text).strip().lower() in ("", "none", "off", "0"):
        return None
    text = str(text).strip()
    now = now or datetime.now()
    with contextlib.suppress(ValueError):
        if "T" in text or "-" in text:
            return datetime.fromisoformat(text).timestamp()
    try:
        hh, mm = (int(x) for x in text.split(":"))
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except ValueError:
        raise ValueError(f"deadline {text!r}: give HH:MM, an ISO date and time, or none") from None
    if t <= now:
        t += timedelta(days=1)
    return t.timestamp()


def deadline_text(deadline: float | None) -> str:
    return datetime.fromtimestamp(deadline).strftime("%Y-%m-%d %H:%M") if deadline else "none"


# ------------------------------------------------------------------------------------------ the phase tables
def finished_ids(phase_dir: str | Path) -> set[str]:
    """Ids with a result (ok or failed) in any run folder of the phase (job tables, merged grid tables, snapshots),
    plus the molecules that crashed their worker twice (grid.json ``failed_without_result``: failed, no row)."""
    from ..chem import phases

    phase_dir = Path(phase_dir)
    done: set[str] = set()
    if not phase_dir.is_dir():
        return done
    for d in phases.expand_run_dirs([phase_dir]):
        if not d.is_dir():
            continue
        try:
            _label, rows = phases.read_phase_rows(d)
        except (FileNotFoundError, OSError, ValueError):
            continue
        done |= {str(r["id"]) for r in rows if r.get("status") in ("ok", "failed")}
    for g in phase_dir.glob("*/grid.json"):
        with contextlib.suppress(OSError, ValueError, TypeError, AttributeError):
            done |= {str(i) for i in json.loads(g.read_text(encoding="utf-8")).get("failed_without_result") or []}
    return done


def private_snapshot(synced: str | Path, dest: str | Path) -> int:
    """The finished molecules of another machine's run folder synced here, as summary/descriptors tables in
    ``dest``. A job table is read from a private copy (reading the synced database in place would make -wal/-shm
    files that SyncThing sends back); a folder with only its CSV tables is read as it is. Returns the number of
    molecules (0: nothing written, ``dest`` left as it was)."""
    from ..chem import jobs

    synced, dest = Path(synced), Path(dest)
    db = synced / jobs.DB_NAME
    with tempfile.TemporaryDirectory(prefix="qb-snap-") as tmp:
        src = synced
        if db.is_file():
            src = Path(tmp) / "run"
            src.mkdir()
            shutil.copyfile(db, src / jobs.DB_NAME)
            wal = db.with_name(db.name + "-wal")
            if wal.is_file():
                shutil.copyfile(wal, src / wal.name)
        elif not (synced / "summary.csv").is_file():
            return 0
        out = Path(tmp) / "tables"
        written = jobs.merge_tables([src], out)
        if not any(p.name == "summary.csv" for p in written):
            return 0
        with open(out / "summary.csv", newline="", encoding="utf-8") as f:
            n = sum(1 for _ in csv.DictReader(f))
        if not n:
            return 0
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
        return n


def remote_run_folders(phase_dir: str | Path) -> list[tuple[str, str]]:
    """(machine, run folder) of every remote share in the grids of the phase whose grid has not merged."""
    out: list[tuple[str, str]] = []
    for g in sorted(Path(phase_dir).glob("*/grid.json")):
        try:
            data = json.loads(g.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("state") == "merged":
            continue
        for sh in data.get("shares") or []:
            folder = str(sh.get("run_folder") or "")
            if sh.get("machine") not in ("", "local", None) and folder and "/" not in folder and "\\" not in folder:
                out.append((str(sh["machine"]), folder))
    return out


def snapshot_remotes(phase_dir: str | Path, synced_root: str | Path, log: Callable[[str], None]) -> int:
    """Copy what the remote shares of the phase's unmerged grids finished (their synced run folders under
    ``synced_root``) into the phase folder. Returns the number of molecules copied."""
    phase_dir, synced_root = Path(phase_dir), Path(synced_root)
    total = 0
    for machine, folder in remote_run_folders(phase_dir):
        synced = synced_root / folder
        if not synced.is_dir():
            log(f"  {machine}: its run folder {folder} has not synced here yet")
            continue
        n = private_snapshot(synced, phase_dir / f"partial_{folder}")
        total += n
        log(f"  {machine}: {n} molecule(s) finished in {folder}" + (" (copied into the phase folder)" if n else ""))
    return total


# --------------------------------------------------------------------------------------------- machines
def ready_machines(spec: str, extra_workers: Sequence[str] = (), include_paused: bool = False,
                   allow_any_host: bool = False, discover: Callable | None = None) -> tuple[list[str] | None, str]:
    """(machines for the next attempt, a note). ``spec`` 'auto': every worker that is ready, plus 'local'; 'local':
    this machine alone; else a comma list, every named worker must be ready (None and the reason otherwise)."""
    from .. import __version__
    from . import client, profile

    spec = (spec or "auto").strip()
    names = None if spec == "auto" else [m.strip() for m in spec.split(",") if m.strip() and m.strip() != "local"]
    if names is not None and not names:
        return ["local"], "this machine alone"
    discover = discover or client.discover
    try:
        refs = discover(list(extra_workers), timeout=5.0, allow_any_host=allow_any_host)
    except Exception as exc:  # noqa: BLE001 - no token, no registry: this machine alone (or stop if names were given)
        refs, why_all = {}, f"cannot ask the workers ({exc})"
    else:
        why_all = ""
    if names is None:
        names = sorted(refs)
        if not names:
            return ["local"], "this machine alone" + (f" ({why_all})" if why_all else " (no workers known)")
    commit = profile.git_commit()
    ok, notes = [], []
    for n in names:
        ref = refs.get(n)
        info = (getattr(ref, "info", None) or {}) if ref is not None else {}
        q = info.get("queue") or {}
        if ref is None:
            why = why_all or "not a known worker"
        elif not getattr(ref, "reachable", False):
            why = f"unreachable ({str(getattr(ref, 'error', ''))[:80]})"
        elif q.get("hold"):
            why = f"busy: {q['hold']}"
        elif q.get("paused") and not include_paused:
            why = "paused"
        elif info.get("quantabench_version") and info.get("quantabench_version") != __version__:
            why = f"QuantaBench {info.get('quantabench_version')}, this machine {__version__}"
        elif commit and info.get("commit") and not client.same_commit(commit, str(info.get("commit"))):
            why = f"commit {info.get('commit')}, this machine {commit} (update it and restart its worker)"
        else:
            ok.append(n)
            notes.append(f"{n} ready" + (" (paused: works once resumed)" if q.get("paused") else ""))
            continue
        notes.append(f"{n} not used: {why}")
        if spec != "auto":
            return None, f"{n} is not ready: {why}"
    return ok + ["local"], "; ".join(notes)


# ------------------------------------------------------------------------------------------------ a batch
@dataclasses.dataclass
class BatchOptions:
    phases: list[str]                         # canonical solvent names, '' = gas
    deadline: float | None = None
    machines: str = "auto"
    include_paused: bool = False
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    pair: bool = True
    weights: str | None = "auto"
    extra_workers: tuple[str, ...] = ()
    allow_any_host: bool = False
    grid_kwargs: dict = dataclasses.field(default_factory=dict)    # the rest of grid_build's keywords


class _Log:
    """The batch's own lines: printed and appended to log.txt (with times)."""

    def __init__(self, path: Path, echo: Callable[[str], None] = print):
        self.path, self.echo = path, echo

    def __call__(self, text: str) -> None:
        self.echo(text)
        with contextlib.suppress(OSError), open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"{now_text()}  {text}\n")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_todo(path: Path, items) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "smiles"])
        for it in items:
            w.writerow([it.id, getattr(it, "smiles", "") or ""])


def run_batch(items: Sequence, cfg, run_dir: str | Path, opts: BatchOptions, *, label: str = "",
              synced_root: str | Path | None = None, echo: Callable[[str], None] = print,
              grid_build: Callable | None = None, ready: Callable | None = None,
              cancel: Callable | None = None, clock: Callable[[], float] = time.time) -> int:
    """Run (or resume) one batch: ``items`` (``MoleculeInput``, unique ids) in each phase of ``opts`` into
    ``run_dir``. Returns an exit code (module docstring). ``grid_build``, ``ready`` and ``cancel`` default to
    ``client.grid_build``, ``ready_machines`` and a worker cancel (tests replace them)."""
    from . import client

    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _Log(run_dir / LOG_FILE, echo)
    grid_build = grid_build or client.grid_build
    ready = ready or (lambda: ready_machines(opts.machines, opts.extra_workers, opts.include_paused,
                                             opts.allow_any_host))
    cancel = cancel or _cancel_remote
    if synced_root is None:
        from .. import settings as settings_mod

        synced_root = settings_mod.load().output_path()
    ids = [str(it.id) for it in items]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        log(f"STOP: duplicate molecule ids: {', '.join(dupes[:10])}")
        return EXIT_USAGE
    label = label or run_dir.name
    info_path = run_dir / BATCH_FILE
    info = _read_json(info_path, {})
    resuming = bool(info)
    from .. import __version__
    from ..chem import phases as phases_mod
    from . import profile

    info.update({"label": label, "molecules": len(items), "phases": [phases_mod.phase_label(p) for p in opts.phases],
                 "machines": opts.machines, "include_paused": opts.include_paused, "deadline": deadline_text(
                     opts.deadline), "method": getattr(cfg, "method", ""), "qb_version": __version__,
                 "commit": profile.git_commit() or ""})
    info.setdefault("started", now_text())
    info.pop("ended", None)
    _write_json(info_path, info)
    log(f"{'resume' if resuming else 'start'} batch {label}: {len(items)} molecule(s), phases "
        f"{', '.join(info['phases'])}, deadline {deadline_text(opts.deadline)}")
    code = EXIT_OK
    for solvent in opts.phases:
        phase = phases_mod.phase_label(solvent)
        code = _run_phase(items, cfg, run_dir, phase, solvent, opts, label, Path(synced_root), log, grid_build,
                          ready, cancel, clock)
        if code != EXIT_OK:
            break
    done = [phases_mod.phase_label(p) for p in opts.phases
            if _read_json(run_dir / phases_mod.phase_label(p) / PHASE_FILE, {}).get("done")]
    if opts.pair and code == EXIT_OK and "gas" in done and len(done) > 1:
        try:
            summary = phases_mod.write_paired([run_dir / p for p in done], run_dir / "paired.csv")
            log(f"paired {summary.molecules} molecule(s) of {', '.join(done)} -> paired.csv "
                f"({summary.paired} ok in every phase)")
        except Exception as exc:  # noqa: BLE001 - the phases are done; a pairing problem must not undo that
            log(f"pair: {exc}")
    info["ended" if code == EXIT_OK else "stopped"] = now_text()
    info["exit"] = code
    _write_json(info_path, info)
    write_report(run_dir)
    if code == EXIT_OK:
        log(f"ALL-DONE batch {label}")
    return code


def _run_phase(items, cfg, run_dir: Path, phase: str, solvent: str, opts: BatchOptions, label: str,
               synced_root: Path, log, grid_build, ready, cancel, clock) -> int:
    pdir = run_dir / phase
    pdir.mkdir(exist_ok=True)
    state_path = pdir / PHASE_FILE
    state = _read_json(state_path, {"phase": phase, "attempts": []})
    if state.get("done"):
        log(f"{phase}: done in an earlier session ({len(state['attempts'])} attempt(s))")
        return EXIT_OK
    if remote_run_folders(pdir):
        log(f"{phase}: what the other machines finished since the last attempt:")
        snapshot_remotes(pdir, synced_root, log)
    phase_cfg = dataclasses.replace(cfg, solvent=solvent or "") if dataclasses.is_dataclass(cfg) else cfg
    while True:
        finished = finished_ids(pdir)
        todo = [it for it in items if str(it.id) not in finished]
        if not todo:
            state["done"] = True
            state["ended"] = now_text()
            _write_json(state_path, state)
            log(f"{phase}: every molecule has a result")
            return EXIT_OK
        if opts.deadline is not None and clock() >= opts.deadline:
            log(f"STOPPED: the deadline ({deadline_text(opts.deadline)}) has passed before {phase} attempt "
                f"{len(state['attempts']) + 1}; {len(todo)} molecule(s) left. Run the same command again to resume.")
            return EXIT_STOPPED
        if len(state["attempts"]) >= opts.max_attempts:
            log(f"STOP: {phase} has had {len(state['attempts'])} attempts; {len(todo)} molecule(s) left "
                "(--max-attempts)")
            return EXIT_FAILED
        machines, note = ready()
        if machines is None:
            log(f"STOP: {note}")
            return EXIT_NOT_READY
        k = len(state["attempts"])
        tag = f"r{k}"
        _write_todo(pdir / f"todo_{tag}.csv", todo)
        attempt = {"tag": tag, "machines": machines, "n": len(todo), "started": now_text(), "note": note}
        state["attempts"].append(attempt)
        _write_json(state_path, state)
        log(f"{phase} {tag}: {len(todo)} molecule(s) on {', '.join(machines)} ({note})")
        t0 = clock()
        grid_log = run_dir / f"grid_{phase}_{tag}.log"
        stopped = {"why": ""}

        def should_stop(stopped=stopped) -> bool:
            if not stopped["why"] and opts.deadline is not None and clock() >= opts.deadline:
                stopped["why"] = "deadline"
            return bool(stopped["why"])

        grid = None
        with open(grid_log, "a", encoding="utf-8", errors="replace") as fh:
            def glog(text: str) -> None:
                fh.write(f"{text}\n")
                fh.flush()

            try:
                grid = grid_build(todo, phase_cfg, machines, opts.weights, label=f"{label}_{phase}_{tag}", wait=True,
                                  output_dir=str(pdir), log=glog, should_stop=should_stop,
                                  extra_workers=opts.extra_workers, allow_any_host=opts.allow_any_host,
                                  **opts.grid_kwargs)
            except KeyboardInterrupt:
                stopped["why"] = "interrupted"
            except Exception as exc:  # noqa: BLE001 - a grid that could not start or follow: this attempt failed
                attempt["error"] = f"{type(exc).__name__}: {exc}"
        attempt.update({"ended": now_text(), "minutes": round((clock() - t0) / 60.0, 1),
                        "grid": getattr(grid, "id", ""), "grid_state": getattr(grid, "state", ""),
                        "missing": len(getattr(grid, "missing", []) or [])})
        merged = getattr(grid, "state", "") == "merged"
        if stopped["why"] == "deadline" and grid is not None and not merged:
            for line in cancel(grid, opts.allow_any_host):
                log(f"  {line}")
        if not merged:
            snapshot_remotes(pdir, synced_root, log)
        attempt["stopped"] = stopped["why"]
        _write_json(state_path, state)
        log(f"{phase} {tag}: {attempt.get('grid_state') or 'not started'} after {attempt['minutes']:g} min"
            + (f" ({stopped['why']})" if stopped["why"] else "") + (f": {attempt['error']}" if "error" in attempt
                                                                   else ""))
        if stopped["why"] == "deadline":
            log(f"STOPPED at the deadline during {phase} {tag}; run the same command again to resume (only "
                "unfinished molecules run)")
            return EXIT_STOPPED
        if stopped["why"] == "interrupted":
            log(f"STOPPED: interrupted during {phase} {tag}; run the same command again to resume")
            return EXIT_STOPPED
        if "error" in attempt and k + 1 >= opts.max_attempts:
            log(f"STOP: {phase} {tag} failed ({attempt['error']}); resumable")
            return EXIT_FAILED
        # merged, possibly with molecules no machine returned: the next round runs those


def _cancel_remote(grid, allow_any_host: bool = False) -> list[str]:
    """Cancel the remote shares of ``grid`` that are not done (they stop after their current molecule)."""
    from .. import settings as settings_mod
    from . import client

    token = settings_mod.load().dist.worker_token
    lines = []
    for sh in getattr(grid, "shares", []) or []:
        if sh.machine == client.LOCAL or not sh.job_id or sh.state in client.DONE_STATES or not sh.url:
            continue
        try:
            st = client.WorkerClient(sh.url, token, 10.0, allow_any_host).cancel(sh.job_id)
            lines.append(f"{sh.machine}: job {sh.job_id} {st.get('state', 'cancelled')} (stops after its current "
                         "molecule)")
        except client.WorkerError as exc:
            lines.append(f"{sh.machine}: cancel failed ({exc}); its job keeps running there")
    return lines


# ------------------------------------------------------------------------------------------------ report
def write_report(run_dir: str | Path) -> Path:
    """``report.md``: the batch's phases and attempts (what ran where, how long, what is left)."""
    run_dir = Path(run_dir)
    info = _read_json(run_dir / BATCH_FILE, {})
    lines = [f"# Batch {info.get('label', run_dir.name)}", "",
             f"- {info.get('molecules', '?')} molecule(s); phases {', '.join(info.get('phases', []))}; method "
             f"{info.get('method') or '-'}; QuantaBench {info.get('qb_version', '?')} ({info.get('commit') or '?'})",
             f"- started {info.get('started', '?')}"
             + (f", ended {info['ended']}" if info.get("ended") else
                f", stopped {info['stopped']} (exit {info.get('exit')}; resumable)" if info.get("stopped") else ""),
             f"- machines {info.get('machines', 'auto')}" + (", paused ones included" if info.get("include_paused")
                                                             else "") + f"; deadline {info.get('deadline', 'none')}",
             "", "| phase | attempt | machines | molecules | minutes | grid state | left without a result | note |",
             "|---|---|---|---|---|---|---|---|"]
    for phase in info.get("phases", []):
        st = _read_json(run_dir / phase / PHASE_FILE, {})
        for a in st.get("attempts", []):
            note = a.get("stopped") or a.get("error", "")
            lines.append(f"| {phase} | {a.get('tag')} | {', '.join(a.get('machines', []))} | {a.get('n')} | "
                         f"{a.get('minutes', '')} | {a.get('grid_state', '')} | {a.get('missing', '')} | {note} |")
        n_done = len(finished_ids(run_dir / phase))
        lines.append(f"| {phase} | total | | {n_done} with a result | | {'done' if st.get('done') else 'open'} | | |")
    if (run_dir / "paired.csv").is_file():
        lines += ["", "Paired table: `paired.csv`."]
    path = run_dir / REPORT_FILE
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def status_lines(run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    info = _read_json(run_dir / BATCH_FILE, {})
    if not info:
        return [f"{run_dir}: not a batch run folder (no {BATCH_FILE})"]
    out = [f"{info.get('label', run_dir.name)}: {info.get('molecules')} molecule(s), started {info.get('started')}"
           + (f", ended {info['ended']}" if info.get("ended") else
              f", stopped {info['stopped']} (exit {info.get('exit')})" if info.get("stopped") else ", running or "
              "interrupted")]
    total = int(info.get("molecules") or 0)
    for phase in info.get("phases", []):
        st = _read_json(run_dir / phase / PHASE_FILE, {})
        n = len(finished_ids(run_dir / phase))
        out.append(f"  {phase:8s} {n}/{total} with a result, {len(st.get('attempts', []))} attempt(s)"
                   + (", done" if st.get("done") else ""))
    return out


# ---------------------------------------------------------------------------------------------------- CLI
def register_cli(subparsers) -> None:
    from .client import _build_args, _worker_opts

    p = subparsers.add_parser("batch", help="staged batches: phases, attempts, a deadline and resume (BATCH-1)",
                              description="Run one molecule list (or a --list of them) as phases on the machines that "
                                          "answer, with a deadline; running the same command again resumes. See "
                                          "docs/WORKERS.md, Batches.")
    bs = p.add_subparsers(dest="batch_command", required=True)
    r = bs.add_parser("run", help="run or resume a batch (same inputs and science options as `grid build`)")
    _build_args(r)
    r.add_argument("--phases", default="gas", help="gas and/or solvents in order, e.g. gas,water (default gas)")
    r.add_argument("--run-dir", help="the batch's run folder (default <output folder>/<date>_<label>; the same "
                                     "folder resumes)")
    r.add_argument("--list", metavar="FILE", help="several batches: one input file per line, each to "
                                                  "<root>/<file stem>")
    r.add_argument("--root", help="with --list: the folder of the batches' run folders (default the output folder)")
    r.add_argument("--label", help="the batch's name (default the run folder's name)")
    r.add_argument("--deadline", default="none", help="stop at HH:MM (the next one), an ISO time, or none (default)")
    r.add_argument("--machines", default="auto", help="auto (every ready worker plus this machine), local, or a "
                                                      "comma list of workers that must be ready")
    r.add_argument("--include-paused", action="store_true",
                   help="a paused machine joins too (its molecules move at once; it gets work after its resume)")
    r.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                   help=f"attempts per phase (default {DEFAULT_MAX_ATTEMPTS})")
    r.add_argument("--no-pair", dest="pair", action="store_false", help="do not pair the phases at the end")
    r.add_argument("--weights", default="auto", help="as in `grid build`")
    r.add_argument("--away-after", type=float, metavar="S", help="as in `grid build`")
    r.add_argument("--no-share-work", dest="share_work", action="store_false", help="as in `grid build`")
    r.add_argument("--poll", type=float, default=5.0, help="seconds between progress checks")
    r.add_argument("--sync-wait", type=float, default=60.0, help="as in `grid build`")
    r.add_argument("--timeout", type=float, default=30.0, help="seconds per request to a worker")
    r.add_argument("--allow-version-mismatch", action="store_true")
    _worker_opts(r)
    r.set_defaults(func=_cli_run)
    s = bs.add_parser("status", help="phases and attempts of a batch run folder")
    s.add_argument("run_dir")
    s.set_defaults(func=_cli_status)


def _read_list(path: str) -> list[Path]:
    base = Path(path).expanduser().resolve().parent
    out = []
    for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            p = Path(line).expanduser()
            out.append(p if p.is_absolute() else base / p)
    return out


def _cli_run(args) -> int:
    from .. import cli as cli_mod
    from .. import settings as settings_mod
    from ..chem import phases as phases_mod

    try:
        deadline = parse_deadline(args.deadline)
        solvents = phases_mod.parse_phases(args.phases)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return EXIT_USAGE
    if args.max_attempts < 1:
        print("--max-attempts must be 1 or more", file=sys.stderr)
        return EXIT_USAGE
    if args.list and (args.file or args.molecules or args.run_dir):
        print("--list runs its own files: give no --file, molecules or --run-dir with it", file=sys.stderr)
        return EXIT_USAGE
    cfg = cli_mod.config_from_args(args)
    keep = tuple(name for name, value in (("threads", args.threads), ("gpu", args.gpu), ("priority", args.priority),
                                          ("free_threads", args.free_threads)) if value is not None)
    opts = BatchOptions(phases=solvents, deadline=deadline, machines=args.machines,
                        include_paused=args.include_paused, max_attempts=args.max_attempts, pair=args.pair,
                        weights=args.weights, extra_workers=tuple(args.worker), allow_any_host=args.allow_any_host,
                        grid_kwargs={"allow_version_mismatch": args.allow_version_mismatch, "keep_resources": keep,
                                     "poll": args.poll, "sync_wait": args.sync_wait, "timeout": args.timeout,
                                     "share_work": args.share_work, "away_after_s": args.away_after})
    print(f"Priority here: {cli_mod.lower_this_process(cfg.priority)}")
    out_root = Path(args.root or getattr(args, "out", None) or settings_mod.load().output_path()).expanduser()
    if args.list:
        try:
            files = _read_list(args.list)
        except OSError as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE
        batches = [([str(f)], out_root / f.stem, f.stem) for f in files]
    else:
        name = args.label or (Path(args.file[0]).stem if args.file else "batch")
        run_dir = Path(args.run_dir).expanduser() if args.run_dir else out_root / f"{datetime.now():%Y-%m-%d}_{name}"
        batches = [(args.file or [], run_dir, args.label or "")]
    code = EXIT_OK
    for files, run_dir, label in batches:
        if args.list:
            args.file, args.molecules = files, []
        try:
            items = cli_mod.read_inputs(args)
        except (ValueError, OSError) as exc:
            print(exc, file=sys.stderr)
            return EXIT_USAGE
        if not items:
            print(f"{run_dir.name}: no molecules", file=sys.stderr)
            return EXIT_USAGE
        if not getattr(args, "no_review", False):
            from ..cli_review import print_header

            print_header(items, cfg)
        code = run_batch(items, cfg, run_dir, opts, label=label)
        if code != EXIT_OK:
            break
    return code


def _cli_status(args) -> int:
    lines = status_lines(args.run_dir)
    for line in lines:
        print(line)
    return 0 if (Path(args.run_dir) / BATCH_FILE).is_file() else 1
