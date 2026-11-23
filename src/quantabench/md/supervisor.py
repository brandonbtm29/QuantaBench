"""The MD run supervisor (SIM-1): runs a run folder's protocol as segments, one LAMMPS process at a time.

    quantabench-cli md run DIR [--variant auto|cpu|gpu] [--threads N] [--detach]
    quantabench-cli md status DIR [--json]
    quantabench-cli md pause DIR [--now]
    quantabench-cli md resume DIR [--variant ...] [--threads N] [--detach]

``Supervisor(folder).run()`` takes the folder's ``run.json`` (``md.sim``), and for each part not yet done (the liquid
slab's bulk, then its slab; the steered pull's one part) starts one engine process, a *segment*, through
``chem.external.run_program``: at the run priority, within a thread budget, killed with its whole process tree if
the supervisor stops. While it runs, the supervisor reads the segment's log (``md.telemetry``) and writes
``progress.json`` every few seconds: the part and stage, the step, ns done of ns in all, ns/day over the last
minutes, the ETA, alerts. When the process ends it records the segment (machine, LAMMPS build, variant, threads,
first and last step, ns/day, exit) in ``run.json``, and goes on with the next part, or stops:

* **paused** (``md pause``: a file named STOP; the inputs' ``fix halt`` ends the run at the next multiple of
  ``pause_every`` steps and writes ``<part>.pause.restart`` and ``<part>.pause.data``);
* **stopped** (``md pause --now``: after ``GRACE_S`` with STOP in place LAMMPS is stopped; or the supervisor was told
  to stop): the next segment starts from the newest periodic restart file;
* **failed** (an ERROR, ``QB_ERROR``, a non-zero exit, or an input that ended without its done tag);
* **done**, after the last part.

``md resume`` (``Supervisor(..., resume=True)``) removes STOP and continues: from the newest checkpoint that this
machine and build may use (``choose_checkpoint``: a binary restart file written by this LAMMPS here, else the pause
data file, which any machine can read), after cutting the part's outputs at that step (``md.sim.cut_outputs``), with
an input written for the segment (``md.liquid.resume_input``). A part that resumes by "rerun" starts again.

The supervisor holds ``supervisor.lock`` in the folder while it runs, so two cannot run one folder. It never runs in
the GUI process: the GUI starts it as a process of its own (``start_detached``), so closing the window leaves the run
going, and it follows ``progress.json`` and ``run.json`` (``read_status``).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import sim
from . import telemetry as T

GRACE_S = 30.0                  # "Stop now": the STOP file's chance to end the run cleanly before LAMMPS is stopped
PROGRESS_EVERY_S = 2.0
STALE_S = 300.0                 # a heartbeat this old from another machine: that supervisor is gone
ENGINE_TIMEOUT_S = 365 * 86400.0   # no engine time limit for MD (the settings' per-program limit is for builds)
ACCELERATORS = ("auto", "cpu", "gpu", "kokkos")
DETACHED_ENV = "QUANTABENCH_MD_DETACHED"     # set for a supervisor started by start_detached (its stdout is the log)


def _log_line(folder: Path, text: str, echo: Callable[[str], None] | None) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    with contextlib.suppress(OSError), open(folder / sim.SUPERVISOR_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if echo is not None:
        echo(text)


def machine_name() -> str:
    try:
        from ..dist import profile

        return profile.machine_name()
    except Exception:  # noqa: BLE001 - a name is only a label
        import socket

        return socket.gethostname().split(".")[0].lower()


# ------------------------------------------------------------------------------------------- the run record
def load_run(folder: str | Path, *, save: bool = True) -> sim.Run:
    """The folder's ``run.json``; for an MD folder written before SIM-1, one built from its setup JSON (``legacy``:
    its inputs have no STOP lines) and saved."""
    folder = Path(folder)
    if (folder / sim.RUN_FILE).is_file():
        return sim.Run.load(folder)
    if (folder / "liquid_setup.json").is_file() and (folder / "in.slab").is_file():
        from . import liquid as L

        tm, cfg, meta = L.load_folder(folder)
        run = L.run_record(tm, cfg, meta, legacy=True)
    elif (folder / "smd_setup.json").is_file() and (folder / "in.smd").is_file():
        from . import lammps as M

        meta = json.loads((folder / "smd_setup.json").read_text(encoding="utf-8"))
        from dataclasses import fields

        known = {f.name for f in fields(M.SMDConfig)}
        cfg = M.SMDConfig(**{k: v for k, v in (meta.get("config") or {}).items() if k in known})
        run = M.smd_run_record(cfg, int(meta.get("smd_steps") or cfg.smd_steps), int(meta.get("n_atoms") or 0),
                               int(meta.get("n_surface") or 0), meta.get("molecules") or [],
                               float(meta.get("net_charge") or 0.0), meta.get("box") or [], "steered pull", [])
        run.legacy = True
        run.notes.append("in.smd written before SIM-1: no STOP lines, so 'md pause' stops it at once")
    else:
        raise sim.RunError(f"{folder}: not an MD run folder (no run.json, liquid_setup.json or smd_setup.json)")
    if save:
        run.save(folder)
    return run


# ------------------------------------------------------------------------------------------- the lock
def _pid_alive(pid: int | None, create_time: float | None = None) -> bool:
    if not pid:
        return False
    try:
        import psutil

        p = psutil.Process(int(pid))
        if create_time and abs(p.create_time() - float(create_time)) > 2.0:
            return False                      # the pid was reused
        return p.status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001 - gone, or not ours to see
        return False


def lock_holder(folder: str | Path) -> dict | None:
    """The live supervisor of ``folder`` ({"pid", "machine", "started", ...}), or None. A lock from another machine
    counts while its heartbeat in ``progress.json`` is fresher than ``STALE_S``."""
    folder = Path(folder)
    info = sim.read_json(folder / sim.LOCK_FILE)
    if not info:
        return None
    if info.get("machine") == machine_name():
        return info if _pid_alive(info.get("pid"), info.get("create_time")) else None
    prog = sim.read_json(folder / sim.PROGRESS_FILE) or {}
    beat = float(prog.get("heartbeat") or 0.0)
    return info if time.time() - beat < STALE_S else None


@contextlib.contextmanager
def held_lock(folder: Path, force: bool = False):
    path = folder / sim.LOCK_FILE
    me = {"pid": os.getpid(), "machine": machine_name(), "started": sim.now_iso()}
    with contextlib.suppress(Exception):
        import psutil

        me["create_time"] = psutil.Process().create_time()
    for _attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock_holder(folder)
            if holder is not None and not force:
                where = "here" if holder.get("machine") == me["machine"] else f"on {holder.get('machine')}"
                raise sim.RunError(f"{folder}: a supervisor already runs it {where} (pid {holder.get('pid')}, since "
                                   f"{holder.get('started')})") from None
            with contextlib.suppress(OSError):
                path.unlink()             # stale: its supervisor is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(me, fh)
        break
    else:
        raise sim.RunError(f"{folder}: could not take {sim.LOCK_FILE}")
    try:
        yield me
    finally:
        with contextlib.suppress(OSError):
            info = sim.read_json(path) or {}
            if info.get("pid") == me["pid"] and info.get("machine") == me["machine"]:
                path.unlink()


# ------------------------------------------------------------------------------------------- STOP
def read_stop(folder: str | Path) -> str | None:
    """None (no STOP file), "pause" or "now"."""
    p = Path(folder) / sim.STOP_FILE
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError:
        return "pause"
    return "now" if text.strip().lower().startswith("now") else "pause"


def request_pause(folder: str | Path, now: bool = False) -> str:
    """``md pause``: create STOP. Refused when no supervisor runs the folder (a STOP left behind would end the next
    run at its first check); then the run is already stopped."""
    folder = Path(folder)
    if lock_holder(folder) is None:
        st = read_status(folder)
        raise sim.RunError(f"{folder}: not running ({st.get('state')}); nothing to pause")
    (folder / sim.STOP_FILE).write_text(("now" if now else "pause") + f" {sim.now_iso()}\n", encoding="utf-8")
    return (f"Stopping now: the run gets {GRACE_S:.0f} s to end at a pause point, then LAMMPS is stopped"
            if now else "Pausing at the next pause point (a checkpoint is written there)")


def clear_stop(folder: str | Path) -> bool:
    p = Path(folder) / sim.STOP_FILE
    if p.exists():
        with contextlib.suppress(OSError):
            p.unlink()
            return True
    return False


# ------------------------------------------------------------------------------------------- the engine
def find_engine(path: str | None = None):
    """(LammpsInfo, why not): the LAMMPS this machine runs (``md.bench.find_lammps``), or the one at ``path``."""
    from . import bench

    if path:
        from .. import proctree

        probe = proctree.run_probe([path, "-h"], timeout=120)
        h = bench.parse_lmp_help(probe.text or "")
        if not h["version"] and not h["packages"]:
            return None, f"{path} -h did not print a LAMMPS version"
        return bench.LammpsInfo(path, "given", h["version"], h["packages"], h["accelerators"], h["gpu_present"]), ""
    return bench.find_lammps()


def build_record(info) -> dict:
    gpu = (info.accelerators or {}).get("GPU", {})
    return {"version": info.version, "path": info.path, "source": info.source, "packages": list(info.packages),
            "gpu_api": gpu.get("api", ""), "gpu_precision": gpu.get("precision", "")}


def choose_accelerator(run: sim.Run, info, requested: str | None) -> tuple[str, str]:
    """(accelerator, why) for the next segment: "cpu", "gpu", "kokkos", or "none" (the input sets its own: a steered
    pull written with ``gpu``). ``requested``: None/"auto", an accelerator, or a variant name ("gpu-cutoff16")."""
    from . import bench
    from . import long_range as LR

    req = (requested or "").strip().lower() or "auto"
    if run.template == "smd":
        if run.variant == "gpu":
            return "none", "in.smd sets the GPU package itself"
        return "cpu", "the rigid pull runs on the CPU"
    disp = run.protocol.dispersion or "ewald"
    if "-" in req:
        v = bench.parse_variant(req)
        if v.dispersion != bench.parse_variant(f"cpu-{disp}").dispersion:
            raise sim.RunError(f"variant {requested!r}: this folder's dispersion is {disp} (its inputs set it)")
        req = v.accelerator
    if req not in ACCELERATORS:
        raise sim.RunError(f"variant {requested!r}: {', '.join(ACCELERATORS)} (or a name like gpu-{disp})")
    if req == "auto":
        req = run.variant or "auto"
    if req == "auto":
        ok, why = bench.can_run(info, bench.Variant("gpu", bench.parse_variant(f"cpu-{disp}").dispersion))
        if ok and LR._gpu_check_failed():
            ok, why = False, "this machine's GPU variants failed the CPU/GPU pressure check (machines --bench md)"
        if ok:
            return "gpu", f"LAMMPS here has the GPU package and a GPU, and {disp} has GPU styles"
        return "cpu", f"CPU: {why}" if why else "CPU"
    ok, why = bench.can_run(info, bench.Variant(req, bench.parse_variant(f"cpu-{disp}").dispersion))
    if not ok:
        raise sim.RunError(f"{req}: {why}")
    return req, "as asked"


def engine_command(info, input_name: str, accelerator: str, threads: int, log_name: str | None
                   ) -> tuple[list[str], int, str]:
    """(command, CPU threads, short label). ``log_name`` None: the input opens its own log."""
    from . import bench

    if accelerator == "none":
        args, n, label = [], 1, "input"
    else:
        args, n, label = bench.accelerator_args(info, accelerator, threads)
    cmd = [info.path, "-nocite", "-log", log_name or "none", *args, "-in", input_name]
    return cmd, n, label


def _input_has_log(path: Path) -> bool:
    with contextlib.suppress(OSError):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("log ") or line.strip() == "log":
                return True
    return False


# ------------------------------------------------------------------------------------------- checkpoints
def choose_checkpoint(folder: str | Path, run: sim.Run, part: sim.Part, machine: str, build: dict
                      ) -> sim.Checkpoint:
    """Where the next segment of ``part`` starts: the furthest checkpoint this machine and build may read. Binary
    restart files (the pause restart, the periodic ones) only when the last segment of the part that ran ran here
    with this LAMMPS; the pause data file always (portable). A "rerun" part, or a part with no checkpoint, starts
    from its beginning."""
    folder = Path(folder)
    segs = run.part_segments(part.name)
    if part.resume == "rerun" or not segs:
        return sim.Checkpoint("start")
    ran = [s for s in segs if s.last_step is not None or s.status != "running"]
    binary_ok = bool(ran) and ran[-1].same_build(machine, build)
    cands: list[tuple[int, int, sim.Checkpoint]] = []       # (step, preference, checkpoint)
    if binary_ok:
        for name in [part.pause_restart, *part.restart_files]:
            s = sim.restart_step(folder / name)
            if s is not None and part.first_step <= s <= part.last_step:
                cands.append((s, 2, sim.Checkpoint("restart", name, s)))
    s = sim.data_step(folder / part.pause_data)
    if s is not None and part.first_step <= s <= part.last_step:
        cands.append((s, 1, sim.Checkpoint("data", part.pause_data, s)))
    if not cands:
        return sim.Checkpoint("start")
    return max(cands, key=lambda c: (c[0], c[1]))[2]


# ------------------------------------------------------------------------------------------- status
def read_status(folder: str | Path) -> dict:
    """What ``md status`` and the GUI show: ``run.json``'s state and segments, ``progress.json``'s latest numbers,
    and whether a supervisor is alive. A run whose state says running with no live supervisor is reported as
    "stopped" (``"stale": True``)."""
    folder = Path(folder)
    out: dict = {"folder": str(folder)}
    try:
        run = load_run(folder, save=False)
    except sim.RunError as exc:
        return {**out, "state": "unknown", "error": str(exc)}
    prog = sim.read_json(folder / sim.PROGRESS_FILE) or {}
    holder = lock_holder(folder)
    state = run.state
    alive = holder is not None
    if state in ("running", "pausing") and not alive:
        state, out["stale"] = "stopped", True
    elif alive and prog.get("state") in ("running", "pausing"):
        state = prog["state"]
    out.update(state=state, alive=alive, name=run.name, template=run.template, legacy=run.legacy,
               ns_total=run.ns_total, segments=[s.__dict__ for s in run.segments],
               parts=[{"name": p.name, "done": run.part_done(p.name), "ns": p.ns} for p in run.protocol.parts])
    if holder:
        out["supervisor"] = holder
    if alive and prog:
        out["progress"] = prog
        out["ns_done"] = prog.get("ns_done")
    else:
        out["ns_done"] = run.ns_done()
    out["stop_requested"] = read_stop(folder)
    return out


def format_status(st: dict) -> str:
    if st.get("state") == "unknown":
        return st.get("error", "unknown")
    L = [f"{st.get('name') or st['folder']}: {st['state']}"
         + (" (no supervisor is running it: md resume continues)" if st.get("stale") else "")]
    p = st.get("progress") or {}
    done, total = st.get("ns_done") or 0.0, st.get("ns_total") or 0.0
    line = f"  {done:.3g} / {total:.3g} ns"
    if p:
        line += f" · part {p.get('part')} ({p.get('stage')}) step {p.get('step'):,}" if p.get("step") is not None \
            else f" · part {p.get('part')}"
        if p.get("ns_per_day"):
            line += f" · {p['ns_per_day']:.3g} ns/day"
        if p.get("eta_text"):
            line += f" · ETA {p['eta_text']}"
        if p.get("machine"):
            line += f" · on {p['machine']}, {p.get('variant', '')} x {p.get('threads', '')} threads"
    L.append(line)
    for a in p.get("alerts") or []:
        L.append(f"  alert: {a}")
    if st.get("stop_requested"):
        L.append(f"  STOP requested ({st['stop_requested']})")
    for part in st.get("parts", []):
        L.append(f"  part {part['name']}: {'done' if part['done'] else 'to do'} ({part['ns']:.3g} ns)")
    for s in st.get("segments", []):
        speed = f"{s['ns_per_day']:.3g} ns/day" if s.get("ns_per_day") else "-"
        steps = f"{s.get('first_step')}-{s.get('last_step')}" if s.get("last_step") is not None else "-"
        start = (s.get("start") or {}).get("kind", "start")
        L.append(f"  segment {s['n']} ({s['part']}): {s['status']}, steps {steps}, from {start}, {s.get('variant')} "
                 f"x {s.get('threads')} on {s.get('machine')}, {speed}" + (f"; {s['error']}" if s.get("error") else ""))
    return "\n".join(L)


# ------------------------------------------------------------------------------------------- the supervisor
class Supervisor:
    def __init__(self, folder: str | Path, *, variant: str | None = None, threads: int | None = None,
                 lammps: str | None = None, priority: str | None = None, resume: bool = False,
                 echo: Callable[[str], None] | None = print, should_stop: Callable[[], bool] | None = None,
                 grace_s: float = GRACE_S, progress_every_s: float = PROGRESS_EVERY_S, force: bool = False):
        self.folder = Path(folder).resolve()
        self.variant = variant
        self.threads = threads
        self.lammps_path = lammps
        self.priority = priority
        self.resume = resume
        self.echo = echo
        self.external_stop = should_stop
        self.grace_s = float(grace_s)
        self.progress_every_s = float(progress_every_s)
        self.force = force
        self.machine = machine_name()
        self.run_record: sim.Run | None = None

    def log(self, text: str) -> None:
        _log_line(self.folder, text, self.echo)

    # ------------------------------------------------------------------ the loop
    def run(self) -> sim.Run:
        folder = self.folder
        if not folder.is_dir():
            raise sim.RunError(f"{folder}: no such folder")
        with held_lock(folder, self.force):
            run = load_run(folder)
            self.run_record = run
            if run.state == "done" and run.next_part() is None:
                self.log("The run is already done.")
                return run
            if run.segments and not self.resume and run.state not in ("written", "queued"):
                raise sim.RunError(f"{folder}: this run was started before ({run.state}); continue it with "
                                   "quantabench-cli md resume")
            if clear_stop(folder):
                self.log("Removed a STOP file left from before.")
            info, why = find_engine(self.lammps_path)
            if info is None:
                raise sim.RunError(f"LAMMPS: {why}")
            build = build_record(info)
            accel, why_accel = choose_accelerator(run, info, self.variant)
            threads = int(self.threads or self._default_threads())
            level = self._priority_level()
            self.log(f"Run {run.name or folder.name} ({run.template}) on {self.machine}: LAMMPS {info.version} "
                     f"({info.path}), {accel} ({why_accel}), {threads} threads, priority {level}")
            from .. import priority as prio_mod

            with prio_mod.children_at(level):
                while True:
                    part = run.next_part()
                    if part is None:
                        run.state = "done"
                        run.save(folder)
                        self._progress(run, None, None, state="done", message="done")
                        self.log("Done.")
                        return run
                    missing = [n for n in part.needs if not (folder / n).is_file()]
                    if missing:
                        run.state = "failed"
                        run.save(folder)
                        raise sim.RunError(f"part {part.name}: {', '.join(missing)} missing")
                    seg = self._segment(run, part, info, build, accel, threads)
                    if seg.status != "done":
                        run.state = {"paused": "paused", "stopped": "stopped"}.get(seg.status, "failed")
                        run.save(folder)
                        self._progress(run, part, seg, state=run.state, message=seg.error or seg.status)
                        self.log(f"Segment {seg.n} ({part.name}) {seg.status}"
                                 + (f": {seg.error}" if seg.error else "")
                                 + ("; continue with: quantabench-cli md resume " + str(folder)
                                    if run.state in sim.RESUMABLE else ""))
                        return run

    def _default_threads(self) -> int:
        from . import bench

        return bench.default_threads()

    def _priority_level(self) -> str:
        from .. import priority as prio_mod

        if self.priority:
            return prio_mod.normalize(self.priority)
        try:
            from .. import settings as settings_mod

            return settings_mod.load().priority_for()
        except Exception:  # noqa: BLE001 - defaults without settings
            return prio_mod.DEFAULT_PRIORITY

    # ------------------------------------------------------------------ one segment
    def _segment(self, run: sim.Run, part: sim.Part, info, build: dict, accel: str, threads: int) -> sim.Segment:
        from ..chem import external
        from ..chem.optimize import Cancelled

        folder = self.folder
        n = len(run.segments) + 1
        ckpt = choose_checkpoint(folder, run, part, self.machine, build)
        if ckpt.kind == "start":
            input_name = part.input
            log_name = part.log
            if run.part_segments(part.name):
                self.log(f"Part {part.name}: no checkpoint to continue from; it starts again")
        else:
            cut = sim.cut_outputs(folder, part, int(ckpt.step))
            if any(cut.values()):
                self.log(f"Part {part.name}: outputs cut at step {ckpt.step:,} before resuming ("
                         + ", ".join(f"{k} -{v}" for k, v in cut.items() if v) + ")")
            input_name = self._write_segment_input(run, part, ckpt, n)
            log_name = f"log.{part.name}.seg{n}"
        has_log = _input_has_log(folder / input_name)
        cmd, used_threads, label = engine_command(info, input_name, accel, threads, None if has_log else log_name)
        disp = run.protocol.dispersion
        seg = sim.Segment(n=n, part=part.name, machine=self.machine, platform=sys.platform, build=build,
                          variant=f"{accel}-{disp}" if disp and accel != "none" else accel, threads=used_threads,
                          command=[Path(cmd[0]).name, *cmd[1:]], input=input_name, log=log_name,
                          screen=f"screen.{part.name}.seg{n}.txt", start=ckpt.__dict__, started=sim.now_iso(),
                          first_step=None)
        run.segments.append(seg)
        run.state = "running"
        run.save(folder)
        self.log(f"Segment {n}: part {part.name} from "
                 + (f"{ckpt.kind} {ckpt.file} (step {ckpt.step:,})" if ckpt.kind != "start" else "its start")
                 + f": {' '.join(seg.command)}")
        tail = T.LogTail(folder / log_name)
        meter = T.SpeedMeter()
        t0 = time.time()
        stop = {"seen": None, "why": "", "last": 0.0, "pid": None}

        def should_stop() -> bool:
            now = time.time()
            if self.external_stop is not None and self.external_stop():
                stop["why"] = "the supervisor was told to stop"
                return True
            req = read_stop(folder)
            if req == "now":
                stop["seen"] = stop["seen"] or now
                if now - stop["seen"] >= self.grace_s:
                    stop["why"] = "stopped now (md pause --now)"
                    return True
            if now - stop["last"] >= self.progress_every_s:
                stop["last"] = now
                tail.poll()
                meter.add(tail.state.last_step, now)
                if stop["pid"] is None:
                    stop["pid"] = _engine_pid(info.path)
                    seg.pid = stop["pid"]
                self._progress(run, part, seg, tail=tail, meter=meter,
                               state="pausing" if req else "running")
            return False

        returncode, error = None, ""
        try:
            res = external.run_program(cmd, folder, threads=used_threads, log_name=seg.screen,
                                       should_stop=should_stop, check=False, timeout=ENGINE_TIMEOUT_S, poll_s=0.25)
            returncode = res.returncode
        except Cancelled:
            seg.status = "stopped"
            error = stop["why"] or "stopped"
        except external.ExternalProgramError as exc:
            seg.status = "failed"
            error = str(exc).splitlines()[0]
        except BaseException:
            seg.status, seg.error, seg.ended = "stopped", "the supervisor stopped", sim.now_iso()
            run.state = "stopped"
            with contextlib.suppress(Exception):
                run.save(folder)
            raise
        tail.poll()
        state = tail.state
        seg.ended = sim.now_iso()
        seg.wall_s = round(time.time() - t0, 1)
        seg.returncode = returncode
        seg.first_step = state.first_step
        seg.last_step = state.paused_step if state.paused_step is not None else state.last_step
        best = state.best_run
        if best:
            seg.ns_per_day = best.get("ns_per_day")
            seg.steps_per_s = best.get("timesteps_per_s")
        if seg.first_step is not None and seg.last_step is not None:
            seg.ns = round(part.ns_at(seg.last_step) - part.ns_at(seg.first_step), 6)
            if seg.ns_per_day is None and seg.ns and seg.wall_s:       # stopped: no Performance line; wall time
                seg.ns_per_day = round(seg.ns / (seg.wall_s / 86400.0), 4)
        if seg.status == "running":
            if state.paused_step is not None and returncode == 0:
                seg.status = "paused"
                if part.resume != "rerun":
                    seg.end = {"kind": "restart", "file": part.pause_restart, "data": part.pause_data,
                               "step": state.paused_step}
            elif part.done_tag and state.has_tag(part.done_tag) and returncode == 0:
                seg.status = "done"
            else:
                seg.status = "failed"
                errs = state.errors or T.summarize_log(folder / seg.screen).errors
                error = (errs[-1] if errs else f"LAMMPS exited with code {returncode}"
                         + ("" if returncode else f" before {part.done_tag or 'the end of its input'}"))
        seg.error = error
        if seg.status in ("paused", "stopped"):
            clear_stop(folder)
        run.save(folder)
        return seg

    def _write_segment_input(self, run: sim.Run, part: sim.Part, ckpt: sim.Checkpoint, n: int) -> str:
        if run.template == "liquid-slab":
            from . import liquid as L

            return L.write_segment_input(self.folder, part.name, ckpt, n)
        raise sim.RunError(f"{run.template}: no segment inputs (it resumes by running again)")

    # ------------------------------------------------------------------ progress.json
    def _progress(self, run: sim.Run, part: sim.Part | None, seg: sim.Segment | None, *, tail=None, meter=None,
                  state: str = "running", message: str = "") -> None:
        step = tail.state.last_step if tail is not None else None
        stage = part.stage_at(step) if part is not None and step is not None else None
        dt = float(stage.timestep_fs) if stage else 1.0
        sps = meter.steps_per_s() if meter is not None else None
        left = run.steps_left(part.name if part else None, step)
        eta = T.eta_seconds(left, sps)
        prog = {"state": state, "part": part.name if part else None, "stage": stage.name if stage else None,
                "segment": seg.n if seg else None, "step": step,
                "part_end": part.last_step if part else None,
                "ns_done": round(run.ns_done(part.name if part else None, step), 6), "ns_total": run.ns_total,
                "steps_per_s": sps, "ns_per_day": meter.ns_per_day(dt) if meter is not None else None,
                "eta_s": eta, "eta_text": T.format_eta(eta),
                "machine": self.machine, "variant": seg.variant if seg else "", "threads": seg.threads if seg else 0,
                "engine_pid": seg.pid if seg else None, "supervisor_pid": os.getpid(),
                "updated": sim.now_iso(), "heartbeat": time.time(), "message": message,
                "alerts": T.alerts(tail.state, target_T=self._target_T(run), running=True) if tail is not None else []}
        if run.ns_total:
            prog["fraction"] = prog["ns_done"] / run.ns_total
        with contextlib.suppress(OSError):
            sim.write_json_atomic(self.folder / sim.PROGRESS_FILE, prog)

    def _target_T(self, run: sim.Run) -> float | None:
        with contextlib.suppress(Exception):
            if run.template == "liquid-slab":
                meta = sim.read_json(self.folder / "liquid_setup.json") or {}
                return float((meta.get("config") or {}).get("temperature"))
        return None


def _engine_pid(exe: str) -> int | None:
    """The pid of the LAMMPS process this supervisor started (a child or grandchild), for the record."""
    with contextlib.suppress(Exception):
        import psutil

        name = Path(exe).name.lower()
        for p in psutil.Process().children(recursive=True):
            with contextlib.suppress(Exception):
                if p.name().lower().startswith(name.split(".")[0]):
                    return p.pid
    return None


# ------------------------------------------------------------------------------------------- detached start
def supervisor_command(folder: str | Path, *, resume: bool = False, variant: str | None = None,
                       threads: int | None = None, lammps: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "quantabench", "md", "resume" if resume else "run", str(Path(folder).resolve())]
    if variant:
        cmd += ["--variant", variant]
    if threads:
        cmd += ["--threads", str(int(threads))]
    if lammps:
        cmd += ["--lammps", lammps]
    return cmd


def start_detached(folder: str | Path, *, resume: bool = False, variant: str | None = None,
                   threads: int | None = None, lammps: str | None = None) -> subprocess.Popen:
    """Start the supervisor as a process of its own that outlives the caller (the GUI, a terminal): no console
    window, its own process group, out of the caller's job object where Windows allows it; its output is appended to
    ``supervisor.log`` in the folder."""
    folder = Path(folder).resolve()
    cmd = supervisor_command(folder, resume=resume, variant=variant, threads=threads, lammps=lammps)
    out = open(folder / sim.SUPERVISOR_LOG, "a", encoding="utf-8")  # noqa: SIM115 - handed to the child
    env = {**os.environ, "PYTHONUNBUFFERED": "1", DETACHED_ENV: "1"}   # its own lines go to the log once
    kwargs: dict = {"cwd": str(folder), "stdin": subprocess.DEVNULL, "stdout": out, "stderr": subprocess.STDOUT,
                    "close_fds": True, "env": env}
    try:
        if sys.platform == "win32":
            flags = (getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                     | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
            try:
                # CREATE_BREAKAWAY_FROM_JOB (0x01000000): not ended with the caller's job
                return subprocess.Popen(cmd, creationflags=flags | 0x01000000, **kwargs)  # noqa: S603
            except OSError:              # the caller's job forbids breaking away: the run then ends with that job
                return subprocess.Popen(cmd, creationflags=flags, **kwargs)  # noqa: S603
        return subprocess.Popen(cmd, start_new_session=True, **kwargs)  # noqa: S603
    finally:
        out.close()


def wait_started(folder: str | Path, proc: subprocess.Popen | None = None, timeout: float = 30.0) -> bool:
    """True once a supervisor holds the folder's lock (or the run already finished), False if ``proc`` died first."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if lock_holder(folder) is not None:
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.2)
    return False


def install_signal_handlers() -> None:
    """SIGTERM (and SIGBREAK on Windows) end the supervisor like Ctrl+C, so it stops LAMMPS and records the segment."""
    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, handler)
