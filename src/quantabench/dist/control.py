"""Machine control (DIST-7): how a grid uses each machine, and pausing a machine so its work moves to the others.

Brandon (2026-11-11): "Can we make it a feature about how we balance the machine split and also how many processors /
much processing is used? ... I'm often opening and closing my laptop throughout the day. Is there a way we include a
pause button on a machine and the work is redistributed?" (docs/WORKERS.md, "Pausing a machine, a closing laptop").

Where each piece of state lives, and who writes it (one writer per file, so SyncThing never has to choose):

    settings.json (synced; any machine, only by a person)    each machine's split choice, run priority, free threads,
                                                              lanes, pause-on-battery (``settings.update_machine``: a
                                                              fresh read, the change, a write; stamped per entry)
    <worker state folder>/pause.json (not synced)             this machine's pause: written by its worker (HTTP
                                                              ``POST /v1/pause``), ``worker pause``, the Machines dock;
                                                              read by the worker, a ``grid build``'s local share and
                                                              the GUI's own batch. It survives restarts.
    <settings>/machines/status/<name>.json (synced)           what a worker last said about itself (paused, last
                                                              wake, battery): written only by that machine's worker,
                                                              so the dock can show a sleeping laptop as paused

``ClockWatch`` notices that the machine slept: the wall clock jumps against the time the watcher expected to have
slept (``time.monotonic`` stops during sleep on macOS and Linux). ``battery`` reads ``psutil.sensors_battery``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

PAUSE_FILE = "pause.json"
STATUS_DIR = "status"                 # <settings>/machines/status/<name>.json (not a profile: a sub-folder)
PAUSE_MODES = ("drain", "now")
PAUSE_REASONS = ("user", "battery")
#: a gap this much longer than the watcher's own sleep means the machine slept (or was frozen)
WAKE_GAP_S = 30.0
CLOCK_INTERVAL_S = 5.0


def _noop(*_a) -> None:
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(5):           # Windows: SyncThing or an antivirus may hold the file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


# ------------------------------------------------------------------------------------------ pause state
def state_dir(folder: str | Path | None = None) -> Path:
    """The worker state folder (``worker.default_state_dir``: per machine, never synced) unless ``folder``."""
    if folder:
        return Path(folder).expanduser()
    from .worker import default_state_dir

    return default_state_dir()


def pause_path(folder: str | Path | None = None) -> Path:
    return state_dir(folder) / PAUSE_FILE


def read_pause(folder: str | Path | None = None) -> dict:
    """This machine's pause state: {"paused": bool, "mode": "drain" | "now", "since", "by", "reason"}. A missing or
    unreadable file is "not paused". Never raises."""
    out = {"paused": False, "mode": "drain", "since": "", "by": "", "reason": ""}
    try:
        data = json.loads(pause_path(folder).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    out["paused"] = data.get("paused") is True
    if data.get("mode") in PAUSE_MODES:
        out["mode"] = data["mode"]
    for k in ("since", "by", "reason"):
        if isinstance(data.get(k), str):
            out[k] = data[k][:200]
    return out


def write_pause(folder: str | Path | None, paused: bool, mode: str = "drain", by: str = "",
                reason: str = "user") -> dict:
    """Save this machine's pause state (atomic); returns it."""
    if mode not in PAUSE_MODES:
        raise ValueError(f"pause mode must be one of {', '.join(PAUSE_MODES)}, not {mode!r}")
    state = {"paused": bool(paused), "mode": mode, "since": _now(), "by": str(by)[:120],
             "reason": reason if reason in PAUSE_REASONS else "user"}
    _atomic_write(pause_path(folder), json.dumps(state, indent=1) + "\n")
    return state


def pause_gate(folder: str | Path | None = None) -> dict:
    """``jobs.BatchControl.source`` for a batch that is not a worker job (the GUI's own): paused = hold (no new
    molecule; the batch goes on at resume), "now" also stops the molecules in progress."""
    st = read_pause(folder)
    if not st["paused"]:
        return {"mode": "run"}
    return {"mode": "hold", "interrupt": st["mode"] == "now"}


# ------------------------------------------------------------------------------------------ sleep / wake
class ClockWatch:
    """Notices that this machine slept: ``check()`` (call it about every ``interval`` s) returns how long the machine
    was away, or 0. It compares the wall clock with what passed since the last check: after a sleep the wall clock
    has run on, while the watcher's own sleep was ``interval`` long (``time.monotonic`` also stands still during
    sleep on macOS and Linux, so either clock shows it). A clock change of more than ``gap`` s looks the same and
    is taken as a wake too (harmless: the worker only asks its grid client before it goes on)."""

    def __init__(self, interval: float = CLOCK_INTERVAL_S, gap: float = WAKE_GAP_S,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self.interval, self.gap = float(interval), float(gap)
        self.clock, self.wall = clock, wall
        self._mono, self._wall = clock(), wall()
        self.last_gap_s = 0.0
        self.last_wake = ""                    # ISO time of the last wake noticed
        self.asleep_from = 0.0                 # wall time of the last check before that sleep

    def check(self) -> float:
        mono, wall = self.clock(), self.wall()
        d_mono, d_wall = mono - self._mono, wall - self._wall
        before = self._wall
        self._mono, self._wall = mono, wall
        # the time the machine was suspended: the wall clock ran on while the monotonic clock stood still (macOS,
        # Linux), or the check simply came that much later than its interval (a monotonic clock that counts sleep)
        away = max(d_wall - d_mono, max(d_wall, d_mono) - self.interval)
        if away > self.gap:
            self.last_gap_s = away
            self.last_wake = _now()
            self.asleep_from = before
            return away
        return 0.0


def battery() -> dict:
    """{"battery": bool (the machine has one), "on_battery": bool | None, "percent": float | None}. Never raises."""
    try:
        import psutil

        b = psutil.sensors_battery()
    except Exception:  # noqa: BLE001 - no battery support on this OS
        b = None
    if b is None:
        return {"battery": False, "on_battery": None, "percent": None}
    plugged = getattr(b, "power_plugged", None)
    percent = getattr(b, "percent", None)
    return {"battery": True, "on_battery": None if plugged is None else not plugged,
            "percent": float(percent) if isinstance(percent, (int, float)) else None}


# ------------------------------------------------------------------------------------------ status files
def status_dir() -> Path:
    from . import profile

    return profile.registry_dir() / STATUS_DIR


def publish_status(name: str, data: dict) -> Path | None:
    """``machines/status/<name>.json`` in the synced settings folder: what this machine's worker last said about
    itself (its only writer). Never raises; None when it could not be written."""
    from .client import _safe_name

    if not _safe_name(name):
        return None
    path = status_dir() / f"{name}.json"
    try:
        _atomic_write(path, json.dumps({"name": name, "updated": _now(), **data}, indent=1, default=str) + "\n")
    except OSError:
        return None
    return path


def load_statuses() -> dict[str, dict]:
    """Every machine's last status file ({machine key: data}); unreadable files are skipped."""
    from ..settings import machine_key

    out: dict[str, dict] = {}
    d = status_dir()
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        if path.name.startswith(".") or ".sync-conflict" in path.name:
            continue
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out[machine_key(data.get("name") or path.stem)] = data
    return out


# ------------------------------------------------------------------------------------------ processing budget
def free_threads_for_max_cpu(percent: float, logical: int) -> int:
    """The free threads that keep a run at about ``percent`` % of ``logical`` threads (the Machines dock's "Max CPU",
    ``machines set --max-cpu``): the automatic budgets then use the rest. At least one thread always works."""
    logical = max(1, int(logical))
    pct = min(100.0, max(1.0, float(percent)))
    return max(0, min(logical - 1, logical - max(1, math.floor(logical * pct / 100.0 + 1e-9))))


def max_cpu_for(free_threads: int, logical: int) -> float:
    logical = max(1, int(logical))
    return round(100.0 * (logical - min(max(0, int(free_threads)), logical - 1)) / logical, 1)


# ------------------------------------------------------------------------------------------ the split
#: what each --weights word means (grid build; the Machines dock's split column follows "auto")
WEIGHT_WORDS = {
    "auto": "each machine as the Machines dock sets it: Auto = measured speed, Manual = its weight, Off = left out",
    "measured": "each machine's measured speed: molecules/hour in the last grid of this protocol, else its benchmark",
    "equal": "the same share for every machine",
}
RATE_FIELDS = ("method", "solvent", "conformer_search", "crest_method", "crest_quick", "max_conformers", "refine_top",
               "frequencies", "qm_method", "rank_method")


def rate_key(config: dict) -> str:
    """What molecules/hour depend on, from a grid's config: the method and the protocol's cost settings."""
    cfg = config or {}
    return json.dumps({k: cfg.get(k) for k in RATE_FIELDS}, sort_keys=True, default=str)


def learned_rates(config: dict, names, output_dir: str | Path | None = None,
                  max_grids: int = 40) -> tuple[dict[str, float], str]:
    """({machine: molecules/hour}, grid id) from the newest finished grid with the same protocol (``rate_key``) that
    measured every machine in ``names`` (``grid.json`` ``rates``, written at the end of a grid build); ({}, "") when
    none did. A machine is "local" in a grid when it started it: that is matched by its machine name too."""
    from . import profile
    from .client import GRID_FILE, LOCAL
    from .plan import norm_name

    if output_dir is None:
        from .. import settings as settings_mod

        output_dir = settings_mod.load().output_path()
    out = Path(output_dir).expanduser()
    want = rate_key(config)
    here = norm_name(profile.machine_name())
    try:
        grids = sorted(out.glob(f"*grid*/{GRID_FILE}"), reverse=True)[:max_grids]
    except OSError:
        return {}, ""
    for path in grids:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rates = data.get("rates") if isinstance(data, dict) else None
        if not isinstance(rates, dict) or rate_key(data.get("config") or {}) != want:
            continue
        client = norm_name(str(data.get("client") or ""))
        got: dict[str, float] = {}
        for machine, rate in rates.items():
            if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not rate > 0:
                continue
            key = norm_name(machine)
            if key == LOCAL:
                key = client
            got[key] = float(rate)
        found = {}
        for n in names:
            key = here if n == LOCAL else norm_name(n)
            if key in got:
                found[n] = got[key]
        if len(found) == len(list(names)):
            return found, str(data.get("id") or path.parent.name)
    return {}, ""


# ------------------------------------------------------------------------------------------ CLI
def _logical_of(machine: str) -> int | None:
    from . import profile

    with contextlib.suppress(Exception):
        p = profile.load_profile(machine)
        if p is not None and p.logical_cores:
            return int(p.logical_cores)
    from .. import settings as settings_mod

    if settings_mod.machine_key(machine) == settings_mod.this_machine():
        from .. import resources

        return int(resources.logical_cores())
    return None


def machine_lines(s, machine: str, logical: int | None = None) -> list[str]:
    """What ``machines set`` and ``grid machines`` print for one machine's controls."""
    level, wp = s.priority_for(machine, scope=True)
    free, wf = s.free_threads_for(machine, scope=True)
    c = s.machine_control_for(machine)
    split = {"auto": "auto (measured speed)", "manual": f"manual weight {c['weight']:g}", "off": "off"}[c["split"]]
    lines = [f"split {split}; run priority {level} ({wp}); free threads {free} ({wf}); lanes {s.lanes_for(machine)}"
             + ("; pauses on battery" if c["pause_on_battery"] else "; runs on battery too")]
    if logical:
        from .. import priority

        lines.append(f"  {priority.describe(level, free, logical)}")
    return lines


def _cli_set(args) -> int:
    from .. import settings as settings_mod

    machine = settings_mod.machine_key(args.name) if args.name not in (None, "", "local") else settings_mod.this_machine()
    values: dict = {}
    logical = _logical_of(machine)
    if args.priority is not None:
        values["priority"] = None if args.priority == "default" else args.priority
    if args.free_threads is not None:
        try:
            values["free_threads"] = None if str(args.free_threads).lower() == "default" else int(args.free_threads)
        except ValueError:
            print(f"--free-threads wants a number or 'default', not {args.free_threads!r}", file=sys.stderr)
            return 2
    if args.max_cpu is not None:
        if not logical:
            print(f"--max-cpu needs {machine}'s thread count: it has no machine profile yet (start its worker once, or "
                  "give --free-threads)", file=sys.stderr)
            return 2
        values["free_threads"] = free_threads_for_max_cpu(args.max_cpu, logical)
        values["max_cpu"] = float(args.max_cpu)
    if args.lanes is not None:
        values["lanes"] = "" if args.lanes == "default" else args.lanes
        if values["lanes"]:
            from . import lanes as lanes_mod

            err = lanes_mod.spec_error(values["lanes"])
            if err:
                print(f"--lanes {args.lanes}: {err}", file=sys.stderr)
                return 2
    if args.split is not None:
        values["split"] = args.split
    if args.weight is not None:
        if not args.weight > 0:
            print("--weight must be more than 0 (use --split off to leave a machine out)", file=sys.stderr)
            return 2
        values["weight"] = float(args.weight)
        values.setdefault("split", "manual")
    if args.pause_on_battery is not None:
        values["pause_on_battery"] = bool(args.pause_on_battery)
    if not values:
        s = settings_mod.load()
        print(f"{machine}:")
        for line in machine_lines(s, machine, logical):
            print(f"  {line}")
        return 0
    try:
        s = settings_mod.update_machine(machine, **values)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{machine}: saved in {settings_mod.settings_dir() / settings_mod.SETTINGS_FILE}")
    for line in machine_lines(s, machine, logical):
        print(f"  {line}")
    print("  A running worker there picks these up within about 15 s once the settings have synced: the priority "
          "at once (lowered live), the lanes and free threads with its next job.")
    return 0


def add_machines_set(machines_parser) -> None:
    """``quantabench-cli machines set NAME ...`` on the ``machines`` command (``dist.profile.register_cli``)."""
    sub = machines_parser.add_subparsers(dest="machines_command")
    p = sub.add_parser("set", help="set one machine's split, run priority, free threads and lanes (DIST-7)",
                       description="Stores the values for that machine in the shared settings (a fresh read and "
                                   "write, stamped, so another machine's values are kept). Its worker reads them "
                                   "within about 15 s once SyncThing has brought the file: the run priority is "
                                   "applied to running lanes at once, lanes and free threads with its next job.")
    p.add_argument("name", nargs="?", help="machine name (default: this one; 'local' = this one)")
    p.add_argument("--priority", choices=["normal", "low", "background", "default"])
    p.add_argument("--free-threads", help="logical threads its automatic budgets leave free, or 'default'")
    p.add_argument("--max-cpu", type=float, metavar="PCT",
                   help="about this share of its threads: sets the free threads from its thread count")
    p.add_argument("--lanes", help="its lane spec (1, auto, N, gpu:1x4,cpu:5x2) or 'default'")
    p.add_argument("--split", choices=["auto", "manual", "off"],
                   help="its part of a grid split (grid build --weights auto): auto = measured speed, manual = "
                        "--weight, off = left out")
    p.add_argument("--weight", type=float, help="its manual weight (implies --split manual)")
    p.add_argument("--pause-on-battery", action=argparse_bool(), default=None,
                   help="its worker pauses itself while the machine runs on battery and resumes on power (the "
                        "default; --no-pause-on-battery lets a laptop work on battery)")
    p.set_defaults(func=_cli_set)


def argparse_bool():
    import argparse

    return argparse.BooleanOptionalAction


def _pause_target(name: str | None, extra) -> tuple[str, str | None]:
    """(machine key, worker URL or None for this machine's own pause file only)."""
    from .. import settings as settings_mod
    from . import client as client_mod

    here = settings_mod.this_machine()
    if name in (None, "", "local") and not extra:
        refs = client_mod.known_workers()
        ref = next((r for n, r in refs.items() if settings_mod.machine_key(n) == here), None)
        return here, ref.url if ref is not None else None
    refs = client_mod.known_workers(extra)
    if extra and name in (None, ""):
        ref = next(iter(r for k, r in refs.items() if r.source == "option"), None)
        return (ref.name or here), ref.url if ref else None
    ref = next((r for n, r in refs.items() if settings_mod.machine_key(n) == settings_mod.machine_key(name)), None)
    if ref is None:
        raise ValueError(f"unknown machine {name!r} (known workers: {', '.join(sorted(refs)) or 'none'})")
    return settings_mod.machine_key(name), ref.url


def _cli_pause(args, paused: bool) -> int:
    from .. import settings as settings_mod
    from . import client as client_mod

    mode = "now" if getattr(args, "now", False) else "drain"
    try:
        machine, url = _pause_target(args.name, args.worker)
    except (ValueError, client_mod.GridError) as exc:
        print(exc, file=sys.stderr)
        return 2
    here = settings_mod.this_machine()
    rc = 0
    if machine == here and not args.worker:
        st = write_pause(args.state_dir, paused, mode, by=here)
        print(f"{here} (this machine): {'paused (' + st['mode'] + ')' if paused else 'resumed'}; "
              f"{pause_path(args.state_dir)} (the worker, a grid build's local share and the GUI's batch here follow it)")
    if url:
        token = settings_mod.load().dist.worker_token
        try:
            q = client_mod.WorkerClient(url, token, timeout=10.0, allow_any_host=args.allow_any_host).pause(paused, mode)
            p = q.get("pause") or {}
            print(f"{machine} worker ({url}): " + ("paused" if q.get("paused") else "running")
                  + (f" ({p.get('mode')}: {'molecules in progress stop at their next checkpoint' if p.get('mode') == 'now' else 'molecules in progress finish'})"
                     if q.get("paused") else "")
                  + f"; {len(q.get('running', []))} job(s) running, {len(q.get('queued', []))} queued")
        except client_mod.WorkerError as exc:
            rc = 1 if machine != here else 0
            print(f"{machine} worker ({url}) did not answer ({exc})" +
                  ("; a grid that follows it treats it as away after a while and moves its molecules" if paused
                   else ""), file=sys.stderr if rc else sys.stdout)
    elif machine != here:
        print(f"{machine}: no worker address known", file=sys.stderr)
        rc = 1
    return rc


def add_worker_pause(worker_parser) -> None:
    """``quantabench-cli worker pause|resume [NAME]`` (``dist.worker.register_cli``)."""
    sub = worker_parser.add_subparsers(dest="worker_command")
    for verb, paused in (("pause", True), ("resume", False)):
        p = sub.add_parser(verb, help=("pause a machine: it takes no new molecules, and a grid moves its unstarted "
                                       "ones to the other machines" if paused else "resume a paused machine"),
                           description="No NAME: this machine (its pause file, which its worker, a grid build's local "
                                       "share and the GUI's batch follow, and its worker if one runs). NAME: that "
                                       "machine's worker over Tailscale. A pause lasts until resume, also across "
                                       "restarts.")
        p.add_argument("name", nargs="?", help="machine name (default: this machine)")
        if paused:
            g = p.add_mutually_exclusive_group()
            g.add_argument("--drain", action="store_true", help="the molecules in progress finish (the default)")
            g.add_argument("--now", action="store_true",
                           help="the molecules in progress stop at their next checkpoint and are handed back")
        p.add_argument("--worker", action="append", default=[], metavar="[NAME=]URL", help="a worker by URL")
        p.add_argument("--state-dir", help="this machine's worker state folder (default: the per-user one)")
        p.add_argument("--allow-any-host", action="store_true")
        p.set_defaults(func=lambda a, paused=paused: _cli_pause(a, paused))
