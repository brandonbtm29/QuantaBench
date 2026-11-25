"""MD runs as jobs (SIM-1): the run of a written MD folder, its segments, and the files that describe them.

A run folder (``md.liquid`` or ``md.lammps`` writes one) is the product: it runs with ``lmp`` alone. This module is
its description as data, written next to the inputs as ``run.json``::

    Run               one execution of a protocol on a system (``run.json``)
      system          what is simulated (System: components, box, packing)
      parameterization the force field as applied (Parameterization: components, physics, checks)
      protocol        Protocol: parts in order, each one LAMMPS input run as its own process
        parts[]       Part: "bulk" (in.bulk), "slab" (in.slab), "smd" (in.smd); stages with step ranges, the
                      output files it appends to, its restart files and how it resumes
      segments[]      Segment: one engine process. Machine, engine build, variant, threads, the checkpoint it
                      started from and the one it ended with, first and last step, times, exit, ns/day
    progress.json     the supervisor's heartbeat while it runs (``md.supervisor``)
    STOP              created by ``md pause`` (or the GUI): the inputs' ``fix halt`` sees it and the part ends
                      at the next pause point with a checkpoint

Segments, checkpoints and resume (the split test plan's rules)
--------------------------------------------------------------
A pause, a crash, a "Stop now" or a move to another accelerator ends a segment; the next segment of the same part
continues from a checkpoint:

* a **pause checkpoint** (``<part>.pause.restart`` and ``<part>.pause.data`` at the step the pause stopped at);
* else the newest of the part's periodic **binary restart files** (``<part>.restart.a/b``, every
  ``Protocol.restart_every`` steps), whose step ``restart_step`` reads from the file's header.

Binary restart files are only for the same machine and the same LAMMPS build (LAMMPS reads them "only with the same
executable on the same platform"); between machines the portable checkpoint is the ``write_data`` file plus the step
number. Each segment records its machine, build, threads and start and end steps, so the supervisor knows which of
the two it may use. Before a segment resumes from step S, every output the part appends to is cut at S (``cut_at``):
rows a resumed fix writes again (an ``ave/time`` with one sample per row, a dump frame) from S on, the others after S,
so the files stay one continuous series with no duplicate or missing step; the readers also merge by step.

A part that resumes by ``"rerun"`` (the short rigid steered pull) starts again from its beginning after a pause.

Pure Python (no numpy, no LAMMPS), so the CLI, the supervisor and the GUI import it freely.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import struct
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from .. import __version__

RUN_FILE = "run.json"
PROGRESS_FILE = "progress.json"
STOP_FILE = "STOP"
LOCK_FILE = "supervisor.lock"
SUPERVISOR_LOG = "supervisor.log"
RUN_VERSION = 1
#: Run.state values. "written": never started; "pausing": STOP seen, the part ends at its next pause point
STATES = ("written", "queued", "running", "pausing", "paused", "done", "failed", "stopped", "cancelled")
#: states in which nothing runs and ``md resume`` continues the run
RESUMABLE = ("paused", "failed", "stopped")
#: Segment.status values
SEGMENT_STATUS = ("running", "done", "paused", "stopped", "failed")
#: the tags the inputs print (md.liquid, md.lammps); the supervisor reads them in the log
PAUSED_TAG = "QB_PAUSED"
ERROR_TAG = "QB_ERROR"


class RunError(RuntimeError):
    """A run folder that cannot be run, resumed or read."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _from_dict(cls, d: dict | None):
    """A dataclass from a dict, ignoring keys it does not know (a newer QuantaBench's run.json)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


# ------------------------------------------------------------------------------------------- the protocol
@dataclass
class Stage:
    """One stage of a part. Steps are LAMMPS' step counter within the part (``reset_timestep`` 0 at its start):
    the stage runs from ``start`` to ``end`` (``run end upto``). A minimization or the slab's box opening has no
    steps (start == end)."""

    name: str                       # "minimize", "warmup", "compress", "npt", "open-box", "nvt", "soft", "equil", "pull"
    kind: str                       # minimize | nvt | npt | open-box | smd | production
    start: int
    end: int
    timestep_fs: float = 1.0
    description: str = ""

    @property
    def steps(self) -> int:
        return max(0, int(self.end) - int(self.start))

    @property
    def ns(self) -> float:
        return self.steps * float(self.timestep_fs) / 1e6


@dataclass
class OutputFile:
    """A file a part appends to while it runs. ``kind``: "ave/time" (rows ``step values...``), "ave/chunk" (blocks
    ``step nchunks total`` + rows), "dump" (``ITEM: TIMESTEP`` frames), "dcd" (an OpenMM run's binary frames,
    ``md.dcd``) or "print" (``fix print`` rows: the first column is not the step, so such files are never cut). ``rewrites_start``: a resumed run writes the checkpoint's
    step again (an ``ave/time`` with one sample per row writes at its setup; a dump writes its first frame), so a cut
    for a resume at step S removes S too."""

    name: str
    kind: str
    every: int = 0
    rewrites_start: bool = True


@dataclass
class Part:
    name: str                       # "bulk" | "slab" | "smd"
    input: str                      # the folder's input: "in.bulk"
    log: str                        # its log: "log.bulk" (later segments: "log.bulk.seg2", ...)
    stages: list[Stage] = field(default_factory=list)
    outputs: list[OutputFile] = field(default_factory=list)
    restart_files: list[str] = field(default_factory=list)   # the periodic binary restarts, ["bulk.restart.a", ...]
    resume: str = "restart"         # "restart": segments continue from a checkpoint; "rerun": the part starts again
    done_tag: str = ""              # what the input prints when the part is complete: "QB_BULK done"
    product: str = ""               # the file it ends with: "bulk_final.data"
    needs: list[str] = field(default_factory=list)          # files that must exist before it starts

    @property
    def first_step(self) -> int:
        return int(self.stages[0].start) if self.stages else 0

    @property
    def last_step(self) -> int:
        return int(self.stages[-1].end) if self.stages else 0

    @property
    def steps(self) -> int:
        return sum(s.steps for s in self.stages)

    @property
    def ns(self) -> float:
        return sum(s.ns for s in self.stages)

    @property
    def pause_restart(self) -> str:
        return f"{self.name}.pause.restart"

    @property
    def pause_data(self) -> str:
        return f"{self.name}.pause.data"

    def stage_at(self, step: int) -> Stage | None:
        """The stage running at ``step`` (the last one at or after its end)."""
        for s in self.stages:
            if s.steps and step < s.end:
                return s
        return self.stages[-1] if self.stages else None

    def ns_at(self, step: int | None) -> float:
        """Simulated time (ns) of this part up to ``step``."""
        if step is None:
            return 0.0
        done = 0.0
        for s in self.stages:
            if not s.steps:
                continue
            k = min(max(int(step), s.start), s.end) - s.start
            done += k * float(s.timestep_fs) / 1e6
        return done

    def steps_left(self, step: int | None) -> int:
        if step is None:
            return self.steps
        return sum(max(0, s.end - max(int(step), s.start)) for s in self.stages if s.steps)

    @classmethod
    def from_dict(cls, d: dict) -> Part:
        p = _from_dict(cls, d)
        p.stages = [_from_dict(Stage, s) for s in d.get("stages", [])]
        p.outputs = [_from_dict(OutputFile, o) for o in d.get("outputs", [])]
        return p


@dataclass
class Protocol:
    """An ordered list of parts (stages grouped by the input that runs them), from a template."""

    template: str                   # "liquid-slab" | "smd"
    name: str = ""
    parts: list[Part] = field(default_factory=list)
    dispersion: str = ""            # md.liquid: "ewald" | "cutoffN"
    restart_every: int = 0          # steps between periodic binary restarts
    pause_every: int = 0            # steps between the checks of the STOP file (pauses land on these steps)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def part(self, name: str) -> Part:
        for p in self.parts:
            if p.name == name:
                return p
        raise RunError(f"no part {name!r} in this protocol ({', '.join(p.name for p in self.parts)})")

    @property
    def ns(self) -> float:
        return sum(p.ns for p in self.parts)

    @classmethod
    def from_dict(cls, d: dict) -> Protocol:
        p = _from_dict(cls, d)
        p.parts = [Part.from_dict(x) for x in d.get("parts", [])]
        return p


# ------------------------------------------------------------------------------------------- system and force field
@dataclass
class System:
    """What is simulated. ``components``: [{"name", "smiles", "count", "role": "liquid" | "adsorbate" | "surface" |
    "ion"}]; ``box``: lengths in Å ({"lateral", "slab_thickness", "vacuum"} or {"cell": [lx, ly, lz]}); ``packing``:
    {"method", "density_g_cm3", "density_source", "tolerance_A", "seed", "min_contact_A"}."""

    name: str = ""
    kind: str = ""                  # "liquid-slab" | "surface" | "bulk" | (v2) "mixture" | "solution"
    components: list[dict] = field(default_factory=list)
    surface: dict | None = None
    box: dict = field(default_factory=dict)
    packing: dict = field(default_factory=dict)
    n_atoms: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Parameterization:
    """The force field as applied to the system (the parameterization builder's node tree, SIMULATE_DESIGN.md
    §9.1). v1 fills one component node and the physics: ``physics`` = {"vdw": {...}, "coulomb": {...},
    "mixing", "special_lj", "special_coul"}; the full parameters stay in the setup JSON next to it."""

    components: list[dict] = field(default_factory=list)
    physics: dict = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ------------------------------------------------------------------------------------------- runs and segments
@dataclass
class Checkpoint:
    kind: str                       # "restart" (binary, same machine and build) | "data" (write_data, portable) | "start"
    file: str = ""
    step: int | None = None


@dataclass
class Segment:
    """One engine process."""

    n: int                          # 1, 2, ... over the whole run
    part: str
    machine: str = ""
    platform: str = ""
    build: dict = field(default_factory=dict)       # {"version", "path", "source", "packages", "gpu_api"}
    variant: str = ""               # "cpu-cutoff16", "gpu-cutoff16", "cpu-ewald", "cpu" (smd)
    threads: int = 0
    command: list[str] = field(default_factory=list)
    input: str = ""
    log: str = ""
    screen: str = ""
    pid: int | None = None
    start: dict | None = None       # Checkpoint as a dict: where it started ({"kind": "start"} = the part's start)
    end: dict | None = None         # the pause checkpoint it wrote, if any
    first_step: int | None = None
    last_step: int | None = None
    started: str = ""
    ended: str = ""
    status: str = "running"
    returncode: int | None = None
    error: str = ""
    ns_per_day: float | None = None      # LAMMPS' "Performance:" of the segment's longest run
    steps_per_s: float | None = None
    wall_s: float | None = None
    ns: float | None = None              # simulated in this segment

    def same_build(self, machine: str, build: dict) -> bool:
        """Whether this segment's binary restart files are readable by ``build`` on ``machine``."""
        return (self.machine == machine and str(self.build.get("version", "")) == str(build.get("version", ""))
                and str(self.build.get("path", "")) == str(build.get("path", "")))


@dataclass
class Run:
    template: str                   # "liquid-slab" | "smd"
    name: str = ""
    system: System = field(default_factory=System)
    parameterization: Parameterization = field(default_factory=Parameterization)
    protocol: Protocol = field(default_factory=lambda: Protocol(""))
    kind: str = "md"
    engine: str = "lammps"
    variant: str = ""               # the preferred accelerator ("" = auto: the GPU where the variant has one)
    resources: dict = field(default_factory=dict)   # {"threads": N, "priority": "low"}; empty = the settings'
    state: str = "written"
    segments: list[Segment] = field(default_factory=list)
    created: str = field(default_factory=now_iso)
    generator: str = f"QuantaBench {__version__}"
    run_version: int = RUN_VERSION
    legacy: bool = False            # inputs written before SIM-1 (no STOP lines): pause is "Stop now" only
    notes: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Run:
        r = _from_dict(cls, d)
        r.system = _from_dict(System, d.get("system"))
        r.parameterization = _from_dict(Parameterization, d.get("parameterization"))
        r.protocol = Protocol.from_dict(d.get("protocol") or {"template": d.get("template", "")})
        r.segments = [_from_dict(Segment, s) for s in d.get("segments", [])]
        return r

    def save(self, folder: str | Path) -> Path:
        path = Path(folder) / RUN_FILE
        write_json_atomic(path, self.to_dict())
        return path

    @classmethod
    def load(cls, folder: str | Path) -> Run:
        path = Path(folder) / RUN_FILE
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            raise RunError(f"{folder}: no {RUN_FILE}") from None
        except (OSError, ValueError, TypeError) as exc:
            raise RunError(f"{path}: unreadable ({exc})") from exc

    # ------------------------------------------------------------------ where the run is
    def part_segments(self, part: str) -> list[Segment]:
        return [s for s in self.segments if s.part == part]

    def part_done(self, part: str) -> bool:
        return any(s.status == "done" for s in self.part_segments(part))

    def next_part(self) -> Part | None:
        for p in self.protocol.parts:
            if not self.part_done(p.name):
                return p
        return None

    def last_step(self, part: str) -> int | None:
        """The furthest step a segment of ``part`` is known to have reached."""
        steps = [s.last_step for s in self.part_segments(part) if s.last_step is not None]
        return max(steps) if steps else None

    @property
    def ns_total(self) -> float:
        return self.protocol.ns

    def ns_done(self, current_part: str | None = None, current_step: int | None = None) -> float:
        """Simulated ns of the finished parts, plus ``current_part`` up to ``current_step``."""
        out = 0.0
        for p in self.protocol.parts:
            if self.part_done(p.name):
                out += p.ns
            elif p.name == current_part:
                out += p.ns_at(current_step)
            else:
                out += p.ns_at(self._resume_step(p))
        return out

    def _resume_step(self, part: Part) -> int | None:
        """Where ``part`` stands between segments: the step of its last pause checkpoint, if any."""
        for s in reversed(self.part_segments(part.name)):
            if s.end and s.end.get("step") is not None:
                return int(s.end["step"])
            if s.status in ("stopped", "failed"):
                return None
        return None

    def steps_left(self, current_part: str | None = None, current_step: int | None = None) -> list[tuple[int, float]]:
        """(steps, timestep fs) still to run, per stage, from ``current_part`` at ``current_step`` on."""
        out: list[tuple[int, float]] = []
        for p in self.protocol.parts:
            if self.part_done(p.name):
                continue
            step = current_step if p.name == current_part else self._resume_step(p)
            for s in p.stages:
                if s.steps:
                    left = s.end - max(int(step) if step is not None else s.start, s.start)
                    if left > 0:
                        out.append((left, float(s.timestep_fs)))
        return out


# ------------------------------------------------------------------------------------------- files
def write_json_atomic(path: str | Path, data) -> None:
    """Write JSON through a ``.tmp`` file and a rename, so a reader (or SyncThing) never sees half of it."""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows: a reader has the file open for a moment
            if attempt == 19:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            time.sleep(0.05)


def read_json(path: str | Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_run_folder(folder: str | Path) -> bool:
    """A folder ``md run`` can run: a ``run.json``, or an MD setup written before SIM-1 (``liquid_setup.json`` with
    ``in.bulk``/``in.slab``, ``smd_setup.json`` with ``in.smd``)."""
    f = Path(folder)
    return (f / RUN_FILE).is_file() or ((f / "liquid_setup.json").is_file() and (f / "in.slab").is_file()) \
        or ((f / "smd_setup.json").is_file() and (f / "in.smd").is_file())


# ------------------------------------------------------------------------------------------- binary restart files
#: write_restart.cpp's header: the magic string, an endian int and a format revision, then (int flag, value) pairs:
#: VERSION (a string), SMALLINT, TAGINT, BIGINT (ints: their sizes), UNITS (a string), NTIMESTEP (a bigint), ...
#: (22 Jul 2025: flags 0, 1, 48 (IMAGEINT, added later), 2, 3, 4, 5). The reader does not trust the numbering: it
#: finds the units string by its content and takes the bigint after it.
_MAGIC = b"LammpS RestartT"
_UNITS = {b"real", b"lj", b"metal", b"si", b"cgs", b"electron", b"micro", b"nano"}


def restart_step(path: str | Path) -> int | None:
    """The timestep stored in a LAMMPS binary restart file's header, or None (not a restart file, truncated, or a
    layout this reader does not know)."""
    try:
        with open(path, "rb") as fh:
            b = fh.read(1024)
    except OSError:
        return None
    if not b.startswith(_MAGIC) or len(b) < 64:
        return None
    i = len(_MAGIC) + 1
    endian = "<" if struct.unpack_from("<i", b, i)[0] == 1 else ">"
    i += 8                                    # endian, format revision
    try:
        (flag,) = struct.unpack_from(endian + "i", b, i)      # VERSION: a string
        (n,) = struct.unpack_from(endian + "i", b, i + 4)
        if not 0 < n < 256:
            return None
        i += 8 + n
        prev = 8
        for _ in range(32):
            (_flag, value) = struct.unpack_from(endian + "ii", b, i)
            text = b[i + 8:i + 8 + value - 1] if 2 <= value <= 10 else b""
            if text in _UNITS and b[i + 8 + value - 1:i + 8 + value] == b"\0":
                j = i + 8 + value                              # the next pair: NTIMESTEP, a bigint of BIGINT's size,
                fmt = "i" if prev == 4 else "q"                # the int just before UNITS
                (step,) = struct.unpack_from(endian + fmt, b, j + 4)
                return int(step) if step >= 0 else None
            prev = value
            i += 8
    except struct.error:
        return None
    return None


def data_step(path: str | Path) -> int | None:
    """The step in a ``write_data`` file's first line ("LAMMPS data file via write_data, version ..., timestep = 100,
    units = real"), or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return None
    m = re.search(r"timestep\s*=\s*(\d+)", first)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------- cutting outputs
def _step_of(line: str) -> int | None:
    s = line.split(None, 1)
    if not s:
        return None
    try:
        return int(float(s[0]))
    except ValueError:
        return None


def cut_at(path: str | Path, kind: str, step: int, keep_step: bool) -> int:
    """Cut an output file before a segment resumes at ``step``: drop every row (block, frame) after ``step``, and the
    one at ``step`` too unless ``keep_step``. Comment and header lines stay. Returns the rows dropped. Written through
    a temporary file and a rename; a missing file or a ``print`` file is left alone (0)."""
    path = Path(path)
    if kind == "print" or not path.is_file():
        return 0
    if kind == "dcd":                       # OpenMM runs' frames (md.dcd: binary, fixed-size frames)
        from . import dcd

        return dcd.truncate(path, step, keep_step)

    def beyond(s: int | None) -> bool:
        return s is not None and (s > step or (s == step and not keep_step))

    tmp = path.with_name(path.name + ".cut.tmp")
    dropped = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as src, \
            open(tmp, "w", encoding="utf-8", newline="") as dst:
        if kind == "dump":
            frame: list[str] = []
            for line in src:
                if line.startswith("ITEM: TIMESTEP") and frame:
                    dropped += _flush_frame(frame, dst, beyond)
                    frame = []
                frame.append(line)
            if frame:
                dropped += _flush_frame(frame, dst, beyond)
        elif kind == "ave/chunk":
            want = 0
            skip = False
            for line in src:
                s = line.strip()
                if not s or s.startswith("#"):
                    if not skip:
                        dst.write(line)
                    continue
                if want == 0:
                    t = s.split()
                    try:
                        want = int(float(t[1]))
                    except (IndexError, ValueError):
                        want = 0
                    skip = beyond(_step_of(s))
                    dropped += skip
                    if not skip:
                        dst.write(line)
                    continue
                want -= 1
                if not skip:
                    dst.write(line)
        else:                               # ave/time and any "step values..." file
            for line in src:
                s = line.strip()
                if not s or s.startswith("#"):
                    dst.write(line)
                    continue
                if beyond(_step_of(s)):
                    dropped += 1
                    continue
                dst.write(line)
    os.replace(tmp, path)
    return dropped


def _flush_frame(frame: list[str], dst, beyond) -> int:
    step = _step_of(frame[1]) if len(frame) > 1 and frame[0].startswith("ITEM: TIMESTEP") else None
    if beyond(step):
        return 1
    dst.writelines(frame)
    return 0


def cut_outputs(folder: str | Path, part: Part, step: int) -> dict[str, int]:
    """``cut_at`` for every output of ``part`` before a resume at ``step``: {file: rows dropped}."""
    out = {}
    for o in part.outputs:
        out[o.name] = cut_at(Path(folder) / o.name, o.kind, step, keep_step=not o.rewrites_start)
    return out


# ------------------------------------------------------------------------------------------- merge by step
def merge_by_step(rows: Iterable[Sequence[float]]) -> list[Sequence[float]]:
    """Rows keyed by their first column (a step), the last occurrence of each step kept, in step order: what a
    reader does with a file appended to by resumed segments that were not cut (``cut_at``)."""
    latest: dict[int, Sequence[float]] = {}
    for r in rows:
        latest[int(r[0])] = r
    return [latest[k] for k in sorted(latest)]
