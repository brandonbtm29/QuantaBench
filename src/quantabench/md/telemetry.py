"""Live telemetry of an MD run, read from the engine's own files (SIM-1). Nothing here asks LAMMPS anything.

``LogTail`` follows a growing LAMMPS log: it reads only what was appended since the last poll (a partial last line
waits for the next one), and keeps the thermo header, the latest rows, the ``Performance:`` lines, the errors and the
tags the QuantaBench inputs print (``QB_PAUSED``, ``QB_ERROR``, ``QB_BULK done``, ...). ``SpeedMeter`` turns
(wall time, step) samples into steps/s and ns/day over a sliding window; ``alerts`` lists what is worth a look:

* an ERROR or ``QB_ERROR`` line (lost atoms, a GPU neighbor-list failure, a missing style);
* the temperature off its target (the mean of the latest rows, once a stage has run for a while);
* the run stopped writing: the log has not grown for a long time while LAMMPS still runs.

``summarize_log`` reads a finished segment's log in one go: its first and last step, the longest run's
``Performance:`` line, a pause, a done tag, errors. Pure Python.
"""

from __future__ import annotations

import collections
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
PERF_RE = re.compile(rf"^Performance:\s*({_NUM}) ns/day,\s*({_NUM}) hours/ns,\s*({_NUM}) timesteps/s")
LOOP_RE = re.compile(rf"^Loop time of ({_NUM}) on (\d+) procs for (\d+) steps with (\d+) atoms")
PAUSED_RE = re.compile(r"^QB_PAUSED (\w+) step (\d+)")
KEEP_ROWS = 400


def _is_row(tokens: list[str], width: int) -> bool:
    if len(tokens) != width or not re.fullmatch(r"-?\d+", tokens[0]):
        return False
    try:
        for t in tokens[1:]:
            float(t)
    except ValueError:
        return False
    return True


@dataclass
class LogState:
    header: list[str] = field(default_factory=list)
    rows: collections.deque = field(default_factory=lambda: collections.deque(maxlen=KEEP_ROWS))
    first_step: int | None = None
    last_step: int | None = None
    runs: list[dict] = field(default_factory=list)        # one per "Loop time" + "Performance" pair
    errors: list[str] = field(default_factory=list)
    warnings: int = 0
    paused_step: int | None = None
    tags: list[str] = field(default_factory=list)         # QB_* lines (other than QB_PAUSED / QB_ERROR)
    grew: float = 0.0                                     # wall time the file last grew (time.time())
    size: int = 0

    def value(self, column: str, row: int = -1) -> float | None:
        if column not in self.header or not self.rows:
            return None
        return float(self.rows[row][self.header.index(column)])

    def mean(self, column: str, n: int) -> float | None:
        if column not in self.header or not self.rows:
            return None
        i = self.header.index(column)
        vals = [r[i] for r in list(self.rows)[-n:]]
        return sum(vals) / len(vals)

    def has_tag(self, tag: str) -> bool:
        return any(t.startswith(tag) for t in self.tags)

    @property
    def best_run(self) -> dict | None:
        """The run with the most steps (a segment's production run, not its ``run 0`` checks)."""
        runs = [r for r in self.runs if r.get("ns_per_day")]
        return max(runs, key=lambda r: r.get("steps", 0)) if runs else None


class LogTail:
    """Follow a LAMMPS log as it grows. ``poll()`` returns the number of new thermo rows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state = LogState()
        self._offset = 0
        self._partial = ""
        self._loop: dict | None = None

    def poll(self) -> int:
        try:
            size = self.path.stat().st_size
        except OSError:
            return 0
        if size < self._offset:             # rewritten (a rerun): start over
            self.__init__(self.path)
        if size == self._offset:
            return 0
        with open(self.path, "rb") as fh:
            fh.seek(self._offset)
            chunk = fh.read(size - self._offset)
        self._offset = size
        self.state.size = size
        self.state.grew = time.time()
        text = self._partial + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        self._partial = lines.pop()          # "" when the chunk ended with a newline
        return sum(self._line(ln.rstrip("\r")) for ln in lines)

    def _line(self, line: str) -> int:
        st = self.state
        s = line.strip()
        if not s:
            return 0
        tok = s.split()
        if tok[0] == "Step":
            st.header = tok
            return 0
        if st.header and _is_row(tok, len(st.header)):
            row = [float(x) for x in tok]
            st.rows.append(row)
            step = int(tok[0])
            st.first_step = step if st.first_step is None else st.first_step
            st.last_step = step
            return 1
        m = LOOP_RE.match(s)
        if m:
            self._loop = {"loop_s": float(m.group(1)), "procs": int(m.group(2)), "steps": int(m.group(3)),
                          "atoms": int(m.group(4))}
            return 0
        m = PERF_RE.match(s)
        if m:
            run = dict(self._loop or {})
            run.update(ns_per_day=float(m.group(1)), hours_per_ns=float(m.group(2)), timesteps_per_s=float(m.group(3)))
            st.runs.append(run)
            self._loop = None
            return 0
        if s.startswith("ERROR") or s.startswith("QB_ERROR"):
            st.errors.append(s[:300])
            return 0
        if s.startswith("WARNING"):
            st.warnings += 1
            return 0
        m = PAUSED_RE.match(s)
        if m:
            st.paused_step = int(m.group(2))
            return 0
        if s.startswith("QB_") and not s.startswith("QB_RUN0"):
            st.tags.append(s[:200])
        return 0


def summarize_log(path: str | Path) -> LogState:
    """Everything ``LogTail`` keeps, for a whole (finished) log."""
    t = LogTail(path)
    t.poll()
    if t._partial:
        t._line(t._partial)
    return t.state


def read_thermo(path: str | Path) -> list[tuple[list[str], list[list[float]]]]:
    """Every thermo block of a LAMMPS log: [(header, rows)], for plots and checks."""
    out: list[tuple[list[str], list[list[float]]]] = []
    head: list[str] | None = None
    rows: list[list[float]] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        tok = line.split()
        if tok and tok[0] == "Step":
            if head and rows:
                out.append((head, rows))
            head, rows = tok, []
            continue
        if head and _is_row(tok, len(head)):
            rows.append([float(x) for x in tok])
        elif head and line.startswith("Loop time"):
            if rows:
                out.append((head, rows))
            head, rows = None, []
    if head and rows:
        out.append((head, rows))
    return out


# ------------------------------------------------------------------------------------------- speed
class SpeedMeter:
    """steps/s over the last ``window_s`` of (wall time, step) samples; ns/day for a time step."""

    def __init__(self, window_s: float = 600.0):
        self.window_s = float(window_s)
        self.samples: collections.deque = collections.deque()

    def add(self, step: int | None, t: float | None = None) -> None:
        if step is None:
            return
        t = time.time() if t is None else t
        if self.samples and step < self.samples[-1][1]:      # a new segment or a rerun: the counter went back
            self.samples.clear()
        if self.samples and step == self.samples[-1][1]:
            return
        self.samples.append((t, int(step)))
        while len(self.samples) > 2 and t - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def steps_per_s(self) -> float | None:
        if len(self.samples) < 2:
            return None
        (t0, s0), (t1, s1) = self.samples[0], self.samples[-1]
        if t1 <= t0 or s1 <= s0:
            return None
        return (s1 - s0) / (t1 - t0)

    def ns_per_day(self, timestep_fs: float) -> float | None:
        r = self.steps_per_s()
        return None if r is None else r * float(timestep_fs) * 86400.0 / 1e6


def eta_seconds(steps_left: list[tuple[int, float]], steps_per_s: float | None) -> float | None:
    """Seconds to run ``steps_left`` [(steps, timestep)] at ``steps_per_s`` (the cost of a step does not depend on
    its length in fs)."""
    if not steps_per_s or steps_per_s <= 0:
        return None
    return sum(n for n, _dt in steps_left) / steps_per_s


def format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return ""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 3 * 86400:
        h, m = divmod(int(seconds) // 60, 60)
        return f"{h} h {m:02d} min"
    return f"{seconds / 86400:.1f} days"


# ------------------------------------------------------------------------------------------- alerts
def alerts(state: LogState, *, target_T: float | None = None, running: bool = True,
           stall_s: float = 1800.0, now: float | None = None, min_rows: int = 20) -> list[str]:
    """What is worth a look in a running (or finished) segment's log."""
    out = [f"error: {e}" for e in state.errors[-3:]]
    if target_T and len(state.rows) >= min_rows:
        t = state.mean("Temp", min_rows)
        if t is not None and abs(t - target_T) > max(10.0, 0.05 * target_T):
            out.append(f"temperature off target: {t:.1f} K over the last {min_rows} thermo rows (target "
                       f"{target_T:g} K)")
    if running and state.grew:
        idle = (time.time() if now is None else now) - state.grew
        if idle > stall_s:
            out.append(f"the log has not grown for {idle / 60:.0f} min while LAMMPS runs")
    return out
