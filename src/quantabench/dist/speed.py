"""Speed-aware lanes: the last molecules of a batch go to the lanes that finish them first (DIST-5).

Lanes pull molecules first come, first served. When lanes differ a lot in speed (``--lanes auto`` for AIMNet2
on BTM NOVA: one GPU lane about 10x faster per molecule than each of four CPU lanes), a CPU lane can start one
of the last molecules when the GPU lane would have finished it, and everything queued behind it, long before.
For a small batch the run can then take longer than the GPU lane alone.

The rule (``should_dispatch``) for an idle lane L and the molecules q1..qn still queued, in queue order:

    L takes q1 unless the lanes clearly faster than L (at least MIN_SPEEDUP times, per size unit) would
    finish *all* of q1..qn before L would finish q1 alone (with HOLD_MARGIN to spare). The faster lanes are
    simulated taking q1..qn in order, each molecule going to the one that finishes it first, starting from
    when each is free (its current molecule's expected remainder).

For molecules of one size and idle fast lanes this is the rule of thumb "a lane T times slower leaves the
last ~0.9 T x (number of faster lanes) molecules to them"; the simulation generalizes it to fast lanes that
are busy and to molecules of different sizes. A lane that holds back simply stays idle; the pool asks again
every ``lanes.POLL_S`` and whenever a lane reports.

Estimates (``SpeedPolicy``), in seconds per size unit (atoms including H, ``molecule_size``), per lane
class (device, threads): lanes of one class share one estimate, so lanes of the same kind never hold back
for each other.

1. measured: the mean of the last ``WINDOW`` (per lane) successful molecules of the class, timed in the
   parent from hand-out to result. A lane's first molecule (model loading, kernel compilation) only counts
   until the class has later ones.
2. else this machine's benchmark (``dist.profile`` registry, ``throughput["<method>@<device>"]``, see
   ``registry_seeds``), scaled to this run: the benchmark protocol is shorter than a production build, so
   the factor is the median measured/benchmark ratio of the classes that have both; before any molecule
   has finished, the smallest factor under which no running molecule is overdue (at most COLD_SCALE_MAX).
3. else unknown: a lane of unknown speed never holds back and is never counted on (today's behavior).

Never stuck: only lanes that are ready (or still starting for the first time, at most STARTUP_GRACE_S)
are counted on, so a lane that died, is restarting or was retired releases the molecules held for it at
once. A lane whose molecule runs past its estimate is assumed to need as long again as it is late, so a
stalled lane stops being counted on after about one slow-lane molecule. The fastest available lane never
holds back, so the queue always drains.
"""

from __future__ import annotations

import itertools
import statistics
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

#: hold back only when the faster lanes finish everything this fraction earlier than the slow lane would.
#: 0 is the exact rule, and leaning towards holding back is the cheaper mistake: a wrong hold costs at most
#: one fast-lane molecule per held lane, a wrong start up to a whole slow-lane molecule (simulated in
#: tests/test_speed.py, 0 beat 0.1 for GPU lanes 4-20x faster)
HOLD_MARGIN = 0.0
#: a lane counts as faster only when it is at least this many times faster (per size unit): lanes within
#: 1.5x of each other never hold back for each other
MIN_SPEEDUP = 1.5
#: molecules per lane in the running mean of a lane class
WINDOW = 8
#: a lane still starting for the first time is counted on for this long (lane start-up is 2-10 s)
STARTUP_GRACE_S = 60.0
#: before any molecule has finished, benchmark times are scaled by at most this (production / benchmark)
COLD_SCALE_MAX = 100.0
#: at most this many queued molecules are simulated per decision (beyond that: take the molecule)
MAX_SIMULATED = 10_000
#: size of a molecule that cannot be read (atoms, a PFAS-sized molecule)
DEFAULT_SIZE = 20.0

READY, STARTING, DOWN = "ready", "starting", "down"


# ------------------------------------------------------------------------------------ the decision
@dataclass(frozen=True)
class LaneView:
    """What the decision needs to know about one lane at one moment."""
    name: str
    per_unit: float | None           # estimated seconds per size unit; None = speed unknown
    busy_size: float | None = None   # size of the molecule it is running; None = idle (or starting)
    elapsed: float = 0.0             # seconds that molecule has been running
    available: bool = True           # ready, or starting for the first time: it will take work soon


def residual(v: LaneView) -> float:
    """Expected seconds until lane ``v`` is free. A molecule running past its estimate is assumed to need
    as long again as it is late, so a stalled lane is counted on less and less."""
    if v.busy_size is None or v.per_unit is None:
        return 0.0
    late = v.elapsed - v.per_unit * v.busy_size
    return -late if late <= 0 else late


def faster_lanes(lane: LaneView, others: Iterable[LaneView], min_speedup: float = MIN_SPEEDUP) -> list[LaneView]:
    """The available lanes of known speed at least ``min_speedup`` times faster than ``lane``."""
    if lane.per_unit is None or lane.per_unit <= 0:
        return []
    return [v for v in others if v.available and v.per_unit is not None and v.per_unit > 0
            and v.per_unit * min_speedup <= lane.per_unit]


def should_dispatch(lane: LaneView, others: Iterable[LaneView], queue_sizes: Iterable[float], *,
                    margin: float = HOLD_MARGIN, min_speedup: float = MIN_SPEEDUP,
                    limit: int = MAX_SIMULATED) -> bool:
    """Whether idle ``lane`` should take the next queued molecule now (see the module docstring).

    ``others``: every other lane of the pool; ``queue_sizes``: the sizes of the queued molecules in queue
    order (read lazily, so a long queue costs only the molecules the faster lanes could finish in time).
    False only when the faster lanes would finish every queued molecule before ``lane`` finished the first.
    """
    sizes = iter(queue_sizes)
    head = next(sizes, None)
    if head is None:
        return False                       # nothing to take
    fast = faster_lanes(lane, others, min_speedup)
    if not fast:
        return True                        # unknown speed, or no faster lane to leave it to
    deadline = (1.0 - margin) * lane.per_unit * max(float(head), 1e-9)
    free = [residual(v) for v in fast]
    rates = [float(v.per_unit) for v in fast]
    lanes_ix = range(len(fast))
    for n, size in enumerate(itertools.chain((head,), sizes)):
        if n >= limit:
            return True
        size = max(float(size), 1e-9)
        best = min(lanes_ix, key=lambda i: free[i] + rates[i] * size)
        done = free[best] + rates[best] * size
        if done >= deadline:
            return True                    # the faster lanes cannot do it all in time: this lane helps
        free[best] = done
    return False


# ------------------------------------------------------------------------------------- estimates
def molecule_size(inp) -> float:
    """Size proxy of a ``MoleculeInput`` for the time per molecule: atoms including hydrogens."""
    try:
        from rdkit import rdBase

        with rdBase.BlockLogs():           # a bad SMILES is reported by its lane, not here
            mol = inp.to_mol()
        return float(mol.GetNumAtoms() + sum(a.GetTotalNumHs() for a in mol.GetAtoms())) or DEFAULT_SIZE
    except Exception:  # noqa: BLE001 - a molecule that cannot be read fails in its lane; any size will do
        return DEFAULT_SIZE


def bench_size(bench_set: str = "pfas") -> float:
    """Mean size of a benchmark set (``profile.BENCH_SETS``), to turn molecules/hour into s per size unit."""
    from ..chem.inputs import MoleculeInput
    from .profile import BENCH_SETS

    items = BENCH_SETS.get(bench_set) or BENCH_SETS["pfas"]
    sizes = [molecule_size(MoleculeInput(id=i, smiles=s)) for i, s in items]
    return sum(sizes) / len(sizes)


def registry_seeds(lanes: Sequence, method: str, profile=None, solvent: str = "") -> dict[str, float | None]:
    """{lane name: benchmark seconds per size unit} from this machine's registry profile (``machines
    --refresh --bench <method>``) for the lane's device, None where the method was not measured there.
    ``solvent``: the run's; a run in water only uses ``<method>+water`` keys (``profile.bench_method``)."""
    from . import profile as profile_mod

    out: dict[str, float | None] = {x.name: None for x in lanes}
    try:
        p = profile if profile is not None else profile_mod.load_profile(profile_mod.machine_name())
        if p is None or not method:
            return out
        name = profile_mod.bench_method(method, solvent)
        for x in lanes:
            rate = profile_mod.throughput_for(p, name, x.device)
            if rate and rate > 0:
                unit = bench_size(profile_mod.set_of_key(p, f"{name}@{x.device}"))
                out[x.name] = 3600.0 / float(rate) / unit
    except Exception:  # noqa: BLE001 - a missing or unreadable registry means unknown speeds
        return {x.name: None for x in lanes}
    return out


def _class_of(lane) -> tuple[str, int]:
    return (str(getattr(lane, "device", "cpu")), int(getattr(lane, "threads", 1) or 1))


class SpeedPolicy:
    """Per-lane-class speed estimates and the hold-back decision for one pool (``lanes.run_parallel``).

    No processes and an injectable clock, so the whole policy runs under a simulated clock in the tests.
    The pool reports ``started``/``finished``/``lost`` for every molecule it hands out and asks ``allow``
    before handing one to an idle lane, with each lane's state (READY, STARTING or DOWN).
    """

    def __init__(self, lanes: Sequence, seeds: Mapping[str, float | None] | None = None, *,
                 clock: Callable[[], float] = time.monotonic, margin: float = HOLD_MARGIN,
                 min_speedup: float = MIN_SPEEDUP, window: int = WINDOW):
        self.names = [x.name for x in lanes]
        self.cls = {x.name: _class_of(x) for x in lanes}
        self.clock = clock
        self.margin, self.min_speedup = margin, min_speedup
        counts: dict[tuple[str, int], int] = {}
        for c in self.cls.values():
            counts[c] = counts.get(c, 0) + 1
        self.seed: dict[tuple[str, int], float] = {}
        for name, value in (seeds or {}).items():
            if name in self.cls and value is not None and value > 0:
                self.seed.setdefault(self.cls[name], float(value))
        self.window = {c: deque(maxlen=max(1, window) * n) for c, n in counts.items()}
        self.firsts: dict[tuple[str, int], list[float]] = {c: [] for c in counts}
        self.timed: set[str] = set()                    # lanes that have finished a molecule
        self.busy: dict[str, tuple[float, float]] = {}  # lane -> (size, start time)
        self.done: dict[str, int] = {}

    # -- what the pool reports
    def started(self, name: str, size: float) -> None:
        self.busy[name] = (max(float(size), 1e-9), self.clock())

    def finished(self, name: str, ok: bool = True) -> None:
        """Lane ``name`` returned its molecule; only successful ones are timed (a quick failure says
        nothing about the lane's speed)."""
        job = self.busy.pop(name, None)
        if job is None or not ok:
            return
        size, t0 = job
        per_unit = max(self.clock() - t0, 0.0) / size
        c = self.cls[name]
        if name in self.timed:
            self.window[c].append(per_unit)
        else:
            self.timed.add(name)
            self.firsts[c].append(per_unit)
        self.done[name] = self.done.get(name, 0) + 1

    def lost(self, name: str) -> None:
        """Lane ``name`` died (or was stopped) with its molecule: no time recorded."""
        self.busy.pop(name, None)

    # -- estimates
    def measured(self, c: tuple[str, int]) -> float | None:
        w = self.window.get(c)
        if w:
            return sum(w) / len(w)
        f = self.firsts.get(c)
        return sum(f) / len(f) if f else None

    def scale(self, now: float | None = None) -> float:
        """Production seconds per benchmark second: the median measured/benchmark ratio of the classes that
        have both; before that, the smallest factor >= 1 under which no running molecule is overdue."""
        ratios = [m / self.seed[c] for c in self.window if c in self.seed and (m := self.measured(c))]
        if ratios:
            return statistics.median(ratios)
        now = self.clock() if now is None else now
        factor = 1.0
        for name, (size, t0) in self.busy.items():
            s = self.seed.get(self.cls[name])
            if s:
                factor = max(factor, (now - t0) / (s * size))
        return min(factor, COLD_SCALE_MAX)

    def estimate(self, name: str, scale: float | None = None) -> float | None:
        """Seconds per size unit for lane ``name``: measured, else benchmark x ``scale``, else None."""
        c = self.cls[name]
        m = self.measured(c)
        if m is not None:
            return m
        s = self.seed.get(c)
        if s is None:
            return None
        return s * (self.scale() if scale is None else scale)

    def source(self, name: str) -> str:
        c = self.cls[name]
        if self.measured(c) is not None:
            return "measured"
        return "benchmark" if c in self.seed else "unknown"

    def views(self, states: Mapping[str, str], now: float | None = None) -> dict[str, LaneView]:
        now = self.clock() if now is None else now
        k = self.scale(now)
        out = {}
        for name in self.names:
            state = states.get(name, DOWN)
            per_unit = self.estimate(name, k)
            if state == DOWN:
                out[name] = LaneView(name, per_unit, available=False)
            elif name in self.busy:
                size, t0 = self.busy[name]
                out[name] = LaneView(name, per_unit, size, now - t0)
            else:
                out[name] = LaneView(name, per_unit)
        return out

    # -- the decision
    def allow(self, name: str, queue_sizes: Iterable[float], states: Mapping[str, str]) -> bool:
        """Whether idle lane ``name`` should take the next molecule now (``should_dispatch``)."""
        views = self.views(states)
        return should_dispatch(views[name], [v for n, v in views.items() if n != name], queue_sizes,
                               margin=self.margin, min_speedup=self.min_speedup)

    # -- what the log says
    def hold_message(self, held: Sequence[str], n_left: int, typical_size: float,
                     states: Mapping[str, str]) -> str:
        """One line for the log when lanes start holding back."""
        views = self.views(states)
        fast: dict[str, LaneView] = {}
        for name in held:
            for v in faster_lanes(views[name], [x for n, x in views.items() if n != name], self.min_speedup):
                fast.setdefault(v.name, v)

        def each(names: Sequence[str]) -> str:
            secs = [views[n].per_unit * typical_size for n in names if views[n].per_unit is not None]
            if not secs:
                return ", ".join(names)
            lo, hi = min(secs), max(secs)
            when = f"~{_fmt_s(lo)}" if hi < 1.15 * lo else f"~{_fmt_s(lo)}-{_fmt_s(hi)}"
            src = sorted({self.source(n) for n in names})
            return f"{', '.join(names)} ({when} per molecule, {'/'.join(src)})"

        faster = [n for n in self.names if n in fast]
        leave, finish = ("leaves" if len(held) == 1 else "leave"), ("finishes" if len(faster) == 1 else "finish")
        return (f"Speed-aware lanes: {each(list(held))} {leave} the last {n_left} molecule(s) to {each(faster)}, "
                f"which {finish} them sooner (off: --no-speed-aware-lanes, or Settings > Compute)")

    def summary(self, typical_size: float) -> str:
        """'Lane speeds: gpu0 5.1 s (measured), cpu1-cpu4 48 s (benchmark) per molecule of ~22 atoms'."""
        k = self.scale()
        groups: dict[tuple[str, int], list[str]] = {}
        for name in self.names:
            groups.setdefault(self.cls[name], []).append(name)
        parts = []
        for names in groups.values():
            est = self.estimate(names[0], k)
            label = names[0] if len(names) == 1 else f"{names[0]}-{names[-1]}"
            if est is None:
                parts.append(f"{label} unknown")
            else:
                parts.append(f"{label} {_fmt_s(est * typical_size)} ({self.source(names[0])})")
        return f"Lane speeds: {', '.join(parts)} per molecule of ~{typical_size:.0f} atoms"


def _fmt_s(seconds: float) -> str:
    if seconds >= 10:
        return f"{seconds:.0f} s"
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds:.2f} s"
