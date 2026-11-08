"""Dynamic CREST threads: a CREST search may also use the threads of idle lanes.

Lanes are planned once per run, each a process with a fixed thread count (``dist.lanes``). A running CREST
cannot change its thread count, but each CREST call is a new program whose ``-T`` is chosen when it starts. So
the parent of a parallel run (``lanes.run_parallel``) owns the run's thread budget (the threads of its CPU lanes,
including lanes left out because there are fewer molecules than lanes) and, when it hands a CPU lane a molecule,
also says how many threads that molecule's CREST call may use (``BuildConfig.crest_threads``). The extra threads
are the free threads of idle lanes. They go back when the CREST call ends (the molecule's next stage), and all of
the lane's threads when the molecule ends, fails, or its lane dies or is killed. A lane starts a molecule only when
its own threads are free, so the threads in use never exceed the budget. Everything after CREST (refinement,
properties) runs on the lane's own threads.

    ThreadBudget(budget)              the accounting: take(lane, base, extra), give_back(lane), release(lane)
    choose_width(cost, base, free, budget, rest) -> int     the policy below (a pure function)
    DynamicThreads(budget, cost_of)   one run's policy state: the budget, the running CREST calls, calibration

Policy (chosen with a discrete-event simulation of batches on the Mac's and NOVA's lane plans, docs/ENGINES.md):
a molecule gets more than its lane's threads only when it would set the end of the batch on its own: its
predicted CREST time at the lane's width is more than ``DOMINANCE`` (2) times the rest of the work spread over
the other threads. The size model is off by a factor of about 1.8 per molecule, and the factor 2 keeps a
molecule that only looks large from taking lanes the real straggler needs. Its width is then a whole number of
lanes' worth (base, 2 x base, ... up to the free threads): for each width the predicted end of the batch is the
larger of this CREST call at that width and the rest of the work spread over the threads left, and the
narrowest width wins unless a wider one predicts an end at least ``MARGIN`` (10 %) earlier. With nothing else
to do, it gets every whole lane free. Predicted CREST cost: ``size ** COST_POWER`` with ``speed.molecule_size``
(atoms with hydrogens; the fitted exponent is 3.9 for GFN-FF and 4.5 for GFN2 on the Mac), speedup
``(width / base) ** SPEEDUP_POWER``. The rest is the queued molecules' cost plus what the running CREST calls
have left (calibrated in seconds from the CREST calls already finished, else their whole cost).

In a large batch every lane is busy until the queue is nearly empty, so nothing is widened and nothing changes;
the policy widens a molecule that would otherwise set the end of the batch on its own (a 54-heavy-atom PFAS on a
2-thread lane), and gives the idle threads to the molecules of a batch smaller than its lanes.

A machine's own constants: ``quantabench-cli timings --fit`` fits time = A x atoms^p x threads^-q to the CREST
stages recorded in its run folders (``chem.telemetry``) and saves them per CREST level in
``<settings>/machines/cost-models/<machine>.json`` (``CostModel``, ``load_cost_model``). A run uses them when they
are there and pass ``usable``, else the built-in ``COST_POWER`` and ``SPEEDUP_POWER``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

#: predicted CREST time ~ size ** COST_POWER (size = atoms with hydrogens)
COST_POWER = 4.0
#: CREST speedup from base to w threads ~ (w / base) ** SPEEDUP_POWER (Mac 4 -> 13 threads: 0.48; NOVA 5 -> 23: 0.42;
#: Mac 2 -> 4: 0.9)
SPEEDUP_POWER = 0.6
#: a wider CREST call must predict an end of the batch at least this much earlier than the narrower one
MARGIN = 0.10
#: only a molecule whose predicted time at its lane's width exceeds this many times the rest's work over the other
#: threads is widened (the size model scatters x1.8 per molecule)
DOMINANCE = 2.0
#: calibration samples kept (seconds per cost unit of the finished CREST calls)
CALIBRATION_SAMPLES = 50
#: fitted constants outside these ranges are not used (a fit on odd data); the built-ins are used instead
POWER_RANGE = (1.0, 8.0)
SPEEDUP_RANGE = (0.0, 1.2)
#: folder of the fitted cost models, in the machine registry (``dist.profile.registry_dir``). A subfolder: the
#: registry reads every ``*.json`` of its own folder as a machine profile
COST_MODEL_DIR = "cost-models"


def crest_cost(size: float, power: float = COST_POWER) -> float:
    """Predicted CREST time of a molecule of ``size`` atoms (with hydrogens) at base width, in relative units."""
    return max(1.0, float(size)) ** power


def speedup(width: int, base: int, power: float = SPEEDUP_POWER) -> float:
    return (max(1, int(width)) / max(1, int(base))) ** power


def predicted_end(cost: float, width: int, base: int, budget: int, rest: float,
                  power: float = SPEEDUP_POWER) -> float:
    """The batch's predicted end (cost units) when this CREST call runs on ``width`` threads: the larger of its own
    time and ``rest`` (thread x cost units of everything else) spread over the threads left."""
    own = cost / speedup(width, base, power)
    left = budget - width
    if left <= 0:
        return own if rest <= 0 else math.inf
    return max(own, rest / left)


def choose_width(cost: float, base: int, free: int, budget: int, rest: float, power: float = SPEEDUP_POWER) -> int:
    """Threads for a CREST call of predicted ``cost`` started on a lane of ``base`` threads, with ``free`` threads
    free in a budget of ``budget`` and ``rest`` (thread x cost units) of other work to do. ``base`` unless the call
    would set the end of the batch on its own (``DOMINANCE``); then whole lanes' worth, the narrowest unless a
    wider width predicts an end ``MARGIN`` earlier. Never more than ``free``, never less than ``base``. With
    nothing else to do (``rest`` 0) the threads have no other use: all whole lanes' worth free. ``power``: the
    speedup exponent (``SPEEDUP_POWER``, or a machine's fitted one)."""
    base = max(1, int(base))
    if rest <= 0:
        return max(base, int(free) // base * base)
    if budget <= base or cost <= DOMINANCE * rest / (budget - base):
        return base
    best_w = base
    best = predicted_end(cost, base, base, budget, rest, power)
    w = 2 * base
    while w <= free:
        end = predicted_end(cost, w, base, budget, rest, power)
        if end < best * (1 - MARGIN):
            best_w, best = w, end
        w += base
    return best_w


# ------------------------------------------------------------------------------------- the cost model
@dataclass(frozen=True)
class CostModel:
    """CREST time = ``scale_s`` x size^``power`` x threads^-``speedup_power`` (size: atoms with hydrogens). The
    built-in one has no scale (relative units only); a fitted one (``quantabench-cli timings --fit``) predicts
    seconds too."""

    power: float = COST_POWER
    speedup_power: float = SPEEDUP_POWER
    scale_s: float | None = None
    source: str = "built-in"          # "built-in" | "fit <machine> <level> (<n> molecules, <date>)"
    scatter: float | None = None      # the fit's scatter factor per molecule (exp of the rms log residual)

    def cost(self, size: float) -> float:
        return crest_cost(size, self.power)

    def seconds(self, size: float, width: int) -> float | None:
        """Predicted seconds of a CREST call on ``width`` threads (None without a fitted scale)."""
        if not self.scale_s:
            return None
        return self.scale_s * max(1.0, float(size)) ** self.power * max(1, int(width)) ** -self.speedup_power

    def to_dict(self) -> dict:
        return asdict(self)


BUILT_IN = CostModel()


def crest_level(cfg) -> str:
    """The CREST level a cost model is kept for: 'gfnff quick', 'gfn2 quick', 'gfn2' (a full run), ... The solvent
    is not part of it (ALPB changes the scale, not the exponents, much)."""
    method = str(getattr(cfg, "crest_method", "gfn2") or "gfn2").strip().lower()
    method = "gfnff" if method == "gff" else method
    return f"{method}{' quick' if getattr(cfg, 'crest_quick', True) else ''}"


def cost_model_path(machine: str) -> Path:
    from .profile import registry_dir

    return registry_dir() / COST_MODEL_DIR / f"{machine}.json"


def usable(entry: dict) -> tuple[bool, str]:
    """Whether a stored fit may be used: finite constants within ``POWER_RANGE`` and ``SPEEDUP_RANGE``."""
    try:
        p, q = float(entry["power"]), float(entry["speedup_power"])
        a = entry.get("scale_s")
        a = None if a is None else float(a)
    except (KeyError, TypeError, ValueError):
        return False, "no power/speedup_power"
    if not (math.isfinite(p) and POWER_RANGE[0] <= p <= POWER_RANGE[1]):
        return False, f"power {p:g} outside {POWER_RANGE}"
    if not (math.isfinite(q) and SPEEDUP_RANGE[0] <= q <= SPEEDUP_RANGE[1]):
        return False, f"speedup power {q:g} outside {SPEEDUP_RANGE}"
    if a is not None and not (math.isfinite(a) and a > 0):
        return False, f"scale {a!r} not positive"
    return True, ""


def read_cost_models(machine: str) -> dict:
    """The file's {"machine", "updated", "models": {level: {...}}}; {} when missing or unreadable."""
    try:
        data = json.loads(cost_model_path(machine).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and isinstance(data.get("models"), dict) else {}


def load_cost_model(machine: str, level: str, log: Callable[[str], None] | None = None) -> CostModel:
    """``machine``'s fitted cost model for CREST ``level`` (``crest_level``), else ``BUILT_IN``. Never raises."""
    try:
        entry = read_cost_models(machine).get("models", {}).get(level)
        if not isinstance(entry, dict):
            return BUILT_IN
        ok, why = usable(entry)
        if not ok:
            if log is not None:
                log(f"Cost model of {machine} ({level}) not used: {why}; built-in constants")
            return BUILT_IN
        a = entry.get("scale_s")
        n = entry.get("n_molecules", "?")
        return CostModel(float(entry["power"]), float(entry["speedup_power"]), None if a is None else float(a),
                         f"fit {machine} {level} ({n} molecules, {str(entry.get('updated', ''))[:10]})",
                         entry.get("scatter") if isinstance(entry.get("scatter"), (int, float)) else None)
    except Exception:  # noqa: BLE001 - a broken file must not stop a run
        return BUILT_IN


def cost_model_for(cfg, log: Callable[[str], None] | None = None) -> CostModel | None:
    """This machine's cost model for a CREST run of ``cfg``; None when the run has no CREST search."""
    if getattr(cfg, "conformer_search", "") != "crest":
        return None
    try:
        from .profile import machine_name

        return load_cost_model(machine_name(), crest_level(cfg), log)
    except Exception:  # noqa: BLE001
        return BUILT_IN


class ThreadBudget:
    """Threads in use per lane (its own ``base`` plus ``extra`` borrowed for a CREST call) within ``budget``."""

    def __init__(self, budget: int):
        self.budget = max(0, int(budget))
        self.held: dict[str, list[int]] = {}          # lane -> [base, extra]

    def in_use(self) -> int:
        return sum(b + e for b, e in self.held.values())

    def free(self) -> int:
        return self.budget - self.in_use()

    def borrowed(self, lane: str) -> int:
        return self.held.get(lane, [0, 0])[1]

    def take(self, lane: str, base: int, extra: int = 0) -> None:
        """Lane ``lane`` starts a molecule on ``base`` threads plus ``extra`` borrowed ones. Raises ValueError
        rather than go over the budget (the caller checks ``free()`` first) or when the lane holds threads."""
        base, extra = int(base), max(0, int(extra))
        if lane in self.held:
            raise ValueError(f"lane {lane} already holds {sum(self.held[lane])} threads")
        if base + extra > self.free():
            raise ValueError(f"{base + extra} threads for lane {lane}, but only {self.free()} of {self.budget} free")
        self.held[lane] = [base, extra]

    def give_back(self, lane: str) -> int:
        """The lane's CREST call ended: its borrowed threads go back. Returns how many."""
        held = self.held.get(lane)
        if not held:
            return 0
        extra, held[1] = held[1], 0
        return extra

    def release(self, lane: str) -> int:
        """The lane's molecule ended (or the lane died): all its threads go back. Returns how many."""
        held = self.held.pop(lane, None)
        return sum(held) if held else 0


class DynamicThreads:
    """One parallel run's dynamic CREST threads (see the module docstring). ``cost_of(mol_id)``: the predicted CREST
    cost (``crest_cost`` of the molecule's size). The parent of the run is the only caller; ``now`` defaults to
    ``time.monotonic()``."""

    def __init__(self, budget: int, cost_of: Callable[[str], float], speedup_power: float = SPEEDUP_POWER):
        self.pool = ThreadBudget(budget)
        self.cost_of = cost_of
        self.power = float(speedup_power)
        self.running: dict[str, dict] = {}     # lane -> {mol, base, width, t0, crest}
        self.samples: list[float] = []         # seconds per cost unit at base width, finished CREST calls
        self.widened = 0                       # CREST calls given borrowed threads
        self.borrowed_thread_s = 0.0           # borrowed threads x seconds

    @property
    def budget(self) -> int:
        return self.pool.budget

    def can_start(self, base: int) -> bool:
        return self.pool.free() >= base

    def _seconds_per_cost(self) -> float | None:
        if not self.samples:
            return None
        s = sorted(self.samples[-CALIBRATION_SAMPLES:])
        return s[len(s) // 2]

    def predicted_seconds(self, mol_id: str, base: int, width: int) -> float | None:
        """Seconds of ``mol_id``'s CREST call on ``width`` threads, calibrated on this run's finished CREST calls;
        None before the first one has finished."""
        k = self._seconds_per_cost()
        if k is None:
            return None
        return k * self.cost_of(mol_id) / speedup(width, base, self.power)

    def running_work(self, base: int, now: float) -> float:
        """Thread x cost units the running CREST calls have left (their whole cost before any CREST call has
        finished; afterwards what is left of their predicted time)."""
        k = self._seconds_per_cost()
        work = 0.0
        for r in self.running.values():
            if not r["crest"]:
                continue
            c = self.cost_of(r["mol"]) / speedup(r["width"], r["base"], self.power)   # its time at its width
            if k is not None:
                c = max(0.0, c - (now - r["t0"]) / k)
            work += c * r["width"]
        return work

    def width(self, base: int, mol_id: str, queued: Iterable[str], now: float | None = None) -> int:
        """CREST threads for ``mol_id`` started now on a lane of ``base`` threads; ``queued``: the molecules
        still waiting (not this one). The queue is only read when at least two lanes' worth is free."""
        free = self.pool.free()
        if free < 2 * base:
            return base
        now = time.monotonic() if now is None else now
        rest = sum(self.cost_of(m) for m in queued) * base + self.running_work(base, now)
        return choose_width(self.cost_of(mol_id), base, free, self.pool.budget, rest, self.power)

    def start(self, lane: str, base: int, width: int, mol_id: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.pool.take(lane, base, width - base)
        self.running[lane] = {"mol": mol_id, "base": base, "width": width, "t0": now, "crest": True}
        if width > base:
            self.widened += 1

    def crest_started(self, lane: str, now: float | None = None) -> None:
        """The lane's molecule reached its conformer stage: CREST starts now (the clock for the calibration)."""
        r = self.running.get(lane)
        if r is not None and r["crest"]:
            r["t0"] = time.monotonic() if now is None else now

    def crest_done(self, lane: str, now: float | None = None) -> tuple[int, float]:
        """The lane's molecule left its conformer stage: the borrowed threads go back. Returns (threads given
        back, seconds the CREST stage took); (0, 0.0) when it was already done."""
        r = self.running.get(lane)
        if r is None or not r["crest"]:
            return 0, 0.0
        now = time.monotonic() if now is None else now
        r["crest"] = False
        seconds = max(0.0, now - r["t0"])
        cost = self.cost_of(r["mol"])
        if cost > 0 and seconds > 0:
            self.samples.append(seconds * speedup(r["width"], r["base"], self.power) / cost)
        back = self.pool.give_back(lane)
        self.borrowed_thread_s += back * seconds
        return back, seconds

    def finish(self, lane: str, now: float | None = None) -> int:
        """The lane's molecule ended, failed or was lost with its lane: every thread it held goes back."""
        r = self.running.pop(lane, None)
        if r is not None and r["crest"] and self.pool.borrowed(lane):
            now = time.monotonic() if now is None else now
            self.borrowed_thread_s += self.pool.borrowed(lane) * max(0.0, now - r["t0"])
        return self.pool.release(lane)

    def summary(self) -> str:
        if not self.widened:
            return (f"Dynamic threads: no CREST call needed borrowed threads (every lane had work; budget "
                    f"{self.budget} threads)")
        return (f"Dynamic threads: {self.widened} CREST call(s) ran on borrowed threads "
                f"({self.borrowed_thread_s / 3600:.2f} borrowed thread-hours; budget {self.budget} threads)")


def setting() -> bool:
    """``settings.resources.dynamic_threads`` (on unless switched off)."""
    from .. import settings as settings_mod

    try:
        return bool(settings_mod.load().resources.dynamic_threads)
    except Exception:  # noqa: BLE001 - unreadable settings: the default
        return True
