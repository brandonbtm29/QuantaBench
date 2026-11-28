"""Device planner (SCHED-3, PR 1): the device assignment that finishes a job soonest on a machine, and whether two
jobs finish sooner side by side than one after the other.

    quantabench-cli jobs plan JOB [JOB ...] [--machine NAME] [--threads N] [--timings [PATH ...]]
                              [--slowdown F] [--include-unverified] [--json]

Brandon (2026-11-23): "The point of the scheduler is to compare what each job needs, then optimize and make one job go
faster by dedicating just GPU, CPU, or both." (2026-11-22): "If one type of job is better suited for the CPU and
another for a GPU, then we can tell the user that both jobs can run efficiently at the same time."
(docs/SIMULATE_DESIGN.md, section 7.) This module only plans and explains; it starts nothing.

A JOB is an MD run folder (``run.json`` or ``liquid_setup.json``: its atoms, the ns still to run, its dispersion,
its engine once it has started, and its earlier segments) or ``kind:key=value,...``::

    md:atoms=9780,ns=13,dispersion=cutoff16[,engine=lammps|openmm][,geometry=slab|bulk]
    build:method=aimnet2,molecules=5000[,solvent=water][,atoms=30][,threads=N]
    crest:molecules=1318,atoms=40[,rotors=10][,method=gfnff][,quick=yes][,threads=10]

(each also takes ``name=...``; ``threads=N``: the job runs on exactly N threads, as its lane spec says;
``device=cpu|gpu``: only there; ``running=yes``: it runs already, so one after the other it comes first: give it its
threads and device too, and the work it has left).

What a job can use (``evaluate``, ``assignments``)
--------------------------------------------------
An assignment is the CPU only on N threads, the GPU only (with the host threads it needs: 1 for MD, the GPU lane's
``lanes.GPU_LANE_THREADS`` for a build), or the GPU plus N threads. Its time to finish comes from the machine's
stored data, best source first:

* **MD** (``md_throughput``, MD-BENCH-1/2): each measured variant of the job's dispersion, scaled to its atoms and
  box (``md.bench.estimate_hours``: ``scaled_rate``, a cutoff from the nearest measured one). This run's own
  segments on this machine (``run.json``, SIM-1) replace the benchmark for their variant: they timed this system.
  ``cpu-*`` (LAMMPS OpenMP) and ``openmm-cpu-*`` use the CPU only; ``openmm-gpu-*`` the GPU and one CPU thread;
  ``gpu-*`` (LAMMPS' GPU package, ``pair/only on``) the GPU *and* threads: the pair style runs on the GPU, PPPM and
  the bonded terms on OpenMP threads, 75-86 % of the step in the pilots' GPU runs. So its time per step is
  g + (1 - g) x A(t) / A(t_measured): g the step's wait for the GPU (the record's ``breakdown["Pair"]``, else
  ``GPU_WAIT_SHARE``), A(t) = s + (1 - s) / t Amdahl's law for the CPU work with ``MD_SERIAL_FRACTION``. CPU-only
  variants scale by A alone. Threads beyond the physical cores add nothing (LAMMPS' OpenMP styles gain little from
  SMT; the benchmark times at most the physical cores). Ewald dispersion (``lj/long/coul/long`` + ``pppm/disp``) has
  no LAMMPS GPU style; OpenMM's LJ-PME does.
* **Builds** (a molecule batch through the pipeline, ``profile.throughput``): batches run as lanes (``dist.lanes``),
  so a batch's rate is the sum of its lanes'. CPU lanes follow ``plan_auto``'s rules for the engine's kind (MMFF and
  in-process native code: one lane per physical core; PyTorch engines: 2-thread lanes on the physical cores;
  external programs: 2-thread lanes on all threads), without its memory cap. A lane of t threads runs at the
  benchmark's rate (one process on the benchmark's threads T) x (t / T)^q, q from ``THREAD_POWER``. A GPU lane runs
  at the benchmark's GPU rate, beside CPU lanes on the other threads ("GPU + N threads"). Learned rates from finished
  runs (``--timings``: ``dist.timings`` rows of this machine, the method and solvent, per lane device) replace the
  benchmark; molecule size scales both (seconds per atom). Without a benchmark the CPU falls back to
  ``plan.ESTIMATE_PER_CORE`` per physical core (rough), the GPU to unknown. Benchmark rates are those of its short
  protocol (4 conformers, 2 refined): a production run takes longer, but the ranking of the assignments holds.
  dxtb (``gfn1-xtb``, ``gfn2-xtb``) stays on the CPU unless its GPU rate was measured (dxtb GPU = on): 5-8x slower on
  the workstation's GPU, and never GPU and CPU lanes in one run (float32 next to float64).
* **CREST** (``crest:``, or a build whose conformer search is CREST): the CPU only (through WSL on Windows), in lanes of
  ``lanes.CREST_LANE_THREADS``, each molecule's time from this machine's fitted cost model (``timings --fit``:
  A x atoms^p x (1 + rotors)^r x threads^-q, ``dist.threads``) at the mean size; without a fit the time is unknown.

The choice (``plan_job``)
-------------------------
The fastest assignment with a known time; of those within ``TIE`` of it, the one with the fewest threads, then
without the GPU. With a GPU it also gives the **lean** assignment: the fewest threads within ``LEAN`` of the fastest
(GPU MD: the GPU plus a few threads), what the job takes when it shares the machine. Every option comes with its
numbers and where they come from, and the choice with why.

Side by side (``side_by_side``)
-------------------------------
For two jobs on one machine: one after the other (the shorter first, each on its fastest assignment) against every
split of the machine (at most one job per GPU, threads within the budget). A job keeps its assignment to its end:
lanes are planned once per run and an MD segment keeps its threads. Side by side is advised only when it finishes
both jobs ``SIDE_MARGIN`` sooner, or neither job later than one after the other (within ``TIE``) and the pair sooner;
otherwise one after the other, with both sets of times. The slowdown of two jobs sharing memory bandwidth and heat is
not measured yet: ``slowdown`` (``--slowdown 0.1``: each 10 % slower side by side) shows what it would change.

Not yet (SCHED-3 PR 2): the ledger in the GUI queue that starts a job when its profile fits beside the running ones,
RAM and GPU memory (MON-1's samples) in the profiles, the measured slowdown, learning an MD run's thread scaling from
its segments, and a job taking the threads the other one frees.
"""

from __future__ import annotations

import functools
import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: GPU MD (LAMMPS' GPU package, pair/only on): the share of a step that waits for the GPU, when the benchmark record
#: has no timing breakdown. Pilots B and C (docs/SIMULATE_DESIGN.md section 7): "Pair" 9 % and 4 %
GPU_WAIT_SHARE = 0.07
#: Amdahl serial fraction of an MD step's CPU work (PPPM, bonded terms, neighbor lists) on OpenMP threads. From the
#: pilots' GPU runs' CPU use: B 237 % of 600 % on 6 threads (0.31), C 173 % of 400 % on 4 threads (0.44)
MD_SERIAL_FRACTION = 0.35
#: a build process's speed ~ threads^q (benchmarks time one process on many threads, lanes run 1-2-thread processes
#: side by side). The workstation, 2026-11-05 (``dist.lanes``): MMFF 12 x 1-thread lanes 3.6x one 12-thread process
#: (q 0.48); GFN1-xTB (dxtb) 6 x 2-thread lanes 3.4x (q 0.31)
THREAD_POWER = {"mmff": 0.48, "torch": 0.31}
DEFAULT_THREAD_POWER = 0.4
#: a thread beyond the physical cores (SMT) is worth this much of a core. The workstation (12 cores, 24 threads), tblite
#: lanes (``dist.lanes``): 24 x 1 ran 9 % faster than 12 x 1, 12 x 2 14 %: about a tenth of a core per extra thread
SMT_GAIN = 0.12
#: options within this fraction of the fastest count as as fast: the one that uses less is chosen
TIE = 0.03
#: the lean assignment: the fewest threads within this fraction of the fastest
LEAN = 0.10
#: side by side is advised when it finishes the pair at least this much sooner than one after the other
SIDE_MARGIN = 0.10
#: host threads of a GPU-only MD run (OpenMM's GPU platform uses one CPU thread)
MD_HOST_THREADS = 1
KINDS = ("md", "build", "crest")
DXTB = ("gfn1-xtb", "gfn2-xtb")


# ------------------------------------------------------------------------------------------- the inputs
@dataclass
class PlanJob:
    """What the planner needs to know about a job (not ``gui.job_queue.Job``: no molecules, no process)."""

    kind: str                           # "md" | "build" | "crest"
    name: str = ""
    # MD
    atoms: int = 0
    ns: float = 0.0                     # still to run
    dispersion: str = ""                # "ewald" | "cutoff12" | "cutoff16" | ...
    geometry: str = "slab"
    box: tuple[float, float, float] | None = None
    engine: str = ""                    # "lammps" | "openmm" | "" (either: not started yet)
    part: str = ""                      # the part that holds most of the ns left (its segments are learned from)
    segments: list[dict] = field(default_factory=list)   # this run's earlier segments (md.sim.Segment dicts)
    folder: str = ""
    # builds and CREST
    method: str = ""                    # a build's engine; CREST: its method (gfnff, gfn2)
    solvent: str = ""
    molecules: int = 0
    size: float = 0.0                   # mean atoms with hydrogens (0 = the benchmark set's)
    rotors: float = 0.0                 # CREST: mean rotatable bonds
    crest_quick: bool = True
    threads: int = 0                    # 0: the planner chooses; N: the job runs on exactly N threads
    device: str = ""                    # "": the planner chooses; "cpu" | "gpu": the job runs there
    running: bool = False               # already running: it goes first one after the other, and keeps its assignment
    notes: list[str] = field(default_factory=list)

    def title(self) -> str:
        if self.name:
            return self.name
        if self.kind == "md":
            return f"MD {self.atoms:,} atoms {self.dispersion}"
        if self.kind == "crest":
            return f"CREST {self.molecules:,} molecules"
        return f"{self.method} {self.molecules:,} molecules"

    def what(self) -> str:
        """'MD: 9,780 atoms, 13 ns, cutoff16' / 'build: aimnet2, 5,000 molecules of ~30 atoms'."""
        if self.kind == "md":
            eng = f", {self.engine}" if self.engine else ""
            return f"MD: {self.atoms:,} atoms, {self.ns:g} ns to run, {self.dispersion}, {self.geometry}{eng}"
        size = f" of ~{self.size:g} atoms" if self.size else ""
        if self.kind == "crest":
            return (f"CREST ({self.crest_level()}): {self.molecules:,} molecules{size}"
                    + (f", ~{self.rotors:g} rotatable bonds" if self.rotors else ""))
        return f"build: {self.method}{'+' + self.solvent if self.solvent else ''}, {self.molecules:,} molecules{size}"

    def crest_level(self) -> str:
        method = (self.method or "gfnff").strip().lower()
        return f"{'gfnff' if method == 'gff' else method}{' quick' if self.crest_quick else ''}"


@dataclass
class Machine:
    """One machine as the planner sees it: its registry profile and the threads a job may use there."""

    profile: object                     # dist.profile.MachineProfile
    budget: int
    physical: int
    gpus: int
    learned: dict = field(default_factory=dict)       # learn_build_rates(): (machine, key, "cpu"|"gpu") -> rate
    cost_models: dict | None = None     # CREST level -> threads.CostModel; None = the registry's (load_cost_model)
    include_unverified: bool = False    # MD variants whose CPU/GPU check failed

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def windows(self) -> bool:
        return str(getattr(self.profile, "os", "")).lower().startswith("windows")

    def describe(self) -> str:
        gpu = (", ".join(str(g.get("name", "GPU")) for g in self.profile.gpus) if self.profile.gpus
               else ("a GPU" if self.gpus else "no GPU"))
        return (f"{self.name} ({self.budget} thread{'s' if self.budget != 1 else ''}, {self.physical} core"
                f"{'s' if self.physical != 1 else ''}, {gpu})")


def machine_view(p, threads: int = 0, learned: Mapping | None = None, cost_models: Mapping | None = None,
                 include_unverified: bool = False, settings=None) -> Machine:
    """A ``Machine`` for profile ``p``. ``threads``: the budget (0: the machine's own, from its free threads and
    priority in the settings and its logical threads). A GPU counts when torch saw one or an MD variant ran on one."""
    from .. import priority

    logical = int(getattr(p, "logical_cores", 0) or 0) or int(getattr(p, "physical_cores", 0) or 0) or 1
    physical = int(getattr(p, "physical_cores", 0) or 0) or max(1, logical // 2)
    if threads and threads > 0:
        budget = min(int(threads), logical)
    else:
        free, level = priority.DEFAULT_FREE_THREADS, priority.DEFAULT_PRIORITY
        try:
            if settings is None:
                from .. import settings as settings_mod

                settings = settings_mod.load()
            free, level = settings.free_threads_for(p.name), settings.priority_for(p.name)
        except Exception:  # noqa: BLE001, S110 - the defaults without settings
            pass
        budget = priority.thread_budget(0, free, level, logical)
    gpus = len(getattr(p, "gpus", []) or [])
    if not gpus:
        variants = ((getattr(p, "md_throughput", None) or {}).get("variants") or {})
        if any(str(k).startswith(("gpu-", "openmm-gpu-")) and (v or {}).get("ns_per_day") for k, v in variants.items()):
            gpus = 1
    return Machine(p, max(1, budget), max(1, physical), gpus, dict(learned or {}),
                   dict(cost_models) if cost_models is not None else None, include_unverified)


@dataclass
class Assignment:
    """One way to run a job: the devices, its predicted time and where the numbers come from."""

    gpu: bool
    threads: int                        # CPU threads it uses (with the GPU: its host threads)
    hours: float | None = None          # predicted wall hours to finish; None = unknown
    rate: float | None = None           # ns/day (MD) or molecules/hour (builds, CREST)
    unit: str = ""
    source: str = ""                    # "benchmark" | "this run's segments" | "run timings" | "cost model" | "rough"
    detail: str = ""                    # the numbers behind it
    variant: str = ""                   # MD: the variant ("gpu-cutoff16")
    lanes: str = ""                     # builds: "1 GPU lane x 4 + 4 CPU lanes x 2"
    possible: bool = True
    why_not: str = ""                   # not possible, or unknown: why
    notes: list[str] = field(default_factory=list)
    host: int = 1                       # host threads of a GPU-only run of this job

    @property
    def known(self) -> bool:
        return self.possible and self.hours is not None

    @property
    def label(self) -> str:
        if not self.gpu:
            return f"CPU only, {self.threads} thread{'s' if self.threads != 1 else ''}"
        if self.threads <= self.host:
            return f"GPU only (+{self.threads} host thread{'s' if self.threads != 1 else ''})"
        return f"GPU + {self.threads} threads"

    def time_text(self) -> str:
        if not self.possible:
            return "not possible"
        if self.hours is None:
            return "unknown"
        text = _hours(self.hours)
        return text if text.startswith("under") else f"~{text}"

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.label}


def _hours(h: float) -> str:
    from ..md.bench import format_hours

    sec = h * 3600.0
    if sec < 1.0:
        return "under 1 s"
    return f"{sec:.0f} s" if sec < 60.0 else format_hours(h)


def capacity(threads: int, cores: int) -> float:
    """Core-equivalents of ``threads`` busy threads on ``cores`` physical cores (``SMT_GAIN`` per thread beyond them)."""
    t = max(0, int(threads))
    return min(t, cores) + SMT_GAIN * max(0, t - cores)


def thread_speed(threads: int, cores: int) -> float:
    """The speed of each of ``threads`` busy threads relative to a thread with a core to itself: 1 up to the physical
    cores, less beyond (two jobs side by side share it: ``side_by_side``)."""
    t = max(1, int(threads))
    return capacity(t, cores) / t


# ------------------------------------------------------------------------------------------- MD
@dataclass
class _MDOption:
    variant: str                        # "gpu-cutoff16", "openmm-gpu-cutoff16", ...
    kind: str                           # "cpu" | "gpu" (host thread only) | "gpu+cpu" (LAMMPS' GPU package)
    rate: float                         # ns/day of this job's system at ``threads``
    threads: int
    gpu_wait: float = GPU_WAIT_SHARE
    source: str = "benchmark"
    detail: str = ""
    notes: list[str] = field(default_factory=list)


def _amdahl(t: float, s: float = MD_SERIAL_FRACTION) -> float:
    return s + (1.0 - s) / max(1.0, float(t))


def _md_kind(accel: str, rec: Mapping | None = None) -> str:
    if accel in ("cpu", "openmm-cpu"):
        return "cpu"
    if accel == "openmm-gpu":
        return "gpu"
    if accel == "gpu":
        return "gpu+cpu"
    if accel == "kokkos":
        return "gpu" if str((rec or {}).get("accelerator", "")) == "kokkos-gpu" else "cpu"
    return ""


def _segment_accel(variant: str) -> str:
    """The accelerator of a segment's variant: "gpu-cutoff16" -> "gpu"; "openmm-opencl-mixed" -> "openmm-gpu"."""
    s = str(variant or "").lower()
    if s.startswith("openmm-"):
        return "openmm-cpu" if s.startswith("openmm-cpu") else "openmm-gpu"
    head = s.split("-", 1)[0]
    return head if head in ("cpu", "gpu", "kokkos") else ""


def _md_options(job: PlanJob, m: Machine) -> tuple[list[_MDOption], list[str]]:
    """(the job's MD options on ``m``, notes): the benchmark's variants of its dispersion scaled to its system, replaced
    per accelerator by this run's latest segment on this machine (of the part with most of the ns left)."""
    from ..md import bench

    notes: list[str] = []
    md = getattr(m.profile, "md_throughput", None) or {}
    out: dict[str, _MDOption] = {}
    try:
        disp = bench.parse_variant(f"cpu-{job.dispersion}").dispersion if job.dispersion else None
    except ValueError:
        disp = None
    if bench.rated(md):
        ests = bench.estimate_hours(md, max(1, job.atoms), 1.0, dispersion=disp, geometry=job.geometry, box=job.box,
                                    include_unverified=m.include_unverified)
        for e in ests:
            try:
                v = bench.parse_variant(e.variant)
            except ValueError:
                continue
            rec = (md.get("variants") or {}).get(e.key) or {}
            kind = _md_kind(v.accelerator, rec)
            engine = "openmm" if v.accelerator.startswith("openmm") else "lammps"
            if not kind or (job.engine and job.engine != engine) or v.accelerator in out:
                continue
            br = rec.get("breakdown") if isinstance(rec.get("breakdown"), Mapping) else {}
            pair = br.get("Pair") if isinstance(br.get("Pair"), (int, float)) else None
            wait = float(pair) / 100.0 if pair is not None and 0 <= pair <= 100 else GPU_WAIT_SHARE
            opt_notes = [e.note] if e.note else []
            if not e.verified:
                opt_notes.append(f"unverified: its CPU/GPU check said {e.check}")
            prec = str(rec.get("precision") or "")
            if prec == "single":
                opt_notes.append("single precision (no FP64 on this GPU): σ comes from a double-precision rerun of "
                                 "its frames afterwards")
            detail = (f"{e.variant}: {bench.fmt_ns(e.measured_ns_per_day)} ns/day on {e.measured_atoms:,} atoms "
                      f"({e.geometry} benchmark, {e.threads} thread{'s' if e.threads != 1 else ''}), "
                      f"{bench.fmt_ns(e.ns_per_day)} at {job.atoms:,}")
            out[v.accelerator] = _MDOption(e.variant, kind, e.ns_per_day, max(1, int(e.threads or 1)), wait,
                                           "benchmark", detail, opt_notes)
    # this run's own segments on this machine: they timed this very system
    segs = [s for s in job.segments if str(s.get("machine", "")) == m.name and (s.get("ns_per_day") or 0) > 0
            and int(s.get("threads") or 0) > 0 and (not job.part or s.get("part") == job.part)]
    for s in sorted(segs, key=lambda s: int(s.get("n") or 0)):
        accel = _segment_accel(str(s.get("variant", "")))
        kind = _md_kind(accel)
        if not kind:
            continue
        old = out.get(accel)
        wait = old.gpu_wait if old is not None else GPU_WAIT_SHARE
        name = old.variant if old is not None else (str(s.get("variant")) if accel not in ("openmm-gpu", "openmm-cpu")
                                                    else f"{accel}-{job.dispersion}")
        out[accel] = _MDOption(name, kind, float(s["ns_per_day"]), int(s["threads"]), wait, "this run's segments",
                               f"segment {s.get('n')} ({s.get('part')}) ran {bench.fmt_ns(float(s['ns_per_day']))} "
                               f"ns/day on {int(s['threads'])} thread{'s' if int(s['threads']) != 1 else ''}"
                               + (f", {s.get('variant')}" if s.get("variant") else ""), [])
    if not out:
        notes.append(f"no MD benchmark of {job.dispersion or 'this'} dispersion on {m.name}: `quantabench-cli machines "
                     "--bench md` there" if not bench.rated(md) else
                     f"no MD variant of {job.dispersion or 'this'} dispersion"
                     + (f" for {job.engine}" if job.engine else "") + f" measured on {m.name}")
    return list(out.values()), notes


def _md_rate(opt: _MDOption, threads: int, physical: int, speed: float = 1.0) -> float:
    """ns/day of ``opt`` on ``threads`` OpenMP threads (at most the physical cores), each at ``speed`` (1: a core to
    itself): the CPU work takes A(t) / speed of a step, the GPU's share does not change."""
    t, tb = min(max(1, threads), physical), min(opt.threads, physical)
    if opt.kind == "gpu":
        return opt.rate
    cpu = _amdahl(t) / max(speed, 1e-6) / _amdahl(tb)
    if opt.kind == "cpu":
        return opt.rate / cpu
    g = min(max(opt.gpu_wait, 0.0), 1.0)
    return opt.rate / (g + (1.0 - g) * cpu)


def _md_eval(job: PlanJob, m: Machine, gpu: bool, threads: int, cache: dict, speed: float | None) -> Assignment:
    if "md" not in cache:
        cache["md"] = _md_options(job, m)
    opts, notes = cache["md"]
    want = ("gpu", "gpu+cpu") if gpu else ("cpu",)
    mine = [o for o in opts if o.kind in want]
    a = Assignment(gpu, max(1, threads), unit="ns/day", host=MD_HOST_THREADS)
    disp, _ = _dispersion_kind(job.dispersion)
    if gpu and not mine:
        if m.gpus == 0 and not any(o.kind != "cpu" for o in opts):
            a.possible, a.why_not = False, f"no GPU in {m.name}'s profile (`machines --refresh` there finds one)"
        elif disp == "ewald" and job.engine == "lammps":
            a.possible, a.why_not = False, ("LAMMPS has no GPU styles for Ewald dispersion (lj/long/coul/long, "
                                            "pppm/disp): it runs on the CPU (OpenMM's LJ-PME runs on GPUs)")
        else:
            a.why_not = (f"no GPU variant of {job.dispersion} measured on {m.name}"
                         + (" (Ewald dispersion: OpenMM's LJ-PME only)" if disp == "ewald" else "")
                         + ": `quantabench-cli machines --bench md --md-variants default,openmm` there")
        return a
    if not mine:
        a.why_not = "; ".join(notes) or f"no CPU variant measured on {m.name}"
        return a
    best, best_rate, used = None, 0.0, threads
    for o in mine:
        t = MD_HOST_THREADS if o.kind == "gpu" else min(max(1, threads), m.physical)
        r = _md_rate(o, t, m.physical, thread_speed(t, m.physical) if speed is None else speed)
        if r > best_rate * (1.0 + 1e-9) or (best is not None and abs(r - best_rate) <= 1e-9 * r and t < used):
            best, best_rate, used = o, r, t
    a.threads, a.rate, a.variant, a.source = used, best_rate, best.variant, best.source
    a.hours = job.ns / best_rate * 24.0 if job.ns > 0 else 0.0
    how = ""
    if best.kind == "gpu+cpu":
        how = (f"; the pair style on the GPU, PPPM and the bonded terms on {used} CPU thread"
               f"{'s' if used != 1 else ''} ({(1 - best.gpu_wait):.0%} of its step on the CPU)")
    elif used != best.threads and best.kind == "cpu":
        how = f"; scaled to {used} threads by Amdahl's law (serial fraction {MD_SERIAL_FRACTION:g})"
    a.detail = best.detail + how
    a.notes = list(best.notes)
    if threads > m.physical and best.kind != "gpu":
        a.notes.append(f"at most the {m.physical} physical cores: LAMMPS' OpenMP threads gain little from SMT")
    return a


def _dispersion_kind(dispersion: str) -> tuple[str, float]:
    from ..md import bench

    try:
        return bench.dispersion_cutoff(dispersion or "cutoff16")
    except ValueError:
        return "", 0.0


# ------------------------------------------------------------------------------------------- builds
@functools.lru_cache(maxsize=8)
def _bench_size(bench_set: str) -> float:
    from . import speed

    return speed.bench_size(bench_set)


def _workload(method: str, solvent: str) -> tuple[str, str, bool]:
    """(engine, kind, gpu_capable) as ``lanes.workload`` has them (method "auto" resolved)."""
    from types import SimpleNamespace

    from . import lanes

    cfg = SimpleNamespace(method=method or "auto", solvent=solvent or "", conformer_search="rdkit", qm_method="")
    return lanes.workload(cfg)


def _cpu_lanes(kind: str, threads: int, cores: int) -> list[int]:
    """The CPU lanes ``plan_auto`` plans on ``threads`` threads and ``cores`` physical cores for an engine of ``kind``
    (threads per lane), without its memory cap; one lane with every thread where only one fits."""
    threads, cores = max(1, int(threads)), max(1, min(int(cores), int(threads)))
    if kind == "mmff":
        n, per = cores, 1
    elif kind == "native":
        n = cores
        per = max(1, threads // n)
    elif kind == "external":
        n, per = threads // 2, 2
    else:                                   # in-process PyTorch (and unknown): 2-thread lanes on the physical cores
        n, per = cores // 2, 2
    n = min(n, cores)                       # never more lanes than physical cores
    if n <= 1:
        return [threads]
    return [per] * n


def _lanes_text(lanes: Sequence[int], gpu_threads: int = 0) -> str:
    parts = [f"1 GPU lane x {gpu_threads}"] if gpu_threads else []
    groups: dict[int, int] = {}
    for t in lanes:
        groups[t] = groups.get(t, 0) + 1
    for t, n in groups.items():
        parts.append(f"{n} CPU lane{'s' if n != 1 else ''} x {t}")
    return " + ".join(parts)


def _gpu_key(p, key: str) -> tuple[str, float] | None:
    """(device, molecules/h) of the fastest GPU benchmark of ``key`` ("aimnet2", "aimnet2+water"), or None."""
    from .profile import _split_key

    hits = [(dev, float(v)) for k, v in p.throughput.items() if v and v > 0
            for mm, dev in [_split_key(k)] if mm == key and dev != "cpu" and not dev.startswith("lanes:")]
    return max(hits, key=lambda x: x[1]) if hits else None


def _rate_source(m: Machine, key: str, device: str, size: float) -> tuple[float, int, str, str] | None:
    """(molecules/h of one process on the benchmark's threads at ``size`` atoms, those threads, source, detail) for
    ``key`` on ``device`` ("cpu" or a GPU): learned from run timings first, else the benchmark."""
    from .profile import _fmt_rate, set_of_key, throughput_for

    p = m.profile
    cls = "cpu" if device == "cpu" else "gpu"
    got = m.learned.get((m.name, key, cls))
    if got and got.get("s_per_atom"):
        unit = size or float(got.get("size") or 0) or _bench_size("pfas")
        rate = 3600.0 / (float(got["s_per_atom"]) * unit)
        t = max(1, int(got.get("threads") or 1))
        return rate, t, "run timings", (f"{got.get('molecules', '?')} molecules of earlier {key} runs here on "
                                        f"{cls.upper()} lanes of {t} thread{'s' if t != 1 else ''}: "
                                        f"{_fmt_rate(rate)}/h per lane at ~{unit:.0f} atoms")
    rate = throughput_for(p, key, device)
    if not rate:
        return None
    info = (getattr(p, "benchmarks", {}) or {}).get(f"{key}@{device}") or {}
    t = int(info.get("threads") or 0) or max(1, int(getattr(p, "logical_cores", 0) or 2) - 1)
    unit = _bench_size(set_of_key(p, f"{key}@{device}"))
    scaled = rate * (unit / size if size else 1.0)
    return scaled, t, "benchmark", (f"benchmark {key}@{device} {_fmt_rate(rate)}/h on {t} threads"
                                    + (f" (molecules of ~{unit:.0f} atoms; ~{size:g} here)" if size else ""))


def _build_eval(job: PlanJob, m: Machine, gpu: bool, threads: int, cache: dict, speed: float | None) -> Assignment:
    from . import lanes as lanes_mod
    from .plan import DEFAULT_PER_CORE, ESTIMATE_PER_CORE
    from .profile import bench_method

    if "w" not in cache:
        cache["w"] = _workload(job.method, job.solvent)
    method, kind, gpu_capable = cache["w"]
    key = bench_method(method, job.solvent)
    q = THREAD_POWER.get(kind, DEFAULT_THREAD_POWER)
    g = lanes_mod.GPU_LANE_THREADS
    a = Assignment(gpu, max(1, threads), unit="molecules/h", host=g)
    notes = []
    cpu = _rate_source(m, key, "cpu", job.size)
    if cpu is None:
        per_core = ESTIMATE_PER_CORE.get(method, DEFAULT_PER_CORE)
        unit = _bench_size("pfas")
        cpu = (per_core * (unit / job.size if job.size else 1.0), 1, "rough",
               f"no benchmark of {key} on {m.name}: a rough {per_core:,.0f}/h per physical core "
               f"(`quantabench-cli machines --bench {method}` there)")
        q_cpu = 1.0                         # per core: a 1-thread "process" per lane
    else:
        q_cpu = q

    def lane_rate(t: int, f: float) -> float:
        # one process on the benchmark's threads gains nothing from SMT: it is anchored at the physical cores it had
        r, tb, _src, _d = cpu
        return r * (t / max(1, min(tb, m.physical))) ** q_cpu * f

    if not gpu:
        lanes = _cpu_lanes(kind, threads, m.physical)
        a.threads = sum(lanes)
        f = thread_speed(a.threads, m.physical) if speed is None else speed
        a.rate = sum(lane_rate(t, f) for t in lanes)
        a.lanes = _lanes_text(lanes)
        a.source, a.detail = cpu[2], cpu[3]
    else:
        if not gpu_capable:
            a.possible, a.why_not = False, f"{method} runs on the CPU only"
            return a
        gdev = _gpu_key(m.profile, key)
        learned_gpu = m.learned.get((m.name, key, "gpu"))
        if gdev is None and not learned_gpu:
            if method in DXTB:
                a.possible = False
                a.why_not = ("dxtb stays on the CPU: 5-8x slower on the workstation's GPU (docs/ENGINES.md); its "
                             "GPU rate was not measured here (dxtb GPU = on, then `machines --bench " + method + "`)")
            elif m.gpus == 0:
                a.possible, a.why_not = False, (f"no GPU in {m.name}'s profile (PyTorch saw none when it was made; "
                                                "`machines --refresh` there)")
            else:
                a.why_not = f"no GPU benchmark of {key} on {m.name}: `quantabench-cli machines --bench {method}` there"
            return a
        src = _rate_source(m, key, gdev[0] if gdev else "gpu", job.size)
        if src is None:
            a.why_not = f"no GPU benchmark of {key} on {m.name}: `quantabench-cli machines --bench {method}` there"
            return a
        r_gpu, _tb, gsrc, gdetail = src
        lane_g = min(max(1, threads), g)
        rest = max(0, threads - lane_g) if method not in DXTB else 0
        lanes = _cpu_lanes(kind, rest, max(1, m.physical - lane_g)) if rest else []
        a.threads = lane_g + sum(lanes)
        f = thread_speed(a.threads, m.physical) if speed is None else speed
        a.rate = r_gpu + sum(lane_rate(t, f) for t in lanes)
        a.lanes = _lanes_text(lanes, lane_g)
        a.source = gsrc if not lanes else ("/".join(sorted({gsrc, cpu[2]})))
        a.detail = gdetail + (f"; CPU lanes: {cpu[3]}" if lanes else "")
        if method in DXTB and threads > lane_g:
            notes.append("dxtb: one precision per run, so no CPU lanes beside its GPU lane")
    a.hours = job.molecules / a.rate if a.rate and job.molecules > 0 else (0.0 if job.molecules <= 0 else None)
    if a.source in ("benchmark", "rough") or "benchmark" in a.source:
        notes.append("at the benchmark's short protocol (4 conformers, 2 refined): a production run takes longer; "
                     "the ranking holds")
    a.notes = notes
    return a


def _crest_eval(job: PlanJob, m: Machine, gpu: bool, threads: int, cache: dict, speed: float | None) -> Assignment:
    from . import lanes as lanes_mod
    from . import threads as threads_mod

    a = Assignment(gpu, max(1, threads), unit="molecules/h", host=lanes_mod.GPU_LANE_THREADS)
    where = " (through WSL on Windows)" if m.windows else ""
    if gpu:
        a.possible, a.why_not = False, f"CREST runs on the CPU only{where}"
        return a
    level = job.crest_level()
    if "model" not in cache:
        models = m.cost_models
        cache["model"] = (models.get(level) if models is not None else
                          threads_mod.load_cost_model(m.name, level)) or threads_mod.BUILT_IN
    model = cache["model"]
    per = min(lanes_mod.CREST_LANE_THREADS, max(1, threads))
    n = max(1, threads // per)
    extra = max(0, threads - n * per)
    widths = [per + (1 if i < extra else 0) for i in range(n)]
    a.threads = sum(widths)
    a.lanes = _lanes_text(widths)
    if not job.size:
        a.why_not = "give the molecules' mean size (atoms=...): CREST's time grows about as atoms^4"
        return a
    f = thread_speed(a.threads, m.physical) if speed is None else speed
    secs = [model.seconds(job.size, w, job.rotors) for w in widths]
    if any(s is None for s in secs):
        a.why_not = (f"no fitted CREST cost model for {m.name} ({level}): `quantabench-cli timings --fit` there "
                     "(the built-in one has no time scale)")
        return a
    a.rate = sum(3600.0 / s for s in secs) * f
    a.hours = job.molecules / a.rate if job.molecules > 0 else 0.0
    a.source = "cost model"
    a.detail = (f"{model.source}: ~{secs[0]:,.0f} s per molecule of {job.size:g} atoms on {widths[0]} threads"
                + (f", {job.rotors:g} rotatable bonds" if job.rotors else ""))
    a.notes = [f"at the mean size: the largest molecules take much longer (atoms^{model.power:.1f}"
               + (f", scatter x{model.scatter:.1f} per molecule" if model.scatter else "") + ")"]
    if f < 0.999:
        a.notes.append(f"{a.threads} threads on {m.physical} cores: each at {f:.0%} of a core (SMT)")
    if where:
        a.notes.append("CREST runs through WSL on Windows")
    return a


# ------------------------------------------------------------------------------------------- one job
def evaluate(job: PlanJob, m: Machine, gpu: bool, threads: int, cache: dict | None = None,
             speed: float | None = None) -> Assignment:
    """The assignment of ``job`` to the GPU (``gpu``) and/or ``threads`` CPU threads of ``m``: its time and where the
    numbers come from. ``speed``: each CPU thread's speed (``thread_speed`` of every busy thread on the machine; None:
    the job alone). ``cache``: a dict kept between calls for the same job and machine."""
    cache = {} if cache is None else cache
    if job.threads:
        threads = min(threads, job.threads) if gpu else job.threads
    threads = max(1, min(int(threads), m.budget))
    if job.device in ("cpu", "gpu") and gpu != (job.device == "gpu"):
        why = "it runs on the " + job.device.upper() + (" (running)" if job.running else " (device=" + job.device + ")")
        return Assignment(gpu, threads, possible=False, why_not=why, host=host_threads(job))
    memo = cache.setdefault("eval", {})
    key = (gpu, threads, None if speed is None else round(float(speed), 6))
    if key not in memo:
        fn = _md_eval if job.kind == "md" else _crest_eval if job.kind == "crest" else _build_eval
        memo[key] = fn(job, m, gpu, threads, cache, speed)
    return memo[key]


def host_threads(job: PlanJob) -> int:
    """The CPU threads a GPU-only run of ``job`` takes."""
    from . import lanes as lanes_mod

    return MD_HOST_THREADS if job.kind == "md" else lanes_mod.GPU_LANE_THREADS


@dataclass
class JobPlan:
    job: PlanJob
    machine: str
    options: list[Assignment]           # CPU only, GPU only, GPU + threads (in this order)
    best: Assignment | None
    lean: Assignment | None = None      # with the GPU: the fewest threads within LEAN of the best
    why: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"job": asdict(self.job), "machine": self.machine, "options": [a.to_dict() for a in self.options],
                "best": self.best.to_dict() if self.best else None,
                "lean": self.lean.to_dict() if self.lean else None, "why": self.why}


def assignments(job: PlanJob, m: Machine, cache: dict | None = None) -> list[Assignment]:
    """CPU only (every thread), GPU only (its host threads) and GPU + every thread, for ``job`` on ``m``."""
    cache = {} if cache is None else cache
    full = job.threads or m.budget
    cpu = evaluate(job, m, False, full, cache)
    gpu_only = evaluate(job, m, True, host_threads(job), cache)
    both = evaluate(job, m, True, full, cache)
    out = [cpu, gpu_only]
    same = (both.known and gpu_only.known and both.threads <= gpu_only.threads) or (
        not both.known and not gpu_only.known and both.why_not == gpu_only.why_not)
    if not same:                            # (OpenMM: the GPU and one thread, whatever it is given)
        out.append(both)
    return out


def _choose(cands: Iterable[Assignment]) -> Assignment | None:
    known = [a for a in cands if a.known]
    if not known:
        return None
    fastest = min(a.hours for a in known)
    near = [a for a in known if a.hours <= fastest * (1.0 + TIE) + 1e-12]
    return min(near, key=lambda a: (a.threads, a.gpu, a.hours))


def plan_job(job: PlanJob, m: Machine, cache: dict | None = None) -> JobPlan:
    """The fastest assignment of ``job`` on ``m`` (module docstring), with the reasons in plain words."""
    cache = {} if cache is None else cache
    options = assignments(job, m, cache)
    best = _choose(options)
    plan = JobPlan(job, m.name, options, best)
    if best is None:
        why = [a.why_not for a in options if a.why_not]
        plan.why = [f"No prediction for {job.title()} on {m.name}: " + ("; ".join(dict.fromkeys(why)) or "no data")]
        return plan
    if best.gpu:
        for t in range(1, best.threads):
            a = evaluate(job, m, True, t, cache)
            if a.known and a.hours <= best.hours * (1.0 + LEAN) and a.threads < best.threads:
                plan.lean = a
                break
    plan.why = _why(plan, m)
    return plan


def _why(plan: JobPlan, m: Machine) -> list[str]:
    best, job = plan.best, plan.job
    lines = [f"Fastest: {best.label}, {best.time_text()}" + (f" ({best.variant})" if best.variant else "")
             + (f" [{best.lanes}]" if best.lanes else "") + "."]
    cpu, gpu_only = plan.options[0], plan.options[1]
    if job.kind == "md" and best.gpu and best.threads > best.host:
        lines.append("LAMMPS' GPU package runs the pair style on the GPU and PPPM and the bonded terms on the CPU "
                     "threads, so a GPU MD job needs threads as well as the GPU.")
    elif job.kind == "md" and best.gpu:
        lines.append("OpenMM's GPU platform runs the whole step on the GPU with one CPU thread; the other threads stay "
                     "free." if best.variant.startswith("openmm") else "The GPU runs the step with one CPU thread.")
    if best.gpu and best.variant.startswith("openmm") and job.engine != "openmm":
        lines.append("OpenMM runs much faster on GPUs at reduced arithmetic precision; σ is computed afterwards in "
                     "double precision. The engine is chosen when the folder is written (`md liquid-slab --engine "
                     "openmm`) or before it starts (`md run --engine openmm`).")
    if not best.gpu:
        for a in plan.options[1:]:
            if a.known:
                lines.append(f"{a.label} would take {a.time_text()}: the GPU is slower for this here.")
                break
        else:
            reason = next((a.why_not for a in plan.options[1:] if a.why_not), "")
            if reason:
                lines.append(f"The GPU: {reason}.")
    elif cpu.known and cpu.hours > best.hours:
        lines.append(f"CPU only would take {cpu.time_text()} ({cpu.hours / best.hours:.1f}x as long)."
                     if best.hours > 0 else f"CPU only would take {cpu.time_text()}.")
    elif not cpu.known and cpu.why_not:
        lines.append(f"CPU only: {cpu.why_not}.")
    if plan.lean is not None:
        free = m.budget - plan.lean.threads
        lines.append(f"{plan.lean.label} is within {LEAN:.0%} ({plan.lean.time_text()}) and leaves {free} thread"
                     f"{'s' if free != 1 else ''} for other work: what it takes beside another job.")
    if job.kind == "md" and gpu_only.known and best.gpu and best.threads > best.host and gpu_only.hours > best.hours:
        lines.append(f"On the GPU with one thread it would take {gpu_only.time_text()}.")
    return lines


# ------------------------------------------------------------------------------------------- two jobs
@dataclass
class SideBySide:
    a: JobPlan
    b: JobPlan
    verdict: str                        # "side by side" | "one after the other" | "unknown"
    serial_order: list[str] = field(default_factory=list)
    serial_finish: list[float] = field(default_factory=list)     # hours, in serial_order
    serial_each: list[float] = field(default_factory=list)       # [a, b] hours one after the other
    side: list[Assignment] = field(default_factory=list)          # [a's, b's]
    side_finish: list[float] = field(default_factory=list)       # [a, b] hours
    slowdown: float = 0.0
    lines: list[str] = field(default_factory=list)

    @property
    def serial_hours(self) -> float | None:
        return max(self.serial_finish) if self.serial_finish else None

    @property
    def side_hours(self) -> float | None:
        return max(self.side_finish) if self.side_finish else None

    def to_dict(self) -> dict:
        return {"a": self.a.job.title(), "b": self.b.job.title(), "verdict": self.verdict,
                "serial_order": self.serial_order, "serial_finish_h": self.serial_finish, "serial_h": self.serial_each,
                "side_by_side": [x.to_dict() for x in self.side], "side_finish_h": self.side_finish,
                "slowdown": self.slowdown, "lines": self.lines}


def _gpu_leaning(plan: JobPlan) -> bool:
    x = plan.lean or plan.best
    return bool(x and x.gpu)


def side_by_side(a: PlanJob, b: PlanJob, m: Machine, *, slowdown: float = 0.0,
                 plans: tuple[JobPlan, JobPlan] | None = None,
                 caches: tuple[dict, dict] | None = None) -> SideBySide:
    """Whether ``a`` and ``b`` finish sooner side by side on ``m`` than one after the other (module docstring)."""
    ca, cb = caches if caches is not None else ({}, {})
    pa, pb = plans if plans is not None else (plan_job(a, m, ca), plan_job(b, m, cb))
    out = SideBySide(pa, pb, "unknown", slowdown=float(slowdown or 0.0))
    if pa.best is None or pb.best is None:
        missing = pa if pa.best is None else pb
        other = pb if missing is pa else pa
        why = next((x.why_not for x in missing.options if x.why_not), "no data")
        if other.best is not None and _gpu_leaning(other) and not any(x.gpu and x.known for x in missing.options):
            use = other.lean or other.best
            out.lines = [f"{other.job.title()} leans on the GPU ({use.label}) and {missing.job.title()} on the CPU: "
                         f"they can share {m.name}, but how much sooner is not predicted ({missing.job.title()}: {why})."]
        else:
            out.lines = [f"Not predicted ({missing.job.title()}: {why})."]
        return out
    first, second = sorted((pa, pb), key=lambda p: (not p.job.running, p.best.hours))
    out.serial_order = [first.job.title(), second.job.title()]
    out.serial_finish = [first.best.hours, first.best.hours + second.best.hours]
    out.serial_each = out.serial_finish if first is pa else out.serial_finish[::-1]
    factor = 1.0 + max(0.0, out.slowdown)
    splits: list[tuple[list[float], list[Assignment]]] = []
    jobs = ((a, ca, 0), (b, cb, 1))
    for (j1, c1, i1), (j2, c2, i2) in (jobs, jobs[::-1]):
        for g1 in (False, True):
            for k in range(1, m.budget):
                x1 = evaluate(j1, m, g1, k, c1)
                rest = m.budget - x1.threads
                if not x1.known or rest < 1:
                    continue
                for g2 in (False, True):
                    if g1 and g2 and m.gpus < 2:
                        continue        # one GPU-heavy job per GPU
                    x2 = evaluate(j2, m, g2, rest, c2)
                    if not x2.known or x1.threads + x2.threads > m.budget:
                        continue
                    # the threads of both share the physical cores: each runs at the speed of all of them
                    f = thread_speed(x1.threads + x2.threads, m.physical)
                    y1, y2 = evaluate(j1, m, g1, k, c1, f), evaluate(j2, m, g2, rest, c2, f)
                    if not (y1.known and y2.known):
                        continue
                    h, pair = [0.0, 0.0], [y1, y2]
                    h[i1], h[i2] = y1.hours * factor, y2.hours * factor
                    pair[i1], pair[i2] = y1, y2
                    splits.append((h, pair))
    if not splits:
        out.verdict = "one after the other"
        out.lines = _pair_lines(out, m)
        return out
    # the shortest makespan; of the splits within TIE of it, the one whose two finish times add up to the least (a
    # split that starves the shorter job for a hair of makespan is not worth it)
    shortest = min(max(h) for h, _ in splits)
    h, pair = min((s for s in splits if max(s[0]) <= shortest * (1.0 + TIE) + 1e-12),
                  key=lambda s: (sum(s[0]), max(s[0])))
    out.side, out.side_finish = pair, h
    sa, sb = out.serial_each
    neither_later = h[0] <= sa * (1.0 + TIE) and h[1] <= sb * (1.0 + TIE) and max(h) < out.serial_hours
    much_sooner = max(h) <= (1.0 - SIDE_MARGIN) * out.serial_hours
    out.verdict = "side by side" if (neither_later or much_sooner) else "one after the other"
    out.lines = _pair_lines(out, m)
    return out


def _pair_lines(out: SideBySide, m: Machine) -> list[str]:
    pa, pb = out.a, out.b
    first, second = (pa, pb) if out.serial_each[0] <= out.serial_each[1] else (pb, pa)
    order = "it runs already" if first.job.running else "the shorter one"
    lines = [f"One after the other ({first.job.title()} first, {order}; each on its fastest assignment): "
             f"{first.job.title()} done after {_hours(first.best.hours)} ({first.best.label}), {second.job.title()} "
             f"after {_hours(out.serial_hours)} ({second.best.label})."]
    if not out.side:
        lines.append(f"Side by side: no split of {m.name} fits both (one GPU, {m.budget} threads).")
        lines.append("-> Run them one after the other.")
        return lines
    xa, xb = out.side
    lines.append(f"Side by side: {pa.job.title()} on {xa.label} done after {_hours(out.side_finish[0])}, "
                 f"{pb.job.title()} on {xb.label} after {_hours(out.side_finish[1])}.")
    gain = out.serial_hours - out.side_hours
    parts = []
    for p, side, serial in zip((pa, pb), out.side_finish, out.serial_each):
        if side < serial * (1.0 - TIE):
            parts.append(f"{p.job.title()} done {_hours(serial - side)} sooner (after {_hours(side)} instead of "
                         f"{_hours(serial)})")
        elif side > serial * (1.0 + TIE):
            parts.append(f"{p.job.title()} {_hours(side - serial)} later (after {_hours(side)} instead of "
                         f"{_hours(serial)})")
        else:
            parts.append(f"{p.job.title()} about as soon (after {_hours(side)})")
    each = "; ".join(parts)
    if out.verdict == "side by side":
        lines.append(f"-> Run them side by side: {each}. Both are done {_hours(gain)} sooner "
                     f"({gain / out.serial_hours:.0%}).")
    elif gain > 0:
        lines.append(f"-> Run them one after the other: side by side would finish the pair only {_hours(gain)} "
                     f"({gain / out.serial_hours:.0%}) sooner: {each}.")
    else:
        lines.append(f"-> Run them one after the other: side by side would take {_hours(-gain)} longer: {each}.")
    lines.append("Each keeps its assignment to its end (lanes are planned once per run; an MD segment keeps its "
                 "threads). " + (f"Side-by-side times include a {out.slowdown:.0%} slowdown (--slowdown)."
                                 if out.slowdown else "The slowdown of sharing the machine (memory bandwidth, heat) is "
                                                      "not measured yet: none assumed (--slowdown 0.1 shows 10 %)."))
    return lines


def plan_queue(jobs: Sequence[PlanJob], m: Machine, *, slowdown: float = 0.0) -> tuple[list[JobPlan], list[SideBySide]]:
    """Each job's plan, and the side-by-side check of every pair (more than two at once is the ledger's, PR 2)."""
    titles = [j.title() for j in jobs]
    for i, j in enumerate(jobs):
        if titles.count(titles[i]) > 1:
            j.name = f"{titles[i]} (job {i + 1})"          # tell the pairs apart
    caches = [{} for _ in jobs]
    plans = [plan_job(j, m, c) for j, c in zip(jobs, caches)]
    pairs = []
    for i in range(len(jobs)):
        for k in range(i + 1, len(jobs)):
            pairs.append(side_by_side(jobs[i], jobs[k], m, slowdown=slowdown, plans=(plans[i], plans[k]),
                                      caches=(caches[i], caches[k])))
    return plans, pairs


# ------------------------------------------------------------------------------------------- text
def plan_lines(plan: JobPlan, m: Machine) -> list[str]:
    """'C6F14 slab: MD: ...' then one line per option and the reasons."""
    lines = [f"{plan.job.title()} ({plan.job.what()}) on {m.describe()}:"]
    shown: set[str] = set()
    for x in plan.options:
        mark = "*" if x is plan.best else " "
        text = f" {mark} {x.label:<28s} {x.time_text():>16s}"
        if x.known:
            text += f"  {x.detail}" + (f" [{x.lanes}]" if x.lanes else "")
        elif x.why_not:
            text += f"  {x.why_not}"
        lines.append(text)
        for n in x.notes:
            if n not in shown:
                shown.add(n)
                lines.append(f"   {'':28s} {'':16s}  ({n})")
    lines += [f"   {w}" for w in plan.why]
    lines += [f"   note: {n}" for n in plan.job.notes]
    return lines


def dialog_text(cfg, molecules: int, p, budget: int | None = None, size: float = 0.0, rotors: float = 0.0) -> str:
    """One paragraph for the job dialog's Resources page: the devices of this build on this computer (``p``: its
    registry profile, None when it has none; ``size``, ``rotors``: the molecules' mean atoms with hydrogens and
    rotatable bonds, 0 = unknown)."""
    if p is None:
        return ("Devices: this computer has no benchmark yet (Machines dock, or `quantabench-cli machines --bench "
                "default`), so the planner cannot compare the CPU and the GPU for this job.")
    from ..chem.search_rule import may_use_crest

    if str(getattr(cfg, "conformer_search", "rdkit")) == "crest":
        job = PlanJob("crest", method=str(getattr(cfg, "crest_method", "gfnff") or "gfnff"),
                      crest_quick=bool(getattr(cfg, "crest_quick", True)), molecules=molecules, size=size,
                      rotors=rotors, threads=int(getattr(cfg, "threads", 0) or 0))
    else:
        job = PlanJob("build", method=str(getattr(cfg, "method", "auto") or "auto"),
                      solvent=str(getattr(cfg, "solvent", "") or ""), molecules=molecules, size=size,
                      threads=int(getattr(cfg, "threads", 0) or 0))
    m = machine_view(p, threads=budget or 0)
    plan = plan_job(job, m)
    parts = [f"{x.label} {x.time_text()}" for x in plan.options if x.possible]
    head = (f"Devices here ({m.budget} thread{'s' if m.budget != 1 else ''}): fastest {plan.best.label}, "
            f"{plan.best.time_text()} for "
            f"{molecules:,} molecule{'s' if molecules != 1 else ''}" if plan.best else "Devices here: no prediction")
    text = head + (f" (all: {'; '.join(parts)})" if parts else "") + "."
    if plan.best is None and plan.why:
        text += " " + plan.why[0]
    gpu_why = next((x.why_not for x in plan.options[1:] if not x.possible and x.why_not), "")
    if gpu_why and plan.best is not None:
        text += f" GPU: {gpu_why}."
    if job.kind == "build" and plan.best is not None and "benchmark" in plan.best.source:
        text += " Times at the benchmark's short protocol; the ranking holds."
    if job.kind == "build" and may_use_crest(cfg):
        text += " Conformers auto: CREST for some molecules, not in this estimate."
    return text


# ------------------------------------------------------------------------------------------- learned rates
def learn_build_rates(rows: Iterable[Mapping]) -> dict[tuple[str, str, str], dict]:
    """{(machine, method key, "cpu" | "gpu"): {"s_per_atom", "threads", "molecules", "size"}} from ``dist.timings``
    rows: per molecule that ended ok on a lane, its wall time per atom (median), the lanes' threads (median). CREST
    searches are left out (their cost model is ``timings --fit``)."""
    from .profile import bench_method

    attempts: dict[tuple, dict] = {}
    crest: set[tuple] = set()
    for r in rows:
        k = (r.get("run_id"), r.get("mol_id"), r.get("attempt"))
        if r.get("stage") == "conformers" and str(r.get("engine") or "").startswith("crest"):
            crest.add(k)
        if k in attempts or r.get("attempt_outcome") != "ok" or r.get("duplicate"):
            continue
        wall, size = r.get("attempt_wall_s"), r.get("size") or r.get("n_atoms")
        method = r.get("mol_method") or r.get("method")
        if not wall or wall <= 0 or not size or not method:
            continue
        dev = str(r.get("device") or r.get("lane_device") or "cpu")
        try:
            key = bench_method(str(method), str(r.get("solvent") or ""))
        except Exception:  # noqa: BLE001 - an engine this QuantaBench does not know
            key = str(method)
        attempts[k] = {"machine": str(r.get("machine") or "?"), "key": key,
                       "cls": "cpu" if dev in ("cpu", "auto", "") else "gpu", "s_per_atom": float(wall) / float(size),
                       "threads": int(r.get("lane_threads") or r.get("threads_given") or 1), "size": float(size)}
    groups: dict[tuple, list[dict]] = {}
    for k, x in attempts.items():
        if k not in crest:
            groups.setdefault((x["machine"], x["key"], x["cls"]), []).append(x)
    return {g: {"s_per_atom": statistics.median(x["s_per_atom"] for x in xs),
                "threads": int(statistics.median(x["threads"] for x in xs)), "molecules": len(xs),
                "size": statistics.median(x["size"] for x in xs)} for g, xs in groups.items()}


# ------------------------------------------------------------------------------------------- job specs
_COMMON_KEYS = {"name", "threads", "device", "running"}
_SPEC_KEYS = {
    "md": {"atoms", "ns", "dispersion", "engine", "geometry", "box"} | _COMMON_KEYS,
    "build": {"method", "molecules", "solvent", "atoms"} | _COMMON_KEYS,
    "crest": {"molecules", "atoms", "rotors", "method", "quick", "solvent"} | _COMMON_KEYS,
}
_YES, _NO = ("yes", "true", "1"), ("no", "false", "0")


def parse_job(text: str) -> PlanJob:
    """A ``kind:key=value,...`` spec or an MD run folder (module docstring); ValueError when it is neither."""
    s = str(text).strip()
    kind, sep, rest = s.partition(":")
    if sep and kind.strip().lower() in KINDS and not Path(s).exists():
        return _parse_spec(kind.strip().lower(), rest)
    return md_job_from_folder(s)


def _parse_spec(kind: str, rest: str) -> PlanJob:
    vals: dict[str, str] = {}
    for item in (x for x in rest.split(",") if x.strip()):
        k, eq, v = item.partition("=")
        k = k.strip().lower()
        if not eq or k not in _SPEC_KEYS[kind]:
            raise ValueError(f"{kind}: {item.strip()!r}: give key=value with a key of {', '.join(sorted(_SPEC_KEYS[kind]))}")
        vals[k] = v.strip()

    def num(key: str, cast=float, required: bool = False, default=0):
        if key not in vals:
            if required:
                raise ValueError(f"{kind}: needs {key}=...")
            return default
        try:
            x = cast(vals[key])
        except ValueError:
            raise ValueError(f"{kind}: {key}={vals[key]!r} is not a number") from None
        if x < 0 or (isinstance(x, float) and not math.isfinite(x)):
            raise ValueError(f"{kind}: {key} must be 0 or more")
        return x

    def yes(key: str, default: bool) -> bool:
        v = vals.get(key, "").lower()
        if v and v not in _YES + _NO:
            raise ValueError(f"{kind}: {key}=yes or {key}=no")
        return default if not v else v in _YES

    device = vals.get("device", "").lower()
    if device not in ("", "cpu", "gpu"):
        raise ValueError(f"{kind}: device=cpu or device=gpu")
    common = {"threads": num("threads", int), "device": device, "running": yes("running", False)}
    name = vals.get("name", "")
    if kind == "md":
        from ..md import bench

        if "dispersion" not in vals:
            raise ValueError("md: needs dispersion=ewald|cutoff12|cutoff16|... (the folder's long-range choice)")
        disp = bench.parse_variant(f"cpu-{vals['dispersion']}").dispersion
        engine = vals.get("engine", "").lower()
        if engine not in ("", "lammps", "openmm"):
            raise ValueError("md: engine=lammps or engine=openmm")
        geometry = vals.get("geometry", "slab").lower()
        if geometry not in ("slab", "bulk"):
            raise ValueError("md: geometry=slab or geometry=bulk")
        box = None
        if vals.get("box"):
            try:
                box = tuple(float(x) for x in vals["box"].lower().split("x"))
            except ValueError:
                box = ()
            if len(box) != 3:
                raise ValueError("md: box=LXxLYxLZ in Å, e.g. box=40x40x200")
        return PlanJob("md", name, atoms=num("atoms", int, True), ns=num("ns", float, True), dispersion=disp,
                       geometry=geometry, box=box, engine=engine, **common)
    if kind == "build":
        if not vals.get("method"):
            raise ValueError("build: needs method=... (an engine, e.g. aimnet2, gfn2-xtb, mmff)")
        return PlanJob("build", name, method=vals["method"].lower(), solvent=vals.get("solvent", ""),
                       molecules=num("molecules", int, True), size=num("atoms"), **common)
    return PlanJob("crest", name, method=vals.get("method", "gfnff").lower(), solvent=vals.get("solvent", ""),
                   molecules=num("molecules", int, True), size=num("atoms", float, True), rotors=num("rotors"),
                   crest_quick=yes("quick", True), **common)


def md_job_from_folder(folder: str | Path) -> PlanJob:
    """An MD job from a run folder: ``run.json`` (SIM-1: ns left, engine, segments) and/or ``liquid_setup.json``."""
    from ..md import sim

    d = Path(folder).expanduser()
    if not d.is_dir():
        raise ValueError(f"{folder}: not a folder, and not a job spec (md:..., build:..., crest:...)")
    setup = sim.read_json(d / "liquid_setup.json") or {}
    cfg = setup.get("config") or {}
    job = PlanJob("md", d.name, folder=str(d))
    run = None
    if (d / sim.RUN_FILE).is_file():
        try:
            run = sim.Run.load(d)
        except sim.RunError as exc:
            raise ValueError(str(exc)) from None
    if run is None and not setup:
        raise ValueError(f"{d}: no run.json or liquid_setup.json (not an MD run folder)")
    if run is not None:
        if run.template == "smd":
            raise ValueError(f"{d}: a steered pull (short, on the CPU): not planned")
        job.name = run.name or d.name
        job.atoms = int(run.system.n_atoms or 0)
        job.ns = max(0.0, run.ns_total - run.ns_done())
        job.dispersion = run.protocol.dispersion or ""
        job.segments = [asdict(s) for s in run.segments]
        job.engine = run.engine if run.segments else ""
        left = {p.name: p.ns - (p.ns if run.part_done(p.name) else p.ns_at(run._resume_step(p)))   # noqa: SLF001
                for p in run.protocol.parts}
        if left and max(left.values()) > 0:
            job.part = max(left, key=left.get)
        if run.segments:
            job.notes.append(f"engine {run.engine}: fixed by its {len(run.segments)} segment(s)")
        elif run.engine:
            job.notes.append(f"written for {run.engine}; `md run --engine` switches a folder that has not started")
        if run.state == "done":
            job.notes.append("this run is done")
        if run.state in ("running", "pausing") and run.segments:
            seg = run.segments[-1]
            job.running = True
            job.threads = int(seg.threads or 0)
            job.device = "cpu" if _md_kind(_segment_accel(seg.variant)) == "cpu" else "gpu"
            job.notes.append(f"running: segment {seg.n} keeps {seg.variant} on {seg.threads} thread(s) until it pauses "
                             "(`md resume --threads N` continues with others)")
    job.atoms = job.atoms or int(setup.get("n_atoms") or 0)
    if not job.ns and run is None:
        job.ns = sum(float(cfg.get(k) or 0.0) for k in ("bulk_ns", "settle_ns", "production_ns"))
    job.dispersion = job.dispersion or str(cfg.get("dispersion") or "")
    target = setup.get("box_target_A") or []
    if len(target) == 3:
        job.box = (float(target[0]), float(target[1]), float(target[2]) + 2.0 * float(cfg.get("vacuum") or 0.0))
    if not job.atoms or not job.dispersion:
        raise ValueError(f"{d}: its atoms or dispersion are not recorded")
    return job


# ------------------------------------------------------------------------------------------- CLI
def register_cli(subparsers) -> None:
    jp = subparsers.add_parser("jobs", help="plan jobs: the fastest devices for each, and which ones to run side by "
                                            "side (SCHED-3)")
    jsub = jp.add_subparsers(dest="jobs_cmd", required=True)
    p = jsub.add_parser("plan", help="for each job, the CPU only, GPU only and GPU + threads times on a machine and "
                                     "the fastest; for two or more, whether pairs finish sooner side by side",
                        description="Plans from the machine registry (benchmarks, MD benchmark), this run's segments "
                                    "and, with --timings, earlier runs' timings. JOB: an MD run folder, or "
                                    "md:atoms=N,ns=NS,dispersion=cutoff16[,engine=lammps|openmm][,geometry=slab|bulk]"
                                    "[,box=LXxLYxLZ], build:method=M,molecules=N[,solvent=water][,atoms=A][,threads=T], "
                                    "crest:molecules=N,atoms=A[,rotors=R][,method=gfnff|gfn2][,quick=yes|no][,threads=T]"
                                    " (each with name=...). Nothing is started.")
    p.add_argument("jobs", nargs="+", metavar="JOB")
    p.add_argument("--machine", help="plan for this machine of the registry (default: this one)")
    p.add_argument("--threads", type=int, default=0,
                   help="the thread budget (default: the machine's logical threads minus its free threads)")
    p.add_argument("--timings", nargs="*", metavar="PATH",
                   help="learn the build rates from these run folders' timings (no PATH: the output folder; slow "
                        "for many runs)")
    p.add_argument("--slowdown", type=float, default=0.0,
                   help="each job's slowdown side by side, e.g. 0.1 for 10 %% (not measured yet; default 0)")
    p.add_argument("--include-unverified", action="store_true",
                   help="also MD variants whose CPU/GPU pressure check failed")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_plan_cli)


def _plan_cli(args) -> int:
    from . import profile as profile_mod

    try:
        jobs = [parse_job(x) for x in args.jobs]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    name = args.machine or profile_mod.machine_name()
    p = profile_mod.load_profile(name)
    if p is None:
        known = ", ".join(x.name for x in profile_mod.load_profiles()) or "none"
        print(f"{name}: not in the machine registry (known: {known}); `quantabench-cli machines --refresh --bench "
              "default` there", file=sys.stderr)
        return 2
    learned: dict = {}
    if args.timings is not None:
        from .. import settings as settings_mod
        from . import timings as timings_mod

        paths = list(args.timings) or [str(settings_mod.load().output_path())]
        got = timings_mod.read_rows(timings_mod.find_job_tables(paths, lambda msg: print(msg, file=sys.stderr)))
        learned = learn_build_rates(got.rows)
        print(f"Learned build rates from {len(got.runs)} run(s): "
              + (", ".join(f"{k[1]}@{k[2]} ({v['molecules']})" for k, v in learned.items() if k[0] == name) or "none "
                 f"of {name}"), file=sys.stderr)
    m = machine_view(p, threads=args.threads, learned=learned, include_unverified=args.include_unverified)
    plans, pairs = plan_queue(jobs, m, slowdown=args.slowdown)
    if args.json:
        print(json.dumps({"machine": m.name, "budget": m.budget, "physical_cores": m.physical, "gpus": m.gpus,
                          "jobs": [x.to_dict() for x in plans], "pairs": [x.to_dict() for x in pairs]}, indent=1,
                         default=str))
        return 0
    for i, plan in enumerate(plans):
        if i:
            print()
        print("\n".join(plan_lines(plan, m)))
    if pairs:
        print()
        print(f"Side by side on {m.name} (pairs; running more than two at once is not planned yet):")
        for x in pairs:
            print(f"  {x.a.job.title()} + {x.b.job.title()}:")
            for line in x.lines:
                print(f"    {line}")
    return 0


def plan_text(jobs: Sequence[PlanJob], m: Machine, slowdown: float = 0.0) -> list[str]:
    """The CLI's text for ``jobs`` on ``m``, as lines."""
    plans, pairs = plan_queue(jobs, m, slowdown=slowdown)
    out: list[str] = []
    for plan in plans:
        out += plan_lines(plan, m)
    for x in pairs:
        out += [f"{x.a.job.title()} + {x.b.job.title()}:"] + [f"  {line}" for line in x.lines]
    return out
