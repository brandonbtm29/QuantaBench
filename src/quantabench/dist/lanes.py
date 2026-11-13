"""Lanes: worker processes that share one machine's CPU cores and GPUs.

One small molecule cannot keep 12 cores busy: RDKit's embedding, dxtb's small matrices, xtb, CREST and Psi4
all scale poorly past a few threads for a PFAS-sized molecule. A batch therefore runs faster as several
**lanes**, each a worker process with its own device and thread budget, each working on its own molecule.

    Lane(name, device, threads)            one worker process; device "cpu" | "cuda:N" | "mps" | "auto"
                                           ("auto": the device the GPU setting selects; only the single
                                           lane of spec "1" uses it)
    parse_lanes(spec, cfg, report=None) -> list[Lane]
        "1"                  one lane, all threads: today's single-process loop (the default)
        "N"                  N CPU lanes with the thread budget // N threads each (at least 1; the budget is
                             ``priority.budget_of(cfg)``: threads 0 = the logical threads minus the free ones)
        "gpu:1x4,cpu:5x2"    explicit: 1 GPU lane with 4 threads + 5 CPU lanes with 2 threads. Items are
                             cpu|gpu|gpuK|mps[:count[xthreads]] or countxthreads (CPU); an item without
                             threads shares what the others leave. "gpu" is the GPU the settings select
                             (cuda:N or mps), "gpuK" is cuda:K; a GPU item without a usable GPU is an error.
        "auto"               from the hardware and the method, see plan_auto()
    jobs.run_resumable(..., lanes=list[Lane] | None)
        None or one lane: the single-process loop. More: run_parallel() below.

How a parallel run works (``run_parallel``):

* one ``multiprocessing`` process per lane, "spawn" start method on every OS (fork is unsafe with torch, Qt
  and threads; Windows only has spawn). The worker is the module-level ``_lane_main``, so it needs no
  ``if __name__ == "__main__"`` guard and starts from the GUI's QThread as well as from the CLI. The thread
  variables (OMP_NUM_THREADS, ...) are set to the lane's budget in the environment the child starts with,
  and the lane applies ``resources.apply_thread_limits`` before it imports the pipeline (and torch).
* run priority (``quantabench.priority``, ``BuildConfig.priority``): each lane lowers its own process before it
  imports anything (``priority.apply_to_process``: below-normal or idle class and lower I/O priority on Windows,
  nice and ionice on Linux, nice and the disk I/O policy on macOS), so the programs it starts inherit it; the
  parent (the GUI, a CLI build) is not lowered by the lanes.
* molecules are handed out pull-style: a lane gets the next molecule when it reports the last one done,
  so a fast lane (the GPU) simply takes more of them. Each lane runs ``pipeline.run_molecule`` and writes
  that molecule's folder (``export.write_molecule`` into run_dir/<id>), then sends the MoleculeResult back
  (RDKit Mol with its conformers and properties) over its pipe, with its log lines (prefixed "[lane]") and
  stage updates. The parent is the only process that writes jobs.sqlite (mark_running, record) and the
  only one that calls ``progress``/``log``/``on_result``, all in the calling thread.
* the largest molecules (atoms with hydrogens) are handed out first (``dispatch_order``), so the end of a
  batch is short molecules and no lane is left alone with a large one while the others are idle; molecules
  that crashed a process still go last. Results are returned in ``todo`` order.
* speed-aware end of a batch (``dist.speed``, on unless ``settings.resources.speed_aware_lanes`` or
  ``build --no-speed-aware-lanes`` turns it off; only for pools with lanes of more than one kind): a free
  lane leaves the next molecule in the queue when the lanes clearly faster than it (a GPU lane next to CPU
  lanes) would finish every queued molecule before it finished that one. Speeds per lane kind come from the
  molecules it finished (seconds per atom), else from this machine's benchmark in the registry, else are
  unknown (no holding back). Only ready lanes (or lanes still starting for the first time) are counted on,
  so a fast lane that dies releases the held molecules at once; one that stalls, after about one slow-lane
  molecule. The first time lanes hold back, one log line says which and why.
* dynamic CREST threads (``dist.threads``, on unless ``settings.resources.dynamic_threads`` or
  ``build --no-dynamic-threads`` turns it off; CREST conformer searches on CPU lanes only): the parent owns the CPU
  lanes' threads (with those of lanes left out because the run has fewer molecules than lanes) and, handing a
  molecule to a lane, says how many threads its CREST call may use: its own, or whole lanes' worth more taken
  from idle lanes when that is predicted to end the batch sooner ("[cpu1] 54950-05-9: CREST with 8 threads
  (borrowed 4 from idle lanes)"). The borrowed threads go back at the molecule's next stage, all of them when the
  molecule ends or its lane dies or is killed; a lane starts a molecule only when its own threads are free, so
  the threads in use never exceed the budget. A run of one molecule on a plan of several lanes gives its CREST
  search all of the plan's threads.
* timings (``chem.telemetry``): each lane measures its molecule's stages and sends the numbers with the result;
  the parent adds the lane, its threads and the CREST threads it lent, and writes the rows in the result's
  transaction. A lane that dies or is killed gets a partial row: its last stage, from when the parent heard of
  it. The CREST cost model is this machine's fitted one when ``timings --fit`` saved one (``threads.CostModel``).
* ``should_stop`` is polled by the parent and copied into a shared stop flag (``StopEvent``: Event semantics,
  no lock) that every lane's ``should_stop`` reads, so a cancel reaches the molecule running in each lane.
* a lane that dies (segfault, OOM kill, ``os._exit``) is restarted. Its molecule's crash count goes up,
  exactly like a crash of the single-process loop (``JobStore.reset_interrupted``): below ``MAX_CRASHES``
  it is tried again at the end of the queue, at ``MAX_CRASHES`` it is recorded as failed ("crashed the
  process N times") and the run goes on. Only a closed pipe means a lane died: an exception from a callback
  (``log``, ``progress``, ``on_result``) or the job table stops the run like any other exception, and a
  molecule that could not be handed to a lane (it died while idle) goes back to the queue uncounted.
* per-molecule time limit (``BuildConfig.molecule_timeout_s`` > 0, ``build --molecule-timeout``): the parent
  notes when it hands each lane its molecule. A lane still on it after the limit is killed (with its programs),
  the molecule is recorded failed with "time limit (N s) in <stage>" (no crash counted, so it is not retried in
  this run; ``--retry-failed`` redoes it) and the lane is restarted if work is left. A result that arrives just
  before the kill wins.
* memory: ``cfg.memory_budget_gb`` 0 (auto) is resolved once, here, to the available memory divided by the
  lanes (``chem.memory_guard``); every lane's dxtb steps are guarded against that share.
* external programs a lane starts (xtb, CREST, Psi4: own process group/session) are recorded by
  ``external.run_program`` in a per-lane folder; when a lane dies or is killed the parent kills what it
  left running (``LanePool.kill_programs``; on Windows with ``taskkill /T`` and, for ``wsl:`` programs, on
  the Linux side too).
* shutdown: lanes get an end-of-work message and are joined; on cancel the parent first waits for the
  running molecules to come back cancelled. On KeyboardInterrupt or an exception the lanes get the stop
  flag, a few seconds to stop their external programs, then are killed with their process trees; the
  molecules they held are recorded 'cancelled' (redone next time, no crash counted). Lane processes are
  daemonic (ended when the parent exits), and a lane whose parent was killed outright ends at its next
  read or write of the pipe, so at most after its current molecule.

``auto`` heuristic (``plan_auto``). Measured 2026-11-05 on BTM NOVA (Ryzen 9 7900X 12C/24T, Windows 11,
AMD's ROCm torch 2.9.1, 12 threads, other jobs sharing the machine) on PFAS: C2-C8 carboxylic and sulfonic
acids, 4:2-8:2 FTOH and FTSA, FBSA, FOSA, HFPO-DA, ADONA (24); wall times include 3-5 s of lane start-up:

    MMFF, 50 conformers, 24 PFAS      1 lane x 12 threads 43.4 s | 6 x 2 13.7 s (3.2x) | 12 x 1 11.9 s (3.6x)
                                      auto = 12 x 1: 11.7 s (3.7x); energies identical
    GFN1-xTB (dxtb, CPU), 6 conformers, 2 refined:
      12 PFAS (C2-C8 PFCA, C2-C6 PFSA) 1 x 12 172 s | 6 x 2 49.9 s (3.4x) | 12 x 1 58.2 s (3.0x)
      24 PFAS                          6 x 2 164 s | 12 x 1 146 s | auto = 6 x 2: 147 s (run-to-run +-10 %)
      lane vs single-process energies agree to < 1e-6 eV (2-thread lanes: < 4e-9 eV)
    2-CPU cloud VM, 8 small molecules: GFN1-xTB 1 x 2 threads 15.6 s, 1 x 1 15.3 s, 2 x 1 9.9 s (1.6x);
      MMFF finishes in ~3-9 s, where the ~2 s lane start-up eats the gain

* threads inside one PFAS-sized molecule buy nothing: GFN1-xTB took 18.8 s with 12 threads and 16.2 s with
  2 for the same three molecules, and 15.3 s with 1 vs 15.6 s with 2 on the cloud VM. Lanes scale until the
  physical cores are used.
* MMFF: one 1-thread lane per physical core.
* tblite (in-process Fortran, no torch): one lane per physical core, with the rest of the thread budget
  shared out (2 threads each on an SMT machine given all its threads). The full-run protocol (GFN2, ALPB
  water, 50 conformers, top 5, frequencies), 49 molecules in the full table's cohort mix, 2026-11-05:
  MacBook (M3 Max, 10 P + 4 E cores, no SMT): 6 x 2 181 s, 7 x 2 171 s, 10 x 1 151 s, 14 x 1 141 s (a second
  thread inside a molecule bought nothing: summed lane-seconds 891 at 6 x 2, 864 at 10 x 1). NOVA (Ryzen 9
  7900X, 12 C / 24 T, the fork's reference-BLAS wheel, one core busy with another job): 6 x 2 427 s,
  12 x 1 363 s, 24 x 1 331 s, 12 x 2 318 s. Energies agree across layouts to 1e-9 eV.
* in-process torch engines on the CPU (dxtb GFN1/GFN2, AIMNet2): 2-thread lanes on the physical cores (as
  fast as 1-thread lanes on all cores, with half the processes: each holds torch, ~0.5-1 GB).
* external programs (an engine with ``.external``, CREST conformer search, a QM stage): 2-thread lanes on
  all threads (the programs spend part of their time in I/O and start-up).
* CREST conformer search on a CPU with known performance cores (Apple Silicon, ``resources.performance_cores``:
  ``hw.perflevel0.physicalcpu``, the physical cores when that cannot be read): ``CREST_LANE_THREADS`` (4)
  threads per lane, lanes = performance cores / 4 rounded half up, at least 1 (the M3 Max's 10: 3 x 4, 12
  threads on 10 performance + 4 efficiency cores). GFN-FF gas phase, 20 PFAS, M3 Max, 2026-11-07: 1 x 12
  1,727 s, 3 x 4 773 s, 5 x 2 1,457 s, 6 x 2 (the 2-thread rule) 3,108 s. With 2 threads per lane the batch
  waits for its largest molecule (54 heavy atoms) on 2 threads while the other lanes are idle; one molecule on
  12 threads scales poorly. Elsewhere CREST keeps the 2-thread rule above: on NOVA (WSL CREST) the layout
  mattered little (4 large ions: 1 x 23 756 s, 4 x 5 769 s). Lanes are planned per run, not per stage: the
  refinement and properties after CREST (tblite, xtb, ...) run in the same lanes with the same threads; only the
  CREST call itself can run wider (dynamic threads, above).
* a GPU-capable engine with a GPU: one GPU lane with 4 threads (RDKit embedding and the host side of the
  batch run on the CPU) plus CPU lanes on the remaining threads by the rules above, when this machine's
  benchmark (``machines --bench <method>``, registry keys ``<method>@<device>``, ``+water`` for a run in
  water) says the GPU is the faster lane: ``<method>@<gpu>`` beats ``<method>@cpu``. Both are timed one
  molecule at a time in one process with the same threads, so this compares one lane with one lane; a GPU
  lane slower than a CPU lane only takes threads from the CPU lanes. The MacBook (AIMNet2: 6,387/h on the
  14-core CPU, 2,978/h on MPS) gets CPU lanes only; NOVA (AIMNet2 about 10x faster on the RX 9060 XT) gets
  its GPU lane. Without both numbers the GPU lane is planned as before, and the reason says so
  (``gpu_benchmark_verdict``). Not for dxtb (GFN1/GFN2):
  it computes in float32 on a GPU and float64 on the CPU, and a GPU lane next to CPU lanes would mix both in
  one table, molecule by molecule as timing decides. ``auto`` gives dxtb CPU lanes only, or with dxtb GPU =
  on (``BuildConfig.dxtb_gpu``) one lane on the GPU. It was not faster either: on NOVA's ROCm wheels (no
  MAGMA, ``gpu_linalg_available``) 1 GPU + 4 x 2 CPU lanes took 77.4 s for the 12 PFAS against 49.9 s for
  6 x 2 CPU lanes.
* lanes are capped by memory (0.75 x RAM over a per-lane working set; Psi4 adds its memory per job) and
  ``settings.resources.parallel_molecules`` (> 0) fixes the number of lanes. The plan never uses more
  threads than the run's thread budget (``priority.budget_of``: with CPU threads 0 the logical threads minus
  ``BuildConfig.free_threads``, at least 25 % and 2 of them at priority ``background``). On Windows a pool runs
  at most ``WINDOWS_MAX_LANES`` (31) lanes: it waits on two handles per lane and Windows waits on at most 63.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

#: Test hook: a lane whose molecule id is listed here (comma separated) calls ``os._exit(TEST_CRASH_EXIT)``,
#: standing in for a segfault or OOM kill; "TFE@1.5" crashes 1.5 s after the lane takes TFE (time for the
#: other lanes to start and be asked). Only tests set it.
TEST_CRASH_ENV = "QUANTABENCH_TEST_LANE_CRASH"
TEST_CRASH_EXIT = 86
#: Test hook: "cpu2=1.5,gpu0=0" makes the named lanes wait that many seconds after each molecule (a slow
#: lane next to a fast one, for the speed-aware tests). Only tests set it.
TEST_DELAY_ENV = "QUANTABENCH_TEST_LANE_DELAY"
#: Test hook: a lane makes a scratch folder holding a 2 MB ``psi.<pid>.97`` and a log when it takes a molecule, as
#: a Psi4 job does (a killed lane leaves it behind). Only tests set it.
TEST_SCRATCH_ENV = "QUANTABENCH_TEST_LANE_SCRATCH"
#: Elements method "auto" is resolved for when a lane plan is made (H C N O F S: PFAS and most organics)
TYPICAL_ELEMENTS = (1, 6, 7, 8, 9, 16)
#: Working set of one lane process in GB, by workload, for the memory cap of the auto plan
LANE_MEMORY_GB = {"mmff": 0.4, "native": 0.6, "torch": 1.2, "external": 0.6}
GPU_LANE_THREADS = 4
#: Threads per lane of the CREST plan on a CPU with known performance cores (Apple Silicon); see plan_auto
CREST_LANE_THREADS = 4
STARTUP_FAILURES = 3          # a lane that dies this often before it is ready is given up
#: Windows waits on at most 63 handles at once (WaitForMultipleObjects) and the pool waits on two per lane
#: (its pipe and its process), so more lanes would make every wait raise ValueError.
WINDOWS_MAX_LANES = 31
SHUTDOWN_GRACE_S = 5.0        # after stop: time for lanes to stop their programs before they are killed
POLL_S = 0.1


@dataclass(frozen=True)
class Lane:
    name: str            # "gpu0", "cpu1", ...
    device: str = "cpu"  # torch device string
    threads: int = 1     # CPU threads for this lane (BLAS, OpenMP, RDKit, external programs)

    def to_dict(self) -> dict:
        return {"name": self.name, "device": self.device, "threads": self.threads}


def _platform() -> str:
    return sys.platform


def max_lanes() -> int | None:
    """Most lanes one pool can run on this OS (``WINDOWS_MAX_LANES`` on Windows), None for no limit."""
    return WINDOWS_MAX_LANES if _platform() == "win32" else None


def _check_lane_count(n: int, spec) -> None:
    cap = max_lanes()
    if cap is not None and n > cap:
        raise ValueError(f"lane spec {spec!r}: {n} lanes is more than the {cap} one process can run on Windows "
                         "(it waits on two handles per lane, at most 63 at once)")


# ------------------------------------------------------------------------------------------ plans
def _cfg(cfg):
    if cfg is not None:
        return cfg
    from ..chem.pipeline import BuildConfig

    return BuildConfig()


@dataclass
class _CPUReport:
    """The part of ``resources.HardwareReport`` a plan needs, without importing torch."""
    physical_cores: int
    logical_cores: int
    ram_gb: float
    gpus: list
    torch_backend: str = "none"
    performance_cores: int = 0       # ``resources.performance_cores()``: Apple Silicon's, 0 = none known


def cpu_report() -> _CPUReport:
    import psutil

    from .. import resources

    return _CPUReport(resources.physical_cores(), resources.logical_cores(),
                      psutil.virtual_memory().total / 1024**3, [], "unknown", resources.performance_cores())


def _gpus(report) -> tuple[str, list]:
    """(backend, gpus) from ``report``; None asks PyTorch. A ``cpu_report()`` knows no GPU ("unknown")."""
    if report is not None:
        return getattr(report, "torch_backend", "none"), list(getattr(report, "gpus", []) or [])
    from .. import resources

    _, backend, gpus = resources.torch_gpus()
    return backend, gpus


def _gpu_device(cfg, report, index: int | None = None, want: str = "") -> str:
    policy = getattr(cfg, "gpu", "auto")
    if policy == "off":
        raise ValueError("the lane spec asks for a GPU lane, but the GPU is switched off (GPU = off in the Compute "
                         "settings, or --gpu off)")
    backend, gpus = _gpus(report)
    if not gpus and backend == "unknown":
        raise ValueError("the lane spec asks for a GPU lane; the GPU has not been detected yet")
    if not gpus:
        raise ValueError("the lane spec asks for a GPU lane, but PyTorch sees no usable GPU (`quantabench-cli "
                         "doctor` shows why); use CPU lanes, e.g. '4' or 'cpu:4x2'")
    if backend == "mps":
        return "mps"
    if want == "mps":
        raise ValueError("'mps' lanes need an Apple GPU; this machine's GPU is CUDA/ROCm: use 'gpu'")
    idx = int(getattr(cfg, "gpu_index", 0)) if index is None else index
    if not 0 <= idx < len(gpus):
        raise ValueError(f"lane spec: GPU {idx} does not exist (PyTorch sees {len(gpus)} GPU(s))")
    return f"cuda:{idx}"


def _gpu_lane_name(device: str, j: int) -> str:
    base = "mps" if device == "mps" else "gpu" + device.partition(":")[2]
    return base if j == 1 else f"{base}.{j}"


#: Engines whose GPU path needs torch.linalg (dxtb's SCF: cholesky, eigh)
GPU_LINALG_ENGINES = ("gfn1-xtb", "gfn2-xtb")


def gpu_linalg_available() -> bool:
    """Whether torch can do linear algebra on the GPU. AMD's Windows ROCm wheels cannot (no MAGMA): dxtb's
    cholesky fails on cuda:0 and every molecule of a GPU lane falls back to the CPU after a failed try."""
    try:
        import torch
    except Exception:
        return False
    return bool(not getattr(torch.version, "hip", None) or getattr(torch.cuda, "has_magma", False))


def workload(cfg) -> tuple[str, str, bool]:
    """(engine, kind, gpu_capable) of a build: kind "mmff" | "native" (in-process compiled code without torch:
    tblite) | "torch" (in-process PyTorch: dxtb, AIMNet2) | "external" (xtb, MOPAC, CREST or Psi4 run as
    programs). Method "auto" is resolved for typical organic elements."""
    from ..chem import engines, solvents

    cfg = _cfg(cfg)
    method = cfg.method
    infos = engines.all_engines()
    if method == "auto" or method not in infos:
        method = engines.resolve_auto(list(TYPICAL_ELEMENTS), solvents.canonical(cfg.solvent))
    info = infos[method]
    if info.external or cfg.conformer_search == "crest" or cfg.qm_method:
        kind = "external"
    elif method == "mmff":
        kind = "mmff"
    elif not info.gpu_capable:
        kind = "native"
    else:
        kind = "torch"
    return method, kind, bool(info.gpu_capable and method != "mmff")


def _qm_memory_gb(cfg=None) -> float:
    """Memory of one Psi4 job (``chem.qm.memory_gb``: ``--qm-memory``, else the setting)."""
    from ..chem import qm

    try:
        return qm.memory_gb(cfg)
    except Exception:
        return 4.0


def _parallel_molecules_setting() -> int:
    from .. import settings as settings_mod

    try:
        return int(settings_mod.load().resources.parallel_molecules)
    except Exception:
        return 0


_THIS_MACHINE = object()


def _this_profile():
    """This machine's registry profile (``dist.profile``), None when it has none or it cannot be read."""
    try:
        from . import profile as profile_mod

        return profile_mod.load_profile(profile_mod.machine_name())
    except Exception:  # noqa: BLE001 - no registry: no benchmark
        return None


def gpu_benchmark_verdict(method: str, device: str, profile, solvent: str = "") -> tuple[bool | None, str]:
    """Whether a GPU lane on ``device`` pays for ``method`` by this machine's benchmark (``machines --bench``):
    True when ``<method>@<gpu>`` beats ``<method>@cpu``, False when it does not, None without both numbers.
    Both benchmarks run one molecule at a time in one process with the same threads, i.e. as one lane each,
    so the comparison is per lane: a GPU lane slower than a CPU lane only takes threads from the CPU lanes.
    Rates come from ``speed.registry_seeds`` (solvent-specific keys, per-set molecule sizes). The text is
    the clause for the plan's reason."""
    from . import speed
    from .profile import bench_method

    if profile is None:
        return None, f"no benchmark of {method} on this machine: GPU lane by default"
    name = bench_method(method, solvent)
    seeds = speed.registry_seeds([Lane("gpu", device), Lane("cpu", "cpu")], name, profile)
    gpu, cpu = seeds.get("gpu"), seeds.get("cpu")
    if gpu is None or cpu is None:
        missing = " and ".join(k for k, v in ((f"{name}@{device}", gpu), (f"{name}@cpu", cpu)) if v is None)
        return None, f"no benchmark {missing} on this machine: GPU lane by default"
    from .profile import _fmt_rate, throughput_for

    rg, rc = throughput_for(profile, name, device) or 0.0, throughput_for(profile, name, "cpu") or 0.0
    numbers = f"benchmark {name}@{device} {_fmt_rate(rg)}/h vs @cpu {_fmt_rate(rc)}/h"
    if gpu < cpu:                     # seconds per size unit: smaller is faster
        return True, numbers
    return False, f"no GPU lane: {numbers}, so a GPU lane would be slower than a CPU lane"


def plan_auto(cfg=None, report=None, parallel_molecules: int = 0, profile=_THIS_MACHINE) -> tuple[list[Lane], str]:
    """The lanes of spec "auto" and a one-line reason. ``report``: a ``resources.HardwareReport`` (or
    ``cpu_report()``, which knows no GPU); None detects the GPU through PyTorch only when the engine can use
    one. ``profile``: the registry profile whose benchmark decides whether a GPU lane pays
    (``gpu_benchmark_verdict``); default this machine's, None = no benchmark. See the module docstring for
    the rules and the measurements behind them."""
    from .. import priority

    cfg = _cfg(cfg)
    total = priority.budget_of(cfg)       # threads 0: all but the free ones; an explicit count as given
    hw = report if report is not None else cpu_report()
    cores = max(1, min(int(hw.physical_cores or total), total))
    perf = max(0, min(int(getattr(hw, "performance_cores", 0) or 0), cores))
    crest = getattr(cfg, "conformer_search", "") == "crest"
    method, kind, gpu_capable = workload(cfg)
    device = "cpu"
    why_gpu = ""
    if gpu_capable and cfg.gpu != "off":
        try:
            device = _gpu_device(cfg, report)
        except ValueError:
            why_gpu = ("GPU not detected yet" if getattr(report, "torch_backend", "") == "unknown"
                       else "no usable GPU")
    elif gpu_capable:
        why_gpu = "GPU off"
    if device != "cpu" and method in GPU_LINALG_ENGINES:
        # dxtb computes in float32 on a GPU and in float64 on the CPU: a GPU lane next to CPU lanes would mix
        # both in one table, molecule by molecule as timing decides. One device per run instead.
        from ..chem.torch_compat import DXTB_GPU_ENV

        if (os.environ.get(DXTB_GPU_ENV, "").strip().lower() or getattr(cfg, "dxtb_gpu", "auto")) == "on":
            return [Lane("main", "auto", total)], (f"{method} with dxtb GPU = on: one lane on {device}, so every "
                                                    "molecule of the run is computed in the same precision")
        if device.startswith("cuda") and not gpu_linalg_available():
            why_gpu = "no GPU lane: this torch build has no GPU linear algebra (MAGMA), which dxtb needs"
        else:
            why_gpu = ("no GPU lane: dxtb would compute its molecules in float32 next to float64 CPU lanes; "
                       "one precision per run (dxtb GPU = on runs everything on the GPU)")
        device = "cpu"
    bench_why = ""
    if device != "cpu":           # a GPU lane only where this machine's benchmark says it is the faster lane
        verdict, bench_why = gpu_benchmark_verdict(method, device, _this_profile() if profile is _THIS_MACHINE
                                                   else profile, getattr(cfg, "solvent", ""))
        if verdict is False:
            why_gpu, bench_why, device = bench_why, "", "cpu"
    cap = max_lanes()
    per_lane_gb = LANE_MEMORY_GB[kind] + (_qm_memory_gb(cfg) if cfg.qm_method else 0.0)
    by_memory = max(1, int(float(hw.ram_gb or 0) * 0.75 / per_lane_gb)) if hw.ram_gb else total

    lanes: list[Lane] = []
    if parallel_molecules > 0:
        n = max(1, min(int(parallel_molecules), total, cap or total))
        per = max(1, total // n)
        if device != "cpu":
            lanes.append(Lane(_gpu_lane_name(device, 1), device, per))
        lanes += [Lane(f"cpu{i + 1}", "cpu", per) for i in range(n - len(lanes))]
        reason = f"parallel molecules = {parallel_molecules} (Compute settings)"
    else:
        budget, cpu_cores = total, cores
        if device != "cpu":
            g = GPU_LANE_THREADS if total >= 2 * GPU_LANE_THREADS else max(1, total // 2)
            lanes.append(Lane(_gpu_lane_name(device, 1), device, g))
            budget -= g
            cpu_cores = max(0, cores - g)
        if kind == "mmff":
            per, wanted = 1, min(cpu_cores, budget)
            reason = "MMFF: one 1-thread lane per physical core"
        elif kind == "native":
            wanted = min(cpu_cores, budget)
            per = max(1, budget // max(1, wanted))
            reason = (f"{method}: one lane per physical core, {per} thread{'s' if per > 1 else ''} each"
                      + (" (all the threads: SMT)" if per > 1 else ""))
        elif kind == "external" and crest and perf:
            # CREST on Apple Silicon: about 4 threads per lane on the performance cores (module docstring)
            fast = max(1, perf - (lanes[0].threads if lanes else 0))     # performance cores a GPU lane leaves
            per = min(CREST_LANE_THREADS, budget)
            wanted = max(1, min(int(fast / CREST_LANE_THREADS + 0.5), budget // per))    # rounded half up
            reason = (f"CREST: {per}-thread lanes, one per {CREST_LANE_THREADS} of the {perf} performance cores "
                      "(rounded)")
        elif kind == "external":
            per, wanted = 2, budget // 2
            reason = ("CREST" if crest else "external programs") + ": 2-thread lanes on all threads"
        else:
            per, wanted = 2, min(cpu_cores, budget) // 2
            reason = f"{method} on the CPU: 2-thread lanes on the physical cores"
        n = max(0, min(wanted, by_memory - len(lanes)))
        if n < wanted:
            reason += f", capped by memory ({hw.ram_gb:.0f} GB)"
            if cfg.qm_method and n > 1:     # Psi4 scales with threads: the lanes that fit share the budget
                per = max(per, budget // n)
                reason += f" ({_qm_memory_gb(cfg):g} GB per QM job): {per} threads each"
        if cap is not None and n + len(lanes) > cap:
            n = cap - len(lanes)
            reason += f", capped at {cap} lanes (Windows)"
        if not lanes and n <= 1:
            return [Lane("main", "auto", total)], f"{reason}: room for one lane only{_free_note(cfg)}"
        lanes += [Lane(f"cpu{i + 1}", "cpu", per) for i in range(n)]
        if lanes[0].device != "cpu":
            reason = f"{method} on {device}: 1 GPU lane with {lanes[0].threads} threads; " + reason
    if why_gpu:
        reason += f" ({why_gpu})"
    if bench_why:
        reason += f" ({bench_why})"
    reason += _free_note(cfg)
    if len(lanes) == 1:
        return [Lane("main", "auto", total)], reason
    return lanes, reason


def _free_note(cfg) -> str:
    """'; 6 of 24 threads left free (background)' when a run leaves more than today's one thread free."""
    from .. import priority, resources

    level, free = priority.of(cfg)
    n = resources.logical_cores()
    left = priority.effective_free_threads(free, level, n)
    if left <= priority.DEFAULT_FREE_THREADS:
        return ""
    return f"; {left} of {n} threads left free" + (f" ({level})" if level == "background" else "")


_ITEM = re.compile(r"^(?:(cpu|gpu|mps)(\d*)(?::(\d+)(?:x(\d+))?)?|(\d+)x(\d+))$")


def _explicit(text: str, cfg, report, total: int) -> list[Lane]:
    from .. import resources

    items = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        m = _ITEM.match(part)
        if m is None:
            raise ValueError(f"lane spec {text!r}: cannot read {part!r}. Use 1, auto, a number of CPU lanes, or "
                             "a list like gpu:1x4,cpu:5x2 (lanes x threads each)")
        kind, gidx, count, threads, n2, t2 = m.groups()
        if n2:
            kind, gidx, count, threads = "cpu", "", n2, t2
        if gidx and kind != "gpu":
            raise ValueError(f"lane spec {text!r}: {part!r}: only 'gpu' takes an index (gpu0, gpu1, ...)")
        count_n = int(count) if count else 1
        threads_n = int(threads) if threads else None
        if count_n < 1 or (threads_n is not None and threads_n < 1):
            raise ValueError(f"lane spec {text!r}: {part!r} needs at least 1 lane and 1 thread")
        if kind == "cpu":
            device = "cpu"
        else:
            device = _gpu_device(cfg, report, int(gidx) if gidx else None, want="mps" if kind == "mps" else "")
        items.append((device, count_n, threads_n))
    if not items:
        raise ValueError("empty lane spec")
    fixed = sum(c * t for _, c, t in items if t is not None)
    free = sum(c for _, c, t in items if t is None)
    shared = max(1, (total - fixed) // free) if free else 0
    lanes, per_gpu, n_cpu = [], {}, 0
    for device, count_n, threads_n in items:
        for _ in range(count_n):
            if device == "cpu":
                n_cpu += 1
                name = f"cpu{n_cpu}"
            else:
                per_gpu[device] = per_gpu.get(device, 0) + 1
                name = _gpu_lane_name(device, per_gpu[device])
            lanes.append(Lane(name, device, threads_n if threads_n is not None else shared))
    logical = int(getattr(report, "logical_cores", 0) or resources.logical_cores())
    used = sum(x.threads for x in lanes)
    if used > logical:
        raise ValueError(f"lane spec {text!r} uses {used} threads, more than the {logical} logical cores of this "
                         "machine")
    _check_lane_count(len(lanes), text)
    # GPU lanes first: a run with fewer molecules than lanes uses the first lanes of the list
    return sorted(lanes, key=lambda x: x.device == "cpu")


def parse_lanes(spec: str | int | None, cfg=None, report=None) -> list[Lane]:
    """The lanes for ``spec`` (see the module docstring). Raises ValueError with a readable message for a spec
    that cannot be used here (bad syntax, a GPU lane without a GPU, more threads than logical cores)."""
    from .. import priority, resources

    cfg = _cfg(cfg)
    total = priority.budget_of(cfg)          # threads 0: all but the free threads
    text = "1" if spec is None else str(spec).strip().lower()
    if text in ("", "1"):
        return [Lane("main", "auto", total)]
    if text == "auto":
        return plan_auto(cfg, report, _parallel_molecules_setting())[0]
    if text.isdigit():
        n = int(text)
        if n < 1:
            raise ValueError(f"lane spec {spec!r}: need at least one lane")
        logical = int(getattr(report, "logical_cores", 0) or resources.logical_cores())
        if n > logical:
            raise ValueError(f"lane spec {spec!r}: {n} lanes is more than the {logical} logical cores of this machine")
        _check_lane_count(n, spec)
        return [Lane(f"cpu{i + 1}", "cpu", max(1, total // n)) for i in range(n)]
    return _explicit(text, cfg, report, total)


def spec_error(spec: str | None) -> str:
    """'' if ``spec`` is well formed, else why not. Says nothing about this machine (a GPU lane on a PC without
    a GPU is well formed), so the settings dialog can store a spec meant for the GPU before detection ran."""
    from .. import resources

    many = resources.HardwareReport(os="", python="", cpu_name="", physical_cores=4096, logical_cores=4096,
                                    ram_gb=1e6, wsl=False, torch_backend="cuda",
                                    gpus=[resources.GPUInfo(i, "", "cuda", 1.0) for i in range(64)])
    try:
        parse_lanes(spec, None, many)
    except ValueError as exc:
        return str(exc)
    return ""


def describe(lanes: list[Lane]) -> str:
    """'1 GPU lane (cuda:0, 4 threads) + 5 CPU lanes (2 threads each) = 14 threads'."""
    def threads(n: int) -> str:
        return f"{n} thread{'s' if n != 1 else ''}"

    if len(lanes) == 1 and lanes[0].device == "auto":
        return f"1 lane (one process), {threads(lanes[0].threads)}, device from the GPU setting"
    groups: dict[tuple[str, int], int] = {}
    for x in lanes:
        groups[(x.device, x.threads)] = groups.get((x.device, x.threads), 0) + 1
    parts = []
    for (device, t), n in groups.items():
        kind = "CPU" if device == "cpu" else "GPU"
        where = f"{device}, " if device != "cpu" else ""
        parts.append(f"{n} {kind} lane{'s' if n > 1 else ''} ({where}{threads(t)}{' each' if n > 1 else ''})")
    return " + ".join(parts) + f" = {threads(sum(x.threads for x in lanes))}"


def plan_text(spec: str | None, cfg=None, report=None, parallel_molecules: int | None = None) -> str:
    """Human-readable plan for a lane spec, never raises (for the settings dialog). ``report`` None uses
    ``cpu_report()``: no torch import, so GPU lanes show as unavailable until hardware detection has run."""
    from .. import priority

    cfg = _cfg(cfg)
    hw = report if report is not None else cpu_report()
    total = priority.budget_of(cfg)
    try:
        text = "1" if spec is None else str(spec).strip().lower()
        if text == "auto":
            pm = _parallel_molecules_setting() if parallel_molecules is None else parallel_molecules
            lanes, reason = plan_auto(cfg, hw, pm)
            out = f"{describe(lanes)} of {total}. {reason}."
        else:
            lanes = parse_lanes(text, cfg, hw)
            out = f"{describe(lanes)} of {total}."
    except Exception as exc:     # noqa: BLE001 - shown to the user as text
        out = f"Not usable here: {exc}"
    if report is None and getattr(hw, "torch_backend", "") == "unknown":
        out += " (GPU not detected yet)"
    return out


# ---------------------------------------------------------------------------------- lane process
_env_lock = threading.Lock()


@contextlib.contextmanager
def _child_env(threads: int):
    """Thread variables of the lane in the environment a child process starts with: libraries the child
    loads before its first line of our code (numpy's BLAS when the main module is re-imported) see them."""
    from .. import resources

    with _env_lock:
        saved = {k: os.environ.get(k) for k in resources.THREAD_ENV_VARS}
        try:
            for k in resources.THREAD_ENV_VARS:
                os.environ[k] = str(threads)
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def lane_device(lane: Lane, cfg, memory_fraction: float) -> str:
    from .. import resources

    if lane.device == "auto":
        return resources.select_device(cfg.gpu, cfg.gpu_index, cfg.gpu_memory_fraction)
    if lane.device.startswith("cuda") and memory_fraction:
        try:
            import torch

            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), torch.device(lane.device))
        except Exception:  # noqa: S110 - a cap is best effort
            pass
    return lane.device


def _test_delay(name: str) -> float:
    for item in os.environ.get(TEST_DELAY_ENV, "").split(","):
        lane, _, seconds = item.partition("=")
        if lane.strip() == name:
            with contextlib.suppress(ValueError):
                return max(0.0, float(seconds))
    return 0.0


def _test_scratch(mol_id: str) -> None:
    """The scratch a Psi4 job leaves while it runs (test hook ``TEST_SCRATCH_ENV``)."""
    from ..chem import external

    d = external.make_scratch_dir(f"psi4_{mol_id}")
    (d / f"psi.{os.getpid()}.97").write_bytes(b"\0" * 2_000_000)
    (d / "output.dat").write_text("==> DiskDFJK: Density-Fitted J/K Matrices <==\n", encoding="utf-8")


def _test_crashes() -> dict[str, float]:
    """Molecule id -> seconds before the crash, from the test hook ``TEST_CRASH_ENV``."""
    out = {}
    for item in os.environ.get(TEST_CRASH_ENV, "").split(","):
        mol_id, _, seconds = item.strip().partition("@")
        if mol_id:
            try:
                out[mol_id] = max(0.0, float(seconds)) if seconds else 0.0
            except ValueError:
                out[mol_id] = 0.0
    return out


def _lane_main(*args, **kwargs) -> None:
    """Entry of one lane process (``_lane_loop``), with a hard exit where the GPU runtime would hang at
    shutdown (``resources.hard_exit_if_needed``: Windows ROCm). Module level, for the spawn start method."""
    _lane_loop(*args, **kwargs)
    from .. import resources

    resources.hard_exit_if_needed(0)


def _lane_loop(lane: Lane, cfg_dict: dict, run_dir: str, conn, stop_event, memory_fraction: float,
               programs_dir: str = "") -> None:
    """Body of one lane process. Module level, so the spawn start method can import it on every OS.
    ``programs_dir``: where ``external.run_program`` records the programs this lane starts, so the parent
    can kill them if the lane dies before it can (``LanePool.kill_programs``)."""
    import signal

    with contextlib.suppress(ValueError, OSError, AttributeError):
        signal.signal(signal.SIGINT, signal.SIG_IGN)    # Ctrl+C is the parent's to handle

    def send(*msg) -> None:
        conn.send((lane.name, *msg))

    try:
        from dataclasses import replace

        from .. import priority, resources

        # this lane process and every program it starts at the run's priority (never the GUI: lanes are its children)
        priority.apply_to_process(cfg_dict.get("priority", priority.DEFAULT_PRIORITY))
        threads = resources.apply_thread_limits(lane.threads)     # before the pipeline (and torch) loads
        from pathlib import Path

        from rdkit import Chem

        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        from ..chem import export, external, jobs, pipeline, telemetry

        external.set_program_registry(programs_dir or None)

        cfg = pipeline.BuildConfig(**{**cfg_dict, "threads": threads})
        # this lane's memory share and time limit for the steps not handed cfg (embedding, Hessian)
        resources.set_molecule_limits(cfg)
        device = lane_device(lane, cfg, memory_fraction)
        crash_ids = _test_crashes()
        delay = _test_delay(lane.name)
    except BaseException:
        with contextlib.suppress(Exception):
            send("fatal", traceback.format_exc())
        return
    send("ready", os.getpid(), device, threads)
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):       # the parent went away
            return
        if task is None:
            return
        k, mol_id, text, *more = task
        # more[0]: threads the parent lends this molecule's CREST call (dist.threads); 0 = the lane's own
        crest_threads = int(more[0]) if more and more[0] else 0
        cfg_mol = replace(cfg, crest_threads=crest_threads) if crest_threads else cfg
        try:
            inp = jobs.input_from_json(text)
            if mol_id in crash_ids:
                time.sleep(crash_ids[mol_id])
                os._exit(TEST_CRASH_EXIT)
            if os.environ.get(TEST_SCRATCH_ENV):
                _test_scratch(mol_id)
            res = pipeline.run_molecule(inp, cfg_mol, device, threads, stop_event.is_set,
                                        log=lambda m: send("log", m),
                                        stage=lambda s, k=k, mol_id=mol_id: send("stage", k, mol_id, s))
            t_end = time.monotonic() + delay
            while time.monotonic() < t_end and not stop_event.is_set():
                time.sleep(min(0.05, max(0.0, t_end - time.monotonic())))
            result_dir = ""
            if cfg.write_files and res.status == "ok":
                with telemetry.export_clock(res, threads):
                    try:
                        files = export.write_molecule(res, Path(run_dir) / mol_id, cfg.formats)
                        res.files = [str(p) for p in files]
                        result_dir = mol_id
                    except Exception as exc:  # a disk or format error must not end an overnight run
                        res.status, res.error = "failed", f"export failed: {type(exc).__name__}: {exc}"
                        send("log", f"  {mol_id}: {res.error}")
            try:
                send("result", k, mol_id, res, result_dir)
            except (OSError, EOFError):
                return
            except Exception as exc:    # the result does not pickle: return it without the RDKit molecule
                res.mol = None
                res.notes.append(f"molecule object not returned from lane {lane.name} ({type(exc).__name__}); "
                                 "the files in the run folder are complete")
                send("result", k, mol_id, res, result_dir)
        except (OSError, EOFError):
            return
        except Exception:
            with contextlib.suppress(Exception):
                send("error", k, mol_id, traceback.format_exc())


# -------------------------------------------------------------------------------------- the pool
class _LaneProcess:
    def __init__(self, lane: Lane):
        self.lane = lane
        self.proc = None
        self.conn = None
        self.ready = False
        self.busy: tuple | None = None   # (k, mol_id, inp)
        self.pid = 0
        self.device = lane.device
        self.startup_failures = 0
        self.retired = False
        self.done = 0
        self.started_at = 0.0       # time.monotonic() of the last start
        self.ever_ready = False     # has been ready once (a restart is not a first start)
        self.busy_since = 0.0       # time.monotonic() when the running molecule was handed over (time limit)
        self.stage = ""             # the running molecule's last stage, for the time-limit error
        self.stage_since = 0.0      # time.time() when that stage began (the partial telemetry row of a lost lane)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.exitcode is None


def _kill_tree(proc) -> None:
    """Kill a lane process and the programs it started (xtb, CREST and Psi4 run in their own process groups,
    so killing the lane alone would leave them running). The lane itself is killed and reaped through
    multiprocessing: psutil must not wait on it, or ``proc.exitcode`` could never be read."""
    import psutil

    try:
        kids = psutil.Process(proc.pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        kids = []
    for p in kids:
        with contextlib.suppress(psutil.Error):
            p.kill()
    with contextlib.suppress(Exception):
        proc.kill()
    if kids:
        psutil.wait_procs(kids, timeout=5)
    proc.join(5)


class StopEvent:
    """The cancel flag every lane polls: ``multiprocessing.Event``'s set()/is_set() on one shared byte. An Event
    takes a lock in is_set(); a lane killed while holding it (an OOM kill lands anywhere, and lanes poll this
    in every optimizer step) would block every other lane and the parent. A byte needs no lock."""

    def __init__(self, ctx):
        self._flag = ctx.RawValue("b", 0)

    def set(self) -> None:
        self._flag.value = 1

    def clear(self) -> None:
        """Only while no lane runs a molecule (a pause "now" that has stopped them all): idle lanes do not read it."""
        self._flag.value = 0

    def is_set(self) -> bool:
        return bool(self._flag.value)


class LanePool:
    """Worker processes for ``lanes``; used by ``run_parallel``. Not thread-safe: one thread drives it."""

    def __init__(self, lanes: list[Lane], cfg, run_dir, log=lambda _m: None):
        import multiprocessing

        self.ctx = multiprocessing.get_context("spawn")
        self.stop_event = StopEvent(self.ctx)
        self.cfg_dict = asdict(cfg)
        self.run_dir = str(run_dir)
        self.log = log
        self.workers = [_LaneProcess(x) for x in lanes]
        shares: dict[str, int] = {}
        for x in lanes:
            shares[x.device] = shares.get(x.device, 0) + 1
        self.fractions = {x.device: (cfg.gpu_memory_fraction or 0.0) / shares[x.device] for x in lanes}
        self.last_fatal = ""
        # external programs each lane started (xtb, CREST, Psi4 run in their own process groups and would
        # outlive a lane that dies or is killed): kill_programs() ends them
        self.programs_dir = Path(tempfile.mkdtemp(prefix="qb_lane_programs_"))
        self.lane_pids: set[int] = set()        # every lane process started: close() sweeps their scratch

    # -- processes
    def start(self, w: _LaneProcess) -> None:
        parent_conn, child_conn = self.ctx.Pipe(duplex=True)
        proc = self.ctx.Process(target=_lane_main, name=f"quantabench-lane-{w.lane.name}", daemon=True,
                                args=(w.lane, self.cfg_dict, self.run_dir, child_conn, self.stop_event,
                                      self.fractions.get(w.lane.device, 0.0), str(self.programs_of(w))))
        with _child_env(w.lane.threads):
            proc.start()
        child_conn.close()
        w.proc, w.conn, w.ready, w.busy, w.pid = proc, parent_conn, False, None, proc.pid
        self.lane_pids.add(proc.pid)
        w.started_at = time.monotonic()

    def start_all(self) -> None:
        for w in self.workers:
            self.start(w)

    def pids(self) -> list[int]:
        return [w.pid for w in self.workers if w.pid]

    def wait(self, timeout: float = POLL_S) -> list:
        """Pipes with something to read, and sentinels of lanes that ended (both wake the caller)."""
        from multiprocessing.connection import wait

        objs = [w.conn for w in self.workers if w.conn is not None]
        objs += [w.proc.sentinel for w in self.workers if w.proc is not None and w.alive]
        if not objs:
            time.sleep(timeout)
            return []
        return wait(objs, timeout)

    def owner(self, conn) -> _LaneProcess | None:
        return next((w for w in self.workers if w.conn is not None and w.conn is conn), None)

    def programs_of(self, w: _LaneProcess) -> Path:
        return self.programs_dir / w.lane.name

    def kill_programs(self, w: _LaneProcess) -> int:
        """Kill the external programs lane ``w`` recorded and left running (call once it has ended)."""
        from ..chem import external

        try:
            return external.kill_recorded_programs(self.programs_of(w))
        except Exception:  # noqa: BLE001 - best effort, must not end the run
            return 0

    def close(self, graceful: bool = True) -> None:
        """End every lane: an end-of-work message, then join. Lanes still running after the grace time (30 s,
        or SHUTDOWN_GRACE_S after the stop flag when not graceful) are killed with their child processes."""
        if not graceful:
            self.stop_event.set()
        for w in self.workers:
            if w.proc is not None and w.alive and not w.ready and w.busy is None:
                _kill_tree(w.proc)             # still importing and holds no work: no need to wait for it
            elif w.conn is not None:
                with contextlib.suppress(OSError, ValueError):
                    w.conn.send(None)
        deadline = time.monotonic() + (30.0 if graceful else SHUTDOWN_GRACE_S)
        while time.monotonic() < deadline and any(w.alive for w in self.workers):
            for obj in self.wait(POLL_S):           # keep reading, so no lane blocks in a send
                w = self.owner(obj)
                if w is None:
                    continue
                try:
                    while w.conn is not None and w.conn.poll():
                        w.conn.recv()
                except (EOFError, OSError, ValueError):
                    self._close_conn(w)
        for w in self.workers:
            if w.proc is not None:
                if w.alive:
                    _kill_tree(w.proc)
                w.proc.join(5)
                with contextlib.suppress(ValueError):
                    w.proc.close()
                w.proc = None
            self._close_conn(w)
            n = self.kill_programs(w)
            if n:
                self.log(f"[{w.lane.name}] stopped {n} external program(s) the lane left running")
        shutil.rmtree(self.programs_dir, ignore_errors=True)
        if self.lane_pids:                  # what lanes killed at the end (a deadline, a cancel) left in scratch
            from ..chem import external

            external.sweep_scratch_quietly(owners=set(self.lane_pids), log=self.log, what="of the lanes")

    @staticmethod
    def _close_conn(w: _LaneProcess) -> None:
        if w.conn is not None:
            with contextlib.suppress(OSError):
                w.conn.close()
            w.conn = None


def speed_aware_setting() -> bool:
    """``settings.resources.speed_aware_lanes`` (on unless switched off)."""
    from .. import settings as settings_mod

    try:
        return bool(settings_mod.load().resources.speed_aware_lanes)
    except Exception:  # noqa: BLE001 - unreadable settings: the default
        return True


def speed_policy(lanes: list[Lane], cfg):
    """The ``speed.SpeedPolicy`` for a pool, seeded from this machine's benchmark; None when every lane is of
    one kind (same device and threads): such lanes never hold back for each other."""
    from . import speed

    if len({(x.device, x.threads) for x in lanes}) < 2:
        return None
    method = cfg.method
    if method == "auto":
        try:
            method = workload(cfg)[0]
        except Exception:  # noqa: BLE001 - no seed then
            method = ""
    return speed.SpeedPolicy(lanes, speed.registry_seeds(lanes, method, solvent=getattr(cfg, "solvent", "")))


def dispatch_order(todo, size_of, crashes: dict[str, int] | None = None) -> list[int]:
    """Indices into ``todo`` ([(id, MoleculeInput)]) in the order a parallel run hands the molecules out:
    largest first (``size_of(id)``: ``speed.molecule_size``, atoms with hydrogens), so the end of the batch is
    short molecules and no lane is left alone with a large one while the others are idle (the CREST runs on
    the Mac waited 51 min for a 54-heavy-atom molecule on 2 threads). Molecules that crashed a process
    (``crashes``) still go last, as in ``JobStore.todo``; equal sizes keep the input order."""
    crashes = crashes or {}
    return sorted(range(len(todo)), key=lambda j: (crashes.get(todo[j][0], 0), -size_of(todo[j][0])))


def dynamic_threads_for(cfg, lanes: list[Lane], enabled: bool | None = None, budget: int = 0,
                        logical: int | None = None) -> int:
    """The thread budget of dynamic CREST threads (``dist.threads``) for a run of ``cfg`` on ``lanes``, 0 when they
    are off: only for CREST conformer searches, on CPU lanes, when ``enabled`` (None: the setting) and the budget
    (``budget``, at least the CPU lanes' threads) has room for two lanes' worth of the narrowest lane.

    Borrowed threads never eat into the free threads: with CPU threads 0 (auto) the budget is cut to
    ``priority.cap_of`` (the logical threads, ``logical`` or this machine's, minus the run's free threads) minus the
    GPU lanes' threads, but never below the CPU lanes' own threads (an explicit lane spec may use more; it then
    borrows nothing). An explicit thread count keeps the plan's budget, as before."""
    from .. import priority
    from . import threads as threads_mod

    cpu = [x.threads for x in lanes if x.device == "cpu"]
    if not cpu or getattr(cfg, "conformer_search", "") != "crest":
        return 0
    if enabled is None:
        enabled = threads_mod.setting()
    if not enabled:
        return 0
    own = sum(cpu)
    total = max(int(budget or 0), own)
    cap = priority.cap_of(cfg, logical)
    if cap is not None:
        total = max(own, min(total, cap - sum(x.threads for x in lanes if x.device != "cpu")))
    return total if total >= 2 * min(cpu) else 0


def run_parallel(store, todo, cfg, lanes: list[Lane], run_dir, *, total: int, done_before: int,
                 progress, should_stop, log, on_result, keep_results: bool = True,
                 speed_aware: bool | None = None, handoff=None, dynamic_threads: bool | None = None,
                 thread_budget: int = 0, telemetry=None, cost_model=None, control=None) -> list:
    """Run ``todo`` ([(id, MoleculeInput)], from ``JobStore.todo``) in ``lanes`` worker processes. The
    calling thread is the only one that writes ``store`` and calls the callbacks. Returns the results in
    ``todo`` order (molecules not started because of a cancel are absent).

    ``speed_aware`` (None: ``settings.resources.speed_aware_lanes``): near the end of the batch, a lane
    leaves the last molecules to clearly faster lanes when they would finish them first (``dist.speed``).
    Off, every free lane takes the next molecule.

    ``dynamic_threads`` (None: ``settings.resources.dynamic_threads``): a CREST search may also use the threads of
    idle CPU lanes (``dist.threads``), within ``thread_budget`` (the planned CPU lanes' threads, at least those
    of ``lanes``).

    ``handoff`` (``jobs.Handoff``): molecules not started yet can be handed to another machine; they are
    taken from the end of the queue between dispatches and marked 'moved'.

    ``telemetry`` (``chem.telemetry.RunTelemetry``): the run's timing rows; each lane sends its stage numbers with
    the result, and this process writes them next to the result (a lane that dies or is killed gets a partial row
    from its last stage). ``cost_model`` (``threads.CostModel``): the CREST cost model of the dynamic threads;
    None = this machine's fitted one, else the built-in.

    ``control`` (``jobs.BatchControl``, DIST-7): while its mode is "hold" or "drain" no molecule is handed out (the
    lanes stay up; releases are still served); "drain" returns once no lane is busy. With ``interrupt`` the lanes'
    molecules stop at their next checkpoint (the shared stop flag, cleared again once every lane is idle) and are
    recorded 'cancelled', i.e. not started: a hold puts them back at the front of the queue. A new ``priority`` is
    applied to the running lane processes and the programs they started (``priority.apply_to_pids``) and to lanes
    started later."""
    from dataclasses import replace

    from ..chem import jobs
    from ..chem import pipeline as pipeline_mod
    from ..chem import telemetry as telemetry_mod
    from ..chem.pipeline import MoleculeResult
    from . import speed
    from . import threads as threads_mod

    tm = telemetry if telemetry is not None else telemetry_mod.NullTelemetry()
    crest_run = getattr(cfg, "conformer_search", "") == "crest"

    limit = max(0.0, float(getattr(cfg, "molecule_timeout_s", 0) or 0))
    cap = max_lanes()
    if cap is not None and len(lanes) > cap:
        log(f"Lanes: {len(lanes)} requested, {cap} is the most one process can run on Windows; using the first {cap}")
        lanes = lanes[:cap]
    if speed_aware is None:
        speed_aware = speed_aware_setting()
    policy = speed_policy(lanes, cfg) if speed_aware else None
    if policy is None:
        log(f"Lanes: {describe(lanes)}; molecules go to whichever lane is free")
    else:
        seeded = "benchmark speeds until measured" if policy.seed else "speeds measured as molecules finish"
        log(f"Lanes: {describe(lanes)}; molecules go to whichever lane is free, the last ones to the lanes that "
            f"finish them first (speed-aware, {seeded})")
    if cfg.method in GPU_LINALG_ENGINES and len({x.device == "cpu" for x in lanes}) == 2:
        log(f"Note: {cfg.method} on GPU and CPU lanes: where dxtb stays on the GPU (torch_compat.dxtb_device) it "
            "computes in float32, on the CPU lanes in float64, so one table mixes both precisions. CPU lanes "
            "only (or one GPU lane) keep one precision per run.")
    # each lane gets its share of the machine's memory for the dxtb steps (chem.memory_guard), resolved once here
    cfg = replace(cfg, memory_budget_gb=pipeline_mod.resolved_budget(cfg, len(lanes), log))
    pool = LanePool(lanes, cfg, run_dir, log)
    texts = {mol_id: jobs.input_to_json(inp) for mol_id, inp in todo}
    inputs = dict(todo)
    position = {mol_id: j for j, (mol_id, _) in enumerate(todo)}     # results are returned in todo order
    results: dict[int, MoleculeResult] = {}
    n_done = 0
    stopping = False
    interrupting = False          # a pause "now" (control.interrupt): the stop flag is set until every lane is idle
    paused_logged = False
    t0 = time.perf_counter()
    announced = False
    held_logged = False
    sizes: dict[str, float] = {}

    def size_of(mol_id: str) -> float:
        if mol_id not in sizes:
            sizes[mol_id] = speed.molecule_size(inputs[mol_id])
        return sizes[mol_id]

    def typical_size() -> float:
        return sum(sizes.values()) / len(sizes) if sizes else speed.DEFAULT_SIZE

    # dynamic CREST threads (dist.threads): this process owns the CPU lanes' thread budget
    budget = dynamic_threads_for(cfg, lanes, dynamic_threads, thread_budget)
    model = (cost_model or threads_mod.cost_model_for(cfg, log) or threads_mod.BUILT_IN) if crest_run else None
    dyn = (threads_mod.DynamicThreads(budget, lambda m: model.cost(size_of(m)), model.speedup_power)
           if budget else None)
    if dyn is not None:
        log(f"Dynamic threads: a CREST search may also use idle lanes' threads, in whole lanes, {budget} threads in "
            "all (build --no-dynamic-threads turns it off)")
        if model.source != threads_mod.BUILT_IN.source:
            log(f"  cost model: {model.source}: time ~ atoms^{model.power:.2f}, speedup (threads/lane)^"
                f"{model.speedup_power:.2f} (quantabench-cli timings --fit)")

    def threads_back(w: _LaneProcess) -> None:
        """Every thread lane ``w``'s molecule held goes back to the budget (its molecule ended or was lost)."""
        if dyn is not None:
            dyn.finish(w.lane.name)

    # k: the dispatch position (progress counts in it), largest molecules first
    queue = deque((k, todo[j][0], todo[j][1]) for k, j in enumerate(dispatch_order(todo, size_of,
                                                                                   store.crash_counts())))

    def lane_state(w: _LaneProcess) -> str:
        if w.conn is None or not w.alive:
            return speed.DOWN
        if w.ready:
            return speed.READY
        if (not w.ever_ready and w.startup_failures == 0
                and time.monotonic() - w.started_at < speed.STARTUP_GRACE_S):
            return speed.STARTING           # first start: counted on (a restart is not: it may never come back)
        return speed.DOWN

    def take(n: int, dest: str, want=None) -> list[str]:
        """``Handoff``: up to ``n`` molecules from the end of the queue (none of them started); ``want``: exactly
        those of these ids that are still queued."""
        if want is not None:
            wanted = set(want)
            picked = [item for item in queue if item[1] in wanted]
            for item in picked:
                queue.remove(item)
            picked.reverse()
        else:
            picked = [queue.pop() for _ in range(min(n, len(queue)))]
        moved = set(store.mark_moved([mol_id for _, mol_id, _ in picked], dest))
        queue.extend(item for item in reversed(picked) if item[1] not in moved)   # could not move: kept
        if moved:
            log(f"Work sharing: {len(moved)} molecule(s) not started here go to {dest or 'another machine'}")
        return [mol_id for _, mol_id, _ in picked if mol_id in moved]

    def finish(w: _LaneProcess | None, k: int, mol_id: str, res, result_dir: str = "") -> None:
        nonlocal n_done
        # the attempt's timing rows, in the transaction of its result (a lost lane's rows were written already)
        tm.end_attempt(mol_id, res, stage=w.stage if w is not None else "",
                       stage_since=w.stage_since if w is not None else None)
        store.record(res, result_dir)
        results[position[mol_id]] = res if keep_results else jobs.light_result(res)
        n_done += 1
        if w is not None:
            w.done += 1
        on_result(res)
        progress(done_before + n_done, total, res.status, mol_id)

    def scratch_of(pid: int | None, what: str) -> None:
        """Scratch a killed or dead lane left (Psi4's DF integrals can be 10-20 GB per molecule)."""
        if pid:
            from ..chem import external

            external.sweep_scratch_quietly(owners={pid}, log=log, what=what)

    def lane_died(w: _LaneProcess) -> None:
        pool._close_conn(w)
        code = None
        pid = w.proc.pid if w.proc is not None else None
        if w.proc is not None:
            w.proc.join(2)                         # the pipe can report EOF a moment before the exit
            if w.proc.exitcode is None:
                _kill_tree(w.proc)
                w.proc.join(5)
            code = w.proc.exitcode
            with contextlib.suppress(ValueError):
                w.proc.close()
        w.proc = None
        left = pool.kill_programs(w)        # xtb/CREST/Psi4 runs in its own session and outlives the lane
        if left:
            log(f"[{w.lane.name}] stopped {left} external program(s) the lane left running")
        scratch_of(pid, f"of lane {w.lane.name}")
        threads_back(w)                     # its programs are gone: the threads it held are free again
        if w.busy is not None:
            k, mol_id, inp = w.busy
            w.busy = None
            if policy is not None:
                policy.lost(w.lane.name)
            tm.end_attempt(mol_id, None, outcome="crashed", error=f"lane {w.lane.name} died (exit code {code})",
                           stage=w.stage, stage_since=w.stage_since)
            crashes = store.record_crash(mol_id)
            if crashes >= jobs.MAX_CRASHES:
                msg = f"crashed the process {crashes} times"
                log(f"[{w.lane.name}] lane died (exit code {code}) running {mol_id}; {mol_id} {msg}: marked "
                    "failed (--retry-failed redoes it)")
                finish(None, k, mol_id, MoleculeResult(input=inp, status="failed", error=msg, device=w.device))
            else:
                log(f"[{w.lane.name}] lane died (exit code {code}) running {mol_id}; restarting the lane, "
                    f"{mol_id} is tried again at the end")
                tm.requeued(mol_id)
                queue.append((k, mol_id, inp))
                progress(done_before + k, total, "queued", mol_id)
        elif not w.ready:
            w.startup_failures += 1
            if w.startup_failures >= STARTUP_FAILURES:
                w.retired = True
                log(f"[{w.lane.name}] lane failed to start {w.startup_failures} times (exit code {code}); "
                    "continuing without it" + (f": {pool.last_fatal.strip().splitlines()[-1]}" if pool.last_fatal
                                                else ""))
        else:
            log(f"[{w.lane.name}] lane died while idle (exit code {code}); restarting it")
        if not w.retired and not stopping and queue:
            pool.start(w)

    def lane_timed_out(w: _LaneProcess) -> None:
        """The molecule of lane ``w`` ran past the time limit: kill the lane (and the programs it started), record
        the molecule failed with "time limit (N s)" (not a crash: it is not tried again in this run;
        --retry-failed redoes it) and restart the lane if work is left."""
        k, mol_id, inp = w.busy
        w.busy = None
        if policy is not None:
            policy.lost(w.lane.name)
        err = pipeline_mod.time_limit_error(limit, w.stage)
        elapsed = time.monotonic() - w.busy_since
        pool._close_conn(w)
        pid = w.proc.pid if w.proc is not None else None
        if w.proc is not None:
            _kill_tree(w.proc)
            with contextlib.suppress(ValueError):
                w.proc.close()
        w.proc = None
        left = pool.kill_programs(w)
        scratch_of(pid, f"of lane {w.lane.name} ({mol_id})")
        threads_back(w)
        log(f"[{w.lane.name}] {mol_id}: {err} after {elapsed:.0f} s; lane killed"
            + (f" with {left} external program(s)" if left else "")
            + ("; restarting it" if queue and not stopping else "") + " (--retry-failed redoes the molecule)")
        tm.end_attempt(mol_id, None, outcome="timeout", error=err, stage=w.stage, stage_since=w.stage_since)
        finish(None, k, mol_id, MoleculeResult(input=inp, status="failed", error=err, device=w.device))
        if not w.retired and not stopping and queue:
            pool.start(w)

    def drain(w: _LaneProcess) -> bool:
        """Handle every message waiting on ``w``'s pipe; False once the pipe is closed (the lane ended).
        Only reading the pipe counts as the lane's failure: an exception from ``handle`` (a ``log``,
        ``progress`` or ``on_result`` callback, the job table) propagates and stops the run."""
        while w.conn is not None:
            try:
                if not w.conn.poll():
                    return True
                msg = w.conn.recv()
            except (EOFError, OSError):
                return False
            handle(w, msg)
        return False

    def handle(w: _LaneProcess, msg) -> None:
        nonlocal announced
        kind = msg[1]
        if kind == "log":
            log(f"[{w.lane.name}] {msg[2]}")
        elif kind == "stage":
            _, _, k, mol_id, stage = msg
            if w.busy is not None and w.busy[1] == mol_id:
                if stage != w.stage:
                    w.stage_since = time.time()
                w.stage = stage
                if dyn is not None:
                    if stage == "conformers":
                        dyn.crest_started(w.lane.name)
                    elif stage not in ("start", "prepare"):       # past the conformer search: CREST has ended
                        back, seconds = dyn.crest_done(w.lane.name)
                        if back:
                            log(f"[{w.lane.name}] {mol_id}: conformer search done in {seconds:.0f} s; "
                                f"{back} borrowed thread(s) back")
            progress(done_before + k, total, stage, mol_id)
        elif kind == "result":
            _, _, k, mol_id, res, result_dir = msg
            w.busy = None
            threads_back(w)
            if interrupting and not stopping and res.status == "cancelled":
                # stopped by a pause "now": recorded as not started, and back at the front of the queue
                if policy is not None:
                    policy.lost(w.lane.name)
                tm.end_attempt(mol_id, res, stage=w.stage, stage_since=w.stage_since)
                store.record(res, "")
                queue.appendleft((k, mol_id, inputs[mol_id]))
                progress(done_before + k, total, "queued", mol_id)
                log(f"[{w.lane.name}] {mol_id}: stopped at a checkpoint (paused now); it starts again on resume")
                return
            if policy is not None:
                policy.finished(w.lane.name, ok=res.status == "ok")
            finish(w, k, mol_id, res, result_dir)
        elif kind == "error":
            _, _, k, mol_id, tb = msg
            inp = inputs[mol_id]
            w.busy = None
            threads_back(w)
            if policy is not None:
                policy.finished(w.lane.name, ok=False)
            err = tb.strip().splitlines()[-1]
            log(f"[{w.lane.name}] {mol_id}: lane error: {err}")
            finish(w, k, mol_id, MoleculeResult(input=inp, status="failed", error=f"lane error: {err}",
                                                device=w.device))
        elif kind == "ready":
            _, _, pid, device, _threads = msg
            w.ready, w.pid, w.device, w.startup_failures = True, pid, device, 0
            w.ever_ready = True
            if not announced and all(x.ready or x.retired for x in pool.workers):
                announced = True
                log(f"{sum(x.ready for x in pool.workers)} lanes ready in {time.perf_counter() - t0:.1f} s")
        elif kind == "fatal":
            pool.last_fatal = msg[2]
            log(f"[{w.lane.name}] lane could not start: {msg[2].strip().splitlines()[-1]}")

    try:
        pool.start_all()
        while True:
            if not stopping and should_stop():
                stopping = True
                pool.stop_event.set()
                busy = sum(w.busy is not None for w in pool.workers)
                log(f"Cancelling: {busy} running molecule(s) stop after their current step")
            if handoff is not None and not stopping:
                handoff.serve(take)
            mode = control.poll() if control is not None else "run"
            if control is not None and not stopping:
                busy_now = sum(w.busy is not None for w in pool.workers)
                if control.priority and control.priority != control.applied_priority:
                    from .. import priority as priority_mod

                    level = control.priority
                    note = priority_mod.apply_to_pids(pool.pids(), level, children=True)
                    pool.cfg_dict["priority"] = priority_mod.normalize(level)    # lanes started from now on
                    control.applied_priority = level
                    log(f"Run priority of the lanes: {note}")
                if mode != "run" and control.interrupt and busy_now and not interrupting:
                    interrupting = True
                    pool.stop_event.set()
                    log(f"Pausing now: {busy_now} running molecule(s) stop at their next checkpoint")
                if interrupting and not busy_now:
                    pool.stop_event.clear()
                    interrupting = False
                    control.interrupt = False
                elif mode != "run" and control.interrupt and not busy_now:
                    control.interrupt = False
                if mode != "run" and not paused_logged:
                    paused_logged = True
                    log(f"Paused ({mode}): no new molecule starts; {len(queue)} waiting, {busy_now} running")
                elif mode == "run" and paused_logged:
                    paused_logged = False
                    log("Resumed")
                if mode == "drain" and not busy_now:
                    log(f"Paused: {len(queue)} molecule(s) not started stay in the job table")
                    control.drained = True
                    break
            if not stopping and mode == "run" and not interrupting:
                states, held = None, []
                for w in pool.workers:
                    if queue and w.ready and w.busy is None and w.conn is not None and w.alive:
                        shared = dyn is not None and w.lane.device == "cpu"
                        if shared and not dyn.can_start(w.lane.threads):
                            continue            # its threads are lent to a CREST search: wait until they are back
                        if policy is not None:
                            if states is None:
                                states = {x.lane.name: lane_state(x) for x in pool.workers}
                            if not policy.allow(w.lane.name, (size_of(m) for _, m, _ in queue), states):
                                held.append(w.lane.name)   # faster lanes finish the rest first: ask again later
                                continue
                        k, mol_id, inp = queue.popleft()
                        width = (dyn.width(w.lane.threads, mol_id, (m for _, m, _ in queue)) if shared
                                 else w.lane.threads)
                        try:
                            w.conn.send((k, mol_id, texts[mol_id], width if width > w.lane.threads else 0))
                        except (OSError, ValueError):
                            # the lane died while idle (its end is handled below): the molecule never
                            # reached it, so it goes back to the front of the queue without a crash
                            queue.appendleft((k, mol_id, inp))
                            continue
                        if shared:
                            dyn.start(w.lane.name, w.lane.threads, width, mol_id)
                            if width > w.lane.threads:
                                log(f"[{w.lane.name}] {mol_id}: CREST with {width} threads (borrowed "
                                    f"{width - w.lane.threads} from idle lanes)")
                        w.busy = (k, mol_id, inp)
                        w.busy_since, w.stage, w.stage_since = time.monotonic(), "", time.time()
                        if policy is not None:
                            policy.started(w.lane.name, size_of(mol_id))
                        # its timing row goes out with mark_running's commit (chem.telemetry)
                        tm.dispatched(mol_id, inp, lane=w.lane, width=width if crest_run else None,
                                      size=size_of(mol_id), k=k, pid=w.pid,
                                      predicted_s=dyn.predicted_seconds(mol_id, w.lane.threads, width)
                                      if shared and dyn is not None else None)
                        store.mark_running(mol_id)
                        progress(done_before + k, total, "start", mol_id)
                if held and not held_logged:
                    held_logged = True
                    log(policy.hold_message(held, len(queue), typical_size(), states))
            busy = any(w.busy is not None for w in pool.workers)
            if dyn is not None and not busy and dyn.pool.in_use():
                for name in list(dyn.pool.held):    # only a busy lane holds threads; never leave the others waiting
                    dyn.finish(name)
            if not busy and (stopping or not queue):
                break
            if not stopping and queue and all(w.retired for w in pool.workers):
                raise RuntimeError("no lane could start" + (f": {pool.last_fatal.strip().splitlines()[-1]}"
                                                            if pool.last_fatal else ""))
            for obj in pool.wait(POLL_S):
                w = pool.owner(obj)
                if w is None:
                    continue                      # a sentinel: handled below
                if not drain(w):
                    lane_died(w)
            if limit and not stopping:
                now = time.monotonic()
                for w in pool.workers:
                    if w.busy is not None and w.proc is not None and now - w.busy_since > limit:
                        if not drain(w):          # a result that just arrived wins over the limit
                            lane_died(w)
                        elif w.busy is not None and w.proc is not None:
                            lane_timed_out(w)
            for w in pool.workers:
                if w.proc is not None and not w.alive:
                    drain(w)                      # what the lane sent before it ended, then its death
                    lane_died(w)
                elif w.proc is None and not w.retired and not stopping and queue:
                    pool.start(w)
        pool.close(graceful=True)
    except BaseException:
        held = [(w.busy, w.stage, w.stage_since) for w in pool.workers if w.busy is not None]
        pool.close(graceful=False)
        for (_k, mol_id, inp), stage, since in held:   # interrupted, not crashed: redone next time, no crash counted
            tm.end_attempt(mol_id, None, outcome="cancelled", stage=stage, stage_since=since)
            with contextlib.suppress(Exception):
                store.record(MoleculeResult(input=inp, status="cancelled", error="cancelled"))
        raise
    done = {w.lane.name: w.done for w in pool.workers if w.done}
    if done:
        log("Molecules per lane: " + ", ".join(f"{name} {n}" for name, n in done.items()))
    if policy is not None and policy.done:
        log(policy.summary(typical_size()))
    if dyn is not None:
        log(dyn.summary())
    return [results[k] for k in sorted(results)]


# ------------------------------------------------------------------------------------------- CLI
def register_cli(subparsers) -> None:
    p = subparsers.add_parser("lanes", help="show how a lane spec splits this machine (build --lanes)")
    p.add_argument("spec", nargs="?", help="1, auto, N, or a list like gpu:1x4,cpu:5x2 (default: the lanes setting)")
    p.add_argument("--method", help="engine the plan is for (default from settings)")
    p.add_argument("--threads", type=int, help="CPU threads to plan with (0 = all but the free threads)")
    p.add_argument("--gpu", choices=["auto", "on", "off"])
    from ..cli import add_priority_args

    add_priority_args(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cli)


def _cli(args) -> int:
    import json
    from dataclasses import replace

    from .. import resources
    from .. import settings as settings_mod
    from ..chem.pipeline import BuildConfig

    s = settings_mod.load()
    cfg = BuildConfig.from_settings(s)
    cfg = replace(cfg, **{k: v for k, v in {"method": args.method, "threads": args.threads, "gpu": args.gpu,
                                            "priority": getattr(args, "priority", None),
                                            "free_threads": getattr(args, "free_threads", None)}.items()
                          if v is not None})
    spec = args.spec or s.lanes_for()
    report = resources.detect()
    try:
        lanes = parse_lanes(spec, cfg, report)
    except ValueError as exc:
        import sys

        print(exc, file=sys.stderr)
        return 2
    aware = bool(getattr(s.resources, "speed_aware_lanes", True))
    wide = dynamic_threads_for(cfg, lanes, bool(getattr(s.resources, "dynamic_threads", True)))
    if args.json:
        print(json.dumps({"spec": spec, "lanes": [x.to_dict() for x in lanes], "speed_aware": aware,
                          "dynamic_threads": wide}, indent=2))
    else:
        print(f"{spec}: {plan_text(spec, cfg, report, s.resources.parallel_molecules)}")
        for x in lanes:
            print(f"  {x.name:8s} {x.device:8s} {x.threads} thread(s)")
        if len({(x.device, x.threads) for x in lanes}) > 1:
            print("  end of a batch: " + ("slower lanes leave the last molecules to faster ones (speed-aware; "
                                          "build --no-speed-aware-lanes turns it off)" if aware else
                                          "every free lane takes the next molecule (speed-aware lanes off)"))
        if wide:
            print(f"  CREST: a search started while lanes are idle may use their threads, {wide} in all (dynamic "
                  "threads; build --no-dynamic-threads turns it off)")
    return 0
