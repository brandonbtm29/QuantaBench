"""Lanes: worker processes that share one machine's CPU cores and GPUs.

One small molecule cannot keep 12 cores busy: RDKit's embedding, dxtb's small matrices, xtb, CREST and Psi4
all scale poorly past a few threads for a PFAS-sized molecule. A batch therefore runs faster as several
**lanes**, each a worker process with its own device and thread budget, each working on its own molecule.

    Lane(name, device, threads)            one worker process; device "cpu" | "cuda:N" | "mps" | "auto"
                                           ("auto": the device the GPU setting selects; only the single
                                           lane of spec "1" uses it)
    parse_lanes(spec, cfg, report=None) -> list[Lane]
        "1"                  one lane, all threads: today's single-process loop (the default)
        "N"                  N CPU lanes with resolve_threads(cfg.threads) // N threads each (at least 1)
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
* molecules are handed out pull-style: a lane gets the next molecule when it reports the last one done,
  so a fast lane (the GPU) simply takes more of them. Each lane runs ``pipeline.run_molecule`` and writes
  that molecule's folder (``export.write_molecule`` into run_dir/<id>), then sends the MoleculeResult back
  (RDKit Mol with its conformers and properties) over its pipe, with its log lines (prefixed "[lane]") and
  stage updates. The parent is the only process that writes jobs.sqlite (mark_running, record) and the
  only one that calls ``progress``/``log``/``on_result``, all in the calling thread.
* ``should_stop`` is polled by the parent and copied into a shared stop flag (``StopEvent``: Event semantics,
  no lock) that every lane's ``should_stop`` reads, so a cancel reaches the molecule running in each lane.
* a lane that dies (segfault, OOM kill, ``os._exit``) is restarted. Its molecule's crash count goes up,
  exactly like a crash of the single-process loop (``JobStore.reset_interrupted``): below ``MAX_CRASHES``
  it is tried again at the end of the queue, at ``MAX_CRASHES`` it is recorded as failed ("crashed the
  process N times") and the run goes on.
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
* in-process torch engines on the CPU (dxtb GFN1/GFN2, AIMNet2): 2-thread lanes on the physical cores (as
  fast as 1-thread lanes on all cores, with half the processes: each holds torch, ~0.5-1 GB).
* external programs (an engine with ``.external``, CREST conformer search, a QM stage): 2-thread lanes on
  all threads (the programs spend part of their time in I/O and start-up).
* a GPU-capable engine with a GPU: one GPU lane with 4 threads (RDKit embedding and the host side of the
  batch run on the CPU) plus CPU lanes on the remaining threads by the rules above. Not for dxtb when torch
  has no GPU linear algebra (``gpu_linalg_available``): on NOVA's ROCm wheels (no MAGMA) dxtb's cholesky
  fails on cuda:0, the GPU lane's molecules fall back to the CPU, and 1 GPU + 4 x 2 CPU lanes took 77.4 s
  for the 12 PFAS against 49.9 s for 6 x 2 CPU lanes. ``auto`` gives 6 x 2 CPU lanes there.
* lanes are capped by memory (0.75 x RAM over a per-lane working set; Psi4 adds its memory per job) and
  ``settings.resources.parallel_molecules`` (> 0) fixes the number of lanes. The plan never uses more
  threads than ``resolve_threads(cfg.threads)``.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass

#: Test hook: a lane whose molecule id is listed here (comma separated) calls ``os._exit(TEST_CRASH_EXIT)``,
#: standing in for a segfault or OOM kill. Only tests set it.
TEST_CRASH_ENV = "QUANTABENCH_TEST_LANE_CRASH"
TEST_CRASH_EXIT = 86
#: Elements method "auto" is resolved for when a lane plan is made (H C N O F S: PFAS and most organics)
TYPICAL_ELEMENTS = (1, 6, 7, 8, 9, 16)
#: Working set of one lane process in GB, by workload, for the memory cap of the auto plan
LANE_MEMORY_GB = {"mmff": 0.4, "torch": 1.2, "external": 0.6}
GPU_LANE_THREADS = 4
STARTUP_FAILURES = 3          # a lane that dies this often before it is ready is given up
SHUTDOWN_GRACE_S = 5.0        # after stop: time for lanes to stop their programs before they are killed
POLL_S = 0.1


@dataclass(frozen=True)
class Lane:
    name: str            # "gpu0", "cpu1", ...
    device: str = "cpu"  # torch device string
    threads: int = 1     # CPU threads for this lane (BLAS, OpenMP, RDKit, external programs)

    def to_dict(self) -> dict:
        return {"name": self.name, "device": self.device, "threads": self.threads}


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


def cpu_report() -> _CPUReport:
    import psutil

    from .. import resources

    return _CPUReport(resources.physical_cores(), resources.logical_cores(),
                      psutil.virtual_memory().total / 1024**3, [], "unknown")


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
    """(engine, kind, gpu_capable) of a build: kind "mmff" | "torch" (in-process) | "external" (xtb, MOPAC,
    CREST or Psi4 run as programs). Method "auto" is resolved for typical organic elements."""
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
    else:
        kind = "torch"
    return method, kind, bool(info.gpu_capable and method != "mmff")


def _qm_memory_gb() -> float:
    from .. import settings as settings_mod

    try:
        return float(settings_mod.load().engines.qm_memory_gb)
    except Exception:
        return 4.0


def _parallel_molecules_setting() -> int:
    from .. import settings as settings_mod

    try:
        return int(settings_mod.load().resources.parallel_molecules)
    except Exception:
        return 0


def plan_auto(cfg=None, report=None, parallel_molecules: int = 0) -> tuple[list[Lane], str]:
    """The lanes of spec "auto" and a one-line reason. ``report``: a ``resources.HardwareReport`` (or
    ``cpu_report()``, which knows no GPU); None detects the GPU through PyTorch only when the engine can use
    one. See the module docstring for the rules and the measurements behind them."""
    from .. import resources

    cfg = _cfg(cfg)
    total = resources.resolve_threads(cfg.threads)
    hw = report if report is not None else cpu_report()
    cores = max(1, min(int(hw.physical_cores or total), total))
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
    if device.startswith("cuda") and method in GPU_LINALG_ENGINES and not gpu_linalg_available():
        device, why_gpu = "cpu", "no GPU lane: this torch build has no GPU linear algebra (MAGMA), which dxtb needs"
    per_lane_gb = LANE_MEMORY_GB[kind] + (_qm_memory_gb() if cfg.qm_method else 0.0)
    by_memory = max(1, int(float(hw.ram_gb or 0) * 0.75 / per_lane_gb)) if hw.ram_gb else total

    lanes: list[Lane] = []
    if parallel_molecules > 0:
        n = max(1, min(int(parallel_molecules), total))
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
        elif kind == "external":
            per, wanted = 2, budget // 2
            reason = "external programs: 2-thread lanes on all threads"
        else:
            per, wanted = 2, min(cpu_cores, budget) // 2
            reason = f"{method} on the CPU: 2-thread lanes on the physical cores"
        n = max(0, min(wanted, by_memory - len(lanes)))
        if n < wanted:
            reason += f", capped by memory ({hw.ram_gb:.0f} GB)"
        if not lanes and n <= 1:
            return [Lane("main", "auto", total)], f"{reason}: room for one lane only"
        lanes += [Lane(f"cpu{i + 1}", "cpu", per) for i in range(n)]
        if lanes[0].device != "cpu":
            reason = f"{method} on {device}: 1 GPU lane with {lanes[0].threads} threads; " + reason
    if why_gpu:
        reason += f" ({why_gpu})"
    if len(lanes) == 1:
        return [Lane("main", "auto", total)], reason
    return lanes, reason


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
    # GPU lanes first: a run with fewer molecules than lanes uses the first lanes of the list
    return sorted(lanes, key=lambda x: x.device == "cpu")


def parse_lanes(spec: str | int | None, cfg=None, report=None) -> list[Lane]:
    """The lanes for ``spec`` (see the module docstring). Raises ValueError with a readable message for a spec
    that cannot be used here (bad syntax, a GPU lane without a GPU, more threads than logical cores)."""
    from .. import resources

    cfg = _cfg(cfg)
    total = resources.resolve_threads(cfg.threads)
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
    from .. import resources

    cfg = _cfg(cfg)
    hw = report if report is not None else cpu_report()
    total = resources.resolve_threads(cfg.threads)
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


def _lane_main(lane: Lane, cfg_dict: dict, run_dir: str, conn, stop_event, memory_fraction: float) -> None:
    """Body of one lane process. Module level, so the spawn start method can import it on every OS."""
    import signal

    with contextlib.suppress(ValueError, OSError, AttributeError):
        signal.signal(signal.SIGINT, signal.SIG_IGN)    # Ctrl+C is the parent's to handle

    def send(*msg) -> None:
        conn.send((lane.name, *msg))

    try:
        from .. import resources

        threads = resources.apply_thread_limits(lane.threads)     # before the pipeline (and torch) loads
        from pathlib import Path

        from rdkit import Chem

        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        from ..chem import export, jobs, pipeline

        cfg = pipeline.BuildConfig(**{**cfg_dict, "threads": threads})
        device = lane_device(lane, cfg, memory_fraction)
        crash_ids = {x for x in os.environ.get(TEST_CRASH_ENV, "").split(",") if x}
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
        k, mol_id, text = task
        try:
            inp = jobs.input_from_json(text)
            if mol_id in crash_ids:
                os._exit(TEST_CRASH_EXIT)
            res = pipeline.run_molecule(inp, cfg, device, threads, stop_event.is_set,
                                        log=lambda m: send("log", m),
                                        stage=lambda s, k=k, mol_id=mol_id: send("stage", k, mol_id, s))
            result_dir = ""
            if cfg.write_files and res.status == "ok":
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

    # -- processes
    def start(self, w: _LaneProcess) -> None:
        parent_conn, child_conn = self.ctx.Pipe(duplex=True)
        proc = self.ctx.Process(target=_lane_main, name=f"quantabench-lane-{w.lane.name}", daemon=True,
                                args=(w.lane, self.cfg_dict, self.run_dir, child_conn, self.stop_event,
                                      self.fractions.get(w.lane.device, 0.0)))
        with _child_env(w.lane.threads):
            proc.start()
        child_conn.close()
        w.proc, w.conn, w.ready, w.busy, w.pid = proc, parent_conn, False, None, proc.pid

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

    @staticmethod
    def _close_conn(w: _LaneProcess) -> None:
        if w.conn is not None:
            with contextlib.suppress(OSError):
                w.conn.close()
            w.conn = None


def run_parallel(store, todo, cfg, lanes: list[Lane], run_dir, *, total: int, done_before: int,
                 progress, should_stop, log, on_result) -> list:
    """Run ``todo`` ([(id, MoleculeInput)], from ``JobStore.todo``) in ``lanes`` worker processes. The
    calling thread is the only one that writes ``store`` and calls the callbacks. Returns the results in
    ``todo`` order (molecules not started because of a cancel are absent)."""
    from ..chem import jobs
    from ..chem.pipeline import MoleculeResult

    log(f"Lanes: {describe(lanes)}; molecules go to whichever lane is free")
    pool = LanePool(lanes, cfg, run_dir, log)
    queue = deque((k, mol_id, inp) for k, (mol_id, inp) in enumerate(todo))
    texts = {mol_id: jobs.input_to_json(inp) for mol_id, inp in todo}
    inputs = dict(todo)
    results: dict[int, MoleculeResult] = {}
    n_done = 0
    stopping = False
    t0 = time.perf_counter()
    announced = False

    def finish(w: _LaneProcess | None, k: int, mol_id: str, res, result_dir: str = "") -> None:
        nonlocal n_done
        store.record(res, result_dir)
        results[k] = res
        n_done += 1
        if w is not None:
            w.done += 1
        on_result(res)
        progress(done_before + n_done, total, res.status, mol_id)

    def lane_died(w: _LaneProcess) -> None:
        pool._close_conn(w)
        code = None
        if w.proc is not None:
            w.proc.join(2)                         # the pipe can report EOF a moment before the exit
            if w.proc.exitcode is None:
                _kill_tree(w.proc)
                w.proc.join(5)
            code = w.proc.exitcode
            with contextlib.suppress(ValueError):
                w.proc.close()
        w.proc = None
        if w.busy is not None:
            k, mol_id, inp = w.busy
            w.busy = None
            crashes = store.record_crash(mol_id)
            if crashes >= jobs.MAX_CRASHES:
                msg = f"crashed the process {crashes} times"
                log(f"[{w.lane.name}] lane died (exit code {code}) running {mol_id}; {mol_id} {msg}: marked "
                    "failed (--retry-failed redoes it)")
                finish(None, k, mol_id, MoleculeResult(input=inp, status="failed", error=msg, device=w.device))
            else:
                log(f"[{w.lane.name}] lane died (exit code {code}) running {mol_id}; restarting the lane, "
                    f"{mol_id} is tried again at the end")
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

    def handle(w: _LaneProcess, msg) -> None:
        nonlocal announced
        kind = msg[1]
        if kind == "log":
            log(f"[{w.lane.name}] {msg[2]}")
        elif kind == "stage":
            _, _, k, mol_id, stage = msg
            progress(done_before + k, total, stage, mol_id)
        elif kind == "result":
            _, _, k, mol_id, res, result_dir = msg
            w.busy = None
            finish(w, k, mol_id, res, result_dir)
        elif kind == "error":
            _, _, k, mol_id, tb = msg
            inp = inputs[mol_id]
            w.busy = None
            err = tb.strip().splitlines()[-1]
            log(f"[{w.lane.name}] {mol_id}: lane error: {err}")
            finish(w, k, mol_id, MoleculeResult(input=inp, status="failed", error=f"lane error: {err}",
                                                device=w.device))
        elif kind == "ready":
            _, _, pid, device, _threads = msg
            w.ready, w.pid, w.device, w.startup_failures = True, pid, device, 0
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
            if not stopping:
                for w in pool.workers:
                    if queue and w.ready and w.busy is None and w.conn is not None and w.alive:
                        k, mol_id, inp = queue.popleft()
                        store.mark_running(mol_id)
                        w.busy = (k, mol_id, inp)
                        progress(done_before + k, total, "start", mol_id)
                        try:
                            w.conn.send((k, mol_id, texts[mol_id]))
                        except (OSError, ValueError):
                            pass                 # the lane died: its sentinel/EOF is handled below
            busy = any(w.busy is not None for w in pool.workers)
            if not busy and (stopping or not queue):
                break
            if not stopping and queue and all(w.retired for w in pool.workers):
                raise RuntimeError("no lane could start" + (f": {pool.last_fatal.strip().splitlines()[-1]}"
                                                            if pool.last_fatal else ""))
            for obj in pool.wait(POLL_S):
                w = pool.owner(obj)
                if w is None:
                    continue                      # a sentinel: handled below
                try:
                    while w.conn is not None and w.conn.poll():
                        handle(w, w.conn.recv())
                except (EOFError, OSError):
                    lane_died(w)
            for w in pool.workers:
                if w.proc is not None and not w.alive:
                    # read what the lane sent before it ended, then deal with its death
                    try:
                        while w.conn is not None and w.conn.poll():
                            handle(w, w.conn.recv())
                    except (EOFError, OSError):
                        pass
                    lane_died(w)
                elif w.proc is None and not w.retired and not stopping and queue:
                    pool.start(w)
        pool.close(graceful=True)
    except BaseException:
        held = [w.busy for w in pool.workers if w.busy is not None]
        pool.close(graceful=False)
        for _k, _mol_id, inp in held:         # interrupted, not crashed: redone next time, no crash counted
            with contextlib.suppress(Exception):
                store.record(MoleculeResult(input=inp, status="cancelled", error="cancelled"))
        raise
    done = {w.lane.name: w.done for w in pool.workers if w.done}
    if done:
        log("Molecules per lane: " + ", ".join(f"{name} {n}" for name, n in done.items()))
    return [results[k] for k in sorted(results)]


# ------------------------------------------------------------------------------------------- CLI
def register_cli(subparsers) -> None:
    p = subparsers.add_parser("lanes", help="show how a lane spec splits this machine (build --lanes)")
    p.add_argument("spec", nargs="?", help="1, auto, N, or a list like gpu:1x4,cpu:5x2 (default: the lanes setting)")
    p.add_argument("--method", help="engine the plan is for (default from settings)")
    p.add_argument("--threads", type=int, help="CPU threads to plan with (0 = all but one)")
    p.add_argument("--gpu", choices=["auto", "on", "off"])
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
    cfg = replace(cfg, **{k: v for k, v in {"method": args.method, "threads": args.threads, "gpu": args.gpu}.items()
                          if v is not None})
    spec = args.spec or s.dist.lanes
    report = resources.detect()
    try:
        lanes = parse_lanes(spec, cfg, report)
    except ValueError as exc:
        import sys

        print(exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"spec": spec, "lanes": [x.to_dict() for x in lanes]}, indent=2))
    else:
        print(f"{spec}: {plan_text(spec, cfg, report, s.resources.parallel_molecules)}")
        for x in lanes:
            print(f"  {x.name:8s} {x.device:8s} {x.threads} thread(s)")
    return 0
