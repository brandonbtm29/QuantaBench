"""Hardware detection and resource control (CPU threads, GPU device, memory caps).

Everything that does numerical work asks this module how many threads it may use and which
device to run on, so one setting controls RDKit, NumPy/BLAS, PyTorch and the engines.

GPU support goes through PyTorch. One code path covers:
  * AMD ROCm (``torch.version.hip`` is set; the device is still called ``"cuda"`` in PyTorch),
  * NVIDIA CUDA,
  * Apple Metal (``"mps"``), reserved for the future macOS port.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field

import psutil

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)


@dataclass
class GPUInfo:
    index: int
    name: str
    backend: str          # "rocm" | "cuda" | "mps"
    total_memory_gb: float
    arch: str = ""


@dataclass
class HardwareReport:
    os: str
    python: str
    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_gb: float
    wsl: bool
    torch_version: str = ""
    torch_backend: str = "none"   # "rocm" | "cuda" | "mps" | "cpu" | "none" (torch missing) | "unknown" (not probed)
    gpus: list[GPUInfo] = field(default_factory=list)
    optional_packages: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    performance_cores: int = 0    # Apple Silicon's performance cores (``performance_cores()``); 0 = none known

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        gpu = ", ".join(f"{g.name} ({g.total_memory_gb:.0f} GB, {g.backend.upper()})" for g in self.gpus) or "none usable"
        return (f"{self.cpu_name}: {self.cores_text()} / {self.logical_cores} threads, "
                f"{self.ram_gb:.0f} GB RAM | GPU: {gpu}")

    def cores_text(self) -> str:
        """'14 cores (10 performance)' on Apple Silicon, '12 cores' elsewhere."""
        perf = int(self.performance_cores or 0)
        return f"{self.physical_cores} cores" + (f" ({perf} performance)" if 0 < perf < self.physical_cores else "")


OPTIONAL_PACKAGES = ("torch", "aimnet", "dxtb", "tad_libcint", "dimorphite_dl", "ovito", "openmm", "torch_sim",
                     "lammps", "openpyxl")


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


#: What NVIDIA Warp's native library prints (from C, on every process start) when no NVIDIA driver loads.
WARP_NO_DRIVER_MESSAGE = "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver"
_WARP_INIT: list[bool] = []


def init_warp_quietly() -> bool:
    """Initialize NVIDIA Warp (which aimnet imports) without its "Could not find or load the NVIDIA CUDA
    driver" line on machines with no NVIDIA driver (AMD, Apple, CPU-only): every lane process printed it.

    Warp writes that line from its C library during ``warp.init()``, before ``warp.config`` is consulted, so
    no Warp setting silences it. This captures file descriptor 2 for the duration of ``warp.init()`` and
    passes every other line through, so a real Warp error (e.g. a too-old NVIDIA driver) still shows; the
    line only appears when the driver is missing, so NVIDIA machines see no difference. Later
    ``import aimnet`` finds Warp initialized and prints nothing. Call before importing aimnet or
    nvalchemiops; returns True once Warp is initialized (False without Warp)."""
    if _WARP_INIT:
        return _WARP_INIT[0]
    if not has_package("warp"):
        _WARP_INIT.append(False)
        return False
    import tempfile

    try:
        import warp
    except Exception:  # noqa: BLE001 - reported where aimnet is imported
        _WARP_INIT.append(False)
        return False
    if hasattr(warp, "LOG_WARNING"):           # Warp >= 1.10: no "Warp x.y initialized" banner (as nvalchemiops)
        warp.config.log_level = warp.LOG_WARNING
    warp.config.quiet = True                   # older Warp
    captured = b""
    ok = False
    try:
        sys.stderr.flush()
        saved = os.dup(2)
    except (OSError, ValueError, AttributeError):   # no usable fd 2 (pythonw, some embedders): plain init
        saved = None
    with tempfile.TemporaryFile() as tmp:
        if saved is not None:
            os.dup2(tmp.fileno(), 2)
        try:
            warp.init()
            ok = True
        except Exception:  # noqa: BLE001 - aimnet's own import reports a broken Warp
            ok = False
        finally:
            if saved is not None:
                os.dup2(saved, 2)
                os.close(saved)
                tmp.seek(0)
                captured = tmp.read()
    rest = [ln for ln in captured.decode("utf-8", "replace").splitlines(keepends=True)
            if not ln.startswith(WARP_NO_DRIVER_MESSAGE)]
    if rest:
        sys.stderr.write("".join(rest))
        sys.stderr.flush()
    _WARP_INIT.append(ok)
    return ok


def is_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def cpu_name() -> str:
    name = platform.processor() or ""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return name or platform.machine()


def logical_cores() -> int:
    return psutil.cpu_count(logical=True) or os.cpu_count() or 1


def physical_cores() -> int:
    return psutil.cpu_count(logical=False) or max(1, logical_cores() // 2)


def _os_arch() -> tuple[str, str]:
    """(sys.platform, platform.machine()); a function so tests can fake another machine."""
    return sys.platform, platform.machine().lower()


def _sysctl_int(name: str) -> int | None:
    """An integer sysctl value on macOS (``sysctl -n NAME``), None when it cannot be read."""
    import shutil
    import subprocess

    exe = shutil.which("sysctl") or "/usr/sbin/sysctl"
    try:
        out = subprocess.run([exe, "-n", name], capture_output=True, text=True, timeout=5, check=False)  # noqa: S603
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


_perf_cores: int | None = None     # performance_cores(), detected once


def performance_cores() -> int:
    """Performance cores of a CPU that has performance and efficiency cores; 0 where none are known.

    Apple Silicon: ``hw.perflevel0.physicalcpu`` (the M3 Max: 10, next to ``hw.perflevel1.physicalcpu`` = 4
    efficiency cores), or the physical cores when that cannot be read. Intel Macs have no perflevel sysctls.
    Elsewhere 0: plans then use the physical cores. Only the CREST lane plan uses it (``dist.lanes.plan_auto``);
    hybrid Intel CPUs (P and E cores) are not told apart yet."""
    global _perf_cores
    if _perf_cores is not None:
        return _perf_cores
    system, machine = _os_arch()
    n = 0
    if system == "darwin":
        n = _sysctl_int("hw.perflevel0.physicalcpu") or 0     # works under Rosetta too
        if n <= 0 and machine in ("arm64", "aarch64"):
            n = physical_cores()
    _perf_cores = max(0, int(n))
    return _perf_cores


def resolve_threads(requested: int) -> int:
    """0 means 'all logical cores but one', so the desktop stays responsive."""
    total = logical_cores()
    if requested <= 0:
        return max(1, total - 1)
    return max(1, min(requested, total))


_applied_threads: int | None = None


def apply_thread_limits(threads: int) -> int:
    """Set thread caps for BLAS/OpenMP (env vars, effective for libraries loaded afterwards) and
    for PyTorch if it is already imported. Torch is not imported here (a ROCm/CUDA torch takes seconds
    to load, and a lane running MMFF or xtb never needs it); one imported later reads OMP_NUM_THREADS.
    Returns the thread count in force."""
    global _applied_threads
    n = resolve_threads(threads)
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits  # optional, comes with scikit-learn/scipy stacks

        threadpool_limits(n)
    except Exception:  # noqa: S110 - optional
        pass
    if "torch" in sys.modules:
        try:
            import torch

            torch.set_num_threads(n)
        except Exception:  # noqa: S110
            pass
    _applied_threads = n
    return n


@dataclass
class MoleculeLimits:
    """The per-molecule limits of a run (``BuildConfig.memory_budget_gb``/``molecule_timeout_s``) for the steps
    that are not handed the config: the conformer embedding (``chem.conformers``) and the frequencies' Hessian
    (``chem.vibrations`` through ``chem.memory_guard``). None = not set by a run: those steps use their own
    defaults (memory: auto for one process; embedding: no time limit)."""
    memory_budget_gb: float | None = None
    molecule_timeout_s: float | None = None


_molecule_limits = MoleculeLimits()


def set_molecule_limits(cfg=None, *, memory_budget_gb: float | None = None,
                        molecule_timeout_s: float | None = None) -> MoleculeLimits:
    """Record this process's per-molecule limits, from ``cfg`` (a ``BuildConfig``) or the keywords; like
    ``apply_thread_limits``, set by whatever runs molecules (a lane, the CLI build, the GUI, a grid worker)
    before the first one. ``set_molecule_limits()`` clears them."""
    global _molecule_limits
    if cfg is not None:
        memory_budget_gb = getattr(cfg, "memory_budget_gb", memory_budget_gb)
        molecule_timeout_s = getattr(cfg, "molecule_timeout_s", molecule_timeout_s)
    _molecule_limits = MoleculeLimits(None if memory_budget_gb is None else float(memory_budget_gb),
                                      None if molecule_timeout_s is None else float(molecule_timeout_s))
    return _molecule_limits


def molecule_limits() -> MoleculeLimits:
    return _molecule_limits


HARD_EXIT_ENV = "QUANTABENCH_HARD_EXIT"     # "1" always, "0" never, unset: only where needed (below)


def needs_hard_exit() -> bool:
    """Whether this process should leave with ``os._exit`` instead of a normal interpreter shutdown.

    AMD's Windows ROCm runtime (torch 2.9.1+rocm7.2.1 on BTM NOVA) can hang forever in its exit-time
    teardown once a process has used the GPU: ``main()`` returns, output is complete, and the process
    then spins one core until it is killed. So when this process initialized a ROCm device on Windows,
    entry points flush and exit at once. Torch is never imported here."""
    env = os.environ.get(HARD_EXIT_ENV, "").strip()
    if env in ("0", "1"):
        return env == "1"
    if sys.platform != "win32":
        return False
    torch = sys.modules.get("torch")
    if torch is None or not getattr(getattr(torch, "version", None), "hip", None):
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return False


def hard_exit_if_needed(code: int) -> None:
    """``os._exit(code)`` after flushing output and logging, when ``needs_hard_exit()``; else return.
    Call it only at the very end of a process's entry point, after every file and database is closed."""
    if not needs_hard_exit():
        return
    import logging

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: S110 - a closed console must not stop the exit
            pass
    logging.shutdown()
    os._exit(int(code or 0))


def current_threads() -> int:
    return _applied_threads or resolve_threads(0)


# ------------------------------------------------------------------------ GPU
def torch_gpus() -> tuple[str, str, list[GPUInfo]]:
    """(torch version, backend, gpus). Never raises."""
    if not has_package("torch"):
        return "", "none", []
    try:
        import torch
    except Exception as exc:  # broken install
        return f"import failed: {exc}", "none", []
    version = torch.__version__
    gpus: list[GPUInfo] = []
    try:
        if torch.cuda.is_available():
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                arch = getattr(props, "gcnArchName", "") or f"sm_{props.major}{props.minor}"
                gpus.append(GPUInfo(i, props.name, backend, props.total_memory / 1024**3, arch))
            return version, backend, gpus
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return version, "mps", [GPUInfo(0, "Apple GPU", "mps", psutil.virtual_memory().total / 1024**3)]
    except Exception:  # noqa: S110 - driver errors mean "no usable GPU"
        pass
    return version, "cpu", []


def detect() -> HardwareReport:
    version, backend, gpus = torch_gpus()
    rep = HardwareReport(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu_name=cpu_name(),
        physical_cores=physical_cores(),
        logical_cores=logical_cores(),
        ram_gb=psutil.virtual_memory().total / 1024**3,
        wsl=is_wsl(),
        torch_version=version,
        torch_backend=backend,
        gpus=gpus,
        optional_packages={p: has_package(p) for p in OPTIONAL_PACKAGES},
        performance_cores=performance_cores(),
    )
    if backend == "none":
        rep.notes.append("PyTorch is not installed: ML potentials and xTB are unavailable; MMFF still works. "
                         "See docs/INSTALL.md for the build that matches your GPU.")
    elif backend == "cpu":
        if "+cpu" in version or (platform.system() == "Windows" and "rocm" not in version):
            rep.notes.append(f"PyTorch {version} is a CPU-only build (the generic one pip pulls in from PyPI). "
                             "For an AMD Radeon on Windows, install AMD's ROCm wheels over it; see docs/INSTALL.md.")
        else:
            rep.notes.append("PyTorch found no GPU. Check the GPU driver; on WSL2, ROCm needs AMD's WSL driver "
                             "(see docs/INSTALL.md).")
    if rep.wsl and not gpus:
        rep.notes.append("Running under WSL2 without GPU pass-through; native Windows may be faster for GPU work.")
    return rep


def detect_light(note: str = "") -> HardwareReport:
    """``detect()`` without importing torch: CPU, RAM and the installed torch version, no GPU list."""
    from importlib import metadata

    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        version = ""
    rep = HardwareReport(
        os=f"{platform.system()} {platform.release()}", python=platform.python_version(), cpu_name=cpu_name(),
        physical_cores=physical_cores(), logical_cores=logical_cores(), ram_gb=psutil.virtual_memory().total / 1024**3,
        wsl=is_wsl(), torch_version=version, torch_backend="unknown" if version else "none",
        optional_packages={p: has_package(p) for p in OPTIONAL_PACKAGES}, performance_cores=performance_cores())
    if note:
        rep.notes.append(note)
    return rep


_PROBE_MARK = "QB-PROBE-JSON "
_ISOLATED: tuple[HardwareReport, dict] | None = None


def _probe_child() -> None:
    """Child side of ``detect_isolated``: one JSON line on stdout, then ``os._exit`` (no torch teardown, which
    can hang with AMD's Windows ROCm runtime, FORK-11)."""
    import json

    out: dict = {"hardware": None, "checks": {}}
    try:
        out["hardware"] = detect().to_dict()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from .chem import engines

        out["checks"]["libcint"] = list(engines.libcint_status())
    except Exception as exc:
        out["checks"]["libcint"] = [False, f"{type(exc).__name__}: {exc}"]
    sys.stdout.write(_PROBE_MARK + json.dumps(out) + "\n")
    sys.stdout.flush()
    os._exit(0)


def detect_isolated(timeout: float = 180.0, cancel=None) -> tuple[HardwareReport, dict] | None:
    """``detect()`` and the checks that import torch (``{"libcint": (ok, reason)}``), run in a child process
    with this interpreter, so this process never imports torch (the GUI: ~500 MB and seconds of start-up for
    a status-bar label) and a broken GPU stack cannot crash it. Cached once it succeeds. ``cancel``: a
    ``threading.Event``; once set, the child is killed and None returned. If the child fails, the report is
    ``detect_light`` with a note saying why, and no checks."""
    global _ISOLATED
    if _ISOLATED is not None:
        return _ISOLATED
    if cancel is not None and cancel.is_set():
        # cancelled before it started: without torch the child can finish inside the first 0.25 s wait, before
        # the loop below looks at ``cancel`` (CI's core install: the result came back instead of None)
        return None
    import json
    import subprocess
    from pathlib import Path

    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        exe = exe.with_name("python.exe")        # pythonw has no stdout to read the result from
    root = str(Path(__file__).resolve().parent.parent)
    code = f"import sys; sys.path.insert(0, {root!r}); from quantabench.resources import _probe_child; _probe_child()"
    kwargs: dict = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "stdin": subprocess.DEVNULL,
                    "text": True, "errors": "replace"}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    why = ""
    try:
        proc = subprocess.Popen([str(exe), "-c", code], **kwargs)  # noqa: S603
    except OSError as exc:
        return detect_light(f"GPU detection could not start a Python process ({exc}); GPU list unknown."), {}
    t0 = time.monotonic()
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=0.25)
            if cancel is not None and cancel.is_set():
                return None          # cancelled while the child ran: its result is not wanted
            break
        except subprocess.TimeoutExpired:
            if (cancel is not None and cancel.is_set()) or time.monotonic() - t0 > timeout:
                proc.kill()
                proc.communicate()
                if cancel is not None and cancel.is_set():
                    return None
                return detect_light(f"GPU detection timed out after {timeout:.0f} s; GPU list unknown."), {}
    for line in reversed((stdout or "").splitlines()):
        if line.startswith(_PROBE_MARK):
            try:
                out = json.loads(line[len(_PROBE_MARK):])
                d = out["hardware"]
                if d is None:
                    why = out.get("error", "no report")
                    break
                rep = HardwareReport(**{**d, "gpus": [GPUInfo(**g) for g in d.get("gpus", [])]})
                _ISOLATED = (rep, dict(out.get("checks") or {}))
                return _ISOLATED
            except (ValueError, KeyError, TypeError) as exc:
                why = f"unreadable result ({exc})"
                break
    else:
        tail = (stderr or "").strip().splitlines()
        why = f"exit code {proc.returncode}" + (f": {tail[-1]}" if tail else "")
    return detect_light(f"GPU detection failed ({why}); GPU list unknown. Run `quantabench-cli doctor`."), {}


def select_device(policy: str = "auto", index: int = 0, memory_fraction: float | None = None) -> str:
    """Return a torch device string for the policy: "auto" (GPU if usable), "on" (GPU, error if none),
    "off" (CPU)."""
    if policy == "off":
        return "cpu"
    _, backend, gpus = torch_gpus()
    if not gpus:
        if policy == "on":
            raise RuntimeError("GPU requested but PyTorch sees no usable GPU (run `quantabench-cli doctor`).")
        return "cpu"
    if backend == "mps":
        return "mps"
    index = index if 0 <= index < len(gpus) else 0
    if memory_fraction:
        try:
            import torch

            torch.cuda.set_per_process_memory_fraction(float(memory_fraction), index)
        except Exception:  # noqa: S110
            pass
    return f"cuda:{index}"


def gpu_memory_used_gb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        return torch.cuda.memory_allocated(torch.device(device)) / 1024**3
    except Exception:
        return None


def matmul_benchmark(device: str, size: int = 2048, repeats: int = 5) -> float:
    """Rough FP32 GFLOP/s, to confirm a device actually computes (and how fast)."""
    import torch

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    (a @ b).sum().item()  # warm-up
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    c.sum().item()
    dt = time.perf_counter() - t
    return 2 * size**3 * repeats / dt / 1e9


# ------------------------------------------------------------------ core health
CORE_MODULES = (
    ("RDKit", "rdkit.Chem.AllChem"),
    ("RDKit SASA", "rdkit.Chem.rdFreeSASA"),
    ("ASE", "ase.optimize"),
    ("Qt (PySide6)", "PySide6.QtWidgets"),
    ("VTK rendering", "vtkmodules.vtkRenderingOpenGL2"),
)
OPTIONAL_MODULES = (
    ("PyTorch", "torch"),
    ("dxtb", "dxtb"),
    ("AIMNet2", "aimnet.calculators"),
)


def explain_import_error(message: str) -> str:
    if "Application Control policy" in message or "WinError 4551" in message:
        return ("Windows Smart App Control (or an App Control policy) blocked one of this library's DLLs. "
                "See docs/INSTALL.md, 'Smart App Control'.")
    if "DLL load failed" in message:
        return ("A DLL failed to load. Install the latest Microsoft Visual C++ Redistributable (x64) and "
                "reinstall the package.")
    return ""


def import_checks() -> list[tuple[str, str, bool, str]]:
    """Import each core and installed optional module. Returns (label, module, ok, message)."""
    import importlib

    out = []
    for label, mod in CORE_MODULES + OPTIONAL_MODULES:
        top = mod.split(".")[0]
        if (label, mod) in OPTIONAL_MODULES and not has_package(top):
            continue
        try:
            if top == "aimnet":
                init_warp_quietly()     # no "Could not find or load the NVIDIA CUDA driver" without NVIDIA
            importlib.import_module(mod)
            out.append((label, mod, True, ""))
        except Exception as exc:  # ImportError, OSError from blocked DLLs, ...
            out.append((label, mod, False, f"{type(exc).__name__}: {exc}"))
    return out
