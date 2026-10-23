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
    torch_backend: str = "none"   # "rocm" | "cuda" | "mps" | "cpu" | "none" (torch missing)
    gpus: list[GPUInfo] = field(default_factory=list)
    optional_packages: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        gpu = ", ".join(f"{g.name} ({g.total_memory_gb:.0f} GB, {g.backend.upper()})" for g in self.gpus) or "none usable"
        return (f"{self.cpu_name}: {self.physical_cores} cores / {self.logical_cores} threads, "
                f"{self.ram_gb:.0f} GB RAM | GPU: {gpu}")


OPTIONAL_PACKAGES = ("torch", "aimnet", "dxtb", "tad_libcint", "dimorphite_dl", "ovito", "openmm", "torch_sim")


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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


def resolve_threads(requested: int) -> int:
    """0 means 'all logical cores but one', so the desktop stays responsive."""
    total = logical_cores()
    if requested <= 0:
        return max(1, total - 1)
    return max(1, min(requested, total))


_applied_threads: int | None = None


def apply_thread_limits(threads: int) -> int:
    """Set thread caps for BLAS/OpenMP (env vars, effective for libraries loaded afterwards) and
    for PyTorch if it is already importable. Returns the thread count in force."""
    global _applied_threads
    n = resolve_threads(threads)
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits  # optional, comes with scikit-learn/scipy stacks

        threadpool_limits(n)
    except Exception:  # noqa: S110 - optional
        pass
    if has_package("torch"):
        try:
            import torch

            torch.set_num_threads(n)
        except Exception:  # noqa: S110
            pass
    _applied_threads = n
    return n


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
            importlib.import_module(mod)
            out.append((label, mod, True, ""))
        except Exception as exc:  # ImportError, OSError from blocked DLLs, ...
            out.append((label, mod, False, f"{type(exc).__name__}: {exc}"))
    return out
