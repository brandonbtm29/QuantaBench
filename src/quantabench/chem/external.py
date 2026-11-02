"""External quantum chemistry programs (xtb, CREST, MOPAC, Psi4) run as subprocesses.

Why subprocesses: these programs are Fortran/C++ codes with their own OpenMP and BLAS runtimes. Run
in-process they fight PyTorch and NumPy over threads and libraries, and a crash (segfault, stack
overflow) would take the GUI down with it. As subprocesses they are isolated, they can live in their
own conda environment, they can be cancelled by killing the process tree, and on Windows a Linux-only
program (CREST) can run inside WSL.

This module is the shared contract every external engine uses:

``find_program(name)``      where is it? (settings path, ``QUANTABENCH_<NAME>`` env var, PATH, the
                            running environment, common conda/mamba environments; ``wsl:<cmd>`` paths)
``find_python_with(mod)``   a Python interpreter that can import ``mod`` (Psi4 lives in its own env)
``program_env(threads)``    OpenMP/BLAS thread caps, ``OMP_STACKSIZE``, ``PSI_SCRATCH`` for the child
``work_dir(prefix)``        a scratch job folder, removed on success unless ``keep_scratch`` is set
``run_program(cmd, cwd)``   run, stream output to a log file, honour cancellation and timeouts,
                            raise ``ExternalProgramError`` with the log tail on failure

Engines pass **file names relative to ``cwd``** in ``cmd``, so the same command works natively and
through WSL (where ``cwd`` is translated to ``/mnt/c/...``).
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from .optimize import Cancelled

IS_WINDOWS = sys.platform == "win32"
WSL_PREFIX = "wsl:"


class ExternalProgramError(RuntimeError):
    """An external program is missing, failed, or produced no usable output."""


@dataclass(frozen=True)
class ProgramSpec:
    name: str                          # key used in settings.engines.paths and QUANTABENCH_<NAME>
    executables: tuple[str, ...]       # names to look for, in order (".exe" is added on Windows)
    version_args: tuple[str, ...] = ("--version",)
    version_regex: str = r"(\d+\.\d+(?:\.\d+)?)"
    install_hint: str = ""


PROGRAMS: dict[str, ProgramSpec] = {
    "xtb": ProgramSpec("xtb", ("xtb",), ("--version",), r"xtb version (\S+)",
                       "conda install -c conda-forge 'xtb>=6.7.1', or the Windows zip from "
                       "https://github.com/grimme-lab/xtb/releases"),
    "crest": ProgramSpec("crest", ("crest",), ("--version",), r"Version (\d[\w.]*)",
                         "Linux/WSL/macOS: conda install -c conda-forge crest (no Windows build: use WSL)"),
    "mopac": ProgramSpec("mopac", ("mopac", "MOPAC2016"), ("--version",), r"MOPAC version (\S+)",
                         "conda install -c conda-forge mopac, or the installer from https://openmopac.net"),
    "psi4": ProgramSpec("psi4", ("psi4",), ("--version",), r"(\d+\.\d+(?:\.\d+)?)",
                        "conda install -c conda-forge psi4 'libint=2.9' (in its own environment)"),
}


@dataclass(frozen=True)
class ProgramInfo:
    name: str
    path: str            # absolute path, or "wsl:<command>" for a program inside WSL
    source: str          # "settings" | "environment variable" | "PATH" | "this environment" | "conda env <x>"
    version: str = ""

    @property
    def via_wsl(self) -> bool:
        return self.path.startswith(WSL_PREFIX)

    def command(self) -> list[str]:
        """argv prefix that starts the program."""
        if self.via_wsl:
            return ["wsl.exe", "-e", self.path[len(WSL_PREFIX):]]
        return [self.path]


# ----------------------------------------------------------------------------------- settings access
def _engine_settings():
    """The ``engines`` section of the user's settings (or defaults if settings cannot be read)."""
    from .. import settings as settings_mod

    try:
        return settings_mod.load().engines
    except Exception:
        return settings_mod.EngineSettings()


# ----------------------------------------------------------------------------------- discovery
_CACHE: dict[str, ProgramInfo | None] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _exe_names(spec: ProgramSpec) -> list[str]:
    names = list(spec.executables)
    if IS_WINDOWS:
        names = [n if n.lower().endswith(".exe") else n + ".exe" for n in names] + names
    return names


def _env_bin_dirs(prefix: Path) -> list[Path]:
    """Folders holding executables in a venv/conda env rooted at ``prefix``."""
    if IS_WINDOWS:
        return [prefix / "Library" / "bin", prefix / "Scripts", prefix, prefix / "bin"]
    return [prefix / "bin"]


def conda_env_prefixes() -> list[Path]:
    """Conda/mamba environments in the usual places (the base env first, then ``envs/*``)."""
    home = Path.home()
    roots = []
    for var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    for name in ("miniforge3", "mambaforge", "miniconda3", "anaconda3", "micromamba", ".local/share/mamba",
                 "mamba", "AppData/Local/miniforge3", "AppData/Local/miniconda3"):
        roots.append(home / name)
    if IS_WINDOWS:
        for var in ("ProgramData", "LOCALAPPDATA"):
            if os.environ.get(var):
                roots += [Path(os.environ[var]) / "miniforge3", Path(os.environ[var]) / "miniconda3"]
    out: list[Path] = []
    for root in roots:
        for p in [root, *sorted((root / "envs").glob("*"))] if (root / "envs").is_dir() else [root]:
            if p.is_dir() and p not in out:
                out.append(p)
    return out


def _probe_version(info: ProgramInfo, spec: ProgramSpec, timeout: float = 15.0) -> str:
    if not spec.version_args:
        return ""
    try:
        cp = subprocess.run([*info.command(), *spec.version_args], capture_output=True, text=True,
                            timeout=timeout, stdin=subprocess.DEVNULL, errors="replace",
                            creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(spec.version_regex, (cp.stdout or "") + (cp.stderr or ""))
    return m.group(1) if m else ""


def _locate(spec: ProgramSpec) -> tuple[str, str] | None:
    """(path, source) of the first match, without running anything."""
    cfg_path = str(getattr(_engine_settings(), "paths", {}).get(spec.name, "") or "").strip()
    if cfg_path:
        if cfg_path.startswith(WSL_PREFIX):
            return cfg_path, "settings"
        p = Path(os.path.expanduser(cfg_path))
        if p.is_file():
            return str(p), "settings"
    env_path = os.environ.get(f"QUANTABENCH_{spec.name.upper()}", "").strip()
    if env_path:
        if env_path.startswith(WSL_PREFIX):
            return env_path, "environment variable"
        p = Path(os.path.expanduser(env_path))
        if p.is_file():
            return str(p), "environment variable"
    names = _exe_names(spec)
    for n in names:
        found = shutil.which(n)
        if found:
            return found, "PATH"
    for d in _env_bin_dirs(Path(sys.prefix)):
        for n in names:
            if (d / n).is_file():
                return str(d / n), "this environment"
    for prefix in conda_env_prefixes():
        for d in _env_bin_dirs(prefix):
            for n in names:
                if (d / n).is_file():
                    return str(d / n), f"conda env {prefix.name}"
    return None


def find_program(name: str, *, version: bool = True, refresh: bool = False) -> ProgramInfo | None:
    """Locate an external program. ``None`` if it is not installed anywhere QuantaBench looks.

    Order: ``settings.engines.paths[name]`` (may be ``wsl:<command>``), ``QUANTABENCH_<NAME>``,
    PATH, the running Python environment, conda/mamba environments in the usual places.
    Results are cached per process; ``refresh=True`` looks again (e.g. after the user edits settings).
    """
    spec = PROGRAMS[name]
    if refresh:
        _CACHE.pop(name, None)
    if name in _CACHE:
        return _CACHE[name]
    hit = _locate(spec)
    info = None
    if hit is not None:
        info = ProgramInfo(name, hit[0], hit[1])
        if version:
            info = ProgramInfo(name, hit[0], hit[1], _probe_version(info, spec))
    _CACHE[name] = info
    return info


def require_program(name: str) -> ProgramInfo:
    info = find_program(name)
    if info is None:
        spec = PROGRAMS[name]
        raise ExternalProgramError(f"{name} not found. Install it ({spec.install_hint}) or set its path in "
                                   f"Settings > Engines (or the QUANTABENCH_{name.upper()} environment variable).")
    return info


def version_tuple(version: str) -> tuple[int, ...]:
    """'6.7.1' -> (6, 7, 1); unknown parts become 0, so comparisons never raise."""
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.\-+]", version or "0")[:3])


def find_python_with(module: str, setting_key: str = "") -> str | None:
    """A Python interpreter that can ``import module`` (checked without importing it here).

    Order: ``settings.engines.paths[setting_key]``, ``QUANTABENCH_<SETTING_KEY>``, this interpreter,
    then the ``python`` next to the module's own console script (e.g. the ``psi4`` launcher of a conda env).
    """
    import importlib.util

    candidates: list[str] = []
    if setting_key:
        cfg = str(getattr(_engine_settings(), "paths", {}).get(setting_key, "") or "").strip()
        if cfg:
            candidates.append(os.path.expanduser(cfg))
        env = os.environ.get(f"QUANTABENCH_{setting_key.upper()}", "").strip()
        if env:
            candidates.append(os.path.expanduser(env))
    for c in candidates:
        if Path(c).is_file():
            return c
    if importlib.util.find_spec(module) is not None:
        return sys.executable
    spec = PROGRAMS.get(module)
    if spec is not None:
        hit = _locate(spec)
        if hit is not None and not hit[0].startswith(WSL_PREFIX):
            exe = Path(hit[0])
            for d in (exe.parent, exe.parent.parent):
                for n in (("python.exe",) if IS_WINDOWS else ("python3", "python")):
                    if (d / n).is_file():
                        return str(d / n)
    return None


# ----------------------------------------------------------------------------------- WSL
def to_wsl_path(path: str | Path) -> str:
    """``C:\\Users\\Brand\\x`` -> ``/mnt/c/Users/Brand/x`` (paths already POSIX are returned as is)."""
    s = str(path)
    if s.startswith("/"):
        return s
    p = PureWindowsPath(s)
    if not p.drive or not p.drive.endswith(":"):
        return p.as_posix()
    rest = "/".join(p.parts[1:])
    return f"/mnt/{p.drive[0].lower()}/{rest}".rstrip("/")


def wsl_available() -> bool:
    return IS_WINDOWS and shutil.which("wsl.exe") is not None


# ----------------------------------------------------------------------------------- environment
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def program_env(threads: int, extra: dict[str, str] | None = None, *, via_wsl: bool = False,
                scratch: str | Path | None = None) -> dict[str, str]:
    """Environment for a child program: thread caps, stack size, scratch, plus ``extra``.

    ``OMP_STACKSIZE`` (default 4G, ``settings.engines.omp_stacksize``) avoids the stack overflows xtb and
    CREST hit on larger molecules; it reserves address space per thread, not RAM.
    ``OMP_MAX_ACTIVE_LEVELS=1`` stops nested OpenMP from oversubscribing the cores.
    """
    es = _engine_settings()
    env = dict(os.environ)
    n = str(max(1, int(threads)))
    for var in THREAD_VARS:
        env[var] = n
    env["OMP_MAX_ACTIVE_LEVELS"] = "1"
    env["OMP_STACKSIZE"] = str(getattr(es, "omp_stacksize", "") or "4G")
    scratch = Path(scratch) if scratch else scratch_root()
    env["PSI_SCRATCH"] = to_wsl_path(scratch) if via_wsl else str(scratch)
    env.update(extra or {})
    if via_wsl:  # WSLENV lists the variables forwarded into the Linux side (/u = Windows -> WSL only)
        names = [*THREAD_VARS, "OMP_MAX_ACTIVE_LEVELS", "OMP_STACKSIZE", "PSI_SCRATCH", *(extra or {})]
        existing = [x for x in env.get("WSLENV", "").split(":") if x]
        env["WSLENV"] = ":".join(dict.fromkeys(existing + [f"{v}/u" for v in names]))
    return env


# ----------------------------------------------------------------------------------- scratch
def scratch_root() -> Path:
    es = _engine_settings()
    root = str(getattr(es, "scratch_dir", "") or "").strip()
    p = Path(os.path.expanduser(root)) if root else Path(tempfile.gettempdir()) / "QuantaBench"
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextlib.contextmanager
def work_dir(prefix: str, keep: bool | None = None) -> Iterator[Path]:
    """A fresh scratch folder for one job. Removed when the block succeeds (unless ``keep`` or
    ``settings.engines.keep_scratch``); kept on an exception so the input and log can be inspected,
    and the exception message gets the folder appended."""
    if keep is None:
        keep = bool(getattr(_engine_settings(), "keep_scratch", False))
    safe = re.sub(r"[^\w.-]+", "_", prefix)[:60] or "job"
    d = Path(tempfile.mkdtemp(prefix=f"{safe}_", dir=scratch_root()))
    try:
        yield d
    except Cancelled:
        shutil.rmtree(d, ignore_errors=True)
        raise
    except ExternalProgramError as exc:
        raise ExternalProgramError(f"{exc}\n(job files kept in {d})") from exc
    else:
        if not keep:
            shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------------- running
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0


@dataclass
class ProgramRun:
    returncode: int
    seconds: float
    log_file: Path

    def text(self) -> str:
        try:
            return self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def tail(self, n: int = 40) -> str:
        return "\n".join(self.text().splitlines()[-n:])


def _raise_stack_limit() -> None:  # POSIX child: xtb/CREST want a large main-thread stack (ulimit -s)
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if hard == resource.RLIM_INFINITY or hard > soft:
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    except Exception:
        pass


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           creationflags=_NO_WINDOW, timeout=20)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(10)


def run_program(cmd: Sequence[str], cwd: str | Path, *, threads: int = 1, env: dict[str, str] | None = None,
                stdin_text: str | None = None, timeout: float | None = None, log_name: str = "program.log",
                should_stop: Callable[[], bool] | None = None, log: Callable[[str], None] | None = None,
                check: bool = True, poll_s: float = 0.2) -> ProgramRun:
    """Run ``cmd`` in ``cwd``; stdout+stderr go to ``cwd/log_name``.

    ``cmd[0]`` may be a ``ProgramInfo.command()`` prefix, including the WSL launcher; file arguments
    must be relative to ``cwd``. ``env`` is a full environment (use ``program_env``); ``None`` means
    ``program_env(threads)``. ``should_stop`` is polled every ``poll_s`` seconds: when it returns True the
    whole process tree is killed and ``Cancelled`` is raised. A timeout raises ``ExternalProgramError``,
    so does a non-zero exit when ``check`` (with the last lines of the log).
    """
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [str(c) for c in cmd]
    via_wsl = bool(cmd) and Path(cmd[0]).name.lower() in ("wsl.exe", "wsl")
    if via_wsl:
        cmd = [cmd[0], "--cd", to_wsl_path(cwd.resolve()), *cmd[1:]]
    if env is None:
        env = program_env(threads, via_wsl=via_wsl)
    log_file = cwd / log_name
    t0 = time.perf_counter()
    if log:
        log(f"  $ {' '.join(cmd)}")
    with open(log_file, "wb") as out:
        kwargs: dict = {"cwd": str(cwd), "env": env, "stdout": out, "stderr": subprocess.STDOUT,
                        "stdin": subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL}
        if IS_WINDOWS:
            kwargs["creationflags"] = _NO_WINDOW
        else:
            kwargs["start_new_session"] = True     # own process group, so cancellation kills children
            kwargs["preexec_fn"] = _raise_stack_limit
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except OSError as exc:
            raise ExternalProgramError(f"could not start {cmd[0]}: {exc}") from exc
        if stdin_text is not None:
            try:
                proc.stdin.write(stdin_text.encode())
                proc.stdin.close()
            except OSError:
                pass
        while True:
            try:
                proc.wait(poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            if should_stop is not None and should_stop():
                _kill_tree(proc)
                raise Cancelled()
            if timeout is not None and time.perf_counter() - t0 > timeout:
                _kill_tree(proc)
                run = ProgramRun(-1, time.perf_counter() - t0, log_file)
                raise ExternalProgramError(f"{Path(cmd[0]).name} timed out after {timeout:.0f} s\n{run.tail()}")
    run = ProgramRun(proc.returncode, time.perf_counter() - t0, log_file)
    if check and run.returncode != 0:
        raise ExternalProgramError(f"{' '.join(cmd[:3])} exited with code {run.returncode}\n{run.tail()}")
    return run


# ----------------------------------------------------------------------------------- reporting
def report(names: Sequence[str] | None = None) -> list[tuple[str, ProgramInfo | None, str]]:
    """(name, info or None, install hint) for ``doctor`` and the Settings dialog."""
    out = []
    for name in names or PROGRAMS:
        out.append((name, find_program(name), PROGRAMS[name].install_hint))
    return out
