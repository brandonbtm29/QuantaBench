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
#: POSIX children run as ``sh -c STACK_WRAPPER prog args...``: raise the stack limit (ulimit -s; xtb and CREST
#: overflow the default 8 MB on larger molecules), then exec the program (same PID, same process group).
#: This replaces a Python ``preexec_fn``, which is unsafe in the forked child of a multi-threaded GUI.
STACK_WRAPPER = 'ulimit -s unlimited 2>/dev/null || ulimit -s "$(ulimit -Hs)" 2>/dev/null; exec "$0" "$@"'
#: The same inside WSL, plus the Linux PID in .qb_pid (taskkill on wsl.exe does not reach the Linux side).
WSL_WRAPPER = ('ulimit -s unlimited 2>/dev/null || ulimit -s "$(ulimit -Hs)" 2>/dev/null; echo $$ > .qb_pid; '
               'exec "$0" "$@"')


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
                       "conda install -c conda-forge \"xtb>=6.7.1\", or the Windows zip from "
                       "https://github.com/grimme-lab/xtb/releases"),
    "crest": ProgramSpec("crest", ("crest",), ("--version",), r"Version (\d[\w.]*)",
                         "Linux/WSL/macOS: conda install -c conda-forge crest (no Windows build: use WSL)"),
    "mopac": ProgramSpec("mopac", ("mopac", "MOPAC2016"), ("--version",), r"MOPAC version (\S+)",
                         "conda install -c conda-forge mopac, or the installer from https://openmopac.net"),
    "psi4": ProgramSpec("psi4", ("psi4",), ("--version",), r"(\d+\.\d+(?:\.\d+)?)",
                        "conda install -c conda-forge psi4 \"libint=2.9\" \"pint<0.25\" (in its own environment; docs/ENGINES.md)"),
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
        """argv prefix that starts the program. Through WSL, a small ``sh`` wrapper raises the stack limit
        (xtb/CREST need it) and records the Linux PID in ``.qb_pid`` so a cancel can kill the Linux side."""
        if self.via_wsl:
            return ["wsl.exe", "-e", "sh", "-c", WSL_WRAPPER, self.path[len(WSL_PREFIX):]]
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
_CACHE: dict[str, tuple[ProgramInfo | None, str]] = {}   # name -> (info, how far its version was probed)


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
    # no wrapper here: the WSL one writes .qb_pid into the working directory, and a probe has no job folder
    cmd = ["wsl.exe", "-e", info.path[len(WSL_PREFIX):]] if info.via_wsl else [info.path]
    try:
        cp = subprocess.run([*cmd, *spec.version_args], capture_output=True, text=True, cwd=tempfile.gettempdir(),
                            timeout=timeout, stdin=subprocess.DEVNULL, errors="replace",
                            creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(spec.version_regex, (cp.stdout or "") + (cp.stderr or ""))
    return m.group(1) if m else ""


def _configured_paths() -> dict:
    paths = getattr(_engine_settings(), "paths", None)
    return paths if isinstance(paths, dict) else {}


def _locate(spec: ProgramSpec) -> tuple[str, str] | None:
    """(path, source) of the first match, without running anything. A configured path that does not exist
    (or a ``wsl:`` path on a machine that is not Windows, e.g. settings synced from Windows to Linux) is
    skipped, and the search goes on; the source of what is found then says so."""
    skipped = []
    for value, source in ((str(_configured_paths().get(spec.name, "") or "").strip(), "settings"),
                          (os.environ.get(f"QUANTABENCH_{spec.name.upper()}", "").strip(), "environment variable")):
        if not value:
            continue
        note = f" ({'; '.join(skipped)})" if skipped else ""
        if value.startswith(WSL_PREFIX):
            if IS_WINDOWS:
                return value, source + note
            skipped.append(f"{source}: {value} is for Windows")
            continue
        p = Path(os.path.expanduser(value))
        if p.is_file():
            return str(p), source + note
        skipped.append(f"{source}: {value} not found")
    hit = _search(spec)
    if hit is not None and skipped:
        return hit[0], f"{hit[1]} ({'; '.join(skipped)})"
    return hit


def _search(spec: ProgramSpec) -> tuple[str, str] | None:
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


def _version_cache_file() -> Path:
    """Per-machine cache (not in the synced settings folder); ``QUANTABENCH_CACHE_DIR`` overrides it."""
    from platformdirs import user_cache_dir

    root = os.environ.get("QUANTABENCH_CACHE_DIR", "").strip() or user_cache_dir("QuantaBench", appauthor=False)
    return Path(root) / "program_versions.json"


def _cached_version(info: ProgramInfo, spec: ProgramSpec, refresh: bool) -> str:
    """Version of a local program, cached on disk by (path, size, mtime): building the GUI's engine menus
    then never waits for ``--version`` runs after the first start."""
    import json

    try:
        st = Path(info.path).stat()
        key = f"{info.path}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return _probe_version(info, spec)
    path = _version_cache_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    if not refresh and isinstance(data.get(key), str) and data[key]:
        return data[key]
    version = _probe_version(info, spec)
    if version:
        data = {k: v for k, v in data.items() if not k.startswith(f"{info.path}|")}
        data[key] = version
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass
    return version


def find_program(name: str, *, version: bool = True, refresh: bool = False,
                 probe_wsl: bool = False) -> ProgramInfo | None:
    """Locate an external program. ``None`` if it is not installed anywhere QuantaBench looks.

    Order: ``settings.engines.paths[name]`` (may be ``wsl:<command>``), ``QUANTABENCH_<NAME>``,
    PATH, the running Python environment, conda/mamba environments in the usual places.
    Results are cached per process, and versions on disk by (path, size, mtime); ``refresh=True`` looks
    and probes again (e.g. after the user edits settings). A ``wsl:`` program's version is only probed
    with ``probe_wsl=True`` (starting WSL can take seconds; ``doctor`` and Settings > Detect now do it).
    """
    spec = PROGRAMS[name]
    if refresh:
        _CACHE.pop(name, None)
    want = "wsl" if probe_wsl else ("local" if version else "none")   # how much probing the caller needs
    rank = {"none": 0, "local": 1, "wsl": 2}
    if name in _CACHE:
        info, probed = _CACHE[name]
        if info is None or rank[probed] >= rank[want] or (want == "wsl" and not info.via_wsl and probed == "local"):
            return info
    hit = _locate(spec)
    info = None
    if hit is not None:
        info = ProgramInfo(name, hit[0], hit[1])
        if version and not info.via_wsl:
            info = ProgramInfo(name, hit[0], hit[1], _cached_version(info, spec, refresh))
        elif version and probe_wsl:
            info = ProgramInfo(name, hit[0], hit[1], _probe_version(info, spec, timeout=60.0))
    _CACHE[name] = (info, want)
    return info


def inspect_path(name: str, path: str) -> ProgramInfo | None:
    """``ProgramInfo`` for an explicit path (or ``wsl:<command>``) the user typed, with its version;
    ``None`` if the file does not exist. Does not touch the cache."""
    spec = PROGRAMS[name]
    path = path.strip()
    if not path.startswith(WSL_PREFIX):
        p = Path(os.path.expanduser(path))
        if not p.is_file():
            return None
        path = str(p)
    info = ProgramInfo(name, path, "typed path")
    return ProgramInfo(name, path, "typed path", _probe_version(info, spec))


def python_module_version(python: str, module: str, timeout: float = 60.0) -> str | None:
    """``module.__version__`` as seen by interpreter ``python`` (None if it cannot import it)."""
    try:
        cp = subprocess.run([python, "-I", "-c", f"import {module}; print({module}.__version__)"],
                            capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                            errors="replace", creationflags=_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    out = (cp.stdout or "").strip().splitlines()
    return out[-1] if cp.returncode == 0 and out else None


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


def ascii_name(text: str, limit: int = 60) -> str:
    """A file-name-safe ASCII version of ``text`` (molecule ids like "α-pinene" -> "pinene"): some programs
    (Psi4 on Windows) open files through narrow ANSI paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-")[:limit] or "job"


@contextlib.contextmanager
def work_dir(prefix: str, keep: bool | None = None) -> Iterator[Path]:
    """A fresh scratch folder for one job. Removed when the block succeeds (unless ``keep`` or
    ``settings.engines.keep_scratch``); kept on an exception so the input and log can be inspected,
    and the exception message gets the folder appended."""
    if keep is None:
        keep = bool(getattr(_engine_settings(), "keep_scratch", False))
    safe = ascii_name(prefix)
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


def _kill_wsl_side(cwd: Path) -> None:
    """Kill the Linux process a ``wsl:`` program started (its PID is in ``cwd/.qb_pid``)."""
    if not (cwd / ".qb_pid").is_file():
        return
    script = ('p=$(cat .qb_pid 2>/dev/null); [ -n "$p" ] || exit 0; kill -TERM "$p" 2>/dev/null; '
              'for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$p" 2>/dev/null || exit 0; sleep 0.5; done; '
              'kill -KILL "$p" 2>/dev/null; true')
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(["wsl.exe", "--cd", to_wsl_path(cwd.resolve()), "-e", "sh", "-c", script],
                       capture_output=True, timeout=30, stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)


def _kill_tree(proc: subprocess.Popen, cwd: Path | None = None, via_wsl: bool = False) -> None:
    """Stop ``proc`` and everything it started: the process group on POSIX (SIGTERM, then SIGKILL for
    whatever is left), ``taskkill /T`` on Windows, and the Linux side of a WSL program."""
    if via_wsl and cwd is not None:
        _kill_wsl_side(cwd)
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            cp = subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                                creationflags=_NO_WINDOW, timeout=20)
            if cp.returncode != 0 and proc.poll() is None:
                proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                pass
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)   # members that ignored SIGTERM, or a stuck leader
    except (OSError, subprocess.SubprocessError):
        with contextlib.suppress(OSError):
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
    if timeout is None:
        limit = float(getattr(_engine_settings(), "timeout_s", 0) or 0)
        timeout = limit if limit > 0 else None
    shown = cmd
    if not IS_WINDOWS and not via_wsl:
        cmd = ["/bin/sh", "-c", STACK_WRAPPER, *cmd]
    log_file = cwd / log_name
    t0 = time.perf_counter()
    if log:
        log(f"  $ {' '.join(shown)}")
    with open(log_file, "wb") as out:
        kwargs: dict = {"cwd": str(cwd), "env": env, "stdout": out, "stderr": subprocess.STDOUT,
                        "stdin": subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL}
        if IS_WINDOWS:
            kwargs["creationflags"] = _NO_WINDOW
        else:
            kwargs["start_new_session"] = True     # own process group, so cancellation kills children
            if not Path(shown[0]).is_file() and shutil.which(shown[0]) is None:
                raise ExternalProgramError(f"could not start {shown[0]}: no such program")
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except OSError as exc:
            raise ExternalProgramError(f"could not start {shown[0]}: {exc}") from exc
        if stdin_text is not None:
            try:
                proc.stdin.write(stdin_text.encode())
                proc.stdin.close()
            except OSError:
                pass
        try:
            while True:
                try:
                    proc.wait(poll_s)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if should_stop is not None and should_stop():
                    raise Cancelled()
                if timeout is not None and time.perf_counter() - t0 > timeout:
                    _kill_tree(proc, cwd, via_wsl)
                    run = ProgramRun(-1, time.perf_counter() - t0, log_file)
                    raise ExternalProgramError(f"{Path(shown[0]).name} timed out after {timeout:.0f} s\n"
                                               f"{run.tail()}")
        except BaseException:
            # Cancel, Ctrl+C (the child has its own session, so SIGINT never reaches it), or any error in
            # should_stop/log: never leave the program running.
            _kill_tree(proc, cwd, via_wsl)
            raise
    run = ProgramRun(proc.returncode, time.perf_counter() - t0, log_file)
    if check and run.returncode != 0:
        raise ExternalProgramError(f"{' '.join(shown[:3])} exited with code {run.returncode}\n{run.tail()}")
    return run


# ----------------------------------------------------------------------------------- reporting
def report(names: Sequence[str] | None = None) -> list[tuple[str, ProgramInfo | None, str]]:
    """(name, info or None, install hint) for ``doctor`` and the Settings dialog."""
    out = []
    for name in names or PROGRAMS:
        out.append((name, find_program(name, probe_wsl=True), PROGRAMS[name].install_hint))
    return out
