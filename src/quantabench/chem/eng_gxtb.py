"""g-xTB (grimme-lab/g-xtb) single points, for the conformer rank step (``chem.rank``).

g-xTB v2 is the official build of the xtb program (a modified xtb 6.7.1 interfacing a modified tblite) run with
``--gxtb``. Its binary is called ``xtb`` like the regular one, so it is a program entry of its own, ``gxtb``
(``external.PROGRAMS``): this machine's path in the settings (``engines.machine_paths``, which the installers set with
``quantabench-cli settings --engine-path gxtb=...``; never the shared ``engines.paths`` for a path that belongs to one
machine), ``QUANTABENCH_GXTB``, or a program named ``gxtb`` on PATH or in a conda environment. A plain ``xtb`` on PATH
is never taken for it.

Builds (release v2.0.1 of 2026-05-14, assets ``xtb-6.7.1-gxtb-140526-<platform>``): Linux x86-64 (static), macOS
arm64 and x86-64 (shared libraries in ``../lib``), Windows x86-64 (DLLs next to ``xtb.exe``). ``KNOWN_BUILDS`` maps the
SHA-256 of each build's executable to its release, so a run records which g-xTB it used (``build_info``: release,
platform, SHA-256, xtb's own version line); a binary with another SHA-256 still runs, recorded as an unknown build.
The release notes call it "a preliminary release for testing purposes" whose parameters are still being optimized:
pin the binary for a production run (scripts/fork-packages.json does).

Single point (``single_point``): ``xtb calc.xyz --gxtb --chrg Q --norestart`` in a job folder, and ``--uhf N`` only for
open shells (any ``--uhf``, even 0, switches g-xTB to an unrestricted wavefunction). The energy is the log's
``:: total energy`` line (v2.0.1 exits with an error when the SCF does not converge). ``--norestart`` keeps every
single point independent of the one before, so cached energies do not depend on the order. The build writes no
``xtbout.json``. ``OMP_STACKSIZE`` is capped at 1G (the Windows build's libgomp rejects the 4G the other programs get),
and the executable's folder goes first on PATH (the Windows DLLs). Threads: ``OMP_NUM_THREADS`` = the molecule's
threads.

License: the g-xtb repository is GPL-3.0; xtb itself is LGPL-3.0. QuantaBench (MIT) runs it as the user's own separate
program and never bundles or links it, as with Open Babel (docs/ENGINES.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import external

PROGRAM = "gxtb"
RELEASE_URL = "https://github.com/grimme-lab/g-xtb/releases/tag/v2.0.1"
INSTALL_HINT = ("the official build from https://github.com/grimme-lab/g-xtb/releases (v2.0.1; the installers' "
                "--with-engines / -WithEngines fetch it), then set its xtb executable as this machine's gxtb path: "
                "quantabench-cli settings --engine-path gxtb=/path/to/bin/xtb")
#: SHA-256 of the executable -> (release, platform): the builds of grimme-lab/g-xtb v2.0.1 (checked 2026-11-11)
KNOWN_BUILDS: dict[str, tuple[str, str]] = {
    "1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26": ("2.0.1", "linux-x86_64"),
    "874673d108e5d30c0989e972b3e3596ba2624395a159c8c6ee4d6248aa7979bf": ("2.0.1", "macos-arm64"),
    "e0616f9bba60125941a6cd3efa1c4dcc777c7f1c54231a0f39374d3ff593e62e": ("2.0.1", "macos-x86_64"),
    "c2af5f3db91794cc40b5e38712f28f0d24acb9af040abb0bfb627313e926cff6": ("2.0.1", "windows-x86_64"),
}
MAX_STACKSIZE = "1G"
INPUT = "calc.xyz"
LOG = "gxtb.log"
#: files a g-xTB single point writes; removed before the next one in the same folder
_OUTPUTS = ("charges", "wbo", "xtbrestart", "xtbtopo.mol", "energy", "gradient", ".sccnotconverged", "xtbout.json",
            "NOT_CONVERGED", LOG)

@dataclass(frozen=True)
class BuildInfo:
    """Which g-xTB a run used: the release and platform of a known build ("" when the SHA-256 is unknown), the
    executable's SHA-256, xtb's version line ("6.7.1 (26dd68d)") and the path."""

    release: str
    platform: str
    sha256: str
    xtb_version: str
    path: str

    @property
    def known(self) -> bool:
        return bool(self.release)

    def label(self) -> str:
        """'g-xTB 2.0.1 (sha256 1b4e30b68ed4)', or 'g-xTB, unknown build (sha256 ...)'."""
        return (f"g-xTB {self.release}" if self.release else "g-xTB, unknown build") + f" (sha256 {self.sha256[:12]})"

    def to_dict(self) -> dict:
        return asdict(self)


_BUILD_CACHE: dict[str, BuildInfo] = {}


def clear_cache() -> None:
    _BUILD_CACHE.clear()
    _RUNS.clear()


def find():
    """The g-xTB program (``external.ProgramInfo``) or None."""
    return external.find_program(PROGRAM)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_cache_file() -> Path:
    return external._version_cache_file().with_name("program_sha256.json")


def _cached_sha256(path: str) -> str:
    """The executable's SHA-256, cached on disk by (path, size, mtime) like the program versions (the Linux build is
    100 MB: hashing it takes a few tenths of a second)."""
    try:
        st = Path(path).stat()
    except OSError:
        return ""
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    cache = _sha_cache_file()
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    if isinstance(data.get(key), str) and len(data[key]) == 64:
        return data[key]
    try:
        digest = file_sha256(path)
    except OSError:
        return ""
    data = {k: v for k, v in data.items() if not k.startswith(f"{path}|")}
    data[key] = digest
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass
    return digest


def build_info(program=None) -> BuildInfo | None:
    """The g-xTB build in use (``KNOWN_BUILDS``), None when it is not found. A ``wsl:`` path is not hashed."""
    program = program or find()
    if program is None:
        return None
    hit = _BUILD_CACHE.get(program.path)
    if hit is not None:
        return hit
    sha = "" if program.via_wsl else _cached_sha256(program.path)
    release, platform = KNOWN_BUILDS.get(sha, ("", ""))
    info = BuildInfo(release, platform, sha, program.version or "", program.path)
    _BUILD_CACHE[program.path] = info
    return info


_RUNS: dict[str, tuple[bool, str]] = {}


def starts(program) -> tuple[bool, str]:
    """(ok, why not): does the program start? A version line from the program search says it does; without one,
    ``xtb --version`` runs once per path in this process with g-xTB's environment and the first line of its error is
    kept. The macOS arm64 build of v2.0.1 ships no libgcc_s.1.1.dylib, so on a Mac without Homebrew's gcc it stops at
    once in dyld ("Library not loaded: @rpath/libgcc_s.1.1.dylib"; docs/ENGINES.md, install_macos.sh fixes it)."""
    if program.via_wsl or program.version:
        return True, ""
    if program.path in _RUNS:
        return _RUNS[program.path]
    import subprocess
    import tempfile

    try:
        cp = subprocess.run([program.path, "--version"], capture_output=True, text=True, errors="replace", timeout=30,
                            cwd=tempfile.gettempdir(), env=environment(program, 1), stdin=subprocess.DEVNULL,
                            creationflags=external._NO_WINDOW)
        text = (cp.stdout or "") + (cp.stderr or "")
        ok = cp.returncode == 0 and "xtb version" in text
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), f"exit code {cp.returncode}")
        out = (True, "") if ok else (False, f"g-xTB at {program.path} does not start ({first[:200]})")
    except (OSError, subprocess.SubprocessError) as exc:
        out = (False, f"g-xTB at {program.path} does not start ({type(exc).__name__}: {exc})")
    _RUNS[program.path] = out
    return out


def available() -> tuple[bool, str]:
    """(ok, why not). Cheap after the first call (the program search, its start and the SHA-256 are cached)."""
    program = find()
    if program is None:
        return False, f"g-xTB not found: {INSTALL_HINT}"
    return starts(program)


def command(program, charge: int = 0, multiplicity: int = 1, input_name: str = INPUT) -> list[str]:
    """The single-point command line (file names relative to the job folder)."""
    cmd = [*program.command(), input_name, "--gxtb", "--chrg", str(int(charge)), "--norestart"]
    if int(multiplicity) > 1:     # --uhf (even 0) makes g-xTB unrestricted: only for open shells
        cmd += ["--uhf", str(int(multiplicity) - 1)]
    return cmd


def _stacksize(value: str) -> str:
    """``value`` (an OMP_STACKSIZE such as '4G' or '512M') capped at ``MAX_STACKSIZE``."""
    m = re.fullmatch(r"\s*(\d+)\s*([kKmMgG]?)\s*", str(value or ""))
    if not m:
        return MAX_STACKSIZE
    scale = {"": 1, "k": 2**10, "m": 2**20, "g": 2**30}[m.group(2).lower()]
    return value if int(m.group(1)) * scale <= 2**30 else MAX_STACKSIZE


def environment(program, threads: int) -> dict[str, str]:
    env = external.program_env(threads, via_wsl=program.via_wsl)
    env["OMP_STACKSIZE"] = _stacksize(env.get("OMP_STACKSIZE", ""))
    if not program.via_wsl:      # the Windows DLLs (and nothing else) next to xtb.exe
        env["PATH"] = str(Path(program.path).parent) + os.pathsep + env.get("PATH", "")
    return env


def parse_energy(text: str) -> float | None:
    """The total energy (Eh) of a g-xTB single point's log, None when there is none."""
    from .eng_xtb import parse_log

    return parse_log(text)["energy_eh"]


def _clean(folder: Path) -> None:
    for name in _OUTPUTS:
        try:
            (folder / name).unlink()
        except FileNotFoundError:
            pass


def single_point(folder: str | Path, numbers, positions_A, charge: int = 0, multiplicity: int = 1, *,
                 threads: int = 1, program=None, should_stop: Callable[[], bool] | None = None,
                 log: Callable[[str], None] | None = None) -> float:
    """One g-xTB single point in ``folder`` (reused: the previous one's files are removed first); the energy in Eh.
    Raises ``ExternalProgramError`` with the log tail when g-xTB fails."""
    from .eng_xtb import write_xyz

    program = program or find()
    if program is None:
        raise external.ExternalProgramError(f"g-xTB not found: {INSTALL_HINT}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _clean(folder)
    write_xyz(folder / INPUT, numbers, np.asarray(positions_A, float), "QuantaBench g-xTB single point")
    run = external.run_program(command(program, charge, multiplicity), folder, threads=threads,
                               env=environment(program, threads), log_name=LOG, should_stop=should_stop, log=log,
                               check=False)
    text = run.text()
    energy = parse_energy(text)
    if run.returncode != 0 or energy is None:
        why = f"exit code {run.returncode}" if run.returncode != 0 else "no total energy in its output"
        if "unknown" in text.lower() and "--gxtb" in text:
            why += f"; {program.path} does not know --gxtb: it is not the g-xTB build of xtb"
        raise external.ExternalProgramError(f"g-xTB single point failed ({why})\n{run.tail(15)}")
    return float(energy)
