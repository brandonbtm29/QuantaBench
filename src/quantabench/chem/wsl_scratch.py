"""Job folders inside WSL for programs that run through WSL on Windows (CREST; CREST-SCRATCH-1).

Why: a ``wsl:`` program used to run in its Windows job folder (``external.work_dir``), which Linux reaches as
``/mnt/c/...``. Every file it writes then crosses WSL's 9P bridge to NTFS. CREST writes a lot: MTD trajectories, the
optimization of every snapshot, many small temporary files per thread, in every lane at once. On the workstation,
just after a CREST batch started, that was about 20 MB/s of disk writes (some 70 GB an hour).

Now the program runs in a folder of WSL's own file system, and only what the pipeline reads comes back:

- ``/dev/shm/quantabench/<job>`` (RAM) when /dev/shm and WSL's available memory both have room for
  ``engines.crest_wsl_shm_min_gb`` (default 2 GB) plus ``SHM_MARGIN`` times the job's expected size. WSL's memory is
  capped (by default half the PC's), and several lanes run at once.
- otherwise ``${XDG_CACHE_HOME:-~/.cache}/quantabench/scratch/<job>``, on WSL's ext4 disk.

``engines.crest_wsl_scratch``: "auto" (the above), "shm", "disk", or "off" (the Windows folder through /mnt, as
before). A native program (Linux, macOS, Windows) is not affected: it runs in its job folder, as always.

The Windows job folder still exists. It holds the inputs, the program's output (stdout, which QuantaBench writes
through the pipe, not 9P), and what is copied back:

- success: the files the caller fetches (``Place.fetch``; CREST: crest_conformers.xyz, crest.energies, crest_best.xyz);
- failure or timeout: the folder's top-level files up to ``external.KEEP_FAILED_MAX_BYTES``, for diagnosis;
- ``engines.keep_scratch``: everything;
- cancelled: nothing (the Windows folder goes too).

The folder inside WSL is removed in every case (``finally``). A process killed meanwhile cannot do that, so:

- the Windows job folder names its twin (``external.WSL_TWIN_FILE``), and ``external.sweep_scratch`` (every run start,
  every lane kill, ``quantabench-cli scratch --clean``) removes the twin of a left-behind folder;
- every folder inside WSL has the owner record of a Windows one (``external.OWNER_FILE``, the Windows process), and
  each new job removes the folders of owners that have ended (``stale``; ``sweep`` for ``scratch --clean``);
- /dev/shm empties itself whenever WSL shuts down.

Each step is one POSIX shell script run as ``wsl.exe -e sh -c SCRIPT sh ARGS...`` (``run_sh``). The tests run the
same scripts on Linux through a stand-in ``wsl.exe``.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import external
from .optimize import Cancelled

MODES = ("auto", "shm", "disk", "off")
DEFAULT_MODE = "auto"
#: the RAM folder (its parent is the tmpfs whose free space is checked)
SHM_ROOT = "/dev/shm/quantabench"
#: the disk folder, under ``${XDG_CACHE_HOME:-$HOME/.cache}`` inside WSL
DISK_SUBDIR = "quantabench/scratch"
MEMINFO = "/proc/meminfo"
#: /dev/shm and WSL's available memory must keep at least this much free (``engines.crest_wsl_shm_min_gb``) ...
DEFAULT_SHM_MIN_GB = 2.0
#: ... after this many times the job's expected size
SHM_MARGIN = 2
SH_TIMEOUT_S = 60.0              # one shell step (WSL may have to start)
COPY_ALL_TIMEOUT_S = 900.0       # keep_scratch: the whole folder comes back
GB = 1e9
#: files a program writes that never come back (the WSL wrapper's PID file)
NEVER_COPIED = (".qb_pid",)


class WslScratchError(RuntimeError):
    """A step inside WSL failed (WSL missing or not answering, no room, a copy that failed)."""


# --------------------------------------------------------------------------------------- the shell steps
def _one_line(script: str) -> str:
    """A script written one statement per line, each ending in ';' (or in do, then, else or a brace), as one line:
    it goes to ``wsl.exe`` as a command-line argument, like ``external.WSL_WRAPPER``. No comments in the scripts."""
    return " ".join(ln.strip() for ln in script.strip().splitlines() if ln.strip())


#: what the choice needs: the disk folder, free space on /dev/shm, available memory, and the job folders there
#: (path, age in s, size in KB, owner record). Args: SHM_ROOT, DISK_SUBDIR, MEMINFO.
PROBE_SCRIPT = _one_line(r'''
shm=$1; cache="${XDG_CACHE_HOME:-$HOME/.cache}"; disk="$cache/$2";
echo "disk_root=$disk";
echo "shm_free_kb=$(df -Pk -- "$(dirname -- "$shm")" 2>/dev/null | awk 'NR == 2 {print $4}')";
echo "mem_available_kb=$(awk '/^MemAvailable:/ {print $2}' "$3" 2>/dev/null)";
now=$(date +%s);
for root in "$shm" "$disk"; do
  [ -d "$root" ] || continue;
  for d in "$root"/*/; do
    [ -d "$d" ] || continue;
    d=${d%/};
    age=$(( now - $(stat -c %Y -- "$d" 2>/dev/null || echo "$now") ));
    kb=$(du -sk -- "$d" 2>/dev/null | cut -f1);
    owner=$(cat "$d/.qb_owner.json" 2>/dev/null | tr -d '\n\t');
    printf 'job\t%s\t%s\t%s\t%s\n' "$d" "$age" "${kb:-0}" "$owner";
  done;
done;
exit 0
''')

#: make the job folder (mktemp under the root), write the owner record, copy the inputs from the Windows folder.
#: Args: root, name, owner JSON, the Windows folder as a WSL path, input names...
PREPARE_SCRIPT = _one_line(r'''
root=$1; name=$2; owner=$3; src=$4; shift 4;
mkdir -p "$root" || exit 3;
d=$(mktemp -d "$root/${name}_XXXXXX") || exit 3;
printf '%s\n' "$owner" > "$d/.qb_owner.json" || { rm -rf "$d"; exit 3; };
for f in "$@"; do
  cp "$src/$f" "$d/$f" || { rm -rf "$d"; exit 4; };
done;
printf 'dir=%s\n' "$d"
''')

#: copy back to the Windows folder. Args: job folder, the Windows folder as a WSL path, mode, max bytes, names...
#: "names": these files (those that exist); "small": every top-level file up to max bytes; "all": everything (also
#: subfolders). In "small" and "all" the names are the ones never to copy (the Windows folder's own files).
FETCH_SCRIPT = _one_line(r'''
src=$1; dest=$2; mode=$3; max=$4; shift 4;
cd "$src" || exit 3;
rc=0;
if [ "$mode" = names ]; then
  for f in "$@"; do
    [ -f "$f" ] || continue;
    cp "$f" "$dest/$f" || rc=4;
  done;
  exit $rc;
fi;
for f in * .[!.]* ..?*; do
  [ -e "$f" ] || continue;
  skip=0;
  for s in "$@"; do [ "$s" = "$f" ] && skip=1; done;
  [ $skip = 1 ] && continue;
  if [ "$mode" = all ]; then
    cp -R "$f" "$dest/" || rc=4;
  elif [ -f "$f" ]; then
    size=$(wc -c < "$f");
    [ $size -le "$max" ] && { cp "$f" "$dest/$f" || rc=4; };
  fi;
done;
exit $rc
''')

#: remove job folders; only paths with a quantabench folder above them (never a root, never anything else)
REMOVE_SCRIPT = _one_line(r'''
rc=0;
for d in "$@"; do
  case "$d" in
    /*/quantabench/?*) rm -rf -- "$d" || rc=4 ;;
    *) echo "not a QuantaBench job folder: $d" >&2; rc=5 ;;
  esac;
done;
exit $rc
''')


def run_sh(script: str, *args: str, timeout: float = SH_TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run a POSIX shell script inside WSL: ``wsl.exe -e sh -c SCRIPT sh ARGS...`` (the default distribution and
    user). ``WslScratchError`` when it cannot start, times out or exits non-zero."""
    cmd = ["wsl.exe", "-e", "sh", "-c", script, "sh", *(str(a) for a in args)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, stdin=subprocess.DEVNULL, creationflags=external._NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        raise WslScratchError(f"wsl.exe: {exc}") from exc
    if cp.returncode != 0:
        # wsl.exe's own messages (no distribution, ...) are UTF-16: drop the NULs
        why = ((cp.stderr or "") + (cp.stdout or "")).replace("\x00", "").strip()
        raise WslScratchError(f"exit code {cp.returncode}" + (f": {why[-300:]}" if why else ""))
    return cp


# --------------------------------------------------------------------------------------- settings
def mode_setting() -> str:
    """``engines.crest_wsl_scratch`` ("auto" when unset or not one of ``MODES``)."""
    mode = str(getattr(external._engine_settings(), "crest_wsl_scratch", DEFAULT_MODE) or DEFAULT_MODE)
    mode = mode.strip().lower()
    return mode if mode in MODES else DEFAULT_MODE


def shm_min_bytes() -> int:
    """``engines.crest_wsl_shm_min_gb`` in bytes (never negative)."""
    gb = getattr(external._engine_settings(), "crest_wsl_shm_min_gb", DEFAULT_SHM_MIN_GB)
    try:
        return int(max(0.0, float(gb)) * GB)
    except (TypeError, ValueError):
        return int(DEFAULT_SHM_MIN_GB * GB)


def _keep() -> bool:
    """``engines.keep_scratch``: everything comes back, and nothing left behind is removed."""
    return bool(getattr(external._engine_settings(), "keep_scratch", False))


def in_use() -> bool:
    """Does this machine run CREST through WSL (Windows, a ``wsl:`` path)? Then ``quantabench-cli scratch`` also looks
    inside WSL. Finds the program without starting WSL."""
    if not external.IS_WINDOWS:
        return False
    info = external.find_program("crest", version=False)
    return info is not None and info.via_wsl


# --------------------------------------------------------------------------------------- what is there
@dataclass
class JobFolder:
    """A job folder inside WSL (``PROBE_SCRIPT``)."""
    path: str
    age_s: float
    size: int                    # bytes
    owner: dict | None


@dataclass
class Probe:
    disk_root: str
    shm_free: int | None         # bytes free on /dev/shm (None: unknown)
    mem_available: int | None    # WSL's MemAvailable in bytes (None: unknown)
    jobs: list[JobFolder] = field(default_factory=list)


def _kb(text: str) -> int | None:
    text = text.strip()
    return int(text) * 1024 if text.isdigit() else None


def parse_probe(text: str) -> Probe:
    """``PROBE_SCRIPT``'s output -> ``Probe``."""
    values: dict[str, str] = {}
    jobs: list[JobFolder] = []
    for line in text.splitlines():
        if line.startswith("job\t"):
            parts = line.split("\t", 4)
            if len(parts) < 4:
                continue
            owner = None
            if len(parts) == 5 and parts[4].strip():
                try:
                    owner = json.loads(parts[4])
                except ValueError:
                    owner = None
                if not isinstance(owner, dict):
                    owner = None
            try:
                age = float(parts[2])
            except ValueError:
                age = 0.0
            jobs.append(JobFolder(parts[1], age, _kb(parts[3]) or 0, owner))
        elif "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    disk = values.get("disk_root", "")
    if not disk.startswith("/"):
        raise WslScratchError(f"no home folder inside WSL ({disk!r})")
    return Probe(disk, _kb(values.get("shm_free_kb", "")), _kb(values.get("mem_available_kb", "")), jobs)


def probe() -> Probe:
    """Free space on /dev/shm, WSL's available memory, the disk folder and the job folders (one ``wsl.exe`` call)."""
    return parse_probe(run_sh(PROBE_SCRIPT, SHM_ROOT, DISK_SUBDIR, MEMINFO).stdout)


def stale(jobs: Sequence[JobFolder], owners: set[int] | None = None) -> list[JobFolder]:
    """The job folders whose owner (a Windows process) has ended, as ``external.sweep_scratch`` decides it: a running
    owner's folder, or one of another host, is never stale; one without an owner record only after
    ``external.UNOWNED_GRACE_H`` hours unchanged. ``owners``: only these pids' folders."""
    out = []
    for j in jobs:
        if j.owner is not None:
            if owners is not None and j.owner.get("pid") not in owners:
                continue
            if external._owner_running(j.owner):
                continue
        elif owners is not None or j.age_s < external.UNOWNED_GRACE_H * 3600:
            continue
        out.append(j)
    return out


def _is_job_folder(path: str) -> bool:
    parts = path.split("/")
    return (path.startswith("/") and "quantabench" in parts[1:-1] and bool(parts[-1])
            and ".." not in parts and parts[-1] not in ("quantabench", "scratch"))


def remove(paths: Sequence[str]) -> None:
    """Remove job folders inside WSL (one ``wsl.exe`` call). Paths that are not QuantaBench job folders are refused."""
    paths = [p for p in paths if p]
    bad = [p for p in paths if not _is_job_folder(p)]
    if bad:
        raise WslScratchError(f"not a QuantaBench job folder inside WSL: {bad[0]}")
    if paths:
        run_sh(REMOVE_SCRIPT, *paths)


def sweep(owners: set[int] | None = None, dry_run: bool = False) -> external.SweepResult:
    """Remove the job folders inside WSL whose owner has ended (``stale``); ``quantabench-cli scratch --clean`` on a
    machine that runs CREST through WSL. ``folders``/``removed``: stale ones, ``running``: the others. Off with
    ``engines.keep_scratch``, like ``external.sweep_scratch``."""
    res = external.SweepResult()
    if _keep():
        return res
    pr = probe()
    old = stale(pr.jobs, owners)
    res.folders = len(old)
    res.running = len(pr.jobs) - len(old)
    if old and not dry_run:
        remove([j.path for j in old])
    res.removed = len(old)
    res.freed_bytes = sum(j.size for j in old)
    return res


# --------------------------------------------------------------------------------------- the choice
def need_bytes(expected: int, min_bytes: int) -> int:
    """What /dev/shm (and WSL's memory) must have free for a job of ``expected`` bytes."""
    return int(min_bytes) + SHM_MARGIN * max(0, int(expected))


def choose(mode: str, pr: Probe, expected: int, min_bytes: int) -> tuple[str, str]:
    """("shm" | "disk", why) for ``mode`` "auto", "shm" or "disk"."""
    if mode in ("shm", "disk"):
        return mode, f"crest_wsl_scratch = {mode}"
    need = need_bytes(expected, min_bytes)
    if pr.shm_free is None or pr.mem_available is None:
        return "disk", "free memory of /dev/shm unknown"
    room = (f"/dev/shm {pr.shm_free / GB:.1f} GB free, {pr.mem_available / GB:.1f} GB of WSL's memory available, "
            f"{need / GB:.1f} GB wanted")
    if pr.shm_free >= need and pr.mem_available >= need:
        return "shm", room
    return "disk", room


# --------------------------------------------------------------------------------------- the job folder
@dataclass
class Place:
    """Where a program runs: ``linux_dir`` inside WSL ("shm" | "disk"), or its Windows/local job folder (``linux_dir``
    None; "windows": a ``wsl:`` program through /mnt, "local": a native program). ``note``: one line for the
    molecule's notes ("CREST scratch: WSL /dev/shm (...)"; empty for a native program)."""
    win_dir: Path
    kind: str = "local"
    linux_dir: str | None = None
    note: str = ""

    def fetch(self, names: Sequence[str]) -> None:
        """Copy these files (those that exist) from the folder inside WSL to the job folder; nothing when the program
        ran there."""
        if self.linux_dir:
            _copy_back(self, "names", names)


def _copy_back(place: Place, mode: str, names: Sequence[str] = ()) -> None:
    if mode != "names":         # never overwrite what the job folder has (the log, the inputs, the owner record)
        try:
            names = [*NEVER_COPIED, *(p.name for p in place.win_dir.iterdir())]
        except OSError:
            names = [*NEVER_COPIED, external.OWNER_FILE, external.WSL_TWIN_FILE]
    dest = external.to_wsl_path(place.win_dir.resolve())
    try:
        run_sh(FETCH_SCRIPT, place.linux_dir or "", dest, mode, str(int(external.KEEP_FAILED_MAX_BYTES)), *names,
               timeout=COPY_ALL_TIMEOUT_S if mode == "all" else SH_TIMEOUT_S)
    except WslScratchError as exc:
        raise external.ExternalProgramError(f"could not copy the job's files out of WSL ({place.linux_dir}): "
                                            f"{exc}") from exc


def _discard(place: Place) -> None:
    """Remove the folder inside WSL, then the job folder's record of it (kept when the removal failed, for the
    sweep)."""
    try:
        remove([place.linux_dir or ""])
    except WslScratchError:
        return
    with contextlib.suppress(OSError):
        (place.win_dir / external.WSL_TWIN_FILE).unlink()


def _prepare(win_dir: Path, mode: str, inputs: Sequence[str], expected: int, what: str,
             log: Callable[[str], None]) -> Place:
    pr = probe()
    old = [] if _keep() else stale(pr.jobs)
    if old:
        try:
            remove([j.path for j in old])
            log(f"  {what} scratch: removed {len(old)} job folder(s) of ended runs inside WSL "
                f"({sum(j.size for j in old) / GB:.2f} GB)")
        except WslScratchError as exc:
            log(f"  {what} scratch: could not remove job folders of ended runs inside WSL ({exc})")
    kind, why = choose(mode, pr, expected, shm_min_bytes())
    roots = [("shm", SHM_ROOT), ("disk", pr.disk_root)] if kind == "shm" else [("disk", pr.disk_root)]
    owner = json.dumps({**external.owner_record(), "win_dir": str(win_dir)})
    src = external.to_wsl_path(win_dir.resolve())
    errors: list[str] = []
    for used, root in roots:
        try:
            out = run_sh(PREPARE_SCRIPT, root, external.ascii_name(win_dir.name), owner, src, *inputs).stdout
        except WslScratchError as exc:
            errors.append(f"{root}: {exc}")
            continue
        linux = next((ln[4:].strip() for ln in out.splitlines() if ln.startswith("dir=")), "")
        if not _is_job_folder(linux):
            errors.append(f"{root}: no job folder ({out.strip()[-200:]!r})")
            continue
        with contextlib.suppress(OSError):
            (win_dir / external.WSL_TWIN_FILE).write_text(json.dumps({"dir": linux, "kind": used}), encoding="utf-8")
        where = "/dev/shm" if used == "shm" else "disk (ext4)"
        if errors:
            why = f"/dev/shm failed: {errors[-1]}"
        return Place(win_dir, used, linux, f"{what} scratch: WSL {where} ({why})")
    raise WslScratchError("; ".join(errors))


@contextlib.contextmanager
def job_folder(program, win_dir: str | Path, *, inputs: Sequence[str] = (), expected_bytes: int = 0,
               what: str = "CREST", log: Callable[[str], None] | None = None) -> Iterator[Place]:
    """Where ``program`` (an ``external.ProgramInfo``) runs for the job in ``win_dir`` (an ``external.work_dir``
    folder holding ``inputs``). Pass ``place.linux_dir`` to ``run_program(program_cwd=...)`` and ``place.fetch`` the
    results the caller reads.

    A native program, or ``engines.crest_wsl_scratch`` "off": ``win_dir`` itself, as before. A ``wsl:`` program: a
    new folder inside WSL with the inputs (module docstring), copied back from on a failure (small files) or with
    ``keep_scratch`` (everything), and removed whatever happens. When no folder can be made inside WSL the job runs
    in ``win_dir`` through /mnt, with a warning."""
    log = log or (lambda _m: None)
    win_dir = Path(win_dir)
    if not getattr(program, "via_wsl", False):
        yield Place(win_dir)
        return
    mode = mode_setting()
    if mode == "off":
        yield Place(win_dir, "windows", note=f"{what} scratch: the Windows job folder through /mnt "
                                             "(crest_wsl_scratch = off)")
        return
    try:
        place = _prepare(win_dir, mode, inputs, expected_bytes, what, log)
    except WslScratchError as exc:
        note = f"{what} scratch: the Windows job folder through /mnt (no folder inside WSL: {exc})"
        log(f"  WARNING: {note}")
        yield Place(win_dir, "windows", note=note)
        return
    keep = _keep()
    try:
        yield place
    except Cancelled:
        raise
    except BaseException:
        with contextlib.suppress(Exception):
            _copy_back(place, "all" if keep else "small")
        raise
    else:
        if keep:
            with contextlib.suppress(Exception):
                _copy_back(place, "all")
    finally:
        _discard(place)
