"""Updating QuantaBench: ``quantabench-cli update [--check]`` and Help > Check for updates...

A source checkout (how QuantaBench is installed today, docs/INSTALL.md) is updated with git:

1. ``check()`` runs ``git fetch`` and reports how far the checkout is behind ``origin/main``.
2. ``update()`` runs ``git pull --ff-only``. It refuses when tracked files have local changes, when the
   checkout is on another branch, or when it has commits of its own that ``origin/main`` lacks, and says why.
3. If ``pyproject.toml``'s dependencies or entry points changed, it runs ``pip install -e`` with the running
   interpreter (the venv), passing the extras that are already installed. PyTorch is pinned to the
   installed build with a constraints file, so a ROCm/CUDA/CPU build is never swapped for another.
4. The caller restarts the app (``relaunch()`` starts a new GUI process).

An install that is not a git checkout gets instructions instead. Nothing here needs the GUI.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__

REMOTE = "origin"
BRANCH = "main"
REPO_URL = "https://github.com/brandonbtm29/QuantaBench"
IS_WINDOWS = sys.platform == "win32"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
# pyproject.toml tables whose change needs `pip install -e` (new packages or new console scripts)
_INSTALL_KEYS = ("dependencies", "optional-dependencies", "requires-python", "scripts", "gui-scripts",
                 "entry-points")

HOW_TO_UPDATE_NO_GIT = (
    "This QuantaBench is not a git checkout, so it cannot update itself.\n"
    "  - Installed with pip from a wheel or PyPI: pip install --upgrade quantabench\n"
    f"  - Downloaded as a zip: get the newest one from {REPO_URL}, or switch to a clone so updates are one "
    f"click:\n      git clone {REPO_URL}.git\n"
    "    then run the install script for your OS in scripts/ (see docs/INSTALL.md).")


@dataclass
class UpdateStatus:
    """Where a checkout stands relative to ``origin/main``."""
    state: str                       # up-to-date | behind | ahead | diverged | not-a-checkout | error
    repo: Path | None = None
    branch: str = ""
    head: str = ""                   # short commit id of HEAD
    upstream: str = f"{REMOTE}/{BRANCH}"
    behind: int = 0                  # commits on origin/main that HEAD lacks
    ahead: int = 0                   # commits on HEAD that origin/main lacks
    incoming: list[str] = field(default_factory=list)   # `git log --oneline` of the commits to pull
    dirty: list[str] = field(default_factory=list)      # tracked files with local changes
    fetched: bool | None = None      # None: not attempted; False: `git fetch` failed (offline?)
    message: str = ""                # extra detail (an error, a fetch failure)

    @property
    def blocked_reason(self) -> str:
        """Why ``update()`` would refuse ('' when it can go ahead or there is nothing to do)."""
        if self.state in ("not-a-checkout", "error"):
            return self.message
        if self.state not in ("behind", "diverged"):
            return ""
        if self.branch != BRANCH:
            return (f"This checkout is on branch '{self.branch or 'detached HEAD'}', not '{BRANCH}'. "
                    f"Switch back with `git switch {BRANCH}` to update.")
        if self.dirty:
            files = ", ".join(self.dirty[:5]) + (f" and {len(self.dirty) - 5} more" if len(self.dirty) > 5 else "")
            return (f"Local changes to tracked files: {files}. Pulling now would mix them with the new code, "
                    "or stop halfway. Commit them, or put them aside with `git stash`, and try again.")
        if self.state == "diverged":
            return (f"This checkout has {self.ahead} commit(s) that {self.upstream} does not, and {self.upstream} "
                    f"has {self.behind} new one(s): a fast-forward is impossible. Push or rebase your commits "
                    "(`git pull --rebase`) first.")
        return ""

    @property
    def can_update(self) -> bool:
        return self.state == "behind" and not self.blocked_reason

    def summary(self) -> str:
        """A few lines for the CLI and the GUI."""
        if self.state == "not-a-checkout":
            return f"QuantaBench {__version__}\n{self.message}"
        if self.state == "error":
            return f"QuantaBench {__version__}: could not check for updates.\n{self.message}"
        lines = [f"QuantaBench {__version__}, git checkout {self.repo} (branch {self.branch or 'detached'}, "
                 f"{self.head})"]
        if self.fetched is False:
            lines.append(f"Could not reach the remote, so this compares with the last fetched {self.upstream}."
                         + (f"\n{self.message}" if self.message else ""))
        if self.state == "up-to-date":
            lines.append(f"Up to date with {self.upstream}.")
        elif self.state == "ahead":
            lines.append(f"Up to date: {self.ahead} local commit(s) not on {self.upstream} yet, nothing to pull.")
        else:
            lines.append(f"{self.behind} new commit(s) on {self.upstream}:")
            lines += [f"  {c}" for c in self.incoming[:15]]
            if len(self.incoming) > 15:
                lines.append(f"  ... and {len(self.incoming) - 15} more")
        if self.blocked_reason:
            lines.append(self.blocked_reason)
        return "\n".join(lines)


@dataclass
class UpdateResult:
    ok: bool
    updated: bool = False            # HEAD moved
    old_head: str = ""
    new_head: str = ""
    changed: list[str] = field(default_factory=list)   # pyproject.toml keys that changed
    pip_ran: bool = False
    pip_ok: bool = True
    message: str = ""
    log: str = ""                    # git/pip output, for a details box

    @property
    def restart_needed(self) -> bool:
        return self.updated


# ---------------------------------------------------------------------------------------------- git
def _git_exe() -> str | None:
    return shutil.which("git")


def _run(cmd: list[str], cwd: Path | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run a command without a console window (the GUI runs under pythonw on Windows) and never prompt."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="", SSH_ASKPASS="")
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",  # noqa: S603
                          errors="replace", timeout=timeout, stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)


def _git(repo: Path, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return _run([_git_exe() or "git", "-C", str(repo), *args], timeout=timeout)


def _out(cp: subprocess.CompletedProcess) -> str:
    return (cp.stdout or "").strip()


def _err(cp: subprocess.CompletedProcess) -> str:
    return ((cp.stderr or "") + (cp.stdout or "")).strip()


def find_checkout(start: Path | None = None) -> Path | None:
    """The git work tree QuantaBench runs from (``start`` or this package's source checkout), or None."""
    if start is None:
        from .settings import repo_root

        start = repo_root()
        if start is None:
            return None
    start = Path(start)
    if _git_exe() is None or not start.is_dir():
        return None
    try:
        cp = _git(start, "rev-parse", "--show-toplevel", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if cp.returncode != 0 or not _out(cp):
        return None
    return Path(_out(cp))


def check(repo: Path | None = None, fetch: bool = True, remote: str = REMOTE, branch: str = BRANCH) -> UpdateStatus:
    """Fetch (unless ``fetch`` is False) and compare HEAD with ``remote/branch``. Never raises."""
    upstream = f"{remote}/{branch}"
    try:
        return _check(repo, fetch, remote, branch, upstream)
    except subprocess.TimeoutExpired as exc:
        return UpdateStatus("error", repo=repo, upstream=upstream, message=f"git timed out: {exc.cmd}")
    except (OSError, subprocess.SubprocessError) as exc:
        return UpdateStatus("error", repo=repo, upstream=upstream, message=f"{type(exc).__name__}: {exc}")


def _check(repo, fetch, remote, branch, upstream) -> UpdateStatus:
    root = find_checkout(repo)
    if root is None:
        src = Path(repo) if repo is not None else _source_tree()
        if src is not None and (src / ".git").exists() and _git_exe() is None:
            return UpdateStatus("not-a-checkout", repo=src, upstream=upstream,
                                message="This is a git checkout, but git is not installed or not on PATH. "
                                        "Install Git (https://git-scm.com) and try again.")
        return UpdateStatus("not-a-checkout", repo=src, upstream=upstream, message=HOW_TO_UPDATE_NO_GIT)
    st = UpdateStatus("up-to-date", repo=root, upstream=upstream)
    cp = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", timeout=30)
    st.branch = _out(cp) if cp.returncode == 0 else ""
    st.head = _out(_git(root, "rev-parse", "--short", "HEAD", timeout=30))
    if fetch:
        cp = _git(root, "fetch", "--quiet", remote, branch, timeout=120)
        st.fetched = cp.returncode == 0
        if not st.fetched:
            st.message = _err(cp)
    cp = _git(root, "rev-parse", "--verify", "--quiet", f"refs/remotes/{upstream}", timeout=30)
    if cp.returncode != 0:
        st.state = "error"
        st.message = (f"There is no {upstream} to compare with" + (f": {st.message}" if st.message else ".")
                      + f"\nCheck the remote with `git remote -v` (it should be {REPO_URL}.git).")
        return st
    cp = _git(root, "rev-list", "--left-right", "--count", f"HEAD...{upstream}", timeout=60)
    if cp.returncode != 0:
        st.state, st.message = "error", _err(cp)
        return st
    ahead, behind = (int(x) for x in _out(cp).split())
    st.ahead, st.behind = ahead, behind
    if behind:
        st.incoming = _out(_git(root, "log", "--oneline", "--no-decorate", f"HEAD..{upstream}", timeout=60)).splitlines()
    cp = _git(root, "status", "--porcelain=v1", "--untracked-files=no", timeout=60)
    st.dirty = [line[3:] for line in (cp.stdout or "").splitlines() if line.strip()]
    st.state = ("diverged" if ahead and behind else "behind" if behind else "ahead" if ahead else "up-to-date")
    return st


def _source_tree() -> Path | None:
    from .settings import repo_root

    return repo_root()


# ----------------------------------------------------------------------------------- pyproject.toml
def _load_toml(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


def dependency_changes(old_text: str, new_text: str) -> list[str]:
    """Keys of ``pyproject.toml`` that changed in a way that needs ``pip install -e`` (dependencies, extras,
    requires-python, entry points, build system). An unreadable file counts as changed."""
    try:
        old, new = _load_toml(old_text), _load_toml(new_text)
    except ValueError:
        return ["pyproject.toml"]
    changed = []
    po, pn = old.get("project", {}), new.get("project", {})
    for key in _INSTALL_KEYS:
        if po.get(key) != pn.get(key):
            changed.append(f"project.{key}")
    if old.get("build-system") != new.get("build-system"):
        changed.append("build-system")
    return changed


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


_REQ = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?")


def _marker_applies(marker: str) -> bool:
    try:
        from packaging.markers import Marker
    except ImportError:            # pip always has packaging; without it, assume the marker does not apply
        return False
    try:
        return Marker(marker).evaluate()
    except Exception:
        return False


def _have_distribution(name: str) -> bool:
    from importlib import metadata

    try:
        metadata.distribution(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def installed_extras(pyproject_text: str, have=_have_distribution) -> list[str]:
    """The extras of ``pyproject_text`` whose requirements are all installed in this interpreter, so an
    update reinstalls with the same extras (``all``, ``dev``, ...)."""
    try:
        project = _load_toml(pyproject_text).get("project", {})
    except ValueError:
        return []
    me = _norm(project.get("name", "quantabench"))
    extras = project.get("optional-dependencies", {}) or {}

    def requirements(extra: str, seen: set[str]) -> list[str]:
        if extra in seen:
            return []
        seen.add(extra)
        out = []
        for req in extras.get(extra, []):
            spec, _, marker = req.partition(";")
            if marker.strip() and not _marker_applies(marker.strip()):
                continue                                      # e.g. tblite on Windows
            m = _REQ.match(spec)
            if not m:
                continue
            if _norm(m.group(1)) == me:                       # quantabench[ml,xtb] inside "all"
                for sub in (m.group(2) or "").split(","):
                    out += requirements(sub.strip(), seen)
            else:
                out.append(m.group(1))
        return out

    found = []
    for extra in extras:
        reqs = requirements(extra, set())
        if reqs and all(have(r) for r in reqs):
            found.append(extra)
    return sorted(found)


def _torch_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


def _console_python(python: str | None = None) -> str:
    """The interpreter for pip: ``python.exe`` next to ``pythonw.exe`` on Windows (same venv)."""
    exe = Path(python or sys.executable)
    if IS_WINDOWS and exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        return str(exe.with_name("python.exe"))
    return str(exe)


def pip_command(repo: Path, extras: list[str], constraints: Path | None, python: str | None = None) -> list[str]:
    target = str(repo) + (f"[{','.join(extras)}]" if extras else "")
    cmd = [_console_python(python), "-m", "pip", "install", "--disable-pip-version-check", "-e", target]
    if constraints is not None:
        cmd += ["--constraint", str(constraints)]
    return cmd


# ------------------------------------------------------------------------------------------- update
def update(repo: Path | None = None, install: bool = True, remote: str = REMOTE, branch: str = BRANCH,
           python: str | None = None, log=None, status: UpdateStatus | None = None) -> UpdateResult:
    """Pull ``remote/branch`` (fast-forward only) and reinstall when the dependencies changed.

    ``status`` is a fresh ``check()`` result to reuse (else this checks again). ``log`` gets one line per
    step (the CLI prints them). Never raises for git/pip failures."""
    say = log or (lambda _msg: None)
    st = status if status is not None else check(repo, fetch=True, remote=remote, branch=branch)
    if st.state in ("up-to-date", "ahead"):
        return UpdateResult(ok=True, old_head=st.head, new_head=st.head, message=st.summary())
    if not st.can_update:
        return UpdateResult(ok=False, old_head=st.head, new_head=st.head,
                            message=st.blocked_reason or st.summary())
    root = st.repo
    pyproject = root / "pyproject.toml"
    try:
        old_text = pyproject.read_text(encoding="utf-8")
    except OSError:
        old_text = ""
    old_full = _out(_git(root, "rev-parse", "HEAD"))
    say(f"Pulling {st.behind} commit(s) from {st.upstream} (fast-forward only)...")
    try:
        cp = _git(root, "pull", "--ff-only", "--quiet", remote, branch, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return UpdateResult(ok=False, old_head=st.head, new_head=st.head, message=f"git pull failed: {exc}")
    log_text = _err(cp)
    new_full = _out(_git(root, "rev-parse", "HEAD"))
    new_head = _out(_git(root, "rev-parse", "--short", "HEAD"))
    if cp.returncode != 0:
        return UpdateResult(ok=False, updated=new_full != old_full, old_head=st.head, new_head=new_head,
                            message="git pull --ff-only failed; nothing was changed.\n" + log_text, log=log_text)
    res = UpdateResult(ok=True, updated=new_full != old_full, old_head=st.head, new_head=new_head, log=log_text)
    try:
        new_text = pyproject.read_text(encoding="utf-8")
    except OSError:
        new_text = old_text
    res.changed = dependency_changes(old_text, new_text) if old_text else ["pyproject.toml"]
    lines = [f"Updated {st.head} -> {new_head} ({st.behind} commit(s))."]
    if res.changed and install:
        extras = installed_extras(old_text) if old_text else []
        say(f"pyproject.toml changed ({', '.join(res.changed)}): reinstalling"
            + (f" with extras [{','.join(extras)}]" if extras else "") + "...")
        res.pip_ran = True
        ok, out = _pip_install(root, extras, python)
        res.pip_ok = ok
        res.log = (res.log + "\n" + out).strip()
        if ok:
            lines.append("Dependencies changed: reinstalled with pip (PyTorch left as it was).")
        else:
            res.ok = False
            lines.append("Dependencies changed, but `pip install -e` failed (output below). The code is updated; "
                         "rerun the install script for your OS in scripts/ to finish.")
            lines.append(out[-2000:])
    elif res.changed:
        lines.append(f"pyproject.toml changed ({', '.join(res.changed)}): run `pip install -e .` before restarting.")
    lines.append("Restart QuantaBench to use the new version.")
    res.message = "\n".join(lines)
    return res


def _pip_install(repo: Path, extras: list[str], python: str | None) -> tuple[bool, str]:
    """``pip install -e repo[extras]`` with torch pinned to the installed build."""
    torch = _torch_version()
    tmp = None
    try:
        if torch:
            fd, name = tempfile.mkstemp(prefix="qb-constraints-", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"torch=={torch}\n")
            tmp = Path(name)
        cmd = pip_command(repo, extras, tmp, python)
        try:
            cp = _run(cmd, cwd=repo, timeout=3600)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{' '.join(cmd)}\n{type(exc).__name__}: {exc}"
        return cp.returncode == 0, f"$ {' '.join(cmd)}\n{_err(cp)}"
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------ restart
def gui_command() -> list[str]:
    """The command that starts the GUI with this interpreter (``pythonw`` on Windows: no console)."""
    exe = Path(sys.executable)
    if IS_WINDOWS and exe.name.lower() == "python.exe" and exe.with_name("pythonw.exe").is_file():
        exe = exe.with_name("pythonw.exe")
    return [str(exe), "-m", "quantabench"]


def relaunch(cwd: Path | None = None) -> subprocess.Popen:
    """Start a new, detached QuantaBench GUI process (the caller then quits)."""
    kwargs: dict = {"cwd": str(cwd or Path.home()), "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                    "stderr": subprocess.DEVNULL, "close_fds": True}
    if IS_WINDOWS:
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(gui_command(), **kwargs)  # noqa: S603
