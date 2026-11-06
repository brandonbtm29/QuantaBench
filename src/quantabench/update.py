"""Updating QuantaBench: ``quantabench-cli update [--check] [--upgrade-packages]`` and Help > Check for updates...

A source checkout (how QuantaBench is installed today, docs/INSTALL.md) is updated with git:

1. ``check()`` runs ``git fetch`` and reports how far the checkout is behind ``origin/main``.
2. ``update()`` runs ``git pull --ff-only``. It refuses when tracked files have local changes, when the
   checkout is on another branch, or when it has commits of its own that ``origin/main`` lacks, and says why.
3. If ``pyproject.toml``'s dependencies or entry points changed, or this platform's entry of
   ``scripts/fork-packages.json`` (the packages the installers add: forks from a git tag or a release page,
   platform pins), it runs ``pip install -e`` with the running interpreter (the venv), passing the extras that
   are already installed and the fork packages. PyTorch is pinned to the installed build with a constraints
   file, so a ROCm/CUDA/CPU build is never swapped for another. A fork package whose entry changed is then
   reinstalled (``--force-reinstall --no-deps``), because a new build often keeps its version number.
4. ``upgrade_packages=True`` (``--upgrade-packages``) also upgrades what is installed: ``pip install -e
   .[extras] <fork packages> --upgrade --upgrade-strategy eager`` (or ``only-if-needed``), with PyTorch pinned
   the same way, then reinstalls the fork packages that come from a git tag or a release page. It works with
   nothing to pull, too. The result lists what pip changed (``pip list`` before and after).
5. The caller restarts the app (``relaunch()`` starts a new GUI process). A running ``quantabench-cli worker``
   keeps the old code until it is restarted (docs/WORKERS.md).

An install that is not a git checkout gets instructions instead. Nothing here needs the GUI.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__

REMOTE = "origin"
BRANCH = "main"
REPO_URL = "https://github.com/brandonbtm29/QuantaBench"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
# the packages the installers add on top of pyproject.toml, per platform (read by scripts/install_*.{ps1,sh} too)
FORK_FILE = "scripts/fork-packages.json"
UPGRADE_STRATEGIES = ("eager", "only-if-needed")
# pip's output when Windows refuses to replace a file that a running program has loaded
_LOCKED = re.compile(r"WinError (5|32)\b|Access is denied|being used by another process", re.IGNORECASE)
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
    forks_changed: bool = False      # this platform's entry of scripts/fork-packages.json changed in the pull
    pip_ran: bool = False
    pip_ok: bool = True
    upgraded: bool = False           # the installed packages were upgraded (upgrade_packages=True)
    package_changes: list[str] = field(default_factory=list)   # "numpy 2.1.0 -> 2.3.4", "rich 14.1.0 (new)", ...
    reinstalled: list[str] = field(default_factory=list)       # fork packages reinstalled from their pinned source
    message: str = ""
    log: str = ""                    # git/pip output, for a details box

    @property
    def restart_needed(self) -> bool:
        """The running process has old code or old packages loaded."""
        return self.updated or bool(self.package_changes) or bool(self.reinstalled)


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


# ------------------------------------------------------------------------- scripts/fork-packages.json
def platform_key(platform: str | None = None) -> str:
    """The entry of ``scripts/fork-packages.json`` for ``platform`` (default ``sys.platform``)."""
    platform = platform or sys.platform
    return "windows" if platform == "win32" else "macos" if platform == "darwin" else "linux"


@dataclass(frozen=True)
class ForkPackages:
    """One platform's entry of ``scripts/fork-packages.json``: what the installers add to ``pip install -e``."""
    packages: tuple[str, ...] = ()                  # requirement strings ("dxtb @ git+https://...@tag", "tblite")
    only_binary: tuple[str, ...] = ()               # --only-binary NAME
    find_links: tuple[tuple[str, str], ...] = ()    # (package name, URL): --find-links URL

    def flags(self) -> list[str]:
        out: list[str] = []
        for name in self.only_binary:
            out += ["--only-binary", name]
        for _name, url in self.find_links:
            out += ["--find-links", url]
        return out

    def link(self, requirement: str) -> str:
        name = requirement_name(requirement)
        return next((url for pkg, url in self.find_links if _norm(pkg) == name), "")

    def pinned_source(self, requirement: str) -> bool:
        """True for a package from a git tag or a release page (a direct URL, or a ``find_links`` entry): a new
        build there often keeps the version number, so pip has to be told to reinstall it."""
        return "@" in requirement.split(";", 1)[0] or bool(self.link(requirement))

    def pinned(self) -> list[str]:
        return [req for req in self.packages if self.pinned_source(req)]


def requirement_name(requirement: str) -> str:
    m = _REQ.match(requirement)
    return _norm(m.group(1)) if m else requirement.strip().lower()


def fork_packages(text: str, platform: str | None = None) -> ForkPackages:
    """``platform``'s entry of a ``fork-packages.json`` text; empty for an empty text or a platform without one.
    Raises ValueError when the text is not valid JSON or the entry has the wrong shape."""
    if not text.strip():
        return ForkPackages()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{FORK_FILE} is not valid JSON: {exc}") from None
    entry = data.get(platform_key(platform)) if isinstance(data, dict) else None
    if entry is None:
        return ForkPackages()

    def strings(value) -> bool:
        return isinstance(value, list) and all(isinstance(s, str) for s in value)

    packages = entry.get("packages") or [] if isinstance(entry, dict) else None
    only_binary = entry.get("only_binary") or [] if isinstance(entry, dict) else None
    links = entry.get("find_links") or {} if isinstance(entry, dict) else None
    if not (strings(packages) and strings(only_binary) and isinstance(links, dict)
            and all(isinstance(url, str) for url in links.values())):
        raise ValueError(f"{FORK_FILE}: the '{platform_key(platform)}' entry needs lists of strings for packages "
                         "and only_binary, and find_links as {package: URL}")
    return ForkPackages(tuple(packages), tuple(only_binary), tuple(links.items()))


def fork_changes(old_text: str, new_text: str, platform: str | None = None) -> tuple[bool, list[str]]:
    """(this platform's entry changed, the fork packages to reinstall) between two ``fork-packages.json`` texts.

    A package from a pinned source (git tag, release page) is reinstalled when its requirement or its find_links
    URL changed; pip handles the others (a changed version range, a new package) by itself. A checkout from
    before the file existed counts as unchanged: its installer used the same pins as the file's first version.
    An unreadable new file changes nothing (the caller reports it)."""
    if not old_text.strip():
        return False, []
    try:
        new = fork_packages(new_text, platform)
    except ValueError:
        return False, []
    try:
        old = fork_packages(old_text, platform)
    except ValueError:
        old = ForkPackages()
    if new == old:
        return False, []
    reinstall = [req for req in new.pinned()
                 if req not in old.packages or new.link(req) != old.link(req)]
    return True, reinstall


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


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


def pip_command(repo: Path, extras: list[str], constraints: Path | None, python: str | None = None,
                forks: ForkPackages | None = None, upgrade: str | None = None) -> list[str]:
    """``pip install -e repo[extras] <fork packages> <their flags> [--upgrade --upgrade-strategy S]``, as the
    installers run it (plus the upgrade)."""
    target = str(repo) + (f"[{','.join(extras)}]" if extras else "")
    cmd = [_console_python(python), "-m", "pip", "install", "--disable-pip-version-check", "-e", target]
    if forks is not None:
        cmd += [*forks.packages, *forks.flags()]
    if upgrade:
        cmd += ["--upgrade", "--upgrade-strategy", upgrade]
    if constraints is not None:
        cmd += ["--constraint", str(constraints)]
    return cmd


def reinstall_command(requirements: Sequence[str], constraints: Path | None, python: str | None = None,
                      forks: ForkPackages | None = None) -> list[str]:
    """Reinstall fork packages from their pinned source, without touching their dependencies (those were
    resolved by ``pip_command`` just before)."""
    cmd = [_console_python(python), "-m", "pip", "install", "--disable-pip-version-check", "--force-reinstall",
           "--no-deps", *requirements]
    if forks is not None:
        cmd += forks.flags()
    if constraints is not None:
        cmd += ["--constraint", str(constraints)]
    return cmd


def pip_list(python: str | None = None) -> dict[str, tuple[str, str]] | None:
    """{normalized name: (name, version)} of the interpreter's installed packages, or None if pip cannot say."""
    cmd = [_console_python(python), "-m", "pip", "list", "--format=json", "--disable-pip-version-check"]
    try:
        cp = _run(cmd, timeout=300)
        rows = json.loads(cp.stdout or "") if cp.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(rows, list):
        return None
    return {_norm(r["name"]): (r["name"], str(r.get("version", ""))) for r in rows
            if isinstance(r, dict) and isinstance(r.get("name"), str)}


def package_changes(before: dict[str, tuple[str, str]], after: dict[str, tuple[str, str]]) -> list[str]:
    """One line per package whose version changed, that appeared or that went away, sorted by name."""
    lines = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old is None and new is not None:
            lines.append(f"{new[0]} {new[1]} (new)")
        elif new is None and old is not None:
            lines.append(f"{old[0]} {old[1]} (removed)")
        elif old is not None and new is not None and old[1] != new[1]:
            lines.append(f"{new[0]} {old[1]} -> {new[1]}")
    return lines


# ------------------------------------------------------------------------------------------- update
def update(repo: Path | None = None, install: bool = True, remote: str = REMOTE, branch: str = BRANCH,
           python: str | None = None, log=None, status: UpdateStatus | None = None,
           upgrade_packages: bool = False, upgrade_strategy: str = "eager") -> UpdateResult:
    """Pull ``remote/branch`` (fast-forward only) and reinstall when the dependencies or this platform's fork
    packages changed. ``upgrade_packages`` also upgrades the installed packages (only that when there is
    nothing to pull); ``upgrade_strategy`` is pip's: ``eager`` upgrades every dependency to the newest version
    allowed, ``only-if-needed`` only QuantaBench and the fork packages (and dependencies that must move).

    ``status`` is a fresh ``check()`` result to reuse (else this checks again). ``log`` gets one line per
    step (the CLI prints them). Never raises for git/pip failures."""
    if upgrade_strategy not in UPGRADE_STRATEGIES:
        raise ValueError(f"upgrade_strategy must be one of {', '.join(UPGRADE_STRATEGIES)}, not {upgrade_strategy!r}")
    say = log or (lambda _msg: None)
    st = status if status is not None else check(repo, fetch=True, remote=remote, branch=branch)
    pull = st.state not in ("up-to-date", "ahead")
    if not pull and not upgrade_packages:
        return UpdateResult(ok=True, old_head=st.head, new_head=st.head, message=st.summary())
    if pull and not st.can_update:
        return UpdateResult(ok=False, old_head=st.head, new_head=st.head,
                            message=st.blocked_reason or st.summary())
    root = st.repo
    pyproject, fork_file = root / "pyproject.toml", root / FORK_FILE
    old_text, old_forks = _read_text(pyproject), _read_text(fork_file)
    reinstall: list[str] = []
    if pull:
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
        new_text, new_forks = _read_text(pyproject) or old_text, _read_text(fork_file)
        res.changed = dependency_changes(old_text, new_text) if old_text else ["pyproject.toml"]
        res.forks_changed, reinstall = fork_changes(old_forks, new_forks)
        lines = [f"Updated {st.head} -> {new_head} ({st.behind} commit(s))."]
    else:
        res = UpdateResult(ok=True, old_head=st.head, new_head=st.head)
        new_text, new_forks = old_text, old_forks
        lines = [f"Nothing to pull from {st.upstream} ({st.head})."]
    try:
        forks = fork_packages(new_forks)
        bad_forks = "" if new_forks.strip() else f"{FORK_FILE} is missing"
    except ValueError as exc:
        forks, bad_forks = ForkPackages(), str(exc)
    if bad_forks and upgrade_packages:
        lines.append(f"{bad_forks}, so the packages were not upgraded: without the fork packages on its command "
                     "line, pip would replace the forks with PyPI's releases.")
        upgrade_packages = False
    elif bad_forks and res.changed:
        lines.append(f"{bad_forks}: reinstalling without the fork packages (the installed ones stay).")
    if upgrade_packages:
        reinstall = list(dict.fromkeys([*reinstall, *forks.pinned()]))
    if (res.changed or res.forks_changed or upgrade_packages) and install:
        extras = installed_extras(old_text) if old_text else []
        with_extras = f" with extras [{','.join(extras)}]" if extras else ""
        if upgrade_packages:
            say(f"Upgrading the installed packages with pip (--upgrade-strategy {upgrade_strategy}){with_extras}; "
                "this can take several minutes...")
        else:
            why = [f"pyproject.toml changed ({', '.join(res.changed)})"] if res.changed else []
            why += [f"{FORK_FILE} changed"] if res.forks_changed else []
            say(f"{'; '.join(why)}: reinstalling{with_extras}...")
        before = pip_list(python)
        res.pip_ran = True
        ok, out = _pip_install(root, extras, python, forks=forks,
                               upgrade=upgrade_strategy if upgrade_packages else None, reinstall=reinstall)
        after = pip_list(python) if before is not None else None
        res.pip_ok = ok
        res.ok = res.ok and ok
        res.upgraded = ok and upgrade_packages
        res.reinstalled = [requirement_name(r) for r in reinstall] if ok else []
        res.package_changes = package_changes(before, after) if before is not None and after is not None else []
        res.log = (res.log + "\n" + out).strip()
        lines += _pip_report(res, ok, out, upgrade_packages, upgrade_strategy, listed=after is not None)
        relinked = _relink_openmp(python)
        if relinked:
            lines.append(relinked)
    elif res.changed or res.forks_changed:
        what = f"pyproject.toml changed ({', '.join(res.changed)})" if res.changed else f"{FORK_FILE} changed"
        lines.append(f"{what}: run `pip install -e .` before restarting.")
    if res.restart_needed:
        lines.append("Restart QuantaBench to use the new version.")
    res.message = "\n".join(lines)
    return res


def _pip_report(res: UpdateResult, ok: bool, out: str, upgrade: bool, strategy: str, listed: bool) -> list[str]:
    """The lines of ``update()``'s message about the pip run."""
    torch = _torch_version()
    kept = f"PyTorch kept at {torch}" if torch else "PyTorch left as it was"
    lines = []
    if ok and upgrade:
        lines.append(f"Upgraded the installed packages with pip (--upgrade-strategy {strategy}; {kept}).")
    elif ok:
        lines.append(f"{'Dependencies' if res.changed else 'The fork packages'} changed: reinstalled with pip ({kept}).")
    elif upgrade:
        lines.append("The package upgrade failed (pip's output below)."
                     + (" The code is updated." if res.updated else "")
                     + (" What pip changed before it stopped is listed." if res.package_changes else ""))
    else:
        lines.append("Dependencies changed, but `pip install -e` failed (output below). The code is updated; "
                     "rerun the install script for your OS in scripts/ to finish.")
    if res.reinstalled:
        lines.append(f"Reinstalled from {FORK_FILE}: {', '.join(res.reinstalled)}.")
    if res.package_changes:
        lines.append(f"Package changes ({len(res.package_changes)}):")
        lines += [f"  {c}" for c in res.package_changes]
    elif ok and upgrade and listed:
        lines.append("No package changed version: each was already the newest one allowed.")
    elif not listed:
        lines.append("(`pip list` did not work, so the changes cannot be listed.)")
    if not ok:
        if _LOCKED.search(out):
            lines.append("Windows keeps the files of a running program locked. Close QuantaBench and any "
                         "`quantabench-cli worker`, then run `quantabench-cli update --upgrade-packages` in a "
                         "terminal, or the install script for your OS.")
        lines.append(out[-2000:])
    return lines


def _pip_install(repo: Path, extras: list[str], python: str | None, forks: ForkPackages | None = None,
                 upgrade: str | None = None, reinstall: Sequence[str] = ()) -> tuple[bool, str]:
    """``pip install -e repo[extras] <fork packages>`` (``--upgrade`` with ``upgrade``), then ``reinstall``
    from their pinned source, with torch pinned to the installed build in both."""
    torch = _torch_version()
    tmp = None
    outputs = []
    try:
        if torch:
            fd, name = tempfile.mkstemp(prefix="qb-constraints-", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"torch=={torch}\n")
            tmp = Path(name)
        cmds = [pip_command(repo, extras, tmp, python, forks=forks, upgrade=upgrade)]
        if reinstall:
            cmds.append(reinstall_command(reinstall, tmp, python, forks=forks))
        for cmd in cmds:
            try:
                cp = _run(cmd, cwd=repo, timeout=3600)
            except (OSError, subprocess.SubprocessError) as exc:
                outputs.append(f"$ {' '.join(cmd)}\n{type(exc).__name__}: {exc}")
                return False, "\n".join(outputs)
            outputs.append(f"$ {' '.join(cmd)}\n{_err(cp)}")
            if cp.returncode != 0:
                return False, "\n".join(outputs)
        return True, "\n".join(outputs)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


# what install_macos.sh runs after pip: tblite's and PyTorch's bundled libomp.dylib, when they are two files
_DUPLICATE_OPENMP_CODE = ("from quantabench.chem.eng_tblite import duplicate_openmp as d; r = d(); "
                          "print('' if r is None else str(r[0]) + chr(10) + str(r[1]))")


def _relink_openmp(python: str | None = None) -> str:
    """macOS: after pip (re)installed tblite, link its ``libomp.dylib`` to PyTorch's again, as
    ``scripts/install_macos.sh`` does, so one OpenMP runtime loads (``chem.eng_tblite.duplicate_openmp``).
    Returns a line for the message ('' when there was nothing to do)."""
    if not IS_MACOS:
        return ""
    try:
        cp = _run([_console_python(python), "-c", _DUPLICATE_OPENMP_CODE], cwd=Path.home(), timeout=120)
    except (OSError, subprocess.SubprocessError):
        return ""
    paths = _out(cp).splitlines() if cp.returncode == 0 else []
    if len(paths) != 2:
        return ""
    ours, theirs = Path(paths[0]), Path(paths[1])
    try:
        os.replace(ours, ours.with_name(ours.name + ".qb-orig"))
        ours.symlink_to(theirs)
    except OSError as exc:
        return (f"Could not link tblite's libomp.dylib to PyTorch's ({exc}): runs that load both abort. "
                "Rerun scripts/install_macos.sh.")
    return "Linked tblite's libomp.dylib to PyTorch's again (one OpenMP runtime per process)."


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
