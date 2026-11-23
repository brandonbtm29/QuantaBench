"""quantabench.update, `quantabench-cli update` and Help > Check for updates..., on throwaway git repos
(a bare "origin", a clone that plays the user's checkout, and a second clone that pushes new commits)."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from quantabench import cli, update

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")

PYPROJECT = """\
[project]
name = "quantabench"
version = "0.0.1"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
ml = ["aimnet>=0.2"]
dev = ["pytest>=8"]
win = ["tblite>=0.4; sys_platform == 'no-such-platform'"]
all = ["quantabench[ml]"]

[project.scripts]
quantabench-cli = "quantabench.cli:main"
"""


WIN1 = "https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win1"
WIN2 = "https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win2"
DXTB = "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1"


def forks_json(windows_link: str = WIN1, dxtb: str = DXTB, macos_extra: tuple[str, ...] = (),
               libcint: str = "tad-libcint>=0.4") -> str:
    return json.dumps({
        "_about": ["test copy of scripts/fork-packages.json"],
        "windows": {"packages": [dxtb, libcint, "tblite"], "only_binary": ["tblite"],
                    "find_links": {"tblite": windows_link}},
        "macos": {"packages": [dxtb, libcint, *macos_extra], "only_binary": ["tblite"], "find_links": {}},
        "linux": {"packages": ["tad-libcint<0.4"], "only_binary": [], "find_links": {}},
    }, indent=2) + "\n"


@pytest.fixture(autouse=True)
def no_real_pip(monkeypatch):
    """Nothing here runs the real pip: `pip list` sees no packages, so no package changes are reported."""
    monkeypatch.setattr(update, "pip_list", lambda python=None: {})


class FakePip:
    """Stands in for update._pip_install and records each call."""

    def __init__(self, ok=True, out="installed"):
        self.calls, self.ok, self.out = [], ok, out

    def __call__(self, root, extras, python, forks=None, upgrade=None, reinstall=()):
        self.calls.append({"root": root, "extras": extras, "python": python, "forks": forks, "upgrade": upgrade,
                           "reinstall": list(reinstall)})
        return self.ok, self.out


def git(cwd: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    return cp.stdout.strip()


def commit(repo: Path, files: dict[str, str], message: str, push: bool = True) -> str:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
        git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)
    if push:
        git(repo, "push", "-q", "origin", "HEAD:main")
    return git(repo, "rev-parse", "--short", "HEAD")


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """(origin, upstream clone that pushes, the user's checkout). Git is isolated from the real config
    (no commit signing, no credential helpers) and never looks above tmp_path."""
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Test\n\temail = test@example.com\n[commit]\n\tgpgsign = false\n"
                   "[tag]\n\tgpgsign = false\n[init]\n\tdefaultBranch = main\n[advice]\n\tdetachedHead = false\n",
                   encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    git(origin, "symbolic-ref", "HEAD", "refs/heads/main")
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "clone", "-q", str(origin), str(upstream)], check=True, capture_output=True)
    git(upstream, "switch", "-q", "-c", "main")
    commit(upstream, {"pyproject.toml": PYPROJECT, "README.md": "QuantaBench\n"}, "initial")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    return origin, upstream, work


# ------------------------------------------------------------------------------------------- check()
def test_up_to_date(repos):
    _, _, work = repos
    st = update.check(work)
    assert st.state == "up-to-date" and st.fetched is True
    assert (st.behind, st.ahead, st.branch) == (0, 0, "main")
    assert not st.can_update and st.blocked_reason == ""
    assert "Up to date with origin/main" in st.summary()
    assert update.find_checkout(work).resolve() == work.resolve()


def test_behind_then_fast_forward(repos):
    _, upstream, work = repos
    commit(upstream, {"a.txt": "1\n"}, "first new commit")
    head = commit(upstream, {"b.txt": "2\n"}, "second new commit")
    st = update.check(work)
    assert st.state == "behind" and st.behind == 2 and st.can_update
    assert len(st.incoming) == 2 and "second new commit" in st.incoming[0]
    assert "2 new commit(s) on origin/main" in st.summary()

    lines = []
    res = update.update(work, log=lines.append)
    assert res.ok and res.updated and res.restart_needed
    assert res.new_head == head and res.changed == [] and not res.pip_ran
    assert (work / "b.txt").exists() and lines and "fast-forward only" in lines[0]
    assert update.check(work).state == "up-to-date"


def test_local_changes_refuse_and_are_kept(repos):
    _, upstream, work = repos
    commit(upstream, {"README.md": "QuantaBench, upstream edit\n"}, "edit readme")
    (work / "README.md").write_text("my local edit\n", encoding="utf-8")
    st = update.check(work)
    assert st.state == "behind" and st.dirty == ["README.md"]
    assert not st.can_update and "Local changes" in st.blocked_reason and "README.md" in st.blocked_reason
    before = git(work, "rev-parse", "HEAD")
    res = update.update(work)
    assert not res.ok and not res.updated and "Local changes" in res.message
    assert git(work, "rev-parse", "HEAD") == before
    assert (work / "README.md").read_text(encoding="utf-8") == "my local edit\n"


def test_untracked_files_do_not_block(repos):
    _, upstream, work = repos
    commit(upstream, {"a.txt": "1\n"}, "new")
    (work / "notes.txt").write_text("scratch\n", encoding="utf-8")
    st = update.check(work)
    assert st.can_update and st.dirty == []


def test_ahead_has_nothing_to_pull(repos):
    _, _, work = repos
    commit(work, {"local.txt": "mine\n"}, "local work", push=False)
    st = update.check(work)
    assert st.state == "ahead" and st.ahead == 1 and not st.can_update
    res = update.update(work)
    assert res.ok and not res.updated and not res.restart_needed


def test_diverged_refuses(repos):
    _, upstream, work = repos
    commit(work, {"local.txt": "mine\n"}, "local work", push=False)
    commit(upstream, {"a.txt": "1\n"}, "remote work")
    st = update.check(work)
    assert st.state == "diverged" and (st.ahead, st.behind) == (1, 1)
    assert not st.can_update and "fast-forward is impossible" in st.blocked_reason
    assert not update.update(work).ok


def test_other_branch_refuses(repos):
    _, upstream, work = repos
    git(work, "switch", "-q", "-c", "feature")
    commit(upstream, {"a.txt": "1\n"}, "remote work")
    st = update.check(work)
    assert st.state == "behind" and st.branch == "feature"
    assert not st.can_update and "git switch main" in st.blocked_reason


def test_offline_compares_with_last_fetch(repos):
    _, upstream, work = repos
    commit(upstream, {"a.txt": "1\n"}, "remote work")
    assert update.check(work).behind == 1                     # fetched once
    git(work, "remote", "set-url", "origin", str(work.parent / "gone.git"))
    st = update.check(work)
    assert st.fetched is False and st.behind == 1
    assert "Could not reach the remote" in st.summary()


def test_not_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    st = update.check(plain)
    assert st.state == "not-a-checkout" and not st.can_update
    assert "pip install --upgrade quantabench" in st.summary() and "git clone" in st.summary()
    res = update.update(plain)
    assert not res.ok and not res.updated


# --------------------------------------------------------------------------------- dependencies, pip
def test_dependency_change_reinstalls_with_installed_extras(repos, monkeypatch):
    _, upstream, work = repos
    pip = FakePip()
    monkeypatch.setattr(update, "_pip_install", pip)
    monkeypatch.setattr(update, "installed_extras", lambda text: ["all", "dev"])
    commit(upstream, {"pyproject.toml": PYPROJECT.replace('["numpy>=1.26"]', '["numpy>=1.26", "rich"]')},
           "new dependency")
    res = update.update(work, python="/venv/bin/python")
    assert res.ok and res.pip_ran and res.pip_ok and not res.upgraded and not res.forks_changed
    assert res.changed == ["project.dependencies"]
    assert len(pip.calls) == 1 and Path(pip.calls[0]["root"]).resolve() == work.resolve()
    assert (pip.calls[0]["extras"], pip.calls[0]["python"]) == (["all", "dev"], "/venv/bin/python")
    assert pip.calls[0]["upgrade"] is None and pip.calls[0]["reinstall"] == []
    assert pip.calls[0]["forks"] == update.ForkPackages()          # this repo has no fork-packages.json
    assert "reinstalled" in res.message


def test_pip_failure_goes_back_to_the_old_commit(repos, monkeypatch):
    """The new code needs what pip could not install: the checkout goes back (git reset --keep), the app keeps
    working on the old version, and the message says why (Brandon's other apps do the same)."""
    _, upstream, work = repos
    before = git(work, "rev-parse", "HEAD")
    (work / "notes.txt").write_text("untracked, kept\n", encoding="utf-8")
    monkeypatch.setattr(update, "_pip_install", FakePip(ok=False, out="ERROR: no matching dist"))
    commit(upstream, {"pyproject.toml": PYPROJECT.replace("pytest>=8", "pytest>=9")}, "bump dev")
    res = update.update(work)
    assert not res.ok and not res.pip_ok and res.rolled_back and not res.updated and not res.restart_needed
    assert git(work, "rev-parse", "HEAD") == before and git(work, "status", "--porcelain", "--untracked-files=no") == ""
    assert (work / "notes.txt").exists() and "pytest>=8" in (work / "pyproject.toml").read_text(encoding="utf-8")
    assert "went back to" in res.message and "no matching dist" in res.message
    # the next check offers the same update again
    assert update.check(work).state == "behind"


def test_failed_upgrade_without_a_dependency_change_keeps_the_new_code(repos, monkeypatch):
    _, upstream, work = repos
    commit(upstream, {update.FORK_FILE: forks_json()}, "fork pins")
    git(work, "pull", "-q", "--ff-only")
    monkeypatch.setattr(update, "_pip_install", FakePip(ok=False, out="ERROR: boom"))
    commit(upstream, {"a.txt": "1\n"}, "code only")
    res = update.update(work, upgrade_packages=True)
    assert not res.ok and res.updated and not res.rolled_back and (work / "a.txt").exists()
    assert "package upgrade failed" in res.message and "The code is updated" in res.message


def test_dependency_changes():
    assert update.dependency_changes(PYPROJECT, PYPROJECT.replace('version = "0.0.1"', 'version = "0.0.2"')) == []
    assert update.dependency_changes(PYPROJECT, PYPROJECT.replace("aimnet>=0.2", "aimnet>=0.3")) == \
        ["project.optional-dependencies"]
    assert update.dependency_changes(PYPROJECT, PYPROJECT.replace("quantabench.cli:main", "quantabench.cli:run")) == \
        ["project.scripts"]
    assert update.dependency_changes(PYPROJECT, "not [valid toml") == ["pyproject.toml"]


def test_installed_extras():
    assert update.installed_extras(PYPROJECT, have=lambda name: name == "pytest") == ["dev"]
    # "all" expands quantabench[ml]; "win" has no requirement on this platform, so it is not reported
    assert update.installed_extras(PYPROJECT, have=lambda name: name in ("pytest", "aimnet")) == ["all", "dev", "ml"]
    assert update.installed_extras("broken [") == []


def test_pip_install_pins_the_installed_torch(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, cwd=None, timeout=0):
        seen["cmd"] = cmd
        if "--constraint" in cmd:
            seen["constraint"] = Path(cmd[cmd.index("--constraint") + 1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(update, "_run", fake_run)
    monkeypatch.setattr(update, "_torch_version", lambda: "2.9.1+rocm7.2.1")
    python = str(tmp_path / "venv" / "bin" / "python")
    ok, out = update._pip_install(tmp_path, ["all", "dev"], python)
    assert ok and "pip install" in out
    assert seen["cmd"][:4] == [python, "-m", "pip", "install"]
    assert seen["cmd"][seen["cmd"].index("-e") + 1] == f"{tmp_path}[all,dev]"
    assert seen["constraint"] == "torch==2.9.1+rocm7.2.1\n"
    assert not Path(seen["cmd"][seen["cmd"].index("--constraint") + 1]).exists()      # temp file removed

    monkeypatch.setattr(update, "_torch_version", lambda: None)             # no torch: nothing to pin
    update._pip_install(tmp_path, [], None)
    assert "--constraint" not in seen["cmd"] and seen["cmd"][-1] == str(tmp_path)


def test_windows_uses_pythonw_for_the_gui_and_python_for_pip(tmp_path, monkeypatch):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    for name in ("python.exe", "pythonw.exe"):
        (scripts / name).write_bytes(b"")
    monkeypatch.setattr(update, "IS_WINDOWS", True)
    monkeypatch.setattr(sys, "executable", str(scripts / "python.exe"))
    assert update.gui_command() == [str(scripts / "pythonw.exe"), "-m", "quantabench"]
    assert update._console_python(str(scripts / "pythonw.exe")) == str(scripts / "python.exe")
    monkeypatch.setattr(update, "IS_WINDOWS", False)
    assert update.gui_command()[0] == str(scripts / "python.exe")


# ------------------------------------------------------------------------- fork packages, upgrades
def test_fork_packages_per_platform():
    assert [update.platform_key(p) for p in ("win32", "darwin", "linux", "cygwin")] == \
        ["windows", "macos", "linux", "linux"]
    win = update.fork_packages(forks_json(), "win32")
    assert win.packages == (DXTB, "tad-libcint>=0.4", "tblite") and win.only_binary == ("tblite",)
    assert win.find_links == (("tblite", WIN1),)
    assert win.flags() == ["--only-binary", "tblite", "--find-links", WIN1]
    assert win.pinned() == [DXTB, "tblite"]                     # a git tag and a release page; not tad-libcint
    assert win.link("tblite") == WIN1 and win.link("tad-libcint>=0.4") == ""
    mac = update.fork_packages(forks_json(), "darwin")
    assert mac.flags() == ["--only-binary", "tblite"] and mac.pinned() == [DXTB]
    assert update.fork_packages(forks_json(), "linux").pinned() == []
    assert update.fork_packages("", "win32") == update.ForkPackages()
    assert update.fork_packages('{"linux": {"packages": []}}', "win32") == update.ForkPackages()
    with pytest.raises(ValueError, match="not valid JSON"):
        update.fork_packages("{broken", "win32")
    with pytest.raises(ValueError, match="lists of strings"):
        update.fork_packages('{"windows": {"packages": "tblite"}}', "win32")
    with pytest.raises(ValueError, match="find_links"):
        update.fork_packages('{"windows": {"find_links": ["https://x"]}}', "win32")


def test_fork_changes():
    old = forks_json()
    assert update.fork_changes(old, old, "win32") == (False, [])
    # a new release page: the package is reinstalled even though its version may be the same
    assert update.fork_changes(old, forks_json(windows_link=WIN2), "win32") == (True, ["tblite"])
    assert update.fork_changes(old, forks_json(windows_link=WIN2), "darwin") == (False, [])   # not this platform
    new_dxtb = DXTB.replace("post1", "post2")
    assert update.fork_changes(old, forks_json(dxtb=new_dxtb), "darwin") == (True, [new_dxtb])
    # a version range or a new index package: pip's normal install takes care of it
    assert update.fork_changes(old, forks_json(libcint="tad-libcint>=0.5"), "win32") == (True, [])
    assert update.fork_changes(old, forks_json(macos_extra=("rich",)), "darwin") == (True, [])
    # a checkout from before the file existed used the same pins; an unreadable new file changes nothing
    assert update.fork_changes("", forks_json(windows_link=WIN2), "win32") == (False, [])
    assert update.fork_changes(old, "{broken", "win32") == (False, [])
    assert update.fork_changes("{broken", forks_json(), "win32") == (True, [DXTB, "tblite"])


def test_pip_commands_with_forks_and_upgrade(tmp_path):
    forks = update.fork_packages(forks_json(), "win32")
    c = tmp_path / "c.txt"
    py = str(tmp_path / "python")
    assert update.pip_command(tmp_path, ["all", "dev"], c, py, forks=forks, upgrade="eager") == [
        py, "-m", "pip", "install", "--disable-pip-version-check", "-e", f"{tmp_path}[all,dev]",
        DXTB, "tad-libcint>=0.4", "tblite", "--only-binary", "tblite", "--find-links", WIN1,
        "--upgrade", "--upgrade-strategy", "eager", "--constraint", str(c)]
    assert update.pip_command(tmp_path, [], None, py, forks=forks) == [
        py, "-m", "pip", "install", "--disable-pip-version-check", "-e", str(tmp_path),
        DXTB, "tad-libcint>=0.4", "tblite", "--only-binary", "tblite", "--find-links", WIN1]
    assert update.reinstall_command(["tblite"], c, py, forks=forks) == [
        py, "-m", "pip", "install", "--disable-pip-version-check", "--force-reinstall", "--no-deps", "tblite",
        "--only-binary", "tblite", "--find-links", WIN1, "--constraint", str(c)]


def test_pip_install_reinstalls_after_the_install(tmp_path, monkeypatch):
    cmds = []
    codes = [0, 0]

    def fake_run(cmd, cwd=None, timeout=0):
        cmds.append(cmd)
        if "--constraint" in cmd:
            assert Path(cmd[cmd.index("--constraint") + 1]).read_text(encoding="utf-8") == "torch==2.9.1+cpu\n"
        return subprocess.CompletedProcess(cmd, codes[len(cmds) - 1], "", f"step {len(cmds)}")

    monkeypatch.setattr(update, "_run", fake_run)
    monkeypatch.setattr(update, "_torch_version", lambda: "2.9.1+cpu")
    forks = update.fork_packages(forks_json(), "win32")
    ok, out = update._pip_install(tmp_path, ["all"], None, forks=forks, upgrade="only-if-needed",
                                  reinstall=[DXTB, "tblite"])
    assert ok and len(cmds) == 2 and "step 1" in out and "step 2" in out
    assert cmds[0][cmds[0].index("--upgrade-strategy") + 1] == "only-if-needed"
    assert cmds[1][cmds[1].index("--no-deps") + 1:][:2] == [DXTB, "tblite"]
    cmds.clear()
    codes[:] = [1, 0]                                        # the install fails: nothing is reinstalled
    ok, out = update._pip_install(tmp_path, ["all"], None, forks=forks, reinstall=["tblite"])
    assert not ok and len(cmds) == 1


def test_pip_list_and_package_changes(monkeypatch):
    rows = [{"name": "numpy", "version": "2.1.0"}, {"name": "PySide6_Essentials", "version": "6.8.0"},
            {"name": "quantabench", "version": "0.3.0.dev3", "editable_project_location": "/repo"}]
    monkeypatch.undo()                                       # the real pip_list, with a fake pip behind it
    monkeypatch.setattr(update, "_run", lambda cmd, cwd=None, timeout=0: subprocess.CompletedProcess(
        cmd, 0, json.dumps(rows), "WARNING: Ignoring invalid distribution ~umpy"))
    before = update.pip_list("/venv/bin/python")
    assert before == {"numpy": ("numpy", "2.1.0"), "pyside6-essentials": ("PySide6_Essentials", "6.8.0"),
                      "quantabench": ("quantabench", "0.3.0.dev3")}
    after = dict(before, numpy=("numpy", "2.3.4"), rich=("rich", "14.1.0"))
    del after["pyside6-essentials"]
    assert update.package_changes(before, after) == ["numpy 2.1.0 -> 2.3.4", "PySide6_Essentials 6.8.0 (removed)",
                                                     "rich 14.1.0 (new)"]
    assert update.package_changes(before, before) == []
    monkeypatch.setattr(update, "_run", lambda cmd, cwd=None, timeout=0: subprocess.CompletedProcess(cmd, 1, "", "no"))
    assert update.pip_list() is None


def _with_forks(repos, monkeypatch, platform="windows"):
    """The repos with a fork-packages.json (win1) in the user's checkout, read as ``platform``."""
    _, upstream, work = repos
    commit(upstream, {update.FORK_FILE: forks_json()}, "fork packages")
    git(work, "pull", "-q", "--ff-only")
    monkeypatch.setattr(update, "platform_key", lambda platform_=None, _p=platform: _p)
    pip = FakePip()
    monkeypatch.setattr(update, "_pip_install", pip)
    monkeypatch.setattr(update, "installed_extras", lambda text: ["all", "dev"])
    return upstream, work, pip


def test_new_fork_release_is_reinstalled_on_update(repos, monkeypatch):
    upstream, work, pip = _with_forks(repos, monkeypatch)
    commit(upstream, {update.FORK_FILE: forks_json(windows_link=WIN2)}, "tblite: the OpenBLAS build")
    res = update.update(work)
    assert res.ok and res.updated and res.forks_changed and res.changed == []
    assert len(pip.calls) == 1 and pip.calls[0]["upgrade"] is None
    assert pip.calls[0]["reinstall"] == ["tblite"] and pip.calls[0]["forks"].find_links == (("tblite", WIN2),)
    assert res.reinstalled == ["tblite"] and res.restart_needed
    assert "Reinstalled from scripts/fork-packages.json: tblite." in res.message
    assert "The fork packages changed: reinstalled with pip" in res.message


def test_other_platforms_fork_change_needs_no_pip(repos, monkeypatch):
    upstream, work, pip = _with_forks(repos, monkeypatch, platform="windows")
    commit(upstream, {update.FORK_FILE: forks_json(macos_extra=("rich",))}, "macOS only")
    res = update.update(work)
    assert res.ok and res.updated and not res.forks_changed and not res.pip_ran and pip.calls == []


def test_fork_file_arriving_with_a_pull_needs_no_pip(repos, monkeypatch):
    """The commit that adds fork-packages.json reaches checkouts whose installer used the same pins."""
    _, upstream, work = repos
    pip = FakePip()
    monkeypatch.setattr(update, "_pip_install", pip)
    commit(upstream, {update.FORK_FILE: forks_json()}, "fork packages")
    res = update.update(work)
    assert res.ok and res.updated and not res.forks_changed and pip.calls == []


def test_dependency_change_keeps_the_fork_packages(repos, monkeypatch):
    upstream, work, pip = _with_forks(repos, monkeypatch)
    commit(upstream, {"pyproject.toml": PYPROJECT.replace("pytest>=8", "pytest>=9")}, "bump dev")
    res = update.update(work)
    assert res.ok and res.changed == ["project.optional-dependencies"] and not res.forks_changed
    forks = pip.calls[0]["forks"]
    assert DXTB in forks.packages and forks.find_links == (("tblite", WIN1),) and pip.calls[0]["reinstall"] == []


def test_upgrade_packages_with_nothing_to_pull(repos, monkeypatch):
    _, work, pip = _with_forks(repos, monkeypatch)
    listings = iter([{"numpy": ("numpy", "1.26.4"), "old": ("old", "0.1")},
                     {"numpy": ("numpy", "2.3.4"), "new": ("new", "1.0")}])
    monkeypatch.setattr(update, "pip_list", lambda python=None: next(listings))
    lines = []
    res = update.update(work, upgrade_packages=True, log=lines.append)
    assert res.ok and not res.updated and res.pip_ran and res.upgraded
    assert len(pip.calls) == 1 and pip.calls[0]["upgrade"] == "eager"
    assert pip.calls[0]["reinstall"] == [DXTB, "tblite"]          # fresh builds from the pinned sources
    assert res.package_changes == ["new 1.0 (new)", "numpy 1.26.4 -> 2.3.4", "old 0.1 (removed)"]
    assert res.restart_needed and res.reinstalled == ["dxtb", "tblite"]
    assert res.message.startswith("Nothing to pull from origin/main")
    assert "Upgraded the installed packages with pip (--upgrade-strategy eager" in res.message
    assert "  numpy 1.26.4 -> 2.3.4" in res.message and "Restart QuantaBench" in res.message
    assert any("Upgrading the installed packages" in line for line in lines)


def test_upgrade_packages_on_linux_reinstalls_nothing(repos, monkeypatch):
    _, work, pip = _with_forks(repos, monkeypatch, platform="linux")
    res = update.update(work, upgrade_packages=True, upgrade_strategy="only-if-needed")
    assert res.ok and pip.calls[0]["upgrade"] == "only-if-needed" and pip.calls[0]["reinstall"] == []
    assert pip.calls[0]["forks"].packages == ("tad-libcint<0.4",)
    assert not res.restart_needed and "No package changed version" in res.message
    with pytest.raises(ValueError, match="upgrade_strategy"):
        update.update(work, upgrade_packages=True, upgrade_strategy="all")


def test_upgrade_packages_with_a_pull_runs_pip_once(repos, monkeypatch):
    upstream, work, pip = _with_forks(repos, monkeypatch)
    commit(upstream, {update.FORK_FILE: forks_json(windows_link=WIN2),
                      "pyproject.toml": PYPROJECT.replace("pytest>=8", "pytest>=9")}, "both")
    res = update.update(work, upgrade_packages=True)
    assert res.ok and res.updated and res.upgraded and res.forks_changed and len(pip.calls) == 1
    assert pip.calls[0]["upgrade"] == "eager" and pip.calls[0]["reinstall"] == ["tblite", DXTB]


def test_upgrade_is_refused_when_the_pull_is(repos, monkeypatch):
    upstream, work, pip = _with_forks(repos, monkeypatch)
    commit(upstream, {"README.md": "upstream edit\n"}, "edit readme")
    (work / "README.md").write_text("my local edit\n", encoding="utf-8")
    res = update.update(work, upgrade_packages=True)
    assert not res.ok and "Local changes" in res.message and pip.calls == []


def test_upgrade_failure_on_locked_files(repos, monkeypatch):
    _, work, _ = _with_forks(repos, monkeypatch)
    monkeypatch.setattr(update, "_pip_install", FakePip(ok=False, out=(
        "ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: "
        "'C:\\\\QuantaBench\\\\.venv\\\\Lib\\\\site-packages\\\\PySide6\\\\Qt6Core.dll'")))
    res = update.update(work, upgrade_packages=True)
    assert not res.ok and not res.pip_ok and not res.upgraded and res.reinstalled == []
    assert "The package upgrade failed" in res.message and "Windows keeps the files" in res.message
    assert "WinError 5" in res.message


def test_bad_fork_file_stops_the_upgrade(repos, monkeypatch):
    """Without the fork list, pip --upgrade would swap the dxtb fork for PyPI's dxtb."""
    upstream, work, pip = _with_forks(repos, monkeypatch)
    commit(upstream, {update.FORK_FILE: "{not json"}, "oops")
    res = update.update(work, upgrade_packages=True)
    assert res.ok and res.updated and not res.upgraded and pip.calls == []
    assert "not valid JSON" in res.message and "were not upgraded" in res.message
    commit(upstream, {"pyproject.toml": PYPROJECT.replace("pytest>=8", "pytest>=9")}, "bump dev")
    res = update.update(work)                                # a dependency change still reinstalls
    assert res.ok and len(pip.calls) == 1 and pip.calls[0]["upgrade"] is None
    assert pip.calls[0]["forks"] == update.ForkPackages() and "reinstalling without the fork" in res.message
    (work / update.FORK_FILE).unlink()                       # missing: no upgrade either
    res = update.update(work, upgrade_packages=True)
    assert "is missing, so the packages were not upgraded" in res.message and len(pip.calls) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="creating a symlink needs a privilege on Windows (WinError 1314)")
def test_macos_relinks_tblite_openmp_after_pip(tmp_path, monkeypatch):
    ours = tmp_path / "tblite" / ".dylibs" / "libomp.dylib"
    theirs = tmp_path / "torch" / "lib" / "libomp.dylib"
    for p, text in ((ours, "tblite's"), (theirs, "torch's")):
        p.parent.mkdir(parents=True)
        p.write_text(text, encoding="utf-8")
    monkeypatch.setattr(update, "_run", lambda cmd, cwd=None, timeout=0: subprocess.CompletedProcess(
        cmd, 0, f"{ours}\n{theirs}\n", ""))
    monkeypatch.setattr(update, "IS_MACOS", False)
    assert update._relink_openmp() == ""
    monkeypatch.setattr(update, "IS_MACOS", True)
    assert "Linked tblite's libomp.dylib" in update._relink_openmp()
    assert ours.is_symlink() and ours.resolve() == theirs.resolve()
    assert ours.with_name("libomp.dylib.qb-orig").read_text(encoding="utf-8") == "tblite's"
    monkeypatch.setattr(update, "_run", lambda cmd, cwd=None, timeout=0: subprocess.CompletedProcess(cmd, 0, "\n", ""))
    assert update._relink_openmp() == ""                     # nothing to do


# ---------------------------------------------------------------------------------------------- CLI
def _status(state, **kw):
    return update.UpdateStatus(state, repo=Path("/repo"), branch="main", head="abc1234", fetched=True, **kw)


@pytest.mark.parametrize(("state", "kw", "check_code", "update_code"), [
    ("up-to-date", {}, 0, 0),
    ("ahead", {"ahead": 1}, 0, 0),
    ("behind", {"behind": 2, "dirty": ["src/x.py"]}, 0, 1),
    ("diverged", {"behind": 1, "ahead": 1}, 0, 1),
    ("not-a-checkout", {"message": update.HOW_TO_UPDATE_NO_GIT}, 1, 1),
    ("error", {"message": "boom"}, 1, 1),
])
def test_cli_update_exit_codes(monkeypatch, capsys, state, kw, check_code, update_code):
    monkeypatch.setattr(update, "check", lambda *a, **k: _status(state, **kw))
    monkeypatch.setattr(update, "update", lambda *a, **k: pytest.fail("must not update"))
    assert cli.main(["update", "--check"]) == check_code
    assert cli.main(["update"]) == update_code
    assert "QuantaBench" in capsys.readouterr().out


def test_cli_update_pulls(monkeypatch, capsys):
    st = _status("behind", behind=3, incoming=["abc new thing"])
    monkeypatch.setattr(update, "check", lambda *a, **k: st)
    called = {}

    def fake_update(*a, **k):
        called.update(k)
        return update.UpdateResult(ok=True, updated=True, message="Updated abc -> def. Restart QuantaBench.")

    monkeypatch.setattr(update, "update", fake_update)
    assert cli.main(["update", "--check"]) == 0
    assert "quantabench-cli update" in capsys.readouterr().out and not called
    assert cli.main(["update"]) == 0
    assert called["status"] is st and "Restart QuantaBench" in capsys.readouterr().out


@pytest.mark.parametrize(("state", "kw", "code", "upgrades"), [
    ("up-to-date", {}, 0, True),
    ("ahead", {"ahead": 1}, 0, True),
    ("behind", {"behind": 2}, 0, True),
    ("behind", {"behind": 2, "dirty": ["src/x.py"]}, 1, False),
    ("diverged", {"behind": 1, "ahead": 1}, 1, False),
    ("error", {"message": "boom"}, 1, False),
])
def test_cli_update_upgrade_packages(monkeypatch, capsys, state, kw, code, upgrades):
    st = _status(state, **kw)
    monkeypatch.setattr(update, "check", lambda *a, **k: st)
    called = []

    def fake_update(*a, **k):
        called.append(k)
        return update.UpdateResult(ok=True, upgraded=True, message="Upgraded the installed packages.")

    monkeypatch.setattr(update, "update", fake_update)
    assert cli.main(["update", "--upgrade-packages", "--upgrade-strategy", "only-if-needed"]) == code
    if upgrades:
        assert called == [{"status": st, "log": print, "upgrade_packages": True,
                           "upgrade_strategy": "only-if-needed"}]
        assert "Upgraded the installed packages." in capsys.readouterr().out
    else:
        assert called == []
    called.clear()
    assert cli.main(["update", "--upgrade-packages", "--check"]) == (1 if state == "error" else 0)
    assert called == []                                        # --check never changes anything


def test_cli_upgrade_strategies_match():
    parser = cli.build_parser()
    for strategy in update.UPGRADE_STRATEGIES:
        assert parser.parse_args(["update", "--upgrade-strategy", strategy]).upgrade_strategy == strategy
    assert parser.parse_args(["update"]).upgrade_strategy == "eager"
    with pytest.raises(SystemExit):
        parser.parse_args(["update", "--upgrade-strategy", "all"])


def test_cli_update_on_real_checkout(repos, monkeypatch, capsys):
    _, upstream, work = repos
    commit(upstream, {"a.txt": "1\n"}, "remote work")
    monkeypatch.setattr(update, "find_checkout", lambda start=None, _f=update.find_checkout: _f(start or work))
    assert cli.main(["update", "--check"]) == 0
    assert "1 new commit(s)" in capsys.readouterr().out and not (work / "a.txt").exists()
    assert cli.main(["update"]) == 0
    assert (work / "a.txt").exists() and "Restart QuantaBench" in capsys.readouterr().out


# ------------------------------------------------------------------- CHANGELOG, release, worker, envs (DIST-1)
CHANGELOG_OLD = """# Changelog

## Unreleased (after 0.3.0)

### Old feature (A-1, 2026-11-20)
- did a thing

## 0.3.0 (2026-11-18)

### Released thing (R-1)
- shipped
"""


def test_changelog_news():
    new = CHANGELOG_OLD.replace("## Unreleased (after 0.3.0)\n\n", "## Unreleased (after 0.3.0)\n\n"
                                "### New feature (B-2, 2026-11-23)\n- **new** stuff\n  - nested\n\n")
    new = new.replace("- did a thing", "- did a thing\n- and a follow-up")
    news = update.changelog_news(CHANGELOG_OLD, new)
    assert [(e.heading, e.changed) for e in news] == [("New feature (B-2, 2026-11-23)", False),
                                                      ("Old feature (A-1, 2026-11-20)", True)]
    assert news[0].release == "Unreleased (after 0.3.0)" and news[0].body == "- **new** stuff\n  - nested"
    assert news[1].title() == "Old feature (A-1, 2026-11-20) (updated)" and news[0].markdown().startswith("### New")
    # a release that is cut moves the sections under a new heading: not news
    cut = CHANGELOG_OLD.replace("## Unreleased (after 0.3.0)", "## Unreleased (after 0.4.0)\n\n## 0.4.0 (2026-12-01)")
    assert update.changelog_news(CHANGELOG_OLD, cut) == []
    assert update.changelog_news(CHANGELOG_OLD, CHANGELOG_OLD) == []
    assert [e.heading for e in update.changelog_news("", CHANGELOG_OLD)] == ["Old feature (A-1, 2026-11-20)",
                                                                             "Released thing (R-1)"]


def test_check_reports_changelog_news_and_the_release(repos):
    _, upstream, work = repos
    commit(upstream, {"CHANGELOG.md": CHANGELOG_OLD}, "changelog")
    git(upstream, "tag", "v0.3.0")
    git(upstream, "push", "-q", "origin", "v0.3.0")
    git(work, "pull", "-q", "--ff-only")
    st = update.check(work)
    assert st.state == "up-to-date" and st.news == [] and (st.release, st.since_release) == ("v0.3.0", 0)
    new = CHANGELOG_OLD.replace("## Unreleased (after 0.3.0)\n\n",
                                "## Unreleased (after 0.3.0)\n\n### Update button (DIST-1)\n- pulls main\n\n")
    commit(upstream, {"CHANGELOG.md": new}, "news")
    commit(upstream, {"b.txt": "2\n"}, "more")
    st = update.check(work)
    assert st.state == "behind" and [e.heading for e in st.news] == ["Update button (DIST-1)"]
    assert (st.release, st.since_release) == ("v0.3.0", 2)
    text = st.summary()
    assert "What's new (CHANGELOG.md):\n  - Update button (DIST-1)" in text
    assert "Newest release: v0.3.0; origin/main is 2 commit(s) past it." in text
    assert "### Update button (DIST-1)\n- pulls main" in st.news_markdown()


def test_news_are_capped(repos, monkeypatch):
    _, upstream, work = repos
    monkeypatch.setattr(update, "NEWS_LIMIT", 3)
    commit(upstream, {"CHANGELOG.md": "# Changelog\n\n## Unreleased\n\n" + "".join(
        f"### Item {i}\n- x\n\n" for i in range(5))}, "five")
    st = update.check(work)
    assert len(st.news) == 3 and st.news_more == 2 and "... and 2 more" in st.summary()
    assert "2 more section(s)" in st.news_markdown()


class _Worker:
    def __init__(self, busy, affected=True, manager="launchd"):
        self.busy, self.affected, self.manager, self.pid = busy, affected, manager, 7

    @property
    def can_restart(self):
        return self.affected and not self.busy and self.manager in ("launchd", "task")

    def busy_reason(self):
        return "worker busy with job-1" if self.busy and self.affected else ""

    def describe(self):
        return "The worker on this computer (pid 7) is idle."

    def restart_hint(self):
        return "by hand"


def test_a_busy_worker_holds_the_update(repos, monkeypatch):
    _, upstream, work = repos
    monkeypatch.setattr(update, "probe_worker", lambda root: _Worker(busy=True))
    commit(upstream, {"a.txt": "1\n"}, "remote work")
    st = update.check(work)
    assert st.state == "behind" and not st.can_update and st.blocked_reason == ""
    assert "worker busy with job-1" in st.summary()
    res = update.update(work, status=st)
    assert not res.ok and not res.updated and res.message == "worker busy with job-1"
    assert not (work / "a.txt").exists()
    # pip under a running job is refused too
    monkeypatch.setattr(update, "_pip_install", FakePip())
    git(work, "pull", "-q", "--ff-only")
    res = update.update(work, upgrade_packages=True)
    assert not res.ok and not res.pip_ran and "worker busy" in res.message


def test_an_idle_worker_is_told_to_restart(repos, monkeypatch):
    _, upstream, work = repos
    monkeypatch.setattr(update, "probe_worker", lambda root: _Worker(busy=False))
    commit(upstream, {"a.txt": "1\n"}, "remote work")
    res = update.update(work)
    assert res.ok and res.updated and res.worker_needs_restart
    assert "still runs the old code" in res.message and "--restart-worker" in res.message
    monkeypatch.setattr(update, "probe_worker", lambda root: _Worker(busy=False, manager="manual"))
    commit(upstream, {"b.txt": "1\n"}, "more")
    res = update.update(work)
    assert "by hand" in res.message


def _fake_env(root: Path, name: str) -> Path:
    from quantabench import launchers

    venv = root / name
    py = launchers.Variant(venv).python(windowed=False)
    py.parent.mkdir(parents=True)
    py.write_text("", encoding="utf-8")
    if sys.platform == "win32":
        (py.parent / "pythonw.exe").write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    return venv


def test_dependency_change_reinstalls_the_other_environments(repos, monkeypatch):
    from quantabench import launchers

    _, upstream, work = repos
    main_env, rocm = _fake_env(work, ".venv"), _fake_env(work, ".venv-rocm")
    _fake_env(work, ".venv-other")                             # another checkout's: left alone
    main_py = str(launchers.Variant(main_env).python(windowed=False))
    rocm_py = str(launchers.Variant(rocm).python(windowed=False))
    rows = {rocm_py: [{"name": "quantabench", "version": "0.3.0", "editable_project_location": str(work)},
                      {"name": "torch", "version": "2.9.1+rocm7.2.1"}, {"name": "aimnet", "version": "0.2"}],
            str(launchers.Variant(work / ".venv-other").python(windowed=False)):
                [{"name": "quantabench", "version": "0.3.0", "editable_project_location": "/elsewhere"}]}
    monkeypatch.setattr(update, "_pip_rows", lambda python=None: rows.get(python, []))
    calls = []

    def fake_pip(root, extras, python, forks=None, upgrade=None, reinstall=(), torch=update._RUNNING):
        calls.append((python, extras, torch))
        return True, "ok"

    monkeypatch.setattr(update, "_pip_install", fake_pip)
    monkeypatch.setattr(update, "installed_extras", lambda text, have=None: ["all"] if have is None else
                        [e for e, pkg in (("ml", "aimnet"), ("dev", "pytest")) if have(pkg)])
    commit(upstream, {"pyproject.toml": PYPROJECT.replace('["numpy>=1.26"]', '["numpy>=1.26", "rich"]')}, "dep")
    assert update.other_envs(work, main_py) == [work / ".venv-other", rocm]   # pip tells them apart
    res = update.update(work, python=main_py)
    assert res.ok and res.updated
    assert calls[0] == (main_py, ["all"], update._RUNNING)                 # this environment first
    assert calls[1] == (rocm_py, ["ml"], "2.9.1+rocm7.2.1")                # then .venv-rocm with its own torch
    assert len(calls) == 2 and res.other_envs == [(".venv-rocm", True,
                                                   "reinstalled in .venv-rocm (PyTorch kept at 2.9.1+rocm7.2.1)")]
    assert "other environments use the same code" in res.message


def test_failed_reinstall_in_another_environment_is_reported(repos, monkeypatch):
    from quantabench import launchers

    _, upstream, work = repos
    main_env, rocm = _fake_env(work, ".venv"), _fake_env(work, ".venv-rocm")
    rocm_py = str(launchers.Variant(rocm).python(windowed=False))
    monkeypatch.setattr(update, "_pip_rows", lambda python=None: [{"name": "QuantaBench", "version": "0.3.0"}]
                        if python == rocm_py else [])
    monkeypatch.setattr(update, "_pip_install", lambda root, extras, python, **kw: (
        python != rocm_py, "ERROR: [WinError 32] being used by another process"))
    commit(upstream, {"pyproject.toml": PYPROJECT.replace("pytest>=8", "pytest>=9")}, "bump dev")
    res = update.update(work, python=str(launchers.Variant(main_env).python(windowed=False)))
    assert not res.ok and res.updated and not res.rolled_back               # this environment is fine
    assert res.other_envs[0][:2] == (".venv-rocm", False) and "Close the QuantaBench windows" in res.other_envs[0][2]
    assert "FAILED: pip failed in .venv-rocm" in res.message


def test_cli_restart_worker(monkeypatch, capsys):
    from quantabench import local_worker

    worker = _Worker(busy=False)
    monkeypatch.setattr(update, "check", lambda *a, **k: _status("behind", behind=1, worker=worker))
    monkeypatch.setattr(update, "update", lambda *a, **k: update.UpdateResult(ok=True, updated=True, worker=worker,
                                                                              message="Updated."))
    restarted = []
    monkeypatch.setattr(local_worker, "restart", lambda w: restarted.append(w) or (True, "Restarted the worker."))
    assert cli.main(["update"]) == 0 and not restarted
    assert cli.main(["update", "--restart-worker"]) == 0 and restarted == [worker]
    assert "Restarted the worker." in capsys.readouterr().out


def test_cli_refuses_while_the_worker_is_busy(monkeypatch, capsys):
    monkeypatch.setattr(update, "check", lambda *a, **k: _status("up-to-date", worker=_Worker(busy=True)))
    monkeypatch.setattr(update, "update", lambda *a, **k: pytest.fail("must not update"))
    assert cli.main(["update", "--upgrade-packages"]) == 1
    assert "worker busy" in capsys.readouterr().out
    monkeypatch.setattr(update, "check", lambda *a, **k: _status("behind", behind=1, worker=_Worker(busy=True)))
    assert cli.main(["update"]) == 1 and "worker busy" in capsys.readouterr().out


def test_git_and_pip_never_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.update(kw["env"]) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))
    update._run(["git", "status"])
    assert seen["GIT_TERMINAL_PROMPT"] == "0" and seen["GCM_INTERACTIVE"] == "never" and seen["PIP_NO_INPUT"] == "1"


# ---------------------------------------------------------------------------------------------- GUI
@pytest.fixture
def window(request):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication, QMainWindow

    class Window(QMainWindow):
        refuse_close = False
        worker = None                         # MainWindow.worker: the running build

        def closeEvent(self, event):          # Qt override
            if self.refuse_close:             # "A build is running. Cancel it and quit?" -> No
                event.ignore()
            else:
                super().closeEvent(event)

    app = QApplication.instance() or QApplication([])
    win = Window()
    win.about = QAction("About QuantaBench", win)
    win.menuBar().addMenu("&Help").addAction(win.about)
    win.logged = []
    win.log = win.logged.append
    yield win
    waiter = getattr(win, "_update_wait", None)
    if waiter is not None:
        waiter.timer.stop()
    win.refuse_close = False
    win.close()
    win.deleteLater()
    app.processEvents()


def _wait(cond, timeout=10.0):
    from PySide6.QtWidgets import QApplication

    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert cond()


class FakeWorker:
    """Stands in for local_worker.LocalWorker."""

    def __init__(self, busy=False, manager="launchd", affected=True):
        self.busy, self.manager, self.affected = busy, manager, affected
        self.pid = 4242

    @property
    def can_restart(self):
        return self.affected and not self.busy and self.manager in ("launchd", "task")

    def busy_reason(self):
        busy = self.busy and self.affected               # as LocalWorker.busy_reason: another checkout's is no reason
        return "The worker on this computer runs this checkout and has 1 job(s) running." if busy else ""

    def describe(self):
        return f"The worker on this computer (pid {self.pid}) is {'busy' if self.busy else 'idle'}."

    def restart_hint(self):
        return "restart it by hand"


def test_help_menu_entries_before_about(window):
    from quantabench.gui import update_action

    update_action.install(window)
    help_menu = [a.menu() for a in window.menuBar().actions() if a.menu() is not None][0]
    texts = [a.text() for a in help_menu.actions()]
    assert texts[:2] == ["Check for updates...", "Create app shortcuts..."] and texts[-1] == "About QuantaBench"
    assert window.a_check_updates.isEnabled() and window.a_create_shortcuts.isEnabled()


def test_check_runs_on_a_worker_and_offers_update(window, monkeypatch):
    import threading

    from quantabench.gui import update_action

    update_action.install(window)
    main_thread = threading.get_ident()
    seen = {}

    def fake_check(repo=None):
        seen["thread"] = threading.get_ident()
        return _status("behind", behind=1, incoming=["abc new"])

    monkeypatch.setattr(update, "check", fake_check)
    def fake_ask(win, status):
        seen["status"] = status
        return False                                       # "Later"

    monkeypatch.setattr(update_action, "_ask_update", fake_ask)
    window.a_check_updates.trigger()
    assert not window.a_check_updates.isEnabled()          # one check at a time
    _wait(lambda: "status" in seen)
    assert seen["thread"] != main_thread and seen["status"].can_update
    assert window.a_check_updates.isEnabled()


@pytest.mark.parametrize(("choice", "restarts"), [((True, False), 1), ((False, False), 0)])
def test_update_then_offers_restart(window, monkeypatch, choice, restarts):
    from quantabench.gui import update_action

    update_action.install(window)
    restarted, shown = [], []
    monkeypatch.setattr(update, "update", lambda repo=None: update.UpdateResult(
        ok=True, updated=True, message="Updated abc -> def."))
    monkeypatch.setattr(update_action, "_show_result", lambda win, res: shown.append(res) or choice)
    monkeypatch.setattr(update_action, "restart", lambda win: restarted.append(win))
    update_action.run_update(window)
    _wait(lambda: shown)
    assert "Updated abc -> def." in window.logged and len(restarted) == restarts
    if not restarts:
        assert "restart it" in window.statusBar().currentMessage()


def _click_dialog(monkeypatch, button_text, tick=False, seen=None):
    """UpdateDialog.exec stand-in: optionally ticks "Also upgrade", then presses the button with this text."""
    from quantabench.gui import update_action

    def fake_exec(dlg):
        if seen is not None:
            seen.update(buttons=dlg.button_texts(), check=dlg.check.text() if dlg.check else None,
                        headline=dlg.headline.text(), reason=dlg.reason.text() if dlg.reason.isVisibleTo(dlg) else "",
                        news=dlg.news.toPlainText() if dlg.news is not None else None)
        if tick:
            dlg.check.setChecked(True)
        dlg.click(button_text)
        return 0

    monkeypatch.setattr(update_action.UpdateDialog, "exec", fake_exec)


@pytest.mark.parametrize(("state", "kw", "button", "tick", "choice"), [
    ("behind", {"behind": 1}, "Update", False, "update"),
    ("behind", {"behind": 1}, "Update", True, "upgrade"),
    ("behind", {"behind": 1}, "Later", True, False),
    ("up-to-date", {}, "Upgrade packages...", False, "upgrade"),
    ("up-to-date", {}, "Close", False, False),
    ("error", {"message": "boom"}, "Close", False, False),
    ("diverged", {"behind": 1, "ahead": 1}, "Close", False, False),
])
def test_dialog_choices(window, monkeypatch, state, kw, button, tick, choice):
    from quantabench.gui import update_action

    seen = {}
    _click_dialog(monkeypatch, button, tick=tick, seen=seen)
    assert update_action._ask_update(window, _status(state, **kw)) == choice
    assert (seen["check"] == update_action.UPGRADE_TEXT) == (state == "behind")
    assert ("Upgrade packages..." in seen["buttons"]) == (state in ("up-to-date",))
    assert ("Update" in seen["buttons"]) == (state == "behind")
    if state == "diverged":
        assert "fast-forward is impossible" in seen["reason"] and seen["buttons"] == ["Close"]


def test_dialog_shows_the_changelog_news_and_why_main(window, monkeypatch):
    from quantabench.gui import update_action

    news = [update.ChangelogEntry("Unreleased", "Update button everywhere (DIST-1, 2026-11-23)", "- **Bold** item"),
            update.ChangelogEntry("Unreleased", "Old entry (X-1)", "- longer now", changed=True)]
    st = _status("behind", behind=2, incoming=["abc1234 Update button", "def5678 Fix"], news=news,
                 release="v0.3.0", since_release=40)
    seen = {}
    _click_dialog(monkeypatch, "Later", seen=seen)
    assert update_action._ask_update(window, st) is False
    assert "What's new" in seen["news"] and "Update button everywhere" in seen["news"] and "Bold item" in seen["news"]
    assert "Old entry (X-1) (updated)" in seen["news"] and "Commits (2)" in seen["news"] and "def5678" in seen["news"]
    assert "2 new commit(s) on origin/main" in seen["headline"] and "v0.3.0" in seen["headline"]
    assert "**" not in seen["news"]                                   # rendered, not raw Markdown


def test_busy_worker_offers_update_when_idle(window, monkeypatch):
    from quantabench.gui import update_action

    st = _status("behind", behind=1, worker=FakeWorker(busy=True))
    assert not st.can_update and "1 job(s) running" in st.busy_reason
    seen = {}
    _click_dialog(monkeypatch, "Update when idle", seen=seen)
    assert update_action._ask_update(window, st) == "update"
    assert "1 job(s) running" in seen["reason"]
    # a worker of another checkout does not hold the update
    other = _status("behind", behind=1, worker=FakeWorker(busy=True, affected=False))
    assert other.can_update


def test_running_build_makes_the_update_wait(window, monkeypatch):
    from quantabench.gui import update_action

    monkeypatch.setattr(update_action, "WAIT_MS", 20)
    monkeypatch.setattr(update_action, "_ask_update", lambda win, status: "update")
    asked, ran, probes = [], [], []
    monkeypatch.setattr(update_action, "_ask_wait", lambda win, what: asked.append(what) or True)
    monkeypatch.setattr(update_action, "run_update", lambda win, repo=None, upgrade_packages=False:
                        ran.append(upgrade_packages))
    monkeypatch.setattr(update, "find_checkout", lambda start=None: Path("/repo"))
    monkeypatch.setattr(update, "probe_worker", lambda root: probes.append(root))
    window.worker = object()                                        # a build runs
    update_action.show_status(window, _status("behind", behind=1))
    assert asked == ["a build is running in this window"] and window._update_wait is not None
    _wait(lambda: "Update waiting until a build" in window.statusBar().currentMessage())
    time.sleep(0.1)
    assert not ran                                                   # still running
    window.worker = None                                             # the build ended
    _wait(lambda: ran)
    assert ran == [False] and probes and window._update_wait is None


def test_waiting_update_can_be_cancelled(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from quantabench.gui import update_action

    update_action.install(window)
    monkeypatch.setattr(update_action, "WAIT_MS", 20)
    monkeypatch.setattr(update_action, "_ask_update", lambda win, status: "upgrade")
    monkeypatch.setattr(update_action, "_ask_wait", lambda win, what: True)
    monkeypatch.setattr(update_action, "run_update", lambda *a, **k: pytest.fail("cancelled: must not update"))
    monkeypatch.setattr(update, "check", lambda *a, **k: pytest.fail("a waiting update offers to cancel instead"))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.worker = object()
    update_action.show_status(window, _status("up-to-date"))
    assert window._update_wait is not None
    window.a_check_updates.trigger()
    assert window._update_wait is None and "cancelled" in window.statusBar().currentMessage()


def test_busy_worker_update_waits_for_it(window, monkeypatch):
    from quantabench.gui import update_action

    monkeypatch.setattr(update_action, "WAIT_MS", 10)
    monkeypatch.setattr(update_action, "WORKER_EVERY", 2)
    monkeypatch.setattr(update_action, "_ask_update", lambda win, status: "update")
    monkeypatch.setattr(update_action, "_ask_wait", lambda win, what: pytest.fail("the dialog's Update when idle "
                                                                                  "was the answer"))
    ran = []
    monkeypatch.setattr(update_action, "run_update", lambda win, repo=None, upgrade_packages=False: ran.append(1))
    monkeypatch.setattr(update, "find_checkout", lambda start=None: Path("/repo"))
    answers = [FakeWorker(busy=True), FakeWorker(busy=True), FakeWorker(busy=False)]
    monkeypatch.setattr(update, "probe_worker", lambda root: answers.pop(0) if len(answers) > 1 else answers[0])
    update_action.show_status(window, _status("behind", behind=1, worker=FakeWorker(busy=True)))
    _wait(lambda: ran)
    assert window._update_wait is None


def test_upgrade_with_a_busy_worker_asks_first(window, monkeypatch):
    """With nothing to pull the dialog did not mention the worker: the wait question does."""
    from quantabench.gui import update_action

    monkeypatch.setattr(update_action, "_ask_update", lambda win, status: "upgrade")
    asked = []
    monkeypatch.setattr(update_action, "_ask_wait", lambda win, what: asked.append(what) or False)
    monkeypatch.setattr(update_action, "run_update", lambda *a, **k: pytest.fail("refused: must not update"))
    update_action.show_status(window, _status("up-to-date", worker=FakeWorker(busy=True)))
    assert asked == ["the worker on this computer runs jobs from this checkout"]
    assert getattr(window, "_update_wait", None) is None


def test_result_lists_changes_and_offers_restart(window, monkeypatch):
    from quantabench.gui import update_action

    changes = [f"pkg{i:02d} 1.0 -> 2.0" for i in range(30)]
    res = update.UpdateResult(ok=True, upgraded=True, package_changes=changes, log="$ pip install ...",
                              message="\n".join(["Upgraded the installed packages with pip.", "Package changes (30):",
                                                 *[f"  {c}" for c in changes], "Restart QuantaBench to use the new version."]))
    seen = {}
    _click(monkeypatch, "Restart now", seen=seen)
    assert update_action._show_result(window, res) == (True, False)
    assert "pkg19" in seen["text"] and "pkg20" not in seen["text"] and "... and 10 more" in seen["text"]
    assert "pkg29 1.0 -> 2.0" in seen["details"] and "$ pip install" in seen["details"]
    assert seen["check"] is None                                     # no worker here
    # nothing changed: no restart offered
    _click(monkeypatch, "Close", seen=seen)
    assert update_action._show_result(window, update.UpdateResult(ok=True, message="No change.")) == (False, False)
    assert "Restart now" not in seen["buttons"]


@pytest.mark.parametrize(("worker", "check_shown"), [
    (FakeWorker(), True), (FakeWorker(manager="task"), True), (FakeWorker(manager="manual"), False),
    (FakeWorker(busy=True), False), (FakeWorker(affected=False), False)])
def test_result_offers_to_restart_an_idle_service_worker(window, monkeypatch, worker, check_shown):
    from quantabench.gui import update_action

    res = update.UpdateResult(ok=True, updated=True, worker=worker, message="Updated abc -> def.")
    seen = {}
    _click(monkeypatch, "Later", seen=seen)
    assert update_action._show_result(window, res) == (False, check_shown)
    assert (seen["check"] is not None) == check_shown
    if check_shown:
        assert ("LaunchAgent" if worker.manager == "launchd" else "scheduled task") in seen["check"]


def test_restart_worker_then_the_app(window, monkeypatch):
    from quantabench import local_worker
    from quantabench.gui import update_action

    events = []
    worker = FakeWorker()
    monkeypatch.setattr(update, "update", lambda repo=None: update.UpdateResult(
        ok=True, updated=True, worker=worker, message="Updated abc -> def."))
    monkeypatch.setattr(update_action, "_show_result", lambda win, res: (True, True))
    monkeypatch.setattr(local_worker, "restart", lambda w: events.append(("worker", w)) or (True, "Restarted it."))
    monkeypatch.setattr(update_action, "restart", lambda win: events.append(("app", win)))
    update_action.run_update(window)
    _wait(lambda: len(events) == 2)
    assert events == [("worker", worker), ("app", window)] and "Restarted it." in window.logged


def test_upgrade_choice_reaches_update(window, monkeypatch):
    from quantabench.gui import update_action

    update_action.install(window)
    calls, results = [], []
    monkeypatch.setattr(update_action, "_ask_update", lambda win, status: "upgrade")
    monkeypatch.setattr(update, "update", lambda repo=None, **kw: calls.append(kw) or update.UpdateResult(
        ok=True, upgraded=True, package_changes=["numpy 2.1.0 -> 2.3.4"],
        message="Nothing to pull.\nUpgraded the installed packages with pip.\nPackage changes (1):\n"
                "  numpy 2.1.0 -> 2.3.4\nRestart QuantaBench to use the new version."))
    monkeypatch.setattr(update_action, "_show_result", lambda win, res: results.append(res) or (False, False))
    update_action.show_status(window, _status("up-to-date"))
    _wait(lambda: results)
    assert calls == [{"upgrade_packages": True}]
    assert results[0].restart_needed and "  numpy 2.1.0 -> 2.3.4" in window.logged
    assert "restart it" in window.statusBar().currentMessage()     # "Later": no restart


def _click(monkeypatch, button_text, tick=False, seen=None):
    """QMessageBox.exec stand-in: optionally ticks the box's check box, then clicks the button with this text."""
    from PySide6.QtWidgets import QMessageBox

    def fake_exec(box):
        if seen is not None:
            seen.update(text=box.text(), details=box.detailedText(), buttons=[b.text() for b in box.buttons()],
                        check=box.checkBox().text() if box.checkBox() else None)
        if tick:
            box.checkBox().setChecked(True)
        button = next(b for b in box.buttons() if b.text().replace("&", "") == button_text)
        button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)


def test_create_shortcuts_asks_then_makes_them(window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from quantabench import launchers
    from quantabench.gui import update_action

    made, shown = [], []
    monkeypatch.setattr(launchers, "create", lambda repo, desktop=False, variants=None: made.append(desktop)
                        or ["/apps/quantabench.desktop"])
    monkeypatch.setattr(QMessageBox, "information", lambda *a: shown.append(a[2]))
    seen = {}
    _click(monkeypatch, "Create", tick=True, seen=seen)
    update_action.create_shortcuts(window)
    _wait(lambda: shown)
    assert made == [True] and "/apps/quantabench.desktop" in shown[0]
    assert "without a terminal window" in seen["text"] and seen["check"] == "Also on the Desktop"
    _click(monkeypatch, "Cancel")
    update_action.create_shortcuts(window)
    assert made == [True]


def test_restart_relaunches_once_the_window_closes(window, monkeypatch):
    from quantabench.gui import update_action

    events = []
    monkeypatch.setattr(update, "relaunch", lambda cwd=None: events.append("relaunch"))
    monkeypatch.setattr(update_action, "_quit_app", lambda: events.append("quit"))
    window.show()
    assert update_action.restart(window) is True
    assert events == ["relaunch", "quit"]


def test_restart_is_cancelled_when_the_window_stays_open(window, monkeypatch):
    from quantabench.gui import update_action

    events = []
    monkeypatch.setattr(update, "relaunch", lambda cwd=None: events.append("relaunch"))
    monkeypatch.setattr(update_action, "_quit_app", lambda: events.append("quit"))
    window.refuse_close = True
    window.show()
    assert update_action.restart(window) is False
    assert events == [] and "restart it" in window.statusBar().currentMessage()
