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


def test_pip_failure_is_reported(repos, monkeypatch):
    _, upstream, work = repos
    monkeypatch.setattr(update, "_pip_install", FakePip(ok=False, out="ERROR: no matching dist"))
    commit(upstream, {"pyproject.toml": PYPROJECT.replace("pytest>=8", "pytest>=9")}, "bump dev")
    res = update.update(work)
    assert not res.ok and res.updated and res.restart_needed and not res.pip_ok
    assert "pip install -e` failed" in res.message and "no matching dist" in res.message


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


# ---------------------------------------------------------------------------------------------- GUI
@pytest.fixture
def window(request):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QApplication, QMainWindow

    class Window(QMainWindow):
        refuse_close = False

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


def test_help_menu_entry_before_about(window):
    from quantabench.gui import update_action

    update_action.install(window)
    help_menu = [a.menu() for a in window.menuBar().actions() if a.menu() is not None][0]
    texts = [a.text() for a in help_menu.actions()]
    assert texts[0] == "Check for updates..." and texts[-1] == "About QuantaBench"
    assert window.a_check_updates.isEnabled()


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


def test_update_and_restart(window, monkeypatch):
    from quantabench.gui import update_action

    update_action.install(window)
    restarted = []
    monkeypatch.setattr(update, "update", lambda repo=None: update.UpdateResult(
        ok=True, updated=True, message="Updated abc -> def."))
    monkeypatch.setattr(update_action, "restart", lambda win: restarted.append(win))
    update_action.run_update(window)
    _wait(lambda: restarted)
    assert restarted == [window] and "Updated abc -> def." in window.logged


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


@pytest.mark.parametrize(("state", "kw", "button", "tick", "choice"), [
    ("behind", {"behind": 1}, "Update and restart", False, "update"),
    ("behind", {"behind": 1}, "Update and restart", True, "upgrade"),
    ("behind", {"behind": 1}, "Later", True, False),
    ("up-to-date", {}, "Upgrade packages...", False, "upgrade"),
    ("up-to-date", {}, "Close", False, False),
    ("error", {"message": "boom"}, "Close", False, False),
])
def test_result_box_choices(window, monkeypatch, state, kw, button, tick, choice):
    from quantabench.gui import update_action

    seen = {}
    _click(monkeypatch, button, tick=tick, seen=seen)
    assert update_action._ask_update(window, _status(state, **kw)) == choice
    assert (seen["check"] == update_action.UPGRADE_TEXT) == (state == "behind")
    assert ("Upgrade packages..." in seen["buttons"]) == (state == "up-to-date")


def test_upgrade_choice_reaches_update(window, monkeypatch):
    from quantabench.gui import update_action

    update_action.install(window)
    calls, results = [], []
    monkeypatch.setattr(update_action, "_ask_update", lambda win, status: "upgrade")
    monkeypatch.setattr(update, "update", lambda repo=None, **kw: calls.append(kw) or update.UpdateResult(
        ok=True, upgraded=True, package_changes=["numpy 2.1.0 -> 2.3.4"],
        message="Nothing to pull.\nUpgraded the installed packages with pip.\nPackage changes (1):\n"
                "  numpy 2.1.0 -> 2.3.4\nRestart QuantaBench to use the new version."))
    monkeypatch.setattr(update_action, "_show_upgrade_result", lambda win, res: results.append(res) or False)
    update_action.show_status(window, _status("up-to-date"))
    _wait(lambda: results)
    assert calls == [{"upgrade_packages": True}]
    assert results[0].restart_needed and "  numpy 2.1.0 -> 2.3.4" in window.logged
    assert "restart it" in window.statusBar().currentMessage()     # "Later": no restart


def test_upgrade_result_lists_changes_and_restarts(window, monkeypatch):
    from quantabench.gui import update_action

    changes = [f"pkg{i:02d} 1.0 -> 2.0" for i in range(30)]
    res = update.UpdateResult(ok=True, upgraded=True, package_changes=changes, log="$ pip install ...",
                              message="\n".join(["Upgraded the installed packages with pip.", "Package changes (30):",
                                                 *[f"  {c}" for c in changes], "Restart QuantaBench to use the new version."]))
    seen = {}
    _click(monkeypatch, "Restart now", seen=seen)
    assert update_action._show_upgrade_result(window, res) is True
    assert "pkg19" in seen["text"] and "pkg20" not in seen["text"] and "... and 10 more" in seen["text"]
    assert "pkg29 1.0 -> 2.0" in seen["details"] and "$ pip install" in seen["details"]
    # nothing changed: no restart offered
    _click(monkeypatch, "Close", seen=seen)
    assert update_action._show_upgrade_result(window, update.UpdateResult(ok=True, message="No change.")) is False
    assert "Restart now" not in seen["buttons"]

    restarted = []
    monkeypatch.setattr(update, "update", lambda repo=None, **kw: res)
    monkeypatch.setattr(update_action, "_show_upgrade_result", lambda win, r: True)
    monkeypatch.setattr(update_action, "restart", lambda win: restarted.append(win))
    update_action.run_update(window, upgrade_packages=True)
    _wait(lambda: restarted)


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
