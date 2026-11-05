"""quantabench.update, `quantabench-cli update` and Help > Check for updates..., on throwaway git repos
(a bare "origin", a clone that plays the user's checkout, and a second clone that pushes new commits)."""

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


def git(cwd: Path, *args: str) -> str:
    cp = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    return cp.stdout.strip()


def commit(repo: Path, files: dict[str, str], message: str, push: bool = True) -> str:
    for name, text in files.items():
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
    calls = []
    monkeypatch.setattr(update, "_pip_install", lambda root, extras, python: (calls.append((root, extras, python))
                                                                              or (True, "installed")))
    monkeypatch.setattr(update, "installed_extras", lambda text: ["all", "dev"])
    commit(upstream, {"pyproject.toml": PYPROJECT.replace('["numpy>=1.26"]', '["numpy>=1.26", "rich"]')},
           "new dependency")
    res = update.update(work, python="/venv/bin/python")
    assert res.ok and res.pip_ran and res.pip_ok
    assert res.changed == ["project.dependencies"]
    assert len(calls) == 1 and Path(calls[0][0]).resolve() == work.resolve()
    assert calls[0][1:] == (["all", "dev"], "/venv/bin/python")
    assert "reinstalled" in res.message


def test_pip_failure_is_reported(repos, monkeypatch):
    _, upstream, work = repos
    monkeypatch.setattr(update, "_pip_install", lambda root, extras, python: (False, "ERROR: no matching dist"))
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
