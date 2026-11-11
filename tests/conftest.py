import os
import sys

import pytest

_EXIT_STATUS = [0]

#: Qt can open windows: always on Windows and macOS (Cocoa; a Mac never sets $DISPLAY, which used to skip every
#: GUI test there, on the CI runner and on the MacBook alike), on Linux with an X display (xvfb-run in CI).
HAS_DISPLAY = os.name == "nt" or sys.platform == "darwin" or bool(os.environ.get("DISPLAY"))


#: Session-wide stand-ins for the user's folders. The per-test ``isolated_settings`` fixture points these at the
#: test's tmp_path through ``monkeypatch``; a test that calls ``monkeypatch.undo()`` (or code that runs after the
#: test, such as a window closed later) then falls back to these, never to the real settings folder that
#: ``settings_location.json`` points at. That fallback once let a test overwrite the shared settings.json
#: (2026-11-07: output_dir set to a pytest folder, the per-machine engine paths dropped).
_SESSION_DIRS = {"QUANTABENCH_SETTINGS_DIR": "settings", "QUANTABENCH_WINDOW_STATE": "window_state.ini",
                 "QUANTABENCH_CACHE_DIR": "cache", "QUANTABENCH_WORKER_STATE": "worker_state"}
_SESSION_ROOT: list[str] = []


def pytest_configure(config):
    """Point the settings, window state and cache at a session folder before any test runs (see _SESSION_DIRS).

    The real-sketcher test (tests/test_gui_sketcher.py, QB_TEST_SKETCHER=1) runs in a child pytest with
    QB_TEST_SKETCHER_CHILD=1: Qt WebEngine accepts the sketcher's URL scheme only before the first QApplication,
    and its Chromium helper processes live as long as the process (the lane tests check that no child processes
    are left). Every other run never loads Qt WebEngine."""
    import tempfile

    root = tempfile.mkdtemp(prefix="quantabench-tests-")
    _SESSION_ROOT.append(root)
    for var, name in _SESSION_DIRS.items():
        os.environ[var] = os.path.join(root, name)
    if os.environ.get("QB_TEST_SKETCHER_CHILD") == "1" and HAS_DISPLAY:
        from quantabench.gui import sketcher_web

        sketcher_web.prepare_before_app()


def pytest_sessionfinish(session, exitstatus):
    _EXIT_STATUS[0] = int(exitstatus)


def pytest_unconfigure(config):
    """AMD's Windows ROCm runtime can hang at interpreter shutdown after GPU tests (BTM NOVA): leave at once
    once pytest has reported, keeping its exit status (``resources.hard_exit_if_needed``)."""
    import shutil

    from quantabench import resources

    for root in _SESSION_ROOT:
        shutil.rmtree(root, ignore_errors=True)
    resources.hard_exit_if_needed(_EXIT_STATUS[0])


@pytest.fixture(autouse=True, scope="module")
def _collect_in_main_thread():
    """Collect each test module's reference cycles here, in the main thread. Left to chance, a cycle that
    holds a Qt window (a panel and its window refer to each other) may be collected by a worker thread of a
    later module (the HTTP worker tests), and Qt aborts the process when a widget is destroyed outside the
    GUI thread. The app itself collects in its GUI thread on a timer (gui/app.py)."""
    yield
    import gc

    gc.collect()


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Never touch the real settings folder (or the GUI's saved window layout) from tests."""
    monkeypatch.setenv("QUANTABENCH_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setenv("QUANTABENCH_WINDOW_STATE", str(tmp_path / "window_state.ini"))
    monkeypatch.setenv("QUANTABENCH_CACHE_DIR", str(tmp_path / "cache"))      # program version cache
    # this machine's worker state (its pause file, DIST-7): a grid's local share and the GUI read it
    monkeypatch.setenv("QUANTABENCH_WORKER_STATE", str(tmp_path / "worker_state"))
    yield tmp_path


@pytest.fixture(autouse=True)
def _keep_the_test_process_priority(monkeypatch):
    """``build``, ``grid build`` and ``worker`` lower their own process (``priority.apply_to_process``); in a test that
    process is pytest's, and a lowered nice value cannot be raised again. Here it only records the level (which
    ``run_program`` then uses for its programs) and forgets it after the test. Lanes are child processes and still
    lower themselves for real; tests/test_priority.py checks the real thing in a child process."""
    from quantabench import priority

    monkeypatch.setattr(priority, "_process_level", None)

    def record_only(level) -> str:
        priority._process_level = priority.lower_of(priority.normalize(level), priority.current_level())
        return f"{priority._process_level} (not applied to the test process)"

    monkeypatch.setattr(priority, "apply_to_process", record_only)
    yield


PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
K_PFOS = "[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
FBSA = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)N"
