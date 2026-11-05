import pytest

_EXIT_STATUS = [0]


def pytest_sessionfinish(session, exitstatus):
    _EXIT_STATUS[0] = int(exitstatus)


def pytest_unconfigure(config):
    """AMD's Windows ROCm runtime can hang at interpreter shutdown after GPU tests (BTM NOVA): leave at once
    once pytest has reported, keeping its exit status (``resources.hard_exit_if_needed``)."""
    from quantabench import resources

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
    yield tmp_path


PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
K_PFOS = "[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
FBSA = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)N"
