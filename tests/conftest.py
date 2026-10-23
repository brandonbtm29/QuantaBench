import pytest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Never touch the real settings folder (or the GUI's saved window layout) from tests."""
    monkeypatch.setenv("QUANTABENCH_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setenv("QUANTABENCH_WINDOW_STATE", str(tmp_path / "window_state.ini"))
    yield tmp_path


PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
K_PFOS = "[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
FBSA = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)N"
