import os

import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (run under xvfb-run on Linux)")


def test_main_window_builds(tmp_path):
    from PySide6.QtWidgets import QApplication

    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    s = sm.Settings(output_dir=str(tmp_path))
    w = MainWindow(s)
    w.add_inputs([from_smiles("CCO", id="ethanol")])
    cfg = w.current_config()
    assert cfg.output_dir == str(tmp_path)
    assert w.mol_list.count() == 1
    w.close()
    app.processEvents()
