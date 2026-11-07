"""The methods guide in the GUI: the info text under each pipeline choice, the combo tooltips, the "More…" links
and Help > Methods guide... (search, links between entries)."""

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    import time

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


@pytest.fixture
def window(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    yield w
    w.close()
    assert _wait(app, lambda: w.closed)


def test_info_text_follows_the_choices(window):
    from PySide6.QtCore import Qt

    from quantabench.chem import engines
    from quantabench.chem import method_guide as guide

    w = window
    w._set_combo(w.w_method, "tblite-gfn2")
    text = w.engine_info.text()
    assert text.startswith("<b>GFN2-xTB</b>: ") and '<a href="guide:gfn2-xtb">More…</a>' in text
    assert "This engine: tblite" in text
    w._set_combo(w.w_method, "mopac-pm7")
    assert w.engine_info.text().startswith("<b>PM7</b>") and "This engine: MOPAC" in w.engine_info.text()
    # the solvent line names the model this engine uses; an engine without one says gas phase (and keeps the warning)
    w._set_combo(w.w_solvent, "water")
    assert w.solvent_info.text().startswith("<b>COSMO</b>")
    w._set_combo(w.w_method, "tblite-gfn2")
    assert w.solvent_info.text().startswith("<b>ALPB</b>")
    w._set_combo(w.w_method, "mmff")
    assert w.solvent_info.text().startswith("<b>Gas phase</b>")
    assert "No implicit solvent model for water: this engine runs in the gas phase." in w.engine_info.text()
    w._set_combo(w.w_method, "auto")
    assert w.engine_info.text().startswith("<b>Auto (best available)</b>")
    assert "Uses the best installed engine" in w.engine_info.text()
    assert w.solvent_info.text().startswith("<b>Implicit solvation</b>")
    w._set_combo(w.w_solvent, "")
    assert w.solvent_info.text().startswith("<b>Gas phase</b>")
    # an engine that is not installed keeps its availability line
    missing = [k for k in engines.method_keys()[1:] if not engines.available(k, quick=True)[0]]
    if missing:
        w._set_combo(w.w_method, missing[0])
        assert "Not available: " in w.engine_info.text()

    w._set_combo(w.w_optimizer, "fire")
    assert w.optimizer_info.text().startswith("<b>FIRE (ASE)</b>")
    w._set_combo(w.w_optimizer, "native")
    w._set_combo(w.w_method, "tblite-gfn2")
    assert "This engine has none: L-BFGS." in w.optimizer_info.text()
    w._set_combo(w.w_method, "mopac-pm7")
    assert "This engine has none" not in w.optimizer_info.text()

    w._set_combo(w.w_confsearch, "rdkit")
    assert w.search_info.text().startswith("<b>RDKit ETKDGv3 + MMFF94s</b>")
    assert not w.w_crest_method.isEnabled() and w.crest_info.isEnabled()      # the link works before CREST is on
    w._set_combo(w.w_confsearch, "crest")
    assert w.search_info.text().startswith("<b>CREST iMTD-GC</b>") and w.w_crest_method.isEnabled()
    for key, entry in guide.CREST_LEVEL_ENTRIES.items():
        if w.w_crest_method.findData(key) >= 0:
            w._set_combo(w.w_crest_method, key)
            assert w.crest_info.text().startswith(f"<b>{guide.get(entry).name}</b>")

    w.w_qm_func.setCurrentText("B3LYP-D3(BJ)")             # typed spellings are matched loosely
    assert w.qm_func_info.text().startswith("<b>B3LYP-D3(BJ)</b>")
    w.w_qm_func.setCurrentText("b3lyp")
    assert w.qm_func_info.text().startswith("<b>B3LYP (no dispersion)</b>")
    w.w_qm_func.setCurrentText("blyp-nl")
    assert w.qm_func_info.text().startswith("No guide entry for blyp-nl")
    w.w_qm_basis.setCurrentText("6-31G(d)")
    assert w.qm_basis_info.text().startswith("<b>6-31G*</b>")

    # every item of the guided combos has the summary as its tooltip
    for combo in (w.w_method, w.w_optimizer, w.w_confsearch, w.w_crest_method, w.w_qm_func, w.w_qm_basis):
        for i in range(combo.count()):
            tip = combo.itemData(i, Qt.ItemDataRole.ToolTipRole)
            assert tip and ": " in tip, (combo, combo.itemText(i))
    crest_tip = w.w_confsearch.itemData(w.w_confsearch.findData("crest"), Qt.ItemDataRole.ToolTipRole)
    assert crest_tip.startswith("CREST iMTD-GC: ") and "Linux/WSL/macOS" in crest_tip
    assert "ALPB/GBSA (xtb, tblite, CREST): water" in w.w_solvent.itemData(w.w_solvent.findData("water"),
                                                                            Qt.ItemDataRole.ToolTipRole)
    # the pipeline settings are unchanged by all this
    assert w.current_config().optimizer == "native"


def test_methods_guide_window_opens_searches_and_follows_links(window):
    from PySide6.QtCore import QUrl

    from quantabench.chem import method_guide as guide

    w = window
    help_menu = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "Help")
    texts = [a.text() for a in help_menu.actions() if not a.isSeparator()]
    assert "Methods guide..." in texts and texts[-1] == "About QuantaBench"
    assert w.a_methods_guide.shortcut().toString() == "F1"
    w.a_methods_guide.trigger()
    g = w.methods_guide
    assert g.isVisible() and g.count_label.text() == f"{len(guide.ENTRIES)} entries"
    assert set(g.visible_keys()) == set(guide.ENTRIES)
    assert g.tree.topLevelItemCount() == len(guide.FAMILIES)
    assert "Methods guide" in g.browser.toPlainText() and "ωB97X-D" in g.browser.toPlainText()

    g.search.setText("anion")                              # any words of the text
    keys = g.visible_keys()
    assert "def2-svpd" in keys and "basis-sets" in keys and "lbfgs" not in keys
    assert g.count_label.text() == f"{len(keys)} of {len(guide.ENTRIES)}"
    g.search.setText("wB97X-D")                            # a name: that entry alone, shown at once
    assert g.visible_keys() == ["wb97x-d"] and g.current_key == "wb97x-d"
    assert "Chai, Head-Gordon" in g.browser.toPlainText()
    g.search.setText("no such words here")
    assert g.visible_keys() == [] and "No entry matches" in g.browser.toPlainText()
    g.search.clear()
    assert len(g.visible_keys()) == len(guide.ENTRIES)

    # links between entries (B3LYP -> dispersion) stay in the window
    g.show_entry("b3lyp")
    assert 'href="guide:dispersion"' in guide.html_entry(guide.get("b3lyp"))
    g.browser.anchorClicked.emit(QUrl("guide:dispersion"))
    assert g.current_key == "dispersion" and g.tree.currentItem() is g.items["dispersion"]
    # an entry the filter hides clears the filter
    g.search.setText("anion")
    assert g.items["fire"].isHidden()
    g.browser.anchorClicked.emit(QUrl("guide:fire"))
    assert g.current_key == "fire" and g.search.text() == "" and not g.items["fire"].isHidden()
    g.browser.anchorClicked.emit(QUrl("guide:6-31gs"))      # anchors spell *, + and / as s, p and -
    assert g.current_key == "6-31g*"

    # "More…" under a pipeline choice opens the same window at that entry
    g.close()
    w._set_combo(w.w_method, "xtb-gfn2")
    w.engine_info.linkActivated.emit("guide:gfn2-xtb")
    assert w.methods_guide is g and g.isVisible() and g.current_key == "gfn2-xtb"
    assert "xtb-gfn2" in g.browser.toPlainText()
    w.qm_basis_info.linkActivated.emit("guide:def2-svp")
    assert g.current_key == "def2-svp"
    g.close()
