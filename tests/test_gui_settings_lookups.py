"""Settings > Lookups (SETTINGS-1): the URL templates and PubChem's cache reach ``settings.lookups``, and the molecule
list's right-click *Look up* menu opens each template with the molecule's identifiers."""

import time

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY,
                                reason="needs a display (run under xvfb-run on Linux)")

PFOA_KEY = "SNGREZUHAYWORS-UHFFFAOYSA-N"


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        time.sleep(0.02)
    return cond()


def test_lookups_tab_round_trip(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.settings_dialog import SettingsDialog

    app = _app()
    d = SettingsDialog(sm.Settings(), None)
    try:
        t = d.lookup_table
        assert t.rowCount() == 4 and t.item(0, 0).text() == "PubChem"
        assert d.lookup_preview.text() == f"PFOA: https://pubchem.ncbi.nlm.nih.gov/#query={PFOA_KEY}"
        t.setCurrentCell(1, 1)
        assert d.lookup_preview.text() == "PFOA: https://webbook.nist.gov/cgi/cbook.cgi?ID=335-67-1&Units=SI"
        assert "%inchikey" in d.lookup_note.text() and "Not science settings" in d.lookup_note.text()
        assert d.pubchem_age.text() == "No limit" and d.pubchem_where.text().startswith("In use now:")

        d._add_lookup()
        r = t.rowCount() - 1
        t.item(r, 0).setText("Wikipedia")
        t.item(r, 1).setText("https://en.wikipedia.org/wiki/%name")
        t.setCurrentCell(r, 1)
        assert d.lookup_preview.text() == "PFOA: https://en.wikipedia.org/wiki/PFOA"
        d._add_lookup()
        t.item(t.rowCount() - 1, 0).setText("Bad")
        t.item(t.rowCount() - 1, 1).setText("file:///etc/passwd")
        assert t.item(t.rowCount() - 1, 1).toolTip().endswith("not saved")
        t.clearSelection()
        t.selectRow(2)                                   # nmrshiftdb2
        d._remove_lookup()
        d.pubchem_cache.setText("~/qb-pubchem")
        d.pubchem_age.setValue(30)
        out = d.result_settings()
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()
    assert [u["name"] for u in out.lookups.urls] == ["PubChem", "NIST WebBook", "EPA CompTox", "Wikipedia"]
    assert (out.lookups.pubchem_cache_dir, out.lookups.pubchem_max_age_days) == ("~/qb-pubchem", 30.0)
    sm.save(out)
    d = SettingsDialog(sm.load(), None)
    try:
        assert d.lookup_table.rowCount() == 4 and d.lookup_table.item(3, 0).text() == "Wikipedia"
        assert d.pubchem_cache.text() == "~/qb-pubchem" and d.pubchem_age.value() == 30
        d._reset_lookups()
        assert len(d.result_settings().lookups.urls) == len(sm.DEFAULT_LOOKUP_URLS)
    finally:
        d.reject()
        d.deleteLater()
        app.processEvents()


def test_molecule_list_look_up_menu(tmp_path, monkeypatch):
    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import lookup_menu
    from quantabench.gui.main_window import MainWindow

    app = _app()
    opened: list[str] = []
    monkeypatch.setattr(lookup_menu, "open_url", lambda url: opened.append(url) or True)
    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))
    try:
        w.show()
        pfoa = from_smiles("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", id="335-67-1", name="PFOA")
        k_pfos = from_smiles("[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", id="kpfos")
        w.add_inputs([pfoa, k_pfos])
        assert _wait(app, lambda: w.model.row("kpfos") is not None)
        from PySide6.QtCore import Qt

        assert w.mol_list.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

        menu = lookup_menu.molecule_menu(w, "335-67-1")
        sub = menu.findChild(type(menu), "lookup_menu")
        acts = {a.text(): a for a in sub.actions() if a.text()}
        assert acts["PubChem"].isEnabled() and acts["PubChem"].toolTip().endswith(PFOA_KEY)
        assert acts["NIST WebBook"].data() == "https://webbook.nist.gov/cgi/cbook.cgi?ID=335-67-1&Units=SI"
        acts["EPA CompTox"].trigger()
        assert opened == [f"https://comptox.epa.gov/dashboard/dsstoxdb/results?search={PFOA_KEY}"]
        assert "Edit lookups..." in acts

        sub = lookup_menu.molecule_menu(w, "kpfos").findChild(type(menu), "lookup_menu")
        acts = {a.text(): a for a in sub.actions() if a.text()}
        assert not acts["NIST WebBook (no CAS RN)"].isEnabled()              # no CAS RN for this one
        values = lookup_menu.molecule_values(w, "kpfos")
        assert values["inchikey"] and values["inchikey"] != PFOA_KEY and "K" in values["inchi"]   # the salt's record

        w.s.lookups.urls = []
        sub = lookup_menu.molecule_menu(w, "kpfos").findChild(type(menu), "lookup_menu")
        assert [a.text() for a in sub.actions() if a.text()][0] == "No lookups (Settings > Lookups)"
    finally:
        w.close()
        assert _wait(app, lambda: w.closed)
