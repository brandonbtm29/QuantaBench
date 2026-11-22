"""gui.clipboard: Edit > Paste from each clipboard source and text kind (fake QMimeData), the Import Options on a
paste, Copy as round trips, the molecule list's right-click menu, and drag and drop of each file kind (UI-4)."""

import time

import pytest
from conftest import HAS_DISPLAY, PFOA
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from test_interchange import GENX, _cdx_of

pytest.importorskip("PySide6")
needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")

WIN_CDX = 'application/x-qt-windows-mime;value="ChemDraw Interchange Format"'
WIN_MDLCT = 'application/x-qt-windows-mime;value="MDLCT"'
TFA = "OC(=O)C(F)(F)F"
PFOA_KEY = "SNGREZUHAYWORS-UHFFFAOYSA-N"


def canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _block(smiles: str, name: str = "") -> str:
    m = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(m)
    if name:
        m.SetProp("_Name", name)
    return Chem.MolToMolBlock(m)


def _embedded(smiles: str, seed: int = 7):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def _mime(**formats):
    from PySide6.QtCore import QMimeData

    m = QMimeData()
    text = formats.pop("text", None)
    for fmt, data in formats.items():
        m.setData(fmt, data)
    if text is not None:
        m.setText(text)
    return m


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


def _skip_without_system_clipboard(app):
    """Some sessions have no usable system clipboard (a Windows process started outside the desktop's window
    station reads back nothing); the tests with ``_FakeClipboard`` cover the logic, these need the real one."""
    from PySide6.QtGui import QGuiApplication

    clip = QGuiApplication.clipboard()
    clip.setText("qb-clipboard-probe")
    _wait(app, lambda: clip.text() == "qb-clipboard-probe", timeout=2.0)
    if clip.text() != "qb-clipboard-probe":
        pytest.skip("the system clipboard does not keep what this process puts on it")


class _FakeClipboard:
    """Stands in for the system clipboard (``gui.clipboard._clipboard``): what Copy as puts there, Paste reads."""

    def __init__(self):
        self.mime = None

    def setMimeData(self, mime):  # noqa: N802 - QClipboard's name
        self.mime = mime

    def mimeData(self):  # noqa: N802
        return self.mime

    def text(self) -> str:
        return self.mime.text() if self.mime is not None else ""


@pytest.fixture
def fake_clipboard(monkeypatch):
    from quantabench.gui import clipboard as cb

    clip = _FakeClipboard()
    monkeypatch.setattr(cb, "_clipboard", lambda: clip)
    return clip


@pytest.fixture
def pubchem_keys(monkeypatch):
    """InChIKeys resolve without the network: the key of each structure used here, as PubChem would answer."""
    from quantabench.chem import pubchem

    known = {Chem.MolToInchiKey(Chem.MolFromSmiles(s)): s for s in (PFOA, GENX, TFA, "CCO", "OCCO")}

    def lookup(key, timeout=15.0, client=None, namespace="name"):
        if namespace != "inchikey" or key not in known:
            raise pubchem.PubChemError(f"not found: {key}")
        return {"cid": 1, "smiles": known[key], "title": key}

    monkeypatch.setattr(pubchem, "lookup", lookup)
    return known


def _window(tmp_path, ask: bool = False):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    w.s.import_.ask = ask             # no Import Options dialog unless a test asks for it
    return w


def _smiles_in(w) -> list[str]:
    return [canon(w.model.row(i).smiles) for i in w.model.ids()]


def _log(w) -> str:
    return w.log_view.toPlainText()


def _result(smiles: str, ident: str, n_conf: int = 2):
    """A built MoleculeResult of ``smiles`` with ``n_conf`` conformers."""
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import ConformerResult, MoleculeResult

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(AllChem.EmbedMultipleConfs(mol, n_conf, randomSeed=11))
    confs = [ConformerResult(rank=k, conf_id=cid, energy_ev=-100.0 - k) for k, cid in enumerate(ids)]
    return MoleculeResult(input=from_smiles(smiles, id=ident), status="ok", mol=mol, conformers=confs,
                          method="mmff94")


# ------------------------------------------------------------------ reading mime data (no window)


def test_format_names_and_order():
    from quantabench.gui import clipboard as cb

    assert cb.format_name(WIN_CDX) == "chemdraw interchange format"
    assert cb._kind(WIN_CDX) == "cdx"
    assert cb._kind("com.cambridgesoft.chemdraw.cdx") == "cdx"
    assert cb._kind("COM.CambridgeSoft.ChemDraw.CDXML") == "cdxml"        # case does not matter
    assert cb._kind(WIN_MDLCT) == cb._kind('application/x-qt-windows-mime;value="mdlct"') == "mdlct"
    assert cb._kind("chemical/x-mdl-molfile") == "mol" and cb._kind("text/html") is None
    mime = _mime(**{"chemical/x-mdl-molfile": _block("CCO").encode(), WIN_CDX: _cdx_of(PFOA),
                    "chemical/x-cdxml": b"<CDXML/>", "text": "CCO"})
    assert [k for _, k, _ in cb.candidates(mime)] == ["cdx", "cdxml", "mol", "text"]


@pytest.mark.parametrize("source", ["cdx", "cdx-mac", "cdxml", "mdlct", "molfile", "sdfile", "text-mol",
                                    "text-smiles"])
def test_each_clipboard_source(source):
    from quantabench.chem import interchange
    from quantabench.gui import clipboard as cb

    block = _block(GENX, "GenX")
    mime = {
        "cdx": lambda: _mime(**{WIN_CDX: _cdx_of(GENX)}),
        "cdx-mac": lambda: _mime(**{"com.cambridgesoft.chemdraw.cdx": _cdx_of(GENX)}),
        "cdxml": lambda: _mime(**{"chemical/x-cdxml": interchange.to_cdxml(Chem.MolFromSmiles(GENX)).encode()}),
        "mdlct": lambda: _mime(**{WIN_MDLCT: interchange.mdlct_encode(block)}),
        "molfile": lambda: _mime(**{"chemical/x-mdl-molfile": block.encode()}),
        "sdfile": lambda: _mime(**{"chemical/x-mdl-sdfile": (block + "$$$$\n").encode()}),
        "text-mol": lambda: _mime(text=block),
        "text-smiles": lambda: _mime(text=GENX + " GenX"),
    }[source]()
    label, items, _ = cb.read_candidates(cb.candidates(mime))
    assert [canon(i.smiles) for i in items] == [canon(GENX)]
    assert label.split(" (")[0] == cb.format_name(mime.formats()[0])


def test_a_format_that_fails_falls_back_to_the_next():
    from quantabench.gui import clipboard as cb

    broken = _cdx_of(PFOA)[:60]
    label, items, _ = cb.read_candidates(cb.candidates(_mime(**{WIN_CDX: broken}, text=PFOA)))
    assert label.startswith("text/plain") and [canon(i.smiles) for i in items] == [canon(PFOA)]
    with pytest.raises(ValueError, match="chemdraw interchange format"):
        cb.read_candidates(cb.candidates(_mime(**{WIN_CDX: broken})))


def test_copy_mime_formats():
    from quantabench.chem import interchange
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import clipboard as cb

    one, two = from_smiles(GENX, id="GenX"), from_smiles(PFOA, id="PFOA")
    mime, problems = cb.copy_mime("smiles", [one])
    assert not problems and canon(mime.text()) == canon(GENX) and "chemical/x-daylight-smiles" in mime.formats()
    mime, _ = cb.copy_mime("smiles", [one, two])
    assert mime.text().splitlines()[1].endswith(" PFOA")
    mime, _ = cb.copy_mime("mol", [one], platform="win32")
    assert "M  END" in mime.text() and bytes(mime.data("chemical/x-mdl-molfile").data()).decode() == mime.text()
    assert interchange.mdlct_decode(bytes(mime.data(WIN_MDLCT).data())).splitlines() == mime.text().splitlines()
    mime, _ = cb.copy_mime("mol", [one], platform="linux")
    assert WIN_MDLCT not in mime.formats()
    mime, _ = cb.copy_mime("mol", [one, two])
    assert mime.text().count("$$$$") == 2 and "chemical/x-mdl-sdfile" in mime.formats()
    mime, _ = cb.copy_mime("cdxml", [one])
    assert "<CDXML" in mime.text() and "chemical/x-cdxml" in mime.formats()
    mime, _ = cb.copy_mime("inchi", [one, two])
    assert [ln.split()[1] for ln in mime.text().splitlines()] == ["GenX", "PFOA"]
    assert mime.text().startswith("InChI=1S/") and "chemical/x-inchi" in mime.formats()
    mime, _ = cb.copy_mime("inchikey", [two])
    assert mime.text() == PFOA_KEY
    # SMILES as entered keeps the text; the formats that read the structure leave a bad one out and name it
    entered = from_smiles("C(O)C", id="ethanol")
    assert cb.copy_mime("smiles", [entered])[0].text() == "C(O)C"
    assert cb.copy_mime("canonical", [entered])[0].text() == "CCO"
    bad = from_smiles("CCO", id="bad")
    bad.smiles = "C1CC("
    for kind in ("canonical", "inchi", "mol", "cdxml"):
        mime, problems = cb.copy_mime(kind, [bad, two])
        assert len(problems) == 1 and problems[0].startswith("bad:"), kind
    mime, problems = cb.copy_mime("canonical", [bad])
    assert not mime.formats() and problems


def test_copy_as_xyz_frames():
    from quantabench.chem.inputs import MoleculeInput, from_smiles
    from quantabench.gui import clipboard as cb

    built = _result(TFA, "TFA", n_conf=3)
    shown = ("TFA", built.mol, built.conformers[2].conf_id, cb._xyz_comment("TFA", built.conformers[2], 3, "mmff94"))
    file_3d = MoleculeInput(id="eth", mol=_embedded("CCO"), smiles="CCO")
    text, problems, n = cb.copy_text("xyz", [built, shown, file_3d, from_smiles("OCCO", id="glycol")])
    assert n == 4 and not problems
    comments = [ln for ln in text.splitlines() if ln.split() and ln.split()[0] in ("TFA", "eth", "glycol")]
    assert comments[0].startswith("TFA conformer 1 of 3, E = -100.000000 eV (mmff94)")
    assert comments[1].startswith("TFA conformer 3 of 3")
    assert "input coordinates" in comments[2] and "preview geometry" in comments[3]
    first = text.splitlines()[2:2 + built.mol.GetNumAtoms()]
    pos = built.mol.GetConformer(built.conformers[0].conf_id).GetAtomPosition(0)
    assert first[0].split() == ["O", f"{pos.x:.8f}", f"{pos.y:.8f}", f"{pos.z:.8f}"]


@pytest.mark.parametrize("kind", ["smiles", "canonical", "inchi", "inchikey", "mol", "xyz", "cdxml"])
def test_copy_then_paste_gives_the_same_structure(kind, pubchem_keys):
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import clipboard as cb

    if kind == "xyz":
        items = [_result(GENX, "GenX"), _result(TFA, "TFA")]
    else:
        items = [from_smiles(GENX, id="GenX"), from_smiles(TFA, id="TFA")]
    for subset in (items[:1], items):
        mime, problems = cb.copy_mime(kind, subset)
        assert not problems
        _label, back, _ = cb.read_candidates(cb.candidates(mime))
        got = sorted(canon(s) for b in back for s in b.smiles.split("."))
        assert got == sorted(canon(Chem.MolToSmiles(Chem.MolFromSmiles(i.input.smiles if kind == "xyz" else i.smiles)))
                             for i in subset), kind
        if kind not in ("cdxml", "inchikey") and len(subset) > 1:
            assert [b.id for b in back] == ["GenX", "TFA"], kind


# ------------------------------------------------------------------ the window: paste


@needs_display
def test_edit_menu_paste_and_copy_round_trips(tmp_path, pubchem_keys):
    from PySide6.QtGui import QGuiApplication

    app = _app()
    _skip_without_system_clipboard(app)
    w = _window(tmp_path)
    titles = [a.text().replace("&", "") for a in w.menuBar().actions()]
    assert titles[:3] == ["File", "Edit", "Build"]
    assert w.a_paste.shortcut().toString() == "Ctrl+V"
    clip = QGuiApplication.clipboard()

    clip.setMimeData(_mime(**{WIN_CDX: _cdx_of(GENX)}))     # what ChemDraw puts there on Windows
    w.a_paste.trigger()
    assert _wait(app, lambda: w.model.rowCount() == 1)
    assert _smiles_in(w) == [canon(GENX)]
    assert "chemdraw interchange format (CDX)" in _log(w)
    assert w.model.row(w.model.ids()[0]).input.mol.GetConformer().Is3D()      # Import Options: 2D -> 3D

    w.mol_list.select_id(w.model.ids()[0])
    for kind in ("mol", "cdxml", "smiles", "canonical", "inchi", "inchikey", "xyz"):
        n = w.model.rowCount()
        w.a_copy_as[kind].trigger()
        assert clip.text(), kind
        w.a_paste.trigger()
        assert _wait(app, lambda n=n: w.model.rowCount() == n + 1), kind
        assert _smiles_in(w)[-1] == canon(GENX), kind
        w.mol_list.select_id(w.model.ids()[0])
    _close(app, w)


@needs_display
def test_paste_into_a_text_field_and_the_molecules_box(tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QGuiApplication, QKeyEvent
    from PySide6.QtWidgets import QApplication

    from quantabench.gui import clipboard as cb

    app = _app()
    _skip_without_system_clipboard(app)
    w = _window(tmp_path)
    clip = QGuiApplication.clipboard()

    # Edit > Paste with the focus in a text field pastes the text there
    clip.setText("CCO")
    with pytest.MonkeyPatch.context() as mp:     # not monkeypatch.undo(): that would also undo isolated_settings
        mp.setattr(cb.QApplication, "focusWidget", staticmethod(lambda: w.search))
        w.a_paste.trigger()
        assert w.search.text() == "CCO" and w.model.rowCount() == 0

    def ctrl_v(kind):
        ev = QKeyEvent(kind, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier, "v")
        QApplication.sendEvent(w.entry, ev)
        return ev

    # Ctrl+V in the Molecules box: one line of text goes into the box as usual
    clip.setText(PFOA)
    ctrl_v(QEvent.Type.KeyPress)
    assert w.entry.text() == PFOA and w.model.rowCount() == 0
    w.entry.clear()
    # ... a MOL block (several lines) or a ChemDraw format becomes molecules instead
    clip.setText(_block(GENX, "GenX"))
    assert ctrl_v(QEvent.Type.ShortcutOverride).isAccepted()
    ctrl_v(QEvent.Type.KeyPress)
    assert _wait(app, lambda: w.model.rowCount() == 1)
    assert w.entry.text() == "" and w.model.ids() == ["GenX"]
    _close(app, w)


@needs_display
def test_paste_each_text_kind(tmp_path, pubchem_keys):
    """SMILES lines, InChI, InChIKey, a MOL block, SD text, XYZ, CSV and spreadsheet cells, through the window's
    paste (the same path as Ctrl+V, given the data instead of the system clipboard)."""
    from quantabench.chem import interchange
    from quantabench.gui import clipboard as cb

    app = _app()
    w = _window(tmp_path)
    sd = "".join(_block(s, n) + "$$$$\n" for s, n in ((PFOA, "PFOA"), (TFA, "TFA")))
    kinds = {
        "smiles lines": (f"{PFOA} PFOA\n\n{TFA}\nCCO ethanol\n", [PFOA, TFA, "CCO"], ["PFOA", None, "ethanol"]),
        "InChI": (interchange.to_inchi(Chem.MolFromSmiles(GENX)) + " GenX", [GENX], ["GenX"]),
        "InChIKey": (f"{PFOA_KEY}\n{Chem.MolToInchiKey(Chem.MolFromSmiles(TFA))} TFA", [PFOA, TFA], [PFOA_KEY, "TFA"]),
        "MOL block": (_block(GENX, "GenX"), [GENX], ["GenX"]),
        "SD text": (sd, [PFOA, TFA], ["PFOA", "TFA"]),
        "XYZ": (interchange.to_xyz(_embedded(TFA), comment="TFA from ORCA"), [TFA], ["TFA"]),
        "CSV": (f"name,SMILES\nPFOA,{PFOA}\nTFA,{TFA}\n", [PFOA, TFA], ["PFOA", "TFA"]),
        "spreadsheet cells": (f"DTXSID\tSMILES\nDTXSID8031865\t{PFOA}\n", [PFOA], ["DTXSID8031865"]),
    }
    for what, (text, smiles, ids) in kinds.items():
        w.clear_all()
        k = len(smiles)
        assert cb.paste(w, _mime(text=text)), what
        assert _wait(app, lambda k=k: w.model.rowCount() == k), what
        assert _smiles_in(w) == [canon(s) for s in smiles], what
        assert [i for i, want in zip(w.model.ids(), ids) if want] == [want for want in ids if want], what
    assert "Paste: 1 molecule(s) from text/plain (XYZ text)" in _log(w)
    assert "Paste: 1 molecule(s) from text/plain (table)" in _log(w)
    _close(app, w)


@needs_display
def test_paste_takes_the_import_options_and_posts_fields(tmp_path, monkeypatch):
    from PySide6.QtCore import Qt

    from quantabench.gui import clipboard as cb
    from quantabench.gui import import_options as gui_imp
    from quantabench.gui.molecule_list import COLUMNS

    app = _app()
    w = _window(tmp_path, ask=True)
    shown = []

    def accept(dlg):
        shown.append(dlg.findChildren(gui_imp.QLabel)[0].text())       # what the paste holds
        assert "You can change this later in Tools > Settings > Import" in dlg.where.text()
        dlg.form.post_fields.setChecked(True)
        return 1

    monkeypatch.setattr(gui_imp.ImportOptionsDialog, "exec", accept)
    cells = f"DTXSID\tPREFERRED_NAME\tSMILES\tlogKow\nDTXSID8031865\tPFOA\t{PFOA}\t4.81\nDTXSID0\tEthylamine HCl\tCCN.Cl\t-0.13\n"
    assert cb.paste(w, _mime(text=cells))
    assert _wait(app, lambda: w.model.rowCount() == 2)
    assert len(shown) == 1 and shown[0].startswith("2 structure(s) in the clipboard: 3 data field(s)")
    assert ("Paste: 2 molecule(s) from text/plain (table); Import Options asked; you can change them later in "
            "Tools > Settings > Import") in _log(w)
    rows = {r.id: r for r in w.model.rows()}
    assert rows["DTXSID0"].smiles == "CCN"                                   # the HCl filter, as for a file
    assert w.model.extra_columns() == ["DTXSID", "PREFERRED_NAME", "logKow"]
    col = len(COLUMNS) + 2
    assert w.model.headerData(col, Qt.Orientation.Horizontal) == "logKow"
    assert w.model.data(w.model.index(w.model.position("DTXSID8031865"), col)) == "4.81"
    # a 2D drawing has choices too; Cancel adds nothing
    monkeypatch.setattr(gui_imp.ImportOptionsDialog, "exec", lambda dlg: 0)
    assert cb.paste(w, _mime(text=_block(GENX, "GenX")))
    assert _wait(app, lambda: "cancelled in Import Options" in _log(w))
    assert w.model.rowCount() == 2
    _close(app, w)


@needs_display
def test_paste_with_nothing_readable_says_so(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from quantabench.gui import clipboard as cb

    app = _app()
    w = _window(tmp_path)
    said = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: said.append(a[2])))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: said.append(a[2])))
    assert not cb.paste(w, _mime(**{"image/png": b"\x89PNG"}))
    assert "no structure or text found" in _log(w)
    assert said and "image/png" in said[-1] and "QuantaBench reads" in said[-1]
    for bad in ("C1CC(", "2\nodd electrons\nC 0 0 0\nH 0 0 1.09\n", "id\tsmiles\na\t\n"):
        n = len(said)
        assert cb.paste(w, _mime(text=bad))
        assert _wait(app, lambda n=n: len(said) > n), bad
        assert "could be read as a molecule" in said[-1] and "No molecule could be read" in said[-1], bad
    assert "bond orders" in said[-2]
    assert w.model.rowCount() == 0
    assert not cb.paste(w, _mime())                                         # an empty clipboard
    _close(app, w)


# ------------------------------------------------------------------ the window: copy and the right-click menu


@needs_display
def test_molecule_list_context_menu_and_edit_menu(tmp_path, fake_clipboard):
    from quantabench.chem.inputs import from_smiles

    app = _app()
    w = _window(tmp_path)
    menu = w.mol_list.context_menu()
    assert [a.text() for a in menu.actions()] == ["Copy molecules", "Copy as", "Paste"]
    assert not w.copy_as_menu.isEnabled() and not w.a_copy_molecules.isEnabled()      # nothing to copy yet
    labels = [a.text() for a in w.copy_as_menu.actions()]
    assert labels == ["SMILES", "Canonical SMILES", "InChI", "InChIKey", "MOL / SDF", "XYZ (current conformer)",
                      "CDXML"]
    menu.deleteLater()
    w.add_inputs([from_smiles("C(O)C", id="ethanol"), from_smiles(TFA, id="TFA")])
    w.mol_list.selectAll()
    menu = w.mol_list.context_menu()
    assert w.copy_as_menu.isEnabled()
    menu.deleteLater()
    w.a_copy_as["smiles"].trigger()
    assert fake_clipboard.text() == "C(O)C ethanol\nOC(=O)C(F)(F)F TFA\n"
    w.a_copy_as["canonical"].trigger()
    assert fake_clipboard.text().splitlines()[0] == "CCO ethanol"
    edit = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "Edit")
    assert [a.text() for a in edit.actions() if not a.isSeparator()] == ["Paste", "Copy molecules", "Copy as"]
    _close(app, w)


@needs_display
def test_copy_as_xyz_of_the_shown_conformer(tmp_path, fake_clipboard):
    app = _app()
    w = _window(tmp_path)
    res = _result(TFA, "TFA", n_conf=3)
    w.add_result(res)
    w.mol_list.select_id("TFA")
    assert _wait(app, lambda: w.conf_table.rowCount() == 3)
    w.conf_table.setCurrentCell(1, 0)                                      # the views show conformer 2
    w.a_copy_as["xyz"].trigger()
    lines = fake_clipboard.text().splitlines()
    assert lines[0] == str(res.mol.GetNumAtoms()) and lines[1].startswith("TFA conformer 2 of 3")
    pos = res.mol.GetConformer(res.conformers[1].conf_id).GetAtomPosition(0)
    assert lines[2].split()[1:] == [f"{pos.x:.8f}", f"{pos.y:.8f}", f"{pos.z:.8f}"]
    assert "chemical/x-xyz" in fake_clipboard.mime.formats()
    _close(app, w)


@needs_display
def test_copy_as_of_14000_molecules_stays_fast(tmp_path, fake_clipboard):
    """SCALE-1: SMILES of every row copies at once; the formats that read structures go to a worker thread."""
    from quantabench.chem.inputs import MoleculeInput
    from quantabench.gui import clipboard as cb

    app = _app()
    w = _window(tmp_path)
    n = 14000
    w.add_inputs([MoleculeInput(id=f"m{i}", smiles="C" * (1 + i % 12) + "O", source="test") for i in range(n)])
    w.mol_list.selectAll()
    t = time.perf_counter()
    assert cb.copy_as(w, "smiles")
    smiles_s = time.perf_counter() - t
    assert len(fake_clipboard.text().splitlines()) == n and fake_clipboard.text().startswith("CO m0\n")
    fake_clipboard.mime = None
    t = time.perf_counter()
    assert cb.copy_as(w, "canonical")                    # returns once the worker has started
    started_s = time.perf_counter() - t
    assert fake_clipboard.mime is None and "Copying 14,000 molecules" in w.statusBar().currentMessage()
    assert _wait(app, lambda: fake_clipboard.mime is not None, timeout=120)
    assert len(fake_clipboard.text().splitlines()) == n
    assert smiles_s < 3.0 and started_s < 3.0, (smiles_s, started_s)
    _close(app, w)


# ------------------------------------------------------------------ the window: drag and drop


def _drag(app, target, mime):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import QApplication

    enter = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(target, enter)
    if not enter.isAccepted():
        return False
    QApplication.sendEvent(target, QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime,
                                              Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    return True


def _urls(*paths):
    from PySide6.QtCore import QMimeData, QUrl

    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


@needs_display
def test_drop_files_folders_and_text(tmp_path, monkeypatch):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QDragEnterEvent
    from PySide6.QtWidgets import QApplication, QMessageBox

    from quantabench.gui import import_action

    app = _app()
    w = _window(tmp_path)
    assert w.acceptDrops() and w.mol_list.viewport().acceptDrops()
    opened, imported, folders, projects, said = [], [], [], [], []
    monkeypatch.setattr(w, "open_files", lambda paths: opened.append(paths))
    monkeypatch.setattr(import_action, "import_files", lambda win, paths: imported.append(paths))
    monkeypatch.setattr(w, "open_run_folder", lambda _checked=False, folder=None: folders.append(folder))
    monkeypatch.setattr(w.project, "open_project", lambda _checked=False, path=None: projects.append(path))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: said.append(a[2])))
    smi, out, run = tmp_path / "list.smi", tmp_path / "water.out", tmp_path / "run1"
    smi.write_text("CCO ethanol\n")
    out.write_text("not really ORCA\n")
    run.mkdir()
    (run / "summary.csv").write_text("id,status\n")

    assert _drag(app, w, _urls(smi, out, run))
    assert opened == [[str(smi)]] and imported == [[str(out)]] and folders == [str(run)]

    # a plain folder: the files in it that QuantaBench reads; on the molecule list as on the window
    plain = tmp_path / "set"
    plain.mkdir()
    for name in ("b.sdf", "a.xyz", "notes.docx", "calc.log"):
        (plain / name).write_text("x")
    (plain / "sub").mkdir()
    opened.clear()
    imported.clear()
    assert _drag(app, w.mol_list.viewport(), _urls(plain))
    assert opened == [[str(plain / "a.xyz"), str(plain / "b.sdf")]] and imported == [[str(plain / "calc.log")]]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _drag(app, w, _urls(empty)) and "No file QuantaBench reads in empty" in said[-1]
    from quantabench import project as qproj

    proj = tmp_path / "study.qbproj"
    proj.mkdir()
    (proj / qproj.PROJECT_FILE).write_text(f'{{"format": "{qproj.FORMAT}"}}')
    assert qproj.is_project(proj)
    assert _drag(app, w, _urls(proj)) and projects == [str(proj)]

    assert _drag(app, w.mol_list.viewport(), _mime(text=_block(PFOA, "PFOA")))   # text dragged from an editor
    assert _wait(app, lambda: w.model.rowCount() == 1)
    assert w.model.ids() == ["PFOA"] and "Drop: 1 molecule(s)" in _log(w)

    png = _mime(**{"image/png": b"x"})                      # the event does not keep its mime data alive
    ignored = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, png,
                              Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(w, ignored)
    assert not ignored.isAccepted()
    _close(app, w)


def _fixtures(folder):
    """One small file per format ``inputs.read_file`` reads (written by RDKit or by hand), each with its own id, so
    a drop of all of them at once adds every one: {path: SMILES}."""
    from test_import_formats import _xyz_lines, cdx_bytes, mol2_block

    from quantabench.chem import interchange

    tfa3d = _embedded(TFA)
    eth3d = _embedded("CCO")
    files = {}

    def put(ext, data, smiles):
        p = folder / f"{ext}_sample.{ext}"
        (p.write_bytes if isinstance(data, bytes) else p.write_text)(data)
        files[p] = smiles

    put("sdf", _block(PFOA, "PFOA_sdf") + "$$$$\n", PFOA)
    put("sd", Chem.MolToMolBlock(tfa3d) + "$$$$\n", TFA)
    put("mol", _block(GENX), GENX)
    put("xyz", interchange.to_xyz(tfa3d, comment="TFA"), TFA)
    put("pdb", Chem.MolToPDBBlock(eth3d), "CCO")
    put("mol2", mol2_block(eth3d, "ethanol"), "CCO")
    put("mrv", Chem.MolToMrvBlock(Chem.MolFromSmiles(TFA)), TFA)
    put("cdx", cdx_bytes(TFA), TFA)
    put("cdxml", interchange.to_cdxml(Chem.MolFromSmiles(GENX)), GENX)
    put("cml", Chem.MolToCMLBlock(eth3d), "CCO")
    put("gjf", f"# B3LYP/6-31G(d)\n\nTFA\n\n0 1\n{_xyz_lines(tfa3d)}\n\n", TFA)
    put("com", f"# HF/STO-3G\n\nTFA\n\n0 1\n{_xyz_lines(tfa3d)}\n\n", TFA)
    put("inp", f"! XTB2\n* xyz 0 1\n{_xyz_lines(eth3d)}\n*\n", "CCO")
    put("smi", f"{PFOA} PFOA_smi\n", PFOA)
    put("smiles", f"{TFA} TFA_smiles\n", TFA)
    put("txt", "OCCO glycol\n", "OCCO")
    put("inchi", Chem.MolToInchi(Chem.MolFromSmiles(TFA)) + " TFA_inchi\n", TFA)
    put("csv", f"name,SMILES\nPFOA_csv,{PFOA}\n", PFOA)
    try:
        import openpyxl

        wb = openpyxl.Workbook()
        wb.active.append(["DTXSID", "SMILES"])
        wb.active.append(["DTXSID_xlsx", TFA])
        wb.save(folder / "xlsx_sample.xlsx")
        files[folder / "xlsx_sample.xlsx"] = TFA
    except ImportError:
        pass
    try:
        import ase.io
        from ase import Atoms

        pos = tfa3d.GetConformer().GetPositions()
        cell = Atoms([a.GetSymbol() for a in tfa3d.GetAtoms()], positions=pos - pos.mean(axis=0) + 6.0,
                     cell=[12, 12, 12], pbc=True)
        ase.io.write(folder / "cif_sample.cif", cell, format="cif")
        files[folder / "cif_sample.cif"] = TFA
    except ImportError:
        pass
    return files


@needs_display
def test_drop_each_file_kind(tmp_path, monkeypatch):
    """Every format File > Open reads, dropped one at a time and then all at once (several files, one drop),
    through File > Open's reader and Import Options; an unknown type gets File > Open's message."""
    from PySide6.QtWidgets import QMessageBox

    from quantabench.chem import formats, inputs

    app = _app()
    w = _window(tmp_path / "out")
    folder = tmp_path / "files"
    folder.mkdir()
    files = _fixtures(folder)
    assert {p.suffix for p in files} >= inputs.FILE_TYPES - {".xls", ".xlsm", ".xlsx", ".cif"}
    for k, (path, smiles) in enumerate(files.items()):
        n = w.model.rowCount()
        target = w if k % 2 else w.mol_list.viewport()          # on the window and on the molecule list
        assert _drag(app, target, _urls(path)), path.name
        assert _wait(app, lambda n=n: w.model.rowCount() == n + 1), path.name
        assert canon(w.model.row(w.model.ids()[-1]).smiles) == canon(smiles), path.name
    n = w.model.rowCount()
    assert _drag(app, w, _urls(*files))
    assert _wait(app, lambda: w.model.rowCount() == n + len(files))
    monkeypatch.setattr(formats, "openbabel", lambda: None)
    said = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: said.append(a[2])))
    (tmp_path / "photo.png").write_bytes(b"\x89PNG")
    assert _drag(app, w, _urls(tmp_path / "photo.png"))
    assert _wait(app, lambda: bool(said))
    assert "photo.png" in said[0] and "Unsupported file type .png" in said[0]
    _close(app, w)


# ------------------------------------------------------------------ Copy molecules and the named actions (UI-9 keymap)


def _three(w):
    from quantabench.chem.import_options import FIELDS
    from quantabench.chem.inputs import from_smiles

    items = [from_smiles(GENX, id="GenX"), from_smiles(PFOA, id="PFOA"), from_smiles("C(O)C", id="ethanol")]
    for it, (logkow, note) in zip(items, ((3.4, "one\tstereocenter"), (4.81, "C8"), (-0.31, ""))):
        it.meta[FIELDS] = {"logKow": logkow, "note": note}
    w.add_inputs(items)
    return items


@needs_display
def test_copy_molecules_round_trip(tmp_path, fake_clipboard, monkeypatch):
    from quantabench.gui import clipboard as cb
    from quantabench.gui import import_options as gui_imp
    from quantabench.gui.molecule_list import COLUMNS

    app = _app()
    w = _window(tmp_path)
    items = _three(w)
    w.mol_list.selectAll()
    w.a_copy_molecules.trigger()
    lines = fake_clipboard.text().splitlines()
    assert lines[0].split("\t") == ["ID", "SMILES", "Name", "logKow", "note"]
    assert lines[1].split("\t") == ["GenX", GENX, "GenX", "3.4", "one stereocenter"]     # no tab inside a cell
    assert lines[3].split("\t")[:2] == ["ethanol", "C(O)C"]                          # SMILES as entered
    assert cb.MOLECULES_MIME in fake_clipboard.mime.formats()
    copied = fake_clipboard.mime

    # back into QuantaBench: same ids and structures (the stored options: no fields posted, no dialog)
    w.clear_all()
    assert cb.paste(w, copied)
    assert _wait(app, lambda: w.model.rowCount() == 3)
    assert w.model.ids() == ["GenX", "PFOA", "ethanol"]
    assert _smiles_in(w) == [canon(i.smiles) for i in items]
    assert "QuantaBench molecules" in _log(w) and "Tools > Settings > Import" in _log(w)

    # a spreadsheet's copy (text only) reads as a table too; posting the fields brings the columns back
    w.clear_all()
    w.s.import_.ask = True
    monkeypatch.setattr(gui_imp.ImportOptionsDialog, "exec",
                        lambda dlg: dlg.form.post_fields.setChecked(True) or 1)
    assert cb.paste(w, _mime(text=copied.text()))
    assert _wait(app, lambda: w.model.rowCount() == 3)
    assert w.model.ids() == ["GenX", "PFOA", "ethanol"] and _smiles_in(w) == [canon(i.smiles) for i in items]
    assert w.model.extra_columns() == ["ID", "Name", "logKow", "note"]    # as File > Open posts a table's columns

    # a hidden field column is left out
    w.mol_list.setColumnHidden(len(COLUMNS) + w.model.extra_columns().index("note"), True)
    w.mol_list.selectRow(0)
    w.a_copy_molecules.trigger()
    assert fake_clipboard.text().splitlines()[0].split("\t") == ["ID", "SMILES", "Name", "logKow"]   # no "ID" twice
    _close(app, w)


@needs_display
def test_actions_are_named_with_their_default_keys(tmp_path, fake_clipboard):
    """UI-9's keymap rebinds the actions by objectName; the Molecules box's Ctrl+V follows the paste action."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent, QKeySequence, QShortcut

    from quantabench.gui import clipboard as cb

    app = _app()
    w = _window(tmp_path)
    names = w.actions_by_name
    assert {"edit.paste", "edit.copy_molecules", "edit.copy_as.smiles", "edit.copy_as.xyz"} <= set(names)
    for name, action in names.items():
        assert action.objectName() == name and w.findChild(type(action), name) is action
        assert action.shortcuts() == cb.default_shortcuts(name)
    assert names["edit.paste"].shortcuts() == QKeySequence.keyBindings(QKeySequence.StandardKey.Paste)
    assert [k.toString() for k in names["edit.copy_molecules"].shortcuts()] == ["Ctrl+Shift+C"]
    assert names["edit.copy_molecules"].shortcutContext() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert names["edit.copy_molecules"] in w.mol_dock.actions()
    assert not cb.default_shortcuts("edit.copy_as.smiles")
    assert not [s for s in w.findChildren(QShortcut) if s.key().toString() in ("Ctrl+V", "Ctrl+Shift+C")]

    def ctrl(key, mods=Qt.KeyboardModifier.ControlModifier):     # does the Molecules box hand the paste on?
        return w._clipboard_filter.eventFilter(w.entry, QKeyEvent(QEvent.Type.ShortcutOverride, key, mods))

    fake_clipboard.setMimeData(_mime(text=_block(PFOA, "PFOA")))       # several lines: the box hands them on
    assert ctrl(Qt.Key.Key_V)
    names["edit.paste"].setShortcut(QKeySequence("Ctrl+Alt+V"))           # rebound (Settings > Controls)
    assert not ctrl(Qt.Key.Key_V)
    assert ctrl(Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
    _close(app, w)


@needs_display
def test_ctrl_shift_c_in_the_list_and_in_the_analyze_table(tmp_path, fake_clipboard):
    """Ctrl+Shift+C copies molecules from the Molecules dock; the Analyze table keeps its own Ctrl+Shift+C (copy with
    column titles); Ctrl+C in the list stays Qt's copy of the current cell."""
    import warnings

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    app = _app()
    w = _window(tmp_path)
    w.show()
    w.activateWindow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        QApplication.setActiveWindow(w)
    _wait(app, lambda: QApplication.activeWindow() is w, timeout=3)
    if QApplication.activeWindow() is not w:
        _close(app, w)
        pytest.skip("this session cannot activate a window, so no shortcut reaches it")
    _three(w)
    w.mol_list.selectAll()
    w.mol_list.setFocus()
    app.processEvents()
    both = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    QTest.keyClick(w.mol_list, Qt.Key.Key_C, both)
    assert fake_clipboard.text().startswith("ID\tSMILES\tName\n") or fake_clipboard.text().startswith("ID\tSMILES")
    fake_clipboard.mime = None
    QTest.keyClick(w.mol_list, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert fake_clipboard.mime is None                       # Qt's cell copy: the system clipboard, not ours
    sheets = [v for v in w.findChildren(object) if type(v).__name__ == "SheetView"]
    if sheets:
        called = []
        sheets[0].copy = lambda headers=False: called.append(headers)
        sheets[0].setFocus()
        app.processEvents()
        QTest.keyClick(sheets[0], Qt.Key.Key_C, both)
        assert called == [True] and fake_clipboard.mime is None
    assert QGuiApplication.clipboard() is not None
    _close(app, w)
