"""Spectra panel (gui.spectra_panel) and File > Import calculation output (gui.import_action).

The imported Psi4 water result (tests/data/psi4_water_opt_freq.out) gives real modes, IR intensities and the
frequency geometry without running a calculation; one test also runs GFN1-xTB frequencies through the pipeline.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import HAS_DISPLAY

from quantabench.gui import spectra_panel as sp

DATA = Path(__file__).parent / "data"
PSI4 = DATA / "psi4_water_opt_freq.out"
XTB = DATA / "xtb_water" / "xtb.out"

needs_display = pytest.mark.skipif(not HAS_DISPLAY,
                                   reason="needs a display (run under xvfb-run on Linux)")


# ------------------------------------------------------------------------------------- helpers
def _app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def _pump(app, seconds):
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        time.sleep(0.005)


def _window(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    pytest.importorskip("pyqtgraph")         # the window leaves the spectra dock out without it (optional panel)
    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    w.show()
    w.set_workspace("analyze")              # UI-1: the workspace that shows the Spectra dock (a tab next to Results)
    w.spectra_dock.raise_()
    app.processEvents()
    return app, w


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


@pytest.fixture
def gui(tmp_path):
    """(app, MainWindow) that is always closed afterwards, even when the test fails: a window garbage-collected
    while its hardware-detection thread runs would abort the whole pytest process."""
    app, w = _window(tmp_path)
    yield app, w
    if not w.closed:
        _close(app, w)


def _imported():
    pytest.importorskip("cclib")
    from quantabench.chem import io_cclib

    return io_cclib.to_result(io_cclib.read_output(PSI4), PSI4)


# ------------------------------------------------------------------------------------- no GUI
def test_broaden_matches_vibrations_ir_spectrum():
    from quantabench.chem import vibrations

    freqs, ints = [-120.0, 1600.0, 1650.0, 3700.0], [5.0, 60.0, 20.0, 40.0]
    ref = vibrations.ir_spectrum(freqs, ints, fwhm=25.0, scale=0.96)
    y = sp.broaden(freqs, ints, 25.0, 0.96, ref.wavenumber_cm1)
    assert np.allclose(y / y.max(), ref.intensity)
    iso = sp.broaden([2000.0], [42.0], 10.0, 1.0, np.array([2000.0]))
    assert iso[0] == pytest.approx(42.0)                       # an isolated band peaks at its intensity
    x, yy = sp.sticks_xy(freqs, ints, 1.0)
    assert list(x) == [1600, 1600, 1650, 1650, 3700, 3700] and list(yy) == [0, 60, 0, 20, 0, 40]


def test_mode_sets():
    disp = np.zeros((2, 3, 3)).tolist()
    res = SimpleNamespace(method="gfn1-xtb", qm={"level": "hf/sto-3g", "vibrations": {
        "frequencies_cm1": [1700.0, 3800.0], "displacements": np.zeros((2, 4, 3)).tolist(),
        "positions_A": np.zeros((3, 3)).tolist()}},
        vibrations={"method": "gfn1-xtb", "frequencies_cm1": [1600.0, 3700.0], "ir_intensities_km_mol": [1.0],
                    "displacements": disp, "positions_A": np.zeros((3, 3)).tolist(), "scale_factor": 0.97})
    sets = sp.mode_sets(res)
    assert [s.key for s in sets] == ["computed", "qm"]
    assert sets[0].label == "GFN1-xTB (dxtb)" and sets[0].scale == 0.97
    assert sets[0].ints is None and sets[0].uniform            # wrong length: treated as missing
    assert sets[0].disp.shape == (2, 3, 3) and sets[0].base.shape == (3, 3)
    assert sets[1].label == "QM hf/sto-3g" and sets[1].disp.shape == (2, 4, 3)
    assert sets[1].base is None                                # 3 positions for 4-atom modes: not used
    assert sp.mode_sets(None) == [] and sp.mode_sets(SimpleNamespace(vibrations=None, qm=None)) == []


# ------------------------------------------------------------------------------------- GUI
@needs_display
def test_selecting_a_mode_with_numpy_frequencies(gui):
    """Frequencies are NumPy floats, so ``f > 0`` is a numpy.bool, which Qt's setVisible refuses under NumPy 2
    (TypeError: 'numpy.bool' object cannot be interpreted as an integer; first seen on the MacBook). No cclib
    needed, so this also runs in the core CI install."""
    _app_, w = gui
    panel = w.spectra_panel
    res = SimpleNamespace(method="gfn1-xtb", qm=None, vibrations={
        "method": "gfn1-xtb", "frequencies_cm1": [-50.0, 1600.0, 3700.0], "ir_intensities_km_mol": [1.0, 60.0, 20.0]})
    panel.set_result(res)
    assert panel.selected == 1 and "1600.0 cm⁻¹" in panel.info.text() and "60.0 km/mol" in panel.info.text()
    assert panel.marker.isVisible()
    panel.select_mode(0)                                     # imaginary mode: no marker
    assert "50.0i cm⁻¹" in panel.info.text() and not panel.marker.isVisible()


@needs_display
def test_dock_menus_plot_table_and_overlay(tmp_path, gui):
    from quantabench.chem import spectra_io

    app, w = gui
    dock, panel = w.spectra_dock, w.spectra_panel
    assert dock.objectName() == "spectra" and dock.windowTitle() == "Spectra"
    assert dock in w.tabifiedDockWidgets(w.results_dock) and not dock.isHidden()
    view = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "View")
    texts = [a.text() for a in view.actions()]
    assert texts[texts.index("Results") + 1] == "Spectra"
    file_menu = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "File")
    assert "Import calculation output..." in [a.text() for a in file_menu.actions()]
    assert panel.sets == [] and "built molecule" in panel.status.text()

    res = _imported()
    w.add_result(res)
    app.processEvents()
    assert w.current_result() is res and len(panel.sets) == 1
    ms = panel.active_set
    assert ms.label == "psi4: wB97X-D/def2-SVP" and panel.table.rowCount() == 3
    assert panel.selected == 0 and panel.table.currentRow() == 0      # strongest band: 1645 cm-1, 61 km/mol
    assert "1644.7 cm⁻¹" in panel.info.text() and "61.1 km/mol" in panel.info.text()

    # plot click <-> table <-> marker
    assert panel.select_near(3990.0) == 2 and panel.table.currentRow() == 2
    assert panel.marker.value() == pytest.approx(3995.445)
    panel.table.setCurrentCell(1, 0)
    assert panel.selected == 1 and panel.marker.value() == pytest.approx(3893.7441)
    assert panel.select_near(2500.0) is None and panel.selected == 1  # nothing near: selection kept

    # km/mol vs normalized, FWHM, scale factor
    x, y = panel.curve()
    band = (x > 1500) & (x < 1800)
    # band height = its intensity, at the 1 cm-1 grid point next to 1644.67: 61.0563 / (1 + (0.33/10)^2)
    assert y[band].max() == pytest.approx(61.0563 / (1 + (0.3334 / 10.0) ** 2), rel=1e-4)
    panel.normalize.setChecked(True)
    x, yn = panel.curve()
    assert yn.max() == pytest.approx(1.0)

    def width(xx, yy):                                                 # FWHM of the 1645 cm-1 band
        sel = (xx > 1400) & (xx < 1900)
        return np.count_nonzero(yy[sel] >= 0.5 * yy[sel].max())

    panel.fwhm.setValue(40.0)
    xw, wide = panel.curve()
    assert width(x, yn) == pytest.approx(20, abs=1) and width(xw, wide) == pytest.approx(40, abs=1)
    panel.scale.setValue(0.95)
    x, ys = panel.curve()
    band = (x > 1400) & (x < 1800)
    assert abs(x[band][np.argmax(ys[band])] - 0.95 * 1644.67) <= 1.0
    assert panel.table.item(0, 2).text() == f"{0.95 * 1644.6666:.1f}"
    assert panel.select_near(0.95 * 3893.7) == 1

    # experimental overlay (JCAMP, %T) normalized to the computed maximum; export as shown
    jdx = tmp_path / "exp.jdx"
    jdx.write_text("##TITLE= synthetic water bands\n##XUNITS= 1/CM\n##YUNITS= %T\n##FIRSTX= 4000\n"
                   "##LASTX= 1000\n##NPOINTS= 4\n##XYDATA= (X++(Y..Y))\n4000 90 10 50 100\n##END=\n")
    assert panel.load_experimental(jdx) and panel.clear_btn.isEnabled()
    ex, ey = panel.experimental_curve()
    assert ey.max() == pytest.approx(1.0)                              # normalized mode: computed max is 1
    panel.normalize.setChecked(False)
    _, ey = panel.experimental_curve()
    cx, cy = panel.curve()
    inside = (cx >= 400) & (cx <= 4000)
    assert ey.max() == pytest.approx(cy[inside].max())
    legend = [label.text for _sample, label in panel.legend.items]
    assert any("Experimental: synthetic water bands" in t for t in legend)
    out = panel.export_csv(tmp_path / "ir.csv")
    lines = out.read_text().splitlines()
    assert lines[0] == "wavenumber_cm-1,intensity_km_mol" and len(lines) == len(cx) + 1
    back = spectra_io.read_spectrum(out)
    assert np.allclose(back.y, np.interp(back.x, cx, cy), atol=1e-3 * cy.max())
    panel.clear_experimental()
    assert panel.experimental is None and panel.experimental_curve() is None


@needs_display
def test_mode_animation(gui):
    from ase.build import molecule

    from quantabench.chem.inputs import from_smiles

    app, w = gui
    panel = w.spectra_panel
    res = _imported()
    w.add_result(res)
    app.processEvents()
    viewer = w.viewer
    if viewer.plotter is None:
        pytest.skip(viewer.error)
    calls = []
    viewer.set_interactive = calls.append              # the optional viewer hook is used when present
    before = viewer.atoms.positions.copy()
    shown = viewer.atoms
    panel.select_mode(0)
    panel.amp.setValue(30)
    assert panel.play() and panel.playing and not panel.play_btn.isEnabled() and panel.stop_btn.isEnabled()
    _pump(app, 0.5)
    assert len(panel.frame_times_ms) >= 2 and viewer.atoms is shown
    disp = np.asarray(res.vibrations["displacements"][0])
    base = np.asarray(res.vibrations["positions_A"])
    delta = viewer.atoms.positions - base
    scale = np.sqrt((disp**2).sum(1)).max()
    a = (delta * disp).sum() / (disp**2).sum()                          # delta = a d: along the mode
    assert np.allclose(delta, a * disp, atol=1e-9) and abs(a) * scale <= 0.30 + 1e-9
    panel.stop()
    assert not panel.playing and np.array_equal(viewer.atoms.positions, before)
    assert calls == [True, False]

    # selecting another mode while playing restarts on that mode
    panel.play()
    panel.select_mode(2)
    assert panel.playing and panel.selected == 2
    _pump(app, 0.2)
    # the molecule changes: the animation stops (the new structure is left alone)
    w.add_inputs([from_smiles("C", id="methane")])
    app.processEvents()
    assert not panel.playing and len(viewer.atoms) == 5 and panel.sets == []
    # back to water: closing the dock while playing stops and restores
    w.mol_list.setCurrentRow(0)
    app.processEvents()
    assert panel.active_set is not None
    before = viewer.atoms.positions.copy()
    assert panel.play()
    _pump(app, 0.2)
    w.spectra_dock.close()
    app.processEvents()
    assert not panel.playing and np.array_equal(viewer.atoms.positions, before)
    w.spectra_dock.show()
    # another structure with a different atom count in the 3D view: refused with a message
    viewer.show_atoms(molecule("CH4"))
    assert not panel.play() and "cannot animate" in panel.info.text()
    # closing the window while playing stops the timer before the 3D view is torn down
    w.mol_list.setCurrentRow(1)
    w.mol_list.setCurrentRow(0)
    app.processEvents()
    assert panel.play()
    _close(app, w)
    assert not panel.playing


@needs_display
def test_pipeline_frequencies_and_qm_trace(tmp_path):
    pytest.importorskip("dxtb")
    pytest.importorskip("cclib")
    from quantabench.chem import io_cclib, pipeline
    from quantabench.chem.inputs import from_smiles

    cfg = pipeline.BuildConfig(method="gfn1-xtb", frequencies=True, max_conformers=1, refine_top=1,
                               write_files=False, xtb_properties=False)
    res = pipeline.run_molecule(from_smiles("O", id="water"), cfg)
    assert res.status == "ok" and res.vibrations and "displacements" in res.vibrations
    res.qm = {"level": "wB97X-D/def2-SVP", "vibrations": io_cclib.to_vibrations_dict(io_cclib.read_output(PSI4))}
    app, w = _window(tmp_path)
    try:
        _check_two_traces(app, w, res, tmp_path)
    finally:
        _close(app, w)


def _check_two_traces(app, w, res, tmp_path):
    panel = w.spectra_panel
    w.add_result(res)                                     # gfn1-xtb is an engine key: no label fallback needed
    app.processEvents()
    assert [s.key for s in panel.sets] == ["computed", "qm"] and panel.set_box.count() == 2
    legend = [label.text for _sample, label in panel.legend.items]
    assert legend[0].startswith("GFN1-xTB") and legend[1].startswith("QM wB97X-D/def2-SVP")
    computed = panel.sets[0]
    assert computed.base is not None and np.allclose(computed.base, res.vibrations["positions_A"])
    panel.select_set(1)
    assert panel.active == 1 and panel.table.item(0, 1).text() == "1644.7"
    panel.scale.setValue(0.95)
    assert panel.sets[1].scale == 0.95
    panel.select_set(0)
    assert panel.scale.value() == pytest.approx(computed.vib["scale_factor"])
    out = panel.export_csv(tmp_path / "both.csv")
    # without IR intensities (native Windows: no dxtb libcint) every mode has the same weight: normalized
    unit = "km_mol" if res.vibrations.get("ir_intensities_km_mol") else "normalized"
    assert out.read_text().splitlines()[0] == f"wavenumber_cm-1,intensity_{unit},qm_intensity_{unit}"
    if w.viewer.plotter is not None:
        shown = w.viewer.atoms.positions.copy()
        assert panel.play()                              # first frame: sin(0) = 0 -> the frequency geometry
        assert np.allclose(w.viewer.atoms.positions, computed.base, atol=0.05)
        panel.stop()
        assert np.array_equal(w.viewer.atoms.positions, shown)


@needs_display
def test_import_action(tmp_path, monkeypatch, gui):
    pytest.importorskip("cclib")
    from PySide6.QtWidgets import QMessageBox

    from quantabench.gui import import_action

    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a[2])))
    app, w = gui
    bad = tmp_path / "bad.out"
    bad.write_text("not an output file\n")
    import_action.import_files(w, [str(PSI4), str(XTB), str(bad)])
    assert _wait(app, lambda: len(w.results) == 2 and warnings)
    assert set(w.results) == {"psi4_water_opt_freq", "xtb"}
    assert "bad.out" in warnings[0]
    res = w.current_result()
    assert res.input.id == "xtb" and res.method == "xtb: GFN2-xTB"
    assert w.spectra_panel.active_set.disp is not None           # normal modes from xtb's g98.out
    assert w.mol_list.count() == 2 and "xtb: GFN2-xTB" in w.mol_list.row_text(1)
    log = w.log_view.toPlainText()
    assert "Imported psi4_water_opt_freq.out" in log


@needs_display
def test_add_imported_result_shows_its_method(gui):
    _app_, w = gui
    w.add_result(_imported())
    assert w.spectra_panel.sets
