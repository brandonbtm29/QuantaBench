"""Surfaces panel: dock, View toggle, charge-ESP and QM-cube surfaces in the 3D view, molecule/conformer changes."""

import numpy as np
import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY,
                                reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    import time

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


@pytest.fixture(scope="module")
def built():
    """Pentanoic acid (several MMFF conformers) and TFA, built without xTB; conformer 0 gets charges."""
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule
    from quantabench.gui.surface_panel import gasteiger_charges

    cfg = BuildConfig(method="mmff", max_conformers=12, refine_top=4, rmsd_threshold=0.1, write_files=False,
                      xtb_properties=False, protonate=False)
    acid = run_molecule(from_smiles("CCCCC(=O)O", id="pentanoic"), cfg)
    tfa = run_molecule(from_smiles("OC(=O)C(F)(F)F", id="TFA"), cfg)
    assert acid.status == tfa.status == "ok"
    assert len(acid.conformers) >= 2
    for res in (acid, tfa):
        for c in res.conformers:
            c.charges = None
        res.conformers[0].charges = [float(x) for x in gasteiger_charges(res.mol)]
    return acid, tfa


def _window(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    if w.viewer.plotter is None:
        pytest.skip(w.viewer.error)
    w.resize(1200, 800)
    w.show()
    w.set_workspace("analyze")              # UI-1: the workspace that shows the Surfaces dock
    return w


def _actors(w):
    return w.viewer.plotter.renderer.actors


def test_panel_installs_and_draws_charge_surface(tmp_path, built):
    from PySide6.QtWidgets import QDockWidget

    from quantabench.gui import surface_panel as sp

    acid, tfa = built
    app = _app()
    w = _window(tmp_path)
    panel = w.surface_panel
    dock = w.findChild(QDockWidget, "surfaces")
    assert dock is w.surface_dock and dock.windowTitle() == "Surfaces"
    assert dock in w.tabifiedDockWidgets(w.appearance_dock) and not dock.isHidden()      # Analyze's right tabs
    view = sp._find_menu(w, "View")
    assert dock.toggleViewAction() in view.actions()

    w.add_result(acid)
    app.processEvents()
    assert not panel.actor_shown()                       # Show is off by default
    assert not panel.source_box.model().item(0).isEnabled()   # no QM cube for this molecule
    assert panel.source() == "vdw"
    panel.show_box.setChecked(True)
    assert _wait(app, panel.actor_shown)
    actor = _actors(w)[sp.ACTOR]
    assert not actor.GetPickable()
    assert "conformer 1 charges" in panel.message.text()
    np.testing.assert_allclose(panel.current.atom_positions, w.viewer.atoms.positions)
    d = panel.current.descriptors
    assert d["esp_min_kcal"] < 0 < d["esp_max_kcal"] and "Area" in panel.readout.text()

    # coloring changes only restyle the actor
    data = panel.current
    assert panel.range_value.suffix() == " kJ/mol" and panel.range_value.stored() == 30.0   # 30 kcal/mol inside
    panel.range_box.setCurrentIndex(panel.range_box.findData("fixed"))
    panel.range_value.setValue(12.5)                                      # kJ/mol (Settings > Display)
    panel.cmap_box.setCurrentIndex(panel.cmap_box.findData("coolwarm"))
    panel.opacity_box.setValue(0.5)
    app.processEvents()
    assert panel.current is data
    mapper = _actors(w)[sp.ACTOR].GetMapper()
    kcal = 12.5 / 4.184                                                   # the mesh's ESP is in kcal/mol
    assert mapper.GetLookupTable().GetRange() == pytest.approx((-kcal, kcal))
    assert _actors(w)[sp.ACTOR].GetProperty().GetOpacity() == pytest.approx(0.5)
    assert "+12.5" in panel.legend_hi.text() and "red = negative" in panel.legend_text.text()   # default
    assert panel.legend_text.text().startswith("ESP, kJ/mol:")
    assert "kJ/mol" in panel.readout.text() and "(kJ/mol)²" in panel.readout.text()
    vmin = f"{d['esp_min_kcal'] * 4.184:.2f}"
    assert f"<td align='right'>{vmin}</td><td>kJ/mol</td>" in panel.readout.text()
    panel.reverse_box.setChecked(False)
    assert "blue = negative" in panel.legend_text.text()
    panel.reverse_box.setChecked(True)
    # other units: the same colours and range, shown in kcal/mol
    w.s.display.relative_energy_unit = "kcal/mol"
    w.apply_display()
    assert panel.range_value.suffix() == " kcal/mol" and panel.range_value.value() == pytest.approx(2.99)
    assert mapper.GetLookupTable().GetRange() == pytest.approx((-kcal, kcal))
    assert panel.color_range() == pytest.approx(kcal)
    assert panel.legend_hi.text() == "+3.0" and panel.legend_text.text().startswith("ESP, kcal/mol:")
    assert f"<td align='right'>{d['esp_min_kcal']:.2f}</td><td>kcal/mol</td>" in panel.readout.text()
    assert "(kcal/mol)²" in panel.readout.text() and panel.current is data
    w.s.display.relative_energy_unit = "kJ/mol"
    w.apply_display()
    assert panel.range_value.value() == pytest.approx(12.5) and panel.color_range() == pytest.approx(kcal)

    # atom picking goes through the surface (the viewer ray-casts its own spheres)
    disp = w.viewer.world_to_display(w.viewer.atoms.positions)
    front = int(np.argmin(disp[:, 2]))
    hit = w.viewer.pick_atom(disp[front, 0], disp[front, 1])
    assert hit is not None
    w.viewer._click(disp[front, 0], disp[front, 1], ctrl=False)
    assert w.selection.atoms == [hit] and panel.actor_shown()

    # a style change rebuilds the scene but keeps the (still valid) surface
    w.viewer.set_style("Space filling")
    app.processEvents()
    assert panel.actor_shown() and panel.current is data

    # another conformer: the stale surface goes, a new one for the new geometry comes
    w.conf_table.setCurrentCell(1, 0)
    assert _wait(app, lambda: panel.actor_shown() and panel.current is not data)
    np.testing.assert_allclose(panel.current.atom_positions, w.viewer.atoms.positions)
    assert "best conformer's charges" in panel.message.text()

    # another molecule
    w.add_result(tfa)
    assert _wait(app, lambda: panel.actor_shown() and len(panel.current.atom_positions) == tfa.mol.GetNumAtoms())

    # an unbuilt molecule: nothing to draw
    from quantabench.chem.inputs import from_smiles

    w.add_inputs([from_smiles("CCO", id="ethanol")])
    assert _wait(app, lambda: not panel.actor_shown() and not panel.busy)
    assert "built molecule" in panel.message.text()

    # back to TFA from the cache, then Show off removes it
    w.mol_list.setCurrentRow(1)
    assert _wait(app, panel.actor_shown)
    panel.show_box.setChecked(False)
    assert _wait(app, lambda: not panel.actor_shown())

    # the worker-thread path (large molecules) gives the same surface
    panel.sync_atom_limit = 0
    panel._cache.clear()
    panel.source_box.setCurrentIndex(panel.source_box.findData("sas"))
    panel.show_box.setChecked(True)
    assert _wait(app, lambda: panel.actor_shown() and not panel.busy)
    assert panel.current.descriptors["surf_area_A2"] > 150
    _close(app, w)


def _rotation(axis, deg):
    from quantabench.gui.viewer3d import rotation_matrix

    return rotation_matrix(axis, np.radians(deg))


def _fake_qm_cubes(res, folder, rot, shift):
    """Density: sum of atomic exponentials; ESP: the conformer charges, on a modest grid around conformer 0,
    with the molecule moved by ``rot``/``shift`` (a QM program that recenters and reorients)."""
    from quantabench.chem import cube as cm

    pos_A = res.mol.GetConformer(res.conformers[0].conf_id).GetPositions() @ rot.T + shift
    z = np.array([a.GetAtomicNum() for a in res.mol.GetAtoms()])
    q = np.asarray(res.conformers[0].charges)
    h = 0.3
    lo = pos_A.min(axis=0) / cm.BOHR_A - 5.0
    n = np.ceil((pos_A.max(axis=0) / cm.BOHR_A + 5.0 - lo) / h).astype(int) + 1
    assert n.max() < 100
    proto = cm.Cube(lo, np.eye(3) * h, np.zeros(tuple(n)), z, pos_A / cm.BOHR_A, "fake\ndensity")
    p = proto.points_A() / cm.BOHR_A
    rho = np.zeros(len(p))
    pot = np.zeros(len(p))
    for zi, qi, r in zip(z, q, pos_A / cm.BOHR_A):
        d = np.linalg.norm(p - r, axis=1)
        rho += 0.3 * zi * np.exp(-2.0 * d)
        pot += qi / np.maximum(d, 0.1)
    dens = cm.write_cube(folder / "Dt.cube", proto.with_data(rho))
    esp = cm.write_cube(folder / "ESP.cube", proto.with_data(pot))
    return {"files": {"density": str(dens), "esp": str(esp)}}


def test_panel_qm_cube_surface(tmp_path, built):
    import copy

    from quantabench.chem.surfaces import DESCRIPTOR_KEYS
    from quantabench.gui import surface_panel as sp

    _, tfa = built
    res = copy.copy(tfa)
    res.qm = _fake_qm_cubes(tfa, tmp_path, _rotation([1, 2, 0.5], 70), np.array([2.0, -1.0, 0.5]))
    app = _app()
    w = _window(tmp_path)
    panel = w.surface_panel
    w.add_result(res)
    app.processEvents()
    assert panel.source_box.model().item(0).isEnabled() and panel.source() == "qm"
    assert panel.iso_box.isEnabled()
    panel.show_box.setChecked(True)
    assert _wait(app, lambda: panel.actor_shown() and not panel.busy)
    assert panel.message.text().startswith("QM density 0.002 au")
    d = panel.current.descriptors
    assert set(d) == set(DESCRIPTOR_KEYS)
    assert not _actors(w)[sp.ACTOR].GetPickable()

    def centroid_offset():               # the surface sits on the displayed atoms, not in the cube's frame
        return np.linalg.norm(np.asarray(panel.current.mesh.points).mean(axis=0)
                              - w.viewer.atoms.positions.mean(axis=0))

    assert centroid_offset() < 0.5 and "RMSD" not in panel.status.text()
    small = d["surf_volume_A3"]
    panel.iso_box.setValue(0.004)                         # tighter envelope, recomputed on the worker
    assert _wait(app, lambda: panel.actor_shown() and not panel.busy
                 and panel.current.descriptors["surf_volume_A3"] < small)
    # Move mode turns the molecule rigidly: the QM surface follows it
    before = np.asarray(panel.current.mesh.points).mean(axis=0)
    w.viewer.translate_atoms(range(len(w.viewer.atoms)), [3.0, 0, 0])
    assert _wait(app, lambda: panel.actor_shown() and not panel.busy
                 and np.asarray(panel.current.mesh.points).mean(axis=0)[0] > before[0] + 2.9)
    assert centroid_offset() < 0.5
    # a different geometry (not a rigid motion): the cubes do not belong to it, the QM surface is withdrawn
    pos = w.viewer.atoms.positions.copy()
    pos[0] += [0.0, 2.5, 0.0]
    w.viewer.set_positions(pos)
    assert _wait(app, lambda: not panel.actor_shown() and not panel.busy)
    assert "another geometry" in panel.message.text()
    _close(app, w)


def test_rigid_fit_recovers_a_rotation():
    from quantabench.gui.surface_panel import rigid_fit

    rng = np.random.default_rng(1)
    a = rng.normal(size=(9, 3)) * 2.0
    rot, shift = _rotation([0.3, -1, 2], 133), np.array([1.0, -4.0, 0.25])
    r, t, rmsd = rigid_fit(a, a @ rot.T + shift)
    np.testing.assert_allclose(r, rot, atol=1e-9)
    np.testing.assert_allclose(t, shift, atol=1e-9)
    assert rmsd < 1e-9
    mirrored = a * [1, 1, -1]                             # a mirror image is not a rigid motion
    assert rigid_fit(a, mirrored)[2] > 0.1


def test_panel_without_3d_view():
    """No OpenGL: the panel says so and does nothing."""
    from types import SimpleNamespace

    from quantabench.gui.surface_panel import SurfacePanel

    _app()
    win = SimpleNamespace(viewer=SimpleNamespace(plotter=None, error="3D view unavailable: no GL"))
    panel = SurfacePanel(win)
    assert "not available" in panel.message.text() and not panel.show_box.isEnabled()
    panel.show_box.setChecked(True)
    panel.refresh()
    assert not panel.actor_shown() and panel.current is None
