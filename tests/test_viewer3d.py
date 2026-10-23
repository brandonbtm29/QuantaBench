"""3D viewer: projection/picking math, selection highlights, measurements, move mode, screenshots.

Display coordinates are VTK's: pixels from the bottom-left of the render window.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")

from quantabench.gui import viewer3d as v3d  # noqa: E402
from quantabench.gui.selection import SelectionModel, format_measurement, measure  # noqa: E402

needs_display = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                   reason="needs a display (run under xvfb-run on Linux)")

PFOA_ANION = "[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"


# ====================================================================== pure helpers (no GUI)
def test_ray_sphere_pick_nearest_and_miss():
    centers = np.array([[0, 0, -10.0], [0, 0, -5.0], [3, 0, -5.0]])
    radii = np.array([1.0, 0.5, 0.5])
    hit = v3d.ray_sphere_pick(np.zeros(3), np.array([0, 0, -1.0]), centers, radii)
    assert hit is not None and hit[0] == 1 and np.isclose(hit[1], 4.5)
    assert v3d.ray_sphere_pick(np.zeros(3), np.array([0, 1.0, 0]), centers, radii) is None
    # grazing ray just inside / outside the radius
    assert v3d.ray_sphere_pick(np.array([0.49, 0, 0]), np.array([0, 0, -1.0]), centers[1:2], radii[1:2])
    assert v3d.ray_sphere_pick(np.array([0.51, 0, 0]), np.array([0, 0, -1.0]), centers[1:2], radii[1:2]) is None
    # spheres behind the ray origin are not hit
    assert v3d.ray_sphere_pick(np.array([0, 0, -20.0]), np.array([0, 0, -1.0]), centers, radii) is None


def test_rotation_matrix_is_proper_rotation():
    r = v3d.rotation_matrix([1, 2, 3], 0.7)
    assert np.allclose(r @ r.T, np.eye(3))
    assert np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(v3d.rotation_matrix([0, 0, 1], np.pi / 2) @ [1, 0, 0], [0, 1, 0])


def test_diverging_colors_are_symmetric_and_signed():
    rgb = v3d.diverging_rgb(np.array([-1.0, 0.0, 1.0]), 1.0)
    assert rgb[0, 0] > rgb[0, 2]          # negative -> red
    assert rgb[2, 2] > rgb[2, 0]          # positive -> blue
    assert np.all(rgb[1] > 240)           # zero -> near white
    assert np.array_equal(v3d.diverging_rgb([5.0], 1.0), v3d.diverging_rgb([1.0], 1.0))


def test_dashed_segments_cover_the_line():
    pts, lines = v3d.dashed_segments(np.array([[0, 0, 0], [1.0, 0, 0]]), [(0, 1)], dash=0.2, gap=0.05)
    assert len(lines) % 3 == 0 and len(pts) == 2 * (len(lines) // 3)
    assert np.isclose(pts[:, 0].min(), 0) and np.isclose(pts[:, 0].max(), 1.0)


def test_sphere_resolution_drops_for_large_systems():
    assert v3d.sphere_resolution(30) > v3d.sphere_resolution(3000) > v3d.sphere_resolution(20000)


def test_shim_reexports():
    from quantabench.gui import viewer

    assert viewer.MoleculeViewer is v3d.MoleculeViewer3D
    assert viewer.STYLES == v3d.STYLES
    assert viewer.bonds_from_rdkit is v3d.bonds_from_rdkit and viewer.element_counts is v3d.element_counts


# ====================================================================== widget
@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app):
    w = v3d.MoleculeViewer3D()
    if w.error:
        pytest.skip(w.error)
    w.resize(640, 480)
    w.show()
    app.processEvents()
    yield w
    w.close_viewer()
    w.close()
    app.processEvents()


@pytest.fixture(scope="module")
def pfoa():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(PFOA_ANION))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def _spread_atoms():
    """Three atoms far apart, so the projected center of each is unambiguous."""
    from ase import Atoms

    return Atoms("CON", positions=[[0, 0, 0], [6, 0, 0], [0, 6, 0]])


@needs_display
def test_construct_and_show_pfoa(viewer, pfoa):
    viewer.show_rdkit(pfoa)
    assert len(viewer.atoms) == pfoa.GetNumAtoms() == 25
    assert {"atoms", "bonds"} <= viewer._names
    assert viewer._atom_pd.n_points == 25
    assert viewer._bond_pd.n_points == 2 * pfoa.GetNumBonds()   # two half-bonds per bond
    for style in v3d.STYLES:
        viewer.set_style(style)
    viewer.set_style("Ball and stick")


@needs_display
def test_projection_matches_vtk(viewer, pfoa):
    viewer.show_rdkit(pfoa)
    ren = viewer.plotter.renderer
    for i in (0, 7, 20):
        p = viewer.atoms.positions[i]
        ren.SetWorldPoint(*p, 1.0)
        ren.WorldToDisplay()
        x, y, _ = ren.GetDisplayPoint()
        assert np.allclose(viewer.world_to_display(p)[0, :2], (x, y), atol=0.5)
        q = viewer.display_to_world(*viewer.world_to_display(p)[0])
        assert np.allclose(q, p, atol=1e-6)


@needs_display
def test_pick_atom_and_empty_space(viewer, pfoa):
    viewer.show_atoms(_spread_atoms())
    for ortho in (False, True):             # perspective, then orthographic
        assert bool(viewer.plotter.renderer.GetActiveCamera().GetParallelProjection()) == ortho
        for i, xy in enumerate(viewer.world_to_display(viewer.atoms.positions)[:, :2]):
            assert viewer.pick_atom(*xy) == i
        assert viewer.pick_atom(3, 3) is None
        viewer.toggle_projection()
    # PFOA: the atom nearest to the camera is always pickable at its projected center
    viewer.show_rdkit(pfoa)
    d = viewer.world_to_display(viewer.atoms.positions)
    front = int(np.argmin(d[:, 2]))
    assert viewer.pick_atom(*d[front, :2]) == front


@needs_display
def test_hidden_hydrogens_keep_full_indices(viewer):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("OCC(F)(F)F"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    viewer.show_rdkit(mol)
    viewer.set_show_hydrogens(False)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    assert list(viewer._vis) == heavy
    d = viewer.world_to_display(viewer.atoms.positions)
    picked = {viewer.pick_atom(*d[i, :2]) for i in range(len(d))}
    assert picked - {None} <= set(heavy)
    front = heavy[int(np.argmin(d[heavy, 2]))]
    assert viewer.pick_atom(*d[front, :2]) == front


@needs_display
def test_click_select_toggle_clear(viewer):
    viewer.show_atoms(_spread_atoms())
    xy = viewer.world_to_display(viewer.atoms.positions)[:, :2]

    def click(i=None, ctrl=False, at=None):
        x, y = xy[i] if at is None else at
        viewer._mouse_press(x, y, ctrl=ctrl)
        viewer._mouse_release(x + 1, y + 1)       # < 4 px jitter is still a click

    click(0)
    assert viewer.selection.atoms == [0]
    click(1)
    assert viewer.selection.atoms == [1]           # plain click replaces
    click(2, ctrl=True)
    assert viewer.selection.atoms == [1, 2]
    click(1, ctrl=True)
    assert viewer.selection.atoms == [2]           # ctrl toggles off
    click(at=(3, 3))
    assert viewer.selection.atoms == []            # empty space clears


@needs_display
def test_real_qt_click_and_drag(viewer, app):
    """Through Qt -> VTK events: a click selects, a drag rotates the camera and keeps the selection."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    viewer.show_atoms(_spread_atoms())
    widget = viewer.plotter.interactor
    ratio = widget.devicePixelRatioF()
    x, y = viewer.world_to_display(viewer.atoms.positions[1])[0, :2]
    pt = QPoint(round(x / ratio), round(widget.height() - 1 - y / ratio))
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pt)
    app.processEvents()
    assert viewer.selection.atoms == [1]
    cam_before = np.array(viewer.plotter.camera.position)
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(100, 100))
    for k in range(1, 6):
        QTest.mouseMove(widget, QPoint(100 + 20 * k, 100))
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(200, 100))
    app.processEvents()
    assert not np.allclose(viewer.plotter.camera.position, cam_before)   # trackball rotated
    assert viewer.selection.atoms == [1]                                  # a drag is not a click


@needs_display
def test_real_qt_keys(viewer, app):
    """P toggles projection; pyvista's/VTK's own single-letter keys (q = close!) are disabled."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    viewer.show_atoms(_spread_atoms())
    widget = viewer.plotter.interactor
    cam = viewer.plotter.renderer.GetActiveCamera()
    QTest.keyClick(widget, Qt.Key.Key_P)
    app.processEvents()
    assert cam.GetParallelProjection()
    for key in (Qt.Key.Key_Q, Qt.Key.Key_E, Qt.Key.Key_W):
        QTest.keyClick(widget, key)
    app.processEvents()
    assert not getattr(viewer.plotter, "_closed", False)
    assert "atoms" in viewer._names and viewer.plotter.renderer.actors["atoms"].GetProperty().GetRepresentation() == 2


@needs_display
def test_highlight_follows_shared_selection(app, pfoa):
    shared = SelectionModel()
    w = v3d.MoleculeViewer3D(selection=shared)
    if w.error:
        pytest.skip(w.error)
    try:
        w.resize(400, 300)
        w.show_rdkit(pfoa)
        assert "halo" not in w._names
        shared.select([0, 5, 9])                   # e.g. a click in the 2D view
        assert "halo" in w._names
        halo = w.plotter.renderer.actors["halo"]
        assert halo.GetMapper().GetInput().GetNumberOfPoints() == 3
        shared.clear()
        assert "halo" not in w._names
    finally:
        w.close_viewer()
        w.close()


@needs_display
def test_measurement_labels(viewer, pfoa):
    viewer.show_rdkit(pfoa)
    seen = []
    viewer.measurementChanged.connect(seen.append)
    pos = viewer.atoms.positions
    for sel in ([0, 1], [0, 1, 3], [0, 1, 3, 6]):
        viewer.selection.select(sel)
        expected = format_measurement(measure(pos, sel))
        assert viewer.measurement_text == expected
        assert seen[-1] == expected
        assert {"measure_lines", "measure_label"} <= viewer._names
        label = viewer.plotter.renderer.actors["measure_label"].GetMapper().GetInput()
        assert label.GetPointData().GetAbstractArray("label").GetValue(0) == expected
    assert seen[0].startswith("distance:") and seen[1].startswith("angle:") and seen[2].startswith("dihedral:")
    viewer.selection.select([0])
    assert viewer.measurement_text == "" and seen[-1] == ""
    assert "measure_label" not in viewer._names


def _slab_with_pfoa(pfoa):
    from ase import Atoms

    from quantabench.build.surfaces import TAG_ADSORBATE, graphite_slab

    slab = graphite_slab(nx=4, ny=3, layers=2, vacuum=15.0)
    conf = pfoa.GetConformer()
    mol = Atoms([a.GetSymbol() for a in pfoa.GetAtoms()], positions=np.array(conf.GetPositions()))
    mol.positions += [slab.cell[0, 0] / 2, slab.cell[1, 1] / 2, 12.0] - mol.positions.mean(axis=0)
    mol.set_tags([TAG_ADSORBATE] * len(mol))
    return slab + mol, len(slab)


def _pdist(p):
    return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)


@needs_display
def test_move_mode_rotation_is_rigid(viewer, pfoa):
    system, n_slab = _slab_with_pfoa(pfoa)
    viewer.show_atoms(system)
    viewer.set_mode("move")
    emitted = []
    viewer.coordinatesChanged.connect(emitted.append)
    before = viewer.atoms.positions.copy()
    assert np.array_equal(viewer.move_group(), np.arange(n_slab, len(system)))   # tag == 1 adsorbate

    viewer._mouse_press(320, 240)
    for k in range(1, 8):
        viewer._mouse_move(320 + 15 * k, 240 + 6 * k)
    viewer._mouse_release(320 + 15 * 7, 240 + 6 * 7)

    after = viewer.atoms.positions
    assert len(emitted) == 1 and np.allclose(emitted[0], after)
    assert np.array_equal(after[:n_slab], before[:n_slab])                       # slab untouched
    mol_b, mol_a = before[n_slab:], after[n_slab:]
    assert not np.allclose(mol_a, mol_b)
    assert np.allclose(_pdist(mol_a), _pdist(mol_b), atol=1e-9)                 # rigid
    assert np.allclose(mol_a.mean(axis=0), mol_b.mean(axis=0), atol=1e-9)       # about the centroid
    # the drawn glyphs follow the coordinates without a rebuild
    assert np.allclose(viewer._atom_pd.points, after[viewer._vis])


@needs_display
def test_move_mode_selection_translate_in_view_plane(viewer, pfoa):
    viewer.show_rdkit(pfoa)
    viewer.set_mode("move")
    group = [0, 1, 2]
    viewer.selection.select(group)
    before = viewer.atoms.positions.copy()
    x, y, _ = viewer.world_to_display(before[0])[0]       # press on a selected atom: the selection moves
    assert viewer.pick_atom(x, y) in group
    viewer._mouse_press(x, y, shift=True)
    viewer._mouse_move(x + 40, y + 30)
    viewer._mouse_release(x + 40, y + 30)
    shift = viewer.atoms.positions - before
    others = np.setdiff1d(np.arange(len(before)), group)
    assert np.allclose(shift[others], 0)
    assert np.allclose(shift[group], shift[group][0]) and np.linalg.norm(shift[0]) > 0.1
    dop = np.array(viewer.plotter.renderer.GetActiveCamera().GetDirectionOfProjection())
    assert abs(np.dot(shift[0], dop)) < 1e-6 * np.linalg.norm(shift[0]) + 1e-9   # stays in the view plane
    # the group's centroid (the grab point) follows the mouse exactly, also in perspective
    d0 = viewer.world_to_display(before[group].mean(axis=0))[0]
    d1 = viewer.world_to_display(viewer.atoms.positions[group].mean(axis=0))[0]
    assert np.allclose(d1[:2] - d0[:2], (40, 30), atol=0.05)


def test_fragment_of():
    bonds = np.array([[0, 1], [1, 2], [3, 4]])
    assert v3d.fragment_of(6, bonds, 2).tolist() == [0, 1, 2]
    assert v3d.fragment_of(6, bonds, 4).tolist() == [3, 4]
    assert v3d.fragment_of(6, bonds, 5).tolist() == [5]


def _press_drag(viewer, x, y, dx, dy, shift=False):
    viewer._mouse_press(x, y, shift=shift)
    for k in range(1, 5):
        viewer._mouse_move(x + dx * k / 4, y + dy * k / 4)
    viewer._mouse_release(x + dx, y + dy)


@needs_display
def test_move_mode_press_on_atom_grabs_whole_molecule(viewer, pfoa):
    """Avogadro's manipulate tool: a click does not change the selection, and a drag that starts on an
    atom moves its whole molecule, not just that atom (which would tear it out of the molecule)."""
    system, n_slab = _slab_with_pfoa(pfoa)
    viewer.show_atoms(system)
    viewer.set_mode("move")
    ads = np.arange(n_slab, len(system))
    target = n_slab + 3
    x, y, _ = viewer.world_to_display(viewer.atoms.positions[target])[0]
    assert viewer.pick_atom(x, y) in ads
    viewer._mouse_press(x, y)
    viewer._mouse_release(x, y)                     # a click in move mode
    assert viewer.selection.atoms == []
    before = viewer.atoms.positions.copy()
    _press_drag(viewer, x, y, 60, 0, shift=True)
    shift = viewer.atoms.positions - before
    assert np.allclose(shift[:n_slab], 0)
    assert np.allclose(shift[ads], shift[ads][0]) and np.linalg.norm(shift[ads][0]) > 0.5
    # an earlier selection of one adsorbate atom does not split the molecule when another atom is grabbed
    viewer.selection.select([n_slab])
    picked = viewer.pick_atom(*viewer.world_to_display(viewer.atoms.positions[target])[0][:2])
    assert picked is not None and picked != n_slab
    assert np.array_equal(viewer.grab_group(picked), ads)
    assert np.array_equal(viewer.grab_group(n_slab), [n_slab])           # pressing the selected atom
    assert np.array_equal(viewer.grab_group(0), ads)                     # a slab atom: the adsorbate
    # a molecule without tags: the connected fragment
    viewer.selection.clear()
    viewer.show_rdkit(pfoa)
    assert len(viewer.grab_group(0)) == pfoa.GetNumAtoms()


@needs_display
def test_label_font_not_scaled_twice_and_theme_follows(viewer, app):
    from PySide6.QtGui import QColor, QPalette

    actor, _ = viewer._text_actor(np.zeros((1, 3)), ["H"], 12)
    # the render window DPI (72 x devicePixelRatio, set by pyvistaqt) already scales point sizes
    assert actor.GetMapper().GetLabelTextProperty().GetFontSize() == 12
    viewer.show_atoms(_spread_atoms())
    pal = QPalette(viewer.palette())
    dark = not viewer._dark
    pal.setColor(QPalette.ColorRole.Window, QColor("#202020" if dark else "#f0f0f0"))
    viewer.setPalette(pal)
    app.processEvents()
    assert viewer._dark == dark


@needs_display
def test_select_mode_box(viewer):
    viewer.show_atoms(_spread_atoms())
    viewer.set_mode("select")
    w, h = viewer.plotter.renderer.GetSize()
    viewer._mouse_press(0, 0)
    viewer._mouse_move(w / 2, h / 2)
    viewer._mouse_release(w - 1, h - 1)
    assert sorted(viewer.selection.atoms) == [0, 1, 2]


@needs_display
def test_double_click_recenters_and_keys(viewer):
    viewer.show_atoms(_spread_atoms())
    x, y = viewer.world_to_display(viewer.atoms.positions[2])[0, :2]
    viewer._mouse_press(x, y, double=True)
    viewer._mouse_release(x, y)
    assert np.allclose(viewer.plotter.camera.focal_point, viewer.atoms.positions[2])
    cam = viewer.plotter.renderer.GetActiveCamera()
    assert viewer._on_key("p") and cam.GetParallelProjection()
    assert viewer._on_key("P") and not cam.GetParallelProjection()
    viewer._on_key("z")
    assert np.allclose(cam.GetDirectionOfProjection(), (0, 0, -1), atol=1e-6)
    viewer._on_key("X")
    assert np.allclose(cam.GetDirectionOfProjection(), (1, 0, 0), atol=1e-6)
    assert viewer._on_key("r")
    assert not viewer._on_key("w")                 # VTK's wireframe key is not ours


@needs_display
def test_label_and_color_modes(viewer, pfoa):
    q = np.linspace(-0.5, 0.5, pfoa.GetNumAtoms())
    viewer.show_rdkit(pfoa, charges=q)
    viewer.set_label_mode("element+index")
    labels = viewer.plotter.renderer.actors["labels"].GetMapper().GetInput().GetPointData().GetAbstractArray("label")
    assert labels.GetValue(0) == "O0" and labels.GetNumberOfValues() == 25
    viewer.set_label_mode("charge")
    labels = viewer.plotter.renderer.actors["labels"].GetMapper().GetInput().GetPointData().GetAbstractArray("label")
    assert labels.GetValue(0) == "-0.50"
    viewer.set_color_mode("charge")
    assert "charge_bar" in viewer._names
    assert np.array_equal(viewer._colors, v3d.diverging_rgb(q, 0.5))
    viewer.set_color_mode("tag")
    assert "charge_bar" not in viewer._names
    viewer.set_label_mode("none")
    assert "labels" not in viewer._names
    with pytest.raises(ValueError):
        viewer.set_color_mode("rainbow")
    with pytest.raises(ValueError):
        viewer.set_mode("fly")


@needs_display
def test_screenshot_png(viewer, pfoa, tmp_path):
    viewer.show_rdkit(pfoa)
    viewer.selection.select([0, 20])
    out = tmp_path / "shot.png"
    viewer.screenshot(out, scale=2)
    assert out.exists() and out.stat().st_size > 1000
    from PySide6.QtGui import QImage

    img = QImage(str(out))
    w, h = viewer.plotter.renderer.GetSize()
    assert (img.width(), img.height()) == (2 * w, 2 * h)


def test_fallback_without_vtk(app, monkeypatch):
    import pyvistaqt

    def broken(*_a, **_k):
        raise RuntimeError("no OpenGL")

    monkeypatch.setattr(pyvistaqt, "QtInteractor", broken)
    w = v3d.MoleculeViewer3D()
    assert w.plotter is None and "no OpenGL" in w.error
    w.show_atoms(_spread_atoms())               # all calls are harmless no-ops
    w.selection.select([0, 1])
    assert w.pick_atom(10, 10) is None
    w.screenshot("unused.png")
    w.clear()
    w.close_viewer()
