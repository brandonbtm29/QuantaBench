"""2D structure viewer: depiction, index mapping, picking, rotation, highlights and export."""

import os

import numpy as np
import pytest
from conftest import K_PFOS, PFOA
from rdkit import Chem

from quantabench.chem.conformers import embed
from quantabench.chem.standardize import prepare

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtSvgWidgets")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY") and os.name != "nt",
                                reason="needs a display (run under xvfb-run on Linux)")

FTCA_62 = "OC(=O)CCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"   # 6:2 fluorotelomer carboxylic acid


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def built(smiles: str, ph: float | None = 7.0, keep_largest_fragment: bool = True) -> Chem.Mol:
    """The pipeline's representation: standardized, protonated, explicit H, one 3D conformer."""
    mol, _ = prepare(Chem.MolFromSmiles(smiles), ph=ph, keep_largest_fragment=keep_largest_fragment)
    embed(mol, 1)
    return mol


@pytest.fixture(scope="module")
def pfoa_anion():
    return built(PFOA)


@pytest.fixture
def viewer(app):
    from quantabench.gui.selection import SelectionModel
    from quantabench.gui.viewer2d import MoleculeViewer2D

    w = MoleculeViewer2D(selection=SelectionModel())
    w.resize(800, 420)
    w.show()
    app.processEvents()
    yield w
    w.close()
    app.processEvents()


def test_renders_pfoa_anion_and_desalted_k_pfos(viewer, pfoa_anion, app):
    viewer.show_rdkit(pfoa_anion)
    assert viewer.error == ""
    assert "<svg" in viewer.svg and len(viewer.depicted_to_full) == pfoa_anion.GetNumAtoms()

    pfos = built(K_PFOS)
    assert all(a.GetSymbol() != "K" for a in pfos.GetAtoms())
    viewer.show_rdkit(pfos)
    app.processEvents()
    assert viewer.error == ""
    assert sum(a.GetFormalCharge() for a in pfos.GetAtoms()) == -1

    salt = built(K_PFOS, keep_largest_fragment=False)   # counter-ion kept: still draws
    viewer.show_rdkit(salt)
    assert viewer.error == ""


def test_hidden_hydrogen_index_map_uses_full_indices(viewer):
    mol = built(FTCA_62)   # carboxylate plus CH2 hydrogens
    # shuffle the atom order so hydrogens sit among the heavy atoms: an identity map would fail
    order = list(np.random.default_rng(7).permutation(mol.GetNumAtoms()))
    mol = Chem.RenumberAtoms(mol, [int(i) for i in order])
    viewer.show_rdkit(mol)
    assert viewer.error == ""
    to_full = viewer.depicted_to_full
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    assert sorted(to_full) == heavy
    assert to_full != list(range(len(to_full)))

    depict = viewer._depict
    for k, full in enumerate(to_full):
        a2, a3 = depict.GetAtomWithIdx(k), mol.GetAtomWithIdx(full)
        assert a2.GetSymbol() == a3.GetSymbol()
        assert a2.GetTotalNumHs() == a3.GetTotalNumHs(includeNeighbors=True)
        assert sorted(to_full[n.GetIdx()] for n in a2.GetNeighbors()) == sorted(
            n.GetIdx() for n in a3.GetNeighbors() if n.GetAtomicNum() > 1)

    o_minus_full = mol.GetSubstructMatch(Chem.MolFromSmarts("[OX1-]C(=O)"))[0]
    o_minus_dep = next(a.GetIdx() for a in depict.GetAtoms() if a.GetFormalCharge() == -1)
    assert to_full[o_minus_dep] == o_minus_full

    viewer.set_show_hydrogens(True)
    assert viewer.depicted_to_full == list(range(mol.GetNumAtoms()))


def test_stereocentre_hydrogen_kept_and_flip_keeps_configuration(viewer):
    from quantabench.gui.viewer2d import prepare_for_drawing

    mol = built("C[C@H](F)C(=O)O", ph=None)
    ref = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)
    viewer.show_rdkit(mol)
    shown = [mol.GetAtomWithIdx(i).GetSymbol() for i in viewer.depicted_to_full]
    assert shown.count("H") == 1          # the H on the stereocentre stays, so it can be wedged

    def drawn_centres():
        # read the configuration back from the wedges exactly as drawn
        back = Chem.MolFromMolBlock(Chem.MolToMolBlock(prepare_for_drawing(viewer._depict)), removeHs=False)
        return Chem.FindMolChiralCenters(back, useLegacyImplementation=False)

    centre = {viewer.depicted_to_full.index(i): lab for i, lab in ref}
    assert dict(drawn_centres()) == centre
    viewer.flip_horizontal()
    assert dict(drawn_centres()) == centre
    viewer.rotate(73)
    assert dict(drawn_centres()) == centre


def test_long_perfluoroalkyl_chain_is_a_zigzag(viewer):
    viewer.show_rdkit(built(K_PFOS))
    depict = viewer._depict
    chain = depict.GetSubstructMatch(Chem.MolFromSmarts("S" + "C(F)(F)" * 7 + "C(F)(F)F"))
    carbons = [i for i in chain if depict.GetAtomWithIdx(i).GetSymbol() in "CS"]
    xy = viewer.coords_2d[carbons]
    turns = []
    for a, b, c in zip(xy, xy[1:], xy[2:]):
        u, v = a - b, c - b
        ang = np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
        assert ang == pytest.approx(120.0, abs=2.0)
        turns.append(np.sign(u[0] * v[1] - u[1] * v[0]))
    # a zig-zag alternates its turn direction at every atom (a crankshaft or curl would not)
    assert all(t1 == -t2 for t1, t2 in zip(turns, turns[1:]))


def test_atom_at_and_click_picking(viewer, pfoa_anion, app):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest

    viewer.show_rdkit(pfoa_anion)
    app.processEvents()
    for full in (0, 1, 5):
        p = viewer.atom_scene_pos(full)
        assert viewer.atom_at(p) == full
        assert viewer.atom_at(QPointF(p.x() + 3, p.y() - 2)) == full
    assert viewer.atom_at(QPointF(5000.0, 5000.0)) is None

    vp = viewer.view.viewport()

    def click(full, mods=Qt.KeyboardModifier.NoModifier):
        pos = viewer.view.mapFromScene(viewer.atom_scene_pos(full))
        QTest.mouseClick(vp, Qt.MouseButton.LeftButton, mods, pos)
        app.processEvents()

    click(3)
    assert viewer.selection.atoms == [3]
    click(6)
    assert viewer.selection.atoms == [6]                  # a plain click replaces
    click(9, Qt.KeyboardModifier.ControlModifier)
    assert viewer.selection.atoms == [6, 9]               # Ctrl+click adds, in order
    click(6, Qt.KeyboardModifier.ControlModifier)
    assert viewer.selection.atoms == [9]                  # and toggles off
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     viewer.view.mapFromScene(QPointF(3000.0, 3000.0)))
    app.processEvents()
    assert viewer.selection.atoms == []


def _rotated_about_centroid(before, after, degrees):
    t = np.radians(degrees)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = before.mean(axis=0)
    return np.allclose((before - c) @ rot.T + c, after, atol=1e-6 if degrees % 1 == 0 else 1e-3)


def test_rotate_turns_coordinates_about_centroid(viewer, pfoa_anion, app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    viewer.show_rdkit(pfoa_anion)
    before = viewer.coords_2d
    p0 = viewer.atom_scene_pos(0)
    viewer.rotate(90)
    after = viewer.coords_2d
    assert _rotated_about_centroid(before, after, 90)
    p1 = viewer.atom_scene_pos(0)
    # screen y points down, so +90° (counter-clockwise) maps scene (x, y) to (y, -x)
    assert (p1.x(), p1.y()) == pytest.approx((p0.y(), -p0.x()), abs=0.5)
    assert viewer.atom_at(p1) == 0                       # picking follows the rotation
    # and the on-screen scale does not drift as the drawing turns
    from quantabench.gui.viewer2d import BOND_PX, DEPICTION_BOND

    for deg in (0, 37, 50):
        viewer.rotate(deg)
        a, b = viewer.atom_scene_pos(1), viewer.atom_scene_pos(3)
        model = np.linalg.norm(viewer.coords_2d[1] - viewer.coords_2d[3])
        assert np.hypot(a.x() - b.x(), a.y() - b.y()) == pytest.approx(BOND_PX * model / DEPICTION_BOND, rel=1e-3)
    viewer.rotate(-87)
    after = viewer.coords_2d

    viewer.view.setFocus()
    QTest.keyClick(viewer.view, Qt.Key.Key_BracketLeft)
    assert _rotated_about_centroid(after, viewer.coords_2d, 15)

    # another conformer of the same molecule keeps the user's orientation
    kept = viewer.coords_2d
    viewer.show_rdkit(Chem.Mol(pfoa_anion), reset_camera=False)
    assert np.allclose(viewer.coords_2d, kept)

    before = viewer.coords_2d
    viewer.flip_vertical()
    c = before.mean(axis=0)
    assert np.allclose(viewer.coords_2d, np.column_stack([before[:, 0], 2 * c[1] - before[:, 1]]))


def test_right_drag_rotates_with_the_mouse_angle(viewer, pfoa_anion, app):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    viewer.show_rdkit(pfoa_anion)
    app.processEvents()
    before = viewer.coords_2d
    vp = viewer.view.viewport()
    c = QPointF(viewer.view.mapFromScene(QPointF(0.0, 0.0)))
    r = 100.0
    path = [QPointF(c.x() + r * np.cos(np.radians(a)), c.y() - r * np.sin(np.radians(a))) for a in range(0, 91, 10)]

    def send(kind, pos, button, buttons):
        ev = QMouseEvent(kind, pos, vp.mapToGlobal(pos), button, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(vp, ev)

    right = Qt.MouseButton.RightButton
    send(QEvent.Type.MouseButtonPress, path[0], right, right)
    for p in path[1:]:
        send(QEvent.Type.MouseMove, p, Qt.MouseButton.NoButton, right)
    send(QEvent.Type.MouseButtonRelease, path[-1], right, Qt.MouseButton.NoButton)
    app.processEvents()
    # dragging a quarter turn counter-clockwise on screen rotates the molecule +90°
    assert _rotated_about_centroid(before, viewer.coords_2d, 90)
    assert viewer.selection.atoms == []                  # rotating never selects


def test_selection_drives_highlights(app, pfoa_anion):
    from quantabench.gui.selection import SelectionModel
    from quantabench.gui.viewer2d import MoleculeViewer2D

    shared = SelectionModel()
    w = MoleculeViewer2D(selection=shared)
    w.show_rdkit(pfoa_anion)
    assert w.highlighted_atoms == [] and "fill-opacity" not in w.svg
    shared.select([0, 1])                                # e.g. picked in the 3D view
    assert w.highlighted_atoms == [0, 1]
    assert "class='atom-0'" in w.svg and "fill:#FFAF00" in w.svg
    assert "#FFAF0099" not in w.svg                      # 8-digit hex is rewritten for QtSvg/SVG 1.1
    shared.clear()
    assert w.highlighted_atoms == [] and "fill:#FFAF00" not in w.svg

    mol = built(FTCA_62)
    w.show_rdkit(mol)
    h = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    shared.select([h, 0])                               # a hidden H cannot be highlighted in 2D
    assert w.highlighted_atoms == [0]
    w.close()


def test_options_charges_and_dark_palette(viewer, app):
    from PySide6.QtGui import QColor, QPalette

    light_pal = QPalette()  # explicit light baseline: the desktop may already be in dark mode (Windows)
    light_pal.setColor(QPalette.ColorRole.Window, QColor("#f4f5f7"))
    light_pal.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    viewer.setPalette(light_pal)
    app.processEvents()
    mol = built(FTCA_62)
    viewer.show_rdkit(mol)
    with pytest.raises(ValueError):
        viewer.set_options(charges=[0.0] * 3)
    rng = np.random.default_rng(1)
    q = rng.normal(0, 0.2, mol.GetNumAtoms())
    q += (-1.0 - q.sum()) / len(q)                       # net charge -1 e
    viewer.set_options(atom_indices=True, charges=q, cip_labels=True)
    assert viewer.error == ""
    # group charges on the drawn heavy atoms still add up to the net charge
    shown = sum(viewer._group_charge(i) for i in viewer.depicted_to_full)
    assert shown == pytest.approx(-1.0)
    assert "with H" in viewer.describe_atom(next(i for i in viewer.depicted_to_full
                                                if mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) > 0))
    light = viewer.svg

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#202124"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#202124"))
    viewer.setPalette(pal)
    app.processEvents()
    assert viewer._is_dark() and viewer.svg != light
    assert "<rect" not in viewer.svg.split("<!-- END OF HEADER -->")[1][:200]   # transparent background


def test_export_svg_and_png(viewer, pfoa_anion, tmp_path):
    from PySide6.QtGui import QImage

    with pytest.raises(RuntimeError):
        viewer.export(tmp_path / "empty.svg")
    viewer.show_rdkit(pfoa_anion)
    viewer.selection.select([0, 1])
    svg = viewer.export(tmp_path / "pfoa.svg")
    png = viewer.export(tmp_path / "pfoa.png", scale=2.0, background="white")
    assert svg.stat().st_size > 1000 and svg.read_text(encoding="latin-1").lstrip().startswith("<?xml")
    img = QImage(str(png))
    assert png.stat().st_size > 1000 and img.width() > 300 and not img.isNull()
    with pytest.raises(ValueError):
        viewer.export(tmp_path / "pfoa.bmp")
    viewer.clear()
    assert viewer.depicted_to_full == [] and viewer.atom_at((0.0, 0.0)) is None
