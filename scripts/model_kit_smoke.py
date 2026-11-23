"""Smoke test of the Model Kit (UI-7) in the real main window, with screenshots.

    xvfb-run -a -s "-screen 0 1700x1200x24" python scripts/model_kit_smoke.py OUT_DIR     # Linux, headless
    python scripts/model_kit_smoke.py OUT_DIR                                             # Windows, macOS

Opens the window with throwaway settings, switches to the Build workspace and Build mode, builds PFBA
(perfluorobutanoic acid) atom by atom through the kit's element and geometry pickers and click handlers, minimizes
it, builds cisplatin from the periodic table (a square-planar Pt: Minimize uses UFF) and GenX from fragments, adds
them to the molecule list and checks their SMILES. Saves PNGs of the window at each step. ``QWidget.grab()`` cannot
read the 3D view's OpenGL surface (offscreen it comes out black), so each picture is the window's grab with the 3D
view's own VTK screenshot painted into its place. Events are pumped as in the tests, deferred deletes included (Qt
retires old dock tab bars with deleteLater; without them a scripted run shows stale tab bars a real event loop never
shows). Exits 1 on a wrong structure.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PFBA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)F"
CISPLATIN_ATOMS = {"Pt": 1, "Cl": 2, "N": 2}
GENX = "OC(=O)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="qb_kit_smoke_"))
    for var, sub in (("QUANTABENCH_SETTINGS_DIR", "settings"), ("QUANTABENCH_WINDOW_STATE", "window.ini"),
                     ("QUANTABENCH_CACHE_DIR", "cache"), ("QUANTABENCH_WORKER_STATE", "worker"),
                     ("QUANTABENCH_SCRATCH_DIR", "scratch")):
        os.environ[var] = str(tmp / sub)       # never the user's own settings or window layout

    from PySide6.QtCore import QEvent, QPoint, QRect
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from rdkit import Chem

    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow(sm.Settings(output_dir=str(tmp / "runs")))
    win.resize(1600, 1120)
    win.show()
    app.processEvents()
    if win.viewer.error:
        print(win.viewer.error)
        return 2

    def settle(seconds: float = 0.3) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            time.sleep(0.02)

    def shot(name: str) -> None:
        settle()
        pm = win.grab()
        v = win.viewer
        v.render_now()
        arr = v.plotter.screenshot(None, return_img=True)
        h, w = arr.shape[:2]
        img = QImage(arr.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        widget = v.plotter.interactor
        p = QPainter(pm)
        p.drawImage(QRect(widget.mapTo(win, QPoint(0, 0)), widget.size()), img)
        p.end()
        pm.save(str(out / f"{name}.png"))
        print(f"saved {out / name}.png")

    def side_view() -> None:
        """Look at the molecule's plane (its two longest axes), fitted: the default view looks down a chain."""
        import numpy as np

        v = win.viewer
        pos = v.atoms.positions
        c = pos.mean(axis=0)
        axes = np.linalg.svd(pos - c)[2]
        v.plotter.camera_position = [tuple(c + 30 * axes[2]), tuple(c), tuple(axes[1])]
        v.plotter.renderer.reset_camera(render=False)
        v.plotter.camera_set = True
        v.request_render()

    def canon(smiles: str) -> str:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    kit = win.model_kit
    win.set_workspace("build")
    win.label_box.setCurrentIndex(win.label_box.findData("element"))
    win.set_view_mode("build")                       # Ctrl+4: an empty model, nothing is selected
    settle()

    def rows(atom: int) -> list[int]:
        return [r for r, v in enumerate(kit.valences) if v.atom == atom and v.h is not None]

    # ---- PFBA, atom by atom: C (trigonal planar) in empty space, O by a double bond and O (bent) on its stubs,
    # then C (tetrahedral) and F
    kit.element_buttons["C"].click()
    kit.choose_geometry("sp2")
    kit.click_space()
    shot("01_first_atom_double_valences")           # a C sp2: three double free valences (magenta stubs)
    kit.choose("=O")
    kit.click_stub(rows(0)[0])
    kit.choose("O sp3")
    kit.click_stub(rows(0)[0])
    kit.choose("C sp3")
    c = kit.click_stub(rows(0)[0])
    win.viewer.reset_camera()
    kit.choose("=O")                                # refused: an sp3 carbon has no double free valence
    n = kit.stack.count()
    if kit.click_stub(rows(c)[0]) is not None or kit.stack.count() != n:
        print("=O on an sp3 carbon was not refused")
        return 1
    shot("02_acid_head_stubs_and_a_refused_edit")
    cur = c
    for k in range(3):
        if k < 2:
            kit.choose("C sp3")
            nxt = kit.click_stub(rows(cur)[0])
        kit.choose("F")
        while rows(cur):
            kit.click_stub(rows(cur)[0])
        cur = nxt
    if kit.model.to_smiles(isomeric=False) != canon(PFBA):
        print("PFBA: wrong structure", kit.model.to_smiles())
        return 1
    res = kit.minimize()
    side_view()
    print("PFBA", kit.model.to_smiles(), res, f"{kit.stack.count()} edits on the undo stack")
    kit.element_buttons["S"].click()                # the geometry chooser for S: lone pairs drawn as rings
    shot("03_pfba_built_minimized_s_geometries")
    kit.choose("C sp3")
    win.a_undo.trigger()                            # Ctrl+Z: before Minimize
    win.a_redo.trigger()                            # Ctrl+Y
    win.set_view_mode("rotate")                     # select two atoms for the edit tools: the C=O
    win.selection.select([0, 1])
    shot("04_rotate_mode_selection_value")
    kit.add_to_list()

    # ---- cisplatin from the periodic table: Pt square planar with charge -2, two Cl, two NH3 drawn as N+ (the
    # formal-charge picture of its dative bonds); Minimize uses UFF
    kit.new()
    win.set_view_mode("build")
    kit.table_section.set_open(True)
    kit.charge_box.setCurrentIndex(kit.charge_box.findData(-2))
    kit.table_buttons["Pt"].click()
    kit.choose_geometry("square planar")
    pt = kit.click_space()
    for ligand in ("Cl", "Cl", "N+ sp3", "N+ sp3"):
        kit.choose(ligand)
        kit.click_stub(next(r for r, v in enumerate(kit.valences) if v.atom == pt))
    kit.charge_box.setCurrentIndex(kit.charge_box.findData(-2))
    kit.table_buttons["Pt"].click()                 # show Pt's coordination geometries in the dock
    kit.choose_geometry("square planar")
    res = kit.minimize()
    side_view()
    counts = {}
    for i in range(kit.model.n_atoms):
        counts[kit.model.symbol(i)] = counts.get(kit.model.symbol(i), 0) + 1
    if res is None or res.forcefield != "UFF" or any(counts.get(k) != v for k, v in CISPLATIN_ATOMS.items()):
        print("cisplatin: wrong structure or force field", counts, res)
        return 1
    print("cisplatin", kit.model.to_smiles(), res.forcefield, res.note)
    if kit.model.to_smiles(isomeric=False) != canon("Cl[Pt-2](Cl)([NH3+])[NH3+]"):
        print("cisplatin: wrong SMILES", kit.model.to_smiles())
        return 1
    shot("05_cisplatin_square_planar_uff")
    kit.add_to_list()
    kit.table_section.set_open(False)

    # ---- GenX from groups: COOH, then on its CH3 carbon F, CF3, ether O and C3F7
    kit.new()                                       # an empty model (Build mode would edit the selected PFBA)
    win.set_view_mode("build")
    kit.choose("COOH")
    entry = kit.click_space().entry
    kit.choose("C sp3")
    c = kit.click_stub(next(r for r, v in enumerate(kit.valences) if v.h == entry))
    hs = [v.h for v in kit.valences if v.atom == c]
    kit.choose("F")
    kit.click_stub(next(r for r, v in enumerate(kit.valences) if v.h == hs[0]))
    kit.choose("CF3")
    kit.click_stub(next(r for r, v in enumerate(kit.valences) if v.h == hs[1]))
    kit.choose("ether O")
    ether = kit.click_stub(next(r for r, v in enumerate(kit.valences) if v.h == hs[2]))
    kit.choose("perfluoropropyl")
    kit.click_stub(next(r for r, v in enumerate(kit.valences) if v.h == ether.exit))
    kit.minimize()
    side_view()
    if kit.model.to_smiles(isomeric=False) != canon(GENX):
        print("GenX: wrong structure", kit.model.to_smiles())
        return 1
    print("GenX", kit.model.to_smiles())
    shot("06_genx_from_fragments")
    kit.add_to_list()
    side_view()
    shot("07_added_to_the_list")
    ids = [r.id for r in win.model.rows()]
    print("molecule list:", ids, [r.source for r in win.model.rows()])
    win.close()
    t = time.time()
    while not win.closed and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.02)
    return 0 if len(ids) == 3 else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "model_kit_screenshots")))
