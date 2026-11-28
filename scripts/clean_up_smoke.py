"""Smoke test of Clean up (BUILD-CLEAN) in the real main window, with screenshots.

    xvfb-run -a -s "-screen 0 1700x1200x24" python scripts/clean_up_smoke.py OUT_DIR     # Linux, headless
    python scripts/clean_up_smoke.py OUT_DIR                                             # Windows, macOS

Opens the window with throwaway settings, lists GenX ((R)-HFPO-DA) and PFOA, and saves PNGs of: the Molecules dock's
right-click menu with its Clean up submenu; GenX opened in the Model Kit with a distorted geometry (every atom moved
at random, one bond stretched); the same after the kit's Clean up (one undoable step; the message line gives the
energies and that the bonds and the stereo are unchanged); a refused clean-up (two fluorines held on top of each
other); the 2D view rotated and flipped, then laid out again by Build > Clean up the 2D layout; with Qt WebEngine and
the downloaded editor, a messy drawing in the Sketch dock before and after its Clean up. As in
``scripts/model_kit_smoke.py``, each picture is the window's grab with the 3D view's own VTK screenshot painted into
its place (``QWidget.grab()`` cannot read the OpenGL surface), and menus, which are windows of their own, are painted
in where they would open. Exits 1 when a step does not do what it should.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GENX_R = "OC(=O)[C@@](F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"
PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="qb_clean_smoke_"))
    for var, sub in (("QUANTABENCH_SETTINGS_DIR", "settings"), ("QUANTABENCH_WINDOW_STATE", "window.ini"),
                     ("QUANTABENCH_CACHE_DIR", "cache"), ("QUANTABENCH_WORKER_STATE", "worker"),
                     ("QUANTABENCH_SCRATCH_DIR", "scratch")):
        os.environ[var] = str(tmp / sub)       # never the user's own settings or window layout

    import numpy as np
    from PySide6.QtCore import QEvent, QPoint, QRect
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication
    from rdkit import Chem

    from quantabench import sketcher_assets
    from quantabench.gui import sketcher_web

    sketch_too = sketcher_web.prepare_before_app() and sketcher_assets.locate().ok   # before the QApplication

    from quantabench import settings as sm
    from quantabench.chem.inputs import from_smiles
    from quantabench.gui import clean_up
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

    def grab():
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
        return pm

    def save(pm, name: str) -> None:
        pm.save(str(out / f"{name}.png"))
        print(f"saved {out / name}.png")

    def side_view() -> None:
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
    win.add_inputs([from_smiles(GENX_R, id="GenX", name="(R)-HFPO-DA"), from_smiles(PFOA, id="PFOA")])
    win.mol_list.select_id("GenX")
    win.mol_list.currentIdChanged.emit("GenX")
    settle()

    # ---- the Molecules dock's right-click menu, Clean up open
    menu = win.mol_list.context_menu()
    item = next(a for a in menu.actions() if a.text() == "Clean up")
    sub = item.menu()
    for m in (menu, sub):
        m.ensurePolished()
        m.resize(m.sizeHint())
    pm = grab()
    at = win.mol_list.viewport().mapTo(win, QPoint(90, 12))
    p = QPainter(pm)
    p.drawPixmap(at, menu.grab())
    p.drawPixmap(at + QPoint(menu.width() - 4, menu.actionGeometry(item).top() - 4), sub.grab())
    p.end()
    save(pm, "01_molecules_dock_right_click_clean_up")
    menu.deleteLater()

    # ---- GenX in the Model Kit, distorted
    if not kit.edit_selected():
        print("Edit selected failed")
        return 1
    model = kit.model
    pos = model.positions + np.random.default_rng(3).normal(0.0, 0.18, model.positions.shape)
    pos[1] += np.array([0.6, 0.0, 0.0])
    model.set_positions(pos)
    side_view()
    kit.say("A distorted GenX (every atom moved at random, the C=O stretched): Clean up.")
    save(grab(), "02_distorted_in_the_model_kit")
    res = kit.clean_up()
    side_view()
    if res is None or model.to_smiles() != canon(GENX_R) or kit.stack.undoText() != "Clean up":
        print("Clean up 3D failed:", kit.message)
        return 1
    print("GenX", kit.message)
    save(grab(), "03_cleaned_up_one_undoable_step")

    # ---- refused: two fluorines of different carbons held on top of each other
    kit.end()                                       # drop the edit without asking
    kit.say("")
    win.mol_list.select_id("PFOA")
    win.mol_list.currentIdChanged.emit("PFOA")
    kit.edit_selected()
    model = kit.model
    fs = [i for i in range(model.n_atoms) if model.symbol(i) == "F"]
    a, b = fs[0], fs[6]
    pos = model.positions.copy()
    pos[b] = pos[a] + np.array([0.35, 0.0, 0.0])
    model.set_positions(pos)
    win.selection.select([a, b])
    kit.freeze_box.setChecked(True)
    before = model.positions.copy()
    if kit.clean_up() is not None or not np.array_equal(model.positions, before) or "refused" not in kit.message:
        print("the clean-up was not refused:", kit.message)
        return 1
    side_view()
    print("refused:", kit.message)
    save(grab(), "04_refused_atoms_held_on_top_of_each_other")
    kit.freeze_box.setChecked(False)
    kit.end()
    kit.say("")

    # ---- the 2D view: rotated and flipped, then laid out again
    win.mol_list.select_id("GenX")
    win.mol_list.currentIdChanged.emit("GenX")
    win.viewer2d.rotate(35.0)
    win.viewer2d.flip_horizontal()
    save(grab(), "05_2d_view_rotated_and_flipped")
    if clean_up.clean_2d(win) is None:
        print("Clean up 2D failed:", win.statusBar().currentMessage())
        return 1
    save(grab(), "06_2d_view_laid_out_again")

    # ---- the Sketch dock (only with Qt WebEngine and the downloaded editor): a messy drawing, then Clean up
    if sketch_too:
        from quantabench.gui import sketcher_dock

        sketcher_dock.open_dock(win)
        win.sketch_dock.setFloating(True)            # its own window, at the size open_dock gives a narrow dock
        win.sketch_dock.resize(980, 680)
        panel = win.sketch_panel
        end = time.time() + 240
        while not panel.ready and time.time() < end:
            settle(0.2)
        if not panel.ready:
            print("the sketcher did not start:", panel.status.text())
            return 1
        m = Chem.MolFromSmiles(GENX_R)
        from quantabench.chem import clean

        for seed in range(1, 50):                   # a hand drawing that still says (R)-GenX
            m = Chem.MolFromSmiles(GENX_R)
            clean.layout_2d(m)
            conf = m.GetConformer()
            rng = np.random.default_rng(seed)
            for i in range(m.GetNumAtoms()):
                q = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (q.x + rng.normal(0, 0.35), q.y + rng.normal(0, 0.35), 0.0))
            m.SetIntProp("_MolFileChiralFlag", 1)  # absolute stereo, as the sketcher marks a drawn wedge
            messy = Chem.MolToV3KMolBlock(m)
            if Chem.MolToSmiles(Chem.MolFromMolBlock(messy)) == canon(GENX_R):
                break
        panel.view.import_text(messy, clear=True)
        settle(1.5)
        panel._say("A messy drawing of (R)-GenX: Clean up.")
        dock = win.sketch_dock
        dock.grab().save(str(out / "07_sketch_messy_drawing.png"))
        print(f"saved {out / '07_sketch_messy_drawing'}.png")
        panel.clean_up()
        end = time.time() + 30
        while "Cleaned up" not in panel.status.text() and time.time() < end:
            settle(0.2)
        settle(1.5)
        if "Cleaned up" not in panel.status.text():
            print("the sketch was not cleaned up:", panel.status.text())
            return 1
        dock.grab().save(str(out / "08_sketch_cleaned_up.png"))
        print(f"saved {out / '08_sketch_cleaned_up'}.png")
    win.close()
    settle(1.0)
    if sketch_too:
        win.sketch_panel.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "clean_up_smoke")))
