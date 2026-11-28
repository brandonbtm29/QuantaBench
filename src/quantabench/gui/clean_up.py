"""Clean up a molecule's 2D layout or 3D structure (BUILD-CLEAN): the named actions and the Molecules dock's entries.

    install(win)    adds Build > Clean up the 2D layout (``build.clean_2d``) and Build > Clean up the 3D structure
                    (``build.clean_3d``) to a MainWindow, named actions of ``gui.keymap`` without default keys (as
                    Minimize; Settings > Controls gives them one), and a Clean up submenu to the molecule list's
                    right-click menu

    clean_2d(win)   lays the 2D view's molecule out again (``MoleculeViewer2D.clean_up``: RDKit's CoordGen, the zig-zag
                    rule for long chains kept, the drawing checked for connectivity and stereo) and undoes rotations
                    and flips: the selected molecule, or the Model Kit's while an edit is open. The structure itself
                    is not changed.
    clean_3d(win)   cleans up the 3D structure in the Model Kit (``ModelKitPanel.clean_up``: Minimize until it
                    converges, MMFF94 or UFF, refused when the bonds or the stereo would change; one undoable step).
                    With no edit open it first opens the selected molecule there, as Edit selected does (its shown
                    conformer when it is built); Replace <id> or Add to list then hands the result back, Close drops
                    it. ``hold``: keep the atoms selected in the views fixed (default: the kit's Freeze selected box).

The Sketch dock has a Clean up button of its own for the drawing in the sketcher (``gui.sketcher_dock``).
"""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from ..chem.clean import CleanError

TIP_2D = ("Lay the 2D view's molecule out again (RDKit's CoordGen; long chains as a zig-zag) and undo rotations and "
          "flips; refused when the drawing would change the connectivity or the stereo")
TIP_3D = ("Clean up the 3D structure in the Model Kit: Minimize until it converges (MMFF94; UFF where MMFF94 has no "
          "parameters), refused when the bonds or the stereo would change; one step to undo. Opens the selected "
          "molecule in the Model Kit when no edit is open")


def _say(win, text: str, error: bool = False) -> None:
    win.statusBar().showMessage(text, 8000 if error else 5000)
    if error:
        win.log(text)


def clean_2d(win):
    """Lay the 2D view's molecule out again. Returns the ``chem.clean.Clean2D``, or None (the status bar says why)."""
    try:
        res = win.viewer2d.clean_up()
    except CleanError as exc:
        _say(win, f"Clean up 2D: {exc}", error=True)
        return None
    _say(win, f"2D layout redone ({res.layout}): connectivity and stereo unchanged.")
    return res


def _kit_dock_up(win) -> None:
    dock = getattr(win, "model_kit_dock", None)
    if dock is not None:
        dock.show()
        dock.raise_()


def clean_3d(win, hold: bool | None = None):
    """Clean up the 3D structure in the Model Kit (module docstring). Returns the ``chem.edit.CleanResult``, or None
    when there is nothing to clean up or it was refused (the Model Kit's message line says why)."""
    kit = getattr(win, "model_kit", None)
    if kit is None:
        _say(win, "Clean up 3D needs the Model Kit, which is not available.", error=True)
        return None
    if hold is None:
        hold = kit.freeze_box.isChecked()
    if not kit.active:
        if win._current_row() is None:
            _say(win, "Clean up 3D: select a molecule in the list first.", error=True)
            return None
        held = list(win.selection.atoms) if hold else []     # the same atom indices in the edit (Edit selected)
        if not kit.edit_selected():
            return None
        if held:
            win.selection.select([i for i in held if 0 <= i < kit.model.n_atoms])
    _kit_dock_up(win)
    res = kit.clean_up(kit.selected() if hold else [])
    if res is None:
        _say(win, f"Clean up 3D: {kit.message}", error=True)
    else:
        _say(win, kit.message)
    return res


def _list_menu(win, menu: QMenu) -> None:
    """The molecule list's right-click menu: a Clean up submenu, when a molecule is shown (or a Model Kit edit is
    open)."""
    kit = getattr(win, "model_kit", None)
    editing = kit is not None and kit.active
    if win._current_row() is None and not editing:
        return
    menu.addSeparator()
    sub = menu.addMenu("Clean up")
    sub.setObjectName("molecules.clean_up")
    sub.setToolTipsVisible(True)
    a2 = sub.addAction("2D layout")
    a2.setToolTip(TIP_2D)
    a2.triggered.connect(lambda _c=False: clean_2d(win))
    a3 = sub.addAction("3D structure (the open Model Kit edit)" if editing else "3D structure (in the Model Kit)")
    a3.setToolTip(TIP_3D)
    a3.setEnabled(kit is not None)
    a3.triggered.connect(lambda _c=False: clean_3d(win, hold=False))
    n = len(win.selection.atoms)
    held = sub.addAction(f"3D structure, holding the {n} selected atom{'s' if n != 1 else ''} fixed" if n else
                         "3D structure, holding the selected atoms fixed (select atoms first)")
    held.setToolTip("As above, with the atoms selected in the 2D or 3D view kept where they are")
    held.setEnabled(bool(n) and kit is not None)
    held.triggered.connect(lambda _c=False: clean_3d(win, hold=True))


def install(win) -> None:
    from . import keymap

    a2 = keymap.bind(win, QAction("Clean up the 2D layout", win), "build.clean_2d", TIP_2D)
    a2.triggered.connect(lambda _checked=False: clean_2d(win))
    a3 = keymap.bind(win, QAction("Clean up the 3D structure", win), "build.clean_3d", TIP_3D)
    a3.triggered.connect(lambda _checked=False: clean_3d(win))
    build = next((a.menu() for a in win.menuBar().actions()
                  if a.menu() is not None and a.text().replace("&", "") == "Build"), None)
    if build is not None:
        build.addAction(a2)
        build.addAction(a3)
    win.a_clean_2d, win.a_clean_3d = a2, a3
    kit = getattr(win, "model_kit", None)
    if kit is not None and isinstance(getattr(win, "keymap", None), keymap.Keymap):
        win.keymap.follow_tooltip(kit.b_clean, "build.clean_3d", kit.b_clean.toolTip())
    win.mol_list.menu_hooks.append(lambda m: _list_menu(win, m))
