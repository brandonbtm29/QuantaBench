"""File > Import calculation output... (cclib): Gaussian, ORCA, Psi4, xtb, MOPAC, ... output files.

CONTRACT (implemented by the spectra/interop builder):

    install(win: MainWindow) -> None
        Adds the File menu action. Reads the chosen file with ``chem.io_cclib.read_output`` on a worker thread
        (``gui.workers.FunctionWorker``), turns it into a built ``MoleculeResult`` (one conformer, method from
        the file, charges, vibrations dict when the file has frequencies) and calls ``win.add_result(res)``.

Several files can be chosen at once; each is parsed on the same worker and added in order. Files that cannot be
read are listed in one message box; the others are still added.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMessageBox

FILE_FILTER = ("Calculation outputs (*.out *.log *.dat *.txt *.output *.lis);;"
               "Gaussian (*.log *.out);;ORCA / Psi4 / xtb / MOPAC (*.out *.dat);;All files (*)")


def read_files(paths: list[str]) -> tuple[list, list[str]]:
    """Parse ``paths`` (worker thread): (MoleculeResults, error messages)."""
    from ..chem import io_cclib

    results, errors = [], []
    for p in paths:
        try:
            calc = io_cclib.read_output(p)
            results.append(io_cclib.to_result(calc, p))
        except Exception as exc:
            errors.append(f"{Path(p).name}: {exc}")
    return results, errors


def import_files(win, paths: list[str] | None = None) -> None:
    """Ask for output files (unless given) and add them to the molecule list when parsed."""
    from ..chem import io_cclib
    from . import workers

    ok, why = io_cclib.available()
    if not ok:
        QMessageBox.warning(win, "Import calculation output", f"cclib is not installed: {why}\n"
                            "(pip install quantabench[interop])")
        return
    if not paths:
        paths, _ = QFileDialog.getOpenFileNames(win, "Import calculation output", "", FILE_FILTER)
    if not paths:
        return
    win.statusBar().showMessage(f"Reading {len(paths)} output file(s)...")

    def done(value):
        results, errors = value
        win.statusBar().clearMessage()
        for res in results:
            try:
                win.add_result(res)
            except Exception as exc:  # one result the window cannot show must not drop the others
                errors.append(f"{res.input.name}: added, but the window could not show it "
                              f"({type(exc).__name__}: {exc})")
                continue
            win.log(f"Imported {res.input.name} as {res.input.id}: {res.method}"
                    + (f", {res.vibrations['n_modes']} vibrational modes" if res.vibrations else ""))
            for note in res.notes[1:]:
                win.log(f"  note: {note}")
        if errors:
            QMessageBox.warning(win, "Import calculation output", "\n".join(errors))

    def failed(msg):
        win.statusBar().clearMessage()
        QMessageBox.warning(win, "Import calculation output", msg)

    workers.start(workers.FunctionWorker(read_files, list(paths)), win, on_done=done, on_failed=failed)


def install(win) -> None:
    action = QAction("Import calculation output...", win)
    action.setToolTip("Read a Gaussian, ORCA, Psi4, Q-Chem, xtb, MOPAC, ... output file with cclib "
                      "(geometry, energies, charges, frequencies and IR intensities)")
    action.triggered.connect(lambda _checked=False: import_files(win))
    menu = None
    for a in win.menuBar().actions():
        if a.menu() is not None and a.text().replace("&", "") == "File":
            menu = a.menu()
            break
    if menu is not None:
        anchor = getattr(win, "a_export", None)
        actions = menu.actions()
        if anchor in actions:                      # right after Export...
            idx = actions.index(anchor)
            if idx + 1 < len(actions):
                menu.insertAction(actions[idx + 1], action)
            else:
                menu.addAction(action)
        else:
            menu.addAction(action)
    win.a_import_output = action
