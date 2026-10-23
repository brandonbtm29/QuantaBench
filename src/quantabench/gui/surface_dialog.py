"""Surface builder: graphite slab with the selected molecule placed above it (PFAS-on-carbon systems).

The dialog is modeless and previews in the main window's 3D view, so the user can keep working in that
view: switch it to Move mode (the dialog's "Move adsorbate" button, or the toolbar) and drag to rotate
the adsorbate rigidly, Shift+drag to slide it. Every finished drag arrives through ``set_positions``
(wired to the viewer's ``coordinatesChanged``), updates the system that Save and the LAMMPS export
write, and re-runs the overlap report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..build import surfaces
from . import workers

BOND_WARN_A = 0.5     # Å; a hand move that changes a bonded distance more than this split the molecule


class SurfaceDialog(QDialog):
    moveModeRequested = Signal(bool)     # the "Move adsorbate" button: True = move mode, False = rotate

    def __init__(self, molecule_atoms=None, molecule_name: str = "", preview_cb=None, parent=None, gpu: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Surface Builder: graphite slab")
        self.molecule = molecule_atoms
        self.molecule_name = molecule_name or "molecule"
        self.preview_cb = preview_cb
        self.gpu = gpu
        self.system = None
        self._previewed = False
        form = QFormLayout()
        self.nx, self.ny = QSpinBox(), QSpinBox()
        for w, v in ((self.nx, 16), (self.ny, 9)):
            w.setRange(1, 200)
            w.setValue(v)
            w.valueChanged.connect(self._update_size)
        self.layers = QSpinBox()
        self.layers.setRange(1, 20)
        self.layers.setValue(3)
        self.fix = QSpinBox()
        self.fix.setRange(0, 20)
        self.fix.setValue(1)
        self.vacuum = QDoubleSpinBox()
        self.vacuum.setRange(5, 200)
        self.vacuum.setValue(30)
        self.vacuum.setSuffix(" Å")
        self.height = QDoubleSpinBox()
        self.height.setRange(1.5, 50)
        self.height.setValue(3.5)
        self.height.setSuffix(" Å")
        self.orientation = QComboBox()
        self.orientation.addItems(["flat", "upright", "as-is"])
        self.size_label = QLabel()
        form.addRow("Cells along x (2.46 Å)", self.nx)
        form.addRow("Cells along y (4.26 Å)", self.ny)
        form.addRow("", self.size_label)
        form.addRow("Graphene layers", self.layers)
        form.addRow("Fixed bottom layers", self.fix)
        form.addRow("Vacuum above surface", self.vacuum)
        mol_text = f"{molecule_name} ({len(molecule_atoms)} atoms)" if molecule_atoms is not None else "none (slab only)"
        form.addRow("Molecule", QLabel(mol_text))
        form.addRow("Gap above surface", self.height)
        form.addRow("Orientation", self.orientation)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(360)
        hint = QLabel("Move adsorbate: drag in the 3D view to rotate it, Shift+drag to slide it. "
                      "Pressing on the molecule grabs all of it (only the selection, if you press on a "
                      "selected atom).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        preview = QPushButton("Build && Preview")
        preview.clicked.connect(self.build)
        self.move_btn = QPushButton("Move adsorbate")
        self.move_btn.setCheckable(True)
        self.move_btn.setEnabled(molecule_atoms is not None)
        self.move_btn.toggled.connect(self._on_move_toggled)
        save = QPushButton("Save...")
        save.clicked.connect(self.save)
        self.smd_btn = QPushButton("Export LAMMPS SMD setup...")
        self.smd_btn.setToolTip("Steered MD: the adsorbate is pulled off the frozen slab (quantabench.md)")
        self.smd_btn.setEnabled(molecule_atoms is not None)
        self.smd_btn.clicked.connect(self.export_smd)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        for b in (preview, self.move_btn, save, self.smd_btn):
            buttons.addButton(b, QDialogButtonBox.ButtonRole.ActionRole)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.status)
        if molecule_atoms is not None:
            lay.addWidget(hint)
        lay.addWidget(buttons)
        self._update_size()

    def _update_size(self) -> None:
        self.size_label.setText(f"= {self.nx.value() * surfaces.A_GRAPHENE:.1f} x "
                                f"{self.ny.value() * surfaces.B_RECT:.1f} Å")

    def _on_move_toggled(self, on: bool) -> None:
        if on and self.system is None:
            self.build()
        self.moveModeRequested.emit(on)

    def set_move_checked(self, on: bool) -> None:
        """Mirror the main window's mode without re-emitting."""
        self.move_btn.blockSignals(True)
        self.move_btn.setChecked(on)
        self.move_btn.blockSignals(False)

    # ------------------------------------------------------------------ system
    def build(self):
        slab = surfaces.graphite_slab(self.nx.value(), self.ny.value(), self.layers.value(), self.vacuum.value(),
                                      self.fix.value())
        system = slab
        if self.molecule is not None:
            system = surfaces.place_on_surface(slab, self.molecule, self.height.value(), self.orientation.currentText())
        self.system = system
        self._remember_bonds()
        self._report()
        if self.preview_cb:
            # keep the user's view on rebuilds with the same box; refit the first time
            self.preview_cb(system, not self._previewed)
            self._previewed = True
        return system

    def set_positions(self, positions) -> None:
        """New coordinates from the 3D view (after a move-mode drag)."""
        if self.system is None:
            return
        pos = np.asarray(positions, float).reshape(-1, 3)
        if len(pos) != len(self.system):
            return  # the view shows something else now
        self.system.positions = pos
        self._report(moved=True)

    def _remember_bonds(self) -> None:
        """Bonded pairs and lengths inside the adsorbate, to warn when a hand move tears the molecule."""
        from .viewer3d import bonds_from_distances

        self._ads_bonds = np.zeros((0, 2), int)
        self._ads_lengths = np.zeros(0)
        tags = self.system.get_tags()
        ads = np.where(tags == surfaces.TAG_ADSORBATE)[0] if len(tags) else np.zeros(0, int)
        if len(ads) > 1:
            pairs = ads[bonds_from_distances(self.system[ads])]
            self._ads_bonds = pairs.reshape(-1, 2)
            self._ads_lengths = self._bond_lengths()

    def _bond_lengths(self) -> np.ndarray:
        pos = self.system.positions
        return np.linalg.norm(pos[self._ads_bonds[:, 0]] - pos[self._ads_bonds[:, 1]], axis=1)

    def max_bond_change(self) -> float:
        """Largest change (Å) of a bonded distance inside the adsorbate since it was built."""
        if self.system is None or not len(getattr(self, "_ads_bonds", ())):
            return 0.0
        return float(np.max(np.abs(self._bond_lengths() - self._ads_lengths)))

    def _report(self, moved: bool = False) -> None:
        system = self.system
        clashes = surfaces.min_distance_report(system, 1.0)
        msg = f"{len(system)} atoms, cell {system.cell.lengths().round(2).tolist()} Å."
        if self.molecule is not None:
            gap = surfaces.surface_adsorbate_gap(system)
            msg += f" Closest surface-molecule distance {gap:.2f} Å."
        msg += f" {len(clashes)} overlapping pairs (<1 Å)." if clashes else " No overlaps."
        if moved:
            msg += " (adsorbate moved by hand)"
            torn = self.max_bond_change()
            if torn > BOND_WARN_A:
                msg += (f"\nWarning: a bond in the molecule changed by {torn:.2f} Å, so only part of it was "
                        "moved. Undo with Build && Preview, then press on the molecule (not a selected atom) "
                        "to grab all of it.")
        self.status.setText(msg)

    def save(self) -> None:
        system = self.system or self.build()
        path, _ = QFileDialog.getSaveFileName(self, "Save system", "system.extxyz",
                                              "Extended XYZ (*.extxyz);;LAMMPS data (*.data);;XYZ (*.xyz);;"
                                              "VASP POSCAR (*.vasp);;CIF (*.cif)")
        if not path:
            return
        from ase.io import write

        try:
            if Path(path).suffix == ".data":
                write(path, system, format="lammps-data", atom_style="charge", units="real", masses=True)
            else:
                write(path, system)
        except Exception as exc:
            QMessageBox.warning(self, "Save", f"Could not save:\n{exc}")
            return
        self.status.setText(self.status.text() + f"\nSaved {path}")

    def export_smd(self, folder: str | None = None) -> None:
        """Write a LAMMPS steered-MD folder (system.data, in.smd, README.txt) for the current system."""
        system = self.system or self.build()
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Folder for the LAMMPS SMD setup")
        if not folder:
            return
        from ..md.lammps import write_smd_setup

        self.smd_btn.setEnabled(False)
        self.status.setText(self.status.text() + "\nWriting LAMMPS setup...")
        w = workers.FunctionWorker(write_smd_setup, system.copy(), folder, gpu=self.gpu,
                                   title=f"{self.molecule_name} on graphite")
        w.done.connect(self._smd_done)        # bound methods: Qt drops them if the dialog is gone
        w.failed.connect(self._smd_failed)
        # The thread must outlive the dialog: closing it deletes the dialog (deleteLater), and a QThread
        # destroyed while running aborts the process. The main window also waits for its workers on quit.
        workers.start(w, self.parent() or self)

    def _smd_done(self, setup) -> None:
        self.smd_btn.setEnabled(True)
        notes = "".join(f"\n  {n}" for n in setup.notes)
        self.status.setText(self.status.text().replace("\nWriting LAMMPS setup...", "")
                            + f"\nLAMMPS SMD setup written to {setup.out_dir} (README.txt has the run commands)"
                            + notes)

    def _smd_failed(self, msg: str) -> None:
        self.smd_btn.setEnabled(True)
        self.status.setText(self.status.text().replace("\nWriting LAMMPS setup...", ""))
        QMessageBox.warning(self, "LAMMPS SMD setup", msg)
