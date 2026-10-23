"""Surface builder: graphite slab with the selected molecule placed above it (PFAS-on-carbon systems)."""

from __future__ import annotations

from pathlib import Path

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


class SurfaceDialog(QDialog):
    def __init__(self, molecule_atoms=None, molecule_name: str = "", preview_cb=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Surface Builder: graphite slab")
        self.molecule = molecule_atoms
        self.preview_cb = preview_cb
        self.system = None
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
        preview = QPushButton("Build && Preview")
        preview.clicked.connect(self.build)
        save = QPushButton("Save...")
        save.clicked.connect(self.save)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.addButton(preview, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(save, QDialogButtonBox.ButtonRole.ActionRole)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.status)
        lay.addWidget(buttons)
        self._update_size()

    def _update_size(self) -> None:
        self.size_label.setText(f"= {self.nx.value() * surfaces.A_GRAPHENE:.1f} x "
                                f"{self.ny.value() * surfaces.B_RECT:.1f} Å")

    def build(self):
        slab = surfaces.graphite_slab(self.nx.value(), self.ny.value(), self.layers.value(), self.vacuum.value(),
                                      self.fix.value())
        system = slab
        if self.molecule is not None:
            system = surfaces.place_on_surface(slab, self.molecule, self.height.value(), self.orientation.currentText())
        clashes = surfaces.min_distance_report(system, 1.0)
        gap = surfaces.surface_adsorbate_gap(system) if self.molecule is not None else float("nan")
        msg = f"{len(system)} atoms, cell {system.cell.lengths().round(2).tolist()} Å."
        if self.molecule is not None:
            msg += f" Closest surface-molecule distance {gap:.2f} Å."
        msg += f" {len(clashes)} overlapping pairs (<1 Å)." if clashes else " No overlaps."
        self.status.setText(msg)
        self.system = system
        if self.preview_cb:
            self.preview_cb(system)
        return system

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
