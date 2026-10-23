"""Settings dialog: compute resources, build defaults, folders."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import resources
from .. import settings as settings_mod
from ..chem import engines

METHODS = ["auto", "aimnet2", "gfn2-xtb", "gfn1-xtb", "mmff"]


def method_label(key: str) -> str:
    if key == "auto":
        return "Auto (best available)"
    info = engines.ENGINES[key]
    ok, why = engines.available(key)
    return info.label + ("" if ok else "  (not installed)")


class SettingsDialog(QDialog):
    def __init__(self, s: settings_mod.Settings, report: resources.HardwareReport | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QuantaBench Settings")
        self.setMinimumWidth(560)
        self.s = s
        tabs = QTabWidget()
        tabs.addTab(self._resources_tab(report), "Compute")
        tabs.addTab(self._build_tab(), "Build defaults")
        tabs.addTab(self._folders_tab(), "Folders")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(buttons)

    # ------------------------------------------------------------------ tabs
    def _resources_tab(self, report) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        r = self.s.resources
        logical = resources.logical_cores()
        self.threads = QSpinBox()
        self.threads.setRange(0, logical)
        self.threads.setSpecialValueText(f"Auto ({resources.resolve_threads(0)} of {logical})")
        self.threads.setValue(r.threads)
        form.addRow("CPU threads", self.threads)
        self.gpu = QComboBox()
        self.gpu.addItems(["auto", "on", "off"])
        self.gpu.setCurrentText(r.gpu)
        self.gpu.setToolTip("auto: use the GPU when PyTorch sees one; on: require it; off: CPU only")
        form.addRow("GPU", self.gpu)
        self.gpu_index = QComboBox()
        gpus = report.gpus if report else []
        for g in gpus:
            self.gpu_index.addItem(f"{g.index}: {g.name} ({g.total_memory_gb:.0f} GB, {g.backend.upper()})", g.index)
        if not gpus:
            self.gpu_index.addItem("No GPU visible to PyTorch", 0)
        self.gpu_index.setCurrentIndex(max(0, min(r.gpu_index, self.gpu_index.count() - 1)))
        form.addRow("Device", self.gpu_index)
        self.mem = QDoubleSpinBox()
        self.mem.setRange(0.1, 1.0)
        self.mem.setSingleStep(0.05)
        self.mem.setValue(r.gpu_memory_fraction)
        self.mem.setToolTip("Upper limit on the share of GPU memory PyTorch may allocate")
        form.addRow("GPU memory cap", self.mem)
        if report:
            info = QLabel(report.summary() + ("\n" + "\n".join(report.notes) if report.notes else ""))
            info.setWordWrap(True)
            form.addRow(info)
        return w

    def _build_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        b = self.s.build
        self.method = QComboBox()
        for m in METHODS:
            self.method.addItem(method_label(m), m)
        self.method.setCurrentIndex(max(0, METHODS.index(b.refine_method) if b.refine_method in METHODS else 0))
        form.addRow("Refinement method", self.method)
        self.ph = QDoubleSpinBox()
        self.ph.setRange(0, 14)
        self.ph.setSingleStep(0.5)
        self.ph.setValue(b.ph)
        form.addRow("pH", self.ph)
        self.protonate = QCheckBox("Assign protonation state at this pH")
        self.protonate.setChecked(b.protonate)
        form.addRow(self.protonate)
        self.desalt = QCheckBox("Keep largest fragment (strip counter-ions)")
        self.desalt.setChecked(b.keep_largest_fragment)
        form.addRow(self.desalt)
        self.max_confs = QSpinBox()
        self.max_confs.setRange(1, 1000)
        self.max_confs.setValue(b.max_conformers)
        form.addRow("Max conformers", self.max_confs)
        self.window = QDoubleSpinBox()
        self.window.setRange(0.5, 50)
        self.window.setSuffix(" kcal/mol")
        self.window.setValue(b.energy_window_kcal)
        form.addRow("Energy window", self.window)
        self.rmsd = QDoubleSpinBox()
        self.rmsd.setRange(0.05, 3.0)
        self.rmsd.setSingleStep(0.05)
        self.rmsd.setSuffix(" Å")
        self.rmsd.setValue(b.rmsd_threshold)
        form.addRow("Duplicate RMSD", self.rmsd)
        self.fmax = QDoubleSpinBox()
        self.fmax.setDecimals(3)
        self.fmax.setRange(0.001, 0.5)
        self.fmax.setSingleStep(0.005)
        self.fmax.setSuffix(" eV/Å")
        self.fmax.setValue(b.fmax)
        form.addRow("Force convergence", self.fmax)
        return w

    def _folders_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        out_box = QGroupBox("Output folder")
        ol = QHBoxLayout(out_box)
        self.output = QLineEdit(self.s.output_dir)
        self.output.setPlaceholderText(str(settings_mod.Settings().output_path()))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_output)
        ol.addWidget(self.output)
        ol.addWidget(browse)
        lay.addWidget(out_box)

        set_box = QGroupBox("Settings folder")
        sl = QVBoxLayout(set_box)
        info = QLabel("Where settings.json lives. Pick a folder inside SyncThing (for example "
                      "~/SyncThing/QuantaBench/Settings) to share settings between computers.")
        info.setWordWrap(True)
        sl.addWidget(info)
        self.settings_label = QLabel()
        self.settings_label.setWordWrap(True)
        sl.addWidget(self.settings_label)
        row = QHBoxLayout()
        for text, fn in (("Change...", self._change_settings_dir), ("Use default", self._default_settings_dir),
                         ("Open folder", lambda: QDesktopServices.openUrl(
                             QUrl.fromLocalFile(str(settings_mod.settings_dir()))))):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            row.addWidget(btn)
        sl.addLayout(row)
        lay.addWidget(set_box)
        lay.addStretch(1)
        self._refresh_settings_label()
        return w

    # ------------------------------------------------------------------ actions
    def _browse_output(self) -> None:
        start = self.output.text() or str(settings_mod.Settings().output_path())
        path = QFileDialog.getExistingDirectory(self, "Output folder", str(Path(start).expanduser()))
        if path:
            self.output.setText(settings_mod.portable(path))

    def _refresh_settings_label(self) -> None:
        cur = settings_mod.settings_dir()
        tag = "  (default)" if cur == settings_mod.default_settings_dir() else ""
        self.settings_label.setText(f"<b>Current:</b> {cur}{tag}")

    def _change_settings_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Settings folder", str(settings_mod.settings_dir()))
        if path:
            self._apply_settings_dir(path)

    def _default_settings_dir(self) -> None:
        self._apply_settings_dir(None)

    def _apply_settings_dir(self, path) -> None:
        try:
            new = settings_mod.set_settings_dir(path)
        except OSError as exc:
            QMessageBox.warning(self, "Settings folder", f"Could not change the settings folder:\n{exc}")
            return
        self._refresh_settings_label()
        QMessageBox.information(self, "Settings folder", f"Settings now live in:\n{new}\n\n"
                                "Settings already in that folder are used from now on.")

    def result_settings(self) -> settings_mod.Settings:
        s = self.s
        s.resources.threads = self.threads.value()
        s.resources.gpu = self.gpu.currentText()
        s.resources.gpu_index = int(self.gpu_index.currentData() or 0)
        s.resources.gpu_memory_fraction = self.mem.value()
        b = s.build
        b.refine_method = self.method.currentData()
        b.ph = self.ph.value()
        b.protonate = self.protonate.isChecked()
        b.keep_largest_fragment = self.desalt.isChecked()
        b.max_conformers = self.max_confs.value()
        b.energy_window_kcal = self.window.value()
        b.rmsd_threshold = self.rmsd.value()
        b.fmax = self.fmax.value()
        s.output_dir = self.output.text().strip()
        return s
