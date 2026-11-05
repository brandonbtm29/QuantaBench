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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import resources
from .. import settings as settings_mod
from ..chem import engines


def method_keys() -> list[str]:
    """'auto' and every registered engine (plugins included), grouped by family."""
    return engines.method_keys()


def method_label(key: str) -> str:
    if key == "auto":
        return "Auto (best available)"
    info = engines.all_engines()[key]
    ok, why = engines.available(key)
    return info.label + ("" if ok else "  (not installed)")


class SettingsDialog(QDialog):
    def __init__(self, s: settings_mod.Settings, report: resources.HardwareReport | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("QuantaBench Settings")
        self.setMinimumWidth(560)
        self.setMinimumHeight(640)   # the Engines tab: four paths, the detection report and run options
        self._closed = False
        self.s = s
        tabs = QTabWidget()
        tabs.addTab(self._resources_tab(report), "Compute")
        tabs.addTab(self._build_tab(), "Build defaults")
        tabs.addTab(self._engines_tab(), "Engines")
        tabs.addTab(self._folders_tab(), "Folders")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(buttons)
        # the lane plan depends on the threads, the GPU and the method (Build defaults tab)
        self.lanes.currentTextChanged.connect(self._update_lane_plan)
        self.threads.valueChanged.connect(self._update_lane_plan)
        self.gpu.currentIndexChanged.connect(self._update_lane_plan)
        self.gpu_index.currentIndexChanged.connect(self._update_lane_plan)
        self.method.currentIndexChanged.connect(self._update_lane_plan)
        self._update_lane_plan()

    #: suggestions in the editable "Parallel lanes" box
    LANE_PRESETS = ("1", "auto", "2", "4", "6", "gpu:1x4,cpu:5x2")

    def _lane_config(self):
        from ..chem.pipeline import BuildConfig

        b = self.s.build
        return BuildConfig(method=self.method.currentData() or b.refine_method, threads=self.threads.value(),
                           gpu=self.gpu.currentText(), gpu_index=int(self.gpu_index.currentData() or 0),
                           conformer_search=b.conformer_search, qm_method=b.qm_method, solvent=b.solvent)

    def _update_lane_plan(self, *_args) -> None:
        """Show what the lane spec means on this machine (no torch import: uses the detection report if any)."""
        from ..dist import lanes as lanes_mod

        text = self.lanes.currentText().strip() or "1"
        self.lane_plan.setText(lanes_mod.plan_text(text, self._lane_config(), self._report,
                                                   self.s.resources.parallel_molecules))

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
        self.dxtb_gpu = QComboBox()
        self.dxtb_gpu.addItems(["auto", "on", "off"])
        self.dxtb_gpu.setCurrentText(getattr(r, "dxtb_gpu", "auto") or "auto")
        self.dxtb_gpu.setToolTip("auto: dxtb (GFN1/GFN2-xTB) uses the GPU only where it has the linear algebra dxtb\n"
                                 "needs (CUDA); on AMD's Windows ROCm build and Apple GPUs it measured 5-8x slower\n"
                                 "than the CPU, so it stays on the CPU. on: use the GPU anyway; off: always the CPU.")
        form.addRow("dxtb on the GPU", self.dxtb_gpu)
        self.linalg = QComboBox()
        self.linalg.addItems(["auto", "cpu", "device"])
        self.linalg.setCurrentText(getattr(r, "linalg", "auto") or "auto")
        self.linalg.setToolTip("Dense linear algebra a GPU build lacks (Cholesky on Windows ROCm, most of it on MPS):\n"
                               "auto: run just those operations on the CPU; cpu: all of them on the CPU;\n"
                               "device: never route (errors where the GPU lacks them).")
        form.addRow("GPU linear algebra", self.linalg)
        self._report = report
        self.lanes = QComboBox()
        self.lanes.setEditable(True)
        self.lanes.addItems(self.LANE_PRESETS)
        self.lanes.setCurrentText(self.s.dist.lanes or "1")
        self.lanes.setToolTip("Batches run in this many worker processes, each on its own molecule:\n"
                              "1 = one process with all threads; auto = from the hardware and the method;\n"
                              "N = N CPU lanes sharing the threads; or a list such as gpu:1x4,cpu:5x2\n"
                              "(1 GPU lane with 4 threads + 5 CPU lanes with 2 threads)")
        form.addRow("Parallel lanes", self.lanes)
        self.lane_plan = QLabel()
        self.lane_plan.setWordWrap(True)
        form.addRow("", self.lane_plan)
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
        methods = method_keys()
        for m in methods:
            self.method.addItem(method_label(m), m)
        self.method.setCurrentIndex(max(0, methods.index(b.refine_method) if b.refine_method in methods else 0))
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

    #: (settings key, label, file-dialog title) of the external programs QuantaBench can run
    PROGRAM_FIELDS = (("xtb", "xtb", "xtb executable"), ("crest", "CREST", "crest executable"),
                      ("mopac", "MOPAC", "MOPAC executable"), ("psi4_python", "Psi4 Python", "Python with psi4"))

    def _engines_tab(self) -> QWidget:
        """Paths of external programs (blank = find automatically), scratch folder, OpenMP stack, QM memory."""
        w = QWidget()
        lay = QVBoxLayout(w)
        e = self.s.engines
        box = QGroupBox("External programs")
        outer = QVBoxLayout(box)
        hint = QLabel("Leave blank to search PATH and the usual conda/mamba environments.<br>"
                      "On Windows, <code>wsl:/full/path/to/crest</code> runs a Linux build inside WSL.")
        outer.addWidget(hint)
        form = QFormLayout()
        outer.addLayout(form)
        self.program_paths: dict[str, QLineEdit] = {}
        for key, label, title in self.PROGRAM_FIELDS:
            edit = QLineEdit(e.paths.get(key, ""))
            edit.setPlaceholderText("automatic")
            browse = QPushButton("...")
            browse.setFixedWidth(32)
            browse.clicked.connect(lambda _c=False, ed=edit, t=title: self._browse_program(ed, t))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(browse)
            form.addRow(label, row)
            self.program_paths[key] = edit
        self.program_report = QPlainTextEdit()      # a fixed-height box: long paths scroll instead of resizing
        self.program_report.setReadOnly(True)
        self.program_report.setFixedHeight(self.fontMetrics().lineSpacing() * 5 + 12)
        self.program_report.setPlaceholderText("Press Detect now to see which programs QuantaBench finds.")
        detect = QPushButton("Detect now")
        detect.clicked.connect(self._detect_programs)
        outer.addWidget(detect)
        outer.addWidget(self.program_report)
        lay.addWidget(box)

        run = QGroupBox("Runs")
        rf = QFormLayout(run)
        self.scratch = QLineEdit(e.scratch_dir)
        self.scratch.setPlaceholderText("system temp folder / QuantaBench")
        rf.addRow("Scratch folder", self.scratch)
        self.keep_scratch = QCheckBox("Keep job folders of successful runs (for debugging)")
        self.keep_scratch.setChecked(e.keep_scratch)
        rf.addRow(self.keep_scratch)
        self.stacksize = QLineEdit(e.omp_stacksize)
        self.stacksize.setToolTip("OMP_STACKSIZE for xtb, CREST, MOPAC and Psi4 (reserves address space per thread)")
        rf.addRow("OpenMP stack size", self.stacksize)
        self.qm_memory = QDoubleSpinBox()
        self.qm_memory.setRange(0.5, 512)
        self.qm_memory.setSuffix(" GB")
        self.qm_memory.setValue(e.qm_memory_gb)
        rf.addRow("Psi4 memory per job", self.qm_memory)
        self.ext_timeout = QDoubleSpinBox()
        self.ext_timeout.setRange(0, 7 * 24 * 3600)
        self.ext_timeout.setDecimals(0)
        self.ext_timeout.setSuffix(" s")
        self.ext_timeout.setSpecialValueText("No limit")
        self.ext_timeout.setValue(e.timeout_s)
        rf.addRow("Time limit per run", self.ext_timeout)
        lay.addWidget(run)
        lay.addStretch(1)
        return w

    def _browse_program(self, edit: QLineEdit, title: str) -> None:
        start = edit.text() if edit.text() and not edit.text().startswith("wsl:") else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, str(Path(start).expanduser()))
        if path:
            edit.setText(settings_mod.portable(path))

    def _detect_programs(self) -> None:
        """Check the paths typed here (not saved yet), or search automatically for blank ones; off the GUI thread."""
        from ..chem import external

        typed = {k: ed.text().strip() for k, ed in self.program_paths.items()}

        def detect():
            rows = []
            for name in ("xtb", "crest", "mopac"):
                if typed.get(name):
                    info = external.inspect_path(name, typed[name])
                    rows.append((name, info, "typed path does not exist" if info is None else ""))
                else:
                    rows.append((name, external.find_program(name, refresh=True), ""))
            py = typed.get("psi4_python") or external.find_python_with("psi4", "psi4_python") or ""
            version = external.python_module_version(py, "psi4") if py else None
            return rows, py, version

        def done(result):
            if self._closed:
                return                    # the dialog was closed while detection ran: drop the result
            rows, py, version = result
            lines = [f"{n}: " + (f"{i.version or 'version ?'} at {i.path} ({i.source})" if i
                                 else (why or "not found")) for n, i, why in rows]
            lines.append("Psi4: " + (f"{version} via {py}" if version else
                                     (f"{py} cannot import psi4" if py else "not found")))
            self.program_report.setPlainText("\n".join(lines))

        def failed(msg):
            if not self._closed:
                self.program_report.setPlainText(msg.splitlines()[0])

        self.program_report.setPlainText("Searching...")
        from . import workers

        # The main window owns the thread (it waits for its workers before closing), so closing this dialog
        # never blocks on a slow probe (a WSL start) and no QThread is destroyed while running.
        owner = self.parent() if self.parent() is not None else self
        workers.start(workers.FunctionWorker(detect), owner, done, failed)

    def done(self, result: int) -> None:
        self._closed = True
        super().done(result)

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
        s.resources.dxtb_gpu = self.dxtb_gpu.currentText()
        s.resources.linalg = self.linalg.currentText()
        from ..dist import lanes as lanes_mod

        lanes = self.lanes.currentText().strip().lower() or "1"
        if not lanes_mod.spec_error(lanes):       # a malformed spec keeps the previous one (the plan line says why)
            s.dist.lanes = lanes
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
        e = s.engines
        e.paths = {k: ed.text().strip() for k, ed in self.program_paths.items() if ed.text().strip()}
        e.scratch_dir = self.scratch.text().strip()
        e.keep_scratch = self.keep_scratch.isChecked()
        e.omp_stacksize = self.stacksize.text().strip() or "4G"
        e.qm_memory_gb = self.qm_memory.value()
        e.timeout_s = self.ext_timeout.value()
        return s
