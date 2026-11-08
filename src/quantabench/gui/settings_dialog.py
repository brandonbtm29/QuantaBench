"""Settings dialog: compute resources, build defaults, import options, surfaces and radii, display (energy units),
engines, folders."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
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
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import resources, units
from .. import settings as settings_mod
from ..chem import engines
from .energy_widgets import EnergySpinBox


def method_keys() -> list[str]:
    """'auto' and every registered engine (plugins included), grouped by family."""
    return engines.method_keys()


def method_label(key: str) -> str:
    if key == "auto":
        return "Auto (best available)"
    info = engines.all_engines()[key]
    ok, _why = engines.available(key, quick=True)     # no torch import for a menu label
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
        tabs.addTab(self._import_tab(), "Import")
        tabs.addTab(self._surfaces_tab(), "Surfaces && radii")    # "&&": a literal & (not a shortcut)
        tabs.addTab(self._display_tab(), "Display")
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
            known = report is not None and report.torch_backend != "unknown"   # the probe may still be running
            self.gpu_index.addItem("No GPU visible to PyTorch" if known else "GPU not detected yet", 0)
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
        self.lanes.setCurrentText(self.s.lanes_for())         # this machine's spec (the file is shared)
        self.lanes.setToolTip("Batches run in this many worker processes, each on its own molecule:\n"
                              "1 = one process with all threads; auto = from the hardware and the method;\n"
                              "N = N CPU lanes sharing the threads; or a list such as gpu:1x4,cpu:5x2\n"
                              "(1 GPU lane with 4 threads + 5 CPU lanes with 2 threads)")
        form.addRow("Parallel lanes", self.lanes)
        self.lane_plan = QLabel()
        self.lane_plan.setWordWrap(True)
        form.addRow("", self.lane_plan)
        self.speed_aware = QCheckBox("Speed-aware lanes: leave the last molecules to faster lanes")
        self.speed_aware.setChecked(bool(getattr(r, "speed_aware_lanes", True)))
        self.speed_aware.setToolTip("With lanes of different speed (a GPU lane next to CPU lanes), a slower lane does\n"
                                    "not start one of the last molecules when the faster lanes would finish all of\n"
                                    "them sooner. Off: every free lane takes the next molecule.")
        form.addRow("", self.speed_aware)
        self.dynamic_threads = QCheckBox("Dynamic CREST threads: a CREST search may use idle lanes' threads")
        self.dynamic_threads.setChecked(bool(getattr(r, "dynamic_threads", True)))
        self.dynamic_threads.setToolTip("With CREST conformers and several lanes, a CREST search started while other\n"
                                        "lanes are idle may use their threads too (in whole lanes, when that is\n"
                                        "predicted to end the batch sooner). The threads go back when the search\n"
                                        "ends; the refinement after it runs on the lane's own threads.")
        form.addRow("", self.dynamic_threads)
        self.memory_budget = QDoubleSpinBox()
        self.memory_budget.setRange(-1.0, 1024.0)
        self.memory_budget.setSingleStep(0.5)
        self.memory_budget.setDecimals(1)
        self.memory_budget.setSuffix(" GB")
        self.memory_budget.setSpecialValueText("Off (no memory guard)")
        self.memory_budget.setValue(max(-1.0, float(getattr(r, "memory_budget_gb", 0.0) or 0.0)))
        self.memory_budget.setToolTip("Memory each process (lane) may use for the dxtb steps (GFN1/GFN2-xTB refinement,\n"
                                      "xTB properties, frequencies). A step estimated to go over it takes a path with\n"
                                      "less memory (tblite polarizability, finite-difference Hessian) or is skipped with\n"
                                      "a note, instead of the process being killed for running out of memory.\n"
                                      "0 = auto (the available memory divided by the lanes, at least 1.5 GB); Off = no guard.")
        form.addRow("Memory per lane", self.memory_budget)
        self.molecule_timeout = QSpinBox()
        self.molecule_timeout.setRange(0, 7 * 86400)
        self.molecule_timeout.setSingleStep(300)
        self.molecule_timeout.setSuffix(" s")
        self.molecule_timeout.setSpecialValueText("Off (no time limit)")
        self.molecule_timeout.setValue(int(round(float(getattr(r, "molecule_timeout_s", 1800.0) or 0.0))))
        self.molecule_timeout.setToolTip("Wall-clock limit per molecule in a batch. A lane still on a molecule past it is\n"
                                         "restarted; one process stops the molecule at its next checkpoint. The molecule\n"
                                         "is marked failed 'time limit (N s)' and --retry-failed redoes it. The conformer\n"
                                         "embedding gets a quarter of it and keeps the conformers it has by then.\n"
                                         "Off = no limit.")
        form.addRow("Time limit per molecule", self.molecule_timeout)
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
        # stored in kcal/mol (a science setting), shown in the relative-energy unit of the Display tab
        self.window = EnergySpinBox(0.5, 50, units.KCAL, units.EnergyDisplay.of(self.s).relative)
        self.window.set_stored(b.energy_window_kcal)
        self.window.setToolTip("Conformers more than this above the lowest one are dropped. Shown in the relative-energy\n"
                               "unit (Settings > Display); runs store it in kcal/mol (energy_window_kcal).")
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

    def _import_tab(self) -> QWidget:
        """The stored choices of File > Open's Import Options dialog (``Settings.import_``), also the defaults of
        the command line's --file inputs."""
        from ..chem.import_options import ImportOptions
        from .import_options import ImportOptionsWidget

        w = QWidget()
        lay = QVBoxLayout(w)
        im = self.s.import_
        self.import_ask = QCheckBox("Show the Import Options dialog when a file has choices")
        self.import_ask.setToolTip("2D structures, data fields, several fragments or several structures of one\n"
                                   "constitution. Off: these choices are used without asking.")
        self.import_ask.setChecked(im.ask)
        lay.addWidget(self.import_ask)
        self.import_form = ImportOptionsWidget(ImportOptions.from_settings(im),
                                               [f for f in (im.id_field, im.name_field) if f])
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.import_form)
        lay.addWidget(scroll, 1)
        note = QLabel("<code>quantabench-cli build --file</code> uses these too; its import flags "
                      "(<code>--keep-coords</code>, <code>--filter</code>, ...) override them.")
        note.setWordWrap(True)
        lay.addWidget(note)
        return w

    #: columns of the radii table
    RADII_COLUMNS = ("Element", "Z", "Bondi (Å)", "Spartan (Å)", "Your value (Å)")
    _OVERRIDE_COL = 4

    def _surfaces_tab(self) -> QWidget:
        """Radii set and per-element radii (Spartan's Preferences > VDW Radii), and the SASA probe radius."""
        from ase.data import chemical_symbols

        from ..chem import radii as radii_mod

        w = QWidget()
        lay = QVBoxLayout(w)
        sf = self.s.surfaces
        box = QGroupBox("Areas and volumes")
        form = QFormLayout(box)
        self.radii_set = QComboBox()
        for key in radii_mod.SETS:
            self.radii_set.addItem(radii_mod.LABELS[key], key)
        self.radii_set.setCurrentIndex(max(0, self.radii_set.findData((sf.radii_set or "bondi").lower())))
        self.radii_set.setToolTip("Radii of the atom spheres for sasa_A2, vdw_area_A2 and vdw_volume_A3.\n"
                                  "cpk_area_A2 and cpk_volume_A3 always use Spartan's default radii, so they\n"
                                  "compare with Spartan's CPK area and volume whatever is chosen here.")
        form.addRow("Radii set", self.radii_set)
        self.sasa_probe = QDoubleSpinBox()
        self.sasa_probe.setRange(0.0, 3.0)
        self.sasa_probe.setDecimals(2)
        self.sasa_probe.setSingleStep(0.1)
        self.sasa_probe.setSuffix(" Å")
        self.sasa_probe.setValue(float(sf.sasa_probe_A))
        self.sasa_probe.setToolTip("Solvent probe radius rolled over the atom spheres for sasa_A2: 1.4 Å for water.\n"
                                   "Spartan's accessible area uses 1.0 Å on its density surface, not on spheres.")
        form.addRow("SASA probe radius", self.sasa_probe)
        note = QLabel("Changes apply to molecules built from now on; finished runs keep their values. "
                      "The ESP polar areas use Spartan's fixed thresholds, |V| > 75, 100 and 125 kJ/mol.")
        note.setWordWrap(True)
        form.addRow(note)
        lay.addWidget(box)

        rbox = QGroupBox("Van der Waals radii")
        rl = QVBoxLayout(rbox)
        hint = QLabel("Type a radius in <i>Your value</i> to replace the set's radius for that element; "
                      "leave it blank to use the set's.")
        hint.setWordWrap(True)
        rl.addWidget(hint)
        self.radii_filter = QLineEdit()
        self.radii_filter.setPlaceholderText("Find an element (symbol or Z)")
        self.radii_filter.textChanged.connect(self._filter_radii)
        rl.addWidget(self.radii_filter)
        bondi, spartan = radii_mod.table("bondi"), radii_mod.table("spartan")
        over = radii_mod.normalize_overrides(sf.radii_overrides)
        self.radii_table = QTableWidget(len(bondi), len(self.RADII_COLUMNS))
        self.radii_table.setHorizontalHeaderLabels(list(self.RADII_COLUMNS))
        self.radii_table.verticalHeader().setVisible(False)
        self.radii_table.verticalHeader().setDefaultSectionSize(self.fontMetrics().lineSpacing() + 8)
        self._radii_loading = True
        for row, z in enumerate(sorted(bondi)):
            cells = (chemical_symbols[z], str(z), f"{bondi[z]:.4f}", f"{spartan[z]:.4f}",
                     f"{over[z]:g}" if z in over else "")
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col != self._OVERRIDE_COL:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.radii_table.setItem(row, col, item)
        self._radii_loading = False
        self.radii_table.itemChanged.connect(self._radius_edited)
        self.radii_table.resizeColumnsToContents()
        self.radii_table.horizontalHeader().setStretchLastSection(True)
        rl.addWidget(self.radii_table, 1)
        row = QHBoxLayout()
        self.radii_status = QLabel()
        row.addWidget(self.radii_status, 1)
        for text, fn in (("Reset selected", self._reset_selected_radii), ("Reset all", self._reset_all_radii)):
            btn = QPushButton(text)
            btn.clicked.connect(fn)
            row.addWidget(btn)
        rl.addLayout(row)
        lay.addWidget(rbox, 1)
        self.radii_set.currentIndexChanged.connect(self._mark_radii_set)
        self._mark_radii_set()
        return w

    @staticmethod
    def _parse_radius(text: str) -> float | None:
        try:
            r = float(text.strip().replace(",", "."))
        except ValueError:
            return None
        return r if 0 < r <= 10 else None

    def _radius_edited(self, item) -> None:
        if self._radii_loading or item.column() != self._OVERRIDE_COL:
            return
        text = item.text().strip()
        bad = bool(text) and self._parse_radius(text) is None
        item.setForeground(Qt.GlobalColor.red if bad else self.radii_table.palette().text().color())
        item.setToolTip("Not a radius in Å (0-10): ignored" if bad else "")
        self._update_radii_status()

    def _mark_radii_set(self, *_args) -> None:
        """Bold the column of the set in use."""
        active = 2 if self.radii_set.currentData() == "bondi" else 3
        for col in (2, 3):
            head = self.radii_table.horizontalHeaderItem(col)
            font = head.font()
            font.setBold(col == active)
            head.setFont(font)
        self._update_radii_status()

    def _update_radii_status(self) -> None:
        n = len(self.radii_overrides())
        name = self.radii_set.currentText()
        self.radii_status.setText(f"{name}" + (f", {n} element{'s' if n != 1 else ''} changed" if n else ""))

    def _filter_radii(self, text: str) -> None:
        t = text.strip().lower()
        for row in range(self.radii_table.rowCount()):
            sym, z = self.radii_table.item(row, 0).text().lower(), self.radii_table.item(row, 1).text()
            self.radii_table.setRowHidden(row, bool(t) and not (sym.startswith(t) or z == t))

    def _reset_selected_radii(self) -> None:
        for row in sorted({i.row() for i in self.radii_table.selectedIndexes()}):
            self.radii_table.item(row, self._OVERRIDE_COL).setText("")

    def _reset_all_radii(self) -> None:
        for row in range(self.radii_table.rowCount()):
            self.radii_table.item(row, self._OVERRIDE_COL).setText("")

    def radii_overrides(self) -> dict[str, float]:
        """``{element symbol: Å}`` of the rows with a valid value typed in *Your value*."""
        out: dict[str, float] = {}
        for row in range(self.radii_table.rowCount()):
            r = self._parse_radius(self.radii_table.item(row, self._OVERRIDE_COL).text())
            if r is not None:
                out[self.radii_table.item(row, 0).text()] = r
        return out

    def _display_tab(self) -> QWidget:
        """Energy units and number format for the GUI's tables and panels (Spartan's Preferences > Document)."""
        w = QWidget()
        lay = QVBoxLayout(w)
        shown = units.EnergyDisplay.of(self.s)
        box = QGroupBox("Energy units")
        form = QFormLayout(box)
        self.energy_unit, self.relative_unit = QComboBox(), QComboBox()
        for combo, current in ((self.energy_unit, shown.absolute), (self.relative_unit, shown.relative)):
            for key in units.UNITS:
                combo.addItem(units.LABELS[key], key)
            combo.setCurrentIndex(max(0, combo.findData(current)))
        self.energy_unit.setToolTip("Total energies: the conformer table's E, the best energy in the molecule list,\n"
                                    "the QM energy, and H and G of the thermochemistry.")
        self.relative_unit.setToolTip("Energy differences: the conformers' ΔE and the energy window, ZPE, H − E and\n"
                                      "G − E (entropy follows: J/(mol·K) with kJ/mol), and the electrostatic potential\n"
                                      "on the Surfaces panel (its colour range, legend and descriptors).")
        form.addRow("Energies", self.energy_unit)
        form.addRow("Relative energies", self.relative_unit)
        lay.addWidget(box)

        nbox = QGroupBox("Numbers")
        nform = QFormLayout(nbox)
        self.sig_digits = QSpinBox()
        self.sig_digits.setRange(0, units.MAX_DIGITS)
        self.sig_digits.setSpecialValueText("Automatic")
        self.sig_digits.setValue(shown.digits)
        self.sig_digits.setToolTip("Automatic: each energy with QuantaBench's usual precision, converted to the unit\n"
                                   "(0.01 kcal/mol is 0.01 kJ/mol, 0.0001 eV). A number: that many significant digits.")
        nform.addRow("Significant digits", self.sig_digits)
        self.number_style = QComboBox()
        for key in units.STYLES:
            self.number_style.addItem(units.STYLE_LABELS[key], key)
        self.number_style.setCurrentIndex(max(0, self.number_style.findData(shown.style)))
        self.number_style.setToolTip("Decimal: never an exponent. Scientific: always (-2.62550e+06).\n"
                                     "Automatic: decimal, with an exponent only where the decimal form would hide\n"
                                     "the value (or, with significant digits, as a spreadsheet's General format).")
        nform.addRow("Number style", self.number_style)
        self.display_preview = QLabel()
        self.display_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        nform.addRow("Example", self.display_preview)
        lay.addWidget(nbox)
        note = QLabel("Display only: runs and exported files (CSV, JSON, SDF) keep their columns and units, such as "
                      "<code>rel_kcal</code> and <code>best_energy_eV</code>.")
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addStretch(1)
        for combo in (self.energy_unit, self.relative_unit, self.number_style):
            combo.currentIndexChanged.connect(self._update_display_preview)
        self.sig_digits.valueChanged.connect(self._update_display_preview)
        self._update_display_preview()
        return w

    #: (absolute E in eV, ΔE in kcal/mol) of the Display tab's example: a PFAS-sized total energy
    PREVIEW_VALUES = (-48276.81234, 1.2345)

    def display_settings(self) -> units.EnergyDisplay:
        """What the Display tab shows now (not saved yet)."""
        return units.EnergyDisplay(settings_mod.DisplaySettings(
            energy_unit=self.energy_unit.currentData(), relative_energy_unit=self.relative_unit.currentData(),
            significant_digits=self.sig_digits.value(), number_style=self.number_style.currentData()))

    def _update_display_preview(self, *_args) -> None:
        d = self.display_settings()
        e_ev, de_kcal = self.PREVIEW_VALUES
        self.display_preview.setText(f"E = {d.with_unit(e_ev, units.EV, 'absolute', 5)}, "
                                     f"ΔE = {d.with_unit(de_kcal, units.KCAL, 'relative', 2)}")
        self.window.set_unit(d.relative)          # Build defaults' energy window follows the relative unit

    #: (settings key, label, file-dialog title) of the external programs QuantaBench can run
    PROGRAM_FIELDS = (("xtb", "xtb", "xtb executable"), ("crest", "CREST", "crest executable"),
                      ("mopac", "MOPAC", "MOPAC executable"), ("psi4_python", "Psi4 Python", "Python with psi4"),
                      ("obabel", "Open Babel", "obabel executable (file formats QuantaBench does not read)"))

    def _engines_tab(self) -> QWidget:
        """Paths of external programs (blank = find automatically), scratch folder, OpenMP stack, QM memory."""
        w = QWidget()
        lay = QVBoxLayout(w)
        e = self.s.engines
        box = QGroupBox("External programs")
        outer = QVBoxLayout(box)
        machine = settings_mod.this_machine()
        hint = QLabel("Leave blank to search PATH and the usual conda/mamba environments.<br>"
                      "On Windows, <code>wsl:/full/path/to/crest</code> runs a Linux build inside WSL.<br>"
                      f"Paths are saved for this machine (<code>{machine}</code>) only: the settings file is shared "
                      "by every machine.")
        outer.addWidget(hint)
        form = QFormLayout()
        outer.addLayout(form)
        self.program_paths: dict[str, QLineEdit] = {}
        current = self.s.engine_paths_for()            # this machine's paths, else the shared defaults
        for key, label, title in self.PROGRAM_FIELDS:
            path, scope = current.get(key, ("", ""))
            edit = QLineEdit(path)
            edit.setPlaceholderText("automatic")
            if scope == "all machines":
                edit.setToolTip("From the default of every machine (engines.paths); a change is saved for this "
                                "machine only")
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
            for name in ("xtb", "crest", "mopac", "obabel"):
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
        s.resources.speed_aware_lanes = self.speed_aware.isChecked()
        s.resources.dynamic_threads = self.dynamic_threads.isChecked()
        budget = float(self.memory_budget.value())
        s.resources.memory_budget_gb = -1.0 if budget < 0 else budget       # below 0 is off
        s.resources.molecule_timeout_s = float(self.molecule_timeout.value())
        from ..dist import lanes as lanes_mod

        lanes = self.lanes.currentText().strip().lower() or "1"
        if not lanes_mod.spec_error(lanes):       # a malformed spec keeps the previous one (the plan line says why)
            s.set_lanes_for(lanes)                # this machine only: other machines keep their own spec
        b = s.build
        b.refine_method = self.method.currentData()
        b.ph = self.ph.value()
        b.protonate = self.protonate.isChecked()
        b.keep_largest_fragment = self.desalt.isChecked()
        b.max_conformers = self.max_confs.value()
        b.energy_window_kcal = self.window.stored()
        b.rmsd_threshold = self.rmsd.value()
        b.fmax = self.fmax.value()
        self.import_form.options().to_settings(s)
        s.import_.ask = self.import_ask.isChecked()
        sf = s.surfaces
        sf.radii_set = self.radii_set.currentData() or "bondi"
        sf.radii_overrides = self.radii_overrides()
        sf.sasa_probe_A = self.sasa_probe.value()
        d = s.display
        d.energy_unit = self.energy_unit.currentData()
        d.relative_energy_unit = self.relative_unit.currentData()
        d.significant_digits = self.sig_digits.value()
        d.number_style = self.number_style.currentData()
        s.output_dir = self.output.text().strip()
        e = s.engines
        for key, ed in self.program_paths.items():   # this machine only: other machines keep their own paths
            s.set_engine_path_for(key, ed.text().strip())
        e.scratch_dir = self.scratch.text().strip()
        e.keep_scratch = self.keep_scratch.isChecked()
        e.omp_stacksize = self.stacksize.text().strip() or "4G"
        e.qm_memory_gb = self.qm_memory.value()
        e.timeout_s = self.ext_timeout.value()
        return s
