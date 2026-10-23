"""Main window.

Layout borrows from OVITO: the structure views are the center of attention, the "pipeline" on the right
lists the processing stages (select one to edit its parameters, see its status for the selected
molecule), the molecule list sits on the left, and conformers + log share the bottom panel.

The center is a splitter with a 2D depiction (RDKit, left, narrower) and the 3D viewer (PyVista).
Both views hold the same :class:`~quantabench.gui.selection.SelectionModel`, so an atom picked in
either one is highlighted in both, and the measurement for the last 2-4 picked atoms is shown in the
status bar.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, resources
from .. import settings as settings_mod
from ..chem import engines, export
from ..chem import inputs as inputs_mod
from ..chem.pipeline import BuildConfig, MoleculeResult
from . import workers
from .selection import SelectionModel
from .settings_dialog import METHODS, SettingsDialog, method_label
from .viewer2d import MoleculeViewer2D
from .viewer3d import COLOR_MODES, LABEL_MODES, STYLES, MoleculeViewer3D

MODE_ACTIONS = (("rotate", "Rotate", "Ctrl+1", "Drag rotates the view (click selects, Ctrl+click adds)"),
                ("select", "Select", "Ctrl+2", "Drag draws a selection box"),
                ("move", "Move", "Ctrl+3", "Drag on a molecule rotates it rigidly, Shift+drag slides it "
                                           "(press on a selected atom to move only the selection)"))
BATCHED_CHOICES = (("auto", "Auto (batched on a GPU)"), ("on", "Always batched"), ("off", "One at a time (ASE)"))

STAGES = [
    ("input", "Input", "Structure source"),
    ("prepare", "Standardize && protonate", "Largest fragment, neutral parent, charge state at pH"),
    ("conformers", "Conformer search", "ETKDGv3 embedding, MMFF94s pre-optimization, pruning"),
    ("refine", "Refinement", "Re-optimize the lowest conformers with an ML potential or xTB"),
    ("properties", "Properties", "Boltzmann weights, descriptors, xTB charges/dipole/polarizability"),
    ("export", "Export", "Files written for every molecule"),
]


def _window_settings() -> QSettings:
    """Window geometry and dock layout (the registry on Windows). ``QUANTABENCH_WINDOW_STATE`` points it at
    an INI file instead, so the tests never overwrite the user's own layout."""
    ini = os.environ.get("QUANTABENCH_WINDOW_STATE")
    if ini:
        return QSettings(ini, QSettings.Format.IniFormat)
    return QSettings("QuantaBench", "QuantaBench")


STATUS_ICON = {"ok": "✔", "failed": "✖", "cancelled": "○", "pending": "…", "running": "▶"}


class MainWindow(QMainWindow):
    def __init__(self, s: settings_mod.Settings):
        super().__init__()
        self.s = s
        self.report: resources.HardwareReport | None = None
        self.inputs: list[inputs_mod.MoleculeInput] = []
        self.results: dict[str, MoleculeResult] = {}
        self.running: dict[str, str] = {}
        self.worker = None
        self.last_run_dir: Path | None = None
        self._closing = False
        self._quit_pending = False         # close() asked while workers run: closes when they finish
        self.closed = False
        self.setWindowTitle(f"QuantaBench {__version__}")
        self.resize(1500, 920)

        self.surface_dialog = None
        self._shown_key = None            # (molecule id, built result or None) in the views; see _views_key
        self._surface_preview = False     # the 3D view shows the Surface Builder's system, not the molecule
        self.selection = SelectionModel(self)
        self.viewer = MoleculeViewer3D(self, selection=self.selection)
        self.viewer2d = MoleculeViewer2D(self, selection=self.selection)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("views")
        self.splitter.addWidget(self.viewer2d)
        self.splitter.addWidget(self.viewer)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([380, 620])
        self.splitter.setCollapsible(1, False)
        self.setCentralWidget(self.splitter)
        self._build_molecule_dock()
        self._build_pipeline_dock()
        self._build_bottom_dock()
        self._build_toolbar_and_menus()
        self._build_statusbar()
        self._load_stage_values()
        self._restore_window()
        QTimer.singleShot(50, self._detect_hardware)

    # ================================================================== UI construction
    def _build_molecule_dock(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("SMILES, name, or CAS (Enter to add)")
        self.entry.returnPressed.connect(self.add_from_entry)
        add = QPushButton("Add")
        add.clicked.connect(self.add_from_entry)
        row.addWidget(self.entry)
        row.addWidget(add)
        lay.addLayout(row)
        self.mol_list = QListWidget()
        self.mol_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.mol_list.currentRowChanged.connect(self._on_molecule_selected)
        lay.addWidget(self.mol_list)
        btns = QHBoxLayout()
        for text, fn in (("Open file...", self.open_files), ("Remove", self.remove_selected), ("Clear", self.clear_all)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)
        self.mol_info = QLabel()
        self.mol_info.setWordWrap(True)
        self.mol_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.mol_info)
        dock = QDockWidget("Molecules", self)
        dock.setObjectName("molecules")
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.mol_dock = dock

    def _build_pipeline_dock(self) -> None:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.stage_tree = QTreeWidget()
        self.stage_tree.setHeaderLabels(["Pipeline stage", "Status"])
        self.stage_tree.setRootIsDecorated(False)
        self.stage_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.stage_items = {}
        for key, label, tip in STAGES:
            it = QTreeWidgetItem([label.replace("&&", "&"), ""])
            it.setToolTip(0, tip)
            it.setData(0, Qt.ItemDataRole.UserRole, key)
            self.stage_tree.addTopLevelItem(it)
            self.stage_items[key] = it
        self.stage_tree.setMaximumHeight(170)
        self.stage_tree.setMinimumHeight(150)
        lay.addWidget(self.stage_tree)
        self.stage_stack = QStackedWidget()
        self.stage_pages = {}
        for key, _label, tip in STAGES:
            page = self._stage_page(key, tip)
            self.stage_pages[key] = self.stage_stack.addWidget(page)
        scroll = QScrollArea()             # the Properties page can list dozens of descriptors
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.stage_stack)
        lay.addWidget(scroll, 1)
        self.stage_tree.currentItemChanged.connect(self._on_stage_selected)
        self.stage_tree.setCurrentItem(self.stage_items["refine"])
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run all")
        self.run_btn.clicked.connect(lambda: self.run_build(selected_only=False))
        self.run_sel_btn = QPushButton("Run selected")
        self.run_sel_btn.clicked.connect(lambda: self.run_build(selected_only=True))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_build)
        for b in (self.run_btn, self.run_sel_btn, self.cancel_btn):
            run_row.addWidget(b)
        lay.addLayout(run_row)
        dock = QDockWidget("Pipeline", self)
        dock.setObjectName("pipeline")
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.pipeline_dock = dock

    def _stage_page(self, key: str, tip: str) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        head = QLabel(tip)
        head.setWordWrap(True)
        head.setStyleSheet("color: gray;")
        form.addRow(head)
        if key == "input":
            self.input_detail = QLabel("Select a molecule.")
            self.input_detail.setWordWrap(True)
            self.input_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(self.input_detail)
            self.csv_smiles_col = QLineEdit()
            self.csv_smiles_col.setPlaceholderText("auto (SMILES, QSAR_READY_SMILES, ...)")
            self.csv_id_col = QLineEdit()
            self.csv_id_col.setPlaceholderText("auto (DTXSID, ID, Name, ...)")
            self.csv_sheet = QLineEdit()
            self.csv_sheet.setPlaceholderText("first sheet (name or 0-based index)")
            self.smiles_pref = QComboBox()
            for key, label in (("smiles", "SMILES, then QSAR-ready, then MS-ready"), ("qsar", "QSAR-ready first"),
                               ("ms", "MS-ready first")):
                self.smiles_pref.addItem(label, key)
            form.addRow("Table SMILES column", self.csv_smiles_col)
            form.addRow("Table ID column", self.csv_id_col)
            form.addRow("Excel sheet", self.csv_sheet)
            form.addRow("SMILES preference", self.smiles_pref)
        elif key == "prepare":
            self.w_desalt = QCheckBox("Keep largest fragment (strip counter-ions)")
            self.w_protonate = QCheckBox("Assign protonation state")
            self.w_ph = QDoubleSpinBox()
            self.w_ph.setRange(0, 14)
            self.w_ph.setSingleStep(0.5)
            self.w_prot_method = QComboBox()
            self.w_prot_method.addItem("Built-in rules (PFAS-aware)", "rules")
            self.w_prot_method.addItem("Dimorphite-DL" + ("" if resources.has_package("dimorphite_dl")
                                                          else " (not installed)"), "dimorphite")
            form.addRow(self.w_desalt)
            form.addRow(self.w_protonate)
            form.addRow("pH", self.w_ph)
            form.addRow("Method", self.w_prot_method)
        elif key == "conformers":
            self.w_maxconf = QSpinBox()
            self.w_maxconf.setRange(1, 1000)
            self.w_window = QDoubleSpinBox()
            self.w_window.setRange(0.5, 50)
            self.w_window.setSuffix(" kcal/mol")
            self.w_rmsd = QDoubleSpinBox()
            self.w_rmsd.setRange(0.05, 3)
            self.w_rmsd.setSingleStep(0.05)
            self.w_rmsd.setSuffix(" Å")
            self.w_seed = QSpinBox()
            self.w_seed.setRange(0, 2**31 - 1)
            form.addRow("Max conformers", self.w_maxconf)
            form.addRow("Energy window", self.w_window)
            form.addRow("Duplicate RMSD", self.w_rmsd)
            form.addRow("Random seed", self.w_seed)
        elif key == "refine":
            self.w_method = QComboBox()
            for m in METHODS:
                self.w_method.addItem(method_label(m), m)
            self.w_refine_top = QSpinBox()
            self.w_refine_top.setRange(1, 1000)
            self.w_refine_top.setValue(20)
            self.w_fmax = QDoubleSpinBox()
            self.w_fmax.setDecimals(3)
            self.w_fmax.setRange(0.001, 0.5)
            self.w_fmax.setSingleStep(0.005)
            self.w_fmax.setSuffix(" eV/Å")
            self.w_steps = QSpinBox()
            self.w_steps.setRange(10, 10000)
            self.w_optimizer = QComboBox()
            self.w_optimizer.addItems(["lbfgs", "fire"])
            self.w_batched = QComboBox()
            for key, label in BATCHED_CHOICES:
                self.w_batched.addItem(label, key)
            self.w_batched.setToolTip("Refine all conformers of a molecule in one torch batch (xTB, AIMNet2). "
                                      "Same results as one at a time; much faster on a GPU.")
            self.engine_info = QLabel()
            self.engine_info.setWordWrap(True)
            self.w_method.currentIndexChanged.connect(self._update_engine_info)
            form.addRow("Method", self.w_method)
            form.addRow("Refine lowest", self.w_refine_top)
            form.addRow("Force convergence", self.w_fmax)
            form.addRow("Max steps", self.w_steps)
            form.addRow("Optimizer", self.w_optimizer)
            form.addRow("Conformer batching", self.w_batched)
            form.addRow(self.engine_info)
        elif key == "properties":
            self.w_xtbprops = QCheckBox("GFN1-xTB charges, dipole, polarizability (best conformer)")
            self.w_xtbprops.setEnabled(resources.has_package("dxtb"))
            self.w_freq = QCheckBox("Frequencies + thermochemistry (best conformer)")
            self.w_freq.setToolTip("Hessian, normal modes, IR intensities, ZPE/H/S/G (ideal gas, qRRHO). "
                                   "Adds the IR spectrum to the exported files. Minutes per molecule on a CPU.")
            self.w_symmetry = QSpinBox()
            self.w_symmetry.setRange(0, 24)
            self.w_symmetry.setSpecialValueText("Auto (per molecule)")
            self.w_symmetry.setValue(0)
            self.w_symmetry.setToolTip("Rotational symmetry number σ. Auto detects it from each molecule's point group "
                                       "(1 for most PFAS, 2 for water, 12 for methane); a fixed value applies to "
                                       "every molecule in the run")
            self.w_temp = QDoubleSpinBox()
            self.w_temp.setRange(1, 2000)
            self.w_temp.setSuffix(" K")
            form.addRow(self.w_xtbprops)
            form.addRow(self.w_freq)
            form.addRow("Symmetry number", self.w_symmetry)
            form.addRow("Temperature", self.w_temp)
            self.props_view = QLabel()
            self.props_view.setWordWrap(True)
            self.props_view.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(self.props_view)
        elif key == "export":
            self.w_formats = {}
            for fmt, desc in export.FORMATS.items():
                cb = QCheckBox(f"{fmt}: {desc}")
                self.w_formats[fmt] = cb
                form.addRow(cb)
            self.out_label = QLabel()
            self.out_label.setWordWrap(True)
            form.addRow(self.out_label)
            open_btn = QPushButton("Open output folder")
            open_btn.clicked.connect(self.open_output_folder)
            form.addRow(open_btn)
        return page

    def _build_bottom_dock(self) -> None:
        tabs = QTabWidget()
        self.conf_table = QTableWidget(0, 7)
        self.conf_table.setHorizontalHeaderLabels(["#", "ΔE (kcal/mol)", "Weight (%)", "E (eV)", "Converged", "Steps",
                                                   "Dipole (D)"])
        self.conf_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.conf_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.conf_table.verticalHeader().setVisible(False)
        self.conf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.conf_table.currentCellChanged.connect(self._on_conformer_selected)
        tabs.addTab(self.conf_table, "Conformers")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(20000)
        tabs.addTab(self.log_view, "Log")
        self.bottom_tabs = tabs
        dock = QDockWidget("Results", self)
        dock.setObjectName("results")
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.bottom_dock = dock

    def _build_toolbar_and_menus(self) -> None:
        tb = QToolBar("Main")
        tb.setObjectName("main_toolbar")
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setToolTip(tip)
            return a

        self.a_open = act("Open...", self.open_files, "Ctrl+O", "Open SDF, MOL, XYZ, PDB, SMI or CSV files")
        self.a_run = act("Run", lambda: self.run_build(False), "F5", "Build every molecule in the list")
        # Not Esc: the views use Esc to clear the selection, and a window-wide Esc would cancel the batch
        self.a_cancel = act("Cancel", self.cancel_build, "Shift+Esc", "Stop the build after the current step "
                            "(Shift+Esc)")
        self.a_cancel.setEnabled(False)
        self.a_export = act("Export...", self.export_selected, "Ctrl+E", "Save the selected molecule")
        self.a_surface = act("Surface builder...", self.open_surface_builder, None,
                             "Graphite slab with the selected molecule on top")
        self.a_settings = act("Settings...", self.open_settings, "Ctrl+,")
        for a in (self.a_open, self.a_run, self.a_cancel, self.a_export, self.a_surface, self.a_settings):
            tb.addAction(a)
        vt = QToolBar("View")
        vt.setObjectName("view_toolbar")
        self.addToolBar(vt)
        self.view_toolbar = vt
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_actions = {}
        for key, text, shortcut, tip in MODE_ACTIONS:
            a = act(text, lambda _checked=False, k=key: self.set_view_mode(k), shortcut, f"{tip} ({shortcut})")
            a.setCheckable(True)
            self.mode_group.addAction(a)
            self.mode_actions[key] = a
            vt.addAction(a)
        self.mode_actions["rotate"].setChecked(True)
        vt.addSeparator()
        self.style_box = QComboBox()
        self.style_box.addItems(STYLES)
        self.style_box.currentTextChanged.connect(self.viewer.set_style)
        vt.addWidget(QLabel(" Style "))
        vt.addWidget(self.style_box)
        self.label_box = QComboBox()
        for m in LABEL_MODES:
            self.label_box.addItem(m.replace("+", " + ").capitalize(), m)
        self.label_box.setToolTip("Atom labels in both views (indices are 0-based, like ASE and RDKit)")
        self.label_box.currentIndexChanged.connect(self._on_label_mode)
        vt.addWidget(QLabel(" Labels "))
        vt.addWidget(self.label_box)
        self.color_box = QComboBox()
        for m in COLOR_MODES:
            self.color_box.addItem(m.capitalize(), m)
        self.color_box.setToolTip("Element (Jmol), Charge (red = negative, blue = positive), "
                                  "Tag (surface gray, adsorbate orange)")
        self.color_box.currentIndexChanged.connect(lambda _i: self.viewer.set_color_mode(self.color_box.currentData()))
        vt.addWidget(QLabel(" Color "))
        vt.addWidget(self.color_box)
        self.h_box = QCheckBox("Hydrogens")
        self.h_box.setChecked(True)
        self.h_box.setToolTip("Hydrogens in the 3D view (the 2D view: View > Hydrogens in 2D)")
        self.h_box.toggled.connect(self.viewer.set_show_hydrogens)
        vt.addWidget(self.h_box)
        self.a_show2d = act("Show 2D", self._set_2d_visible, "Ctrl+D", "Show the 2D structure next to the 3D view")
        self.a_show2d.setCheckable(True)
        self.a_show2d.setChecked(True)
        vt.addAction(self.a_show2d)
        vt.addAction(act("Reset view", self.reset_views, "R", "Default orientation, fit both views"))
        vt.addAction(act("Screenshot...", self.save_screenshot))

        m = self.menuBar()
        f = m.addMenu("&File")
        for a in (self.a_open, self.a_export):
            f.addAction(a)
        f.addAction(act("Open output folder", self.open_output_folder))
        f.addSeparator()
        f.addAction(act("Quit", self.close, "Ctrl+Q"))
        b = m.addMenu("&Build")
        for a in (self.a_run, self.a_cancel):
            b.addAction(a)
        b.addAction(self.a_surface)
        v = m.addMenu("&View")
        v.addAction(self.a_show2d)
        self.a_h2d = act("Hydrogens in 2D", lambda on: self.viewer2d.set_show_hydrogens(on))
        self.a_h2d.setCheckable(True)
        v.addAction(self.a_h2d)
        mm = v.addMenu("Mouse mode")
        for a in self.mode_actions.values():
            mm.addAction(a)
        v.addSeparator()
        for d in (self.mol_dock, self.pipeline_dock, self.bottom_dock):
            v.addAction(d.toggleViewAction())
        self.a_ovito = act("Open in OVITO viewport", self.open_in_ovito)
        self.a_ovito.setEnabled(resources.has_package("ovito"))
        if not resources.has_package("ovito"):
            self.a_ovito.setText("Open in OVITO viewport (pip install ovito)")
        v.addAction(self.a_ovito)
        t = m.addMenu("&Tools")
        t.addAction(self.a_settings)
        t.addAction(act("Hardware report", self.show_hardware_report))
        h = m.addMenu("&Help")
        h.addAction(act("About QuantaBench", self.about))

    def _build_statusbar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.measure_label = QLabel("")
        self.measure_label.setToolTip("Click 2 atoms for a distance, 3 for an angle, 4 for a dihedral "
                                      "(Ctrl+click adds atoms)")
        self.measure_label.setStyleSheet("font-weight: bold; padding: 0 8px;")
        self.statusBar().addPermanentWidget(self.measure_label)
        self.viewer.measurementChanged.connect(self._on_measurement)
        self.hw_label = QLabel("Detecting hardware...")
        self.statusBar().addPermanentWidget(self.hw_label)
        self.statusBar().addPermanentWidget(self.progress)

    # ================================================================== settings <-> widgets
    def _load_stage_values(self) -> None:
        b = self.s.build
        self.w_desalt.setChecked(b.keep_largest_fragment)
        self.w_protonate.setChecked(b.protonate)
        self.w_ph.setValue(b.ph)
        self.w_maxconf.setValue(b.max_conformers)
        self.w_window.setValue(b.energy_window_kcal)
        self.w_rmsd.setValue(b.rmsd_threshold)
        self.w_seed.setValue(b.random_seed)
        self.w_method.setCurrentIndex(METHODS.index(b.refine_method) if b.refine_method in METHODS else 0)
        self.w_fmax.setValue(b.fmax)
        self.w_steps.setValue(b.max_steps)
        self.w_temp.setValue(b.temperature_k)
        self.w_xtbprops.setChecked(resources.has_package("dxtb"))
        self.w_freq.setChecked(b.frequencies)
        keys = [k for k, _ in BATCHED_CHOICES]
        self.w_batched.setCurrentIndex(keys.index(b.batched) if b.batched in keys else 0)
        for fmt, cb in self.w_formats.items():
            cb.setChecked(fmt in export.DEFAULT_FORMATS)
        self.out_label.setText(f"Output folder: {self.s.output_path()}")
        self._update_engine_info()

    def _store_stage_values(self) -> None:
        b = self.s.build
        b.keep_largest_fragment = self.w_desalt.isChecked()
        b.protonate = self.w_protonate.isChecked()
        b.ph = self.w_ph.value()
        b.max_conformers = self.w_maxconf.value()
        b.energy_window_kcal = self.w_window.value()
        b.rmsd_threshold = self.w_rmsd.value()
        b.random_seed = self.w_seed.value()
        b.refine_method = self.w_method.currentData()
        b.fmax = self.w_fmax.value()
        b.max_steps = self.w_steps.value()
        b.temperature_k = self.w_temp.value()
        b.frequencies = self.w_freq.isChecked()
        b.batched = self.w_batched.currentData()

    def current_config(self) -> BuildConfig:
        self._store_stage_values()
        formats = tuple(f for f, cb in self.w_formats.items() if cb.isChecked()) or export.DEFAULT_FORMATS
        if self.w_freq.isChecked() and "spectrum" not in formats:
            formats = (*formats, "spectrum")
        return BuildConfig.from_settings(
            self.s, refine_top=self.w_refine_top.value(), optimizer=self.w_optimizer.currentText(),
            protonation_method=self.w_prot_method.currentData(), xtb_properties=self.w_xtbprops.isChecked(),
            symmetry_number=self.w_symmetry.value(), formats=formats)

    def _update_engine_info(self) -> None:
        key = self.w_method.currentData()
        if key == "auto":
            order = [k for k in ("aimnet2", "gfn2-xtb", "gfn1-xtb") if engines.available(k)[0]]
            text = "Uses the best installed engine the molecule's elements allow: " + \
                   (" > ".join(engines.ENGINES[k].label for k in order) or "none") + " > MMFF94s."
        else:
            info = engines.ENGINES[key]
            ok, why = engines.available(key)
            text = info.description + ("" if ok else f"\nNot available: {why}")
        self.engine_info.setText(text)

    # ================================================================== hardware
    def _detect_hardware(self) -> None:
        if self._closing:
            return
        w = workers.FunctionWorker(resources.detect)
        w.done.connect(self._on_hardware)
        workers.start(w, self, on_failed=lambda msg: self.hw_label.setText(f"Hardware detection failed: {msg}"))

    def _on_hardware(self, rep) -> None:
        self.report = rep
        gpu = rep.gpus[0] if rep.gpus else None
        threads = resources.resolve_threads(self.s.resources.threads)
        gpu_text = f"{gpu.name} {gpu.total_memory_gb:.0f} GB ({gpu.backend.upper()})" if gpu else "no GPU (CPU only)"
        if self.s.resources.gpu == "off":
            gpu_text += " [disabled]"
        self.hw_label.setText(f"CPU {threads}/{rep.logical_cores} threads | GPU: {gpu_text}")
        self.hw_label.setToolTip(rep.summary() + ("\n" + "\n".join(rep.notes) if rep.notes else ""))
        self.log(f"Hardware: {rep.summary()}")
        for n in rep.notes:
            self.log(f"note: {n}")

    def show_hardware_report(self) -> None:
        rep = self.report or resources.detect()
        lines = [rep.summary(), f"OS: {rep.os}{' (WSL2)' if rep.wsl else ''}, Python {rep.python}",
                 f"PyTorch: {rep.torch_version or 'not installed'} ({rep.torch_backend})", "", "Engines:"]
        for key, info in engines.ENGINES.items():
            ok, why = engines.available(key)
            lines.append(f"  {'✔' if ok else '✖'} {info.label} {why}")
        lines += ["", "Settings: " + str(settings_mod.settings_dir() / settings_mod.SETTINGS_FILE)] + rep.notes
        QMessageBox.information(self, "Hardware report", "\n".join(lines))

    # ================================================================== molecules
    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def add_inputs(self, items: list[inputs_mod.MoleculeInput]) -> None:
        existing = {m.id for m in self.inputs}
        for it in items:
            base, n = it.id, 2
            while it.id in existing:
                it.id = f"{base}_{n}"
                n += 1
            existing.add(it.id)
            self.inputs.append(it)
            self.mol_list.addItem(QListWidgetItem(self._item_text(it.id)))
        if items:
            self.mol_list.setCurrentRow(self.mol_list.count() - 1)
            self.log(f"Added {len(items)} molecule(s).")

    def _item_text(self, ident: str) -> str:
        if ident in self.running:
            return f"{STATUS_ICON['running']} {ident}  ({self.running[ident]})"
        res = self.results.get(ident)
        icon = STATUS_ICON[res.status] if res else STATUS_ICON["pending"]
        extra = f"  {res.method}, {len(res.conformers)} conf." if res and res.status == "ok" else ""
        return f"{icon} {ident}{extra}"

    def _refresh_item(self, ident: str) -> None:
        for i, m in enumerate(self.inputs):
            if m.id == ident:
                self.mol_list.item(i).setText(self._item_text(ident))

    def add_from_entry(self) -> None:
        text = self.entry.text().strip()
        if not text:
            return
        self.entry.setEnabled(False)
        self.statusBar().showMessage(f"Looking up {text}...")
        w = workers.FunctionWorker(inputs_mod.parse_text, text)  # names go to PubChem: keep the UI responsive

        def done(items):
            self.entry.setEnabled(True)
            self.entry.clear()
            self.statusBar().clearMessage()
            self.add_inputs(items)

        def failed(msg):
            self.entry.setEnabled(True)
            self.statusBar().clearMessage()
            QMessageBox.warning(self, "Add molecule", msg)

        workers.start(w, self, on_done=done, on_failed=failed)   # relayed: the callbacks touch widgets

    def open_files(self, paths: list[str] | None = None) -> None:
        if not paths:
            patterns = " ".join(f"*{ext}" for ext in sorted(inputs_mod.FILE_TYPES))
            paths, _ = QFileDialog.getOpenFileNames(self, "Open structures", "",
                                                    f"Structures and tables ({patterns});;All files (*)")
        if not paths:
            return
        sheet = self.csv_sheet.text().strip() or None
        if sheet is not None and sheet.isdigit():
            sheet = int(sheet)
        opts = dict(smiles_col=self.csv_smiles_col.text().strip() or None, id_col=self.csv_id_col.text().strip() or None,
                    sheet=sheet, smiles_preference=self.smiles_pref.currentData())

        def read_all():  # off the GUI thread: a CompTox table can have thousands of rows to parse
            out, messages, errors = [], [], []
            for p in paths:
                try:
                    out += inputs_mod.read_file(p, log=messages.append, **opts)
                except Exception as exc:
                    errors.append(f"{Path(p).name}: {exc}")
            return out, messages, errors

        def done(result):
            items, messages, errors = result
            self.statusBar().clearMessage()
            for m in messages:
                self.log(m)
            self.add_inputs(items)
            if errors:
                QMessageBox.warning(self, "Open", "\n".join(errors))

        self.statusBar().showMessage(f"Reading {len(paths)} file(s)...")
        w = workers.FunctionWorker(read_all)
        workers.start(w, self, on_done=done, on_failed=lambda msg: QMessageBox.warning(self, "Open", msg))

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.mol_list.selectedIndexes()}, reverse=True)
        for r in rows:
            ident = self.inputs[r].id
            if ident in self.running:
                continue
            self.inputs.pop(r)
            self.results.pop(ident, None)
            self.mol_list.takeItem(r)

    def clear_all(self) -> None:
        if self.worker is not None:
            return
        self.inputs.clear()
        self.results.clear()
        self.mol_list.clear()
        self.conf_table.setRowCount(0)
        self.viewer.clear()
        self.viewer2d.clear()
        self._shown_key = None

    def selected_input(self) -> inputs_mod.MoleculeInput | None:
        row = self.mol_list.currentRow()
        return self.inputs[row] if 0 <= row < len(self.inputs) else None

    def _on_molecule_selected(self, row: int) -> None:
        inp = self.inputs[row] if 0 <= row < len(self.inputs) else None
        if inp is None:
            return
        res = self.results.get(inp.id)
        self.input_detail.setText(f"<b>{inp.id}</b> {inp.name}<br>SMILES: {inp.smiles}<br>Source: {inp.source}")
        self._update_stage_status(res)
        self._fill_conformers(res)
        built = res is not None and res.status == "ok"
        if built:
            self._show_info(res)
        else:
            self.mol_info.setText(res.error if res and res.error else "Not built yet. Press Run.")
            self.props_view.setText("")
        if self._surface_preview:
            return                       # the open Surface Builder owns the 3D view until it is closed
        key = (inp.id, res if built else None)
        if self._shown_key is None or self._shown_key[0] != key[0] or self._shown_key[1] is not key[1]:
            # Indices belong to the previous structure. The input preview (AddHs of the raw input) and the
            # built molecule (standardized, protonated) number their atoms differently, so a swap from
            # preview to built, or to a rebuilt result, clears the selection too.
            self.selection.clear()
            self._shown_key = key
        if built:
            self._show_conformer(res, 0, reset_camera=True)
        else:
            self._preview_input(inp)

    def _show_conformer(self, res: MoleculeResult, row: int, reset_camera: bool) -> None:
        """Show conformer ``row`` of a built molecule in both views, with its charges when it has them."""
        c = res.conformers[row]
        self.viewer.show_rdkit(res.mol, c.conf_id, reset_camera=reset_camera, charges=c.charges)
        self.viewer2d.show_rdkit(res.mol, c.conf_id, reset_camera=reset_camera)
        self._apply_2d_options(c.charges)

    def _preview_input(self, inp) -> None:
        """Quick single-conformer preview so the user sees something right away."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = inp.to_mol()
            if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol, randomSeed=1) != 0:
                    return
                AllChem.MMFFOptimizeMolecule(mol)
            self.viewer.show_rdkit(mol, -1)
            self.viewer2d.show_rdkit(mol, -1)
            self._apply_2d_options(None)
        except Exception as exc:
            self.log(f"preview failed for {inp.id}: {exc}")

    def _show_info(self, res: MoleculeResult) -> None:
        d = res.descriptors
        bits = [f"<b>{res.input.id}</b> {d.get('formula', '')}, charge {res.charge:+d}",
                f"{engines.ENGINES[res.method].label} on {res.device}",
                f"{res.n_embedded} embedded → {len(res.conformers)} conformers"]
        for key, label, unit in (("volume_A3_boltz", "Volume", "Å³"), ("sasa_A2_boltz", "SASA", "Å²"),
                                 ("dipole_debye_boltz", "Dipole", "D"), ("xtb_dipole_debye", "Dipole (xTB)", "D"),
                                 ("xtb_polarizability_iso_A3", "Polarizability (xTB)", "Å³")):
            if key in d:
                bits.append(f"{label}: {d[key]:.2f} {unit}")
        vib = res.vibrations
        thermo = (vib or {}).get("thermo")
        if thermo:
            bits.append(f"ZPE {thermo['zpe_kcal']:.2f} kcal/mol, S {thermo['entropy_cal_molK']:.1f} cal/(mol·K), "
                        f"G − E {thermo['g_corr_kcal']:.2f} kcal/mol at {thermo['temperature_k']:.0f} K"
                        + (f"; <b>{vib['n_imaginary']} imaginary</b>" if vib["n_imaginary"] else ""))
        bits += [f"<i>{n}</i>" for n in res.notes]
        self.mol_info.setText("<br>".join(bits))
        lines = []
        if thermo:
            lines += [f"<b>Thermochemistry</b> ({vib['method']}, {vib['hessian_method']} Hessian, "
                      f"{thermo['temperature_k']:.2f} K, {'qRRHO' if thermo['qrrho'] else 'RRHO'})",
                      f"ZPE: {thermo['zpe_kcal']:.3f} kcal/mol",
                      f"H: {thermo['enthalpy_kcal']:.3f} kcal/mol (H − E {thermo['h_corr_kcal']:.3f})",
                      f"S: {thermo['entropy_cal_molK']:.3f} cal/(mol·K)",
                      f"G: {thermo['gibbs_kcal']:.3f} kcal/mol (G − E {thermo['g_corr_kcal']:.3f})",
                      f"{vib['n_modes']} modes, {vib['n_imaginary']} imaginary, lowest "
                      f"{vib['lowest_freq_cm1']:.1f} cm⁻¹", ""]
        lines += [f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}" for k, v in d.items()
                  if not k.startswith("vib_")]
        self.props_view.setText("<br>".join(lines))

    def _update_stage_status(self, res: MoleculeResult | None) -> None:
        for key, it in self.stage_items.items():
            text = ""
            if res is not None:
                t = res.timings.get(key if key != "properties" else "descriptors")
                if t is not None:
                    text = f"{STATUS_ICON['ok']} {t:.1f} s"
                elif key == "input":
                    text = STATUS_ICON["ok"]
                elif key == "export" and res.files:
                    text = f"{STATUS_ICON['ok']} {len(res.files)} files"
                elif res.status == "failed":
                    text = STATUS_ICON["failed"]
            it.setText(1, text)

    def _fill_conformers(self, res: MoleculeResult | None) -> None:
        self.conf_table.blockSignals(True)
        self.conf_table.setRowCount(0)
        if res and res.status == "ok":
            self.conf_table.setRowCount(len(res.conformers))
            for r, c in enumerate(res.conformers):
                vals = [str(c.rank + 1), f"{c.rel_kcal:.2f}", f"{100 * c.weight:.1f}", f"{c.energy_ev:.5f}",
                        "yes" if c.converged else "no", str(c.steps),
                        f"{c.dipole_debye:.2f}" if c.dipole_debye is not None else ""]
                for col, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.conf_table.setItem(r, col, item)
        self.conf_table.blockSignals(False)

    def _on_conformer_selected(self, row, _col, _prow, _pcol) -> None:
        inp = self.selected_input()
        res = self.results.get(inp.id) if inp else None
        if res and 0 <= row < len(res.conformers) and not self._surface_preview:
            self._show_conformer(res, row, reset_camera=False)

    def _on_stage_selected(self, cur, _prev) -> None:
        if cur is not None:
            self.stage_stack.setCurrentIndex(self.stage_pages[cur.data(0, Qt.ItemDataRole.UserRole)])

    # ================================================================== running
    def run_build(self, selected_only: bool = False) -> None:
        if self.worker is not None:
            return
        if selected_only:
            rows = sorted({i.row() for i in self.mol_list.selectedIndexes()})
            todo = [self.inputs[r] for r in rows]
        else:
            todo = list(self.inputs)
        if not todo:
            QMessageBox.information(self, "Run", "Add a molecule first (SMILES, name, or a file).")
            return
        cfg = self.current_config()
        settings_mod.save(self.s)
        self.worker = workers.BuildWorker(todo, cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log)
        self.worker.molecule_done.connect(self._on_molecule_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.running = {m.id: "queued" for m in todo}
        for m in todo:
            self._refresh_item(m.id)
        self._set_running(True)
        self.progress.setRange(0, len(todo))
        self.progress.setValue(0)
        self.bottom_tabs.setCurrentWidget(self.log_view)
        workers.start(self.worker, self)

    def cancel_build(self, _checked=False, confirm: bool = True) -> None:
        if self.worker is None:
            return
        remaining = len(self.running)
        if confirm and remaining > 1 and QMessageBox.question(
                self, "Cancel build", f"Stop the build? {remaining} molecule(s) have not finished.\n"
                "Finished molecules are kept; a batch can be resumed with "
                "quantabench-cli build --resume.") != QMessageBox.StandardButton.Yes:
            return
        self.worker.cancel()
        self.log("Cancelling after the current step...")

    def _set_running(self, on: bool) -> None:
        self.progress.setVisible(on)
        for w in (self.run_btn, self.run_sel_btn, self.a_run):
            w.setEnabled(not on)
        self.cancel_btn.setEnabled(on)
        self.a_cancel.setEnabled(on)

    def _on_progress(self, i, n, stage, ident) -> None:
        if stage in ("ok", "failed", "cancelled"):
            self.progress.setValue(i)
            return
        self.running[ident] = stage
        self._refresh_item(ident)
        self.statusBar().showMessage(f"[{i + 1}/{n}] {ident}: {stage}")

    def _on_molecule_done(self, res: MoleculeResult) -> None:
        self.running.pop(res.input.id, None)
        self.results[res.input.id] = res
        self._refresh_item(res.input.id)
        inp = self.selected_input()
        if inp is not None and inp.id == res.input.id:
            self._on_molecule_selected(self.mol_list.currentRow())

    def _on_finished(self, results, run_dir) -> None:
        for ident in list(self.running):
            self.running.pop(ident)
            self._refresh_item(ident)
        self.worker = None
        self._set_running(False)
        self.last_run_dir = run_dir
        ok = sum(r.status == "ok" for r in results)
        if self._quit_pending:
            return                       # closing: keep "Closing after the current step..." on screen
        self.statusBar().showMessage(f"Done: {ok}/{len(results)} built" + (f", saved to {run_dir}" if run_dir else ""),
                                     15000)
        if run_dir:
            self.out_label.setText(f"Last run: {run_dir}")
        self.bottom_tabs.setCurrentWidget(self.conf_table)

    def _on_failed(self, msg) -> None:
        self.worker = None
        self.running.clear()
        self._set_running(False)
        self.log(msg)
        if not self._quit_pending:
            QMessageBox.critical(self, "Build failed", msg.splitlines()[0])

    # ================================================================== export & tools
    def export_selected(self) -> None:
        inp = self.selected_input()
        res = self.results.get(inp.id) if inp else None
        if not res or res.status != "ok":
            QMessageBox.information(self, "Export", "Select a molecule that has been built.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export to folder", str(self.s.output_path()))
        if not folder:
            return
        formats = tuple(f for f, cb in self.w_formats.items() if cb.isChecked()) or export.DEFAULT_FORMATS
        files = export.write_molecule(res, Path(folder), formats)
        self.log(f"Exported {len(files)} files to {folder}")

    def open_output_folder(self) -> None:
        folder = self.last_run_dir or self.s.output_path()
        Path(folder).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def save_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "quantabench.png",
                                              "3D view, PNG (*.png);;2D structure, SVG (*.svg);;2D structure, PNG (*.2d.png)")
        if not path:
            return
        if path.endswith(".svg") or path.endswith(".2d.png"):
            self.viewer2d.export(path, background="white" if path.endswith(".png") else None)
        else:
            self.viewer.screenshot(path)

    def _selected_atoms(self):
        inp = self.selected_input()
        res = self.results.get(inp.id) if inp else None
        if not res or res.status != "ok":
            return None, "", None
        best = res.conformers[0]
        atoms = export.mol_to_atoms(res.mol, best.conf_id)
        if best.charges is not None:
            atoms.set_initial_charges(best.charges)
        return atoms, res.input.id, res

    def open_surface_builder(self) -> None:
        """Modeless, so the 3D view stays usable: Move mode grabs the adsorbate, and every finished drag
        updates the dialog's system (Save and the LAMMPS export write the moved coordinates)."""
        from .surface_dialog import SurfaceDialog

        if self.surface_dialog is not None:
            self.surface_dialog.raise_()
            self.surface_dialog.activateWindow()
            return
        atoms, name, _ = self._selected_atoms()
        dlg = SurfaceDialog(atoms, name, preview_cb=self._preview_surface, parent=self,
                            gpu=self.s.resources.gpu != "off")
        dlg.setModal(False)
        dlg.moveModeRequested.connect(lambda on: self.set_view_mode("move" if on else "rotate"))
        self.viewer.coordinatesChanged.connect(dlg.set_positions)
        dlg.finished.connect(self._surface_dialog_closed)
        self.surface_dialog = dlg
        dlg.show()

    def _preview_surface(self, system, reset_camera: bool = True) -> None:
        self.selection.clear()           # molecule indices mean other atoms in the slab system
        self._shown_key = None
        self._surface_preview = True
        self.viewer2d.clear()            # the 2D depiction is for molecules, not slabs: hide it meanwhile
        self.viewer2d.setVisible(False)
        self.viewer.show_atoms(system, reset_camera=reset_camera)
        if self.color_box.currentData() == "element" and reset_camera:
            self.color_box.setCurrentIndex(COLOR_MODES.index("tag"))

    def _surface_dialog_closed(self, _result=0) -> None:
        dlg, self.surface_dialog = self.surface_dialog, None
        if dlg is not None:
            try:
                self.viewer.coordinatesChanged.disconnect(dlg.set_positions)
            except (RuntimeError, TypeError):
                pass
            dlg.deleteLater()
        self._surface_preview = False
        self.set_view_mode("rotate")
        self.viewer2d.setVisible(self.a_show2d.isChecked())
        if self.color_box.currentData() == "tag":
            self.color_box.setCurrentIndex(COLOR_MODES.index("element"))
        row = self.mol_list.currentRow()
        if 0 <= row < len(self.inputs):
            self._shown_key = None
            self._on_molecule_selected(row)

    # ================================================================== views
    def set_view_mode(self, mode: str) -> None:
        """'rotate' | 'select' | 'move' for the 3D view (the toolbar group and the surface dialog follow)."""
        self.viewer.set_mode(mode)
        self.mode_actions[mode].setChecked(True)
        if self.surface_dialog is not None:
            self.surface_dialog.set_move_checked(mode == "move")
        hint = {"rotate": "", "select": "Select mode: drag a box to select atoms",
                "move": "Move mode: drag on a molecule rotates it, Shift+drag slides it"}[mode]
        if hint:
            self.statusBar().showMessage(hint, 8000)

    def _on_label_mode(self, _index=0) -> None:
        self.viewer.set_label_mode(self.label_box.currentData())
        inp = self.selected_input()
        res = self.results.get(inp.id) if inp else None
        row = max(0, self.conf_table.currentRow())
        charges = res.conformers[row].charges if res and res.status == "ok" and row < len(res.conformers) else None
        self._apply_2d_options(charges)

    def _apply_2d_options(self, charges) -> None:
        mode = self.label_box.currentData()
        try:
            self.viewer2d.set_options(atom_indices=mode in ("index", "element+index"),
                                      charges=charges if mode == "charge" else None)
        except ValueError as exc:   # charges for another atom count
            self.log(f"2D labels: {exc}")

    def _on_measurement(self, text: str) -> None:
        self.measure_label.setText(text)

    def _set_2d_visible(self, on: bool) -> None:
        self.viewer2d.setVisible(bool(on))
        if self.a_show2d.isChecked() != bool(on):
            self.a_show2d.setChecked(bool(on))

    def reset_views(self) -> None:
        self.viewer.reset_camera()
        self.viewer2d.fit()

    def open_in_ovito(self) -> None:
        atoms, name, _ = self._selected_atoms()
        if atoms is None:
            QMessageBox.information(self, "OVITO", "Select a molecule that has been built.")
            return
        try:
            from .ovito_view import ovito_dock

            dock = ovito_dock(atoms, f"OVITO: {name}", self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        except Exception as exc:
            QMessageBox.warning(self, "OVITO", f"Could not open the OVITO viewport:\n{exc}")

    def open_settings(self) -> None:
        before = settings_mod.settings_dir()
        self._store_stage_values()
        dlg = SettingsDialog(self.s, self.report, self)
        if not dlg.exec():
            return
        s = dlg.result_settings()
        if settings_mod.settings_dir() != before:
            loaded = settings_mod.load()
            if (settings_mod.settings_dir() / settings_mod.SETTINGS_FILE).exists():
                s = loaded
        self.s = s
        settings_mod.save(self.s)
        resources.apply_thread_limits(self.s.resources.threads)
        self._load_stage_values()
        if self.report:
            self._on_hardware(self.report)

    def about(self) -> None:
        QMessageBox.about(self, "About QuantaBench",
                          f"<b>QuantaBench {__version__}</b><br>Open, GPU-aware molecular modeling workbench.<br>"
                          "RDKit · ASE · PyTorch · dxtb · AIMNet2 · PySide6 · PyVista<br>MIT License<br>"
                          "<a href='https://github.com/brandonbtm29/QuantaBench'>github.com/brandonbtm29/QuantaBench</a>")

    # ================================================================== window state
    def _restore_window(self) -> None:
        qs = _window_settings()
        geo, state = qs.value("geometry"), qs.value("state")
        self.resizeDocks([self.bottom_dock], [190], Qt.Orientation.Vertical)
        self.resizeDocks([self.mol_dock, self.pipeline_dock], [270, 440], Qt.Orientation.Horizontal)
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)
        split = qs.value("splitter")
        if split is not None:
            self.splitter.restoreState(split)
        show2d = qs.value("show2d")
        if show2d is not None:
            self._set_2d_visible(str(show2d).lower() in ("true", "1"))

    def _busy_threads(self) -> list:
        """Worker threads that are still running (build, PubChem lookups, file reads, LAMMPS export)."""
        out = []
        for worker in list(self.__dict__.get("_live_workers", ())):
            thread = getattr(worker, "_thread", None)
            try:
                if thread is not None and thread.isRunning():
                    out.append(thread)
            except RuntimeError:          # already deleted by deleteLater
                pass
        return out

    def _close_when_idle(self) -> None:
        if self.worker is None and not self._busy_threads():
            self._quit_timer.stop()
            self.close()

    def closeEvent(self, event) -> None:
        """Never blocks: with work still running, cancel it, say so, and close again once every worker thread
        has stopped. (Waiting here froze the UI, and a QThread destroyed while running aborts the process,
        which could leave jobs.sqlite mid-write.) Cancellation takes effect at the pipeline's next checkpoint."""
        if self.worker is not None and not self._quit_pending:
            if QMessageBox.question(self, "Quit", "A build is running. Cancel it and quit?") != \
                    QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.cancel()
        if self.worker is not None or self._busy_threads():
            self._quit_pending = True
            self._closing = True
            self.statusBar().showMessage("Closing after the current step...")
            self.centralWidget().setEnabled(False)
            for d in (self.mol_dock, self.pipeline_dock):
                d.setEnabled(False)
            if not hasattr(self, "_quit_timer"):
                self._quit_timer = QTimer(self)
                self._quit_timer.setInterval(200)
                self._quit_timer.timeout.connect(self._close_when_idle)
            self._quit_timer.start()
            event.ignore()
            return
        self._closing = True
        qs = _window_settings()
        qs.setValue("geometry", self.saveGeometry())
        qs.setValue("state", self.saveState())
        qs.setValue("splitter", self.splitter.saveState())
        qs.setValue("show2d", self.a_show2d.isChecked())
        try:
            self._store_stage_values()
            settings_mod.save(self.s)
        except OSError:
            pass
        if self.surface_dialog is not None:
            self.surface_dialog.close()
        self.viewer.close_viewer()
        self.viewer2d.close_viewer()
        self.closed = True
        super().closeEvent(event)
