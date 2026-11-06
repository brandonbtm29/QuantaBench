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
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl, Signal
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

from .. import __version__, resources, units
from .. import settings as settings_mod
from ..chem import engines, export, jobs, solvents
from ..chem import inputs as inputs_mod
from ..chem.pipeline import BuildConfig, MoleculeResult, fallback_count
from . import molecule_list, workers
from .energy_widgets import EnergySpinBox
from .molecule_list import (
    STATUS_FILTERS,
    STATUS_ICON,
    MoleculeListView,
    MoleculeRow,
    MoleculeTableModel,
    ResultStore,
)
from .selection import SelectionModel
from .settings_dialog import SettingsDialog, method_keys, method_label
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
    ("conformers", "Conformer search", "ETKDGv3 embedding + MMFF94s, or CREST metadynamics (xtb); pruning"),
    ("refine", "Refinement", "Re-optimize the lowest conformers with an ML potential, xTB, tblite or MOPAC, "
                             "optionally in an implicit solvent"),
    ("properties", "Properties", "Boltzmann weights, descriptors, the engine's charges/dipole/polarizability, "
                                 "frequencies"),
    ("qm", "QM (DFT)", "Psi4 single points or optimizations of the lowest conformers: properties, polarizability, "
                       "density/ESP cubes, frequencies"),
    ("export", "Export", "Files written for every molecule"),
]
QM_FUNCTIONALS = ("wb97x-d", "b3lyp-d3bj", "pbe0-d3bj", "r2scan", "m06-2x", "hf")
QM_BASES = ("def2-svp", "def2-svpd", "def2-tzvp", "6-31g*", "6-31+g*", "6-311+g**", "aug-cc-pvdz")
LOG_LINES = 5000          # Log view's line cap (see _build_bottom_dock)
CREST_METHODS = (("gfn2", "GFN2-xTB"), ("gfnff", "GFN-FF (fast, large molecules)"),
                 ("gfn2//gfnff", "GFN2-xTB // GFN-FF (needs xtb)"), ("gfn1", "GFN1-xTB"))


def _window_settings() -> QSettings:
    """Window geometry and dock layout (the registry on Windows). ``QUANTABENCH_WINDOW_STATE`` points it at
    an INI file instead, so the tests never overwrite the user's own layout."""
    ini = os.environ.get("QUANTABENCH_WINDOW_STATE")
    if ini:
        return QSettings(ini, QSettings.Format.IniFormat)
    return QSettings("QuantaBench", "QuantaBench")


def _built(res) -> bool:
    """A result the views can show: built, with its molecule and at least one conformer."""
    return res is not None and res.status == "ok" and res.mol is not None and bool(res.conformers)


class MainWindow(QMainWindow):
    #: The selected molecule changed, or its results did (MoleculeResult, or None when not built).
    #: Panels (spectra, surfaces) listen to this instead of reaching into the molecule list.
    resultChanged = Signal(object)
    #: Settings > Display changed (a ``units.EnergyDisplay``): panels showing energies redraw them.
    displayChanged = Signal(object)

    def __init__(self, s: settings_mod.Settings):
        super().__init__()
        self.s = s
        self.report: resources.HardwareReport | None = None
        # The list holds light rows (molecule_list.MoleculeRow); full results are read from their folders on
        # selection and only the last few stay in memory (ResultStore), so 14,000 molecules fit (SCALE-1).
        self.model = MoleculeTableModel(self, display=s)
        self.results = ResultStore(self.model)
        self.running: dict[str, str] = {}
        self.worker = None
        self.last_run_dir: Path | None = None
        self.run_folder: Path | None = None    # the run folder the list was opened from (File > Open run folder)
        self._batch_dir: Path | None = None    # run folder of the batch now running (job table), if any
        self._batch_ids: list[str] = []
        self._loading: str | None = None       # molecule whose result is being read (one read at a time)
        self._wanted: str | None = None        # molecule to read next
        self._shown_res: MoleculeResult | None = None   # built result in the views
        self._closing = False
        self._probe_cancel = threading.Event()   # stops the hardware probe's child process (closeEvent)
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
        # The Pipeline widgets hold settings only once _load_stage_values has filled them: until then, storing
        # them (current_config, e.g. from a panel built during start-up) would write their minimums into the
        # settings, and closing the window would save those (pH 0, 1 conformer, 1 K).
        self._stage_loaded = False
        self._build_molecule_dock()
        self._build_pipeline_dock()
        self._build_bottom_dock()
        self._build_toolbar_and_menus()
        self._build_statusbar()
        self._load_stage_values()
        self._install_panels()
        self._restore_window()
        QTimer.singleShot(50, self._detect_hardware)

    # ================================================================== UI construction
    def _install_panels(self) -> None:
        """Optional analysis panels; one that fails to load is logged and skipped."""
        from . import import_action, machines_panel, spectra_panel, surface_panel, update_action

        for panel in (spectra_panel, surface_panel, import_action, machines_panel, update_action):
            try:
                panel.install(self)
            except Exception as exc:  # a broken panel must not stop the app from starting
                self.log(f"{panel.__name__.rsplit('.', 1)[-1]} unavailable: {type(exc).__name__}: {exc}")

    def current_result(self) -> MoleculeResult | None:
        """Result of the selected molecule (None if nothing is selected, it is not built, or it is still being
        read from its folder: ``resultChanged`` follows when it is)."""
        row = self._current_row()
        res = self.results.cached(row.id) if row is not None else None
        return res if _built(res) else None

    def _full_current(self) -> MoleculeResult | None:
        """The selected molecule's built result, read from its folder now if it is not loaded yet."""
        row = self._current_row()
        res = self.results.get(row.id) if row is not None else None
        return res if _built(res) else None

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
        find = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search ID, name, SMILES, formula")
        self.search.setClearButtonEnabled(True)
        self.status_filter = QComboBox()
        for key, label in STATUS_FILTERS:
            self.status_filter.addItem(label, key)
        find.addWidget(self.search, 1)
        find.addWidget(self.status_filter)
        lay.addLayout(find)
        self.mol_list = MoleculeListView(self.model)
        self.mol_list.currentIdChanged.connect(self._on_molecule_selected)
        lay.addWidget(self.mol_list)
        self._search_timer = QTimer(self)          # filter once typing pauses, not per keystroke
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(150)
        self._search_timer.timeout.connect(self._apply_search)
        self.search.textChanged.connect(self._search_timer.start)
        self.search.returnPressed.connect(self._apply_search)
        self.status_filter.currentIndexChanged.connect(self._apply_search)
        self.count_label = QLabel()
        self.count_label.setStyleSheet("color: gray;")
        lay.addWidget(self.count_label)
        self._count_timer = QTimer(self)           # at most a few count updates a second during a batch
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(300)
        self._count_timer.timeout.connect(self._update_counts)
        for sig in (self.model.rowsInserted, self.model.rowsRemoved, self.model.modelReset, self.model.dataChanged,
                    self.mol_list.proxy.layoutChanged):
            sig.connect(self._counts_later)
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
        self._update_counts()

    def _apply_search(self, *_args) -> None:
        self._search_timer.stop()
        self.mol_list.proxy.set_text(self.search.text())
        self.mol_list.proxy.set_status(self.status_filter.currentData() or "")
        self._update_counts()

    def _counts_later(self, *_args) -> None:
        if not self._count_timer.isActive():
            self._count_timer.start()

    def _update_counts(self) -> None:
        n = self.model.rowCount()
        if not n:
            self.count_label.setText("No molecules. Add SMILES or names, open a file or a run folder.")
            return
        c = self.model.counts()
        todo = c["pending"] + c["cancelled"] + c["running"]
        text = f"{n:,} molecule{'s' if n != 1 else ''}: {c['ok']:,} built, {c['failed']:,} failed, {todo:,} not built"
        shown = self.mol_list.count()
        if shown != n:
            text += f" ({shown:,} shown)"
        self.count_label.setText(text)

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
        self.stage_tree.setMaximumHeight(195)
        self.stage_tree.setMinimumHeight(170)
        lay.addWidget(self.stage_tree)
        self.stage_stack = QStackedWidget()
        self.stage_pages = {}
        for key, _label, tip in STAGES:
            page = self._stage_page(key, tip)
            for combo in page.findChildren(QComboBox):   # long item texts must not widen the dock
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(14)
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
        self.pipeline_dock = dock
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

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
            from ..chem import crest

            self.w_confsearch = QComboBox()
            self.w_confsearch.addItem("RDKit ETKDGv3 + MMFF94s", "rdkit")
            ok, why = crest.available()
            self.w_confsearch.addItem("CREST iMTD-GC" + ("" if ok else " (not installed)"), "crest")
            self.w_confsearch.setItemData(1, "CREST metadynamics conformer search (Grimme). Minutes per molecule; "
                                          "Linux/WSL/macOS" + ("" if ok else f". {why}"), Qt.ItemDataRole.ToolTipRole)
            self.w_crest_method = QComboBox()
            for k, label in CREST_METHODS:
                self.w_crest_method.addItem(label, k)
            self.w_crest_quick = QCheckBox("Quick mode (--quick)")
            self.w_crest_quick.setChecked(True)
            self.w_crest_fallback = QCheckBox("If CREST fails, use RDKit's conformers")
            self.w_crest_fallback.setChecked(True)
            self.w_crest_fallback.setToolTip(
                "Checked: a molecule CREST fails on is built with RDKit ETKDGv3 + MMFF94s and flagged (a note, "
                "summary.csv conformer_fallback, a count at the end of the batch). Unchecked: the molecule fails.")
            self.w_confsearch.currentIndexChanged.connect(self._update_crest_widgets)
            form.addRow("Search", self.w_confsearch)
            form.addRow("CREST level", self.w_crest_method)
            form.addRow(self.w_crest_quick)
            form.addRow(self.w_crest_fallback)
            self.w_maxconf = QSpinBox()
            self.w_maxconf.setRange(1, 1000)
            # stored in kcal/mol (BuildDefaults.energy_window_kcal, a science setting), shown in the display unit
            self.w_window = EnergySpinBox(0.5, 50, units.KCAL, self.energy_display().relative)
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
            for m in method_keys():
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
            for k, label in (("lbfgs", "L-BFGS (ASE)"), ("fire", "FIRE (ASE)"),
                             ("native", "Engine's own optimizer (xtb, MOPAC; else L-BFGS)")):
                self.w_optimizer.addItem(label, k)
            self.w_solvent = QComboBox()
            self.w_solvent.addItem("Gas phase", "")
            for name, sv in solvents.SOLVENTS.items():
                self.w_solvent.addItem(f"{sv.label} (ε = {sv.epsilon:g})", name)
            self.w_solvent.setToolTip("Implicit solvent for engines that have one: ALPB (xtb, tblite), COSMO (MOPAC), "
                                      "PCM (Psi4). Engines without one run in the gas phase (noted per molecule).")
            self.w_solvent.currentIndexChanged.connect(self._update_engine_info)
            self.w_batched = QComboBox()
            for key, label in BATCHED_CHOICES:
                self.w_batched.addItem(label, key)
            self.w_batched.setToolTip("Refine all conformers of a molecule in one torch batch (xTB, AIMNet2). "
                                      "Same results as one at a time; much faster on a GPU.")
            self.engine_info = QLabel()
            self.engine_info.setWordWrap(True)
            self.w_method.currentIndexChanged.connect(self._update_engine_info)
            form.addRow("Method", self.w_method)
            form.addRow("Solvent", self.w_solvent)
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
        elif key == "qm":
            from ..chem import qm

            ok, why = qm.available()
            self.w_qm_on = QCheckBox("Run a QM stage" + ("" if ok else " (Psi4 not found)"))
            self.w_qm_on.setToolTip("Psi4 runs as a separate program (its own conda environment). " +
                                    ("" if ok else why + ". Set its Python in Settings > Engines."))
            self.w_qm_func = QComboBox()
            self.w_qm_func.setEditable(True)
            self.w_qm_func.addItems(QM_FUNCTIONALS)
            self.w_qm_basis = QComboBox()
            self.w_qm_basis.setEditable(True)
            self.w_qm_basis.addItems(QM_BASES)
            self.w_qm_task = QComboBox()
            self.w_qm_task.addItem("Single point", "energy")
            self.w_qm_task.addItem("Optimize", "optimize")
            self.w_qm_top = QSpinBox()
            self.w_qm_top.setRange(1, 100)
            self.w_qm_top.setToolTip("How many of the lowest refined conformers get the QM stage")
            self.w_qm_task.setToolTip("Single point at the refined geometry, or a QM optimization (optking)")
            self.w_qm_props = QCheckBox("Properties and polarizability")
            self.w_qm_props.setToolTip("Best conformer: dipole, quadrupole, Mulliken/Löwdin/MBIS charges, HOMO/LUMO, "
                                       "static polarizability (analytic or finite field)")
            self.w_qm_props.setChecked(True)
            self.w_qm_freq = QCheckBox("Frequencies + thermochemistry")
            self.w_qm_freq.setToolTip("Best conformer: QM Hessian, IR intensities, ZPE/H/S/G. Slow for DFT "
                                      "(finite differences: about 6N gradients)")
            self.w_qm_cubes = QCheckBox("Density/ESP cubes + surfaces")
            self.w_qm_cubes.setToolTip("0.002 au density surface: area, enclosed volume, ESP extrema and "
                                       "Murray-Politzer statistics. Shown in the Surfaces panel.")
            self.w_qm_on.toggled.connect(self._update_qm_widgets)
            form.addRow(self.w_qm_on)
            form.addRow("Functional", self.w_qm_func)
            form.addRow("Basis set", self.w_qm_basis)
            form.addRow("Task", self.w_qm_task)
            form.addRow("Conformers", self.w_qm_top)
            form.addRow(self.w_qm_props)
            form.addRow(self.w_qm_freq)
            form.addRow(self.w_qm_cubes)
        elif key == "export":
            self.w_formats = {}
            for fmt, desc in export.FORMATS.items():
                cb = QCheckBox(fmt)
                cb.setToolTip(desc)
                lbl = QLabel(desc)
                lbl.setWordWrap(True)
                self.w_formats[fmt] = cb
                form.addRow(cb, lbl)
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
        self.conf_table.setHorizontalHeaderLabels(self._conformer_headers(self.energy_display()))
        self.conf_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.conf_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.conf_table.verticalHeader().setVisible(False)
        self.conf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.conf_table.currentCellChanged.connect(self._on_conformer_selected)
        tabs.addTab(self.conf_table, "Conformers")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        # Each line costs ~7 KB of RSS (its laid-out text block); a batch logs ~3 lines per molecule, so this cap,
        # not the molecule count, bounds the log: ~35 MB at 5,000 lines (20,000 lines were ~140 MB, SCALE-1).
        self.log_view.setMaximumBlockCount(LOG_LINES)
        tabs.addTab(self.log_view, "Log")
        self.bottom_tabs = tabs
        dock = QDockWidget("Results", self)
        dock.setObjectName("results")
        self.results_dock = dock
        dock.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.bottom_dock = dock

    @staticmethod
    def _conformer_headers(d: units.EnergyDisplay) -> list[str]:
        return ["#", d.header("ΔE", "relative"), "Weight (%)", d.header("E", "absolute"), "Converged", "Steps",
                "Dipole (D)"]

    def energy_display(self) -> units.EnergyDisplay:
        """Units and number format of Settings > Display, for everything in the window that shows an energy."""
        return units.EnergyDisplay.of(self.s)

    def apply_display(self) -> None:
        """Redraw energies in the units of Settings > Display: conformer table, molecule list, the molecule's info
        and properties, the energy window (its stored kcal/mol value is kept), and the panels (``displayChanged``)."""
        d = self.energy_display()
        self.conf_table.setHorizontalHeaderLabels(self._conformer_headers(d))
        self.model.set_display(d)
        self.w_window.set_unit(d.relative)
        res = self._shown_res
        if res is not None:
            row = self.conf_table.currentRow()
            self._fill_conformers(res)
            if 0 <= row < self.conf_table.rowCount():
                self.conf_table.blockSignals(True)          # same conformer: the views stay as they are
                self.conf_table.setCurrentCell(row, 0)
                self.conf_table.blockSignals(False)
            self._show_info(res)
        self.displayChanged.emit(d)

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
        self.a_open_run = act("Open run folder...", self.open_run_folder, "Ctrl+Shift+O",
                              "List the molecules of a run folder (finished or partial); results load when selected")
        self.a_resume = act("Resume run folder", self.resume_run, None,
                            "Continue the unfinished molecules of the opened (or last) run folder with its stored "
                            "settings")
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
        for a in (self.a_open, self.a_open_run, self.a_export):
            f.addAction(a)
        f.addAction(act("Open output folder", self.open_output_folder))
        f.addSeparator()
        f.addAction(act("Quit", self.close, "Ctrl+Q"))
        b = m.addMenu("&Build")
        for a in (self.a_run, self.a_cancel, self.a_resume):
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
        self.w_window.set_stored(b.energy_window_kcal)
        self.w_rmsd.setValue(b.rmsd_threshold)
        self.w_seed.setValue(b.random_seed)
        methods = method_keys()
        self.w_method.setCurrentIndex(methods.index(b.refine_method) if b.refine_method in methods else 0)
        self._set_combo(self.w_confsearch, b.conformer_search)
        self._set_combo(self.w_solvent, solvents.canonical(b.solvent) if b.solvent in solvents.SOLVENTS
                        or not b.solvent else "")
        level = (b.qm_method or "").split("/", 1)
        self.w_qm_on.setChecked(bool(b.qm_method))
        if b.qm_method:
            self.w_qm_func.setCurrentText(level[0])
            self.w_qm_basis.setCurrentText(level[1] if len(level) > 1 else "def2-svp")
        self._update_crest_widgets()
        self._update_qm_widgets()
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
        self._stage_loaded = True

    def _store_stage_values(self) -> None:
        if not getattr(self, "_stage_loaded", False):
            return                                   # widgets not filled from the settings yet
        b = self.s.build
        b.keep_largest_fragment = self.w_desalt.isChecked()
        b.protonate = self.w_protonate.isChecked()
        b.ph = self.w_ph.value()
        b.max_conformers = self.w_maxconf.value()
        b.energy_window_kcal = self.w_window.stored()
        b.rmsd_threshold = self.w_rmsd.value()
        b.random_seed = self.w_seed.value()
        b.refine_method = self.w_method.currentData()
        b.fmax = self.w_fmax.value()
        b.max_steps = self.w_steps.value()
        b.temperature_k = self.w_temp.value()
        b.frequencies = self.w_freq.isChecked()
        b.batched = self.w_batched.currentData()
        b.conformer_search = self.w_confsearch.currentData()
        b.solvent = self.w_solvent.currentData()
        b.qm_method = self.qm_level() if self.w_qm_on.isChecked() else ""

    def qm_level(self) -> str:
        """'functional/basis' from the QM page."""
        func = self.w_qm_func.currentText().strip().lower() or "wb97x-d"
        basis = self.w_qm_basis.currentText().strip().lower() or "def2-svp"
        return f"{func}/{basis}"

    @staticmethod
    def _set_combo(combo: QComboBox, data) -> None:
        i = combo.findData(data)
        combo.setCurrentIndex(i if i >= 0 else 0)

    def _update_crest_widgets(self, _index=0) -> None:
        on = self.w_confsearch.currentData() == "crest"
        self.w_crest_method.setEnabled(on)
        self.w_crest_quick.setEnabled(on)
        self.w_crest_fallback.setEnabled(on)

    def _update_qm_widgets(self, _checked=False) -> None:
        on = self.w_qm_on.isChecked()
        for w in (self.w_qm_func, self.w_qm_basis, self.w_qm_task, self.w_qm_top, self.w_qm_props, self.w_qm_freq,
                  self.w_qm_cubes):
            w.setEnabled(on)

    def current_config(self) -> BuildConfig:
        self._store_stage_values()
        formats = tuple(f for f, cb in self.w_formats.items() if cb.isChecked()) or export.DEFAULT_FORMATS
        if (self.w_freq.isChecked() or (self.w_qm_on.isChecked() and self.w_qm_freq.isChecked())) \
                and "spectrum" not in formats:
            formats = (*formats, "spectrum")
        return BuildConfig.from_settings(
            self.s, refine_top=self.w_refine_top.value(), optimizer=self.w_optimizer.currentData(),
            protonation_method=self.w_prot_method.currentData(), xtb_properties=self.w_xtbprops.isChecked(),
            symmetry_number=self.w_symmetry.value(), formats=formats,
            crest_method=self.w_crest_method.currentData(), crest_quick=self.w_crest_quick.isChecked(),
            crest_fallback="rdkit" if self.w_crest_fallback.isChecked() else "fail",
            qm_task=self.w_qm_task.currentData(), qm_top=self.w_qm_top.value(),
            qm_properties=self.w_qm_props.isChecked(), qm_frequencies=self.w_qm_freq.isChecked(),
            qm_cubes=self.w_qm_cubes.isChecked())

    def _refresh_engine_lists(self) -> None:
        """Re-label methods/CREST/QM after the user changed program paths (availability may differ)."""
        from ..chem import crest, qm

        current = self.w_method.currentData()
        self.w_method.blockSignals(True)
        self.w_method.clear()
        for m in method_keys():
            self.w_method.addItem(method_label(m), m)
        self._set_combo(self.w_method, current)
        self.w_method.blockSignals(False)
        ok, _why = crest.available()
        self.w_confsearch.setItemText(1, "CREST iMTD-GC" + ("" if ok else " (not installed)"))
        ok, _why = qm.available()
        self.w_qm_on.setText("Run a QM stage" + ("" if ok else " (Psi4 not found)"))
        self._update_engine_info()

    def _update_engine_info(self) -> None:
        key = self.w_method.currentData()
        if key == "auto":
            order = [k for k in engines.AUTO_ORDER if k in engines.all_engines() and engines.available(k, quick=True)[0]]
            text = "Uses the best installed engine the molecule's elements allow: " + \
                   (" > ".join(engines.ENGINES[k].label for k in order) or "none") + " > MMFF94s."
        else:
            info = engines.all_engines()[key]
            ok, why = engines.available(key, quick=True)
            text = info.description + ("" if ok else f"\nNot available: {why}")
            solvent = self.w_solvent.currentData() if hasattr(self, "w_solvent") else ""
            if solvent and not engines.supports_solvent(key, solvent):
                text += f"\nNo implicit solvent model for {solvent}: this engine runs in the gas phase."
        self.engine_info.setText(text)

    # ================================================================== hardware
    def _detect_hardware(self) -> None:
        """GPU list and the checks that need torch (GFN2's libcint backend), in a child process: the window never
        imports torch just to label them (~500 MB). Engines import it when a run uses them."""
        if self._closing:
            return
        w = workers.FunctionWorker(resources.detect_isolated, cancel=self._probe_cancel)
        w.done.connect(self._on_probe)
        workers.start(w, self, on_failed=lambda msg: self.hw_label.setText(f"Hardware detection failed: {msg}"))

    def _on_probe(self, result) -> None:
        if result is None or self._closing:
            return                        # cancelled (the window is closing)
        rep, checks = result
        if "libcint" in checks:
            before = engines.libcint_status(check_import=False)
            engines.set_probed_libcint_status(checks["libcint"])
            if engines.libcint_status(check_import=False) != before:
                for i in range(self.w_method.count()):    # the version-based guess was wrong: relabel GFN2
                    self.w_method.setItemText(i, method_label(self.w_method.itemData(i)))
                self._update_engine_info()
        self._on_hardware(rep)

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
        rep = self.report or resources.detect_light("GPU detection is still running.")
        lines = [rep.summary(), f"OS: {rep.os}{' (WSL2)' if rep.wsl else ''}, Python {rep.python}",
                 f"PyTorch: {rep.torch_version or 'not installed'} ({rep.torch_backend})", "", "Engines:"]
        for key, info in engines.ENGINES.items():
            ok, why = engines.available(key, quick=True)
            lines.append(f"  {'✔' if ok else '✖'} {info.label} {why}")
        lines += ["", "Settings: " + str(settings_mod.settings_dir() / settings_mod.SETTINGS_FILE)] + rep.notes
        QMessageBox.information(self, "Hardware report", "\n".join(lines))

    # ================================================================== molecules
    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def add_inputs(self, items: list[inputs_mod.MoleculeInput], built: list[MoleculeResult] = ()) -> None:
        """Append molecules (one batched insert, however many) and select the last. Ids are made unique.
        ``built``: results of some of ``items`` (imports), kept in memory since they exist nowhere else."""
        existing = set(self.model.ids())
        rows = []
        for it in items:
            base, n = it.id, 2
            while it.id in existing:
                it.id = f"{base}_{n}"
                n += 1
            existing.add(it.id)
            rows.append(MoleculeRow(id=it.id, name=it.name or "", smiles=it.smiles or "", source=it.source or "",
                                    input=it))
        by_id = {r.id: r for r in rows}
        for res in built:                     # res.input is one of items: already renamed
            self.results.pin(res)
            for k, v in molecule_list.row_updates(res).items():
                setattr(by_id[res.input.id], k, v)
        self.model.add_rows(rows)
        if rows:
            self.mol_list.select_id(rows[-1].id)
            self.log(f"Added {len(rows)} molecule(s).")

    def add_result(self, res: MoleculeResult) -> None:
        """Add an already-built molecule (e.g. imported from another program's output) and select it."""
        self.add_inputs([res.input], built=[res])   # may rename the id to keep ids unique

    def _current_row(self) -> MoleculeRow | None:
        ident = self.mol_list.current_id()
        return self.model.row(ident) if ident else None

    def _input_of(self, row: MoleculeRow) -> inputs_mod.MoleculeInput:
        """The row's input: in memory, else from its run folder's job table (rows of an opened folder)."""
        if row.input is not None:
            return row.input
        if row.run_dir:
            try:
                found = jobs.load_inputs(row.run_dir, [row.id])
                if found:
                    return found[0]
            except Exception as exc:  # noqa: BLE001 - a moved or locked folder: fall back to the SMILES
                self.log(f"{row.id}: could not read its input from {row.run_dir} ({exc})")
        return inputs_mod.MoleculeInput(id=row.id, smiles=row.smiles, name=row.name, source=row.source)

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
        ids = [i for i in self.mol_list.selected_ids() if i not in self.running]
        self.model.remove_ids(ids)
        for ident in ids:
            self.results.discard(ident)

    def clear_all(self) -> None:
        if self.worker is not None:
            return
        self.model.clear()
        self.results.clear()
        self.run_folder = None
        self.conf_table.setRowCount(0)
        self.viewer.clear()
        self.viewer2d.clear()
        self._shown_key = None
        self._shown_res = None
        self.mol_info.setText("")
        self.props_view.setText("")
        self.setWindowTitle(f"QuantaBench {__version__}")

    def selected_input(self) -> inputs_mod.MoleculeInput | None:
        row = self._current_row()
        return self._input_of(row) if row is not None else None

    # -- results are read from their folders on selection, one at a time, off the GUI thread
    def _request_load(self, ident: str) -> None:
        self._wanted = ident
        if self._loading is None:
            self._start_load(ident)

    def _start_load(self, ident: str) -> None:
        row = self.model.row(ident)
        if row is None or self._closing:
            self._wanted = None
            return
        self._loading = ident
        w = workers.FunctionWorker(molecule_list.load_result, row.id, row.folder, row.run_dir, row.input)
        workers.start(w, self, on_done=self._on_loaded, on_failed=self._on_load_failed)

    def _on_loaded(self, res: MoleculeResult) -> None:
        ident, self._loading = self._loading, None
        if self.model.row(ident) is not None and res is not None:
            self.results.put(res)
        self._after_load(ident)

    def _on_load_failed(self, msg: str) -> None:
        ident, self._loading = self._loading, None
        row = self.model.row(ident)
        if row is not None:
            self.log(f"{ident}: could not read its result from {row.folder or row.run_dir}: {msg}")
            stand_in = self.results.stand_in(row)
            stand_in.status, stand_in.error = "ok", f"Built, but its files could not be read: {msg}"
            self.results.put(stand_in)       # not retried on every selection; reopening the folder does
        self._after_load(ident)

    def _after_load(self, ident: str) -> None:
        wanted, self._wanted = self._wanted, None
        if self._closing:
            return
        if self.mol_list.current_id() == ident:
            self._on_molecule_selected()
        if wanted and wanted != ident and self.results.cached(wanted) is None and self.results.loadable(wanted):
            self._request_load(wanted)

    def _on_molecule_selected(self, *_args) -> None:
        row = self._current_row()
        if row is None:
            return
        res = self.results.cached(row.id)
        if res is None and row.status == "ok" and not row.stage and self.results.loadable(row.id):
            self._request_load(row.id)       # shown when read (a few ms); the views keep the last one meanwhile
            self.input_detail.setText(f"<b>{row.id}</b> {row.name}<br>SMILES: {row.smiles}")
            self.mol_info.setText(f"<b>{row.id}</b>: reading the results...")
            self._fill_conformers(None)
            self._shown_res = None
            return
        if res is None and row.status in ("failed", "cancelled"):
            res = self.results.stand_in(row)
        inp = res.input if res is not None else self._input_of(row)
        self.input_detail.setText(f"<b>{inp.id}</b> {inp.name}<br>SMILES: {inp.smiles}<br>Source: {inp.source}")
        self._update_stage_status(res)
        self._fill_conformers(res)
        built = _built(res)
        self._shown_res = res if built else None
        if built:
            self._show_info(res)
        else:
            self.mol_info.setText(res.error if res and res.error else "Not built yet. Press Run.")
            self.props_view.setText("")
        if self._surface_preview:
            self.resultChanged.emit(res if built else None)
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
        self.resultChanged.emit(res if built else None)   # after the views: panels add to the 3D scene

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
        u = self.energy_display()
        info = engines.all_engines().get(res.method)
        bits = [f"<b>{res.input.id}</b> {d.get('formula', '')}, charge {res.charge:+d}",
                f"{info.label} on {res.device}" if info is not None else res.method,
                f"{res.n_embedded} embedded → {len(res.conformers)} conformers"
                if res.conformer_search != "imported" else res.input.source]
        if res.solvent:
            bits[1] += f", {res.solvent} (implicit)"
        for key, label, unit in (("volume_A3_boltz", "Volume", "Å³"), ("sasa_A2_boltz", "SASA", "Å²"),
                                 ("dipole_debye_boltz", "Dipole", "D"), ("xtb_dipole_debye", "Dipole (xTB)", "D"),
                                 ("xtb_polarizability_iso_A3", "Polarizability (xTB)", "Å³"),
                                 ("qm_dipole_debye", "Dipole (QM)", "D"),
                                 ("qm_polarizability_iso_A3", "Polarizability (QM)", "Å³"),
                                 ("qm_surf_volume_A3", "Volume, 0.002 au (QM)", "Å³"),
                                 ("qm_gap_ev", "HOMO-LUMO gap (QM)", "eV")):     # an orbital energy: stays in eV
            if isinstance(d.get(key), (int, float)):
                bits.append(f"{label}: {d[key]:.2f} {unit}")
        if res.qm:
            e = res.qm.get("energy_ev")
            bits.append(f"QM: {res.qm.get('level', '')} ({res.qm.get('program', '')})"
                        + (f", E = {u.with_unit(e, units.EV, 'absolute', 5)}" if isinstance(e, (int, float))
                           else ""))
        vib = res.vibrations
        thermo = (vib or {}).get("thermo")
        if thermo:
            bits.append(f"ZPE {u.with_unit(thermo['zpe_kcal'], units.KCAL, 'relative', 2)}, "
                        f"S {u.entropy_text(thermo['entropy_cal_molK'], 1)}, "
                        f"G − E {u.with_unit(thermo['g_corr_kcal'], units.KCAL, 'relative', 2)} "
                        f"at {thermo['temperature_k']:.0f} K"
                        + (f"; <b>{vib['n_imaginary']} imaginary</b>" if vib["n_imaginary"] else ""))
        bits += [f"<i>{n}</i>" for n in res.notes]
        self.mol_info.setText("<br>".join(bits))
        lines = []
        if thermo:
            def corr(key):            # H − E, G − E: a relative energy, with its unit when it differs from H's
                text = u.text(thermo[key], units.KCAL, "relative", 3)
                return text if u.relative == u.absolute else f"{text} {u.label('relative')}"

            lines += [f"<b>Thermochemistry</b> ({vib['method']}, {vib['hessian_method']} Hessian, "
                      f"{thermo['temperature_k']:.2f} K, {'qRRHO' if thermo['qrrho'] else 'RRHO'})",
                      f"ZPE: {u.with_unit(thermo['zpe_kcal'], units.KCAL, 'relative', 3)}",
                      f"H: {u.with_unit(thermo['enthalpy_kcal'], units.KCAL, 'absolute', 3)} "
                      f"(H − E {corr('h_corr_kcal')})",
                      f"S: {u.entropy_text(thermo['entropy_cal_molK'], 3)}",
                      f"G: {u.with_unit(thermo['gibbs_kcal'], units.KCAL, 'absolute', 3)} "
                      f"(G − E {corr('g_corr_kcal')})",
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
            u = self.energy_display()
            for r, c in enumerate(res.conformers):
                vals = [str(c.rank + 1), u.text(c.rel_kcal, units.KCAL, "relative", 2), f"{100 * c.weight:.1f}",
                        u.text(c.energy_ev, units.EV, "absolute", 5),
                        "yes" if c.converged else "no", str(c.steps),
                        f"{c.dipole_debye:.2f}" if c.dipole_debye is not None else ""]
                for col, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.conf_table.setItem(r, col, item)
        self.conf_table.blockSignals(False)

    def _on_conformer_selected(self, row, _col, _prow, _pcol) -> None:
        res = self._shown_res
        if res and 0 <= row < len(res.conformers) and not self._surface_preview:
            self._show_conformer(res, row, reset_camera=False)

    def _on_stage_selected(self, cur, _prev) -> None:
        if cur is not None:
            self.stage_stack.setCurrentIndex(self.stage_pages[cur.data(0, Qt.ItemDataRole.UserRole)])

    # ================================================================== running
    def run_build(self, selected_only: bool = False) -> None:
        if self.worker is not None:
            return
        ids = self.mol_list.selected_ids() if selected_only else self.model.ids()
        if not ids:
            QMessageBox.information(self, "Run", "Add a molecule first (SMILES, name, or a file).")
            return
        rows = [self.model.row(i) for i in ids]
        cfg = self.current_config()
        settings_mod.save(self.s)
        if all(r.input is not None for r in rows):
            todo = [r.input for r in rows]
        else:       # rows of an opened run folder: their inputs are read from its job table on the worker thread
            from functools import partial

            todo = partial(molecule_list.gather_inputs,
                           [(r.id, r.run_dir, r.input, r.smiles, r.name, r.source) for r in rows])
        run_dir = None
        if len(rows) > 1 and cfg.write_files:  # a batch goes through a job table: resumable, results stay on disk
            from ..chem.pipeline import new_run_dir

            run_dir = new_run_dir(cfg.output_dir or self.s.output_path(), f"batch_{len(rows)}")
        self._start_worker(todo, cfg, run_dir, ids, total=len(ids), done=0)

    def _start_worker(self, todo, cfg: BuildConfig, run_dir: Path | None, ids: list[str], total: int,
                      done: int) -> None:
        self.worker = workers.BuildWorker(todo, cfg, run_dir=run_dir)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log)
        self.worker.molecule_done.connect(self._on_molecule_done)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self._batch_dir = Path(run_dir) if run_dir is not None else None
        self._batch_ids = list(ids)
        self.running = dict.fromkeys(ids, "queued")
        self.model.update_many(ids, stage="queued")
        self._set_running(True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.bottom_tabs.setCurrentWidget(self.log_view)
        workers.start(self.worker, self)

    def cancel_build(self, _checked=False, confirm: bool = True) -> None:
        if self.worker is None:
            return
        remaining = len(self.running)
        if confirm and remaining > 1 and QMessageBox.question(
                self, "Cancel build", f"Stop the build? {remaining} molecule(s) have not finished.\n"
                "Finished molecules are kept; Build > Resume run folder (or quantabench-cli build --resume) "
                "continues the batch.") != QMessageBox.StandardButton.Yes:
            return
        self.worker.cancel()
        self.log("Cancelling after the current step...")

    # ================================================================== run folders
    def open_run_folder(self, _checked=False, folder: str | Path | None = None) -> None:
        """List the molecules of a run folder (finished or partial) from its job table, or the summary.csv of a
        non-resumable batch. Only light rows are read (a 14,000-molecule folder in well under a second); each
        molecule's full result is read from its folder when it is selected."""
        if self.worker is not None:
            QMessageBox.information(self, "Open run folder", "Wait for the build to finish (or cancel it) first.")
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Open run folder", str(self.s.output_path()))
        if not folder:
            return
        folder = Path(folder)
        if not (folder / jobs.DB_NAME).is_file() and not (folder / "summary.csv").is_file():
            QMessageBox.warning(self, "Open run folder", f"{folder} is not a QuantaBench run folder "
                                f"(no {jobs.DB_NAME} or summary.csv).")
            return
        n = self.model.rowCount()
        if n and QMessageBox.question(self, "Open run folder", f"Replace the {n:,} molecule(s) in the list with "
                                      f"those of {folder.name}?") != QMessageBox.StandardButton.Yes:
            return
        self.statusBar().showMessage(f"Reading {folder}...")

        def failed(msg):
            self.statusBar().clearMessage()
            QMessageBox.warning(self, "Open run folder", msg)

        workers.start(workers.FunctionWorker(molecule_list.read_run_folder, folder), self,
                      on_done=lambda value: self._on_run_folder_read(folder, value), on_failed=failed)

    def _on_run_folder_read(self, folder: Path, value) -> None:
        rows, status = value
        self.statusBar().clearMessage()
        if self.worker is not None or self._closing:
            return
        self.clear_all()
        self.model.set_rows(rows)
        self.run_folder = self.last_run_dir = folder
        self.out_label.setText(f"Run folder: {folder}")
        self.setWindowTitle(f"QuantaBench {__version__} - {folder.name}")
        self.log(f"Opened {folder}: " + (status.summary() if status is not None else
                                          f"{len(rows)} molecule(s) from summary.csv (no job table: not resumable)"))
        if status is not None and status.remaining and not status.active:
            self.log("  Build > Resume run folder continues the unfinished ones with the folder's settings.")
        if rows:
            self.mol_list.setCurrentRow(0)
        self._update_counts()

    def resume_run(self, _checked=False, folder: str | Path | None = None) -> None:
        """Continue the unfinished molecules of the opened run folder (or of the last batch) with the science
        settings stored in it (method, conformers, solvent, ...); threads, GPU and lanes are this machine's."""
        if self.worker is not None:
            return
        folder = Path(folder) if folder else (self.run_folder or self.last_run_dir)
        if folder is None or not (folder / jobs.DB_NAME).is_file():
            QMessageBox.information(self, "Resume run folder", "Open a run folder first (File > Open run folder...): "
                                    "the folder of a batch, with its jobs.sqlite.")
            return
        try:
            st = jobs.open_run(folder)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Resume run folder", str(exc))
            return
        if st.active:
            QMessageBox.warning(self, "Resume run folder", f"{folder} is being run by another process right now.")
            return
        if not st.remaining:
            QMessageBox.information(self, "Resume run folder", f"Every molecule of {folder.name} is finished "
                                    f"({st.summary()}).")
            return
        cfg = jobs.stored_config(st.config, BuildConfig.from_settings(self.s))
        ids = [r.id for r in self.model.rows()
               if r.status not in ("ok", "failed") and Path(r.run_dir or "") == folder]
        self.log(f"Resuming {folder}: {st.remaining} molecule(s) to do, method {cfg.method}")
        self._start_worker(None, cfg, folder, ids, total=st.total, done=st.finished)

    def _set_running(self, on: bool) -> None:
        self.progress.setVisible(on)
        for w in (self.run_btn, self.run_sel_btn, self.a_run):
            w.setEnabled(not on)
        self.cancel_btn.setEnabled(on)
        self.a_cancel.setEnabled(on)

    def _on_progress(self, i, n, stage, ident) -> None:
        if self.progress.maximum() != n:
            self.progress.setMaximum(max(1, n))
        if stage in ("ok", "failed", "cancelled"):
            self.progress.setValue(i)
            return
        self.running[ident] = stage
        self.model.update(ident, stage=stage)
        self.statusBar().showMessage(f"[{i + 1}/{n}] {ident}: {stage}")

    def _on_molecule_done(self, res: MoleculeResult) -> None:
        """A finished molecule becomes a summary row; its full result is kept only if it is selected (or exists
        nowhere else), so a batch of any size runs with flat memory."""
        ident = res.input.id
        self.running.pop(ident, None)
        fields = molecule_list.row_updates(res)
        if self._batch_dir is not None:
            fields["run_dir"] = str(self._batch_dir)
        self.model.update(ident, **fields)
        selected = self.mol_list.current_id() == ident
        self.results.discard(ident)                  # an older result of it must not come back from the cache
        if res.status == "ok" and not fields["folder"]:
            self.results.pin(res)                    # nothing on disk (file writing off): memory is all there is
        elif selected:
            self.results.put(res)
        if selected:
            self._on_molecule_selected()

    def _end_batch(self) -> None:
        ids = list(self.running)
        self.running.clear()
        self.model.update_many(ids, stage="")        # queued but not started (cancel, failure)
        if self._batch_dir is not None and (self._batch_dir / jobs.DB_NAME).is_file():
            self.model.update_many(self._batch_ids, run_dir=str(self._batch_dir))   # Resume finds them
        self._batch_dir, self._batch_ids = None, []
        self._count_timer.stop()
        self._update_counts()                        # the final counts now, not after the debounce
        self.worker = None

    def _on_finished(self, results, run_dir) -> None:
        self._end_batch()
        self._set_running(False)
        self.last_run_dir = run_dir
        ok = sum(r.status == "ok" for r in results)
        fell_back = fallback_count(results)
        if fell_back:
            self.log(f"WARNING: {fell_back} of {len(results)} molecule(s) got RDKit conformers because CREST failed "
                     "(their notes, summary.csv conformer_fallback)")
        if self._quit_pending:
            return                       # closing: keep "Closing after the current step..." on screen
        self.statusBar().showMessage(f"Done: {ok}/{len(results)} built"
                                     + (f", {fell_back} with RDKit conformers (CREST failed)" if fell_back else "")
                                     + (f", saved to {run_dir}" if run_dir else ""), 15000)
        if run_dir:
            self.out_label.setText(f"Last run: {run_dir}")
        self.bottom_tabs.setCurrentWidget(self.conf_table)

    def _on_failed(self, msg) -> None:
        self._end_batch()
        self._set_running(False)
        self.log(msg)
        if not self._quit_pending:
            QMessageBox.critical(self, "Build failed", msg.splitlines()[0])

    # ================================================================== export & tools
    def export_selected(self) -> None:
        res = self._full_current()
        if res is None:
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
        res = self._full_current()
        if res is None:
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
        if self._current_row() is not None:
            self._shown_key = None
            self._on_molecule_selected()

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
        res = self._shown_res
        row = max(0, self.conf_table.currentRow())
        charges = res.conformers[row].charges if res is not None and row < len(res.conformers) else None
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
        accepted = dlg.exec()
        dlg.deleteLater()
        if not accepted:
            return
        s = dlg.result_settings()
        if settings_mod.settings_dir() != before:
            loaded = settings_mod.load()
            if (settings_mod.settings_dir() / settings_mod.SETTINGS_FILE).exists():
                s = loaded
        self.s = s
        settings_mod.save(self.s)
        resources.apply_thread_limits(self.s.resources.threads)
        from ..chem import external, torch_compat

        torch_compat.apply_settings(self.s, strict=False)   # dxtb linear algebra / GPU modes (Settings > Compute)
        external.clear_cache()            # program paths may have changed
        self._refresh_engine_lists()
        self._load_stage_values()
        self.apply_display()              # energy units (Settings > Display)
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
        self._probe_cancel.set()
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
