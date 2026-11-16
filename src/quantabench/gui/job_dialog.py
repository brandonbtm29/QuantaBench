"""The job dialog (UI-5): submit molecules as one job, a level of theory × a task × resources.

Build > New job... (Ctrl+J) and the Pipeline dock's New job... open it, filled from the Pipeline dock's settings. Four
pages:

    Job         the name, which molecules (selected or all), a recipe (``chem.recipes``: Screen, Standard, Careful and
                the user's own) and the task (conformer distribution, optimize, single point, frequencies; scan later)
    Level       the level of theory by family (force field, tight binding, semi-empirical, ML potential; engines that
                are not installed are listed, disabled, with why), solvent, charge and multiplicity, the conformer
                search, the rank step and the DFT (QM) stage
    Resources   where it runs (this computer; the workers are listed for DIST-3), threads, priority, GPU, memory and
                the time limit per molecule
    Review      the review of the molecules for this job (``chem.advisor``, as the Review window): findings, the Fast
                screen / Recommended / Careful presets, and Apply, which sets the dialog's fields

Submit puts the job in the queue (``gui.job_queue``; the Jobs dock shows it); it starts at once when nothing runs.
The dialog never changes the Pipeline dock: a job carries its own settings. Save as recipe... keeps the dialog's level,
task and conformer settings as a recipe in the settings.
"""

from __future__ import annotations

import html
import threading
from dataclasses import replace

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..chem import recipes, solvents
from ..chem.pipeline import BuildConfig
from . import workers

QM_FUNCTIONALS = ("wb97x-d", "b3lyp-d3bj", "pbe0-d3bj", "r2scan", "m06-2x", "hf")
QM_BASES = ("def2-svp", "def2-svpd", "def2-tzvp", "6-31g*", "6-31+g*", "6-311+g**", "aug-cc-pvdz")
CREST_METHODS = (("gfnff", "GFN-FF (fast; the Nov 14 test's choice)"), ("gfn2", "GFN2-xTB"),
                 ("gfn2//gfnff", "GFN2-xTB // GFN-FF (needs xtb)"), ("gfn1", "GFN1-xTB"))
PRIORITIES = (("normal", "Normal"), ("low", "Low"), ("background", "Background (Brandon's runs: half capacity)"))
GPU_CHOICES = (("auto", "Auto (when PyTorch sees one)"), ("on", "On (require it)"), ("off", "Off (CPU only)"))
FROM_STRUCTURE = "From the structure"
ID_ROLE = Qt.ItemDataRole.UserRole
REVIEW_TAB = 3


def _set(combo: QComboBox, value) -> bool:
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)
    return i >= 0


def _note(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    lab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    lab.setStyleSheet("color: gray;")
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def _split_qm(level: str) -> tuple[str, str]:
    func, _, basis = (level or "").partition("/")
    return func, basis


class _Progress(QObject):
    changed = Signal(int, int)


class JobDialog(QDialog):
    """See the module docstring. ``ids``: the molecules (default: the selected ones, else all)."""

    submitted = Signal(object)          # the queued gui.job_queue.Job

    def __init__(self, win, ids: list[str] | None = None, available: set[str] | None = None,
                 programs: set[str] | None = None):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("New job")
        self.setObjectName("job_dialog")
        self.resize(760, 640)
        self.base: BuildConfig = win.current_config()
        self.levels = recipes.levels(quick=True)
        self.available = available if available is not None else {lv.key for lv in self.levels if lv.available}
        self.programs = programs if programs is not None else recipes._programs()
        self.review = None
        self._loading = False
        self._review_gen = 0
        self._review_stop = threading.Event()
        self._progress = _Progress(self)
        self._progress.changed.connect(self._on_review_progress)
        sel = list(win.mol_list.selected_ids())
        self.all_ids = list(win.model.ids())
        self.sel_ids = sel
        self.fixed_ids = list(ids) if ids is not None else None

        self.tabs = QTabWidget()
        self.tabs.addTab(self._job_page(), "Job")
        self.tabs.addTab(self._level_page(), "Level of theory")
        self.tabs.addTab(self._resources_page(), "Resources")
        self.tabs.addTab(self._review_page(), "Review")
        self.tabs.currentChanged.connect(self._on_tab)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight: bold;")
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet("color: #d35400;")
        self.save_btn = QPushButton("Save as recipe...")
        self.save_btn.setToolTip("Keep this job's level of theory, task and conformer settings as a recipe "
                                 "(Settings: synced to your other machines)")
        self.save_btn.clicked.connect(lambda _c=False: self.save_recipe())
        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setDefault(True)
        self.submit_btn.setToolTip("Queue the job (Jobs dock); it starts at once when nothing else runs")
        self.submit_btn.clicked.connect(lambda _c=False: self.submit())
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.submit_btn)
        buttons.addWidget(cancel)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(self.summary)
        lay.addWidget(self.problems)
        lay.addLayout(buttons)

        self.load_config(self.base)
        self._connect_changes()
        self._update()

    # ================================================================== pages
    def _job_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        form = QFormLayout()
        outer.addLayout(form)
        outer.addStretch(1)
        self.w_name = QLineEdit()
        self.w_name.setPlaceholderText("automatic (recipe: task, molecules)")
        form.addRow("Name", self.w_name)
        self.w_sel = QRadioButton()
        self.w_all = QRadioButton()
        mols = QHBoxLayout()
        mols.addWidget(self.w_sel)
        mols.addWidget(self.w_all)
        mols.addStretch(1)
        form.addRow("Molecules", mols)
        self.w_recipe = QComboBox()
        self.w_recipe.setObjectName("job_recipe")
        self._fill_recipes()
        self.w_recipe_del = QPushButton("Delete")
        self.w_recipe_del.setToolTip("Delete this recipe of yours from the settings")
        self.w_recipe_del.clicked.connect(lambda _c=False: self.delete_recipe())
        row = QHBoxLayout()
        row.addWidget(self.w_recipe, 1)
        row.addWidget(self.w_recipe_del)
        form.addRow("Recipe", row)
        self.recipe_info = _note()
        form.addRow("", self.recipe_info)
        self.w_task = QComboBox()
        self.w_task.setObjectName("job_task")
        model = QStandardItemModel(self.w_task)
        self.w_task.setModel(model)
        for t in recipes.TASKS.values():
            self.w_task.addItem(t.label, t.key)
            item = model.item(self.w_task.count() - 1)
            item.setToolTip(t.description if t.available else f"{t.description} {t.why}")
            if not t.available:
                item.setEnabled(False)
        form.addRow("Task", self.w_task)
        self.task_info = _note()
        form.addRow("", self.task_info)
        if self.fixed_ids is not None:
            self.w_sel.setText(f"These {len(self.fixed_ids):,}")
            self.w_sel.setChecked(True)
            self.w_all.setVisible(False)
        else:
            self.w_sel.setText(f"Selected ({len(self.sel_ids):,})")
            self.w_all.setText(f"All in the list ({len(self.all_ids):,})")
            self.w_sel.setEnabled(bool(self.sel_ids))
            (self.w_sel if len(self.sel_ids) > 1 else self.w_all).setChecked(True)
        return page

    def _fill_recipes(self) -> None:
        cur = self.w_recipe.currentData() if self.w_recipe.count() else recipes.CUSTOM
        self.w_recipe.blockSignals(True)
        self.w_recipe.clear()
        self.w_recipe.addItem("The Pipeline dock's settings", recipes.CUSTOM)
        for r in recipes.all_recipes(self.win.s):
            label = r.name if r.builtin else f"{r.name} (yours)"
            self.w_recipe.addItem(label, r.key)
            self.w_recipe.setItemData(self.w_recipe.count() - 1, r.description, Qt.ItemDataRole.ToolTipRole)
        _set(self.w_recipe, cur)
        self.w_recipe.blockSignals(False)

    def _level_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        form = QFormLayout()
        self.w_method = QComboBox()
        self.w_method.setObjectName("job_method")
        model = QStandardItemModel(self.w_method)
        self.w_method.setModel(model)
        family = None
        for lv in self.levels:
            if lv.family != family and lv.family:
                family = lv.family
                self.w_method.addItem(f"— {recipes.FAMILY_LABELS.get(family, family)} —", None)
                model.item(self.w_method.count() - 1).setEnabled(False)
            label = lv.label + ("" if lv.available else "  (not installed)")
            self.w_method.addItem(label, lv.key)
            item = model.item(self.w_method.count() - 1)
            item.setToolTip(lv.why or lv.label)
            if not lv.available:
                item.setEnabled(False)
        form.addRow("Level of theory", self.w_method)
        self.w_solvent = QComboBox()
        self.w_solvent.addItem("Gas phase", "")
        for name, sv in solvents.SOLVENTS.items():
            self.w_solvent.addItem(f"{sv.label} (ε = {sv.epsilon:g})", name)
        form.addRow("Solvent", self.w_solvent)
        self.level_info = _note()
        form.addRow("", self.level_info)
        self.w_charge = QSpinBox()
        self.w_charge.setRange(-11, 10)
        self.w_charge.setSpecialValueText(FROM_STRUCTURE)
        self.w_charge.setValue(-11)
        self.w_charge.setToolTip("The total charge of every molecule in the job; 'From the structure' takes each one's "
                                 "own (after protonation at the pH)")
        self.w_mult = QSpinBox()
        self.w_mult.setRange(0, 7)
        self.w_mult.setSpecialValueText(FROM_STRUCTURE)
        self.w_mult.setToolTip("Spin multiplicity 2S+1 of every molecule; 'From the structure' takes each one's own")
        cm = QHBoxLayout()
        cm.addWidget(QLabel("Charge"))
        cm.addWidget(self.w_charge)
        cm.addSpacing(12)
        cm.addWidget(QLabel("Multiplicity"))
        cm.addWidget(self.w_mult)
        cm.addStretch(1)
        form.addRow("", cm)
        lay.addLayout(form)

        conf = QGroupBox("Conformers")
        cf = QFormLayout(conf)
        self.w_search = QComboBox()
        self.w_search.addItem("RDKit ETKDGv3 + MMFF94s", "rdkit")
        crest_ok = "crest" in self.programs
        self.w_search.addItem("CREST iMTD-GC" + ("" if crest_ok else "  (not found: RDKit instead, flagged)"),
                              "crest")
        self.w_search.addItem("Auto: CREST for flexible neutral molecules, RDKit for the rest"
                              + ("" if crest_ok else "  (CREST not found)"), "auto")
        self.w_crest_method = QComboBox()
        for k, label in CREST_METHODS:
            self.w_crest_method.addItem(label, k)
        self.w_crest_quick = QCheckBox("Quick")
        self.w_crest_quick.setToolTip("CREST --quick: fewer metadynamics runs, minutes instead of hours")
        crest_row = QHBoxLayout()
        crest_row.addWidget(self.w_crest_method, 1)
        crest_row.addWidget(self.w_crest_quick)
        self.w_crest_min_rotors = QSpinBox()
        self.w_crest_min_rotors.setRange(0, 100)
        self.w_crest_min_rotors.setSuffix(" rotatable bonds")
        self.w_crest_min_rotors.setToolTip("Auto: a neutral molecule with at least this many rotatable bonds gets CREST; "
                                           "ions, zwitterions and smaller molecules get RDKit")
        self.w_crest_with_rdkit = QCheckBox("Add RDKit's conformers to CREST's pool")
        self.w_crest_with_rdkit.setToolTip("Where CREST runs, RDKit's search runs too and the lowest of each pool are "
                                           "refined, so the refinement keeps the lower of both")
        self.w_max_confs = QSpinBox()
        self.w_max_confs.setRange(1, 5000)
        self.w_refine_top = QSpinBox()
        self.w_refine_top.setRange(1, 1000)
        self.w_rank = QComboBox()
        from ..chem import rank

        for k, label in rank.CHOICES:
            self.w_rank.addItem(label, k)
        cf.addRow("Search", self.w_search)
        cf.addRow("CREST level", crest_row)
        cf.addRow("CREST from", self.w_crest_min_rotors)
        cf.addRow("", self.w_crest_with_rdkit)
        cf.addRow("Conformers (RDKit)", self.w_max_confs)
        cf.addRow("Refine the lowest", self.w_refine_top)
        cf.addRow("Rank step", self.w_rank)
        self.conf_box = conf
        lay.addWidget(conf)

        qm = QGroupBox("DFT (QM stage, Psi4)")
        qf = QFormLayout(qm)
        psi4 = "psi4" in self.programs
        self.w_qm_on = QCheckBox("DFT on the lowest conformers" + ("" if psi4 else "  (Psi4 not found)"))
        self.w_qm_on.setEnabled(psi4)
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
        lvl = QHBoxLayout()
        lvl.addWidget(self.w_qm_func, 1)
        lvl.addWidget(QLabel("/"))
        lvl.addWidget(self.w_qm_basis, 1)
        qf.addRow(self.w_qm_on)
        qf.addRow("Functional / basis", lvl)
        qf.addRow("Task", self.w_qm_task)
        qf.addRow("Conformers", self.w_qm_top)
        self.qm_box = qm
        lay.addWidget(qm)
        lay.addStretch(1)
        return page

    def _resources_page(self) -> QWidget:
        from .. import priority
        from ..dist import client, profile

        page = QWidget()
        outer = QVBoxLayout(page)
        form = QFormLayout()
        outer.addLayout(form)
        outer.addStretch(1)
        self.w_run_on = QComboBox()
        self.w_run_on.setObjectName("job_run_on")
        model = QStandardItemModel(self.w_run_on)
        self.w_run_on.setModel(model)
        me = profile.machine_name()
        self.w_run_on.addItem(f"This computer ({me})", recipes.LOCAL)
        try:
            from .. import settings as settings_mod

            others = [n for n in client.known_workers() if settings_mod.machine_key(n) != settings_mod.machine_key(me)]
        except Exception:  # noqa: BLE001 - the registry is optional here
            others = []
        for name in others:
            self.w_run_on.addItem(f"{name} (later: DIST-3)", name)
            item = model.item(self.w_run_on.count() - 1)
            item.setEnabled(False)
            item.setToolTip("Routing a job to a worker from this dialog is the next step (DIST-3). Today "
                            "quantabench-cli grid build runs a batch across machines.")
        form.addRow("Run on", self.w_run_on)
        self.w_threads = QSpinBox()
        self.w_threads.setRange(0, 512)
        self.w_threads.setSpecialValueText(f"Auto ({priority.budget_of(self.base)} here)")
        self.w_threads.setToolTip("CPU threads per molecule; Auto: this machine's threads minus the free threads at the "
                                  "priority (Settings)")
        form.addRow("Threads", self.w_threads)
        self.w_priority = QComboBox()
        for k, label in PRIORITIES:
            self.w_priority.addItem(label, k)
        self.w_priority.setToolTip("CPU and I/O priority of the work (never the window's)")
        form.addRow("Priority", self.w_priority)
        self.w_gpu = QComboBox()
        for k, label in GPU_CHOICES:
            self.w_gpu.addItem(label, k)
        self.w_gpu_index = QSpinBox()
        self.w_gpu_index.setRange(0, 15)
        self.w_gpu_index.setPrefix("device ")
        gpu = QHBoxLayout()
        gpu.addWidget(self.w_gpu, 1)
        gpu.addWidget(self.w_gpu_index)
        form.addRow("GPU", gpu)
        self.w_qm_mem = QDoubleSpinBox()
        self.w_qm_mem.setRange(0, 1024)
        self.w_qm_mem.setDecimals(1)
        self.w_qm_mem.setSuffix(" GB")
        self.w_qm_mem.setSpecialValueText("Settings (Engines)")
        self.w_qm_mem.setToolTip("Memory per Psi4 job. The b03 ESP pass: give 12-16 GB, or DF-SCF goes to disk and "
                                 "molecules time out")
        form.addRow("Memory per DFT job", self.w_qm_mem)
        self.w_mem = QDoubleSpinBox()
        self.w_mem.setRange(-1, 1024)
        self.w_mem.setDecimals(1)
        self.w_mem.setSuffix(" GB")
        self.w_mem.setSpecialValueText("No guard")
        self.w_mem.setToolTip("Memory budget of one process for the dxtb steps; 0 = auto (available memory / lanes)")
        form.addRow("Memory budget, tight binding", self.w_mem)
        self.w_timeout = QSpinBox()
        self.w_timeout.setRange(0, 7 * 24 * 3600)
        self.w_timeout.setSingleStep(300)
        self.w_timeout.setSuffix(" s")
        self.w_timeout.setSpecialValueText("None")
        self.w_timeout.setToolTip("Wall-clock limit per molecule in a batch; a molecule past it is marked failed")
        form.addRow("Time limit per molecule", self.w_timeout)
        form.addRow("", _note("Lanes (parallel molecules) are this machine's Settings > Resources > Parallel lanes."))
        return page

    def _review_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        head = QHBoxLayout()
        self.review_title = QLabel("The review runs when this page opens.")
        self.review_title.setWordWrap(True)
        self.review_btn = QPushButton("Review again")
        self.review_btn.clicked.connect(lambda _c=False: self.start_review())
        head.addWidget(self.review_title, 1)
        head.addWidget(self.review_btn)
        lay.addLayout(head)
        self.findings = QTreeWidget()
        self.findings.setHeaderLabels(["Finding", "Molecules"])
        self.findings.setRootIsDecorated(False)
        self.findings.setWordWrap(True)
        self.findings.header().setStretchLastSection(False)
        from PySide6.QtWidgets import QHeaderView

        self.findings.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.findings.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.findings.currentItemChanged.connect(self._on_finding)
        self.finding_detail = QTextBrowser()
        self.finding_detail.setOpenLinks(False)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.findings)
        split.addWidget(self.finding_detail)
        split.setSizes([260, 160])
        lay.addWidget(split, 1)
        row = QHBoxLayout()
        self.apply_finding_btn = QPushButton("Apply the finding")
        self.apply_finding_btn.setEnabled(False)
        self.apply_finding_btn.clicked.connect(lambda _c=False: self.apply_finding())
        row.addWidget(self.apply_finding_btn)
        row.addSpacing(16)
        row.addWidget(QLabel("Presets:"))
        from ..chem import advisor

        self.preset_buttons = {}
        for key, name in advisor.PRESET_NAMES.items():
            b = QPushButton(name)
            b.setEnabled(False)
            b.clicked.connect(lambda _c=False, k=key: self.apply_preset(k))
            self.preset_buttons[key] = b
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.review_cost = _note()
        lay.addWidget(self.review_cost)
        return page

    # ================================================================== the configuration
    def _connect_changes(self) -> None:
        for w in (self.w_method, self.w_solvent, self.w_search, self.w_crest_method, self.w_rank, self.w_qm_func,
                  self.w_qm_basis, self.w_qm_task, self.w_run_on, self.w_priority, self.w_gpu):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.w_qm_func, self.w_qm_basis):
            w.editTextChanged.connect(self._changed)
        for w in (self.w_max_confs, self.w_refine_top, self.w_qm_top, self.w_threads, self.w_gpu_index,
                  self.w_timeout, self.w_charge, self.w_mult):
            w.valueChanged.connect(self._changed)
        for w in (self.w_qm_mem, self.w_mem):
            w.valueChanged.connect(self._changed)
        self.w_crest_min_rotors.valueChanged.connect(self._changed)
        for w in (self.w_crest_quick, self.w_crest_with_rdkit, self.w_qm_on, self.w_sel, self.w_all):
            w.toggled.connect(self._changed)
        self.w_name.textChanged.connect(self._changed)
        self.w_task.currentIndexChanged.connect(self._changed)
        self.w_recipe.currentIndexChanged.connect(self._on_recipe)

    def load_config(self, cfg: BuildConfig) -> None:
        """Set the fields from ``cfg`` (the Pipeline dock's, a recipe's, a preset's)."""
        self._loading = True
        try:
            if not _set(self.w_method, cfg.method):
                _set(self.w_method, "auto")
            _set(self.w_solvent, solvents.canonical(cfg.solvent) if cfg.solvent else "")
            _set(self.w_search, cfg.conformer_search)
            _set(self.w_crest_method, cfg.crest_method)
            self.w_crest_quick.setChecked(cfg.crest_quick)
            self.w_crest_min_rotors.setValue(cfg.crest_min_rotors)
            self.w_crest_with_rdkit.setChecked(cfg.crest_with_rdkit)
            self.w_max_confs.setValue(cfg.max_conformers)
            self.w_refine_top.setValue(cfg.refine_top)
            _set(self.w_rank, cfg.rank_method)
            func, basis = _split_qm(cfg.qm_method)
            self.w_qm_on.setChecked(bool(func) and self.w_qm_on.isEnabled())
            if func:
                self.w_qm_func.setCurrentText(func)
                self.w_qm_basis.setCurrentText(basis)
            _set(self.w_qm_task, cfg.qm_task)
            self.w_qm_top.setValue(cfg.qm_top)
            self.w_threads.setValue(cfg.threads)
            _set(self.w_priority, cfg.priority)
            _set(self.w_gpu, cfg.gpu)
            self.w_gpu_index.setValue(cfg.gpu_index)
            self.w_qm_mem.setValue(cfg.qm_memory_gb)
            self.w_mem.setValue(cfg.memory_budget_gb)
            self.w_timeout.setValue(int(cfg.molecule_timeout_s))
            self.cfg = cfg
        finally:
            self._loading = False

    def config(self) -> BuildConfig:
        """The dialog's configuration before the task's changes (``spec().config()`` has them)."""
        func, basis = self.w_qm_func.currentText().strip().lower(), self.w_qm_basis.currentText().strip().lower()
        qm_method = f"{func}/{basis}" if self.w_qm_on.isChecked() and func else ""
        return replace(self.cfg, method=self.w_method.currentData() or "auto", solvent=self.w_solvent.currentData(),
                       conformer_search=self.w_search.currentData(), crest_method=self.w_crest_method.currentData(),
                       crest_quick=self.w_crest_quick.isChecked(), max_conformers=self.w_max_confs.value(),
                       crest_min_rotors=self.w_crest_min_rotors.value(),
                       crest_with_rdkit=self.w_crest_with_rdkit.isChecked(),
                       refine_top=self.w_refine_top.value(), rank_method=self.w_rank.currentData(),
                       qm_method=qm_method, qm_task=self.w_qm_task.currentData(), qm_top=self.w_qm_top.value(),
                       threads=self.w_threads.value(), priority=self.w_priority.currentData(),
                       gpu=self.w_gpu.currentData(), gpu_index=self.w_gpu_index.value(),
                       qm_memory_gb=float(self.w_qm_mem.value()), memory_budget_gb=float(self.w_mem.value()),
                       molecule_timeout_s=float(self.w_timeout.value()))

    def ids(self) -> list[str]:
        if self.fixed_ids is not None:
            return list(self.fixed_ids)
        return list(self.sel_ids) if self.w_sel.isChecked() else list(self.all_ids)

    def spec(self) -> recipes.JobSpec:
        cfg = self.config()
        task = self.w_task.currentData() or recipes.DEFAULT_TASK
        if task == "frequencies" and "spectrum" not in cfg.formats:
            cfg = replace(cfg, formats=(*cfg.formats, "spectrum"))
        recipe = self.w_recipe.currentData() or recipes.CUSTOM
        name = self.w_name.text().strip()
        if not name:
            r = recipes.find(recipe, self.win.s)
            n = len(self.ids())
            name = (f"{r.name if r is not None else 'Job'}: {recipes.TASKS[task].label.lower()}, "
                    f"{n:,} molecule{'s' if n != 1 else ''}")
        return recipes.JobSpec(name=name, recipe=recipe, task=task, cfg=cfg,
                               charge=None if self.w_charge.value() == self.w_charge.minimum() else self.w_charge.value(),
                               multiplicity=self.w_mult.value() or None, run_on=self.w_run_on.currentData())

    # ================================================================== reacting
    def _on_recipe(self, _i=0) -> None:
        r = recipes.find(self.w_recipe.currentData() or "", self.win.s)
        cfg = r.config(self.config(), self.available) if r is not None else replace(
            self.config(), **{k: getattr(self.base, k) for k in recipes.RECIPE_FIELDS})
        self.load_config(cfg)
        if r is not None:
            _set(self.w_task, r.task)
        self._changed()

    def _changed(self, *_a) -> None:
        if self._loading:
            return
        self.review = None                     # the review was for other settings
        self._update()

    def recipe_modified(self) -> bool:
        r = recipes.find(self.w_recipe.currentData() or "", self.win.s)
        if r is None:
            return False
        want, _notes = r.resolved(self.available)
        cfg = self.config()
        return any(getattr(cfg, k) != v for k, v in want.items()) or (self.w_task.currentData() != r.task)

    def _update(self) -> None:
        r = recipes.find(self.w_recipe.currentData() or "", self.win.s)
        self.w_recipe_del.setEnabled(r is not None and not r.builtin)
        if r is None:
            self.recipe_info.setText("The settings of the Pipeline dock (Calculate). Change any field here: the job "
                                     "keeps its own settings, the Pipeline dock stays as it is.")
        else:
            needs = r.needs(self.programs)
            _changes, notes = r.resolved(self.available)
            text = f"{html.escape(r.description)}<br><b>Steps:</b> {' → '.join(html.escape(s) for s in r.steps)}"
            if needs:
                text += f"<br><span style='color:#d35400'>Needs {', '.join(needs)}: not found here.</span>"
            for n in notes:
                text += f"<br><span style='color:#d35400'>{html.escape(n)}</span>"
            if self.recipe_modified():
                text += "<br><i>Changed from the recipe below.</i>"
            self.recipe_info.setText(text)
        task = recipes.TASKS.get(self.w_task.currentData() or "")
        self.task_info.setText(task.description if task is not None else "")
        conformers = task is not None and not task.keeps_coordinates
        self.conf_box.setEnabled(conformers)
        search = self.w_search.currentData()
        crest = search in ("crest", "auto")
        self.w_crest_method.setEnabled(crest)
        self.w_crest_quick.setEnabled(crest)
        self.w_crest_with_rdkit.setEnabled(crest)
        self.w_crest_min_rotors.setEnabled(search == "auto")
        self.w_max_confs.setEnabled(search != "crest" or self.w_crest_with_rdkit.isChecked())
        qm = self.w_qm_on.isChecked()
        for w in (self.w_qm_func, self.w_qm_basis, self.w_qm_task, self.w_qm_top):
            w.setEnabled(qm)
        self.level_info.setText(self._level_note())
        spec = self.spec()
        problems = spec.problems()
        if not spec.cfg.qm_method and self.w_qm_on.isChecked():
            problems.append("DFT: give a functional")
        self.summary.setText(spec.summary() if not problems else spec.name)
        self.problems.setText("; ".join(problems))
        self.problems.setVisible(bool(problems))
        self.submit_btn.setEnabled(not problems and bool(self.ids()))

    def _level_note(self) -> str:
        from ..chem import engines

        key = self.w_method.currentData() or "auto"
        solvent = self.w_solvent.currentData()
        lv = next((x for x in self.levels if x.key == key), None)
        parts = []
        if lv is not None and lv.key != "auto":
            parts.append(f"{recipes.FAMILY_LABELS.get(lv.family, lv.family)}" + (", GPU-capable" if lv.gpu else ""))
        if solvent and key != "auto" and not engines.supports_solvent(key, solvent):
            parts.append(f"{lv.label if lv else key} has no implicit solvent: it runs in the gas phase (the rank step "
                         "and the DFT stage still use the solvent where they can)")
        return ". ".join(parts)

    def _on_tab(self, i: int) -> None:
        if i == REVIEW_TAB and self.review is None:
            self.start_review()

    # ================================================================== the review
    def start_review(self) -> bool:
        from ..chem import advisor
        from .molecule_list import gather_inputs

        ids = self.ids()
        if not ids:
            self.review_title.setText("No molecules: add some first.")
            return False
        spec = self.spec()
        if spec.problems():
            self.review_title.setText("Fix the job first: " + "; ".join(spec.problems()))
            return False
        rows = [self.win.model.row(i) for i in ids]
        specs = [(r.id, r.run_dir, r.input, r.smiles, r.name, r.source) for r in rows]
        cfg = spec.config()
        self._review_gen += 1
        gen = self._review_gen
        self._review_stop.set()
        stop = self._review_stop = threading.Event()
        progress = self._progress

        def job():
            inputs = recipes.job_inputs(gather_inputs(specs), spec)
            return gen, advisor.review(inputs, cfg, progress=lambda i, n: progress.changed.emit(i, n),
                                       should_stop=stop.is_set)

        self.review_title.setText(f"Reviewing {len(ids):,} molecule{'s' if len(ids) != 1 else ''} for this job...")
        self.findings.clear()
        self.finding_detail.clear()
        for b in self.preset_buttons.values():
            b.setEnabled(False)
        workers.start(workers.FunctionWorker(job), self.win, on_done=self._review_done, on_failed=self._review_failed)
        return True

    @Slot(int, int)
    def _on_review_progress(self, i: int, n: int) -> None:
        if n > 200:
            self.review_title.setText(f"Reviewing... {i:,} / {n:,}")

    def _review_done(self, value) -> None:
        gen, review = value
        if gen != self._review_gen:
            return
        try:
            self.show_review(review)
        except RuntimeError:          # the dialog was closed meanwhile
            pass

    def _review_failed(self, msg: str) -> None:
        try:
            self.review_title.setText("The review failed; the job can still be submitted. "
                                      + (msg.splitlines()[0][:200] if msg else ""))
        except RuntimeError:
            pass

    def show_review(self, r) -> None:
        from .review_panel import LEVEL_ICONS, LEVEL_NAMES

        self.review = r
        counts = r.counts()
        self.review_title.setText(f"Review of {r.n:,} molecule{'s' if r.n != 1 else ''}: "
                                  + ", ".join(f"{counts[k]} {LEVEL_NAMES[k].lower()}{'s' if counts[k] != 1 else ''}"
                                              for k in counts))
        self.findings.clear()
        style = self.style()
        for f in r.findings:
            it = QTreeWidgetItem([f"{f.topic}: {f.message}", f"{f.n:,}" if f.n else ""])
            it.setIcon(0, style.standardIcon(LEVEL_ICONS[f.level]))
            it.setToolTip(0, f"{LEVEL_NAMES[f.level]}: {f.message}\n\n{f.why}")
            it.setData(0, ID_ROLE, f.key)
            self.findings.addTopLevelItem(it)
        if self.findings.topLevelItemCount():
            self.findings.setCurrentItem(self.findings.topLevelItem(0))
        else:
            self.finding_detail.setHtml("<p>No findings.</p>")
        for p in r.presets:
            b = self.preset_buttons.get(p.key)
            if b is not None:
                b.setEnabled(True)
                sets = ", ".join(f"{k} = {v}" for k, v in p.diff.items()) if p.diff else "no change"
                b.setToolTip(f"{p.rationale}\n\nEstimated: {p.cost.text()}.\n\nSets in this job: {sets}")
        self.review_cost.setText(f"This job: {r.cost.text()}.")

    def _finding(self):
        it = self.findings.currentItem()
        key = it.data(0, ID_ROLE) if it is not None else None
        return next((f for f in self.review.findings if f.key == key), None) if self.review is not None else None

    def _on_finding(self, _cur=None, _prev=None) -> None:
        from .review_panel import finding_html

        f = self._finding()
        self.finding_detail.setHtml(finding_html(f) if f is not None else "")
        self.apply_finding_btn.setEnabled(bool(f is not None and f.changes))

    def apply_changes(self, changes: dict) -> None:
        """Set ``changes`` (BuildConfig fields, from a finding or a preset) in the dialog's fields."""
        cfg = replace(self.config(), **{k: v for k, v in changes.items() if hasattr(self.cfg, k)})
        self.load_config(cfg)
        self.review = None
        self._update()

    def apply_finding(self) -> None:
        f = self._finding()
        if f is not None and f.changes:
            self.apply_changes(f.changes)
            self.start_review()

    def apply_preset(self, key: str) -> None:
        p = self.review.preset(key) if self.review is not None else None
        if p is not None:
            self.apply_changes(p.changes)
            self.start_review()

    # ================================================================== recipes and submitting
    def save_recipe(self, name: str | None = None) -> recipes.Recipe | None:
        from .. import settings as settings_mod

        if name is None:
            r = recipes.find(self.w_recipe.currentData() or "", self.win.s)
            name, ok = QInputDialog.getText(self, "Save as recipe", "Recipe name:",
                                            text=r.name if r is not None and not r.builtin else "")
            if not ok:
                return None
        try:
            rec = recipes.save_recipe(self.win.s, name, self.config(), task=self.w_task.currentData(),
                                      description=f"Saved from the job dialog: {self.spec().summary()}")
        except ValueError as exc:
            QMessageBox.warning(self, "Save as recipe", str(exc))
            return None
        try:
            settings_mod.save(self.win.s)
        except OSError as exc:
            QMessageBox.warning(self, "Save as recipe", f"The settings could not be written: {exc}")
        self._fill_recipes()
        _set(self.w_recipe, rec.key)
        self._update()
        return rec

    def delete_recipe(self) -> bool:
        from .. import settings as settings_mod

        r = recipes.find(self.w_recipe.currentData() or "", self.win.s)
        if r is None or r.builtin:
            return False
        recipes.delete_recipe(self.win.s, r.name)
        try:
            settings_mod.save(self.win.s)
        except OSError:
            pass
        self.w_recipe.setCurrentIndex(0)
        self._fill_recipes()
        self._update()
        return True

    def submit(self):
        spec = self.spec()
        problems = spec.problems()
        if problems:
            QMessageBox.warning(self, "New job", "; ".join(problems))
            return None
        ids = self.ids()
        if not ids:
            QMessageBox.information(self, "New job", "Add a molecule first.")
            return None
        job = submit_spec(self.win, spec, ids)
        self._review_stop.set()
        self.submitted.emit(job)
        self.accept()
        return job

    def done(self, r: int) -> None:          # noqa: D401 - Qt's name
        self._review_stop.set()
        super().done(r)


def submit_spec(win, spec: recipes.JobSpec, ids: list[str]):
    """Queue ``spec`` for ``ids`` in ``win``: the molecules with the job's charge, multiplicity and starting
    coordinates, the job's configuration, a run folder made when it starts."""
    cfg = spec.config()
    todo, cfg, run_dir = win.prepare_run(ids, cfg, transform=lambda inputs: recipes.job_inputs(inputs, spec))
    return win.submit_job(todo, cfg, run_dir, ids, name=spec.name, summary=spec.summary(), recipe=spec.recipe,
                          task=spec.task, run_on=spec.run_on)
