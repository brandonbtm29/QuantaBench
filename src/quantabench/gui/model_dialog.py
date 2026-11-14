"""The Table dock's Model... dialog (QSAR-4): a property model on the table's columns, cross-validated, its
predictions joined back into the table (``analysis.models`` does the fitting).

    target       a numeric column: a measured property, joined from a CSV file (Join...) or an imported field
    model        ridge regression (NumPy), or elastic net, random forest, gradient boosting (scikit-learn, the qsar
                 extra; greyed out without it)
    features     numeric columns, ticked in a filterable list; "Liquid groups" ticks the dimensionless groups of
                 DESC-3 (Lorentz-Lorenz, dipole orientation ratio, packing fraction, sphericity), "α, μ, V" the raw
                 descriptors
    σ*           model target / (kT/v_m^⅔) and multiply the predictions back (the column sigma_scale_kT_mN_m, or the
                 formula column sigma_scale); the reduced surface tension is nearly constant across a homologous
                 series, so a model of σ* extrapolates where a model of σ does not
    validation   k-fold (the seed fixes the folds), leave one group out by a column (a chemical class, a family),
                 and a holdout column: rows where it is non-zero (or "true", "test") never train and are scored
                 separately

Rows: the rows the table shows (its filter applies). Fit runs on a worker thread and shows the metrics and the
coefficients (or importances); Add to table writes ``<name>.csv`` (``id``, ``<name>_pred`` by the model fitted on all
training rows, ``<name>_oof`` the out-of-fold prediction) and ``<name>.json`` into the project's models/ folder (with
no project open: models/ in the output folder) and joins the CSV, so the columns persist with the table and a
project; Plot 1:1 plots the target against the out-of-fold prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..analysis import formula as fm
from ..analysis import models as mdl

NONE = ""
#: the scale of the reduced target: the descriptor first, then the Features menu's formula column
SCALE_ALIASES = ("sigma_scale_kT_mN_m", "sigma_scale", "gas_sigma_scale_kT_mN_m")
#: preset -> one entry per feature: the aliases that can stand for it, the first the table has wins
FEATURE_PRESETS: dict[str, tuple[tuple[str, ...], ...]] = {
    "Liquid groups": (("lorentz_lorenz_bosque", "LL_bosque", "gas_lorentz_lorenz_bosque"),
                      ("dipole_orientation_ratio", "ORI", "gas_dipole_orientation_ratio"),
                      ("packing_fraction_vdw", "eta_vdw", "gas_packing_fraction_vdw"),
                      ("sphericity_vdw", "psi_vdw", "gas_sphericity_vdw")),
    "α, μ, V": (("polarizability_bosque_A3", "gas_polarizability_bosque_A3"),
                ("dipole_debye_boltz", "dipole_debye_best", "gas_dipole_debye_boltz", "gas_dipole_debye_best"),
                ("vdw_volume_A3_best", "gas_vdw_volume_A3_best"),
                ("molar_volume_packing_cm3", "gas_molar_volume_packing_cm3")),
}
#: columns that are never features (bookkeeping, not chemistry)
NOT_FEATURES = frozenset(("seq",))
TARGET_WORDS = ("sigma", "surface_tension", "st25", "gamma", "measured", "exp")


def preset_aliases(name: str, aliases) -> list[str]:
    have = set(aliases)
    out = []
    for choices in FEATURE_PRESETS[name]:
        got = next((a for a in choices if a in have), None)
        if got:
            out.append(got)
    return out


class ModelDialog(QDialog):
    """Fit a model of a target column on feature columns (module docstring)."""

    def __init__(self, panel, parent=None):
        super().__init__(parent or panel)
        self.panel = panel
        self.model = panel.model
        self.result: mdl.ModelResult | None = None
        self.ids: list[str] = []
        self.written: tuple[Path, Path] | None = None
        self.running = False
        self.setWindowTitle("Model a property")
        self.setObjectName("model_dialog")
        self.resize(820, 680)

        self.name = QLineEdit(self._free_name("model1"))
        self.name.setObjectName("model_name")
        self.name.setToolTip("Names the files (<name>.csv, <name>.json) and the columns <name>_pred, <name>_oof")
        self.target = QComboBox()
        self.target.setObjectName("model_target")
        self.target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target.setMinimumContentsLength(28)
        self.kind = QComboBox()
        self.kind.setObjectName("model_kind")
        have_sk = mdl.sklearn_available()
        for k, (label, needs_sk) in mdl.KINDS.items():
            self.kind.addItem(label if have_sk or not needs_sk else f"{label}: pip install scikit-learn", k)
            if needs_sk and not have_sk:
                item = self.kind.model().item(self.kind.count() - 1)
                item.setEnabled(False)
                item.setToolTip(mdl.SKLEARN_MISSING)
        self.reduced = QCheckBox("Model σ* = target / (kT/v_m^⅔) and multiply back")
        self.reduced.setObjectName("model_reduced")

        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter the features")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._filter)
        self.features = QListWidget()
        self.features.setObjectName("model_features")
        self.features.itemChanged.connect(self._on_checked)
        self.picked = QLabel()
        self.picked.setStyleSheet("color: gray;")
        preset_row = QHBoxLayout()
        preset_row.addWidget(self.filter, 1)
        self.preset_btns = {}
        for name in FEATURE_PRESETS:
            b = QPushButton(name)
            b.setToolTip("Tick " + ", ".join(c[0] for c in FEATURE_PRESETS[name]) + " (or the formula columns of the "
                         "Features menu that stand for them)")
            b.clicked.connect(lambda _c=False, n=name: self.tick_preset(n))
            preset_row.addWidget(b)
            self.preset_btns[name] = b
        clear = QPushButton("None")
        clear.clicked.connect(lambda: self.set_features([]))
        preset_row.addWidget(clear)

        self.folds = QSpinBox()
        self.folds.setRange(2, 20)
        self.folds.setValue(5)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(20261113)
        self.seed.setToolTip("Fixes the folds (and the trees' randomness): the same seed gives the same numbers")
        self.groups = QComboBox()
        self.groups.setObjectName("model_groups")
        self.groups.setToolTip("Leave one group out: each value of this column (a class, a family) is predicted by a "
                               "model that never saw it. Tells how far the model extrapolates.")
        self.holdout = QComboBox()
        self.holdout.setObjectName("model_holdout")
        self.holdout.setToolTip("Rows where this column is non-zero (or reads true, yes, test) never train: they are "
                                "predicted and scored separately")
        val = QHBoxLayout()
        val.addWidget(QLabel("Folds"))
        val.addWidget(self.folds)
        val.addWidget(QLabel("Seed"))
        val.addWidget(self.seed)
        val.addStretch(1)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("model_fit")
        self.fit_btn.clicked.connect(lambda: self.run())
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Cross-validated MAE, RMSE and R², and the coefficients, appear here.")
        self.add_btn = QPushButton("Add to table")
        self.add_btn.setObjectName("model_add")
        self.add_btn.setToolTip("Write <name>.csv and <name>.json to the models folder and join <name>_pred and "
                                "<name>_oof into the table")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(lambda: self.add_to_table())
        self.plot_btn = QPushButton("Plot 1:1")
        self.plot_btn.setToolTip("The target against the out-of-fold prediction, with the 1:1 line")
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(lambda: self.plot())
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        help_ = QLabel("Uses the rows the table shows (its filter applies). Missing feature values are filled with "
                       "the training rows' medians inside each fold; features are standardized, so the "
                       "coefficients compare. Ridge regression needs only NumPy; the other models need scikit-learn.")
        help_.setWordWrap(True)
        help_.setStyleSheet("color: gray;")

        form = QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("Target column", self.target)
        form.addRow("", self.reduced)
        form.addRow("Model", self.kind)
        form.addRow("Leave group out by", self.groups)
        form.addRow("Holdout column", self.holdout)
        form.addRow("", val)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("Features"))
        lay.addLayout(preset_row)
        lay.addWidget(self.features, 2)
        lay.addWidget(self.picked)
        lay.addWidget(help_)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.status, 1)
        fit_row.addWidget(self.fit_btn)
        lay.addLayout(fit_row)
        lay.addWidget(self.report, 2)
        bottom = QHBoxLayout()
        bottom.addWidget(self.add_btn)
        bottom.addWidget(self.plot_btn)
        bottom.addStretch(1)
        bottom.addWidget(close)
        lay.addLayout(bottom)

        self.target.currentIndexChanged.connect(self._on_target)
        self.fill()

    # ------------------------------------------------------------------ columns
    def fill(self) -> None:
        """(Re)list the table's columns, keeping the choices that still exist."""
        m = self.model
        keep_target = self.target.currentData()
        keep_feats = set(self.chosen_features())
        keep_groups, keep_hold = self.groups.currentData(), self.holdout.currentData()
        numeric = [c for c in m.numeric_columns() if c.alias not in NOT_FEATURES]
        self.target.blockSignals(True)
        self.target.clear()
        for c in numeric:
            self.target.addItem(c.alias if c.alias == m.title(c.key) else f"{c.alias}   ({m.title(c.key)})", c.key)
        i = self.target.findData(keep_target) if keep_target else -1
        if i < 0:
            guess = [n for n, c in enumerate(numeric) if c.source in ("joined", "field")
                     and any(w in c.alias.lower() for w in TARGET_WORDS)]
            guess = guess or [n for n, c in enumerate(numeric) if c.source in ("joined", "field")]
            i = guess[0] if guess else 0
        self.target.setCurrentIndex(i)
        self.target.blockSignals(False)
        for box, keep in ((self.groups, keep_groups), (self.holdout, keep_hold)):
            box.clear()
            box.addItem("(none)", NONE)
            for c in m.columns():
                if c.alias not in NOT_FEATURES:
                    box.addItem(c.alias, c.key)
            j = box.findData(keep) if keep else 0
            box.setCurrentIndex(max(j, 0))
        self.features.blockSignals(True)
        self.features.clear()
        for c in numeric:
            it = QListWidgetItem(c.alias if c.alias == m.title(c.key) else f"{c.alias}   ({m.title(c.key)})")
            it.setData(Qt.ItemDataRole.UserRole, c.alias)
            it.setToolTip(f"{c.group}" + (f": {c.note}" if c.note else "") + (f" = {c.expression}" if c.expression
                                                                              else ""))
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if c.alias in keep_feats else Qt.CheckState.Unchecked)
            self.features.addItem(it)
        self.features.blockSignals(False)
        for name, b in self.preset_btns.items():
            b.setEnabled(bool(preset_aliases(name, m.aliases())))
        self._on_target()
        self._filter(self.filter.text())

    def _free_name(self, base: str) -> str:
        taken = set(self.model.aliases())
        stem = base.rstrip("0123456789") or "model"
        k = 1
        name = base
        while f"{name}_pred" in taken or f"{name}_oof" in taken:
            k += 1
            name = f"{stem}{k}"
        return name

    def _items(self):
        return [self.features.item(i) for i in range(self.features.count())]

    def chosen_features(self) -> list[str]:
        return [it.data(Qt.ItemDataRole.UserRole) for it in self._items() if it.checkState() == Qt.CheckState.Checked]

    def set_features(self, aliases) -> None:
        want = set(aliases)
        self.features.blockSignals(True)
        for it in self._items():
            it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in want
                             else Qt.CheckState.Unchecked)
        self.features.blockSignals(False)
        self._on_checked()

    def tick_preset(self, name: str) -> list[str]:
        """Tick the preset's features (added to those ticked); the aliases ticked."""
        got = preset_aliases(name, [it.data(Qt.ItemDataRole.UserRole) for it in self._items()])
        self.set_features(set(self.chosen_features()) | set(got))
        return got

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()
        for it in self._items():
            it.setHidden(bool(needle) and needle not in it.text().lower())

    def _on_checked(self, *_args) -> None:
        n = len(self.chosen_features())
        self.picked.setText(f"{n} feature(s) ticked")
        self._invalidate()

    def scale_alias(self) -> str:
        have = set(self.model.aliases())
        return next((a for a in SCALE_ALIASES if a in have), "")

    def _on_target(self, *_args) -> None:
        scale = self.scale_alias()
        self.reduced.setEnabled(bool(scale))
        self.reduced.setToolTip(f"Divide the target by {scale} (kT/v_m^⅔ per molecule, mN/m), model that, multiply "
                                "the predictions back: for a surface tension" if scale else
                                "Needs the column sigma_scale_kT_mN_m (descriptors of a run built after DESC-3) or "
                                "the Features menu's sigma_scale")
        if not scale:
            self.reduced.setChecked(False)
        key = self.target.currentData()
        col = self.model.column(key) if key else None
        if col is not None and not self.reduced.isChecked() and scale:
            self.reduced.setChecked(any(w in col.alias.lower() for w in ("sigma", "surface_tension", "st25",
                                                                         "gamma")) and "star" not in col.alias)
        self._invalidate()

    def _invalidate(self) -> None:
        if self.result is not None and not self.running:
            self.add_btn.setEnabled(False)
            self.status.setText("The choices changed: Fit again.")
        self.result = None

    # ------------------------------------------------------------------ fit
    def spec(self) -> mdl.ModelSpec:
        name = self.name.text().strip()
        if not fm.is_name(name) or not fm.is_name(f"{name}_pred"):
            raise mdl.ModelError(f"{name!r} is not a usable name: letters, digits and _, starting with a letter")
        key = self.target.currentData()
        col = self.model.column(key) if key else None
        if col is None:
            raise mdl.ModelError("choose a target column")
        g = self.model.column(self.groups.currentData()) if self.groups.currentData() else None
        h = self.model.column(self.holdout.currentData()) if self.holdout.currentData() else None
        return mdl.ModelSpec(name=name, kind=self.kind.currentData(), features=self.chosen_features(),
                             target=col.alias, reduced=self.reduced.isChecked(),
                             scale=self.scale_alias() if self.reduced.isChecked() else "", folds=self.folds.value(),
                             seed=self.seed.value(), groups=g.alias if g else "", holdout=h.alias if h else "")

    def inputs(self, spec: mdl.ModelSpec) -> dict:
        """The arrays ``analysis.models.run`` takes, for the rows the table shows."""
        m = self.model
        view = m.view_positions()

        def col(alias: str, numeric: bool = True):
            c = m.column_by_alias(alias)
            v = m.values(c.key)[view]
            return np.asarray(v, dtype=float) if numeric and c.numeric else v

        X = np.column_stack([col(a) for a in spec.features]) if spec.features else np.empty((len(view), 0))
        out = dict(X=X, y=col(spec.target), scale=col(spec.scale) if spec.reduced else None,
                   groups=col(spec.groups, numeric=False) if spec.groups else None,
                   holdout=col(spec.holdout) if spec.holdout else None)
        self.ids = [str(v) for v in m.values("id")[view]]
        return out

    def run(self, background: bool = True) -> None:
        from . import workers

        if self.running:
            return
        try:
            spec = self.spec()
            if spec.kind != mdl.RIDGE and not mdl.sklearn_available():
                raise mdl.ModelError(f"{mdl.KINDS[spec.kind][0]} {mdl.SKLEARN_MISSING}")
            kw = self.inputs(spec)
        except mdl.ModelError as exc:
            self._failed(str(exc))
            return
        self.written = None
        self.plot_btn.setEnabled(False)
        if not background:
            try:
                self._done(mdl.run(spec, **kw))
            except mdl.ModelError as exc:
                self._failed(str(exc))
            return
        self.running = True
        self.fit_btn.setEnabled(False)
        self.status.setStyleSheet("")
        self.status.setText(f"Fitting {mdl.KINDS[spec.kind][0].split(' (')[0].lower()} on {len(spec.features)} "
                            "feature(s)...")
        owner = getattr(self.panel, "win", None) or self
        workers.start(workers.FunctionWorker(mdl.run, spec, **kw), owner, on_done=self._done, on_failed=self._failed)

    def _done(self, res: mdl.ModelResult) -> None:
        self.running = False
        self.fit_btn.setEnabled(True)
        self.result = res
        self.report.setPlainText(res.report())
        cv = res.cv
        self.status.setStyleSheet("")
        self.status.setText(f"{res.spec.folds}-fold CV: MAE {cv['mae']:.4g}, RMSE {cv['rmse']:.4g}, R² {cv['r2']:.4f} "
                            f"(n {cv['n']:,})")
        self.add_btn.setEnabled(True)

    def _failed(self, msg: str) -> None:
        self.running = False
        self.fit_btn.setEnabled(True)
        self.result = None
        self.add_btn.setEnabled(False)
        self.status.setStyleSheet("color: #c0392b;")
        self.status.setText(msg.split(": ", 1)[1] if msg.startswith("ModelError: ") else msg)

    # ------------------------------------------------------------------ out
    def folder(self) -> Path:
        """models/ of the open project, else of the output folder."""
        from .. import project as proj

        win = getattr(self.panel, "win", None)
        ctl = getattr(win, "project", None)
        if ctl is not None and getattr(ctl, "path", None) is not None:
            return proj.models_dir(ctl.path)
        return Path(win.s.output_path()) / proj.MODELS_DIR

    def add_to_table(self) -> tuple[Path, Path] | None:
        res = self.result
        if res is None:
            return None
        try:
            paths = mdl.write(res, self.ids, self.folder())
        except OSError as exc:
            self._failed(f"cannot write the model: {exc}")
            return None
        name = res.spec.name
        self.panel.join_csv(paths[0], "id", [f"{name}_pred", f"{name}_oof"], label=name)
        self.written = paths
        self.add_btn.setEnabled(False)
        self.plot_btn.setEnabled(True)
        self.status.setText(f"Added {name}_pred and {name}_oof; files in {paths[0].parent}")
        self.name.setText(self._free_name(name))
        return paths

    def plot(self) -> bool:
        res, win = self.result, getattr(self.panel, "win", None)
        plot = getattr(win, "plot_panel", None)
        if res is None or plot is None:
            return False
        name = res.spec.name
        x = self.model.column_by_alias(res.spec.target)
        y = next((c for c in self.model.columns() if c.key == f"join:{name}::{name}_oof"), None)
        if x is None or y is None:
            return False
        plot.set_axes(x.key, y.key, parity=True)
        dock = getattr(win, "plot_dock", None)
        if dock is not None:
            dock.show()
            dock.raise_()
        return True
