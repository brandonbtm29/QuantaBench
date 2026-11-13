"""Dialogs of the Table dock (UI-3): the column chooser, a formula column, and the coefficient fit.

    ColumnChooser   show or hide columns (live), grouped by where they come from, with each one's formula name
    FormulaDialog   a new or edited formula column, checked as it is typed (``analysis.formula``)
    FitDialog       fit a formula's parameters to a target column (``formula.fit`` on a worker thread) and add the
                    fitted formula as a column
    JoinDialog      columns of a CSV file joined by id (DESC-3): the id column, the columns, how many rows match
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..analysis import formula as fm

GROUPS = ("Molecule", "Imported data fields", "Summary and descriptors", "Joined from files", "Formula columns")
SYNTAX = ("Numbers, column names (their formula names: Columns...), + - * / ** and parentheses; functions log "
          "(natural), log10, exp, sqrt, abs, min(a, b, ...), max(a, b, ...), pow(a, b). A missing value gives an "
          "empty cell in that row. Energies are in eV (best_energy_eV).")


def _set_status(label: QLabel, text: str, ok: bool) -> None:
    label.setText(text)
    label.setStyleSheet("" if ok else "color: #c0392b;")


def _insert_box(model, editor: QLineEdit, placeholder: str = "Insert a column...") -> QComboBox:
    """A combo box that types the chosen column's formula name into ``editor`` at the cursor."""
    box = QComboBox()
    box.addItem(placeholder, "")
    for c in model.numeric_columns():
        box.addItem(c.alias if c.alias == model.title(c.key) else f"{c.alias}   ({model.title(c.key)})", c.alias)
    box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(22)

    def pick(i: int) -> None:
        alias = box.itemData(i)
        if alias:
            editor.insert(alias)
            editor.setFocus()
        box.blockSignals(True)
        box.setCurrentIndex(0)
        box.blockSignals(False)

    box.currentIndexChanged.connect(pick)
    return box


class ColumnChooser(QDialog):
    """Columns to show, applied as they are ticked."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Table columns")
        self.resize(560, 620)
        self._filling = False
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a column (name or formula name)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.set_filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Column", "In formulas", "Kind"])
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.itemChanged.connect(self._on_changed)
        note = QLabel("A formula names a column by its formula name (second column): the column's name where that "
                      "is a plain identifier, otherwise one made from it (letters, digits and _).")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        show = QPushButton("Show listed")
        show.clicked.connect(lambda: self.set_listed(True))
        hide = QPushButton("Hide listed")
        hide.clicked.connect(lambda: self.set_listed(False))
        defaults = QPushButton("Defaults")
        defaults.setToolTip("Forget the choices: the light molecule columns, the main descriptors and every formula "
                            "column")
        defaults.clicked.connect(self.defaults)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btns = QHBoxLayout()
        for b in (show, hide, defaults):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.search)
        lay.addWidget(self.tree, 1)
        lay.addWidget(note)
        lay.addLayout(btns)
        self.fill()

    def fill(self) -> None:
        self._filling = True
        self.tree.clear()
        groups = {}
        for g in GROUPS:
            it = QTreeWidgetItem([g])
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            groups[g] = it
        for c in self.model.columns():
            it = QTreeWidgetItem([self.model.title(c.key), c.alias if c.numeric else "",
                                  "formula" if c.source == "formula" else "number" if c.numeric else "text"])
            it.setData(0, Qt.ItemDataRole.UserRole, c.key)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Checked if self.model.is_shown(c.key) else Qt.CheckState.Unchecked)
            if c.expression:
                it.setToolTip(0, f"= {c.expression}")
            groups[c.group].addChild(it)
        for g in GROUPS:
            if groups[g].childCount():
                self.tree.addTopLevelItem(groups[g])
                groups[g].setExpanded(True)
        self.tree.resizeColumnToContents(1)
        self._filling = False
        self.set_filter(self.search.text())

    def items(self):
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            for k in range(g.childCount()):
                yield g.child(k)

    def item(self, key: str):
        return next((it for it in self.items() if it.data(0, Qt.ItemDataRole.UserRole) == key), None)

    def set_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            shown = 0
            for k in range(g.childCount()):
                it = g.child(k)
                hit = not needle or needle in it.text(0).lower() or needle in it.text(1).lower()
                it.setHidden(not hit)
                shown += hit
            g.setHidden(shown == 0)

    def _on_changed(self, item, column: int) -> None:
        if self._filling or column != 0:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            self.model.set_shown(key, item.checkState(0) == Qt.CheckState.Checked)

    def set_listed(self, on: bool) -> None:
        keys = {it.data(0, Qt.ItemDataRole.UserRole): on for it in self.items() if not it.isHidden()}
        self.model.set_shown_many(keys)
        self.fill()

    def defaults(self) -> None:
        self.model.reset_choices()
        self.fill()


class FormulaDialog(QDialog):
    """A new formula column, or an existing one (``name``) to edit."""

    def __init__(self, model, name: str | None = None, parent=None):
        super().__init__(parent)
        self.model = model
        self.old_name = name
        spec = next((f for f in model.formulas() if f.name == name), None) if name else None
        self.setWindowTitle("Edit formula column" if spec else "New formula column")
        self.resize(640, 260)
        self.name = QLineEdit(spec.name if spec else self._free_name())
        self.name.setObjectName("formula_name")
        self.expr = QLineEdit(spec.expression if spec else "")
        self.expr.setObjectName("formula_expression")
        self.expr.setPlaceholderText("e.g. polarizability_bosque_A3 / molar_volume_packing_cm3")
        self.insert = _insert_box(model, self.expr)
        self.status = QLabel()
        self.status.setWordWrap(True)
        help_ = QLabel(SYNTAX)
        help_.setWordWrap(True)
        help_.setStyleSheet("color: gray;")
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form = QFormLayout()
        form.addRow("Name", self.name)
        row = QHBoxLayout()
        row.addWidget(self.expr, 1)
        row.addWidget(self.insert)
        form.addRow("Formula", row)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.status)
        lay.addWidget(help_)
        lay.addStretch(1)
        lay.addWidget(self.buttons)
        self.name.textChanged.connect(self.check)
        self.expr.textChanged.connect(self.check)
        self.check()

    def _free_name(self) -> str:
        taken = set(self.model.aliases())
        k = 1
        while f"f{k}" in taken:
            k += 1
        return f"f{k}"

    def check(self, *_args) -> bool:
        ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if not self.expr.text().strip():
            _set_status(self.status, "Type a formula.", True)
            ok_btn.setEnabled(False)
            return False
        try:
            f = self.model.check_formula(self.name.text(), self.expr.text(), old_name=self.old_name)
            values = f.evaluate(self.model.formula_env(), n=self.model.total_rows())
        except fm.FormulaError as exc:
            _set_status(self.status, str(exc), False)
            ok_btn.setEnabled(False)
            return False
        n = len(values)
        have = int(np.isfinite(values).sum())
        _set_status(self.status, f"OK: a value for {have:,} of {n:,} molecule{'s' if n != 1 else ''}.", True)
        ok_btn.setEnabled(True)
        return True

    def accept(self) -> None:
        if not self.check():
            return
        try:
            if self.old_name:
                self.model.update_formula(self.old_name, self.name.text(), self.expr.text())
            else:
                self.model.add_formula(self.name.text(), self.expr.text())
        except (fm.FormulaError, KeyError) as exc:
            _set_status(self.status, str(exc), False)
            return
        super().accept()


class FitDialog(QDialog):
    """Fit the parameters of a formula (its names that are not columns) to a target column, on the rows the table
    shows where every value is present (``formula.fit``)."""

    EXAMPLE = "((a*polarizability_bosque_A3 + b*dipole_debye**2 + c)/molar_volume_packing_cm3)**4"

    def __init__(self, model, panel=None, parent=None):
        super().__init__(parent or panel)
        self.model = model
        self.panel = panel
        self.result: fm.FitResult | None = None
        self.running = False
        self.setWindowTitle("Fit a formula")
        self.resize(720, 560)
        self.expr = QLineEdit()
        self.expr.setObjectName("fit_formula")
        self.expr.setPlaceholderText(f"e.g. {self.EXAMPLE}")
        self.insert = _insert_box(model, self.expr)
        self.target = QComboBox()
        self.target.setObjectName("fit_target")
        for c in model.numeric_columns():
            self.target.addItem(c.alias if c.alias == model.title(c.key) else f"{c.alias}   ({model.title(c.key)})",
                                c.key)
        self.target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target.setMinimumContentsLength(28)
        self.params = QTableWidget(0, 4)
        self.params.setHorizontalHeaderLabels(["Parameter", "Start", "Fitted", "± (std. error)"])
        self.params.verticalHeader().setVisible(False)
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.params.setMaximumHeight(150)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.clicked.connect(lambda: self.run_fit())
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Parameters, n, MAE, RMSE and R² appear here.")
        self.col_name = QLineEdit("fit1")
        self.col_name.setObjectName("fit_column_name")
        self.add_btn = QPushButton("Add as column")
        self.add_btn.setToolTip("A formula column with the fitted values written in")
        self.add_btn.setEnabled(False)
        self.add_btn.clicked.connect(lambda: self.add_column())
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        help_ = QLabel("Parameters are the names in the formula that are not columns (a, b, c ...). The fit uses the "
                       "rows the table shows (its filter applies) where the target and every column used have a "
                       "value: scipy.optimize.least_squares from the start values. " + SYNTAX)
        help_.setWordWrap(True)
        help_.setStyleSheet("color: gray;")
        form = QFormLayout()
        row = QHBoxLayout()
        row.addWidget(self.expr, 1)
        row.addWidget(self.insert)
        form.addRow("Formula", row)
        form.addRow("Target column", self.target)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(help_)
        lay.addWidget(self.params)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.status, 1)
        fit_row.addWidget(self.fit_btn)
        lay.addLayout(fit_row)
        lay.addWidget(self.report, 1)
        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Column name"))
        add_row.addWidget(self.col_name)
        add_row.addWidget(self.add_btn)
        add_row.addStretch(1)
        add_row.addWidget(close)
        lay.addLayout(add_row)
        self.expr.textChanged.connect(self._on_formula)
        self.target.currentIndexChanged.connect(self._on_target)
        self._on_target()
        self._on_formula()

    # -- parameters
    def _starts(self) -> dict[str, float]:
        out = {}
        for r in range(self.params.rowCount()):
            name = self.params.item(r, 0).text()
            try:
                out[name] = float(self.params.item(r, 1).text())
            except (TypeError, ValueError, AttributeError):
                out[name] = 1.0
        return out

    def _on_target(self, *_args) -> None:
        key = self.target.currentData()
        col = self.model.column(key) if key else None
        if col is not None:
            base = f"fit_{col.alias}"[:40]
            taken = set(self.model.aliases())
            name, k = base, 2
            while name in taken:
                name, k = f"{base}_{k}", k + 1
            self.col_name.setText(name)

    def _on_formula(self, *_args) -> None:
        self.result = None
        self.add_btn.setEnabled(False)
        starts = self._starts()
        text = self.expr.text().strip()
        params: list[str] = []
        if text:
            try:
                params = fm.check(text, self.model.aliases(), allow_params=True).params(self.model.aliases())
            except fm.FormulaError as exc:
                _set_status(self.status, str(exc), False)
                self.fit_btn.setEnabled(False)
                self._fill_params([], starts)
                return
        self._fill_params(params, starts)
        self.fit_btn.setEnabled(bool(params) and not self.running)
        _set_status(self.status, f"Parameters: {', '.join(params)}" if params else "Type a formula with parameters.",
                    True)

    def _fill_params(self, params: list[str], starts: dict[str, float]) -> None:
        self.params.setRowCount(len(params))
        for r, p in enumerate(params):
            name = QTableWidgetItem(p)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.params.setItem(r, 0, name)
            self.params.setItem(r, 1, QTableWidgetItem(f"{starts.get(p, 1.0):g}"))
            for c in (2, 3):
                it = QTableWidgetItem("")
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.params.setItem(r, c, it)

    # -- fit
    def set_start(self, name: str, value: float) -> None:
        for r in range(self.params.rowCount()):
            if self.params.item(r, 0).text() == name:
                self.params.item(r, 1).setText(f"{value:g}")

    def run_fit(self, background: bool = True) -> None:
        from . import workers

        text = self.expr.text().strip()
        key = self.target.currentData()
        if not text or not key or self.running:
            return
        try:
            f = fm.check(text, self.model.aliases(), allow_params=True)
            view = self.model.view_positions()
            env = self.model.formula_env()
            cols = {a: np.array(env[a], dtype=float)[view] for a in f.columns(self.model.aliases())}
        except fm.FormulaError as exc:
            _set_status(self.status, str(exc), False)
            return
        target = np.array(self.model.values(key), dtype=float)[view]
        args = (f, cols, target)
        kwargs = dict(start=self._starts(), target_name=self.model.column(key).alias)
        if not background:
            try:
                self._done(fm.fit(*args, **kwargs))
            except (fm.FitError, fm.FormulaError) as exc:
                self._failed(str(exc))
            return
        self.running = True
        self.fit_btn.setEnabled(False)
        _set_status(self.status, "Fitting...", True)
        owner = getattr(self.panel, "win", None) or self.parent() or self
        workers.start(workers.FunctionWorker(fm.fit, *args, **kwargs), owner, on_done=self._done,
                      on_failed=self._failed)

    def _done(self, res: fm.FitResult) -> None:
        self.running = False
        self.fit_btn.setEnabled(True)
        self.result = res
        for r in range(self.params.rowCount()):
            p = self.params.item(r, 0).text()
            if p in res.params:
                self.params.item(r, 2).setText(f"{res.params[p]:.6g}")
                se = res.stderr.get(p)
                self.params.item(r, 3).setText(f"{se:.2g}" if se is not None and np.isfinite(se) else "")
        self.report.setPlainText(res.report() + f"\n\nFitted formula:\n{res.fitted}")
        _set_status(self.status, f"n = {res.n}, R² = {res.r2:.4f}, RMSE = {res.rmse:.4g}"
                    + ("" if res.success else " (not converged)"), res.success)
        self.add_btn.setEnabled(True)

    def _failed(self, msg: str) -> None:
        self.running = False
        self.fit_btn.setEnabled(True)
        self.result = None
        _set_status(self.status, msg, False)

    def add_column(self):
        res = self.result
        if res is None:
            return None
        note = f"Fit to {res.target}: n {res.n}, R² {res.r2:.4f}, RMSE {res.rmse:.4g}, MAE {res.mae:.4g} ({res.formula})"
        try:
            col = self.model.add_formula(self.col_name.text(), res.fitted, note=note)
        except fm.FormulaError as exc:
            _set_status(self.status, str(exc), False)
            return None
        _set_status(self.status, f"Added the column {col.alias}.", True)
        self.add_btn.setEnabled(False)
        return col


class JoinDialog(QDialog):
    """Columns of a CSV file joined to the table by id (``TablePanel.join_csv``): pick the column that holds the
    molecule ids and the columns to bring in; the count of matching rows updates as the id column changes."""

    def __init__(self, panel, path: str, parent=None):
        from pathlib import Path

        from ..analysis import sheet

        super().__init__(parent or panel)
        self.panel, self.path = panel, str(Path(path).expanduser())
        self.setWindowTitle("Join columns from a file")
        self.setMinimumWidth(460)
        self.joined = None
        try:
            names = sheet.csv_columns(self.path)
        except (OSError, UnicodeDecodeError) as exc:
            names = []
            self._error = f"Cannot read {self.path}: {exc}"
        else:
            self._error = "" if names else f"{self.path} has no header row"
        self.id_box = QComboBox()
        self.id_box.setObjectName("join_id")
        for n in names:
            self.id_box.addItem(n, n)
        low = [n.lower() for n in names]
        guess = next((low.index(w) for w in ("id", "cas", "cas_rn", "casrn", "identifier", "name") if w in low), 0)
        self.id_box.setCurrentIndex(guess)
        self.columns = QListWidget()
        self.columns.setObjectName("join_columns")
        for n in names:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.columns.addItem(it)
        self.all_box = QCheckBox("All columns")
        self.all_box.setChecked(True)
        self.all_box.toggled.connect(self._all)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form = QFormLayout()
        form.addRow("File", QLabel(Path(self.path).name))
        form.addRow("Molecule ids in", self.id_box)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("Columns to join (matched on the molecule id, ignoring case and spaces):"))
        lay.addWidget(self.columns, 1)
        lay.addWidget(self.all_box)
        lay.addWidget(self.status)
        lay.addWidget(self.buttons)
        self.id_box.currentIndexChanged.connect(self._mark_id)
        self.columns.itemChanged.connect(lambda _it: self._check())
        self._mark_id()

    def _mark_id(self, *_args) -> None:
        """The id column is not one to join: its item is greyed out."""
        ident = self.id_box.currentData()
        self.columns.blockSignals(True)
        for i in range(self.columns.count()):
            it = self.columns.item(i)
            name = it.data(Qt.ItemDataRole.UserRole) or it.text()
            it.setData(Qt.ItemDataRole.UserRole, name)
            on = name != ident
            it.setFlags((it.flags() | Qt.ItemFlag.ItemIsEnabled) if on else (it.flags() & ~Qt.ItemFlag.ItemIsEnabled))
            it.setText(name if on else f"{name}   (the molecule ids)")
        self.columns.blockSignals(False)
        self._check()

    def _all(self, on: bool) -> None:
        self.columns.blockSignals(True)
        for i in range(self.columns.count()):
            self.columns.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.columns.blockSignals(False)
        self._check()

    def chosen(self) -> list[str]:
        ident = self.id_box.currentData()
        out = []
        for i in range(self.columns.count()):
            it = self.columns.item(i)
            name = it.data(Qt.ItemDataRole.UserRole) or it.text()
            if it.checkState() == Qt.CheckState.Checked and name != ident:
                out.append(name)
        return out

    def _check(self) -> None:
        from ..analysis import sheet

        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if self._error:
            _set_status(self.status, self._error, False)
            ok.setEnabled(False)
            return
        cols, ident = self.chosen(), self.id_box.currentData()
        if not cols:
            _set_status(self.status, "Choose at least one column besides the id column.", False)
            ok.setEnabled(False)
            return
        try:
            got = sheet.read_join(sheet.JoinSpec(self.path, ident, cols))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            _set_status(self.status, str(exc), False)
            ok.setEnabled(False)
            return
        have = set(got.block.ids)
        rows = [r for r in self.panel.model.molecules.rows()]
        n = sum(1 for r in rows if sheet.norm_id(r.id) in have)
        extra = []
        if got.duplicates:
            extra.append(f"{got.duplicates} repeated id(s): the last row wins")
        if got.no_id:
            extra.append(f"{got.no_id} row(s) without an id")
        _set_status(self.status, f"{n:,} of the table's {len(rows):,} molecules match ({got.rows:,} rows in the "
                    "file)" + (f"; {'; '.join(extra)}" if extra else "") + ".", n > 0)
        ok.setEnabled(True)

    def accept(self) -> None:
        try:
            self.panel.join_csv(self.path, self.id_box.currentData(), self.chosen())
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            _set_status(self.status, str(exc), False)
            return
        self.joined = self.chosen()
        super().accept()
