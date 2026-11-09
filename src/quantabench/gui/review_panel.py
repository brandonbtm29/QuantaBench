"""The Review window (ADV-1): the model advisor's review of the molecules, before a run and on demand.

CONTRACT:

    install(win)
        Adds a "Review..." button next to the pipeline's Run buttons, Build > Review molecules... (F6) and
        Build > Review before each run (checkable, on by default, remembered in the window state), and sets
        ``win.review_before_run``: ``MainWindow.run_build`` calls it first, and when it returns True the run waits
        for the Review window ("Run anyway" / "Run" starts it, "Cancel run" drops it).
    show_review(win, selected_only=False, before_run=False) -> ReviewDialog
        Opens (or raises) ``win``'s Review window and starts a review of the list (or the selected molecules) with
        the pipeline's current settings.

The window (``ReviewDialog``, non-modal, so the Molecules list and the Methods guide stay usable):

- **Batch**: the profile (``chem.advisor.Profile``) as a tree: each section with its counts; a count opens to its
  molecule ids (the first ``advisor.MAX_IDS``), and double-clicking an id selects that molecule in the Molecules
  dock. Then the methods that can take the batch, and where each step runs (gas or the solvent model).
- **Findings**, most severe first, with an icon per level: "Apply" sets the finding's ``changes`` in the pipeline
  pages and reviews again; "Explain" opens the Methods guide at the finding's first entry; the line below holds the
  reason and the measured result behind it, with links to every guide entry and molecule.
- **Presets**: Fast screen, Recommended, Careful, each a button that applies the preset, with its rationale and
  estimated time as the tooltip and below the buttons.
- Buttons: "Run" (no problems or warnings) or "Run anyway", "Cancel run" and "Close".

The review runs on a worker thread owned by the main window (``workers.start``; 14,000 molecules take seconds the
first time, a fraction of a second after an Apply: ``advisor`` caches each molecule's profile). Its progress comes
back through a signal of a GUI-thread object. The window never blocks a run: a failed review says so and the run
can go ahead.
"""

from __future__ import annotations

import html
import threading

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..chem import advisor
from ..chem import method_guide as guide
from . import workers

ID_ROLE = Qt.ItemDataRole.UserRole
SETTING_KEY = "review/before_run"
LEVEL_ICONS = {"problem": QStyle.StandardPixmap.SP_MessageBoxCritical,
               "warning": QStyle.StandardPixmap.SP_MessageBoxWarning,
               "info": QStyle.StandardPixmap.SP_MessageBoxInformation}
LEVEL_NAMES = {"problem": "Problem", "warning": "Warning", "info": "Note"}


# ============================================================================================ applying settings
def _set_combo(combo, value) -> bool:
    i = combo.findData(value)
    if i < 0:
        return False
    combo.setCurrentIndex(i)
    return True


def apply_changes(win, changes: dict) -> list[str]:
    """Set ``changes`` (BuildConfig fields) in the main window's pipeline pages; returns what could not be set."""
    skipped = []
    for key, value in changes.items():
        ok = True
        if key == "method":
            ok = _set_combo(win.w_method, value)
        elif key == "solvent":
            ok = _set_combo(win.w_solvent, value or "")
        elif key == "conformer_search":
            ok = _set_combo(win.w_confsearch, value)
        elif key == "crest_method":
            ok = _set_combo(win.w_crest_method, value)
        elif key == "crest_quick":
            win.w_crest_quick.setChecked(bool(value))
        elif key == "max_conformers":
            win.w_maxconf.setValue(int(value))
        elif key == "refine_top":
            win.w_refine_top.setValue(int(value))
        elif key == "energy_window_kcal":
            win.w_window.set_stored(float(value))
        elif key == "qm_method":
            win.w_qm_on.setChecked(bool(value))
            if value:
                func, _sep, basis = str(value).partition("/")
                win.w_qm_func.setCurrentText(func)
                win.w_qm_basis.setCurrentText(basis or "def2-svp")
        elif key == "qm_top":
            win.w_qm_top.setValue(int(value))
        elif key == "qm_task":
            ok = _set_combo(win.w_qm_task, value)
        elif key == "keep_largest_fragment":
            win.w_desalt.setChecked(bool(value))
        elif key == "protonate":
            win.w_protonate.setChecked(bool(value))
        elif key == "ph":
            win.w_ph.setValue(float(value))
        elif key == "frequencies":
            win.w_freq.setChecked(bool(value))
        elif key == "xtb_properties":
            win.w_xtbprops.setChecked(bool(value))
        else:
            ok = False
        if not ok:
            skipped.append(key)
    if hasattr(win, "_update_crest_widgets"):
        win._update_crest_widgets()
        win._update_qm_widgets()
        win._update_engine_info()
    return skipped


def describe_changes(changes: dict) -> str:
    return ", ".join(f"{k} = {v!r}" for k, v in changes.items())


# ============================================================================================ the window
class _Progress(QObject):
    """Lives in the GUI thread; the worker emits through it (a queued connection to the dialog's slot)."""
    changed = Signal(int, int)


class ReviewDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Review before the run")
        self.setObjectName("review")
        self.setModal(False)
        self.resize(1100, 760)
        self.review: advisor.Review | None = None
        self.selected_only = False
        self.before_run = False
        self.running_review = False
        self._stop = threading.Event()
        self._generation = 0
        self._progress = _Progress(self)
        self._progress.changed.connect(self._on_progress)
        self.apply_buttons: dict[str, QPushButton] = {}
        self.explain_buttons: dict[str, QPushButton] = {}

        self.title = QLabel("Review")
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-weight: bold;")
        self.status = QLabel("")
        self.status.setStyleSheet("color: gray;")
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(260)
        self.bar.setVisible(False)
        head = QHBoxLayout()
        head.addWidget(self.title, 1)
        head.addWidget(self.status)
        head.addWidget(self.bar)

        self.profile_tree = QTreeWidget()
        self.profile_tree.setHeaderLabels(["Batch", "Molecules"])
        self.profile_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.profile_tree.itemDoubleClicked.connect(self._on_profile_item)
        self.profile_tree.setToolTip("Double-click a molecule id to select it in the Molecules list")

        self.findings_tree = QTreeWidget()
        self.findings_tree.setHeaderLabels(["Finding", "Molecules", ""])
        self.findings_tree.setRootIsDecorated(False)
        self.findings_tree.setWordWrap(True)
        hdr = self.findings_tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.findings_tree.currentItemChanged.connect(self._on_finding)
        self.detail = QTextBrowser()
        self.detail.setOpenLinks(False)
        self.detail.anchorClicked.connect(self._on_link)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.findings_tree)
        right.addWidget(self.detail)
        right.setSizes([430, 220])
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.profile_tree)
        split.addWidget(right)
        split.setSizes([380, 720])

        self.preset_buttons: dict[str, QPushButton] = {}
        presets = QHBoxLayout()
        presets.addWidget(QLabel("Presets:"))
        for key, name in advisor.PRESET_NAMES.items():
            b = QPushButton(name)
            b.setEnabled(False)
            b.clicked.connect(lambda _c=False, k=key: self.apply_preset(k))
            self.preset_buttons[key] = b
            presets.addWidget(b)
        presets.addStretch(1)
        self.preset_info = QLabel("")
        self.preset_info.setWordWrap(True)
        self.preset_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.before_box = QCheckBox("Review before each run")
        self.before_box.setChecked(review_before_run_enabled(win))
        self.before_box.toggled.connect(lambda on: set_review_before_run(win, on))
        self.refresh_btn = QPushButton("Review again")
        self.refresh_btn.clicked.connect(lambda _c=False: self.start())
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_now)
        self.cancel_btn = QPushButton("Cancel run")
        self.cancel_btn.clicked.connect(self.cancel_run)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(self.before_box)
        buttons.addStretch(1)
        for b in (self.refresh_btn, self.run_btn, self.cancel_btn, self.close_btn):
            buttons.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addWidget(split, 1)
        lay.addLayout(presets)
        lay.addWidget(self.preset_info)
        lay.addLayout(buttons)
        self._update_buttons()

    # ------------------------------------------------------------------ the review
    def start(self, selected_only: bool | None = None, before_run: bool | None = None) -> bool:
        """Review the list (or the selected molecules) with the pipeline's settings now, on a worker thread."""
        if selected_only is not None:
            self.selected_only = selected_only
        if before_run is not None:
            self.before_run = before_run
        win = self.win
        ids = win.mol_list.selected_ids() if self.selected_only else win.model.ids()
        if not ids:
            self.title.setText("No molecules to review: add some first.")
            self._update_buttons()
            return False
        rows = [win.model.row(i) for i in ids]
        specs = [(r.id, r.run_dir, r.input, r.smiles, r.name, r.source) for r in rows]
        cfg = win.current_config()
        self._generation += 1
        gen = self._generation
        self._stop.set()                       # a review still running for older settings stops early
        stop = self._stop = threading.Event()
        progress = self._progress

        def job():
            from .molecule_list import gather_inputs

            inputs = gather_inputs(specs)
            return gen, advisor.review(inputs, cfg, progress=lambda i, n: progress.changed.emit(i, n),
                                       should_stop=stop.is_set)

        self.running_review = True
        self.title.setText(f"Reviewing {len(ids):,} molecule{'s' if len(ids) != 1 else ''}...")
        self.status.setText("")
        self.bar.setRange(0, len(ids))
        self.bar.setValue(0)
        self.bar.setVisible(len(ids) > 500)
        self._update_buttons()
        w = workers.FunctionWorker(job)
        workers.start(w, win, on_done=self._done, on_failed=self._failed)
        return True

    @Slot(int, int)
    def _on_progress(self, i: int, n: int) -> None:
        self.bar.setRange(0, max(1, n))
        self.bar.setValue(i)

    def _done(self, value) -> None:
        gen, review = value
        if gen != self._generation:
            return                              # an older review, for settings since changed
        self.running_review = False
        self.bar.setVisible(False)
        self.show_review(review)

    def _failed(self, msg: str) -> None:
        self.running_review = False
        self.bar.setVisible(False)
        self.review = None
        self.title.setText("The review failed; the run can still go ahead.")
        self.status.setText(msg.splitlines()[0][:200] if msg else "")
        self.win.log(f"Review failed: {msg}")
        self._update_buttons()

    # ------------------------------------------------------------------ showing it
    def show_review(self, r: advisor.Review) -> None:
        self.review = r
        counts = r.counts()
        self.title.setText(f"Review of {r.n:,} molecule{'s' if r.n != 1 else ''}: "
                                         f"{advisor.settings_line(r)}")
        self.status.setText(f"{counts['problem']} problems, {counts['warning']} warnings, {counts['info']} notes "
                            f"({r.seconds:.1f} s)")
        self._fill_profile(r)
        self._fill_findings(r)
        self._fill_presets(r)
        self._update_buttons()

    def _ids_items(self, parent: QTreeWidgetItem, ids: list[str], n: int) -> None:
        for ident in ids:
            it = QTreeWidgetItem([ident, ""])
            it.setData(0, ID_ROLE, ident)
            parent.addChild(it)
        if n > len(ids):
            parent.addChild(QTreeWidgetItem([f"... {n - len(ids):,} more", ""]))

    def _fill_profile(self, r: advisor.Review) -> None:
        t = self.profile_tree
        t.clear()
        p = r.profile
        labels = dict(advisor.SECTION_LABELS)
        if r.settings.get("protonate"):
            labels["charge"] = f"Charge class at pH {r.settings['ph']:g}"
        for section, groups in p.sections.items():
            if not groups:
                continue
            top = QTreeWidgetItem([labels.get(section, section), ""])
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            if section == "pfas":
                top.setToolTip(0, advisor.PFAS_RULE)
            t.addTopLevelItem(top)
            for g in groups:
                it = QTreeWidgetItem([g.label, f"{g.n:,}"])
                top.addChild(it)
                self._ids_items(it, g.ids, g.n)
            top.setExpanded(section in ("charge", "pfas", "groups", "flags"))
        st = p.stats
        if st["heavy"]["median"] is not None:
            top = QTreeWidgetItem(["Size (median, range)", ""])
            top.addChild(QTreeWidgetItem([f"heavy atoms {st['heavy']['median']:g} ({st['heavy']['min']}-"
                                          f"{st['heavy']['max']})", ""]))
            top.addChild(QTreeWidgetItem([f"rotatable bonds {st['rotatable']['median']:g} ({st['rotatable']['min']}-"
                                          f"{st['rotatable']['max']})", ""]))
            t.addTopLevelItem(top)
            top.setExpanded(True)
        cov = QTreeWidgetItem(["Methods that can take these molecules", ""])
        t.addTopLevelItem(cov)
        n_ok = p.n - p.count("flags", "not_computable")
        for c in r.coverage:
            here = "" if c.installed is None else " (installed)" if c.installed else " (not found here)"
            it = QTreeWidgetItem([f"{c.label}: {c.kind}{here}", f"{c.n_ok:,}/{n_ok:,}"])
            if c.note:
                it.setToolTip(0, c.note)
            cov.addChild(it)
            if c.ids_excluded:
                ex = QTreeWidgetItem([f"cannot take ({c.n_elements:,} elements, {c.n_spin:,} open shell)", ""])
                it.addChild(ex)
                self._ids_items(ex, c.ids_excluded, c.n_elements + c.n_spin)
        steps = QTreeWidgetItem(["Where each step runs", ""])
        t.addTopLevelItem(steps)
        for step, where in r.steps:
            steps.addChild(QTreeWidgetItem([f"{step}: {where}", ""]))
        steps.setExpanded(True)

    def _fill_findings(self, r: advisor.Review) -> None:
        t = self.findings_tree
        t.clear()
        self.apply_buttons.clear()
        self.explain_buttons.clear()
        style = self.style()
        for f in r.findings:
            it = QTreeWidgetItem([f"{f.topic}: {f.message}", f"{f.n:,}" if f.n else ""])
            it.setIcon(0, style.standardIcon(LEVEL_ICONS[f.level]))
            it.setToolTip(0, f"{LEVEL_NAMES[f.level]}: {f.message}\n\n{f.why}")
            it.setData(0, ID_ROLE, f.key)
            t.addTopLevelItem(it)
            box = QWidget()
            row = QHBoxLayout(box)
            row.setContentsMargins(0, 0, 0, 0)
            if f.changes:
                b = QPushButton("Apply")
                b.setToolTip("Set in the pipeline: " + describe_changes(f.changes))
                b.clicked.connect(lambda _c=False, key=f.key: self.apply_finding(key))
                row.addWidget(b)
                self.apply_buttons[f.key] = b
            if f.guide:
                e = QPushButton("Explain")
                e.setToolTip("Methods guide: " + ", ".join(guide.ENTRIES[k].name for k in f.guide if k in guide.ENTRIES))
                e.clicked.connect(lambda _c=False, key=f.key: self.explain(key))
                row.addWidget(e)
                self.explain_buttons[f.key] = e
            t.setItemWidget(it, 2, box)
        if t.topLevelItemCount():
            t.setCurrentItem(t.topLevelItem(0))
        else:
            self.detail.setHtml("<p>No findings.</p>")

    def finding(self, key: str) -> advisor.Finding | None:
        if self.review is None:
            return None
        return next((f for f in self.review.findings if f.key == key), None)

    def _on_finding(self, cur, _prev=None) -> None:
        f = self.finding(cur.data(0, ID_ROLE)) if cur is not None else None
        self.detail.setHtml(finding_html(f) if f is not None else "")

    def _fill_presets(self, r: advisor.Review) -> None:
        for p in r.presets:
            b = self.preset_buttons[p.key]
            b.setEnabled(True)
            needs = f" Needs {', '.join(p.needs)} (not found here)." if p.needs else ""
            b.setToolTip(f"{p.rationale}\n\nEstimated: {p.cost.text()}.{needs}\n\nSets: {describe_changes(p.diff)}")
        rec = r.preset(r.recommended)
        if rec is not None:
            self.preset_info.setText(f"Recommended: {rec.rationale} Estimated {rec.cost.text()}."
                                     + (f" Needs {', '.join(rec.needs)}." if rec.needs else ""))

    # ------------------------------------------------------------------ actions
    def apply_finding(self, key: str) -> None:
        f = self.finding(key)
        if f is None or not f.changes:
            return
        skipped = apply_changes(self.win, f.changes)
        self.win.log(f"Review: applied {describe_changes(f.changes)}"
                     + (f" (not set: {', '.join(skipped)})" if skipped else ""))
        self.start()

    def apply_preset(self, key: str) -> None:
        p = self.review.preset(key) if self.review is not None else None
        if p is None:
            return
        skipped = apply_changes(self.win, p.changes)
        self.win.log(f"Review: preset {p.name}: {describe_changes(p.diff) or 'no change'}"
                     + (f" (not set: {', '.join(skipped)})" if skipped else ""))
        self.start()

    def explain(self, key: str) -> None:
        f = self.finding(key)
        if f is not None and f.guide:
            self.win.show_methods_guide(f.guide[0])

    def _on_link(self, url) -> None:
        text = url.toString()
        if text.startswith("guide:"):
            self.win.show_methods_guide(text.split(":", 1)[1])
        elif text.startswith("mol:"):
            self.win.mol_list.select_id(text.split(":", 1)[1])

    def _on_profile_item(self, item, _col=0) -> None:
        ident = item.data(0, ID_ROLE)
        if ident:
            self.win.mol_list.select_id(ident)

    def _update_buttons(self) -> None:
        r = self.review
        counts = r.counts() if r is not None else {"problem": 0, "warning": 0}
        self.run_btn.setVisible(self.before_run)
        self.cancel_btn.setVisible(self.before_run)
        self.run_btn.setText("Run anyway" if (counts["problem"] or counts["warning"]) else "Run")
        self.run_btn.setEnabled(self.before_run and self.win.worker is None)
        self.refresh_btn.setEnabled(not self.running_review)

    def run_now(self) -> None:
        """Start the run the review was opened for, with the pipeline settings as they are now."""
        if not self.before_run:
            return
        selected = self.selected_only
        self.before_run = False
        self._stop.set()
        self.hide()
        self.win.run_build(selected)

    def cancel_run(self) -> None:
        self.before_run = False
        self._stop.set()
        self.win.log("Run cancelled from the review.")
        self.hide()

    def closeEvent(self, event) -> None:
        if self.before_run:
            self.win.log("Review closed: the run was not started.")
        self.before_run = False
        self._stop.set()
        super().closeEvent(event)


def finding_html(f: advisor.Finding) -> str:
    """One finding as HTML for the detail pane: level, message, why, molecules (links), what to do, guide links."""
    parts = [f"<p><b>{LEVEL_NAMES[f.level]} ({html.escape(f.topic)}):</b> {html.escape(f.message)}</p>"]
    if f.why:
        parts.append(f"<p><b>Why:</b> {html.escape(f.why)}"
                     + (f' <span style="color: gray;">({html.escape(f.evidence)})</span>' if f.evidence else "")
                     + "</p>")
    if f.recommendation or f.changes:
        parts.append(f"<p><b>What to do:</b> {html.escape(f.recommendation)}"
                     + (f" <i>Apply sets: {html.escape(describe_changes(f.changes))}</i>" if f.changes else "")
                     + "</p>")
    if f.ids:
        links = ", ".join(f'<a href="mol:{html.escape(i, quote=True)}">{html.escape(i)}</a>' for i in f.ids)
        more = f" ... and {f.n - len(f.ids):,} more" if f.n > len(f.ids) else ""
        parts.append(f"<p><b>Molecules ({f.n:,}):</b> {links}{more}</p>")
    if f.guide:
        links = ", ".join(f'<a href="guide:{guide.ENTRIES[k].anchor}">{html.escape(guide.ENTRIES[k].name)}</a>'
                          for k in f.guide if k in guide.ENTRIES)
        parts.append(f"<p><b>Explain:</b> {links}</p>")
    return "\n".join(parts)


# ============================================================================================ installing it
def _qsettings():
    from .main_window import _window_settings

    return _window_settings()


def review_before_run_enabled(win) -> bool:
    try:
        value = _qsettings().value(SETTING_KEY, True)
    except Exception:  # noqa: BLE001
        return True
    return value not in (False, "false", "0", 0)


def set_review_before_run(win, on: bool) -> None:
    try:
        _qsettings().setValue(SETTING_KEY, bool(on))
    except Exception:  # noqa: BLE001, S110 - a read-only settings store: the choice lasts this session
        pass
    act = getattr(win, "a_review_before", None)
    if act is not None and act.isChecked() != bool(on):
        act.setChecked(bool(on))
    dlg = getattr(win, "review_dialog", None)
    if dlg is not None and dlg.before_box.isChecked() != bool(on):
        dlg.before_box.setChecked(bool(on))


def show_review(win, selected_only: bool = False, before_run: bool = False) -> ReviewDialog:
    dlg = getattr(win, "review_dialog", None)
    if dlg is None:
        dlg = ReviewDialog(win)
        win.review_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    dlg.start(selected_only=selected_only, before_run=before_run)
    return dlg


def install(win) -> None:
    def before_run(selected_only: bool) -> bool:
        if not review_before_run_enabled(win):
            return False
        show_review(win, selected_only=selected_only, before_run=True)
        return True

    win.review_before_run = before_run
    btn = QPushButton("Review...")
    btn.setToolTip("Review the molecules with the pipeline's settings: what is in the batch, which methods fit, what "
                   "to change (F6)")
    btn.clicked.connect(lambda _c=False: show_review(win, selected_only=False))
    run_row = win.run_btn.parentWidget().layout() if win.run_btn.parentWidget() is not None else None
    # the Run buttons' row: insert the button before "Run all"
    for lay in _layouts(run_row):
        idx = lay.indexOf(win.run_btn)
        if idx >= 0:
            lay.insertWidget(idx, btn)
            break
    win.review_btn = btn
    act = QAction("Review molecules...", win)
    act.setShortcut(QKeySequence("F6"))
    act.setToolTip("The model advisor's review of the list with the pipeline's settings")
    act.triggered.connect(lambda _c=False: show_review(win, selected_only=False))
    before = QAction("Review before each run", win)
    before.setCheckable(True)
    before.setChecked(review_before_run_enabled(win))
    before.toggled.connect(lambda on: set_review_before_run(win, on))
    win.a_review = act
    win.a_review_before = before
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == "Build":
            menu = action.menu()
            first = menu.actions()[0] if menu.actions() else None
            menu.insertAction(first, act)
            menu.insertAction(first, before)
            menu.insertSeparator(first)
            break


def _layouts(root):
    """``root`` and every layout nested in it (the Run row is a QHBoxLayout inside the dock's QVBoxLayout)."""
    if root is None:
        return []
    out, stack = [], [root]
    while stack:
        lay = stack.pop()
        out.append(lay)
        for i in range(lay.count()):
            child = lay.itemAt(i).layout()
            if child is not None:
                stack.append(child)
    return out
