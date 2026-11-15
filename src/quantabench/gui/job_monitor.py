"""The Jobs dock (UI-5): the job queue, each job's progress and log, and what the running job uses.

    #  Job                               Status      Progress        Time
    1  Screen: conformer distribution..  done        12 / 12         4 min
    2  Careful: frequencies, 3 mol...    running     1 / 3 (aspirin: refine)
    3  Run all: 40 molecules             queued      40 molecules

Below the list: Cancel (a queued job never starts; the running job stops after its current step, as the Cancel
button does), Up / Down (queued jobs), Open output folder, Clear finished; the selected job's settings, run folder and
log; and a line of what the running job uses: its threads, GPU and memory settings, and this computer's live load,
temperatures and memory, mirrored from the machine monitor strip (MON-1) in the Pipeline dock.

``install(win)`` makes the queue (``win.job_queue``), the dock (object name "jobs", in the Calculate workspace next
to Results), Build > New job... (Ctrl+J) and a New job... button next to the Pipeline dock's Run buttons.
"""

from __future__ import annotations

import html
import time

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .job_queue import FINISHED, JobQueue

COLUMNS = ("#", "Job", "Status", "Progress", "Time")
NUMBER_ROLE = Qt.ItemDataRole.UserRole
STATUS_COLORS = {"running": "#2471a3", "cancelling": "#d35400", "done": "#1e8449", "failed": "#c0392b",
                 "cancelled": "gray", "queued": ""}


def _duration(s: float) -> str:
    if s <= 0:
        return ""
    if s < 1:
        return "< 1 s"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


class JobMonitor(QWidget):
    """The dock's widget. ``win``: the main window (its ``job_queue``, ``cancel_build`` and machine monitor)."""

    def __init__(self, win, queue: JobQueue, parent=None):
        super().__init__(parent)
        self.win = win
        self.queue = queue
        self.items: dict[int, QTreeWidgetItem] = {}
        self.tree = QTreeWidget()
        self.tree.setObjectName("jobs_tree")
        self.tree.setHeaderLabels(list(COLUMNS))
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (0, 2, 3, 4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.currentItemChanged.connect(lambda *_a: self._show_selected())
        self.empty = QLabel("No jobs yet. Run all, Run selected or Build > New job... (Ctrl+J) put one here.")
        self.empty.setStyleSheet("color: gray;")
        self.empty.setWordWrap(True)

        self.new_btn = QPushButton("New job...")
        self.new_btn.setToolTip("Submit molecules with a level of theory, a task and resources (Ctrl+J)")
        self.new_btn.clicked.connect(lambda _c=False: open_job_dialog(self.win))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("A queued job: it never starts. The running job: stop it after the current step")
        self.cancel_btn.clicked.connect(lambda _c=False: self.cancel_selected())
        self.up_btn = QPushButton("Up")
        self.up_btn.clicked.connect(lambda _c=False: self.move_selected(-1))
        self.down_btn = QPushButton("Down")
        self.down_btn.clicked.connect(lambda _c=False: self.move_selected(1))
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.clicked.connect(lambda _c=False: self.open_selected())
        self.clear_btn = QPushButton("Clear finished")
        self.clear_btn.clicked.connect(lambda _c=False: self.queue.clear_finished())
        buttons = QHBoxLayout()
        for b in (self.new_btn, self.cancel_btn, self.up_btn, self.down_btn, self.open_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.clear_btn)

        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.log = QPlainTextEdit()
        self.log.setObjectName("job_log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setPlaceholderText("The selected job's log")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.detail)
        rl.addWidget(self.log, 1)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.tree, 1)
        ll.addWidget(self.empty)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([560, 440])

        self.usage = QLabel()
        self.usage.setObjectName("job_usage")
        self.usage.setWordWrap(True)
        self.usage.setTextFormat(Qt.TextFormat.RichText)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.addWidget(split, 1)
        lay.addLayout(buttons)
        lay.addWidget(self.usage)

        queue.added.connect(self._on_added)
        queue.changed.connect(self._on_changed)
        queue.removed.connect(self._on_removed)
        self._shown_log = (0, 0)               # (job number, lines shown)
        self.timer = QTimer(self)               # elapsed times and the machine line, while visible
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.tick)
        self._update_buttons()
        self._update_usage()

    # ------------------------------------------------------------------ the list
    def _on_added(self, number: int) -> None:
        job = self.queue.job(number)
        if job is None:
            return
        it = QTreeWidgetItem([str(job.number), job.name, "", "", ""])
        it.setData(0, NUMBER_ROLE, job.number)
        self.items[number] = it
        self.tree.addTopLevelItem(it)
        self._fill(it, job)
        self._reorder()
        if self.tree.currentItem() is None or job.status == "running":
            self.tree.setCurrentItem(it)
        self.empty.setVisible(False)

    def _on_changed(self, number: int) -> None:
        job = self.queue.job(number)
        it = self.items.get(number)
        if job is None or it is None:
            return
        self._fill(it, job)
        self._reorder()
        if self.selected() is job:
            self._show_selected()
        self._update_buttons()
        self._update_usage()

    def _on_removed(self, number: int) -> None:
        it = self.items.pop(number, None)
        if it is not None:
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))
        self.empty.setVisible(not self.items)
        self._show_selected()
        self._update_buttons()

    def _reorder(self) -> None:
        """The tree in the queue's order (Up / Down reorder queued jobs)."""
        order = [j.number for j in self.queue.jobs]
        now = [self.tree.topLevelItem(i).data(0, NUMBER_ROLE) for i in range(self.tree.topLevelItemCount())]
        if now == order:
            return
        cur = self.selected()
        self.tree.blockSignals(True)
        for n in order:
            it = self.items.get(n)
            if it is not None:
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(it))
                self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        if cur is not None and cur.number in self.items:
            self.tree.setCurrentItem(self.items[cur.number])

    def _fill(self, it: QTreeWidgetItem, job) -> None:
        it.setText(1, job.name)
        it.setToolTip(1, job.summary or job.name)
        status = job.status + (f": {job.current}" if job.active and job.current else "")
        it.setText(2, job.status)
        it.setToolTip(2, status + (f"\n{job.error}" if job.error else ""))
        color = STATUS_COLORS.get(job.status, "")
        it.setForeground(2, self.palette().text() if not color else _brush(color))
        it.setText(3, job.progress_text() + (f"  ({job.current})" if job.active and job.current else ""))
        it.setText(4, _duration(job.elapsed()))

    def selected(self):
        it = self.tree.currentItem()
        return self.queue.job(it.data(0, NUMBER_ROLE)) if it is not None else None

    def select(self, number: int) -> None:
        it = self.items.get(number)
        if it is not None:
            self.tree.setCurrentItem(it)

    def _show_selected(self) -> None:
        job = self.selected()
        if job is None:
            self.detail.setText("")
            self.log.clear()
            self._shown_log = (0, 0)
            self._update_buttons()
            return
        lines = [f"<b>{html.escape(job.name)}</b> ({job.status})"]
        if job.summary:
            lines.append(html.escape(job.summary))
        cfg = job.cfg
        if cfg is not None:
            lines.append(html.escape(f"threads {cfg.threads or 'auto'}, priority {cfg.priority}, GPU {cfg.gpu}"
                                     + (f", {cfg.qm_memory_gb:g} GB per DFT job" if cfg.qm_memory_gb else "")
                                     + (f", time limit {cfg.molecule_timeout_s:g} s" if cfg.molecule_timeout_s
                                        else "")))
        if job.output_dir is not None:
            lines.append(f"Run folder: {html.escape(str(job.output_dir))}")
        if job.error:
            lines.append(f"<span style='color:#c0392b'>{html.escape(job.error)}</span>")
        self.detail.setText("<br>".join(lines))
        # the log: append what is new for the same job, else refill
        shown_job, shown = self._shown_log
        log = list(job.log)
        if shown_job != job.number or shown > len(log):
            self.log.setPlainText("\n".join(log))
        elif len(log) > shown:
            self.log.appendPlainText("\n".join(log[shown:]))
        self._shown_log = (job.number, len(log))
        self._update_buttons()

    def _update_buttons(self) -> None:
        job = self.selected()
        self.cancel_btn.setEnabled(job is not None and job.status in ("queued", "running"))
        queued = self.queue.queued()
        self.up_btn.setEnabled(job is not None and job.status == "queued" and queued and queued[0] is not job)
        self.down_btn.setEnabled(job is not None and job.status == "queued" and queued and queued[-1] is not job)
        self.open_btn.setEnabled(job is not None and job.output_dir is not None)
        self.clear_btn.setEnabled(any(j.status in FINISHED for j in self.queue.jobs))

    # ------------------------------------------------------------------ actions
    def cancel_selected(self) -> bool:
        job = self.selected()
        if job is None:
            return False
        if job.status == "queued":
            return self.queue.cancel(job.number)
        if job.status == "running" and job is self.queue.running:
            self.win.cancel_build()
            return job.status == "cancelling"
        return False

    def move_selected(self, by: int) -> bool:
        job = self.selected()
        return job is not None and self.queue.move(job.number, by)

    def open_selected(self) -> bool:
        job = self.selected()
        if job is None or job.output_dir is None:
            return False
        return QDesktopServices.openUrl(QUrl.fromLocalFile(str(job.output_dir)))

    # ------------------------------------------------------------------ live
    def tick(self) -> None:
        for n, it in self.items.items():
            job = self.queue.job(n)
            if job is not None and job.active:
                it.setText(4, _duration(job.elapsed(time.time())))
        self._update_usage()

    def _update_usage(self) -> None:
        self.usage.setText(usage_html(self.win, self.queue.running))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.timer.start()
        self.tick()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.timer.stop()


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(color))


def usage_html(win, job) -> str:
    """What the running job uses (its resource settings) and this computer's live load (the machine monitor's
    latest sample, never a new reading: nothing here blocks)."""
    from .. import priority

    parts = []
    if job is not None and job.cfg is not None:
        cfg = job.cfg
        try:
            threads = priority.budget_of(cfg)
        except Exception:  # noqa: BLE001
            threads = cfg.threads
        parts.append(f"<b>Job {job.number}</b>: {threads} thread{'s' if threads != 1 else ''}"
                     f"{' (auto)' if not cfg.threads else ''}, GPU {html.escape(cfg.gpu)}, priority "
                     f"{html.escape(cfg.priority)}")
    else:
        parts.append("<b>No job running</b>")
    strip = getattr(win, "machine_monitor", None)
    sample = None
    if strip is not None:
        try:
            sample = strip.monitor.latest(wake=False)
        except Exception:  # noqa: BLE001 - a mirror, never worth an error
            sample = None
    if sample is not None:
        from .machine_monitor import cpu_html, gpu_html, ram_html

        parts += [cpu_html(sample), gpu_html(sample), ram_html(sample)]
    elif strip is not None:
        parts.append("<span style='color:gray'>machine metrics: the machine monitor in the Pipeline dock samples "
                     "them while it is shown</span>")
    return "&nbsp;&nbsp;·&nbsp;&nbsp;".join(parts)


# ============================================================================================ installing
def open_job_dialog(win, ids: list[str] | None = None, show: bool = True):
    from PySide6.QtWidgets import QMessageBox

    from .job_dialog import JobDialog

    if not win.model.ids():
        QMessageBox.information(win, "New job", "Add a molecule first (SMILES, name, or a file).")
        return None
    dlg = JobDialog(win, ids=ids)
    dlg.submitted.connect(lambda job, w=win: _on_submitted(w, job))
    win.job_dialog = dlg
    if show:
        dlg.open()
    return dlg


def _on_submitted(win, job) -> None:
    dock = getattr(win, "jobs_dock", None)
    if dock is not None and job is not None:
        dock.show()
        dock.raise_()
        win.job_monitor.select(job.number)


def install(win) -> None:
    queue = JobQueue(win._start_job, win._can_start_job, win)
    win.job_queue = queue
    monitor = JobMonitor(win, queue)
    win.job_monitor = monitor
    dock = QDockWidget("Jobs", win)
    dock.setObjectName("jobs")
    dock.setWidget(monitor)
    win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    dock.hide()                               # the workspaces show it (Calculate: next to Results)
    win.jobs_dock = dock
    a = QAction("New job...", win)
    a.setShortcut(QKeySequence("Ctrl+J"))
    a.setToolTip("Submit molecules as a job: a level of theory, a task and resources (Ctrl+J)")
    a.triggered.connect(lambda _c=False: open_job_dialog(win))
    win.a_new_job = a
    build = None
    for act in win.menuBar().actions():
        if act.text().replace("&", "") == "Build" and act.menu() is not None:
            build = act.menu()
            break
    if build is not None:
        first = build.actions()[0] if build.actions() else None
        build.insertAction(first, a)
    view = next((act.menu() for act in win.menuBar().actions() if act.text().replace("&", "") == "View"
                 and act.menu() is not None), None)
    if view is not None:
        view.addAction(dock.toggleViewAction())
    row = _run_row(win)
    if row is not None:
        btn = QPushButton("New job...")
        btn.setObjectName("new_job_btn")
        btn.setToolTip(a.toolTip())
        btn.clicked.connect(lambda _c=False: open_job_dialog(win))
        row.insertWidget(0, btn)
        win.new_job_btn = btn


def _run_row(win):
    """The Pipeline dock's row of Run buttons (a QHBoxLayout)."""
    btn = getattr(win, "run_btn", None)
    parent = btn.parentWidget() if btn is not None else None
    if parent is None or parent.layout() is None:
        return None
    lay = parent.layout()
    for i in range(lay.count()):
        sub = lay.itemAt(i).layout()
        if sub is not None and sub.indexOf(btn) >= 0:
            return sub
    return None
