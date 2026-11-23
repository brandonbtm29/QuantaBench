"""The Jobs dock (UI-5): the job queue, each job's progress and log, and what the running job uses.

    #  Job                               Machine         Status      Progress        Time
    1  Screen: conformer distribution..  this computer   done        12 / 12         4 min
    2  Careful: frequencies, 3 mol...    this computer   running     1 / 3 (aspirin: refine)
    3  Standard: 40 molecules            macbook-pro     running     12 / 40 (PFOS: refine)
    4  Run all: 40 molecules             this computer   queued      40 molecules

Below the list: Cancel (a queued job never starts; the running job stops after its current step, as the Cancel
button does), Up / Down (queued jobs), Open output folder, Clear finished; the selected job's settings, run folder and
log; and a line of what the running job uses: its threads, GPU and memory settings, and this computer's live load,
temperatures and memory, mirrored from the machine monitor strip (MON-1) in the Pipeline dock.

A job on a worker (the job dialog's Run on, DIST-3) is listed with its machine; Cancel sends the cancel there, and
when it ends its molecules' rows in the Molecules list take the results (``apply_remote_rows``).

An MD job (SIM-1: a run folder run by its own supervisor process) shows its progress in ns with ns/day and the ETA;
Cancel pauses it at its next pause point (a checkpoint is written), and Resume queues the folder again to continue.
Simulate > Run MD folder... (``run_md_folder``) submits one, or follows a run that is already going.

``install(win)`` makes the queue (``win.job_queue``, with ``make_follower`` for jobs on workers), the dock (object
name "jobs", in the Calculate workspace next to Results), Build > New job... (Ctrl+J) and a New job... button next to
the Pipeline dock's Run buttons.
"""

from __future__ import annotations

import html
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
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

from . import keymap
from .job_queue import FINISHED, JobQueue

COLUMNS = ("#", "Job", "Machine", "Status", "Progress", "Time")
C_NAME, C_MACHINE, C_STATUS, C_PROGRESS, C_TIME = 1, 2, 3, 4, 5
HERE = "this computer"
NUMBER_ROLE = Qt.ItemDataRole.UserRole
STATUS_COLORS = {"running": "#2471a3", "cancelling": "#d35400", "done": "#1e8449", "failed": "#c0392b",
                 "cancelled": "gray", "queued": "", "paused": "#b7950b", "pausing": "#d35400"}


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
        hdr.setSectionResizeMode(C_NAME, QHeaderView.ResizeMode.Stretch)
        for c in (0, C_MACHINE, C_STATUS, C_PROGRESS, C_TIME):
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
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.setObjectName("md_resume_btn")
        self.resume_btn.setToolTip("An MD run that paused, stopped or failed: continue it from its newest checkpoint")
        self.resume_btn.clicked.connect(lambda _c=False: self.resume_selected())
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.clicked.connect(lambda _c=False: self.open_selected())
        self.clear_btn = QPushButton("Clear finished")
        self.clear_btn.clicked.connect(lambda _c=False: self.queue.clear_finished())
        buttons = QHBoxLayout()
        for b in (self.new_btn, self.cancel_btn, self.resume_btn, self.up_btn, self.down_btn, self.open_btn):
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
        it = QTreeWidgetItem([str(job.number), job.name, "", "", "", ""])
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
        it.setText(C_NAME, job.name)
        it.setToolTip(C_NAME, job.summary or job.name)
        machine = job.machine or HERE if job.is_remote else HERE
        it.setText(C_MACHINE, machine + (" (unreachable)" if job.unreachable else ""))
        it.setToolTip(C_MACHINE, f"{machine}: worker job {job.remote_id}" if job.remote_id else machine)
        live = job.active or (job.is_remote and job.status == "queued")
        shown = "pausing" if job.is_md and job.status == "cancelling" else job.status
        status = shown + (f": {job.current}" if live and job.current else "")
        it.setText(C_STATUS, shown)
        it.setToolTip(C_STATUS, status + (f"\n{job.error}" if job.error else ""))
        color = STATUS_COLORS.get(shown, "")
        it.setForeground(C_STATUS, self.palette().text() if not color else _brush(color))
        it.setText(C_PROGRESS, job.progress_text() + (f"  ({job.current})" if live and job.current else ""))
        it.setText(C_TIME, _duration(job.elapsed()))

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
        shown = "pausing" if job.is_md and job.status == "cancelling" else job.status
        lines = [f"<b>{html.escape(job.name)}</b> ({shown})"]
        if job.summary:
            lines.append(html.escape(job.summary))
        cfg = job.cfg
        if job.is_md:
            lines += md_detail_lines(job)
        elif job.is_remote:
            where = job.machine or job.run_on
            lines.append(html.escape(f"on {where}" + (f", worker job {job.remote_id}" if job.remote_id else "")
                                     + ": that machine's threads, GPU and priority"
                                     + (f"; {job.current}" if job.current else "")))
        elif cfg is not None:
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
        self.cancel_btn.setEnabled(job is not None and job.status in ("queued", "running")
                                   and (not job.is_remote or job.follower is not None))
        md_running = job is not None and job.is_md and job.status == "running"
        self.cancel_btn.setText("Pause" if md_running else "Cancel")
        self.cancel_btn.setToolTip("Pause the MD run at its next pause point (a checkpoint is written there)"
                                   if md_running else
                                   "A queued job: it never starts. The running job: stop it after the current step")
        self.resume_btn.setVisible(any(j.is_md for j in self.queue.jobs))
        self.resume_btn.setEnabled(job is not None and job.is_md and job.status in ("paused", "failed", "cancelled")
                                   and job.output_dir is not None)
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
        if job.status == "queued" or job.is_remote or job.is_md:
            return self.queue.cancel(job.number)
        if job.status == "running" and job is self.queue.running:
            self.win.cancel_build()
            return job.status == "cancelling"
        return False

    def resume_selected(self):
        job = self.selected()
        if job is None:
            return None
        new = self.queue.resume_md(job.number)
        if new is not None:
            self.select(new.number)
        return new

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
                it.setText(C_TIME, _duration(job.elapsed(time.time())))
        self._update_usage()

    def _update_usage(self) -> None:
        remote = [j.machine for j in self.queue.jobs if j.is_remote and j.status not in FINISHED]
        self.usage.setText(usage_html(self.win, self.queue.running, remote))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.timer.start()
        self.tick()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.timer.stop()


def md_detail_lines(job) -> list[str]:
    """The selected MD job's details (HTML lines): its folder's state, where it is, speed, ETA, alerts, segments."""
    st = job.md or {}
    prog = st.get("progress") or {}
    out = []
    state = st.get("state")
    if state:
        out.append(html.escape(f"run: {state}" + (" (no supervisor: Resume continues it)" if st.get("stale") else "")))
    if prog.get("step") is not None:
        out.append(html.escape(f"part {prog.get('part')} ({prog.get('stage')}), step {prog['step']:,} of "
                               f"{prog.get('part_end') or 0:,}; {prog.get('variant')} x {prog.get('threads')} threads "
                               f"on {prog.get('machine')}"))
    if prog.get("ns_per_day") or prog.get("eta_text"):
        out.append(html.escape(f"{(prog.get('ns_per_day') or 0):.3g} ns/day over the last minutes"
                               + (f", ETA {prog['eta_text']}" if prog.get("eta_text") else "")))
    for a in prog.get("alerts") or []:
        out.append(f"<span style='color:#d35400'>{html.escape(a)}</span>")
    segs = st.get("segments") or []
    if segs:
        out.append(html.escape("segments: " + "; ".join(
            f"{s.get('n')} {s.get('part')} {s.get('status')}"
            + (f" {s.get('ns_per_day'):.3g} ns/day" if s.get("ns_per_day") else "") for s in segs[-4:])))
    return out


def run_md_folder(win, folder=None):
    """Simulate > Run MD folder...: run (or resume, or follow) a folder written by md liquid-slab, md smd or the
    Surface builder as an MD job. Returns the job, or None."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from ..md import sim
    from .job_queue import md_job

    if folder is None:
        start = str(getattr(getattr(win, "s", None), "output_dir", "") or "")
        folder = QFileDialog.getExistingDirectory(win, "Run an MD folder", start)
        if not folder:
            return None
    if not sim.is_run_folder(folder):
        QMessageBox.information(win, "Run an MD folder", f"{folder} is not an MD run folder: md liquid-slab, md smd "
                                "or the Surface builder writes one (run.json, or a liquid_setup.json or smd_setup.json "
                                "with its inputs).")
        return None
    queue = win.job_queue
    if any(j.is_md and j.output_dir is not None and str(j.output_dir) == str(folder) and j.status not in FINISHED
           for j in queue.jobs):
        return None
    if choose_md_engine(win, folder) is None:
        return None
    job = queue.submit(md_job(folder))
    dock = getattr(win, "jobs_dock", None)
    if dock is not None:
        dock.show()
        dock.raise_()
    if getattr(win, "job_monitor", None) is not None:
        win.job_monitor.select(job.number)
    return job


def choose_md_engine(win, folder) -> str | None:
    """The MD engine of a liquid-slab folder that has not started (OPENMM-1): asked with the note on what each
    engine gives, the folder's own engine first; the folder is switched if another is chosen. Other folders (a run
    that has started, a steered pull) keep theirs. None: cancelled."""
    from PySide6.QtWidgets import QInputDialog, QMessageBox

    from ..md import liquid as L
    from ..md import sim

    folder = Path(folder)
    if not (folder / "liquid_setup.json").is_file():
        return "lammps"
    try:
        run = sim.Run.load(folder)
    except sim.RunError:
        return "lammps"
    if run.segments:
        return run.engine
    labels = {"lammps": "LAMMPS (the reference engine)",
              "openmm": "OpenMM (much faster on GPUs, reduced arithmetic precision)"}
    order = [run.engine, *(e for e in L.ENGINES if e != run.engine)]
    import textwrap

    item, ok = QInputDialog.getItem(win, "MD engine", f"Run {folder.name} with:\n\n" + textwrap.fill(L.ENGINE_NOTE, 80),
                                    [labels[e] for e in order], 0, False)
    if not ok:
        return None
    engine = next(e for e, text in labels.items() if text == item)
    if engine != run.engine:
        try:
            L.set_engine(folder, engine)
        except (sim.RunError, ValueError) as exc:
            QMessageBox.warning(win, "MD engine", str(exc))
            return None
    return engine


def _brush(color: str):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(color))


def usage_html(win, job, remote: list[str] | None = None) -> str:
    """What the running job uses (its resource settings) and this computer's live load (the machine monitor's
    latest sample, never a new reading: nothing here blocks). ``remote``: the machines of jobs on workers."""
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
    md = getattr(getattr(win, "job_queue", None), "md_running", None)
    if md is not None:
        prog = (md.md or {}).get("progress") or {}
        what = (f"{html.escape(str(prog.get('variant') or 'LAMMPS'))} x {prog.get('threads')} threads"
                if prog.get("threads") else "LAMMPS in its own process")
        parts.append(f"<b>MD job {md.number}</b>: {what}, beside the builds")
    elif job is None or job.cfg is None:
        parts.append("<b>No job running here</b>" if remote else "<b>No job running</b>")
    if remote:
        counts: dict[str, int] = {}
        for m in remote:
            counts[m] = counts.get(m, 0) + 1
        parts.append(", ".join(f"{n} on {html.escape(m)}" for m, n in counts.items()))
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


def make_follower(win, job, on_update, on_local, local_busy):
    """A started ``dist.remote_job.RemoteJob`` for a job on a worker (``JobQueue.make_follower``): the worker token
    from the settings, this computer's output folder (where a worker's synced run folder arrives), the job's own
    folder (a project's) for a downloaded archive."""
    from .. import settings as settings_mod
    from ..dist import profile, remote_job

    s = settings_mod.load()
    out = s.output_path()
    return remote_job.RemoteJob(
        job.run_on, job.todo, job.cfg, remote_job.label_for(job.name, job.number),
        token=s.dist.worker_token, out_dir=out,
        download_dir=getattr(job.cfg, "output_dir", "") or out, on_update=on_update, on_local=on_local,
        local_busy=local_busy, client_name=profile.machine_name()).start()


def apply_remote_rows(win, number: int, rows) -> int:
    """A worker job's results in the Molecules list: each of its molecules' rows takes the status, method,
    conformers, energy, folder and run folder of the run folder that came back (``JobQueue.remote_rows``)."""
    job = win.job_queue.job(number)
    ids = set(job.ids) if job is not None else set()
    n = 0
    for r in rows:
        if r.id not in ids or win.model.row(r.id) is None:
            continue
        fields = dict(status=r.status, stage="", method=r.method, n_conformers=r.n_conformers,
                      energy_ev=r.energy_ev, error=r.error, folder=r.folder, run_dir=r.run_dir)
        if r.formula:
            fields["formula"] = r.formula
        win.model.update(r.id, **fields)
        win.results.discard(r.id)            # an older result of it must not come back from the cache
        n += 1
    if win.mol_list.current_id() in ids:
        win._on_molecule_selected()
    counts = getattr(win, "_update_counts", None)
    if callable(counts):
        counts()
    return n


def _mark_remote(win, number: int, text: str) -> None:
    job = win.job_queue.job(number)
    if job is None:
        return
    ids = [i for i in job.ids if win.model.row(i) is not None]
    if ids:
        win.model.update_many(ids, stage=text)


def install(win) -> None:
    queue = JobQueue(win._start_job, win._can_start_job, win,
                     make_follower=lambda job, on_update, on_local, busy: make_follower(win, job, on_update,
                                                                                         on_local, busy))
    queue.remote_rows.connect(lambda n, rows: apply_remote_rows(win, n, rows))
    queue.remote_marked.connect(lambda n, text: _mark_remote(win, n, text))
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
    job_tip = "Submit molecules as a job: a level of theory, a task and resources"
    keymap.bind(win, a, "build.new_job", job_tip)
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
        if isinstance(getattr(win, "keymap", None), keymap.Keymap):
            win.keymap.follow_tooltip(btn, "build.new_job", job_tip)
        else:
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
