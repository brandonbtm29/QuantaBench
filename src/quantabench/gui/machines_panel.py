"""Machines dock: this computer's lanes and the machines in the registry (workers) - DIST-3, minimal.

Shows the machine registry (``dist.profile``: one JSON per machine in the synced settings folder),
each worker's live status from ``dist.client.discover`` (fetched off the GUI thread on Refresh), and
what the "Parallel lanes" setting means on this computer for the current pipeline settings
(``dist.lanes.plan_text``). Splitting a batch across machines is ``quantabench-cli grid build`` for now;
a "Run on" choice for the Run buttons is the next step (backlog DIST-3).
"""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QEvent, QObject, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

COLUMNS = ("Machine", "Status", "Hardware", "Throughput (molecules/h)", "Updated")


def _hardware(p) -> str:
    gpus = ", ".join(g.get("name", "") for g in (p.gpus or []) if g.get("name"))
    cores = f"{p.physical_cores}C/{p.logical_cores}T" if p.physical_cores else ""
    ram = f"{p.ram_gb:.0f} GB" if p.ram_gb else ""
    return "; ".join(x for x in (p.os, " ".join(y for y in (p.cpu, cores) if y), ram, gpus) if x)


def _throughput(p) -> str:
    items = sorted((p.throughput or {}).items(), key=lambda kv: -float(kv[1]))
    if not items:
        return "not measured"
    return ", ".join(f"{k} {float(v):,.0f}" for k, v in items[:4]) + (" ..." if len(items) > 4 else "")


def _sensor_text(raw) -> str:
    """"; CPU 37 % 4.62 GHz 61 °C" from a worker's /v1/info "sensors" (MON-1 workers; older ones have none)."""
    if not isinstance(raw, dict):
        return ""
    from .. import sensors

    s = sensors.Sample.from_dict(raw)
    parts = [f"CPU {s.cpu_percent:.0f} %"] if s.cpu_percent is not None else []
    if s.cpu_clock_mhz is not None:
        parts.append(sensors.fmt_clock(s.cpu_clock_mhz))
    if s.cpu_temp_c is not None:
        parts.append(f"{s.cpu_temp_c:.0f} °C")
    if s.gpu_percent is not None:
        parts.append(f"GPU {s.gpu_percent:.0f} %")
    return "; " + " ".join(parts) if parts else ""


def _age(p) -> str:
    from ..dist import profile

    try:
        return profile.format_age(profile.profile_age_s(p))
    except Exception:
        return p.updated or ""


class MachinesPanel(QWidget):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self._refs: dict = {}
        self._checked_at = 0.0
        self._busy = False
        self._benching = False
        self._bench_stop = threading.Event()
        lay = QVBoxLayout(self)
        self.here = QLabel()
        self.here.setWordWrap(True)
        self.here.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.here)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)
        self.note = QLabel()
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        row = QHBoxLayout()
        self.refresh_btn = QPushButton("Check workers")
        self.refresh_btn.setToolTip("Ask every known worker for its status (over Tailscale; a few seconds)")
        self.refresh_btn.clicked.connect(self.check_workers)
        self.bench_btn = QPushButton("Benchmark this computer")
        self.bench_btn.setToolTip("Measure molecules/hour for each installed engine on this computer's CPU and GPU "
                                  "(a few minutes, all cores), so a split gives it the right share")
        self.bench_btn.clicked.connect(self.benchmark_here)
        folder = QPushButton("Open registry folder")
        folder.clicked.connect(self._open_registry)
        row.addWidget(self.refresh_btn)
        row.addWidget(self.bench_btn)
        row.addWidget(folder)
        row.addStretch(1)
        lay.addLayout(row)
        self.refresh()

    # ------------------------------------------------------------------ content
    def _lane_text(self) -> str:
        from ..dist import lanes as lanes_mod
        from ..dist import profile

        s = self.win.s
        try:
            cfg = self.win.current_config()
        except Exception:
            cfg = None
        spec = s.lanes_for()
        plan = lanes_mod.plan_text(spec, cfg, getattr(self.win, "report", None), s.resources.parallel_molecules)
        return (f"<b>{profile.machine_name()}</b> (this computer). Parallel lanes <code>{spec}</code>: {plan}<br>"
                "Change it in Settings &gt; Compute; split a list across machines with "
                "<code>quantabench-cli grid build</code> (docs/WORKERS.md).")

    def refresh(self) -> None:
        """Re-read the registry (cheap: a few small JSON files) and the lane plan."""
        from ..dist import plan, profile

        try:
            self.here.setText(self._lane_text())
        except Exception as exc:
            self.here.setText(f"Lane plan unavailable ({type(exc).__name__}: {exc})")
        try:
            profiles = profile.load_profiles()
        except Exception as exc:
            profiles = []
            self.note.setText(f"Could not read the machine registry ({exc})")
        me = plan.norm_name(profile.machine_name())
        by_name = {plan.norm_name(p.name): p for p in profiles}
        names = sorted(set(by_name) | {me}, key=lambda n: (n != me, n))     # this computer always has a row
        for n in self._refs:                     # workers known only from settings or --worker
            if plan.norm_name(n) not in by_name:
                names.append(n)
        self.table.setRowCount(len(names))
        for r, n in enumerate(names):
            p = by_name.get(plan.norm_name(n))
            ref = self._refs.get(n) or next((v for k, v in self._refs.items() if plan.norm_name(k) == n), None)
            cells = [
                (p.name if p else n) + ("  (this computer)" if n == me else ""),
                self._status(ref, p, n == me),
                _hardware(p) if p else "",
                _throughput(p) if p else ("not measured" if n == me else ""),
                _age(p) if p else "",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 1 and ref is not None and ref.error:
                    item.setToolTip(ref.error)
                self.table.setItem(r, c, item)
        if not profiles and not self._refs:
            self.note.setText("No machines in the registry yet. Click 'Benchmark this computer' here on each "
                              "computer, or start `quantabench-cli worker` (it measures a new machine once), with "
                              "the settings folder synced between them.")
        elif not self.note.text().startswith("Could not"):
            when = time.strftime("%H:%M", time.localtime(self._checked_at)) if self._checked_at else ""
            self.note.setText(f"Workers last checked at {when}." if when else
                              "Status shows the registry only; 'Check workers' asks each worker.")

    def _status(self, ref, p, here: bool) -> str:
        if ref is None:
            if p is not None and p.address:
                return "worker (not checked)"
            return "this computer" if here else "no worker"
        if ref.error:
            return "unreachable"
        q = (ref.info or {}).get("queue") or {}
        running, queued = len(q.get("running", [])), len(q.get("queued", []))
        state = ("paused" if q.get("paused") else "benchmarking" if q.get("hold") else
                 "busy" if running else "idle")
        extra = f", {queued} queued" if queued else ""
        from .. import __version__

        version = (ref.info or {}).get("quantabench_version", "")
        mismatch = f", version {version} (here {__version__})" if version and version != __version__ else ""
        return f"{state}{extra}{mismatch}{_sensor_text((ref.info or {}).get('sensors'))}"

    # ------------------------------------------------------------------ actions
    def check_workers(self) -> None:
        if self._busy:
            return
        from ..dist import client
        from . import workers

        self._busy = True
        self.refresh_btn.setEnabled(False)
        self.note.setText("Checking workers...")

        def done(refs):
            self._busy = False
            self.refresh_btn.setEnabled(True)
            self._refs = dict(refs or {})
            self._checked_at = time.time()
            self.refresh()

        def failed(msg):
            self._busy = False
            self.refresh_btn.setEnabled(True)
            self.note.setText(f"Could not check the workers: {msg.splitlines()[0]}")

        workers.start(workers.FunctionWorker(client.discover, timeout=3.0), self.win, done, failed)

    def benchmark_here(self) -> None:
        """This computer's profile with fresh throughput numbers, saved to the registry (off the GUI thread;
        closing the window stops it after the current molecule, saving nothing)."""
        if self._benching:
            return
        from ..dist import profile
        from . import workers

        self._benching = True
        self._bench_stop.clear()
        self.bench_btn.setEnabled(False)
        self.note.setText("Benchmarking this computer (a few minutes; the app stays usable, but other work "
                          "here slows the measurement down)...")

        stop = self._bench_stop           # the thread holds the event, not the panel

        def run():
            prof = profile.local_profile(bench=True, should_stop=stop.is_set)
            if stop.is_set():
                return None
            profile.save_profile(prof)
            return prof

        def done(prof):
            self._benching = False
            if self._bench_stop.is_set():
                return                    # the window is closing
            self.bench_btn.setEnabled(True)
            self.refresh()
            self.note.setText("Benchmark saved: " + _throughput(prof) if prof is not None else "Benchmark stopped")

        def failed(msg):
            self._benching = False
            if not self._bench_stop.is_set():
                self.bench_btn.setEnabled(True)
                self.note.setText(f"Benchmark failed: {msg.splitlines()[0]}")

        workers.start(workers.FunctionWorker(run), self.win, done, failed)

    def stop_benchmark(self) -> None:
        self._bench_stop.set()

    def _open_registry(self) -> None:
        from ..dist import profile

        path = profile.registry_dir()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def on_visibility(self, visible: bool) -> None:
        if visible:
            self.refresh()


class _CloseWatcher(QObject):
    """Stops a running benchmark when the main window starts closing (it waits for its threads)."""

    def __init__(self, panel: MachinesPanel, win):
        super().__init__(panel)
        self.panel = panel
        win.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Close:
            self.panel.stop_benchmark()
        return False


def install(win) -> None:
    panel = MachinesPanel(win)
    panel._close_watcher = _CloseWatcher(panel, win)
    dock = QDockWidget("Machines", win)
    dock.setObjectName("machines")
    dock.setWidget(panel)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    pipeline = getattr(win, "pipeline_dock", None)
    if pipeline is not None:
        win.tabifyDockWidget(pipeline, dock)
        pipeline.raise_()
    toggle = dock.toggleViewAction()
    toggle.setText("Machines")
    toggle.setToolTip("This computer's parallel lanes and the other machines (workers)")
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == "View":
            action.menu().addAction(toggle)
            break
    dock.visibilityChanged.connect(panel.on_visibility)
    win.machines_dock = dock
    win.machines_panel = panel
