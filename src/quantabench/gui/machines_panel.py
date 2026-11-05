"""Machines dock: this computer's lanes and the machines in the registry (workers) - DIST-3, minimal.

Shows the machine registry (``dist.profile``: one JSON per machine in the synced settings folder),
each worker's live status from ``dist.client.discover`` (fetched off the GUI thread on Refresh), and
what the "Parallel lanes" setting means on this computer for the current pipeline settings
(``dist.lanes.plan_text``). Splitting a batch across machines is ``quantabench-cli grid build`` for now;
a "Run on" choice for the Run buttons is the next step (backlog DIST-3).
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QUrl
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
    return ", ".join(f"{k} {float(v):,.0f}" for k, v in items[:4]) + (" ..." if len(items) > 4 else "")


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
        folder = QPushButton("Open registry folder")
        folder.clicked.connect(self._open_registry)
        row.addWidget(self.refresh_btn)
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
        names = sorted(by_name, key=lambda n: (n != me, n))
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
                _throughput(p) if p else "",
                _age(p) if p else "",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 1 and ref is not None and ref.error:
                    item.setToolTip(ref.error)
                self.table.setItem(r, c, item)
        if not profiles and not self._refs:
            self.note.setText("No machines in the registry yet. Run `quantabench-cli machines --refresh --bench "
                              "mmff` on each computer (or start `quantabench-cli worker`), with the settings "
                              "folder synced between them.")
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
        state = "paused" if q.get("paused") else ("busy" if running else "idle")
        extra = f", {queued} queued" if queued else ""
        from .. import __version__

        version = (ref.info or {}).get("quantabench_version", "")
        mismatch = f", version {version} (here {__version__})" if version and version != __version__ else ""
        return f"{state}{extra}{mismatch}"

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

    def _open_registry(self) -> None:
        from ..dist import profile

        path = profile.registry_dir()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def on_visibility(self, visible: bool) -> None:
        if visible:
            self.refresh()


def install(win) -> None:
    panel = MachinesPanel(win)
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
