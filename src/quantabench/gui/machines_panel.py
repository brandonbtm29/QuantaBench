"""Machines dock: this computer's lanes and the machines in the registry (workers) - DIST-3, and machine control
(DIST-7).

Shows the machine registry (``dist.profile``: one JSON per machine in the synced settings folder), each worker's
live status from ``dist.client.discover`` (fetched off the GUI thread on "Check workers"), and what the "Parallel
lanes" setting means on this computer for the current pipeline settings (``dist.lanes.plan_text``).

Per machine (DIST-7, docs/WORKERS.md "Pausing a machine, a closing laptop"):

- **Split**: Auto (measured speed), Manual weight or Off: its part of ``grid build --weights auto``
  (``Settings.machine_control_for``), and the **Share** of the current molecule list it would get.
- **Pause / Resume**: this computer writes its pause file (its worker, a grid build's local share and this window's
  own batch follow it); another machine is paused through its worker (``POST /v1/pause``). "Pause now" also stops
  the molecules in progress at their next checkpoint. A grid that follows a paused machine moves its unstarted
  molecules to the others.
- **Processing budget** of the selected machine: run priority, free threads (or a max CPU %), lanes and pause on
  battery, saved for that machine only (``settings.update_machine``: a fresh read and write). Its worker reads them
  within about 15 s once the settings have synced (the priority is applied to running lanes at once).

Splitting a batch across machines is ``quantabench-cli grid build``; a "Run on" choice for the Run buttons is the next
step (backlog DIST-3).
"""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QEvent, QObject, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

COLUMNS = ("Machine", "Status", "Split", "Share", "Run", "Hardware", "Throughput (molecules/h)", "Updated")
COL = {name: k for k, name in enumerate(COLUMNS)}
SPLIT_LABELS = {"auto": "Auto (measured speed)", "manual": "Manual weight", "off": "Off"}
PRIORITY_CHOICES = ("default", "normal", "low", "background")


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
        self._statuses: dict = {}
        self._checked_at = 0.0
        self._busy = False
        self._benching = False
        self._bench_stop = threading.Event()
        self._names: list[str] = []          # machine keys of the table's rows, in order
        self._filling = False
        self.shares: dict[str, int] = {}     # machine key -> molecules of the current list in a grid split
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
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(COL["Hardware"], QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        lay.addWidget(self.table, 1)
        lay.addWidget(self._budget_box())
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

    # ------------------------------------------------------------------ the budget box (selected machine)
    def _budget_box(self) -> QGroupBox:
        box = QGroupBox("Processing budget")
        self.budget_box = box
        form = QFormLayout(box)
        self.budget_machine = QLabel("Select a machine")
        form.addRow("Machine", self.budget_machine)
        self.priority = QComboBox()
        self.priority.addItems(PRIORITY_CHOICES)
        self.priority.setToolTip("CPU and I/O priority of its runs (docs/ENGINES.md, Run priority and free threads); "
                                 "'default' = the shared default. A running worker lowers its lanes at once.")
        form.addRow("Run priority", self.priority)
        self.free_threads = QSpinBox()
        self.free_threads.setRange(-1, 512)
        self.free_threads.setSpecialValueText("default")
        self.free_threads.setToolTip("Logical threads its automatic budgets leave free (the lanes of its next job)")
        form.addRow("Free threads", self.free_threads)
        self.max_cpu = QSpinBox()
        self.max_cpu.setRange(0, 100)
        self.max_cpu.setSuffix(" %")
        self.max_cpu.setSpecialValueText("not set")
        self.max_cpu.setToolTip("About this share of its threads: sets the free threads from its thread count")
        self.max_cpu.valueChanged.connect(self._max_cpu_changed)
        form.addRow("Max CPU", self.max_cpu)
        self.lanes = QLineEdit()
        self.lanes.setPlaceholderText("default (the shared lane spec)")
        self.lanes.setToolTip("Its lane spec: 1, auto, N or gpu:1x4,cpu:5x2 (its next job)")
        form.addRow("Lanes", self.lanes)
        self.pause_on_battery = QCheckBox("Pause its worker while it runs on battery")
        form.addRow("", self.pause_on_battery)
        buttons = QHBoxLayout()
        self.apply_btn = QPushButton("Save for this machine")
        self.apply_btn.clicked.connect(self.apply_budget)
        buttons.addWidget(self.apply_btn)
        buttons.addStretch(1)
        form.addRow(buttons)
        box.setEnabled(False)
        return box

    def _logical(self, key: str) -> int | None:
        from ..dist import plan, profile

        try:
            p = next((x for x in profile.load_profiles() if plan.norm_name(x.name) == key), None)
        except Exception:
            p = None
        if p is not None and p.logical_cores:
            return int(p.logical_cores)
        if key == plan.norm_name(profile.machine_name()):
            from .. import resources

            return int(resources.logical_cores())
        return None

    def _max_cpu_changed(self, value: int) -> None:
        if self._filling or not value:
            return
        from ..dist import control

        logical = self._logical(self.selected() or "")
        if logical:
            self.free_threads.setValue(control.free_threads_for_max_cpu(value, logical))

    def selected(self) -> str | None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return None
        r = min(rows)
        return self._names[r] if 0 <= r < len(self._names) else None

    def select(self, key: str) -> None:
        if key in self._names:
            self.table.selectRow(self._names.index(key))

    def _selection_changed(self) -> None:
        from .. import settings as settings_mod

        key = self.selected()
        self.budget_box.setEnabled(key is not None)
        if key is None:
            self.budget_machine.setText("Select a machine")
            return
        s = settings_mod.load()
        before, self._filling = self._filling, True
        try:
            self.budget_machine.setText(key)
            level, scope = s.priority_for(key, scope=True)
            self.priority.setCurrentText(level if scope == "this machine" else "default")
            free, fscope = s.free_threads_for(key, scope=True)
            self.free_threads.setValue(free if fscope == "this machine" else -1)
            c = s.machine_control_for(key)
            self.max_cpu.setValue(int(c["max_cpu"]))
            own = next((v for k, v in s.dist.machine_lanes.items() if settings_mod.machine_key(k) == key), "")
            self.lanes.setText(str(own or ""))
            self.pause_on_battery.setChecked(bool(c["pause_on_battery"]))
        finally:
            self._filling = before

    def apply_budget(self) -> None:
        """Save the budget box for the selected machine only (``settings.update_machine``)."""
        from .. import settings as settings_mod
        from ..dist import lanes as lanes_mod

        key = self.selected()
        if key is None:
            return
        spec = self.lanes.text().strip()
        err = lanes_mod.spec_error(spec) if spec else ""
        if err:
            self.note.setText(f"Lanes {spec!r}: {err}")
            return
        level = self.priority.currentText()
        values = {"priority": None if level == "default" else level,
                  "free_threads": None if self.free_threads.value() < 0 else self.free_threads.value(),
                  "lanes": spec, "pause_on_battery": self.pause_on_battery.isChecked(),
                  "max_cpu": float(self.max_cpu.value()) or None}
        try:
            settings_mod.update_machine(key, **values)
        except (OSError, ValueError) as exc:
            self.note.setText(f"Could not save {key}'s budget: {exc}")
            return
        self._sync_window_settings()
        self.note.setText(f"Saved for {key}: priority {level}, free threads "
                          f"{'default' if values['free_threads'] is None else values['free_threads']}, lanes "
                          f"{spec or 'default'}. Its worker reads them within about 15 s once the settings have "
                          "synced (the priority at once, lanes and free threads with its next job).")
        self.refresh()

    def _sync_window_settings(self) -> None:
        """The main window keeps its own Settings; take the per-machine tables just written into it."""
        from .. import settings as settings_mod

        s = getattr(self.win, "s", None)
        if s is None:
            return
        fresh = settings_mod.load()
        for sec, name in settings_mod.PER_MACHINE_TABLES:
            setattr(getattr(s, sec), name, getattr(getattr(fresh, sec), name))
        loaded = getattr(fresh, settings_mod._AS_LOADED, None)
        if loaded is not None:
            setattr(s, settings_mod._AS_LOADED, loaded)
            base = getattr(s, settings_mod._ALL_AS_LOADED, None)
            if isinstance(base, dict):
                for sec, name in settings_mod.PER_MACHINE_TABLES:
                    base.setdefault(sec, {})[name] = getattr(getattr(fresh, sec), name)

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
                "Change it in Settings &gt; Compute or below; split a list across machines with "
                "<code>quantabench-cli grid build</code> (docs/WORKERS.md).")

    def _current_ids(self) -> list[str]:
        model = getattr(self.win, "model", None)
        try:
            return list(model.ids()) if model is not None else []
        except Exception:
            return []

    def _compute_shares(self, names: list[str], here: str) -> str:
        """Molecules of the current list each machine would get from ``grid build --weights auto`` (the split
        settings; machines with a worker address and this computer). Returns a note ('' when all is well)."""
        from .. import settings as settings_mod
        from ..dist import client, plan, profile

        self.shares = {}
        ids = self._current_ids()
        if not ids:
            return ""
        s = settings_mod.load()
        try:
            by_name = {plan.norm_name(p.name): p for p in profile.load_profiles()}
        except Exception:
            by_name = {}
        targets = [n for n in names if n == here or (by_name.get(n) is not None and by_name[n].address)
                   or n in self._refs]
        targets = [n for n in targets if s.machine_control_for(n)["split"] != "off"]
        if not targets:
            return "Every machine's split is off."
        try:
            cfg = self.win.current_config()
            method = cfg.method
        except Exception:
            cfg, method = None, ""
        if not method or method == "auto":
            return "Choose a refinement method (not auto) to see each machine's share of the list."
        grid_names = [client.LOCAL if n == here else n for n in targets]
        try:
            w = client.resolve_weights("auto", grid_names, profile.bench_method(method, getattr(cfg, "solvent", "")),
                                       self._refs, config=client.config_for_workers(cfg) if cfg else None,
                                       output_dir=s.output_path())
        except Exception as exc:  # a machine without a profile, or no positive weight
            return f"Share not available ({exc})"
        split = plan.split(ids, w)
        for n, g in zip(targets, grid_names, strict=True):
            self.shares[n] = len(split.get(g, []))
        return ""

    def refresh(self) -> None:
        """Re-read the registry (cheap: a few small JSON files), the split settings and the lane plan."""
        from .. import settings as settings_mod
        from ..dist import control, plan, profile

        try:
            self.here.setText(self._lane_text())
        except Exception as exc:
            self.here.setText(f"Lane plan unavailable ({type(exc).__name__}: {exc})")
        try:
            profiles = profile.load_profiles()
        except Exception as exc:
            profiles = []
            self.note.setText(f"Could not read the machine registry ({exc})")
        try:
            self._statuses = control.load_statuses()
        except Exception:
            self._statuses = {}
        me = plan.norm_name(profile.machine_name())
        by_name = {plan.norm_name(p.name): p for p in profiles}
        names = sorted(set(by_name) | {me}, key=lambda n: (n != me, n))     # this computer always has a row
        for n in self._refs:                     # workers known only from settings or --worker
            if plan.norm_name(n) not in by_name and plan.norm_name(n) not in names:
                names.append(plan.norm_name(n))
        selected = self.selected()
        share_note = self._compute_shares(names, me)
        s = settings_mod.load()
        total = len(self._current_ids())
        self._filling = True
        try:
            self._names = names
            self.table.setRowCount(len(names))
            for r, n in enumerate(names):
                p = by_name.get(n)
                ref = self._refs.get(n) or next((v for k, v in self._refs.items() if plan.norm_name(k) == n), None)
                status = self._status(ref, p, n == me, n)
                share = self.shares.get(n)
                cells = {
                    "Machine": (p.name if p else n) + ("  (this computer)" if n == me else ""),
                    "Status": status,
                    "Split": SPLIT_LABELS[s.machine_control_for(n)["split"]],
                    "Share": "" if share is None else (f"{share} of {total} ({100 * share / total:.0f} %)"
                                                       if total else "0"),
                    "Run": "",
                    "Hardware": _hardware(p) if p else "",
                    "Throughput (molecules/h)": _throughput(p) if p else ("not measured" if n == me else ""),
                    "Updated": _age(p) if p else "",
                }
                for c, name in enumerate(COLUMNS):
                    item = QTableWidgetItem(cells[name])
                    if name == "Status" and ref is not None and ref.error:
                        item.setToolTip(ref.error)
                    self.table.setItem(r, c, item)
                self.table.setCellWidget(r, COL["Split"], self._split_widget(n, s))
                self.table.setCellWidget(r, COL["Run"], self._pause_widget(n, n == me, ref, status))
        finally:
            self._filling = False
        if selected in names:
            self.select(selected)
        if share_note:
            self.note.setText(share_note)
        elif not profiles and not self._refs:
            self.note.setText("No machines in the registry yet. Click 'Benchmark this computer' here on each "
                              "computer, or start `quantabench-cli worker` (it measures a new machine once), with "
                              "the settings folder synced between them.")
        elif not self.note.text().startswith(("Could not", "Saved", "Paused", "Resumed", "Lanes")):
            when = time.strftime("%H:%M", time.localtime(self._checked_at)) if self._checked_at else ""
            self.note.setText(f"Workers last checked at {when}." if when else
                              "Status shows the registry only; 'Check workers' asks each worker.")

    def _split_widget(self, key: str, s) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        c = s.machine_control_for(key)
        combo = QComboBox()
        for mode, label in SPLIT_LABELS.items():
            combo.addItem(label, mode)
        combo.setCurrentIndex(list(SPLIT_LABELS).index(c["split"]))
        combo.setToolTip("Its part of a grid split (grid build --weights auto): Auto = its measured speed (the last grid "
                         "of this protocol, else its benchmark), Manual = the weight beside it (1 = an average Auto "
                         "machine), Off = left out")
        weight = QDoubleSpinBox()
        weight.setRange(0.05, 1000.0)
        weight.setDecimals(2)
        weight.setSingleStep(0.25)
        weight.setValue(float(c["weight"]))
        weight.setEnabled(c["split"] == "manual")
        weight.setToolTip("Manual weight (1 = an average Auto machine)")
        combo.currentIndexChanged.connect(lambda _i, k=key, cb=combo, w=weight: self.set_split(k, cb.currentData(),
                                                                                                 w.value()))
        weight.editingFinished.connect(lambda k=key, cb=combo, w=weight: self.set_split(k, cb.currentData(),
                                                                                         w.value()))
        lay.addWidget(combo)
        lay.addWidget(weight)
        box.combo, box.weight = combo, weight
        return box

    def set_split(self, key: str, mode: str, weight: float | None = None) -> None:
        """Save ``key``'s split choice (and manual weight) and show the new shares."""
        from .. import settings as settings_mod

        if self._filling:
            return
        values = {"split": mode}
        if mode == "manual" and weight:
            values["weight"] = float(weight)
        try:
            settings_mod.update_machine(key, **values)
        except (OSError, ValueError) as exc:
            self.note.setText(f"Could not save {key}'s split: {exc}")
            return
        self._sync_window_settings()
        self.refresh()

    def _pause_widget(self, key: str, here: bool, ref, status: str) -> QWidget:
        paused = status.startswith("paused")
        btn = QToolButton()
        btn.setText("Resume" if paused else "Pause")
        btn.setToolTip("Resume: it takes molecules again" if paused else
                       "Pause: it takes no new molecules; the ones in progress finish. A grid that follows it gives "
                       "its unstarted molecules to the other machines. The arrow: Pause now (the molecules in "
                       "progress stop at their next checkpoint and are handed back).")
        reachable = here or (ref is not None and not ref.error)
        btn.setEnabled(reachable)
        if not paused:
            menu = QMenu(btn)
            now = QAction("Pause now (stop the running molecules)", menu)
            now.triggered.connect(lambda _c=False, k=key: self.pause(k, True, "now"))
            menu.addAction(now)
            btn.setMenu(menu)
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        btn.clicked.connect(lambda _c=False, k=key, p=paused: self.pause(k, not p, "drain"))
        return btn

    def _status(self, ref, p, here: bool, key: str = "") -> str:
        from ..dist import control

        if here:
            st = control.read_pause()
            if st["paused"]:
                return f"paused ({st['mode']})" + (", on battery" if st["reason"] == "battery" else "")
        if ref is None:
            last = self._statuses.get(key) or {}
            if last.get("paused"):
                return "paused (last said)"
            if p is not None and p.address:
                return "worker (not checked)"
            return "this computer" if here else "no worker"
        if ref.error:
            last = self._statuses.get(key) or {}
            return "unreachable" + (", paused (last said)" if last.get("paused") else
                                    (f", last said {last.get('updated', '')[:16]}" if last.get("updated") else ""))
        q = (ref.info or {}).get("queue") or {}
        running, queued = len(q.get("running", [])), len(q.get("queued", []))
        pause = q.get("pause") or {}
        state = (f"paused ({pause.get('mode', 'drain')})" if q.get("paused") else "benchmarking" if q.get("hold") else
                 "busy" if running else "idle")
        extra = f", {queued} queued" if queued else ""
        if q.get("rejoin"):
            extra += ", waiting for its grid client after a wake"
        from .. import __version__

        version = (ref.info or {}).get("quantabench_version", "")
        mismatch = f", version {version} (here {__version__})" if version and version != __version__ else ""
        return f"{state}{extra}{mismatch}{_sensor_text((ref.info or {}).get('sensors'))}"

    # ------------------------------------------------------------------ actions
    def pause(self, key: str, paused: bool, mode: str = "drain") -> None:
        """Pause or resume ``key``: this computer through its pause file (and its worker if one runs), another
        machine through its worker (off the GUI thread)."""
        from .. import settings as settings_mod
        from ..dist import client, control, plan, profile
        from . import workers

        me = plan.norm_name(profile.machine_name())
        if key == me:
            try:
                control.write_pause(None, paused, mode, by=f"{me} (Machines dock)")
            except OSError as exc:
                self.note.setText(f"Could not save the pause state ({exc})")
                return
            self.note.setText(("Paused this computer" + (" now: the molecules in progress stop at their next "
                                                         "checkpoint" if mode == "now" else
                                                         ": the molecules in progress finish")
                               if paused else "Resumed this computer")
                              + ". Its worker, a grid build's local share and this window's batch follow it.")
        ref = self._refs.get(key) or next((v for k, v in self._refs.items() if plan.norm_name(k) == key), None)
        if ref is None:
            try:
                ref = next((r for n, r in client.known_workers().items() if plan.norm_name(n) == key), None)
            except Exception:
                ref = None
        if ref is None:
            self.refresh()
            return
        token = settings_mod.load().dist.worker_token

        def call():
            return client.WorkerClient(ref.url, token, timeout=10.0).pause(paused, mode, by=f"{me} (Machines dock)")

        def done(q):
            if key != me:
                self.note.setText(f"{'Paused' if q.get('paused') else 'Resumed'} {key}"
                                  + (f" ({mode})" if q.get("paused") else "")
                                  + ". A grid that follows it moves its unstarted molecules to the other machines.")
            self.check_workers()

        def failed(msg):
            if key != me:
                self.note.setText(f"Could not reach {key}: {msg.splitlines()[0]} (a grid that follows it treats it as "
                                  "away after a while)")
            self.refresh()

        workers.start(workers.FunctionWorker(call), self.win, done, failed)

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
            self.note.setText("")
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
    toggle.setToolTip("This computer's parallel lanes, the other machines (workers), their split, budget and pause")
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == "View":
            action.menu().addAction(toggle)
            break
    dock.visibilityChanged.connect(panel.on_visibility)
    win.machines_dock = dock
    win.machines_panel = panel
