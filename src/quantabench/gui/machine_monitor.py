"""Machine monitor strip (MON-1): live load, effective clock, temperatures and memory while a run is set up and runs.

It sits in the Pipeline dock, above Run all / Run selected / Cancel (the page where a run is set up and started), and
shows one block per machine: this computer first, then every worker that answers ``GET /v1/sensors``:

    CPU  37 %  4.62 GHz avg, 5.49 max  61 °C  88 W     [utilization sparkline] [°C sparkline]
    GPU   9 %  36 °C  VRAM 2.0 / 15.9 GB                [utilization sparkline] [°C sparkline]
    RAM  16.8 / 31.1 GB (54 %)                         [RAM % sparkline]

Sparklines cover the last 10 minutes. Temperatures turn amber at 80 °C and red at 90 °C, memory at 85 % and 95 %, and
the clock turns amber when the CPU runs below 80 % of its base clock under load (likely throttling). The tooltip of a
block says where each number comes from and why a missing one is missing (``sensors.Sample.missing``).

Threads: this computer is sampled by a ``sensors.Monitor`` (a daemon thread; macmon is kept running on a Mac) and the
workers by ``RemotePoller`` (one daemon thread, every 5 s, 2.5 s timeout per worker; an unreachable worker is tried
again after a minute, an older one without ``/v1/sensors`` after ten). A QTimer on the GUI thread only copies their
latest samples, so nothing here can block the window. Both stop while the strip is hidden and when the window closes.
"""

from __future__ import annotations

import html
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from PySide6.QtCore import QEvent, QObject, QPointF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import sensors

WARN_COLOR = "#c77800"
HOT_COLOR = "#d03a3a"
LOAD_COLOR = QColor(64, 128, 210)
TEMP_COLOR = QColor(220, 120, 40)
MEM_COLOR = QColor(120, 90, 200)
REMOTE_INTERVAL_S = 5.0
REMOTE_TIMEOUT_S = 2.5
RETRY_UNREACHABLE_S = 60.0
RETRY_OLD_WORKER_S = 600.0
THROTTLE_LOAD_PCT = 50.0
THROTTLE_BASE_FRACTION = 0.8


def _color(text: str, lvl: str) -> str:
    text = html.escape(text)
    if lvl == "hot":
        return f'<span style="color:{HOT_COLOR}; font-weight:600">{text}</span>'
    if lvl == "warn":
        return f'<span style="color:{WARN_COLOR}; font-weight:600">{text}</span>'
    return text


def clock_level(s: sensors.Sample) -> str:
    """"warn" when the CPU runs below THROTTLE_BASE_FRACTION of its base clock while it is busy."""
    if (s.cpu_clock_mhz and s.cpu_base_mhz and (s.cpu_percent or 0) >= THROTTLE_LOAD_PCT
            and s.cpu_clock_mhz < THROTTLE_BASE_FRACTION * s.cpu_base_mhz):
        return "warn"
    return "ok"


def cpu_html(s: sensors.Sample) -> str:
    parts = [f"<b>CPU</b> {html.escape(sensors.fmt_num(s.cpu_percent, '%'))}"]
    if s.cpu_clock_mhz is not None:
        clock = f"{sensors.fmt_clock(s.cpu_clock_mhz)} avg"
        if s.cpu_clock_max_mhz is not None:
            clock += f", {s.cpu_clock_max_mhz / 1000:.2f} max"
        parts.append(_color(clock, clock_level(s)))
    if s.cpu_temp_c is not None:
        parts.append(_color(sensors.fmt_num(s.cpu_temp_c, "°C"), sensors.level("temp", s.cpu_temp_c)))
    if s.cpu_power_w is not None:
        parts.append(html.escape(sensors.fmt_num(s.cpu_power_w, "W")))
    return "&nbsp;&nbsp;".join(parts)


def gpu_html(s: sensors.Sample) -> str:
    parts = ["<b>GPU</b>"]
    if s.gpu_percent is not None:
        parts.append(html.escape(sensors.fmt_num(s.gpu_percent, "%")))
    if s.gpu_temp_c is not None:
        parts.append(_color(sensors.fmt_num(s.gpu_temp_c, "°C"), sensors.level("temp", s.gpu_temp_c)))
    if s.gpu_mem_used_mb is not None:
        total = f" / {s.gpu_mem_total_mb / 1024:.1f}" if s.gpu_mem_total_mb else ""
        parts.append(_color(f"VRAM {s.gpu_mem_used_mb / 1024:.1f}{total} GB", sensors.level("mem", s.gpu_mem_percent)))
    if len(parts) == 1:
        parts.append("-")
    return "&nbsp;&nbsp;".join(parts)


def ram_html(s: sensors.Sample) -> str:
    return f"<b>RAM</b> {_color(sensors.ram_text(s).replace('/', ' / '), sensors.level('mem', s.ram_percent))}"


def hint(s: sensors.Sample) -> str:
    """A short grey note next to the machine's name when its temperatures are missing: how to get them."""
    if s.has("cpu_temp_c"):
        return ""
    why = s.missing.get("cpu_temp_c", "")
    if "macmon" in why:
        return "temperatures and clocks: brew install macmon"
    if "LibreHardwareMonitor" in why:
        return "temperatures: turn on LibreHardwareMonitor's web server"
    if "turned off" in why:
        return "sensors off in the settings"
    if "WSL2" in why:
        return "WSL2: sensors are on the Windows side"
    return "no temperature sensor"


def tooltip(s: sensors.Sample) -> str:
    lines = [f"<b>{html.escape(s.machine or '?')}</b>: {html.escape(s.source)}"]
    if s.cpu_name:
        lines.append(html.escape(s.cpu_name))
    if s.gpu_name:
        lines.append(html.escape(s.gpu_name))
    if s.cpu_base_mhz:
        lines.append(f"Base clock {sensors.fmt_clock(s.cpu_base_mhz)}; effective clock min-max over the cores "
                     f"{sensors.fmt_clock(s.cpu_clock_min_mhz)} - {sensors.fmt_clock(s.cpu_clock_max_mhz)}")
    for key, label in sensors.METRICS:
        if s.has(key):
            src = s.sources.get(key)
            if src:
                lines.append(f"{html.escape(label)}: {html.escape(src)}")
        elif key in s.missing:
            lines.append(f"{html.escape(label)}: <i>{html.escape(s.missing[key])}</i>")
    return "<br>".join(lines)


# --------------------------------------------------------------------------------------- sparkline
class Sparkline(QWidget):
    """A small line of (time, value) points over the last ``span_s`` seconds. ``fixed``: the value range (0-100 for
    percentages); otherwise it follows the data, at least ``floor`` wide (temperatures)."""

    def __init__(self, color: QColor, fixed: tuple[float, float] | None = (0.0, 100.0), floor: float = 20.0,
                 span_s: float = sensors.HISTORY_S, parent=None):
        super().__init__(parent)
        self.color, self.fixed, self.floor, self.span_s = color, fixed, floor, span_s
        self.points: list[tuple[float, float]] = []
        self.setFixedSize(QSize(56, 18))

    def set_points(self, points: list[tuple[float, float]], label: str = "") -> None:
        self.points = [(t, v) for t, v in points if v is not None]
        if self.points:
            vals = [v for _t, v in self.points]
            self.setToolTip(f"{label}, last {self.span_s / 60:.0f} min: {min(vals):.0f}-{max(vals):.0f} "
                            f"(now {vals[-1]:.0f})" if label else "")
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QPen(self.palette().mid().color(), 1))
        p.drawLine(0, h - 1, w, h - 1)
        if len(self.points) < 2:
            p.end()
            return
        t1 = self.points[-1][0]
        t0 = t1 - self.span_s
        vals = [v for _t, v in self.points]
        if self.fixed is not None:
            lo, hi = self.fixed
        else:
            lo, hi = min(vals), max(vals)
            if hi - lo < self.floor:
                mid = (hi + lo) / 2
                lo, hi = mid - self.floor / 2, mid + self.floor / 2
        poly = QPolygonF()
        for t, v in self.points:
            if t < t0:
                continue
            x = (t - t0) / self.span_s * (w - 1)
            y = (h - 2) - (min(hi, max(lo, v)) - lo) / max(1e-9, hi - lo) * (h - 3)
            poly.append(QPointF(x, y))
        p.setPen(QPen(self.color, 1.4))
        p.drawPolyline(poly)
        p.end()


# ------------------------------------------------------------------------------------- one machine
class MachineBlock(QFrame):
    """The lines of one machine."""

    def __init__(self, name: str, here: bool, parent=None):
        super().__init__(parent)
        self.name, self.here = name, here
        grid = QGridLayout(self)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.RichText)
        grid.addWidget(self.title, 0, 0, 1, 3)
        self.lines: dict[str, QLabel] = {}
        self.sparks: dict[str, Sparkline] = {}
        for row, key in enumerate(("cpu", "gpu", "ram"), start=1):
            lab = QLabel()
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # never widens the dock
            grid.addWidget(lab, row, 0)
            self.lines[key] = lab
            load = Sparkline(MEM_COLOR if key == "ram" else LOAD_COLOR)
            grid.addWidget(load, row, 1)
            self.sparks[f"{key}_load"] = load
            if key != "ram":
                temp = Sparkline(TEMP_COLOR, fixed=None)
                grid.addWidget(temp, row, 2)
                self.sparks[f"{key}_temp"] = temp
        grid.setColumnStretch(0, 1)
        self.status = ""

    def show_sample(self, s: sensors.Sample | None, history: list[tuple[float, sensors.Sample]], status: str = "",
                    ) -> None:
        label = html.escape(self.name) + (" (this computer)" if self.here else "")
        if s is None:
            self.title.setText(f"<b>{label}</b> <span style='color:gray'>{html.escape(status or 'waiting...')}</span>")
            for lab in self.lines.values():
                lab.setText("")
            for sp in self.sparks.values():
                sp.set_points([])
            for w in (self.lines["cpu"], self.lines["gpu"], self.lines["ram"], *self.sparks.values()):
                w.setVisible(False)
            self.setToolTip(html.escape(status))
            return
        for w in (self.lines["cpu"], self.lines["ram"], self.sparks["cpu_load"], self.sparks["cpu_temp"],
                  self.sparks["ram_load"]):
            w.setVisible(True)
        note = status or hint(s)
        extra = f" <span style='color:gray'>{html.escape(note)}</span>" if note else ""
        self.title.setText(f"<b>{label}</b>{extra}")
        has_gpu = any(s.has(k) for k in sensors.GPU_KEYS) or any(
            any(h.has(k) for k in sensors.GPU_KEYS) for _t, h in history[-30:])
        self.lines["cpu"].setText(cpu_html(s))
        self.lines["gpu"].setText(gpu_html(s))
        self.lines["ram"].setText(ram_html(s))
        for key in ("gpu",):
            self.lines[key].setVisible(has_gpu)
            self.sparks[f"{key}_load"].setVisible(has_gpu)
            self.sparks[f"{key}_temp"].setVisible(has_gpu)
        series = {"cpu_load": ("cpu_percent", "CPU utilization %"), "cpu_temp": ("cpu_temp_c", "CPU °C"),
                  "gpu_load": ("gpu_percent", "GPU utilization %"), "gpu_temp": ("gpu_temp_c", "GPU °C"),
                  "ram_load": ("ram_percent", "RAM in use %")}
        for key, (attr, text) in series.items():
            self.sparks[key].set_points([(t, getattr(h, attr)) for t, h in history], text)
        tip = tooltip(s)
        self.setToolTip(tip)
        for lab in self.lines.values():
            lab.setToolTip(tip)

    def summary(self, s: sensors.Sample | None) -> str:
        if s is None:
            return f"{self.name}: -"
        out = f"{self.name}: CPU {sensors.fmt_num(s.cpu_percent, '%')}"
        if s.cpu_temp_c is not None:
            out += f" {s.cpu_temp_c:.0f} °C"
        if s.gpu_percent is not None:
            out += f", GPU {s.gpu_percent:.0f} %"
        return out


# --------------------------------------------------------------------------------------- workers
@dataclass
class RemoteState:
    name: str
    url: str = ""
    sample: sensors.Sample | None = None
    status: str = "checking..."
    history: deque = field(default_factory=lambda: deque(maxlen=int(sensors.HISTORY_S / REMOTE_INTERVAL_S) + 2))
    next_try: float = 0.0


def _default_workers() -> tuple[list, str]:
    """(worker refs other than this machine, token) from the registry and the settings; no token: none."""
    from .. import settings as settings_mod
    from ..dist import client, profile

    s = settings_mod.load()
    token = s.dist.worker_token
    if not token:
        return [], ""
    me = settings_mod.machine_key(profile.machine_name())
    return [r for r in client.known_workers().values() if settings_mod.machine_key(r.name) != me], token


def _default_fetch(ref, token: str) -> dict:
    from ..dist import client

    return client.WorkerClient(ref.url, token, timeout=REMOTE_TIMEOUT_S).sensors(timeout=REMOTE_TIMEOUT_S)


class RemotePoller:
    """Asks every known worker for its sensors every ``interval_s``, in one daemon thread. ``snapshot()`` never
    blocks. ``list_workers()`` -> (refs with .name and .url, token); ``fetch(ref, token)`` -> the /v1/sensors dict
    (raises ``WorkerError``; status 404 = a worker from before MON-1)."""

    def __init__(self, interval_s: float = REMOTE_INTERVAL_S, list_workers: Callable[[], tuple[list, str]] | None = None,
                 fetch: Callable[[object, str], dict] | None = None, list_every_s: float = 60.0):
        self.interval_s = interval_s
        self.list_workers = list_workers or _default_workers
        self.fetch = fetch or _default_fetch
        self.list_every_s = list_every_s          # the registry and settings are re-read this often, not every poll
        self._listed: tuple[float, list, str] | None = None
        self._states: dict[str, RemoteState] = {}
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._halt.is_set()

    def start(self) -> None:
        if not self.running:
            self._halt = threading.Event()          # a new one per start: a thread being stopped never runs on
            self._thread = threading.Thread(target=self._run, args=(self._halt,), name="quantabench-remote-sensors",
                                            daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Never waits (a GUI thread calls it); the thread ends after its current request."""
        self._halt.set()
        self._thread = None

    def snapshot(self) -> dict[str, RemoteState]:
        with self._lock:
            return {k: RemoteState(v.name, v.url, v.sample, v.status, deque(v.history), v.next_try)
                    for k, v in self._states.items()}

    def poll_once(self, halt: threading.Event | None = None) -> None:
        halt = halt or self._halt
        if self._listed is None or time.monotonic() - self._listed[0] >= self.list_every_s:
            try:
                refs, token = self.list_workers()
            except Exception:  # noqa: BLE001
                refs, token = [], ""
            self._listed = (time.monotonic(), list(refs), token)
        _t, refs, token = self._listed
        names = set()
        for ref in refs:
            if halt.is_set():
                return
            names.add(ref.name)
            with self._lock:
                st = self._states.setdefault(ref.name, RemoteState(ref.name, ref.url))
                due = time.monotonic() >= st.next_try
            if not due:
                continue
            sample, status, retry = None, "", 0.0
            try:
                data = self.fetch(ref, token)
                raw = data.get("sensors") if isinstance(data, dict) else None
                if raw is None:
                    status = str((data or {}).get("why") or "no sample yet")[:200]
                else:
                    sample = sensors.Sample.from_dict(raw)
                    status = str(data.get("why") or "")[:200]
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "status", 0)
                if code == 404:
                    status, retry = "older worker without sensors (update it)", RETRY_OLD_WORKER_S
                else:
                    status, retry = "unreachable", RETRY_UNREACHABLE_S
            with self._lock:
                st.url = ref.url
                st.status = status
                st.next_try = time.monotonic() + retry
                if sample is not None:
                    st.sample = sample
                    st.history.append((time.time(), sample))
                elif retry:
                    st.sample = None
        with self._lock:
            for gone in set(self._states) - names:
                self._states.pop(gone, None)

    def _run(self, halt: threading.Event) -> None:
        while not halt.is_set():
            t0 = time.monotonic()
            self.poll_once(halt)
            halt.wait(max(0.5, self.interval_s - (time.monotonic() - t0)))


# ------------------------------------------------------------------------------------------ strip
class MachineMonitorStrip(QWidget):
    """The strip: a header (click to fold it to one line) and one ``MachineBlock`` per machine."""

    def __init__(self, parent=None, monitor: sensors.Monitor | None = None, poller: RemotePoller | None = None,
                 interval_s: float | None = None, settings_key: str = "monitor/collapsed"):
        super().__init__(parent)
        cfg = sensors.load_config()
        self.interval_s = max(0.5, float(interval_s or cfg.interval_s))
        self.monitor = monitor or sensors.Monitor(interval_s=self.interval_s, config=cfg)
        self.poller = poller if poller is not None else RemotePoller(max(REMOTE_INTERVAL_S, self.interval_s))
        self.settings_key = settings_key
        self.stopped = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(1)
        head = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setText("Machine monitor")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle.setAutoRaise(True)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolTip("Live CPU and GPU load, effective clock, temperatures and memory of this computer and "
                               "the workers (quantabench-cli monitor; docs/ENGINES.md, Live machine metrics)")
        self.toggle.toggled.connect(self._set_open)
        self.summary = QLabel()
        self.summary.setStyleSheet("color: gray;")
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        head.addWidget(self.toggle)
        head.addWidget(self.summary, 1)
        lay.addLayout(head)
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(4, 0, 0, 0)
        self.body_lay.setSpacing(2)
        lay.addWidget(self.body)
        self.here: MachineBlock | None = None
        self.remote: dict[str, MachineBlock] = {}
        self.timer = QTimer(self)
        self.timer.setInterval(int(min(self.interval_s, 2.0) * 1000))
        self.timer.timeout.connect(self.refresh)
        self._restore_open()

    # ------------------------------------------------------------------ state
    def _restore_open(self) -> None:
        try:
            from .main_window import _window_settings

            value = _window_settings().value(self.settings_key)
        except Exception:  # noqa: BLE001
            value = None
        if value is not None and str(value).lower() in ("true", "1"):
            self.toggle.setChecked(False)

    def _set_open(self, on: bool) -> None:
        self.body.setVisible(on)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        try:
            from .main_window import _window_settings

            _window_settings().setValue(self.settings_key, not on)
        except Exception:  # noqa: BLE001, S110
            pass
        self.refresh()

    # ------------------------------------------------------------------ sampling
    def start(self) -> None:
        if self.stopped:
            return
        self.monitor.start()
        if self.poller is not None:
            self.poller.start()
        if not self.timer.isActive():
            self.timer.start()
        QTimer.singleShot(300, self.refresh)

    def pause(self) -> None:
        self.timer.stop()
        self.monitor.stop(timeout=0)      # never waits on the GUI thread
        if self.poller is not None:
            self.poller.stop()

    def stop(self) -> None:
        """For good (the window closes): threads and macmon stop, timers too."""
        self.stopped = True
        self.pause()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self.pause()

    def refresh(self) -> None:
        """Copy the latest samples into the blocks (GUI thread; never blocks)."""
        try:
            s = self.monitor.latest(wake=False)
            hist = [(h.t, h) for h in self.monitor.history()]
            if self.here is None:
                name = s.machine if s is not None and s.machine else _machine_name()
                self.here = MachineBlock(name, True)
                self.body_lay.insertWidget(0, self.here)
            status = "" if s is not None else (self.monitor.error or "starting...")
            if s is not None and s.t and time.time() - s.t > max(10.0, 5 * self.interval_s):
                status = "stale"
            self.here.show_sample(s, hist, status)
            parts = [self.here.summary(s)]
            remote = self.poller.snapshot() if self.poller is not None else {}
            for name, st in sorted(remote.items()):
                block = self.remote.get(name)
                if block is None:
                    block = self.remote[name] = MachineBlock(name, False)
                    self.body_lay.addWidget(block)
                block.show_sample(st.sample, list(st.history), st.status)
                parts.append(block.summary(st.sample) if st.sample is not None else f"{name}: {st.status}")
            for name in list(self.remote):
                if name not in remote:
                    self.remote.pop(name).deleteLater()
            self.summary.setText("" if self.toggle.isChecked() else "  |  ".join(parts))
        except RuntimeError:          # widgets deleted while closing
            pass


def _machine_name() -> str:
    try:
        from ..dist import profile

        return profile.machine_name()
    except Exception:  # noqa: BLE001
        return "this computer"


class _CloseWatcher(QObject):
    """Stops the strip once the main window has really closed (a close it ignores, to finish a build, does not)."""

    def __init__(self, strip: MachineMonitorStrip, win):
        super().__init__(strip)
        self.strip, self.win = strip, win
        win.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Close:
            QTimer.singleShot(0, self._check)
        return False

    def _check(self) -> None:
        try:
            if getattr(self.win, "closed", False) or not self.win.isVisible():
                self.strip.stop()
        except RuntimeError:
            pass


def install(win) -> None:
    """Put the strip into the Pipeline dock, just above its Run buttons."""
    dock = getattr(win, "pipeline_dock", None)
    if dock is None or dock.widget() is None or dock.widget().layout() is None:
        raise RuntimeError("no Pipeline dock to put the machine monitor in")
    lay = dock.widget().layout()
    strip = MachineMonitorStrip(dock.widget())
    lay.insertWidget(max(0, lay.count() - 1), strip)
    strip._close_watcher = _CloseWatcher(strip, win)
    win.machine_monitor = strip
