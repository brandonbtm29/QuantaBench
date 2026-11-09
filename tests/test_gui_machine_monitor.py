"""The machine monitor strip in the Pipeline dock (MON-1): this computer's and the workers' live metrics, colours by
threshold, sparklines, folding, and that it never blocks or outlives the window. Fake monitors and pollers stand in
for the sensors and the workers."""

import time

import pytest
from conftest import HAS_DISPLAY

pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=15.0):
    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def nova_samples(n=120, hot=False):
    from quantabench import sensors

    now = time.time()
    out = []
    for i in range(n):
        out.append(sensors.Sample(
            t=now - (n - i) * 2, machine="nova-test", source="psutil + Windows performance counters + "
            "LibreHardwareMonitor", cpu_percent=90.0 + (i % 5), cpu_clock_mhz=4620.0, cpu_clock_min_mhz=3850.0,
            cpu_clock_max_mhz=5490.0, cpu_base_mhz=4701.0, cpu_temp_c=(93.0 if hot else 61.0) + (i % 3),
            cpu_power_w=88.0, gpu_percent=9.0, gpu_temp_c=82.0, gpu_mem_used_mb=2048.0, gpu_mem_total_mb=16304.0,
            ram_used_gb=16.8, ram_total_gb=31.1, gpu_name="AMD Radeon RX 9060 XT", cpu_name="AMD Ryzen 9 7900X",
            sources={"cpu_temp_c": "LibreHardwareMonitor", "cpu_clock_mhz": "Windows performance counters"}))
    return out


class FakeMonitor:
    def __init__(self, samples, error=""):
        self.samples, self.error = list(samples), error
        self.started = self.stopped = 0
        self.asked_wake = []

    @property
    def running(self):
        return self.started > self.stopped

    def start(self):
        self.started += 1
        return self

    def stop(self, timeout=3.0):
        assert timeout == 0, "the GUI thread must never wait for the sensor thread"
        self.stopped += 1

    def latest(self, wake=True, wait=0.0):
        self.asked_wake.append(wake)
        return self.samples[-1] if self.samples else None

    def history(self):
        return list(self.samples)


class Ref:
    def __init__(self, name, url):
        self.name, self.url = name, url


def make_poller(replies):
    """A RemotePoller whose workers answer from ``replies`` {name: dict or exception}."""
    from quantabench.gui import machine_monitor as mm

    def workers():
        return [Ref(n, f"http://100.64.0.{i + 2}:8765") for i, n in enumerate(replies)], "token"

    def fetch(ref, token):
        r = replies[ref.name]
        if isinstance(r, Exception):
            raise r
        return r

    return mm.RemotePoller(0.2, workers, fetch, list_every_s=0.0)


def test_strip_shows_this_computer_and_workers(isolated_settings):
    from quantabench import sensors
    from quantabench.dist.client import WorkerError
    from quantabench.gui import machine_monitor as mm

    app = _app()
    mac = sensors.Sample(t=time.time(), machine="mac-test", source="psutil", cpu_percent=64.0, ram_used_gb=21.3,
                         ram_total_gb=36.0, missing={"cpu_temp_c": sensors.MACMON_INSTALL})
    poller = make_poller({"mac-test": {"sensors": mac.to_dict(), "why": ""},
                          "old-box": WorkerError("http://x/v1/sensors: HTTP 404: no such endpoint", 404),
                          "gone-box": WorkerError("cannot reach http://x: timed out")})
    poller.poll_once()
    mon = FakeMonitor(nova_samples(hot=True))
    strip = mm.MachineMonitorStrip(None, monitor=mon, poller=poller, interval_s=1.0)
    strip.resize(440, 300)
    strip.refresh()
    here = strip.here
    assert here.name == "nova-test" and here.here
    cpu = here.lines["cpu"].text()
    assert "<b>CPU</b> 94 %" in cpu and "4.62 GHz avg, 5.49 max" in cpu and "88 W" in cpu
    assert mm.HOT_COLOR in cpu and "95 °C" in cpu                       # 95 °C >= 90: red
    gpu = here.lines["gpu"].text()
    assert "9 %" in gpu and mm.WARN_COLOR in gpu and "82 °C" in gpu and "VRAM 2.0 / 15.9 GB" in gpu
    assert "16.8 / 31.1 GB (54 %)" in here.lines["ram"].text()
    tip = here.toolTip()
    assert "LibreHardwareMonitor" in tip and "Base clock 4.70 GHz" in tip and "AMD Radeon RX 9060 XT" in tip
    assert len(here.sparks["cpu_load"].points) == 120 and len(here.sparks["cpu_temp"].points) == 120
    assert "CPU utilization %" in here.sparks["cpu_load"].toolTip()
    strip.show()
    app.processEvents()
    strip.grab()                                                        # paints the sparklines
    assert set(strip.remote) == {"mac-test", "old-box", "gone-box"}
    m = strip.remote["mac-test"]
    assert "64 %" in m.lines["cpu"].text() and "21.3 / 36.0 GB" in m.lines["ram"].text()
    assert "brew install macmon" in m.title.text() and not m.lines["gpu"].isVisible()
    assert "older worker without sensors" in strip.remote["old-box"].title.text()
    assert "unreachable" in strip.remote["gone-box"].title.text()
    # folding: the body hides and the header keeps one line per machine
    strip.toggle.setChecked(False)
    assert strip.body.isHidden() and "nova-test: CPU 94 % 95 °C, GPU 9 %" in strip.summary.text()
    assert "mac-test: CPU 64 %" in strip.summary.text()
    strip.toggle.setChecked(True)
    assert strip.summary.text() == "" and strip.body.isVisible()
    strip.stop()
    assert mon.stopped >= 1 and not poller.running and not strip.timer.isActive()
    strip.close()


def test_the_clock_turns_amber_below_the_base_clock_under_load():
    from quantabench import sensors
    from quantabench.gui import machine_monitor as mm

    busy = sensors.Sample(cpu_percent=95.0, cpu_clock_mhz=3400.0, cpu_base_mhz=4701.0, cpu_clock_max_mhz=3600.0)
    assert mm.clock_level(busy) == "warn" and mm.WARN_COLOR in mm.cpu_html(busy)
    idle = sensors.Sample(cpu_percent=5.0, cpu_clock_mhz=3400.0, cpu_base_mhz=4701.0)
    assert mm.clock_level(idle) == "ok"
    assert "<b>CPU</b> -" in mm.cpu_html(sensors.Sample())
    assert mm.hint(sensors.Sample(missing={"cpu_temp_c": "x LibreHardwareMonitor y"})).startswith("temperatures: turn")


def test_strip_without_a_sample_yet_and_with_html_in_names(isolated_settings):
    from quantabench import sensors
    from quantabench.gui import machine_monitor as mm

    _app()
    mon = FakeMonitor([], error="")
    evil = sensors.Sample(t=time.time(), machine="<b>x</b>", source="<script>", cpu_percent=1.0,
                          gpu_name="<i>gpu</i>")
    poller = make_poller({"<img src=x>": {"sensors": evil.to_dict(), "why": "<b>why</b>"}})
    poller.poll_once()
    strip = mm.MachineMonitorStrip(None, monitor=mon, poller=poller, interval_s=1.0)
    strip.refresh()
    assert "starting" in strip.here.title.text() and strip.here.lines["cpu"].isHidden()
    block = strip.remote["<img src=x>"]
    assert "&lt;img src=x&gt;" in block.title.text() and "<img" not in block.title.text()
    assert "<script>" not in block.toolTip() and "&lt;b&gt;why" in block.title.text()
    mon.samples = nova_samples(3)
    strip.refresh()
    assert not strip.here.lines["cpu"].isHidden() and "4.62 GHz" in strip.here.lines["cpu"].text()
    strip.stop()


def test_remote_poller_retries_and_forgets():
    from quantabench import sensors
    from quantabench.dist.client import WorkerError

    replies = {"a": WorkerError("cannot reach", 0), "b": {"sensors": None, "why": "no sample yet"}}
    poller = make_poller(replies)
    poller.poll_once()
    snap = poller.snapshot()
    assert snap["a"].status == "unreachable" and snap["a"].next_try > time.monotonic() + 30
    assert snap["b"].status == "no sample yet" and snap["b"].sample is None
    replies["a"] = {"sensors": sensors.Sample(cpu_percent=5.0).to_dict()}
    poller.poll_once()
    assert poller.snapshot()["a"].status == "unreachable"            # not asked again before the retry time
    with poller._lock:
        poller._states["a"].next_try = 0.0
    poller.poll_once()
    assert poller.snapshot()["a"].sample.cpu_percent == 5.0 and len(poller.snapshot()["a"].history) == 1
    del replies["b"]
    poller.poll_once()
    assert set(poller.snapshot()) == {"a"}                            # a worker no longer known is dropped
    poller.start()
    assert poller.running
    poller.stop()
    assert not poller.running


def test_main_window_has_the_strip_above_the_run_buttons(tmp_path):
    from quantabench import settings as sm
    from quantabench.gui import machine_monitor as mm
    from quantabench.gui.main_window import MainWindow

    app = _app()
    w = MainWindow(sm.Settings(output_dir=str(tmp_path)))
    strip = w.machine_monitor
    assert isinstance(strip, mm.MachineMonitorStrip)
    lay = w.pipeline_dock.widget().layout()
    assert lay.indexOf(strip) == lay.count() - 2                       # just above Run all / Run selected / Cancel
    assert not strip.monitor.running                                   # nothing runs before the window is shown
    w.show()
    assert _wait(app, lambda: strip.here is not None and strip.monitor.latest(wake=False) is not None)
    assert _wait(app, lambda: "<b>CPU</b>" in strip.here.lines["cpu"].text() and "%" in strip.here.lines["cpu"].text())
    assert "RAM" in strip.here.lines["ram"].text()
    t = time.perf_counter()
    strip.refresh()
    assert time.perf_counter() - t < 0.2                               # the GUI thread only copies samples
    w.close()
    assert _wait(app, lambda: w.closed and strip.stopped)
    assert _wait(app, lambda: not strip.monitor.running)
