"""Live machine metrics (quantabench.sensors, MON-1): the providers' parsers on synthetic fixtures (an LHM data.json
tree, macmon JSON lines, a fake /sys, mocked PDH values), the never-raise rules, the per-machine settings, the
worker's /v1/sensors, and quantabench-cli monitor. Nothing here needs real hardware sensors."""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import threading
import time
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from quantabench import sensors
from quantabench import settings as settings_mod

POSIX = os.name != "nt"


# ----------------------------------------------------------------------------------------- fixtures
def _sensor(sid, text, kind, value, unit):
    return {"id": 0, "Text": text, "Min": "", "Value": f"{value} {unit}", "Max": "", "SensorId": sid, "Type": kind,
            "RawValue": f"{value} {unit}", "ImageURL": "images/transparent.png", "Children": []}


def _group(text, children):
    return {"id": 0, "Text": text, "Min": "", "Value": "", "Max": "", "ImageURL": "images_icon/x.png",
            "Children": children}


def _hw(hw_id, text, image, groups):
    return {"id": 0, "Text": text, "Min": "", "Value": "", "Max": "", "HardwareId": hw_id,
            "ImageURL": f"images_icon/{image}", "Children": groups}


def lhm_tree(cpu_temp="48.1", fans=(0, 0), decimal="."):
    """A data.json tree shaped like LibreHardwareMonitor 0.9.x's (a 12-core Ryzen, two GPUs, a Super I/O chip)."""
    def d(x) -> str:
        return str(x).replace(".", decimal)

    cores = [_sensor(f"/amdcpu/0/clock/{2 * i + 6}", f"Core #{i + 1}", "Clock", d(f"{4000 + 100 * i}.0"), "MHz")
             for i in range(12)]
    effective = [_sensor(f"/amdcpu/0/clock/{2 * i + 7}", f"Core #{i + 1} (Effective)", "Clock", d("120.5"), "MHz")
                 for i in range(12)]
    cpu = _hw("/amdcpu/0", "AMD Ryzen 9 7900X", "cpu.png", [
        _group("Powers", [_sensor("/amdcpu/0/power/0", "Package", "Power", d("64.2"), "W"),
                          _sensor("/amdcpu/0/power/1", "CPU PPT", "Power", d("62.8"), "W")]),
        _group("Clocks", [_sensor("/amdcpu/0/clock/0", "Bus Speed", "Clock", d("99.8"), "MHz"),
                          _sensor("/amdcpu/0/clock/1", "Cores (Average)", "Clock", d("4618.0"), "MHz"),
                          *cores, *effective]),
        _group("Temperatures", [_sensor("/amdcpu/0/temperature/3", "Package", "Temperature", d("41.3"), "°C"),
                                _sensor("/amdcpu/0/temperature/2", "Core (Tctl/Tdie)", "Temperature", d(cpu_temp),
                                        "°C"),
                                _sensor("/amdcpu/0/temperature/9", "CCDs Max (Tdie)", "Temperature", d("38.5"),
                                        "°C")]),
        _group("Load", [_sensor("/amdcpu/0/load/0", "CPU Total", "Load", d("7.3"), "%")]),
    ])
    small_gpu = _hw("/gpu-amd/0", "AMD Radeon(TM) Graphics", "ati.png", [
        _group("Load", [_sensor("/gpu-amd/0/load/0", "GPU Core", "Load", d("50.0"), "%")]),
        _group("Data", [_sensor("/gpu-amd/0/smalldata/2", "GPU Memory Total", "SmallData", d("512.0"), "MB")]),
    ])
    gpu = _hw("/gpu-amd/2", "AMD Radeon RX 9060 XT", "ati.png", [
        _group("Powers", [_sensor("/gpu-amd/2/power/3", "GPU Package", "Power", d("15.0"), "W")]),
        _group("Clocks", [_sensor("/gpu-amd/2/clock/0", "GPU Core", "Clock", d("837.0"), "MHz"),
                          _sensor("/gpu-amd/2/clock/2", "GPU Memory", "Clock", d("41.0"), "MHz")]),
        _group("Temperatures", [_sensor("/gpu-amd/2/temperature/0", "GPU Core", "Temperature", d("36.0"), "°C"),
                                _sensor("/gpu-amd/2/temperature/7", "GPU Hot Spot", "Temperature", d("37.0"), "°C")]),
        _group("Load", [_sensor("/gpu-amd/2/load/0", "GPU Core", "Load", d("9.0"), "%"),
                        _sensor("/gpu-amd/2/load/2", "D3D 3D", "Load", d("0.7"), "%")]),
        _group("Fans", [_sensor("/gpu-amd/2/fan/0", "GPU Fan", "Fan", str(fans[1]), "RPM")]),
        _group("Data", [_sensor("/gpu-amd/2/smalldata/0", "GPU Memory Used", "SmallData", d("2045.0"), "MB"),
                        _sensor("/gpu-amd/2/smalldata/2", "GPU Memory Total", "SmallData", d("16304.0"), "MB")]),
    ])
    board = _hw("/motherboard", "ASUS B650E", "mainboard.png", [
        _hw("/lpc/nct6701d/0", "Nuvoton NCT6701D", "chip.png", [
            _group("Fans", [_sensor("/lpc/nct6701d/0/fan/0", "Fan #1", "Fan", str(fans[0]), "RPM")]),
            _group("Temperatures", [_sensor("/lpc/nct6701d/0/temperature/1", "Temperature #1", "Temperature",
                                            d("36.0"), "°C")])])])
    ram = _hw("/ram", "Total Memory", "ram.png", [
        _group("Data", [_sensor("/ram/data/0", "Memory Used", "Data", d("16.8"), "GB")])])
    return {"id": 0, "Text": "Sensor", "Min": "Min", "Value": "Value", "Max": "Max", "ImageURL": "", "Children": [
        {"id": 1, "Text": "BTM-NOVA", "Min": "", "Value": "", "Max": "", "ImageURL": "images_icon/computer.png",
         "Children": [board, cpu, ram, small_gpu, gpu]}]}


def old_lhm_tree():
    """An older tree: no HardwareId, SensorId or Type; hardware only by icon, sensor types by their group."""
    def leaf(text, value):
        return {"Text": text, "Value": value, "Min": "", "Max": "", "ImageURL": "images/transparent.png",
                "Children": []}

    cpu = {"Text": "Intel Core i7-8700K", "ImageURL": "images_icon/cpu.png", "Value": "", "Children": [
        {"Text": "Temperatures", "ImageURL": "images_icon/temperature.png", "Value": "", "Children": [
            leaf("CPU Core #1", "60.0 °C"), leaf("CPU Package", "186.8 °F"),
            leaf("CPU Core #1 Distance to TjMax", "40.0 °C")]},
        {"Text": "Powers", "ImageURL": "images_icon/power.png", "Value": "", "Children": [
            leaf("CPU Package", "95,5 W")]},
        {"Text": "Clocks", "ImageURL": "images_icon/clock.png", "Value": "", "Children": [
            leaf("CPU Core #1", "4700 MHz"), leaf("CPU Core #2", "4500 MHz")]}]}
    gpu = {"Text": "NVIDIA GeForce RTX 3080", "ImageURL": "images_icon/nvidia.png", "Value": "", "Children": [
        {"Text": "Load", "ImageURL": "images_icon/load.png", "Value": "", "Children": [leaf("GPU Core", "88.0 %")]},
        {"Text": "Temperatures", "ImageURL": "images_icon/temperature.png", "Value": "", "Children": [
            leaf("GPU Core", "71.0 °C")]},
        {"Text": "Powers", "ImageURL": "images_icon/power.png", "Value": "", "Children": [
            leaf("GPU Power", "310.2 W")]},
        {"Text": "Data", "ImageURL": "images_icon/data.png", "Value": "", "Children": [
            leaf("GPU Memory Used", "8.5 GB"), leaf("GPU Memory Total", "10.0 GB")]}]}
    return {"Text": "Sensor", "Value": "Value", "Children": [
        {"Text": "PC", "ImageURL": "images_icon/computer.png", "Value": "", "Children": [cpu, gpu]}]}


#: a ``macmon pipe`` line as macmon 0.3-0.6 prints it (README format: cluster usage as [MHz, fraction])
MACMON_TUPLES = {"temp": {"cpu_temp_avg": 61.73, "gpu_temp_avg": 48.95},
                 "memory": {"ram_total": 38654705664, "ram_usage": 22011707392, "swap_total": 0, "swap_usage": 0},
                 "ecpu_usage": [2064, 0.41], "pcpu_usage": [3624, 0.87], "gpu_usage": [1398, 0.255],
                 "cpu_power": 31.5, "gpu_power": 6.25, "ane_power": 0.0, "all_power": 38.1, "sys_power": 52.3,
                 "ram_power": 1.2, "gpu_ram_power": 0.1}
#: a newer layout: separate *_freq_mhz fields and per-core lists
MACMON_FIELDS = {"timestamp": "2026-11-09T05:00:00Z", "temp": {"cpu_temp_avg": 0.0, "gpu_temp_avg": 44.0},
                 "memory": {"ram_total": 38654705664, "ram_usage": 19327352832},
                 "ecpu_freq_mhz": 1020, "ecpu_usage_percent": 12.0, "pcpu_freq_mhz": 4056, "pcpu_usage_percent": 70.0,
                 "gpu_freq_mhz": 338, "gpu_usage_percent": 3.5, "pcpu_freqs_mhz": [4056, 3900, 2000],
                 "ecpu_freqs_mhz": [1020, 1020], "cpu_power": 12.0, "gpu_power": 0.4}


# ------------------------------------------------------------------------------------------- sample
def test_sample_put_first_source_wins_and_force_replaces():
    s = sensors.Sample()
    assert s.put("cpu_temp_c", 50.0, "a") and not s.put("cpu_temp_c", 60.0, "b")
    assert s.cpu_temp_c == 50.0 and s.sources["cpu_temp_c"] == "a"
    assert s.put("cpu_temp_c", 61.0, "b", force=True) and s.cpu_temp_c == 61.0 and s.sources["cpu_temp_c"] == "b"
    assert not s.put("cpu_power_w", float("nan"), "a") and not s.put("cpu_power_w", "12 W", "a")
    assert not s.put("cpu_power_w", True, "a") and s.cpu_power_w is None
    assert s.put("cpu_clock_per_mhz", [4000, 4100.06], "a") and s.cpu_clock_per_mhz == [4000.0, 4100.1]
    assert not s.put("cpu_clock_per_mhz", [1, 2], "b") and not s.put("cpu_percent_per", [1, None], "b")
    assert s.put("gpu_name", " RX 9060 XT ", "a") and s.gpu_name == "RX 9060 XT"
    assert s.put("gpu_name", "Other", "b", force=True) and s.gpu_name == "Other"


def test_sample_round_trips_and_rejects_bad_json():
    s = sensors.Sample(t=1.5, machine="m", cpu_percent=12.345, cpu_percent_per=[1, 2], ram_used_gb=3.0,
                       ram_total_gb=6.0, sources={"cpu_percent": "psutil"}, missing={"cpu_temp_c": "no sensor"})
    back = sensors.Sample.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back.cpu_percent == 12.35 and back.cpu_percent_per == [1.0, 2.0] and back.ram_percent == 50.0
    assert back.missing == {"cpu_temp_c": "no sensor"} and back.machine == "m"
    assert sensors.Sample.from_dict(s.to_dict(per_cpu=False)).cpu_percent_per == []
    bad = sensors.Sample.from_dict({"cpu_temp_c": "hot", "cpu_power_w": float("inf"), "gpu_percent": True,
                                    "cpu_percent_per": list(range(5000)), "cpu_clock_per_mhz": [1, "x"],
                                    "machine": "x" * 1000, "missing": {"k": 3, "j": "why"}, "t": "now",
                                    "unknown_key_from_a_newer_peer": 1})
    assert bad.cpu_temp_c is None and bad.cpu_power_w is None and bad.gpu_percent is None
    assert bad.cpu_percent_per == [] and bad.cpu_clock_per_mhz == [] and len(bad.machine) == 200
    assert bad.missing == {"j": "why"} and bad.t == 0.0
    for junk in (None, [], "text", 3):
        assert sensors.Sample.from_dict(junk) == sensors.Sample()


def test_line_and_levels():
    s = sensors.Sample(t=time.time(), cpu_percent=37, cpu_clock_mhz=4620, cpu_clock_min_mhz=3850,
                       cpu_clock_max_mhz=5490, cpu_temp_c=92.0, cpu_power_w=88, gpu_percent=9, gpu_temp_c=36,
                       gpu_mem_used_mb=2048, gpu_mem_total_mb=16304, ram_used_gb=16.8, ram_total_gb=31.1)
    text = sensors.line(s)
    assert "CPU 37 % | 4.62 GHz avg (3.85-5.49) | 92 °C | 88 W" in text
    assert "GPU 9 % | 36 °C | VRAM 2.0/15.9 GB" in text and "RAM 16.8/31.1 GB (54 %)" in text and "[HOT]" in text
    assert sensors.fmt_clock(837) == "837 MHz" and sensors.fmt_clock(None) == "-"
    assert [sensors.level("temp", t) for t in (None, 79.9, 80, 90)] == ["ok", "ok", "warn", "hot"]
    assert [sensors.level("mem", p) for p in (50, 85, 96)] == ["ok", "warn", "hot"]
    assert "GPU" not in sensors.line(sensors.Sample(cpu_percent=1, ram_used_gb=1, ram_total_gb=2))


# ---------------------------------------------------------------------------- LibreHardwareMonitor
def test_lhm_tree_parsing():
    got = sensors.parse_lhm(lhm_tree())
    assert got["cpu_name"] == "AMD Ryzen 9 7900X"
    assert got["cpu_temp_c"] == 48.1                       # Tctl/Tdie before Package
    assert got["cpu_power_w"] == 64.2                      # Package before PPT
    assert got["cpu_clock_per_mhz"] == [4000.0 + 100 * i for i in range(12)]   # "Core #N", not the Effective ones
    assert got["cpu_clock_mhz"] == pytest.approx(4550.0)
    assert (got["cpu_clock_min_mhz"], got["cpu_clock_max_mhz"]) == (4000.0, 5100.0)
    assert got["gpu_name"] == "AMD Radeon RX 9060 XT"     # the GPU with the most memory, not the iGPU
    assert (got["gpu_percent"], got["gpu_temp_c"], got["gpu_clock_mhz"], got["gpu_power_w"]) == (9.0, 36.0, 837.0, 15.0)
    assert (got["gpu_mem_used_mb"], got["gpu_mem_total_mb"]) == (2045.0, 16304.0)
    assert "fan_rpm" not in got and "0 RPM for every fan" in got["_notes"]["fan_rpm"]
    assert sensors.parse_lhm(lhm_tree(fans=(1180, 0)))["fan_rpm"] == 1180.0
    comma = sensors.parse_lhm(lhm_tree(decimal=","))              # a German Windows
    assert comma["cpu_temp_c"] == 48.1 and comma["gpu_mem_total_mb"] == 16304.0


def test_old_lhm_tree_without_ids_and_fahrenheit():
    got = sensors.parse_lhm(old_lhm_tree())
    assert got["cpu_temp_c"] == pytest.approx(86.0)              # CPU Package, 186.8 °F
    assert got["cpu_power_w"] == 95.5 and got["cpu_clock_mhz"] == 4600.0
    assert got["gpu_name"] == "NVIDIA GeForce RTX 3080" and got["gpu_percent"] == 88.0
    assert got["gpu_power_w"] == 310.2 and got["gpu_mem_used_mb"] == pytest.approx(8.5 * 1024)


@pytest.mark.parametrize("junk", [None, [], "text", {"Children": "no"}, {"Children": [None, 3, {"Value": None}]},
                                  {"Text": "x", "Children": [{"Text": "Temperatures", "Children": [
                                      {"Text": "t", "Value": "hot", "Children": []}]}]}])
def test_lhm_parser_never_raises_on_junk(junk):
    assert sensors.parse_lhm(junk) == {"_notes": {}}


def test_lhm_deep_tree_is_cut_off():
    node = {"Text": "leaf", "Value": "1 W", "Type": "Power", "Children": []}
    for _ in range(5000):
        node = {"Text": "n", "Children": [node]}
    assert sensors.parse_lhm(node) == {"_notes": {}}


def test_lhm_provider_fills_prefers_its_gpu_and_backs_off():
    calls = []

    def fetch():
        calls.append(1)
        return lhm_tree()

    p = sensors.LibreHardwareMonitor(fetch=fetch)
    s = sensors.Sample()
    s.put("gpu_percent", 1.0, "Windows performance counters")
    s.put("cpu_clock_mhz", 4960.0, "Windows performance counters")
    p.read(s)
    assert s.cpu_temp_c == 48.1 and s.sources["cpu_temp_c"] == "LibreHardwareMonitor"
    assert s.gpu_percent == 9.0 and s.sources["gpu_percent"] == "LibreHardwareMonitor"   # the driver's GPU busy %
    assert s.cpu_clock_mhz == 4960.0                     # the counters' (Task Manager's) clock stays

    def down():
        calls.append(1)
        from urllib.error import URLError

        raise URLError(ConnectionRefusedError(10061, "refused"))

    p = sensors.LibreHardwareMonitor("http://127.0.0.1:8085/", fetch=down)
    s = sensors.Sample()
    p.read(s)
    p.read(s)                                            # inside the back-off: not asked again
    assert len(calls) == 2
    assert "not answering at http://127.0.0.1:8085" in s.missing["cpu_temp_c"]
    assert "Options > Remote Web Server > Run" in s.missing["cpu_temp_c"]


class _LHMHandler(BaseHTTPRequestHandler):
    tree: dict = {}
    auth = ""

    def log_message(self, *_a):
        pass

    def do_GET(self):
        if self.auth and self.headers.get("Authorization") != "Basic " + base64.b64encode(self.auth.encode()).decode():
            self.send_response(401)
            self.end_headers()
            return
        if self.path != "/data.json":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(self.tree).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def lhm_server():
    servers = []

    def make(auth=""):
        handler = type("H", (_LHMHandler,), {"tree": lhm_tree(), "auth": auth})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield make
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def test_lhm_over_http_with_basic_auth(lhm_server):
    url = lhm_server(auth="brandon:p@ss")
    s = sensors.Sample()
    sensors.LibreHardwareMonitor(url).read(s)
    assert s.cpu_temp_c is None and "asks for a user name and password" in s.missing["cpu_temp_c"]
    host = url.split("//", 1)[1]
    p = sensors.LibreHardwareMonitor(f"http://brandon:p%40ss@{host}/data.json")
    assert p.display_url() == f"http://{host}" and "p%40ss" not in p.display_url()
    s = sensors.Sample()
    p.read(s)
    assert s.cpu_temp_c == 48.1 and s.gpu_mem_total_mb == 16304.0
    s = sensors.Sample()
    sensors.LibreHardwareMonitor(lhm_server()).read(s)       # no auth asked
    assert s.cpu_power_w == 64.2


def test_lhm_unreachable_is_quick():
    import socket

    with socket.socket() as sock:                 # a port nobody listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    t = time.perf_counter()
    s = sensors.Sample()
    sensors.LibreHardwareMonitor(f"http://127.0.0.1:{port}").read(s)
    assert time.perf_counter() - t < 2.0 and "not answering" in s.missing["cpu_temp_c"]


# --------------------------------------------------------------------------------- Windows counters
PDH_VALUES = {
    "base": 4701.0, "perf": 106.0,
    "perf_all": {"_Total": 106.0, "0,_Total": 106.0, "0,10": 80.0, "0,2": 110.0, "0,0": 116.0, "0,1": 60.0},
    "gpu_engine": {
        "pid_100_luid_0x00000000_0x0000D1A2_phys_0_eng_0_engtype_3D": 20.0,
        "pid_200_luid_0x00000000_0x0000D1A2_phys_0_eng_0_engtype_3D": 15.0,
        "pid_300_luid_0x00000000_0x0000D1A2_phys_0_eng_1_engtype_Compute": 55.0,
        "pid_300_luid_0x00000000_0x0000D1A2_phys_0_eng_1_engtype_Compute_1": 30.0,
        "pid_400_luid_0x00000000_0x00001111_phys_0_eng_0_engtype_3D": 90.0,   # the iGPU, busier but smaller
    },
    "gpu_memory": {"luid_0x00000000_0x0000D1A2_phys_0": 2.0 * 2**30, "luid_0x00000000_0x00001111_phys_0": 2**28},
}


def test_pdh_metrics_follow_task_manager():
    got = sensors.pdh_metrics(PDH_VALUES)
    assert got["cpu_base_mhz"] == 4701.0
    assert got["cpu_clock_mhz"] == pytest.approx(4701 * 1.06)          # Processor Frequency x % Performance / 100
    assert got["cpu_clock_per_mhz"] == pytest.approx([4701 * 1.16, 4701 * 0.60, 4701 * 1.10, 4701 * 0.80])
    assert got["cpu_clock_max_mhz"] == pytest.approx(4701 * 1.16) and got["cpu_clock_min_mhz"] == pytest.approx(2820.6)
    assert got["gpu_percent"] == 85.0          # eng_1: 55 + 30 across processes; the dGPU (most dedicated memory)
    assert got["gpu_mem_used_mb"] == 2048.0
    assert sensors.pdh_metrics({}) == {}
    assert sensors.pdh_metrics({"base": None, "perf": 50, "gpu_engine": {"junk": 3}, "gpu_memory": {"x": "y"}}) == {}
    only_total = sensors.pdh_metrics({"base": 3000.0, "perf_all": {"0,0": 50.0, "1,0": 100.0}})
    assert only_total["cpu_clock_mhz"] == 2250.0 and only_total["cpu_clock_per_mhz"] == [1500.0, 3000.0]


class FakeCounters(sensors.WindowsCounters):
    """WindowsCounters with the PDH calls replaced by a mocked function (no ctypes, any OS)."""

    def __init__(self, values, failed=None):
        self.values = values
        self.counters = {k: k for k in sensors.PDH_COUNTERS if k in values}
        self.failed = failed or {}
        self.collected = 0
        self._t, self._first, self._gpu_reg = time.monotonic() - 1, True, None

        class Pdh:
            def PdhCollectQueryData(_s, _q):    # noqa: N802, N805
                self.collected += 1
                return 0

        self.pdh, self.query = Pdh(), None

    def _value(self, h):
        return self.values[h]

    def _array(self, h):
        return self.values[h]


def test_windows_counters_provider_with_mocked_pdh(monkeypatch):
    monkeypatch.setattr(sensors, "windows_gpu_registry", lambda: ("AMD Radeon RX 9060 XT", 16304.0))
    p = FakeCounters(PDH_VALUES)
    s = sensors.Sample()
    p.read(s)
    assert p.collected == 1 and s.sources["cpu_clock_mhz"] == "Windows performance counters"
    assert s.cpu_clock_mhz == pytest.approx(4983.06) and len(s.cpu_clock_per_mhz) == 4
    assert s.gpu_name == "AMD Radeon RX 9060 XT" and s.gpu_mem_total_mb == 16304.0 and s.gpu_percent == 85.0
    missing = FakeCounters({"base": 4701.0}, failed={"perf": "x: not available", "gpu_engine": "y: not available"})
    s = sensors.Sample()
    missing.read(s)
    assert s.cpu_base_mhz == 4701.0 and s.cpu_clock_mhz is None and "not available" in s.missing["cpu_clock_mhz"]
    assert "GPU Engine" in s.missing["gpu_percent"]


@pytest.mark.skipif(sys.platform != "win32", reason="the real performance counters are Windows only")
def test_real_windows_counters():
    p = sensors.WindowsCounters()
    try:
        s = sensors.Sample()
        p.read(s)
        assert s.cpu_base_mhz and s.cpu_clock_mhz and s.cpu_clock_per_mhz
    finally:
        p.close()


# ------------------------------------------------------------------------------------------ macmon
def test_macmon_parsing():
    got = sensors.parse_macmon(MACMON_TUPLES, p_cores=10, e_cores=4)
    assert (got["cpu_temp_c"], got["gpu_temp_c"]) == (61.73, 48.95)
    assert got["cpu_clock_mhz"] == pytest.approx((3624 * 10 + 2064 * 4) / 14)       # weighted by the cores
    assert (got["cpu_clock_min_mhz"], got["cpu_clock_max_mhz"]) == (2064, 3624)
    assert (got["gpu_clock_mhz"], got["gpu_percent"]) == (1398, pytest.approx(25.5))
    assert (got["cpu_power_w"], got["gpu_power_w"]) == (31.5, 6.25)
    assert got["ram_total_gb"] == 36.0 and got["ram_used_gb"] == pytest.approx(20.5, abs=0.01)
    assert sensors.parse_macmon(MACMON_TUPLES)["cpu_clock_mhz"] == pytest.approx((3624 + 2064) / 2)   # cores unknown
    new = sensors.parse_macmon(MACMON_FIELDS, 10, 4)
    assert "cpu_temp_c" not in new and new["gpu_temp_c"] == 44.0                     # 0 = not read
    assert new["cpu_clock_per_mhz"] == [1020, 1020, 4056, 3900, 2000]
    assert new["cpu_clock_mhz"] == pytest.approx(11996 / 5) and new["gpu_percent"] == pytest.approx(3.5)
    for junk in (None, [], {"temp": "x", "ecpu_usage": "y", "memory": [1]}, {"pcpu_usage": [None, None]}):
        assert sensors.parse_macmon(junk) == {}


def test_without_macmon_the_mac_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(sensors, "find_macmon", lambda: None)
    r = sensors.Reader(sensors.SensorConfig(), kind="macos")
    try:
        assert [p.label for p in r.providers] == ["psutil", "macmon"]
        s = r.read()
    finally:
        r.close()
    assert s.cpu_percent is not None and s.ram_used_gb is not None
    assert "brew install macmon" in s.missing["cpu_temp_c"] and "brew install macmon" in s.missing["cpu_clock_mhz"]
    assert "unified memory" in s.missing["gpu_mem_used_mb"]


FAKE_MACMON = r'''
import json, sys, time
line = json.dumps(%s)
args = sys.argv[1:]
if "-s" in args:
    print(line, flush=True)
    sys.exit(0)
interval = int(args[args.index("-i") + 1]) / 1000
while True:
    print(line, flush=True)
    time.sleep(interval)
'''


@pytest.fixture
def fake_macmon(tmp_path):
    script = tmp_path / "macmon_impl.py"
    script.write_text(FAKE_MACMON % repr(MACMON_TUPLES), encoding="utf-8")
    exe = tmp_path / "macmon"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.skipif(not POSIX, reason="the fake macmon is a shell script")
def test_macmon_once_and_continuous(fake_macmon, monkeypatch):
    monkeypatch.setattr(sensors, "_cluster_cores", lambda: (10, 4))
    once = sensors.Macmon(exe=fake_macmon)
    s = sensors.Sample()
    once.read(s)
    assert s.cpu_temp_c == 61.73 and s.sources["cpu_temp_c"] == "macmon" and once.proc is None
    keep = sensors.Macmon(continuous=True, interval_s=0.2, exe=fake_macmon)
    try:
        s = sensors.Sample()
        keep.read(s)
        assert s.gpu_power_w == 6.25 and keep.proc is not None and keep.proc.poll() is None
        proc = keep.proc
        s = sensors.Sample()
        keep.read(s)                                # the same process, its newest line
        assert keep.proc is proc and s.cpu_power_w == 31.5
    finally:
        keep.close()
    assert proc.poll() is not None                  # closed: no macmon left behind


@pytest.mark.skipif(not POSIX, reason="the fake macmon is a shell script")
def test_a_macmon_that_fails_is_retried_later(tmp_path, monkeypatch):
    exe = tmp_path / "macmon"
    exe.write_text("#!/bin/sh\necho 'macmon: no IOReport' >&2\nexit 3\n", encoding="utf-8")
    exe.chmod(0o755)
    for continuous in (False, True):
        p = sensors.Macmon(continuous=continuous, interval_s=0.2, exe=str(exe))
        s = sensors.Sample()
        p.read(s)
        assert "macmon" in s.missing["cpu_temp_c"] and p._retry_at > time.monotonic()
        p.close()


# ------------------------------------------------------------------------------------------- Linux
Freq = namedtuple("Freq", "current min max")
Temp = namedtuple("Temp", "label current high critical")
Fan = namedtuple("Fan", "label current")


def fake_sys(tmp_path: Path) -> Path:
    root = tmp_path / "sys"
    (root / "devices/system/cpu/cpu0/cpufreq").mkdir(parents=True)
    (root / "devices/system/cpu/cpu0/cpufreq/base_frequency").write_text("4700000\n")
    dev = root / "class/drm/card1/device"
    (dev / "hwmon/hwmon4").mkdir(parents=True)
    (root / "class/drm/card1-DP-1").mkdir()                          # a connector, not a card
    for name, value in (("vendor", "0x1002"), ("gpu_busy_percent", "73"), ("mem_info_vram_used", str(3 * 2**30)),
                        ("mem_info_vram_total", str(16 * 2**30))):
        (dev / name).write_text(value + "\n")
    for name, value in (("temp1_input", "61000"), ("power1_average", "142000000"), ("freq1_input", "2650000000")):
        (dev / "hwmon/hwmon4" / name).write_text(value + "\n")
    other = root / "class/drm/card0/device"
    other.mkdir(parents=True)
    (other / "vendor").write_text("0x8086\n")                        # an Intel iGPU: not amdgpu
    (root / "class/powercap/intel-rapl:0").mkdir(parents=True)
    (root / "class/powercap/intel-rapl:0/energy_uj").write_text("1000000\n")
    return root


@pytest.mark.skipif(sys.platform == "win32", reason="the fake sysfs needs names like intel-rapl:0, which Windows forbids")
def test_linux_sysfs_provider(tmp_path, monkeypatch):
    import psutil

    root = fake_sys(tmp_path)
    monkeypatch.setattr(psutil, "cpu_freq", lambda percpu=False: [Freq(4800.0, 400, 5600), Freq(3000.0, 400, 5600)],
                        raising=False)
    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: {
        "acpitz": [Temp("", 30.0, None, None)],
        "k10temp": [Temp("Tccd1", 70.0, None, None), Temp("Tctl", 78.5, None, None)]}, raising=False)
    monkeypatch.setattr(psutil, "sensors_fans", lambda: {"nct6799": [Fan("cpu_fan", 0), Fan("fan2", 1250)]},
                        raising=False)
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(sensors.time, "monotonic", lambda: next(clock))
    p = sensors.LinuxSysfs(root=root)
    s = sensors.Sample()
    p.read(s)
    assert s.cpu_clock_mhz == 3900.0 and (s.cpu_clock_min_mhz, s.cpu_clock_max_mhz) == (3000.0, 4800.0)
    assert s.cpu_base_mhz == 4700.0 and s.cpu_temp_c == 78.5 and s.fan_rpm == 1250
    assert (s.gpu_percent, s.gpu_temp_c, s.gpu_power_w, s.gpu_clock_mhz) == (73.0, 61.0, 142.0, 2650.0)
    assert (s.gpu_mem_used_mb, s.gpu_mem_total_mb) == (3072.0, 16384.0) and s.sources["gpu_percent"] == "amdgpu sysfs"
    assert s.cpu_power_w is None and "two readings" in s.missing["cpu_power_w"]
    (root / "class/powercap/intel-rapl:0/energy_uj").write_text(str(1000000 + 130 * 10**6) + "\n")
    s = sensors.Sample()
    p.read(s)
    assert s.cpu_power_w == pytest.approx(65.0) and s.sources["cpu_power_w"] == "RAPL"


def test_linux_without_sensors_says_why(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: {}, raising=False)
    s = sensors.Sample()
    sensors.LinuxSysfs(root=tmp_path / "empty").read(s)
    assert "no cpufreq" in s.missing["cpu_clock_mhz"] and "k10temp" in s.missing["cpu_temp_c"]
    assert "RAPL" in s.missing["cpu_power_w"]
    assert sensors.pick_cpu_temp({"coretemp": [Temp("Core 0", 50, None, None),
                                               Temp("Package id 0", 64, None, None)]}) == (64, "coretemp Package id 0")
    assert sensors.pick_cpu_temp({"cpu_thermal": [Temp("", 47.2, None, None)]})[0] == 47.2


@pytest.mark.skipif(not POSIX or os.geteuid() == 0, reason="root reads every file")
def test_rapl_readable_by_root_only(tmp_path):
    root = fake_sys(tmp_path)
    (root / "class/powercap/intel-rapl:0/energy_uj").chmod(0o000)
    s = sensors.Sample()
    sensors.LinuxSysfs(root=root)._read_rapl(s)
    assert "root only" in s.missing["cpu_power_w"]


def test_nvidia_smi_parsing():
    text = "NVIDIA GeForce RTX 4090, 97, 71, 2715, 431.20, 20123, 24564\nNVIDIA T400, 0, 40, 300, [N/A], 10, 2048\n"
    got = sensors.NvidiaSmi.parse(text)
    assert got == {"gpu_name": "NVIDIA GeForce RTX 4090", "gpu_percent": 97.0, "gpu_temp_c": 71.0,
                   "gpu_clock_mhz": 2715.0, "gpu_power_w": 431.2, "gpu_mem_used_mb": 20123.0,
                   "gpu_mem_total_mb": 24564.0}
    assert sensors.NvidiaSmi.parse("garbage\n,,\n") == {}


def test_wsl_says_the_sensors_are_on_the_windows_side(tmp_path, monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: {}, raising=False)
    monkeypatch.setattr(sensors.NvidiaSmi, "find", staticmethod(lambda: None))
    cfg = sensors.SensorConfig(lhm_url="http://127.0.0.1:9")      # nothing listens there
    r = sensors.Reader(cfg, kind="wsl")
    try:
        assert [p.label for p in r.providers] == ["psutil", "Linux sysfs", "LibreHardwareMonitor"]
        s = r.read()
    finally:
        r.close()
    assert s.os == "wsl" and s.cpu_percent is not None
    assert "Windows side" in s.missing["cpu_power_w"] or "LibreHardwareMonitor" in s.missing["cpu_power_w"]
    assert "WSL2" in s.missing["gpu_percent"]


# ----------------------------------------------------------------------------------- never raises
class Boom(sensors.Provider):
    label = "boom"

    def read(self, s):
        raise RuntimeError("sensor chip on fire")


class Slow(sensors.Provider):
    label = "slow"

    def __init__(self, seconds):
        self.seconds = seconds

    def read(self, s):
        time.sleep(self.seconds)
        s.put("cpu_temp_c", 55.0, self.label)


def test_a_failing_provider_never_fails_a_read():
    r = sensors.Reader(sensors.SensorConfig(), providers=[Boom(), sensors.CpuLoad(), Boom()], kind="linux")
    s = r.read()
    assert s.cpu_percent is not None and s.source == "psutil"
    assert "boom failed (RuntimeError: sensor chip on fire)" in s.missing["_boom"]
    assert all(key in s.missing or s.has(key) for key, _ in sensors.METRICS)


def test_providers_that_cannot_start_are_reported(monkeypatch):
    def broken():
        raise OSError("pdh.dll not found")

    monkeypatch.setattr(sensors, "WindowsCounters", broken)
    monkeypatch.setattr(sensors.NvidiaSmi, "find", staticmethod(lambda: None))
    cfg = sensors.SensorConfig(lhm_url="http://127.0.0.1:9")
    r = sensors.Reader(cfg, kind="windows")
    s = r.read()
    r.close()
    assert "pdh.dll not found" in s.missing["_Windows performance counters"]
    assert "LibreHardwareMonitor" in s.missing["cpu_temp_c"] and s.cpu_percent is not None
    assert any("!!" in line and "pdh.dll" in line for line in sensors.doctor_lines(s))


def test_module_read_never_raises(monkeypatch):
    sensors.reset()

    def broken(*_a, **_k):
        raise MemoryError("no memory for sensors")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sensors, "Reader", broken)
        s = sensors.read()
    assert s.source == "error" and "MemoryError" in s.missing["_reader"]
    sensors.reset()
    s = sensors.read()
    assert s.cpu_percent is not None and s.machine
    sensors.reset()


def test_module_read_gives_the_last_sample_while_another_thread_reads(monkeypatch):
    sensors.reset()
    try:
        sensors._shared = sensors.Reader(sensors.SensorConfig(), providers=[Slow(0.05)], kind="linux")
        first = sensors.read()
        sensors._shared.providers = [Slow(2.5)]
        t = threading.Thread(target=sensors.read)
        t.start()
        time.sleep(0.2)
        t0 = time.perf_counter()
        got = sensors.read()
        assert time.perf_counter() - t0 < 1.6 and got.cpu_temp_c == first.cpu_temp_c == 55.0
        t.join()
    finally:
        sensors.reset()


def test_sensors_off(monkeypatch):
    monkeypatch.setenv(sensors.ENV_SWITCH, "0")
    cfg = sensors.load_config()
    assert not cfg.enabled
    r = sensors.Reader(cfg, kind="windows")
    s = r.read()
    r.close()
    assert [p.label for p in r.providers] == ["psutil"] and s.cpu_percent is not None
    assert "turned off" in s.missing["cpu_temp_c"]


# ------------------------------------------------------------------------------------------ monitor
class CountingReader:
    def __init__(self, fail_every=0):
        self.n, self.closed, self.fail_every = 0, 0, fail_every

    def read(self):
        self.n += 1
        if self.fail_every and self.n % self.fail_every == 0:
            raise RuntimeError("flaky")
        return sensors.Sample(t=time.time(), cpu_percent=float(self.n))

    def close(self):
        self.closed += 1


def test_monitor_samples_keeps_history_and_survives_errors():
    reader = CountingReader(fail_every=3)
    m = sensors.Monitor(interval_s=0.5, history_s=2.0, reader_factory=lambda: reader)
    m.interval_s = 0.05                               # faster than settings allow, for the test
    assert m.latest(wake=False) is None and not m.running
    first = m.latest(wait=2.0)                        # wake=True starts it
    assert first is not None and m.running
    time.sleep(0.6)
    hist = m.history()
    assert len(hist) <= m._history.maxlen and [h.cpu_percent for h in hist] == sorted(h.cpu_percent for h in hist)
    assert all(h.cpu_percent % 3 for h in hist)       # the failed reads left no sample
    m.stop()
    assert not m.running and reader.closed == 1
    n = reader.n
    assert m.latest() is not None and not m.running   # stopped: latest() does not start it again
    time.sleep(0.2)
    assert reader.n == n


def test_monitor_keeps_its_interval_however_often_it_is_asked():
    reader = CountingReader()
    m = sensors.Monitor(interval_s=0.5, reader_factory=lambda: reader).start()
    t = time.monotonic()
    while time.monotonic() - t < 1.2:            # a CLI or GUI loop polling every few ms
        m.latest()
        time.sleep(0.005)
    m.stop()
    assert 2 <= reader.n <= 4, reader.n


def test_monitor_idles_without_requests_and_wakes():
    reader = CountingReader()
    m = sensors.Monitor(interval_s=0.5, idle_stop_s=0.3, reader_factory=lambda: reader)
    m.interval_s = 0.05
    m.start()
    time.sleep(0.8)
    assert reader.closed == 1                         # idle: providers freed (a Mac's macmon stops)
    n = reader.n
    time.sleep(0.3)
    assert reader.n == n
    assert m.latest() is not None                     # the old sample at once; sampling resumes
    time.sleep(0.3)
    assert reader.n > n
    m.stop()


# ----------------------------------------------------------------------------------------- settings
def test_per_machine_sensor_settings(isolated_settings):
    s = settings_mod.Settings()
    s.set_sensors_for("lhm_url", "http://127.0.0.1:8086", machine="BTM NOVA")
    s.set_sensors_for("enabled", False, machine="brandons-macbook-pro")
    s.set_sensors_for("interval_s", 0.1, machine="btm-nova")
    with pytest.raises(ValueError):
        s.set_sensors_for("password", "x")
    with pytest.raises(ValueError):
        s.set_sensors_for("enabled", "yes")
    assert s.sensors.machine == {"btm-nova": {"lhm_url": "http://127.0.0.1:8086", "interval_s": 0.1},
                                 "brandons-macbook-pro": {"enabled": False}}
    nova = s.sensors_for("btm-nova")
    assert nova.lhm_url == "http://127.0.0.1:8086" and nova.interval_s == 0.5 and nova.enabled   # clamped
    assert not s.sensors_for("Brandons-MacBook-Pro").enabled
    assert s.sensors_for("other").lhm_url == sensors.DEFAULT_LHM_URL
    settings_mod.save(s)
    back = settings_mod.load()
    assert back.sensors.machine == s.sensors.machine
    # another machine's entry written meanwhile (SyncThing) is kept when this one saves its own change
    other = settings_mod.load()
    other.set_sensors_for("lhm_url", "http://10.0.0.2:8085", machine="wsl-box")
    settings_mod.save(other)
    back.set_sensors_for("enabled", None, machine="brandons-macbook-pro")
    settings_mod.save(back)
    final = settings_mod.load().sensors.machine
    assert final["wsl-box"] == {"lhm_url": "http://10.0.0.2:8085"} and "brandons-macbook-pro" not in final
    junk = settings_mod.Settings.from_dict({"sensors": {"machine": {"a": {"enabled": "yes", "interval_s": [1],
                                                                          "lhm_url": 3}, "b": "nope"}}})
    assert junk.sensors.machine == {"a": {"enabled": "yes", "lhm_url": 3}}       # kept as found, but not used:
    a = junk.sensors_for("a")
    assert a.enabled and a.lhm_url == sensors.DEFAULT_LHM_URL                       # wrong types are ignored


def test_load_config_follows_this_machines_settings(isolated_settings):
    from quantabench.dist import profile

    s = settings_mod.Settings()
    s.set_sensors_for("lhm_url", "http://127.0.0.1:9999", machine=profile.machine_name())
    s.sensors.interval_s = 3.0
    settings_mod.save(s)
    cfg = sensors.load_config()
    assert cfg.lhm_url == "http://127.0.0.1:9999" and cfg.interval_s == 3.0 and cfg.enabled


# ------------------------------------------------------------------------------------- worker, CLI
TOKEN = "test-token-" + "y" * 32


def test_worker_serves_its_sensors(tmp_path, isolated_settings):
    from quantabench.dist import client as client_mod
    from quantabench.dist import worker as worker_mod

    w = worker_mod.Worker("127.0.0.1", 0, TOKEN, tmp_path / "runs", state_dir=tmp_path / "state", name="sensor-box",
                          awake=False).start()
    try:
        c = client_mod.WorkerClient(w.url, TOKEN, timeout=10)
        data = c.sensors()
        assert data["machine"] == "sensor-box" and data["interval_s"] > 0
        s = sensors.Sample.from_dict(data["sensors"])
        assert s.cpu_percent is not None and s.ram_total_gb and s.source
        info = c.info()
        assert isinstance(info["sensors"], dict) and info["sensors"]["cpu_percent_per"] == []   # small in /v1/info
        with pytest.raises(client_mod.WorkerError) as e:
            client_mod.WorkerClient(w.url, "wrong-token-" + "z" * 20).sensors()
        assert e.value.status == 401
    finally:
        w.stop(timeout=30)
    assert w._sensors is None


def test_monitor_command(capsys, isolated_settings):
    from quantabench import cli

    assert cli.main(["monitor", "--count", "2", "--interval", "0.5", "--per-cpu"]) == 0
    out = capsys.readouterr().out.splitlines()
    lines = [ln for ln in out if "  CPU " in ln]
    assert len(lines) == 2 and all("RAM" in ln and " GB" in ln for ln in lines)
    assert any(":" in ln and "%" in ln and ln.startswith("          0:") for ln in out)     # per-CPU cells
    assert cli.main(["monitor", "--count", "1", "--json"]) == 0
    one = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert one["cpu_percent"] is not None and "missing" in one


def test_monitor_set_writes_this_machines_entry(capsys, isolated_settings):
    from quantabench import cli

    assert cli.main(["monitor", "--set", "lhm_url=http://127.0.0.1:8090", "--set", "interval_s=1.5"]) == 0
    me = settings_mod.this_machine()
    assert settings_mod.load().sensors.machine == {me: {"lhm_url": "http://127.0.0.1:8090", "interval_s": 1.5}}
    assert "interval_s=1.5" in capsys.readouterr().out
    assert cli.main(["monitor", "--set", "lhm_url=", "--set", "enabled=off", "--machine", "mac-test"]) == 0
    assert settings_mod.load().sensors.machine["mac-test"] == {"enabled": False}
    assert cli.main(["monitor", "--set", "speed=fast"]) == 2
    assert cli.main(["monitor", "--set", "enabled=maybe"]) == 2


def test_monitor_command_for_a_worker(tmp_path, capsys, isolated_settings):
    from quantabench import cli
    from quantabench.dist import worker as worker_mod

    s = settings_mod.Settings()
    s.dist.worker_token = TOKEN
    settings_mod.save(s)
    w = worker_mod.Worker("127.0.0.1", 0, TOKEN, tmp_path / "runs", state_dir=tmp_path / "state", name="box",
                          awake=False).start()
    try:
        assert cli.main(["monitor", "--worker", w.url, "--count", "1", "--interval", "0.5"]) == 0
    finally:
        w.stop(timeout=30)
    out = capsys.readouterr().out
    assert "CPU" in out and "RAM" in out


def test_real_read_is_fast_and_complete():
    """This machine, whatever it is: a read within about a second, every metric either valued or explained."""
    sensors.reset()
    try:
        t = time.perf_counter()
        s = sensors.read()
        assert time.perf_counter() - t < 3.0
        t = time.perf_counter()
        s = sensors.read()
        assert time.perf_counter() - t < 1.5
    finally:
        sensors.reset()
    assert s.cpu_percent is not None and 0 <= s.cpu_percent <= 100 and s.ram_used_gb > 0
    for key, _label in sensors.METRICS:
        assert s.has(key) or s.missing.get(key), key
    assert all(math.isfinite(v) for v in s.cpu_percent_per)
