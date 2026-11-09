"""Live machine metrics (MON-1): CPU and GPU load, effective clocks, temperatures, power and memory.

    from quantabench import sensors
    s = sensors.read()                  # a Sample: never raises, returns within about a second
    s.cpu_clock_mhz, s.cpu_temp_c       # None where this machine cannot tell
    s.missing["cpu_temp_c"]             # why not, and how to enable it
    m = sensors.Monitor(interval_s=2)   # samples in a thread of its own; m.latest() and m.history() never block

Where the numbers come from (docs/ENGINES.md, "Live machine metrics"):

    every OS  psutil: CPU utilization, total and per logical CPU (from this reader's own ``cpu_times`` deltas, so
              it never disturbs ``psutil.cpu_percent``'s state), and RAM in use (total - available)
    Windows   the performance counters through ctypes (PDH; no pywin32): the effective clock as Task Manager shows
              it, ``Processor Frequency`` x ``% Processor Performance`` / 100 of ``\\Processor Information(_Total)``,
              per logical CPU from the ``(0,N)`` instances; GPU utilization by Task Manager's rule (``GPU Engine``:
              per engine the sum over processes, the busiest engine) and dedicated memory (``GPU Adapter Memory``).
              LibreHardwareMonitor's web server when it runs (``/data.json``, default http://127.0.0.1:8085):
              CPU temperature (Tctl/Tdie, CPU Package) and package power, GPU temperature, clock, power and VRAM,
              fans. It is the one source of temperatures on Windows.
    macOS     macmon (``brew install macmon``, MIT, no sudo): CPU and GPU temperatures, the P- and E-cluster and
              GPU clocks, GPU utilization, CPU and GPU power. A Monitor keeps one ``macmon pipe`` running; a single
              ``read()`` runs it once. Without it: utilization and RAM only.
    Linux     psutil ``cpu_freq(percpu=True)`` (cpufreq) and ``sensors_temperatures`` (k10temp Tctl, coretemp
              Package id 0), RAPL package energy (readable by root only on most systems), amdgpu in sysfs (busy %,
              temperature, clock, power, VRAM), ``nvidia-smi`` when installed.
    WSL2      the Linux VM's view (utilization, RAM, nvidia-smi); temperatures, clocks and power are on the Windows
              side: LibreHardwareMonitor's web server is tried at the configured URL (it answers there with WSL's
              mirrored networking, or at the Windows host's address).

Nothing here may fail or stall a caller: every provider catches its own errors (the reason goes to
``Sample.missing``), network reads time out after 0.6 s and back off for 30 s, and ``read()`` gives the last
sample when another thread is mid-read. ``settings.sensors.enabled = false`` (or ``QUANTABENCH_SENSORS=0`` for one
process) turns the temperatures, clocks and power off; utilization and RAM (psutil) are still reported, and the run
timings then record no sensor columns. The settings (``Settings.sensors_for``) are read when a reader is made: the
shared reader of ``read()`` keeps them until ``reset()``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

ENV_SWITCH = "QUANTABENCH_SENSORS"
DEFAULT_LHM_URL = "http://127.0.0.1:8085"
DEFAULT_INTERVAL_S = 2.0
HISTORY_S = 600.0                 # what a Monitor keeps for sparklines: the last 10 minutes
HTTP_TIMEOUT_S = 0.6
RETRY_S = 30.0                    # a source that failed is tried again after this long
MACMON_INSTALL = "install macmon for clocks and temperatures: brew install macmon (MIT, no sudo)"
LHM_ENABLE = ("run LibreHardwareMonitor (0.9.5 or later) and turn on Options > Remote Web Server > Run "
              "(port 8085)")

#: (key, label) of every metric, in display order (doctor, the GUI's tooltips)
METRICS: tuple[tuple[str, str], ...] = (
    ("cpu_percent", "CPU utilization"),
    ("cpu_clock_mhz", "CPU effective clock"),
    ("cpu_base_mhz", "CPU base clock"),
    ("cpu_temp_c", "CPU temperature"),
    ("cpu_power_w", "CPU package power"),
    ("gpu_percent", "GPU utilization"),
    ("gpu_temp_c", "GPU temperature"),
    ("gpu_clock_mhz", "GPU clock"),
    ("gpu_power_w", "GPU power"),
    ("gpu_mem_used_mb", "GPU memory (VRAM)"),
    ("ram_used_gb", "RAM in use"),
    ("fan_rpm", "Fan speed"),
)
#: colour thresholds of the GUI strip and the CLI's flags: (warn, hot)
TEMP_LEVELS_C = (80.0, 90.0)
MEM_LEVELS_PCT = (85.0, 95.0)


def _finite(x) -> float | None:
    """``x`` as a finite float, else None (bools, strings, NaN and inf are not numbers here)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------------------------------- sample
@dataclass
class Sample:
    """One reading of a machine. Units: %, MHz, °C, W, MB (GPU memory), GB (RAM), RPM. ``sources``: metric key ->
    where its value came from; ``missing``: metric key -> why it has no value and how to get one."""

    t: float = 0.0
    machine: str = ""
    os: str = ""
    source: str = ""
    cpu_name: str = ""
    cpu_percent: float | None = None
    cpu_percent_per: list[float] = field(default_factory=list)
    cpu_clock_mhz: float | None = None
    cpu_clock_min_mhz: float | None = None
    cpu_clock_max_mhz: float | None = None
    cpu_clock_per_mhz: list[float] = field(default_factory=list)
    cpu_base_mhz: float | None = None
    cpu_temp_c: float | None = None
    cpu_power_w: float | None = None
    gpu_name: str = ""
    gpu_percent: float | None = None
    gpu_temp_c: float | None = None
    gpu_clock_mhz: float | None = None
    gpu_power_w: float | None = None
    gpu_mem_used_mb: float | None = None
    gpu_mem_total_mb: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    fan_rpm: float | None = None
    sources: dict[str, str] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)

    # -- filling
    def put(self, key: str, value, source: str, force: bool = False) -> bool:
        """Set ``key`` from ``source`` unless an earlier source set it (``force``: replace it). Numbers must be
        finite; lists must hold numbers; strings must be non-empty."""
        current = getattr(self, key)
        if force and isinstance(current, str) and isinstance(value, str) and value.strip():
            current = ""
            setattr(self, key, current)
        elif force and current is not None and not isinstance(current, (str, list)) and _finite(value) is not None:
            current = None
            setattr(self, key, None)
            self.sources.pop(key, None)
        if isinstance(current, list):
            if current or not isinstance(value, (list, tuple)):
                return False
            vals = [_finite(v) for v in value]
            if not vals or any(v is None for v in vals):
                return False
            setattr(self, key, [round(v, 1) for v in vals])
            return True
        if isinstance(current, str):
            if current or not isinstance(value, str) or not value.strip():
                return False
            setattr(self, key, value.strip())
            return True
        if current is not None:
            return False
        v = _finite(value)
        if v is None:
            return False
        setattr(self, key, v)
        if key in dict(METRICS):
            self.sources.setdefault(key, source)
        return True

    def why(self, key: str, reason: str) -> None:
        """Note why ``key`` has no value (the first reason given wins)."""
        if reason:
            self.missing.setdefault(key, reason)

    # -- derived
    @property
    def ram_percent(self) -> float | None:
        if self.ram_used_gb is None or not self.ram_total_gb:
            return None
        return 100.0 * self.ram_used_gb / self.ram_total_gb

    @property
    def gpu_mem_percent(self) -> float | None:
        if self.gpu_mem_used_mb is None or not self.gpu_mem_total_mb:
            return None
        return 100.0 * self.gpu_mem_used_mb / self.gpu_mem_total_mb

    def has(self, key: str) -> bool:
        v = getattr(self, key, None)
        return bool(v) if isinstance(v, (list, str)) else v is not None

    # -- transport
    def to_dict(self, per_cpu: bool = True) -> dict:
        d = asdict(self)
        if not per_cpu:
            d["cpu_percent_per"], d["cpu_clock_per_mhz"] = [], []
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 2)
        return d

    @classmethod
    def from_dict(cls, data) -> Sample:
        """A Sample from JSON (a worker's ``/v1/sensors``): unknown keys are ignored, and every value is checked
        (finite numbers, at most 1024 per-CPU values, strings cut to 200 characters), so a peer of another version
        or a bad reply cannot break the reader."""
        s = cls()
        if not isinstance(data, dict):
            return s
        for f in fields(cls):
            if f.name not in data:
                continue
            v, cur = data[f.name], getattr(s, f.name)
            if isinstance(cur, list):
                if isinstance(v, list) and len(v) <= 1024:
                    nums = [_finite(x) for x in v]
                    if all(n is not None for n in nums):
                        setattr(s, f.name, nums)
            elif isinstance(cur, dict):
                if isinstance(v, dict):
                    setattr(s, f.name, {str(k)[:40]: str(x)[:300] for k, x in list(v.items())[:40]
                                        if isinstance(x, str)})
            elif isinstance(cur, str):
                if isinstance(v, str):
                    setattr(s, f.name, v[:200])
            elif f.name == "t":
                s.t = _finite(v) or 0.0
            else:
                setattr(s, f.name, _finite(v))
        return s


# ------------------------------------------------------------------------------------------ format
def fmt_clock(mhz: float | None) -> str:
    if mhz is None:
        return "-"
    return f"{mhz / 1000:.2f} GHz" if mhz >= 1000 else f"{mhz:.0f} MHz"


def fmt_num(x: float | None, unit: str, digits: int = 0) -> str:
    return "-" if x is None else f"{x:.{digits}f} {unit}"


def level(kind: str, value: float | None) -> str:
    """"ok", "warn" or "hot" for a temperature ("temp") or a memory percentage ("mem")."""
    if value is None:
        return "ok"
    warn, hot = TEMP_LEVELS_C if kind == "temp" else MEM_LEVELS_PCT
    return "hot" if value >= hot else "warn" if value >= warn else "ok"


def cpu_text(s: Sample) -> str:
    """"37 % | 4.62 GHz avg (3.85-5.49) | 61 °C | 88 W" with what is known."""
    parts = [fmt_num(s.cpu_percent, "%")]
    if s.cpu_clock_mhz is not None:
        clock = f"{fmt_clock(s.cpu_clock_mhz)} avg"
        if s.cpu_clock_min_mhz is not None and s.cpu_clock_max_mhz is not None:
            clock += f" ({s.cpu_clock_min_mhz / 1000:.2f}-{s.cpu_clock_max_mhz / 1000:.2f})"
        parts.append(clock)
    if s.cpu_temp_c is not None:
        parts.append(fmt_num(s.cpu_temp_c, "°C"))
    if s.cpu_power_w is not None:
        parts.append(fmt_num(s.cpu_power_w, "W"))
    return " | ".join(parts)


def gpu_text(s: Sample) -> str:
    parts = []
    if s.gpu_percent is not None:
        parts.append(fmt_num(s.gpu_percent, "%"))
    if s.gpu_clock_mhz is not None:
        parts.append(fmt_clock(s.gpu_clock_mhz))
    if s.gpu_temp_c is not None:
        parts.append(fmt_num(s.gpu_temp_c, "°C"))
    if s.gpu_power_w is not None:
        parts.append(fmt_num(s.gpu_power_w, "W"))
    if s.gpu_mem_used_mb is not None:
        total = f"/{s.gpu_mem_total_mb / 1024:.1f}" if s.gpu_mem_total_mb else ""
        parts.append(f"VRAM {s.gpu_mem_used_mb / 1024:.1f}{total} GB")
    return " | ".join(parts) or "-"


def ram_text(s: Sample) -> str:
    if s.ram_used_gb is None:
        return "-"
    total = f"/{s.ram_total_gb:.1f}" if s.ram_total_gb else ""
    pct = f" ({s.ram_percent:.0f} %)" if s.ram_percent is not None else ""
    return f"{s.ram_used_gb:.1f}{total} GB{pct}"


def line(s: Sample) -> str:
    """One compact line for ``quantabench-cli monitor``."""
    stamp = time.strftime("%H:%M:%S", time.localtime(s.t or time.time()))
    out = f"{stamp}  CPU {cpu_text(s)}"
    if any(s.has(k) for k in ("gpu_percent", "gpu_temp_c", "gpu_mem_used_mb", "gpu_power_w")):
        out += f"  ||  GPU {gpu_text(s)}"
    out += f"  ||  RAM {ram_text(s)}"
    if s.fan_rpm is not None:
        out += f"  ||  fan {s.fan_rpm:.0f} RPM"
    flags = []
    if level("temp", s.cpu_temp_c) == "hot" or level("temp", s.gpu_temp_c) == "hot":
        flags.append("HOT")
    if level("mem", s.ram_percent) == "hot":
        flags.append("RAM FULL")
    return out + (f"  [{', '.join(flags)}]" if flags else "")


# ------------------------------------------------------------------------------------------ config
@dataclass
class SensorConfig:
    enabled: bool = True
    interval_s: float = DEFAULT_INTERVAL_S
    lhm_url: str = DEFAULT_LHM_URL


def env_enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "").strip().lower() not in ("0", "off", "false", "no")


def load_config() -> SensorConfig:
    """This machine's sensor settings (``settings.sensors``, with its per-machine entry); defaults when the
    settings cannot be read. ``QUANTABENCH_SENSORS=0`` turns them off."""
    cfg = SensorConfig()
    try:
        from . import settings as settings_mod

        s = settings_mod.load().sensors_for()
        cfg = SensorConfig(bool(s.enabled), float(s.interval_s), str(s.lhm_url or DEFAULT_LHM_URL))
    except Exception:  # noqa: BLE001, S110 - unreadable settings: the defaults
        pass
    if not env_enabled():
        cfg.enabled = False
    return cfg


# --------------------------------------------------------------------------------------- providers
class Provider:
    """One source of numbers. ``read`` fills what it can (``Sample.put``) and notes what it cannot
    (``Sample.why``); it may raise, the Reader catches it. ``close`` frees what it holds (a process, a query)."""

    label = ""

    def read(self, s: Sample) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


def _times_total(t) -> float:
    total = float(sum(t))
    # Linux: guest time is already in user and nice
    return total - getattr(t, "guest", 0.0) - getattr(t, "guest_nice", 0.0)


def _busy_percent(a, b) -> float:
    da = _times_total(b) - _times_total(a)
    if da <= 0:
        return 0.0
    idle = (b.idle - a.idle) + (getattr(b, "iowait", 0.0) - getattr(a, "iowait", 0.0))
    return max(0.0, min(100.0, 100.0 * (da - idle) / da))


class CpuLoad(Provider):
    """CPU utilization (total and per logical CPU) and RAM, from psutil. The deltas are this provider's own, so
    several readers (the GUI's monitor, a run's load samples) do not disturb each other or ``cpu_percent``."""

    label = "psutil"

    def __init__(self):
        import psutil

        self._psutil = psutil
        self._last = psutil.cpu_times(percpu=True)
        self._t = time.monotonic()
        self._first = True

    def read(self, s: Sample) -> None:
        ps = self._psutil
        wait = 0.2 - (time.monotonic() - self._t)
        if self._first and wait > 0:     # the first read right after the start: give the deltas time
            time.sleep(wait)
        self._first = False
        now = ps.cpu_times(percpu=True)
        per = [_busy_percent(a, b) for a, b in zip(self._last, now, strict=False)]
        self._last, self._t = now, time.monotonic()
        if per:
            s.put("cpu_percent_per", per, self.label)
            s.put("cpu_percent", sum(per) / len(per), self.label)
        vm = ps.virtual_memory()
        s.put("ram_used_gb", (vm.total - vm.available) / 2**30, self.label)
        s.put("ram_total_gb", vm.total / 2**30, self.label)


# ------------------------------------------------------------------------- LibreHardwareMonitor
#: LHM's group nodes ("Temperatures", ...) -> sensor type, for versions whose sensors have no "Type"
LHM_GROUPS = {"Voltages": "Voltage", "Clocks": "Clock", "Temperatures": "Temperature", "Load": "Load",
              "Powers": "Power", "Fans": "Fan", "Data": "Data", "Controls": "Control", "Levels": "Level",
              "Throughput": "Throughput", "Currents": "Current", "Energy": "Energy", "Factors": "Factor",
              "Frequencies": "Frequency"}
LHM_CPU_TEMPS = ("Core (Tctl/Tdie)", "CPU Package", "Package", "Core (Tctl)", "Core (Tdie)", "Tdie", "Tctl",
                 "CCDs Max (Tdie)", "Core Max", "Core Average", "CPU Cores")
LHM_CPU_POWER = ("Package", "CPU Package", "CPU PPT", "Total Power", "CPU Cores")
LHM_GPU_POWER = ("GPU Package", "GPU Power", "GPU Total", "GPU Core", "GPU Board Power")
_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_CORE_CLOCK = re.compile(r"^(?:CPU )?(?:[PE]-)?Core #\d+$")
_NOT_A_TEMP = ("distance", "limit", "resolution", "critical", "warning")


def _lhm_value(text) -> tuple[float | None, str]:
    """(number, unit) of an LHM value string ("48.1 °C", "4618,0 MHz", "2045.0 MB")."""
    if not isinstance(text, str):
        return _finite(text), ""
    m = _NUM.search(text)
    if not m:
        return None, ""
    return _finite(float(m.group().replace(",", "."))), text[m.end():].strip()


def _hw_kind(hw_id: str, image: str) -> str:
    hw_id, image = (hw_id or "").lower(), (image or "").lower()
    if hw_id.startswith(("/amdcpu", "/intelcpu", "/cpu")) or image.endswith("/cpu.png"):
        return "cpu"
    if hw_id.startswith("/gpu") or image.endswith(("/ati.png", "/amd.png", "/nvidia.png", "/intel.png")):
        return "gpu"
    if hw_id == "/ram" or image.endswith("/ram.png"):
        return "ram"
    return "other"


def lhm_sensors(tree) -> list[dict]:
    """Every sensor of an LHM ``data.json`` tree: {"hw", "hw_kind", "hw_name", "type", "name", "value", "unit"}.
    Works with sensors that carry ``SensorId``/``Type`` (LHM 0.9.x) and with older trees that only have the group
    nodes ("Temperatures", ...) and hardware icons."""
    out: list[dict] = []

    def walk(node, hw: tuple[str, str, str], group: str, depth: int) -> None:
        if not isinstance(node, dict) or depth > 32:
            return
        text = str(node.get("Text") or "")
        children = node.get("Children") or []
        if node.get("HardwareId"):
            hw = (str(node["HardwareId"]), text, str(node.get("ImageURL") or ""))
        elif (not node.get("SensorId") and text not in LHM_GROUPS and str(node.get("ImageURL") or "").startswith(
                "images_icon/") and not str(node.get("ImageURL")).endswith("computer.png") and children):
            hw = ("", text, str(node["ImageURL"]))
        kind = node.get("Type") or (LHM_GROUPS.get(group) if not children else None)
        if not children and kind and node.get("Value") not in (None, ""):
            value, unit = _lhm_value(node.get("Value"))
            if value is not None:
                if kind == "SmallData":
                    kind = "Data"
                if kind == "Temperature" and unit.endswith("F"):
                    value, unit = (value - 32.0) * 5.0 / 9.0, "°C"
                out.append({"hw": hw[0], "hw_kind": _hw_kind(hw[0], hw[2]), "hw_name": hw[1], "type": kind,
                            "name": text, "value": value, "unit": unit, "id": str(node.get("SensorId") or "")})
        for c in children if isinstance(children, list) else []:
            walk(c, hw, text if children and not node.get("SensorId") else group, depth + 1)

    walk(tree, ("", "", ""), "", 0)
    return out


def _first(sensors: list[dict], kind: str, names: tuple[str, ...]) -> dict | None:
    by_name = {}
    for x in sensors:
        if x["type"] == kind:
            by_name.setdefault(x["name"], x)
    for n in names:
        if n in by_name:
            return by_name[n]
    return None


def _mb(x: dict | None) -> float | None:
    if x is None:
        return None
    unit = x["unit"].upper()
    return x["value"] * (1024.0 if unit.startswith("GB") else 1.0 / 1024 if unit.startswith("KB") else 1.0)


def parse_lhm(tree) -> dict:
    """The metrics in an LHM ``data.json`` tree (``Sample`` keys), plus "_notes" {key: reason} for what it has but
    cannot be used. CPU clocks are the per-core clocks ("Core #N", not the "Effective" ones), the fallback when the
    performance counters are not there."""
    sensors = lhm_sensors(tree)
    out: dict = {}
    notes: dict[str, str] = {}
    cpu = [x for x in sensors if x["hw_kind"] == "cpu"]
    if cpu:
        out["cpu_name"] = cpu[0]["hw_name"]
        t = _first(cpu, "Temperature", LHM_CPU_TEMPS)
        if t is None:
            temps = [x for x in cpu if x["type"] == "Temperature" and not any(w in x["name"].lower() for w in _NOT_A_TEMP)]
            t = max(temps, key=lambda x: x["value"]) if temps else None
        if t is not None:
            out["cpu_temp_c"] = t["value"]
        p = _first(cpu, "Power", LHM_CPU_POWER)
        if p is not None:
            out["cpu_power_w"] = p["value"]
        load = _first(cpu, "Load", ("CPU Total",))
        if load is not None:
            out["cpu_percent"] = load["value"]
        cores = [x["value"] for x in cpu if x["type"] == "Clock" and _CORE_CLOCK.match(x["name"]) and x["value"] > 0]
        if cores:
            out["cpu_clock_per_mhz"] = cores
            out["cpu_clock_mhz"] = sum(cores) / len(cores)
            out["cpu_clock_min_mhz"], out["cpu_clock_max_mhz"] = min(cores), max(cores)
        else:
            avg = _first(cpu, "Clock", ("Cores (Average)", "CPU Cores"))
            if avg is not None:
                out["cpu_clock_mhz"] = avg["value"]
    gpus: dict[str, list[dict]] = {}
    for x in sensors:
        if x["hw_kind"] == "gpu":
            gpus.setdefault(x["hw"] or x["hw_name"], []).append(x)
    if gpus:
        def vram_total(g: list[dict]) -> float:
            return _mb(_first(g, "Data", ("GPU Memory Total", "D3D Dedicated Memory Total"))) or 0.0

        g = max(gpus.values(), key=vram_total)        # several GPUs: the one with the most memory
        out["gpu_name"] = g[0]["hw_name"]
        for key, kind, names in (("gpu_percent", "Load", ("GPU Core", "D3D 3D")),
                                 ("gpu_temp_c", "Temperature", ("GPU Core", "GPU Hot Spot", "GPU Memory")),
                                 ("gpu_clock_mhz", "Clock", ("GPU Core", "GPU Shader")),
                                 ("gpu_power_w", "Power", LHM_GPU_POWER)):
            x = _first(g, kind, names)
            if x is not None:
                out[key] = x["value"]
        used = _mb(_first(g, "Data", ("GPU Memory Used", "D3D Dedicated Memory Used")))
        total = vram_total(g)
        if used is not None:
            out["gpu_mem_used_mb"] = used
        if total:
            out["gpu_mem_total_mb"] = total
    fans = [x["value"] for x in sensors if x["type"] == "Fan"]
    if fans and max(fans) > 0:
        out["fan_rpm"] = max(fans)
    elif fans:
        notes["fan_rpm"] = ("LibreHardwareMonitor reports 0 RPM for every fan (no fan readings from this board, or "
                            "the fans are stopped)")
    out["_notes"] = notes
    return out


class LibreHardwareMonitor(Provider):
    """LibreHardwareMonitor's remote web server (``<url>/data.json``). ``http://user:password@host:port`` sends
    basic authentication. Never through a proxy; a failure is retried after ``RETRY_S``."""

    label = "LibreHardwareMonitor"
    #: metrics LHM replaces even when an earlier source gave them: the GPU's own (the driver's GPU busy % and
    #: memory, as AMD's and NVIDIA's tools show them) beat the performance counters' per-process sums
    PREFERRED = ("gpu_name", "gpu_percent", "gpu_mem_used_mb", "gpu_mem_total_mb")

    def __init__(self, url: str = DEFAULT_LHM_URL, timeout: float = HTTP_TIMEOUT_S, retry_s: float = RETRY_S,
                 fetch: Callable[[], object] | None = None):
        self.url = (url or DEFAULT_LHM_URL).strip().rstrip("/")
        if self.url.endswith("/data.json"):
            self.url = self.url[: -len("/data.json")]
        self.timeout, self.retry_s = timeout, retry_s
        self._fetch = fetch or self._http
        self._retry_at = 0.0
        self.error = ""

    def display_url(self) -> str:
        """The URL without a password."""
        return re.sub(r"//[^/@]*@", "//", self.url)

    def _http(self):
        import base64
        from urllib import request as urlrequest
        from urllib.parse import unquote, urlsplit

        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"not an http URL: {self.display_url()}")
        host = parts.hostname or ""
        netloc = f"[{host}]" if ":" in host else host
        if parts.port:
            netloc += f":{parts.port}"
        req = urlrequest.Request(f"{parts.scheme}://{netloc}{parts.path}/data.json",
                                 headers={"User-Agent": "QuantaBench sensors"})
        if parts.username:
            token = f"{unquote(parts.username)}:{unquote(parts.password or '')}".encode()
            req.add_header("Authorization", "Basic " + base64.b64encode(token).decode("ascii"))
        opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as resp:
            raw = resp.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("reply larger than 16 MB")
        return json.loads(raw.decode("utf-8-sig"))

    def _explain(self, exc: BaseException) -> str:
        from urllib.error import HTTPError, URLError

        where = self.display_url()
        if isinstance(exc, HTTPError) and exc.code == 401:
            return (f"LibreHardwareMonitor's web server at {where} asks for a user name and password: put them in "
                    "this machine's sensors URL (http://user:password@127.0.0.1:8085)")
        if isinstance(exc, (URLError, OSError, TimeoutError)):
            reason = getattr(exc, "reason", exc)
            return f"LibreHardwareMonitor's web server is not answering at {where} ({reason}): {LHM_ENABLE}"
        return f"LibreHardwareMonitor's web server at {where} gave no usable data ({type(exc).__name__}: {exc})"

    def read(self, s: Sample) -> None:
        if time.monotonic() < self._retry_at:
            self._note(s)
            return
        try:
            tree = self._fetch()
            got = parse_lhm(tree)
        except Exception as exc:  # noqa: BLE001
            self.error = self._explain(exc)
            self._retry_at = time.monotonic() + self.retry_s
            self._note(s)
            return
        self.error = ""
        notes = got.pop("_notes", {})
        gpu_complete = "gpu_percent" in got and "gpu_mem_used_mb" in got
        for key, value in got.items():
            s.put(key, value, self.label, force=gpu_complete and key in self.PREFERRED)
        for key, why in notes.items():
            s.why(key, why)

    def _note(self, s: Sample) -> None:
        for key in ("cpu_temp_c", "cpu_power_w", "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w", "fan_rpm"):
            s.why(key, self.error)


# -------------------------------------------------------------------------------- Windows counters
#: counter key -> English counter path (PdhAddEnglishCounterW: the same on every Windows language)
PDH_COUNTERS = {
    "base": r"\Processor Information(_Total)\Processor Frequency",
    "perf": r"\Processor Information(_Total)\% Processor Performance",
    "perf_all": r"\Processor Information(*)\% Processor Performance",
    "gpu_engine": r"\GPU Engine(*)\Utilization Percentage",
    "gpu_memory": r"\GPU Adapter Memory(*)\Dedicated Usage",
}
_PDH_CPU = re.compile(r"^(\d+),(\d+)$")
_PDH_ENGINE = re.compile(r"luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_(\d+)_eng_(\d+)")
_PDH_ADAPTER = re.compile(r"luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_(\d+)")


def pdh_metrics(values: dict) -> dict:
    """Metrics from formatted counter values: {"base": MHz, "perf": %, "perf_all": {instance: %}, "gpu_engine":
    {instance: %}, "gpu_memory": {instance: bytes}} (any may be missing). Effective clock = base x performance / 100
    (Task Manager's "Speed"); per logical CPU from the "group,index" instances. GPU utilization: per adapter and
    engine the sum over processes, the busiest engine of the adapter with the most dedicated memory in use."""
    out: dict = {}
    base = _finite(values.get("base"))
    if base and base > 0:
        out["cpu_base_mhz"] = base
        perf = _finite(values.get("perf"))
        if perf is not None and perf > 0:
            out["cpu_clock_mhz"] = base * perf / 100.0
        per = []
        for name, v in (values.get("perf_all") or {}).items():
            m = _PDH_CPU.match(str(name))
            v = _finite(v)
            if m and v is not None:
                per.append(((int(m.group(1)), int(m.group(2))), base * v / 100.0))
        if per:
            clocks = [c for _k, c in sorted(per)]
            out["cpu_clock_per_mhz"] = clocks
            out["cpu_clock_min_mhz"], out["cpu_clock_max_mhz"] = min(clocks), max(clocks)
            out.setdefault("cpu_clock_mhz", sum(clocks) / len(clocks))
    engines: dict[tuple[str, str, str], float] = {}
    for name, v in (values.get("gpu_engine") or {}).items():
        m = _PDH_ENGINE.search(str(name))
        v = _finite(v)
        if m and v is not None:
            key = (m.group(1).lower(), m.group(2), m.group(3))
            engines[key] = engines.get(key, 0.0) + v
    busy: dict[str, float] = {}
    for (luid, _phys, _eng), v in engines.items():
        busy[luid] = max(busy.get(luid, 0.0), v)
    dedicated: dict[str, float] = {}
    for name, v in (values.get("gpu_memory") or {}).items():
        m = _PDH_ADAPTER.search(str(name))
        v = _finite(v)
        if m and v is not None:
            dedicated[m.group(1).lower()] = dedicated.get(m.group(1).lower(), 0.0) + v
    pick = max(dedicated, key=dedicated.get) if dedicated else (max(busy, key=busy.get) if busy else None)
    if pick is not None:
        if pick in busy:
            out["gpu_percent"] = min(100.0, busy[pick])
        if pick in dedicated:
            out["gpu_mem_used_mb"] = dedicated[pick] / 2**20
    return out


def windows_gpu_registry() -> tuple[str, float | None]:
    """(name, memory MB) of the display adapter with the most memory, from the registry (no DXGI, no torch)."""
    import winreg

    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    best: tuple[str, float | None] = ("", None)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as key:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(key, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(key, sub) as k:
                    name = str(winreg.QueryValueEx(k, "DriverDesc")[0])
                    try:
                        mem = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0]
                    except OSError:
                        mem = winreg.QueryValueEx(k, "HardwareInformation.MemorySize")[0]
            except OSError:
                continue
            if isinstance(mem, bytes):
                mem = int.from_bytes(mem[:8], "little")
            mb = float(mem) / 2**20 if isinstance(mem, int) and mem > 0 else None
            if mb and (best[1] is None or mb > best[1]):
                best = (name, mb)
    return best


class WindowsCounters(Provider):
    """The Windows performance counters (PDH) through ctypes: one query kept open, collected once per read (rate
    counters such as % Processor Performance are averaged over the time since the previous read)."""

    label = "Windows performance counters"
    PDH_FMT_DOUBLE = 0x00000200
    PDH_FMT_NOCAP100 = 0x00008000
    PDH_MORE_DATA = 0x800007D2

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes

        class _Value(ctypes.Union):
            _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double),
                        ("largeValue", ctypes.c_longlong), ("ptr", ctypes.c_void_p)]

        class FmtValue(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("u", _Value)]

        class FmtItem(ctypes.Structure):
            _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", FmtValue)]

        self.FmtValue, self.FmtItem = FmtValue, FmtItem
        pdh = ctypes.WinDLL("pdh.dll")
        handle = ctypes.c_void_p
        pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(handle)]
        pdh.PdhAddEnglishCounterW.argtypes = [handle, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(handle)]
        pdh.PdhCollectQueryData.argtypes = [handle]
        pdh.PdhGetFormattedCounterValue.argtypes = [handle, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                    ctypes.POINTER(FmtValue)]
        pdh.PdhGetFormattedCounterArrayW.argtypes = [handle, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        pdh.PdhCloseQuery.argtypes = [handle]
        for fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhGetFormattedCounterValue",
                   "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(pdh, fn).restype = ctypes.c_ulong
        self.pdh = pdh
        self.query = handle()
        st = pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query))
        if st:
            raise OSError(f"PdhOpenQuery failed (0x{st:08x})")
        self.counters: dict[str, object] = {}
        self.failed: dict[str, str] = {}
        for key, path in PDH_COUNTERS.items():
            h = handle()
            st = pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(h))
            if st:
                self.failed[key] = f"{path}: not available (0x{st:08x})"
            else:
                self.counters[key] = h
        pdh.PdhCollectQueryData(self.query)
        self._t = time.monotonic()
        self._first = True
        self._gpu_reg: tuple[str, float | None] | None = None

    def _value(self, h) -> float | None:
        ct = self._ct
        v = self.FmtValue()
        st = self.pdh.PdhGetFormattedCounterValue(h, self.PDH_FMT_DOUBLE | self.PDH_FMT_NOCAP100, None, ct.byref(v))
        if st or v.CStatus not in (0, 1):
            return None
        return v.u.doubleValue

    def _array(self, h) -> dict[str, float]:
        ct = self._ct
        from ctypes import wintypes

        size, count = wintypes.DWORD(0), wintypes.DWORD(0)
        fmt = self.PDH_FMT_DOUBLE | self.PDH_FMT_NOCAP100
        st = self.pdh.PdhGetFormattedCounterArrayW(h, fmt, ct.byref(size), ct.byref(count), None)
        if st != self.PDH_MORE_DATA or not size.value:
            return {}
        buf = ct.create_string_buffer(size.value)
        st = self.pdh.PdhGetFormattedCounterArrayW(h, fmt, ct.byref(size), ct.byref(count), buf)
        if st:
            return {}
        items = ct.cast(buf, ct.POINTER(self.FmtItem))
        out = {}
        for i in range(count.value):
            it = items[i]
            if it.FmtValue.CStatus in (0, 1) and it.szName:
                out[it.szName] = it.FmtValue.u.doubleValue
        return out

    def read(self, s: Sample) -> None:
        wait = 0.2 - (time.monotonic() - self._t)
        if self._first and wait > 0:     # rate counters need two collections some time apart
            time.sleep(wait)
        self._first = False
        self.pdh.PdhCollectQueryData(self.query)
        self._t = time.monotonic()
        values: dict = {}
        for key, h in self.counters.items():
            values[key] = self._array(h) if key.endswith(("_all", "engine", "memory")) else self._value(h)
        got = pdh_metrics(values)
        if "gpu_percent" in got or "gpu_mem_used_mb" in got:
            if self._gpu_reg is None:
                try:
                    self._gpu_reg = windows_gpu_registry()
                except OSError:
                    self._gpu_reg = ("", None)
            name, total = self._gpu_reg
            if name:
                got.setdefault("gpu_name", name)
            if total:
                got.setdefault("gpu_mem_total_mb", total)
        for key, value in got.items():
            s.put(key, value, self.label)
        if "base" in self.failed or "perf" in self.failed:
            s.why("cpu_clock_mhz", "; ".join(self.failed.get(k, "") for k in ("base", "perf") if k in self.failed))
        if "gpu_engine" in self.failed:
            s.why("gpu_percent", f"no GPU Engine counters ({self.failed['gpu_engine']}; a WDDM 2.0 or later "
                                 "display driver has them)")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.pdh.PdhCloseQuery(self.query)


# ---------------------------------------------------------------------------------------- macmon
MACMON_PATHS = ("/opt/homebrew/bin/macmon", "/usr/local/bin/macmon")


def find_macmon() -> str | None:
    """macmon on PATH, else Homebrew's folders (an app started from the Finder has no Homebrew PATH)."""
    exe = shutil.which("macmon")
    if exe:
        return exe
    return next((p for p in MACMON_PATHS if os.access(p, os.X_OK)), None)


def _pair(obj: dict, name: str) -> tuple[float | None, float | None]:
    """(MHz, utilization 0-1) of a macmon cluster: ``"<name>_usage": [MHz, fraction]`` (macmon 0.3-0.6), or
    ``<name>_freq_mhz`` with ``<name>_usage`` / ``<name>_usage_percent`` as separate fields."""
    v = obj.get(f"{name}_usage")
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return _finite(v[0]), _finite(v[1])
    f = _finite(obj.get(f"{name}_freq_mhz", obj.get(f"{name}_freq")))
    u = _finite(v)
    if u is None:
        u = _finite(obj.get(f"{name}_usage_percent"))
        u = None if u is None else u / 100.0
    return f, u


def _core_list(obj: dict, prefix: str) -> list[float]:
    """Per-core clocks of a cluster if this macmon version gives them (a list of MHz, or of [MHz, fraction])."""
    for key in (f"{prefix}_freqs_mhz", f"{prefix}_cores_freq_mhz", f"{prefix}_cores"):
        v = obj.get(key)
        if isinstance(v, list) and v:
            out = []
            for x in v:
                f = _finite(x[0]) if isinstance(x, (list, tuple)) and x else _finite(
                    x.get("freq_mhz") if isinstance(x, dict) else x)
                if f is None:
                    return []
                out.append(f)
            return out
    return []


def parse_macmon(obj, p_cores: int = 0, e_cores: int = 0) -> dict:
    """The metrics of one ``macmon pipe`` JSON line. ``temp.cpu_temp_avg``/``gpu_temp_avg`` (°C; 0 = not read),
    the E/P clusters' and the GPU's (MHz, utilization), ``cpu_power``/``gpu_power`` (W), ``memory.ram_usage`` /
    ``ram_total`` (bytes). The effective CPU clock is the clusters' clocks weighted by their cores (``p_cores``,
    ``e_cores``; equal weights when unknown); min/max over the clusters or, when given, the cores."""
    out: dict = {}
    if not isinstance(obj, dict):
        return out
    temp = obj.get("temp") if isinstance(obj.get("temp"), dict) else {}
    for key, src in (("cpu_temp_c", "cpu_temp_avg"), ("gpu_temp_c", "gpu_temp_avg")):
        v = _finite(temp.get(src))
        if v is not None and v > 0:
            out[key] = v
    mem = obj.get("memory") if isinstance(obj.get("memory"), dict) else {}
    used, total = _finite(mem.get("ram_usage")), _finite(mem.get("ram_total"))
    if used is not None and total:
        out["ram_used_gb"], out["ram_total_gb"] = used / 2**30, total / 2**30
    (ef, _eu), (pf, _pu), (gf, gu) = _pair(obj, "ecpu"), _pair(obj, "pcpu"), _pair(obj, "gpu")
    cores = _core_list(obj, "ecpu") + _core_list(obj, "pcpu")
    clusters = [(f, n) for f, n in ((ef, e_cores), (pf, p_cores)) if f is not None and f > 0]
    if cores:
        out["cpu_clock_per_mhz"] = cores
        out["cpu_clock_mhz"] = sum(cores) / len(cores)
        out["cpu_clock_min_mhz"], out["cpu_clock_max_mhz"] = min(cores), max(cores)
    elif clusters:
        weights = [n if all(m > 0 for _f, m in clusters) else 1 for _f, n in clusters]
        out["cpu_clock_mhz"] = sum(f * w for (f, _n), w in zip(clusters, weights, strict=True)) / sum(weights)
        out["cpu_clock_min_mhz"] = min(f for f, _n in clusters)
        out["cpu_clock_max_mhz"] = max(f for f, _n in clusters)
    if gf is not None and gf > 0:
        out["gpu_clock_mhz"] = gf
    if gu is not None:
        out["gpu_percent"] = max(0.0, min(100.0, gu * 100.0 if gu <= 1.0 else gu))
    for key, src in (("cpu_power_w", "cpu_power"), ("gpu_power_w", "gpu_power")):
        v = _finite(obj.get(src))
        if v is not None:
            out[key] = v
    return out


def _cluster_cores() -> tuple[int, int]:
    from . import resources

    p = resources._sysctl_int("hw.perflevel0.physicalcpu") or 0   # noqa: SLF001
    e = resources._sysctl_int("hw.perflevel1.physicalcpu") or 0   # noqa: SLF001
    return p, e


class Macmon(Provider):
    """macmon's JSON lines. ``continuous``: one ``macmon pipe -i <ms>`` process kept running and its last line used
    (a Monitor); otherwise ``macmon pipe -s 1`` once per read."""

    label = "macmon"

    def __init__(self, continuous: bool = False, interval_s: float = DEFAULT_INTERVAL_S, exe: str | None = None):
        self.exe = exe or find_macmon()
        self.continuous = continuous
        self.interval_ms = max(200, int(interval_s * 1000))
        self.proc: subprocess.Popen | None = None
        self._latest: tuple[float, dict] | None = None
        self._lock = threading.Lock()
        self._retry_at = 0.0
        self.error = ""
        self._cores: tuple[int, int] | None = None

    def _parse(self, text: str) -> dict | None:
        try:
            obj = json.loads(text)
        except ValueError:
            return None
        if self._cores is None:
            try:
                self._cores = _cluster_cores()
            except Exception:  # noqa: BLE001
                self._cores = (0, 0)
        return parse_macmon(obj, *self._cores)

    def _once(self) -> dict | None:
        out = subprocess.run([self.exe, "pipe", "-s", "1", "-i", "250"], capture_output=True, text=True,  # noqa: S603
                             timeout=3.0, check=False)
        lines = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("{")]
        if not lines:
            raise RuntimeError(f"macmon gave no data (exit {out.returncode}: {out.stderr.strip()[:200]})")
        return self._parse(lines[-1])

    def _start(self) -> None:
        self.proc = subprocess.Popen([self.exe, "pipe", "-i", str(self.interval_ms)],  # noqa: S603
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        threading.Thread(target=self._pump, args=(self.proc,), name="quantabench-macmon", daemon=True).start()

    def _pump(self, proc) -> None:
        with contextlib.suppress(Exception):
            for raw in proc.stdout:
                got = self._parse(raw.strip()) if raw.strip().startswith("{") else None
                if got is not None:
                    with self._lock:
                        self._latest = (time.monotonic(), got)

    def read(self, s: Sample) -> None:
        if not self.exe:
            for key in ("cpu_clock_mhz", "cpu_temp_c", "cpu_power_w", "gpu_percent", "gpu_temp_c", "gpu_clock_mhz",
                        "gpu_power_w"):
                s.why(key, MACMON_INSTALL)
            return
        if time.monotonic() < self._retry_at:
            self._note(s)
            return
        got = None
        try:
            if not self.continuous:
                got = self._once()
            else:
                if self.proc is None or self.proc.poll() is not None:
                    self._start()
                    deadline = time.monotonic() + min(1.0, self.interval_ms / 1000 + 0.6)
                    while time.monotonic() < deadline and self._latest is None and self.proc.poll() is None:
                        time.sleep(0.05)
                with self._lock:
                    latest = self._latest
                if latest is None and self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(f"macmon pipe exited (code {self.proc.returncode}) without data")
                if latest is not None and time.monotonic() - latest[0] < 3 * self.interval_ms / 1000 + 2:
                    got = latest[1]
        except Exception as exc:  # noqa: BLE001
            self.error = f"macmon failed ({type(exc).__name__}: {exc})"
            self._retry_at = time.monotonic() + RETRY_S
            self.close()
        if got is None:
            self.error = self.error or "macmon has not reported yet"
            self._note(s)
            return
        self.error = ""
        for key, value in got.items():
            s.put(key, value, self.label)

    def _note(self, s: Sample) -> None:
        for key in ("cpu_clock_mhz", "cpu_temp_c", "cpu_power_w", "gpu_temp_c", "gpu_power_w"):
            s.why(key, self.error)

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=2)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()
        with self._lock:
            self._latest = None


# ------------------------------------------------------------------------------------------- Linux
TEMP_CHIPS = (("k10temp", ("Tctl", "Tdie", "Tccd1")), ("zenpower", ("Tdie", "Tctl")),
              ("coretemp", ("Package id 0",)), ("cpu_thermal", ("",)), ("soc_thermal", ("",)),
              ("x86_pkg_temp", ("",)), ("acpitz", ("",)))


def pick_cpu_temp(temps: dict) -> tuple[float | None, str]:
    """(°C, "chip label") from ``psutil.sensors_temperatures()``: k10temp Tctl, zenpower Tdie, coretemp Package id 0,
    an ARM SoC zone, then ACPI's."""
    for chip, labels in TEMP_CHIPS:
        entries = temps.get(chip) or []
        for want in labels:
            for e in entries:
                if (getattr(e, "label", "") or "") == want or (not want and e is entries[0]):
                    v = _finite(getattr(e, "current", None))
                    if v is not None and v > 0:
                        return v, f"{chip} {want}".strip()
    return None, ""


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_num(path: Path) -> float | None:
    text = _read(path)
    try:
        return _finite(float(text)) if text is not None else None
    except ValueError:
        return None


def read_amdgpu(drm: Path = Path("/sys/class/drm")) -> dict:
    """amdgpu's sysfs files of the AMD GPU with the most VRAM: gpu_busy_percent, mem_info_vram_used/total (bytes),
    hwmon temp1 (edge, m°C), power1_average or power1_input (µW), freq1 (sclk, Hz)."""
    best: dict = {}
    try:
        cards = sorted(p for p in drm.glob("card[0-9]*") if "-" not in p.name)
    except OSError:
        return {}
    for card in cards:
        dev = card / "device"
        if (_read(dev / "vendor") or "").lower() != "0x1002":
            continue
        got: dict = {"gpu_name": _read(dev / "product_name") or "AMD GPU (amdgpu)"}
        busy = _read_num(dev / "gpu_busy_percent")
        if busy is not None:
            got["gpu_percent"] = busy
        used, total = _read_num(dev / "mem_info_vram_used"), _read_num(dev / "mem_info_vram_total")
        if used is not None:
            got["gpu_mem_used_mb"] = used / 2**20
        if total:
            got["gpu_mem_total_mb"] = total / 2**20
        for hw in sorted((dev / "hwmon").glob("hwmon*")) if (dev / "hwmon").is_dir() else []:
            t = _read_num(hw / "temp1_input")
            if t is not None:
                got["gpu_temp_c"] = t / 1000.0
            p = _read_num(hw / "power1_average")
            if p is None:
                p = _read_num(hw / "power1_input")
            if p is not None:
                got["gpu_power_w"] = p / 1e6
            f = _read_num(hw / "freq1_input")
            if f is not None:
                got["gpu_clock_mhz"] = f / 1e6
            break
        if got.get("gpu_mem_total_mb", 0) >= best.get("gpu_mem_total_mb", -1):
            best = got
    return best


class LinuxSysfs(Provider):
    """Linux: cpufreq clocks (psutil), CPU temperature (psutil hwmon), fans, RAPL package power, amdgpu."""

    label = "Linux sysfs"

    def __init__(self, root: Path = Path("/sys"), wsl: bool = False):
        self.root = root
        self.wsl = wsl
        self._rapl: tuple[float, float] | None = None       # (monotonic s, energy µJ)
        self._rapl_why = ""

    def read(self, s: Sample) -> None:
        import psutil

        if self.wsl:          # the reasons that matter there come first (Sample.why: the first one wins)
            for key, why in DEFAULT_WHY["wsl"].items():
                s.why(key, why)
        cpufreq = self.root / "devices/system/cpu/cpu0/cpufreq"
        if cpufreq.is_dir():
            freqs = [f.current for f in (psutil.cpu_freq(percpu=True) or []) if _finite(f.current)]
            if freqs:
                s.put("cpu_clock_per_mhz", freqs, self.label)
                s.put("cpu_clock_mhz", sum(freqs) / len(freqs), self.label)
                s.put("cpu_clock_min_mhz", min(freqs), self.label)
                s.put("cpu_clock_max_mhz", max(freqs), self.label)
            base = _read_num(cpufreq / "base_frequency")
            if base:
                s.put("cpu_base_mhz", base / 1000.0, self.label)
            s.why("cpu_base_mhz", "the kernel's cpufreq driver does not publish a base frequency")
        else:
            for key in ("cpu_clock_mhz", "cpu_base_mhz"):
                s.why(key, "no cpufreq in /sys (a virtual machine or WSL2: the host knows the clocks)")
        temps = {}
        if hasattr(psutil, "sensors_temperatures"):
            with contextlib.suppress(Exception):
                temps = psutil.sensors_temperatures() or {}
        t, _where = pick_cpu_temp(temps)
        if t is not None:
            s.put("cpu_temp_c", t, self.label)
        else:
            s.why("cpu_temp_c", "no CPU temperature sensor in /sys/class/hwmon (load the k10temp or coretemp "
                                "module; a virtual machine has none)")
        if hasattr(psutil, "sensors_fans"):
            with contextlib.suppress(Exception):
                rpm = [f.current for fs in (psutil.sensors_fans() or {}).values() for f in fs if _finite(f.current)]
                if rpm and max(rpm) > 0:
                    s.put("fan_rpm", max(rpm), self.label)
        self._read_rapl(s)
        for key, value in read_amdgpu(self.root / "class/drm").items():
            s.put(key, value, "amdgpu sysfs")

    def _read_rapl(self, s: Sample) -> None:
        path = self.root / "class/powercap/intel-rapl:0/energy_uj"
        if not path.exists():
            s.why("cpu_power_w", "no RAPL energy counter in /sys/class/powercap")
            return
        try:
            energy = float(path.read_text().strip())
        except PermissionError:
            s.why("cpu_power_w", "the RAPL energy counter (/sys/class/powercap/intel-rapl:0/energy_uj) is readable "
                                 "by root only on this system")
            return
        except (OSError, ValueError) as exc:
            s.why("cpu_power_w", f"RAPL energy counter unreadable ({exc})")
            return
        now = time.monotonic()
        last, self._rapl = self._rapl, (now, energy)
        if last is not None and now > last[0] and energy >= last[1]:
            s.put("cpu_power_w", (energy - last[1]) / 1e6 / (now - last[0]), "RAPL")
        else:
            s.why("cpu_power_w", "RAPL needs two readings (next sample)")


class NvidiaSmi(Provider):
    """``nvidia-smi --query-gpu`` (NVIDIA GPUs on Linux, WSL2 and Windows)."""

    label = "nvidia-smi"
    QUERY = "name,utilization.gpu,temperature.gpu,clocks.sm,power.draw,memory.used,memory.total"

    def __init__(self, exe: str | None = None):
        self.exe = exe
        self._retry_at = 0.0

    @staticmethod
    def find() -> str | None:
        return shutil.which("nvidia-smi") or next(
            (p for p in ("/usr/lib/wsl/lib/nvidia-smi",) if os.access(p, os.X_OK)), None)

    @staticmethod
    def parse(text: str) -> dict:
        best: dict = {}
        for row in text.splitlines():
            cells = [c.strip() for c in row.split(",")]
            if len(cells) < 7:
                continue

            def num(x: str) -> float | None:
                try:
                    return _finite(float(x))
                except ValueError:
                    return None

            got = {"gpu_name": cells[0]}
            for key, cell in zip(("gpu_percent", "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w", "gpu_mem_used_mb",
                                  "gpu_mem_total_mb"), cells[1:7], strict=True):
                v = num(cell)
                if v is not None:
                    got[key] = v
            if got.get("gpu_mem_total_mb", 0) > best.get("gpu_mem_total_mb", -1):
                best = got
        return best

    def read(self, s: Sample) -> None:
        if not self.exe or time.monotonic() < self._retry_at:
            return
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            out = subprocess.run([self.exe, f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],  # noqa: S603
                                 capture_output=True, text=True, timeout=1.5, check=False, creationflags=flags)
            got = self.parse(out.stdout) if out.returncode == 0 else {}
        except (OSError, subprocess.SubprocessError):
            got = {}
        if not got:
            self._retry_at = time.monotonic() + 60.0
            return
        for key, value in got.items():
            s.put(key, value, self.label)


# ------------------------------------------------------------------------------------------ reader
def _platform() -> str:
    """"windows", "macos", "wsl" or "linux" (a function, so tests can fake another OS)."""
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    from . import resources

    return "wsl" if resources.is_wsl() else "linux"


def default_providers(config: SensorConfig, continuous: bool = False, kind: str | None = None) -> list[Provider]:
    """The providers of this OS, in order of precedence (the first to give a value wins). One that cannot start
    (no pdh.dll, ...) is left out; ``Reader`` says why."""
    kind = kind or _platform()
    makers: list[tuple[str, Callable[[], Provider]]] = [("psutil", CpuLoad)]
    if kind == "windows":
        makers += [("Windows performance counters", WindowsCounters),
                   ("LibreHardwareMonitor", lambda: LibreHardwareMonitor(config.lhm_url))]
    elif kind == "macos":
        makers += [("macmon", lambda: Macmon(continuous, config.interval_s))]
    else:
        makers += [("Linux sysfs", lambda: LinuxSysfs(wsl=kind == "wsl"))]
        if kind == "wsl":
            makers += [("LibreHardwareMonitor", lambda: LibreHardwareMonitor(config.lhm_url))]
    smi = NvidiaSmi.find() if kind != "macos" else None
    if smi:
        makers += [("nvidia-smi", lambda: NvidiaSmi(smi))]
    out = []
    for _name, make in makers:
        try:
            out.append(make())
        except Exception as exc:  # noqa: BLE001
            out.append(_Broken(_name, f"{type(exc).__name__}: {exc}"))
    return out


class _Broken(Provider):
    def __init__(self, label: str, error: str):
        self.label, self.error = label, error

    def read(self, s: Sample) -> None:
        s.missing.setdefault("_" + self.label, f"{self.label} unavailable ({self.error})")


#: why a metric has no value when no provider said (per OS)
DEFAULT_WHY = {
    "windows": {k: f"needs LibreHardwareMonitor's web server: {LHM_ENABLE}"
                for k in ("cpu_temp_c", "cpu_power_w", "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w", "fan_rpm")},
    "macos": {**{k: MACMON_INSTALL for k in ("cpu_clock_mhz", "cpu_temp_c", "cpu_power_w", "gpu_percent",
                                             "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w")},
              "cpu_base_mhz": "Apple Silicon has no fixed base clock",
              "gpu_mem_used_mb": "Apple Silicon's GPU shares the RAM (unified memory): see RAM",
              "fan_rpm": "not read on macOS"},
    "wsl": {k: ("WSL2 has no hardware sensors: temperatures, clocks and power are on the Windows side (QuantaBench "
                "on Windows reads them; or point this machine's sensors URL at LibreHardwareMonitor on Windows)")
            for k in ("cpu_clock_mhz", "cpu_base_mhz", "cpu_temp_c", "cpu_power_w", "gpu_temp_c", "gpu_power_w",
                      "fan_rpm")},
    "linux": {},
}


GPU_KEYS = ("gpu_percent", "gpu_temp_c", "gpu_clock_mhz", "gpu_power_w", "gpu_mem_used_mb")
NO_GPU = {"linux": "no GPU sensors found (no amdgpu card in /sys/class/drm, no nvidia-smi)",
          "wsl": "no GPU sensors found in WSL2 (nvidia-smi only; AMD GPUs are read on the Windows side)",
          "windows": "no GPU sensors found (no GPU Engine counters, LibreHardwareMonitor not answering)"}


class Reader:
    """Reads this machine's sensors through ``providers`` (``default_providers``). One reader per user: its CPU
    utilization is the time since its previous read. Not thread-safe; ``read()`` (module) and ``Monitor`` lock."""

    def __init__(self, config: SensorConfig | None = None, continuous: bool = False,
                 providers: list[Provider] | None = None, kind: str | None = None):
        self.config = config or load_config()
        self.kind = kind or _platform()
        self.providers = providers if providers is not None else (
            default_providers(self.config, continuous, self.kind) if self.config.enabled else _off_providers())
        self._machine = ""
        self._cpu_name: str | None = None

    def _machine_name(self) -> str:
        if not self._machine:
            try:
                from .dist import profile

                self._machine = profile.machine_name()
            except Exception:  # noqa: BLE001
                import socket

                self._machine = socket.gethostname().lower()
        return self._machine

    def read(self) -> Sample:
        s = Sample(t=time.time(), machine=self._machine_name(), os=self.kind)
        used = []
        for p in self.providers:
            before = len(s.sources)
            try:
                p.read(s)
            except Exception as exc:  # noqa: BLE001
                s.missing.setdefault("_" + p.label, f"{p.label} failed ({type(exc).__name__}: {exc})")
            if len(s.sources) > before and p.label not in used:
                used.append(p.label)
        for src in s.sources.values():
            if src not in used:
                used.append(src)
        s.source = " + ".join(used) or "none"
        if not self.config.enabled:
            for key, _label in METRICS:
                if not s.has(key):
                    s.why(key, "sensors are turned off (settings sensors.enabled, or QUANTABENCH_SENSORS=0)")
        for key, why in DEFAULT_WHY.get(self.kind, {}).items():
            if not s.has(key):
                s.why(key, why)
        if not any(s.has(k) for k in GPU_KEYS):
            for key in GPU_KEYS:
                s.why(key, NO_GPU.get(self.kind, "no GPU sensors found"))
        for key, _label in METRICS:
            if s.has(key):
                s.missing.pop(key, None)
            else:
                s.why(key, f"not reported by {s.source}")
        if not s.cpu_name:
            if self._cpu_name is None:
                self._cpu_name = ""
                with contextlib.suppress(Exception):
                    from . import resources

                    self._cpu_name = resources.cpu_name()
            s.cpu_name = self._cpu_name
        return s

    def close(self) -> None:
        for p in self.providers:
            with contextlib.suppress(Exception):
                p.close()


def _off_providers() -> list[Provider]:
    """Sensors turned off: utilization and RAM still come from psutil (cheap, no other program involved)."""
    try:
        return [CpuLoad()]
    except Exception as exc:  # noqa: BLE001
        return [_Broken("psutil", str(exc))]


_shared: Reader | None = None
_shared_lock = threading.Lock()
_last: Sample | None = None


def read(config: SensorConfig | None = None) -> Sample:
    """This machine's sensors now (the shared reader: its CPU utilization and Windows clocks are averaged since its
    previous call). Never raises; when another thread is mid-read for over a second, its last sample is returned."""
    global _shared, _last
    try:
        if not _shared_lock.acquire(timeout=1.0):
            return replace(_last) if _last is not None else Sample(t=time.time(), source="busy")
        try:
            if _shared is None:
                _shared = Reader(config)
            s = _shared.read()
            _last = s
            return s
        finally:
            _shared_lock.release()
    except Exception as exc:  # noqa: BLE001
        return Sample(t=time.time(), source="error", missing={"_reader": f"sensors failed ({type(exc).__name__}: {exc})"})


def reset() -> None:
    """Close the shared reader (tests; a settings change)."""
    global _shared, _last
    with _shared_lock:
        if _shared is not None:
            _shared.close()
        _shared, _last = None, None


# ----------------------------------------------------------------------------------------- monitor
class Monitor:
    """Samples in a daemon thread every ``interval_s`` and keeps ``history_s`` of samples. ``latest()`` and
    ``history()`` never block (a GUI timer reads them). ``idle_stop_s``: with nobody asking for that long, the thread
    stops sampling and frees its providers (a worker's macmon) until the next ``latest()``."""

    def __init__(self, interval_s: float | None = None, history_s: float = HISTORY_S, idle_stop_s: float | None = None,
                 config: SensorConfig | None = None, reader_factory: Callable[[], Reader] | None = None):
        self.config = config or load_config()
        self.interval_s = max(0.5, float(interval_s or self.config.interval_s or DEFAULT_INTERVAL_S))
        self.idle_stop_s = idle_stop_s
        self._factory = reader_factory or (lambda: Reader(self.config, continuous=True))
        self._history: deque[Sample] = deque(maxlen=max(10, int(history_s / self.interval_s) + 2))
        self._lock = threading.Lock()
        self._halt = threading.Event()       # a new one per start, so a thread being stopped never runs on
        self._wake = threading.Event()
        self._new = threading.Event()
        self._asked = time.monotonic()
        self._thread: threading.Thread | None = None
        self._stopped = False                # stop() was called: latest() does not start it again
        self.error = ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._halt.is_set()

    def start(self) -> Monitor:
        self._stopped = False
        if not self.running:
            self._halt = threading.Event()
            self._asked = time.monotonic()
            self._thread = threading.Thread(target=self._run, args=(self._halt,), name="quantabench-sensors",
                                            daemon=True)
            self._thread.start()
            _LIVE.add(self)
        return self

    def stop(self, timeout: float = 3.0) -> None:
        """Stop sampling. ``timeout``: how long to wait for the thread (0: not at all, for a GUI thread; the thread
        then frees its providers, such as macmon, after its current read)."""
        self._stopped = True
        self._halt.set()
        self._wake.set()
        t, self._thread = self._thread, None
        if timeout > 0 and t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout)

    def latest(self, wake: bool = True, wait: float = 0.0) -> Sample | None:
        """The newest sample (None before the first). ``wake``: start the thread if it is not running (unless
        ``stop`` was called) and end an idle pause. ``wait``: seconds to wait for a first or fresh sample when the
        newest is older than two intervals (a monitor woken from idle)."""
        self._asked = time.monotonic()
        if wake:
            if not self.running and not self._stopped:
                self.start()
            self._wake.set()
        with self._lock:
            last = self._history[-1] if self._history else None
        if wait > 0 and (last is None or time.time() - last.t > 2 * self.interval_s + 1):
            self._new.clear()
            self._new.wait(wait)
            with self._lock:
                last = self._history[-1] if self._history else None
        return last

    def history(self) -> list[Sample]:
        with self._lock:
            return list(self._history)

    def _run(self, halt: threading.Event) -> None:
        reader = None
        try:
            while not halt.is_set():
                if self.idle_stop_s and time.monotonic() - self._asked > self.idle_stop_s:
                    if reader is not None:
                        reader.close()
                        reader = None
                    self._wake.clear()
                    self._wake.wait(5.0)
                    continue
                t0 = time.monotonic()
                try:
                    if reader is None:
                        reader = self._factory()
                    s = reader.read()
                    if halt.is_set():
                        break
                    with self._lock:
                        self._history.append(s)
                    self._new.set()
                    self.error = ""
                except Exception as exc:  # noqa: BLE001
                    self.error = f"{type(exc).__name__}: {exc}"
                # the interval is kept however often latest() is called (only stop() cuts it short)
                halt.wait(max(0.05, self.interval_s - (time.monotonic() - t0)))
        finally:
            if reader is not None:
                reader.close()


#: monitors started in this process, stopped at exit (a Mac's macmon must not outlive the app)
_LIVE: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _stop_all() -> None:
    for m in list(_LIVE):
        with contextlib.suppress(Exception):
            m.stop(timeout=1.0)


# ------------------------------------------------------------------------------------------ doctor
def doctor_lines(s: Sample | None = None) -> list[str]:
    """``quantabench-cli doctor``'s Sensors section: each metric with its value and source, or why it is missing
    and how to enable it."""
    s = s or read()
    lines = [f"  Sensors       {s.source} (live: `quantabench-cli monitor`)"]
    for key, label in METRICS:
        if s.has(key):
            v = getattr(s, key)
            unit, digits = {"cpu_percent": ("%", 0), "gpu_percent": ("%", 0), "cpu_temp_c": ("°C", 1),
                            "gpu_temp_c": ("°C", 1), "cpu_power_w": ("W", 1), "gpu_power_w": ("W", 1),
                            "fan_rpm": ("RPM", 0), "ram_used_gb": ("GB", 1), "gpu_mem_used_mb": ("MB", 0)
                            }.get(key, ("MHz", 0))
            lines.append(f"    OK  {label:26s} {v:,.{digits}f} {unit} ({s.sources.get(key, s.source)})")
        else:
            lines.append(f"    --  {label:26s} {s.missing.get(key, 'unavailable')}")
    for key, why in s.missing.items():
        if key.startswith("_"):
            lines.append(f"    !!  {why}")
    return lines
