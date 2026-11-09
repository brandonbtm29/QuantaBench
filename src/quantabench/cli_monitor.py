"""``quantabench-cli monitor``: this machine's live metrics, one compact line per sample (``quantabench.sensors``).

    quantabench-cli monitor                      # every 2 s until Ctrl+C
    quantabench-cli monitor --interval 1 --count 10
    quantabench-cli monitor --json --count 1     # one sample as JSON
    quantabench-cli monitor --worker btm-nova    # a worker's sensors (its /v1/sensors), over Tailscale
    quantabench-cli monitor --set lhm_url=http://127.0.0.1:8086     # this machine's sensor settings
    quantabench-cli monitor --set enabled=false --machine brandons-macbook-pro

``register`` adds the command to the parser of ``quantabench.cli``. ``doctor`` shows the same sources with what is
missing and how to enable it (``sensors.doctor_lines``).
"""

from __future__ import annotations

import json
import sys
import time

DESCRIPTION = ("Prints CPU utilization, the effective CPU clock (average, min-max over the cores), CPU temperature "
               "and package power, GPU utilization, clock, temperature, power and VRAM, and RAM, one line per sample. "
               "Windows: performance counters, plus LibreHardwareMonitor's web server for temperatures and power; "
               "macOS: macmon (brew install macmon); Linux: sysfs. `quantabench-cli doctor` says what is missing "
               "and how to enable it (docs/ENGINES.md, Live machine metrics).")


def register(sub) -> None:
    p = sub.add_parser("monitor", help="live CPU/GPU load, effective clock, temperatures, power and memory",
                       description=DESCRIPTION)
    p.add_argument("--interval", type=float, default=None,
                   help="seconds between samples (default from settings, 2)")
    p.add_argument("--count", type=int, default=0, help="stop after N samples (default: until Ctrl+C)")
    p.add_argument("--json", action="store_true", help="one JSON object per sample instead of a line")
    p.add_argument("--per-cpu", action="store_true", help="also print each logical CPU's utilization and clock")
    p.add_argument("--worker", metavar="NAME", help="show a worker's sensors (registry name or URL) instead")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="store a sensor setting for this machine (or --machine NAME): enabled=true|false, "
                        "interval_s=2, lhm_url=http://127.0.0.1:8085; KEY= removes it")
    p.add_argument("--machine", metavar="NAME", help="with --set: the machine (default: this one)")
    p.set_defaults(func=_monitor)


def _parse_value(key: str, text: str):
    from . import settings as settings_mod

    kind = settings_mod.SENSOR_KEYS.get(key)
    if kind is None:
        raise ValueError(f"unknown sensor setting {key!r} (one of {', '.join(settings_mod.SENSOR_KEYS)})")
    text = text.strip()
    if text == "":
        return None
    if kind is bool:
        low = text.lower()
        if low not in ("true", "false", "on", "off", "yes", "no", "1", "0"):
            raise ValueError(f"{key} must be true or false")
        return low in ("true", "on", "yes", "1")
    if kind is float:
        return float(text)
    return text


def _set(args) -> int:
    from . import settings as settings_mod

    s = settings_mod.load()
    machine = args.machine or None
    try:
        for item in args.set:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"--set {item!r}: use KEY=VALUE")
            s.set_sensors_for(key.strip(), _parse_value(key.strip(), value), machine)
    except ValueError as exc:
        print(f"monitor: {exc}", file=sys.stderr)
        return 2
    path = settings_mod.save(s)
    eff = s.sensors_for(machine)
    print(f"Saved {path}: {machine or settings_mod.this_machine()} sensors enabled={eff.enabled}, "
          f"interval_s={eff.interval_s:g}, lhm_url={eff.lhm_url}")
    return 0


def _print(s, args) -> None:
    from . import sensors

    if args.json:
        print(json.dumps(s.to_dict(per_cpu=args.per_cpu)), flush=True)
        return
    print(sensors.line(s), flush=True)
    if args.per_cpu and s.cpu_percent_per:
        clocks = s.cpu_clock_per_mhz if len(s.cpu_clock_per_mhz) == len(s.cpu_percent_per) else []
        cells = [f"{i}:{p:3.0f}%" + (f" {clocks[i] / 1000:.2f}" if clocks else "") for i, p in
                 enumerate(s.cpu_percent_per)]
        for i in range(0, len(cells), 8):
            print("          " + "  ".join(cells[i:i + 8]), flush=True)


def _remote(args, interval: float) -> int:
    from . import sensors
    from . import settings as settings_mod
    from .dist import client

    token = settings_mod.load().dist.worker_token
    if "://" in args.worker:
        try:
            ref = client.WorkerRef(args.worker, client.normalize_url(args.worker), "option")
        except ValueError as exc:
            print(f"monitor: {exc}", file=sys.stderr)
            return 2
    else:
        refs = client.known_workers()
        ref = next((r for k, r in refs.items() if settings_mod.machine_key(k) == settings_mod.machine_key(
            args.worker)), None)
    if ref is None:
        print(f"monitor: no worker {args.worker!r} in the registry or settings (`quantabench-cli machines`)",
              file=sys.stderr)
        return 2
    c = client.WorkerClient(ref.url, token, timeout=5.0)
    n = 0
    try:
        while True:
            try:
                data = c.sensors()
            except client.WorkerError as exc:
                if exc.status == 404:
                    print(f"monitor: {ref.name} is an older worker without /v1/sensors (update it)", file=sys.stderr)
                    return 1
                print(f"monitor: {exc}", file=sys.stderr)
                return 1
            s = sensors.Sample.from_dict(data.get("sensors"))
            if data.get("sensors") is None:
                print(f"{ref.name}: {data.get('why') or 'no sample yet'}", flush=True)
            else:
                if n == 0 and not args.json:
                    print(f"{ref.name}: {s.source}", flush=True)
                _print(s, args)
            n += 1
            if args.count and n >= args.count:
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def _monitor(args) -> int:
    from . import sensors

    if args.set:
        return _set(args)
    cfg = sensors.load_config()
    interval = max(0.5, float(args.interval or cfg.interval_s))
    if args.worker:
        return _remote(args, interval)
    m = sensors.Monitor(interval_s=interval, config=cfg).start()
    n = 0
    try:
        s = m.latest(wait=interval + 2.0)
        if s is None:
            print(f"monitor: no sample ({m.error or 'the sensors did not answer'})", file=sys.stderr)
            return 1
        if not args.json:
            print(f"{s.machine}: {s.source}" + ("" if cfg.enabled else " (sensors turned off in the settings)"))
            gaps = [f"{label}: {s.missing[k]}" for k, label in sensors.METRICS if k in s.missing]
            for g in dict.fromkeys(gaps[:4]):
                print(f"  -- {g}")
            if len(gaps) > 4:
                print(f"  -- and {len(gaps) - 4} more (quantabench-cli doctor)")
        last_t = 0.0
        while True:
            if s is not None and s.t != last_t:
                _print(s, args)
                last_t = s.t
                n += 1
                if args.count and n >= args.count:
                    return 0
            time.sleep(min(0.25, interval / 4))
            s = m.latest()
    except KeyboardInterrupt:
        return 0
    finally:
        m.stop()
