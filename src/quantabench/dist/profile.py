"""Machine profiles: what this computer can run, and how fast.

CONTRACT (implemented by the profile builder; field names are fixed, more may be added):

    MachineProfile (JSON round trip with to_dict/from_dict; unknown keys ignored)
    local_profile(bench: bool = False, methods=None) -> MachineProfile
        hardware from resources.detect(), engines from engines.all_engines()/available(), external
        programs from external.report(); with bench=True, throughput["<method>@<device>"] in molecules per
        hour on a fixed small PFAS set (see benchmark()).
    save_profile(p) -> Path      settings_dir()/machines/<name>.json (atomic write)
    load_profiles() -> list[MachineProfile]    every machine in the registry, newest first
    machine_name() -> str         stable, lowercase hostname (Tailscale MagicDNS-style)
    throughput_for(p, method, device="auto") -> float | None

The registry folder is synced (SyncThing), so it may hold stale files, half-written ones, SyncThing
conflict copies (``name.sync-conflict-....json``) and files written by newer or older QuantaBench versions.
``load_profiles`` skips what it cannot read, coerces wrong types to defaults, and keeps the newest profile
per machine name.

Throughput is measured on ``BENCH_SET`` (five PFAS anions) with a short protocol (``BENCH_PROTOCOL``: 4
conformers, the 2 best refined, no files, no xTB properties). The numbers are molecules per hour *for that
protocol*: a production run with 50 conformers is much slower, but the ratio between machines, which is
all ``dist.plan`` needs, carries over roughly.
"""

from __future__ import annotations

import json
import math
import platform
import socket
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

BENCH_VERSION = 1
#: (id, SMILES) of the benchmark molecules: PFAS anions as they occur at pH 7 (no protonation step needed).
BENCH_SET: tuple[tuple[str, str], ...] = (
    ("TFA", "[O-]C(=O)C(F)(F)F"),
    ("PFBA", "[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)F"),
    ("PFHxA", "[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
    ("PFOA", "[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
    ("PFBS", "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"),
)
#: BuildConfig overrides for the benchmark (the rest are BuildConfig defaults, not the user's settings).
BENCH_PROTOCOL = {"max_conformers": 4, "refine_top": 2, "protonate": False, "xtb_properties": False,
                  "write_files": False, "frequencies": False, "conformer_search": "rdkit", "solvent": "",
                  "qm_method": "", "seed": 42}
DEFAULT_BENCH_METHODS = ("mmff", "gfn1-xtb", "gfn2-xtb", "aimnet2")
DEFAULT_MAX_SECONDS = 60.0         # per method and device: two devices stay within ~2 minutes per method
MIN_SECONDS = 5.0                  # a fast engine repeats the set (up to MAX_PASSES times) until this long
MAX_PASSES = 20
STALE_DAYS = 30.0


@dataclass
class MachineProfile:
    name: str
    os: str = ""
    arch: str = ""
    python: str = ""
    quantabench_version: str = ""
    cpu: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    ram_gb: float = 0.0
    gpus: list[dict] = field(default_factory=list)       # {"index", "name", "memory_gb", "backend"}
    torch: str = ""                                       # torch version, "" if missing
    engines: dict[str, bool] = field(default_factory=dict)
    programs: dict[str, str] = field(default_factory=dict)   # program -> version ("" = missing)
    throughput: dict[str, float] = field(default_factory=dict)  # "<method>@<device>" -> molecules/hour
    address: str = ""                                     # worker URL, e.g. http://100.90.226.110:8765
    updated: str = ""                                     # ISO-8601 UTC
    # -- added by the profile builder
    commit: str = ""                                      # git commit of the checkout QuantaBench runs from
    wsl: bool = False                                     # this profile describes a WSL2 instance
    torch_backend: str = ""                               # "rocm" | "cuda" | "mps" | "cpu" | "none"
    wsl_programs: list[str] = field(default_factory=list)    # programs that run through WSL (``wsl:`` paths)
    benchmarks: dict[str, dict] = field(default_factory=dict)  # key -> {molecules, seconds, threads, date, ...}
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MachineProfile:
        """Profile from JSON. Unknown keys are ignored; a value of the wrong type (a partial or hand-edited
        file) falls back to the default, so one bad field never hides a machine."""
        if not isinstance(d, dict):
            raise TypeError("a machine profile must be a JSON object")
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("a machine profile needs a name")
        out = cls(name=name.strip())
        for f in fields(cls):
            if f.name == "name" or f.name not in d:
                continue
            ok, value = _coerce(getattr(out, f.name), d[f.name], f.name)
            if ok:
                setattr(out, f.name, value)
        return out


def _coerce(default, value, name: str) -> tuple[bool, object]:
    if isinstance(default, bool):
        return isinstance(value, bool), value
    if isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        return ok, float(value) if ok else value
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool), value
    if isinstance(default, str):
        return isinstance(value, str), value
    if isinstance(default, list):
        if not isinstance(value, list):
            return False, value
        if name == "gpus":
            return True, [g for g in value if isinstance(g, dict)]
        return True, [x for x in value if isinstance(x, str)]
    if isinstance(default, dict):
        if not isinstance(value, dict):
            return False, value
        if name == "engines":
            return True, {str(k): v for k, v in value.items() if isinstance(v, bool)}
        if name == "programs":
            return True, {str(k): v for k, v in value.items() if isinstance(v, str)}
        if name == "throughput":
            return True, {str(k): float(v) for k, v in value.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0}
        if name == "benchmarks":
            return True, {str(k): v for k, v in value.items() if isinstance(v, dict)}
        return True, value
    return True, value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def machine_name() -> str:
    """Lowercase short hostname, like a Tailscale MagicDNS name ("BTM-NOVA" -> "btm-nova"). WSL2 takes the
    Windows computer name as its hostname, so a WSL instance gets "-wsl" appended: it has its own engines and
    programs, and must not overwrite the Windows profile of the same computer."""
    from .. import resources
    from .plan import norm_name

    name = norm_name(socket.gethostname().split(".")[0]) or "localhost"
    return f"{name}-wsl" if resources.is_wsl() else name


def registry_dir() -> Path:
    from .. import settings as settings_mod

    return settings_mod.settings_dir() / "machines"


def save_profile(p: MachineProfile) -> Path:
    d = registry_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{p.name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(p.to_dict(), indent=2) + "\n", encoding="utf-8")
    for attempt in range(5):  # on Windows, SyncThing or an editor may hold the file open for a moment
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))
    return path


def _read_profile(path: Path) -> MachineProfile | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and not isinstance(data.get("name"), str):
            data = {**data, "name": path.stem.split(".")[0]}   # partial file: the file name is the machine
        return MachineProfile.from_dict(data)
    except (OSError, ValueError, TypeError):
        return None


def _updated_ts(p: MachineProfile) -> float:
    try:
        dt = datetime.fromisoformat(p.updated)
    except (TypeError, ValueError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def load_profiles() -> list[MachineProfile]:
    """Every machine in the registry, newest first, one entry per machine name. Unreadable files are skipped;
    of several files for one machine (e.g. a SyncThing conflict copy) the most recently updated wins."""
    d = registry_dir()
    if not d.is_dir():
        return []
    best: dict[str, MachineProfile] = {}
    for path in sorted(d.glob("*.json")):
        if path.name.startswith("."):
            continue
        p = _read_profile(path)
        if p is None:
            continue
        old = best.get(p.name)
        if old is None or _updated_ts(p) > _updated_ts(old):
            best[p.name] = p
    return sorted(best.values(), key=lambda p: (_updated_ts(p), p.name), reverse=True)


def load_profile(name: str) -> MachineProfile | None:
    for p in load_profiles():
        if p.name == name:
            return p
    return None


def profile_age_s(p: MachineProfile, now: float | None = None) -> float | None:
    ts = _updated_ts(p)
    if ts <= 0:
        return None
    return max(0.0, (time.time() if now is None else now) - ts)


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown age"
    if seconds < 90:
        return "just now"
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:.0f} {unit} ago"
    return "just now"


# --------------------------------------------------------------------------------- throughput
def _split_key(key: str) -> tuple[str, str]:
    m, _, dev = key.partition("@")
    return m, dev


def best_throughput(p: MachineProfile, method: str) -> tuple[str, float] | None:
    """(key, molecules/hour) of the fastest measured device for ``method``, or None."""
    hits = [(k, float(v)) for k, v in p.throughput.items() if _split_key(k)[0] == method and v > 0]
    return max(hits, key=lambda kv: kv[1]) if hits else None


def throughput_for(p: MachineProfile, method: str, device: str = "auto") -> float | None:
    """Measured molecules/hour for ``method`` on ``device``; "auto" = the machine's fastest device."""
    if device == "auto":
        hit = best_throughput(p, method)
        return hit[1] if hit else None
    for key, value in p.throughput.items():
        m, dev = _split_key(key)
        if m == method and dev == device:
            return float(value)
    return None


# --------------------------------------------------------------------------------- local machine
def git_commit(root: Path | None = None) -> str:
    """Short commit (12 hex digits) of the source checkout this package runs from, "" for an installed
    package. Reads ``.git`` directly (worktrees and packed refs included): no git program needed."""
    from .. import settings as settings_mod

    root = root or settings_mod.repo_root()
    if root is None:
        return ""
    try:
        git = root / ".git"
        if git.is_file():  # worktree or submodule: "gitdir: <path>"
            text = git.read_text(encoding="utf-8").strip()
            if not text.startswith("gitdir:"):
                return ""
            git = (root / text[len("gitdir:"):].strip()).resolve()
        if not git.is_dir():
            return ""
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head[:12]
        ref = head[4:].strip()
        common = git
        if (git / "commondir").is_file():
            common = (git / (git / "commondir").read_text(encoding="utf-8").strip()).resolve()
        for base in (git, common):
            if (base / ref).is_file():
                return (base / ref).read_text(encoding="utf-8").strip()[:12]
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha[:12]
    except OSError:
        return ""
    return ""


def _programs(probe_wsl: bool) -> tuple[dict[str, str], list[str]]:
    from ..chem import external

    progs: dict[str, str] = {}
    via_wsl: list[str] = []
    for name in external.PROGRAMS:
        try:
            info = external.find_program(name, probe_wsl=probe_wsl)
        except Exception:  # a broken settings path must not break the profile
            info = None
        if info is None:
            progs[name] = ""
            continue
        progs[name] = info.version or "unknown"
        if info.via_wsl:
            via_wsl.append(name)
    return progs, via_wsl


def _configured_address(name: str) -> str:
    from .. import settings as settings_mod

    try:
        workers = settings_mod.load().dist.workers
    except Exception:
        return ""
    return str(workers.get(name, "") or "")


def local_profile(bench: bool = False, methods=None, *, devices: Sequence[str] | None = None,
                  probe_wsl: bool = False, keep_previous: bool = True, address: str | None = None,
                  max_seconds: float = DEFAULT_MAX_SECONDS, should_stop: Callable[[], bool] | None = None,
                  log: Callable[[str], None] | None = None) -> MachineProfile:
    """This machine's profile. ``bench``: also measure throughput for ``methods`` (default: the available
    ones of ``DEFAULT_BENCH_METHODS``) on ``devices`` (default: CPU, plus each GPU for GPU-capable engines).

    ``keep_previous``: measurements and the worker address already in this machine's registry file are
    kept unless measured again (a failed re-measurement drops the old number). ``address``: the worker's
    URL (the worker passes it); otherwise ``settings.dist.workers[<name>]``, else the stored one.
    ``probe_wsl``: also ask programs that run through WSL for their version (starting WSL takes seconds).
    """
    from .. import __version__, resources
    from ..chem import engines as engines_mod

    log = log or (lambda _msg: None)
    rep = resources.detect()
    name = machine_name()
    progs, via_wsl = _programs(probe_wsl)
    p = MachineProfile(
        name=name,
        os=rep.os + (" (WSL2)" if rep.wsl else ""),
        arch=platform.machine(),
        python=rep.python,
        quantabench_version=__version__,
        cpu=rep.cpu_name,
        physical_cores=rep.physical_cores,
        logical_cores=rep.logical_cores,
        ram_gb=round(rep.ram_gb, 1),
        gpus=[{"index": g.index, "name": g.name, "memory_gb": round(g.total_memory_gb, 1), "backend": g.backend,
               "arch": g.arch} for g in rep.gpus],
        torch=rep.torch_version if not rep.torch_version.startswith("import failed") else "",
        engines={k: bool(engines_mod.available(k)[0]) for k in engines_mod.all_engines()},
        programs=progs,
        address=address or _configured_address(name),
        updated=_now(),
        commit=git_commit(),
        wsl=rep.wsl,
        torch_backend=rep.torch_backend,
        wsl_programs=via_wsl,
        notes=list(rep.notes),
    )
    measured: dict[str, dict] = {}
    if bench:
        wanted = list(methods) if methods else [m for m in DEFAULT_BENCH_METHODS if p.engines.get(m)]
        p.throughput = benchmark(wanted, devices, max_seconds=max_seconds, should_stop=should_stop, log=log,
                                 details=measured, gpus=rep.gpus)
        p.benchmarks = measured
    if keep_previous:
        prev = load_profile(name)
        if prev is not None:
            for key, value in prev.throughput.items():
                if key not in measured:
                    p.throughput.setdefault(key, value)
            for key, info in prev.benchmarks.items():
                p.benchmarks.setdefault(key, info)
            if not p.address:
                p.address = prev.address
    return p


# --------------------------------------------------------------------------------- benchmark
def bench_molecules():
    from ..chem.inputs import from_smiles

    return [from_smiles(smi, id=ident) for ident, smi in BENCH_SET]


def bench_devices(method: str, gpus=None) -> list[str]:
    """CPU, plus every GPU torch sees when the engine can use one."""
    from .. import resources
    from ..chem import engines as engines_mod

    info = engines_mod.all_engines().get(method)
    if info is None or not info.gpu_capable:
        return ["cpu"]
    if gpus is None:
        _, _, gpus = resources.torch_gpus()
    return ["cpu"] + ["mps" if g.backend == "mps" else f"cuda:{g.index}" for g in gpus]


def _device_policy(device: str) -> dict:
    if device == "cpu":
        return {"gpu": "off"}
    if device.startswith("cuda"):
        _, _, idx = device.partition(":")
        return {"gpu": "on", "gpu_index": int(idx or 0)}
    return {"gpu": "on"}   # mps


def _lanes_key(lanes) -> str:
    groups: dict[tuple[str, int], int] = {}
    for lane in lanes:
        key = (getattr(lane, "device", "cpu"), int(getattr(lane, "threads", 1)))
        groups[key] = groups.get(key, 0) + 1
    return "lanes:" + ",".join(f"{dev}:{n}x{t}" for (dev, t), n in groups.items())


def benchmark(methods: Sequence[str], devices: Sequence[str] | None = None, molecules=None, lanes=None, *,
              max_seconds: float = DEFAULT_MAX_SECONDS, threads: int = 0,
              should_stop: Callable[[], bool] | None = None, log: Callable[[str], None] | None = None,
              details: dict | None = None, gpus=None) -> dict[str, float]:
    """Molecules per hour for each method and device, keyed "<method>@<device>".

    Runs ``molecules`` (default ``BENCH_SET``) through ``pipeline.run_molecule`` with ``BENCH_PROTOCOL``,
    after one untimed warm-up molecule (imports, model loading, kernel compilation). A fast engine repeats the
    set until ``MIN_SECONDS`` have been timed. Each method/device stops after ``max_seconds`` (rate from the
    molecules finished by then); a first molecule that takes more than
    1.5 x ``max_seconds`` is cancelled and gives no number. A device the engine fell back from (e.g. a GPU
    error), a failed molecule, or an unavailable engine give no number either; ``details[key]["error"]``
    says why. ``should_stop`` cancels between and within molecules (keys already measured are returned).

    ``devices``: torch devices, default per method from ``bench_devices``. ``lanes`` (more than one
    ``dist.lanes.Lane``): time the whole set through ``jobs.run_resumable`` with those lanes instead, keyed
    "<method>@lanes:<device>:<count>x<threads>,...".
    """
    from .. import resources
    from ..chem import engines as engines_mod
    from ..chem import pipeline

    log = log or (lambda _msg: None)
    user_stop = should_stop or (lambda: False)
    details = details if details is not None else {}
    mols = list(molecules) if molecules is not None else bench_molecules()
    if not mols:
        raise ValueError("benchmark needs at least one molecule")
    n_threads = resources.apply_thread_limits(threads)
    out: dict[str, float] = {}
    t_all = time.perf_counter()
    for method in methods:
        if user_stop():
            break
        if method not in engines_mod.all_engines():
            log(f"  {method}: unknown method, skipped")
            details[f"{method}@?"] = {"error": "unknown method", "date": _now()}
            continue
        ok, why = engines_mod.available(method)
        if not ok:
            log(f"  {method}: unavailable ({why}), skipped")
            continue
        if lanes and len(lanes) > 1:
            key = f"{method}@{_lanes_key(lanes)}"
            rate = _bench_lanes(method, mols, lanes, key, details, user_stop, log)
            if rate:
                out[key] = rate
            continue
        for device in (list(devices) if devices else bench_devices(method, gpus)):
            if user_stop():
                break
            if device != "cpu" and not engines_mod.all_engines()[method].gpu_capable:
                continue
            key = f"{method}@{device}"
            cfg = pipeline.BuildConfig(method=method, threads=n_threads, **BENCH_PROTOCOL, **_device_policy(device))
            rate = _bench_one(pipeline, cfg, mols, device, n_threads, key, max_seconds, details, user_stop, log)
            if rate:
                out[key] = rate
    log(f"  benchmark took {time.perf_counter() - t_all:.0f} s in total")
    return out


def _bench_one(pipeline, cfg, mols, device: str, threads: int, key: str, max_seconds: float, details: dict,
               user_stop: Callable[[], bool], log: Callable[[str], None]) -> float | None:
    info = {"molecules": 0, "seconds": 0.0, "threads": threads, "device": device, "date": _now(),
            "set": f"pfas-{len(mols)}" if [m.id for m in mols] == [i for i, _ in BENCH_SET] else f"custom-{len(mols)}",
            "bench_version": BENCH_VERSION, "warmup_s": 0.0, "error": ""}
    details[key] = info
    quiet = lambda _msg: None  # noqa: E731

    def run(inp, stop):
        try:
            return pipeline.run_molecule(inp, cfg, device, threads, stop, quiet)
        except Exception as exc:  # run_molecule records most errors itself; this is the rest
            return pipeline.MoleculeResult(input=inp, status="failed", error=f"{type(exc).__name__}: {exc}")

    def problem(res) -> str:
        if res.status != "ok":
            return res.error or res.status
        if res.method != cfg.method:
            return f"ran {res.method} instead of {cfg.method}"
        if res.device != device:
            why = next((n for n in reversed(res.notes) if "fell back" in n), "")
            return f"fell back to {res.device}" + (f": {why}" if why else "")
        return ""

    t = time.perf_counter()
    warm = run(mols[0], lambda: user_stop() or time.perf_counter() - t > 1.5 * max_seconds)
    info["warmup_s"] = round(time.perf_counter() - t, 2)
    if user_stop():
        info["error"] = "cancelled"
        return None
    if problem(warm):
        info["error"] = problem(warm)
        log(f"  {key:22s} FAILED: {info['error']}")
        return None
    done, busy = 0, 0.0
    t0 = time.perf_counter()
    for k in range(len(mols) * MAX_PASSES):   # fast engines go round the set again, for a stable time
        inp = mols[k % len(mols)]
        if user_stop() or (done and busy >= max_seconds) or (done >= len(mols) and busy >= MIN_SECONDS):
            break

        def stop(done=done):
            elapsed = time.perf_counter() - t0
            return user_stop() or (done > 0 and elapsed > max_seconds) or elapsed > 1.5 * max_seconds

        t1 = time.perf_counter()
        res = run(inp, stop)
        if res.status == "cancelled":
            break
        if problem(res):
            info["error"] = f"{inp.id}: {problem(res)}"
            log(f"  {key:22s} FAILED: {info['error']}")
            return None
        busy += time.perf_counter() - t1
        done += 1
    info["molecules"], info["seconds"] = done, round(busy, 2)
    if user_stop():
        info["error"] = "cancelled"
        return None
    if not done:
        info["error"] = f"no molecule finished within {1.5 * max_seconds:.0f} s"
        log(f"  {key:22s} FAILED: {info['error']}")
        return None
    rate = round(done / busy * 3600.0, 1)
    info["per_hour"] = round(rate, 1)
    log(f"  {key:22s} {done} molecule(s) in {busy:6.1f} s (warm-up {info['warmup_s']:.1f} s): "
        f"{rate:,.0f} molecules/hour")
    return rate


def _bench_lanes(method: str, mols, lanes, key: str, details: dict, user_stop, log) -> float | None:
    import tempfile

    from ..chem import jobs, pipeline

    cfg = pipeline.BuildConfig(method=method, gpu="auto", **BENCH_PROTOCOL)
    info = {"molecules": 0, "seconds": 0.0, "lanes": [lane.to_dict() for lane in lanes], "date": _now(),
            "bench_version": BENCH_VERSION, "error": ""}
    details[key] = info
    with tempfile.TemporaryDirectory(prefix="qb_bench_") as tmp:
        t = time.perf_counter()
        results, _ = jobs.run_resumable(mols, cfg, Path(tmp) / "run", should_stop=user_stop, lanes=list(lanes))
        dt = time.perf_counter() - t
    ok = sum(r.status == "ok" and r.method == method for r in results)
    info["molecules"], info["seconds"] = ok, round(dt, 2)
    if user_stop() or not ok:
        info["error"] = "cancelled" if user_stop() else "no molecule finished"
        return None
    rate = round(ok / dt * 3600.0, 1)
    info["per_hour"] = round(rate, 1)
    log(f"  {key}: {ok} molecule(s) in {dt:.1f} s: {rate:,.0f} molecules/hour")
    return rate


# --------------------------------------------------------------------------------- display
def _fmt_rate(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def describe(p: MachineProfile, this: bool = False, now: float | None = None,
             version: str | None = None) -> list[str]:
    """Lines for ``quantabench-cli machines``."""
    age = profile_age_s(p, now)
    head = f"{p.name}{' (this machine)' if this else ''}"
    lines = [f"{head:40s} updated {p.updated or '?'} ({format_age(age)})"]
    sysline = ", ".join(x for x in (p.os, p.arch, f"Python {p.python}" if p.python else "",
                                    f"QuantaBench {p.quantabench_version}" if p.quantabench_version else "") if x)
    if p.commit:
        sysline += f" ({p.commit})"
    if sysline:
        lines.append(f"  {sysline}")
    if p.cpu or p.physical_cores:
        lines.append(f"  {p.cpu or 'CPU'}: {p.physical_cores} cores / {p.logical_cores} threads, {p.ram_gb:.0f} GB RAM")
    for g in p.gpus:
        mem = g.get("memory_gb")
        lines.append(f"  GPU {g.get('index', '?')}: {g.get('name', '?')}"
                     + (f", {float(mem):.0f} GB" if isinstance(mem, (int, float)) else "")
                     + (f" ({g.get('backend')})" if g.get("backend") else ""))
    if p.torch or p.torch_backend:
        lines.append(f"  PyTorch {p.torch or 'not installed'}" + (f" ({p.torch_backend})" if p.torch_backend else ""))
    if p.engines:
        have = [k for k, v in p.engines.items() if v]
        missing = [k for k, v in p.engines.items() if not v]
        lines.append(f"  engines:    {', '.join(have) or 'none'}" + (f"  (not: {', '.join(missing)})" if missing else ""))
    if p.programs:
        have = [f"{k} {v}" + (" (WSL)" if k in p.wsl_programs else "") for k, v in p.programs.items() if v]
        missing = [k for k, v in p.programs.items() if not v]
        lines.append(f"  programs:   {', '.join(have) or 'none'}" + (f"  (not: {', '.join(missing)})" if missing else ""))
    if p.throughput:
        rates = ", ".join(f"{k} {_fmt_rate(v)}" for k, v in sorted(p.throughput.items()))
        lines.append(f"  throughput: {rates} molecules/hour (benchmark set)")
    else:
        lines.append("  throughput: not measured (`quantabench-cli machines --refresh --bench mmff,gfn1-xtb` there)")
    failed = {k: v.get("error") for k, v in p.benchmarks.items() if v.get("error") and k not in p.throughput}
    for k, err in failed.items():
        lines.append(f"  benchmark {k} failed: {err}")
    if p.address:
        lines.append(f"  worker:     {p.address}")
    if age is not None and age > STALE_DAYS * 86400:
        lines.append(f"  note: profile is {age / 86400:.0f} days old; run `quantabench-cli machines --refresh` on {p.name}")
    if version and p.quantabench_version and p.quantabench_version != version:
        lines.append(f"  note: runs QuantaBench {p.quantabench_version}, this machine {version}")
    return lines


def _parse_methods(text: str, engines: dict) -> list[str]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items or items == ["default"]:
        return [m for m in DEFAULT_BENCH_METHODS if engines.get(m)]
    unknown = [m for m in items if m not in engines]
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)} (known: {', '.join(engines)})")
    return items


def refresh_from_cli(bench: bool, methods=None, devices=None, max_seconds: float = DEFAULT_MAX_SECONDS,
                     probe_wsl: bool = False, out=None) -> tuple[MachineProfile | None, Path | None]:
    """Build (and benchmark) this machine's profile and save it, for ``doctor --bench`` and ``machines``.
    Ctrl+C stops the benchmark after the current molecule and saves nothing; returns (None, None) then."""
    import signal
    import threading

    out = out or sys.stdout
    stop = {"now": False}

    def on_sigint(*_a):
        if stop["now"]:
            raise KeyboardInterrupt
        stop["now"] = True
        print("\nStopping the benchmark (Ctrl+C again to abort)...", file=sys.stderr)

    main = threading.current_thread() is threading.main_thread()
    old = signal.signal(signal.SIGINT, on_sigint) if bench and main else None
    try:
        if bench:
            names = ", ".join(methods) if methods else "the available default engines"
            print(f"Benchmarking {names} on {machine_name()} ({len(BENCH_SET)} PFAS anions, up to "
                  f"{max_seconds:.0f} s per method and device)", file=out)
        prof = local_profile(bench=bench, methods=methods, devices=devices, probe_wsl=probe_wsl,
                             max_seconds=max_seconds, should_stop=lambda: stop["now"],
                             log=lambda m: print(m, file=out, flush=True))
    finally:
        if old is not None:
            signal.signal(signal.SIGINT, old)
    if stop["now"]:
        print("Cancelled; the machine registry was not changed.", file=sys.stderr)
        return None, None
    path = save_profile(prof)
    print(f"Saved this machine's profile to {path}", file=out)
    return prof, path


def _machines_cli(args) -> int:
    from .. import __version__
    from ..chem import engines as engines_mod

    this = machine_name()
    if args.refresh or args.bench:
        methods = None
        if args.bench:
            avail = {k: engines_mod.available(k)[0] for k in engines_mod.all_engines()}
            try:
                methods = _parse_methods(args.bench, avail)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 2
            if not methods:
                print("No benchmark method is available on this machine.", file=sys.stderr)
                return 2
        devices = [d.strip() for d in args.devices.split(",") if d.strip()] if args.devices else None
        prof, _ = refresh_from_cli(bool(methods), methods, devices, args.max_seconds, args.wsl,
                                   out=sys.stderr if args.json else sys.stdout)
        if prof is None:
            return 130
    profiles = load_profiles()
    if args.json:
        now = time.time()
        print(json.dumps({"registry": str(registry_dir()), "this_machine": this,
                          "machines": [{**p.to_dict(), "age_s": profile_age_s(p, now)} for p in profiles]}, indent=2))
        return 0
    print(f"Machine registry: {registry_dir()} ({len(profiles)} machine{'s' if len(profiles) != 1 else ''})")
    if not profiles:
        print("  empty: run `quantabench-cli machines --refresh` on each machine "
              "(add --bench mmff,gfn1-xtb to measure throughput)")
        return 0
    for p in profiles:
        print()
        print("\n".join(describe(p, this=p.name == this, version=__version__)))
    if this not in {p.name for p in profiles}:
        print(f"\nThis machine ({this}) is not in the registry: `quantabench-cli machines --refresh`.")
    return 0


def register_cli(subparsers) -> None:
    m = subparsers.add_parser("machines", help="machine registry: what each machine can run and how fast",
                              description="Lists the machine profiles in the synced settings folder "
                                          "(settings/machines/<name>.json). --refresh writes this machine's.")
    m.add_argument("--refresh", action="store_true", help="write this machine's profile to the registry")
    m.add_argument("--bench", metavar="METHODS",
                   help="measure throughput (molecules/hour on a fixed PFAS set) for these methods, e.g. "
                        "mmff,gfn1-xtb, or 'default'; implies --refresh")
    m.add_argument("--devices", help="with --bench: devices to time, e.g. cpu,cuda:0 (default: CPU, plus each "
                                     "GPU for GPU-capable engines)")
    m.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                   help=f"with --bench: time budget per method and device (default {DEFAULT_MAX_SECONDS:.0f})")
    m.add_argument("--wsl", action="store_true", help="also probe versions of programs run through WSL (slow)")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=_machines_cli)
