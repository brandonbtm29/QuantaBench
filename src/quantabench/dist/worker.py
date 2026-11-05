"""``quantabench-cli worker``: run QuantaBench jobs for the other machines on the tailnet.

A small HTTP/JSON service (standard library only) that another machine's ``grid build``, the GUI or
``dist.client.WorkerClient`` submits jobs to. Each job is a molecule list plus the science settings
(``BuildConfig``); the worker runs it with ``jobs.run_resumable`` into its *own* run folder under its own
output folder (one writer per run folder), so the results reach the other machines through SyncThing,
or through ``GET /v1/jobs/<id>/archive`` where there is no synced folder.

API (version 1; every request needs ``Authorization: Bearer <settings.dist.worker_token>``)::

    GET  /v1/info                    machine profile, versions, queue
    GET  /v1/jobs                    every job (newest first)
    POST /v1/jobs                    {inputs, config, label, chunk?, lanes?, client?, allow_version_mismatch?}
                                     -> 201 {id, job}; needs X-QuantaBench-Version equal to the worker's
    GET  /v1/jobs/<id>[?tail=N]      state, counts from the run folder's job table, run folder, log tail
    POST /v1/jobs/<id>/cancel        stop a running job (it stays resumable) or drop a queued one
    GET  /v1/jobs/<id>/archive       zip stream of the run folder (finished jobs)
    POST /v1/pause                   {"paused": true|false}: hold new jobs so this machine stays responsive

Safety: binds the Tailscale address (or loopback) only and refuses 0.0.0.0 unless told otherwise; constant
time token check; size-limited, strictly validated requests (known ``BuildConfig`` fields only, the output
folder is always the worker's own, molecule ids and labels must be folder-safe); jobs from a different
QuantaBench version are refused. See docs/WORKERS.md for setting it up on Windows and macOS.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import errno
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .. import __version__
from .. import settings as settings_mod

API_VERSION = 1
VERSION_HEADER = "X-QuantaBench-Version"
DEFAULT_PORT = 8765
MAX_REQUEST_BYTES = 64 * 1024 * 1024      # a job of a few thousand molecules with molblocks fits easily
MAX_INPUTS = 100_000
MIN_TOKEN_LENGTH = 24
LOG_TAIL = 500                             # log lines kept in memory per job
STATE_FILE = "jobs.json"
STATE_BACKUP = "jobs.json.bak"             # the same list, written after STATE_FILE: one of them is always whole
STATE_LOCK = "worker.lock"
JOB_FILE = "worker.json"                   # in each run folder: the job, the worker's environment, final tables
JOB_LOG = "worker.log"                     # in each run folder: the job's log
TABLES = ("summary.csv", "descriptors.csv")
ARCHIVE_MANIFEST = ".qb-archive.json"      # last member of an archive: every other member and its size
#: SyncThing's temporary files (``~syncthing~name.tmp`` on Windows, ``.syncthing.name.tmp`` elsewhere): they
#: appear and vanish while it pulls, so they never go into an archive
SYNC_TEMP_PREFIXES = ("~syncthing~", ".syncthing.")
_ADDR_NOT_AVAILABLE = {errno.EADDRNOTAVAIL, 10049}   # 10049: WSAEADDRNOTAVAIL (address not on this machine yet)
STATES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
FINAL_STATES = ("done", "failed", "cancelled")
#: BuildConfig fields that only say how fast this machine runs a job. A client normally leaves them out and
#: the worker uses its own settings; the science fields must be the same on every machine.
RESOURCE_FIELDS = ("threads", "gpu", "gpu_index", "gpu_memory_fraction", "linalg", "dxtb_gpu")
TAILSCALE_NET4 = ipaddress.ip_network("100.64.0.0/10")
TAILSCALE_NET6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
TAILSCALE_PATHS = {
    "win32": [r"C:\Program Files\Tailscale\tailscale.exe"],
    "darwin": ["/Applications/Tailscale.app/Contents/MacOS/Tailscale"],
}

_JOB_ID_RE = re.compile(r"[0-9a-f]{12}")
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,59}")
_LANES_RE = re.compile(r"[A-Za-z0-9:,x]{1,64}")
_GRID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,119}")
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

# Allowed values of BuildConfig string fields (method, solvent and qm_method are checked by their modules).
CHOICES: dict[str, tuple[str, ...]] = {
    "protonation_method": ("rules", "dimorphite"),
    "optimizer": ("lbfgs", "fire", "native"),
    "batched": ("auto", "on", "off"),
    "freq_hessian": ("auto", "analytic", "finite"),
    "conformer_search": ("rdkit", "crest"),
    "crest_method": ("gfn2", "gfnff", "gfn2//gfnff", "gfn1"),
    "qm_task": ("energy", "optimize"),
    "qm_program": ("psi4",),
    "gpu": ("auto", "on", "off"),
    "linalg": ("auto", "cpu", "device"),
    "dxtb_gpu": ("auto", "on", "off"),
}
# Inclusive ranges of numeric BuildConfig fields; other numbers only need to be finite.
RANGES: dict[str, tuple[float, float]] = {
    "ph": (0.0, 14.0), "max_conformers": (1, 100_000), "refine_top": (0, 100_000),
    "energy_window_kcal": (0.0, 1e4), "rmsd_threshold": (0.0, 100.0), "fmax": (1e-6, 100.0),
    "max_steps": (0, 10_000_000), "temperature_k": (1e-3, 1e5), "symmetry_number": (0, 1000),
    "seed": (-(2**31), 2**31 - 1), "qm_top": (0, 10_000), "threads": (0, 4096), "gpu_index": (0, 64),
    "gpu_memory_fraction": (0.01, 1.0),
}
_INPUT_LIMITS = {"id": 120, "smiles": 20_000, "name": 1_000, "source": 2_000, "molblock": 16_000_000}


def _noop(*_args) -> None:
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RequestError(Exception):
    """A request the worker refuses; ``status`` is the HTTP status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


# ===================================================================================== validation
def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


def parse_json(raw: bytes) -> dict:
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RequestError(400, f"body is not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise RequestError(400, "body must be a JSON object")
    return data


def _check_str(name: str, value, limit: int, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise RequestError(400, f"{name} must be a string")
    if len(value) > limit:
        raise RequestError(400, f"{name} is longer than {limit} characters")
    if not allow_empty and not value:
        raise RequestError(400, f"{name} must not be empty")
    return value


def check_label(label) -> str:
    """Labels name the run folder, so they must be one folder-safe path component."""
    if not isinstance(label, str) or not _LABEL_RE.fullmatch(label):
        raise RequestError(400, "label must be 1-60 characters of A-Z a-z 0-9 . _ + - and start with a letter "
                                "or digit")
    return label


def check_mol_id(mol_id) -> str:
    """Molecule ids name the molecule's folder inside the run folder: no separators, no '..', no device
    names (``inputs.safe_id`` makes such ids from any text)."""
    from ..chem.inputs import safe_id

    if not isinstance(mol_id, str) or not mol_id or len(mol_id) > _INPUT_LIMITS["id"] or safe_id(mol_id) != mol_id:
        raise RequestError(400, f"molecule id {str(mol_id)[:80]!r} is not folder-safe (A-Z a-z 0-9 . _ + -, "
                                "not starting or ending with . or _)")
    if mol_id.split(".")[0].lower() in _WINDOWS_RESERVED:
        raise RequestError(400, f"molecule id {mol_id!r} is a reserved file name on Windows")
    return mol_id


def validate_inputs(items) -> list:
    """``inputs`` of a job: a list of ``jobs.input_to_json`` texts (or the same as objects)."""
    from ..chem import jobs
    from ..chem.inputs import MoleculeInput

    if not isinstance(items, list) or not items:
        raise RequestError(400, "inputs must be a non-empty list")
    if len(items) > MAX_INPUTS:
        raise RequestError(400, f"at most {MAX_INPUTS} inputs per job")
    known = {f.name for f in dataclasses.fields(MoleculeInput)} - {"mol"} | {"molblock"}
    out: list = []
    seen: dict[str, str] = {}   # casefolded id -> id: Windows and macOS file systems ignore letter case
    for k, item in enumerate(items):
        if isinstance(item, str):
            try:
                item = json.loads(item, parse_constant=_reject_constant)
            except ValueError:
                raise RequestError(400, f"inputs[{k}] is not valid JSON") from None
        if not isinstance(item, dict):
            raise RequestError(400, f"inputs[{k}] must be an object (jobs.input_to_json)")
        unknown = sorted(set(item) - known)
        if unknown:
            raise RequestError(400, f"inputs[{k}]: unknown field(s) {', '.join(unknown)}")
        mol_id = check_mol_id(item.get("id"))
        other = seen.get(mol_id.casefold())
        if other == mol_id:
            raise RequestError(400, f"duplicate molecule id {mol_id!r}")
        if other is not None:
            raise RequestError(400, f"molecule ids {other!r} and {mol_id!r} differ only in letter case: they would "
                                    "share one folder on Windows and macOS (inputs.dedupe_ids renames them)")
        seen[mol_id.casefold()] = mol_id
        for name in ("smiles", "name", "source", "molblock"):
            if name in item:
                _check_str(f"inputs[{k}].{name}", item[name], _INPUT_LIMITS[name])
        for name in ("smiles", "name"):
            if _CONTROL.search(item.get(name) or ""):
                raise RequestError(400, f"inputs[{k}].{name} contains control characters")
        if not (item.get("smiles") or item.get("molblock")):
            raise RequestError(400, f"inputs[{k}] ({mol_id}) has neither smiles nor molblock")
        charge = item.get("charge")
        if charge is not None and (type(charge) is not int or not -50 <= charge <= 50):
            raise RequestError(400, f"inputs[{k}].charge must be null or an integer in [-50, 50]")
        mult = item.get("multiplicity", 1)
        if type(mult) is not int or not 1 <= mult <= 20:
            raise RequestError(400, f"inputs[{k}].multiplicity must be an integer in [1, 20]")
        meta = item.get("meta", {})
        if not isinstance(meta, dict) or len(json.dumps(meta, default=str)) > 65_536:
            raise RequestError(400, f"inputs[{k}].meta must be an object of at most 64 KB")
        try:
            inp = jobs.input_from_json(json.dumps(item))
        except (TypeError, ValueError) as exc:
            raise RequestError(400, f"inputs[{k}] ({mol_id}): {exc}") from None
        if item.get("molblock") and inp.mol is None:
            raise RequestError(400, f"inputs[{k}] ({mol_id}): the molblock could not be read")
        out.append(inp)
    return out


def _check_number(name: str, value, kind: type):
    if kind is bool:
        if not isinstance(value, bool):
            raise RequestError(400, f"config.{name} must be true or false")
        return value
    if kind is int:
        if type(value) is not int:
            raise RequestError(400, f"config.{name} must be an integer")
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RequestError(400, f"config.{name} must be a finite number")
    else:
        value = float(value)
    lo, hi = RANGES.get(name, (-1e12, 1e12))
    if not lo <= value <= hi:
        raise RequestError(400, f"config.{name} must be in [{lo}, {hi}]")
    return value


def validate_config(data) -> dict:
    """Check a ``BuildConfig`` as sent by a client (``dataclasses.asdict``, usually without ``output_dir``
    and the resource fields). Returns the accepted fields; raises ``RequestError`` (400) otherwise."""
    from ..chem import engines, export, solvents
    from ..chem.pipeline import BuildConfig

    if not isinstance(data, dict):
        raise RequestError(400, "config must be an object of BuildConfig fields")
    defaults = BuildConfig()
    known = {f.name for f in dataclasses.fields(BuildConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise RequestError(400, f"unknown config field(s): {', '.join(unknown)}")
    out: dict = {}
    for name, value in data.items():
        default = getattr(defaults, name)
        if name == "output_dir":
            if value not in ("", None):
                raise RequestError(400, "config.output_dir is chosen by the worker (its own output folder); "
                                        "leave it out")
            continue
        if name == "write_files":
            if value is not True:
                raise RequestError(400, "config.write_files must be true on a worker")
            out[name] = True
            continue
        if name == "formats":
            if not isinstance(value, (list, tuple)) or not value or len(value) > len(export.FORMATS):
                raise RequestError(400, f"config.formats must be a list of {', '.join(export.FORMATS)}")
            bad = [x for x in value if not isinstance(x, str) or x not in export.FORMATS]
            if bad or len(set(value)) != len(value):
                raise RequestError(400, f"config.formats: unknown or repeated format(s) {bad or value}; known: "
                                        f"{', '.join(export.FORMATS)}")
            out[name] = list(value)
            continue
        if isinstance(default, (bool, int, float)):
            out[name] = _check_number(name, value, type(default))
            continue
        if isinstance(default, str):
            value = _check_str(f"config.{name}", value, 200)
            if _CONTROL.search(value):
                raise RequestError(400, f"config.{name} contains control characters")
            if name in CHOICES and value not in CHOICES[name]:
                raise RequestError(400, f"config.{name} must be one of {', '.join(CHOICES[name])}")
            out[name] = value
            continue
        raise RequestError(400, f"config.{name} cannot be set through a worker")
    if "method" in out and out["method"] not in engines.method_keys():
        raise RequestError(400, f"config.method {out['method']!r} is not an engine of this worker "
                                f"(known: {', '.join(engines.method_keys())})")
    if out.get("solvent"):
        try:
            solvents.canonical(out["solvent"])
        except ValueError as exc:
            raise RequestError(400, f"config.solvent: {exc}") from None
    if out.get("qm_method"):
        from ..chem import qm

        try:
            qm.parse_level(out["qm_method"])
        except ValueError as exc:
            raise RequestError(400, f"config.qm_method: {exc}") from None
    return out


def build_config(config: dict, output_dir: Path, resources=None):
    """The ``BuildConfig`` a worker runs: the client's science settings on ``BuildConfig`` defaults, this
    machine's resource settings unless the client set them, and always this worker's output folder."""
    from ..chem.pipeline import BuildConfig

    r = resources or settings_mod.load().resources
    values = {"threads": r.threads, "gpu": r.gpu, "gpu_index": r.gpu_index,
              "gpu_memory_fraction": r.gpu_memory_fraction, "linalg": getattr(r, "linalg", "auto"),
              "dxtb_gpu": getattr(r, "dxtb_gpu", "auto")}
    values.update(config)
    if "formats" in values:
        values["formats"] = tuple(values["formats"])
    values["output_dir"] = str(output_dir)
    return dataclasses.replace(BuildConfig(), **values)


def validate_chunk(chunk) -> dict | None:
    """Which part of a split this job is. ``{"index": i, "count": n}``: the worker keeps the molecules of
    ``jobs.chunk_index`` chunk i of n; ``{"index": i, "weights": [...]}``: those that ``plan.weighted_index``
    gives to i (so a client may send the whole list); ``{"index": i}`` alone: already split by the client
    (``grid build``), recorded only. Optional ``"grid"``: the grid id, for the record."""
    if chunk is None:
        return None
    if not isinstance(chunk, dict):
        raise RequestError(400, "chunk must be an object")
    unknown = sorted(set(chunk) - {"index", "count", "weights", "grid"})
    if unknown:
        raise RequestError(400, f"chunk: unknown field(s) {', '.join(unknown)}")
    index = chunk.get("index")
    if type(index) is not int or index < 0:
        raise RequestError(400, "chunk.index must be a non-negative integer")
    out: dict = {"index": index}
    if "count" in chunk and "weights" in chunk:
        raise RequestError(400, "chunk takes count or weights, not both")
    if "count" in chunk:
        count = chunk["count"]
        if type(count) is not int or not index < count <= 10_000:
            raise RequestError(400, "chunk.count must be an integer > index")
        out["count"] = count
    elif "weights" in chunk:
        w = chunk["weights"]
        if (not isinstance(w, list) or not 0 < len(w) <= 256 or index >= len(w)
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                       for x in w) or w[index] <= 0):
            raise RequestError(400, "chunk.weights must be a list of non-negative numbers with a positive "
                                    "weight at index")
        out["weights"] = [float(x) for x in w]
    if "grid" in chunk:
        if not isinstance(chunk["grid"], str) or not _GRID_RE.fullmatch(chunk["grid"]):
            raise RequestError(400, "chunk.grid must be a grid id (folder-safe, at most 120 characters)")
        out["grid"] = chunk["grid"]
    return out


@dataclass
class JobSpec:
    """A validated job request."""
    inputs: list
    config: dict
    label: str
    chunk: dict | None = None
    lanes: str | None = None
    client: str = ""
    client_version: str = ""
    allow_version_mismatch: bool = False


def validate_job(body: dict, client_version: str = "") -> JobSpec:
    allowed = {"inputs", "config", "label", "chunk", "lanes", "client", "allow_version_mismatch"}
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise RequestError(400, f"unknown field(s): {', '.join(unknown)}")
    for name in ("inputs", "config", "label"):
        if name not in body:
            raise RequestError(400, f"{name} is required")
    label = check_label(body["label"])
    config = validate_config(body["config"])
    chunk = validate_chunk(body.get("chunk"))
    lanes = body.get("lanes")
    if lanes is not None:
        if type(lanes) is int and 1 <= lanes <= 256:
            lanes = str(lanes)
        if not isinstance(lanes, str) or not _LANES_RE.fullmatch(lanes):
            raise RequestError(400, "lanes must be a lane spec such as 1, auto, 4 or gpu:1x4,cpu:5x2")
        from . import lanes as lanes_mod

        try:
            lanes_mod.parse_lanes(lanes, build_config(config, Path(".")))
        except (ValueError, TypeError) as exc:
            raise RequestError(400, f"lanes: {exc}") from None
    client = body.get("client", "")
    if client and (not isinstance(client, str) or not _LABEL_RE.fullmatch(client)):
        raise RequestError(400, "client must be a machine name")
    allow = body.get("allow_version_mismatch", False)
    if not isinstance(allow, bool):
        raise RequestError(400, "allow_version_mismatch must be true or false")
    inputs = validate_inputs(body["inputs"])
    if chunk and "weights" in chunk:  # a client may send the whole list: keep this worker's share
        from . import plan

        inputs = [x for x in inputs if plan.assign(x.id, chunk["weights"]) == chunk["index"]]  # = build --chunk-weights
    elif chunk and "count" in chunk:  # run_resumable applies the chunk again and records it
        from ..chem import jobs

        inputs = [x for x in inputs if jobs.in_chunk(x.id, (chunk["index"], chunk["count"]))]
    if not inputs:
        raise RequestError(400, "no input belongs to this chunk")
    return JobSpec(inputs=inputs, config=config, label=label, chunk=chunk, lanes=lanes,
                   client=client or "", client_version=client_version[:40],
                   allow_version_mismatch=allow)


# ======================================================================================= helpers
def tailscale_executable() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for candidate in TAILSCALE_PATHS.get(sys.platform, []):
        if Path(candidate).is_file():
            return candidate
    return None


def tailscale_ipv4(timeout: float = 10.0) -> str:
    """This machine's Tailscale IPv4 address (``tailscale ip -4``)."""
    exe = tailscale_executable()
    if exe is None:
        raise RuntimeError("Tailscale is not installed (no `tailscale` command); install it, or bind "
                           "--bind localhost for a local test")
    try:
        proc = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=timeout,  # noqa: S603
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"`tailscale ip -4` failed: {exc}") from None
    for line in proc.stdout.split():
        with contextlib.suppress(ValueError):
            addr = ipaddress.ip_address(line.strip())
            if addr in TAILSCALE_NET4:
                return str(addr)
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    raise RuntimeError("Tailscale has no IPv4 address yet (is it running and logged in?)"
                       + (f": {detail[-1]}" if detail else ""))


def is_tailscale_address(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr in (TAILSCALE_NET4 if addr.version == 4 else TAILSCALE_NET6)


def resolve_bind(bind: str, allow_any: bool = False, tailscale_wait: float = 0.0,
                 log: Callable[[str], None] = _noop) -> str:
    """The address to listen on: ``tailscale`` (this machine's 100.x address), ``localhost`` (127.0.0.1) or
    an IP address. 0.0.0.0 / :: (every network) only with ``allow_any``."""
    bind = (bind or "tailscale").strip()
    if bind == "tailscale":
        deadline = time.monotonic() + max(0.0, tailscale_wait)
        while True:
            try:
                return tailscale_ipv4()
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise
                log(f"{exc}; retrying")
                time.sleep(5.0)
    if bind == "localhost":
        return "127.0.0.1"
    try:
        addr = ipaddress.ip_address(bind.strip("[]"))
    except ValueError:
        raise ValueError(f"--bind must be tailscale, localhost or an IP address, not {bind!r}") from None
    if addr.is_unspecified and not allow_any:
        raise ValueError(f"refusing to listen on {addr} (every network, including public Wi-Fi). Use "
                         "--bind tailscale, or pass --allow-any-address if a firewall limits the port.")
    return str(addr)


def ensure_token(log: Callable[[str], None] = _noop) -> str:
    """``settings.dist.worker_token``; generated and saved on first use (the synced settings folder shares
    it with the other machines). A settings file that exists but cannot be read is an error, not a first
    start: saving then would replace every setting (and the token the other machines use) with defaults."""
    path = settings_mod.settings_dir() / settings_mod.SETTINGS_FILE
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {path} ({exc}); fix or restore it (SyncThing keeps old versions in "
                             ".stversions) instead of letting the worker overwrite it") from None
        if not isinstance(raw, dict):
            raise ValueError(f"{path} is not a JSON object; fix or restore it")
    s = settings_mod.load()
    token = s.dist.worker_token
    if not token:
        token = secrets.token_urlsafe(32)
        s.dist.worker_token = token
        path = settings_mod.save(s)
        log(f"Generated a worker token and saved it in {path} (the synced settings folder gives it to your "
            "other machines)")
    if len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(f"settings.dist.worker_token is shorter than {MIN_TOKEN_LENGTH} characters; clear it "
                         "to have a strong one generated")
    return token


def git_commit() -> str:
    """Short commit of the source checkout this runs from ('' for an installed package)."""
    root = settings_mod.repo_root()
    git = shutil.which("git")
    if root is None or git is None or not (root / ".git").exists():
        return ""
    try:
        proc = subprocess.run([git, "-C", str(root), "rev-parse", "--short=12", "HEAD"],  # noqa: S603
                              capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _process_start(pid: int) -> float | None:
    import psutil

    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return None


def _lock_owner_alive(info: dict | None) -> bool:
    """Whether the worker that wrote ``worker.lock`` still runs: same host, and a process with that pid that
    started when the lock says (after a reboot or a killed worker the pid may belong to another program, as
    ``jobs._RunLock`` also checks). A lock without a start time (older versions) falls back to the pid."""
    if not info or info.get("host") != socket.gethostname():
        return False
    try:
        pid = int(info.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if pid == os.getpid() or pid <= 0:
        return False
    start = _process_start(pid)
    if start is None:
        return False
    started = info.get("pid_started")
    if not isinstance(started, (int, float)) or isinstance(started, bool):
        return True
    return abs(start - float(started)) < 1.0


@contextlib.contextmanager
def keep_awake(log: Callable[[str], None] = _noop) -> Iterator[None]:
    """Keep the machine from idle-sleeping while a job runs: ``caffeinate -i -w <pid>`` on macOS (it also
    ends if the worker dies), ``SetThreadExecutionState`` on Windows; nothing elsewhere."""
    proc = None
    win = False
    if sys.platform == "darwin":
        exe = shutil.which("caffeinate")
        if exe:
            try:
                proc = subprocess.Popen([exe, "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL,  # noqa: S603
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as exc:
                log(f"caffeinate failed ({exc}); the Mac may sleep during the job")
    elif sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
            win = True
        except (OSError, AttributeError):
            pass
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(5)
        if win:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def default_state_dir() -> Path:
    """Per-machine worker state (job list, queued inputs): never in the synced settings folder."""
    env = os.environ.get("QUANTABENCH_WORKER_STATE", "").strip()
    if env:
        return Path(env).expanduser()
    from platformdirs import user_state_dir

    return Path(user_state_dir(settings_mod.APP_NAME, appauthor=False)) / "worker"


def _fsync_dir(folder: Path) -> None:
    """Make a rename in ``folder`` durable (POSIX; Windows cannot open a folder this way, NTFS journals it)."""
    if os.name == "nt":
        return
    with contextlib.suppress(OSError):
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic_write(path: Path, text: str, durable: bool = False) -> None:
    """Write ``text`` to a temporary file and rename it over ``path``. ``durable``: fsync the data before the
    rename and the folder after it, so a power cut leaves either the old or the new file, never a torn one.
    On Windows a rename onto a file another program holds open for a moment (antivirus, SyncThing, an
    editor) fails with PermissionError, so it is retried briefly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))
    if durable:
        _fsync_dir(path.parent)


def _file_digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    return size, h.hexdigest()


def _csv_ids(path: Path) -> set[str]:
    """The ``id`` column of a table (empty when it is missing or unreadable)."""
    import csv

    try:
        with open(path, newline="", encoding="utf-8") as f:
            return {row.get("id", "") for row in csv.DictReader(f)}
    except (OSError, csv.Error, UnicodeDecodeError):
        return set()


def _tail(path: Path, n: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200 * max(n, 1)))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    return lines[-n:] if n else []


# ========================================================================================== jobs
@dataclass
class Job:
    id: str
    label: str
    run_dir: str
    config: dict
    state: str = "queued"
    created: str = ""
    started: str = ""
    finished: str = ""
    error: str = ""
    chunk: dict | None = None
    lanes: str | None = None
    client: str = ""
    client_version: str = ""
    n_inputs: int = 0
    counts: dict = field(default_factory=dict)
    resumed: int = 0
    cancel_requested: bool = False
    # runtime only
    stop: threading.Event = field(default_factory=threading.Event, repr=False)
    log: collections.deque = field(default_factory=lambda: collections.deque(maxlen=LOG_TAIL), repr=False)
    progress: dict = field(default_factory=dict, repr=False)

    RECORD = ("id", "label", "run_dir", "config", "state", "created", "started", "finished", "error", "chunk",
              "lanes", "client", "client_version", "n_inputs", "counts", "resumed", "cancel_requested")

    def to_record(self) -> dict:
        return {k: getattr(self, k) for k in self.RECORD}

    @classmethod
    def from_record(cls, d: dict) -> Job:
        return cls(**{k: d[k] for k in cls.RECORD if k in d})


class JobManager:
    """The queue: runs jobs one at a time (``max_jobs``) in background threads through
    ``jobs.run_resumable``, persists the job list in ``state_dir`` and resumes unfinished jobs on start.
    Used by the HTTP worker and, without a ``state_dir``, by ``grid build`` for the local share."""

    def __init__(self, output_dir: str | Path, name: str, state_dir: str | Path | None = None,
                 lanes: str | None = None, max_jobs: int = 1, log: Callable[[str], None] = _noop,
                 awake: bool = True):
        self.output_dir = Path(output_dir).expanduser()
        self.name = name
        self.state_dir = Path(state_dir).expanduser() if state_dir else None
        self.lanes = lanes
        self.max_jobs = max(1, int(max_jobs))
        self.environment: dict = {}
        self.paused = False
        self._log = log
        self._awake = awake
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._queue: collections.deque[str] = collections.deque()
        self._memory_inputs: dict[str, list] = {}
        self._threads: list[threading.Thread] = []
        self._stopping = False
        self._locked = False

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.state_dir is not None:
            self._take_state_lock()
            self._load_state()
        for k in range(self.max_jobs):
            t = threading.Thread(target=self._runner, name=f"qb-job-{k}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 60.0) -> None:
        """Stop running jobs at the next molecule boundary (they become 'interrupted' and resume on the next
        start) and end the runner threads."""
        with self._cond:
            self._stopping = True
            for job in self._jobs.values():
                if job.state == "running":
                    job.stop.set()
            self._cond.notify_all()
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.1, deadline - time.monotonic()))
        with self._lock:
            self._persist()
        if self._locked and self.state_dir is not None:
            (self.state_dir / STATE_LOCK).unlink(missing_ok=True)
            self._locked = False

    def _take_state_lock(self) -> None:
        path = self.state_dir / STATE_LOCK
        self.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = None
        if not isinstance(info, dict):
            info = None
        if _lock_owner_alive(info):
            raise RuntimeError(f"another worker (pid {info.get('pid')}) uses {self.state_dir}; stop it or give "
                               "this one --state-dir")
        if info:
            self._log(f"Previous worker (pid {info.get('pid')}) ended without removing {path.name}; taking it over")
        _atomic_write(path, json.dumps({"pid": os.getpid(), "pid_started": _process_start(os.getpid()) or 0.0,
                                        "host": socket.gethostname(), "since": _now()}))
        self._locked = True

    def _read_state(self) -> dict | None:
        """The saved job list: ``jobs.json``, else its backup (a power cut can tear the file being written,
        never both). An unreadable file is kept as ``*.unreadable`` for inspection."""
        for name in (STATE_FILE, STATE_BACKUP):
            path = self.state_dir / name
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("jobs", []), list):
                    raise ValueError("not a job list")
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                self._log(f"Could not read {path} ({exc})" + ("; trying the backup" if name == STATE_FILE else ""))
                with contextlib.suppress(OSError):
                    shutil.copyfile(path, path.with_name(path.name + ".unreadable"))
                continue
            if name == STATE_BACKUP:
                self._log(f"Restored the job list from {path}")
            return data
        return None

    def _load_state(self) -> None:
        data = self._read_state()
        if data is None:
            return
        jobs_list = []
        for rec in data.get("jobs", []):
            try:
                jobs_list.append(Job.from_record(rec))
            except TypeError:
                continue
        with self._lock:
            for job in sorted(jobs_list, key=lambda j: j.created):
                if job.state not in STATES:
                    continue
                self._jobs[job.id] = job
                if job.state not in ("running", "interrupted", "queued"):
                    continue
                if job.cancel_requested:  # cancelled, then the worker stopped before the molecule ended
                    job.state, job.finished = "cancelled", job.finished or _now()
                    self._log(f"Job {job.id} ({job.label}) was cancelled before the worker stopped; not resuming it")
                    self._drop_inputs(job.id)
                    self._finish_run_folder(job)
                    continue
                if job.state != "queued":
                    job.resumed += 1
                    self._log(f"Resuming job {job.id} ({job.label}) in {job.run_dir}")
                job.state = "queued"
                self._queue.append(job.id)
            self._persist()

    def _persist(self) -> bool:
        """Save the job list (``jobs.json`` and then its backup, each fsynced). Never raises: a job list that
        cannot be saved for a moment stays in memory and is saved with the next change, and the job threads
        must not die of it."""
        if self.state_dir is None:
            return True
        jobs_sorted = sorted(self._jobs.values(), key=lambda j: j.created)
        finished = [j for j in jobs_sorted if j.state in FINAL_STATES]
        drop = {j.id for j in finished[:-200]}            # keep the last 200 finished jobs
        for job_id in drop:
            self._jobs.pop(job_id, None)
        data = {"format": 1, "worker": self.name, "updated": _now(),
                "jobs": [j.to_record() for j in jobs_sorted if j.id not in drop]}
        text = json.dumps(data, indent=1, default=str) + "\n"
        try:
            _atomic_write(self.state_dir / STATE_FILE, text, durable=True)
            _atomic_write(self.state_dir / STATE_BACKUP, text, durable=True)
        except OSError as exc:
            self._log(f"Could not save the job list in {self.state_dir} ({exc}); kept in memory, saved with the "
                      "next change")
            return False
        return True

    # -------------------------------------------------------------- inputs
    def _inputs_path(self, job_id: str) -> Path:
        return self.state_dir / "inputs" / f"{job_id}.jsonl"

    def _save_inputs(self, job_id: str, inputs: list) -> None:
        if self.state_dir is None:
            self._memory_inputs[job_id] = list(inputs)
            return
        from ..chem import jobs as jobs_mod

        _atomic_write(self._inputs_path(job_id), "".join(jobs_mod.input_to_json(x) + "\n" for x in inputs),
                      durable=True)

    def _load_inputs(self, job_id: str) -> list | None:
        if self.state_dir is None:
            return self._memory_inputs.get(job_id)
        from ..chem import jobs as jobs_mod

        try:
            text = self._inputs_path(job_id).read_text(encoding="utf-8")
        except OSError:
            return None  # resume from the run folder's job table
        return [jobs_mod.input_from_json(line) for line in text.splitlines() if line.strip()]

    def _drop_inputs(self, job_id: str) -> None:
        self._memory_inputs.pop(job_id, None)
        if self.state_dir is not None:
            try:
                self._inputs_path(job_id).unlink(missing_ok=True)
            except OSError as exc:  # held open for a moment on Windows: a leftover file, not an error
                self._log(f"Could not remove the queued inputs of job {job_id} ({exc})")

    # -------------------------------------------------------------- queue
    def _new_run_dir(self, label: str) -> Path:
        from ..chem.inputs import safe_id

        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base = f"{stamp}_{label}_{safe_id(self.name, 'worker')}"[:120]
        taken = {Path(j.run_dir).name for j in self._jobs.values()}
        name, n = base, 2
        while name in taken or (self.output_dir / name).exists():
            name, n = f"{base}-{n}", n + 1
        run_dir = (self.output_dir / name).resolve()
        if not run_dir.is_relative_to(self.output_dir.resolve()):  # cannot happen with a checked label
            raise RequestError(400, "label leads outside the output folder")
        return run_dir

    def submit(self, spec: JobSpec) -> Job:
        with self._cond:
            if self._stopping:
                raise RequestError(503, "the worker is shutting down")
            job_id = secrets.token_hex(6)
            while job_id in self._jobs:
                job_id = secrets.token_hex(6)
            job = Job(id=job_id, label=spec.label, run_dir=str(self._new_run_dir(spec.label)),
                      config=spec.config, created=_now(), chunk=spec.chunk, lanes=spec.lanes,
                      client=spec.client, client_version=spec.client_version, n_inputs=len(spec.inputs))
            try:
                self._save_inputs(job_id, spec.inputs)
            except OSError as exc:
                raise RequestError(503, f"the worker cannot save the job's inputs ({exc})") from None
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._persist()
            self._cond.notify_all()
        self._log(f"Job {job_id} ({spec.label}, {len(spec.inputs)} molecule(s)"
                  + (f" from {spec.client}" if spec.client else "") + f") queued -> {job.run_dir}")
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state in ("queued", "interrupted"):
                with contextlib.suppress(ValueError):
                    self._queue.remove(job_id)
                job.state, job.finished, job.cancel_requested = "cancelled", _now(), True
                self._drop_inputs(job_id)
                self._persist()
            elif job.state == "running":
                job.cancel_requested = True
                job.stop.set()
                self._persist()
        self._log(f"Job {job_id}: cancel requested")
        return job

    def set_paused(self, paused: bool) -> None:
        with self._cond:
            self.paused = bool(paused)
            self._cond.notify_all()
        self._log("Paused: running jobs finish, new ones wait" if paused else "Resumed")

    def queue_state(self) -> dict:
        with self._lock:
            running = [j.id for j in self._jobs.values() if j.state == "running"]
            return {"running": running, "queued": list(self._queue), "paused": self.paused,
                    "max_jobs": self.max_jobs}

    def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until no job is queued or running (tests and the local grid share)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._queue and not any(j.state == "running" for j in self._jobs.values()):
                    return True
            time.sleep(0.05)
        return False

    # -------------------------------------------------------------- running
    def _runner(self) -> None:
        while True:
            with self._cond:
                while not self._stopping and (self.paused or not self._queue):
                    self._cond.wait(1.0)
                if self._stopping:
                    return
                job = self._jobs[self._queue.popleft()]
                job.state, job.started, job.error = "running", _now(), ""
                job.stop.clear()
                self._persist()
            try:
                self._execute(job)
            except Exception as exc:  # the worker must outlive any job
                job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
                self._log(f"Job {job.id} failed: {job.error}")
            try:
                with self._lock:
                    job.finished = _now() if job.state in FINAL_STATES else job.finished
                    if job.state in FINAL_STATES:
                        self._drop_inputs(job.id)
                    self._persist()
            except Exception as exc:  # bookkeeping must not end this thread: the queue would stall silently
                self._log(f"Job {job.id}: could not record its end ({type(exc).__name__}: {exc})")

    def _execute(self, job: Job) -> None:
        from ..chem import jobs as jobs_mod
        from . import profile as profile_mod

        run_dir = Path(job.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / JOB_LOG, "a", encoding="utf-8") as logf:
            def log(msg: str) -> None:
                line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
                job.log.append(line)
                logf.write(line + "\n")
                logf.flush()

            def progress(i: int, n: int, stage: str, mol_id: str) -> None:
                job.progress = {"done": i, "total": n, "stage": stage, "molecule": mol_id}

            log(f"Job {job.id} ({job.label}) on {self.name}, QuantaBench {__version__}"
                + (f" (resume {job.resumed})" if job.resumed else ""))
            self._write_job_file(job)
            tables = None
            try:
                cfg = build_config(job.config, self.output_dir)
                # the job's own spec or --lanes must fit (an error); this machine's setting falls back to 1 lane
                lanes, _spec = profile_mod.resolve_lanes(job.lanes or self.lanes, cfg, log=log)
                chunk = (job.chunk["index"], job.chunk["count"]) if job.chunk and "count" in job.chunk else None
                inputs = self._load_inputs(job.id)
                with keep_awake(log) if self._awake else contextlib.nullcontext():
                    jobs_mod.run_resumable(inputs or None, cfg, run_dir, progress=progress,
                                           should_stop=job.stop.is_set, log=log, lanes=lanes, chunk=chunk)
                # final counts before the state changes, so no client sees "done" with old counts
                job.counts = self._counts(job)
                tables = self._final_tables(run_dir)
                self._checkpoint(run_dir)
                job.state = ("cancelled" if job.cancel_requested else "interrupted" if job.stop.is_set()
                             else "done")
            except Exception as exc:
                job.counts = self._counts(job)
                job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
                for line in traceback.format_exc().rstrip().splitlines():
                    log(line)
                tables = self._final_tables(run_dir, rebuild=True)
            log(f"Job {job.id}: {job.state}" + (f" ({job.error})" if job.error else "")
                + (" " + json.dumps(job.counts) if job.counts else ""))
            self._write_job_file(job, tables)   # last: a synced copy is complete once this matches its tables
        self._log(f"Job {job.id} ({job.label}): {job.state}")

    @staticmethod
    def _final_tables(run_dir: Path, rebuild: bool = False) -> dict:
        """worker.json's ``tables``: size and SHA-256 of summary.csv and descriptors.csv as this job left them,
        and the failed molecules that have no summary row (they crashed the process twice), so that a client
        can tell a complete synced copy from one SyncThing is still bringing. ``rebuild``: write the tables
        from the job table first (a failed run ends without them), unless another process holds the run."""
        from ..chem import jobs as jobs_mod

        db = run_dir / jobs_mod.DB_NAME
        failures: list = []
        if db.is_file():
            with contextlib.suppress(Exception):  # an unreadable table: report the files that are there
                status = jobs_mod.open_run(run_dir)
                failures = list(status.failures)
                if rebuild and not status.active:
                    jobs_mod.rebuild_tables(run_dir)
        files: dict = {}
        for name in TABLES:
            path = run_dir / name
            with contextlib.suppress(OSError):
                if path.is_file():
                    size, digest = _file_digest(path)
                    files[name] = {"size": size, "sha256": digest}
        in_summary = _csv_ids(run_dir / "summary.csv")
        return {"files": files, "unrecorded_failures": [[i, e] for i, e in failures if i not in in_summary],
                "written": _now()}

    def _finish_run_folder(self, job: Job) -> None:
        """A job that ends without running (cancelled while the worker was down): give its run folder the
        tables and final worker.json that a finished job has, and remove the dead process's run lock."""
        from ..chem import jobs as jobs_mod

        run_dir = Path(job.run_dir)
        if not run_dir.is_dir():
            return
        try:
            lock = run_dir / jobs_mod.LOCK_NAME
            with contextlib.suppress(OSError, ValueError, TypeError):
                if (json.loads(lock.read_text(encoding="utf-8")).get("host") == socket.gethostname()
                        and not jobs_mod.open_run(run_dir).active):
                    lock.unlink()
            tables = self._final_tables(run_dir, rebuild=True)
            self._checkpoint(run_dir)
            self._write_job_file(job, tables)
        except Exception as exc:
            self._log(f"Job {job.id}: could not finish its run folder ({type(exc).__name__}: {exc})")

    @staticmethod
    def _checkpoint(run_dir: Path) -> None:
        """Fold the WAL into jobs.sqlite: a status poll whose read-only connection overlapped the run's last
        commit keeps -wal/-shm alive, and the synced or archived folder is cleaner as one file. Only the
        worker, the run folder's writer, does this."""
        from ..chem import jobs as jobs_mod

        with contextlib.suppress(Exception):
            jobs_mod.JobStore(run_dir, create=False).close()

    def _write_job_file(self, job: Job, tables: dict | None = None) -> None:
        data = {"job": job.to_record(), "environment": self.environment}
        if tables is not None:
            data["tables"] = tables
        try:
            _atomic_write(Path(job.run_dir) / JOB_FILE, json.dumps(data, indent=1, default=str) + "\n")
        except OSError as exc:
            self._log(f"Job {job.id}: could not write {JOB_FILE} ({exc})")

    def _counts(self, job: Job) -> dict:
        from ..chem import jobs as jobs_mod

        if not (Path(job.run_dir) / jobs_mod.DB_NAME).is_file():
            return dict(job.counts)
        try:
            return dict(jobs_mod.open_run(job.run_dir).counts)
        except Exception:  # a table being created, a locked file on Windows: keep the last known counts
            return dict(job.counts)

    def public(self, job: Job, tail: int = 20) -> dict:
        """The job as the API reports it."""
        counts = self._counts(job) if job.state == "running" else dict(job.counts)
        lines = list(job.log)[-tail:] if tail else []
        if tail and not lines:
            lines = _tail(Path(job.run_dir) / JOB_LOG, tail)
        return {**{k: v for k, v in job.to_record().items() if k != "config"},
                "method": job.config.get("method", ""), "run_folder": Path(job.run_dir).name,
                "worker": self.name, "counts": counts, "progress": dict(job.progress), "log_tail": lines}


# ======================================================================================== server
class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR on Windows would let another program bind the same port; use SO_EXCLUSIVEADDRUSE there
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address: tuple[str, int], worker: Worker):
        self.worker = worker
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, _Handler)

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)   # HTTPServer's version does a slow reverse DNS lookup
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), port


_ROUTES = (
    ("GET", re.compile(r"/v1/info"), "info"),
    ("GET", re.compile(r"/v1/jobs"), "list_jobs"),
    ("POST", re.compile(r"/v1/jobs"), "submit"),
    ("GET", re.compile(r"/v1/jobs/([0-9a-f]{12})"), "job"),
    ("POST", re.compile(r"/v1/jobs/([0-9a-f]{12})/cancel"), "cancel"),
    ("GET", re.compile(r"/v1/jobs/([0-9a-f]{12})/archive"), "archive"),
    ("POST", re.compile(r"/v1/pause"), "pause"),
)


class _ZipStream:
    """Write-only, unseekable file object over the response socket (zipfile then uses data descriptors).
    After ``abort()`` every write is dropped, so a ``ZipFile`` closed after an error cannot add the central
    directory that would make a cut-off archive look complete."""

    def __init__(self, raw):
        self.raw = raw
        self.aborted = False

    def write(self, data) -> int:
        if not self.aborted:
            self.raw.write(data)
        return len(data)

    def flush(self) -> None:
        if not self.aborted:
            self.raw.flush()

    def abort(self) -> None:
        self.aborted = True


def archive_members(run_dir: Path) -> list[Path]:
    """The files of a run folder that go into its archive: no links, no run lock, no SyncThing temporary
    files (they come and go while SyncThing pulls), no manifest of an earlier download."""
    from ..chem import jobs as jobs_mod

    files = []
    for root, dirs, names in os.walk(run_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink() and not d.startswith(SYNC_TEMP_PREFIXES))
        for name in sorted(names):
            p = Path(root) / name
            if (name in (jobs_mod.LOCK_NAME, ARCHIVE_MANIFEST) or name.startswith(SYNC_TEMP_PREFIXES)
                    or p.is_symlink() or not p.is_file()):
                continue
            files.append(p)
    return files


class _Handler(BaseHTTPRequestHandler):
    server_version = "QuantaBench-worker"
    sys_version = ""
    timeout = 120  # seconds per socket operation: a stalled client cannot hold a thread forever

    # -------------------------------------------------------------- plumbing
    def log_message(self, format: str, *args) -> None:  # BaseHTTPRequestHandler's signature
        self.server.worker.log(f"{self.client_address[0]} {format % args}")

    def log_request(self, code="-", size="-") -> None:
        code_int = int(code) if str(code).isdigit() or isinstance(code, int) else 0
        if self.command != "GET" or code_int >= 400:    # status polls would flood the log
            self.log_message('"%s" %s', self.requestline, code)

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def _json(self, status: int, data, headers: dict | None = None) -> None:
        body = (json.dumps(data, default=str) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(VERSION_HEADER, __version__)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        scheme, _, given = header.partition(" ")
        expected = self.server.worker.token.encode("utf-8")
        ok = hmac.compare_digest(given.strip().encode("utf-8"), expected)
        return scheme.lower() == "bearer" and ok

    def _body(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(411, "send a Content-Length (chunked bodies are not accepted)")
        length = self.headers.get("Content-Length", "0").strip() or "0"
        if not length.isdigit():
            raise RequestError(400, "bad Content-Length")
        n = int(length)
        if n > self.server.worker.max_request_bytes:
            raise RequestError(413, f"request larger than {self.server.worker.max_request_bytes} bytes")
        if n == 0:
            return {}
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise RequestError(415, "Content-Type must be application/json")
        raw = self.rfile.read(n)
        self._body_read = True
        if len(raw) != n:
            raise RequestError(400, "request body ended early")
        return parse_json(raw)

    def _drain(self) -> None:
        """Read a small unread body before an error reply, so closing the socket does not reset the
        connection before the client has read the reply."""
        length = self.headers.get("Content-Length", "0").strip()
        if self.command == "POST" and not getattr(self, "_body_read", False) and length.isdigit() \
                and 0 < int(length) <= 1_048_576:
            with contextlib.suppress(OSError):
                self.rfile.read(int(length))

    def _handle(self, method: str) -> None:
        worker = self.server.worker
        self._body_read = False
        try:
            if not self._authorized():
                worker.log(f"{self.client_address[0]}: rejected {method} {self.path[:80]} (missing or wrong token)")
                raise RequestError(401, "missing or wrong token (Authorization: Bearer <settings.dist.worker_token>)")
            path, _, query = self.path.partition("?")
            allowed = []
            for verb, pattern, name in _ROUTES:
                m = pattern.fullmatch(path)
                if m:
                    allowed.append(verb)
                    if verb == method:
                        getattr(self, f"_r_{name}")(*m.groups(), query=parse_qs(query))
                        return
            if allowed:
                raise RequestError(405, f"use {' or '.join(allowed)} for {path}")
            raise RequestError(404, f"no such endpoint {path[:80]!r}")
        except RequestError as exc:
            extra = {"WWW-Authenticate": 'Bearer realm="quantabench"'} if exc.status == 401 else None
            self._drain()
            with contextlib.suppress(OSError):
                self._json(exc.status, {"error": exc.message}, extra)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            worker.log("Internal error:\n" + traceback.format_exc())
            with contextlib.suppress(OSError):
                self._json(500, {"error": "internal error (see the worker's log)"})

    def _job_or_404(self, job_id: str) -> Job:
        job = self.server.worker.manager.get(job_id)
        if job is None:
            raise RequestError(404, f"no job {job_id}")
        return job

    # -------------------------------------------------------------- routes
    def _r_info(self, query) -> None:
        self._json(200, self.server.worker.info())

    def _r_list_jobs(self, query) -> None:
        mgr = self.server.worker.manager
        self._json(200, {"jobs": [mgr.public(j, tail=0) for j in mgr.list()]})

    def _r_submit(self, query) -> None:
        worker = self.server.worker
        body = self._body()
        client_version = self.headers.get(VERSION_HEADER, "").strip()
        allow = body.get("allow_version_mismatch") is True or worker.allow_version_mismatch
        if client_version != __version__ and not allow:
            raise RequestError(409, f"QuantaBench version mismatch: client {client_version or '(not sent)'}, "
                                    f"worker {__version__}. Update both machines to the same version (git pull), "
                                    "or allow_version_mismatch.")
        spec = validate_job(body, client_version)
        job = worker.manager.submit(spec)
        self._json(201, {"id": job.id, "job": worker.manager.public(job, tail=0)})

    def _r_job(self, job_id: str, query) -> None:
        job = self._job_or_404(job_id)
        try:
            tail = max(0, min(LOG_TAIL, int(query.get("tail", ["20"])[0])))
        except ValueError:
            raise RequestError(400, "tail must be an integer") from None
        self._json(200, self.server.worker.manager.public(job, tail=tail))

    def _r_cancel(self, job_id: str, query) -> None:
        self._body()
        self._job_or_404(job_id)
        mgr = self.server.worker.manager
        self._json(200, mgr.public(mgr.cancel(job_id), tail=0))

    def _r_pause(self, query) -> None:
        body = self._body()
        if set(body) - {"paused"} or not isinstance(body.get("paused"), bool):
            raise RequestError(400, 'body must be {"paused": true|false}')
        mgr = self.server.worker.manager
        mgr.set_paused(body["paused"])
        self._json(200, mgr.queue_state())

    def _r_archive(self, job_id: str, query) -> None:
        job = self._job_or_404(job_id)
        if job.state in ("queued", "running"):
            raise RequestError(409, f"job {job_id} is {job.state}; download it when it is done")
        run_dir = Path(job.run_dir)
        if not run_dir.is_dir():
            raise RequestError(404, f"the run folder of job {job_id} no longer exists")
        files = archive_members(run_dir)
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="{run_dir.name}.zip"')
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header(VERSION_HEADER, __version__)
        self.end_headers()
        # No Content-Length: the zip is streamed and the connection closes at the end. The manifest (every
        # member and its size) is the last member, and an error drops the central directory, so a cut-off or
        # failed stream reads as damaged or incomplete, never as a shorter result.
        stream = _ZipStream(self.wfile)
        zf = zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6)
        top = run_dir.name
        try:
            entries = []
            for p in files:
                rel = p.relative_to(run_dir).as_posix()
                zf.write(p, f"{top}/{rel}")
                entries.append({"path": rel, "size": zf.infolist()[-1].file_size})
            manifest = {"format": 1, "job": job_id, "run_folder": top, "files": entries, "written": _now()}
            zf.writestr(f"{top}/{ARCHIVE_MANIFEST}", json.dumps(manifest, indent=1) + "\n")
            zf.close()
            self.wfile.flush()
        except Exception as exc:  # the headers are out: no JSON error now, only an incomplete archive
            stream.abort()
            with contextlib.suppress(Exception):
                zf.close()
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                self.server.worker.log(f"Archive of job {job_id} stopped ({type(exc).__name__}: {exc}); the "
                                       "client gets an incomplete archive and refuses it")
        self.close_connection = True


class Worker:
    """The HTTP worker: binds at construction (``port=0`` picks a free port), ``start()`` serves in a
    background thread, ``serve_forever()`` in the calling thread, ``stop()`` ends both."""

    def __init__(self, host: str, port: int, token: str, output_dir: str | Path, *,
                 state_dir: str | Path | None = None, name: str | None = None, lanes: str | None = None,
                 max_jobs: int = 1, allow_version_mismatch: bool = False,
                 max_request_bytes: int = MAX_REQUEST_BYTES, publish: bool | None = None,
                 log: Callable[[str], None] = _noop, awake: bool = True):
        from . import profile as profile_mod

        if not token or len(token) < 8:
            raise ValueError("a worker needs a token")
        self.token = token
        self.log = log
        self.allow_version_mismatch = allow_version_mismatch
        self.max_request_bytes = int(max_request_bytes)
        self.name = name or profile_mod.machine_name()
        self.server = _Server((host, int(port)), self)
        bound_host, bound_port = self.server.server_address[:2]
        self.host, self.port = str(bound_host), int(bound_port)
        url_host = f"[{self.host}]" if ":" in self.host else self.host
        self.url = f"http://{url_host}:{self.port}"
        self.manager = JobManager(output_dir, self.name, state_dir=state_dir, lanes=lanes, max_jobs=max_jobs,
                                  log=log, awake=awake)
        self.commit = git_commit()
        prof = profile_mod.local_profile()
        prof = dataclasses.replace(prof, name=self.name, address=self.url, updated=_now())
        self.profile = prof
        loopback = ipaddress.ip_address(self.host).is_loopback
        self.publish = (not loopback) if publish is None else publish
        self.manager.environment = {"worker": self.name, "url": self.url, "quantabench_version": __version__,
                                    "commit": self.commit, "python": sys.version.split()[0],
                                    "platform": sys.platform, "profile": prof.to_dict()}
        self._thread: threading.Thread | None = None
        self._started = False

    def info(self) -> dict:
        return {"api": API_VERSION, "name": self.name, "url": self.url, "quantabench_version": __version__,
                "commit": self.commit, "python": sys.version.split()[0], "platform": sys.platform,
                "output_dir": str(self.manager.output_dir), "lanes": self.lanes_spec(),
                "queue": self.manager.queue_state(), "profile": self.profile.to_dict()}

    def lanes_spec(self) -> str:
        """The lane spec jobs without their own run with: ``--lanes``, else this machine's setting."""
        if self.manager.lanes:
            return self.manager.lanes
        with contextlib.suppress(Exception):
            return settings_mod.load().lanes_for()
        return "1"

    def _publish_profile(self) -> None:
        """Write this worker's URL into its machine profile in the (synced) registry, keeping measured
        throughput from an earlier ``machines --bench``."""
        from . import profile as profile_mod

        if not self.publish:
            return
        old = next((p for p in profile_mod.load_profiles() if p.name == self.name), None)
        prof = self.profile
        if old is not None and not prof.throughput and old.throughput:
            prof = dataclasses.replace(prof, throughput=dict(old.throughput))
        try:
            path = profile_mod.save_profile(prof)
            self.log(f"Published {self.url} in {path}")
        except OSError as exc:
            self.log(f"Could not write the machine profile ({exc}); clients need --worker {self.url}")

    def start(self) -> Worker:
        self.manager.start()
        self._started = True
        self._publish_profile()
        self._thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.2},
                                        name="qb-worker-http", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.manager.start()
        self._started = True
        self._publish_profile()
        try:
            self.server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            self.log("Stopping: running jobs stop after the current molecule and resume on the next start")
        finally:
            self.stop()

    def stop(self, timeout: float = 60.0) -> None:
        if self._thread is not None:
            self.server.shutdown()
            self._thread.join(10)
            self._thread = None
        self.server.server_close()
        if self._started:
            self._started = False
            self.manager.stop(timeout)

    def __enter__(self) -> Worker:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


# =========================================================================================== CLI
def _stamped_log(log_file: str | None) -> Callable[[str], None]:
    lock = threading.Lock()
    path = Path(log_file).expanduser() if log_file else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with lock:
            print(line, flush=True)
            if path is not None:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
    return log


BIND_RETRY_S = 5.0


def bind_with_retry(make: Callable[[], Worker], wait: float, log: Callable[[str], None] = _noop) -> Worker:
    """``make()`` (which binds the socket), retried for up to ``wait`` seconds while the address is not on
    this machine yet: at login ``tailscale ip -4`` can name the address before the Tailscale adapter has
    it, and the worker would otherwise exit (Windows Task Scheduler does not reliably start it again)."""
    deadline = time.monotonic() + max(0.0, wait)
    while True:
        try:
            return make()
        except OSError as exc:
            if exc.errno not in _ADDR_NOT_AVAILABLE or time.monotonic() >= deadline:
                raise
            log(f"Cannot listen yet ({exc}); the Tailscale interface may still be coming up, retrying")
            time.sleep(BIND_RETRY_S)


def _cli_worker(args) -> int:
    log = _stamped_log(args.log_file)
    s = settings_mod.load()
    try:
        host = resolve_bind(args.bind or s.dist.worker_bind, args.allow_any_address, args.tailscale_wait, log)
        token = ensure_token(log)
    except (ValueError, RuntimeError) as exc:
        print(f"worker: {exc}", file=sys.stderr)
        return 2
    if not (ipaddress.ip_address(host).is_loopback or is_tailscale_address(host)):
        log(f"Warning: {host} is not a Tailscale or loopback address; anyone on that network who has the token "
            "can submit jobs, and traffic is not encrypted")
    output = s.output_path()
    port = s.dist.worker_port if args.port is None else args.port
    if args.lanes:  # an explicit --lanes must fit this machine; the setting falls back to one lane per job
        from . import profile as profile_mod

        try:
            profile_mod.resolve_lanes(args.lanes, build_config({}, output))
        except ValueError as exc:
            print(f"worker: --lanes {args.lanes}: {exc}", file=sys.stderr)
            return 2

    def make() -> Worker:
        return Worker(host, port, token, output, state_dir=args.state_dir or default_state_dir(), name=args.name,
                      lanes=args.lanes or None, max_jobs=args.max_jobs,
                      allow_version_mismatch=args.allow_version_mismatch,
                      max_request_bytes=int(args.max_request_mb * 1024 * 1024), publish=args.publish, log=log)

    tailscale = (args.bind or s.dist.worker_bind or "tailscale").strip() == "tailscale"
    try:
        worker = bind_with_retry(make, args.tailscale_wait if tailscale else 0.0, log)
    except OSError as exc:
        print(f"worker: cannot listen on {host}:{port}: {exc}", file=sys.stderr)
        return 2

    def on_term(_signum, _frame) -> None:
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGBREAK"):   # launchd unload / `kill`; Ctrl+Break on Windows: stop cleanly
        with contextlib.suppress(ValueError, AttributeError, OSError):
            signal.signal(getattr(signal, name), on_term)
    log(f"QuantaBench {__version__} worker '{worker.name}' at {worker.url}; results in {output}; "
        f"lanes {worker.lanes_spec()}, {worker.manager.max_jobs} job(s) at a time. Ctrl+C stops it (running "
        "jobs resume on the next start).")
    try:
        worker.serve_forever()
    except RuntimeError as exc:  # state folder in use
        print(f"worker: {exc}", file=sys.stderr)
        return 2
    return 0


def register_cli(subparsers) -> None:
    w = subparsers.add_parser(
        "worker", help="run jobs for your other machines (HTTP/JSON over Tailscale)",
        description="Serves the QuantaBench worker API on this machine's Tailscale address. Other machines "
                    "submit jobs with `grid build`; results go to this machine's output folder (synced) and "
                    "can be downloaded as a zip. See docs/WORKERS.md.")
    w.add_argument("--bind", help="tailscale (default from settings), localhost, or an IP address")
    w.add_argument("--port", type=int, help=f"TCP port (default from settings, {DEFAULT_PORT})")
    w.add_argument("--lanes", help="lane spec for jobs that do not set one (default from settings)")
    w.add_argument("--max-jobs", type=int, default=1, help="jobs at a time (default 1)")
    w.add_argument("--name", help="worker name (default: this machine's name)")
    w.add_argument("--state-dir", help="folder for the job list and queued inputs (default: the per-user "
                                       "state folder; not the synced settings)")
    w.add_argument("--allow-any-address", action="store_true",
                   help="allow --bind 0.0.0.0 / :: (every network); only behind a firewall rule")
    w.add_argument("--allow-version-mismatch", action="store_true",
                   help="accept jobs from a different QuantaBench version")
    w.add_argument("--max-request-mb", type=float, default=MAX_REQUEST_BYTES / 2**20,
                   help="largest job request accepted, MB")
    w.add_argument("--publish", action=argparse.BooleanOptionalAction, default=None,
                   help="write this worker's URL into its machine profile (default: unless bound to loopback)")
    w.add_argument("--tailscale-wait", type=float, default=120.0,
                   help="with --bind tailscale: seconds to wait for Tailscale to come up (autostart at login)")
    w.add_argument("--log-file", help="also append the worker log to this file")
    w.set_defaults(func=_cli_worker)


__all__ = ("API_VERSION", "Job", "JobManager", "JobSpec", "RequestError", "VERSION_HEADER",
                          "Worker", "build_config", "ensure_token", "resolve_bind", "tailscale_ipv4",
                          "validate_config", "validate_inputs", "validate_job")
