"""Submitting jobs to workers, and ``quantabench-cli grid``: one command that splits a molecule list across
your machines, follows the progress and merges the result tables.

``WorkerClient(url, token)``  info, submit, status, jobs, cancel, pause, wait, download_archive
``known_workers`` / ``discover``  workers from the machine registry (the URL a worker writes into its
                              profile), ``settings.dist.workers`` overrides and ``--worker [NAME=]URL``
``grid_build(...)``           split (``dist.plan``), submit each share, run the "local" share in this process,
                              follow, then merge with ``jobs.merge_tables``. A worker's run folder that has
                              synced here (SyncThing) is read in place; otherwise its zip archive is downloaded
                              into the grid folder first.

A grid run is recorded as ``<output folder>/<stamp>_grid_<label>/grid.json`` (machines, job ids, run folders);
the merged ``summary.csv``/``descriptors.csv`` land next to it. ``grid status|cancel|merge <grid id>`` use it.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import functools
import hashlib
import http.client
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import stat
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlsplit

from .. import __version__
from .. import settings as settings_mod
from .worker import (
    ARCHIVE_MANIFEST,
    DEFAULT_PORT,
    FINAL_STATES,
    JOB_FILE,
    RESOURCE_FIELDS,
    TABLES,
    TAILSCALE_NET4,
    TAILSCALE_NET6,
    VERSION_HEADER,
)

LOCAL = "local"
GRID_FILE = "grid.json"
MAX_JSON_RESPONSE = 32 * 1024 * 1024
#: client-side share state: the worker answered that it does not know the job (404), refused the token (401)
#: or the request (other 4xx). Waiting longer cannot change that, so following stops for this share.
LOST = "lost"
DONE_STATES = (*FINAL_STATES, LOST)
#: a run folder name as a worker makes it (``<stamp>_<label>_<worker>``, maybe with a ``-2`` suffix)
_FOLDER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,159}")


def _noop(*_args) -> None:
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class WorkerError(RuntimeError):
    """A worker refused a request (``status`` = HTTP status) or could not be reached (``status`` 0)."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class GridError(RuntimeError):
    """A grid run cannot start or continue (unknown or unreachable machine, version mismatch, ...)."""


class _NoRedirect(urlrequest.HTTPRedirectHandler):
    """urllib would follow a redirect with the Authorization header still attached."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urlerror.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


def normalize_url(url: str) -> str:
    """'100.90.226.110' / 'btm-nova:8765' / 'http://host:8765/' -> 'http://host:port'."""
    text = (url or "").strip()
    if "://" not in text:
        text = "http://" + text
    parts = urlsplit(text)
    try:
        port = parts.port or DEFAULT_PORT
    except ValueError:
        raise ValueError(f"bad port in worker URL {url!r}") from None
    if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or parts.path not in ("", "/")):
        raise ValueError(f"worker URL must look like http://HOST:PORT, not {url!r}")
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}:{port}"


def _ip(text: str):
    addr = ipaddress.ip_address(str(text).split("%")[0])
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr


def is_tailnet(addr) -> bool:
    """``addr`` (ip_address) is in Tailscale's ranges (100.64.0.0/10, fd7a:115c:a1e0::/48)."""
    return addr in (TAILSCALE_NET4 if addr.version == 4 else TAILSCALE_NET6)


def check_route(peer: str, local: str, allow_any_host: bool = False) -> None:
    """Refuse a connection that the token must not cross; called once the TCP connection is up and before a
    byte is sent. Loopback is fine. A Tailscale peer needs a Tailscale *local* address too: 100.64.0.0/10 is
    also carrier-grade NAT space, and with Tailscale stopped or disconnected here the connection leaves through
    the default route (a café's Wi-Fi), where plain HTTP is readable; ``allow_any_host`` does not change that.
    Any other peer only with ``allow_any_host``. Checking the connected socket, not a name lookup, also
    covers MagicDNS names that resolve elsewhere later (another network, an expired cache)."""
    p, here = _ip(peer), _ip(local)
    if p.is_loopback:
        return
    if is_tailnet(p):
        if is_tailnet(here):
            return
        raise WorkerError(f"Tailscale is not connected on this machine: the connection to {p} goes out through "
                          f"{here}, not the tailnet, so the token would cross that network unencrypted. Start "
                          "Tailscale (`tailscale status`) and try again.")
    if not allow_any_host:
        raise WorkerError(f"{p} is not a Tailscale or loopback address: the token would cross that network "
                          "unencrypted (allow_any_host / --allow-any-host to do it anyway)")


def check_destination(host: str, port: int | None) -> None:
    """Before connecting: every address ``host`` resolves to must be loopback or Tailscale, so that a LAN or
    public address is refused without a packet sent to it. ``check_route`` then checks the connection that
    is actually made (the name may resolve differently by then)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise WorkerError(f"cannot resolve {host}: {exc}") from None
    for info in infos:
        addr = _ip(info[4][0])
        if not (addr.is_loopback or is_tailnet(addr)):
            raise WorkerError(f"{host} is {addr}, not a Tailscale or loopback address: the token would cross that "
                              "network unencrypted (allow_any_host / --allow-any-host to do it anyway)")


class _CheckedHTTPConnection(http.client.HTTPConnection):
    """``HTTPConnection`` that checks every new connection (``check_destination`` before it, ``check_route``
    once it is up), before the request and its Authorization header are written."""

    def __init__(self, *args, allow_any_host: bool = False, **kw):
        super().__init__(*args, **kw)
        self.allow_any_host = allow_any_host

    def connect(self) -> None:
        if not self.allow_any_host:
            check_destination(self.host, self.port)
        super().connect()
        try:
            check_route(self.sock.getpeername()[0], self.sock.getsockname()[0], self.allow_any_host)
        except BaseException:
            self.sock.close()
            self.sock = None
            raise


class _CheckedHTTPHandler(urlrequest.HTTPHandler):
    def __init__(self, allow_any_host: bool = False):
        super().__init__()
        self.allow_any_host = allow_any_host

    def http_open(self, req):
        return self.do_open(functools.partial(_CheckedHTTPConnection, allow_any_host=self.allow_any_host), req)


def _safe_name(name) -> bool:
    """A machine name that can be one folder name (``parts/<machine>``): the label rules."""
    from .worker import _LABEL_RE

    return isinstance(name, str) and bool(_LABEL_RE.fullmatch(name))


def config_for_workers(cfg, keep_resources: Sequence[str] = ()) -> dict:
    """The ``BuildConfig`` as a worker accepts it: no output folder (each worker uses its own) and, unless
    asked, no resource fields (each machine uses its own threads/GPU settings)."""
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    d.pop("output_dir", None)
    for name in RESOURCE_FIELDS:
        if name not in keep_resources:
            d.pop(name, None)
    if "formats" in d:
        d["formats"] = list(d["formats"])
    return d


def job_body(inputs, cfg, label: str, *, chunk: dict | None = None, lanes: str | None = None,
             client: str = "", allow_version_mismatch: bool = False,
             keep_resources: Sequence[str] = ()) -> dict:
    from ..chem import jobs

    body: dict = {"inputs": [x if isinstance(x, str) else jobs.input_to_json(x) for x in inputs],
                  "config": cfg if isinstance(cfg, dict) else config_for_workers(cfg, keep_resources),
                  "label": label}
    if chunk:
        body["chunk"] = chunk
    if lanes:
        body["lanes"] = lanes
    if client:
        body["client"] = client
    if allow_version_mismatch:
        body["allow_version_mismatch"] = True
    return body


def _check_manifest(zf: zipfile.ZipFile, members: list, top: str) -> None:
    """The worker writes ``<top>/.qb-archive.json`` (every other member and its size) as the last member, and
    only once every file is in: a stream that failed or was cut off on the way lacks it or does not match."""
    name = f"{top}/{ARCHIVE_MANIFEST}"
    if not members or members[-1].filename != name:
        raise WorkerError("the archive has no manifest at its end: the download is incomplete, or the worker "
                          "is older than this client (update it)")
    try:
        manifest = json.loads(zf.read(name).decode("utf-8"))
        listed = {f"{top}/{e['path']}": int(e["size"]) for e in manifest["files"]}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise WorkerError(f"the archive manifest cannot be read ({exc})") from None
    actual = {i.filename: i.file_size for i in members[:-1] if not i.is_dir()}
    if listed != actual:
        missing = sorted(set(listed) - set(actual))[:3]
        raise WorkerError("the archive does not match its manifest" + (f" (missing {', '.join(missing)})"
                                                                       if missing else ""))


def safe_extract(zip_path: str | Path, dest: str | Path, expected_top: str | None = None,
                 require_manifest: bool = False) -> Path:
    """Extract a worker archive (one top folder) into ``dest`` and return that folder. Refuses absolute
    paths, '..', drive letters, backslashes and links; a cut-off download fails here. ``expected_top``: the
    run folder the archive must hold (the job's). ``require_manifest``: the worker's manifest must be the
    last member and list exactly the other members (``download_archive``)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            tops = set()
            for info in members:
                name = info.filename
                parts = PurePosixPath(name).parts
                if (not name or name.startswith("/") or "\\" in name or ":" in name or not parts
                        or any(p in ("..", ".") for p in parts)):
                    raise WorkerError(f"unsafe path in archive: {name!r}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise WorkerError(f"archive member {name!r} is a link")
                tops.add(parts[0])
            if len(tops) != 1:
                raise WorkerError("a worker archive holds exactly one run folder")
            top = next(iter(tops))
            if expected_top is not None and top != expected_top:
                raise WorkerError(f"the archive holds {top!r}, not the job's run folder {expected_top!r}")
            if require_manifest:
                _check_manifest(zf, members, top)
            bad = zf.testzip()
            if bad:
                raise WorkerError(f"archive member {bad} is damaged")
            staging = dest / f".extract-{secrets.token_hex(4)}"
            try:
                zf.extractall(staging)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
    except zipfile.BadZipFile as exc:
        raise WorkerError(f"the archive is incomplete or damaged ({exc})") from None
    target = dest / top
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging / top, target)
    shutil.rmtree(staging, ignore_errors=True)
    return target


# ======================================================================================== client
class WorkerClient:
    """One worker's API. ``token`` is ``settings.dist.worker_token`` (the same on every machine)."""

    def __init__(self, url: str, token: str, timeout: float = 30.0, allow_any_host: bool = False):
        self.url = normalize_url(url)
        self.token = token or ""
        self.timeout = timeout
        self.allow_any_host = allow_any_host
        # never via a proxy, never a redirect; every connection is checked before the token is written
        self._opener = urlrequest.build_opener(urlrequest.ProxyHandler({}), _NoRedirect(),
                                               _CheckedHTTPHandler(allow_any_host))

    def __repr__(self) -> str:  # never shows the token
        return f"WorkerClient({self.url!r})"

    def _open(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        if not self.token:
            raise WorkerError("no worker token: settings.dist.worker_token is empty (start a worker once, or wait "
                              "for the synced settings)")
        data = json.dumps(body).encode("utf-8") if body is not None else (b"" if method == "POST" else None)
        req = urlrequest.Request(self.url + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header(VERSION_HEADER, __version__)
        req.add_header("User-Agent", f"QuantaBench/{__version__}")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            return self._opener.open(req, timeout=timeout or self.timeout)
        except urlerror.HTTPError as exc:
            try:
                message = json.loads(exc.read(65_536).decode("utf-8")).get("error", "")
            except (ValueError, OSError, AttributeError):
                message = str(exc.reason)
            raise WorkerError(f"{self.url}{path}: HTTP {exc.code}: {message}", exc.code) from None
        except (urlerror.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise WorkerError(f"cannot reach {self.url}: {reason}") from None

    def _json(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        with self._open(method, path, body, timeout) as resp:
            raw = resp.read(MAX_JSON_RESPONSE + 1)
        if len(raw) > MAX_JSON_RESPONSE:
            raise WorkerError(f"{self.url}{path}: response too large")
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise WorkerError(f"{self.url}{path}: not a QuantaBench worker (no JSON reply)") from None
        if not isinstance(data, dict):
            raise WorkerError(f"{self.url}{path}: unexpected reply")
        return data

    # -------------------------------------------------------------- API
    def info(self, timeout: float | None = None) -> dict:
        return self._json("GET", "/v1/info", timeout=timeout)

    def submit_body(self, body: dict) -> dict:
        return self._json("POST", "/v1/jobs", body, timeout=max(self.timeout, 120.0))

    def submit(self, inputs, cfg, label: str, **kw) -> str:
        """Queue a job; returns its id. ``kw``: chunk, lanes, client, allow_version_mismatch, keep_resources."""
        return self.submit_body(job_body(inputs, cfg, label, **kw))["id"]

    def status(self, job_id: str, tail: int = 20) -> dict:
        return self._json("GET", f"/v1/jobs/{_job_id(job_id)}?tail={int(tail)}")

    def jobs(self) -> list[dict]:
        return self._json("GET", "/v1/jobs")["jobs"]

    def cancel(self, job_id: str) -> dict:
        return self._json("POST", f"/v1/jobs/{_job_id(job_id)}/cancel")

    def pause(self, paused: bool = True) -> dict:
        return self._json("POST", "/v1/pause", {"paused": bool(paused)})

    def wait(self, job_id: str, poll: float = 5.0, timeout: float | None = None,
             on_status: Callable[[dict], None] = _noop) -> dict:
        """Poll until the job is done, failed or cancelled; returns the last status. A worker that is briefly
        unreachable (restart, network) is retried until ``timeout``."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                st = self.status(job_id, tail=0)
                on_status(st)
                if st["state"] in FINAL_STATES:
                    return st
            except WorkerError as exc:
                if exc.status:            # the worker answered: no such job, auth, ...
                    raise
            if deadline is not None and time.monotonic() > deadline:
                raise WorkerError(f"job {job_id} on {self.url} not finished after {timeout:.0f} s")
            time.sleep(poll)

    def download_archive(self, job_id: str, dest_dir: str | Path, expected_top: str | None = None) -> Path:
        """Download the job's run folder as a zip and extract it into ``dest_dir``; returns the run folder.
        The archive must end with the worker's manifest and, with ``expected_top``, hold that run folder."""
        dest_dir = Path(dest_dir).expanduser()
        dest_dir.mkdir(parents=True, exist_ok=True)
        part = dest_dir / f".{_job_id(job_id)}.zip.part"
        try:
            with self._open("GET", f"/v1/jobs/{job_id}/archive", timeout=max(self.timeout, 300.0)) as resp, \
                    open(part, "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
            return safe_extract(part, dest_dir, expected_top=expected_top, require_manifest=True)
        except OSError as exc:
            raise WorkerError(f"downloading job {job_id} from {self.url} failed: {exc}") from None
        finally:
            part.unlink(missing_ok=True)


def same_commit(a: str, b: str) -> bool:
    """Two short commits (12 or more hex digits) name the same commit; unknown ('') counts as the same."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    return not a or not b or a.startswith(b) or b.startswith(a)


def _job_id(job_id: str) -> str:
    if not isinstance(job_id, str) or len(job_id) != 12 or any(c not in "0123456789abcdef" for c in job_id):
        raise WorkerError(f"bad job id {job_id!r}")
    return job_id


# ===================================================================================== discovery
@dataclass
class WorkerRef:
    name: str
    url: str
    source: str                    # "registry" | "settings" | "option"
    info: dict | None = None
    error: str = ""

    @property
    def reachable(self) -> bool:
        return self.info is not None


def known_workers(extra: Sequence[str] = ()) -> dict[str, WorkerRef]:
    """Name -> worker: registry profiles' ``address``, then ``settings.dist.workers``, then ``extra``
    (``NAME=URL`` or ``URL``; a bare URL is named after the worker's own answer by ``discover``)."""
    from . import profile

    refs: dict[str, WorkerRef] = {}
    for p in profile.load_profiles():          # newest first
        if p.address and p.name not in refs:
            with contextlib.suppress(ValueError):
                refs[p.name] = WorkerRef(p.name, normalize_url(p.address), "registry")
    for name, url in settings_mod.load().dist.workers.items():
        with contextlib.suppress(ValueError):
            refs[name] = WorkerRef(name, normalize_url(url), "settings")
    for item in extra:
        name, sep, url = item.partition("=")
        if not sep:
            name, url = "", item
        try:
            url = normalize_url(url)
        except ValueError as exc:
            raise GridError(str(exc)) from None
        refs[name.strip() or f"?{url}"] = WorkerRef(name.strip(), url, "option")
    return refs


def discover(extra: Sequence[str] = (), token: str | None = None, timeout: float = 3.0,
             allow_any_host: bool = False) -> dict[str, WorkerRef]:
    """``known_workers`` with each one's ``/v1/info`` fetched in parallel (``info`` or ``error`` set)."""
    refs = known_workers(extra)
    token = settings_mod.load().dist.worker_token if token is None else token

    def probe(ref: WorkerRef) -> None:
        try:
            ref.info = WorkerClient(ref.url, token, timeout, allow_any_host).info(timeout=timeout)
        except WorkerError as exc:
            ref.error = str(exc)

    threads = [threading.Thread(target=probe, args=(r,), daemon=True) for r in refs.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 5)
    from ..chem.inputs import safe_id

    out: dict[str, WorkerRef] = {}
    for key, ref in refs.items():
        if not ref.name:   # a bare --worker URL: named by the worker's answer, which must be folder-safe
            reported = (ref.info or {}).get("name")
            if reported is not None and not _safe_name(reported):
                ref.error = (f"the worker reports the machine name {str(reported)[:60]!r}, which cannot name a "
                             "folder; give it one with --worker NAME=URL")
                ref.info, reported = None, None
            ref.name = reported or safe_id(urlsplit(ref.url).hostname or key, "worker")
        out[ref.name] = ref
    return out


# ========================================================================================== grid
@dataclass
class Share:
    machine: str
    url: str = ""                  # "" for the local share
    n: int = 0
    weight: float = 1.0
    job_id: str = ""
    run_folder: str = ""
    run_dir: str = ""              # as the machine that runs it sees it
    state: str = ""
    counts: dict = field(default_factory=dict)
    source: str = ""               # where the merge read it: "local" | "synced" | "archive" | "partial"
    commit: str = ""               # git commit the machine runs (as its worker reports it)
    error: str = ""                # why following or collecting this share stopped

    @property
    def finished(self) -> int:
        return int(self.counts.get("ok", 0)) + int(self.counts.get("failed", 0))


@dataclass
class GridRun:
    id: str
    folder: str
    label: str
    created: str
    client: str
    quantabench_version: str
    method: str
    config: dict
    weights: dict
    ids: list[str]
    shares: list[Share]
    state: str = "submitted"       # submitted | running | finished | merged | failed
    merged: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    commit: str = ""               # git commit of the client that started the grid (shares record their own)
    failed_without_result: list[str] = field(default_factory=list)   # crashed the process twice: no summary row

    @property
    def path(self) -> Path:
        return Path(self.folder) / GRID_FILE

    def save(self) -> Path:
        d = dataclasses.asdict(self)
        tmp = self.path.with_name(GRID_FILE + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    @classmethod
    def load(cls, path: str | Path) -> GridRun:
        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        d["shares"] = [Share(**{k: v for k, v in s.items() if k in {f.name for f in dataclasses.fields(Share)}})
                       for s in d.get("shares", [])]
        d["folder"] = str(path.parent)           # the folder may have synced to another machine
        return cls(**d)

    def summary(self) -> str:
        total = sum(s.finished for s in self.shares)
        parts = [f"{s.machine} {s.finished}/{s.n} {s.state or '-'}" for s in self.shares if s.n]
        return f"{total}/{len(self.ids)} done: " + " | ".join(parts)


def _target_profiles(names: Sequence[str], refs: dict[str, WorkerRef]) -> dict[str, object]:
    """{grid target: MachineProfile or None}: what a worker reports beats the registry copy (it is fresher)."""
    from . import plan, profile

    registry = {plan.norm_name(p.name): p for p in profile.load_profiles()}
    out: dict[str, object] = {}
    for n in names:
        ref = refs.get(n)
        p = None
        if ref is not None and ref.info and isinstance(ref.info.get("profile"), dict):
            with contextlib.suppress(Exception):
                p = profile.MachineProfile.from_dict(ref.info["profile"])
        if p is None:
            p = registry.get(plan.norm_name(profile.machine_name() if n == LOCAL else n))
        if p is None and n == LOCAL:
            with contextlib.suppress(Exception):
                p = profile.local_profile()
        out[n] = p
    return out


def resolve_weights(spec: str | None, names: Sequence[str], method: str, refs: dict[str, WorkerRef],
                    log: Callable[[str], None] = _noop) -> dict[str, float]:
    """``{target: weight}`` for ``grid build --weights``, with ``plan.parse_weights`` rules:

    - ``"3,1"``: one weight per machine, in ``--machines`` order;
    - ``"btm-nova=3,local=1"``: named (every machine once; "local" or this machine's name for this one);
    - ``"auto"`` (default): each machine's measured throughput for ``method`` from the profile its worker
      reports (else the registry copy), an estimate from its core count without a benchmark, 0 where the
      method is unavailable (``plan.auto_weight``). A target without any profile gets the mean of the others.
    """
    from . import plan, profile

    spec = (spec or "auto").strip()
    profiles = _target_profiles(names, refs)
    if spec.lower() == "auto":
        if not method or method == "auto":
            raise GridError("--weights auto needs an explicit --method (throughputs are per engine)")
        found: dict[str, float | None] = {}
        for n in names:
            p = profiles.get(n)
            if p is None:
                found[n] = None
                continue
            w, source = plan.auto_weight(p, method)
            found[n] = w
            if not source.startswith("benchmark"):
                log(f"{n}: weight {w:g} ({source})")
        known = [v for v in found.values() if v is not None]
        if not known or sum(known) <= 0:
            if known:
                raise GridError(f"none of {', '.join(names)} can run {method}")
            log(f"No profile for {', '.join(names)} (`machines --refresh --bench {method}`): equal weights")
            return dict.fromkeys(names, 1.0)
        missing = [n for n, v in found.items() if v is None]
        if missing:
            mean = sum(known) / len(known)
            log(f"No profile for {', '.join(missing)}: using the mean of the others ({mean:g})")
            found = {n: (mean if v is None else v) for n, v in found.items()}
        return {n: float(found[n]) for n in names}
    try:
        parsed = plan.parse_weights(spec, method, [p for p in profiles.values() if p is not None])
    except ValueError as exc:
        raise GridError(str(exc).replace("--chunk-weights", "--weights")) from None
    if not parsed.named:
        if len(parsed) != len(names):
            raise GridError(f"--weights {spec!r}: need {len(names)} weights (one per machine: {', '.join(names)})")
        out = dict(zip(names, parsed.values_list))
    else:
        key: dict[str, str] = {}
        for n in names:
            key[plan.norm_name(n)] = n
            if n == LOCAL:
                key.setdefault(plan.norm_name(profile.machine_name()), n)
        unknown = [k for k in parsed if k not in key]
        if unknown:
            raise GridError(f"--weights names {', '.join(unknown)}, which are not in --machines ({', '.join(names)})")
        out = dict.fromkeys(names, None)
        for k, v in parsed.items():
            out[key[k]] = float(v)
        missing = [n for n, v in out.items() if v is None]
        if missing:
            raise GridError(f"--weights {spec!r}: no weight for {', '.join(missing)} (give every machine one)")
    for note in getattr(parsed, "notes", []):
        log(f"note: {note}")
    if sum(out.values()) <= 0:
        raise GridError(f"--weights {spec!r}: at least one weight must be positive")
    return {n: float(out[n]) for n in names}


class _LocalTarget:
    def __init__(self, manager, job):
        self.manager, self.job = manager, job

    def status(self) -> dict:
        return self.manager.public(self.job, tail=0)


class _RemoteTarget:
    def __init__(self, client: WorkerClient, job_id: str):
        self.client, self.job_id = client, job_id

    def status(self) -> dict:
        return self.client.status(self.job_id, tail=0)


def follow(grid: GridRun, targets: dict[str, object], poll: float = 5.0, log: Callable[[str], None] = print,
           only: set[str] | None = None, should_stop: Callable[[], bool] = lambda: False) -> None:
    """Print the combined progress until every share (or those in ``only``) is done, failed or cancelled.
    An 'interrupted' share (its worker was stopped) resumes when that worker starts again, and an unreachable
    worker may come back, so both are waited for. A worker that answers with a 4xx (no such job, wrong token)
    will not change its answer: that share becomes 'lost' and following it stops."""
    last = ""
    warned: set[str] = set()
    while True:
        pending = False
        for share in grid.shares:
            target = targets.get(share.machine)
            if target is None or not share.job_id or share.state == LOST:
                continue
            try:
                st = target.status()
                share.state, share.counts = st.get("state", share.state), st.get("counts") or share.counts
                warned.discard(share.machine)
            except WorkerError as exc:
                if 400 <= exc.status < 500:
                    share.state, share.error = LOST, str(exc)
                    log(f"{share.machine}: {exc}; no longer following this share (`grid merge --partial` uses "
                        "a synced copy of its run folder if there is one)")
                elif share.machine not in warned:
                    log(f"{share.machine}: {exc} (retrying)")
                    warned.add(share.machine)
            if (only is None or share.machine in only) and share.state not in DONE_STATES:
                pending = True
        line = grid.summary()
        if line != last:
            log(f"[grid] {line}")
            last = line
        if not pending or should_stop():
            return
        time.sleep(poll)


def _summary_rows(path: Path) -> int:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except (OSError, csv.Error, UnicodeDecodeError):
        return -1


def _summary_ids(path: Path) -> set[str]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return {row.get("id", "") for row in csv.DictReader(f)}
    except (OSError, csv.Error, UnicodeDecodeError):
        return set()


def _job_file(folder: Path) -> dict | None:
    """A run folder's worker.json (``{"job": ..., "environment": ..., "tables"?: ...}``), or None."""
    try:
        data = json.loads((Path(folder) / JOB_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("job"), dict) else None


def is_job_folder(folder: Path, job_id: str) -> bool:
    """``folder`` is the run folder of worker job ``job_id`` (its worker.json says so), not merely a folder of
    the same name (a "local" share and a worker on the same machine, same label, same second)."""
    data = _job_file(folder)
    return data is not None and data["job"].get("id") == job_id


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def final_tables(folder: Path, job_id: str) -> dict | None:
    """The ``tables`` record of job ``job_id``'s worker.json when the job has ended and every table it lists
    is here byte for byte, else None. The worker writes that record last, after the tables, with their sizes
    and SHA-256: SyncThing brings files in any order, so a summary.csv that has arrived says nothing about
    descriptors.csv (or about which version of either this is)."""
    from ..chem import jobs

    folder = Path(folder)
    data = _job_file(folder)
    if data is None or data["job"].get("id") != job_id or data["job"].get("state") not in FINAL_STATES:
        return None
    tables = data.get("tables")
    if not isinstance(tables, dict) or not isinstance(tables.get("files"), dict):
        return None
    if (folder / jobs.LOCK_NAME).exists():
        return None
    for name, meta in tables["files"].items():
        if name not in TABLES:          # a synced file: it names nothing but the two tables
            return None
        path = folder / name
        try:
            if (not isinstance(meta, dict) or not path.is_file() or path.stat().st_size != int(meta["size"])
                    or _file_sha256(path) != meta.get("sha256")):
                return None
        except (OSError, KeyError, TypeError, ValueError):
            return None
    return tables


def _synced(folder: Path, job_id: str, wait: float) -> dict | None:
    """``final_tables`` of the worker's run folder synced here (SyncThing), waiting up to ``wait`` seconds
    for it (10 s at most when the folder is not here at all); None when it does not become complete."""
    deadline = time.monotonic() + (wait if folder.is_dir() else min(wait, 10.0))
    while True:
        tables = final_tables(folder, job_id)
        if tables is not None:
            return tables
        if time.monotonic() >= deadline:
            return None
        time.sleep(2.0)


@contextlib.contextmanager
def _private_copy(folder: Path):
    """A ``JobStore`` on a private copy of another machine's job table (jobs.sqlite and its -wal, which
    SyncThing brings along). Reading the synced files themselves, even read-only, creates -wal/-shm files
    there that SyncThing would send back to the machine writing the folder; ``immutable`` reads leave the
    folder alone but skip the WAL, which holds the most recent molecules of a running job."""
    import sqlite3
    import tempfile

    from ..chem import jobs

    db = Path(folder) / jobs.DB_NAME
    if not db.is_file():
        raise WorkerError(f"no {jobs.DB_NAME} in {folder}")
    with tempfile.TemporaryDirectory(prefix="qb-copy-") as tmp:
        try:
            shutil.copyfile(db, Path(tmp) / jobs.DB_NAME)
            wal = db.with_name(db.name + "-wal")
            if wal.is_file():
                shutil.copyfile(wal, Path(tmp) / wal.name)
            store = jobs.JobStore(tmp, create=False)
        except (OSError, sqlite3.Error) as exc:
            raise WorkerError(f"cannot read the job table of {folder} ({exc})") from None
        try:
            yield store
        except (sqlite3.Error, ValueError) as exc:   # a damaged table, a row that is not JSON
            raise WorkerError(f"cannot read the job table of {folder} ({exc})") from None
        finally:
            store.close()


def copy_counts(folder: Path) -> dict[str, int]:
    """Molecule counts of another machine's run folder (a synced copy), without changing it
    (``_private_copy``)."""
    with _private_copy(folder) as store:
        return store.counts()


def _snapshot(folder: Path, dest: Path) -> Path | None:
    """The tables of a run folder that has not finished (or is not complete here), from a private copy of
    its job table (``_private_copy``), written to ``dest/<run folder>``; None when nothing is finished."""
    from ..chem import export

    with _private_copy(folder) as store:
        stored = store.stored_results()
    if not stored:
        return None
    target = dest / folder.name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    export.write_batch_tables(stored, target)
    return target


def _share_paths(grid: GridRun, share: Share, out: Path) -> tuple[Path, Path]:
    """(synced run folder, download folder ``<grid>/parts/<machine>``) of a share, refusing names from
    grid.json (a synced file) or a worker's answers that would lead out of those folders."""
    if not _safe_name(share.machine):
        raise WorkerError(f"machine name {share.machine!r} cannot name a folder")
    if not isinstance(share.run_folder, str) or not _FOLDER_RE.fullmatch(share.run_folder):
        raise WorkerError(f"run folder name {share.run_folder!r} is not a single folder name")
    parts = Path(os.path.abspath(Path(grid.folder) / "parts"))   # abspath: see worker.JobManager._new_run_dir
    dest = Path(os.path.abspath(parts / share.machine))
    if dest.parent != parts:
        raise WorkerError(f"machine name {share.machine!r} leads out of {parts}")
    return out / share.run_folder, dest


def _own_local(grid: GridRun, share: Share) -> Path | None:
    """The run folder of the "local" share when this machine ran it (it started the grid, and the folder is
    there); None on any other machine, which only has a synced copy (``grid merge --out`` there)."""
    from . import profile

    if share.machine != LOCAL or not share.run_dir or grid.client != profile.machine_name():
        return None
    folder = Path(share.run_dir)
    return folder if folder.is_dir() else None


def _collect_share(grid: GridRun, share: Share, out: Path, token: str, sync_wait: float,
                   log: Callable[[str], None], allow_any_host: bool) -> tuple[Path | None, dict | None]:
    """(run folder to merge or None, its final ``tables`` record or None) for one share; see ``collect``."""
    own = _own_local(grid, share)
    if own is not None:
        share.source = LOCAL         # this machine ran it: its own folder, whatever its state
        return own, (_job_file(own) or {}).get("tables")
    here, dest = _share_paths(grid, share, out)
    if share.state in FINAL_STATES:
        tables = _synced(here, share.job_id, sync_wait)
        if tables is not None:
            share.source = "synced"
            if "summary.csv" not in tables["files"]:
                log(f"{share.machine}: {share.run_folder} has no finished molecule")
                return None, tables
            return here, tables
        if share.url:
            log(f"{share.machine}: {share.run_folder} is not (fully) synced here; downloading it")
            try:
                folder = WorkerClient(share.url, token, allow_any_host=allow_any_host).download_archive(
                    share.job_id, dest, expected_top=share.run_folder)
                share.source = "archive"
                return folder, (_job_file(folder) or {}).get("tables")
            except WorkerError as exc:
                log(f"{share.machine}: downloading failed ({exc}); trying the synced copy")
    # not finished (grid merge --partial), lost, or no download: what the synced copy has finished so far
    if not is_job_folder(here, share.job_id):
        raise WorkerError(f"no copy of {share.run_folder} here" + ("" if share.state in FINAL_STATES else
                                                                  f" (the share is {share.state or 'unknown'})"))
    snap = _snapshot(here, dest)
    share.source = "partial"
    rows = _summary_rows(snap / "summary.csv") if snap is not None else 0
    log(f"{share.machine}: {share.state or 'unknown'}; merging the {max(rows, 0)} molecule(s) finished so far "
        f"in the synced copy of {share.run_folder}")
    return snap, None


def collect(grid: GridRun, token: str, sync_wait: float = 60.0, log: Callable[[str], None] = print,
            allow_any_host: bool = False, local_output: Path | None = None) -> list[Path]:
    """The run folder of every share, readable here: the local share in place; a finished remote one from
    the synced output folder once its worker.json vouches for its tables (``final_tables``), else downloaded
    into ``<grid folder>/parts/<machine>/``; a share that has not finished, or whose worker is gone or lost
    the job, from what the synced copy has finished so far (``grid merge --partial``). A share that cannot be
    collected is logged and left out (its molecules end up in ``grid.missing``); it does not stop the
    others. Molecules that crashed their worker process twice have no summary row; they are recorded in
    ``grid.failed_without_result``."""
    out = Path(local_output) if local_output else Path(grid.folder).parent
    dirs: list[Path] = []
    crashed: set[str] = set()
    for share in grid.shares:
        if not share.job_id or not share.n:
            continue
        try:
            folder, tables = _collect_share(grid, share, out, token, sync_wait, log, allow_any_host)
        except WorkerError as exc:
            share.source, share.error = "", str(exc)
            log(f"{share.machine}: {exc}; its molecules are left out of the merge")
            continue
        for item in (tables or {}).get("unrecorded_failures") or []:
            if isinstance(item, list) and item and isinstance(item[0], str):
                crashed.add(item[0])
        if folder is not None:
            dirs.append(folder)
    wanted = set(grid.ids)
    grid.failed_without_result = sorted(crashed & wanted)
    return dirs


def merge(grid: GridRun, token: str, sync_wait: float = 60.0, log: Callable[[str], None] = print,
          allow_any_host: bool = False, out_dir: Path | None = None) -> list[Path]:
    """Collect every share and write the merged tables into the grid folder (or ``out_dir``); records the
    molecules that are missing (not finished, or their share could not be collected) in ``grid.missing``."""
    from ..chem import jobs

    dirs = collect(grid, token, sync_wait, log, allow_any_host)
    target = Path(out_dir) if out_dir else Path(grid.folder)
    written = jobs.merge_tables(dirs, target, log=log) if dirs else []
    got = _summary_ids(target / "summary.csv") if any(p.name == "summary.csv" for p in written) else set()
    crashed = set(grid.failed_without_result)
    if crashed:
        log(f"{len(crashed)} molecule(s) crashed their worker process twice and have no result row: "
            + ", ".join(sorted(crashed)[:10]) + (" ..." if len(crashed) > 10 else ""))
    grid.merged = [str(p) for p in written]
    grid.missing = [i for i in grid.ids if i not in got and i not in crashed]
    grid.state = "merged"
    return written


def grid_build(items: Sequence, cfg, machines: Sequence[str] = (), weights: str | None = "auto", *,
               label: str | None = None, wait: bool = False, extra_workers: Sequence[str] = (),
               allow_version_mismatch: bool = False, allow_auto_method: bool = False,
               keep_resources: Sequence[str] = (), poll: float = 5.0, sync_wait: float = 60.0,
               output_dir: str | Path | None = None, allow_any_host: bool = False, timeout: float = 30.0,
               log: Callable[[str], None] = print, should_stop: Callable[[], bool] = lambda: False) -> GridRun:
    """Split ``items`` (``MoleculeInput``) across ``machines`` (worker names and/or "local"), submit, and with
    ``wait`` follow and merge. The local share always runs here (this call returns when it is done)."""
    from ..chem.inputs import dedupe_ids
    from ..chem.pipeline import new_run_dir
    from . import plan, profile
    from .worker import JobManager, RequestError, check_label, check_mol_id, validate_job

    s = settings_mod.load()
    token = s.dist.worker_token
    out = Path(output_dir or getattr(cfg, "output_dir", "") or s.output_path()).expanduser()
    warnings: list[str] = []
    items = dedupe_ids(list(items), warnings)
    for w in warnings:
        log(w)
    if not items:
        raise GridError("Nothing to build: give SMILES, names or --file")
    label = label or (items[0].id if len(items) == 1 else f"batch_{len(items)}")
    try:
        check_label(label)
    except RequestError as exc:
        raise GridError(f"--label: {exc}") from None
    bad = []
    for x in items:
        try:
            check_mol_id(x.id)
        except RequestError as exc:
            bad.append(str(exc))
    if bad:  # a worker would refuse the whole share; say so before anything is submitted
        raise GridError(f"{len(bad)} molecule id(s) cannot name a folder on every machine: " + "; ".join(bad[:5]))

    names = list(dict.fromkeys(m.strip() for m in machines if m.strip()))
    refs = discover(extra_workers, token, timeout=min(timeout, 5.0), allow_any_host=allow_any_host) \
        if any(n != LOCAL for n in names) or not names else {}
    if not names:
        names = [n for n, r in refs.items() if r.reachable]
        if not names:
            raise GridError("No reachable worker found: start `quantabench-cli worker` on your machines, or name "
                            "them with --machines (\"local\" = this machine) and --worker NAME=URL")
    if cfg.method == "auto" and len(names) > 1:
        msg = ("method 'auto' can pick a different engine on each machine, which would mix levels of theory in one "
               "table; choose --method explicitly")
        if not allow_auto_method:
            raise GridError(msg + " (or pass --allow-auto-method)")
        log("Warning: " + msg)
    commit = profile.git_commit()
    commits: dict[str, str] = {LOCAL: commit}
    for n in names:
        if n == LOCAL:
            continue
        if not _safe_name(n):
            raise GridError(f"Machine name {n!r} cannot name a folder (letters, digits, . _ + -; start with a "
                            "letter or digit)")
        ref = refs.get(n)
        if ref is None:
            known = ", ".join(sorted(refs)) or "none"
            raise GridError(f"Unknown machine {n!r} (known workers: {known}; \"local\" = this machine). Add it with "
                            f"--worker {n}=http://ADDRESS:{DEFAULT_PORT} or settings.dist.workers")
        if not ref.reachable:
            raise GridError(f"{n} ({ref.url}) is not reachable: {ref.error}")
        version = ref.info.get("quantabench_version", "")
        if version != __version__ and not allow_version_mismatch:
            raise GridError(f"{n} runs QuantaBench {version}, this machine {__version__}: update both (git pull) or "
                            "pass --allow-version-mismatch")
        commits[n] = str(ref.info.get("commit") or "")
        if not same_commit(commit, commits[n]):
            log(f"Warning: {n} runs commit {commits[n]}, this machine {commit}: the version number is the same, "
                "but the code may differ (git pull on both). Both commits are recorded in grid.json.")
        engines = (ref.info.get("profile") or {}).get("engines") or {}
        if cfg.method != "auto" and engines.get(cfg.method) is False:
            raise GridError(f"{n} cannot run {cfg.method} (engine not available there)")
        if n == profile.machine_name() and LOCAL in names:
            log(f"Note: {n} is this machine and \"local\" is listed too; both shares run here")
    w = resolve_weights(weights, names, cfg.method, refs, log)
    ids = [x.id for x in items]
    split = plan.split(ids, w)
    by_id = {x.id: x for x in items}
    log("Split: " + ", ".join(f"{n} {len(split[n])} (weight {w[n]:g})" for n in names))

    folder = new_run_dir(out, f"grid_{label}")
    folder.mkdir(parents=True, exist_ok=True)
    config = config_for_workers(cfg, keep_resources)
    grid = GridRun(id=folder.name, folder=str(folder), label=label, created=_now(), client=profile.machine_name(),
                   quantabench_version=__version__, method=cfg.method, config=config, weights=w, ids=ids,
                   shares=[Share(machine=n, url=refs[n].url if n != LOCAL else "", n=len(split[n]), weight=w[n],
                                 commit=commits.get(n, "")) for n in names],
                   commit=commit)
    grid.save()

    targets: dict[str, object] = {}
    submitted: list[tuple[WorkerClient, str]] = []
    for k, share in enumerate(grid.shares):
        if share.machine == LOCAL or not share.n:
            continue
        client = WorkerClient(share.url, token, timeout, allow_any_host)
        body = job_body([by_id[i] for i in split[share.machine]], config, label,
                        chunk={"index": k, "grid": grid.id[:120]}, client=profile.machine_name(),
                        allow_version_mismatch=allow_version_mismatch)
        try:
            reply = client.submit_body(body)
        except WorkerError as exc:
            for c, job_id in submitted:
                with contextlib.suppress(WorkerError):
                    c.cancel(job_id)
            grid.state = "failed"
            grid.save()
            raise GridError(f"Submitting to {share.machine} failed: {exc}"
                            + ("; cancelled the shares already submitted" if submitted else "")) from None
        try:  # what the worker answers names folders here: check it like any other input
            job = reply["job"]
            share.job_id, share.run_folder, share.run_dir = _job_id(reply["id"]), job["run_folder"], job["run_dir"]
            share.state = job["state"]
            if not isinstance(share.run_folder, str) or not _FOLDER_RE.fullmatch(share.run_folder):
                raise WorkerError(f"bad run folder name {share.run_folder!r}")
        except (KeyError, TypeError, WorkerError) as exc:
            with contextlib.suppress(WorkerError, KeyError, TypeError):
                client.cancel(reply["id"])
            for c, job_id in submitted:
                with contextlib.suppress(WorkerError):
                    c.cancel(job_id)
            grid.state = "failed"
            grid.save()
            raise GridError(f"{share.machine} answered the submission unexpectedly ({exc})") from None
        submitted.append((client, share.job_id))
        targets[share.machine] = _RemoteTarget(client, share.job_id)
        log(f"{share.machine}: {share.n} molecule(s) -> job {share.job_id} ({share.run_folder})")

    manager = None
    local = next((sh for sh in grid.shares if sh.machine == LOCAL and sh.n), None)
    if local is not None:
        body = job_body([by_id[i] for i in split[LOCAL]], config, label, client=profile.machine_name())
        spec = validate_job(json.loads(json.dumps(body)), __version__)  # the checks a worker would make
        here = profile.machine_name()
        same = any(sh.machine == here for sh in grid.shares)     # a worker on this machine is listed too
        # lanes=None: this machine's own spec from the settings (Settings.lanes_for), one lane if it does not fit
        manager = JobManager(out, f"{here}-local" if same else here, state_dir=None, lanes=None)
        manager.start()
        job = manager.submit(spec)
        local.job_id, local.run_dir, local.run_folder = job.id, job.run_dir, Path(job.run_dir).name
        local.state, local.source = job.state, LOCAL
        targets[LOCAL] = _LocalTarget(manager, job)
        log(f"local: {local.n} molecule(s) run here -> {local.run_folder}")
    grid.state = "running"
    grid.save()
    log(f"Grid {grid.id}: {GRID_FILE} in {folder}")

    try:
        if wait or manager is not None:
            follow(grid, targets, poll, log, only=None if wait else {LOCAL}, should_stop=should_stop)
    except KeyboardInterrupt:
        log("Stopped following. Remote shares keep running; the local share stops after its current molecule "
            f"(resume it with `quantabench-cli build --resume {local.run_dir}`)" if local else
            "Stopped following; the shares keep running on their machines.")
        raise
    finally:
        if manager is not None:
            manager.stop()
            if local is not None:
                local.state, local.counts = manager.public(manager.get(local.job_id), tail=0)["state"], \
                    manager.public(manager.get(local.job_id), tail=0)["counts"]
        grid.save()
    if wait and not should_stop():
        grid.state = "finished"
        merge(grid, token, sync_wait, log, allow_any_host)
        grid.save()
        log(f"Merged {len(grid.ids) - len(grid.missing)}/{len(grid.ids)} molecule(s) into {folder}"
            + (f"; missing: {', '.join(grid.missing[:10])}{' ...' if len(grid.missing) > 10 else ''}"
               if grid.missing else ""))
    elif not wait:
        log(f"Follow it with `quantabench-cli grid status {grid.id}`; merge with `quantabench-cli grid merge "
            f"{grid.id}` when it is done.")
    return grid


def find_grid(ident: str, output_dir: str | Path | None = None) -> GridRun:
    """A grid by id (its folder name, or a unique part of it) in the output folder, or by path."""
    p = Path(ident).expanduser()
    for cand in (p, p / GRID_FILE):
        if cand.is_file() and cand.name == GRID_FILE:
            return GridRun.load(cand)
    out = Path(output_dir).expanduser() if output_dir else settings_mod.load().output_path()
    if (out / ident / GRID_FILE).is_file():
        return GridRun.load(out / ident / GRID_FILE)
    matches = sorted(out.glob(f"*grid*/{GRID_FILE}"))
    matches = [m for m in matches if ident in m.parent.name]
    if len(matches) == 1:
        return GridRun.load(matches[0])
    if matches:
        raise GridError(f"{ident!r} matches {len(matches)} grids: " + ", ".join(m.parent.name for m in matches[:5]))
    raise GridError(f"No grid {ident!r} in {out}")


def refresh(grid: GridRun, token: str, allow_any_host: bool = False) -> dict[str, str]:
    """Update each share's state and counts from its worker (or, if unreachable, from a synced copy of its
    run folder, read without changing it: ``copy_counts``); returns per-machine notes. A worker that answers
    with a 4xx for the job marks the share 'lost'."""
    import sqlite3

    from ..chem import jobs

    notes: dict[str, str] = {}
    out = Path(grid.folder).parent
    for share in grid.shares:
        if not share.job_id:
            continue
        if share.machine != LOCAL:
            try:
                st = WorkerClient(share.url, token, 10.0, allow_any_host).status(share.job_id, tail=0)
                share.state, share.counts = st["state"], st.get("counts") or share.counts
                continue
            except WorkerError as exc:
                notes[share.machine] = str(exc)
                if 400 <= exc.status < 500:
                    share.state, share.error = LOST, str(exc)
        own = _own_local(grid, share)
        if own is not None:
            folder = own
        else:
            try:
                folder = _share_paths(grid, share, out)[0]
            except WorkerError as exc:
                notes[share.machine] = notes.get(share.machine, "") + f" ({exc})"
                continue
            if not is_job_folder(folder, share.job_id):
                notes[share.machine] = notes.get(share.machine, "") + " (no synced copy here)"
                continue
        try:
            if own is not None:   # this machine's own run folder
                rs = jobs.open_run(folder)
                share.counts = dict(rs.counts)
                share.state = "running" if rs.active else "done" if not rs.remaining else "interrupted"
                notes.setdefault(share.machine, "")
            else:     # another machine's folder, synced here: never write into it
                share.counts = copy_counts(folder)
                notes[share.machine] = notes.get(share.machine, "") + " (counts from the synced run folder)"
        except (OSError, ValueError, sqlite3.Error, WorkerError) as exc:
            notes.setdefault(share.machine, f"no run folder here ({exc.__class__.__name__})")
    return notes


# =========================================================================================== CLI
def _build_args(b: argparse.ArgumentParser) -> None:
    """The input and science options of ``build`` (the same argument groups as ``cli.build_parser``)."""
    from .. import cli as cli_mod

    cli_mod.add_input_args(b)
    cli_mod.add_engine_args(
        b, method_help="refinement engine; required in practice: 'auto' may differ between machines",
        threads_help="CPU threads on every machine (default: each machine's settings)",
        gpu_help="GPU policy on every machine (default: each machine's settings)",
        out_help="this machine's output folder for the grid record, the merged tables and the local share "
                 "(default from settings)")
    cli_mod.add_build_science_args(b)


def _worker_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--worker", action="append", default=[], metavar="[NAME=]URL",
                   help="a worker not in the machine registry or settings (repeatable)")
    p.add_argument("--allow-any-host", action="store_true",
                   help="send the token to hosts that are not Tailscale or loopback addresses (unencrypted)")


def _cli_machines(args) -> int:
    from . import profile

    try:
        refs = discover(args.worker, timeout=args.timeout, allow_any_host=args.allow_any_host)
    except GridError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([dataclasses.asdict(r) for r in refs.values()], indent=2, default=str))
        return 0 if any(r.reachable for r in refs.values()) else 1
    if not refs:
        print("No workers known. Start `quantabench-cli worker` on a machine (it publishes its address in the "
              "synced settings), or pass --worker NAME=http://ADDRESS:PORT.")
        return 1
    for ref in refs.values():
        if not ref.reachable:
            print(f"{ref.name:24s} {ref.url:30s} unreachable: {ref.error}")
            continue
        info = ref.info
        prof = profile.MachineProfile.from_dict(info.get("profile") or {"name": ref.name})
        q = info.get("queue") or {}
        hw = ", ".join(x for x in (prof.os, prof.cpu, ", ".join(g.get("name", "") for g in prof.gpus)) if x)
        mark = "" if info.get("quantabench_version") == __version__ else f" (this machine: {__version__})"
        print(f"{ref.name:24s} {ref.url:30s} QuantaBench {info.get('quantabench_version')}{mark}"
              f"{' @' + info['commit'] if info.get('commit') else ''}; "
              f"{len(q.get('running', []))} running, {len(q.get('queued', []))} queued"
              f"{', paused' if q.get('paused') else ''}{', ' + q['hold'] if q.get('hold') else ''}" + (f"; {hw}" if hw else ""))
        if prof.throughput:
            print(" " * 25 + "throughput: " + ", ".join(f"{k} {v:g}/h" for k, v in sorted(prof.throughput.items())))
    return 0 if any(r.reachable for r in refs.values()) else 1


def _cli_build(args) -> int:
    from .. import cli as cli_mod

    try:
        items = cli_mod.read_inputs(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    cfg = cli_mod.config_from_args(args)
    keep = tuple(name for name, value in (("threads", args.threads), ("gpu", args.gpu)) if value is not None)
    machines = [m for m in (args.machines or "").split(",") if m.strip()]
    try:
        grid = grid_build(items, cfg, machines, args.weights, label=args.label, wait=args.wait,
                          extra_workers=args.worker, allow_version_mismatch=args.allow_version_mismatch,
                          allow_auto_method=args.allow_auto_method, keep_resources=keep, poll=args.poll,
                          sync_wait=args.sync_wait, output_dir=args.out, allow_any_host=args.allow_any_host,
                          timeout=args.timeout)
    except (GridError, WorkerError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    print(f"grid {grid.id}: {grid.summary()}")
    return 0 if not args.wait or not grid.missing else 1


def _cli_status(args) -> int:
    token = settings_mod.load().dist.worker_token
    if not args.grid_id:
        out = Path(args.out).expanduser() if args.out else settings_mod.load().output_path()
        found = sorted(out.glob(f"*grid*/{GRID_FILE}"), reverse=True)[:20]
        if not found:
            print(f"No grid runs in {out}")
            return 1
        for path in found:
            with contextlib.suppress(OSError, ValueError, TypeError):
                g = GridRun.load(path)
                print(f"{g.id}  {g.state:9s} {len(g.ids)} molecules on {', '.join(s.machine for s in g.shares if s.n)}")
        return 0
    try:
        grid = find_grid(args.grid_id, args.out)
    except GridError as exc:
        print(exc, file=sys.stderr)
        return 2
    notes = refresh(grid, token, args.allow_any_host)
    print(f"{grid.id} ({grid.method}, {len(grid.ids)} molecules, {grid.state})")
    for share in grid.shares:
        if not share.n:
            continue
        c = share.counts
        print(f"  {share.machine:24s} job {share.job_id or '-':12s} {share.state or '-':11s} "
              f"{share.finished}/{share.n} ({c.get('ok', 0)} ok, {c.get('failed', 0)} failed)  {share.run_folder}"
              + (f"  [{notes[share.machine].strip()}]" if notes.get(share.machine) else ""))
    print(f"  total {sum(s.finished for s in grid.shares)}/{len(grid.ids)}")
    return 0


def _cli_cancel(args) -> int:
    token = settings_mod.load().dist.worker_token
    try:
        grid = find_grid(args.grid_id, args.out)
    except GridError as exc:
        print(exc, file=sys.stderr)
        return 2
    rc = 0
    for share in grid.shares:
        if not share.job_id:
            continue
        if share.machine == LOCAL:
            print("local: stop it with Ctrl+C in the window that runs `grid build`")
            continue
        try:
            st = WorkerClient(share.url, token, 10.0, args.allow_any_host).cancel(share.job_id)
            print(f"{share.machine}: job {share.job_id} {st['state']}"
                  + (" (stops after the current molecule)" if st["state"] == "running" else ""))
        except WorkerError as exc:
            print(f"{share.machine}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def _cli_merge(args) -> int:
    from . import profile

    token = settings_mod.load().dist.worker_token
    try:
        grid = find_grid(args.grid_id, None)
    except GridError as exc:
        print(exc, file=sys.stderr)
        return 2
    other = grid.client != profile.machine_name()
    if other and not args.out:
        print(f"{grid.id} was started on {grid.client}; merge it there, or give --out (a folder of this machine) "
              "so the grid folder keeps a single writer", file=sys.stderr)
        return 2
    refresh(grid, token, args.allow_any_host)
    unfinished = [f"{s.machine} ({s.state or 'unknown'})" for s in grid.shares
                  if s.job_id and s.state not in FINAL_STATES]
    if unfinished and not args.partial:
        print(f"Not finished on {', '.join(unfinished)}; wait, or pass --partial to merge what is finished "
              "(read from the synced run folders)", file=sys.stderr)
        return 1
    try:
        written = merge(grid, token, args.sync_wait, print, args.allow_any_host,
                        Path(args.out).expanduser() if args.out else None)
    except WorkerError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not other:
        grid.save()
    for p in written:
        print(f"wrote {p}")
    if grid.missing:
        print(f"{len(grid.missing)} molecule(s) missing: {', '.join(grid.missing[:20])}", file=sys.stderr)
    return 0 if written and not grid.missing else 1


def register_cli(subparsers) -> None:
    g = subparsers.add_parser("grid", help="split a molecule list across your machines (workers), merge results",
                              description="Run one batch on several machines: `grid build` splits the list by "
                                          "weight, submits each share to a `quantabench-cli worker`, runs the "
                                          "'local' share here and merges the tables. See docs/WORKERS.md.")
    gs = g.add_subparsers(dest="grid_command", required=True)

    m = gs.add_parser("machines", help="known workers and whether they answer")
    _worker_opts(m)
    m.add_argument("--timeout", type=float, default=3.0, help="seconds per worker")
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=_cli_machines)

    b = gs.add_parser("build", help="split a list across machines, run, follow and merge",
                      description="Same inputs and science options as `build`. Every machine gets the same "
                                  "BuildConfig (set --method explicitly); threads and GPU follow each machine's "
                                  "own settings unless --threads/--gpu are given.")
    _build_args(b)
    b.add_argument("--machines", help="comma list of worker names and/or 'local' (default: every reachable worker)")
    b.add_argument("--weights", default="auto", help="auto (measured throughput, else equal) or one number per "
                                                     "machine, e.g. 3,1")
    b.add_argument("--wait", action="store_true", help="follow until every share is done, then merge the tables")
    b.add_argument("--label", help="name for the run folders (default batch_<n>)")
    _worker_opts(b)
    b.add_argument("--allow-version-mismatch", action="store_true")
    b.add_argument("--allow-auto-method", action="store_true",
                   help="allow method 'auto' across machines (may mix engines)")
    b.add_argument("--poll", type=float, default=5.0, help="seconds between progress checks")
    b.add_argument("--sync-wait", type=float, default=60.0,
                   help="seconds to wait for SyncThing to bring a worker's run folder before downloading it")
    b.add_argument("--timeout", type=float, default=30.0, help="seconds per request to a worker")
    b.set_defaults(func=_cli_build)

    st = gs.add_parser("status", help="progress of a grid run (no id: list recent grid runs)")
    st.add_argument("grid_id", nargs="?")
    st.add_argument("--out", help="output folder to look in (default from settings)")
    st.add_argument("--allow-any-host", action="store_true")
    st.set_defaults(func=_cli_status)

    c = gs.add_parser("cancel", help="cancel every remote share of a grid run")
    c.add_argument("grid_id")
    c.add_argument("--out", help="output folder to look in (default from settings)")
    c.add_argument("--allow-any-host", action="store_true")
    c.set_defaults(func=_cli_cancel)

    mg = gs.add_parser("merge", help="merge the tables of a finished grid run (downloads unsynced shares)")
    mg.add_argument("grid_id")
    mg.add_argument("--out", help="write the merged tables here instead of the grid folder")
    mg.add_argument("--partial", action="store_true", help="merge even if some shares are not finished")
    mg.add_argument("--sync-wait", type=float, default=60.0)
    mg.add_argument("--allow-any-host", action="store_true")
    mg.set_defaults(func=_cli_merge)
