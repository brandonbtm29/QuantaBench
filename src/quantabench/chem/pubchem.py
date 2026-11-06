"""PubChem client: PUG REST (identifiers, computed properties, synonyms, registry numbers) and PUG View
(experimental property annotations), kept within PubChem's usage policy, with an on-disk response cache.

CONTRACT:

    client = PubChemClient(cache=None, offline=False, rate=4.0)   # cache: a folder or .sqlite file; False = none
    client.cids_by_name(name_or_cas) -> [cid]                     # [] when PubChem has no such name
    client.cids_by_inchikey(key), client.cids_by_smiles(smiles) -> [cid]
    client.properties(cids) -> {cid: {MolecularFormula, MolecularWeight, SMILES, ConnectivitySMILES, InChI, ...}}
    client.synonyms(cids) -> {cid: [synonym]}                     # batched, PubChem's order (most relevant first)
    client.registry_numbers(cids) -> {cid: [RN]}                  # xrefs RN: CAS and EC numbers linked to the CID
    client.view(cid, heading="Experimental Properties") -> PUG View JSON | None
    client.experimental(cid) -> [ExperimentalRecord]              # one request per CID: the whole subtree
    lookup(name) -> {"cid", "smiles", "title"}                    # chem.inputs.lookup_name
    is_cas(text), normalize_cas(text), cas_numbers(strings)       # CAS RN format + check digit
    parse_view(json, cid=None, headings=None) -> [ExperimentalRecord]
    parse_values(text, heading) -> [ParsedValue]                  # "15.2 dynes/cm", "72.8 mN/m at 20 °C", ...
    classify_conditions(records) -> records                       # pure_liquid | aqueous_solution | unknown

Usage policy (https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access)
-----------------------------------------------------------------------
At most 5 requests per second and 400 per minute. Every response carries ``X-Throttling-Control``, e.g.
``Request Count status: Green (0%), Request Time status: Green (0%), Service status: Red (87%)``: the first
two are this client's own load, the last is PubChem's overall load. The limiter spaces requests by
``1/rate`` (default 4/s, capped at 5/s) times a factor from the worst status seen last: Yellow x2 and Red x4
for this client's own load, Yellow x1.5 and Red x2 for the service status alone, and a 60 s pause on Black
(requests are being refused). 503 (server busy) and 504 (timeout) are retried with exponential backoff
(2, 4, 8, 16, 32 s). The User-Agent names QuantaBench and its repository, never a person or an email address.

Cache
-----
Every 200 and 404 (and 400: a malformed identifier stays malformed) is stored, zlib-compressed, in one SQLite
file keyed by method, URL and POST body, committed per response: an interrupted pull of 14,000 molecules resumes
where it stopped, and a rerun makes no request it made before. ``offline=True`` only reads the cache and raises
:class:`NotCached` for anything else. Errors (5xx, network) are never cached.

Experimental values
-------------------
PUG View's ``Experimental Properties`` subtree is fetched once per CID and every Information entry under the
headings in :data:`HEADINGS` becomes :class:`ExperimentalRecord` objects: the raw string, the parsed number
(ranges and ± kept), its unit, the value in the heading's standard unit (surface tension in mN/m; dyn/cm =
mN/m = mJ/m², N/m x 1000), the temperature if stated, the source (HSDB, ICSC, ...), its reference text and DOI.
``Other Experimental Properties`` strings that name one of these properties ("Surface tension: 27.6 dyn/cm at
20 °C") are filed under that heading. :func:`classify_conditions` then sorts surface tension, density,
viscosity and refractive index records into ``pure_liquid``, ``aqueous_solution`` (concentration when stated)
or ``unknown``, from the text, the reference (CRC Handbook, Jasper, Yaws, DIPPR: pure-liquid tables;
surfactant literature: aqueous), and the compound's melting point (a compound that is solid at the value's
temperature cannot have given a pure-liquid surface tension there). Records that cannot be told apart are
``needs_review``. HSDB's 15.2 dyn/cm for PFOA (mp 54 °C; Moody & Field 2000, a review of fluorinated
surfactants in fire-fighting foams) is such a record: very likely an aqueous value at the CMC.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import sqlite3
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .. import __version__

_log = logging.getLogger(__name__)

REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
REPO_URL = "https://github.com/brandonbtm29/QuantaBench"
USER_AGENT = f"QuantaBench/{__version__} (+{REPO_URL})"

MAX_RATE = 5.0              # requests per second (policy)
MAX_PER_MINUTE = 400        # requests per minute (policy)
DEFAULT_RATE = 4.0
#: Spacing factors per throttling color: this client's own load (request count, request time) and the service.
USER_BACKOFF = {"green": 1.0, "yellow": 2.0, "red": 4.0}
SERVICE_BACKOFF = {"green": 1.0, "yellow": 1.5, "red": 2.0}
BLACK_PAUSE_S = 60.0
RETRY_STATUSES = (429, 500, 502, 503, 504)
PROPERTY_BATCH = 100        # CIDs per property request
LIST_BATCH = 50             # CIDs per synonyms / xrefs request (long lists)

#: Computed properties requested in batches. PubChem renamed CanonicalSMILES -> ConnectivitySMILES and
#: IsomericSMILES -> SMILES in 2025; asking for the old names now returns only the new ones, so only the new
#: names are requested and :func:`_normalize_props` also reads the old ones (cached or older responses).
PROPERTIES = ("Title", "MolecularFormula", "MolecularWeight", "ExactMass", "SMILES", "ConnectivitySMILES", "InChI",
              "InChIKey", "XLogP", "TPSA", "Complexity", "HBondDonorCount", "HBondAcceptorCount",
              "RotatableBondCount", "Charge", "HeavyAtomCount")
OLD_PROPERTY_NAMES = {"SMILES": "IsomericSMILES", "ConnectivitySMILES": "CanonicalSMILES"}

#: PUG View headings kept as records: heading -> (dimension, standard unit). Liquid properties are classified.
HEADINGS: dict[str, tuple[str, str]] = {
    "Surface Tension": ("surface_tension", "mN/m"),
    "Density": ("density", "g/cm3"),
    "Boiling Point": ("temperature", "°C"),
    "Melting Point": ("temperature", "°C"),
    "Vapor Pressure": ("pressure", "Pa"),
    "Viscosity": ("viscosity", "mPa.s"),
    "Solubility": ("concentration", "mg/L"),
    "LogP": ("none", ""),
    "Dissociation Constants": ("none", ""),
    "Refractive Index": ("none", ""),
    "Heat of Vaporization": ("molar_energy", "kJ/mol"),
    "Critical Temperature": ("temperature", "°C"),
    "Critical Pressure": ("pressure", "Pa"),
    "Vapor Density": ("none", ""),
    "Physical Description": ("text", ""),
    "Color/Form": ("text", ""),
}
LIQUID_HEADINGS = ("Surface Tension", "Density", "Viscosity", "Refractive Index")
#: "Other Experimental Properties" strings that name a property: filed under that heading.
_OTHER_LABELS = (
    (re.compile(r"surface\s+tension", re.I), "Surface Tension"),
    (re.compile(r"critical\s+temp", re.I), "Critical Temperature"),
    (re.compile(r"critical\s+press", re.I), "Critical Pressure"),
    (re.compile(r"(heat|enthalpy)\s+of\s+vapo", re.I), "Heat of Vaporization"),
    (re.compile(r"\bviscosity\b", re.I), "Viscosity"),
    (re.compile(r"(refractive\s+index|index\s+of\s+refraction)", re.I), "Refractive Index"),
    (re.compile(r"\b(density|specific\s+gravity)\b", re.I), "Density"),
)
CONDITION_CLASSES = ("pure_liquid", "aqueous_solution", "unknown", "solid")


# ============================================================================================ errors
class PubChemError(Exception):
    """Base class: anything PubChem could not answer."""


class PubChemUnavailable(PubChemError):
    """PubChem could not be reached (network, 5xx after retries): try again later."""


class NotCached(PubChemUnavailable):
    """Offline mode and the response is not in the cache."""


class PubChemFault(PubChemError):
    """PubChem answered with a fault other than 'not found' (e.g. 400 bad request)."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"HTTP {status} {code}: {message}".strip())
        self.status, self.code, self.message = status, code, message


# ============================================================================================ CAS numbers
_CAS_RE = re.compile(r"(?<!\d)0*(\d{2,7})-(\d{2})-(\d)(?![\d-])")


def cas_check_digit(body: str) -> int:
    """Check digit of the CAS digits before it (``33567`` for 335-67-?): sum of digit x position from the right."""
    digits = [int(c) for c in reversed(body)]
    return sum(i * d for i, d in enumerate(digits, 1)) % 10


def is_cas(text) -> bool:
    """True for a well-formed CAS RN (2-7, 2 and 1 digits) whose check digit is right. EC numbers (206-397-9) fail
    the format, misprints the check digit."""
    m = re.fullmatch(r"\s*0*(\d{2,7})-(\d{2})-(\d)\s*", str(text or ""))
    return bool(m) and cas_check_digit(m.group(1) + m.group(2)) == int(m.group(3))


def normalize_cas(text) -> str:
    """The first valid CAS RN in ``text`` ('CAS-335-67-1', ' 0335-67-1 ' -> '335-67-1'), else ''."""
    for m in _CAS_RE.finditer(str(text or "")):
        cas = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if is_cas(cas):
            return cas
    return ""


def cas_numbers(strings: Iterable[str]) -> list[str]:
    """Valid CAS RNs among ``strings`` (synonyms or registry numbers), in order, without repeats. Only strings that
    are a CAS RN as a whole count ('CAS-335-67-1' too), not numbers inside names."""
    out: list[str] = []
    for s in strings:
        s = str(s).strip()
        if s.upper().startswith("CAS-"):
            s = s[4:]
        if is_cas(s):
            cas = normalize_cas(s)
            if cas not in out:
                out.append(cas)
    return out


# ============================================================================================ transport
@dataclass
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def header(self, name: str) -> str:
        low = name.lower()
        return next((v for k, v in self.headers.items() if k.lower() == low), "")


Transport = Callable[[str, str, "bytes | None", dict, float], HttpResponse]


def urllib_transport(method: str, url: str, data: bytes | None, headers: dict, timeout: float) -> HttpResponse:
    """The real HTTP call (stdlib only). HTTP errors come back as responses; network errors raise OSError."""
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return HttpResponse(resp.status, resp.read(), dict(resp.headers.items()))
    except urllib.error.HTTPError as exc:
        body = b""
        with contextlib.suppress(Exception):
            body = exc.read()
        return HttpResponse(exc.code, body, dict(exc.headers.items()) if exc.headers else {})


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


# ============================================================================================ throttling
def parse_throttling(header: str) -> dict[str, tuple[str, int]]:
    """``X-Throttling-Control`` -> {"request count": ("green", 0), "request time": (...), "service": (...)}."""
    out: dict[str, tuple[str, int]] = {}
    for part in (header or "").split(","):
        m = re.match(r"\s*(.+?)\s+status:\s*([A-Za-z]+)\s*\((\d+)%\)", part)
        if m:
            out[m.group(1).strip().lower()] = (m.group(2).lower(), int(m.group(3)))
    return out


def throttle_factor(status: dict[str, tuple[str, int]]) -> tuple[float, float]:
    """(spacing factor, pause in s) for a parsed throttling status: the worst component decides."""
    factor, pause = 1.0, 0.0
    for comp, (color, _pct) in status.items():
        if color == "black":
            pause = max(pause, BLACK_PAUSE_S)
            color = "red"
        table = SERVICE_BACKOFF if comp.startswith("service") else USER_BACKOFF
        factor = max(factor, table.get(color, 1.0))
    return factor, pause


class RateLimiter:
    """Spaces requests by ``factor / rate`` and keeps at most ``per_minute`` in any 60 s window."""

    def __init__(self, rate: float = DEFAULT_RATE, per_minute: int = MAX_PER_MINUTE,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] | None = None):
        self.rate = min(float(rate), MAX_RATE) if rate and rate > 0 else DEFAULT_RATE
        self.per_minute = min(int(per_minute), MAX_PER_MINUTE)
        self.clock, self.sleep = clock, sleep or _sleep
        self.factor = 1.0
        self.status: dict[str, tuple[str, int]] = {}
        self._pause_until = 0.0
        self._last: float | None = None
        self._recent: deque[float] = deque()
        self.waited = 0.0

    @property
    def interval(self) -> float:
        return self.factor / self.rate

    def wait(self) -> None:
        now = self.clock()
        t = now if self._last is None else max(now, self._last + self.interval)
        t = max(t, self._pause_until)
        while self._recent and self._recent[0] <= t - 60.0:
            self._recent.popleft()
        if len(self._recent) >= self.per_minute:
            t = max(t, self._recent[0] + 60.0)
        if t > now:
            self.waited += t - now
            self.sleep(t - now)
        self._last = t
        self._recent.append(t)

    def update(self, header: str) -> None:
        status = parse_throttling(header)
        if not status:
            return
        self.status = status
        self.factor, pause = throttle_factor(status)
        if pause:
            self.pause(pause)

    def pause(self, seconds: float) -> None:
        self._pause_until = max(self._pause_until, self.clock() + seconds)


# ============================================================================================ cache
def default_cache_path() -> Path:
    """Per-machine cache (``QUANTABENCH_CACHE_DIR`` overrides the platform cache folder)."""
    import os

    from platformdirs import user_cache_dir

    root = os.environ.get("QUANTABENCH_CACHE_DIR", "").strip() or user_cache_dir("QuantaBench", appauthor=False)
    return Path(root) / "pubchem" / "pubchem_cache.sqlite"


class ResponseCache:
    """SQLite store of PubChem responses keyed by method + URL + POST body; one commit per response."""

    def __init__(self, path: str | Path):
        path = Path(path).expanduser()
        if path.suffix.lower() not in (".sqlite", ".db", ".sqlite3"):
            path = path / "pubchem_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, url TEXT, status INTEGER, "
                         "body BLOB, fetched_utc TEXT)")
        self._db.commit()

    def get(self, key: str) -> tuple[int, bytes] | None:
        with self._lock:
            row = self._db.execute("SELECT status, body FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return int(row[0]), zlib.decompress(row[1]) if row[1] else b""

    def put(self, key: str, url: str, status: int, body: bytes) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                             (key, url, status, zlib.compress(body), datetime.now(UTC).isoformat(timespec="seconds")))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._db.close()


# ============================================================================================ client
def _chunks(items: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _num(v):
    """PubChem sends MolecularWeight and ExactMass as strings: numbers as float, others unchanged."""
    if isinstance(v, str):
        with contextlib.suppress(ValueError):
            return float(v)
    return v


def _normalize_props(p: dict) -> dict:
    out = dict(p)
    for new, old in OLD_PROPERTY_NAMES.items():
        if not out.get(new) and out.get(old):
            out[new] = out[old]
        out.pop(old, None)
    for k in ("MolecularWeight", "ExactMass"):
        if k in out:
            out[k] = _num(out[k])
    return out


class PubChemClient:
    """PUG REST + PUG View within the usage policy, with a response cache. Thread-safe (one request at a time)."""

    def __init__(self, cache: str | Path | ResponseCache | bool | None = None, offline: bool = False,
                 rate: float = DEFAULT_RATE, per_minute: int = MAX_PER_MINUTE, transport: Transport | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] | None = None,
                 timeout: float = 30.0, max_retries: int = 5, log: Callable[[str], None] | None = None):
        if isinstance(cache, ResponseCache):
            self.cache: ResponseCache | None = cache
        elif cache is False:
            self.cache = None
        else:
            try:
                self.cache = ResponseCache(cache if cache not in (None, True) else default_cache_path())
            except (OSError, sqlite3.Error) as exc:
                (log or _log.warning)(f"PubChem cache not available ({exc}); continuing without it")
                self.cache = None
        if offline and self.cache is None:
            raise ValueError("offline mode needs a cache")
        self.offline = offline
        self.transport = transport or urllib_transport
        self.sleep = sleep or _sleep
        self.limiter = RateLimiter(rate, per_minute, clock, self.sleep)
        self.timeout, self.max_retries = timeout, max_retries
        self.log = log or _log.info
        self.headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        self.requests = 0           # HTTP requests sent
        self.cache_hits = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ raw requests
    def fetch(self, url: str, data: dict | None = None) -> tuple[int, bytes]:
        """(status, body) of a GET (or a POST of the form ``data``), from the cache when it has it."""
        body = urllib.parse.urlencode(data).encode() if data else None
        method = "POST" if body else "GET"
        key = f"{method} {url}" + (f"\n{body.decode()}" if body else "")
        with self._lock:
            if self.cache is not None:
                hit = self.cache.get(key)
                if hit is not None:
                    self.cache_hits += 1
                    return hit
            if self.offline:
                raise NotCached(f"not in the cache (offline): {url}")
            headers = dict(self.headers)
            if body:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            last = ""
            for attempt in range(self.max_retries + 1):
                self.limiter.wait()
                try:
                    resp = self.transport(method, url, body, headers, self.timeout)
                except OSError as exc:          # URLError, timeouts, resets
                    last = f"{type(exc).__name__}: {exc}"
                    resp = None
                else:
                    self.requests += 1
                    self.limiter.update(resp.header("X-Throttling-Control"))
                    if resp.status not in RETRY_STATUSES:
                        break
                    last = f"HTTP {resp.status}"
                if attempt < self.max_retries:
                    wait = min(60.0, 2.0 ** (attempt + 1))
                    self.log(f"PubChem: {last}; retrying in {wait:.0f} s")
                    self.limiter.pause(wait)
            else:
                raise PubChemUnavailable(f"PubChem unavailable after {self.max_retries + 1} tries ({last}): {url}")
            if resp.status in (200, 400, 404) and self.cache is not None:
                self.cache.put(key, url, resp.status, resp.body)
            return resp.status, resp.body

    def get_json(self, url: str, data: dict | None = None) -> dict | None:
        """Parsed JSON of a 200; None for 404 (not found); :class:`PubChemFault` for other faults."""
        status, body = self.fetch(url, data)
        if status == 404:
            return None
        try:
            doc = json.loads(body.decode("utf-8")) if body else {}
        except ValueError as exc:
            raise PubChemFault(status, "BadJSON", str(exc)) from exc
        if status != 200:
            fault = doc.get("Fault", {}) if isinstance(doc, dict) else {}
            raise PubChemFault(status, fault.get("Code", ""), "; ".join([fault.get("Message", "")]
                                                                       + list(fault.get("Details", []))))
        return doc

    # ------------------------------------------------------------------ identifiers -> CIDs
    @staticmethod
    def _cids(doc: dict | None) -> list[int]:
        if not doc:
            return []
        return [int(c) for c in doc.get("IdentifierList", {}).get("CID", []) if int(c) > 0]   # CID 0 = unknown

    def cids_by_name(self, name: str) -> list[int]:
        """CIDs of a name, synonym, CAS RN or DTXSID (PubChem's name index)."""
        name = name.strip()
        if any(ch in name for ch in "/\\#?%"):
            return self._cids(self.get_json(f"{REST}/compound/name/cids/JSON", {"name": name}))
        return self._cids(self.get_json(f"{REST}/compound/name/{urllib.parse.quote(name, safe='')}/cids/JSON"))

    def cids_by_inchikey(self, inchikey: str) -> list[int]:
        return self._cids(self.get_json(f"{REST}/compound/inchikey/{inchikey.strip()}/cids/JSON"))

    def cids_by_smiles(self, smiles: str) -> list[int]:
        smiles = smiles.strip()
        url = f"{REST}/compound/smiles/cids/JSON?" + urllib.parse.urlencode({"smiles": smiles})
        if len(url) > 1800:
            return self._cids(self.get_json(f"{REST}/compound/smiles/cids/JSON", {"smiles": smiles}))
        return self._cids(self.get_json(url))

    # ------------------------------------------------------------------ batched data
    def properties(self, cids: Iterable[int], names: Sequence[str] = PROPERTIES) -> dict[int, dict]:
        """Computed properties of ``cids`` in batches of :data:`PROPERTY_BATCH`; old SMILES names normalized."""
        out: dict[int, dict] = {}
        ids = sorted({int(c) for c in cids if c})
        for batch in _chunks(ids, PROPERTY_BATCH):
            url = f"{REST}/compound/cid/{','.join(map(str, batch))}/property/{','.join(names)}/JSON"
            doc = self.get_json(url) or {}
            for p in doc.get("PropertyTable", {}).get("Properties", []):
                out[int(p["CID"])] = _normalize_props(p)
        return out

    def _information(self, cids: Iterable[int], what: str, key: str) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        ids = sorted({int(c) for c in cids if c})
        for batch in _chunks(ids, LIST_BATCH):
            doc = self.get_json(f"{REST}/compound/cid/{','.join(map(str, batch))}/{what}/JSON") or {}
            for info in doc.get("InformationList", {}).get("Information", []):
                out.setdefault(int(info["CID"]), []).extend(info.get(key, []))
        return out

    def synonyms(self, cids: Iterable[int]) -> dict[int, list[str]]:
        return self._information(cids, "synonyms", "Synonym")

    def registry_numbers(self, cids: Iterable[int]) -> dict[int, list[str]]:
        """xrefs RN: the registry numbers (CAS RNs, EC numbers) depositors linked to each CID."""
        return self._information(cids, "xrefs/RN", "RN")

    # ------------------------------------------------------------------ PUG View
    def view(self, cid: int, heading: str = "Experimental Properties") -> dict | None:
        url = f"{VIEW}/data/compound/{int(cid)}/JSON?" + urllib.parse.urlencode({"heading": heading})
        return self.get_json(url)

    def experimental(self, cid: int, headings: Iterable[str] | None = None) -> list[ExperimentalRecord]:
        """Experimental records of one CID ([] when PubChem has no experimental annotations for it)."""
        doc = self.view(cid, "Experimental Properties")
        return parse_view(doc, cid=cid, headings=headings) if doc else []


_DEFAULT: PubChemClient | None = None
_DEFAULT_LOCK = threading.Lock()


def default_client() -> PubChemClient:
    """One shared client per process (GUI name lookups), so its rate limit holds across threads."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = PubChemClient(max_retries=2, timeout=15.0)
        return _DEFAULT


def lookup(name: str, timeout: float = 15.0, client: PubChemClient | None = None) -> dict:
    """{"cid", "smiles", "title"} of a name or CAS RN (one request). Raises :class:`PubChemError` when PubChem
    is unreachable or has no structure for it."""
    client = client or default_client()
    client.timeout = timeout
    quoted = urllib.parse.quote(name.strip(), safe="")
    doc = client.get_json(f"{REST}/compound/name/{quoted}/property/SMILES,ConnectivitySMILES,Title/JSON")
    props = _normalize_props(((doc or {}).get("PropertyTable", {}).get("Properties") or [{}])[0])
    smiles = props.get("SMILES") or props.get("ConnectivitySMILES") or ""
    if not smiles:
        raise PubChemError("no compound with that name" if doc is None else "no structure in the answer")
    return {"cid": props.get("CID"), "smiles": smiles, "title": props.get("Title") or name}


# ============================================================================================ value parsing
_NUM = r"[-+]?\d+(?:\.\d+)?(?:\s*[xX]\s*10\s*\^?\s*[-+]?\s*\d+|[eE][-+]?\d+)?"
_VALUE_RE = re.compile(
    r"(?P<q><=|>=|≤|≥|<|>|~|\bca\.?|\bapprox\.?|\babout)?\s*(?P<a>" + _NUM + r")"
    r"(?:\s*(?:-|to|–)\s*(?P<b>" + _NUM + r"))?(?:\s*(?:±|\+/-)\s*(?P<u>\d+(?:\.\d+)?))?")
#: unit regex (matched right after the number) -> (dimension, unit name)
_UNITS: tuple[tuple[re.Pattern, str, str], ...] = tuple((re.compile(rx, re.I), dim, name) for rx, dim, name in (
    (r"dynes?\s*(?:/\s*cm|cm\s*-1|per\s+cm)|dyn\s*(?:/\s*cm|cm\s*-1)", "surface_tension", "dyn/cm"),
    (r"mN\s*(?:/\s*m\b|m\s*-1|·\s*m\s*-1)|milli\s*newtons?\s*(?:/|per)\s*m", "surface_tension", "mN/m"),
    (r"mJ\s*/\s*m\s*(?:2|²|\^2)", "surface_tension", "mJ/m2"),
    (r"ergs?\s*/\s*(?:sq\s*)?cm\s*(?:2|²|\^2)?", "surface_tension", "erg/cm2"),
    (r"N\s*/\s*m\b|N\s*m\s*-1", "surface_tension", "N/m"),
    (r"g\s*/\s*(?:cm\s*(?:3|³|\^3)|cu\s*cm|cc\b|ml\b)", "density", "g/cm3"),
    (r"kg\s*/\s*(?:m\s*(?:3|³|\^3)|cu\s*m)", "density", "kg/m3"),
    (r"kg\s*/\s*l\b", "density", "g/cm3"),
    (r"°\s*C|º\s*C|deg(?:rees?)?\.?\s*C\b|℃", "temperature", "°C"),
    (r"°\s*F|º\s*F|deg(?:rees?)?\.?\s*F\b", "temperature", "°F"),
    (r"K\b", "temperature", "K"),
    (r"m\s*Pa\s*[.·*-]?\s*s(?:ec)?\b|millipascal", "viscosity", "mPa.s"),
    (r"c[Pp]\b|centipoises?", "viscosity", "cP"),
    (r"Pa\s*[.·*-]?\s*s\b", "viscosity", "Pa.s"),
    (r"\[?\s*mm\s*Hg\s*\]?|torr\b", "pressure", "mmHg"),
    (r"kPa\b", "pressure", "kPa"),
    (r"(?-i:MPa)\b", "pressure", "MPa"),
    (r"(?<!m)Pa\b(?!\s*[.·*-]?\s*s\b)", "pressure", "Pa"),
    (r"atm\b", "pressure", "atm"),
    (r"bar\b", "pressure", "bar"),
    (r"mm\s*(?:2|²)\s*/\s*s|cSt\b|centistokes", "kinematic_viscosity", "mm2/s"),
    (r"mg\s*/\s*l\b", "concentration", "mg/L"),
    (r"(?:ug|µg|μg|microg)\s*/\s*l\b", "concentration", "ug/L"),
    (r"g\s*/\s*l\b", "concentration", "g/L"),
    (r"mg\s*/\s*ml\b", "concentration", "g/L"),
    (r"kJ\s*/\s*mol", "molar_energy", "kJ/mol"),
    (r"kcal\s*/\s*mol", "molar_energy", "kcal/mol"),
    (r"J\s*/\s*mol", "molar_energy", "J/mol"),
))
_TO_STD: dict[str, Callable[[float], float]] = {
    "dyn/cm": lambda v: v, "mN/m": lambda v: v, "mJ/m2": lambda v: v, "erg/cm2": lambda v: v, "N/m": lambda v: v * 1e3,
    "g/cm3": lambda v: v, "kg/m3": lambda v: v / 1e3,
    "°C": lambda v: v, "°F": lambda v: (v - 32.0) * 5.0 / 9.0, "K": lambda v: v - 273.15,
    "mmHg": lambda v: v * 133.322368, "kPa": lambda v: v * 1e3, "MPa": lambda v: v * 1e6, "Pa": lambda v: v,
    "atm": lambda v: v * 101325.0, "bar": lambda v: v * 1e5,
    "mPa.s": lambda v: v, "cP": lambda v: v, "Pa.s": lambda v: v * 1e3,
    "mg/L": lambda v: v, "ug/L": lambda v: v / 1e3, "g/L": lambda v: v * 1e3,
    "kJ/mol": lambda v: v, "kcal/mol": lambda v: v * 4.184, "J/mol": lambda v: v / 1e3,
}
_TEMP_RE = re.compile(r"(?:\bat|@|\(|,|\bof)?\s*(?P<t>-?\d+(?:\.\d+)?)\s*(?P<u>°\s*[CF]|º\s*[CF]|deg(?:rees?)?\.?\s*[CF]\b|℃|K\b)",
                      re.I)
_PRESSURE_RE = re.compile(r"(?:\bat|@)\s*(?P<p>\d+(?:\.\d+)?)\s*\[?\s*(?P<u>mm\s*Hg|torr|kPa|Pa|atm|bar)\b", re.I)


def _float(text: str) -> float:
    t = re.sub(r"\s+", "", text)
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)[xX]10\^?([-+]?\d+)", t)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    return float(t)


def _clean(text: str) -> str:
    return (text.replace("−", "-").replace("–", "-").replace("×", "x").replace(" ", " ")
            .replace(" ", " ").replace(" ", " "))


def _temperature_c(value: float, unit: str) -> float:
    u = unit.replace(" ", "").replace("º", "°").lower()
    if u.endswith("f"):
        return (value - 32.0) * 5.0 / 9.0
    if u == "k":
        return value - 273.15
    return value


@dataclass
class ParsedValue:
    value: float | None = None          # the number (mean of a range)
    low: float | None = None
    high: float | None = None
    qualifier: str = ""
    uncertainty: float | None = None
    unit: str = ""                       # as recognized ("dyn/cm"), "" when none
    value_std: float | None = None       # in the heading's standard unit
    unit_std: str = ""
    temperature_c: float | None = None
    pressure: str = ""                   # "736 mmHg" (boiling points)
    note: str = ""


def _match_unit(rest: str) -> tuple[str, str, int]:
    """(dimension, unit, length) of a unit at the start of ``rest``, else ("", "", 0)."""
    s = rest.lstrip()
    lead = len(rest) - len(s)
    for rx, dim, name in _UNITS:
        m = rx.match(s)
        if m:
            return dim, name, lead + m.end()
    return "", "", 0


def parse_values(text: str, heading: str) -> list[ParsedValue]:
    """The values in one PUG View string for ``heading``: one per ';'-separated segment that holds one, the unit
    carried over from an earlier segment when a later one has none ("22.3 dyn/cm at 20 °C; 21.1 at 30 °C").

    In each segment the value is the first number followed by a unit of the heading's dimension; failing that,
    the first number (unitless headings, and densities written without a unit, which are then taken as g/cm3).
    """
    dim, std = HEADINGS.get(heading, ("none", ""))
    if dim == "text":
        return []
    out: list[ParsedValue] = []
    carry = ""
    for seg in _clean(text).split(";"):
        matches = [m for m in _VALUE_RE.finditer(seg) if m.group("a")]
        if not matches:
            continue
        chosen, unit, uend = None, "", 0
        for m in matches:
            d, name, n = _match_unit(seg[m.end():])
            if d and (d == dim or (dim == "viscosity" and d == "kinematic_viscosity")):
                chosen, unit, uend = m, name, m.end() + n
                break
        if chosen is None:
            chosen = matches[0]
            d, name, n = _match_unit(seg[chosen.end():])
            if d and d != dim and dim != "none":
                if dim == "density" or d == "temperature":   # "1.792 @ 20°C": the number then a temperature
                    name, n = "", 0
                else:
                    continue                    # e.g. a concentration only: no value of this property here
            unit, uend = name, chosen.end() + n
            if not unit and carry and dim not in ("none",):
                unit = carry
        a = _float(chosen.group("a"))
        b = _float(chosen.group("b")) if chosen.group("b") else None
        pv = ParsedValue(qualifier=(chosen.group("q") or "").strip(), unit=unit, unit_std=std)
        if b is not None and b >= a:
            pv.low, pv.high, pv.value = a, b, (a + b) / 2.0
        else:
            pv.value = a
        if chosen.group("u"):
            pv.uncertainty = float(chosen.group("u"))
        if unit in _TO_STD:
            pv.value_std = _TO_STD[unit](pv.value)
        elif dim == "density" and not unit and 0.3 <= pv.value <= 5.0:
            pv.value_std, pv.note = pv.value, "no unit given: taken as g/cm3 (or relative density)"
        elif dim == "none":
            pv.value_std = pv.value
        elif unit == "mm2/s":
            pv.note = "kinematic viscosity (mm2/s): needs the density for mPa.s"
        if dim != "temperature":
            rest = seg[uend:]
            tm = _TEMP_RE.search(rest) or _TEMP_RE.search(seg[:chosen.start()])
            if tm:
                pv.temperature_c = round(_temperature_c(float(tm.group("t")), tm.group("u")), 2)
        pm = _PRESSURE_RE.search(seg[uend:])
        if pm:
            punit = re.sub(r"\s+", "", pm.group("u"))
            pv.pressure = f"{pm.group('p')} {punit}"
        carry = unit or carry
        out.append(pv)
    return out


# ============================================================================================ records
@dataclass
class ExperimentalRecord:
    cid: int | None
    heading: str
    raw: str
    value: float | None = None
    value_low: float | None = None
    value_high: float | None = None
    qualifier: str = ""
    uncertainty: float | None = None
    unit: str = ""
    value_std: float | None = None
    unit_std: str = ""
    temperature_c: float | None = None
    pressure: str = ""
    condition_class: str = ""
    concentration: str = ""
    condition_basis: str = ""
    condition_note: str = ""
    needs_review: bool = False
    parse_note: str = ""
    source_name: str = ""
    source_id: str = ""
    reference: str = ""
    doi: str = ""
    description: str = ""
    url: str = ""
    license_url: str = ""
    anid: int | None = None
    from_heading: str = ""              # "Other Experimental Properties" when re-filed

    def row(self) -> dict:
        d = asdict(self)
        d["needs_review"] = "yes" if self.needs_review else ""
        return {k: ("" if v is None else v) for k, v in d.items()}


RECORD_COLUMNS = tuple(ExperimentalRecord.__dataclass_fields__)


def _strings(value: dict) -> list[str]:
    out = [s.get("String", "") for s in value.get("StringWithMarkup", []) if s.get("String")]
    if "Number" in value:
        unit = value.get("Unit", "")
        out += [f"{n} {unit}".strip() for n in value.get("Number", [])]
    return out


def _records_from_info(cid, heading: str, info: dict, ref: dict | None, keep: set[str] | None
                       ) -> list[ExperimentalRecord]:
    ref = ref or {}
    refs = list(info.get("Reference") or [])
    ext = [e.get("Matched", {}) for e in info.get("ExtendedReference") or []]
    matched = "; ".join(m.get("Citation", "") for m in ext if m.get("Citation"))
    reference = "; ".join(refs)
    if matched and matched not in reference:
        reference = f"{reference} [{matched}]" if reference else matched
    doi = next((m.get("DOI", "") for m in ext if m.get("DOI")), "")
    if not doi:
        m = re.search(r"\b(10\.\d{4,9}/\S+)", " ".join(refs))
        doi = m.group(1).rstrip(".,;)") if m else ""
    base = dict(cid=cid, source_name=ref.get("SourceName", ""), source_id=str(ref.get("SourceID", "")),
                reference=reference, doi=doi, description=info.get("Description", "") or "",
                url=ref.get("URL", ""), license_url=ref.get("LicenseURL", ""), anid=ref.get("ANID"))
    out = []
    for s in _strings(info.get("Value", {})):
        h, origin = heading, ""
        if heading == "Other Experimental Properties":
            h = next((lab for rx, lab in _OTHER_LABELS if rx.search(s)), "")
            origin = heading
            if not h:
                continue
        if keep is not None and h not in keep:
            continue
        values = parse_values(s, h)
        if not values:
            out.append(ExperimentalRecord(heading=h, raw=s, from_heading=origin, **base))
        for pv in values:
            out.append(ExperimentalRecord(heading=h, raw=s, value=pv.value, value_low=pv.low, value_high=pv.high,
                                          qualifier=pv.qualifier, uncertainty=pv.uncertainty, unit=pv.unit,
                                          value_std=pv.value_std, unit_std=pv.unit_std if pv.value_std is not None
                                          else "", temperature_c=pv.temperature_c, pressure=pv.pressure,
                                          parse_note=pv.note, from_heading=origin, **base))
    return out


def parse_view(doc: dict, cid: int | None = None, headings: Iterable[str] | None = None,
               classify: bool = True) -> list[ExperimentalRecord]:
    """Records of a PUG View compound record (any heading's subtree). ``headings``: the headings to keep (default
    :data:`HEADINGS`; ``"all"`` keeps every heading's strings, unparsed where the heading is not known)."""
    rec = (doc or {}).get("Record", {})
    cid = cid if cid is not None else rec.get("RecordNumber")
    refs = {r.get("ReferenceNumber"): r for r in rec.get("Reference", [])}
    keep = None if headings == "all" else set(headings or HEADINGS)
    out: list[ExperimentalRecord] = []

    def walk(sections):
        for sec in sections or []:
            h = sec.get("TOCHeading", "")
            for info in sec.get("Information", []):
                out.extend(_records_from_info(cid, h, info, refs.get(info.get("ReferenceNumber")), keep))
            walk(sec.get("Section"))

    walk(rec.get("Section"))
    return classify_conditions(out) if classify else out


# ============================================================================================ conditions
_AQUEOUS_TEXT = re.compile(r"\baqueous\b|\baq\.|\bin water\b|\bwater solution|\bsolution\b|\bsoln\b|\bcmc\b|"
                           r"critical micelle|\bmicell|\bdilute\b|\bsaturated (?:aqueous )?solution", re.I)
_CONCENTRATION = re.compile(r"(?P<c>\d+(?:\.\d+)?\s*(?:wt\s*%|w/w\s*%|%\s*(?:w/w|w/v|wt)?|mg\s*/\s*l\b|g\s*/\s*l\b|"
                            r"mmol\s*/\s*l\b|mol\s*/\s*l\b|(?-i:mM)\b|ppm\b|g\s*/\s*100\s*ml))", re.I)
_PURE_TEXT = re.compile(r"\bpure\b|\bneat\b|in contact with (?:its own |the )?(?:vapou?r|air)|\bliquid\b", re.I)
#: references whose surface-tension tables are of pure liquids
_PURE_REFS = re.compile(r"crc handbook|handbook of chemistry and physics|\bjasper\b|lange'?s handbook|\byaws\b|"
                        r"\bdippr\b|\briddick\b|handbook of organic chemistry|dean,? j\.? ?a\.?|"
                        r"j\.? ?phys\.? ?chem\.? ?ref\.? ?data|thermophysical properties", re.I)
_SURFACTANT_REFS = re.compile(r"surfactant|micell|\bfoams?\b|amphiphil|\baqueous\b|\bcmc\b|interfac", re.I)
ROOM_T = 25.0


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else None


def _transition_c(records: list[ExperimentalRecord], heading: str) -> float | None:
    vals = []
    for r in records:
        if r.heading == heading and r.value_std is not None:
            if heading == "Boiling Point" and r.pressure and not re.match(r"7[4-7]\d\s*mmHg|10[01](\.\d+)?\s*kPa",
                                                                          r.pressure):
                continue
            vals.append(r.value_std)
    return _median(vals)


def classify_conditions(records: list[ExperimentalRecord]) -> list[ExperimentalRecord]:
    """Set ``condition_class`` (and basis, note, concentration, needs_review) on the liquid-property records
    (:data:`LIQUID_HEADINGS`) of one compound's records; the melting and boiling points among them inform it."""
    by_cid: dict = {}
    for r in records:
        by_cid.setdefault(r.cid, []).append(r)
    for recs in by_cid.values():
        mp, bp = _transition_c(recs, "Melting Point"), _transition_c(recs, "Boiling Point")
        for r in recs:
            if r.heading in LIQUID_HEADINGS:
                _classify(r, mp, bp)
    return records


def _classify(r: ExperimentalRecord, mp: float | None, bp: float | None) -> None:
    text = r.raw
    ref = r.reference
    t = r.temperature_c if r.temperature_c is not None else ROOM_T
    t_note = "" if r.temperature_c is not None else f"temperature not stated (taken as {ROOM_T:g} °C)"
    solid = mp is not None and mp > t + 0.5
    liquid = mp is not None and mp < t and (bp is None or bp > t)
    notes = []
    conc = _CONCENTRATION.search(text)
    if _AQUEOUS_TEXT.search(text) or conc:
        r.condition_class, r.condition_basis = "aqueous_solution", "text"
        r.concentration = conc.group("c").strip() if conc else ""
        if not conc:
            notes.append("concentration not stated")
    elif _PURE_TEXT.search(text):
        r.condition_class, r.condition_basis = "pure_liquid", "text"
    elif solid:
        if r.heading == "Surface Tension":
            r.condition_class, r.condition_basis, r.needs_review = "unknown", "phase", True
            why = (f"compound is solid at {t:g} °C (mp {mp:g} °C), so this is not a pure-liquid value at that "
                   "temperature")
            if _SURFACTANT_REFS.search(ref):
                why += "; the reference is surfactant literature: very likely an aqueous solution at or above the " \
                       "CMC (gamma_CMC)"
            else:
                why += "; likely an aqueous-solution value"
            notes.append(why)
        else:
            r.condition_class, r.condition_basis = "solid", "phase"
            notes.append(f"compound is solid at {t:g} °C (mp {mp:g} °C): a solid-state value, not the liquid's")
    elif _PURE_REFS.search(ref):
        r.condition_class, r.condition_basis = "pure_liquid", "reference"
    elif _SURFACTANT_REFS.search(ref) and r.heading == "Surface Tension":
        r.condition_class, r.condition_basis, r.needs_review = "unknown", "reference", True
        notes.append("the reference is surfactant literature: possibly an aqueous solution at or above the CMC")
    elif liquid:
        r.condition_class, r.condition_basis = "pure_liquid", "phase"
        notes.append(f"inferred: liquid at {t:g} °C (mp {mp:g} °C) and no solution named")
    else:
        r.condition_class, r.condition_basis, r.needs_review = "unknown", "", True
        notes.append("no solution or pure-liquid cue, and no melting point to tell")
    if t_note and r.heading == "Surface Tension":
        notes.append(t_note)
    r.condition_note = "; ".join(notes)


# ============================================================================================ summaries
def best_value(records: list[ExperimentalRecord], heading: str, classes: Sequence[str] = ("pure_liquid",),
               target_t: float = ROOM_T) -> ExperimentalRecord | None:
    """The record of ``heading`` in ``classes`` with a standard value whose temperature is closest to ``target_t``
    (records without a temperature rank after those within 10 K); peer-reviewed sources first among equals."""
    cands = [r for r in records if r.heading == heading and r.value_std is not None
             and (not classes or r.condition_class in classes)]
    if not cands:
        return None

    def rank(r):
        dt = abs(r.temperature_c - target_t) if r.temperature_c is not None else 10.0 + 0.01
        return (dt, 0 if "PEER" in r.description.upper() else 1, r.needs_review)

    return min(cands, key=rank)


def parachor(mw: float | None, sigma_mn_m: float | None, density_g_cm3: float | None) -> float | None:
    """Sugden parachor P = M σ^(1/4) / ρ_l (ρ_vapor neglected far below the critical point), in
    cm³ mol⁻¹ (mN/m)^(1/4): the inverse of σ = (P Δρ / M)^4."""
    if not mw or not sigma_mn_m or not density_g_cm3 or sigma_mn_m <= 0 or density_g_cm3 <= 0:
        return None
    return float(mw) * float(sigma_mn_m) ** 0.25 / float(density_g_cm3)
