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
"""

from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MachineProfile:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def machine_name() -> str:
    return socket.gethostname().split(".")[0].lower()


def registry_dir() -> Path:
    from .. import settings as settings_mod

    return settings_mod.settings_dir() / "machines"


def save_profile(p: MachineProfile) -> Path:
    d = registry_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{p.name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(p.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_profiles() -> list[MachineProfile]:
    out = []
    for path in sorted(registry_dir().glob("*.json")) if registry_dir().is_dir() else []:
        try:
            out.append(MachineProfile.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(out, key=lambda p: p.updated, reverse=True)


def local_profile(bench: bool = False, methods=None) -> MachineProfile:
    """Placeholder until the profile builder lands: name and QuantaBench version only."""
    from .. import __version__

    return MachineProfile(name=machine_name(), quantabench_version=__version__)


def throughput_for(p: MachineProfile, method: str, device: str = "auto") -> float | None:
    for key, value in p.throughput.items():
        m, _, dev = key.partition("@")
        if m == method and (device == "auto" or dev == device):
            return float(value)
    return None
