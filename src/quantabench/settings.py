"""User settings and where they live.

Settings are one JSON file, ``settings.json``. Its folder is chosen in this order:

1. ``QUANTABENCH_SETTINGS_DIR`` environment variable.
2. A pointer file ``settings_location.json`` (``{"settings_dir": "..."}``), looked up first in the
   repository root when running from a source checkout (gitignored), then in the per-user config
   folder. This is how a machine points QuantaBench at a synced folder, e.g.
   ``~/SyncThing/QuantaBench/Settings``.
3. The per-user config folder (``%APPDATA%\\QuantaBench`` on Windows, ``~/.config/QuantaBench`` on Linux).

Paths inside settings may use ``~``; they are stored portably so the same file works on every machine.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from platformdirs import user_config_dir, user_documents_dir

APP_NAME = "QuantaBench"
SETTINGS_FILE = "settings.json"
POINTER_FILE = "settings_location.json"
ENV_SETTINGS_DIR = "QUANTABENCH_SETTINGS_DIR"


@dataclass
class ResourceSettings:
    threads: int = 0                 # 0 = all logical cores minus one (keeps the GUI responsive)
    gpu: str = "auto"                # "auto" | "on" | "off"
    gpu_index: int = 0
    gpu_memory_fraction: float = 0.9  # cap for PyTorch's caching allocator
    parallel_molecules: int = 0      # batch jobs: molecules in flight at once (0 = auto)


@dataclass
class BuildDefaults:
    ph: float = 7.0
    protonate: bool = True
    keep_largest_fragment: bool = True
    max_conformers: int = 50
    energy_window_kcal: float = 6.0
    rmsd_threshold: float = 0.5
    refine_method: str = "auto"       # "none" | "mmff" | "gfn1-xtb" | "gfn2-xtb" | "aimnet2" | "auto"
    fmax: float = 0.03                # eV/Å convergence for the refinement optimizer
    max_steps: int = 500
    temperature_k: float = 298.15
    random_seed: int = 42


@dataclass
class Settings:
    output_dir: str = ""              # "" = Documents/QuantaBench/Runs
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    build: BuildDefaults = field(default_factory=BuildDefaults)
    recent_files: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- paths
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(os.path.expanduser(self.output_dir))
        return Path(user_documents_dir()) / APP_NAME / "Runs"

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        s = cls()
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            current = getattr(s, f.name)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                known = {k: v for k, v in value.items() if k in current.__dataclass_fields__}
                setattr(s, f.name, type(current)(**{**asdict(current), **known}))
            else:
                setattr(s, f.name, value)
        return s

    def to_dict(self) -> dict:
        return asdict(self)


# -------------------------------------------------------------------- location
def repo_root() -> Path | None:
    """The source checkout this module runs from, if any (has pyproject.toml and src/quantabench)."""
    here = Path(__file__).resolve()
    for parent in here.parents[:4]:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "quantabench").is_dir():
            return parent
    return None


def default_settings_dir() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False, roaming=True))


def _read_pointer(folder: Path) -> str:
    try:
        data = json.loads((folder / POINTER_FILE).read_text(encoding="utf-8"))
        return str(data.get("settings_dir") or "").strip()
    except (OSError, ValueError, AttributeError):
        return ""


def pointer_locations() -> list[Path]:
    places = []
    root = repo_root()
    if root is not None:
        places.append(root)
    places.append(default_settings_dir())
    return places


def settings_dir() -> Path:
    env = os.environ.get(ENV_SETTINGS_DIR, "").strip()
    if env:
        return Path(os.path.expanduser(env))
    for place in pointer_locations():
        value = _read_pointer(place)
        if value:
            return Path(os.path.expanduser(value))
    return default_settings_dir()


def portable(path: str | Path) -> str:
    """Store a path with ``~`` when it is under the home folder, so it works on every machine."""
    p = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    try:
        return "~/" + p.relative_to(home).as_posix()
    except ValueError:
        return str(p)


def set_settings_dir(new_dir: str | Path | None, pointer_place: Path | None = None) -> Path:
    """Point QuantaBench at ``new_dir`` ('' or None = back to the default). Copies settings.json
    into the new folder if it has none, never deletes anything. Returns the folder now in use."""
    old = settings_dir()
    place = pointer_place or (repo_root() or default_settings_dir())
    place.mkdir(parents=True, exist_ok=True)
    value = portable(new_dir) if new_dir else ""
    (place / POINTER_FILE).write_text(json.dumps({"settings_dir": value}, indent=2) + "\n", encoding="utf-8")
    new = settings_dir()
    new.mkdir(parents=True, exist_ok=True)
    if (old / SETTINGS_FILE).is_file() and not (new / SETTINGS_FILE).exists():
        (new / SETTINGS_FILE).write_text((old / SETTINGS_FILE).read_text(encoding="utf-8"), encoding="utf-8")
    return new


def load() -> Settings:
    path = settings_dir() / SETTINGS_FILE
    try:
        return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError):
        return Settings()


def save(settings: Settings) -> Path:
    folder = settings_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SETTINGS_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
