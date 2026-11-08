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

import copy
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
    linalg: str = "auto"             # dxtb dense linear algebra a GPU lacks: "auto" | "cpu" | "device" (torch_compat)
    dxtb_gpu: str = "auto"           # dxtb on a GPU: "auto" (only where faster) | "on" | "off" (torch_compat)
    speed_aware_lanes: bool = True   # lanes: slower lanes leave the last molecules to faster ones (dist.speed)
    dynamic_threads: bool = True     # lanes: a CREST search may also use idle lanes' threads (dist.threads)
    # memory budget per process for the dxtb steps (chem.memory_guard), GB: 0 = available memory / lanes
    # (at least 1.5 GB), < 0 = no guard
    memory_budget_gb: float = 0.0
    # wall-clock limit per molecule in a batch, s (0 = none): a lane past it is killed and restarted, one
    # process stops the molecule at its next checkpoint; the molecule is marked failed "time limit (N s)"
    molecule_timeout_s: float = 1800.0


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
    batched: str = "auto"             # batched conformer refinement: "auto" (GPU only) | "on" | "off"
    frequencies: bool = False         # frequencies + thermochemistry of the best conformer
    conformer_search: str = "rdkit"   # "rdkit" (ETKDGv3 + MMFF) | "crest" (iMTD-GC, external)
    solvent: str = ""                 # implicit solvent for engines that support one ("" = gas phase)
    qm_method: str = ""               # final QM level, e.g. "wb97x-d/def2-svp" ("" = no QM stage)


@dataclass
class EngineSettings:
    """External programs (see ``chem/external.py``). Paths may use ``~``; ``wsl:<command>`` runs a
    Linux program inside WSL on Windows (e.g. ``wsl:crest``).

    The settings file is shared by every machine (synced folder), but a program's path usually belongs to one
    machine (NOVA's ``wsl:/home/.../crest``, the Mac's own build): ``machine_paths`` holds each machine's paths
    (keyed by ``machine_key`` of its name, like ``dist.machine_lanes``), and ``paths`` is the default for every
    machine. Use ``Settings.engine_paths_for`` / ``set_engine_path_for``."""
    paths: dict[str, str] = field(default_factory=dict)  # {"xtb": ..., "crest": ..., "mopac": ..., "psi4_python": ...}
    # machine name -> {program: path} of that machine only; wins over ``paths`` there ("" = find it automatically
    # on that machine, even when ``paths`` names one)
    machine_paths: dict[str, dict[str, str]] = field(default_factory=dict)
    scratch_dir: str = ""             # "" = <system temp>/QuantaBench
    keep_scratch: bool = False        # keep job folders of successful external runs
    omp_stacksize: str = "4G"         # OMP_STACKSIZE for xtb/CREST/MOPAC/Psi4 child processes
    qm_memory_gb: float = 4.0         # memory per Psi4 job
    timeout_s: float = 0.0            # 0 = no limit per external run


@dataclass
class DistSettings:
    """Splitting work across lanes and machines (``quantabench.dist``, docs/PLATFORMS.md).

    The settings file is shared by every machine (synced folder), but a lane spec describes one machine's
    hardware: ``machine_lanes`` holds each machine's own spec (keyed by ``machine_key`` of its name), and
    ``lanes`` is only the default for a machine without an entry. Use ``Settings.lanes_for``."""
    lanes: str = "1"                  # default lane spec: "1" (one process), "auto", "N", "gpu:1x4,cpu:5x2"
    machine_lanes: dict[str, str] = field(default_factory=dict)   # machine name -> that machine's lane spec
    worker_port: int = 8765
    worker_bind: str = "tailscale"    # "tailscale" (this machine's 100.x address) | "localhost" | an address
    worker_token: str = ""            # shared secret for workers; generated on first `worker` start
    workers: dict[str, str] = field(default_factory=dict)   # machine name -> URL override


@dataclass
class SurfaceSettings:
    """Areas, volumes and surfaces (``chem.radii``, ``chem.descriptors``), as Spartan's Preferences > VDW Radii and
    the Surface box of its Settings tab."""
    radii_set: str = "bondi"          # radii for sasa_A2 / vdw_area_A2 / vdw_volume_A3: "bondi" | "spartan"
    radii_overrides: dict[str, float] = field(default_factory=dict)   # element symbol -> Å (edited table rows)
    sasa_probe_A: float = 1.4         # solvent probe radius for sasa_A2 (water 1.4 Å; Spartan's accessible area 1.0)
    # The ESP polar areas (esp_polar_area_{75,100,125}_A2) use Spartan's three fixed |ESP| thresholds, so their
    # columns are the same in every run; no setting.


@dataclass
class DisplaySettings:
    """How the GUI shows energies (``quantabench.units``), as Spartan's Preferences > Document. Display only: not
    a science setting (``jobs.SCIENCE_FIELDS``), never sent to grid workers, and exported files keep their columns
    and units (``_kcal``, ``_eV``)."""
    energy_unit: str = "kJ/mol"           # total energies (conformer E, best E, QM E, H, G): kJ/mol | kcal/mol | eV | hartree
    relative_energy_unit: str = "kJ/mol"  # differences (ΔE, energy window, ZPE, H − E, G − E) and the ESP on surfaces
    significant_digits: int = 0           # 0 = automatic (QuantaBench's precision per value), else 1-15
    number_style: str = "auto"            # "auto" | "decimal" | "scientific"


@dataclass
class ImportSettings:
    """Stored choices of the Import Options dialog (File > Open; ``chem.import_options.ImportOptions`` has the same
    fields and says what each does), also the defaults of the command line's ``--file`` inputs. ``ask``: show the
    dialog when a file has choices (its "Don't ask again" box clears it; Settings > Import sets it back)."""
    ask: bool = True
    convert_2d: bool = True           # 2D structures (MOL/SDF 2D flag or every z = 0, ChemDraw) get 3D coordinates
    grow_hydrogens: bool = True       # explicit hydrogens with coordinates
    keep_coords: bool = False         # the file's 3D coordinates are the first conformer of the build
    post_fields: bool = False         # SD properties / table columns become molecule-table and summary.csv columns
    parse_fields: bool = True         # numbers in those fields are parsed (a unit goes into the column name)
    id_field: str = ""                # SD property / column for the id ("" = automatic: DTXSID, ID, the title)
    name_field: str = ""              # ... for the name ("" = the title)
    remove_water: bool = True
    remove_acids: bool = True         # HCl, HBr, HI and other small neutral inorganic fragments
    strip_counterions: bool = False   # Na+, K+, Cl-, SO4 2-, ... (the build's keep-largest-fragment does it anyway)
    keep_largest: bool = False        # only the largest fragment, chosen as the build chooses it
    label_order: str = "read"         # "read" (the file's atom order) | "standard" (RDKit canonical ranks)
    same_constitution: str = "molecules"   # several structures of one constitution: "molecules" | "conformers"


@dataclass
class Settings:
    output_dir: str = ""              # "" = Documents/QuantaBench/Runs
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    build: BuildDefaults = field(default_factory=BuildDefaults)
    engines: EngineSettings = field(default_factory=EngineSettings)
    dist: DistSettings = field(default_factory=DistSettings)
    surfaces: SurfaceSettings = field(default_factory=SurfaceSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    import_: ImportSettings = field(default_factory=ImportSettings)     # "import" is a Python keyword
    recent_files: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- paths
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(os.path.expanduser(self.output_dir))
        return Path(user_documents_dir()) / APP_NAME / "Runs"

    # ---------------------------------------------------------------- io
    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        """Settings from JSON. Unknown keys are ignored, and a value of the wrong type (a hand-edited or
        corrupted file: ``"paths": null``) falls back to the default instead of breaking the app."""
        s = cls()
        if not isinstance(data, dict):
            return s
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            current = getattr(s, f.name)
            if hasattr(current, "__dataclass_fields__"):
                if not isinstance(value, dict):
                    continue
                merged = asdict(current)
                for k, v in value.items():
                    if k in merged:
                        ok, v = _coerce(merged[k], v, tables=(f.name, k) in NESTED_TABLES)
                        if ok:
                            merged[k] = v
                setattr(s, f.name, type(current)(**merged))
            else:
                ok, value = _coerce(current, value)
                if ok:
                    setattr(s, f.name, value)
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    # ---------------------------------------------------------------- per-machine values
    def lanes_for(self, machine: str | None = None) -> str:
        """The lane spec of ``machine`` (default: this one): its ``dist.machine_lanes`` entry, else
        ``dist.lanes``. Resolve it into lanes with ``dist.profile.resolve_lanes``."""
        key = machine_key(machine if machine is not None else _this_machine())
        for name, spec in self.dist.machine_lanes.items():
            if machine_key(name) == key and str(spec).strip():
                return str(spec).strip()
        return (self.dist.lanes or "1").strip() or "1"

    def set_lanes_for(self, spec: str, machine: str | None = None) -> None:
        """Store ``spec`` as the lane spec of ``machine`` (default: this one) only."""
        key = machine_key(machine if machine is not None else _this_machine())
        lanes = {k: v for k, v in self.dist.machine_lanes.items() if machine_key(k) != key}
        lanes[key] = spec
        self.dist.machine_lanes = lanes

    def engine_paths_for(self, machine: str | None = None) -> dict[str, tuple[str, str]]:
        """Program -> (path, scope) on ``machine`` (default: this one), scope "this machine" for its
        ``engines.machine_paths`` entry and "all machines" for ``engines.paths``. A program the machine set to ""
        (find it automatically) is left out, as is any blank path."""
        out = {k: (str(v).strip(), "all machines") for k, v in self.engines.paths.items() if isinstance(v, str)}
        own = self._machine_paths(machine)
        out.update({k: (str(v).strip(), "this machine") for k, v in own.items() if isinstance(v, str)})
        return {k: v for k, v in out.items() if v[0]}

    def set_engine_path_for(self, program: str, path: str, machine: str | None = None) -> None:
        """Set ``program``'s path on ``machine`` (default: this one) only; ``""`` = find it automatically there.
        A value equal to ``engines.paths`` (the default of every machine) is not stored again for the machine."""
        key = machine_key(machine if machine is not None else _this_machine())
        path = str(path or "").strip()
        entry = dict(self._machine_paths(machine))
        shared = str(self.engines.paths.get(program, "") or "").strip()
        if path == shared:
            entry.pop(program, None)
        else:
            entry[program] = path
        table = {k: v for k, v in self.engines.machine_paths.items() if machine_key(k) != key}
        if entry:
            table[key] = entry
        self.engines.machine_paths = table

    def _machine_paths(self, machine: str | None = None) -> dict[str, str]:
        key = machine_key(machine if machine is not None else _this_machine())
        merged: dict[str, str] = {}
        for name, entry in self.engines.machine_paths.items():
            if machine_key(name) == key and isinstance(entry, dict):
                merged.update({str(k): str(v) for k, v in entry.items() if isinstance(v, str)})
        return merged


def machine_key(name: str) -> str:
    """Machine names compared like Tailscale MagicDNS names (lowercase, spaces and underscores as '-'): the
    same rule as ``dist.plan.norm_name``."""
    return "-".join(str(name).strip().lower().replace("_", " ").split())


def this_machine() -> str:
    """This machine's key in the per-machine settings (``machine_key`` of ``dist.profile.machine_name``)."""
    return machine_key(_this_machine())


def _this_machine() -> str:
    from .dist.profile import machine_name

    return machine_name()


#: (section, field) of the settings whose values are tables of strings (machine -> {program: path})
NESTED_TABLES = {("engines", "machine_paths")}


def _coerce(default, value, tables: bool = False) -> tuple[bool, object]:
    """(accepted, value) for a JSON value replacing ``default``: same type, or an int where a float goes.
    ``tables``: a dict of string tables (``NESTED_TABLES``) instead of a dict of strings and numbers."""
    if isinstance(default, bool):
        return isinstance(value, bool), value
    if isinstance(default, float):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, float(value)
        return False, value
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool), value
    for kind in (str, dict, list):
        if isinstance(default, kind):
            if kind is dict and isinstance(value, dict) and tables:
                return True, {str(k): {str(a): b for a, b in v.items() if isinstance(b, str)}
                              for k, v in value.items() if isinstance(v, dict)}
            if kind is dict and isinstance(value, dict):
                # string values (paths, lane specs, URLs) and numbers (per-element radii); nothing else
                return True, {str(k): v for k, v in value.items()
                              if isinstance(v, str) or (isinstance(v, (int, float)) and not isinstance(v, bool))}
            return isinstance(value, kind), value
    return True, value


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


#: (section, field) of the settings that hold one entry per machine (machine name -> value). The file is shared
#: by every machine, so ``save`` writes only the entries this process changed and keeps the others' from disk.
PER_MACHINE_TABLES = (("engines", "machine_paths"), ("dist", "machine_lanes"))
_AS_LOADED = "_per_machine_as_loaded"     # attribute (not a dataclass field: never written) set by load/save


def _machine_tables(s: Settings) -> dict[tuple[str, str], dict]:
    return {(sec, name): {machine_key(k): copy.deepcopy(v) for k, v in getattr(getattr(s, sec), name).items()}
            for sec, name in PER_MACHINE_TABLES}


def _merge_machine_table(disk: dict, before: dict, mine: dict) -> dict:
    """The table to write: the file's entries, with the machines whose entry this process changed (``before`` is
    the table as loaded, ``mine`` as it is now; added, changed or removed) taken from ``mine``."""
    out = {machine_key(k): v for k, v in disk.items()}
    mine = {machine_key(k): v for k, v in mine.items()}
    for key in set(before) | set(mine):
        if before.get(key) != mine.get(key):
            if key in mine:
                out[key] = mine[key]
            else:
                out.pop(key, None)
    return out


def load() -> Settings:
    path = settings_dir() / SETTINGS_FILE
    try:
        s = Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):   # FileNotFoundError included: defaults
        s = Settings()
    setattr(s, _AS_LOADED, _machine_tables(s))
    return s


def save(settings: Settings) -> Path:
    """Write ``settings``. The per-machine tables (``PER_MACHINE_TABLES``) of settings that came from ``load``
    are merged with the file as it is now: an entry this process did not change is taken from the file, so a
    GUI that stayed open for hours does not write back its old copy of another machine's CREST path or lanes.
    Settings made in code (no ``load``) are written as they are."""
    folder = settings_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SETTINGS_FILE
    before = getattr(settings, _AS_LOADED, None)
    if before is not None and path.is_file():
        try:
            disk = _machine_tables(Settings.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            disk = None
        if disk is not None:
            for sec, name in PER_MACHINE_TABLES:
                mine = getattr(getattr(settings, sec), name)
                setattr(getattr(settings, sec), name,
                        _merge_machine_table(disk[(sec, name)], before.get((sec, name), {}), mine))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    setattr(settings, _AS_LOADED, _machine_tables(settings))
    return path
