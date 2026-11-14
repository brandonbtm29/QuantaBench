"""Project documents (UI-2): one folder that holds a molecule list, its results and the settings they were built with.

    <name>.qbproj/
        project.json      the format and version, the name, when it was created and saved, the build settings
                          (``Settings.build``), the Analyze table's state (``analysis.sheet.TableState``: columns,
                          formula columns, joined files, plot), the workspace and the selected molecule
        molecules.jsonl   one line per molecule, in the list's order: its light row (status, method, conformers,
                          best energy, formula, error, result folder, run folder, input order, posted fields) and,
                          for a molecule added in the window, its input (``jobs.input_to_json``: SMILES or the
                          structure with its coordinates); a molecule of a run folder keeps its input there
        runs/             where the project's builds write: one run folder per batch, as the output folder holds them
        results/          results that existed only in memory (imported outputs, builds with file writing off),
                          written there when the project is saved
        models/           property models of the Analyze table (Model...): <name>.csv, the predictions the table
                          joins, and <name>.json, the spec, metrics and coefficients (``analysis.models``)

Paths inside the project folder are stored relative to it with forward slashes, so a project can be moved, synced
or opened on Windows and macOS alike; paths outside it (a run folder elsewhere, a joined CSV) are stored absolute.
Both files are written to a temporary name and renamed, so a crash while saving leaves the last saved project.

Plain Python (no Qt): ``gui.project_actions`` turns the window's rows into ``ProjectData`` and back.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

FORMAT = "quantabench-project"
VERSION = 1
SUFFIX = ".qbproj"
PROJECT_FILE = "project.json"
MOLECULES_FILE = "molecules.jsonl"
RUNS_DIR = "runs"
RESULTS_DIR = "results"
MODELS_DIR = "models"

#: the light row fields a molecule keeps in molecules.jsonl (``gui.molecule_list.MoleculeRow``'s, less the
#: transient ``stage`` and the ``input`` object)
ROW_FIELDS = ("id", "name", "smiles", "source", "status", "method", "n_conformers", "energy_ev", "formula", "error",
              "folder", "run_dir", "seq", "fields")
PATH_FIELDS = ("folder", "run_dir")


class ProjectError(ValueError):
    """Not a project folder, another format, or a newer version than this QuantaBench reads."""


@dataclass
class ProjectData:
    name: str = ""
    molecules: list[dict] = field(default_factory=list)   # ROW_FIELDS + "input" (JSON text) where there is one
    build: dict = field(default_factory=dict)              # Settings.build as a dict
    table: str = ""                                        # TableState.to_json()
    workspace: str = ""
    current: str = ""                                      # the selected molecule's id
    created: str = ""
    saved: str = ""
    quantabench_version: str = ""
    extra: dict = field(default_factory=dict)              # keys of a newer version, kept on save


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def project_path(path: str | os.PathLike) -> Path:
    """``path`` as a project folder: with the ``.qbproj`` suffix (added when missing)."""
    p = Path(path).expanduser()
    return p if p.suffix.lower() == SUFFIX else p.with_name(p.name + SUFFIX)


def is_project(path: str | os.PathLike) -> bool:
    p = Path(path).expanduser()
    if not (p / PROJECT_FILE).is_file():
        return False
    try:
        return json.loads((p / PROJECT_FILE).read_text(encoding="utf-8")).get("format") == FORMAT
    except (OSError, ValueError, AttributeError):
        return False


def find_project(path: str | os.PathLike) -> Path | None:
    """The project folder ``path`` is, is in (its project.json, a run folder of it), or None."""
    p = Path(path).expanduser()
    if p.is_file():
        p = p.parent
    for q in (p, *p.parents):
        if is_project(q):
            return q
    return None


# ------------------------------------------------------------------------------------------------- paths
def to_stored(path: str, root: Path) -> str:
    """A path for the project files: relative to ``root`` with "/" when inside it, else absolute as given."""
    if not path:
        return ""
    p = Path(path).expanduser()
    try:
        rel = p.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return str(p)
    return rel.as_posix() if str(rel) != "." else "."


def from_stored(text: str, root: Path) -> str:
    """The path a stored path means on this machine: a relative one ("/" or "\\" separators) under ``root``."""
    if not text:
        return ""
    if PureWindowsPath(text).is_absolute() or PurePosixPath(text).is_absolute() or text.startswith("~"):
        return str(Path(text).expanduser())
    parts = [x for x in text.replace("\\", "/").split("/") if x not in ("", ".")]
    return str(root.joinpath(*parts)) if parts else str(root)


def _plain(v):
    """A JSON value for a posted field: NumPy scalars as Python numbers, NaN and infinities as None, other objects
    as text."""
    if v is None or isinstance(v, (bool, str, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except (TypeError, ValueError):
            pass
    return str(v)


# ------------------------------------------------------------------------------------------------- save, load
def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def save(folder: str | os.PathLike, data: ProjectData) -> Path:
    """Write ``data`` to the project folder ``folder`` (created, with its runs/ folder, when new). Paths in the
    rows and in the table's joined files are stored relative to the folder where they are inside it."""
    from . import __version__

    root = project_path(folder)
    root.mkdir(parents=True, exist_ok=True)
    (root / RUNS_DIR).mkdir(exist_ok=True)
    lines = []
    for m in data.molecules:
        d = {k: m.get(k) for k in ROW_FIELDS}
        for k in PATH_FIELDS:
            d[k] = to_stored(d.get(k) or "", root)
        if d.get("fields"):
            d["fields"] = {str(k): _plain(v) for k, v in d["fields"].items()}
        if m.get("input"):
            d["input"] = m["input"]
        lines.append(json.dumps(d, ensure_ascii=False, sort_keys=True))
    table = data.table
    if table:
        try:
            t = json.loads(table)
            for j in t.get("joins") or ():
                if isinstance(j, dict) and isinstance(j.get("path"), str):
                    j["path"] = to_stored(j["path"], root)
            table = json.dumps(t, sort_keys=True)
        except (ValueError, AttributeError):
            pass
    head = {**data.extra, "format": FORMAT, "version": VERSION, "name": data.name or root.stem,
            "created": data.created or _now(), "saved": _now(), "quantabench_version": __version__,
            "build": data.build, "table": table, "workspace": data.workspace, "current": data.current,
            "n_molecules": len(lines)}
    _write_atomic(root / MOLECULES_FILE, "\n".join(lines) + ("\n" if lines else ""))
    _write_atomic(root / PROJECT_FILE, json.dumps(head, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return root


def load(folder: str | os.PathLike) -> ProjectData:
    """Read the project folder ``folder`` (``ProjectError`` when it is not one QuantaBench reads)."""
    root = Path(folder).expanduser()
    try:
        head = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ProjectError(f"{root} is not a QuantaBench project (no {PROJECT_FILE})") from None
    except (OSError, ValueError) as exc:
        raise ProjectError(f"{root / PROJECT_FILE} cannot be read: {exc}") from exc
    if not isinstance(head, dict) or head.get("format") != FORMAT:
        raise ProjectError(f"{root} is not a QuantaBench project ({PROJECT_FILE} has another format)")
    if int(head.get("version") or 0) > VERSION:
        raise ProjectError(f"{root} was saved by a newer QuantaBench (project version {head.get('version')}; this "
                           f"one reads {VERSION}): update QuantaBench to open it")
    molecules = []
    path = root / MOLECULES_FILE
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except ValueError as exc:
                    raise ProjectError(f"{path}, line {n}: {exc}") from exc
                if not isinstance(d, dict) or not d.get("id"):
                    continue
                for k in PATH_FIELDS:
                    d[k] = from_stored(d.get(k) or "", root)
                molecules.append(d)
    table = head.get("table") or ""
    if table:
        try:
            t = json.loads(table)
            for j in t.get("joins") or ():
                if isinstance(j, dict) and isinstance(j.get("path"), str):
                    j["path"] = from_stored(j["path"], root)
            table = json.dumps(t, sort_keys=True)
        except (ValueError, AttributeError):
            table = ""
    known = {"format", "version", "name", "created", "saved", "quantabench_version", "build", "table", "workspace",
             "current", "n_molecules"}
    return ProjectData(name=str(head.get("name") or root.stem), molecules=molecules,
                       build=head.get("build") if isinstance(head.get("build"), dict) else {}, table=table,
                       workspace=str(head.get("workspace") or ""), current=str(head.get("current") or ""),
                       created=str(head.get("created") or ""), saved=str(head.get("saved") or ""),
                       quantabench_version=str(head.get("quantabench_version") or ""),
                       extra={k: v for k, v in head.items() if k not in known})


def runs_dir(folder: str | os.PathLike) -> Path:
    return project_path(folder) / RUNS_DIR


def results_dir(folder: str | os.PathLike) -> Path:
    return project_path(folder) / RESULTS_DIR


def models_dir(folder: str | os.PathLike) -> Path:
    return project_path(folder) / MODELS_DIR
