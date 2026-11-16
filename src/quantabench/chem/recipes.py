"""Jobs as the job dialog submits them (UI-5): a level of theory × a task × resources, and named recipes.

UI-agnostic: the dialog (``gui.job_dialog``) and its tests use these, and every job still runs through the pipeline
(``BuildConfig``), so a job is resumable and its run folder is like any other.

    Task       what to compute with the level of theory: a conformer distribution, an optimization, a single point,
               frequencies + thermochemistry (scan: not yet). Each is a set of ``BuildConfig`` changes.
    Recipe     a named multi-level protocol ("Screen: RDKit conformers → GFN2-xTB → AIMNet2 ranks them"): the
               ``BuildConfig`` fields it sets. Three are built in (``BUILTIN``); the user's own are saved in the
               settings (``Settings.recipes``, synced like every setting) by the dialog's Save as recipe...
    JobSpec    what the dialog submits: a name, the recipe it started from, the task, the configuration (the level of
               theory, conformers, ranking, QM and resources), charge and multiplicity overrides, and where it runs.

Method aliases: a recipe may name ``"gfn2"`` for "whichever GFN2-xTB engine is installed here" (dxtb, tblite or the xtb
program, in ``engines.AUTO_ORDER``'s order), so the same recipe works on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from . import import_options as imp
from .pipeline import BuildConfig

# ============================================================================================ tasks
#: the single-structure tasks: one conformer, refined (or not) at the level of theory, ranked by nothing
_ONE_STRUCTURE = {"max_conformers": 1, "refine_top": 1, "conformer_search": "rdkit", "rank_method": "refine",
                  "frequencies": False}


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    description: str
    changes: dict = field(default_factory=dict)       # BuildConfig fields the task sets over the recipe's
    available: bool = True
    why: str = ""                                     # why it is not available
    keeps_coordinates: bool = False                   # a structure file's 3D coordinates are the starting geometry

    def apply(self, cfg: BuildConfig) -> BuildConfig:
        return replace(cfg, **self.changes)


TASKS: dict[str, Task] = {t.key: t for t in (
    Task("conformers", "Conformer distribution",
         "Search conformers, refine the lowest at the level of theory, rank them and weight them (Boltzmann). The "
         "recipe's conformer search and counts apply."),
    Task("optimize", "Optimize",
         "One structure optimized at the level of theory: a structure file's 3D coordinates when it has them (after "
         "an MMFF94s pre-optimization), else one RDKit conformer.",
         dict(_ONE_STRUCTURE), keeps_coordinates=True),
    Task("single_point", "Single point",
         "The energy and properties at the level of theory, without optimizing at that level. The geometry is the "
         "MMFF94s-relaxed structure (the file's coordinates when it has them); the QM stage, if on, runs on it too.",
         {**_ONE_STRUCTURE, "max_steps": 0, "optimizer": "lbfgs", "batched": "off"}, keeps_coordinates=True),
    Task("frequencies", "Frequencies + thermochemistry",
         "A conformer distribution, then the Hessian of the best conformer: IR spectrum, zero-point energy, H, S and "
         "G at the run's temperature (with the QM stage on: also at the QM level, if its Frequencies box is set).",
         {"frequencies": True}),
    Task("scan", "Scan (not yet)",
         "A relaxed scan of a bond, angle or dihedral.", available=False,
         why="Scans need constrained optimizations, which the refinement does not have yet (UI-5 follow-up)."),
)}
DEFAULT_TASK = "conformers"


# ============================================================================================ levels of theory
#: the families of the dialog's level of theory, in its order (engines.EngineInfo.family; DFT is the QM stage)
FAMILIES = ("force field", "tight binding", "semi-empirical", "ML potential", "DFT")
FAMILY_LABELS = {"force field": "Force field", "tight binding": "Tight binding", "semi-empirical": "Semi-empirical",
                 "ML potential": "ML potential", "DFT": "DFT (QM stage)"}
#: ``"gfn2"`` in a recipe: the first installed of these
GFN2_KEYS = ("gfn2-xtb", "tblite-gfn2", "xtb-gfn2")
ALIASES = {"gfn2": GFN2_KEYS}


@dataclass(frozen=True)
class Level:
    key: str                  # engines key, or "auto"
    label: str
    family: str               # FAMILIES, "" for auto
    available: bool
    why: str = ""             # why it is not available
    gpu: bool = False


def levels(quick: bool = True) -> list[Level]:
    """Every refinement engine, by family (``FAMILIES``), with whether it runs here. ``quick``: the GUI's check, which
    never imports torch (``engines.available``)."""
    from . import engines

    out = [Level("auto", "Auto (the best installed for each molecule's elements)", "", True)]
    infos = engines.all_engines()
    order = {f: i for i, f in enumerate(FAMILIES)}
    for key in sorted(infos, key=lambda k: (order.get(infos[k].family, 99), list(infos).index(k))):
        info = infos[key]
        ok, why = engines.available(key, quick=quick)
        out.append(Level(key, info.label, info.family, bool(ok), "" if ok else str(why), info.gpu_capable))
    return out


def installed_methods(quick: bool = True) -> set[str]:
    return {lv.key for lv in levels(quick) if lv.available}


def resolve_method(method: str, available: set[str] | None = None) -> tuple[str, str]:
    """(engine key, note) for a recipe's method: an alias (``ALIASES``) becomes the first installed engine of it, or
    "auto" with a note when none is installed here. Anything else is returned as it is."""
    keys = ALIASES.get(method)
    if keys is None:
        return method, ""
    avail = installed_methods() if available is None else set(available)
    for k in keys:
        if k in avail:
            return k, ""
    return "auto", f"no GFN2-xTB engine is installed here ({', '.join(keys)}): the refinement is Auto"


# ============================================================================================ recipes
#: the BuildConfig fields a recipe may set (the dialog's level of theory, task and conformer settings; not the
#: resources, the input preparation or the output files)
RECIPE_FIELDS = ("method", "solvent", "conformer_search", "crest_method", "crest_quick", "max_conformers", "refine_top",
                 "energy_window_kcal", "rank_method", "rank_window_kj", "fmax", "max_steps", "optimizer",
                 "xtb_properties", "frequencies", "qm_method", "qm_task", "qm_top", "qm_properties",
                 "qm_polarizability", "qm_frequencies", "qm_cubes")
_DEFAULTS = BuildConfig()


@dataclass(frozen=True)
class Recipe:
    key: str
    name: str
    description: str
    steps: tuple[str, ...]
    changes: dict
    task: str = DEFAULT_TASK
    builtin: bool = True

    def resolved(self, available: set[str] | None = None) -> tuple[dict, list[str]]:
        """The changes with method aliases resolved here, and notes on what was substituted."""
        changes = dict(self.changes)
        notes = []
        if "method" in changes:
            changes["method"], note = resolve_method(changes["method"], available)
            if note:
                notes.append(note)
        return changes, notes

    def config(self, base: BuildConfig, available: set[str] | None = None) -> BuildConfig:
        """``base`` (the resources, solvent, input preparation and files) with the recipe's fields."""
        return replace(base, **self.resolved(available)[0])

    def needs(self, programs: set[str] | None = None) -> list[str]:
        """External programs the recipe uses that are not found here ("crest", "psi4")."""
        need = []
        if self.changes.get("conformer_search") == "crest":
            need.append("crest")
        if self.changes.get("qm_method"):
            need.append("psi4")
        if programs is None:
            programs = _programs()
        return [p for p in need if p not in programs]

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "task": self.task, "changes": dict(self.changes)}


def _programs() -> set[str]:
    from . import crest, qm

    out = set()
    if crest.available()[0]:
        out.add("crest")
    if qm.available()[0]:
        out.add("psi4")
    return out


BUILTIN: tuple[Recipe, ...] = (
    Recipe("screen", "Screen",
           "Many molecules at minutes each: GFN2-xTB refines a modest RDKit conformer search and the rank step orders "
           "the refined conformers (AIMNet2 in the gas phase, g-xTB + CPCM in a solvent).",
           ("RDKit ETKDGv3 + MMFF94s, 30 conformers", "GFN2-xTB on the lowest 10",
            "rank: AIMNet2 (gas) or g-xTB + CPCM (solvent)"),
           {"conformer_search": "rdkit", "max_conformers": 30, "method": "gfn2", "refine_top": 10,
            "rank_method": "auto", "qm_method": "", "frequencies": False}),
    Recipe("standard", "Standard",
           "The default protocol: a full RDKit search, the best installed refinement for each molecule (AIMNet2 "
           "where it has parameters for the elements, else GFN2-xTB), and the rank step.",
           ("RDKit ETKDGv3 + MMFF94s, 50 conformers", "Auto refinement (AIMNet2, else GFN2-xTB) on the lowest 20",
            "rank: auto by phase"),
           {"conformer_search": "rdkit", "max_conformers": 50, "method": "auto", "refine_top": 20,
            "rank_method": "auto", "qm_method": "", "frequencies": False}),
    Recipe("careful", "Careful",
           "Flexible molecules and final numbers: CREST's metadynamics search (GFN-FF, quick: the best search for "
           "flexible neutral molecules in the Nov 14 test), the auto refinement and rank step, then ωB97X-D/def2-SVP "
           "single points of the lowest three conformers. Hours per molecule for large ones.",
           ("CREST iMTD-GC, GFN-FF, quick", "Auto refinement on the lowest 20", "rank: auto by phase",
            "DFT: ωB97X-D/def2-SVP single points of the lowest 3"),
           {"conformer_search": "crest", "crest_method": "gfnff", "crest_quick": True, "max_conformers": 50,
            "method": "auto", "refine_top": 20, "rank_method": "auto", "qm_method": "wb97x-d/def2-svp",
            "qm_task": "energy", "qm_top": 3, "frequencies": False}),
)
BY_KEY = {r.key: r for r in BUILTIN}
CUSTOM = ""          # the dialog's "Custom" entry: no recipe


def _ok_value(name: str, value) -> bool:
    default = getattr(_DEFAULTS, name)
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, type(default))


def recipe_from_dict(data, key: str = "") -> Recipe | None:
    """A user recipe from its settings entry; None for an entry that is not one (hand-edited, from a newer version
    with fields this one lacks are dropped, wrong types are dropped)."""
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"].strip():
        return None
    raw = data.get("changes")
    if not isinstance(raw, dict):
        return None
    changes = {k: v for k, v in raw.items() if k in RECIPE_FIELDS and _ok_value(k, v)}
    task = data.get("task") if data.get("task") in TASKS else DEFAULT_TASK
    name = data["name"].strip()
    desc = data.get("description") if isinstance(data.get("description"), str) else ""
    return Recipe(key or f"user:{name}", name, desc, describe_steps(changes), changes, task=task, builtin=False)


def user_recipes(settings) -> list[Recipe]:
    out, seen = [], set()
    for entry in getattr(settings, "recipes", None) or []:
        r = recipe_from_dict(entry)
        if r is not None and r.key not in seen:
            seen.add(r.key)
            out.append(r)
    return out


def all_recipes(settings=None) -> list[Recipe]:
    return list(BUILTIN) + (user_recipes(settings) if settings is not None else [])


def find(key: str, settings=None) -> Recipe | None:
    return next((r for r in all_recipes(settings) if r.key == key), None)


def recipe_changes(cfg: BuildConfig) -> dict:
    """The recipe fields of ``cfg`` (what Save as recipe... stores): all of them, so the recipe does not depend on
    the defaults of the version that runs it."""
    return {k: getattr(cfg, k) for k in RECIPE_FIELDS}


def save_recipe(settings, name: str, cfg: BuildConfig, task: str = DEFAULT_TASK, description: str = "") -> Recipe:
    """Add (or replace, by name) a user recipe in ``settings.recipes``; the caller saves the settings."""
    name = name.strip()
    if not name:
        raise ValueError("a recipe needs a name")
    if name.lower() in {r.name.lower() for r in BUILTIN}:
        raise ValueError(f"{name!r} is a built-in recipe's name")
    entry = {"name": name, "description": description, "task": task if task in TASKS else DEFAULT_TASK,
             "changes": recipe_changes(cfg)}
    kept = [e for e in (settings.recipes or []) if not (isinstance(e, dict) and str(e.get("name", "")).strip().lower()
                                                         == name.lower())]
    settings.recipes = [*kept, entry]
    return recipe_from_dict(entry)


def delete_recipe(settings, name: str) -> bool:
    before = len(settings.recipes or [])
    settings.recipes = [e for e in (settings.recipes or [])
                        if not (isinstance(e, dict) and str(e.get("name", "")).strip().lower() == name.strip().lower())]
    return len(settings.recipes) < before


def describe_steps(changes: dict) -> tuple[str, ...]:
    """The steps of a recipe's changes, in words (a user recipe's)."""
    c = {**{k: getattr(_DEFAULTS, k) for k in RECIPE_FIELDS}, **changes}
    search = (f"CREST, {c['crest_method']}{', quick' if c['crest_quick'] else ''}" if c["conformer_search"] == "crest"
              else f"RDKit ETKDGv3 + MMFF94s, {c['max_conformers']} conformers")
    steps = [search, f"{method_name(c['method'])} on the lowest {c['refine_top']}"]
    if c["rank_method"] != "refine":
        steps.append(f"rank: {c['rank_method']}")
    if c["qm_method"]:
        steps.append(f"DFT: {c['qm_method']} {'optimizations' if c['qm_task'] == 'optimize' else 'single points'} "
                     f"of the lowest {c['qm_top']}")
    if c["frequencies"]:
        steps.append("frequencies of the best conformer")
    return tuple(steps)


def method_name(key: str) -> str:
    if key == "auto":
        return "Auto refinement"
    if key == "gfn2":
        return "GFN2-xTB"
    from . import engines

    info = engines.all_engines().get(key)
    return info.label.split(" (")[0] if info is not None else key


# ============================================================================================ the job
#: where a job runs: this computer, a worker by name (``dist.client.known_workers``), or "auto" (the fastest free
#: machine for the job: ``dist.remote_job.choose_machine``; DIST-3)
LOCAL = "local"
AUTO = "auto"


@dataclass
class JobSpec:
    name: str = ""
    recipe: str = CUSTOM
    task: str = DEFAULT_TASK
    cfg: BuildConfig = field(default_factory=BuildConfig)
    charge: int | None = None           # None: each molecule's own (from the structure after protonation)
    multiplicity: int | None = None     # None: each molecule's own
    run_on: str = LOCAL

    def config(self) -> BuildConfig:
        """The configuration the job runs with: ``cfg`` with the task's changes."""
        task = TASKS.get(self.task)
        if task is None or not task.available:
            raise ValueError(f"task {self.task!r} cannot run" + (f": {task.why}" if task is not None else ""))
        return task.apply(self.cfg)

    def problems(self) -> list[str]:
        """Why the job cannot be submitted as it is ([] = it can)."""
        out = []
        task = TASKS.get(self.task)
        if task is None:
            out.append(f"unknown task {self.task!r}")
        elif not task.available:
            out.append(f"{task.label}: {task.why}")
        if not str(self.run_on or "").strip():
            out.append("Run on: choose this computer, a worker or Auto")
        if self.multiplicity is not None and self.multiplicity < 1:
            out.append("the multiplicity is 1 (singlet) or more")
        return out

    def summary(self) -> str:
        """One line: level · phase · task (counts)."""
        cfg = self.config() if not self.problems() else self.cfg
        task = TASKS.get(self.task)
        parts = [method_name(cfg.method), cfg.solvent.capitalize() if cfg.solvent else "Gas phase"]
        if self.task == "conformers" or self.task == "frequencies":
            search = "CREST" if cfg.conformer_search == "crest" else f"{cfg.max_conformers} conformers"
            parts.append(f"{task.label.lower()} ({search} → lowest {cfg.refine_top})")
        elif task is not None:
            parts.append(task.label.lower())
        if cfg.qm_method:
            parts.append(f"DFT {cfg.qm_method}")
        if self.charge is not None or self.multiplicity is not None:
            parts.append(f"charge {self.charge if self.charge is not None else 'own'}, "
                         f"multiplicity {self.multiplicity if self.multiplicity is not None else 'own'}")
        if self.run_on == AUTO:
            parts.append("on the fastest free machine")
        elif self.run_on not in (LOCAL, ""):
            parts.append(f"on {self.run_on}")
        return " · ".join(parts)


def job_inputs(inputs, spec: JobSpec) -> list:
    """Copies of ``inputs`` (``MoleculeInput``) for the job: the charge and multiplicity overrides, and for the
    single-structure tasks the structure file's coordinates as the starting geometry."""
    task = TASKS.get(spec.task)
    out = []
    for inp in inputs:
        changes = {}
        if spec.charge is not None:
            changes["charge"] = int(spec.charge)
        if spec.multiplicity is not None:
            changes["multiplicity"] = int(spec.multiplicity)
        if task is not None and task.keeps_coordinates and inp.mol is not None and inp.mol.GetNumConformers():
            changes["meta"] = {**inp.meta, imp.KEEP_COORDS: True}
        out.append(replace(inp, **changes) if changes else inp)
    return out


def config_fields() -> tuple[str, ...]:
    return tuple(f.name for f in fields(BuildConfig))
