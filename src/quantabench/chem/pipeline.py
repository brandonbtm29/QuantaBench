"""The geometry builder pipeline.

    input (SMILES / name / file / CSV)
      -> standardize (largest fragment, neutral parent)
      -> protonation state at pH
      -> 3D embedding, conformer search: ETKDGv3 + MMFF94s pre-optimization, or CREST iMTD-GC (``chem.crest``)
      -> energy-window + RMSD pruning
      -> refinement of the lowest conformers with the chosen engine (AIMNet2 / xTB / tblite / MOPAC / MMFF),
         in an implicit solvent when the engine has one; GPU when possible; all conformers in one torch batch
         on a GPU (``chem.batch_opt``), one at a time with ASE (or the program's own optimizer) otherwise
      -> re-pruning, Boltzmann weights, descriptors, the engine's own properties (charges, dipole, ...)
      -> optional: frequencies, IR and RRHO/qRRHO thermochemistry of the best conformer
      -> optional QM stage (``chem.qm``): DFT single point / optimization of the lowest conformers with Psi4,
         properties (dipole, charges, MBIS, orbitals), density/ESP cubes and surface descriptors
      -> export (XYZ, SDF ensemble, extXYZ with charges, PDB, MOL, LAMMPS data, cubes) + batch tables

The pipeline is UI-agnostic: the CLI and the GUI both call ``run`` with a progress callback and a
cancel check.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem

from .. import priority, resources
from ..settings import Settings
from . import batch_opt, engines, estimates, export, properties, telemetry
from . import conformers as confs
from . import descriptors as desc
from . import import_options as imp
from .inputs import ID_NOTE, MoleculeInput
from .optimize import Cancelled, optimize
from .standardize import UnsupportedStructure, formal_charge, inchikey, prepare, spin_multiplicity

KCAL_PER_EV = confs.KCAL_PER_EV


@dataclass
class BuildConfig:
    method: str = "auto"
    ph: float = 7.0
    protonate: bool = True
    protonation_method: str = "rules"     # "rules" | "dimorphite"
    keep_largest_fragment: bool = True
    max_conformers: int = 50
    refine_top: int = 20                  # refine at most this many force-field conformers
    # the refinement level's window: the force-field pool keeps this + 4 kcal/mol, and with rank_method "refine" the
    # refined conformers are pruned with it; a rank step prunes with rank_window_kj instead (chem.rank)
    energy_window_kcal: float = 6.0
    # the level that ranks the refined conformers (chem.rank.RANK_METHODS): "auto" (by phase), "refine" (the
    # refinement's energies, as before RANK-2), "gxtb+cpcm", "gxtb", "aimnet2", "aimnet2+cpcm", "qm"
    rank_method: str = "auto"
    rank_window_kj: float = 30.0          # the rank step keeps conformers within this of its lowest (kJ/mol)
    rmsd_threshold: float = 0.5
    fmax: float = 0.03
    max_steps: int = 500
    optimizer: str = "lbfgs"              # "lbfgs" | "fire" | "native" (the engine's own optimizer, if it has one)
    temperature_k: float = 298.15
    xtb_properties: bool = True           # GFN1-xTB charges, dipole, polarizability of the best conformer
    batched: str = "auto"                 # "auto" (batched on a GPU) | "on" | "off": see use_batched()
    frequencies: bool = False             # Hessian, IR and thermochemistry of the best conformer
    freq_hessian: str = "auto"            # "auto" | "analytic" | "finite" (vibrations.analyze)
    symmetry_number: int = 0              # rotational symmetry number σ for the entropy; 0 = detect per molecule
    freq_qrrho: bool = True               # Grimme's quasi-RRHO entropy for low-frequency modes
    # imaginary modes below -20 cm-1 at the first Hessian: tighten to fmax 0.002 eV/Å, then step along the lowest
    # one both ways and re-optimize, at most this many steps (vibrations.analyze imag_retries); 0 = report only
    freq_imag_retries: int = 3
    seed: int = 42
    solvent: str = ""                     # implicit solvent (chem.solvents name), "" = gas phase
    conformer_search: str = "rdkit"       # "rdkit" (ETKDGv3 + MMFF94s) | "crest" (CREST iMTD-GC, external)
    crest_method: str = "gfn2"            # "gfn2" | "gfnff" | "gfn2//gfnff" | "gfn1"
    crest_quick: bool = True              # CREST --quick (fewer metadynamics runs): minutes instead of hours
    # CREST missing or failed: "rdkit" = RDKit ETKDGv3 + MMFF94s, flagged (note, conformer_fallback, counts at the
    # end of the batch) | "fail" = the molecule fails
    crest_fallback: str = "rdkit"
    # threads of this molecule's CREST call, 0 = the molecule's threads. Set per molecule by a parallel run that
    # lends it the threads of idle lanes (dist.threads); not a setting
    crest_threads: int = 0
    qm_method: str = ""                 # "functional/basis", e.g. "wb97x-d/def2-svp"; "" = no QM stage
    qm_task: str = "energy"               # "energy" (single point) | "optimize"
    qm_top: int = 1                       # how many of the lowest conformers get the QM stage
    qm_properties: bool = True            # dipole, quadrupole, Mulliken/Lowdin/MBIS charges, orbital energies
    qm_frequencies: bool = False          # QM Hessian + IR + thermochemistry of the best QM conformer
    qm_cubes: bool = False                # density + ESP cubes and molecular-surface descriptors (best conformer)
    qm_program: str = "psi4"
    threads: int = 0                      # 0 = auto: the logical threads minus free_threads (quantabench.priority)
    # this machine's run priority ("normal" | "low" | "background") and the logical threads the automatic budgets
    # leave free (quantabench.priority; Settings.priority_for / free_threads_for). Resources, not science
    priority: str = "low"
    free_threads: int = 1
    gpu: str = "auto"
    gpu_index: int = 0
    gpu_memory_fraction: float = 0.9
    linalg: str = "auto"                  # torch_compat linear-algebra policy: "auto" | "cpu" | "device"
    dxtb_gpu: str = "auto"                # dxtb on a GPU: "auto" (where it is faster) | "on" | "off"
    output_dir: str = ""
    formats: tuple[str, ...] = export.DEFAULT_FORMATS
    write_files: bool = True
    # memory budget (GB) of one process for the dxtb steps (chem.memory_guard): 0 = auto (available memory
    # / lanes, resolved when a run starts), < 0 = no guard
    memory_budget_gb: float = 0.0
    molecule_timeout_s: float = 0.0       # wall-clock limit per molecule in a batch (0 = none); see time_limited()
    # areas and volumes (chem.descriptors.SurfaceOptions): radii set for sasa_A2/vdw_* ("bondi" | "spartan"),
    # per-element overrides (symbol -> Å) and the SASA probe radius; cpk_* always use Spartan's table
    radii_set: str = "bondi"
    radii_overrides: dict = field(default_factory=dict)
    sasa_probe_A: float = 1.4

    @classmethod
    def from_settings(cls, s: Settings, **overrides) -> BuildConfig:
        b, r = s.build, s.resources
        cfg = cls(method=b.refine_method, ph=b.ph, protonate=b.protonate, keep_largest_fragment=b.keep_largest_fragment,
                  max_conformers=b.max_conformers, energy_window_kcal=b.energy_window_kcal,
                  rank_method=b.rank_method, rank_window_kj=b.rank_window_kj,
                  rmsd_threshold=b.rmsd_threshold, fmax=b.fmax, max_steps=b.max_steps, temperature_k=b.temperature_k,
                  seed=b.random_seed, batched=b.batched, frequencies=b.frequencies, threads=r.threads,
                  priority=s.priority_for(), free_threads=s.free_threads_for(), gpu=r.gpu, gpu_index=r.gpu_index,
                  gpu_memory_fraction=r.gpu_memory_fraction, linalg=r.linalg, dxtb_gpu=r.dxtb_gpu,
                  memory_budget_gb=r.memory_budget_gb, molecule_timeout_s=r.molecule_timeout_s,
                  output_dir=str(s.output_path()),
                  conformer_search=b.conformer_search, solvent=b.solvent, qm_method=b.qm_method,
                  radii_set=s.surfaces.radii_set, radii_overrides=dict(s.surfaces.radii_overrides),
                  sasa_probe_A=s.surfaces.sasa_probe_A)
        return replace(cfg, **overrides)


@dataclass
class ConformerResult:
    rank: int
    conf_id: int
    energy_ev: float
    rel_kcal: float = 0.0
    weight: float = 0.0
    ff_energy_kcal: float | None = None
    converged: bool = True
    steps: int = 0
    charges: list[float] | None = None
    dipole_debye: float | None = None
    # the rank step (chem.rank): this conformer's energy at the rank level (eV; None when the refinement ranked) and
    # its refinement energy relative to the refinement's lowest of the pool (kcal/mol). energy_ev stays the
    # refinement's; rel_kcal and weight follow the rank level
    rank_energy_ev: float | None = None
    refine_rel_kcal: float | None = None

    @property
    def ranking_energy_ev(self) -> float:
        """The energy that orders the conformers: the rank level's, else the refinement's."""
        return self.energy_ev if self.rank_energy_ev is None else self.rank_energy_ev


@dataclass
class MoleculeResult:
    input: MoleculeInput
    status: str = "pending"            # "ok" | "failed" | "cancelled"
    error: str = ""
    smiles: str = ""
    charge: int = 0
    multiplicity: int = 1
    method: str = ""
    device: str = "cpu"
    forcefield: str = ""
    n_embedded: int = 0
    conformers: list[ConformerResult] = field(default_factory=list)
    mol: Chem.Mol | None = None
    notes: list[str] = field(default_factory=list)
    descriptors: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    vibrations: dict | None = None     # VibrationalResult.to_dict(include_modes=True) when cfg.frequencies
    solvent: str = ""                  # implicit solvent actually used by the refinement ("" = gas phase)
    conformer_search: str = ""         # "rdkit" | "crest" (what actually produced the conformers)
    conformer_fallback: str = ""       # why they are not from the search asked for ("CREST failed: ..."), "" = they are
    qm: dict | None = None            # chem.qm stage results (see qm.QM_RESULT_KEYS)
    input_charge: int | None = None    # total formal charge of the record as drawn (all fragments)
    removed_fragments: list[str] = field(default_factory=list)   # SMILES of fragments standardization removed
    # structure keys, for joining results to other tables (Spartan, a gas/water pair): the computed species
    # after preparation, and the input record as drawn (all fragments); "" with a note when InChI fails
    inchikey: str = ""
    canonical_smiles: str = ""
    input_inchikey: str = ""
    phase: str = ""                    # the phase the run asked for: "gas" or the solvent (BuildConfig.solvent)
    solvation_model: str = ""          # model of ``solvent`` ("alpb", "cosmo", ...); "" = gas phase
    phase_steps: dict = field(default_factory=dict)   # step -> where it was computed ("ALPB(water)", "gas (why)")
    # the rank step (chem.rank.record): the level asked for and used, the pool with every conformer's refinement and
    # rank energies, the pick and whether it changed; None before the ranking (and in results from before RANK-2)
    rank: dict | None = None
    # the stage timings, threads and CPU time of this attempt (chem.telemetry.StageClock), for the job table's
    # telemetry rows; not a result, so not in to_dict() and not in the molecule's files
    telemetry: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("mol", "input", "telemetry")}
        d["input"] = {k: v for k, v in asdict(self.input).items() if k != "mol"}
        return d

    def phase_notes(self) -> str:
        """'refine: ALPB(water); xtb_properties: ALPB(water), tblite GFN1-xTB; ...', or 'gas' for a gas-phase run."""
        if not self.phase_steps:
            return "gas" if self.phase in ("", "gas") else ""
        return "; ".join(f"{k}: {v}" for k, v in self.phase_steps.items())

    def summary_row(self) -> dict:
        best = self.conformers[0] if self.conformers else None
        return {
            "id": self.input.id, "name": self.input.name, "status": self.status, "error": self.error,
            "input_smiles": self.input.smiles, "smiles": self.smiles,
            "inchikey": self.inchikey, "canonical_smiles": self.canonical_smiles, "input_inchikey": self.input_inchikey,
            "input_charge": "" if self.input_charge is None else self.input_charge, "charge": self.charge,
            "multiplicity": self.multiplicity, "removed_fragments": ".".join(self.removed_fragments),
            "method": self.method, "forcefield": self.forcefield,
            "device": self.device, "phase": self.phase, "solvent": self.solvent,
            "solvation_model": self.solvation_model, "phase_notes": self.phase_notes(),
            "conformer_search": self.conformer_search, "conformer_fallback": self.conformer_fallback,
            "n_embedded": self.n_embedded, "n_conformers": len(self.conformers),
            "best_energy_eV": best.energy_ev if best else "", "best_converged": best.converged if best else "",
            "dipole_debye": best.dipole_debye if best and best.dipole_debye is not None else "",
            "qm_method": (self.qm or {}).get("level", ""), "qm_energy_eV": (self.qm or {}).get("energy_ev", ""),
            **self._rank_fields(best),
            "seconds": round(sum(self.timings.values()), 2), "notes": " | ".join(self.notes),
        } | self._posted_fields()

    def _rank_fields(self, best: ConformerResult | None) -> dict:
        """The rank step's summary columns: the method asked for, the level used, the best conformer's rank energy,
        whether the pick differs from the refinement's, and the conformers before the rank window."""
        r = self.rank or {}
        return {"rank_method": r.get("method", ""), "rank_level": r.get("level", ""),
                "rank_energy_eV": best.rank_energy_ev if best is not None and best.rank_energy_ev is not None else "",
                "rank_pick_changed": r.get("pick_changed", "") if r.get("level", "refine") != "refine" else "",
                "rank_n_pool": r.get("n_pool", "") if r else ""}

    def _posted_fields(self) -> dict:
        """The input's fields posted on import (``import_options.FIELDS``: SD properties, table columns) as extra
        summary columns; a name that is already a summary column gets a ``file_`` prefix."""
        posted = (self.input.meta or {}).get(imp.FIELDS) or {}
        out: dict = {}
        for k, v in posted.items() if isinstance(posted, dict) else ():
            key = f"file_{k}" if k in SUMMARY_COLUMNS else str(k)
            out[key] = v
        return out


#: the standard summary.csv columns (posted import fields with one of these names get a ``file_`` prefix)
SUMMARY_COLUMNS = frozenset(
    ("id name status error input_smiles smiles inchikey canonical_smiles input_inchikey input_charge charge "
     "multiplicity removed_fragments method forcefield device phase solvent solvation_model phase_notes "
     "conformer_search conformer_fallback n_embedded n_conformers best_energy_eV best_converged dipole_debye "
     "qm_method qm_energy_eV rank_method rank_level rank_energy_eV rank_pick_changed rank_n_pool seconds notes").split())

ProgressFn = Callable[[int, int, str, str], None]   # (molecule index, total, stage, message)
BATCHED_MODES = ("auto", "on", "off")


def use_batched(cfg: BuildConfig, method: str, device: str) -> bool:
    """Whether ``method`` on ``device`` refines all conformers in one batch (``batch_opt.refine_batched``).

    'auto' batches only on a GPU, where one call for B conformers costs about as much as one call for
    one; on a CPU the sequential ASE path stays the default (batched L-BFGS gives identical results and
    was 2.4x faster on 2 CPUs for PFOA x 8, but it holds all conformers in memory at once). 'on' batches
    on any device, 'off' never. Methods without a batched adapter (MMFF) always run sequentially.
    """
    if cfg.batched not in BATCHED_MODES:
        raise ValueError(f"batched must be one of {BATCHED_MODES}, not {cfg.batched!r}")
    if cfg.batched == "off" or not batch_opt.supports_batched(method):
        return False
    return cfg.batched == "on" or device != "cpu"


def _noop(*_args) -> None:
    pass


DXTB_METHODS = ("gfn1-xtb", "gfn2-xtb")


def memory_budget(cfg: BuildConfig) -> float:
    """The memory budget (GB) of this process for the dxtb steps: ``cfg.memory_budget_gb`` as resolved for the
    run (``memory_guard.resolve_budget_gb``; a run resolves 0 = auto before it starts), 0 = guard off."""
    from . import memory_guard

    return memory_guard.resolve_budget_gb(cfg.memory_budget_gb, 1)


def resolved_budget(cfg: BuildConfig, n_lanes: int, log: Callable[[str], None] = _noop) -> float:
    """``cfg.memory_budget_gb`` resolved for a run of ``n_lanes`` processes (auto: available memory / lanes,
    at least ``memory_guard.FLOOR_GB``), to store in the config the molecules run with; -1 = guard off.
    Logs the budget once when a dxtb step can run."""
    from . import memory_guard

    budget = memory_guard.resolve_budget_gb(cfg.memory_budget_gb, n_lanes)
    if cfg.xtb_properties or cfg.method in (*DXTB_METHODS, "auto"):
        if budget > 0:
            how = "set" if cfg.memory_budget_gb > 0 else f"auto: available memory / {max(1, n_lanes)} lane(s)"
            log(f"Memory guard: {budget:.1f} GB per process for the dxtb steps ({how}; --memory-budget changes it)")
        else:
            log("Memory guard: off (--memory-budget < 0)")
    return budget if budget > 0 else -1.0


def guard_refinement(result: MoleculeResult, numbers, cfg: BuildConfig) -> None:
    """Raise ``MemoryGuardError`` when one dxtb energy + forces evaluation of this molecule would not fit the
    budget. Method "auto" then tries its next engine (with a note); an explicit method fails the molecule with
    the reason. There is no cheaper dxtb path to the same forces (``memory_guard`` module docstring)."""
    if result.method not in DXTB_METHODS:
        return
    from . import memory_guard

    ok, why = memory_guard.fits(result.method, numbers, result.charge, result.multiplicity, memory_budget(cfg))
    if not ok:
        raise memory_guard.MemoryGuardError(f"{engines.ENGINES[result.method].label} refinement not started by the "
                                            f"memory guard: {why}")


def time_limit_error(limit_s: float, stage: str = "") -> str:
    """The error of a molecule that ran past the per-molecule time limit: 'time limit (1800 s) in refine'."""
    return f"time limit ({float(limit_s):g} s)" + (f" in {stage}" if stage else "")


class TimeLimit:
    """``should_stop`` for one molecule with a wall-clock limit (``BuildConfig.molecule_timeout_s``), for the
    single-process loops (``run``, ``jobs.run_resumable`` with one lane). Best effort: the molecule ends at the
    pipeline's next ``should_stop`` checkpoint (every optimizer step, every conformer, between stages, the
    poll of an external program), not inside a step without one (RDKit embedding, one dxtb SCF, the xTB
    properties). Lanes enforce the limit from the parent instead, by killing the lane (``dist.lanes``)."""

    def __init__(self, should_stop: Callable[[], bool], limit_s: float):
        self.should_stop = should_stop
        self.limit = max(0.0, float(limit_s or 0.0))
        self.t0 = time.perf_counter()
        self.stage = ""
        self.hit = False

    def __call__(self) -> bool:
        if self.should_stop():
            return True
        if self.limit and time.perf_counter() - self.t0 > self.limit:
            self.hit = True
            return True
        return False

    def wrap_stage(self, stage: Callable[[str], None]) -> Callable[[str], None]:
        def inner(s: str) -> None:
            self.stage = s
            stage(s)
        return inner

    def finish(self, res: MoleculeResult, log: Callable[[str], None] = _noop) -> MoleculeResult:
        """A molecule stopped by the limit (not by a cancel) becomes failed with the time-limit error."""
        if self.hit and res.status == "cancelled" and not self.should_stop():
            res.status, res.error = "failed", time_limit_error(self.limit, self.stage)
            log(f"  FAILED: {res.error} (stopped at the next checkpoint after {time.perf_counter() - self.t0:.0f} s)")
        return res


def run_molecule_limited(inp: MoleculeInput, cfg: BuildConfig, device: str, threads: int,
                         should_stop: Callable[[], bool], log: Callable[[str], None],
                         stage: Callable[[str], None] = _noop) -> MoleculeResult:
    """``run_molecule`` under ``cfg.molecule_timeout_s`` (``TimeLimit``; no limit when it is 0)."""
    if not cfg.molecule_timeout_s or cfg.molecule_timeout_s <= 0:
        return run_molecule(inp, cfg, device, threads, should_stop, log, stage=stage)
    tl = TimeLimit(should_stop, cfg.molecule_timeout_s)
    return tl.finish(run_molecule(inp, cfg, device, threads, tl, log, stage=tl.wrap_stage(stage)), log)


def _properties_plan(numbers, charge: int, multiplicity: int, cfg: BuildConfig) -> tuple[bool, str, str]:
    """``memory_guard.properties_plan`` for the GFN1-xTB properties step of the best conformer."""
    from ..resources import has_package
    from . import memory_guard

    return memory_guard.properties_plan(numbers, charge, multiplicity, memory_budget(cfg), polarizability=True,
                                        dxtb_polarizability=engines.has_libcint(), tblite=has_package("tblite"))


def calculator_options(result: MoleculeResult, cfg: BuildConfig) -> dict:
    """Options every engine factory receives (see ``engines`` module docstring)."""
    return {"solvent": result.solvent, "threads": priority.budget_of(cfg)}


def surface_options(cfg: BuildConfig) -> desc.SurfaceOptions:
    """The areas' and volumes' radii and probe of a run (``descriptors.SurfaceOptions``)."""
    return desc.SurfaceOptions.make(cfg.radii_set, cfg.radii_overrides, cfg.sasa_probe_A)


def _engine_dipole(atoms, calc, origin: str | None, charge: int, masses: np.ndarray) -> np.ndarray | None:
    """The engine's electronic dipole (e·Å) about the centre of mass at ``atoms``' geometry, or None when the
    engine does not provide one (``origin`` None) or it is missing. ``origin`` is where the engine puts it."""
    if origin == "com":        # xtb: its last run's result (a native optimizer may not leave calc's state)
        mu = calc.results.get("dipole")
    elif origin == "origin":
        try:                   # cached at the final geometry; recomputed if the atoms moved since
            mu = atoms.get_dipole_moment()
        except Exception:      # noqa: BLE001 - not implemented by this calculator: its last result, if any
            mu = calc.results.get("dipole")
    else:
        return None
    if mu is None:
        return None
    mu = np.asarray(mu, float).reshape(3)
    if origin == "origin" and charge:
        pos = atoms.get_positions()
        mu = mu - float(charge) * (pos * masses[:, None]).sum(0) / masses.sum()
    return mu


def _refine(result: MoleculeResult, conf_set: confs.ConformerSet, cfg: BuildConfig, device: str,
            should_stop: Callable[[], bool], log: Callable[[str], None]) -> list[ConformerResult]:
    mol = conf_set.mol
    ids = conf_set.conf_ids[: max(1, cfg.refine_top)]
    ff = conf_set.energies_kcal[: len(ids)]
    if result.method == "mmff":
        return [ConformerResult(rank=0, conf_id=cid, energy_ev=e / KCAL_PER_EV, ff_energy_kcal=e)
                for cid, e in zip(ids, ff)]
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    options = calculator_options(result, cfg)
    native = engines.native_optimizer(result.method) if cfg.optimizer == "native" else None

    plug = engines.plugin(result.method)
    # "com": calc's dipole is about the centre of mass (xtb); "origin": about the coordinate origin (tblite, ASE's
    # convention), moved to the centre of mass here; neither: the point-charge dipole of the engine's charges
    dipole_origin = plug.extra.get("dipole_origin") if plug is not None else None

    guard_refinement(result, numbers, cfg)

    def build(dev):
        extra = {"should_stop": should_stop} if plug is not None else {}
        return engines.make_calculator(result.method, numbers, result.charge, result.multiplicity, dev,
                                       **options, **extra)

    def run_opt(atoms):
        if native is not None:
            return native(atoms, cfg.fmax, cfg.max_steps, should_stop, log, charge=result.charge,
                          multiplicity=result.multiplicity, **options)
        return optimize(atoms, cfg.fmax, cfg.max_steps, "lbfgs" if cfg.optimizer == "native" else cfg.optimizer,
                        should_stop)

    calc = build(device)
    out = []
    for i, (cid, e_ff) in enumerate(zip(ids, ff)):
        if should_stop():
            raise Cancelled()
        atoms = export.mol_to_atoms(mol, cid)
        atoms.calc = calc
        try:
            res = run_opt(atoms)
        except Cancelled:
            raise
        except Exception as exc:
            if result.device == "cpu":
                raise
            result.notes.append(f"{result.method} failed on {result.device} ({type(exc).__name__}: {exc}); "
                                "fell back to CPU")
            log(result.notes[-1])
            result.device = "cpu"
            calc = build("cpu")
            atoms = export.mol_to_atoms(mol, cid)
            atoms.calc = calc
            res = run_opt(atoms)
        conf = mol.GetConformer(cid)
        for j, p in enumerate(atoms.get_positions()):
            conf.SetAtomPosition(j, p.tolist())
        charges = calc.results.get("charges")
        charges = None if charges is None else [float(x) for x in np.ravel(charges)]
        mu = _engine_dipole(atoms, calc, dipole_origin, result.charge, masses)
        if mu is not None:     # the engine's own (electronic) dipole, about the centre of mass
            dip = float(np.linalg.norm(mu)) * desc.DEBYE_PER_E_ANG
        else:                  # point-charge dipole
            dip = desc.dipole_debye(atoms.get_positions(), np.asarray(charges), masses) if charges else None
        out.append(ConformerResult(rank=0, conf_id=cid, energy_ev=res.energy_ev, ff_energy_kcal=e_ff,
                                   converged=res.converged, steps=res.steps, charges=charges, dipole_debye=dip))
        log(f"  conformer {i + 1}/{len(ids)}: E = {res.energy_ev:.5f} eV, {res.steps} steps"
            + ("" if res.converged else " (not converged)"))
    return out


def _frequencies(result: MoleculeResult, conf_set: confs.ConformerSet, best: ConformerResult, cfg: BuildConfig,
                 should_stop: Callable[[], bool], log: Callable[[str], None]) -> bool:
    """Vibrational analysis of the best conformer; a failure becomes a note (and False), not a failed molecule.
    Plugin engines (tblite, xtb, MOPAC) get the refinement's solvent: the Hessian (finite differences of the
    forces), the tight re-optimization and the energy in the thermochemistry all include the continuum."""
    from . import vibrations

    atoms = export.mol_to_atoms(conf_set.mol, best.conf_id)
    stored_positions = atoms.get_positions().copy()
    fmax = min(cfg.fmax, 0.01)    # frequencies need a tighter minimum than the conformer ranking
    calc = None
    if engines.plugin(result.method) is not None:   # plugin engines: pass the solvent/threads options through
        options = calculator_options(result, cfg)
        calc = engines.make_calculator(result.method, atoms.numbers, result.charge, result.multiplicity,
                                       result.device, should_stop=should_stop, **options)
        native = engines.native_optimizer(result.method) if cfg.optimizer == "native" else None
        if native is not None:   # the program's own optimizer is much faster than ASE driving it step by step
            atoms.calc = calc
            try:
                native(atoms, fmax, cfg.max_steps, should_stop, log, charge=result.charge,
                       multiplicity=result.multiplicity, **options)
            except Cancelled:
                raise
            except Exception as exc:
                log(f"  native pre-optimization failed ({type(exc).__name__}); ASE will pre-optimize")
    try:
        vr = vibrations.analyze(atoms, result.method, result.charge, result.multiplicity, device=result.device, calc=calc,
                                temperature=cfg.temperature_k, symmetry_number=cfg.symmetry_number,
                                hessian=cfg.freq_hessian, mol=conf_set.mol, qrrho=cfg.freq_qrrho,
                                preoptimize_fmax=fmax, imag_retries=cfg.freq_imag_retries,
                                should_stop=should_stop, log=log)
    except Cancelled:
        raise
    except Exception as exc:
        result.notes.append(f"frequencies skipped ({type(exc).__name__}: {exc})")
        log(f"  {result.notes[-1]}")
        return False
    # The tight re-optimization moves the atoms (up to a few tenths of an Å on flat perfluoroalkyl torsions).
    # The stored conformer is left alone, so its ranking energy, charges, dipole and 3D descriptors still
    # belong to its exported geometry; the frequency geometry is kept separately (``positions_A``, written
    # as <id>_freq.xyz by the spectrum export) together with its own energy (thermo.potential_energy_ev).
    result.vibrations = vr.to_dict(include_modes=True)   # displacements: the spectra panel animates modes
    result.descriptors.update({f"vib_{k}": v for k, v in vr.descriptors().items()})
    if vr.positions is not None:
        pos = np.asarray(vr.positions, float)
        result.vibrations["symbols"] = atoms.get_chemical_symbols()
        result.vibrations["positions_A"] = np.round(pos, 6).tolist()
        shift = float(np.sqrt(((pos - stored_positions) ** 2).sum(1)).max()) if len(pos) else 0.0
        result.vibrations["geometry_max_shift_A"] = shift
        result.descriptors["vib_geometry_max_shift_A"] = shift
    result.notes += [f"frequencies: {w}" for w in vr.warnings]
    if engines.all_engines()[result.method].family == "semi-empirical":
        result.notes.append("frequencies: MOPAC's energy is the heat of formation, which already includes ZPE and "
                            "298 K thermal terms; use vib_zpe/vib_h_corr/vib_g_corr, not the absolute H and G")
    t = vr.thermo
    log(f"  frequencies ({vr.hessian_method} Hessian): {vr.n_modes} modes, {vr.n_imaginary} imaginary"
        + (f", ZPE {t.zpe_kcal:.2f} kcal/mol, S {t.entropy_cal_molK:.2f} cal/(mol K)" if t else ""))
    return True


def solvation_model(method: str, solvent: str) -> str:
    """The implicit-solvent model ``method`` uses for ``solvent`` by default ("alpb", "cosmo"; "" = gas phase):
    the first of its ``EngineInfo.solvation`` models with parameters for that solvent."""
    if not solvent:
        return ""
    from . import solvents

    info = engines.all_engines().get(method)
    for model in (info.solvation if info else ()):
        if solvents.for_engine(solvent, model) is not None:
            return model
    return ""


def solvation_label(method: str, solvent: str) -> str:
    """'ALPB(water)' for what ``method`` computes in ``solvent``, 'gas' in the gas phase."""
    model = solvation_model(method, solvent)
    return f"{model.upper()}({solvent})" if model else "gas"


def _gas_reason(method: str, solvent: str) -> str:
    return f"gas ({engines.all_engines()[method].label} has no implicit solvent model for {solvent})"


def _solvent_for(method: str, solvent: str, notes: list[str]) -> str:
    """The solvent ``method`` will actually use: ``solvent`` if it can model it, else gas phase (noted)."""
    if not solvent:
        return ""
    if engines.supports_solvent(method, solvent):
        return solvent
    msg = f"{engines.all_engines()[method].label} has no implicit solvent model for {solvent}: gas phase"
    if msg not in notes:
        notes.append(msg)
    return ""


def _forcefield_fallback(result: MoleculeResult, forcefield: str, method: str, log: Callable[[str], None]) -> None:
    """MMFF94s had no parameters for the structure (metals, hypervalent atoms, radicals): say what was used."""
    if forcefield == "UFF":
        msg = "MMFF94s has no parameters for this structure: conformers pre-optimized with UFF"
        if method == "mmff":
            msg += " (the energies are UFF energies)"
    elif method == "mmff":
        raise UnsupportedStructure("neither MMFF94s nor UFF has parameters for this structure, so method 'mmff' "
                                   "has no energies; use an xTB method")
    else:
        msg = ("neither MMFF94s nor UFF has parameters for this structure: the conformers are unoptimized ETKDG "
               "geometries until the refinement")
    result.notes.append(msg)
    log(f"  {msg}")


#: BuildConfig.crest_fallback values: what a molecule gets when CREST is missing or fails
CREST_FALLBACKS = ("rdkit", "fail")


class ConformerSearchError(RuntimeError):
    """The conformer search asked for failed and the settings say not to fall back (``crest_fallback = "fail"``)."""


def fallback_count(results) -> int:
    """How many of ``results`` got their conformers from another search than the one asked for (CREST failed, RDKit
    used). A run folder counts them in its job table (``jobs.RunStatus.fallbacks``)."""
    return sum(bool(getattr(r, "conformer_fallback", "")) for r in results)


def fallback_summary(n: int, total: int) -> str:
    """The end-of-batch line for ``n`` molecules whose CREST search fell back to RDKit ('' when there are none)."""
    if not n:
        return ""
    return (f"WARNING: CREST failed for {n} of {total} molecule(s); their conformers come from RDKit ETKDGv3 + "
            "MMFF94s (summary.csv: conformer_fallback). Rerun them with --crest-fallback fail to see the errors as "
            "failures, or check the CREST install (quantabench-cli doctor).")


def _conformer_search(prepared: Chem.Mol, result: MoleculeResult, cfg: BuildConfig, threads: int,
                      should_stop: Callable[[], bool], log: Callable[[str], None],
                      seeds: list | tuple = ()) -> confs.ConformerSet:
    """RDKit ETKDGv3 + MMFF94s, or CREST. If CREST is missing or fails, ``cfg.crest_fallback`` decides: "rdkit" goes
    on with RDKit's conformers and flags the molecule (a note, ``conformer_fallback``, a warning in the log, a count
    at the end of the batch), "fail" fails the molecule (``ConformerSearchError``). ``seeds``: coordinates from the
    input file (import option *keep coordinates*), the first conformers of RDKit's set; CREST makes its own start."""
    if cfg.conformer_search == "crest" and seeds:
        result.notes.append("CREST makes its own start geometry: the file's coordinates were not used")
    if cfg.conformer_search == "crest":
        from . import crest

        if cfg.crest_fallback not in CREST_FALLBACKS:
            raise ValueError(f"crest_fallback must be one of {', '.join(CREST_FALLBACKS)}, not {cfg.crest_fallback!r}")
        try:
            # cfg.crest_threads: the threads a parallel run lends this CREST call (dist.threads)
            crest_threads = int(cfg.crest_threads) if int(cfg.crest_threads or 0) > 0 else threads
            cs = crest.generate(prepared, cfg, threads=crest_threads, solvent=result.solvent, charge=result.charge,
                                multiplicity=result.multiplicity, should_stop=should_stop, log=log)
            result.conformer_search = "crest"
            return cs
        except Cancelled:
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {crest.failure_reason(str(exc))}"
            if cfg.crest_fallback == "fail":
                raise ConformerSearchError(f"CREST conformer search failed ({reason}); crest_fallback = fail, so "
                                           "the molecule is not built with RDKit's conformers instead") from exc
            result.conformer_fallback = f"CREST failed: {reason}"
            result.notes.append(f"CREST conformer search failed ({reason}); used RDKit ETKDGv3 + MMFF94s instead "
                                "(summary.csv: conformer_fallback; --crest-fallback fail stops the molecule instead)")
            log(f"  WARNING: {result.notes[-1]}")
    elif cfg.conformer_search not in ("rdkit", ""):
        raise ValueError(f"conformer_search must be 'rdkit' or 'crest', not {cfg.conformer_search!r}")
    result.conformer_search = "rdkit"
    # Keep a wider force-field window: the refined ranking can differ from MMFF's.
    return confs.generate(prepared, cfg.max_conformers, cfg.energy_window_kcal + 4.0, cfg.rmsd_threshold,
                          cfg.seed, threads, seeds=seeds)


def _engine_properties(result: MoleculeResult, mol: Chem.Mol, best: ConformerResult, cfg: BuildConfig,
                       log: Callable[[str], None], should_stop: Callable[[], bool] = lambda: False) -> bool | None:
    """Properties from the refinement engine itself (e.g. MOPAC heat of formation, xtb GFN2 dipole), with the
    refinement's solvent. None when the engine has no properties function, False when it failed (noted)."""
    fn = engines.properties_function(result.method)
    if fn is None:
        return None
    atoms = export.mol_to_atoms(mol, best.conf_id)
    try:
        props = dict(fn(atoms.numbers, atoms.positions, result.charge, result.multiplicity,
                        should_stop=should_stop, **calculator_options(result, cfg)))
    except Cancelled:
        raise
    except Exception as exc:
        result.notes.append(f"{engines.ENGINES[result.method].label} properties skipped "
                            f"({type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''})")
        log(f"  {result.notes[-1]}")
        return False
    charges = props.pop("charges", None)
    for note in props.pop("_notes", []):
        result.notes.append(note)
    result.descriptors.update(props)
    if charges is not None and best.charges is None:
        best.charges = [float(x) for x in charges]
        if best.dipole_debye is None:
            prefix = engines.ENGINES[result.method].prop_prefix
            dip = props.get(f"{prefix}_dipole_debye")
            best.dipole_debye = float(dip) if dip is not None else None
        result.notes.append(f"atomic charges: {engines.ENGINES[result.method].label}")
    return True


def _xtb_properties(result: MoleculeResult, mol: Chem.Mol, best: ConformerResult, cfg: BuildConfig,
                    solvated: bool, log: Callable[[str], None]) -> bool:
    """GFN1-xTB charges, dipole and polarizability of the best conformer (``xtb_*`` descriptors): dxtb in the gas
    phase (under the memory guard), tblite with the refinement's solvent when ``solvated``. False when the step
    was skipped (memory guard) or failed (the reason is in the notes)."""
    atoms = export.mol_to_atoms(mol, best.conf_id)
    args = (atoms.numbers, atoms.positions, atoms.get_masses(), result.charge, result.multiplicity)
    if solvated:     # tblite: no dxtb autograd graph, nothing for the memory guard to check
        do_charges, pol_engine, guard_note = True, "tblite", ""
    else:
        do_charges, pol_engine, guard_note = _properties_plan(atoms.numbers, result.charge, result.multiplicity, cfg)
    if guard_note:
        result.notes.append(guard_note)
        log(f"  {guard_note}")
    if not do_charges:
        return False
    try:
        if solvated:
            props = properties.tblite_xtb_properties(*args, solvent=result.solvent)
        else:
            kw = {"polarizability": bool(pol_engine) or not engines.has_libcint(),
                  "polarizability_engine": pol_engine or "dxtb"}
            try:
                props = properties.xtb_properties(*args, device=result.device, **kw)
            except Exception:
                if result.device == "cpu":
                    raise
                props = properties.xtb_properties(*args, device="cpu", **kw)
    except Cancelled:
        raise
    except Exception as exc:
        result.notes.append(f"xTB properties skipped ({type(exc).__name__}: {exc})")
        return False
    charges = props.pop("xtb_charges")
    where = props.pop("_solvation", "")
    for note in props.pop("_notes", []):
        result.notes.append(note)
        log(f"  {note}")
    result.descriptors.update(props)
    if best.charges is None:
        best.charges, best.dipole_debye = charges, props["xtb_dipole_debye"]
        result.notes.append("atomic charges: GFN1-xTB Mulliken" + (f" ({where})" if where else ""))
    log(f"  GFN1-xTB properties{' in ' + where if where else ''}: dipole {props['xtb_dipole_debye']:.2f} D"
        + (f", polarizability {props['xtb_polarizability_iso_A3']:.2f} Å^3"
           if "xtb_polarizability_iso_A3" in props else ""))
    if pol_engine == "tblite" and "xtb_polarizability_iso_A3" in props:
        result.descriptors["xtb_polarizability_method"] = ("tblite GFN1-xTB finite field"
                                                           + (f", {where}" if where else ""))
    return True


def _estimates(result: MoleculeResult, mol: Chem.Mol, best: ConformerResult, cfg: BuildConfig,
               log: Callable[[str], None]) -> None:
    """The cheap estimates of ``chem.estimates`` (liquid density and molar volume, additive and D4
    polarizabilities, Sugden parachor) of the best conformer, in every phase. A method without parameters for an
    element gives no column and a note; an estimate never fails a molecule."""
    vdw = result.descriptors.get("vdw_volume_A3_best") if surface_options(cfg) == desc.DEFAULT_SURFACE else None
    try:
        values, notes = estimates.estimate(mol, best.conf_id, vdw_volume_bondi_A3=vdw)
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - estimates are extras: note and go on
        result.notes.append(f"estimates skipped ({type(exc).__name__}: {exc})")
        return
    result.descriptors.update(values)
    result.notes.extend(notes)


def _gfn2_polarizability(result: MoleculeResult, mol: Chem.Mol, best: ConformerResult, cfg: BuildConfig,
                         log: Callable[[str], None]) -> None:
    """GFN2-xTB's static polarizability (``tblite_gfn2_polarizability_*``) of the best conformer when the
    refinement engine did not report one (AIMNet2, dxtb, MOPAC, ...), with tblite and the run's solvent, so a
    table has both the GFN1-xTB (``xtb_*``) and the GFN2-xTB response whatever refined it."""
    if "tblite_gfn2_polarizability_iso_A3" in result.descriptors or not properties.tblite_available():
        return
    atoms = export.mol_to_atoms(mol, best.conf_id)
    try:
        iso, aniso, notes = properties.tblite_gfn2_polarizability(atoms.numbers, atoms.positions, result.charge,
                                                                  result.multiplicity, solvent=result.solvent)
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - an extra column: note and go on
        result.notes.append(f"GFN2-xTB polarizability skipped ({type(exc).__name__}: {exc})")
        return
    result.notes.extend(notes)
    if iso is not None:
        result.descriptors["tblite_gfn2_polarizability_iso_A3"] = iso
        result.descriptors["tblite_gfn2_polarizability_aniso_A3"] = aniso
        log(f"  GFN2-xTB polarizability (tblite{', ' + result.solvent if result.solvent else ''}): {iso:.2f} Å^3")


def _qm_stage(result: MoleculeResult, cs: confs.ConformerSet, cfg: BuildConfig, threads: int,
              should_stop: Callable[[], bool], log: Callable[[str], None]) -> None:
    """QM (DFT) calculations on the lowest conformers; a failure becomes a note, not a failed molecule."""
    from . import qm

    try:
        qm.run_stage(result, cs, cfg, threads=threads, should_stop=should_stop, log=log)
    except Cancelled:
        raise
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else ""
        result.notes.append(f"QM stage skipped ({type(exc).__name__}: {first})")
        log(f"  {result.notes[-1]}")


def _rank_step(result: MoleculeResult, cs: confs.ConformerSet, refined: list[ConformerResult], cfg: BuildConfig,
               solvent: str, device: str, threads: int, clock, should_stop: Callable[[], bool],
               log: Callable[[str], None], stage: Callable[[str], None]) -> list[ConformerResult]:
    """The conformers to keep, lowest first by the level that ranks them (``chem.rank``), with ``rank_energy_ev``
    and ``refine_rel_kcal`` set when a rank step ran; ``result.rank`` and the phase note record it.

    ``rank_method`` "refine" (or a level that cannot run here): RMSD and ``energy_window_kcal`` pruning at the
    refinement energies, as before RANK-2. Any other level: RMSD pruning only, a single point of every pooled
    conformer at the rank level, then ``rank_window_kj`` around the rank level's lowest. A failed rank step is noted
    and the molecule ranked at the refinement level; it never fails the molecule."""
    from . import rank as rank_mod

    mol = cs.mol
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    ids = [c.conf_id for c in refined]
    e_kcal = [c.energy_ev * KCAL_PER_EV for c in refined]
    by_id = {c.conf_id: c for c in refined}
    steps = result.phase_steps
    plan = rank_mod.resolve(cfg.rank_method, numbers, result.charge, result.multiplicity, solvent, cfg.qm_method)
    refine_label = engines.all_engines()[result.method].label if result.method in engines.all_engines() \
        else result.method

    def at_refinement(why: str = "") -> list[ConformerResult]:
        if why:
            plan.notes.append(why)
            plan.level = "refine"
        note = rank_mod.summary(plan)
        if note and note not in result.notes:
            result.notes.append(note)
            log(f"  {note}")
        keep_ids, _ = confs.prune(mol, ids, e_kcal, cfg.energy_window_kcal, cfg.rmsd_threshold)
        kept = sorted((by_id[i] for i in keep_ids), key=lambda c: c.energy_ev)
        for c in refined:
            c.rank_energy_ev = c.refine_rel_kcal = None
        result.rank = {"method": plan.requested, "level": "refine",
                       "label": rank_mod.label("refine", solvent, refine_label), "n_pool": len(refined),
                       "n_kept": len(kept), "window_kcal": float(cfg.energy_window_kcal), "notes": list(plan.notes)}
        if solvent:
            steps["rank"] = f"refinement ({steps.get('refine', 'gas')})"
        return kept

    if plan.level == "refine":
        return at_refinement()
    stage("rank")
    t0 = time.perf_counter()
    keep_ids, _ = confs.prune(mol, ids, e_kcal, None, cfg.rmsd_threshold)      # duplicates only: no window here
    pool = sorted((by_id[i] for i in keep_ids), key=lambda c: c.energy_ev)
    level_label = rank_mod.label(plan.level, solvent, refine_label, cfg.qm_method)
    clock.start("rank", engine=level_label, n_in=len(pool))
    log(f"  rank: {len(pool)} conformer(s) at {level_label}")
    try:
        if plan.level == "qm":
            ranked = rank_mod.qm_ranked(result, cs, pool, cfg, threads=threads, should_stop=should_stop, log=log)
        else:
            cache = rank_mod.SinglePointCache()
            try:
                ranked = rank_mod.compute(plan.level, mol, pool, charge=result.charge,
                                          multiplicity=result.multiplicity, solvent=solvent,
                                          refine_method=result.method, device=device, threads=threads, cache=cache,
                                          should_stop=should_stop, log=log, job_name=result.input.id)
            finally:
                cache.close()
    except Cancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - the rank step falls back to the refinement level, noted
        clock.abort("failed")
        first = str(exc).splitlines()[0][:300] if str(exc) else ""
        return at_refinement(f"rank {plan.level}: failed ({type(exc).__name__}: {first}); ranked at the refinement "
                             "level")
    energies = {cid: e for cid, e in ranked.energies_ev.items() if e is not None}
    for note in ranked.notes:
        if note not in result.notes:
            result.notes.append(note)
    if not energies:
        clock.abort("failed")
        return at_refinement(f"rank {plan.level}: no conformer got an energy; ranked at the refinement level")
    kept_ids = rank_mod.window_keep(energies, cfg.rank_window_kj)
    e_ref0 = pool[0].energy_ev
    for c in refined:
        c.rank_energy_ev = ranked.energies_ev.get(c.conf_id)
        c.refine_rel_kcal = (c.energy_ev - e_ref0) * KCAL_PER_EV
    kept = [by_id[i] for i in kept_ids]
    seconds = time.perf_counter() - t0
    result.timings["rank"] = seconds
    result.rank = rank_mod.record(plan, pool, ranked, kept_ids, cfg.rank_window_kj, solvent=solvent,
                                  refine_label=refine_label, qm_method=cfg.qm_method, seconds=seconds, threads=threads)
    result.rank["qm_done"] = ranked.qm_done
    note = rank_mod.summary(plan)
    if note and note not in result.notes:
        result.notes.append(note)
        log(f"  {note}")
    if solvent:
        steps["rank"] = level_label
        if ranked.qm_done and result.qm is not None:
            qm_solvent = result.qm.get("solvent") or ""
            steps["qm"] = (f"PCM({qm_solvent})" if qm_solvent else "gas (see the QM notes)") + " (the rank step)"
    r = result.rank
    log(f"  rank: {len(kept)} of {len(pool)} within {cfg.rank_window_kj:g} kJ/mol; "
        + (f"the pick changed: refinement's #{r['pick_refine_order'] + 1}, {r['pick_refine_rel_kj']:+.1f} kJ/mol "
           "above its lowest" if r["pick_changed"] else "the same pick as the refinement")
        + f" [{seconds:.1f} s]")
    clock.stop(n_out=len(kept), engine=rank_mod.engine_label(ranked))
    return kept


def run_molecule(inp: MoleculeInput, cfg: BuildConfig, device: str = "cpu", threads: int = 1,
                 should_stop: Callable[[], bool] = lambda: False,
                 log: Callable[[str], None] = _noop, stage: Callable[[str], None] = _noop) -> MoleculeResult:
    result = MoleculeResult(input=inp, multiplicity=inp.multiplicity, device=device)
    clock = telemetry.StageClock(result, threads)       # stage timings for the job table (never raises)
    t = time.perf_counter()
    clock.start("prepare", engine="rdkit")
    try:
        if device != "cpu":   # GPU policies for dxtb (also set in lane processes, which get cfg, not settings)
            from . import torch_compat

            torch_compat.apply_settings(cfg)
        stage("prepare")
        info: dict = {}
        if inp.meta.get(ID_NOTE):
            result.notes.append(str(inp.meta[ID_NOTE]))
        result.notes += [str(n) for n in inp.meta.get(imp.NOTES) or []]     # what the import did (filters, 2D -> 3D)
        from . import solvents

        result.phase = solvents.canonical(cfg.solvent) or "gas"
        drawn = inp.to_mol()
        result.input_inchikey, why = inchikey(drawn)
        if why:
            result.notes.append(f"no input InChIKey: {why}")
        prepared, notes = prepare(drawn, cfg.ph if cfg.protonate else None, cfg.keep_largest_fragment,
                                  cfg.protonation_method, info=info)
        # fragments the import filters removed count as removed, and the charge as drawn is the file's
        result.input_charge = inp.meta.get(imp.FILE_CHARGE, info.get("input_charge"))
        result.removed_fragments = list(inp.meta.get(imp.REMOVED) or []) + list(info.get("removed_fragments") or [])
        result.smiles = Chem.MolToSmiles(Chem.RemoveHs(prepared))
        result.canonical_smiles = result.smiles
        result.inchikey, why = inchikey(prepared)
        if why:
            notes.append(f"no InChIKey: {why}")
        result.charge = inp.charge if inp.charge is not None else formal_charge(prepared)
        if inp.charge is not None and inp.charge != formal_charge(prepared):
            notes.append(f"charge {inp.charge:+d} given with the input (the structure's formal charge is "
                         f"{formal_charge(prepared):+d})")
        # an explicit multiplicity other than the default singlet wins; otherwise drawn radicals decide
        result.multiplicity, spin_note = spin_multiplicity(prepared, result.charge,
                                                           inp.multiplicity if inp.multiplicity != 1 else None)
        if spin_note:
            notes.append(spin_note)
        result.notes += notes
        numbers = [a.GetAtomicNum() for a in prepared.GetAtoms()]
        solvent = solvents.canonical(cfg.solvent)
        method = cfg.method
        if method == "auto":
            method, why = engines.auto_choice(numbers, solvent)
            if why:
                result.notes.append(why)
                notes.append(why)
        else:
            ok, why = engines.available(method)
            if ok:
                ok, why = engines.supports(method, numbers)
            if not ok:
                fallback = engines.resolve_auto(numbers, solvent)
                result.notes.append(f"{method} not usable ({why}); using {fallback}")
                method = fallback
        result.method = method
        result.solvent = _solvent_for(method, solvent, result.notes)
        if method == "mmff" or not engines.ENGINES[method].gpu_capable:
            result.device = "cpu"
        result.timings["prepare"] = time.perf_counter() - t
        clock.features(prepared)
        clock.stop()
        log(f"{inp.id}: {result.smiles} (charge {result.charge:+d}"
            + (f", multiplicity {result.multiplicity}" if result.multiplicity != 1 else "")
            + f") -> {engines.ENGINES[method].label} on {result.device}"
            + (f" in {result.solvent}" if result.solvent else ""))
        for n in notes:
            log(f"  {n}")

        stage("conformers")
        t = time.perf_counter()
        crest_run = cfg.conformer_search == "crest"
        clock.start("conformers", threads=int(cfg.crest_threads or 0) or threads if crest_run else threads)
        conformer_solvent = result.solvent        # what CREST was given (RDKit's MMFF94s has no solvent)
        seeds: list = []
        if inp.meta.get(imp.KEEP_COORDS):
            seeds, why = confs.seed_coordinates(drawn, prepared)
            result.notes.append(f"the file's coordinates are the first {len(seeds)} conformer(s) (MMFF94s-optimized "
                                "like the others)" if seeds else f"the file's coordinates were not used ({why})")
            log(f"  {result.notes[-1]}")
        cs = _conformer_search(prepared, result, cfg, threads, should_stop, log, seeds=seeds)
        result.forcefield, result.n_embedded = cs.forcefield, cs.n_embedded
        result.timings["conformers"] = time.perf_counter() - t
        clock.stop(n_out=len(cs.conf_ids), engine=_conformer_engine(result, cfg, conformer_solvent))
        log(f"  {cs.n_embedded} embedded, {len(cs.conf_ids)} unique after {cs.forcefield} + pruning")
        for note in cs.notes:
            result.notes.append(note)
            log(f"  {note}")
        if result.conformer_search == "rdkit" and cs.forcefield != "MMFF94s":
            _forcefield_fallback(result, cs.forcefield, method, log)
        if should_stop():
            raise Cancelled()

        stage("refine")
        t = time.perf_counter()
        clock.start("refine", engine=method, n_in=min(len(cs.conf_ids), max(1, cfg.refine_top)))
        candidates = [method]
        if cfg.method == "auto":  # if the first choice cannot start (e.g. model download offline), try the next
            order = engines.auto_candidates(numbers, solvent)
            candidates += order[order.index(method) + 1:] if method in order else []
        for n_try, candidate in enumerate(candidates):
            result.method = candidate
            result.solvent = _solvent_for(candidate, solvent, result.notes if n_try else [])
            result.device = device if engines.ENGINES[candidate].gpu_capable else "cpu"
            if candidate in ("gfn1-xtb", "gfn2-xtb") and result.device != "cpu":
                from . import torch_compat

                run_on, why = torch_compat.dxtb_device(result.device)   # CPU where dxtb is slower on the GPU
                if why:
                    result.device = run_on
                    if why not in result.notes:
                        result.notes.append(why)
                        log(f"  {why}")
            refine = batch_opt.refine_batched if use_batched(cfg, candidate, result.device) else _refine
            if refine is not _refine:
                log(f"  batched refinement of {min(len(cs.conf_ids), max(1, cfg.refine_top))} conformers "
                    f"on {result.device}")
            try:
                refined = refine(result, cs, cfg, result.device, should_stop, log)
                break
            except Cancelled:
                raise
            except Exception as exc:
                if n_try == len(candidates) - 1:
                    raise
                result.notes.append(f"{engines.ENGINES[candidate].label} failed ({type(exc).__name__}: {exc}); "
                                    f"trying {engines.ENGINES[candidates[n_try + 1]].label}")
                log(f"  {result.notes[-1]}")
        method = result.method
        result.timings["refine"] = time.perf_counter() - t
        clock.stop(n_out=len(refined), engine=_engine_label(method, result.device, result.solvent))
        result.solvation_model = solvation_model(method, result.solvent)
        steps = result.phase_steps
        if solvent:                  # a solvent run: where each step was computed, and why a step was not
            if result.conformer_search == "crest":
                steps["conformers"] = f"ALPB({conformer_solvent}) (CREST)" if conformer_solvent else "gas (CREST)"
            else:
                steps["conformers"] = (f"gas ({cs.forcefield or 'MMFF94s'} has no implicit solvent: the same "
                                       "conformer pool in every phase)")
            steps["refine"] = solvation_label(method, result.solvent) if result.solvent else _gas_reason(method, solvent)

        # re-prune on refined geometries, rank (the refinement's energies or the rank step's, chem.rank), weight
        result.mol = cs.mol
        kept = _rank_step(result, cs, refined, cfg, solvent, device, threads, clock, should_stop, log, stage)
        e0 = kept[0].ranking_energy_ev
        weights = confs.boltzmann_weights([(c.ranking_energy_ev - e0) * KCAL_PER_EV for c in kept], cfg.temperature_k)
        for rank, (c, w) in enumerate(zip(kept, weights)):
            c.rank, c.rel_kcal, c.weight = rank, (c.ranking_energy_ev - e0) * KCAL_PER_EV, float(w)
        result.conformers = kept

        stage("descriptors")
        t = time.perf_counter()
        clock.start("descriptors", engine="rdkit" + (f" + {method} properties" if engines.properties_function(method)
                                                     else ""), n_in=len(kept))
        result.descriptors = {**desc.graph_descriptors(cs.mol),
                              **desc.ensemble_descriptors(cs.mol, [c.conf_id for c in kept], weights,
                                                          surface_options(cfg))}
        if kept[0].dipole_debye is not None:
            result.descriptors["dipole_debye_best"] = kept[0].dipole_debye
            dips = [c.dipole_debye for c in kept]
            if all(d is not None for d in dips):
                result.descriptors["dipole_debye_boltz"] = float(np.dot(weights, dips))
        _estimates(result, cs.mol, kept[0], cfg, log)
        result.descriptors["method"] = method
        result.descriptors["rank_method"] = (result.rank or {}).get("level", "refine")   # the level that ranked
        if method == "mmff" and result.forcefield == "UFF":
            result.descriptors["method"] = "UFF"      # MMFF94s had no parameters: the energies are UFF's
        if method == "mmff" and result.forcefield.startswith("CREST"):
            # MMFF refinement keeps CREST's geometries and energies (they are not MMFF energies)
            result.descriptors["method"] = result.forcefield
            result.notes.append(f"no refinement: geometries and energies are {result.forcefield}, not MMFF94s")
        if result.solvent:
            result.descriptors["solvent"] = result.solvent
            if engines.ENGINES[method].family == "semi-empirical":
                result.notes.append("MOPAC COSMO energies are not smooth under small displacements: optimizations "
                                    "may stop short of fmax and COSMO frequencies are approximate (use the gas "
                                    "phase for frequencies)")
        engine_props = _engine_properties(result, cs.mol, kept[0], cfg, log, should_stop)
        if solvent and engine_props is not None:
            steps["properties"] = (("skipped (see notes)" if not engine_props else solvation_label(method, result.solvent))
                                   if result.solvent else _gas_reason(method, solvent))
        # in a solvent, the GFN1-xTB properties come from tblite with the same solvation (dxtb has none); in the gas
        # phase from dxtb, or from tblite where dxtb is not installed
        solvated_props = bool(result.solvent) and properties.tblite_available()
        tblite_props = solvated_props or (not properties.available() and properties.tblite_available())
        if cfg.xtb_properties and (tblite_props or properties.available()):
            clock.stop(n_out=len(kept))
            stage("properties")
            clock.start("properties", engine="tblite gfn1" + (f" ({result.solvent})" if result.solvent else "")
                        if tblite_props else "dxtb gfn1", n_in=1)
            done = _xtb_properties(result, cs.mol, kept[0], cfg, tblite_props, log)
            _gfn2_polarizability(result, cs.mol, kept[0], cfg, log)
            if solvent:
                steps["xtb_properties"] = (
                    "skipped (see notes)" if not done else
                    f"{solvation_label('tblite-gfn1', result.solvent)}, tblite GFN1-xTB" if tblite_props
                    else "gas (the refinement ran in the gas phase)" if not result.solvent
                    else "gas (dxtb GFN1-xTB has no implicit solvent; install tblite for solvated xTB properties)")
                if done and result.solvent and not solvated_props:
                    result.notes.append(f"xTB properties: {steps['xtb_properties']}")
            clock.stop(n_out=int(bool(done)))
        result.timings["descriptors"] = time.perf_counter() - t
        clock.stop(n_out=len(kept))            # the descriptors stage, when no xTB properties ran
        if cfg.frequencies:
            stage("frequencies")
            t = time.perf_counter()
            clock.start("frequencies", engine=f"{method} {cfg.freq_hessian}", n_in=1)
            done = _frequencies(result, cs, kept[0], cfg, should_stop, log)
            if solvent:
                plugin_engine = engines.plugin(method) is not None
                steps["frequencies"] = ("skipped (see notes)" if not done else
                                        solvation_label(method, result.solvent) if result.solvent and plugin_engine
                                        else _gas_reason(method, solvent))
            result.timings["frequencies"] = time.perf_counter() - t
            clock.stop(n_out=int(bool(done)))
        if cfg.qm_method and not (result.rank or {}).get("qm_done"):   # rank level "qm" ran the QM stage
            stage("qm")
            t = time.perf_counter()
            clock.start("qm", engine=f"{cfg.qm_program or 'psi4'} {cfg.qm_method} {cfg.qm_task}",
                        n_in=max(1, int(cfg.qm_top or 1)))
            _qm_stage(result, cs, cfg, threads, should_stop, log)
            if solvent and result.qm is not None:
                qm_solvent = result.qm.get("solvent") or ""
                steps["qm"] = f"PCM({qm_solvent})" if qm_solvent else "gas (see the QM notes)"
            result.timings["qm"] = time.perf_counter() - t
            qm_done = result.qm or {}
            clock.stop(n_out=len(qm_done.get("conformers") or []),
                       engine=f"{qm_done.get('program') or cfg.qm_program or 'psi4'} "
                              f"{qm_done.get('program_version') or ''} {cfg.qm_method} {cfg.qm_task}".replace("  ", " "))
        result.status = "ok"
        log(f"  done: {len(kept)} conformer(s), best E = {kept[0].energy_ev:.5f} eV"
            + (f" (rank {kept[0].rank_energy_ev:.5f} eV)" if kept[0].rank_energy_ev is not None else ""))
    except Cancelled:
        result.status = "cancelled"
        result.error = "cancelled"
        clock.abort("cancelled")
    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        clock.abort("failed")
        log(f"  FAILED: {result.error}")
    return result


def _engine_label(method: str, device: str, solvent: str) -> str:
    """'tblite-gfn2 (water)', 'aimnet2 on cuda:0': the engine of a stage for the telemetry rows."""
    return method + (f" ({solvent})" if solvent else "") + (f" on {device}" if device and device != "cpu" else "")


def _conformer_engine(result: MoleculeResult, cfg: BuildConfig, solvent: str) -> str:
    """'crest gfnff quick (water)' or 'rdkit etkdg + MMFF94s' (with ' (crest failed)' after a fallback)."""
    if result.conformer_search == "crest":
        from ..dist.threads import crest_level

        return f"crest {crest_level(cfg)}" + (f" ({solvent})" if solvent else "")
    return f"rdkit etkdg + {result.forcefield or 'MMFF94s'}" + (" (crest failed)" if result.conformer_fallback else "")


def new_run_dir(base: str | Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(base).expanduser() / f"{stamp}_{label}"[:120]


def run(inputs: list[MoleculeInput], cfg: BuildConfig, progress: ProgressFn = _noop,
        should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = _noop,
        run_dir: Path | None = None,
        on_result: Callable[[MoleculeResult], None] = _noop) -> tuple[list[MoleculeResult], Path | None]:
    threads = resources.apply_thread_limits(priority.budget_of(cfg))   # threads 0: all but the free ones
    device = resources.select_device(cfg.gpu, cfg.gpu_index, cfg.gpu_memory_fraction)
    log(f"Resources: {threads} CPU threads, device {device}")
    cfg = replace(cfg, memory_budget_gb=resolved_budget(cfg, 1, log))
    with priority.children_at(cfg.priority):          # xtb, CREST, Psi4 started here run at the run's priority
        return _run_inputs(inputs, cfg, device, threads, progress, should_stop, log, run_dir, on_result)


def _run_inputs(inputs, cfg, device, threads, progress, should_stop, log, run_dir, on_result):
    if cfg.write_files and run_dir is None:
        label = inputs[0].id if len(inputs) == 1 else f"batch_{len(inputs)}"
        run_dir = new_run_dir(cfg.output_dir or Settings().output_path(), label)
    results = []
    total = len(inputs)
    for i, inp in enumerate(inputs):
        if should_stop():
            break
        progress(i, total, "start", inp.id)
        res = run_molecule_limited(inp, cfg, device, threads, should_stop, log,
                                   stage=lambda s, i=i, inp=inp: progress(i, total, s, inp.id))
        if cfg.write_files and run_dir is not None and res.status == "ok":
            files = export.write_molecule(res, run_dir / inp.id, cfg.formats)
            res.files = [str(p) for p in files]
        results.append(res)
        on_result(res)
        progress(i + 1, total, res.status, inp.id)
    if cfg.write_files and run_dir is not None and results:
        export.write_batch_tables(results, run_dir)
        log(f"Results written to {run_dir}")
    line = fallback_summary(fallback_count(results), len(results))
    if line:
        log(line)
    return results, run_dir
