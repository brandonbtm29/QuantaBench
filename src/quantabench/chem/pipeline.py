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

from .. import resources
from ..settings import Settings
from . import batch_opt, engines, export, properties
from . import conformers as confs
from . import descriptors as desc
from .inputs import MoleculeInput
from .optimize import Cancelled, optimize
from .standardize import formal_charge, prepare

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
    energy_window_kcal: float = 6.0
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
    seed: int = 42
    solvent: str = ""                     # implicit solvent (chem.solvents name), "" = gas phase
    conformer_search: str = "rdkit"       # "rdkit" (ETKDGv3 + MMFF94s) | "crest" (CREST iMTD-GC, external)
    crest_method: str = "gfn2"            # "gfn2" | "gfnff" | "gfn2//gfnff" | "gfn1"
    crest_quick: bool = True              # CREST --quick (fewer metadynamics runs): minutes instead of hours
    qm_method: str = ""                   # "functional/basis", e.g. "wb97x-d/def2-svp"; "" = no QM stage
    qm_task: str = "energy"               # "energy" (single point) | "optimize"
    qm_top: int = 1                       # how many of the lowest conformers get the QM stage
    qm_properties: bool = True            # dipole, quadrupole, Mulliken/Lowdin/MBIS charges, orbital energies
    qm_frequencies: bool = False          # QM Hessian + IR + thermochemistry of the best QM conformer
    qm_cubes: bool = False                # density + ESP cubes and molecular-surface descriptors (best conformer)
    qm_program: str = "psi4"
    threads: int = 0
    gpu: str = "auto"
    gpu_index: int = 0
    gpu_memory_fraction: float = 0.9
    linalg: str = "auto"                  # torch_compat linear-algebra policy: "auto" | "cpu" | "device"
    dxtb_gpu: str = "auto"                # dxtb on a GPU: "auto" (where it is faster) | "on" | "off"
    output_dir: str = ""
    formats: tuple[str, ...] = export.DEFAULT_FORMATS
    write_files: bool = True

    @classmethod
    def from_settings(cls, s: Settings, **overrides) -> BuildConfig:
        b, r = s.build, s.resources
        cfg = cls(method=b.refine_method, ph=b.ph, protonate=b.protonate, keep_largest_fragment=b.keep_largest_fragment,
                  max_conformers=b.max_conformers, energy_window_kcal=b.energy_window_kcal,
                  rmsd_threshold=b.rmsd_threshold, fmax=b.fmax, max_steps=b.max_steps, temperature_k=b.temperature_k,
                  seed=b.random_seed, batched=b.batched, frequencies=b.frequencies, threads=r.threads, gpu=r.gpu, gpu_index=r.gpu_index,
                  gpu_memory_fraction=r.gpu_memory_fraction, linalg=r.linalg, dxtb_gpu=r.dxtb_gpu,
                  output_dir=str(s.output_path()),
                  conformer_search=b.conformer_search, solvent=b.solvent, qm_method=b.qm_method)
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
    qm: dict | None = None             # chem.qm stage results (see qm.QM_RESULT_KEYS)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("mol", "input")}
        d["input"] = {k: v for k, v in asdict(self.input).items() if k != "mol"}
        return d

    def summary_row(self) -> dict:
        best = self.conformers[0] if self.conformers else None
        return {
            "id": self.input.id, "name": self.input.name, "status": self.status, "error": self.error,
            "input_smiles": self.input.smiles, "smiles": self.smiles, "charge": self.charge, "method": self.method,
            "device": self.device, "solvent": self.solvent, "conformer_search": self.conformer_search,
            "n_embedded": self.n_embedded, "n_conformers": len(self.conformers),
            "best_energy_eV": best.energy_ev if best else "", "best_converged": best.converged if best else "",
            "dipole_debye": best.dipole_debye if best and best.dipole_debye is not None else "",
            "qm_method": (self.qm or {}).get("level", ""), "qm_energy_eV": (self.qm or {}).get("energy_ev", ""),
            "seconds": round(sum(self.timings.values()), 2), "notes": " | ".join(self.notes),
        }


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


def calculator_options(result: MoleculeResult, cfg: BuildConfig) -> dict:
    """Options every engine factory receives (see ``engines`` module docstring)."""
    return {"solvent": result.solvent, "threads": resources.resolve_threads(cfg.threads)}


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
    full_dipole = plug is not None and plug.extra.get("dipole_origin") == "com"

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
        mu = calc.results.get("dipole") if full_dipole else None
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
                 should_stop: Callable[[], bool], log: Callable[[str], None]) -> None:
    """Vibrational analysis of the best conformer; a failure becomes a note, not a failed molecule."""
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
                                preoptimize_fmax=fmax, should_stop=should_stop, log=log)
    except Cancelled:
        raise
    except Exception as exc:
        result.notes.append(f"frequencies skipped ({type(exc).__name__}: {exc})")
        log(f"  {result.notes[-1]}")
        return
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


def _conformer_search(prepared: Chem.Mol, result: MoleculeResult, cfg: BuildConfig, threads: int,
                      should_stop: Callable[[], bool], log: Callable[[str], None]) -> confs.ConformerSet:
    """RDKit ETKDGv3 + MMFF94s, or CREST (falls back to RDKit with a note if CREST is missing or fails)."""
    if cfg.conformer_search == "crest":
        from . import crest

        try:
            cs = crest.generate(prepared, cfg, threads=threads, solvent=result.solvent, charge=result.charge,
                                multiplicity=result.multiplicity, should_stop=should_stop, log=log)
            result.conformer_search = "crest"
            return cs
        except Cancelled:
            raise
        except Exception as exc:
            result.notes.append(f"CREST conformer search failed ({type(exc).__name__}: {str(exc).splitlines()[0]}); "
                                "used RDKit ETKDGv3 + MMFF94s")
            log(f"  {result.notes[-1]}")
    elif cfg.conformer_search not in ("rdkit", ""):
        raise ValueError(f"conformer_search must be 'rdkit' or 'crest', not {cfg.conformer_search!r}")
    result.conformer_search = "rdkit"
    # Keep a wider force-field window: the refined ranking can differ from MMFF's.
    return confs.generate(prepared, cfg.max_conformers, cfg.energy_window_kcal + 4.0, cfg.rmsd_threshold,
                          cfg.seed, threads)


def _engine_properties(result: MoleculeResult, mol: Chem.Mol, best: ConformerResult, cfg: BuildConfig,
                       log: Callable[[str], None], should_stop: Callable[[], bool] = lambda: False) -> None:
    """Properties from the refinement engine itself (e.g. MOPAC heat of formation, xtb GFN2 dipole)."""
    fn = engines.properties_function(result.method)
    if fn is None:
        return
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
        return
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


def run_molecule(inp: MoleculeInput, cfg: BuildConfig, device: str = "cpu", threads: int = 1,
                 should_stop: Callable[[], bool] = lambda: False,
                 log: Callable[[str], None] = _noop, stage: Callable[[str], None] = _noop) -> MoleculeResult:
    result = MoleculeResult(input=inp, multiplicity=inp.multiplicity, device=device)
    t = time.perf_counter()
    try:
        if device != "cpu":   # GPU policies for dxtb (also set in lane processes, which get cfg, not settings)
            from . import torch_compat

            torch_compat.apply_settings(cfg)
        stage("prepare")
        prepared, notes = prepare(inp.to_mol(), cfg.ph if cfg.protonate else None, cfg.keep_largest_fragment,
                                  cfg.protonation_method)
        result.notes += notes
        result.smiles = Chem.MolToSmiles(Chem.RemoveHs(prepared))
        result.charge = inp.charge if inp.charge is not None else formal_charge(prepared)
        numbers = [a.GetAtomicNum() for a in prepared.GetAtoms()]
        from . import solvents

        solvent = solvents.canonical(cfg.solvent)
        method = cfg.method
        if method == "auto":
            method = engines.resolve_auto(numbers, solvent)
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
        log(f"{inp.id}: {result.smiles} (charge {result.charge:+d}) -> {engines.ENGINES[method].label} on {result.device}"
            + (f" in {result.solvent}" if result.solvent else ""))
        for n in notes:
            log(f"  {n}")

        stage("conformers")
        t = time.perf_counter()
        cs = _conformer_search(prepared, result, cfg, threads, should_stop, log)
        result.forcefield, result.n_embedded = cs.forcefield, cs.n_embedded
        result.timings["conformers"] = time.perf_counter() - t
        log(f"  {cs.n_embedded} embedded, {len(cs.conf_ids)} unique after {cs.forcefield} + pruning")
        if should_stop():
            raise Cancelled()

        stage("refine")
        t = time.perf_counter()
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

        # re-prune on refined energies/geometries, rank, weight
        ids = [c.conf_id for c in refined]
        e_kcal = [c.energy_ev * KCAL_PER_EV for c in refined]
        keep_ids, _ = confs.prune(cs.mol, ids, e_kcal, cfg.energy_window_kcal, cfg.rmsd_threshold)
        by_id = {c.conf_id: c for c in refined}
        kept = sorted((by_id[i] for i in keep_ids), key=lambda c: c.energy_ev)
        e0 = kept[0].energy_ev
        weights = confs.boltzmann_weights([(c.energy_ev - e0) * KCAL_PER_EV for c in kept], cfg.temperature_k)
        for rank, (c, w) in enumerate(zip(kept, weights)):
            c.rank, c.rel_kcal, c.weight = rank, (c.energy_ev - e0) * KCAL_PER_EV, float(w)
        result.conformers = kept
        result.mol = cs.mol

        stage("descriptors")
        t = time.perf_counter()
        result.descriptors = {**desc.graph_descriptors(cs.mol),
                              **desc.ensemble_descriptors(cs.mol, [c.conf_id for c in kept], weights)}
        if kept[0].dipole_debye is not None:
            result.descriptors["dipole_debye_best"] = kept[0].dipole_debye
            dips = [c.dipole_debye for c in kept]
            if all(d is not None for d in dips):
                result.descriptors["dipole_debye_boltz"] = float(np.dot(weights, dips))
        result.descriptors["method"] = method
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
        _engine_properties(result, cs.mol, kept[0], cfg, log, should_stop)
        if cfg.xtb_properties and properties.available():
            stage("properties")
            best = kept[0]
            atoms = export.mol_to_atoms(cs.mol, best.conf_id)
            args = (atoms.numbers, atoms.positions, atoms.get_masses(), result.charge, result.multiplicity)
            try:
                try:
                    props = properties.xtb_properties(*args, device=result.device)
                except Exception:
                    if result.device == "cpu":
                        raise
                    props = properties.xtb_properties(*args, device="cpu")
                charges = props.pop("xtb_charges")
                for note in props.pop("_notes", []):
                    result.notes.append(note)
                    log(f"  {note}")
                result.descriptors.update(props)
                if best.charges is None:
                    best.charges, best.dipole_debye = charges, props["xtb_dipole_debye"]
                    result.notes.append("atomic charges: GFN1-xTB Mulliken")
                log(f"  GFN1-xTB properties: dipole {props['xtb_dipole_debye']:.2f} D"
                    + (f", polarizability {props['xtb_polarizability_iso_A3']:.2f} Å^3"
                       if "xtb_polarizability_iso_A3" in props else ""))
            except Exception as exc:
                result.notes.append(f"xTB properties skipped ({type(exc).__name__}: {exc})")
        result.timings["descriptors"] = time.perf_counter() - t
        if cfg.frequencies:
            stage("frequencies")
            t = time.perf_counter()
            _frequencies(result, cs, kept[0], cfg, should_stop, log)
            result.timings["frequencies"] = time.perf_counter() - t
        if cfg.qm_method:
            stage("qm")
            t = time.perf_counter()
            _qm_stage(result, cs, cfg, threads, should_stop, log)
            result.timings["qm"] = time.perf_counter() - t
        result.status = "ok"
        log(f"  done: {len(kept)} conformer(s), best E = {e0:.5f} eV")
    except Cancelled:
        result.status = "cancelled"
        result.error = "cancelled"
    except Exception as exc:
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        log(f"  FAILED: {result.error}")
    return result


def new_run_dir(base: str | Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path(base).expanduser() / f"{stamp}_{label}"[:120]


def run(inputs: list[MoleculeInput], cfg: BuildConfig, progress: ProgressFn = _noop,
        should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = _noop,
        run_dir: Path | None = None,
        on_result: Callable[[MoleculeResult], None] = _noop) -> tuple[list[MoleculeResult], Path | None]:
    threads = resources.apply_thread_limits(cfg.threads)
    device = resources.select_device(cfg.gpu, cfg.gpu_index, cfg.gpu_memory_fraction)
    log(f"Resources: {threads} CPU threads, device {device}")
    if cfg.write_files and run_dir is None:
        label = inputs[0].id if len(inputs) == 1 else f"batch_{len(inputs)}"
        run_dir = new_run_dir(cfg.output_dir or Settings().output_path(), label)
    results = []
    total = len(inputs)
    for i, inp in enumerate(inputs):
        if should_stop():
            break
        progress(i, total, "start", inp.id)
        res = run_molecule(inp, cfg, device, threads, should_stop, log,
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
    return results, run_dir
