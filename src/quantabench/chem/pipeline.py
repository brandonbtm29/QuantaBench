"""The geometry builder pipeline.

    input (SMILES / name / file / CSV)
      -> standardize (largest fragment, neutral parent)
      -> protonation state at pH
      -> 3D embedding, conformer search (ETKDGv3) + MMFF94s pre-optimization
      -> energy-window + RMSD pruning
      -> refinement of the lowest conformers with the chosen engine (AIMNet2 / xTB / MMFF), GPU when possible;
         all conformers in one torch batch on a GPU (``chem.batch_opt``), one at a time with ASE otherwise
      -> re-pruning, Boltzmann weights, descriptors
      -> optional: frequencies, IR and RRHO/qRRHO thermochemistry of the best conformer
      -> export (XYZ, SDF ensemble, extXYZ with charges, PDB, MOL, LAMMPS data) + batch tables

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
    optimizer: str = "lbfgs"
    temperature_k: float = 298.15
    xtb_properties: bool = True           # GFN1-xTB charges, dipole, polarizability of the best conformer
    batched: str = "auto"                 # "auto" (batched on a GPU) | "on" | "off": see use_batched()
    frequencies: bool = False             # Hessian, IR and thermochemistry of the best conformer
    freq_hessian: str = "auto"            # "auto" | "analytic" | "finite" (vibrations.analyze)
    symmetry_number: int = 0              # rotational symmetry number σ for the entropy; 0 = detect per molecule
    freq_qrrho: bool = True               # Grimme's quasi-RRHO entropy for low-frequency modes
    seed: int = 42
    threads: int = 0
    gpu: str = "auto"
    gpu_index: int = 0
    gpu_memory_fraction: float = 0.9
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
                  gpu_memory_fraction=r.gpu_memory_fraction, output_dir=str(s.output_path()))
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
    vibrations: dict | None = None     # VibrationalResult.to_dict() when cfg.frequencies

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("mol", "input")}
        d["input"] = {k: v for k, v in asdict(self.input).items() if k != "mol"}
        return d

    def summary_row(self) -> dict:
        best = self.conformers[0] if self.conformers else None
        return {
            "id": self.input.id, "name": self.input.name, "status": self.status, "error": self.error,
            "input_smiles": self.input.smiles, "smiles": self.smiles, "charge": self.charge, "method": self.method,
            "device": self.device, "n_embedded": self.n_embedded, "n_conformers": len(self.conformers),
            "best_energy_eV": best.energy_ev if best else "", "best_converged": best.converged if best else "",
            "dipole_debye": best.dipole_debye if best and best.dipole_debye is not None else "",
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

    def build(dev):
        return engines.make_calculator(result.method, numbers, result.charge, result.multiplicity, dev)

    calc = build(device)
    out = []
    for i, (cid, e_ff) in enumerate(zip(ids, ff)):
        if should_stop():
            raise Cancelled()
        atoms = export.mol_to_atoms(mol, cid)
        atoms.calc = calc
        try:
            res = optimize(atoms, cfg.fmax, cfg.max_steps, cfg.optimizer, should_stop)
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
            res = optimize(atoms, cfg.fmax, cfg.max_steps, cfg.optimizer, should_stop)
        conf = mol.GetConformer(cid)
        for j, p in enumerate(atoms.get_positions()):
            conf.SetAtomPosition(j, p.tolist())
        charges = calc.results.get("charges")
        charges = None if charges is None else [float(x) for x in np.ravel(charges)]
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
    fmax = min(cfg.fmax, 0.01)    # frequencies need a tighter minimum than the conformer ranking
    try:
        vr = vibrations.analyze(atoms, result.method, result.charge, result.multiplicity, device=result.device,
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
    result.vibrations = vr.to_dict()
    result.descriptors.update({f"vib_{k}": v for k, v in vr.descriptors().items()})
    if vr.positions is not None:
        pos = np.asarray(vr.positions, float)
        result.vibrations["symbols"] = atoms.get_chemical_symbols()
        result.vibrations["positions_A"] = np.round(pos, 6).tolist()
        shift = float(np.sqrt(((pos - atoms.get_positions()) ** 2).sum(1)).max()) if len(pos) else 0.0
        result.vibrations["geometry_max_shift_A"] = shift
        result.descriptors["vib_geometry_max_shift_A"] = shift
    result.notes += [f"frequencies: {w}" for w in vr.warnings]
    t = vr.thermo
    log(f"  frequencies ({vr.hessian_method} Hessian): {vr.n_modes} modes, {vr.n_imaginary} imaginary"
        + (f", ZPE {t.zpe_kcal:.2f} kcal/mol, S {t.entropy_cal_molK:.2f} cal/(mol K)" if t else ""))


def run_molecule(inp: MoleculeInput, cfg: BuildConfig, device: str = "cpu", threads: int = 1,
                 should_stop: Callable[[], bool] = lambda: False,
                 log: Callable[[str], None] = _noop, stage: Callable[[str], None] = _noop) -> MoleculeResult:
    result = MoleculeResult(input=inp, multiplicity=inp.multiplicity, device=device)
    t = time.perf_counter()
    try:
        stage("prepare")
        prepared, notes = prepare(inp.to_mol(), cfg.ph if cfg.protonate else None, cfg.keep_largest_fragment,
                                  cfg.protonation_method)
        result.notes += notes
        result.smiles = Chem.MolToSmiles(Chem.RemoveHs(prepared))
        result.charge = inp.charge if inp.charge is not None else formal_charge(prepared)
        numbers = [a.GetAtomicNum() for a in prepared.GetAtoms()]
        method = cfg.method
        if method == "auto":
            method = engines.resolve_auto(numbers)
        else:
            ok, why = engines.available(method)
            if ok:
                ok, why = engines.supports(method, numbers)
            if not ok:
                fallback = engines.resolve_auto(numbers)
                result.notes.append(f"{method} not usable ({why}); using {fallback}")
                method = fallback
        result.method = method
        if method == "mmff" or not engines.ENGINES[method].gpu_capable:
            result.device = "cpu"
        result.timings["prepare"] = time.perf_counter() - t
        log(f"{inp.id}: {result.smiles} (charge {result.charge:+d}) -> {engines.ENGINES[method].label} on {result.device}")
        for n in notes:
            log(f"  {n}")

        stage("conformers")
        t = time.perf_counter()
        # Keep a wider force-field window: the refined ranking can differ from MMFF's.
        cs = confs.generate(prepared, cfg.max_conformers, cfg.energy_window_kcal + 4.0, cfg.rmsd_threshold,
                            cfg.seed, threads)
        result.forcefield, result.n_embedded = cs.forcefield, cs.n_embedded
        result.timings["conformers"] = time.perf_counter() - t
        log(f"  {cs.n_embedded} embedded, {len(cs.conf_ids)} unique after {cs.forcefield} + pruning")
        if should_stop():
            raise Cancelled()

        stage("refine")
        t = time.perf_counter()
        candidates = [method]
        if cfg.method == "auto":  # if the first choice cannot start (e.g. model download offline), try the next
            order = ("aimnet2", "gfn2-xtb", "gfn1-xtb", "mmff")
            candidates += [k for k in order[order.index(method) + 1:]
                           if k == "mmff" or (engines.available(k)[0] and engines.supports(k, numbers)[0])]
        for n_try, candidate in enumerate(candidates):
            result.method = candidate
            result.device = device if engines.ENGINES[candidate].gpu_capable else "cpu"
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
