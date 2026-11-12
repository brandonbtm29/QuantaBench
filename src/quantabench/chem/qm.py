"""QM stage: DFT (or HF) on the lowest conformers with Psi4, run in its own Python environment.

CONTRACT (implemented by the qm builder):

    run_stage(result, cs, cfg, *, threads, should_stop, log) -> None

Fills ``result.qm`` (a JSON-ready dict) and adds ``qm_*`` descriptors to ``result.descriptors``.
Called by ``pipeline.run_molecule`` after refinement when ``cfg.qm_method`` is set; raising is fine (the
pipeline turns it into a note). ``cfg`` fields: ``qm_method`` ("functional/basis", e.g. "wb97x-d/def2-svp",
"b3lyp-d3bj/def2-tzvp", "hf/6-31g*"), ``qm_task`` ("energy" | "optimize"), ``qm_top`` (conformers, lowest
first), ``qm_properties``, ``qm_frequencies`` (best QM conformer only), ``qm_cubes`` (best only), ``solvent``
(PCM via ``solvents.for_engine(s, "pcm")`` when Psi4 has PCMSolver; otherwise note "gas phase").

How Psi4 runs: ``external.find_python_with("psi4", "psi4_python")`` gives an interpreter that can import
psi4 (often a separate conda env). ``chem/psi4_driver.py`` (stdlib + numpy + psi4 only, no quantabench
imports) is run with that interpreter as ``python -I psi4_driver.py task.json`` inside ``external.work_dir``;
it writes ``result.json`` (and cube files). Threads via ``psi4.set_num_threads`` and the environment,
memory from ``settings.engines.qm_memory_gb``, scratch via ``PSI_SCRATCH``.

``result.qm`` keys (QM_RESULT_KEYS): level, program, program_version, task, solvent, conformers (list of
{conf_id, energy_ev, rel_kcal, converged, positions_A (optimize only)}), energy_ev (lowest), properties
(best conformer: dipole_debye, dipole_vector_debye, quadrupole_debye_A, homo_ev, lumo_ev, gap_ev,
charges {mulliken, lowdin, mbis}, mbis_volumes_bohr3, polarizability_iso_A3 when computed), files
({"density": path, "esp": path, "log": path} in a scratch folder: ``export.write_molecule`` copies them next to
the other results), surface (dict from ``surfaces.surface_descriptors`` when cubes were made, with the ESP
distribution under ``esp_profile``; export writes it as ``<stem>_esp_profile.csv``), timings,
warnings. Frequencies: ``result.vibrations``-compatible dict under ``qm["vibrations"]``.

IMPLEMENTATION NOTES

- A driver still running ``EXIT_GRACE_S`` after it wrote ``result.json`` is stopped and its result used, with a
  warning (Psi4 1.9.1 can hang at exit once a failed SCF left its timers on).
- One driver process per molecule runs every conformer and then the extras of the lowest-energy one
  (properties, polarizability, cubes, Hessian). That saves the Psi4 import per conformer, keeps the best
  wavefunction in memory (charges, orbitals and cubes need no new SCF) and lets the polarizability SCFs start
  from its orbitals. Progress lines from the driver are forwarded to ``log`` while it runs; cancellation kills
  the process; ``settings.engines.timeout_s`` (> 0) limits the whole QM stage of one molecule.
- Polarizability: analytic CPSCF when Psi4 has it for the functional (HF, LDA/GGA/hybrids incl. wB97X-D),
  finite field otherwise (meta-GGA, VV10, double hybrids, PCM). ``properties["polarizability_method"]`` says
  which. Anisotropy = largest - smallest eigenvalue, the same convention as the xTB descriptors.
- Solvent: IEFPCM single points. Psi4 has no analytic PCM gradients, so optimizations and Hessians run in the
  gas phase (a warning says so).
- Frequencies: the best conformer is re-optimized tightly (gau_tight) at the QM level first, like the
  pipeline's own frequency step; the Hessian (analytic for HF/LDA, finite differences of gradients otherwise)
  and dipole derivatives come from Psi4, the normal modes, IR intensities and qRRHO thermochemistry from
  ``chem.vibrations`` here, so ``qm["vibrations"]`` has the same layout as ``result.vibrations``.
- Files kept for export (``qm["files"]``): ``log`` (Psi4's output.dat), ``density``/``esp`` (cubes) and, for
  ``qm_task="optimize"``, ``geometry`` (multi-frame XYZ of the QM-optimized conformers, lowest first). The
  stored RDKit conformers keep the refinement geometries, as for the pipeline's frequency step.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .engines import BOHR_ANG, HARTREE_EV

QM_RESULT_KEYS = ("level", "program", "program_version", "task", "solvent", "conformers", "energy_ev", "properties",
                  "files", "surface", "vibrations", "timings", "warnings")

KCAL_PER_EV = 23.060548
DRIVER = Path(__file__).with_name("psi4_driver.py")
TASK_FILE, RESULT_FILE, LOG_FILE, PSI4_OUTPUT = "task.json", "result.json", "driver.log", "output.dat"
PROGRESS = "QB| "          # progress prefix printed by psi4_driver.say()
TASKS = ("energy", "optimize")
#: seconds a driver may keep running after it wrote result.json before it is stopped (run_driver)
EXIT_GRACE_S = 60.0

# Spellings people use for the same thing -> Psi4's name.
METHOD_ALIASES = {"ωb97x-d": "wb97x-d", "wb97xd": "wb97x-d", "ωb97x-d3": "wb97x-d3", "ωb97m-v": "wb97m-v",
                  "ωb97x-v": "wb97x-v", "ωb97x": "wb97x", "b3lyp-d3(bj)": "b3lyp-d3bj", "pbe0-d3(bj)": "pbe0-d3bj",
                  "rhf": "hf", "uhf": "hf", "scf": "hf", "pbeh": "pbe0"}
BASIS_ALIASES = {"6-31g(d)": "6-31g*", "6-31g(d,p)": "6-31g**", "6-31+g(d)": "6-31+g*", "6-31+g(d,p)": "6-31+g**",
                 "6-311g(d)": "6-311g*", "6-311g(d,p)": "6-311g**", "6-311+g(d,p)": "6-311+g**",
                 "def2svp": "def2-svp", "def2tzvp": "def2-tzvp", "def2svpd": "def2-svpd", "def2tzvpp": "def2-tzvpp",
                 "def2qzvp": "def2-qzvp"}


def available() -> tuple[bool, str]:
    from . import external

    if external.find_python_with("psi4", "psi4_python") is None:
        return False, "Psi4 not found (" + external.PROGRAMS["psi4"].install_hint + ")"
    return True, ""


# ----------------------------------------------------------------------------------- level and task
def parse_level(level: str) -> tuple[str, str]:
    """'wB97X-D/def2-SVP' -> ('wb97x-d', 'def2-svp'). Raises ValueError with an example on bad input."""
    text = (level or "").strip().lower().replace(" ", "")
    example = "e.g. 'wb97x-d/def2-svp', 'b3lyp-d3bj/def2-tzvp' or 'hf/6-31g*'"
    if text.count("/") != 1:
        raise ValueError(f"QM level must be 'method/basis' ({example}), not {level!r}")
    method, basis = text.split("/")
    if not method or not basis:
        raise ValueError(f"QM level {level!r} lacks a {'method' if not method else 'basis set'} ({example})")
    if not re.fullmatch(r"[\w+\-*(),ω]+", method) or not re.fullmatch(r"[\w+\-*(),]+", basis):
        raise ValueError(f"QM level {level!r} has unexpected characters ({example})")
    return METHOD_ALIASES.get(method, method), BASIS_ALIASES.get(basis, basis)


def level_string(level: str) -> str:
    method, basis = parse_level(level)
    return f"{method}/{basis}"


def _pcm_solvent(solvent: str, warnings: list[str]) -> tuple[str, str | None]:
    """(canonical solvent, PCMSolver name or None)."""
    from . import solvents

    name = solvents.canonical(solvent)
    if not name:
        return "", None
    pcm = solvents.for_engine(name, "pcm")
    if pcm is None:
        warnings.append(f"PCMSolver has no parameters for {solvents.SOLVENTS[name].label}: QM in the gas phase")
    return name, pcm


def build_task(result, cs, cfg, threads: int = 1, warnings: list[str] | None = None) -> dict:
    """The driver's task for ``result``'s ``cfg.qm_top`` lowest conformers (atom order of ``result.mol``)."""
    from . import export, external

    warnings = warnings if warnings is not None else []
    method, basis = parse_level(cfg.qm_method)
    task_kind = (cfg.qm_task or "energy").lower()
    if task_kind not in TASKS:
        raise ValueError(f"qm_task must be one of {TASKS}, not {cfg.qm_task!r}")
    program = (getattr(cfg, "qm_program", "") or "psi4").lower()
    if program != "psi4":
        raise ValueError(f"QM program {program!r} is not supported (only psi4)")
    mol = result.mol if result.mol is not None else cs.mol
    if not result.conformers:
        raise ValueError("no conformers to run the QM stage on")
    top = result.conformers[: max(1, int(cfg.qm_top or 1))]
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    conformers = [{"id": int(c.conf_id), "positions_A": np.round(export.mol_to_atoms(mol, c.conf_id).positions, 8).tolist()}
                  for c in top]
    solvent, pcm = _pcm_solvent(getattr(cfg, "solvent", ""), warnings)
    es = external._engine_settings()
    return {
        "symbols": symbols, "conformers": conformers, "charge": int(result.charge),
        "multiplicity": int(result.multiplicity or 1), "method": method, "basis": basis, "task": task_kind,
        "properties": bool(cfg.qm_properties), "polarizability": bool(cfg.qm_properties), "cubes": bool(cfg.qm_cubes),
        "hessian": bool(cfg.qm_frequencies) and len(symbols) > 1, "freq_optimize": True,
        "threads": max(1, int(threads)), "memory_gb": float(getattr(es, "qm_memory_gb", 4.0) or 4.0),
        "solvent_pcm": pcm, "solvent": solvent if pcm else "", "temperature_k": float(cfg.temperature_k),
        "output": PSI4_OUTPUT,
    }


# ----------------------------------------------------------------------------------- running
class _ProgressPump:
    """Forwards the driver's progress lines (``QB| ...``) from its log file to ``log`` while it runs.

    ``external.run_program`` polls ``should_stop`` every 0.2 s; calling the pump from there streams progress
    without a second thread."""

    def __init__(self, path: Path, log: Callable[[str], None]):
        self.path, self.log, self.pos, self.buf = path, log, 0, ""

    def __call__(self) -> bool:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
                self.pos = fh.tell()
        except OSError:
            return False
        if chunk:
            self.buf += chunk.decode("utf-8", errors="replace")
            *lines, self.buf = self.buf.split("\n")
            for line in lines:
                if line.startswith(PROGRESS):
                    self.log("  QM: " + line[len(PROGRESS):].rstrip())
        return False


def run_driver(task: dict, wd: Path, python: str, *, threads: int = 1, timeout: float | None = None,
               should_stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = lambda _m: None) -> dict:
    """Write ``task`` into ``wd``, run the driver there and return its result (``ExternalProgramError`` if it
    failed or wrote nothing usable)."""
    from . import external
    from .optimize import Cancelled

    wd = Path(wd)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / TASK_FILE).write_text(json.dumps(task), encoding="utf-8")
    shutil.copy2(DRIVER, wd / DRIVER.name)
    pump = _ProgressPump(wd / LOG_FILE, log)
    res_path = wd / RESULT_FILE
    seen: dict[str, float] = {}

    def finished_but_running() -> bool:
        # the driver writes result.json (atomically) as its last act; a Psi4 process still running EXIT_GRACE_S
        # later is stuck (seen after a failed SCF left Psi4's timers on: it hangs at exit) and is stopped
        if "at" not in seen and res_path.is_file():
            seen["at"] = time.monotonic()
        return "at" in seen and time.monotonic() - seen["at"] > EXIT_GRACE_S

    hung = False
    try:
        run = external.run_program([python, "-I", DRIVER.name, TASK_FILE], wd, threads=threads,
                                   env=external.program_env(threads, scratch=wd), timeout=timeout, log_name=LOG_FILE,
                                   should_stop=lambda: pump() or should_stop() or finished_but_running(), check=False)
    except Cancelled:
        if should_stop() or not res_path.is_file():
            raise
        hung = True
        run = external.ProgramRun(-1, 0.0, wd / LOG_FILE)
    pump()
    try:
        out = json.loads(res_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise external.ExternalProgramError(f"Psi4 driver exited with code {run.returncode} without a result\n"
                                            f"{run.tail()}") from None
    if not out.get("ok"):
        raise external.ExternalProgramError(f"Psi4: {out.get('error', 'unknown error')}\n{run.tail(15)}")
    if hung:
        out.setdefault("warnings", []).append(f"the Psi4 driver did not exit {EXIT_GRACE_S:g} s after writing its "
                                              "result and was stopped (the result is complete)")
    return out


def _keep_dir(stem: str) -> Path:
    """A scratch folder that outlives the job folder, for the files ``export.write_qm`` copies later."""
    from . import external

    safe = external.ascii_name(stem, 40)      # ASCII only: Psi4 on Windows opens files through ANSI paths
    return Path(tempfile.mkdtemp(prefix=f"qm_{safe}_", dir=external.scratch_root()))


def _keep_file(src: Path, dst: Path, warnings: list[str]) -> str | None:
    """Move a result file out of the job folder; copy if moving fails; a warning (not a lost QM result)
    if neither works (a Windows sharing violation, a full disk)."""
    try:
        return str(shutil.move(str(src), dst))
    except OSError:
        try:
            return str(shutil.copy2(src, dst))
        except OSError as exc:
            warnings.append(f"could not keep {src.name} ({exc})")
            return None


def _write_xyz(path: Path, symbols, frames: list[tuple[str, list]]) -> Path:
    lines = []
    for comment, pos in frames:
        lines += [str(len(symbols)), comment]
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, pos)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------------- frequencies
def vibrations_from_hessian(freq: dict, numbers, cfg, charge: int, multiplicity: int, level: str):
    """``VibrationalResult`` from the driver's Hessian (Eh/bohr²) and dipole derivatives, with the same normal
    modes, IR intensities, imaginary-mode handling and (q)RRHO thermochemistry as ``vibrations.analyze``."""
    from ase import Atoms

    from . import vibrations as vib

    numbers = np.asarray(numbers, int)
    pos = np.asarray(freq["positions_A"], float)
    n = len(numbers)
    masses = vib.isotope_masses(numbers)
    e_pot = float(freq["energy_hartree"]) * HARTREE_EV
    sf = vib.SCALE_FACTORS.get(level.lower())
    imag_tol = 20.0
    warnings: list[str] = []
    grad = freq.get("gradient_au")
    fmax = 0.0
    if grad is not None:
        fmax = float(np.sqrt((np.asarray(grad, float).reshape(-1, 3) ** 2).sum(1)).max()) * HARTREE_EV / BOHR_ANG
        if fmax > 0.05:
            warnings.append(f"max force {fmax:.3f} eV/Å: not a stationary point, frequencies are approximate")
    h = vib.symmetrize(np.asarray(freq["hessian_au"], float).reshape(3 * n, 3 * n) * HARTREE_EV / BOHR_ANG**2)
    geometry = vib.detect_geometry(pos, masses)
    sigma = int(cfg.symmetry_number or 0) or vib.rotational_symmetry_number(numbers, pos, masses)
    nm = vib.normal_modes(h, masses, pos, geometry)
    freqs = nm.frequencies_cm1
    dd = freq.get("dipole_derivatives_au")
    ints, ir_source = None, ""
    if dd is not None:
        ints = vib.ir_intensities(np.asarray(dd, float).reshape(3 * n, 3).T, nm, masses)
        kind = freq.get("hessian_kind", "")
        ir_source = ("analytic dipole derivatives (Psi4)" if kind == "analytic"
                     else "dipole derivatives by finite differences (Psi4)")
    else:
        warnings.append("IR intensities: Psi4 gave no dipole derivatives for this method")
    n_imag = int((freqs < -imag_tol).sum())
    small = freqs[(freqs < 0) & (freqs >= -imag_tol)]
    if n_imag:
        warnings.append(f"{n_imag} imaginary frequenc{'y' if n_imag == 1 else 'ies'} (lowest {freqs[0]:.1f} cm-1): "
                        "not a minimum; imaginary modes are excluded from the thermochemistry")
    if len(small):
        warnings.append(f"{len(small)} small imaginary mode(s) above -{imag_tol:g} cm-1 treated as numerical noise "
                        "(used as real frequencies in the thermochemistry)")
    psi4_nu = freq.get("psi4_frequencies_cm1")
    if psi4_nu is not None and len(psi4_nu) == len(freqs):
        dev = float(np.abs(np.sort(freqs) - np.sort(np.asarray(psi4_nu, float))).max()) if len(freqs) else 0.0
        if dev > 2.0:
            warnings.append(f"frequencies differ from Psi4's own analysis by up to {dev:.1f} cm-1")
    thermo_freqs = np.abs(freqs[freqs >= -imag_tol])
    thermo_freqs = thermo_freqs[thermo_freqs > 1e-3]
    thermo = vib.thermochemistry(Atoms(numbers, positions=pos), thermo_freqs, e_pot, cfg.temperature_k, 101325.0,
                                 sigma, multiplicity, geometry, masses, cfg.freq_qrrho)
    hess_method = "analytic" if freq.get("hessian_kind") == "analytic" else "finite"
    return vib.VibrationalResult(
        method=level, hessian_method=hess_method, charge=charge, multiplicity=multiplicity, device="cpu", n_atoms=n,
        geometry=geometry, frequencies_cm1=[float(f) for f in freqs],
        reduced_masses_amu=[float(x) for x in nm.reduced_masses_amu],
        force_constants_mdyn_A=[float(x) for x in nm.force_constants_mdyn_A],
        ir_intensities_km_mol=None if ints is None else [float(x) for x in ints], ir_source=ir_source,
        n_imaginary=n_imag, thermo=thermo, fd_step_A=None, max_force_ev_A=fmax,
        scale_factor=sf.value if sf else 1.0, scale_factor_source=sf.source if sf else "", imag_tol_cm1=imag_tol,
        warnings=warnings, timings={}, displacements=nm.displacements, hessian_ev_A2=h, positions=pos)


def _monatomic_vibrations(numbers, positions, e_hartree: float, cfg, charge: int, multiplicity: int, level: str):
    from ase import Atoms

    from . import vibrations as vib

    masses = vib.isotope_masses(numbers)
    atoms = Atoms(numbers, positions=positions)
    thermo = vib.thermochemistry(atoms, [], e_hartree * HARTREE_EV, cfg.temperature_k, 101325.0, 1, multiplicity,
                                 "monatomic", masses, cfg.freq_qrrho)
    return vib.VibrationalResult(level, "none", charge, multiplicity, "cpu", 1, "monatomic", [], [], [], None, "", 0,
                                 thermo, warnings=["single atom: no vibrations (translational and electronic terms only)"],
                                 positions=np.asarray(positions, float))


# ----------------------------------------------------------------------------------- results
def _surface_prefix(key: str) -> str:
    return "qm_" + key if key.startswith("surf_") else "qm_surf_" + key


def qm_descriptors(qm: dict) -> dict:
    """Scalar ``qm_*`` descriptors from a ``result.qm`` dict."""
    d: dict = {"qm_level": qm.get("level", "")}
    if qm.get("energy_ev") is not None:
        d["qm_energy_ev"] = qm["energy_ev"]
    p = qm.get("properties") or {}
    for key in ("dipole_debye", "homo_ev", "lumo_ev", "gap_ev", "polarizability_iso_A3", "polarizability_aniso_A3"):
        if p.get(key) is not None:
            d[f"qm_{key}"] = float(p[key])
    mbis = (p.get("charges") or {}).get("mbis")
    if mbis:
        d["qm_mbis_charge_min"], d["qm_mbis_charge_max"] = float(min(mbis)), float(max(mbis))
    if p.get("mbis_volumes_bohr3"):
        # sum of atomic <r^3> (bohr^3): the Tkatchenko-Scheffler effective-volume measure, ~ size/polarizability
        d["qm_mbis_r3_sum_bohr3"] = float(sum(p["mbis_volumes_bohr3"]))
    vib = qm.get("vibrations")
    if vib:
        d["qm_n_imaginary"] = vib.get("n_imaginary")
        d["qm_lowest_freq_cm1"] = vib.get("lowest_freq_cm1")
        t = vib.get("thermo") or {}
        for key in ("zpe_kcal", "enthalpy_kcal", "entropy_cal_molK", "gibbs_kcal", "h_corr_kcal", "g_corr_kcal",
                    "symmetry_number"):
            if t.get(key) is not None:
                d[f"qm_{key}"] = t[key]
    for key, val in (qm.get("surface") or {}).items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            d[_surface_prefix(key)] = float(val)
    return d


def _properties(out: dict) -> dict | None:
    p = out.get("properties")
    pol = out.get("polarizability")
    if p is None and pol is None:
        return None
    props = dict(p or {})
    props["conf_id"] = out.get("best_id")
    if pol:
        props.update(polarizability_iso_A3=pol["iso_A3"], polarizability_aniso_A3=pol["aniso_A3"],
                     polarizability_aniso_invariant_A3=pol["aniso_invariant_A3"],
                     polarizability_tensor_A3=pol["tensor_A3"], polarizability_method=pol["method"])
    return props


def run_stage(result, cs, cfg, *, threads: int = 1, should_stop: Callable[[], bool] = lambda: False,
              log: Callable[[str], None] = lambda _m: None) -> None:
    from . import external

    t_start = time.perf_counter()
    warnings: list[str] = []
    task = build_task(result, cs, cfg, threads, warnings)
    level = f"{task['method']}/{task['basis']}"
    python = external.find_python_with("psi4", "psi4_python")
    if python is None:
        raise external.ExternalProgramError("Psi4 not found: " + external.PROGRAMS["psi4"].install_hint
                                            + ", then set its Python in Settings > Engines (psi4_python) or "
                                            "QUANTABENCH_PSI4_PYTHON")
    es = external._engine_settings()
    timeout = float(getattr(es, "timeout_s", 0) or 0) or None
    n_conf = len(task["conformers"])
    extras = [name for name, on in (("properties", task["properties"]), ("cubes", task["cubes"]),
                                    ("frequencies", task["hessian"])) if on]
    log(f"  QM {level} (Psi4): {task['task']} of {n_conf} conformer(s)"
        + (f", then {', '.join(extras)} of the best" if extras else "")
        + (f", PCM {task['solvent']}" if task["solvent_pcm"] else "") + f", {task['threads']} thread(s)")
    for w in warnings:
        log(f"  QM: {w}")
    n_logged = len(warnings)

    with external.work_dir(f"psi4_{result.input.id}") as wd:
        out = run_driver(task, wd, python, threads=task["threads"], timeout=timeout, should_stop=should_stop, log=log)
        warnings += out.get("warnings", [])
        files: dict[str, str] = {}
        try:
            keep = _keep_dir(result.input.id)
        except OSError as exc:
            keep = None
            warnings.append(f"could not keep the QM files ({exc})")
        if keep is not None:
            kept = []
            if (wd / PSI4_OUTPUT).is_file():
                kept.append(("log", wd / PSI4_OUTPUT, keep / "psi4.out"))
            cubes = out.get("cubes") or {}
            for label in ("density", "esp"):
                if cubes.get(label) and (wd / cubes[label]).is_file():
                    kept.append((label, wd / cubes[label], keep / f"{label}.cube"))
            for label, src, dst in kept:
                path = _keep_file(src, dst, warnings)
                if path is not None:
                    files[label] = path

    # conformers: energies (eV), relative energies, optimized geometries
    e_ref = out.get("energy_hartree")
    confs = []
    for rec in out["conformers"]:
        e = rec.get("energy_hartree")
        c = {"conf_id": rec["id"], "energy_ev": None if e is None else e * HARTREE_EV,
             "rel_kcal": None if e is None else (e - e_ref) * HARTREE_EV * KCAL_PER_EV,
             "converged": bool(rec.get("converged")), "energy_hartree": e, "seconds": rec.get("seconds")}
        if rec.get("positions_A") is not None:
            c["positions_A"] = rec["positions_A"]
        if rec.get("error"):
            c["error"] = rec["error"]
            warnings.append(f"conformer {rec['id']}: {rec['error']}")
        confs.append(c)
    symbols = task["symbols"]
    frames = sorted((c for c in confs if c.get("positions_A") and c["energy_ev"] is not None),
                    key=lambda c: c["energy_ev"])
    if frames:      # optimized geometries (qm_task="optimize")
        files["geometry"] = str(_write_xyz(keep / "geometry.xyz", symbols, [
            (f"{result.input.id} conf {c['conf_id']} {level} E = {c['energy_ev']:.6f} eV rel {c['rel_kcal']:.2f} kcal/mol",
             c["positions_A"]) for c in frames]))

    qm: dict = {"level": level, "program": out.get("program", "psi4"), "program_version": out.get("program_version", ""),
                "task": task["task"], "solvent": out.get("solvent", ""), "solvent_model": out.get("solvent_model", ""),
                "reference": out.get("reference", ""), "conformers": confs, "best_conf_id": out.get("best_id"),
                "energy_ev": None if e_ref is None else e_ref * HARTREE_EV, "properties": _properties(out),
                "files": files, "surface": None, "vibrations": None, "cubes": None, "warnings": warnings,
                "timings": {**(out.get("timings") or {})}}
    if cubes:
        qm["cubes"] = {k: v for k, v in cubes.items() if k not in ("density", "esp")}

    # frequencies -> result.vibrations-compatible dict
    numbers = [a.GetAtomicNum() for a in (result.mol if result.mol is not None else cs.mol).GetAtoms()]
    if cfg.qm_frequencies:
        try:
            if len(numbers) == 1:
                vr = _monatomic_vibrations(numbers, out["best_positions_A"], e_ref, cfg, result.charge,
                                           result.multiplicity, level)
            elif out.get("frequencies"):
                vr = vibrations_from_hessian(out["frequencies"], numbers, cfg, result.charge, result.multiplicity, level)
            else:
                vr = None
            if vr is not None:
                vd = vr.to_dict(include_modes=True)
                pos = np.asarray(vr.positions, float)
                vd["symbols"] = symbols
                vd["positions_A"] = np.round(pos, 6).tolist()
                ref = np.asarray(next(c["positions_A"] for c in task["conformers"] if c["id"] == out.get("best_id")))
                vd["geometry_max_shift_A"] = float(np.sqrt(((pos - ref) ** 2).sum(1)).max()) if len(pos) else 0.0
                fr = out.get("frequencies") or {}
                vd.update(program="psi4", hessian_kind=fr.get("hessian_kind", ""),
                          psi4_frequencies_cm1=fr.get("psi4_frequencies_cm1"),
                          psi4_ir_km_mol=fr.get("psi4_ir_km_mol"), reoptimized=fr.get("optimized", False),
                          reoptimization_converged=fr.get("opt_converged"))
                qm["vibrations"] = vd
                warnings += [f"frequencies: {w}" for w in vr.warnings]
                t = vr.thermo
                log(f"  QM frequencies ({fr.get('hessian_kind', vr.hessian_method)} Hessian): {vr.n_modes} modes, "
                    f"{vr.n_imaginary} imaginary" + (f", ZPE {t.zpe_kcal:.2f} kcal/mol" if t else ""))
        except Exception as exc:
            warnings.append(f"QM frequencies skipped ({type(exc).__name__}: {exc})")

    # molecular surface from the cubes (chem.surfaces may not be implemented yet)
    if "density" in files and "esp" in files:
        from . import surfaces

        try:
            qm["surface"] = dict(surfaces.surface_descriptors(files["density"], files["esp"], profile=True),
                                 iso_au=0.002)
        except NotImplementedError:
            warnings.append("surface descriptors are not available yet (chem.surfaces is not implemented); "
                            "the density and ESP cubes are kept")
        except Exception as exc:
            warnings.append(f"surface descriptors failed ({type(exc).__name__}: {exc})")

    qm["timings"]["wall"] = time.perf_counter() - t_start
    result.qm = qm
    result.descriptors.update(qm_descriptors(qm))
    result.notes += [f"QM: {w}" for w in warnings if not w.startswith("pint ")]
    p = qm["properties"] or {}
    log(f"  QM done: E = {qm['energy_ev']:.5f} eV (conf {qm['best_conf_id']})"
        + (f", dipole {p['dipole_debye']:.2f} D" if p.get("dipole_debye") is not None else "")
        + (f", gap {p['gap_ev']:.2f} eV" if p.get("gap_ev") is not None else "")
        + (f", polarizability {p['polarizability_iso_A3']:.2f} Å^3" if p.get("polarizability_iso_A3") is not None else "")
        + f" [{qm['timings']['wall']:.1f} s]")
    for w in warnings[n_logged:]:
        log(f"  QM: {w}")
