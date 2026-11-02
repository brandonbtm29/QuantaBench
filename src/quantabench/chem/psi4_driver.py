"""Psi4 driver for QuantaBench's QM stage. Standalone: stdlib + numpy + psi4, no quantabench imports.

Run with an interpreter that can import psi4 (usually its own conda environment), inside a job folder:

    <psi4 python> -I psi4_driver.py task.json

It writes ``result.json`` (and Psi4's ``output.dat``, cube files under ``cubes/``) next to the task and
exits 0. On failure it still writes ``result.json`` with ``"ok": false`` and an ``"error"`` message,
prints the message to stderr and exits 1 (2 for an invalid task file). Progress lines start with
``QB| `` on stdout; ``chem.qm`` forwards them to the GUI/CLI log while the job runs.

One process handles all conformers of a molecule: that saves the Psi4 import (~1-2 s) per conformer, keeps
the best conformer's wavefunction in memory (so charges, orbitals and cubes need no extra SCF), and lets
the polarizability SCFs start from its converged orbitals (``guess read``). A failed conformer (SCF or
optimizer not converging) is recorded and the others continue.

Task (JSON object)
------------------
symbols         ["O", "H", "H"]
conformers      [{"id": 3, "positions_A": [[x, y, z], ...]}, ...]   (or a single top-level positions_A)
charge, multiplicity
method, basis   Psi4 names, e.g. "wb97x-d" / "def2-svp" (HF or a DFT functional)
task            "energy" | "optimize" (optking)
properties      dipole, quadrupole, Mulliken/Lowdin/MBIS charges, MBIS volumes, Wiberg indices, orbitals
polarizability  static dipole polarizability of the best conformer
cubes           density (Dt.cube) and ESP (ESP.cube) of the best conformer
hessian         Hessian (+ dipole derivatives when Psi4 provides them) of the best conformer, gas phase
threads, memory_gb
solvent_pcm     PCMSolver solvent name, or null for the gas phase
optional        solvent (label), freq_optimize (true: tight re-optimization before the Hessian),
                polarizability_method ("auto" | "analytic" | "finite_field"), field_au (0.002),
                cube_spacing_bohr (0.2), cube_overage_bohr (4.0), temperature_k (298.15),
                opt_convergence ("gau"), geom_maxiter (100), psi4_options ({}), output ("output.dat")

Psi4 notes (1.9)
----------------
- Static polarizability: analytic CPSCF through ``psi4.properties(..., ["dipole_polarizabilities"])`` works
  for RHF/RKS including range-separated hybrids (wB97X-D); it is not available for meta-GGAs, VV10 and
  double hybrids, and with PCM the response of the reaction field is not part of it. Those cases (or
  ``polarizability_method="finite_field"``) use finite fields: 6 SCFs with ``perturb_dipole`` = ±F along
  x, y, z from the converged orbitals, alpha_ij = -dmu_i/dF_j by central differences.
- PCM has no analytic gradients in Psi4 (it falls back to finite differences of energies, ~6N energies per
  gradient), so optimizations and Hessians run in the gas phase and the energies/properties are PCM single
  points at those geometries.
- Hessians: analytic for HF and LDA, finite differences of analytic gradients otherwise (2(3N-6)+1
  gradients in C1). Both give dipole derivatives (IR intensities).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROGRESS = "QB| "
HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903
AU_DEBYE = 2.541746473                    # e*a0 -> debye (CODATA 2018)
AU_POLARIZABILITY_A3 = 0.1481847          # a0^3 -> A^3
MOL_NAME = "qb"

TASKS = ("energy", "optimize")
POL_METHODS = ("auto", "analytic", "finite_field")
DFTD3_HINT = ("the -D3 dispersion correction needs simple-dftd3: conda install -c conda-forge dftd3-python "
              "(in the Psi4 environment)")
DFTD4_HINT = "the -D4 dispersion correction needs dftd4: conda install -c conda-forge dftd4-python"


class TaskError(ValueError):
    """The task file is invalid."""


def say(msg: str) -> None:
    print(PROGRESS + msg, flush=True)


# ----------------------------------------------------------------------------------- task
def load_task(path: str | Path) -> dict:
    """Read and validate a task file; returns it with defaults filled in."""
    try:
        task = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskError(f"cannot read task file {path}: {exc}") from exc
    return validate_task(task)


def validate_task(task: dict) -> dict:
    if not isinstance(task, dict):
        raise TaskError("task must be a JSON object")
    t = dict(task)
    symbols = t.get("symbols")
    if not symbols or not all(isinstance(s, str) and s for s in symbols):
        raise TaskError("symbols must be a non-empty list of element symbols")
    n = len(symbols)
    confs = t.get("conformers")
    if not confs:
        if t.get("positions_A") is None:
            raise TaskError("give conformers [{id, positions_A}] or positions_A")
        confs = [{"id": 0, "positions_A": t["positions_A"]}]
    out = []
    for i, c in enumerate(confs):
        pos = np.asarray(c.get("positions_A"), dtype=float)
        if pos.shape != (n, 3) or not np.all(np.isfinite(pos)):
            raise TaskError(f"conformer {i}: positions_A must be {n} x 3 finite numbers, got shape {pos.shape}")
        out.append({"id": c.get("id", i), "positions_A": pos.tolist()})
    t["conformers"] = out
    t["charge"] = int(t.get("charge", 0))
    t["multiplicity"] = int(t.get("multiplicity", 1))
    if t["multiplicity"] < 1:
        raise TaskError("multiplicity must be >= 1")
    for key in ("method", "basis"):
        if not str(t.get(key) or "").strip():
            raise TaskError(f"{key} is missing (a level is 'method/basis', e.g. wb97x-d/def2-svp)")
        t[key] = str(t[key]).strip().lower()
    t["task"] = str(t.get("task", "energy")).lower()
    if t["task"] not in TASKS:
        raise TaskError(f"task must be one of {TASKS}, not {t['task']!r}")
    for key in ("properties", "polarizability", "cubes", "hessian"):
        t[key] = bool(t.get(key, False))
    t["freq_optimize"] = bool(t.get("freq_optimize", True))
    t["threads"] = max(1, int(t.get("threads", 1)))
    t["memory_gb"] = max(0.5, float(t.get("memory_gb", 2.0)))
    t["solvent_pcm"] = (str(t["solvent_pcm"]).strip() or None) if t.get("solvent_pcm") else None
    t["solvent"] = str(t.get("solvent") or t["solvent_pcm"] or "")
    t["polarizability_method"] = str(t.get("polarizability_method", "auto")).lower()
    if t["polarizability_method"] not in POL_METHODS:
        raise TaskError(f"polarizability_method must be one of {POL_METHODS}")
    t["field_au"] = float(t.get("field_au", 0.002))
    t["cube_spacing_bohr"] = float(t.get("cube_spacing_bohr", 0.2))
    t["cube_overage_bohr"] = float(t.get("cube_overage_bohr", 4.0))
    t["temperature_k"] = float(t.get("temperature_k", 298.15))
    t["opt_convergence"] = str(t.get("opt_convergence", "gau"))
    t["geom_maxiter"] = int(t.get("geom_maxiter", 100))
    t["psi4_options"] = dict(t.get("psi4_options") or {})
    t["output"] = str(t.get("output", "output.dat"))
    return t


def reference_for(method: str, multiplicity: int) -> str:
    """'rhf'/'uhf' for Hartree-Fock, 'rks'/'uks' for DFT (unrestricted for open shells)."""
    hf = method in ("hf", "scf")
    if multiplicity > 1:
        return "uhf" if hf else "uks"
    return "rhf" if hf else "rks"


# ----------------------------------------------------------------------------------- numerics
def alpha_from_dipoles(mu_plus: np.ndarray, mu_minus: np.ndarray, field: float) -> tuple[np.ndarray, int]:
    """Polarizability (a.u.) from dipoles at +F and -F along x, y, z (rows: field direction).

    Central difference d mu_i / d F_j; the sign of Psi4's perturbation convention is fixed by requiring a
    positive trace (a ground-state static polarizability is positive definite). Returns (alpha, sign)."""
    raw = (np.asarray(mu_plus, float) - np.asarray(mu_minus, float)).T / (2.0 * field)   # [i, j] = dmu_i/dF_j
    sign = 1 if np.trace(raw) >= 0 else -1
    alpha = sign * raw
    return 0.5 * (alpha + alpha.T), sign


def polarizability_summary(alpha_au: np.ndarray) -> dict:
    """iso = trace/3; aniso = largest - smallest eigenvalue (QuantaBench's convention, as for xTB);
    aniso_invariant = sqrt(((a1-a2)^2 + (a2-a3)^2 + (a3-a1)^2)/2), the usual Raman invariant."""
    a = 0.5 * (np.asarray(alpha_au, float) + np.asarray(alpha_au, float).T)
    eig = np.linalg.eigvalsh(a)
    inv = math.sqrt(0.5 * ((eig[0] - eig[1]) ** 2 + (eig[1] - eig[2]) ** 2 + (eig[2] - eig[0]) ** 2))
    iso = float(np.trace(a) / 3.0)
    return {"tensor_au": a.tolist(), "eigenvalues_au": eig.tolist(), "iso_au": iso,
            "aniso_au": float(eig[-1] - eig[0]), "aniso_invariant_au": inv,
            "iso_A3": iso * AU_POLARIZABILITY_A3, "aniso_A3": float(eig[-1] - eig[0]) * AU_POLARIZABILITY_A3,
            "aniso_invariant_A3": inv * AU_POLARIZABILITY_A3,
            "tensor_A3": (a * AU_POLARIZABILITY_A3).tolist()}


def read_cube_header_and_data(path: str | Path) -> tuple[tuple[int, int, int], np.ndarray]:
    """Grid shape and values (nx, ny, nz) of a Gaussian cube file (first value per point)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    natoms = int(lines[2].split()[0])
    shape = tuple(int(lines[3 + k].split()[0]) for k in range(3))
    start = 6 + abs(natoms) + (1 if natoms < 0 else 0)
    values = np.array(" ".join(lines[start:]).split(), dtype=float)
    nvals = shape[0] * shape[1] * shape[2]
    per_point = max(1, values.size // nvals)
    data = values[: nvals * per_point].reshape(shape[0], shape[1], shape[2], per_point)[..., 0]
    return shape, data


def boundary_max(data: np.ndarray) -> float:
    """Largest absolute value on the six faces of a 3D grid."""
    d = np.abs(data)
    faces = (d[0], d[-1], d[:, 0], d[:, -1], d[:, :, 0], d[:, :, -1])
    return float(max(f.max() for f in faces))


def homo_lumo(eps_a, eps_b, nalpha: int, nbeta: int, restricted: bool) -> tuple[float | None, float | None]:
    """HOMO and LUMO energies (Hartree) from orbital energies (ascending) and occupations."""
    ea = np.sort(np.asarray(eps_a, float))
    eb = ea if restricted else np.sort(np.asarray(eps_b, float))
    occ = [ea[nalpha - 1]] if nalpha > 0 else []
    vir = [ea[nalpha]] if nalpha < len(ea) else []
    if not restricted:
        occ += [eb[nbeta - 1]] if nbeta > 0 else []
        vir += [eb[nbeta]] if nbeta < len(eb) else []
    return (float(max(occ)) if occ else None, float(min(vir)) if vir else None)


# ----------------------------------------------------------------------------------- psi4 setup
def pint_compat(warnings: list[str]) -> None:
    """qcelemental 0.30 builds its unit registry from CODATA strings like 'C^3 m^3 J^{-2}', which pint >= 0.25
    cannot parse (TypeError: No loader function defined for UnhandledParsingError); every unit conversion
    then fails, optking included. If that happens, rewrite '^{n}' as '^(n)' in pint definitions for this
    process. The proper fix is ``conda install 'pint<0.25'`` in the Psi4 environment."""
    try:
        import qcelemental
    except ImportError:
        return
    try:
        qcelemental.constants.conversion_factor("bohr", "angstrom")
        return
    except TypeError:
        pass
    import pint

    original = pint.UnitRegistry.define
    braced = re.compile(r"\^\{([^{}]*)\}")

    def define(self, definition, *args, **kwargs):
        if isinstance(definition, str) and "^{" in definition:
            definition = braced.sub(r"^(\1)", definition)
        return original(self, definition, *args, **kwargs)

    pint.UnitRegistry.define = define
    qcelemental.constants._ureg = None
    qcelemental.constants.conversion_factor("bohr", "angstrom")
    warnings.append(f"pint {pint.__version__} is incompatible with qcelemental {qcelemental.__version__} "
                    "(patched for this run; fix the environment with: conda install 'pint<0.25')")


def check_level(psi4, method: str, basis: str, mol) -> None:
    from psi4.driver.procrouting import proc_table
    from psi4.driver.procrouting.dft import functionals

    if method not in ("hf", "scf") and method not in functionals:
        import difflib

        known = method in proc_table.procedures["energy"]
        close = difflib.get_close_matches(method, [*functionals, "hf"], n=3)
        raise TaskError(f"{method!r} is " + ("not an SCF method (the QM stage runs HF or DFT)" if known
                                              else "not a Psi4 method")
                        + (f"; did you mean {', '.join(close)}?" if close else ""))
    try:
        psi4.core.BasisSet.build(mol, "ORBITAL", basis, quiet=True)
    except Exception as exc:
        raise TaskError(f"basis set {basis!r} is not available in Psi4 ({type(exc).__name__}: "
                        f"{str(exc).strip().splitlines()[0] if str(exc).strip() else ''}); e.g. def2-svp, def2-tzvp, "
                        "6-31g*, cc-pvdz") from exc


def make_molecule(psi4, symbols, positions_A, charge: int, multiplicity: int):
    """C1 molecule in the input frame (no recentering or reorientation), so positions, dipoles and the
    Hessian are all in the caller's coordinates and atom order."""
    mol = psi4.core.Molecule.from_arrays(elem=list(symbols), geom=np.asarray(positions_A, float), units="Angstrom",
                                         molecular_charge=charge, molecular_multiplicity=multiplicity,
                                         fix_com=True, fix_orientation=True, fix_symmetry="c1", name=MOL_NAME)
    mol.update_geometry()
    return mol


def positions_A(mol) -> list[list[float]]:
    return (np.asarray(mol.geometry()) * BOHR_ANG).round(8).tolist()


def pcm_input(solvent: str) -> str:
    return f"""
       Units = Angstrom
       Medium {{
         SolverType = IEFPCM
         Solvent = {solvent}
       }}
       Cavity {{
         RadiiSet = UFF
         Type = GePol
         Scaling = False
         Area = 0.3
         Mode = Implicit
       }}
    """


def friendly(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}".strip()
    low = text.lower()
    if "dftd3" in low or "s-dftd3" in low:
        text += f" ({DFTD3_HINT})"
    elif "dftd4" in low:
        text += f" ({DFTD4_HINT})"
    return text


# ----------------------------------------------------------------------------------- steps
def scf_or_opt(psi4, t: dict, mol, pcm: bool) -> tuple[float, object, bool]:
    """Energy or optimization of one geometry. Returns (energy Eh, wavefunction, converged)."""
    method = t["method"]
    if t["task"] == "energy":
        e, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
        return float(e), wfn, True
    if pcm:
        psi4.set_options({"pcm": False})
    converged = True
    try:
        e, wfn = psi4.optimize(method, molecule=mol, return_wfn=True)
    except psi4.OptimizationConvergenceError as exc:
        converged = False
        wfn = exc.wfn
        e = wfn.energy()
        mol.set_geometry(wfn.molecule().geometry())
        mol.update_geometry()
    if pcm:
        psi4.set_options({"pcm": True})
        e, wfn = psi4.energy(method, molecule=mol, return_wfn=True)
    return float(e), wfn, converged


# Unpaired electrons of each free atom's ground state (index = Z), as in psi4.p4util.prop_util.
FREE_ATOM_UNPAIRED = (
    0, 1, 0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0,
    1, 0, 1, 2, 5, 6, 5, 4, 3, 0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 0, 1, 0, 3, 4, 5, 6, 7, 8, 5, 4, 3, 2, 1, 0, 1, 2, 3,
    4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0)


def free_atom_volumes(psi4, t: dict, wfn, pcm: bool) -> dict[str, float]:
    """MBIS <r^3> of each free atom (bohr^3), gas phase, same method and basis; stored on ``wfn`` as
    ``MBIS FREE ATOM <X> VOLUME`` so that oeprop can form the volume ratios.

    Psi4's own helper finds the method by matching energy variables, which fails under PCM (the total energy
    then includes the solvation term), and it clears all global variables; this does the same calculation
    explicitly. One atomic SCF per element; nothing for a single atom."""
    mol = wfn.molecule()
    if mol.natom() == 1:
        return {}
    hf = t["method"] in ("hf", "scf")
    out: dict[str, float] = {}
    if pcm:
        psi4.set_options({"pcm": False})
    try:
        for i in range(mol.natom()):
            sym, z = mol.symbol(i).capitalize(), int(mol.Z(i))
            if sym in out:
                continue
            unpaired = FREE_ATOM_UNPAIRED[z] if z < len(FREE_ATOM_UNPAIRED) else 0
            ref = ("uhf" if hf else "uks") if unpaired else ("rhf" if hf else "rks")
            psi4.set_options({"reference": ref})
            atom = psi4.core.Molecule.from_arrays(elem=[sym], geom=[0.0, 0.0, 0.0], molecular_charge=0,
                                                  molecular_multiplicity=1 + unpaired, fix_com=True,
                                                  fix_orientation=True, fix_symmetry="c1", name="qbatom")
            atom.update_geometry()
            _e, awfn = psi4.energy(t["method"], molecule=atom, return_wfn=True)
            psi4.oeprop(awfn, "MBIS_CHARGES", title=f"{sym} free atom", free_atom=True)
            out[sym] = float(np.ravel(np.asarray(awfn.variable("MBIS RADIAL MOMENTS <R^3>"), float))[0])
            wfn.set_variable(f"MBIS FREE ATOM {sym.upper()} VOLUME", out[sym])
    finally:
        psi4.set_options({"reference": reference_for(t["method"], t["multiplicity"])})
        if pcm:
            psi4.set_options({"pcm": True})
        psi4.core.set_active_molecule(mol)
    return out


def properties(psi4, t: dict, wfn, pcm: bool, warnings: list[str]) -> dict:
    """One-electron properties of a converged SCF wavefunction (no new SCF; MBIS volume ratios need one
    small SCF per element for the free atoms)."""
    out: dict = {}
    psi4.oeprop(wfn, "DIPOLE", "QUADRUPOLE", "MULLIKEN_CHARGES", "LOWDIN_CHARGES", "WIBERG_LOWDIN_INDICES",
                title="QB")
    mu = np.asarray(wfn.variable("QB DIPOLE"), float).ravel()
    quad = np.asarray(wfn.variable("QB QUADRUPOLE"), float).reshape(3, 3)
    out["dipole_vector_debye"] = (mu * AU_DEBYE).tolist()
    out["dipole_debye"] = float(np.linalg.norm(mu) * AU_DEBYE)
    out["quadrupole_debye_A"] = (quad * AU_DEBYE * BOHR_ANG).tolist()
    traceless = 1.5 * (quad - np.trace(quad) / 3.0 * np.eye(3))
    out["quadrupole_traceless_debye_A"] = (traceless * AU_DEBYE * BOHR_ANG).tolist()
    out["multipole_origin"] = "center of mass"
    charges = {"mulliken": np.ravel(np.asarray(wfn.variable("MULLIKEN CHARGES"), float)).tolist(),
               "lowdin": np.ravel(np.asarray(wfn.variable("LOWDIN CHARGES"), float)).tolist()}
    wib = np.asarray(wfn.variable("WIBERG LOWDIN INDICES"), float)
    n = wib.shape[0]
    bonds = [[i, j, round(float(wib[i, j]), 4)] for i in range(n) for j in range(i + 1, n) if wib[i, j] >= 0.5]
    out["wiberg"] = {"valences": wib.sum(axis=1).round(4).tolist(), "bonds": bonds, "threshold": 0.5}
    try:
        free = free_atom_volumes(psi4, t, wfn, pcm)
        psi4.oeprop(wfn, "MBIS_CHARGES", *(["MBIS_VOLUME_RATIOS"] if free else []), title="QB", free_atom=True)
        charges["mbis"] = np.ravel(np.asarray(wfn.variable("MBIS CHARGES"), float)).tolist()
        out["mbis_volumes_bohr3"] = np.ravel(np.asarray(wfn.variable("MBIS RADIAL MOMENTS <R^3>"), float)).tolist()
        if free:
            out["mbis_volume_ratios"] = np.ravel(np.asarray(wfn.variable("MBIS VOLUME RATIOS"), float)).tolist()
            out["mbis_free_atom_volumes_bohr3"] = free
    except Exception as exc:
        warnings.append(f"MBIS charges/volumes failed ({friendly(exc)})")
    out["charges"] = charges
    restricted = bool(wfn.same_a_b_orbs())
    eps_a = np.ravel(wfn.epsilon_a_subset("AO", "ALL").to_array(dense=True)).astype(float)
    eps_b = eps_a if restricted else np.ravel(wfn.epsilon_b_subset("AO", "ALL").to_array(dense=True)).astype(float)
    homo, lumo = homo_lumo(eps_a, eps_b, wfn.nalpha(), wfn.nbeta(), restricted)
    out["homo_ev"] = None if homo is None else homo * HARTREE_EV
    out["lumo_ev"] = None if lumo is None else lumo * HARTREE_EV
    out["gap_ev"] = None if homo is None or lumo is None else (lumo - homo) * HARTREE_EV
    out["n_alpha"], out["n_beta"] = int(wfn.nalpha()), int(wfn.nbeta())
    out["n_basis"] = int(wfn.basisset().nbf())
    return out


def write_guess(wfn) -> None:
    """Put ``wfn``'s orbitals where the next SCF with ``guess read`` looks for them (same process and molecule
    name, so the same scratch file name)."""
    wfn.to_file(wfn.get_scratch_filename(180))


def polarizability(psi4, t: dict, mol, wfn, pcm: bool, warnings: list[str]) -> dict:
    method, field = t["method"], t["field_au"]
    choice = t["polarizability_method"]
    reason = ""
    if choice == "auto":
        from psi4.driver.procrouting import proc_table

        func = wfn.functional()
        if pcm:
            reason = "PCM (the analytic response omits the reaction field)"
        elif method not in proc_table.procedures["properties"]:
            reason = f"no analytic properties for {method} in Psi4"
        elif func.is_meta() or func.needs_vv10():
            reason = "no analytic response for meta-GGA/VV10 functionals in Psi4"
        choice = "finite_field" if reason else "analytic"
    if choice == "analytic":
        try:
            write_guess(wfn)
            psi4.set_options({"guess": "read"})
            psi4.properties(method, properties=["dipole_polarizabilities"], molecule=mol)
            alpha = np.array([[psi4.core.variable(f"DIPOLE POLARIZABILITY {a}{b}") for b in "XYZ"] for a in "XYZ"])
            res = polarizability_summary(alpha)
            res["method"] = "analytic (CPSCF linear response)"
            return res
        except Exception as exc:
            if t["polarizability_method"] == "analytic":
                raise
            reason = f"analytic response failed ({friendly(exc)})"
        finally:
            psi4.set_options({"guess": "auto"})
    if reason:
        say(f"polarizability by finite field: {reason}")
    mus: dict[tuple[int, int], np.ndarray] = {}
    try:
        for k in range(3):
            for s in (1, -1):
                f = [0.0, 0.0, 0.0]
                f[k] = s * field
                write_guess(wfn)
                psi4.set_options({"guess": "read", "perturb_h": True, "perturb_with": "dipole", "perturb_dipole": f})
                _e, w = psi4.energy(method, molecule=mol, return_wfn=True)
                mus[(k, s)] = np.asarray(w.variable("CURRENT DIPOLE"), float).ravel()
    finally:
        psi4.set_options({"guess": "auto", "perturb_h": False, "perturb_dipole": [0.0, 0.0, 0.0]})
    alpha, sign = alpha_from_dipoles(np.array([mus[(k, 1)] for k in range(3)]),
                                     np.array([mus[(k, -1)] for k in range(3)]), field)
    res = polarizability_summary(alpha)
    res["method"] = f"finite field (F = {field:g} au, central differences of the dipole, 6 SCFs)"
    res["field_au"] = field
    res["psi4_perturbation_sign"] = sign
    if reason:
        res["reason"] = reason
    return res


def cubes(psi4, t: dict, wfn, warnings: list[str]) -> dict:
    folder = Path("cubes")
    folder.mkdir(exist_ok=True)
    s, o = t["cube_spacing_bohr"], t["cube_overage_bohr"]
    psi4.set_options({"cubeprop_tasks": ["density", "esp"], "cubic_grid_spacing": [s, s, s],
                      "cubic_grid_overage": [o, o, o], "cubeprop_filepath": str(folder)})
    psi4.cubeprop(wfn)
    density, esp = folder / "Dt.cube", folder / "ESP.cube"
    if not density.is_file() or not esp.is_file():
        raise RuntimeError(f"cubeprop wrote no Dt.cube/ESP.cube (found {sorted(p.name for p in folder.iterdir())})")
    for extra in ("Da.cube", "Db.cube", "Ds.cube"):
        (folder / extra).unlink(missing_ok=True)
    shape, data = read_cube_header_and_data(density)
    edge = boundary_max(data)
    if edge >= 0.002:
        warnings.append(f"the 0.002 au density isosurface reaches the cube boundary (max density there {edge:.4f} au); "
                        "increase cube_overage_bohr")
    return {"density": str(density), "esp": str(esp), "shape": list(shape), "spacing_bohr": s, "overage_bohr": o,
            "boundary_density_max_au": edge, "n_points": int(np.prod(shape))}


def hessian_kind(method: str, reference: str) -> str:
    from psi4.driver.procrouting import proc_table

    if method in proc_table.procedures["hessian"] and reference not in ("rohf", "cuhf"):
        return "analytic"
    if method in proc_table.procedures["gradient"]:
        return "finite differences of analytic gradients"
    return "finite differences of energies"


def frequencies(psi4, t: dict, mol0, pcm: bool, warnings: list[str]) -> dict:
    """Gas-phase Hessian (Eh/bohr^2) and dipole derivatives at a (tightly re-optimized) geometry."""
    method = t["method"]
    if pcm:
        psi4.set_options({"pcm": False})
        warnings.append("frequencies in the gas phase: Psi4 has no analytic PCM gradients")
    mol = make_molecule(psi4, t["symbols"], positions_A(mol0), t["charge"], t["multiplicity"])
    out: dict = {"optimized": False, "opt_converged": None}
    if t["freq_optimize"]:
        say("frequencies: tight re-optimization (gau_tight) of the best conformer")
        psi4.set_options({"g_convergence": "gau_tight"})
        out["optimized"] = True
        try:
            psi4.optimize(method, molecule=mol)
            out["opt_converged"] = True
        except psi4.OptimizationConvergenceError as exc:
            out["opt_converged"] = False
            mol.set_geometry(exc.wfn.molecule().geometry())
            mol.update_geometry()
            warnings.append("frequency re-optimization did not converge: frequencies are at a non-stationary point")
        finally:
            psi4.set_options({"g_convergence": t["opt_convergence"]})
    kind = hessian_kind(method, reference_for(method, t["multiplicity"]))
    say(f"frequencies: Hessian ({kind}) for {len(t['symbols'])} atoms")
    e, wfn = psi4.frequency(method, molecule=mol, return_wfn=True)
    hess = np.asarray(wfn.hessian(), float)
    dd = wfn.variables().get("CURRENT DIPOLE GRADIENT")
    grad = wfn.gradient()
    out.update(hessian_au=hess.tolist(), hessian_kind=kind, energy_hartree=float(e), positions_A=positions_A(mol),
               dipole_derivatives_au=None if dd is None else np.asarray(dd, float).tolist(),
               gradient_au=None if grad is None else np.asarray(grad, float).tolist())
    try:
        vib = wfn.frequency_analysis
        trv = np.asarray(vib["TRV"].data)
        omega = np.asarray(vib["omega"].data)[trv == "V"]
        nu = np.where(np.abs(omega.imag) > np.abs(omega.real), -np.abs(omega.imag), omega.real)
        out["psi4_frequencies_cm1"] = np.sort(nu.astype(float)).round(3).tolist()
        if "IR_intensity" in vib:
            ir = np.asarray(vib["IR_intensity"].data, float)[trv == "V"]
            out["psi4_ir_km_mol"] = ir[np.argsort(nu.astype(float))].round(4).tolist()
    except Exception as exc:
        warnings.append(f"could not read Psi4's own frequency analysis ({friendly(exc)})")
    if pcm:
        psi4.set_options({"pcm": True})
    return out


# ----------------------------------------------------------------------------------- main
def run(t: dict) -> dict:
    timings: dict[str, float] = {}
    warnings: list[str] = []
    t0 = time.perf_counter()
    pint_compat(warnings)
    import psi4

    timings["import"] = time.perf_counter() - t0
    psi4.set_output_file(t["output"], False)
    psi4.set_memory(int(t["memory_gb"] * 1e9))
    psi4.set_num_threads(t["threads"])
    ref = reference_for(t["method"], t["multiplicity"])
    opts = {"basis": t["basis"], "reference": ref, "scf_type": "df", "e_convergence": 1e-8, "d_convergence": 1e-8,
            "maxiter": 200, "properties_origin": ["COM"], "g_convergence": t["opt_convergence"],
            "geom_maxiter": t["geom_maxiter"], "t": t["temperature_k"]}
    opts.update(t["psi4_options"])
    psi4.set_options(opts)
    confs = t["conformers"]
    mol = make_molecule(psi4, t["symbols"], confs[0]["positions_A"], t["charge"], t["multiplicity"])
    check_level(psi4, t["method"], t["basis"], mol)

    pcm = False
    if t["solvent_pcm"]:
        if psi4.addons("pcmsolver"):
            psi4.set_options({"pcm": True, "pcm_scf_type": "total"})
            psi4.pcm_helper(pcm_input(t["solvent_pcm"]))
            pcm = True
        else:
            warnings.append(f"Psi4 was built without PCMSolver: gas phase instead of {t['solvent'] or t['solvent_pcm']} "
                            "(conda install -c conda-forge pcmsolver)")

    result: dict = {"ok": True, "program": "psi4", "program_version": psi4.__version__,
                    "level": f"{t['method']}/{t['basis']}", "method": t["method"], "basis": t["basis"],
                    "reference": ref, "task": t["task"], "threads": t["threads"], "memory_gb": t["memory_gb"],
                    "solvent": t["solvent"] if pcm else "",
                    "solvent_model": "IEFPCM (PCMSolver, GePol cavity, UFF radii)" if pcm else "",
                    "conformers": [], "warnings": warnings, "timings": timings}
    say(f"Psi4 {psi4.__version__}: {t['method']}/{t['basis']} ({ref.upper()}), {len(confs)} conformer(s), "
        f"{t['task']}, {t['threads']} thread(s), {t['memory_gb']:g} GB" + (f", PCM {t['solvent_pcm']}" if pcm else ""))
    if pcm and t["task"] == "optimize":
        warnings.append("optimizations in the gas phase (Psi4 has no analytic PCM gradients); energies and "
                        "properties are PCM single points at those geometries")

    best = None
    for i, c in enumerate(confs):
        tc = time.perf_counter()
        rec: dict = {"id": c["id"], "energy_hartree": None, "converged": False, "error": ""}
        mol = make_molecule(psi4, t["symbols"], c["positions_A"], t["charge"], t["multiplicity"])
        try:
            e, wfn, conv = scf_or_opt(psi4, t, mol, pcm)
            rec.update(energy_hartree=e, converged=conv)
            if t["task"] == "optimize":
                rec["positions_A"] = positions_A(mol)
            if best is None or e < best[0]:
                best = (e, i, mol, wfn)
            msg = f"E = {e:.8f} Eh" + ("" if conv else " (optimization not converged)")
        except Exception as exc:
            rec["error"] = friendly(exc)
            msg = f"failed: {rec['error']}"
        rec["seconds"] = time.perf_counter() - tc
        result["conformers"].append(rec)
        say(f"conformer {i + 1}/{len(confs)} (id {c['id']}): {msg} [{rec['seconds']:.1f} s]")
        psi4.core.clean()
    timings["conformers"] = sum(r["seconds"] for r in result["conformers"])
    if best is None:
        raise RuntimeError(f"no conformer finished: {result['conformers'][0]['error']}")
    e_best, i_best, mol, wfn = best
    result["best_index"] = i_best
    result["best_id"] = confs[i_best]["id"]
    result["energy_hartree"] = e_best
    result["best_positions_A"] = positions_A(mol)

    if t["properties"]:
        tp = time.perf_counter()
        result["properties"] = properties(psi4, t, wfn, pcm, warnings)
        timings["properties"] = time.perf_counter() - tp
        p = result["properties"]
        say(f"properties: dipole {p['dipole_debye']:.3f} D, HOMO {p['homo_ev']:.3f} eV, LUMO "
            + (f"{p['lumo_ev']:.3f} eV" if p["lumo_ev"] is not None else "n/a") + f" [{timings['properties']:.1f} s]")
    if t["cubes"]:
        tp = time.perf_counter()
        try:
            result["cubes"] = cubes(psi4, t, wfn, warnings)
            say(f"cubes: density + ESP on a {'x'.join(map(str, result['cubes']['shape']))} grid "
                f"[{time.perf_counter() - tp:.1f} s]")
        except Exception as exc:
            warnings.append(f"cubes failed ({friendly(exc)})")
        timings["cubes"] = time.perf_counter() - tp
    if t["polarizability"]:
        tp = time.perf_counter()
        try:
            result["polarizability"] = polarizability(psi4, t, mol, wfn, pcm, warnings)
            pol = result["polarizability"]
            say(f"polarizability: iso {pol['iso_A3']:.3f} A^3, aniso {pol['aniso_A3']:.3f} A^3 ({pol['method']}) "
                f"[{time.perf_counter() - tp:.1f} s]")
        except Exception as exc:
            warnings.append(f"polarizability failed ({friendly(exc)})")
        timings["polarizability"] = time.perf_counter() - tp
        psi4.core.clean()
    if t["hessian"]:
        tp = time.perf_counter()
        try:
            result["frequencies"] = frequencies(psi4, t, mol, pcm, warnings)
        except Exception as exc:
            warnings.append(f"frequencies failed ({friendly(exc)})")
        timings["frequencies"] = time.perf_counter() - tp
        if "frequencies" in result:
            say(f"frequencies: {len(result['frequencies'].get('psi4_frequencies_cm1', []))} modes "
                f"[{timings['frequencies']:.1f} s]")
        psi4.core.clean()
    timings["total"] = time.perf_counter() - t0
    return result


def write_json(path: str | Path, data: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(argv) != 1:
        print("usage: python -I psi4_driver.py task.json", file=sys.stderr)
        return 2
    task_path = Path(argv[0])
    result_path = task_path.with_name("result.json")
    try:
        task = load_task(task_path)
    except TaskError as exc:
        write_json(result_path, {"ok": False, "error": str(exc)})
        print(f"QuantaBench psi4 driver: {exc}", file=sys.stderr)
        return 2
    try:
        result = run(task)
    except Exception as exc:  # every failure is reported through result.json
        msg = str(exc) if isinstance(exc, TaskError) else friendly(exc)
        write_json(result_path, {"ok": False, "error": msg, "traceback": traceback.format_exc()})
        print(f"QuantaBench psi4 driver: {msg}", file=sys.stderr)
        return 1
    write_json(result_path, result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
