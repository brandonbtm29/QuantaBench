"""tblite engines: GFN2-xTB and GFN1-xTB in process, through the tblite library's Python API.

Why tblite next to dxtb and the xtb program:

- It is the reference Fortran implementation of the GFN Hamiltonians. GFN2, dipoles and the electric-field
  response need nothing else, whereas dxtb's GFN2 and every dxtb field property need the libcint backend
  (no compatible build for native Windows today, see ``engines.has_libcint``).
- It runs in this process: a gradient costs the SCF only (milliseconds for small molecules), not a
  program start and file I/O as with the xtb program, and each step restarts from the previous
  wavefunction, so geometry optimizations and finite-difference Hessians converge in a few SCF cycles.

Engines: ``tblite-gfn2`` and ``tblite-gfn1`` (elements H-Rn, CPU only). Implicit solvent: ALPB (default,
parametrized for every solvent in ``chem.solvents``), GBSA (falls back to ALPB, with a note, for solvents it
has no parameters for) or CPCM (ddX, from the dielectric constant). Pass ``solvation_model`` in the engine
options to choose; the default is ALPB as in the xtb program.

Units: tblite works in atomic units (Bohr, Hartree); the ASE calculator reports energy in eV, forces in
eV/Å, the dipole in e·Å about the origin (ASE's convention, used by the finite-difference IR intensities)
and Mulliken charges in e.

Properties (``properties``): dipole moment (Debye, about the center of mass for ions), HOMO/LUMO/gap (eV),
Mulliken charges and the static dipole polarizability from central finite differences of the dipole in a
uniform electric field (6 SCFs restarted from the field-free wavefunction). The polarizability is the
electronic response of tblite's minimal valence basis: it is much smaller than experiment (GFN2 water
0.37 Å³, experiment 1.45 Å³; out-of-plane components of planar molecules are nearly zero), so use it for
trends or recalibrate it. It is **not** the ``Mol. α(0)`` that the xtb program prints: that number comes
from the D4 dispersion model (charge-scaled atomic reference polarizabilities), not from the SCF. The
polarizability is always computed in the gas phase (a uniform field acting on a molecule in a continuum
is not the molecular property one tabulates).

Threads: tblite parallelizes with OpenMP inside this process (the PyPI wheel links the system libgomp and
a bundled single-threaded reference BLAS/LAPACK; conda-forge builds use OpenBLAS). OpenMP reads
``OMP_NUM_THREADS`` once, when the library is first loaded (the first ``import tblite``), so the thread
count is whatever was in the environment then. ``resources.apply_thread_limits`` sets it before any engine
is built, which is how the pipeline's thread setting reaches tblite; the ``threads`` engine option itself
is ignored. To change it later in the same process use ``threadpoolctl.threadpool_limits(n, "openmp")``.

Install: ``pip install tblite`` (wheels for Linux and macOS). Native Windows: PyPI has no wheel, so install
the brandonbtm29/tblite fork's wheels (``INSTALL_HINT_WINDOWS``, docs/INSTALL.md), or use conda-forge's
``tblite-python`` when running in that conda environment.
"""

from __future__ import annotations

import sys
from typing import ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..resources import has_package
from . import engines
from .engines import BOHR_ANG, HARTREE_EV

DEBYE_PER_E_ANG = 4.80320471
AU_POLARIZABILITY_A3 = BOHR_ANG**3       # 1 bohr³ = 0.148185 Å³
FIELD_STEP_AU = 5e-4                     # finite-field step (Hartree/(e·bohr) = 514 MV/cm per au)
SOLVATION_MODELS = ("alpb", "gbsa", "cpcm")
INSTALL_HINT = "pip install tblite (Linux/macOS) or conda install -c conda-forge tblite-python (Windows)"
# PyPI has no Windows wheel: the brandonbtm29/tblite fork builds them (docs/INSTALL.md, FORK-2)
WINDOWS_WHEELS = "https://github.com/brandonbtm29/tblite/releases/expanded_assets/qb-v0.8.0-win1"
INSTALL_HINT_WINDOWS = (f"pip install tblite --only-binary tblite --find-links {WINDOWS_WHEELS} "
                        "(or, in a conda environment, conda install -c conda-forge tblite-python)")


def available() -> tuple[bool, str]:
    if not has_package("tblite"):
        return False, INSTALL_HINT_WINDOWS if sys.platform == "win32" else INSTALL_HINT
    return True, ""


def solvation_args(solvent: str, model: str = "alpb") -> tuple | None:
    """Arguments of ``tblite.interface.Calculator.add`` for ``solvent`` (canonical name, "" = gas phase)
    with ``model`` ("alpb" | "gbsa" | "cpcm"); ``None`` in the gas phase."""
    if not solvent:
        return None
    from . import solvents

    model = (model or "alpb").lower()
    if model not in SOLVATION_MODELS:
        raise ValueError(f"tblite solvation model must be one of {', '.join(SOLVATION_MODELS)}, not {model!r}")
    if model == "cpcm":
        return ("ddX-solvation", float(solvents.for_engine(solvent, "cpcm")), "cpcm")
    name = solvents.for_engine(solvent, model)
    if name is None:
        raise ValueError(f"no {model.upper()} parameters for {solvent}")
    return (f"{model}-solvation", name)


def _describe(args: tuple | None) -> str:
    if args is None:
        return "gas phase"
    if args[0] == "ddX-solvation":
        return f"CPCM (eps = {args[1]:g})"
    return f"{args[0].split('-')[0].upper()}({args[1]})"


class TBLiteCalculator(Calculator):
    """ASE calculator around ``tblite.interface.Calculator`` (molecules, no periodic boundaries).

    ``electric_field`` is a uniform field in atomic units (Hartree/(e·bohr)). A single point that fails
    (SCF not converged, inconsistent charge/spin, atoms too close) raises ``RuntimeError`` with tblite's
    message; a failure when restarting from the previous wavefunction is first retried once from the
    superposition-of-atomic-densities guess.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces", "dipole", "charges"]

    def __init__(self, numbers, method: str = "GFN2-xTB", charge: int = 0, multiplicity: int = 1,
                 solvent: str = "", solvation_model: str = "alpb", electric_field=None, accuracy: float = 1.0,
                 max_iterations: int = 250, **kwargs):
        super().__init__(**kwargs)
        if method not in ("GFN2-xTB", "GFN1-xTB", "IPEA1-xTB"):
            raise ValueError(f"unknown tblite method {method!r}")
        if int(multiplicity) < 1:
            raise ValueError(f"multiplicity must be >= 1, not {multiplicity}")
        self.method = method
        self.charge = int(charge)
        self.uhf = int(multiplicity) - 1
        self.notes: list[str] = []
        self.solvation = self._resolve_solvation(solvent, solvation_model)
        self.electric_field = None if electric_field is None else np.asarray(electric_field, float).reshape(3)
        self.accuracy = float(accuracy)
        self.max_iterations = int(max_iterations)
        self.n_singlepoints = 0
        self._numbers = np.asarray(numbers, dtype=int).ravel()
        self._api = None        # tblite.interface.Calculator, rebuilt when the atoms change identity
        self._res = None        # tblite.interface.Result: last wavefunction, the restart guess
        self._field_buf = None

    def _resolve_solvation(self, solvent: str, model: str) -> tuple | None:
        try:
            return solvation_args(solvent, model)
        except ValueError:
            if not solvent or (model or "").lower() != "gbsa":
                raise
            self.notes.append(f"tblite: no GBSA parameters for {solvent}, used ALPB")
            return solvation_args(solvent, "alpb")

    @property
    def solvation_label(self) -> str:
        return _describe(self.solvation)

    # ------------------------------------------------------------------ tblite objects
    def _build(self, positions_bohr: np.ndarray):
        from tblite.interface import Calculator as TBCalculator

        try:
            api = TBCalculator(self.method, self._numbers, positions_bohr, float(self.charge), self.uhf,
                               color=False, logger=_silent)
        except TypeError:     # older tblite without context options (verbosity 0 still keeps it quiet)
            api = TBCalculator(self.method, self._numbers, positions_bohr, float(self.charge), self.uhf)
        api.set("verbosity", 0)
        api.set("accuracy", self.accuracy)
        api.set("max-iter", self.max_iterations)
        if self.electric_field is not None:
            from tblite import library

            # the C API takes a double* (a numpy array is rejected); keep the buffer alive with the calculator
            self._field_buf = library.ffi.new("double[3]", [float(x) for x in self.electric_field])
            api.add("electric-field", self._field_buf)
        if self.solvation is not None:
            try:
                api.add(*self.solvation)
            except (RuntimeError, ValueError) as exc:   # GBSA lacks some solvents that ALPB has (ethanol, ...)
                if not self.solvation[0].startswith("gbsa"):   # e.g. a tblite build without ddX (CPCM)
                    raise RuntimeError(f"tblite {self.method} {self.solvation_label}: {exc}") from exc
                self.notes.append(f"tblite: no GBSA parameters for {self.solvation[1]} with {self.method}, used ALPB")
                self.solvation = ("alpb-solvation", self.solvation[1])
                api.add(*self.solvation)
        return api

    def singlepoint(self, positions_ang: np.ndarray, guess=None):
        """Run one SCF at ``positions_ang`` and return the tblite ``Result`` (also kept as the next guess).
        ``guess`` (a ``Result``) replaces the stored restart wavefunction for this call."""
        from tblite.interface import Result

        pos = np.ascontiguousarray(np.asarray(positions_ang, float).reshape(-1, 3) / BOHR_ANG)
        if len(pos) != len(self._numbers):
            raise ValueError(f"{len(pos)} positions for {len(self._numbers)} atoms")
        try:
            if self._api is None:
                self._api = self._build(pos)
            else:
                self._api.update(positions=pos)
        except RuntimeError as exc:
            raise RuntimeError(f"tblite {self.method} setup failed: {exc}") from exc
        restart = Result(guess) if guess is not None else self._res
        try:
            res = self._api.singlepoint(restart)
        except RuntimeError as exc:
            if restart is None:
                raise RuntimeError(f"tblite {self.method} single point failed: {exc}") from exc
            try:   # the restart guess may be poor after a large step: retry from atomic densities
                res = self._api.singlepoint(None)
            except RuntimeError as exc2:
                self._res = None
                raise RuntimeError(f"tblite {self.method} single point failed: {exc2}") from exc2
        self.n_singlepoints += 1
        self._res = res
        return res

    # ------------------------------------------------------------------ ASE interface
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        numbers = self.atoms.numbers
        if len(numbers) != len(self._numbers) or np.any(numbers != self._numbers):
            self._numbers = np.asarray(numbers, dtype=int).copy()
            self._api = self._res = None
        res = self.singlepoint(self.atoms.positions)
        e = float(res.get("energy")) * HARTREE_EV
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = -np.asarray(res.get("gradient"), float) * (HARTREE_EV / BOHR_ANG)
        self.results["dipole"] = np.asarray(res.get("dipole"), float) * BOHR_ANG
        self.results["charges"] = np.asarray(res.get("charges"), float).copy()

    # ------------------------------------------------------------------ derived quantities
    def frontier_orbitals_ev(self) -> tuple[float | None, float | None]:
        """(HOMO, LUMO) in eV from the last SCF. Orbitals are filled with the electron count and the number
        of unpaired electrons (alpha electrons = (N + uhf) / 2), as the xtb program does."""
        if self._res is None:
            raise RuntimeError("no SCF has been run yet")
        return frontier_orbitals(self._res.get("orbital-energies"), self._res.get("orbital-occupations"), self.uhf)

    def polarizability_au(self, positions_ang: np.ndarray, step: float = FIELD_STEP_AU,
                          guess=None) -> np.ndarray:
        """Static dipole polarizability (3x3, atomic units = bohr³) by central differences of the dipole
        in ±``step`` fields along x, y, z, each SCF restarted from ``guess`` (or this calculator's last
        wavefunction). Uses this calculator's charge, spin and solvation."""
        guess = guess if guess is not None else self._res
        alpha = np.zeros((3, 3))
        for k in range(3):
            mu = []
            for sign in (1.0, -1.0):
                field = np.zeros(3)
                field[k] = sign * step
                fcalc = TBLiteCalculator(self._numbers, self.method, self.charge, self.uhf + 1,
                                         electric_field=field, accuracy=min(self.accuracy, 0.01),
                                         max_iterations=self.max_iterations)
                fcalc.solvation = self.solvation
                res = fcalc.singlepoint(positions_ang, guess=guess)
                mu.append(np.asarray(res.get("dipole"), float))       # e·bohr
            alpha[:, k] = (mu[0] - mu[1]) / (2.0 * step)
        return alpha


def _silent(_message: str) -> None:
    pass


def frontier_orbitals(eps_hartree, occupations, uhf: int = 0) -> tuple[float | None, float | None]:
    """(HOMO, LUMO) in eV. ``eps``/``occupations`` have shape (norb,) for a restricted calculation (occupations
    0..2) or (2, norb) for a spin-polarized one (0..1 per channel)."""
    eps = np.asarray(eps_hartree, float) * HARTREE_EV
    occ = np.asarray(occupations, float)
    if eps.ndim == 1:
        n_el = int(round(float(occ.sum())))
        n_alpha = (n_el + int(uhf)) // 2
        homo = float(eps[n_alpha - 1]) if n_alpha >= 1 else None
        lumo = float(eps[n_alpha]) if n_alpha < len(eps) else None
        return homo, lumo
    occupied = [e[o > 0.5] for e, o in zip(eps, occ)]
    virtual = [e[o <= 0.5] for e, o in zip(eps, occ)]
    homo = max((float(x.max()) for x in occupied if x.size), default=None)
    lumo = min((float(x.min()) for x in virtual if x.size), default=None)
    return homo, lumo


def polarizability_summary(alpha_au: np.ndarray) -> tuple[float, float]:
    """(isotropic, anisotropy) in Å³: the mean and the spread (max - min) of the symmetrized tensor's
    eigenvalues, as ``properties.xtb_properties`` reports them."""
    eig = np.linalg.eigvalsh(0.5 * (alpha_au + alpha_au.T)) * AU_POLARIZABILITY_A3
    return float(eig.mean()), float(eig.max() - eig.min())


def _dipole_about_com(numbers, positions_ang, mu_e_ang, charge: int) -> np.ndarray:
    """Shift an origin-based dipole to the center of mass (only changes anything for ions)."""
    from ase.data import atomic_masses

    m = atomic_masses[np.asarray(numbers, int)]
    com = (np.asarray(positions_ang, float) * m[:, None]).sum(0) / m.sum()
    return np.asarray(mu_e_ang, float) - float(charge) * com


# ---------------------------------------------------------------------------- engine hooks
def _factory(method: str):
    def factory(numbers, charge: int = 0, multiplicity: int = 1, device: str = "cpu", **options) -> TBLiteCalculator:
        return TBLiteCalculator(numbers, method, charge, multiplicity, solvent=options.get("solvent") or "",
                                solvation_model=options.get("solvation_model") or "alpb")
    factory.__name__ = f"tblite_{method.split('-')[0].lower()}_factory"
    return factory


def _properties(method: str, prefix: str):
    def properties(numbers, positions, charge: int = 0, multiplicity: int = 1, *, solvent: str = "",
                   polarizability: bool = True, field_step: float = FIELD_STEP_AU, **options) -> dict:
        from ase import Atoms

        calc = TBLiteCalculator(numbers, method, charge, multiplicity, solvent=solvent or "",
                                solvation_model=options.get("solvation_model") or "alpb")
        atoms = Atoms(numbers=np.asarray(numbers, int), positions=np.asarray(positions, float))
        atoms.calc = calc
        atoms.get_potential_energy()
        mu = _dipole_about_com(atoms.numbers, atoms.positions, calc.results["dipole"], charge)
        homo, lumo = calc.frontier_orbitals_ev()
        out: dict = {f"{prefix}_dipole_debye": float(np.linalg.norm(mu) * DEBYE_PER_E_ANG)}
        if homo is not None:
            out[f"{prefix}_homo_ev"] = homo
        if lumo is not None:
            out[f"{prefix}_lumo_ev"] = lumo
        if homo is not None and lumo is not None:
            out[f"{prefix}_gap_ev"] = lumo - homo
        notes = list(calc.notes)
        if polarizability:
            gas = calc
            if calc.solvation is not None:     # the polarizability is a gas-phase property (module docstring)
                gas = TBLiteCalculator(numbers, method, charge, multiplicity)
                gas.singlepoint(atoms.positions, guess=calc._res)
                notes.append(f"{prefix} polarizability: gas phase (other {prefix} properties: {calc.solvation_label})")
            iso, aniso = polarizability_summary(gas.polarizability_au(atoms.positions, field_step))
            out[f"{prefix}_polarizability_iso_A3"] = iso
            out[f"{prefix}_polarizability_aniso_A3"] = aniso
        out["charges"] = [float(x) for x in calc.results["charges"]]
        if notes:
            out["_notes"] = notes
        return out
    properties.__name__ = f"{prefix}_properties"
    return properties


_ELEMENTS = frozenset(range(1, 87))
for _key, _label, _method, _prefix, _desc in (
    ("tblite-gfn2", "GFN2-xTB (tblite)", "GFN2-xTB", "tblite_gfn2",
     "GFN2-xTB in process (tblite library): ALPB/GBSA/CPCM solvation, dipole, polarizability. CPU."),
    ("tblite-gfn1", "GFN1-xTB (tblite)", "GFN1-xTB", "tblite_gfn1",
     "GFN1-xTB in process (tblite library): ALPB/GBSA/CPCM solvation, dipole, polarizability. CPU."),
):
    engines.register(
        engines.EngineInfo(_key, _label, _desc, False, _ELEMENTS, family="tight binding",
                           solvation=SOLVATION_MODELS, external="", prop_prefix=_prefix),
        available, _factory(_method), _properties(_method, _prefix))
