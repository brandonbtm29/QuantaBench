"""Vibrational analysis: Hessian, normal modes, IR intensities, thermochemistry and IR spectra.

This is the Spartan "Calculate frequencies / IR / thermodynamics" step for one (best) conformer.

Pipeline
--------
1. **Hessian** (second derivatives of the energy, eV/Å²):

   - analytic, by automatic differentiation, for GFN1-/GFN2-xTB through dxtb (exact to numerical
     precision, and no step size to tune);
   - central finite differences of the forces for any ASE calculator (MMFF, AIMNet2, xTB, ...):
     ``H[:, j] = -(F(x + h e_j) - F(x - h e_j)) / 2h``, 6N force calls, default h = 0.01 Å.

   The Hessian is then symmetrized, because finite differences (and float noise) leave a small
   antisymmetric part that would make the eigenvalues complex.

2. **Normal modes.** Mass-weight with the most abundant isotope masses (the Gaussian/Spartan
   convention: a measured spectrum is dominated by the main isotopologue, not by an "average atom"),
   project out the three translations and the three rotations (two for a linear molecule) by working in
   the orthogonal complement of the Eckart translation/rotation vectors, and diagonalize. Frequencies are
   in cm⁻¹; imaginary modes are reported as negative numbers (the usual convention).

3. **IR intensities** (km/mol) from the dipole derivative dμ/dQ along each mass-weighted normal mode,
   ``A = N_A π / (3 c² 4πε₀) |dμ/dQ|²``; 1 e²/amu = 974.9 km/mol:

   - dxtb: analytic dipole derivatives (an ElectricField interaction at zero field, as in
     ``chem.properties``);
   - otherwise finite differences of the dipole, taken from the calculator's ``dipole`` or from
     ``Σ qᵢ rᵢ`` when it reports atomic ``charges`` (AIMNet2 does). The charge model ignores
     polarization of the electron density inside each atom, so these are labeled approximate;
   - ``None`` when neither is available (MMFF: its charges are fixed, so they would give fixed-charge
     intensities that are not worth reporting).

4. **Thermochemistry** (ideal gas, rigid rotor, harmonic oscillator; ``ase.thermochemistry``) at T and P,
   with Grimme's quasi-RRHO entropy for low-frequency modes by default (see :func:`qrrho_entropy`).

5. **IR spectrum**: Lorentzian broadening of the stick spectrum, normalized to 1.

Everything heavy (torch, dxtb, aimnet) is imported lazily.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from .engines import BOHR_ANG, HARTREE_EV

KCAL_PER_EV = 23.060548                     # same constant as chem.conformers
CAL_PER_EV = KCAL_PER_EV * 1000.0

# sqrt(eV / (Å² amu)) in rad/s, divided by 2πc: eigenvalue of the mass-weighted Hessian -> cm⁻¹ (≈ 521.47)
_EIG_TO_CM1 = math.sqrt(units._e / (1e-20 * units._amu)) / (2.0 * math.pi * units._c * 100.0)
# |dμ/dQ|² in e²/amu -> IR intensity in km/mol (≈ 974.86). SI form of A = N_A π/(3c²) |dμ/dQ|² / (4πε₀).
KM_MOL_PER_E2_AMU = (units._Nav * math.pi * units._e**2
                     / (3.0 * units._c**2 * 4.0 * math.pi * units._eps0 * units._amu) / 1000.0)
# 1 eV/Å² = 16.02 N/m = 0.1602 mdyn/Å
_EV_A2_TO_MDYN_A = units._e / 1e-20 / 100.0

HessianMode = Literal["auto", "analytic", "finite"]


# --------------------------------------------------------------------------- scale factors
@dataclass(frozen=True)
class ScaleFactor:
    """A harmonic-frequency scale factor and where it comes from."""

    value: float
    source: str


SCALE_FACTORS: dict[str, ScaleFactor] = {
    "mmff": ScaleFactor(1.0, "Unscaled. MMFF94 force constants were fitted to vibrational frequencies "
                             "(Halgren, J. Comput. Chem. 1996, 17, 553-586), so no factor is applied."),
    "gfn1-xtb": ScaleFactor(1.0, "Unscaled: no published general factor for GFN1-xTB is known to us. GFN1-xTB "
                                 "was parameterized partly on harmonic frequencies (Grimme, Bannwarth, "
                                 "Shushkov, J. Chem. Theory Comput. 2017, 13, 1989-2009)."),
    "gfn2-xtb": ScaleFactor(1.0, "Unscaled: no published general factor for GFN2-xTB is known to us "
                                 "(Bannwarth, Ehlert, Grimme, J. Chem. Theory Comput. 2019, 15, 1652-1671)."),
    "aimnet2": ScaleFactor(1.0, "Unscaled: no published factor. AIMNet2 approximates wB97M-D3/def2-TZVPP "
                                "harmonic frequencies (Anstine, Zubatyuk, Isayev, Chem. Sci. 2025); harmonic "
                                "hybrid-DFT frequencies typically overestimate fundamentals by a few percent."),
    # Reference values for comparing with Spartan/Gaussian results computed at these levels.
    "hf/6-31g*": ScaleFactor(0.8953, "Scott & Radom, J. Phys. Chem. 1996, 100, 16502-16513 (fundamentals)."),
    "mp2/6-31g*": ScaleFactor(0.9427, "Scott & Radom, J. Phys. Chem. 1996, 100, 16502-16513 (fundamentals)."),
    "b3lyp/6-31g*": ScaleFactor(0.9614, "Scott & Radom, J. Phys. Chem. 1996, 100, 16502-16513 (fundamentals)."),
}


def scale_factor(method: str) -> float:
    """Frequency scale factor for ``method`` (1.0 when none is tabulated)."""
    sf = SCALE_FACTORS.get(method.lower())
    return sf.value if sf else 1.0


# --------------------------------------------------------------------------- MMFF as an ASE calculator
class RdkitMMFFCalculator(Calculator):
    """MMFF94/MMFF94s (UFF fallback) from RDKit as an ASE calculator.

    The topology (bonds, atom types, charges) comes from ``mol`` and is fixed; only the coordinates
    change. Atom order must match the ``Atoms`` object (``export.mol_to_atoms`` keeps it). MMFF partial
    charges are geometry-independent bond-increment charges, so they are only reported when
    ``report_charges=True`` (they would give fixed-charge IR intensities, which are crude).
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces"]

    def __init__(self, mol, variant: str = "MMFF94s", report_charges: bool = False, **kwargs):
        super().__init__(**kwargs)
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.Mol(mol)
        if mol.GetNumConformers() == 0:
            mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if props is not None:
            self.forcefield = variant
            self._ff = AllChem.MMFFGetMoleculeForceField(mol, props, ignoreInterfragInteractions=False)
            self._charges = np.array([props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())])
        elif AllChem.UFFHasAllMoleculeParams(mol):
            self.forcefield = "UFF"
            self._ff = AllChem.UFFGetMoleculeForceField(mol, ignoreInterfragInteractions=False)
            self._charges = None
        else:
            raise RuntimeError("neither MMFF nor UFF has parameters for this molecule")
        if self._ff is None:
            raise RuntimeError(f"could not set up the {self.forcefield} force field")
        self.n_atoms = mol.GetNumAtoms()
        self.report_charges = report_charges and self._charges is not None
        if self.report_charges:
            self.implemented_properties = [*self.implemented_properties, "charges"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if len(self.atoms) != self.n_atoms:
            raise ValueError(f"{len(self.atoms)} atoms given, the force field has {self.n_atoms}")
        pos = [float(x) for x in self.atoms.get_positions().ravel()]
        e = self._ff.CalcEnergy(pos) / KCAL_PER_EV                       # kcal/mol -> eV
        grad = np.asarray(self._ff.CalcGrad(pos)).reshape(-1, 3) / KCAL_PER_EV
        self.results["energy"] = e
        self.results["free_energy"] = e
        self.results["forces"] = -grad
        if self.report_charges:
            self.results["charges"] = self._charges.copy()


def mol_from_atoms(atoms: Atoms, charge: int = 0):
    """RDKit molecule with bonds perceived from 3D coordinates (for MMFF on bare coordinates)."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    xyz = f"{len(atoms)}\n\n" + "\n".join(f"{s} {x:.8f} {y:.8f} {z:.8f}"
                                           for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions))
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


# --------------------------------------------------------------------------- masses and geometry
def isotope_masses(numbers) -> np.ndarray:
    """Most abundant isotope masses (amu), the convention of Gaussian and Spartan frequency jobs."""
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    return np.array([pt.GetMostCommonIsotopeMass(int(z)) for z in numbers], dtype=float)


def detect_geometry(positions, masses, tol: float = 1e-4) -> str:
    """'monatomic', 'linear' or 'nonlinear', from the principal moments of inertia.

    Linear means the smallest moment is below ``tol`` times the largest, i.e. atoms deviate from a
    line by roughly 1 % of the molecular length or less.
    """
    pos = np.asarray(positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    if len(pos) == 1:
        return "monatomic"
    r = pos - (m[:, None] * pos).sum(0) / m.sum()
    inertia = np.einsum("i,ij,ik->jk", m, r, r)
    inertia = np.trace(inertia) * np.eye(3) - inertia
    moments = np.linalg.eigvalsh(inertia)
    return "linear" if moments[0] < tol * moments[-1] else "nonlinear"


def _rotation_from_pairs(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Proper rotation R (det +1) that best maps the rows of ``p`` onto ``q`` (Kabsch): q ≈ p @ R.T."""
    u, _s, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def rotational_symmetry_number(numbers, positions, masses=None, tol: float = 0.1) -> int:
    """Rotational symmetry number σ: the number of proper rotations (identity included) that map the
    molecule onto itself, atoms matched by element within ``tol`` Å (Gaussian and Spartan detect it from
    the point group in the same way).

    1 for C1/Cs/Ci/C∞v, 2 for C2v (water) and D∞h (CO2), 3 for C3v (ammonia), 6 for D3d (staggered ethane),
    12 for Td (methane) and D6h (benzene), 24 for Oh (SF6). ``tol`` absorbs the residual asymmetry of an
    optimized geometry; a conformer that is only nearly symmetric counts as symmetric within it.
    """
    z = np.asarray(numbers, int)
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(z)
    if n < 2:
        return 1
    m = np.ones(n) if masses is None else np.asarray(masses, dtype=float)
    r = pos - (m[:, None] * pos).sum(0) / m.sum()
    dist = np.linalg.norm(r, axis=1)

    def match(rot: np.ndarray, cutoff: float) -> np.ndarray | None:
        """Image of each atom under ``rot`` (a permutation of same-element atoms), or None."""
        d = np.linalg.norm((r @ rot.T)[:, None, :] - r[None, :, :], axis=2)
        d[z[:, None] != z[None, :]] = np.inf
        img = d.argmin(axis=1)
        if not np.all(d[np.arange(n), img] < cutoff) or len(set(img.tolist())) != n:
            return None
        return img

    def maps_onto_itself(rot: np.ndarray) -> np.ndarray | None:
        """Refined rotation (fitted to all matched atoms, so noise in the two reference atoms does not
        count) when ``rot`` is a symmetry operation, else None."""
        img = match(rot, 3 * tol)
        if img is None:
            return None
        rot = _rotation_from_pairs(r, r[img])
        return rot if match(rot, tol) is not None else None

    if detect_geometry(pos, m) == "linear":
        # D∞h (a C2 perpendicular to the axis) iff the molecule has an inversion center
        d = np.linalg.norm(-r[:, None, :] - r[None, :, :], axis=2)
        d[z[:, None] != z[None, :]] = np.inf
        return 2 if np.all(d.min(axis=1) < tol) else 1

    def candidates(i: int) -> np.ndarray:
        return np.where((z == z[i]) & (np.abs(dist - dist[i]) < 2 * tol))[0]

    off_center = [i for i in range(n) if dist[i] > tol]
    if not off_center:
        return 1
    a = min(off_center, key=lambda i: len(candidates(i)))
    ua = r[a] / dist[a]
    # second reference: not collinear with the first through the center of mass
    others = [i for i in off_center if np.linalg.norm(np.cross(ua, r[i] / dist[i])) > 0.2]
    if not others:
        return 1
    b = min(others, key=lambda i: len(candidates(i)))
    dab = np.linalg.norm(r[a] - r[b])
    found: list[np.ndarray] = []
    for a2 in candidates(a):
        for b2 in candidates(b):
            if a2 == b2 or abs(np.linalg.norm(r[a2] - r[b2]) - dab) > 2 * tol:
                continue
            p = np.array([r[a], r[b], np.cross(r[a], r[b])])
            q = np.array([r[a2], r[b2], np.cross(r[a2], r[b2])])
            rot = maps_onto_itself(_rotation_from_pairs(p, q))
            if rot is not None and not any(np.abs(rot - f).max() < 0.05 for f in found):
                found.append(rot)
    return max(1, len(found))


# --------------------------------------------------------------------------- Hessians
def symmetrize(hessian: np.ndarray) -> np.ndarray:
    h = np.asarray(hessian, dtype=float)
    return 0.5 * (h + h.T)


def _calc_dipole(atoms: Atoms, origin: np.ndarray) -> np.ndarray | None:
    """Dipole (e·Å) from the calculator's last results: its ``dipole`` if any, else Σ qᵢ (rᵢ - origin)."""
    res = atoms.calc.results
    if res.get("dipole") is not None:
        return np.asarray(res["dipole"], dtype=float).reshape(3)
    q = res.get("charges")
    if q is None:
        return None
    return (np.asarray(q, dtype=float).reshape(-1, 1) * (atoms.positions - origin)).sum(0)


def dipole_source(atoms: Atoms) -> str:
    """Which dipole a finite-difference IR would use for this calculator ('' when none)."""
    res = atoms.calc.results
    if res.get("dipole") is not None:
        return "finite-difference dipole from the calculator (approximate)"
    if res.get("charges") is not None:
        return "finite-difference dipole from atomic charges (approximate)"
    return ""


def finite_difference_hessian(atoms: Atoms, step: float = 0.01, dipole: bool = True,
                              should_stop: Callable[[], bool] | None = None,
                              progress: Callable[[int, int], None] | None = None
                              ) -> tuple[np.ndarray, np.ndarray | None]:
    """Central-difference Hessian (eV/Å², not yet symmetrized) and dipole derivative (3, 3N) in e.

    ``atoms`` must carry a calculator. Each Cartesian coordinate is displaced by ±``step`` Å
    (6N force calls). The error is O(step²); 0.01 Å is the usual compromise between truncation
    error and the noise of the force evaluation (SCF convergence, float32 on GPUs).
    The dipole derivative is returned when the calculator reports a dipole or atomic charges.
    """
    from .optimize import Cancelled

    work = atoms.copy()
    work.calc = atoms.calc
    x0 = work.get_positions()
    n = len(work)
    origin = x0.mean(axis=0)
    work.get_forces()
    want_dip = dipole and _calc_dipole(work, origin) is not None
    hess = np.zeros((3 * n, 3 * n))
    dmu = np.zeros((3, 3 * n)) if want_dip else None
    total = 3 * n
    try:
        for j in range(total):
            if should_stop and should_stop():
                raise Cancelled()
            a, k = divmod(j, 3)
            out = []
            for sign in (1.0, -1.0):
                x = x0.copy()
                x[a, k] += sign * step
                work.set_positions(x)
                f = work.get_forces().ravel().copy()
                out.append((f, _calc_dipole(work, origin) if want_dip else None))
            hess[:, j] = -(out[0][0] - out[1][0]) / (2.0 * step)
            if want_dip:
                dmu[:, j] = (out[0][1] - out[1][1]) / (2.0 * step)
            if progress:
                progress(j + 1, total)
    finally:
        work.set_positions(x0)
        work.get_forces()  # leave the calculator's results at the reference geometry
    return hess, dmu


def _dxtb_setup(numbers, positions, method: str, device: str):
    import dxtb
    import torch
    from dxtb.components.field import new_efield

    dev = torch.device(device)
    # float64 even on GPUs: low frequencies come from small differences of large Hessian elements,
    # and float32 Hessians give spurious imaginary modes.
    dtype = torch.float64
    par = dxtb.GFN2_XTB if method == "gfn2-xtb" else dxtb.GFN1_XTB
    opts = {"verbosity": 0}
    if method == "gfn2-xtb":
        opts["int_driver"] = "libcint"
    field = torch.zeros(3, device=dev, dtype=dtype, requires_grad=True)
    z = torch.as_tensor(np.asarray(numbers), device=dev)
    calc = dxtb.Calculator(z, par, interaction=new_efield(field), opts=opts, device=dev, dtype=dtype)

    def pos_tensor():
        return torch.tensor(np.asarray(positions) / BOHR_ANG, device=dev, dtype=dtype, requires_grad=True)

    return calc, pos_tensor


def dxtb_hessian(numbers, positions, method: str = "gfn1-xtb", charge: int = 0, multiplicity: int = 1,
                 device: str = "cpu") -> np.ndarray:
    """Analytic (autograd) GFN-xTB Hessian from dxtb, converted from Hartree/Bohr² to eV/Å²."""
    calc, pos_tensor = _dxtb_setup(numbers, positions, method, device)
    kwargs = {"chrg": charge}
    if multiplicity > 1:
        kwargs["spin"] = multiplicity - 1
    n = len(numbers)
    h = calc.get_hessian(pos_tensor(), **kwargs)
    return h.detach().cpu().double().numpy().reshape(3 * n, 3 * n) * HARTREE_EV / BOHR_ANG**2


# Peak memory of dxtb's autograd Hessian grows roughly with n_AO³ (it keeps the SCF graph for 3N
# backward passes). Measured on CPU, float64, dxtb 0.4: GFN1 heptanoate (62 AOs) 2.9 GB, PFBA anion
# (52 AOs) 1.9 GB, and PFOA anion (100 AOs) was killed at 7 GB; GFN2 heptanoate (49 AOs) 4.4 GB.
# Finite differences of the analytic forces need only ~0.7 GB and took 16 s vs 13 s (22 atoms).
_DXTB_HESS_BASE_GB = 0.8
_DXTB_HESS_GB_PER_NAO3 = {"gfn1-xtb": 1.2e-5, "gfn2-xtb": 3.5e-5}


def estimate_dxtb_hessian_memory_gb(numbers, method: str = "gfn1-xtb") -> float:
    """Rough peak memory (GB) of an analytic dxtb Hessian, from the number of atomic orbitals."""
    import dxtb
    import torch

    par = dxtb.GFN2_XTB if method == "gfn2-xtb" else dxtb.GFN1_XTB
    calc = dxtb.Calculator(torch.as_tensor(np.asarray(numbers)), par, opts={"verbosity": 0}, dtype=torch.float64)
    nao = int(calc.ihelp.nao)
    return _DXTB_HESS_BASE_GB + _DXTB_HESS_GB_PER_NAO3.get(method, 3.5e-5) * nao**3


def available_memory_gb(device: str = "cpu") -> float:
    """Free memory for the Hessian: free GPU memory for CUDA/ROCm devices, else available RAM."""
    if device != "cpu" and (device.startswith("cuda") or device.startswith("hip")):
        try:
            import torch

            free, _total = torch.cuda.mem_get_info(torch.device(device))
            return free / 1e9
        except Exception:  # noqa: S110 - fall back to system RAM
            pass
    import psutil

    return psutil.virtual_memory().available / 1e9


def dxtb_dipole_derivatives(numbers, positions, method: str = "gfn1-xtb", charge: int = 0,
                            multiplicity: int = 1, device: str = "cpu") -> np.ndarray:
    """Analytic dipole derivative dμ/dR, shape (3, 3N), in e (= e·Bohr/Bohr = e·Å/Å)."""
    calc, pos_tensor = _dxtb_setup(numbers, positions, method, device)
    kwargs = {"chrg": charge}
    if multiplicity > 1:
        kwargs["spin"] = multiplicity - 1
    d = calc.get_dipole_derivatives(pos_tensor(), **kwargs)
    return d.detach().cpu().double().numpy().reshape(3, 3 * len(numbers))


def aimnet_hessian(calc, atoms: Atoms) -> np.ndarray:
    """Analytic AIMNet2 Hessian (eV/Å²) through ``AIMNet2Calculator.eval(hessian=True)``.

    Only used with ``hessian='analytic'``; it could not be validated against real model weights in
    development, so 'auto' uses finite differences for AIMNet2.
    """
    import torch

    base = calc.base_calc
    data = {"coord": torch.as_tensor(atoms.positions, dtype=torch.float32, device=base.device),
            "numbers": torch.as_tensor(atoms.numbers, dtype=torch.int64, device=base.device),
            "charge": torch.as_tensor([float(calc.charge)], device=base.device),
            "mult": torch.as_tensor([float(calc.mult)], device=base.device)}
    out = base.eval(data, forces=True, hessian=True)
    h = out["hessian"].detach().cpu().double().numpy()
    n = len(atoms)
    if h.size != 9 * n * n:
        raise RuntimeError(f"unexpected AIMNet2 Hessian shape {h.shape}")
    return h.reshape(3 * n, 3 * n)


# --------------------------------------------------------------------------- normal modes
@dataclass
class NormalModes:
    """Result of a normal-mode analysis (modes sorted by frequency, imaginary first).

    Reduced masses and force constants follow the Gaussian convention: with the Cartesian displacement
    of a mode normalized to 1, μ = 1 / Σᵢ (Lᵢ²/mᵢ) and k = μ ω². For H₂ this gives μ = m_H (not m_H/2)
    and k = 2 k_bond, so compare them with Gaussian/Spartan output, not with textbook diatomics.
    """

    frequencies_cm1: np.ndarray           # (n_vib,), imaginary as negative
    eigenvalues: np.ndarray               # (n_vib,) of the mass-weighted Hessian, eV/(Å² amu)
    modes_mw: np.ndarray                  # (3N, n_vib) orthonormal mass-weighted eigenvectors
    displacements: np.ndarray             # (n_vib, N, 3) Cartesian displacements, each of unit norm
    reduced_masses_amu: np.ndarray        # (n_vib,)
    force_constants_mdyn_A: np.ndarray    # (n_vib,), negative for imaginary modes
    geometry: str                         # 'nonlinear' | 'linear' | 'monatomic'


def _translation_rotation_basis(positions: np.ndarray, masses: np.ndarray, n_tr: int) -> np.ndarray:
    """Orthonormal basis (3N, n_tr) of the mass-weighted Eckart translations and rotations."""
    n = len(masses)
    sm = np.sqrt(masses)
    r = positions - (masses[:, None] * positions).sum(0) / masses.sum()
    vecs = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = sm
        vecs.append(v.ravel())
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        vecs.append((sm[:, None] * np.cross(axis, r)).ravel())
    u, _, _ = np.linalg.svd(np.array(vecs).T, full_matrices=False)
    return u[:, :n_tr]    # for a linear molecule the rotation about the axis is (near) zero: rank 5


def normal_modes(hessian: np.ndarray, masses, positions, geometry: str | None = None) -> NormalModes:
    """Mass-weighted, Eckart-projected normal-mode analysis of a Cartesian Hessian in eV/Å².

    The vibrational space is the orthogonal complement of the translations and rotations, so exactly
    3N-6 (3N-5 for linear) modes come out, with no near-zero "rotations" to filter by threshold.
    """
    m = np.asarray(masses, dtype=float)
    pos = np.asarray(positions, dtype=float)
    n = len(m)
    geometry = geometry or detect_geometry(pos, m)
    if geometry == "monatomic" or n == 1:
        empty = np.zeros(0)
        return NormalModes(empty, empty, np.zeros((3 * n, 0)), np.zeros((0, n, 3)), empty, empty, "monatomic")
    n_tr = 5 if geometry == "linear" else 6
    sm3 = np.repeat(np.sqrt(m), 3)
    hmw = symmetrize(hessian) / np.outer(sm3, sm3)
    tr = _translation_rotation_basis(pos, m, n_tr)
    # Complement of the TR space: eigenvectors of the projector with eigenvalue 1.
    proj = np.eye(3 * n) - tr @ tr.T
    w, v = np.linalg.eigh(proj)
    d = v[:, w > 0.5]
    lam, u = np.linalg.eigh(d.T @ hmw @ d)
    modes = d @ u                                           # (3N, n_vib), orthonormal
    freqs = np.sign(lam) * np.sqrt(np.abs(lam)) * _EIG_TO_CM1
    cart = (modes / sm3[:, None]).T                         # (n_vib, 3N), unnormalized Cartesian
    norm2 = (cart**2).sum(1)
    reduced = 1.0 / norm2
    disp = (cart / np.sqrt(norm2)[:, None]).reshape(-1, n, 3)
    k = lam * reduced * _EV_A2_TO_MDYN_A
    return NormalModes(freqs, lam, modes, disp, reduced, k, geometry)


def ir_intensities(dipole_derivative: np.ndarray, modes: NormalModes, masses) -> np.ndarray:
    """IR intensities (km/mol) from dμ/dR (3, 3N, in e) and the mass-weighted normal modes."""
    sm3 = np.repeat(np.sqrt(np.asarray(masses, dtype=float)), 3)
    dmu_dq = np.asarray(dipole_derivative, dtype=float) @ (modes.modes_mw / sm3[:, None])   # e/amu^½
    return (dmu_dq**2).sum(0) * KM_MOL_PER_E2_AMU


# --------------------------------------------------------------------------- thermochemistry
@dataclass
class Thermochemistry:
    """Ideal-gas RRHO thermochemistry (energies in eV, entropy in eV/K)."""

    temperature_k: float
    pressure_pa: float
    potential_energy_ev: float
    zpe_ev: float
    enthalpy_ev: float               # E + ZPE + ∫Cp dT
    entropy_ev_k: float              # qRRHO entropy when qrrho is on, else harmonic
    gibbs_ev: float                  # H - T S
    entropy_harmonic_ev_k: float     # always the plain RRHO value, for comparison
    qrrho: bool
    qrrho_cutoff_cm1: float
    qrrho_alpha: float
    symmetry_number: int
    spin: float                      # S, from multiplicity = 2S + 1
    geometry: str
    n_modes_used: int

    @property
    def zpe_kcal(self) -> float:
        return self.zpe_ev * KCAL_PER_EV

    @property
    def enthalpy_kcal(self) -> float:
        return self.enthalpy_ev * KCAL_PER_EV

    @property
    def gibbs_kcal(self) -> float:
        return self.gibbs_ev * KCAL_PER_EV

    @property
    def entropy_cal_molK(self) -> float:
        return self.entropy_ev_k * CAL_PER_EV

    @property
    def h_corr_kcal(self) -> float:
        """Thermal correction to the enthalpy, H - E (includes ZPE)."""
        return (self.enthalpy_ev - self.potential_energy_ev) * KCAL_PER_EV

    @property
    def g_corr_kcal(self) -> float:
        """Thermal correction to the Gibbs energy, G - E."""
        return (self.gibbs_ev - self.potential_energy_ev) * KCAL_PER_EV

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(zpe_kcal=self.zpe_kcal, enthalpy_kcal=self.enthalpy_kcal, gibbs_kcal=self.gibbs_kcal,
                 entropy_cal_molK=self.entropy_cal_molK, h_corr_kcal=self.h_corr_kcal,
                 g_corr_kcal=self.g_corr_kcal)
        return d


def harmonic_vib_entropy(frequencies_cm1, temperature: float) -> np.ndarray:
    """Per-mode harmonic-oscillator entropy (eV/K): k[x/(eˣ-1) - ln(1-e⁻ˣ)], x = hcν̃/kT."""
    nu = np.asarray(frequencies_cm1, dtype=float)
    x = nu * units.invcm / (units.kB * temperature)
    return units.kB * (x / np.expm1(x) - np.log(-np.expm1(-x)))


def qrrho_entropy(frequencies_cm1, temperature: float, cutoff_cm1: float = 100.0, alpha: float = 4.0,
                  b_av: float = 1e-44) -> np.ndarray:
    """Per-mode quasi-RRHO vibrational entropy (eV/K), Grimme, Chem. Eur. J. 2012, 18, 9955-9964.

    Low-frequency modes are poorly described as harmonic oscillators (their entropy diverges as
    ν -> 0, and they are often hindered rotations). Each mode's entropy is interpolated between the
    harmonic value S_V and that of a free rotor S_R with the same moment of inertia,

        S = w S_V + (1 - w) S_R,   w = 1 / (1 + (ω₀/ν)^α),   ω₀ = 100 cm⁻¹, α = 4,
        S_R = R [1/2 + ln( (8π³ μ' k T / h²)^½ )],   μ = h / (8π² c ν),   μ' = μ B_av / (μ + B_av),

    where B_av = 1e-44 kg m² caps the moment of inertia of very soft modes. Only the entropy is
    modified (Grimme's original scheme); ZPE and enthalpy stay harmonic.
    """
    nu = np.asarray(frequencies_cm1, dtype=float)
    s_v = harmonic_vib_entropy(nu, temperature)
    nu_m = nu * 100.0                                           # m⁻¹
    mu = units._hplanck / (8.0 * math.pi**2 * units._c * nu_m)  # kg m²
    mu_eff = mu * b_av / (mu + b_av)
    s_r = units.kB * (0.5 + np.log(np.sqrt(8.0 * math.pi**3 * mu_eff * units._k * temperature
                                           / units._hplanck**2)))
    w = 1.0 / (1.0 + (cutoff_cm1 / nu) ** alpha)
    return w * s_v + (1.0 - w) * s_r


def thermochemistry(atoms: Atoms, frequencies_cm1, potential_energy_ev: float, temperature: float = 298.15,
                    pressure: float = 101325.0, symmetry_number: int = 1, multiplicity: int = 1,
                    geometry: str | None = None, masses=None, qrrho: bool = True,
                    qrrho_cutoff_cm1: float = 100.0, qrrho_alpha: float = 4.0) -> Thermochemistry:
    """Ideal-gas thermochemistry with ``ase.thermochemistry.IdealGasThermo``, plus qRRHO entropy.

    ``frequencies_cm1`` must contain only the real vibrational modes to use (imaginary modes removed
    by the caller). ``symmetry_number`` is the rotational symmetry number σ (1 for C1, 2 for water,
    3 for ammonia, 6 for ethane/benzene's D3d/..., 12 for methane); it lowers S_rot by R ln σ.
    """
    from ase.thermochemistry import IdealGasThermo

    work = atoms.copy()
    work.calc = None
    if masses is not None:
        work.set_masses(np.asarray(masses, dtype=float))
    geometry = geometry or detect_geometry(work.positions, work.get_masses())
    nu = np.sort(np.asarray([f for f in frequencies_cm1 if f > 0], dtype=float))
    vib_e = list(nu * units.invcm)
    spin = (multiplicity - 1) / 2.0
    kw = dict(geometry=geometry, potentialenergy=potential_energy_ev, atoms=work,
              symmetrynumber=symmetry_number, spin=spin)
    try:
        thermo = IdealGasThermo(vib_e, vib_selection="all", **kw)
    except TypeError:  # ASE < 3.25 has no vib_selection
        thermo = IdealGasThermo(vib_e, **kw)
    zpe = float(thermo.get_ZPE_correction())
    h = float(thermo.get_enthalpy(temperature, verbose=False))
    s_harm = float(thermo.get_entropy(temperature, pressure, verbose=False))
    s = s_harm
    if qrrho and len(nu):
        s = s_harm - float(harmonic_vib_entropy(nu, temperature).sum()) \
            + float(qrrho_entropy(nu, temperature, qrrho_cutoff_cm1, qrrho_alpha).sum())
    return Thermochemistry(temperature, pressure, float(potential_energy_ev), zpe, h, s, h - temperature * s,
                           s_harm, qrrho, qrrho_cutoff_cm1, qrrho_alpha, symmetry_number, spin, geometry, len(nu))


# --------------------------------------------------------------------------- spectrum
@dataclass
class IRSpectrum:
    """A broadened IR spectrum on a regular wavenumber grid, normalized to a maximum of 1."""

    wavenumber_cm1: np.ndarray
    intensity: np.ndarray
    fwhm_cm1: float
    scale: float
    uniform_sticks: bool = False     # True when no intensities were available (positions only)

    def write_csv(self, path: str | Path) -> Path:
        return write_spectrum_csv(path, self)


def ir_spectrum(frequencies_cm1, intensities=None, fwhm: float = 20.0, scale: float = 1.0,
                x_min: float = 400.0, x_max: float = 4000.0, step: float = 1.0) -> IRSpectrum:
    """Lorentzian-broadened IR spectrum. Imaginary (negative) modes are skipped.

    Each stick at ``scale·ν̃ₖ`` with intensity Iₖ becomes Iₖ (γ/π) / ((x - scale·ν̃ₖ)² + γ²), γ = FWHM/2,
    which keeps band areas proportional to the integrated intensities. The sum is normalized to 1.
    With ``intensities=None`` every mode gets the same weight (``uniform_sticks``), e.g. for MMFF.
    """
    nu = np.asarray(frequencies_cm1, dtype=float)
    ints = np.ones_like(nu) if intensities is None else np.asarray(intensities, dtype=float)
    if ints.shape != nu.shape:
        raise ValueError("frequencies and intensities differ in length")
    keep = nu > 0
    centers = nu[keep] * scale
    ints = ints[keep]
    x = np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    g = 0.5 * fwhm
    y = (ints[None, :] * (g / math.pi) / ((x[:, None] - centers[None, :]) ** 2 + g**2)).sum(1)
    if y.size and y.max() > 0:
        y = y / y.max()
    return IRSpectrum(x, y, fwhm, scale, intensities is None)


def write_spectrum_csv(path: str | Path, spectrum: IRSpectrum) -> Path:
    """Write ``wavenumber_cm-1,intensity_normalized`` rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["wavenumber_cm-1", "intensity_normalized"])
        for x, y in zip(spectrum.wavenumber_cm1, spectrum.intensity):
            w.writerow([f"{x:.2f}", f"{y:.6f}"])
    return path


def write_modes_csv(path: str | Path, result: VibrationalResult, scale: float | None = None) -> Path:
    """Write the mode table: frequency (raw and scaled), IR intensity, reduced mass, force constant."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = scale_factor(result.method) if scale is None else scale
    ints = result.ir_intensities_km_mol
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "frequency_cm-1", "scaled_frequency_cm-1", "ir_intensity_km_mol",
                    "reduced_mass_amu", "force_constant_mdyn_A"])
        for i, f in enumerate(result.frequencies_cm1):
            w.writerow([i + 1, f"{f:.2f}", f"{f * scale:.2f}", "" if ints is None else f"{ints[i]:.3f}",
                        f"{result.reduced_masses_amu[i]:.4f}", f"{result.force_constants_mdyn_A[i]:.4f}"])
    return path


def mode_trajectory(atoms: Atoms, result: VibrationalResult, index: int, amplitude: float = 0.5,
                    n_frames: int = 20) -> list[Atoms]:
    """Frames of one normal mode for animation: x(t) = x₀ + A sin(2πt/n) d, d of unit norm (Å)."""
    disp = result.displacements[index]
    frames = []
    for t in range(n_frames):
        a = atoms.copy()
        a.calc = None
        a.positions = atoms.positions + amplitude * math.sin(2 * math.pi * t / n_frames) * disp
        frames.append(a)
    return frames


# --------------------------------------------------------------------------- one entry point
@dataclass
class VibrationalResult:
    """Frequencies, IR and thermochemistry for one structure."""

    method: str
    hessian_method: str                         # 'analytic' | 'finite'
    charge: int
    multiplicity: int
    device: str
    n_atoms: int
    geometry: str
    frequencies_cm1: list[float]                # imaginary as negative, ascending
    reduced_masses_amu: list[float]
    force_constants_mdyn_A: list[float]
    ir_intensities_km_mol: list[float] | None
    ir_source: str                              # how intensities were obtained ('' = none)
    n_imaginary: int
    thermo: Thermochemistry | None
    fd_step_A: float | None = None
    max_force_ev_A: float = 0.0
    scale_factor: float = 1.0
    scale_factor_source: str = ""
    imag_tol_cm1: float = 20.0
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    displacements: np.ndarray | None = field(default=None, repr=False)   # (n_modes, N, 3), unit norm
    hessian_ev_A2: np.ndarray | None = field(default=None, repr=False)   # symmetrized Cartesian Hessian
    positions: np.ndarray | None = field(default=None, repr=False)       # geometry analyzed (Å)

    @property
    def n_modes(self) -> int:
        return len(self.frequencies_cm1)

    @property
    def lowest_freq_cm1(self) -> float | None:
        return float(self.frequencies_cm1[0]) if self.frequencies_cm1 else None

    def descriptors(self) -> dict:
        """Scalars for descriptors.csv. Absolute H and G include the method's electronic energy, so they
        compare only within one method; ``h_corr_kcal``/``g_corr_kcal`` are the thermal corrections."""
        d: dict = {"n_imaginary": self.n_imaginary, "lowest_freq_cm1": self.lowest_freq_cm1}
        t = self.thermo
        if t is not None:
            d.update(zpe_kcal=t.zpe_kcal, enthalpy_kcal=t.enthalpy_kcal, entropy_cal_molK=t.entropy_cal_molK,
                     gibbs_kcal=t.gibbs_kcal, h_corr_kcal=t.h_corr_kcal, g_corr_kcal=t.g_corr_kcal,
                     symmetry_number=t.symmetry_number)
        return d

    def to_dict(self, include_modes: bool = False) -> dict:
        """JSON-ready summary (lists and floats only). Displacements only on request (they are 3N² numbers)."""
        d = {k: v for k, v in asdict(self).items()
             if k not in ("displacements", "hessian_ev_A2", "positions", "thermo")}
        d["thermo"] = self.thermo.to_dict() if self.thermo else None
        d["n_modes"] = self.n_modes
        d["lowest_freq_cm1"] = self.lowest_freq_cm1
        if include_modes and self.displacements is not None:
            d["displacements"] = np.round(self.displacements, 5).tolist()
        return d

    def spectrum(self, fwhm: float = 20.0, scale: float | None = None, **kw) -> IRSpectrum:
        scale = self.scale_factor if scale is None else scale
        return ir_spectrum(self.frequencies_cm1, self.ir_intensities_km_mol, fwhm, scale, **kw)


_DXTB = ("gfn1-xtb", "gfn2-xtb")


def _make_calc(method: str, atoms: Atoms, charge: int, multiplicity: int, device: str, mol) -> Calculator:
    if method == "mmff":
        return RdkitMMFFCalculator(mol if mol is not None else mol_from_atoms(atoms, charge))
    from . import engines

    return engines.make_calculator(method, atoms.numbers, charge, multiplicity, device)


def analyze(atoms: Atoms, method: str = "gfn1-xtb", charge: int = 0, multiplicity: int = 1, device: str = "cpu",
            temperature: float = 298.15, pressure: float = 101325.0, symmetry_number: int | None = 0,
            hessian: HessianMode = "auto", *, calc: Calculator | None = None, mol=None, fd_step: float = 0.01,
            ir: bool = True, qrrho: bool = True, qrrho_cutoff_cm1: float = 100.0, qrrho_alpha: float = 4.0,
            imag_tol_cm1: float = 20.0, preoptimize_fmax: float | None = None,
            max_analytic_memory_gb: float | None = None,
            should_stop: Callable[[], bool] | None = None,
            progress: Callable[[int, int], None] | None = None,
            log: Callable[[str], None] | None = None) -> VibrationalResult:
    """Frequencies, IR intensities and thermochemistry of ``atoms`` (ideally an optimized minimum).

    method:     'mmff' | 'gfn1-xtb' | 'gfn2-xtb' | 'aimnet2' (or any label when ``calc`` is given).
    hessian:    'auto' = analytic for dxtb methods when it fits in memory (see
                ``max_analytic_memory_gb``), finite differences otherwise;
                'analytic' = dxtb (or AIMNet2's autograd Hessian); 'finite' = central differences.
    calc:       use this ASE calculator instead of building one from ``method``.
    mol:        RDKit molecule with the topology for MMFF (bonds are perceived from 3D if omitted).
    imag_tol_cm1: modes below -tol count as imaginary and are left out of the thermochemistry;
                smaller imaginary values are numerical noise (loose optimization, finite differences)
                and are used as real frequencies |ν̃| in the thermochemistry, as Goodvibes' --invertfreq.
    symmetry_number: rotational symmetry number σ; 0 or None detects it from the analyzed geometry
                (``rotational_symmetry_number``), so each molecule of a batch gets its own.
    preoptimize_fmax: tighten the geometry (LBFGS) to this max force (eV/Å) before the Hessian.
    max_analytic_memory_gb: memory allowed for dxtb's autograd Hessian, whose peak grows ~n_AO³
                (a PFOA anion needs >7 GB); default half of the free RAM (or GPU memory).
    The atoms object is not modified; constraints are ignored.
    """
    from .optimize import optimize

    say = log or (lambda _msg: None)
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    work = Atoms(atoms.numbers, positions=atoms.get_positions())
    n = len(work)
    masses = isotope_masses(work.numbers)
    sf = SCALE_FACTORS.get(method.lower())

    if hessian not in ("auto", "analytic", "finite"):
        raise ValueError(f"hessian must be 'auto', 'analytic' or 'finite', not {hessian!r}")
    if n == 1:
        # No vibrations, but H and G must still include the electronic energy, as for every polyatomic
        # molecule, or ion-pairing and reaction ΔH/ΔG built from descriptors.csv are off by E_el of the ion.
        try:
            work.calc = calc or _make_calc(method, work, charge, multiplicity, device, mol)
            e_pot = float(work.get_potential_energy())
            warnings.append("single atom: no vibrations (translational and electronic terms only)")
        except Exception as exc:
            e_pot = 0.0
            warnings.append(f"single atom: no vibrations; electronic energy not available "
                            f"({type(exc).__name__}: {exc}), potential energy taken as 0")
        thermo = thermochemistry(work, [], e_pot, temperature, pressure, 1, multiplicity, "monatomic", masses,
                                 qrrho, qrrho_cutoff_cm1, qrrho_alpha)
        return VibrationalResult(method, "none", charge, multiplicity, device, 1, "monatomic", [], [], [], None, "",
                                 0, thermo, warnings=warnings, positions=work.get_positions())

    use_dxtb = method in _DXTB and calc is None
    if hessian == "analytic" and not use_dxtb and method != "aimnet2":
        raise ValueError(f"no analytic Hessian for {method!r}; use hessian='finite'")
    analytic = hessian == "analytic" or (hessian == "auto" and use_dxtb)
    if use_dxtb:
        need = estimate_dxtb_hessian_memory_gb(work.numbers, method)
        budget = max_analytic_memory_gb if max_analytic_memory_gb is not None else 0.5 * available_memory_gb(device)
        if hessian == "auto" and need > budget:
            analytic = False
            say(f"analytic Hessian would need ~{need:.1f} GB (budget {budget:.1f} GB): "
                "using finite differences of the analytic forces")
        elif analytic and need > budget:
            warnings.append(f"analytic Hessian may need ~{need:.1f} GB (budget {budget:.1f} GB)")

    calc = calc or _make_calc(method, work, charge, multiplicity, device, mol)
    work.calc = calc
    if preoptimize_fmax is not None:
        t = time.perf_counter()
        opt = optimize(work, fmax=preoptimize_fmax, steps=1000, should_stop=should_stop)
        timings["preoptimize"] = time.perf_counter() - t
        if not opt.converged:
            warnings.append(f"pre-optimization did not reach fmax {preoptimize_fmax} eV/Å "
                            f"in {opt.steps} steps (fmax {opt.fmax:.4f})")
    e_pot = float(work.get_potential_energy())
    fmax = float(np.sqrt((work.get_forces() ** 2).sum(1)).max())
    if fmax > 0.05:
        warnings.append(f"max force {fmax:.3f} eV/Å: not a stationary point, frequencies are approximate "
                        "(tighten the optimization or use preoptimize_fmax)")

    t = time.perf_counter()
    dmu = None
    ir_source = ""
    if analytic:
        say(f"Hessian: analytic ({method}) for {n} atoms")
        if method == "aimnet2" and not use_dxtb:
            h = aimnet_hessian(calc, work)
        else:
            try:
                h = dxtb_hessian(work.numbers, work.positions, method, charge, multiplicity, device)
            except Exception as exc:
                if device == "cpu":
                    raise
                warnings.append(f"analytic Hessian failed on {device} ({type(exc).__name__}: {exc}); used CPU")
                device = "cpu"
                h = dxtb_hessian(work.numbers, work.positions, method, charge, multiplicity, device)
        hessian_method, step = "analytic", None
    else:
        say(f"Hessian: central finite differences, step {fd_step} Å, {6 * n} force calls")
        h, dmu = finite_difference_hessian(work, fd_step, dipole=ir and not use_dxtb,
                                           should_stop=should_stop, progress=progress)
        if dmu is not None:
            ir_source = dipole_source(work)
        hessian_method, step = "finite", fd_step
    h = symmetrize(h)
    timings["hessian"] = time.perf_counter() - t

    if ir and use_dxtb:
        t = time.perf_counter()
        try:
            dmu = dxtb_dipole_derivatives(work.numbers, work.positions, method, charge, multiplicity, device)
            ir_source = f"analytic dipole derivatives ({method}, dxtb)"
        except Exception as exc:
            warnings.append(f"IR intensities unavailable ({type(exc).__name__}: {exc})")
        timings["dipole_derivatives"] = time.perf_counter() - t

    geometry = detect_geometry(work.positions, masses)
    if not symmetry_number:              # 0 / None: detect σ from this (optimized) geometry
        symmetry_number = rotational_symmetry_number(work.numbers, work.positions, masses)
        say(f"rotational symmetry number σ = {symmetry_number} (detected)")
    nm = normal_modes(h, masses, work.positions, geometry)
    freqs = nm.frequencies_cm1
    ints = ir_intensities(dmu, nm, masses) if dmu is not None else None
    if ints is None and ir and not ir_source:
        say("IR intensities: not available for this engine")

    n_imag = int((freqs < -imag_tol_cm1).sum())
    small = freqs[(freqs < 0) & (freqs >= -imag_tol_cm1)]
    if n_imag:
        warnings.append(f"{n_imag} imaginary frequenc{'y' if n_imag == 1 else 'ies'} "
                        f"(lowest {freqs[0]:.1f} cm-1): not a minimum; imaginary modes are excluded from the "
                        "thermochemistry")
    if len(small):
        warnings.append(f"{len(small)} small imaginary mode(s) above -{imag_tol_cm1:g} cm-1 treated as numerical "
                        "noise (used as real frequencies in the thermochemistry)")
    thermo_freqs = np.abs(freqs[freqs >= -imag_tol_cm1])
    thermo_freqs = thermo_freqs[thermo_freqs > 1e-3]

    t = time.perf_counter()
    thermo = thermochemistry(work, thermo_freqs, e_pot, temperature, pressure, symmetry_number, multiplicity,
                             geometry, masses, qrrho, qrrho_cutoff_cm1, qrrho_alpha)
    timings["thermo"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t_start
    for w in warnings:
        say(f"  {w}")

    return VibrationalResult(
        method=method, hessian_method=hessian_method, charge=charge, multiplicity=multiplicity, device=device,
        n_atoms=n, geometry=geometry, frequencies_cm1=[float(f) for f in freqs],
        reduced_masses_amu=[float(x) for x in nm.reduced_masses_amu],
        force_constants_mdyn_A=[float(x) for x in nm.force_constants_mdyn_A],
        ir_intensities_km_mol=None if ints is None else [float(x) for x in ints], ir_source=ir_source,
        n_imaginary=n_imag, thermo=thermo, fd_step_A=step, max_force_ev_A=fmax,
        scale_factor=sf.value if sf else 1.0, scale_factor_source=sf.source if sf else "",
        imag_tol_cm1=imag_tol_cm1, warnings=warnings, timings=timings, displacements=nm.displacements,
        hessian_ev_A2=h, positions=work.get_positions())


def analyze_many(items: Sequence[Atoms], **kwargs) -> list[VibrationalResult]:
    """Convenience loop (e.g. a few conformers); each call is independent."""
    return [analyze(a, **kwargs) for a in items]
