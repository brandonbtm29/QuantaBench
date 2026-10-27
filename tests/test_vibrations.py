"""Vibrational analysis: Hessians, normal modes, IR intensities, thermochemistry, spectra.

dxtb tests use water and CO2 (3 atoms) so each analytic Hessian takes a couple of seconds on CPU;
MMFF tests need only RDKit.
"""

import json

import numpy as np
import pytest
from ase import Atoms
from ase import units as ase_units
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem import AllChem

from quantabench.chem import engines
from quantabench.chem import vibrations as vib
from quantabench.chem.optimize import optimize


def _dxtb():
    pytest.importorskip("torch")
    pytest.importorskip("dxtb")


def _mmff_mol(smiles: str, seed: int = 7) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=2000)
    return mol


def _atoms(mol: Chem.Mol) -> Atoms:
    return Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions())


@pytest.fixture(scope="module")
def water_gfn1():
    _dxtb()
    return vib.analyze(molecule("H2O"), "gfn1-xtb", symmetry_number=2, preoptimize_fmax=1e-4)


# --------------------------------------------------------------------------- dxtb
def test_water_gfn1_modes_and_ir(water_gfn1):
    r = water_gfn1
    assert r.hessian_method == "analytic" and r.geometry == "nonlinear"
    assert r.n_modes == 3 and r.n_imaginary == 0
    bend, sym, asym = r.frequencies_cm1
    assert 1400 < bend < 1800
    assert 3200 < sym < 4000 and 3200 < asym < 4000
    ints = r.ir_intensities_km_mol
    if not engines.has_libcint():  # native Windows: frequencies only, IR skipped with a warning
        assert ints is None and any("IR intensities skipped" in w for w in r.warnings)
    else:
        assert ints is not None and "analytic" in r.ir_source
        assert all(i > 0 for i in ints)
        assert ints[2] > ints[1]              # asymmetric stretch stronger than symmetric
    # reduced masses: bend and stretches of water are ~1.0-1.1 amu
    assert all(0.9 < m < 1.2 for m in r.reduced_masses_amu)


def test_water_thermo(water_gfn1):
    t = water_gfn1.thermo
    assert t.zpe_ev > 0 and 10 < t.zpe_kcal < 15
    assert t.gibbs_ev < t.enthalpy_ev
    # Experimental standard entropy of water vapor is 45.1 cal/(mol K); RRHO with sigma = 2 is close.
    assert 43 < t.entropy_cal_molK < 47
    # all modes are far above 100 cm-1, so qRRHO equals the harmonic entropy here
    assert t.entropy_ev_k == pytest.approx(t.entropy_harmonic_ev_k, rel=1e-6)
    d = water_gfn1.descriptors()
    assert set(d) >= {"zpe_kcal", "enthalpy_kcal", "entropy_cal_molK", "gibbs_kcal", "n_imaginary",
                      "lowest_freq_cm1"}
    assert d["h_corr_kcal"] > d["zpe_kcal"] > 0
    json.dumps(water_gfn1.to_dict(include_modes=True))      # JSON-ready


def test_analytic_vs_finite_difference(water_gfn1):
    _dxtb()
    a = Atoms("OH2", positions=water_gfn1.positions)
    # 'auto' switches to finite differences when the analytic Hessian would not fit in memory
    fd = vib.analyze(a, "gfn1-xtb", hessian="auto", max_analytic_memory_gb=0.0, symmetry_number=2)
    assert fd.hessian_method == "finite" and fd.fd_step_A == 0.01
    assert np.allclose(fd.frequencies_cm1, water_gfn1.frequencies_cm1, atol=15.0)
    assert np.allclose(fd.hessian_ev_A2, water_gfn1.hessian_ev_A2, atol=0.05)


def test_co2_is_linear():
    _dxtb()
    r = vib.analyze(molecule("CO2"), "gfn1-xtb", symmetry_number=2, preoptimize_fmax=1e-4)
    assert r.geometry == "linear" and r.n_modes == 4 and r.n_imaginary == 0
    f = r.frequencies_cm1
    assert f[0] == pytest.approx(f[1], abs=1.0)         # degenerate bend
    ints = r.ir_intensities_km_mol
    if engines.has_libcint():
        assert ints[2] < 1e-3                          # symmetric stretch is IR-inactive
        assert ints[3] > ints[0] > 1.0                 # asymmetric stretch is the strongest band
    assert r.thermo.geometry == "linear"


# --------------------------------------------------------------------------- MMFF
def test_mmff_calculator_matches_rdkit():
    mol = _mmff_mol("CCO")
    calc = vib.RdkitMMFFCalculator(mol)
    a = _atoms(mol)
    a.calc = calc
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    e_ref = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
    assert a.get_potential_energy() == pytest.approx(e_ref / vib.KCAL_PER_EV, rel=1e-8)
    # forces are minus the numerical gradient of the energy
    x0 = a.get_positions()
    h = 1e-4
    xp, xm = x0.copy(), x0.copy()
    xp[0, 2] += h
    xm[0, 2] -= h
    a.positions = xp
    ep = a.get_potential_energy()
    a.positions = xm
    em = a.get_potential_energy()
    a.positions = x0
    assert a.get_forces()[0, 2] == pytest.approx(-(ep - em) / (2 * h), abs=1e-4)


def test_ethane_mmff_finite_difference():
    mol = _mmff_mol("CC")
    r = vib.analyze(_atoms(mol), "mmff", mol=mol, symmetry_number=6, preoptimize_fmax=1e-4)
    assert r.hessian_method == "finite"
    assert r.n_modes == 18 and r.n_imaginary == 0
    assert r.ir_intensities_km_mol is None and r.ir_source == ""
    # torsion lowest (~300 cm-1), C-H stretches highest (~2900-3100 cm-1)
    assert 150 < r.lowest_freq_cm1 < 450
    assert 2700 < r.frequencies_cm1[-1] < 3300
    assert r.thermo.zpe_ev > 0 and r.thermo.gibbs_ev < r.thermo.enthalpy_ev


def test_mmff_bond_perception_without_mol():
    mol = _mmff_mol("CC")
    r = vib.analyze(_atoms(mol), "mmff", preoptimize_fmax=1e-3)
    assert r.n_modes == 18 and r.n_imaginary == 0


def test_planar_ammonia_is_a_saddle():
    """NH3 optimized with all atoms held in a plane is the inversion transition state."""
    from ase.constraints import FixCartesian

    mol = _mmff_mol("N")
    a = _atoms(mol)
    a.positions[:, 2] = 0.0
    a.calc = vib.RdkitMMFFCalculator(mol)
    a.set_constraint(FixCartesian(range(len(a)), mask=(False, False, True)))
    optimize(a, fmax=1e-4, steps=500)
    r = vib.analyze(a, "mmff", mol=mol, symmetry_number=3)
    assert r.n_imaginary >= 1
    assert r.frequencies_cm1[0] < -200                 # umbrella mode
    assert any("imaginary" in w for w in r.warnings)
    assert r.thermo.n_modes_used == r.n_modes - r.n_imaginary


def test_charges_finite_difference_ir_and_aimnet_mock(monkeypatch):
    """Fixed point charges: dmu/dR_i = q_i exactly, so the FD dipole path can be checked analytically.

    The same calculator stands in for AIMNet2 (which reports charges), since its weights cannot be
    downloaded in CI."""
    mol = _mmff_mol("C=O")
    calc = vib.RdkitMMFFCalculator(mol, report_charges=True)
    monkeypatch.setattr(engines, "make_calculator", lambda *a, **k: calc)
    r = vib.analyze(_atoms(mol), "aimnet2", preoptimize_fmax=1e-4, symmetry_number=2)
    assert r.hessian_method == "finite"
    assert "charges" in r.ir_source and "approximate" in r.ir_source
    q = calc._charges
    masses = vib.isotope_masses(_atoms(mol).numbers)
    nm = vib.normal_modes(r.hessian_ev_A2, masses, r.positions)
    dmu = np.zeros((3, 3 * len(q)))
    for i, qi in enumerate(q):
        dmu[:, 3 * i:3 * i + 3] = qi * np.eye(3)
    expected = vib.ir_intensities(dmu, nm, masses)
    assert np.allclose(r.ir_intensities_km_mol, expected, rtol=1e-4, atol=1e-3)


def test_explicit_calculator_and_bad_hessian_mode():
    mol = _mmff_mol("O")
    with pytest.raises(ValueError):
        vib.analyze(_atoms(mol), "mmff", mol=mol, hessian="analytic")
    r = vib.analyze(_atoms(mol), "custom", calc=vib.RdkitMMFFCalculator(mol), preoptimize_fmax=1e-4)
    assert r.n_modes == 3 and r.scale_factor == 1.0


# --------------------------------------------------------------------------- pure numerics
class Harmonic(Calculator):
    """Diatomic spring: k = 5 eV/Å², r0 = 1 Å."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        d = atoms.positions[1] - atoms.positions[0]
        r = np.linalg.norm(d)
        f = -5.0 * (r - 1.0) * d / r
        self.results = {"energy": 2.5 * (r - 1.0) ** 2, "forces": np.array([-f, f])}


def test_diatomic_spring_frequency():
    """ν̃ = sqrt(k/μ)/(2πc): 5 eV/Å² with two H masses gives an analytic reference."""
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    r = vib.analyze(a, "spring", calc=Harmonic(), hessian="finite", fd_step=1e-3, qrrho=False)
    m = vib.isotope_masses([1])[0]
    expected = np.sqrt(5.0 / (m / 2)) * vib._EIG_TO_CM1
    assert r.geometry == "linear" and r.n_modes == 1
    assert r.frequencies_cm1[0] == pytest.approx(expected, rel=1e-5)
    # Gaussian convention (unit-norm Cartesian displacement): H2 has mu = m_H, k = mu * omega^2 = 2 k_bond
    assert r.reduced_masses_amu[0] == pytest.approx(m, rel=1e-6)
    assert r.force_constants_mdyn_A[0] == pytest.approx(10.0 * vib._EV_A2_TO_MDYN_A, rel=1e-5)


def test_unit_constants():
    assert vib._EIG_TO_CM1 == pytest.approx(521.47, rel=1e-4)
    assert vib.KM_MOL_PER_E2_AMU == pytest.approx(974.86, rel=1e-3)


def test_qrrho_limits():
    nu = np.array([5.0, 50.0, 100.0, 1000.0])
    s_h = vib.harmonic_vib_entropy(nu, 298.15)
    s_q = vib.qrrho_entropy(nu, 298.15)
    assert s_q[-1] == pytest.approx(s_h[-1], rel=1e-3)     # high frequencies: harmonic
    assert s_q[0] < s_h[0]                                 # very soft modes: capped by the free rotor
    w = 1 / (1 + (100 / 100.0) ** 4)
    assert w == 0.5
    # with the cutoff pushed to ~0 the interpolation is purely harmonic
    assert np.allclose(vib.qrrho_entropy(nu, 298.15, cutoff_cm1=1e-6), s_h)


def test_spectrum_peaks_match_frequencies(tmp_path):
    freqs = [-150.0, 800.0, 1700.0, 3000.0]
    ints = [10.0, 20.0, 100.0, 50.0]
    s = vib.ir_spectrum(freqs, ints, fwhm=20.0, x_min=400, x_max=4000, step=0.5)
    y = s.intensity
    peaks = s.wavenumber_cm1[1:-1][(y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])]
    assert np.allclose(peaks, [800.0, 1700.0, 3000.0], atol=0.5)
    assert y.max() == pytest.approx(1.0)
    assert s.wavenumber_cm1[np.argmax(y)] == pytest.approx(1700.0, abs=0.5)
    scaled = vib.ir_spectrum(freqs, ints, scale=0.95, step=0.5)
    assert scaled.wavenumber_cm1[np.argmax(scaled.intensity)] == pytest.approx(1615.0, abs=0.5)
    path = vib.write_spectrum_csv(tmp_path / "ir.csv", s)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "wavenumber_cm-1,intensity_normalized" and len(lines) == len(y) + 1


def test_modes_csv_and_trajectory(tmp_path, water_gfn1):
    p = vib.write_modes_csv(tmp_path / "modes.csv", water_gfn1)
    rows = p.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4 and rows[0].startswith("mode,frequency_cm-1")
    a = Atoms("OH2", positions=water_gfn1.positions)
    frames = vib.mode_trajectory(a, water_gfn1, 2, amplitude=0.3, n_frames=8)
    assert len(frames) == 8 and np.allclose(frames[0].positions, a.positions)
    assert np.linalg.norm(frames[2].positions - a.positions) == pytest.approx(0.3, rel=1e-6)


def test_single_atom():
    r = vib.analyze(Atoms("Ar"), "mmff")
    assert r.n_modes == 0 and r.geometry == "monatomic" and r.thermo.zpe_ev == 0.0


def test_geometry_detection():
    m = np.ones(3)
    assert vib.detect_geometry([[0, 0, 0], [0, 0, 1], [0, 0, 2]], m) == "linear"
    assert vib.detect_geometry([[0, 0, 0], [0, 0, 1], [0, 1, 2]], m) == "nonlinear"
    assert vib.detect_geometry([[0, 0, 0]], [1.0]) == "monatomic"


def test_rotational_symmetry_number():
    expected = {"H2O": 2, "NH3": 3, "CH4": 12, "C6H6": 12, "C2H6": 6, "C2H4": 4, "CH3Cl": 3, "BF3": 6,
                "CO2": 2, "HCl": 1, "H2": 2, "CH3CH2OH": 1, "C3H8": 2}
    rng = np.random.default_rng(1)
    for name, sigma in expected.items():
        a = molecule(name)
        assert vib.rotational_symmetry_number(a.numbers, a.positions, a.get_masses()) == sigma, name
        noisy = a.positions + rng.normal(0, 0.005, a.positions.shape)   # an optimizer's residual asymmetry
        assert vib.rotational_symmetry_number(a.numbers, noisy, a.get_masses()) == sigma, name


def test_symmetry_number_detected_per_molecule():
    """Methane's entropy must include -R ln 12 without the user setting σ (experiment 44.5 cal/(mol K))."""
    auto = vib.analyze(molecule("CH4"), "mmff", hessian="finite")
    one = vib.analyze(molecule("CH4"), "mmff", hessian="finite", symmetry_number=1)
    assert auto.thermo.symmetry_number == 12 and one.thermo.symmetry_number == 1
    r_ln12 = 1.987204 * np.log(12)
    assert one.thermo.entropy_cal_molK - auto.thermo.entropy_cal_molK == pytest.approx(r_ln12, abs=0.01)
    assert auto.descriptors()["symmetry_number"] == 12


def test_single_atom_includes_electronic_energy():
    class Fixed(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 3.0, "forces": np.zeros((len(atoms), 3))}

    r = vib.analyze(Atoms("Na"), "custom", charge=1, calc=Fixed())
    t = r.thermo
    assert t.potential_energy_ev == 3.0
    assert t.enthalpy_ev == pytest.approx(3.0 + 2.5 * ase_units.kB * 298.15, rel=1e-6)   # E + 5/2 kT
    assert not any("taken as 0" in w for w in r.warnings)
