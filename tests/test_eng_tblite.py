"""tblite engines (chem.eng_tblite): registration, solvation mapping, and GFN1/GFN2 numbers checked against
the xtb program 6.7.1 and dxtb's analytic polarizability at ASE's reference geometries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule

from quantabench.chem import eng_tblite as et
from quantabench.chem import engines

EH = engines.HARTREE_EV

# References at ase.build.molecule geometries: energies, HOMO/LUMO and dipole printed by xtb 6.7.1
# (``xtb x.xyz --gfn N``, ``--alpb water``); polarizabilities from dxtb 0.4 analytic (libcint), same Hamiltonian.
REF = {
    ("H2O", "GFN2-xTB"): {"energy": -5.070222287, "homo": -12.1663, "lumo": 2.0111, "dipole_debye": 2.286,
                          "alpha_iso_au": 2.51448},
    ("H2O", "GFN1-xTB"): {"energy": -5.768449488, "homo": -13.6057, "lumo": -4.3471, "alpha_iso_au": 1.95614},
    ("CH4", "GFN2-xTB"): {"energy": -4.175074574, "homo": -12.7051, "lumo": 4.2711, "alpha_iso_au": 6.52958},
}
WATER_ALPB_GFN2 = -5.084802152


@pytest.fixture
def tb(monkeypatch):
    """tblite, loaded with one OpenMP thread (OpenMP reads OMP_NUM_THREADS once, at library load)."""
    if "tblite" not in sys.modules:
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
    return pytest.importorskip("tblite")


# ----------------------------------------------------------------------------- no tblite needed
def test_engines_registered():
    eng = engines.all_engines()
    for key, prefix in (("tblite-gfn2", "tblite_gfn2"), ("tblite-gfn1", "tblite_gfn1")):
        info = eng[key]
        assert info.family == "tight binding" and info.solvation == ("alpb", "gbsa", "cpcm")
        assert info.external == "" and not info.gpu_capable and info.prop_prefix == prefix
        assert info.elements == frozenset(range(1, 87))
        assert engines.properties_function(key) is not None and engines.native_optimizer(key) is None
        assert engines.supports_solvent(key, "water") and engines.supports_solvent(key, "octanol")
    assert eng["tblite-gfn2"].label == "GFN2-xTB (tblite)"


def test_unavailable_gives_install_hint(monkeypatch):
    monkeypatch.setattr(et, "has_package", lambda name: False)
    ok, why = engines.available("tblite-gfn2")
    assert not ok and "pip install tblite" in why and "conda-forge tblite-python" in why


def test_import_is_cheap():
    """Importing the plugins must not import tblite or start MOPAC (fresh interpreter)."""
    src = str(Path(et.__file__).resolve().parents[2])
    code = ("import sys, subprocess\n"
            "calls = []\n"
            "orig = subprocess.Popen.__init__\n"
            "def spy(self, *a, **k):\n"
            "    calls.append(str(a[0] if a else k.get('args')))\n"
            "    return orig(self, *a, **k)\n"
            "subprocess.Popen.__init__ = spy\n"
            "from quantabench.chem import engines\n"
            "engines.load_plugins()\n"
            "assert 'tblite' not in sys.modules, 'tblite imported'\n"
            "assert not [c for c in calls if 'mopac' in c.lower()], calls\n"
            "assert 'tblite-gfn2' in engines.ENGINES and 'mopac-pm7' in engines.ENGINES, engines.PLUGIN_ERRORS\n")
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cp = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert cp.returncode == 0, cp.stderr


def test_solvation_args():
    assert et.solvation_args("", "alpb") is None
    assert et.solvation_args("water", "alpb") == ("alpb-solvation", "water")
    assert et.solvation_args("dichloromethane", "gbsa") == ("gbsa-solvation", "ch2cl2")
    assert et.solvation_args("water", "cpcm") == ("ddX-solvation", pytest.approx(78.36), "cpcm")
    with pytest.raises(ValueError):
        et.solvation_args("water", "smd")


def test_frontier_orbitals_closed_open_and_spin_polarized():
    eps = np.array([-0.6, -0.5, -0.4, 0.1, 0.2])
    homo, lumo = et.frontier_orbitals(eps, [2, 2, 2, 0, 0])
    assert homo == pytest.approx(-0.4 * EH) and lumo == pytest.approx(0.1 * EH)
    homo, lumo = et.frontier_orbitals(eps, [2, 2, 1, 0, 0], uhf=1)        # restricted doublet: SOMO is the HOMO
    assert homo == pytest.approx(-0.4 * EH) and lumo == pytest.approx(0.1 * EH)
    homo, lumo = et.frontier_orbitals(eps, [2, 2, 1.9, 0.1, 0])           # Fermi smearing does not move them
    assert homo == pytest.approx(-0.4 * EH)
    eps2 = np.array([[-0.6, -0.4, 0.1], [-0.55, -0.2, 0.15]])
    homo, lumo = et.frontier_orbitals(eps2, np.array([[1, 1, 0], [1, 0, 0]]))
    assert homo == pytest.approx(-0.4 * EH) and lumo == pytest.approx(-0.2 * EH)


def test_polarizability_summary_units():
    iso, aniso = et.polarizability_summary(np.diag([1.0, 2.0, 3.0]))
    assert iso == pytest.approx(2.0 * 0.148184711) and aniso == pytest.approx(2.0 * 0.148184711)


# ----------------------------------------------------------------------------- tblite
@pytest.mark.parametrize("name,method", list(REF))
def test_energy_orbitals_and_polarizability_match_references(tb, name, method):
    ref = REF[(name, method)]
    atoms = molecule(name)
    calc = et.TBLiteCalculator(atoms.numbers, method)
    atoms.calc = calc
    assert atoms.get_potential_energy() / EH == pytest.approx(ref["energy"], abs=1e-6)
    homo, lumo = calc.frontier_orbitals_ev()
    assert homo == pytest.approx(ref["homo"], abs=2e-3) and lumo == pytest.approx(ref["lumo"], abs=2e-3)
    if "dipole_debye" in ref:
        assert np.linalg.norm(atoms.get_dipole_moment()) * et.DEBYE_PER_E_ANG == pytest.approx(ref["dipole_debye"],
                                                                                              abs=2e-3)
    alpha = calc.polarizability_au(atoms.positions)
    assert np.trace(alpha) / 3 == pytest.approx(ref["alpha_iso_au"], abs=1e-3)
    assert np.allclose(alpha, alpha.T, atol=1e-3)
    assert np.trace(calc.polarizability_au(atoms.positions, step=1e-3)) / 3 == pytest.approx(ref["alpha_iso_au"],
                                                                                              abs=1e-3)


def test_forces_match_finite_differences_and_restart_is_exact(tb):
    atoms = molecule("CH3OH")
    atoms.rattle(0.05, seed=1)
    calc = et.TBLiteCalculator(atoms.numbers, "GFN2-xTB")
    atoms.calc = calc
    forces = atoms.get_forces()
    h = 1e-4
    for i, k in ((0, 0), (1, 2), (4, 1)):
        b = atoms.copy()
        b.calc = calc                      # restarts from the previous wavefunction every time
        b.positions[i, k] += h
        ep = b.get_potential_energy()
        b.positions[i, k] -= 2 * h
        em = b.get_potential_energy()
        assert -(ep - em) / (2 * h) == pytest.approx(forces[i, k], abs=2e-4)
    fresh = et.TBLiteCalculator(atoms.numbers, "GFN2-xTB")
    assert fresh.singlepoint(b.positions).get("energy") * EH == pytest.approx(b.get_potential_energy(), abs=1e-6)
    assert calc.n_singlepoints >= 7


def test_solvation_models(tb):
    atoms = molecule("H2O")
    e_gas = et.TBLiteCalculator(atoms.numbers).singlepoint(atoms.positions).get("energy")
    alpb = et.TBLiteCalculator(atoms.numbers, solvent="water")
    assert alpb.solvation_label == "ALPB(water)"
    assert alpb.singlepoint(atoms.positions).get("energy") == pytest.approx(WATER_ALPB_GFN2, abs=1e-6)
    cpcm = et.TBLiteCalculator(atoms.numbers, solvent="water", solvation_model="cpcm")
    assert cpcm.singlepoint(atoms.positions).get("energy") < e_gas
    gbsa = et.TBLiteCalculator(atoms.numbers, solvent="ethanol", solvation_model="gbsa")   # no GBSA parameters
    alpb_e = et.TBLiteCalculator(atoms.numbers, solvent="ethanol").singlepoint(atoms.positions).get("energy")
    assert gbsa.singlepoint(atoms.positions).get("energy") == pytest.approx(alpb_e, abs=1e-8)
    assert gbsa.solvation_label == "ALPB(ethanol)" and any("used ALPB" in n for n in gbsa.notes)


def test_failures_raise_clear_runtime_errors(tb):
    atoms = molecule("H2O")
    with pytest.raises(RuntimeError, match="not converged"):
        et.TBLiteCalculator(atoms.numbers, max_iterations=1).singlepoint(atoms.positions)
    with pytest.raises(RuntimeError, match="unpaired"):
        et.TBLiteCalculator(atoms.numbers, multiplicity=2).singlepoint(atoms.positions)
    with pytest.raises(RuntimeError, match="close"):
        et.TBLiteCalculator(atoms.numbers).singlepoint(np.zeros((3, 3)))


def test_cation_doublet(tb):
    atoms = molecule("H2O")
    calc = et.TBLiteCalculator(atoms.numbers, charge=1, multiplicity=2)
    calc.singlepoint(atoms.positions)
    homo, lumo = calc.frontier_orbitals_ev()
    occ = calc._res.get("orbital-occupations")
    eps = calc._res.get("orbital-energies") * EH
    assert homo == pytest.approx(eps[3]) and lumo == pytest.approx(eps[4]) and occ[3] == pytest.approx(1.0)


def test_properties_function(tb):
    atoms = molecule("H2O")
    props = engines.properties_function("tblite-gfn2")(atoms.numbers, atoms.positions, 0, 1, solvent="", threads=1)
    assert props["tblite_gfn2_dipole_debye"] == pytest.approx(2.286, abs=2e-3)
    assert props["tblite_gfn2_gap_ev"] == pytest.approx(2.0111 + 12.1663, abs=3e-3)
    assert props["tblite_gfn2_polarizability_iso_A3"] == pytest.approx(2.51448 * 0.148184711, abs=2e-4)
    assert props["tblite_gfn2_polarizability_aniso_A3"] > 0.5      # planar molecule: out-of-plane alpha ~ 0
    assert len(props["charges"]) == 3 and sum(props["charges"]) == pytest.approx(0.0, abs=1e-6)
    assert "_notes" not in props
    solv = engines.properties_function("tblite-gfn1")(atoms.numbers, atoms.positions, 0, 1, solvent="water")
    assert solv["tblite_gfn1_dipole_debye"] > 0 and solv["tblite_gfn1_polarizability_iso_A3"] == pytest.approx(
        1.95614 * 0.148184711, abs=2e-4)
    assert any("gas phase" in n for n in solv["_notes"])


def test_ion_dipole_is_about_center_of_mass(tb):
    atoms = molecule("H2O")
    props = et._properties("GFN2-xTB", "p")(atoms.numbers, atoms.positions, 1, 2, polarizability=False)
    shifted = et._properties("GFN2-xTB", "p")(atoms.numbers, atoms.positions + 5.0, 1, 2, polarizability=False)
    assert props["p_dipole_debye"] == pytest.approx(shifted["p_dipole_debye"], abs=1e-4)


def test_pipeline_with_solvent(tb):
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    cfg = BuildConfig(method="tblite-gfn2", solvent="water", max_conformers=2, refine_top=1, write_files=False,
                      xtb_properties=False, threads=1, frequencies=True)
    res = run_molecule(from_smiles("CO", id="methanol"), cfg)
    assert res.status == "ok", res.error
    assert res.method == "tblite-gfn2" and res.solvent == "water"
    d = res.descriptors
    assert d["tblite_gfn2_dipole_debye"] > 1.5 and d["tblite_gfn2_polarizability_iso_A3"] > 0.5
    assert res.conformers[0].converged and res.conformers[0].charges is not None
    vib = res.vibrations
    assert vib and vib["n_imaginary"] == 0 and len(vib["frequencies_cm1"]) == 12
    assert "dipole from the calculator" in vib["ir_source"] and max(vib["ir_intensities_km_mol"]) > 10
