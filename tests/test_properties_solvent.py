"""The water phase of the pipeline: every step that has an implicit solvent computes in it (refinement, the
engine's properties, the GFN1-xTB properties, the Hessian), and summary.csv says per step where it ran.
The finite-field polarizability with ALPB is checked against the second derivative of the energy."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from ase.build import molecule

from quantabench.chem import pipeline, properties
from quantabench.chem.inputs import from_smiles

A3 = 0.529177210903**3


@pytest.fixture
def tb(monkeypatch):
    if "tblite" not in sys.modules:
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
    return pytest.importorskip("tblite")


@pytest.fixture
def gas_xtb():
    """The gas-phase xtb_* columns come from dxtb (torch): without it the gas runs have none to compare."""
    if not properties.available():
        pytest.skip("dxtb/torch not installed: no gas-phase GFN1-xTB properties")


@pytest.mark.parametrize("method", ["GFN2-xTB", "GFN1-xTB"])
def test_alpb_finite_field_polarizability_matches_the_energy_second_derivative(tb, method):
    """tblite takes a uniform field and ALPB in one SCF. The dipole-derivative polarizability (what the pipeline
    reports) equals -d2E/dF2 of the same calculations, in water and in the gas phase, and the reaction field makes
    the solute more polarizable."""
    from quantabench.chem.eng_tblite import TBLiteCalculator

    atoms = molecule("CH3OH")
    iso = {}
    for solvent in ("", "water"):
        calc = TBLiteCalculator(atoms.numbers, method, solvent=solvent, accuracy=0.01)
        calc.singlepoint(atoms.positions)
        alpha = calc.polarizability_au(atoms.positions)
        h = 2e-3
        second = np.zeros(3)
        for k in range(3):
            e = []
            for s in (1, 0, -1):
                field = np.zeros(3)
                field[k] = s * h
                fc = TBLiteCalculator(atoms.numbers, method, solvent=solvent, accuracy=0.001,
                                      electric_field=field if s else None)
                e.append(float(fc.singlepoint(atoms.positions).get("energy")))
            second[k] = -(e[0] - 2 * e[1] + e[2]) / h**2
        assert np.diag(alpha) * A3 == pytest.approx(second * A3, abs=2e-3)
        assert np.allclose(alpha, alpha.T, atol=1e-3)
        iso[solvent] = np.trace(alpha) / 3 * A3
    assert iso[""] * 1.02 < iso["water"] < iso[""] * 1.4


def test_tblite_gfn1_properties_match_dxtb_in_the_gas_phase(tb):
    """The gas phase keeps dxtb for the xtb_* columns, the water phase uses tblite: the same GFN1-xTB numbers."""
    pytest.importorskip("dxtb")
    if not properties.available():
        pytest.skip("dxtb/torch not usable")
    atoms = molecule("CH3OH")
    args = (atoms.numbers, atoms.positions, atoms.get_masses(), 0, 1)
    ref = properties.xtb_properties(*args, polarizability=False)
    tbl = properties.tblite_xtb_properties(*args, polarizability=False)
    assert np.allclose(ref["xtb_charges"], tbl["xtb_charges"], atol=2e-4)
    assert tbl["xtb_dipole_debye"] == pytest.approx(ref["xtb_dipole_debye"], abs=2e-3)
    assert tbl["_solvation"] == "gas phase"


def test_tblite_gfn1_properties_in_water(tb):
    atoms = molecule("CH3OH")
    args = (atoms.numbers, atoms.positions, atoms.get_masses(), 0, 1)
    gas = properties.tblite_xtb_properties(*args)
    wat = properties.tblite_xtb_properties(*args, solvent="water")
    assert wat["_solvation"] == "ALPB(water)"
    assert wat["xtb_dipole_debye"] > gas["xtb_dipole_debye"]             # the reaction field polarizes the solute
    assert wat["xtb_polarizability_iso_A3"] > gas["xtb_polarizability_iso_A3"]
    assert abs(np.array(wat["xtb_charges"]) - np.array(gas["xtb_charges"])).max() > 1e-3
    assert sum(wat["xtb_charges"]) == pytest.approx(0.0, abs=1e-6)


def _run(solvent, method="tblite-gfn2", smiles="CC(=O)[O-]", **kw):
    cfg = pipeline.BuildConfig(method=method, solvent=solvent, max_conformers=2, refine_top=1, write_files=False,
                               threads=1, protonate=False, **kw)
    res = pipeline.run_molecule(from_smiles(smiles, id="acetate"), cfg)
    assert res.status == "ok", res.error
    return res


def test_water_phase_is_solvated_in_every_step(tb, gas_xtb):
    """tblite-gfn2 in water: refinement, engine properties (polarizability included), the GFN1-xTB properties
    (tblite, not dxtb's gas phase) and the Hessian all ran in ALPB water, and summary.csv says so."""
    wat = _run("water", frequencies=True)
    gas = _run("", frequencies=True)
    steps = wat.phase_steps
    assert steps["refine"] == steps["properties"] == steps["frequencies"] == "ALPB(water)"
    assert steps["xtb_properties"] == "ALPB(water), tblite GFN1-xTB"
    assert steps["conformers"].startswith("gas (MMFF94s has no implicit solvent")
    row = wat.summary_row()
    assert row["phase"] == "water" and row["solvent"] == "water" and row["solvation_model"] == "alpb"
    assert "frequencies: ALPB(water)" in row["phase_notes"] and "xtb_properties: ALPB(water)" in row["phase_notes"]
    assert gas.summary_row()["phase"] == "gas" and gas.summary_row()["phase_notes"] == "gas"
    assert gas.summary_row()["solvation_model"] == ""
    dw, dg = wat.descriptors, gas.descriptors
    assert dw["xtb_polarizability_method"] == "tblite GFN1-xTB finite field, ALPB(water)"
    assert dw["tblite_gfn2_solvation"] == "ALPB(water)" and -90 < dw["tblite_gfn2_gsolv_kcal"] < -50   # acetate
    for key in ("tblite_gfn2_polarizability_iso_A3", "xtb_polarizability_iso_A3", "tblite_gfn2_dipole_debye",
                "xtb_dipole_debye"):
        assert dw[key] > dg[key], key           # the continuum polarizes the anion
    # the frequencies' energy is the solvated one: G(water) - G(gas) is a solvation free energy of the right size
    assert -90 < dw["vib_gibbs_kcal"] - dg["vib_gibbs_kcal"] < -50
    assert wat.vibrations["n_imaginary"] == 0
    # the structure keys of the computed species and of the input
    assert row["inchikey"] == row["input_inchikey"] == "QTBSBXVTEAMEQO-UHFFFAOYSA-M"
    assert row["canonical_smiles"] == "CC(=O)[O-]"


def test_a_step_without_a_solvent_model_says_so(tb, gas_xtb):
    """MMFF has no implicit solvent: the water run computes in the gas phase and says why, per step."""
    res = _run("water", method="mmff", smiles="CCO")
    assert res.solvent == "" and res.solvation_model == ""
    assert res.phase_steps["refine"] == "gas (MMFF94s (force field) has no implicit solvent model for water)"
    assert res.phase_steps["xtb_properties"] == "gas (the refinement ran in the gas phase)"
    assert "refine: gas (MMFF94s" in res.summary_row()["phase_notes"]
