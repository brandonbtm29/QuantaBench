"""Importing other programs' output files through cclib (chem.io_cclib) and the CLI ``import`` command.

Fixtures in tests/data (real program output, water):

- psi4_water_opt_freq.out: Psi4 1.9.1 (conda-forge), wB97X-D/def2-SVP optimization + finite-difference
  frequencies. The finite-difference gradient sub-runs were cut (a note marks the cut) to keep it at 64 KB;
  the optimization, the frequency analysis and the thermochemistry are complete. The input (echoed at the top)
  carries a two-line pint workaround needed by that Psi4 environment.
- xtb_water/xtb.out + g98.out: xtb 6.7.1 ``xtb water.xyz --ohess`` (GFN2-xTB), with the g98.out xtb writes.
- mopac_water_force.out: MOPAC v23.2.5, PM7 optimization then FORCE (OLDGEO) in one input.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cclib")

from quantabench.chem import export, io_cclib  # noqa: E402
from quantabench.chem.engines import HARTREE_EV  # noqa: E402

DATA = Path(__file__).parent / "data"
PSI4 = DATA / "psi4_water_opt_freq.out"
XTB = DATA / "xtb_water" / "xtb.out"
MOPAC = DATA / "mopac_water_force.out"


@pytest.fixture(scope="module")
def psi4():
    return io_cclib.read_output(PSI4)


def test_fixtures_are_small_text():
    for p in (PSI4, XTB, XTB.with_name("g98.out"), MOPAC):
        assert p.stat().st_size < 100_000, p
        p.read_text(encoding="utf-8")


def test_psi4_output(psi4):
    c = psi4
    assert c.program == "Psi4" and c.version == "1.9.1"
    assert (c.method, c.basis, c.label) == ("wB97X-D", "def2-SVP", "psi4: wB97X-D/def2-SVP")
    assert c.numbers == [8, 1, 1] and c.charge == 0 and c.multiplicity == 1
    assert c.trajectory_A.shape == (3, 3, 3) and c.optimized is True and c.success is True
    assert np.allclose(c.positions_A, c.trajectory_A[-1])
    # "@DF-RKS Final Energy: -76.33703097666631" of the optimized geometry, in QuantaBench's eV
    assert c.energy_ev == pytest.approx(-76.33703097666631 * HARTREE_EV, abs=2e-6)
    assert len(c.energies_ev) == 3 and c.energies_ev[-1] == c.energy_ev
    assert c.frequencies_cm1 == pytest.approx([1644.6666, 3893.7441, 3995.4450])
    assert c.ir_intensities_km_mol == pytest.approx([61.0563, 10.6330, 42.3592])
    assert c.displacements.shape == (3, 3, 3)
    assert c.reduced_masses_amu is not None and len(c.reduced_masses_amu) == 3
    assert c.homo_ev < 0 < c.lumo_ev and c.dipole_debye == pytest.approx(2.03, abs=0.01)
    assert c.zpve_ev == pytest.approx(0.02171972 * HARTREE_EV, rel=1e-6)    # Psi4's "Vibrational ZPVE"
    assert any("Git: Rev" in s for s in c.compat) and any("thermochemistry" in s for s in c.compat)
    assert c.warnings == []


def test_psi4_compat_shims_only_touch_their_lines():
    compat = []
    text = "  Git: Rev {} zzzzzzz \n  ==> Thermochemistry Energy Analysis <==\n    Vibrational ZPVE  1 [kcal/mol]\n"
    out = io_cclib._psi4_compat(text, compat)
    assert "Git: Rev zzzzzzz" in out and "(ZPVE layout)" in out and len(compat) == 2
    compat = []
    old = "  Git: Rev {master} 8b8d4f2 \n  ==> Thermochemistry Energy Analysis <==\n    Vibrational ZPE 1\n"
    assert io_cclib._psi4_compat(old, compat) == old and compat == []


def test_vibrations_dict_matches_quantabench_shape_and_psi4_thermo(psi4):
    from quantabench.chem.vibrations import VibrationalResult

    v = io_cclib.to_vibrations_dict(psi4)
    ref = VibrationalResult("x", "finite", 0, 1, "cpu", 3, "nonlinear", [1.0] * 3, [1.0] * 3, [1.0] * 3, None, "",
                            0, None, displacements=np.zeros((3, 3, 3))).to_dict(include_modes=True)
    assert set(ref) <= set(v)
    assert {"symbols", "positions_A"} <= set(v)
    assert v["method"] == "psi4: wB97X-D/def2-SVP" and v["hessian_method"] == "imported"
    assert v["n_modes"] == 3 and v["n_imaginary"] == 0 and v["lowest_freq_cm1"] == pytest.approx(1644.6666)
    d = np.asarray(v["displacements"])
    assert d.shape == (3, 3, 3) and np.allclose((d**2).sum((1, 2)), 1.0, atol=1e-4)    # unit Cartesian norm
    assert v["symbols"] == ["O", "H", "H"] and np.allclose(v["positions_A"], psi4.positions_A, atol=1e-6)
    t = v["thermo"]
    # QuantaBench's RRHO/qRRHO on Psi4's frequencies vs Psi4's own thermochemistry printout (298.15 K, 1 atm)
    assert t["symmetry_number"] == 2
    assert t["zpe_kcal"] == pytest.approx(13.629, abs=2e-3)
    assert t["h_corr_kcal"] == pytest.approx(16.001, abs=2e-3)
    assert t["g_corr_kcal"] == pytest.approx(2.556, abs=2e-3)
    assert t["entropy_cal_molK"] == pytest.approx(45.096, abs=5e-3)
    assert v["program_zpve_kcal"] == pytest.approx(t["zpe_kcal"], abs=1e-3)


def test_export_write_vibrations_accepts_imported_dict(psi4, tmp_path):
    paths = export.write_vibrations(io_cclib.to_vibrations_dict(psi4), tmp_path, "w")
    assert {p.name for p in paths} == {"w_ir.csv", "w_modes.csv", "w_freq.xyz"}
    rows = (tmp_path / "w_modes.csv").read_text().splitlines()
    assert len(rows) == 4 and rows[1].startswith("1,1644.67,1644.67,61.056")


def test_to_mol_and_to_atoms(psi4):
    from rdkit import Chem

    mol = io_cclib.to_mol(psi4)
    assert mol.GetNumAtoms() == 3 and mol.GetNumBonds() == 2 and mol.GetProp("qb_bonds") == "rdDetermineBonds"
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == "O"
    atoms = io_cclib.to_atoms(psi4)
    assert atoms.get_chemical_formula() == "H2O" and np.allclose(atoms.positions, psi4.positions_A)


def test_to_mol_falls_back_to_connectivity():
    calc = io_cclib.read_output(PSI4)
    calc.charge = 3                                  # no Lewis structure of H2O with charge +3
    mol = io_cclib.to_mol(calc)
    assert mol.GetNumBonds() == 2 and mol.GetProp("qb_bonds") == "connectivity only"
    assert any("connectivity only" in w for w in calc.warnings)


def test_to_result(psi4):
    res = io_cclib.to_result(psi4, PSI4)
    assert res.status == "ok" and res.method == "psi4: wB97X-D/def2-SVP"
    assert res.input.id == "psi4_water_opt_freq" and res.smiles == "O" and res.charge == 0
    assert len(res.conformers) == 1 and res.mol.GetNumConformers() == 1
    c = res.conformers[0]
    assert c.energy_ev == psi4.energy_ev and c.converged and c.dipole_debye == psi4.dipole_debye
    assert c.charges is None or len(c.charges) == 3
    assert res.notes[0].startswith("imported from psi4_water_opt_freq.out")
    assert res.vibrations["n_modes"] == 3 and res.descriptors["formula"] == "H2O"
    assert res.descriptors["vib_zpe_kcal"] == pytest.approx(13.629, abs=2e-3)
    assert res.to_dict()["method"] == res.method          # the summary JSON can be written


def test_xtb_output_with_g98_modes():
    c = io_cclib.read_output(XTB)
    assert c.program == "xTB" and c.version == "6.7.1" and c.label == "xtb: GFN2-xTB" and c.basis == ""
    assert c.frequencies_cm1 == pytest.approx([1539.10, 3642.25, 3650.69])       # the six zero modes are gone
    assert c.ir_intensities_km_mol == pytest.approx([133.26, 6.78, 16.68])
    assert c.displacements.shape == (3, 3, 3)
    assert any("g98.out" in s for s in c.compat) and any("translation/rotation" in s for s in c.compat)
    assert c.energy_ev == pytest.approx(-5.070544374391 * HARTREE_EV, abs=1e-5)  # "TOTAL ENERGY" of xtb
    assert c.homo_ev == pytest.approx(-12.1461, abs=1e-3) and c.lumo_ev == pytest.approx(2.2385, abs=1e-3)
    assert "mulliken" in c.charges and len(c.charges["mulliken"]) == 3
    assert c.zpve_ev * 23.060548 == pytest.approx(12.63, abs=0.01)             # eV in cclib's xtb parser
    assert any("multiplicity" in w for w in c.warnings) and c.multiplicity == 1
    v = io_cclib.to_vibrations_dict(c)
    assert v["thermo"]["zpe_kcal"] == pytest.approx(v["program_zpve_kcal"], abs=0.01)
    assert all(k > 0 for k in v["force_constants_mdyn_A"])                    # g98.out prints zeros: k = μω²


def test_xtb_without_g98_has_no_modes(tmp_path):
    shutil.copy(XTB, tmp_path / "xtb.out")
    c = io_cclib.read_output(tmp_path / "xtb.out")
    assert c.displacements is None and any("g98.out" in w for w in c.warnings)
    v = io_cclib.to_vibrations_dict(c)
    assert "displacements" not in v and v["reduced_masses_amu"] == pytest.approx([2.15, 1.55, 2.14])


def test_mopac_v23_output(tmp_path):
    c = io_cclib.read_output(MOPAC)
    assert c.program == "MOPAC" and c.method == "PM7" and c.label == "mopac: PM7"
    assert any("MOPAC 22+" in s for s in c.compat)
    assert c.frequencies_cm1 == pytest.approx([1394.41, 2809.63, 2856.36])
    assert c.ir_intensities_km_mol is None and c.displacements is None
    assert c.energy_kind == "heat of formation"
    assert c.energy_ev * 23.060548 == pytest.approx(-57.79986, abs=1e-3)       # FINAL HEAT OF FORMATION, kcal/mol
    v = io_cclib.to_vibrations_dict(c)
    assert v["ir_intensities_km_mol"] is None and np.isnan(v["reduced_masses_amu"]).all()
    assert any("heat of formation" in w for w in v["warnings"])
    paths = export.write_vibrations(v, tmp_path, "m")               # uniform sticks, "nan" in the mode table
    assert len(paths) == 3 and "nan" in (tmp_path / "m_modes.csv").read_text()


def test_unreadable_files(tmp_path):
    p = tmp_path / "notes.out"
    p.write_text("just some text\nnothing to parse\n")
    with pytest.raises(ValueError, match="does not recognize"):
        io_cclib.read_output(p)
    with pytest.raises(FileNotFoundError):
        io_cclib.read_output(tmp_path / "missing.out")


def test_cli_import(tmp_path, capsys):
    from quantabench import cli

    out = tmp_path / "out"
    bad = tmp_path / "bad.out"
    bad.write_text("nothing\n")
    rc = cli.main(["import", str(PSI4), str(XTB), str(bad), "--out", str(out)])
    text = capsys.readouterr()
    assert rc == 1 and "bad.out: FAILED" in text.err
    assert "wB97X-D/def2-SVP" in text.out and "strongest IR bands" in text.out
    for name in ("psi4_water_opt_freq.xyz", "psi4_water_opt_freq_conformers.sdf", "psi4_water_opt_freq_ir.csv",
                 "psi4_water_opt_freq_modes.csv", "psi4_water_opt_freq_summary.json", "xtb_ir.csv",
                 "summary.csv"):
        assert (out / name).is_file(), name
    rc = cli.main(["import", str(MOPAC), "--out", str(out), "--formats", "xyz"])
    assert rc == 0 and (out / "mopac_water_force.xyz").is_file()
    assert not (out / "mopac_water_force_ir.csv").exists()
