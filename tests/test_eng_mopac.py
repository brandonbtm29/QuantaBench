"""MOPAC engines (chem.eng_mopac): input generation, .aux/.out parsing (captured MOPAC 23.2.5 output), a fake
``mopac`` for the calculator/optimizer plumbing, and checks against the real program (marked ``external``)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from quantabench import settings as settings_mod
from quantabench.chem import eng_mopac as em
from quantabench.chem import engines, external

K = em.KCAL_PER_EV

# Captured from MOPAC 23.2.5: "PM7 1SCF GRADIENTS CHARGE=0 AUX(PRECISION=9) THREADS=1", ASE's water.
AUX_WATER = """\
 START OF MOPAC PROGRAM
 START OF MOPAC FILE
 ####################################
 #       Start of Input data        #
 ####################################
 MOPAC_VERSION=23.2.5
 DATE="Sun Nov  1 19:39:17 2026"
 METHOD=PM7
 TITLE="water"
 KEYWORDS=" PM7 1SCF GRADIENTS CHARGE=0 AUX(PRECISION=9) THREADS=1"
 ATOM_EL[03]=
  O  H  H
 ATOM_X:ANGSTROMS[09]=
    0.0000000000000    0.0000000000000    0.1192620000000
    0.0000000000000    0.7632390000000   -0.4770470000000
    0.0000000000000   -0.7632390000000   -0.4770470000000
 NUM_ELECTRONS=08
 EMPIRICAL_FORMULA="H2 O  =     3 atoms"
 HEAT_OF_FORMATION:KCAL/MOL=-0.576961586870593D+02
 GRADIENT_NORM:KCAL/MOL/ANGSTROM=+0.144262954787895D+02
 POINT_GROUP=C2v
 DIPOLE:DEBYE=+0.215889941624465D+01
 DIP_VEC:DEBYE[3]= -0.15815239196146D-15 -0.18950683226470D-10 -0.21588994162447D+01
 AREA:SQUARE ANGSTROMS=+0.424967862749595D+02
 VOLUME:CUBIC ANGSTROMS=+0.252469884177143D+02
 IONIZATION_POTENTIAL:EV=+0.120335024456450D+02
 ATOM_CHARGES[03]=
 -0.64260288775767 +0.32130144387613 +0.32130144388154
 GRADIENTS:KCAL/MOL/ANGSTROM[09]=
   0.0000000481986  -0.0000001308067   9.9552887628338   0.0000000238769   5.4523333355298  -4.9624905036319   0.0000000238769  -5.4523334828719  -4.9927983817822
 OVERLAP_MATRIX[021]=
 #  Lower half triangle only
   1.0000000000000   0.0000000000000   1.0000000000000   0.0000000000000   0.0000000000000   1.0000000000000   0.0000000000000   0.0000000000000   0.0000000000000   1.0000000000000
   0.1348317493382  -0.0000000000000   0.2919235539828  -0.2280761891779   1.0000000000000   0.1348317493382  -0.0000000000000  -0.2919235539828  -0.2280761891779   0.2384489883941
   1.0000000000000
 SET_OF_MOS=       1       6
 M.O.SYMMETRY_LABELS[06]=
   1a1     1b2     2a1     1b1     3a1     2b2
 EIGENVALUES[06]=
  -29.390475510068  -17.613590664172  -14.256833876835  -12.033502445645    3.970718473144    6.210427826716
 MOLECULAR_ORBITAL_OCCUPANCIES[00006]=
 2.0000000000000 2.0000000000000 2.0000000000000 2.0000000000000 0.0000000000000 0.0000000000000
 CPU_TIME:SECONDS[1]=        0.02
 END OF MOPAC FILE
 END OF MOPAC PROGRAM
"""

# Captured: methyl radical, "PM7 1SCF GRADIENTS PRECISE CHARGE=0 UHF DOUBLET ..." (.out: ALPHA SOMO -9.905,
# BETA LUMO 0.606).
AUX_CH3_UHF = """\
 HEAT_OF_FORMATION:KCAL/MOL=+0.285485883301053D+02
 NUM_ALPHA_ELECTRONS=4
 NUM_BETA_ELECTRONS=3
 ALPHA_EIGENVALUES[07]=
  -23.343535441491  -14.322618387419  -14.317450995798   -9.904604458603    6.063747509438    6.069861580868    6.307364840439
 BETA_EIGENVALUES[07]=
  -22.885417145335  -14.045916777837  -14.040784084088    0.606175946052    6.252607058646    6.258812295896    6.622037482036
 ALPHA_MOLECULAR_ORBITAL_OCCUPANCIES[00007]=
 1 1 1 1 0 0 0
 BETA_MOLECULAR_ORBITAL_OCCUPANCIES[00007]=
 1 1 1 0 0 0 0
"""

OUT_ERROR = """\
 *  MISSING VAN DER WAALS RADIUS FOR PO                                       *
 ******************************************************************************
 *                                                                            *
 *     Error and normal termination messages reported in this calculation     *
 *                                                                            *
 *  MISSING VAN DER WAALS RADIUS FOR PO                                       *
 * JOB ENDED NORMALLY                                                         *
 *                                                                            *
 ******************************************************************************
"""

OUT_POLAR = """\
 ENERGY OF "REORIENTED" SYSTEM WITHOUT FIELD:      -57.6961588431
 *****************************************************************
 CALCULATION OF STATIC FIELD QUANTITIES
 *****************************************************************

 +++++ ALPHA AT       0.00000 EV.

                           COMPONENTS OF ALPHA

                     *X           *Y           *Z
           *=X:       9.2974       0.3851       0.1121
           *=Y:       0.0000       8.6079       0.0236
           *=Z:      -0.0000       0.0000       7.5819

  ISOTROPIC AVERAGE ALPHA =       8.49573 A.U. =      1.25894 ANG.**3

 +++++ ALPHA AT       0.00000 EV.
  ISOTROPIC AVERAGE ALPHA =       8.49040 A.U. =      1.25815 ANG.**3
"""


def _fake_program(tmp_path: Path, name: str, body: str) -> Path:
    """A tiny Python script that behaves like an external program (as in test_external)."""
    script = tmp_path / f"{name}_impl.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    if sys.platform == "win32":
        exe = tmp_path / f"{name}.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        exe = tmp_path / name
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


# A fake MOPAC: dHf = 10 * sum((x - 0.1)^2) kcal/mol with its exact gradient; an optimization jumps to the
# minimum in two cycles; "Po" in the input gives MOPAC's error box and no .aux heat of formation.
FAKE_MOPAC = '''
import os, sys
from pathlib import Path
if "--version" in sys.argv:
    print("MOPAC version 23.2.5")
    raise SystemExit(0)
mop = Path(sys.argv[1])
text = mop.read_text()
rec = os.environ.get("FAKE_MOPAC_RECORD")
if rec:
    with open(rec, "a") as f:
        f.write(text + "@@@\\n")
lines = text.splitlines()
kw = lines[0].split()
atoms = [ln.split() for ln in lines[3:] if ln.strip()]
x = [float(a[i]) for a in atoms for i in (1, 3, 5)]
stem = mop.with_suffix("")
if any(a[0] == "Po" for a in atoms):
    Path(f"{stem}.out").write_text(" *     Error and normal termination messages reported in this calculation     *\\n"
                                   " *  MISSING VAN DER WAALS RADIUS FOR PO                                       *\\n"
                                   " * JOB ENDED NORMALLY                                                         *\\n"
                                   " ****************************************\\n")
    raise SystemExit(0)
def block(name, vals):
    return f" {name}[{len(vals):02d}]=\\n " + " ".join(f"{v:.10f}" for v in vals) + "\\n"
aux = []
opt = "1SCF" not in kw
if "CYCLES=1" in kw:   # cycle limit: per-cycle results only, no final SCF (as MOPAC does)
    g = [20 * (v - 0.1) for v in x]
    Path(f"{stem}.aux").write_text(" HEAT_OF_FORM_UPDATED:KCAL/MOL=7.0\\n" + block("ATOM_X_UPDATED:ANGSTROMS", x)
                                   + block("GRADIENTS_UPDATED:KCAL/MOL/ANGSTROM", g)
                                   + ' ERROR_MESSAGE="EXCESS NUMBER OF OPTIMIZATION CYCLES"\\n')
    Path(f"{stem}.out").write_text("")
    raise SystemExit(0)
if opt:
    for e in (5.0, 1.0):
        aux.append(f" HEAT_OF_FORM_UPDATED:KCAL/MOL={e}\\n")
    x = [0.1] * len(x)
e = 10 * sum((v - 0.1) ** 2 for v in x)
g = [20 * (v - 0.1) for v in x]
aux.append(" HEAT_OF_FORMATION:KCAL/MOL=" + f"{e:+.15E}".replace("E", "D") + "\\n")   # Fortran exponent
aux.append(" DIPOLE:DEBYE=+0.100000000000000D+01\\n DIP_VEC:DEBYE[3]= 1.0 0.0 0.0\\n")
aux.append(" AREA:SQUARE ANGSTROMS=42.5\\n VOLUME:CUBIC ANGSTROMS=25.25\\n IONIZATION_POTENTIAL:EV=12.0\\n")
if opt:
    aux.append(block("ATOM_X_OPT:ANGSTROMS", x))
aux.append(block("ATOM_CHARGES", [0.1 * (i + 1) for i in range(len(atoms))]))
if "GRADIENTS" in kw or opt:
    aux.append(block("GRADIENTS:KCAL/MOL/ANGSTROM", g))
aux.append(" EIGENVALUES[03]=\\n -15.0 -12.0 4.0\\n MOLECULAR_ORBITAL_OCCUPANCIES[00003]=\\n 2.0 2.0 0.0\\n")
Path(f"{stem}.aux").write_text(" START OF MOPAC FILE\\n" + "".join(aux) + " END OF MOPAC FILE\\n")
out = " FINAL HEAT OF FORMATION\\n"
if any(k.startswith("POLAR") for k in kw):
    out += " CALCULATION OF STATIC FIELD QUANTITIES\\n  ISOTROPIC AVERAGE ALPHA =   8.0 A.U. =   1.18548 ANG.**3\\n"
Path(f"{stem}.out").write_text(out)
'''


@pytest.fixture(autouse=True)
def _scratch(tmp_path):
    s = settings_mod.Settings()
    s.engines.scratch_dir = str(tmp_path / "scratch")
    settings_mod.save(s)
    external.clear_cache()
    yield
    external.clear_cache()


@pytest.fixture
def fake_mopac(tmp_path, monkeypatch):
    exe = _fake_program(tmp_path, "mopac", FAKE_MOPAC)
    record = tmp_path / "inputs.txt"
    monkeypatch.setenv("QUANTABENCH_MOPAC", str(exe))
    monkeypatch.setenv("FAKE_MOPAC_RECORD", str(record))
    return record


@pytest.fixture
def real_mopac():
    info = external.find_program("mopac", refresh=True)
    if info is None or not em.available()[0]:
        pytest.skip("MOPAC >= 22 not found (set QUANTABENCH_MOPAC)")
    return info


# ----------------------------------------------------------------------------- registration, input
def test_engines_registered():
    eng = engines.all_engines()
    for key, label, prefix in (("mopac-pm7", "PM7 (MOPAC)", "pm7"), ("mopac-pm6", "PM6-D3H4X (MOPAC)", "pm6")):
        info = eng[key]
        assert info.label == label and info.prop_prefix == prefix and info.family == "semi-empirical"
        assert info.solvation == ("cosmo",) and info.external == "mopac" and not info.gpu_capable
        assert engines.native_optimizer(key) is not None and engines.properties_function(key) is not None
    elems = eng["mopac-pm7"].elements
    assert {1, 6, 9, 16, 35, 53, 57, 71, 83} <= elems and not {58, 70, 84, 86} & elems
    assert engines.supports_solvent("mopac-pm7", "water")


def test_keywords():
    assert em.keywords("PM7") == "PM7 1SCF GRADIENTS PRECISE CHARGE=0 AUX(PRECISION=9) THREADS=1"
    kw = em.keywords("PM6-D3H4X", -1, 2, solvent="water", threads=4)
    assert kw.split()[:3] == ["PM6-D3H4X", "1SCF", "GRADIENTS"]
    assert "CHARGE=-1" in kw and "UHF DOUBLET" in kw and "EPS=78.36" in kw and "THREADS=4" in kw
    opt = em.keywords("PM7", task="opt", gnorm=0.6918, cycles=300)
    assert "GNORM=0.6918" in opt and "CYCLES=300" in opt and "1SCF" not in opt and "LET" not in opt.split()
    assert "LET" in em.keywords("PM7", task="opt", gnorm=0.005).split()
    assert "POLAR(E=(0.0),BETA=0,GAMMA=0)" in em.keywords("PM7", task="sp", polar=True).split()
    with pytest.raises(ValueError):
        em.keywords("PM7", multiplicity=10)


def test_write_input(tmp_path):
    em.write_input(tmp_path / "job.mop", [8, 1], [[0, 0, 0.1], [0, 0.76, -0.48]], "PM7 1SCF", "water\nwet")
    lines = (tmp_path / "job.mop").read_text().splitlines()
    assert lines[0] == "PM7 1SCF" and lines[1] == "water wet"
    assert lines[3].split() == ["O", "0.0000000000", "1", "0.0000000000", "1", "0.1000000000", "1"]
    assert lines[4].split()[0] == "H" and len(lines) == 5


def test_check_electrons():
    em.check_electrons([6, 1, 1, 1], 0, 2)
    em.check_electrons([6, 1, 1, 1], 1, 1)
    with pytest.raises(ValueError, match="multiplicity"):
        em.check_electrons([6, 1, 1, 1], 0, 1)


# ----------------------------------------------------------------------------- parsing captured output
def test_parse_aux_water():
    aux = em.parse_aux(AUX_WATER)
    assert em.last(aux, "HEAT_OF_FORMATION:KCAL/MOL") == pytest.approx(-57.6961586870593)
    assert em.last(aux, "ATOM_EL") == ["O", "H", "H"]
    assert em.last(aux, "TITLE") == "water" and em.last(aux, "POINT_GROUP") == "C2v"
    assert em.last(aux, "KEYWORDS").startswith("PM7 1SCF")
    assert em.last(aux, "AREA:SQUARE ANGSTROMS") == pytest.approx(42.4967862749595)
    assert em.last(aux, "VOLUME:CUBIC ANGSTROMS") == pytest.approx(25.2469884177143)
    assert em.last(aux, "OVERLAP_MATRIX").shape == (21,)
    assert em.last(aux, "CPU_TIME:SECONDS")[0] == pytest.approx(0.02)
    assert em.last(aux, "EMPIRICAL_FORMULA") == "H2 O  =     3 atoms"
    res = em._results(em.MopacJob(aux, "", 0.0), 3)
    assert res["energy"] == pytest.approx(-57.6961586870593 / K)
    assert res["forces"][0] == pytest.approx([0.0, 0.0, -9.9552887628338 / K], abs=1e-8)
    assert res["dipole"] == pytest.approx([0, 0, -2.1588994162447 * 0.20819434], abs=1e-8)
    assert res["charges"] == pytest.approx([-0.64260288775767, 0.32130144387613, 0.32130144388154])
    assert em.frontier_orbitals(aux) == (pytest.approx(-12.033502445645), pytest.approx(3.970718473144))


def test_frontier_orbitals_uhf():
    homo, lumo = em.frontier_orbitals(em.parse_aux(AUX_CH3_UHF))
    assert homo == pytest.approx(-9.904604458603) and lumo == pytest.approx(0.606175946052)
    assert em.frontier_orbitals({}) == (None, None)


def test_messages_and_polarizability_from_out():
    assert em.mopac_messages(OUT_ERROR) == ["MISSING VAN DER WAALS RADIUS FOR PO"]
    assert em.mopac_messages(" JOB ENDED NORMALLY\n") == []
    assert em.polarizability_from_out(OUT_POLAR) == pytest.approx(1.25894)
    assert 8.49573 * engines.BOHR_ANG**3 == pytest.approx(1.25894, abs=1e-5)     # MOPAC's a.u. -> Å³
    assert em.polarizability_from_out("no polar here") is None


# ----------------------------------------------------------------------------- availability
@pytest.mark.parametrize("version_line,ok,match", [
    ("MOPAC version 23.2.5", True, ""),
    ("MOPAC version 16.111", False, "too old"),
    ("", False, "could not read the version"),
])
def test_available_checks_version(tmp_path, monkeypatch, version_line, ok, match):
    exe = _fake_program(tmp_path, "mopac", f"print({version_line!r})\n")
    monkeypatch.setenv("QUANTABENCH_MOPAC", str(exe))
    got, why = engines.available("mopac-pm7")
    assert got is ok and match in why


def test_unavailable_hint(monkeypatch):
    monkeypatch.setattr(external, "_locate", lambda spec: None)
    ok, why = em.available()
    assert not ok and "conda install" in why


# ----------------------------------------------------------------------------- fake program
def test_calculator_with_fake_program(fake_mopac):
    atoms = Atoms("OH2", positions=[[0, 0, 0.12], [0, 0.76, -0.48], [0, -0.76, -0.48]])
    calc = engines.make_calculator("mopac-pm7", atoms.numbers, 0, 1, solvent="water", threads=3)
    atoms.calc = calc
    e = atoms.get_potential_energy()
    x = atoms.positions.ravel()
    assert e == pytest.approx(10 * ((x - 0.1) ** 2).sum() / K)
    f = atoms.get_forces()
    assert f.ravel() == pytest.approx(-20 * (x - 0.1) / K)
    b = atoms.copy()
    b.calc = calc
    b.positions[1, 1] += 1e-3
    ep = b.get_potential_energy()
    b.positions[1, 1] -= 2e-3
    assert -(ep - b.get_potential_energy()) / 2e-3 == pytest.approx(f[1, 1], abs=1e-6)   # energy/force consistent
    assert atoms.get_dipole_moment() == pytest.approx([0.20819434, 0, 0])
    assert atoms.get_charges() == pytest.approx([0.1, 0.2, 0.3])
    first = fake_mopac.read_text().split("@@@")[0].splitlines()
    assert first[0] == "PM7 1SCF GRADIENTS PRECISE CHARGE=0 EPS=78.36 AUX(PRECISION=9) THREADS=3"
    assert calc.n_jobs == 4      # start, +h, -h, and back to the start for the dipole (ASE caches one point)


def test_native_optimize_with_fake_program(fake_mopac):
    atoms = Atoms("OH2", positions=[[0, 0, 0.12], [0, 0.76, -0.48], [0, -0.76, -0.48]])
    atoms.calc = engines.make_calculator("mopac-pm6", atoms.numbers, 0, 1)
    native = engines.native_optimizer("mopac-pm6")
    res = native(atoms, 0.03, 123, lambda: False, lambda m: None, charge=0, multiplicity=1, solvent="", threads=1)
    assert res.converged and res.steps == 2 and res.fmax == pytest.approx(0.0) and res.energy_ev == pytest.approx(0.0)
    assert atoms.positions == pytest.approx(np.full((3, 3), 0.1))
    n_inputs = fake_mopac.read_text().count("@@@")
    assert atoms.get_potential_energy() == pytest.approx(0.0) and atoms.get_charges() == pytest.approx([0.1, 0.2, 0.3])
    assert fake_mopac.read_text().count("@@@") == n_inputs           # the final point was cached, not rerun
    kw = fake_mopac.read_text().split("@@@")[0].splitlines()[0]
    assert kw.startswith("PM6-D3H4X GNORM=0.6918 CYCLES=123 PRECISE CHARGE=0")


def test_native_optimize_cycle_limit_is_not_an_error(fake_mopac):
    atoms = Atoms("OH2", positions=[[0, 0, 0.12], [0, 0.76, -0.48], [0, -0.76, -0.48]])
    start = atoms.positions.copy()
    logs: list[str] = []
    res = engines.native_optimizer("mopac-pm7")(atoms, 0.03, 1, None, logs.append)
    assert not res.converged and res.steps == 1 and res.energy_ev == pytest.approx(7.0 / K)
    assert res.fmax == pytest.approx(np.sqrt(((20 * (start - 0.1) / K) ** 2).sum(1)).max())
    assert any("EXCESS NUMBER OF OPTIMIZATION CYCLES" in m for m in logs)
    with pytest.raises(external.ExternalProgramError, match="EXCESS NUMBER"):   # not accepted for a single point
        em.run_job(atoms.numbers, atoms.positions, em.keywords("PM7", task="opt", cycles=1))


def test_properties_with_fake_program(fake_mopac):
    props = engines.properties_function("mopac-pm7")([8, 1, 1], [[0, 0, 0.1]] * 3, 0, 1, solvent="", threads=1)
    assert props["pm7_polarizability_iso_A3"] == pytest.approx(1.18548)
    assert props["pm7_dipole_debye"] == 1.0 and props["pm7_ionization_ev"] == 12.0
    assert props["pm7_homo_ev"] == -12.0 and props["pm7_lumo_ev"] == 4.0 and props["pm7_gap_ev"] == 16.0
    assert props["pm7_cosmo_area_A2"] == 42.5 and props["pm7_cosmo_volume_A3"] == 25.25
    assert props["charges"] == pytest.approx([0.1, 0.2, 0.3]) and "_notes" not in props
    inputs = fake_mopac.read_text().split("@@@")
    assert "POLAR(E=(0.0),BETA=0,GAMMA=0)" in inputs[0] and "1SCF" in inputs[0] and "GRADIENTS" not in inputs[0]
    solv = engines.properties_function("mopac-pm7")([8, 1, 1], [[0, 0, 0.1]] * 3, 0, 1, solvent="water")
    assert solv["pm7_polarizability_iso_A3"] == pytest.approx(1.18548)
    assert any("gas phase" in n for n in solv["_notes"])
    rad = engines.properties_function("mopac-pm7")([6, 1, 1, 1], [[0, 0, 0.1]] * 4, 0, 2)
    assert "pm7_polarizability_iso_A3" not in rad and any("UHF" in n for n in rad["_notes"])


def test_failure_reports_mopac_message(fake_mopac, tmp_path):
    with pytest.raises(external.ExternalProgramError, match="MISSING VAN DER WAALS RADIUS FOR PO"):
        em.run_job([84, 1, 1], [[0, 0, 0], [0, 0, 1.7], [1.7, 0, 0]], em.keywords("PM7"))
    assert any((tmp_path / "scratch").glob("mopac_*/job.mop"))       # job files kept for inspection


# ----------------------------------------------------------------------------- real MOPAC
@pytest.mark.external
def test_real_water_energy_forces_and_properties(real_mopac):
    atoms = molecule("H2O")
    calc = em.MopacCalculator("PM7")
    atoms.calc = calc
    assert atoms.get_potential_energy() * K == pytest.approx(-57.69616, abs=1e-3)   # .out: -57.69616 KCAL/MOL
    f = atoms.get_forces()
    # PRECISE gradient; MOPAC's default SCF prints 9.955 (its gradients carry ~0.02 kcal/mol/Å of noise)
    assert f[0, 2] * K == pytest.approx(-9.9447, abs=2e-3)
    b = atoms.copy()
    b.calc = calc
    b.positions[1, 1] += 1e-3
    ep = b.get_potential_energy()
    b.positions[1, 1] -= 2e-3
    assert -(ep - b.get_potential_energy()) / 2e-3 == pytest.approx(f[1, 1], abs=5e-4)
    props = engines.properties_function("mopac-pm7")(atoms.numbers, atoms.positions, 0, 1, solvent="", threads=1)
    # values printed in MOPAC's .out for this geometry
    assert props["pm7_heat_of_formation_kcal"] == pytest.approx(-57.69616, abs=1e-3)
    assert props["pm7_dipole_debye"] == pytest.approx(2.159, abs=1e-3)
    assert props["pm7_ionization_ev"] == pytest.approx(12.033502, abs=1e-4)
    assert props["pm7_homo_ev"] == pytest.approx(-12.034, abs=1e-3)
    assert props["pm7_lumo_ev"] == pytest.approx(3.971, abs=1e-3)
    assert props["pm7_cosmo_area_A2"] == pytest.approx(42.50, abs=0.01)
    assert props["pm7_cosmo_volume_A3"] == pytest.approx(25.25, abs=0.01)
    assert props["pm7_polarizability_iso_A3"] == pytest.approx(1.25894, abs=1e-3)
    assert props["charges"][0] == pytest.approx(-0.642603, abs=1e-4)
    cosmo = engines.properties_function("mopac-pm7")(atoms.numbers, atoms.positions, 0, 1, solvent="water")
    assert cosmo["pm7_heat_of_formation_kcal"] == pytest.approx(-65.2076, abs=2e-3)
    assert cosmo["pm7_dipole_debye"] == pytest.approx(2.4515, abs=2e-3)
    assert cosmo["pm7_polarizability_iso_A3"] == pytest.approx(1.25894, abs=1e-3)


@pytest.mark.external
def test_real_native_optimize_pm6(real_mopac):
    atoms = molecule("CH3OH")
    atoms.rattle(0.05, seed=4)
    atoms.calc = engines.make_calculator("mopac-pm6", atoms.numbers, 0, 1, threads=1)
    res = engines.native_optimizer("mopac-pm6")(atoms, 0.03, 200, lambda: False, lambda m: None, charge=0,
                                               multiplicity=1, solvent="", threads=1)
    assert res.converged and res.fmax <= 0.03 and res.steps > 1
    assert res.energy_ev * K == pytest.approx(-43.51, abs=0.02)
    n = atoms.calc.n_jobs
    assert np.abs(atoms.get_forces()).max() <= 0.03 and atoms.calc.n_jobs == n
    b = molecule("CH3OH")
    b.rattle(0.1, seed=1)
    logs: list[str] = []
    short = engines.native_optimizer("mopac-pm7")(b, 0.03, 3, None, logs.append)
    assert not short.converged and short.steps == 3 and short.fmax > 0.03
    assert any("EXCESS NUMBER OF OPTIMIZATION CYCLES" in m for m in logs)


@pytest.mark.external
def test_real_doublet_and_error_message(real_mopac):
    ch3 = Atoms("CH3", positions=[[0, 0, 0], [1.08, 0, 0], [-0.54, 0.935307, 0.05], [-0.54, -0.935307, 0]])
    props = engines.properties_function("mopac-pm7")(ch3.numbers, ch3.positions, 0, 2)
    assert props["pm7_heat_of_formation_kcal"] == pytest.approx(28.5486, abs=2e-3)
    assert props["pm7_homo_ev"] == pytest.approx(-9.905, abs=1e-3) and props["pm7_lumo_ev"] == pytest.approx(0.606, abs=1e-3)
    assert any("UHF" in n for n in props["_notes"])
    with pytest.raises(external.ExternalProgramError, match="VAN DER WAALS RADIUS FOR PO"):
        em.run_job([84, 1, 1], [[0, 0, 0], [0, 0, 1.7], [1.7, 0, 0]], em.keywords("PM7"))


@pytest.mark.external
def test_real_pipeline_native_cosmo(real_mopac):
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    cfg = BuildConfig(method="mopac-pm7", optimizer="native", solvent="water", max_conformers=2, refine_top=1,
                      write_files=False, xtb_properties=False, threads=1, frequencies=True)
    res = run_molecule(from_smiles("CO", id="methanol"), cfg)
    assert res.status == "ok", res.error
    assert res.solvent == "water" and res.conformers[0].converged
    d = res.descriptors
    assert d["pm7_heat_of_formation_kcal"] < -50 and d["pm7_polarizability_iso_A3"] == pytest.approx(3.0, abs=0.2)
    assert res.conformers[0].charges is not None
    vib = res.vibrations
    assert vib and vib["n_imaginary"] == 0 and len(vib["frequencies_cm1"]) == 12, res.notes
