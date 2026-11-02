"""The xtb program as engines (chem.eng_xtb): parsers on captured xtb 6.7.1 output, command lines, the ASE
calculator, the native optimizer and the properties function against a fake xtb, plus a few real-xtb
checks (marked ``external``; skipped when xtb >= 6.5 is not installed)."""

from __future__ import annotations

import json
import sys
import textwrap
import threading
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from quantabench import settings as settings_mod
from quantabench.chem import eng_xtb, engines, external
from quantabench.chem.optimize import Cancelled

HARTREE_EV = engines.HARTREE_EV
BOHR = engines.BOHR_ANG


@pytest.fixture(autouse=True)
def _scratch(isolated_settings, tmp_path):
    """Scratch folders under tmp_path, no cached program lookups between tests."""
    s = settings_mod.Settings()
    s.engines.scratch_dir = str(tmp_path / "scratch")
    settings_mod.save(s)
    external.clear_cache()
    yield tmp_path / "scratch"
    external.clear_cache()


def _fake_program(folder: Path, name: str, body: str) -> Path:
    """A tiny Python script that behaves like an external program (works on Windows and Linux)."""
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / f"{name}_impl.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    if sys.platform == "win32":
        exe = folder / f"{name}.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        exe = folder / name
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


# A fake xtb: E = -5 + k/2 Σ x² (bohr), gradient k·x, charges q = (-0.6, +0.3, +0.3, ...) + charge/N,
# ALPB lowers E by 0.01 Eh. --opt scales the geometry by 0.99 and "converges" after 4 cycles.
FAKE_XTB = r'''
import json, os, sys, time
from pathlib import Path
BOHR = 0.529177210903
args = sys.argv[1:]
if "--version" in args:
    print("      * xtb version " + os.environ.get("FAKE_XTB_VERSION", "6.7.1") + " (fake) compiled by test")
    sys.exit(0)
log = os.environ.get("FAKE_XTB_LOG")
if log:
    with open(log, "a") as f:
        f.write(json.dumps({"argv": args, "threads": os.environ.get("OMP_NUM_THREADS"),
                            "restart": Path("xtbrestart").exists(), "cwd": os.getcwd()}) + "\n")
if os.environ.get("FAKE_XTB_SLEEP"):
    time.sleep(float(os.environ["FAKE_XTB_SLEEP"]))
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
inp = args[0]
lines = Path(inp).read_text().splitlines()
n = int(lines[0])
sym = [ln.split()[0] for ln in lines[2:2 + n]]
pos = [[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + n]]
if os.environ.get("FAKE_XTB_FAIL") == "scf":
    print("   iter      E             dE")
    print("########################################################################")
    print("[ERROR] Program stopped due to fatal error")
    print("-1- scf: Self consistent charge iterator did not converge")
    Path(".sccnotconverged").touch()
    sys.exit(1)
charge = int(opt("--chrg", "0"))
k = 0.02
steps, converged = 0, True
if "--opt" in args:
    pos = [[0.99 * v for v in p] for p in pos]
    cycles = int(opt("--cycles", "200"))
    steps, converged = min(4, cycles), cycles >= 4
x = [[v / BOHR for v in p] for p in pos]
energy = -5.0 + 0.5 * k * sum(v * v for p in x for v in p) - (0.01 if "--alpb" in args else 0.0)
grad = [[k * v for v in p] for p in x]
gnorm = sum(v * v for p in grad for v in p) ** 0.5
q = [-0.6] + [0.6 / (n - 1)] * (n - 1) if n > 1 else [0.0]
q = [qi + charge / n for qi in q]
Path("xtbrestart").write_text("restart")
stem = Path(inp).stem
eg = ["#", "# Number of atoms", "#", f"{n:10d}", "#", "# The current total energy in Eh", "#", f"{energy:20.12f}",
      "#", "# The current gradient in Eh/bohr", "#"] + [f"{v:20.12f}" for p in grad for v in p] + \
     ["#", "# The atomic numbers and current coordinates in Bohr", "#"] + \
     [f"   1 {p[0]:12.7f} {p[1]:12.7f} {p[2]:12.7f}" for p in x]
if "--grad" in args:
    Path(stem + ".engrad").write_text("\n".join(eg) + "\n")
gfnff = "--gfnff" in args
Path("gfnff_charges" if gfnff else "charges").write_text("\n".join(f"{v:14.8f}" for v in q) + "\n")
dip = [sum(qi * p[i] for qi, p in zip(q, x)) for i in range(3)]
if "--json" in args and not gfnff:
    Path("xtbout.json").write_text(json.dumps({"total energy": round(energy, 8), "HOMO-LUMO gap / eV": 10.5,
                                               "dipole / a.u.": dip, "partial charges": q}))
if "--opt" in args:
    out = [str(n), f" energy: {energy:.12f} gnorm: {gnorm:.12f} xtb: 6.7.1 (fake)"]
    out += [f"{s} {p[0]:.14f} {p[1]:.14f} {p[2]:.14f}" for s, p in zip(sym, pos)]
    Path("xtbopt.xyz").write_text("\n".join(out) + "\n")
    Path(".xtboptok").touch()
    for c in range(1, steps + 1):
        print(f"........................... CYCLE {c:4d} ...........................")
    if converged:
        print(f"   *** GEOMETRY OPTIMIZATION CONVERGED AFTER {steps} ITERATIONS ***")
    else:
        print(f"   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN {steps} ITERATIONS ***")
print(f"         :: total energy        {energy:20.12f} Eh    ::")
print(f"         :: gradient norm       {gnorm:20.12f} Eh/a0 ::")
if not gfnff:
    print("         :: HOMO-LUMO gap             10.500000000000 eV    ::")
if "--alpb" in args:
    print("         :: -> Gsolv                  -0.012000000000 Eh    ::")
if opt("--gfn") == "2":
    print(" Mol. α(0) /au        :          9.428750")
print("normal termination of xtb")
'''


@pytest.fixture
def fake_xtb(tmp_path, monkeypatch):
    exe = _fake_program(tmp_path / "bin", "xtb", FAKE_XTB)
    log = tmp_path / "xtb_calls.jsonl"
    monkeypatch.setenv("QUANTABENCH_XTB", str(exe))
    monkeypatch.setenv("FAKE_XTB_LOG", str(log))

    def calls():
        return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []

    return calls


WATER = Atoms("OH2", positions=[[0, 0, 0.117790], [0, 0.755453, -0.471161], [0, -0.755453, -0.471161]])


# ----------------------------------------------------------------------------- registry
def test_engines_registered_with_contract_fields():
    all_e = engines.all_engines()
    for key, prefix, family in (("xtb-gfn2", "gfn2", "tight binding"), ("xtb-gfn1", "gfn1", "tight binding"),
                                ("gfnff", "gfnff", "force field")):
        info = all_e[key]
        assert info.prop_prefix == prefix and info.family == family and info.external == "xtb"
        assert info.solvation == ("alpb", "gbsa") and not info.gpu_capable
        assert info.elements == frozenset(range(1, 87))
        assert engines.properties_function(key) is not None and engines.native_optimizer(key) is not None
        assert key in engines.method_keys() and engines.plugin(key).extra["dipole_origin"] == "com"
    assert all_e["xtb-gfn2"].label == "GFN2-xTB (xtb program)" and all_e["gfnff"].label == "GFN-FF (xtb program)"
    assert engines.supports_solvent("xtb-gfn2", "water") and engines.supports_solvent("gfnff", "dichloromethane")


def test_import_starts_no_program(tmp_path):
    src = Path(eng_xtb.__file__).parents[2]
    # dependencies first (numpy.testing, pulled in by scipy/ase, runs a subprocess when imported)
    code = ("import subprocess, numpy.testing, ase.data, ase.calculators.calculator\n"
            "import quantabench.chem.engines, quantabench.chem.external\n"
            "def boom(*a, **k):\n    raise AssertionError('subprocess at import time')\n"
            "subprocess.Popen = subprocess.run = boom\n"
            "import quantabench.chem.eng_xtb as m\nprint('ok', sorted(m.METHODS))\n")
    env = external.program_env(1, {"PYTHONPATH": str(src)})
    run = external.run_program([sys.executable, "-c", code], tmp_path / "imp", env=env)
    assert "ok ['gfnff', 'xtb-gfn1', 'xtb-gfn2']" in run.text()


def test_available_missing_old_and_new(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "_locate", lambda spec: None)
    ok, why = eng_xtb.available()
    assert not ok and "conda install" in why
    monkeypatch.undo()
    exe = _fake_program(tmp_path / "old", "xtb", FAKE_XTB)
    monkeypatch.setenv("QUANTABENCH_XTB", str(exe))
    monkeypatch.setenv("FAKE_XTB_VERSION", "6.4.1")
    external.clear_cache()
    ok, why = eng_xtb.available()
    assert not ok and "6.4.1" in why and ">= 6.5" in why and "6.7.1 recommended" in why
    monkeypatch.setenv("FAKE_XTB_VERSION", "6.7.1")
    external.clear_cache()
    assert eng_xtb.available() == (True, "")
    assert engines.available("xtb-gfn2") == (True, "")


def test_opt_level_guarantees_fmax():
    assert eng_xtb.opt_level(0.03) == "vtight"      # tight's 8e-4 Eh/bohr is 0.041 eV/Å
    assert eng_xtb.opt_level(0.05) == "tight"
    assert eng_xtb.opt_level(0.1) == "normal"
    assert eng_xtb.opt_level(0.2) == "lax"          # loose is 0.206 eV/Å
    assert eng_xtb.opt_level(1.0) == "crude"
    assert eng_xtb.opt_level(0.001) == "extreme"
    for name, _de, g in eng_xtb.OPT_LEVELS:      # each level is chosen as soon as its threshold suffices
        assert eng_xtb.opt_level(g * eng_xtb.EV_PER_EH_BOHR * 1.0001) == name


def test_command_lines_are_relative_and_complete():
    wsl = external.ProgramInfo("xtb", "wsl:xtb", "settings", "6.7.1")
    cmd = eng_xtb.command("xtb-gfn2", charge=-1, multiplicity=2, solvent="dichloromethane", program=wsl)
    assert cmd == ["wsl.exe", "-e", "xtb", "calc.xyz", "--gfn", "2", "--grad", "--chrg", "-1", "--uhf", "1",
                   "--alpb", "ch2cl2", "--json"]
    native = external.ProgramInfo("xtb", "/opt/xtb/bin/xtb", "PATH", "6.7.1")
    cmd = eng_xtb.command("gfnff", solvent="water", model="gbsa", opt_level_name="vtight", cycles=50,
                          accuracy=0.1, program=native)
    assert cmd == ["/opt/xtb/bin/xtb", "calc.xyz", "--gfnff", "--opt", "vtight", "--cycles", "50", "--grad",
                   "--chrg", "0", "--uhf", "0", "--gbsa", "water"]
    cmd = eng_xtb.command("xtb-gfn1", grad=False, accuracy=0.1, program=native)
    assert cmd[2:4] == ["--gfn", "1"] and "--grad" not in cmd and cmd[-3:] == ["--acc", "0.1", "--json"]
    with pytest.raises(ValueError):
        eng_xtb.solvent_args("unobtainium")


# ----------------------------------------------------------------------------- parsers (captured xtb 6.7.1)
ENGRAD_WATER_GFN2 = """#
# Number of atoms
#
         3
#
# The current total energy in Eh
#
     -5.070325128562
#
# The current gradient in Eh/bohr
#
       0.000000000000
      -0.000000000000
       0.003637705841
       0.000000000000
      -0.004782488097
      -0.001818852920
      -0.000000000000
       0.004782488097
      -0.001818852920
#
# The atomic numbers and current coordinates in Bohr
#
   8     0.0000000    0.0000000    0.2225908
   1     0.0000000    1.4275993   -0.8903653
   1     0.0000000   -1.4275993   -0.8903653
"""

TURBOMOLE_GRADIENT = """$grad
  cycle =      1    SCF energy =    -5.07032512856   |dE/dxyz| =  0.008099
    0.00000000000000      0.00000000000000      0.22259084021904      O
    0.00000000000000      1.42759927002289     -0.89036525060229      H
    0.00000000000000     -1.42759927002289     -0.89036525060229      H
   2.4396499117129E-20  -6.9716362619870E-17   3.6377058406327E-03
   9.3529552451851E-21  -4.7824880967663E-03  -1.8188529203164E-03
  -3.3749454362314E-20   4.7824880967664E-03  -1.8188529203163E-03
$end
"""

# GFN2-xTB, water in ALPB water: summary and property printout of xtb 6.7.1
LOG_ALPB = """
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         ::                     SUMMARY                     ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         :: total energy              -5.084788707376 Eh    ::
         :: total w/o Gsasa/hb        -5.073167740152 Eh    ::
         :: gradient norm              0.009577743000 Eh/a0 ::
         :: HOMO-LUMO gap             15.157425814181 eV    ::
         ::.................................................::
         :: SCC energy                -5.119690478605 Eh    ::
         :: -> isotropic ES            0.049102970800 Eh    ::
         :: -> Gsolv                  -0.019208209254 Eh    ::
         ::    -> Gelec               -0.007587242030 Eh    ::
         ::    -> Gsasa                0.001955239293 Eh    ::
         :: repulsion energy           0.034901771229 Eh    ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
 Mol. C6AA /au·bohr⁶  :         44.866089
 Mol. C8AA /au·bohr⁸  :        800.219440
 Mol. α(0) /au        :          9.532856
"""

LOG_OPT_OK = """
.............................. CYCLE    5 ..............................
   gradient norm :     0.0003735 Eh/α   predicted    -0.2557718E-05 ( -21.09%)
.............................. CYCLE    6 ..............................
   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 6 ITERATIONS ***
         :: total energy              -8.226118342198 Eh    ::
"""
LOG_OPT_FAILED = """
.............................. CYCLE    1 ..............................
.............................. CYCLE    2 ..............................
   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 2 ITERATIONS ***
"""
LOG_SCF_FAILED = """
[ERROR] Program stopped due to fatal error
-3- Single point calculation terminated
-2- xtb_calculator_singlepoint: Electronic structure method terminated
-1- scf: Self consistent charge iterator did not converge
"""


def test_parse_engrad_and_turbomole_gradient():
    e, g, coords = eng_xtb.parse_engrad(ENGRAD_WATER_GFN2)
    assert e == -5.070325128562 and g.shape == (3, 3)
    assert g[0, 2] == pytest.approx(0.003637705841) and g[1, 1] == pytest.approx(-0.004782488097)
    assert coords[1, 1] == pytest.approx(1.4275993)
    e2, g2 = eng_xtb.parse_turbomole_gradient(TURBOMOLE_GRADIENT, 3)
    assert e2 == pytest.approx(e, abs=1e-10) and np.allclose(g2, g, atol=1e-11)


def test_parse_log_summary_property_and_opt_lines():
    info = eng_xtb.parse_log(LOG_ALPB)
    assert info["energy_eh"] == -5.084788707376 and info["gsolv_eh"] == -0.019208209254
    assert info["gap_ev"] == 15.157425814181 and info["alpha_au"] == 9.532856
    assert info["alpha_au"] * eng_xtb.AU_POLARIZABILITY_A3 == pytest.approx(1.412623, abs=1e-6)
    assert not info["scf_failed"] and not info["opt_converged"]
    ok = eng_xtb.parse_log(LOG_OPT_OK)
    assert ok["opt_converged"] and ok["opt_steps"] == 6 and ok["energy_eh"] == -8.226118342198
    bad = eng_xtb.parse_log(LOG_OPT_FAILED)
    assert not bad["opt_converged"] and bad["opt_steps"] == 2
    assert eng_xtb.parse_log(LOG_SCF_FAILED)["scf_failed"]


# CF3COO- (GFN2): xtb's json dipole at the original coordinates and after shifting them by +10 Å in x
TFA_ANION = """7

O      1.381886   -1.282415    0.160828
C      0.939841   -0.096904    0.002738
O      1.637337    0.957751   -0.151509
C     -0.615806    0.070862   -0.002254
F     -1.162386   -0.386633    1.159252
F     -1.000613    1.369603   -0.144944
F     -1.180260   -0.632264   -1.024110
"""


def test_ion_dipole_is_shifted_to_the_centre_of_mass():
    (symbols, pos, _), = eng_xtb.read_xyz_frames(TFA_ANION)
    numbers = Atoms(symbols).numbers
    mu0 = [-1.81925300, 0.21305064, -0.00702682]        # xtb 6.7.1, as printed in xtbout.json
    mu10 = [-20.71651419, 0.21305064, -0.00702682]      # same molecule, x + 10 Å: xtb says 52.7 D
    a = eng_xtb.dipole_about_com(numbers, pos, -1, mu0)
    b = eng_xtb.dipole_about_com(numbers, pos + [10.0, 0, 0], -1, mu10)
    assert np.allclose(a, b, atol=1e-6)
    assert np.linalg.norm(a) * eng_xtb.AU_DEBYE == pytest.approx(5.1393, abs=1e-3)
    neutral = eng_xtb.dipole_about_com(numbers, pos, 0, mu0)
    assert np.allclose(neutral, mu0)


def test_read_xyz_frames_multi():
    text = "2\n energy: -1.5 gnorm: 0.001\nH 0 0 0\nH 0 0 0.74\n2\n-1.4\nh 0 0 0\nh 0 0 0.8\n"
    frames = eng_xtb.read_xyz_frames(text)
    assert [f[0] for f in frames] == [["H", "H"], ["H", "H"]] and frames[1][1][1, 2] == 0.8
    with pytest.raises(ValueError):
        eng_xtb.read_xyz_frames("3\n\nH 0 0 0\n")


# ----------------------------------------------------------------------------- fake xtb
def _expected(positions_A, alpb=False):
    x = np.asarray(positions_A) / BOHR
    e = -5.0 + 0.5 * 0.02 * (x ** 2).sum() - (0.01 if alpb else 0.0)
    return e, 0.02 * x


def test_calculator_units_restart_and_cleanup(fake_xtb, _scratch):
    calc = engines.make_calculator("xtb-gfn2", WATER.numbers, charge=0, multiplicity=1, solvent="water", threads=2)
    assert isinstance(calc, eng_xtb.XtbCalculator)
    a = WATER.copy()
    a.calc = calc
    e_eh, g = _expected(a.positions, alpb=True)
    assert a.get_potential_energy() == pytest.approx(e_eh * HARTREE_EV, abs=1e-8)
    assert np.allclose(a.get_forces(), -g * HARTREE_EV / BOHR, atol=1e-8)
    q = np.array([-0.6, 0.3, 0.3])
    assert np.allclose(calc.results["charges"], q)
    x = a.positions / BOHR
    m = np.array([15.999, 1.008, 1.008])
    com = (x * m[:, None]).sum(0) / m.sum()
    assert np.allclose(calc.results["dipole"], ((x - com) * q[:, None]).sum(0) * BOHR, atol=1e-6)  # e·Å
    folder = calc.directory_path
    assert folder.parent == _scratch and (folder / "xtbrestart").is_file()
    p = a.positions.copy()
    p[0, 2] += 0.01
    a.positions = p
    a.get_potential_energy()
    calls = fake_xtb()
    assert [c["restart"] for c in calls] == [False, True]          # the second run starts from xtbrestart
    assert calls[0]["threads"] == "2" and Path(calls[0]["cwd"]).resolve() == folder.resolve()
    assert calls[0]["argv"] == ["calc.xyz", "--gfn", "2", "--grad", "--chrg", "0", "--uhf", "0", "--acc", "0.1",
                                "--alpb", "water", "--json"]
    calc.close()
    assert not folder.exists()


def test_calculator_gfnff_reads_its_charges(fake_xtb):
    calc = engines.make_calculator("gfnff", WATER.numbers, threads=1)
    a = WATER.copy()
    a.calc = calc
    a.get_potential_energy()
    assert np.allclose(calc.results["charges"], [-0.6, 0.3, 0.3]) and "dipole" in calc.results
    assert "--json" not in fake_xtb()[0]["argv"] and "--gfnff" in fake_xtb()[0]["argv"]
    calc.close()


def test_calculator_scf_failure_raises_with_log_tail(fake_xtb, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_FAIL", "scf")
    a = WATER.copy()
    a.calc = engines.make_calculator("xtb-gfn2", a.numbers, threads=1)
    with pytest.raises(external.ExternalProgramError, match="SCF did not converge") as err:
        a.get_potential_energy()
    assert "Self consistent charge iterator did not converge" in str(err.value)
    a.calc.close()


def test_native_optimize_uses_calculator_folder_and_sets_results(fake_xtb):
    a = WATER.copy()
    start = a.positions.copy()
    calc = engines.make_calculator("xtb-gfn2", a.numbers, threads=1)
    a.calc = calc
    native = engines.native_optimizer("xtb-gfn2")
    res = native(a, 0.03, 77, lambda: False, None, charge=0, multiplicity=1, solvent="", threads=1)
    argv = fake_xtb()[-1]["argv"]
    assert argv[argv.index("--opt") + 1] == "vtight" and argv[argv.index("--cycles") + 1] == "77"
    assert np.allclose(a.positions, 0.99 * start, atol=1e-10)
    e_eh, g = _expected(a.positions)
    assert res.energy_ev == pytest.approx(e_eh * HARTREE_EV, abs=1e-8) and res.steps == 4
    assert res.fmax == pytest.approx(np.linalg.norm(g, axis=1).max() * eng_xtb.EV_PER_EH_BOHR, rel=1e-6)
    assert res.converged == (res.fmax <= 0.03)
    n_calls = len(fake_xtb())
    assert np.allclose(a.get_forces(), -g * eng_xtb.EV_PER_EH_BOHR, atol=1e-8)   # cached: no new run
    assert len(fake_xtb()) == n_calls and calc.results["charges"] is not None
    calc.close()


def test_native_optimize_not_converged_and_without_calculator(fake_xtb, _scratch):
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.1]])       # tiny forces: fmax is always met
    native = engines.native_optimizer("gfnff")
    res = native(a, 0.5, 2, lambda: False, None, charge=0, multiplicity=1, solvent="water", threads=1)
    assert not res.converged and res.steps == 2                 # xtb ran out of cycles
    res = native(a, 0.5, 10, lambda: False, None, charge=0, multiplicity=1, solvent="water", threads=1)
    assert res.converged and res.steps == 4
    assert "--alpb" in fake_xtb()[-1]["argv"] and "--opt" in fake_xtb()[-1]["argv"]
    assert not any(_scratch.iterdir())                           # its own job folders were removed


def test_native_optimize_cancel_kills_xtb(fake_xtb, monkeypatch):
    monkeypatch.setenv("FAKE_XTB_SLEEP", "30")
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    a = WATER.copy()
    with pytest.raises(Cancelled):
        engines.native_optimizer("xtb-gfn2")(a, 0.03, 50, flag.is_set, None, charge=0, multiplicity=1, threads=1)


def test_properties_gas_reference_and_keys(fake_xtb):
    props = engines.properties_function("xtb-gfn2")(WATER.numbers, WATER.positions, 0, 1, solvent="water", threads=1)
    e_solv, _ = _expected(WATER.positions, alpb=True)
    assert props["gfn2_energy_ev"] == pytest.approx(e_solv * HARTREE_EV, abs=1e-6)
    assert props["gfn2_gsolv_kcal"] == pytest.approx(-0.01 * eng_xtb.KCAL_PER_HARTREE, abs=1e-4)
    assert props["gfn2_gsolv_term_kcal"] == pytest.approx(-0.012 * eng_xtb.KCAL_PER_HARTREE, abs=1e-6)
    assert props["gfn2_gap_ev"] == 10.5
    assert props["gfn2_d4_polarizability_A3"] == pytest.approx(9.42875 * 0.1481847)
    assert props["gfn2_dipole_debye"] > 0 and props["charges"] == pytest.approx([-0.6, 0.3, 0.3])
    runs = [c["argv"] for c in fake_xtb()]
    assert len(runs) == 2 and "--alpb" in runs[0] and "--alpb" not in runs[1]
    assert "--grad" not in runs[0]
    p1 = engines.properties_function("xtb-gfn1")(WATER.numbers, WATER.positions, 0, 1, threads=1)
    assert "gfn1_d4_polarizability_A3" not in p1 and "gfn1_gsolv_kcal" not in p1 and p1["_notes"] == []
    pff = engines.properties_function("gfnff")(WATER.numbers, WATER.positions, 0, 1, threads=1)
    assert "gfnff_gap_ev" not in pff and pff["gfnff_dipole_debye"] > 0 and pff["_notes"]


def test_pipeline_with_fake_xtb_native_optimizer(fake_xtb):
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    cfg = BuildConfig(method="xtb-gfn2", solvent="water", optimizer="native", max_conformers=3, refine_top=2,
                      write_files=False, xtb_properties=False, threads=1)
    res = run_molecule(from_smiles("CO", id="methanol"), cfg)
    assert res.status == "ok", res.error
    assert res.method == "xtb-gfn2" and res.solvent == "water" and res.device == "cpu"
    assert res.conformers[0].charges is not None and res.conformers[0].steps == 4
    assert "gfn2_gsolv_kcal" in res.descriptors and "gfn2_dipole_debye" in res.descriptors
    assert any("--opt" in c["argv"] for c in fake_xtb())


# ----------------------------------------------------------------------------- the real program
def _real_xtb():
    ok, why = eng_xtb.available()
    if not ok:
        pytest.skip(f"xtb >= 6.5 not available: {why}")
    return external.find_program("xtb")


@pytest.mark.external
def test_real_singlepoints_match_xtb_printout():
    _real_xtb()
    # total energies xtb 6.7.1 prints for this geometry (default accuracy)
    ref = {"xtb-gfn2": -5.070325128562, "xtb-gfn1": -5.768602223514, "gfnff": -0.327252506734}
    for key, e_ref in ref.items():
        calc = eng_xtb.XtbCalculator(WATER.numbers, key, threads=1, accuracy=1.0)
        a = WATER.copy()
        a.calc = calc
        assert a.get_potential_energy() / HARTREE_EV == pytest.approx(e_ref, abs=1e-9)
        f = a.get_forces()
        h = 1e-3                                  # central difference of one force component
        e_pm = []
        for sign in (1, -1):
            p = WATER.positions.copy()
            p[1, 1] += sign * h
            a.positions = p
            e_pm.append(a.get_potential_energy())
        assert -(e_pm[0] - e_pm[1]) / (2 * h) == pytest.approx(f[1, 1], abs=2e-4)
        calc.close()
    calc = eng_xtb.XtbCalculator(WATER.numbers, "xtb-gfn2", threads=1)
    a = WATER.copy()
    a.calc = calc
    a.get_potential_energy()
    assert calc.results["dipole"][2] == pytest.approx(-0.89636557 * BOHR, abs=1e-4)   # xtbout.json, e·Å
    assert calc.results["charges"] == pytest.approx([-0.56331856, 0.28165928, 0.28165928], abs=1e-4)
    calc.close()


@pytest.mark.external
def test_real_properties_water_in_water():
    _real_xtb()
    p = eng_xtb.properties(WATER.numbers, WATER.positions, 0, 1, method="xtb-gfn2", solvent="water", threads=1)
    # xtb 6.7.1: E(ALPB) -5.084788707376, E(gas) -5.070325128562, "-> Gsolv" -0.019208209254 Eh,
    # gap 15.157425814181 eV, Mol. α(0) 9.532856 au
    assert p["gfn2_energy_ev"] / HARTREE_EV == pytest.approx(-5.084788707376, abs=1e-9)
    assert p["gfn2_gsolv_kcal"] == pytest.approx(-9.0760, abs=1e-3)
    assert p["gfn2_gsolv_term_kcal"] == pytest.approx(-12.0533, abs=1e-3)
    assert p["gfn2_gap_ev"] == pytest.approx(15.1574, abs=1e-3)
    assert p["gfn2_d4_polarizability_A3"] == pytest.approx(1.4126, abs=1e-3)
    assert p["gfn2_dipole_debye"] == pytest.approx(2.5456, abs=1e-3)


@pytest.mark.external
def test_real_native_optimize_methanol():
    _real_xtb()
    a = Atoms("COH4", positions=[[-0.397027, -0.011019, -0.002816], [0.885059, -0.428016, -0.396370],
                                 [-0.393616, -0.119606, 1.106740], [-1.118120, -0.663864, -0.510319],
                                 [-0.483571, 1.051486, -0.304893], [1.507274, 0.171019, 0.107658]])
    a.calc = eng_xtb.XtbCalculator(a.numbers, "xtb-gfn2", threads=1)
    e0 = a.get_potential_energy()
    res = eng_xtb.native_optimize(a, 0.03, 100, None, None, method="xtb-gfn2", threads=1)
    assert res.converged and res.fmax <= 0.03 and 0 < res.steps < 40
    assert res.energy_ev < e0 and res.energy_ev / HARTREE_EV == pytest.approx(-8.2261183, abs=2e-6)
    assert np.linalg.norm(a.get_forces(), axis=1).max() == pytest.approx(res.fmax, abs=2e-3)
    a.calc.close()


@pytest.mark.external
def test_real_native_optimize_anion_dipole_matches_a_fresh_single_point():
    """xtb reports an ion's dipole about its own origin; after --opt that frame can differ from xtbopt.xyz."""
    _real_xtb()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles("[O-]C(=O)C(F)(F)F"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    a = Atoms([x.GetAtomicNum() for x in m.GetAtoms()], positions=m.GetConformer().GetPositions() + 4.0)
    a.calc = eng_xtb.XtbCalculator(a.numbers, "xtb-gfn2", charge=-1, threads=1)
    eng_xtb.native_optimize(a, 0.03, 200, None, None, method="xtb-gfn2", charge=-1, threads=1)
    mu_opt = np.asarray(a.calc.results["dipole"])
    b = a.copy()
    b.calc = eng_xtb.XtbCalculator(b.numbers, "xtb-gfn2", charge=-1, threads=1)
    b.get_potential_energy()
    assert np.linalg.norm(mu_opt) == pytest.approx(np.linalg.norm(b.calc.results["dipole"]), rel=1e-3)
    a.calc.close()
    b.calc.close()
