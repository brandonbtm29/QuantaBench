"""MD-2: liquid-slab surface tension. OPLS-AA typing (md.oplsaa, md.forcefield), the LAMMPS writers (md.liquid),
the σ analysis (md.surface_tension) on synthetic data, the CLI, and a LAMMPS smoke test (skipped without the
``lammps`` module). The openff/gaff2 backends run only where AmberTools is found (marked ``external``)."""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quantabench import cli
from quantabench.md import forcefield as F
from quantabench.md import liquid as L
from quantabench.md import oplsaa as O
from quantabench.md import surface_tension as S

C6F14 = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
HEXANE = "CCCCCC"
C6F6 = "Fc1c(F)c(F)c(F)c(F)c1F"


@pytest.fixture(scope="module")
def c6f14():
    return F.type_molecule(C6F14, "C6F14", "oplsaa")


@pytest.fixture(scope="module")
def hexane():
    return F.type_molecule(HEXANE, "hexane", "oplsaa")


# ------------------------------------------------------------------------------------------------- typing
def test_perfluorohexane_types_charges_and_terms(c6f14):
    tm = c6f14
    by_el = {}
    for el, t in zip(tm.elements, tm.atom_types):
        by_el.setdefault(el, set()).add(t)
    assert by_el == {"F": {"opls_965"}, "C": {"opls_961", "opls_962"}}
    assert tm.atom_types.count("opls_961") == 2 and tm.atom_types.count("opls_962") == 4     # CF3 ends, CF2 middle
    q = dict(zip(tm.atom_types, tm.charges))
    assert q == {"opls_961": 0.36, "opls_962": 0.24, "opls_965": -0.12}
    assert abs(tm.net_charge) < 1e-12 and tm.formal_charge == 0
    assert (len(tm.bonds), len(tm.angles), len(tm.dihedrals), len(tm.impropers)) == (19, 36, 45, 0)
    assert tm.mixing == "geometric" and tm.special_lj == (0.0, 0.0, 0.5) and tm.special_coul == (0.0, 0.0, 0.5)
    f, c = tm.lj_types["opls_965"], tm.lj_types["opls_962"]
    assert math.isclose(f.sigma, 2.95) and math.isclose(f.epsilon, 0.221752 / 4.184)
    assert math.isclose(c.sigma, 3.50) and math.isclose(c.epsilon, 0.276144 / 4.184)
    assert "ffnonbonded.itp line 819" in f.source and "watkins2001" in f.source
    assert abs(tm.mass - 338.04) < 0.05


def test_perfluoro_backbone_torsion_is_watkins_jorgensen(c6f14):
    """C-C-C-C of perfluoroalkyl carbons gets GROMACS' dih_perfluoroalkane_CT_CT_CT_CT, not the alkane torsion."""
    tm = c6f14
    cccc = [d for d in tm.dihedrals if all(tm.elements[i] == "C" for i in d[:4])]
    assert len(cccc) == 3 and {d[4] for d in cccc} == {"perfluoro CT-CT-CT-CT"}
    v = tm.dihedral_types["perfluoro CT-CT-CT-CT"].coeffs
    assert np.allclose(v, [6.622, 0.948, -1.388, -2.118], atol=1e-3)      # kcal/mol, OPLS V1..V4
    assert "2485" in tm.dihedral_types["perfluoro CT-CT-CT-CT"].source
    assert np.allclose(tm.dihedral_types["F-CT-CT-F"].coeffs, [-2.5, 0, 0, 0], atol=1e-9)
    assert np.allclose(tm.dihedral_types["CT-CT-CT-F"].coeffs, [0.3, 0, 0.4, 0], atol=1e-9)


def test_rb_to_opls_inverts_the_gromacs_relations():
    for rb, _line, _c in O.DIHEDRALS_RB.values():
        v1, v2, v3, v4 = (x * O.KJ_PER_KCAL for x in O.rb_to_opls(rb))
        back = (v2 + 0.5 * (v1 + v3), 0.5 * (-v1 + 3 * v3), -v2 + 4 * v4, -2 * v3, -4 * v4, 0.0)
        assert np.allclose(back, rb, atol=1e-9)
    with pytest.raises(ValueError):
        O.rb_to_opls((0, 0, 0, 0, 0, 1.0))


def test_hexane_and_hexafluorobenzene(hexane):
    tm = hexane
    assert sorted(set(tm.atom_types)) == ["opls_135", "opls_136", "opls_140"]
    assert abs(tm.net_charge) < 1e-12
    assert np.allclose(tm.dihedral_types["CT-CT-CT-CT"].coeffs, [1.3, -0.05, 0.2, 0.0], atol=1e-9)
    k, r0 = tm.bond_types["CT-CT"].coeffs
    assert math.isclose(k, 268.0, rel_tol=1e-9) and math.isclose(r0, 1.529)
    tm6 = F.type_molecule(C6F6, "C6F6", "oplsaa")
    assert sorted(set(tm6.atom_types)) == ["opls_718", "opls_719"] and abs(tm6.net_charge) < 1e-12
    assert len(tm6.impropers) == 6 and tm6.improper_types["Z-CA-X-Y"].coeffs == [pytest.approx(1.1), -1, 2]
    assert np.allclose(tm6.dihedral_types["X-CA-CA-X"].coeffs, [0, 30.334 / 4.184, 0, 0])
    nbrs = {}
    for a, b, _lab in tm6.bonds:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    for i, j, c, k, _lab in tm6.impropers:      # the aromatic carbon is the third atom, bonded to the other three
        assert tm6.elements[c] == "C" and nbrs[c] == {i, j, k}


@pytest.mark.parametrize("smiles, why", [
    ("FC(F)(F)C(F)(F)CCC", "junction"),          # semifluorinated
    ("CCO", "other than C, H or F"),
    ("Cc1ccccc1", "aromatic"),                    # toluene: alkyl on a ring
    ("FC(F)C(F)(F)F", "both H and F"),            # HFC-125
])
def test_oplsaa_refuses_what_it_cannot_type(smiles, why):
    with pytest.raises(F.ForceFieldError, match=why):
        F.type_molecule(smiles, "x", "oplsaa")


def test_more_perfluoro_and_alkane_types():
    assert set(F.type_molecule("FC(F)(F)F", "CF4").atom_types) == {"opls_964", "opls_965"}
    branched = F.type_molecule("FC(F)(F)C(F)(C(F)(F)F)C(F)(F)F", "i-C4F10")
    assert branched.atom_types.count("opls_963") == 1
    assert set(F.type_molecule("C", "methane").atom_types) == {"opls_138", "opls_140"}
    assert F.type_molecule("CC(C)(C)C", "neopentane").atom_types.count("opls_139") == 1


def test_lj_overrides_hook(hexane):
    tm = F.TypedMolecule.from_dict(json.loads(json.dumps(hexane.to_dict())))
    tm.apply_overrides({"lj": {"opls_140": {"epsilon": 0.02}}, "source": "test"})
    assert tm.lj_types["opls_140"].epsilon == 0.02 and tm.lj_types["opls_140"].sigma == 2.5
    assert tm.overrides["opls_140"]["source"] == "test" and "was" in tm.lj_types["opls_140"].source
    with pytest.raises(F.ForceFieldError, match="unknown atom type"):
        tm.apply_overrides({"lj": {"nope": {"epsilon": 1}}})


def test_typed_molecule_round_trips_through_json(c6f14):
    tm = F.TypedMolecule.from_dict(json.loads(json.dumps(c6f14.to_dict())))
    tm.check()
    assert tm.atom_types == c6f14.atom_types and tm.dihedrals == c6f14.dihedrals
    assert np.allclose(tm.positions, c6f14.positions, atol=1e-5)


# ------------------------------------------------------------------------------------------------ writers
LAMMPS_COMMANDS = {
    "units", "atom_style", "boundary", "bond_style", "angle_style", "dihedral_style", "improper_style",
    "special_bonds", "pair_style", "pair_modify", "read_data", "include", "kspace_style", "kspace_modify",
    "neighbor", "neigh_modify", "variable", "thermo_style", "thermo", "thermo_modify", "min_style", "minimize",
    "reset_timestep", "velocity", "timestep", "fix", "unfix", "run", "write_data", "print", "log", "compute",
    "uncompute", "change_box", "set", "restart", "if", "clear", "dump", "dump_modify",
}


def _check_input_syntax(text: str) -> None:
    lines = text.splitlines()
    cont = False
    for line in lines:
        s = line.split("#")[0].strip() if '"' not in line else line.strip()
        if not s:
            cont = False
            continue
        if not cont:
            assert s.split()[0] in LAMMPS_COMMANDS, line
        assert line.count('"') % 2 == 0, line
        assert "{" not in s.replace("${", "") and "}" in s or "{" not in s, line   # no unformatted Python braces
        cont = s.endswith("&")
    for v in set(re.findall(r"\$\{(\w+)\}", text)):
        assert re.search(rf"variable\s+{v}\s", text), f"${{{v}}} is used but never defined"


def _counts(data: str) -> dict:
    head = {}
    for kind in ("atoms", "bonds", "angles", "dihedrals", "impropers", "atom types", "bond types", "angle types",
                 "dihedral types", "improper types"):
        m = re.search(rf"^(\d+) {kind}$", data, re.M)
        head[kind] = int(m.group(1)) if m else 0
    return head


def _section_rows(data: str, name: str) -> int:
    m = re.search(rf"^{name}( #[^\n]*)?\n\n(.*?)(\n\n|\Z)", data, re.S | re.M)
    return len(m.group(2).strip().splitlines()) if m else 0


def test_write_liquid_slab_lattice(hexane, tmp_path):
    cfg = L.LiquidSlabConfig(lateral=26.0, slab_thickness=24.0, packing="lattice", dispersion="ewald")
    s = L.write_liquid_slab(hexane, tmp_path, cfg, density_estimate=0.66)
    data = (tmp_path / "system.data").read_text(encoding="utf-8")
    c = _counts(data)
    n = s.n_molecules
    assert n == round(26 * 26 * 24 / (hexane.mass / 0.66 / 6.02214076e23 * 1e24))
    assert c["atoms"] == 20 * n and c["bonds"] == 19 * n and c["angles"] == 36 * n and c["dihedrals"] == 45 * n
    assert c["atom types"] == 3 and c["bond types"] == 2 and c["angle types"] == 3 and c["dihedral types"] == 3
    assert c["impropers"] == 0 and "improper types" not in data
    for sec, k in (("Atoms", "atoms"), ("Bonds", "bonds"), ("Angles", "angles"), ("Dihedrals", "dihedrals")):
        assert _section_rows(data, sec) == c[k], sec
    mols = {int(line.split()[1]) for line in data.split("Atoms # full\n\n")[1].split("\n\n")[0].splitlines()}
    assert mols == set(range(1, n + 1))
    ff = (tmp_path / "ff.params").read_text(encoding="utf-8")
    assert len(re.findall(r"^pair_coeff (\d) \1 ", ff, re.M)) == 3
    assert "dihedral_coeff 3 1.3 -0.05 0.2 0" in ff
    for name in ("in.bulk", "in.slab", "in.run"):
        _check_input_syntax((tmp_path / name).read_text(encoding="utf-8"))
    bulk, slab = (tmp_path / "in.bulk").read_text(encoding="utf-8"), (tmp_path / "in.slab").read_text(encoding="utf-8")
    assert "pair_style      lj/long/coul/long long long ${rc}" in bulk and "pppm/disp" in slab
    assert "force/disp/real 0.0001 force/disp/kspace 0.002" in slab
    assert "tail yes" not in bulk + slab and "mix/disp" not in slab        # geometric mixing: no mix/disp geom
    assert "fix             squeeze all deform" in bulk                    # lattice start: compression stage
    assert "npt temp ${T} ${T} ${tdamp} iso ${P} ${P} ${pdamp}" in bulk
    assert "c_thermo_press[1] c_thermo_press[2] c_thermo_press[3] file pressure_tensor.txt" in slab
    assert "fix             ptens all ave/time ${p_every} 1 ${p_every}" in slab and "variable p_every        index 10" in slab
    assert "bin/1d z lower ${bin} units box" in slab and "change_box      all z final" in slab
    assert "special_bonds   lj 0 0 0.5 coul 0 0 0.5" in bulk
    meta = json.loads((tmp_path / "liquid_setup.json").read_text(encoding="utf-8"))
    assert meta["n_molecules"] == n and meta["packing"] == "lattice" and meta["molecule"]["forcefield"] == "oplsaa"
    assert all(t["source"] for t in meta["molecule"]["lj_types"].values())
    assert all(t["source"] for t in meta["molecule"]["dihedral_types"].values())
    readme = (tmp_path / "README.txt").read_text(encoding="utf-8")
    assert "lmp.exe -sf omp -pk omp 12 -in in.bulk" in readme and "no GPU version" in readme
    assert s.min_contact > 1.0


def test_cutoff_variant_has_tail_in_bulk_only(c6f14, tmp_path):
    cfg = L.LiquidSlabConfig(lateral=34.0, slab_thickness=20.0, packing="lattice", dispersion="cutoff16")
    L.write_liquid_slab(c6f14, tmp_path, cfg, density_estimate=1.67)
    bulk, slab = (tmp_path / "in.bulk").read_text(encoding="utf-8"), (tmp_path / "in.slab").read_text(encoding="utf-8")
    assert "lj/cut/coul/long ${rc}" in bulk and "variable rc             index 16" in bulk
    assert "pair_modify     mix geometric tail yes" in bulk
    assert "tail yes" not in slab and "kspace_style    pppm ${kacc}" in slab and "pppm/disp" not in slab
    assert "-sf gpu -pk gpu 1 -in in.slab" in (tmp_path / "README.txt").read_text(encoding="utf-8")


def test_parse_dispersion_and_box_plan(hexane):
    assert L.parse_dispersion("ewald") == ("ewald", 12.0)
    assert L.parse_dispersion("cutoff18") == ("cutoff", 18.0)
    for bad in ("cutoff", "pme", "cutoff5"):
        with pytest.raises(ValueError):
            L.parse_dispersion(bad)
    cfg = L.LiquidSlabConfig(n_molecules=100).resolved()
    n, box = L.plan_box(hexane, 0.655, cfg)
    assert n == 100 and box[:2] == (40.0, 40.0)
    assert math.isclose(box[2] * 40 * 40 * 0.655, 100 * hexane.mass / 6.02214076e23 * 1e24)
    with pytest.raises(ValueError, match="cutoff"):
        L.plan_box(hexane, 0.655, L.LiquidSlabConfig(lateral=20.0).resolved())


def test_lattice_packing_keeps_molecules_apart(c6f14):
    pos, box = L.pack_lattice(c6f14, 50, (30.0, 30.0, 40.0), 2.0, seed=1)
    assert pos.shape == (50 * 20, 3)
    assert all(b >= t for b, t in zip(box, (30.0, 30.0, 40.0)))
    assert np.all(pos > -1e-6) and np.all(pos < np.array(box) + 1e-6)
    assert L.min_contact(pos, 20, box) >= 2.0 - 1e-6


# ----------------------------------------------------------------------------------------------- analysis
def test_unit_conversion_and_sigma_series():
    assert math.isclose(S.ATM_A_TO_MN_M, 101325.0 * 1e-10 * 1e3)
    p = np.array([[0, 5.0, -5.0, 10.0], [10, 1.0, 3.0, 12.0]])      # Pzz - (Pxx+Pyy)/2 = 10 atm both rows
    assert np.allclose(S.sigma_series(p, 200.0), 100.0 * 10.0 * 0.0101325)


def _ar1(n, phi, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_block_average_and_statistical_inefficiency_on_ar1():
    phi, n = 0.9, 200_000
    x = _ar1(n, phi)
    g = S.statistical_inefficiency(x)
    assert abs(g - (1 + phi) / (1 - phi)) / 19.0 < 0.15                  # exact g = 19 for AR(1), φ = 0.9
    true_sem = math.sqrt(1 / (1 - phi ** 2) * 19.0 / n)
    ba = S.block_average(x, 5000)
    assert ba.n_blocks == 40 and 0.6 < ba.sem / true_sem < 1.5
    curve = S.blocking_curve(x)
    assert curve[0][1] < 0.4 * true_sem and 0.6 < curve[-6][1] / true_sem < 1.6   # rises to the plateau
    assert S.statistical_inefficiency(np.random.default_rng(1).normal(size=10_000)) < 1.3


def _tanh_profile(z, rl, rv, centre, z0, d, lz):
    dz = (z - centre + lz / 2) % lz - lz / 2
    return 0.5 * (rl + rv) - 0.5 * (rl - rv) * np.tanh(2 * (np.abs(dz) - z0) / d)


def test_profile_centring_and_tanh_fit():
    lz, zlo = 200.0, -50.0
    z = zlo + (np.arange(400) + 0.5) * 0.5
    rho = np.array([_tanh_profile(z, 1.6, 0.004, c, 45.0, 6.0, lz) for c in (140.0, 145.0, 148.0)])   # crosses zhi
    dist, rolled, centres = S.centred_profiles(z, rho, zlo, lz)
    assert np.allclose(centres, [140.0, 145.0, 148.0], atol=0.3)
    fit = S.fit_tanh(dist, rolled.mean(axis=0))
    assert abs(fit["rho_liquid"] - 1.6) < 1e-3 and abs(fit["z0"] - 45.0) < 0.3 and abs(fit["d"] - 6.0) < 0.3
    assert abs(S.unwrap_centres(np.array([190.0, 5.0]), 200.0)[1] - 205.0) < 1e-9


def _synthetic_run(folder: Path, sigma=12.0, lz=200.0, ns=6.0, settle_ns=1.0, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "slab_box.txt").write_text(f"QB_SLAB lx 40 ly 40 lz {lz} zlo -50 zhi {lz - 50} dt 1 p_every 10 "
                                         f"settle_steps {int(settle_ns * 1e6)} production_steps {int(ns * 1e6)} T 298.15 rc 12 bulk_lz 100\n")
    steps = np.arange(0, int((ns + settle_ns) * 1e6), 10)
    excess = sigma / (0.5 * lz * S.ATM_A_TO_MN_M)                       # atm
    noise = _ar1(len(steps), 0.95, seed) * 30.0
    pxx = rng.normal(0, 50, len(steps))
    pyy = -pxx - 2.0 * excess                                            # (Pxx + Pyy) / 2 = -excess
    pzz = noise
    np.savetxt(folder / "pressure_tensor.txt", np.column_stack([steps, pxx, pyy, pzz]), header="TimeStep Pxx Pyy Pzz")
    z = -50 + (np.arange(400) + 0.5) * 0.5
    with open(folder / "density_profile.txt", "w") as fh:
        fh.write("# Chunk-averaged data\n# Timestep Number-of-chunks Total-count\n# Chunk Coord1 Ncount density/mass\n")
        for k, st in enumerate(range(10_000, int((ns + settle_ns) * 1e6) + 1, 10_000)):
            r = _tanh_profile(z, 1.66, 0.003, 50.0 + 0.001 * k, 48.0, 5.0, lz) + rng.normal(0, 0.005, len(z))
            fh.write(f"{st} {len(z)} 9520\n" + "".join(f"  {i + 1} {zz:.2f} 1 {rr:.6f}\n" for i, (zz, rr) in enumerate(zip(z, r))))
    b = np.column_stack([np.arange(1000, 1_000_001, 1000), rng.normal(1.66, 0.003, 1000), np.full(1000, 1.5e5),
                         np.full(1000, 40.0), np.full(1000, 40.0), np.full(1000, 95.0), np.full(1000, -1e4)])
    np.savetxt(folder / "bulk_density.txt", b)
    (folder / "liquid_setup.json").write_text(json.dumps({"config": {"dispersion": "ewald", "settle_ns": settle_ns,
                                                                     "reference_sigma": 12.33, "reference_density": 1.67},
                                                          "n_molecules": 476, "molecule": {"name": "syn"}}))


def test_analyze_synthetic_slab(tmp_path):
    _synthetic_run(tmp_path, sigma=12.0)
    r = S.analyze(tmp_path)
    s = r["sigma"]
    assert s["n_blocks"] == 6 and math.isclose(s["block_ns"], 1.0) and s["production_ns"] == pytest.approx(6.0)
    assert abs(s["mean_mN_m"] - 12.0) < 4 * s["sem_mN_m"] and 0 < s["sem_mN_m"] < 1.0
    assert s["statistical_inefficiency"] > 5
    d = r["density"]
    assert abs(d["plateau_g_cm3"] - 1.66) < 0.002 and abs(d["tanh_fit"]["z0"] - 48.0) < 0.3
    assert d["vapour_g_cm3"] < 0.01 and abs(d["centre_drift_A"]) < 1.0
    assert abs(r["bulk"]["density_g_cm3"] - 1.66) < 0.002
    assert {c["check"] for c in r["checks"]} >= {"production length", "slab centre at the bulk density", "slab drift"}
    assert r["reference"]["sigma_mN_m"] == 12.33
    js, md = S.write_report(tmp_path, r)
    assert json.loads(js.read_text(encoding="utf-8"))["sigma"]["mean_mN_m"] == s["mean_mN_m"]
    assert "**σ (mN/m)**" in md.read_text(encoding="utf-8")
    r2 = S.analyze(tmp_path, skip_ns=4.0)                                    # 3 ns left: 4 shorter blocks
    assert r2["sigma"]["n_blocks"] == 4 and any("rough error bar" in n for n in r2["notes"])


def test_cli_liquid_slab_and_surface_tension(tmp_path):
    out = tmp_path / "hex"
    assert cli.main(["md", "liquid-slab", "--smiles", HEXANE, "--name", "hexane", "-o", str(out), "--lateral", "26",
                     "--thickness", "20", "--packing", "lattice", "--dispersion", "cutoff12", "--sigma-ref", "17.9"]) == 0
    assert (out / "in.slab").is_file() and json.loads((out / "liquid_setup.json").read_text(encoding="utf-8"))["config"]["reference_sigma"] == 17.9
    assert cli.main(["md", "liquid-slab", "--smiles", "CCO", "-o", str(tmp_path / "x")]) == 2      # refused by OPLS rules
    assert cli.main(["md", "liquid-slab", "--list-ff"]) == 0
    _synthetic_run(tmp_path / "syn")
    assert cli.main(["md", "surface-tension", str(tmp_path / "syn")]) == 0
    assert (tmp_path / "syn" / "surface_tension.md").is_file()
    assert cli.main(["md", "surface-tension", str(tmp_path)]) == 2           # no slab_box.txt


def test_md_modules_import_without_heavy_dependencies():
    code = ("import sys; import quantabench.md.liquid, quantabench.md.forcefield, quantabench.md.surface_tension, "
            "quantabench.cli_md; print(any(m in sys.modules for m in ('lammps', 'torch', 'openff', 'parmed')))")
    src = str(Path(L.__file__).resolve().parents[2])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": src})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------------------------------- LAMMPS and AmberTools
@pytest.mark.parametrize("dispersion", ["ewald", "cutoff12"])
def test_lammps_smoke_run(hexane, tmp_path, dispersion):
    """A few hundred steps of every stage on a tiny box: the inputs parse and run, the slab opens with whole
    molecules, and the analysis gives a finite σ (not a result: far too short)."""
    pytest.importorskip("lammps")
    cfg = L.LiquidSlabConfig(lateral=25.5, slab_thickness=18.0, packing="lattice", dispersion=dispersion,
                             thermo_every=100, dump_every=0)
    s = L.write_liquid_slab(hexane, tmp_path, cfg, density_estimate=0.66)
    L.run_liquid_slab(tmp_path, variables=dict(warmup_ps=0.1, compress_ps=0.2, bulk_ns=0.0003, settle_ns=0.0001,
                                               production_ns=0.0003, prof_every=10, prof_repeat=10, prof_freq=100))
    for name in ("bulk_final.data", "slab_final.data", "pressure_tensor.txt", "density_profile.txt", "slab_box.txt"):
        assert (tmp_path / name).is_file(), name
    box = S.read_slab_box(tmp_path / "slab_box.txt")
    assert math.isclose(box["lz"], box["bulk_lz"] + 2 * 50.0, rel_tol=1e-6)
    from quantabench.md.lammps import read_data_atoms

    d = read_data_atoms(tmp_path / "slab_final.data")
    lens = np.array([box["lx"], box["ly"], box["lz"]])
    unwrapped = d["x"] + d["image"] * lens
    for m in range(1, s.n_molecules + 1):          # every molecule whole: its atoms within a few Å of each other
        xyz = unwrapped[d["mol"] == m]
        assert np.ptp(xyz, axis=0).max() < 12.0
    r = S.analyze(tmp_path, skip_ns=0.0)
    assert math.isfinite(r["sigma"]["mean_mN_m"]) and r["performance_ns_day"]["slab"] > 0


@pytest.mark.external
@pytest.mark.parametrize("ff", ["openff", "gaff2"])
def test_crosscheck_backends(ff):
    ok, why = F.backend_available(ff)
    if not ok:
        pytest.skip(why)
    if ff == "openff" and subprocess.run([F.driver_python(), "-c", "import openff.interchange"],
                                         capture_output=True).returncode != 0:
        pytest.skip("no OpenFF in the AmberTools environment")
    tm = F.type_molecule(C6F14, "C6F14", ff)
    assert abs(tm.net_charge) < 1e-9 and tm.mixing == "arithmetic"
    assert tm.special_lj[2] == pytest.approx(0.5) and tm.special_coul[2] == pytest.approx(1 / 1.2)
    assert len(tm.bonds) == 19 and len(tm.angles) == 36 and len(tm.dihedrals) >= 45
    assert all(t.source for t in tm.lj_types.values()) and tm.versions.get("ambertools")
    q = tm.charges[[i for i, e in enumerate(tm.elements) if e == "F"]]
    assert np.all(q < 0)
