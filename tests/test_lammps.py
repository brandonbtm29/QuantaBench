
"""LAMMPS steered-MD bridge: parameters, data/input writers, a tiny real run, and the log readers."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from quantabench.build import surfaces
from quantabench.chem import export, pipeline
from quantabench.chem.inputs import from_smiles
from quantabench.md import lammps as qlmp
from quantabench.md import lammps_log as qlog
from quantabench.md import params as P

TFA = "OC(=O)C(F)(F)F"   # deprotonated to trifluoroacetate at pH 7 by the pipeline


@pytest.fixture(scope="module")
def tfa():
    """Optimized TFA anion with partial charges: GFN1-xTB if dxtb runs here, else Gasteiger (RDKit)."""
    from quantabench.chem import properties

    use_xtb = properties.available()
    cfg = pipeline.BuildConfig(method="mmff", xtb_properties=use_xtb, max_conformers=3, refine_top=1,
                               write_files=False, gpu="off")
    res = pipeline.run_molecule(from_smiles(TFA, id="TFA"), cfg)
    assert res.status == "ok", res.error
    assert res.charge == -1
    best = res.conformers[0]
    atoms = export.mol_to_atoms(res.mol, best.conf_id)
    charges = best.charges
    if charges is None:
        from rdkit.Chem import AllChem

        AllChem.ComputeGasteigerCharges(res.mol)
        charges = [a.GetDoubleProp("_GasteigerCharge") for a in res.mol.GetAtoms()]
    atoms.set_initial_charges(charges)
    assert abs(sum(charges) + 1) < 1e-3
    return atoms


@pytest.fixture(scope="module")
def tfa_on_graphite(tfa):
    slab = surfaces.graphite_slab(6, 4, layers=2, vacuum=20)
    return surfaces.place_on_surface(slab, tfa, height=3.5)


SHORT = dict(soft_steps=100, equil_steps=200, smd_steps=300)
SHORT_CFG = qlmp.SMDConfig(thermo_every=50, dump_every=100, pull_output_every=20)


# ------------------------------------------------------------------------------------------- parameters


def test_uff_spot_checks_against_rappe_1992():
    # Table 1 of Rappé et al. 1992: x_i (Å), D_i (kcal/mol)
    assert P.UFF_LJ["F"] == (3.364, 0.050)
    assert P.UFF_LJ["C"] == (3.851, 0.105)
    assert P.UFF_LJ["H"] == (2.886, 0.044)
    assert P.UFF_LJ["O"] == (3.500, 0.060)
    assert P.UFF_LJ["S"] == (4.035, 0.274)
    for el in ("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"):
        assert el in P.UFF_LJ
    f = P.uff_lj("F")
    assert f.epsilon == 0.050
    assert math.isclose(f.sigma, 3.364 / 2 ** (1 / 6))
    assert math.isclose(f.r_min, 3.364)          # LJ minimum sits at UFF's x_i
    with pytest.raises(KeyError):
        P.uff_lj("Xx")


def test_uff_table_matches_rdkit():
    """Independent check: RDKit's UFF implementation carries the same x_i, D_i."""
    from rdkit import Chem
    from rdkit.Chem import rdForceFieldHelpers

    for el, smi in [("C", "CC"), ("N", "NN"), ("O", "OO"), ("F", "FF"), ("Cl", "ClCl"), ("Br", "BrBr"),
                    ("I", "II"), ("S", "SS"), ("P", "PP"), ("Si", "[SiH3][SiH3]")]:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        x, d = rdForceFieldHelpers.GetUFFVdWParams(mol, 0, 1)
        assert (round(x, 3), round(d, 3)) == P.UFF_LJ[el], el


def test_graphite_presets_mixing_and_overrides():
    amber, steele = P.graphite_lj("amber"), P.graphite_lj("steele")
    assert (amber.epsilon, amber.sigma) == (0.0860, 3.400)
    assert math.isclose(steele.epsilon, 28.0 * 0.0019872, rel_tol=2e-3)   # 28 K in kcal/mol
    m = P.mix_geometric(amber, P.uff_lj("F"))
    assert math.isclose(m.epsilon, math.sqrt(0.086 * 0.05))
    labels = {"C_surf": "C", "C": "C", "F": "F"}
    out = P.resolve_type_params(labels, {"C_surf"}, overrides={"C": (0.07, 3.55), "F": (0.061, 2.94)})
    assert out["C_surf"] == amber.__class__(0.086, 3.4, amber.source)   # element override skips the surface
    assert (out["C"].epsilon, out["F"].sigma) == (0.07, 2.94)
    out = P.resolve_type_params(labels, {"C_surf"}, overrides={"C_surf": (0.0556, 3.4)})
    assert out["C_surf"].epsilon == 0.0556 and out["C"].epsilon == 0.105


# ------------------------------------------------------------------------------------------------ writers


def _data_header(text: str) -> dict:
    lines = text.splitlines()
    return {"atoms": int(next(ln for ln in lines if ln.endswith(" atoms")).split()[0]),
            "types": int(next(ln for ln in lines if ln.endswith(" atom types")).split()[0]),
            "bonds": any("bonds" in ln and ln.split()[0].isdigit() for ln in lines)}


def test_write_setup_files(tfa_on_graphite, tmp_path):
    setup = qlmp.write_smd_setup(tfa_on_graphite, tmp_path / "smd", temperature=310)
    for p in setup.paths:
        assert p.exists()
    assert setup.type_labels == ["C_surf", "O", "C", "F"]          # surface carbon is its own type
    assert setup.n_surface == 6 * 4 * 4 * 2
    assert len(setup.molecules) == 1 and len(setup.molecules[0]) == 7
    assert math.isclose(setup.net_charge, -1.0, abs_tol=1e-3)

    data = qlmp.read_data_atoms(setup.data)
    hdr = _data_header(setup.data.read_text())
    assert hdr == {"atoms": len(tfa_on_graphite), "types": 4, "bonds": False}
    assert set(data["mol"][: setup.n_surface]) == {1} and set(data["mol"][setup.n_surface:]) == {2}
    assert np.all(data["q"][: setup.n_surface] == 0)
    assert np.allclose(data["q"][setup.n_surface:], tfa_on_graphite.get_initial_charges()[setup.n_surface:],
                       atol=1e-6)
    assert np.allclose(data["unwrapped"], tfa_on_graphite.positions, atol=1e-6)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = setup.box
    assert math.isclose(xhi, tfa_on_graphite.cell[0, 0]) and math.isclose(yhi, tfa_on_graphite.cell[1, 1])
    assert zhi >= tfa_on_graphite.positions[:, 2].max() + setup.config.pull_distance  # room for the pull
    assert zlo < 0

    inp = setup.input.read_text()
    assert "boundary        p p f" in inp and "atom_style      full" in inp
    assert "pair_style      lj/cut/coul/long 12" in inp and "kspace_style    pppm 0.0001" in inp
    assert "kspace_modify   slab 3" in inp and "pair_modify     mix geometric" in inp
    assert "neigh_modify    exclude molecule/intra adsorbate" in inp
    assert "neigh_modify    exclude molecule/intra surface" in inp
    assert "pair_coeff      1 1 0.08600 3.40000" in inp
    assert f"pair_coeff      4 4 0.05000 {3.364 / 2 ** (1 / 6):.5f}" in inp
    assert "rigid/small molecule langevin" in inp
    assert "smd cvel ${kspring} ${vpull} tether NULL NULL" in inp
    assert "variable T           index 310 " in inp
    assert "variable smd_steps   index 150000" in inp            # 15 Å / (1e-4 Å/fs x 1 fs)
    assert not any(ln.split()[2] == "surface" for ln in inp.splitlines() if ln.startswith("fix "))
    assert "package gpu" not in inp and "suffix gpu" not in inp
    readme = setup.readme.read_text()
    assert "lmp.exe -sf gpu -pk gpu 1 neigh no -in in.smd" in readme and "lmp -in in.smd" in readme
    assert "Rappe" in readme and "rigid" in readme and "frozen" in readme


def test_gpu_flags_and_uncharged_system(tfa_on_graphite, tmp_path):
    setup = qlmp.write_smd_setup(tfa_on_graphite, tmp_path / "gpu", gpu=True)
    inp = setup.input.read_text()
    assert "package gpu 1 neigh no" in inp and "suffix gpu" in inp
    assert inp.index("package gpu") < inp.index("read_data")        # must precede the box
    assert "lmp.exe -in in.smd" in setup.readme.read_text()

    neutral = qlmp.write_smd_setup(tfa_on_graphite, tmp_path / "lj", charges=np.zeros(len(tfa_on_graphite)))
    inp = neutral.input.read_text()
    assert "pair_style      lj/cut 12" in inp and "kspace" not in inp.replace("KSpace", "")


def test_overlap_and_bad_inputs_are_rejected(tfa, tmp_path):
    slab = surfaces.graphite_slab(6, 4, layers=2, vacuum=20)
    close = surfaces.place_on_surface(slab, tfa, height=0.8)
    with pytest.raises(ValueError, match="contact"):
        qlmp.write_smd_setup(close, tmp_path / "x")
    with pytest.raises(ValueError, match="surface"):
        qlmp.write_smd_setup(slab, tmp_path / "y")                     # no adsorbate
    ok = surfaces.place_on_surface(slab, tfa, height=3.5)
    with pytest.raises(ValueError, match="charges"):
        qlmp.write_smd_setup(ok, tmp_path / "z", charges=[0.0, 1.0])


def test_counter_ion_becomes_point_particle(tfa, tmp_path):
    slab = surfaces.graphite_slab(6, 4, layers=2, vacuum=20)
    system = surfaces.place_on_surface(slab, tfa, height=3.5)
    system += Atoms("Na", positions=[[2.0, 2.0, slab.positions[:, 2].max() + 3.0]], charges=[1.0], tags=[2])
    setup = qlmp.write_smd_setup(system, tmp_path / "ion", **SHORT, config=SHORT_CFG)
    assert [len(m) for m in setup.molecules] == [7, 1]
    assert abs(setup.net_charge) < 1e-3
    inp = setup.input.read_text()
    assert "group           ions molecule 3" in inp and "group           pulled molecule 2" in inp
    pytest.importorskip("lammps")
    log = qlog.parse_log(qlmp.run_lammps(setup.out_dir))
    assert log.ok and not log.lost_atoms and len(log.blocks) == 3


def test_counter_ion_uses_ion_parameters_and_stays_stable(tfa, tmp_path):
    """Na+ next to a carboxylate in vacuum: with UFF's neutral-atom LJ the ion collapsed onto oxygen
    and the run blew up on Windows for some seeds. Ion LJ + nve/limit must survive every seed."""
    assert P.ion_lj("Na").sigma == pytest.approx(2.439, abs=1e-3)
    slab = surfaces.graphite_slab(6, 4, layers=2, vacuum=20)
    system = surfaces.place_on_surface(slab, tfa, height=3.5)
    system += Atoms("Na", positions=[[2.0, 2.0, slab.positions[:, 2].max() + 3.0]], charges=[1.0], tags=[2])
    base = qlmp.write_smd_setup(system, tmp_path / "base", **SHORT, config=SHORT_CFG)
    assert "Na_ion" in base.data.read_text() and "nve/limit 0.1" in base.input.read_text()
    pytest.importorskip("lammps")
    import re
    import shutil

    for seed in (1001, 1004, 1005, 4928459):      # seeds that failed on Windows with UFF Na
        d = tmp_path / f"s{seed}"
        shutil.copytree(base.out_dir, d)
        inp = d / "in.smd"
        inp.write_text(re.sub(r"(variable seed\s+index\s+)\d+", rf"\g<1>{seed}", inp.read_text()))
        log = qlog.parse_log(qlmp.run_lammps(d))
        assert log.ok and not log.lost_atoms, f"seed {seed}"


def test_molecule_across_periodic_boundary_stays_whole(tfa, tmp_path):
    slab = surfaces.graphite_slab(6, 4, layers=2, vacuum=20)
    system = surfaces.place_on_surface(slab, tfa, height=3.5, xy=(0.2, 0.2))   # straddles x = 0 and y = 0
    setup = qlmp.write_smd_setup(system, tmp_path / "pbc", **SHORT, config=SHORT_CFG)
    data = qlmp.read_data_atoms(setup.data)
    ns = setup.n_surface
    assert (data["image"][ns:, :2] != 0).any()                     # some atoms carry image flags
    assert ((data["x"][:, 0] >= 0) & (data["x"][:, 0] < setup.box[0][1])).all()
    d_in = np.linalg.norm(system.positions[ns:, None] - system.positions[None, ns:], axis=-1)
    d_data = np.linalg.norm(data["unwrapped"][ns:, None] - data["unwrapped"][None, ns:], axis=-1)
    assert np.allclose(d_in, d_data, atol=1e-6)
    pytest.importorskip("lammps")
    log = qlog.parse_log(qlmp.run_lammps(setup.out_dir))
    assert log.ok and not log.lost_atoms


# ----------------------------------------------------------------------------------------------- real run


def test_smd_run_with_lammps_module(tfa_on_graphite, tmp_path):
    pytest.importorskip("lammps")
    setup = qlmp.write_smd_setup(tfa_on_graphite, tmp_path / "run", **SHORT, config=SHORT_CFG)
    log_path = qlmp.run_lammps(setup.out_dir)
    text = log_path.read_text()
    assert "ERROR" not in text and "Lost atoms" not in text

    log = qlog.parse_log(log_path)
    assert log.ok and not log.lost_atoms and log.version
    assert [len(b) for b in log.blocks] == [100 // 50 + 1, 200 // 50 + 1, 300 // 50 + 1]
    assert "f_pull[7]" in log.blocks[2].columns and np.all(np.isfinite(log.blocks[2]["PotEng"]))
    assert len(log.concatenated()["Step"]) == 1 + (100 + 200 + 300) // 50

    before = qlmp.read_data_atoms(setup.data)
    after = qlmp.read_data_atoms(setup.out_dir / "final.data")
    ns = setup.n_surface
    # frozen slab: not a single coordinate changed
    assert np.allclose(after["unwrapped"][:ns], before["unwrapped"][:ns], atol=1e-7)
    # rigid adsorbate: all internal distances preserved
    d0 = np.linalg.norm(before["unwrapped"][ns:, None] - before["unwrapped"][None, ns:], axis=-1)
    d1 = np.linalg.norm(after["unwrapped"][ns:, None] - after["unwrapped"][None, ns:], axis=-1)
    assert np.abs(d1 - d0).max() < 1e-5
    # ... while the body itself did move (it was thermalized and pulled)
    assert np.abs(after["unwrapped"][ns:] - before["unwrapped"][ns:]).max() > 1e-3

    pull = qlog.read_pull_force(setup.out_dir / "pull_force.txt")
    assert pull.columns[:5] == ["time_fs", "com_z_A", "displacement_A", "force_z_kcal_mol_A", "work_kcal_mol"]
    assert len(pull.data) >= 300 // 20 and np.all(np.isfinite(pull.data))
    assert pull.time_fs[0] == 0 and math.isclose(pull.time_fs[-1], 300.0)
    est = qlog.estimate_detachment(pull)
    assert math.isfinite(est.work_kcal_mol) and math.isfinite(est.peak_force_kcal_mol_A)
    assert not est.plateau_reached           # 300 fs pulls the spring only 0.03 Å
    assert (setup.out_dir / "smd.lammpstrj").exists() and (setup.out_dir / "equilibrated.data").exists()


# ------------------------------------------------------------------------------------------------ readers


LOG = """LAMMPS (22 Jul 2025 - Update 4)
units real
   Step          Time           Temp          PotEng
         0   0              0             -43.1
WARNING: Neighbor exclusions used with KSpace solver may give inconsistent Coulombic energies (x)
       100   50             67.9          -43.2
Loop time of 0.03 on 1 procs for 100 steps with 199 atoms

   Step          Time           Temp          PotEng       f_pull[3]
       100   50             67.9          -43.2          0
       200   150            300.1         -44.0          1.5
ERROR: Lost atoms: original 199 current 198 (src/thermo.cpp:500)
"""


def test_parse_log_blocks_warnings_and_errors(tmp_path):
    p = tmp_path / "log.lammps"
    p.write_text(LOG)
    log = qlog.parse_log(p)
    assert log.version.startswith("22 Jul 2025")
    assert [len(b) for b in log.blocks] == [2, 2]
    assert log.blocks[0].loop_time == 0.03 and log.blocks[1].loop_time is None
    assert np.allclose(log.blocks[1]["f_pull[3]"], [0, 1.5])
    assert len(log.warnings) == 1 and log.lost_atoms and not log.ok
    joined = log.concatenated()
    assert list(joined) == ["Step", "Time", "Temp", "PotEng"] and list(joined["Step"]) == [0, 100, 200]


def test_estimate_detachment_on_synthetic_pull(tmp_path):
    t = np.arange(0, 150_001, 100.0)                       # fs
    z = 1e-4 * t                                           # Å of spring travel
    rng = np.random.default_rng(0)
    force = 4.0 * np.exp(-0.5 * ((z - 2.0) / 0.8) ** 2) + rng.normal(0, 0.3, len(t))
    work = np.concatenate([[0.0], np.cumsum(0.5 * (force[1:] + force[:-1]) * np.diff(z))])
    p = tmp_path / "pull_force.txt"
    rows = "\n".join(f"{a} {b} {c} {d} {e}" for a, b, c, d, e in zip(t, z + 7, z, force, work))
    p.write_text("# time_fs com_z_A displacement_A force_z_kcal_mol_A work_kcal_mol\n" + rows + "\n")
    est = qlog.estimate_detachment(qlog.read_pull_force(p))
    exact = 4.0 * 0.8 * math.sqrt(2 * math.pi)            # ∫ Gaussian = 8.02 kcal/mol
    assert est.plateau_reached
    assert abs(est.work_kcal_mol - exact) < 0.3
    assert abs(est.peak_force_kcal_mol_A - 4.0) < 0.3 and abs(est.peak_displacement_A - 2.0) < 0.3
    assert math.isclose(est.work_kj_mol, est.work_kcal_mol * 4.184)
    assert math.isclose(est.peak_force_pN, est.peak_force_kcal_mol_A * 69.4786)
    assert "plateau reached" in est.summary()


def test_langevin_drag_estimate():
    # 113 g/mol dragged at 1e-4 Å/fs over 15 Å with τ = 1000 fs: M v d / τ in kcal/mol
    w = qlmp.langevin_drag_work(113.0, 1e-4, 15.0, 1000.0)
    assert math.isclose(w, 113.0 * 1e-4 / 1000.0 * 48.88821291 ** 2 * 15.0)
    assert 0.3 < w < 0.5


def test_md_package_imports_without_heavy_dependencies():
    code = ("import sys; import quantabench.md.lammps, quantabench.md.lammps_log, quantabench.md.params; "
            "print(any(m in sys.modules for m in ('lammps', 'torch', 'dxtb', 'aimnet', 'ovito')))")
    src = str(Path(qlmp.__file__).resolve().parents[2])      # this checkout's src/, not an installed copy
    env = {**os.environ, "PYTHONPATH": src}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_estimate_detachment_removes_langevin_drag(tmp_path):
    """A fast pull adds a friction work F_drag * d that never plateaus; it is subtracted when known."""
    t = np.arange(0, 15_001, 10.0)                        # fs
    z = 1e-3 * t                                          # Å, 1e-3 Å/fs
    drag = qlmp.langevin_drag_work(113.0, 1e-3, 1.0, 1000.0)          # kcal/mol/Å
    rng = np.random.default_rng(0)
    bind = 4.0 * np.exp(-0.5 * ((z - 2.0) / 0.8) ** 2)
    force = bind + drag + rng.normal(0, 0.2, len(t))
    work = np.concatenate([[0.0], np.cumsum(0.5 * (force[1:] + force[:-1]) * np.diff(z))])
    p = tmp_path / "pull_force.txt"
    p.write_text("# time_fs com_z_A displacement_A force_z_kcal_mol_A work_kcal_mol\n"
                 + "\n".join(f"{a} {b} {c} {d} {e}" for a, b, c, d, e in zip(t, z + 7, z, force, work)) + "\n")
    pull = qlog.read_pull_force(p)
    raw = qlog.estimate_detachment(pull)
    exact = 4.0 * 0.8 * math.sqrt(2 * math.pi)
    assert raw.drag_work_kcal_mol is None and raw.work_kcal_mol > exact + 3.0      # inflated by ~F_drag * 15 Å
    (tmp_path / "smd_setup.json").write_text(json.dumps({"langevin_drag_work_estimate_kcal_mol": drag * 15.0,
                                                         "config": {"pull_distance": 15.0}}))
    f_drag = qlog.drag_force_from_setup(tmp_path)
    assert math.isclose(f_drag, drag)
    est = qlog.estimate_detachment(pull, drag_force_kcal_mol_A=f_drag)
    assert est.plateau_reached and abs(est.work_kcal_mol - exact) < 0.3
    assert est.raw_work_kcal_mol > est.work_kcal_mol + 3.0 and "Langevin drag" in est.summary()
